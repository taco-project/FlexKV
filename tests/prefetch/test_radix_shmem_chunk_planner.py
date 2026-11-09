"""Unit tests for RadixShmemChunkPlanner + PrefetchCoordinator policy wiring."""

from __future__ import annotations

from types import SimpleNamespace as NS

import numpy as np
import pytest

from flexkv.prefetch.coordinator import PrefetchCoordinator
from flexkv.prefetch.types import PrefetchBusy, PrefetchOptions


class Clock:
    now = 0.0

    def __call__(self):
        return self.now


class FakeJob:
    def __init__(self, local_hit, planned_hit, common_hit=None):
        self.local_hit = local_hit
        self.planned_hit = planned_hit
        self._common = planned_hit if common_hit is None else common_hit
        self._done = False
        self.cancelled = False
        self.wait_count = 0
        self.job_id = id(self)

    def done(self):
        return self._done

    @property
    def published_hit(self):
        return self._common if self._done else self.local_hit

    def complete(self, common_hit=None):
        if common_hit is not None:
            self._common = common_hit
        self._done = True

    def wait(self, timeout=None):
        assert self._done
        self.wait_count += 1
        return NS(common_hit=self._common, finalize=lambda: None)

    def cancel(self):
        self.cancelled = True


class FakeCPU:
    peer_enabled = True

    def __init__(self):
        self.prefetch_calls = []
        self._jobs = []

    def prefetch(self, sequence, **kwargs):
        self.prefetch_calls.append(kwargs)
        # Local already covers hash_start..; planned to hash_end.
        start = int(kwargs.get("hash_start", 0) or 0)
        end = int(kwargs.get("hash_end") or kwargs.get("query_end") or 0)
        job = FakeJob(local_hit=start, planned_hit=end, common_hit=end)
        self._jobs.append(job)
        return job


class FakeCache:
    def __init__(self, cpu):
        self.cpu_cache_engine = cpu
        self._cache_tree_lock = NS(__enter__=lambda s: None, __exit__=lambda *a: None)


class FakeEngine:
    def __init__(self, block_size=1, block_bytes=1):
        self.cache_config = NS(tokens_per_block=block_size)
        self.model_config = NS(
            num_kv_heads=8,
            effective_tp_size_per_node=1,
        )
        self.cpu = FakeCPU()
        self.cache_engine = FakeCache(self.cpu)
        self._prefetch = NS(reserved_bytes=0)


def _patch_block_bytes(monkeypatch, nbytes=1):
    monkeypatch.setattr(
        "flexkv.prefetch.radixshmem_planner.block_size_in_bytes_for_cache",
        lambda *a, **k: nbytes,
        raising=False,
    )
    # Import path used inside __init__
    import flexkv.common.config as cfg

    monkeypatch.setattr(
        cfg, "block_size_in_bytes_for_cache", lambda *a, **k: nbytes, raising=False
    )


@pytest.fixture
def planner(monkeypatch):
    from flexkv.prefetch import radixshmem_planner as mod

    monkeypatch.setattr(
        "flexkv.common.config.block_size_in_bytes_for_cache",
        lambda *a, **k: 1,
    )
    monkeypatch.setattr(mod, "PREFETCH_MAX_INFLIGHT", 8)
    engine = FakeEngine(block_size=1, block_bytes=1)
    # Bypass RankInfo / block_size_in_bytes by constructing manually.
    p = mod.RadixShmemChunkPlanner.__new__(mod.RadixShmemChunkPlanner)
    p.engine = engine
    p.cache = engine.cache_engine
    p.block_size = 1
    p.block_bytes = 1
    p.max_pinned_bytes = 10_000
    p.pinned_bytes = 0
    p.chunk_timeout_ms = 5000
    from concurrent.futures import ThreadPoolExecutor

    p.executor = ThreadPoolExecutor(max_workers=1)
    p._submit_executor = ThreadPoolExecutor(max_workers=1)
    p._inflight = {}
    p._default_mask = 1
    yield p, engine
    p._submit_executor.shutdown(wait=True)
    p.executor.shutdown(wait=True)


def test_reserve_serializes_inflight(planner):
    p, engine = planner
    ctx = NS(sequence=NS())
    options = PrefetchOptions(chunk_max_blocks=4, max_inflight_chunks=2)
    c1 = p.reserve(ctx, 0, 16, options, 10_000)
    assert c1 is not None and c1.begin == 0 and c1.end == 4
    p._inflight[c1.graph.graph_id] = c1
    with pytest.raises(PrefetchBusy):
        p.reserve(ctx, 4, 16, options, 10_000)


def test_submit_and_poll_advances_published_end(planner):
    p, engine = planner
    from flexkv.common.block import SequenceMeta

    tokens = np.arange(16, dtype=np.int64)
    seq = SequenceMeta(tokens, 1, None)
    seq.gen_hashes = lambda: None  # type: ignore[method-assign]
    seq.block_hashes = np.arange(16, dtype=np.int64)
    ctx = NS(sequence=seq, match=None, pinned_bytes=0, reusable_end=0, namespace=None, tokens=tokens)
    options = PrefetchOptions(chunk_max_blocks=4)
    chunk = p.reserve(ctx, 0, 16, options, 10_000)
    p.submit_batch([chunk])
    chunk.submit_future.result(timeout=5)
    assert len(engine.cpu.prefetch_calls) == 1
    assert engine.cpu.prefetch_calls[0]["hash_end"] == 4
    job = engine.cpu._jobs[0]
    assert not job.done()
    assert p.poll_completions() == []
    job.complete(common_hit=4)
    comps = p.poll_completions()
    assert len(comps) == 1 and comps[0].graph_id == chunk.graph.graph_id
    assert chunk.done and chunk.published_end == 4


def test_coordinator_timeout_with_radix_backend(planner):
    """PrefetchCoordinator seal/drain works when completions come from poll_completions."""
    p, engine = planner
    from flexkv.common.block import SequenceMeta

    tokens = np.arange(20, dtype=np.int64)
    seq = SequenceMeta(tokens, 1, None)
    seq.gen_hashes = lambda: None  # type: ignore[method-assign]
    seq.block_hashes = np.arange(20, dtype=np.int64)

    # Stub begin/resolve/pin so we don't need a real radix client.
    def begin(tokens, namespace, client_id, options):
        return NS(
            sequence=seq,
            tokens=tokens,
            match=None,
            pinned_bytes=0,
            reusable_end=0,
            namespace=namespace,
            future=NS(done=lambda: True, result=lambda: (seq, 0, 20), cancel=lambda: None),
            cancel_query=NS(set=lambda: None, is_set=lambda: False),
            released=False,
            component_mask=1,
        )

    p.begin = begin  # type: ignore[method-assign]
    p.resolve = lambda ctx: (0, 20)  # type: ignore[method-assign]
    def pin(ctx, end, **kw):
        ctx.reusable_end = end
        return True
    p._pin_prefix = pin  # type: ignore[method-assign]
    p.release = lambda ctx: None  # type: ignore[method-assign]
    p.cancel_query = lambda ctx: None  # type: ignore[method-assign]
    p.result_end = lambda ctx, committed: committed  # type: ignore[method-assign]

    clock = Clock()
    coord = PrefetchCoordinator(p, clock=clock, max_reserved_bytes=10_000)
    engine._prefetch = coord
    handle = coord.start(
        tokens, PrefetchOptions(policy="timeout", chunk_max_blocks=4, timeout_budget_s=5)
    )
    coord.tick()
    for chunk in p._inflight.values():
        chunk.submit_future.result(timeout=5)
    assert len(engine.cpu._jobs) == 1
    clock.now = 5
    coord.tick()
    assert coord.snapshot(handle).stop_reason == "deadline"
    engine.cpu._jobs[0].complete(common_hit=4)
    coord.tick()
    snap = coord.snapshot(handle)
    assert snap.terminal
    assert snap.loaded_tokens == 4


def test_coordinator_best_effort_demand(planner):
    p, engine = planner
    from flexkv.common.block import SequenceMeta

    tokens = np.arange(20, dtype=np.int64)
    seq = SequenceMeta(tokens, 1, None)
    seq.gen_hashes = lambda: None  # type: ignore[method-assign]
    seq.block_hashes = np.arange(20, dtype=np.int64)

    def begin(tokens, namespace, client_id, options):
        return NS(
            sequence=seq,
            tokens=tokens,
            match=None,
            pinned_bytes=0,
            reusable_end=0,
            namespace=namespace,
            future=NS(done=lambda: True, result=lambda: (seq, 0, 20), cancel=lambda: None),
            cancel_query=NS(set=lambda: None, is_set=lambda: False),
            released=False,
            component_mask=1,
        )

    p.begin = begin  # type: ignore[method-assign]
    p.resolve = lambda ctx: (0, 20)  # type: ignore[method-assign]
    def pin(ctx, end, **kw):
        ctx.reusable_end = end
        return True
    p._pin_prefix = pin  # type: ignore[method-assign]
    p.release = lambda ctx: None  # type: ignore[method-assign]
    p.cancel_query = lambda ctx: None  # type: ignore[method-assign]
    p.result_end = lambda ctx, committed: committed  # type: ignore[method-assign]

    clock = Clock()
    coord = PrefetchCoordinator(p, clock=clock, max_reserved_bytes=10_000)
    engine._prefetch = coord
    handle = coord.start(
        tokens, PrefetchOptions(policy="best_effort", chunk_max_blocks=4)
    )
    coord.tick()
    for chunk in p._inflight.values():
        chunk.submit_future.result(timeout=5)
    coord.demand([handle])
    coord.tick()
    assert coord.snapshot(handle).stop_reason == "demand"
    engine.cpu._jobs[0].complete(common_hit=4)
    coord.tick()
    assert coord.snapshot(handle).terminal
    assert coord.snapshot(handle).loaded_tokens == 4


def test_chunk_transfer_timeout_ms_env(monkeypatch):
    from flexkv.server import shm_radix_bootstrap as bootstrap

    monkeypatch.delenv("FLEXKV_RADIXSHMEM_CHUNK_TIMEOUT_MS", raising=False)
    assert bootstrap.chunk_transfer_timeout_ms({}) == bootstrap.PREFETCH_TIMEOUT_MS
    assert bootstrap.chunk_transfer_timeout_ms({"chunk_timeout_ms": 1234}) == 1234
    monkeypatch.setenv("FLEXKV_RADIXSHMEM_CHUNK_TIMEOUT_MS", "2500")
    assert bootstrap.chunk_transfer_timeout_ms({}) == 2500


def test_prefetch_options_ignores_backend_only_keys():
    opts = PrefetchOptions.from_mapping(
        {"policy": "timeout", "chunk_timeout_ms": 999, "chunk_max_blocks": 32}
    )
    assert opts.policy == "timeout"
    assert opts.chunk_max_blocks == 32
    assert not hasattr(opts, "chunk_timeout_ms")


def test_begin_rejects_swa_aware(planner):
    p, _ = planner
    with pytest.raises(ValueError, match="does not support SWA"):
        p.begin(
            np.arange(4, dtype=np.int64),
            namespace=None,
            client_id=0,
            options=PrefetchOptions(swa_aware=True),
        )


def _session_setup(p, engine, *, target=12, local=0):
    from concurrent.futures import Future
    from flexkv.common.block import SequenceMeta
    from flexkv.prefetch.radixshmem_planner import RadixPrefetchContext

    def begin(tokens, namespace, client_id, options):
        ctx = RadixPrefetchContext(tokens, namespace, client_id, component_mask=1)
        ctx.sequence = SequenceMeta(tokens, p.block_size, namespace)
        ctx.future = Future()
        ctx.future.set_result((ctx.sequence, local, target))
        return ctx

    p.begin = begin
    engine.cpu.match = lambda seq, **kw: NS(
        num_matched_blocks=len(seq.token_ids) // p.block_size, release=lambda: None
    )
    coord = PrefetchCoordinator(p, max_reserved_bytes=10000)
    engine._prefetch = coord
    return coord


def _finish_jobs(p, engine, coord):
    for chunk in list(p._inflight.values()):
        if chunk.submit_future is not None:
            chunk.submit_future.result(timeout=5)
    for job in engine.cpu._jobs:
        if not job.done():
            job.complete()
    coord.tick()


def test_batch_claim_busy_retry_and_fairness(planner):
    p, engine = planner
    coord = _session_setup(p, engine)
    opts = PrefetchOptions(policy="wait_complete", chunk_max_blocks=4)
    a = coord.start(np.arange(12, dtype=np.int64), opts)
    b = coord.start(np.arange(12, dtype=np.int64), opts)
    coord.tick()
    assert len(p._inflight) == 1
    assert coord.snapshot(b).stop_reason is None
    # B must get the next claim, before A can reserve its second chunk.
    _finish_jobs(p, engine, coord)
    assert coord.snapshot(b).submitted_chunks == 1
    for _ in range(10):
        _finish_jobs(p, engine, coord)
        if coord.snapshot(a).terminal and coord.snapshot(b).terminal:
            break
    assert coord.snapshot(a).loaded_tokens == 12
    assert coord.snapshot(b).loaded_tokens == 12
    coord.release(a)
    coord.release(b)
    assert coord.reserved_bytes == p.pinned_bytes == 0
    assert not p._inflight


@pytest.mark.parametrize("resident", [0, 3, 8])
def test_resolve_uses_current_pinned_prefix(planner, resident):
    p, engine = planner
    coord = _session_setup(p, engine, target=8, local=8)
    engine.cpu.match = lambda seq, **kw: NS(
        num_matched_blocks=resident, release=lambda: None
    )
    h = coord.start(np.arange(8, dtype=np.int64))
    ctx = coord.sessions[h].context
    assert p.resolve(ctx) == (resident, 8)
    assert p.result_end(ctx, 8) == resident
    assert ctx.pinned_bytes == resident
    p.release(ctx)
    assert p.pinned_bytes == 0


def test_already_done_job_consumed_once(planner):
    p, engine = planner
    coord = _session_setup(p, engine, target=4)
    job = FakeJob(0, 4)
    job.complete()
    engine.cpu.prefetch = lambda *a, **kw: job
    h = coord.start(np.arange(4, dtype=np.int64))
    coord.tick()
    _finish_jobs(p, engine, coord)
    assert coord.snapshot(h).terminal
    assert coord.snapshot(h).loaded_tokens == 4
    assert job.wait_count == 1
    coord.tick()
    assert job.wait_count == 1


@pytest.mark.parametrize("limit,accepted", [(3, False), (4, True)])
def test_pin_budget_includes_compressed_node_tail(planner, limit, accepted):
    p, engine = planner
    coord = _session_setup(p, engine, target=1, local=1)
    engine.cpu.register_chunk_blocks = 4
    p.max_pinned_bytes = limit
    h = coord.start(np.arange(1, dtype=np.int64))
    ctx = coord.sessions[h].context
    assert p._pin_prefix(ctx, 1) is accepted
    assert ctx.pinned_bytes == (4 if accepted else 0)
    p.release(ctx)
    assert p.pinned_bytes == 0


def test_submit_failure_is_session_local(planner):
    p, engine = planner
    coord = _session_setup(p, engine, target=4)
    def fail(*a, **kw):
        raise RuntimeError("index unavailable")
    engine.cpu.prefetch = fail
    h = coord.start(np.arange(4, dtype=np.int64))
    coord.tick()
    for chunk in p._inflight.values():
        with pytest.raises(RuntimeError):
            chunk.submit_future.result(timeout=5)
    coord.tick()
    assert coord.snapshot(h).terminal
    assert coord.snapshot(h).loaded_tokens == 0
    assert coord.snapshot(h).outcome == "failed"
    assert "index unavailable" in coord.snapshot(h).error
    assert coord.reserved_bytes == 0
    engine.cpu.prefetch = lambda *a, **kw: None
    other = coord.start(np.arange(4, dtype=np.int64))
    coord.tick()
    _finish_jobs(p, engine, coord)
    assert coord.snapshot(other).terminal


def test_blocked_submit_does_not_block_demand_and_drains(planner):
    import threading
    p, engine = planner
    coord = _session_setup(p, engine, target=12)
    entered, unblock = threading.Event(), threading.Event()
    original = engine.cpu.prefetch
    def blocked(*a, **kw):
        entered.set()
        assert unblock.wait(5)
        return original(*a, **kw)
    engine.cpu.prefetch = blocked
    h = coord.start(np.arange(12, dtype=np.int64),
                    PrefetchOptions(policy="best_effort", chunk_max_blocks=4))
    try:
        coord.tick()
        assert entered.wait(5)
        coord.demand([h])
        coord.tick()
        assert coord.snapshot(h).state == "draining"
    finally:
        unblock.set()
    _finish_jobs(p, engine, coord)
    assert coord.snapshot(h).terminal
    assert coord.snapshot(h).stop_reason == "demand"
    assert coord.snapshot(h).loaded_tokens == 4
