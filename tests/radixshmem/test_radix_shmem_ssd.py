"""The radixshmem SSD tier: an index-only radix-server whose slot ids are
FlexKV's SSD blocks, next to the data-mode CPU server.

Drives `RadixShmemCacheEngine` on real in-process servers (standalone, so the
SSD matches are local; a peer's SSD run is forced through a synthetic match):
PUT writes through (D2H -> H2DISK, each tier published by its own writers),
GET falls back to the local SSD, PREFETCH stages the SSD tail into the CPU
tree, SWA rides along, a stale index is cleared at attach, a replaced server
segment turns the tiers off; plus the
`KVTaskEngine` side of a prefetch carrying both a pull job and a graph.
Skipped without radixshmem or `flexkv.c_ext`.
"""
from __future__ import annotations

import contextlib
import copy
import os
from types import SimpleNamespace

import numpy as np
import pytest

try:
    import radixshmem
except ImportError as exc:
    pytest.skip(f"radixshmem unusable ({exc})", allow_module_level=True)

try:
    import torch

    from flexkv.cache.cache_engine import DEFAULT_CACHE_STRATEGY
    from flexkv.cache.radix_shmem_engine import ShmRadixMatch
    from flexkv.cache.radix_shmem_planner import RadixShmemCacheEngine
    from flexkv.common.block import SequenceMeta
    from flexkv.common.config import (GLOBAL_CONFIG_FROM_ENV, CacheConfig, ModelConfig,
                                      SWAPoolConfig)
    from flexkv.common.pool import PoolId
    from flexkv.common.transfer import TransferType
    from flexkv.server import shm_radix_bootstrap as bootstrap
except Exception as exc:  # pragma: no cover - environment-dependent
    pytest.skip(f"RadixShmemCacheEngine unavailable (needs flexkv.c_ext): {exc}",
                allow_module_level=True)

TPB = 4
CPU_BLOCKS = 64
SSD_BLOCKS = 256
SWA_W = 4
FULL_MASK = int(radixshmem.COMPONENT_MASK_FULL)
JOINT_MASK = FULL_MASK | int(radixshmem.COMPONENT_MASK_SWA)
SWA = radixshmem.ComponentType.SWA


def _sweep(name: str) -> None:
    base = name.lstrip("/")
    for path in (f"/dev/shm/{base}", f"/dev/shm/{base}.sock", f"/dev/shm/{base}_data"):
        with contextlib.suppress(FileNotFoundError, IsADirectoryError):
            os.remove(path)


@contextlib.contextmanager
def _engine(swa: bool = False, ssd_blocks: int = SSD_BLOCKS, reuse_ssd_server=None,
            tag: str = ""):
    """A `RadixShmemCacheEngine` on a data-mode CPU server and an index-only
    SSD server (both in-process). Yields (engine, ssd_server)."""
    pid = os.getpid()
    cpu_name, ssd_name = f"/ssdtcpu{pid}{tag}", f"/ssdtssd{pid}"
    saved = {k: getattr(GLOBAL_CONFIG_FROM_ENV, k) for k in
             ("enable_radixshmem", "radixshmem_server_name", "radixshmem_ssd_server_name")}
    GLOBAL_CONFIG_FROM_ENV.enable_radixshmem = True
    GLOBAL_CONFIG_FROM_ENV.radixshmem_server_name = cpu_name
    GLOBAL_CONFIG_FROM_ENV.radixshmem_ssd_server_name = ssd_name
    servers, engine = [], None
    try:
        cache_config = CacheConfig(tokens_per_block=TPB, enable_cpu=True, enable_ssd=True,
                                   enable_remote=False, num_cpu_blocks=CPU_BLOCKS,
                                   num_ssd_blocks=ssd_blocks)
        swa_slots = 4 * SWA_W
        if swa:
            cache_config.swa = SWAPoolConfig(enabled=True, num_slots=swa_slots,
                                             num_ssd_slots=4 * swa_slots, num_swa_layers=1,
                                             bytes_per_token_per_layer=64,
                                             window_blocks=SWA_W)
            cache_config.enable_swa_transfer = True
        model_config = ModelConfig(num_layers=2, num_kv_heads=4, head_size=64,
                                   dtype=torch.float16, tp_size=1, dp_size=1)
        geo = bootstrap.expected_geometry(model_config, cache_config)
        data_bytes = CPU_BLOCKS * geo.full_slot_bytes + (swa_slots * geo.swa_slot_bytes if swa else 0)
        _sweep(cpu_name)
        servers.append(radixshmem.RadixServer(radixshmem.ServerConfig(
            name=cpu_name, data_bytes=data_bytes, prefault=False,
            swa_ratio=(swa_slots * geo.swa_slot_bytes / data_bytes) if swa else 0.0)).start())
        if reuse_ssd_server is None:
            _sweep(ssd_name)
            ssd_server = radixshmem.RadixServer(radixshmem.ServerConfig(name=ssd_name)).start()
            servers.append(ssd_server)
        else:
            ssd_server = reuse_ssd_server
        engine = RadixShmemCacheEngine(cache_config, model_config)
        yield engine, ssd_server
    finally:
        if engine is not None:
            for tier in (engine.cpu_cache_engine, engine.ssd_cache_engine):
                if tier is not None:
                    tier.close()
        for server in reversed(servers):
            server.close()
        for k, v in saved.items():
            setattr(GLOBAL_CONFIG_FROM_ENV, k, v)


def _request(num_blocks: int, base: int = 0):
    n = num_blocks * TPB
    token_ids = np.arange(base, base + n, dtype=np.int64)
    mask = np.ones(n, dtype=np.bool_)
    slot_mapping = (np.repeat(np.arange(1000, 1000 + num_blocks), TPB) * TPB
                    + np.tile(np.arange(TPB), num_blocks)).astype(np.int64)
    return token_ids, mask, slot_mapping


def _seq(token_ids):
    return SequenceMeta(token_ids=np.asarray(token_ids).copy(), tokens_per_block=TPB)


def _ops(graph):
    out = {}
    for op in graph._op_map.values():
        out.setdefault((op.transfer_type, op.pool_id), []).append(op)
    return out


def _put(engine, token_ids, mask, slot_mapping, rid=1, complete=True):
    graph, ret, cb, op_cbs, end = engine.put(request_id=rid, token_ids=token_ids,
                                             token_mask=mask, slot_mapping=slot_mapping,
                                             dp_client_id=0)
    if complete:
        for op_id in sorted(op_cbs):
            op_cbs[op_id]()
        cb()
    return graph, ret, cb, op_cbs


def _get(engine, token_ids, mask, slot_mapping, rid=2, prefetch=False, swa_aware=False):
    strategy = copy.deepcopy(DEFAULT_CACHE_STRATEGY)
    if prefetch:
        strategy.ignore_gpu = True
        strategy.ignore_gds = True
    return engine.get(request_id=rid, token_ids=token_ids, token_mask=mask,
                      slot_mapping=slot_mapping, dp_client_id=0,
                      temp_cache_strategy=strategy, swa_aware=swa_aware)


def _hit(tier, token_ids, mask=FULL_MASK) -> int:
    m = tier.match(_seq(token_ids), component_mask=mask)
    try:
        return m.num_matched_blocks
    finally:
        m.release()


def test_put_writes_through_and_publishes_each_tier_at_its_writer():
    with _engine() as (engine, _):
        cpu, ssd = engine.cpu_cache_engine, engine.ssd_cache_engine
        tok, mask, sm = _request(12)
        graph, ret, cb, op_cbs = _put(engine, tok, mask, sm, complete=False)
        ops = _ops(graph)
        d2h = ops[(TransferType.D2H, PoolId.FULL_KV)]
        h2disk = ops[(TransferType.H2DISK, PoolId.FULL_KV)]
        assert len(d2h) == 1 and sum(len(o.dst_block_ids) for o in h2disk) == 12
        for op in h2disk:
            assert d2h[0].op_id in op.predecessors
            np.testing.assert_array_equal(op.src_block_ids, d2h[0].dst_block_ids)
        assert bool(ret.all())
        assert _hit(cpu, tok) == 0 and _hit(ssd, tok) == 0

        op_cbs[d2h[0].op_id]()                      # D2H done: CPU readable
        assert _hit(cpu, tok) == 12 and _hit(ssd, tok) == 0
        for op in h2disk:                            # H2DISK done: SSD readable
            op_cbs[op.op_id]()
        assert _hit(ssd, tok) == 12
        cb()                                         # nothing re-published, no pin left
        assert _hit(cpu, tok) == 12
        assert len(cpu.take(CPU_BLOCKS)) == CPU_BLOCKS   # every block evictable again


def test_put_extension_writes_the_ssd_from_cached_and_new_cpu_blocks():
    with _engine() as (engine, _):
        cpu, ssd = engine.cpu_cache_engine, engine.ssd_cache_engine
        tok, mask, sm = _request(8)
        _put(engine, tok, mask, sm)
        ssd.reset()                                  # SSD lost the prefix, CPU kept it
        tok2, mask2, sm2 = _request(12)
        graph, ret, _, _ = _put(engine, tok2, mask2, sm2, rid=3)
        h2disk = _ops(graph)[(TransferType.H2DISK, PoolId.FULL_KV)]
        assert sum(len(o.src_block_ids) for o in h2disk) == 12
        assert _hit(ssd, tok2) == 12 and _hit(cpu, tok2) == 12


def test_put_of_a_cpu_cached_prefix_does_not_copy_it_to_the_ssd():
    with _engine() as (engine, _):
        ssd = engine.ssd_cache_engine
        tok, mask, sm = _request(8)
        _put(engine, tok, mask, sm)
        ssd.reset()
        graph, ret, _, _ = _put(engine, tok, mask, sm, rid=3)
        assert graph.num_ops == 0 and _hit(ssd, tok) == 0


def test_get_falls_back_to_the_local_ssd_and_publishes_into_cpu():
    with _engine() as (engine, _):
        cpu = engine.cpu_cache_engine
        tok, mask, sm = _request(10)
        _put(engine, tok, mask, sm)
        cpu.reset()                                  # CPU lost it, SSD has it
        graph, ret, cb, _, end = _get(engine, tok, mask, sm)
        ops = _ops(graph)
        disk2h = ops[(TransferType.DISK2H, PoolId.FULL_KV)]
        h2d = ops[(TransferType.H2D, PoolId.FULL_KV)]
        assert len(h2d) == 1 and len(h2d[0].src_block_ids) == 10
        assert all(op.op_id in h2d[0].predecessors for op in disk2h)
        staged = np.concatenate([op.dst_block_ids for op in disk2h[:1]])
        np.testing.assert_array_equal(h2d[0].src_block_ids, staged)
        assert int(ret.sum()) == 10 * TPB
        assert _hit(cpu, tok) == 0
        cb()
        assert _hit(cpu, tok) == 10


def test_get_reads_cpu_then_ssd_for_the_tail():
    with _engine() as (engine, _):
        cpu, ssd = engine.cpu_cache_engine, engine.ssd_cache_engine
        tok, mask, sm = _request(10)
        _put(engine, tok, mask, sm)
        tok_short, mask_s, sm_s = _request(4)
        cpu.reset()
        _put(engine, tok_short, mask_s, sm_s, rid=5)  # CPU [0,4), SSD [0,10)
        assert _hit(cpu, tok) == 4 and _hit(ssd, tok) == 10
        graph, ret, cb, _, _ = _get(engine, tok, mask, sm)
        ops = _ops(graph)
        disk2h = ops[(TransferType.DISK2H, PoolId.FULL_KV)]
        assert sum(len(o.dst_block_ids) for o in disk2h[:1]) == 6
        assert len(ops[(TransferType.H2D, PoolId.FULL_KV)][0].src_block_ids) == 10
        cb()
        assert _hit(cpu, tok) == 10


def test_prefetch_stages_the_ssd_tail_into_cpu():
    with _engine() as (engine, _):
        cpu = engine.cpu_cache_engine
        tok, mask, sm = _request(9)
        _put(engine, tok, mask, sm)
        cpu.reset()
        graph, ret, cb, _, _ = _get(engine, tok, mask, sm, prefetch=True)
        ops = _ops(graph)
        assert (TransferType.H2D, PoolId.FULL_KV) not in ops
        assert (TransferType.DISK2H, PoolId.FULL_KV) in ops
        assert int(ret.sum()) == 9 * TPB
        assert cb.prefetch_job is None and cb.prefetch_ssd_end() == 0
        cb()
        assert _hit(cpu, tok) == 9 and cb.prefetch_ssd_end() == 9


def test_prefetch_abort_returns_the_staging_slots():
    with _engine() as (engine, _):
        cpu = engine.cpu_cache_engine
        tok, mask, sm = _request(9)
        _put(engine, tok, mask, sm)
        cpu.reset()
        _, _, cb, _, _ = _get(engine, tok, mask, sm, prefetch=True)
        cb.abort()
        assert _hit(cpu, tok) == 0
        assert len(cpu.take(CPU_BLOCKS)) == CPU_BLOCKS


def test_prefetch_plans_a_peer_ssd_run_as_peerssd2h():
    """A cluster-wide SSD match ending in a peer's run: the local part is a
    DISK2H, the peer part a PEERSSD2H naming the peer's blocks and rank."""
    with _engine() as (engine, _):
        cpu, ssd = engine.cpu_cache_engine, engine.ssd_cache_engine
        tok, mask, sm = _request(10)
        _put(engine, tok, mask, sm)
        cpu.reset()
        local = ssd.match(_seq(tok))
        local_slots = local.local_slots[:6].copy()
        released = []
        fake = ShmRadixMatch(num_matched_blocks=10, local_slots=local_slots,
                             remote_rank=3, remote_slots=np.arange(500, 504, dtype=np.int64),
                             finalize=lambda: (released.append(1), local.release()))
        seen = {}

        def _match(seq, **kw):
            seen.update(kw)
            return fake
        ssd.match = _match                           # type: ignore[method-assign]
        graph, ret, cb, _, _ = _get(engine, tok, mask, sm, prefetch=True)
        assert seen.get("local_only") is False
        ops = _ops(graph)
        disk2h = ops[(TransferType.DISK2H, PoolId.FULL_KV)]
        peer = ops[(TransferType.PEERSSD2H, PoolId.FULL_KV)]
        assert sum(len(o.src_block_ids) for o in disk2h[:1]) == 6
        assert len(peer) == 1
        np.testing.assert_array_equal(peer[0].src_block_ids, np.arange(500, 504))
        np.testing.assert_array_equal(peer[0].src_block_node_ids, [3, 3, 3, 3])
        assert int(ret.sum()) == 10 * TPB
        assert released == []
        cb()
        assert released == [1] and _hit(cpu, tok) == 10


def test_swa_rides_through_the_ssd():
    with _engine(swa=True) as (engine, _):
        cpu, ssd = engine.cpu_cache_engine, engine.ssd_cache_engine
        tok, mask, sm = _request(10)
        graph, _, _, _ = _put(engine, tok, mask, sm)
        ops = _ops(graph)
        assert len(ops[(TransferType.H2DISK, PoolId.SWA)][0].dst_block_ids) == SWA_W
        assert _hit(ssd, tok, JOINT_MASK) == 10
        cpu.reset()
        graph, ret, cb, _, _ = _get(engine, tok, mask, sm, swa_aware=True)
        ops = _ops(graph)
        swa_read = ops[(TransferType.DISK2H, PoolId.SWA)]
        swa_h2d = ops[(TransferType.H2D, PoolId.SWA)]
        assert len(swa_read[0].src_block_ids) == SWA_W
        assert swa_read[0].op_id in swa_h2d[0].predecessors
        cb()
        assert _hit(cpu, tok, JOINT_MASK) == 10


def test_stale_ssd_index_is_cleared_at_attach():
    with _engine() as (engine, ssd_server):
        tok, mask, sm = _request(6)
        _put(engine, tok, mask, sm)
        assert _hit(engine.ssd_cache_engine, tok) == 6
        engine.ssd_cache_engine.close()
        engine.ssd_cache_engine = None
        with _engine(reuse_ssd_server=ssd_server, tag="b") as (engine2, _):
            assert _hit(engine2.ssd_cache_engine, tok) == 0


def test_a_replaced_radix_server_segment_turns_the_tiers_off():
    """A radix-server restarted under a live FlexKV recreates its segments; the
    old mapping must not be used for GET or PUT any more."""
    with _engine() as (engine, _):
        tok, mask, sm = _request(6)
        _put(engine, tok, mask, sm)
        assert int(_get(engine, tok, mask, sm)[1].sum()) == 6 * TPB
        path = "/dev/shm/" + engine.cpu_cache_engine.shm_name.lstrip("/")
        os.rename(path, path + ".old")               # what a restarted server leaves behind
        try:
            with open(path, "wb"):
                pass
            engine._segments._next = 0.0
            assert int(_get(engine, tok, mask, sm)[1].sum()) == 0
            tok2, mask2, sm2 = _request(4, base=10_000)
            assert int(_put(engine, tok2, mask2, sm2)[1].sum()) == 0
            assert _hit(engine.cpu_cache_engine, tok2) == 0
        finally:
            os.replace(path + ".old", path)


def _bare_task_engine():
    from flexkv.kvtask import KVTaskEngine
    mgr = KVTaskEngine.__new__(KVTaskEngine)
    mgr.cache_config = CacheConfig(tokens_per_block=TPB, num_cpu_blocks=64)
    mgr.cache_engine = SimpleNamespace(supports_prefetch_inherit=False)
    mgr.tasks, mgr.prefetch_jobs, mgr.graph_to_task = {}, {}, {}
    mgr.transfer_handles = []
    return mgr


class _Job:
    def __init__(self):
        self.local_hit, self.planned_hit, self.job_id, self.cancelled = 1, 2, 9, False
        self._r = None

    def done(self):
        return self._r is not None

    def wait(self, timeout=None):
        return self._r

    def cancel(self):
        self.cancelled = True

    def finish(self):
        self._r = SimpleNamespace(common_hit=2, remote_blocks=1, remote_bytes=0,
                                  source_rank=1, finalize=lambda: None)


def _job_and_graph_task(job, ssd_end):
    from flexkv.common.transfer import TransferOp, TransferOpGraph
    from flexkv.kvtask import KVTask, TaskStatus, TaskType
    graph = TransferOpGraph()
    graph.add_transfer_op(TransferOp(graph_id=graph.graph_id, transfer_type=TransferType.DISK2H,
                                     src_block_ids=np.arange(2), dst_block_ids=np.arange(2)))
    published = []

    class _Handle:
        """What KVTaskEngine reads off a radixshmem prefetch plan handle."""
        prefetch_job = job

        def __call__(self):
            published.append(1)

        def prefetch_ssd_end(self):
            return ssd_end if published else 0

    n = 4 * TPB
    task = KVTask(task_id=1, task_type=TaskType.PREFETCH, task_end_op_id=-1,
                  task_end_op_finished=False, status=TaskStatus.RUNNING,
                  token_ids=np.arange(n), slot_mapping=np.zeros(n, dtype=np.int64),
                  token_mask=np.ones(n, dtype=np.bool_), graph=graph,
                  return_mask=np.zeros(n, dtype=np.bool_), callback=_Handle(),
                  op_callback_dict={}, prefetch_job=job, prefetch_local_hit_blocks=1,
                  prefetch_planned_hit_blocks=4)
    return task, published


@pytest.mark.parametrize("graph_first", [True, False])
def test_task_engine_waits_for_both_the_job_and_the_graph(graph_first):
    from flexkv.kvtask import TaskStatus
    mgr = _bare_task_engine()
    job = _Job()
    task, published = _job_and_graph_task(job, ssd_end=4)
    mgr.tasks[1], mgr.prefetch_jobs[1] = task, job
    if graph_first:
        assert mgr._prefetch_job_pending(1)          # graph done, job not
        assert task.status == TaskStatus.RUNNING and published == []
        job.finish()
        mgr._poll_prefetch_jobs()
    else:
        job.finish()
        mgr._poll_prefetch_jobs()                    # job done, graph not
        assert task.status == TaskStatus.RUNNING and 1 in mgr.prefetch_jobs
        assert mgr._prefetch_job_pending(1)
    assert task.status == TaskStatus.COMPLETED and published == [1]
    assert not bool(task.return_mask[:TPB].any())
    assert bool(task.return_mask[TPB:4 * TPB].all())


def test_full_only_put_skips_the_swa_window():
    """An intermediate (FULL-only) PUT stores Full KV in CPU and SSD with no
    SWA op and no SWA slot; the final PUT, one block longer, adds the window."""
    from flexkv.cache.cache_engine import FULL_ONLY_PUT_STRATEGY
    with _engine(swa=True) as (engine, _):
        cpu, ssd = engine.cpu_cache_engine, engine.ssd_cache_engine
        tok, mask, sm = _request(9)
        graph, _, cb, op_cbs, _ = engine.put(request_id=1, token_ids=tok, token_mask=mask,
                                             slot_mapping=sm, dp_client_id=0,
                                             temp_cache_strategy=FULL_ONLY_PUT_STRATEGY)
        assert all(op.pool_id != PoolId.SWA for op in graph._op_map.values())
        for op_id in sorted(op_cbs):
            op_cbs[op_id]()
        cb()
        assert _hit(cpu, tok) == 9 and _hit(ssd, tok) == 9
        assert _hit(cpu, tok, JOINT_MASK) == 0          # no window yet
        tok2, mask2, sm2 = _request(10)
        graph, _, _, _ = _put(engine, tok2, mask2, sm2, rid=2)
        ops = _ops(graph)
        assert len(ops[(TransferType.D2H, PoolId.FULL_KV)][0].dst_block_ids) == 1
        assert (TransferType.D2H, PoolId.SWA) in ops
        assert _hit(cpu, tok2, JOINT_MASK) == 10 and _hit(ssd, tok2, JOINT_MASK) == 10


@contextlib.contextmanager
def _dedup_peer(engine, local_blocks: int, peer_blocks: int, peer: int = 3,
                take_ok: bool = True):
    """Global dedup against a synthetic peer whose SSD holds [0, peer_blocks)
    while this node's SSD holds [0, local_blocks). Yields the recorded calls."""
    ssd = engine.ssd_cache_engine
    engine._ssd_global_dedup = True
    ssd.peer_enabled = True
    calls = {"take": [], "insert": [], "recycle": [], "local_only": []}
    real_match = ssd.match

    def _match(seq, **kw):
        calls["local_only"].append(kw.get("local_only", True))
        if kw.get("local_only", True):
            return real_match(seq, **kw)
        local = real_match(seq, local_only=True)
        nl = min(local.num_local_blocks, local_blocks)
        return ShmRadixMatch(num_matched_blocks=max(nl, peer_blocks),
                             local_slots=local.local_slots[:nl].copy(),
                             remote_rank=peer if peer_blocks > nl else -1,
                             remote_slots=np.arange(900, 900 + max(0, peer_blocks - nl),
                                                    dtype=np.int64),
                             finalize=local.release)

    def _take(rank, n_full, n_swa=0):
        calls["take"].append((rank, n_full, n_swa))
        if not take_ok:
            return None, None, "allocate error MEMPOOL_FULL"
        return (np.arange(500, 500 + n_full, dtype=np.int64),
                np.arange(700, 700 + n_swa, dtype=np.int64) if n_swa else None, "")

    def _insert(rank, seq, path_end, full, swa=None):
        calls["insert"].append((rank, path_end, full.tolist(), None if swa is None else swa.tolist()))
        return len(full)

    def _recycle(rank, slots, component=None):
        calls["recycle"].append((rank, list(slots), component))
        return True
    ssd.match, ssd.remote_take = _match, _take        # type: ignore[method-assign]
    ssd.remote_insert, ssd.remote_recycle = _insert, _recycle  # type: ignore[method-assign]
    yield calls


def test_global_dedup_skips_what_a_peer_ssd_holds():
    with _engine() as (engine, _):
        tok, mask, sm = _request(10)
        with _dedup_peer(engine, local_blocks=0, peer_blocks=10) as calls:
            graph, ret, _, _ = _put(engine, tok, mask, sm)
        ops = _ops(graph)
        assert calls["local_only"] == [False]
        assert (TransferType.H2DISK, PoolId.FULL_KV) not in ops
        assert (TransferType.H2PEERSSD, PoolId.FULL_KV) not in ops
        assert calls["take"] == [] and _hit(engine.ssd_cache_engine, tok) == 0
        assert _hit(engine.cpu_cache_engine, tok) == 10


@pytest.mark.parametrize("ok", [True, False])
def test_global_dedup_writes_the_rest_to_the_peer_holding_the_prefix(ok):
    """[0, 6) on peer 3 -> [6, 10) goes to peer 3's SSD (H2PEERSSD into slots
    allocated there) and is published there once written; a failed write
    gives the slots back to the peer instead and fails nothing."""
    from flexkv.common.transfer import CompletedOp
    with _engine() as (engine, _):
        tok, mask, sm = _request(10)
        with _dedup_peer(engine, local_blocks=2, peer_blocks=6) as calls:
            graph, ret, cb, op_cbs = _put(engine, tok, mask, sm, complete=False)
            ops = _ops(graph)
            assert (TransferType.H2DISK, PoolId.FULL_KV) not in ops
            (peer_op,) = ops[(TransferType.H2PEERSSD, PoolId.FULL_KV)]
            (d2h,) = ops[(TransferType.D2H, PoolId.FULL_KV)]
            assert calls["take"] == [(3, 4, 0)]
            np.testing.assert_array_equal(peer_op.dst_block_ids, np.arange(500, 504))
            np.testing.assert_array_equal(peer_op.src_block_node_ids, [3] * 4)
            np.testing.assert_array_equal(peer_op.src_block_ids, d2h.dst_block_ids[6:])
            assert d2h.op_id in peer_op.predecessors
            op_cbs[d2h.op_id]()
            done = CompletedOp(graph_id=graph.graph_id, op_id=peer_op.op_id,
                               block_results=(ok,) * 4)
            op_cbs[peer_op.op_id](done)
            cb()
        if ok:
            assert calls["insert"] == [(3, 10, [500, 501, 502, 503], None)]
            assert calls["recycle"] == []
        else:
            assert calls["insert"] == []
            assert [(r, s) for r, s, _ in calls["recycle"]] == [(3, [500, 501, 502, 503])]
        assert _hit(engine.ssd_cache_engine, tok) == 0     # nothing written locally
        taken = engine.cpu_cache_engine.take(CPU_BLOCKS)
        assert len(taken) == CPU_BLOCKS                    # no pin left
        engine.cpu_cache_engine.recycle(taken)
        if not ok:   # the peer is written around for a while: the next PUT writes locally
            tok2, mask2, sm2 = _request(12, base=1000)
            graph2, _, _, _ = _put(engine, tok2, mask2, sm2, rid=2)
            assert (TransferType.H2PEERSSD, PoolId.FULL_KV) not in _ops(graph2)
            assert engine._ssd_dedup_stats.get("fallback_backoff") == 1


def test_global_dedup_abort_returns_the_peer_slots():
    with _engine() as (engine, _):
        tok, mask, sm = _request(10)
        with _dedup_peer(engine, local_blocks=0, peer_blocks=4) as calls:
            graph, ret, cb, op_cbs = _put(engine, tok, mask, sm, complete=False)
            cb.abort()
        assert calls["insert"] == []
        assert [(r, len(s)) for r, s, _ in calls["recycle"]] == [(3, 6)]


def test_global_dedup_writes_locally_when_the_peer_cannot_take_it():
    with _engine() as (engine, _):
        tok, mask, sm = _request(10)
        with _dedup_peer(engine, local_blocks=2, peer_blocks=6, take_ok=False) as calls:
            graph, ret, _, _ = _put(engine, tok, mask, sm)
        ops = _ops(graph)
        assert (TransferType.H2PEERSSD, PoolId.FULL_KV) not in ops
        h2disk = ops[(TransferType.H2DISK, PoolId.FULL_KV)]
        assert sum(len(o.dst_block_ids) for o in h2disk[:1]) == 10  # this node's own run: none
        assert engine._ssd_dedup_stats.get("fallback_alloc") == 1


def test_global_dedup_local_fraction_skips_the_cluster_lookup():
    with _engine() as (engine, _):
        tok, mask, sm = _request(10)
        with _dedup_peer(engine, local_blocks=0, peer_blocks=10) as calls:
            engine._ssd_dedup_local_fraction = 1.0
            graph, ret, _, _ = _put(engine, tok, mask, sm)
        assert calls["local_only"] == [True]
        assert (TransferType.H2DISK, PoolId.FULL_KV) in _ops(graph)


def test_global_dedup_carries_the_swa_window_to_the_peer():
    from flexkv.common.transfer import CompletedOp
    with _engine(swa=True) as (engine, _):
        tok, mask, sm = _request(10)
        with _dedup_peer(engine, local_blocks=0, peer_blocks=6) as calls:
            graph, ret, cb, op_cbs = _put(engine, tok, mask, sm, complete=False)
            ops = _ops(graph)
            (swa_op,) = ops[(TransferType.H2PEERSSD, PoolId.SWA)]
            (full_op,) = ops[(TransferType.H2PEERSSD, PoolId.FULL_KV)]
            assert (TransferType.H2DISK, PoolId.SWA) not in ops
            assert calls["take"] == [(3, 4, SWA_W)]
            for op_id in sorted(op_cbs):
                if op_id in (swa_op.op_id, full_op.op_id):
                    continue
                op_cbs[op_id]()
            op_cbs[full_op.op_id](CompletedOp(graph_id=graph.graph_id, op_id=full_op.op_id,
                                              block_results=(True,) * 4))
            assert calls["insert"] == []                # waits for the window too
            op_cbs[swa_op.op_id](CompletedOp(graph_id=graph.graph_id, op_id=swa_op.op_id,
                                             block_results=(True,) * SWA_W))
            cb()
        assert calls["insert"] == [(3, 10, [500, 501, 502, 503], list(range(700, 700 + SWA_W)))]


def test_ssd_write_keeps_cpu_slots_a_concurrent_put_published_first():
    """Two PUTs of the same blocks planned together: the second one's CPU
    insert finds them published and does not take its slots, which its SSD
    write still reads -- they go back only once that write is done."""
    with _engine() as (engine, _):
        cpu, ssd = engine.cpu_cache_engine, engine.ssd_cache_engine
        tok, mask, sm = _request(8)
        g1, _, cb1, cbs1 = _put(engine, tok, mask, sm, rid=1, complete=False)
        g2, _, cb2, cbs2 = _put(engine, tok, mask, sm, rid=2, complete=False)
        for op_id in sorted(cbs1):
            cbs1[op_id]()
        cb1()
        free = cpu.num_free_blocks
        (d2h,) = _ops(g2)[(TransferType.D2H, PoolId.FULL_KV)]
        cbs2[d2h.op_id]()                               # publish: all 8 already there
        assert cpu.num_free_blocks == free              # not recycled under the SSD write
        for op_id in sorted(cbs2):
            if op_id != d2h.op_id:
                cbs2[op_id]()
        assert cpu.num_free_blocks == free + 8
        cb2()
        assert _hit(ssd, tok) == 8
