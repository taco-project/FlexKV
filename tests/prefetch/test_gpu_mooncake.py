"""Opt-in real CUDA IPC + Mooncake RDMA/TCP tests; no transfer mocks.

Set FLEXKV_TEST_REAL_MOONCAKE=1 and FLEXKV_MOONCAKE_STORE_CONFIG_PATH.
Use a dedicated Mooncake master: the fixture owns its namespace and CPU pool.
"""

import json
import os
import time
import uuid
from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest
import torch

if os.getenv("FLEXKV_TEST_REAL_MOONCAKE") != "1":
    pytest.skip("opt-in real GPU/Mooncake validation", allow_module_level=True)

from common_utils import create_gpu_kv_layout, GPU_LAYOUT_SGLANG
from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.common.request import KVResponseStatus
from flexkv.kvmanager import KVManager
from flexkv.prefetch.types import PrefetchOptions
from flexkv.server.client import KVTPClient

assert torch.cuda.is_available(), "real validation requires CUDA"


def until(fn, timeout=30):
    deadline = time.monotonic() + timeout
    while True:
        result = fn()
        if result:
            return result
        assert time.monotonic() < deadline, "real transfer did not finish in time"
        time.sleep(0.001)


def record(label, snapshot, **fields):
    print(
        "REAL_PREFETCH " + json.dumps(dict(case=label, **asdict(snapshot), **fields)),
        flush=True,
    )


@pytest.fixture(scope="module")
def cluster():
    torch.set_num_threads(2)
    model = ModelConfig(
        num_layers=8,
        num_kv_heads=8,
        head_size=128,
        dtype=torch.bfloat16,
        kv_dim=2,
        tp_size=1,
        dp_size=1,
    )
    cfg = CacheConfig(
        tokens_per_block=16,
        num_cpu_blocks=768,
        enable_cpu=True,
        enable_ssd=False,
        enable_remote=True,
        num_remote_blocks=1024,
        use_mooncake_store_backend=True,
        mooncake_store_config_path=os.environ["FLEXKV_MOONCAKE_STORE_CONFIG_PATH"],
        enable_chunked_prefetch=True,
    )
    manager = KVManager(
        model, cfg, server_recv_port="ipc:///tmp/prefetch-real-" + uuid.uuid4().hex
    )
    manager.start()
    engine = manager.kv_task_engine
    layout = create_gpu_kv_layout(model, cfg, 512, GPU_LAYOUT_SGLANG)
    tensors = [
        torch.empty(tuple(layout.kv_shape[2:]), dtype=model.dtype, device="cuda")
        for _ in range(model.num_layers * model.kv_dim)
    ]
    client = KVTPClient(manager.gpu_register_port, dp_client_id=0, device_id=0)
    client.register_to_server(tensors, layout)
    until(manager.is_ready, 180)
    # Every element varies with its physical offset and layer/KV identity.
    pattern = torch.arange(
        tensors[0].numel(), device="cuda", dtype=torch.int32
    ).reshape(tensors[0].shape)
    expected = [
        ((pattern + i * 13) % 251 + 1).to(model.dtype) for i in range(len(tensors))
    ]
    for target, source in zip(tensors, expected):
        target.copy_(source)
    torch.cuda.synchronize()
    tokens = np.arange(512 * 16, dtype=np.int64)
    namespace = ["prefetch-real-" + uuid.uuid4().hex]
    slots = np.repeat(np.arange(512, dtype=np.int64) * 16, 16)
    task = manager.put_async(tokens, slots, namespace=namespace)
    response = manager.wait(task, timeout=120, completely=True)[task]
    assert response.status == KVResponseStatus.SUCCESS, response
    assert response.return_mask.all()
    ctx = SimpleNamespace(
        manager=manager,
        engine=engine,
        tensors=tensors,
        expected=expected,
        tokens=tokens,
        namespace=namespace,
        slots=slots,
        cfg=cfg,
    )
    print(
        "REAL_SEED "
        + json.dumps(
            dict(
                blocks=512,
                tokens=len(tokens),
                bytes=sum(t.numel() * t.element_size() for t in tensors),
                namespace=namespace,
                layout="sglang",
                torch=torch.__version__,
            )
        ),
        flush=True,
    )
    try:
        yield ctx
    finally:
        manager.shutdown()


@pytest.fixture
def clean(cluster):
    c = cluster

    def clear():
        assert c.engine._prefetch.drained
        for h in list(c.engine._prefetch.sessions):
            c.engine._prefetch.release(h)
        assert c.engine._prefetch.reserved_bytes == 0
        assert c.engine._prefetch_backend.pinned_bytes == 0
        c.engine._clear_cpu_cache()
        assert c.engine.cache_engine.cpu_cache_engine.mempool.num_free_blocks == 768

    c.engine._runtime.call(clear)
    for t in c.tensors:
        t.zero_()
    torch.cuda.synchronize()
    yield c

    def stop_sessions():
        for h in list(c.engine._prefetch.sessions):
            c.engine._prefetch.stop(h, "request_abort")

    c.engine._runtime.call(stop_sessions)
    until(lambda: c.engine._runtime.call(lambda: c.engine._prefetch.drained))
    c.engine._runtime.call(clear)


def start(c, **options):
    return c.manager.start_prefetch(c.tokens, PrefetchOptions(**options), c.namespace)


def snapshot(c, h):
    return c.manager.poll_prefetch([h])[h]


def verify_load(c, h, snap):
    assert snap.terminal and snap.inflight_chunks == snap.inflight_bytes == 0
    end = snap.reusable_prefix_end_token
    assert 0 <= end <= len(c.tokens) and end % 16 == 0
    task, mask = c.manager.get_match(c.tokens, cpu_only=True, namespace=c.namespace)
    assert mask[:end].all() and not mask[end:].any()
    # Handoff: GET holds its own references before releasing the prefetch lease.
    c.manager.release_prefetch(h)
    if end:
        c.manager.launch(task, c.slots[:end])
        response = c.manager.wait(task, timeout=60, completely=True)[task]
        assert response.status == KVResponseStatus.SUCCESS, response
    else:
        c.manager.cancel(task)
    torch.cuda.synchronize()
    blocks = end // 16
    for actual, expected in zip(c.tensors, c.expected):
        assert torch.equal(
            actual[:blocks], expected[:blocks]
        ), "GPU KV content mismatch"
        assert (
            torch.count_nonzero(actual[blocks:]).item() == 0
        ), "unreturned suffix overwritten"
    return end


@pytest.mark.parametrize("policy", ["wait_complete", "timeout", "best_effort"])
@pytest.mark.parametrize("window,chunk", [(1, 8), (2, 8), (2, 32), (4, 8), (2, 128)])
def test_real_full_roundtrip(clean, policy, window, chunk):
    c = clean
    before = time.monotonic()
    h = start(
        c,
        policy=policy,
        max_inflight_chunks=window,
        chunk_max_blocks=chunk,
        timeout_budget_s=30,
    )
    snap = c.manager.wait_prefetch(h, 60)
    record(
        "full",
        snap,
        policy=policy,
        window=window,
        chunk=chunk,
        elapsed_ms=(time.monotonic() - before) * 1000,
    )
    assert snap.loaded_tokens == len(c.tokens)
    # 512 KiB/block: the 32 MiB byte cap also limits a 128-block option to 64.
    assert snap.submitted_chunks == 512 // min(chunk, 64)
    assert verify_load(c, h, snap) == len(c.tokens)


@pytest.mark.parametrize("reason", ["demand", "request_abort"])
@pytest.mark.parametrize("window", [1, 2])
def test_real_stop_with_inflight(clean, reason, window):
    c = clean
    h = start(c, policy="best_effort", chunk_max_blocks=8, max_inflight_chunks=window)
    before = until(lambda: (s if (s := snapshot(c, h)).inflight_chunks else None))
    stop_started = time.monotonic()
    if reason == "demand":
        sealed = c.manager.notify_prefetch_demand([h])[h]
    else:
        sealed = c.manager.stop_prefetch(h, reason)
    snap = c.manager.wait_prefetch(h, 30)
    record(
        "stop_inflight",
        snap,
        before=asdict(before),
        sealed=asdict(sealed),
        window=window,
        stop_to_terminal_ms=(time.monotonic() - stop_started) * 1000,
    )
    assert snap.stop_reason == reason
    assert snap.submitted_chunks == snap.sealed_submit_seq == sealed.submitted_chunks
    assert 0 < snap.loaded_tokens < len(c.tokens)
    verify_load(c, h, snap)


@pytest.mark.parametrize("budget", [0, 0.05])
def test_real_timeout(clean, budget):
    c = clean
    before = time.monotonic()
    h = start(c, policy="timeout", chunk_max_blocks=1, timeout_budget_s=budget)
    snap = c.manager.wait_prefetch(h, 30)
    record(
        "deadline", snap, budget_s=budget, elapsed_ms=(time.monotonic() - before) * 1000
    )
    assert snap.stop_reason == "deadline"
    assert snap.submitted_chunks == snap.sealed_submit_seq
    if budget == 0:
        assert snap.submitted_chunks == snap.loaded_tokens == 0
    else:
        assert 0 < snap.loaded_tokens < len(c.tokens)
    verify_load(c, h, snap)


def test_real_overlapping_sessions(clean):
    c = clean
    handles = [start(c, chunk_max_blocks=8) for _ in range(4)]
    snapshots = [c.manager.wait_prefetch(h, 60) for h in handles]
    for s in snapshots:
        record("overlapping", s)
        assert s.reusable_prefix_end_token == len(c.tokens)
    for h in handles[1:]:
        c.manager.release_prefetch(h)
    verify_load(c, handles[0], snapshots[0])


def test_real_foreground_get_during_prefetch(clean):
    c = clean
    h = start(c, chunk_max_blocks=1)
    until(lambda: snapshot(c, h).reusable_prefix_end_token >= 32)
    task, mask = c.manager.get_match(c.tokens, cpu_only=True, namespace=c.namespace)
    end = int(mask.sum())
    assert 0 < end < len(c.tokens)
    c.manager.launch(task, c.slots[:end])
    assert (
        c.manager.wait(task, timeout=30, completely=True)[task].status
        == KVResponseStatus.SUCCESS
    )
    torch.cuda.synchronize()
    for actual, expected in zip(c.tensors, c.expected):
        assert torch.equal(actual[: end // 16], expected[: end // 16])
    snap = c.manager.wait_prefetch(h, 60)
    record("foreground_overlap", snap, foreground_tokens=end)
    verify_load(c, h, snap)


def test_real_reset_drains_and_changes_epoch(clean):
    c = clean
    h = start(c, chunk_max_blocks=1)
    until(lambda: snapshot(c, h).inflight_chunks > 0)

    def reset():
        try:
            c.manager.reset()
            return True
        except RuntimeError as exc:
            assert "draining" in str(exc)
            return False

    until(reset)
    with pytest.raises((ValueError, KeyError), match="epoch|stale|unknown"):
        snapshot(c, h)
    new = start(c, policy="timeout", timeout_budget_s=0)
    snap = c.manager.wait_prefetch(new)
    assert new.epoch != h.epoch
    record("reset_epoch", snap)
    c.manager.release_prefetch(new)


def test_real_key_disappears_after_metadata_query(clean, monkeypatch):
    """Delete one owned key after matching; the unchanged RDMA worker sees failure.

    This is a control-plane failure injection. Metadata, allocation, transfer,
    success bitmap, publication, suffix recycling and H2D all remain real.
    """
    from flexkv.external.mooncake_store_keys import PoolKind

    c = clean
    planner = c.engine._prefetch_backend
    query = planner._query

    def query_then_delete(context):
        seq, matched = query(context)
        assert matched == 512
        key = c.engine.cache_engine.remote_cache_engine._build_pool_key(
            str(seq.block_hashes[12]), PoolKind.KV
        )
        # Fault injection deliberately revokes the lease on this owned key.
        assert planner._query_client._store.remove(key, True) == 0
        return seq, matched

    monkeypatch.setattr(planner, "_query", query_then_delete)
    h = start(c, chunk_max_blocks=8, max_inflight_chunks=2)
    snap = c.manager.wait_prefetch(h, 30)
    record("real_missing_key_after_query", snap)
    assert snap.stop_reason == "partial_read"
    assert snap.loaded_tokens == 12 * 16
    assert snap.submitted_chunks == snap.sealed_submit_seq
    assert verify_load(c, h, snap) == 12 * 16
