"""Real Python/C++ radix, native hash, physical CPU slots and completion payloads.

Only the remote network/data worker is substituted by a deterministic byte store;
this suite does not claim a GPU/Mooncake end-to-end run.
"""

from concurrent.futures import Future
from types import SimpleNamespace as NS
import threading
import time

import numpy as np
import pytest

pytest.importorskip("flexkv.c_ext")
from flexkv.cache.cache_engine import CacheEngine, CacheEngineAccel, GlobalCacheEngine
from flexkv.common.block import SequenceMeta
from flexkv.common.config import CacheConfig, ModelConfig, LayerGroupSpec
from flexkv.common.transfer import CompletedOp, DeviceType
from flexkv.prefetch.coordinator import PrefetchCoordinator
from flexkv.prefetch.planner import MooncakeChunkPlanner
from flexkv.prefetch.types import PrefetchOptions


@pytest.fixture(
    params=[CacheEngine, CacheEngineAccel], ids=["python_radix", "cpp_radix"]
)
def env(request):
    cpu = request.param(DeviceType.CPU, 64, 4, 0.1)
    cache = GlobalCacheEngine.__new__(GlobalCacheEngine)
    cache.cache_engines = {DeviceType.CPU: cpu}
    cache.cpu_cache_engine = cpu
    cache._cache_tree_lock = threading.RLock()
    cache.remote_cache_engine = NS(_build_pool_key=lambda h, kind: h)
    submitted = []
    engine = NS(
        cache_engine=cache,
        cache_config=CacheConfig(tokens_per_block=4),
        model_config=ModelConfig(num_layers=1, num_kv_heads=2, head_size=2),
        transfer_handles=[NS(submit_batch=lambda graphs: submitted.extend(graphs))],
    )
    planner = MooncakeChunkPlanner(engine, 1 << 20)
    planner._query_client = NS(batch_exists=lambda keys: len(keys))
    coordinator = PrefetchCoordinator(planner, max_reserved_bytes=1 << 20)
    engine._prefetch = coordinator
    result = NS(
        cpu=cpu,
        engine=engine,
        planner=planner,
        c=coordinator,
        sent=submitted,
        tokens=np.arange(40, dtype=np.int64),
        data=np.zeros((64, 16), dtype=np.uint8),
    )
    yield result
    for h in list(coordinator.sessions):
        coordinator.release(h)
    # Complete outstanding operations to exercise release after abort.
    for graph in submitted:
        if graph.graph_id in coordinator.graphs:
            finish(result, graph)
    coordinator.tick()
    planner.shutdown()
    assert planner.pinned_bytes == 0
    assert coordinator.reserved_bytes == 0


def start(env, **options):
    h = env.c.start(env.tokens, PrefetchOptions(chunk_max_blocks=2, **options))
    env.c.sessions[h].context.future.result(timeout=2)
    env.c.tick()
    return h


def finish(env, graph, bitmap=None, failed=False, missing=False):
    op = next(iter(graph._op_map.values()))
    bitmap = bitmap if bitmap is not None else (True,) * len(op.dst_block_ids)
    # Remote values keyed by the full prefix hash; place in actual allocated slots.
    for hash_value, dst, ok in zip(
        op.mooncake_store_block_hashes, op.dst_block_ids, bitmap
    ):
        if ok:
            env.data[dst] = int(hash_value) & 255
    if not missing:
        env.c.on_completion(
            CompletedOp(
                graph.graph_id, op.op_id, num_blocks=len(bitmap), block_results=bitmap
            )
        )
    env.c.on_completion(CompletedOp(graph.graph_id, -1, failed=failed))


@pytest.mark.parametrize("block_bytes", [64, 8 * 1024 * 1024])
def test_chunk_size_uses_blocks_across_model_geometries(env, block_bytes):
    # A larger model must not silently shrink an eight-block chunk to 32 MiB.
    env.planner.block_bytes = block_bytes
    env.planner.max_pinned_bytes = env.c.max_reserved_bytes = 1 << 30
    handle = env.c.start(env.tokens, PrefetchOptions(chunk_max_blocks=8))
    env.c.sessions[handle].context.future.result(timeout=2)
    env.c.tick()
    chunk = next(iter(env.c.sessions[handle].chunks.values()))
    assert chunk.end - chunk.begin == 8 * env.planner.block_size
    assert chunk.nbytes == 8 * block_bytes


@pytest.mark.parametrize(
    "reserved_blocks,pinned_blocks,expected", [(3, 64, 3), (64, 2, 2)]
)
def test_byte_budgets_still_bound_block_sized_chunks(
    env, reserved_blocks, pinned_blocks, expected
):
    env.c.max_reserved_bytes = reserved_blocks * env.planner.block_bytes
    env.planner.max_pinned_bytes = pinned_blocks * env.planner.block_bytes
    handle = env.c.start(env.tokens, PrefetchOptions(chunk_max_blocks=8))
    env.c.sessions[handle].context.future.result(timeout=2)
    env.c.tick()
    chunk = next(iter(env.c.sessions[handle].chunks.values()))
    assert chunk.end - chunk.begin == expected * env.planner.block_size
    assert env.c.reserved_bytes == expected * env.planner.block_bytes


def test_real_data_chunk_hash_offsets_and_invisible_staging(env):
    h = start(env)
    env.c.tick()
    assert len(env.sent) == 2
    seq = SequenceMeta(env.tokens, 4)
    assert env.cpu.match(seq).num_ready_matched_blocks == 0
    np.testing.assert_array_equal(
        next(iter(env.sent[1]._op_map.values())).mooncake_store_block_hashes,
        seq.block_hashes[2:4],
    )
    finish(env, env.sent[1])
    env.c.tick()
    assert env.cpu.match(seq).num_ready_matched_blocks == 0
    finish(env, env.sent[0])
    for _ in range(20):
        env.c.tick()
        for graph in list(env.sent):
            if (
                graph.graph_id in env.c.graphs
                and not env.c.sessions[h].chunks[graph.graph_id].done
            ):
                finish(env, graph)
        if env.c.snapshot(h).terminal:
            break
    snapshot = env.c.snapshot(h)
    assert snapshot.terminal and snapshot.loaded_tokens == 40
    match = env.cpu.match(seq)
    assert match.num_ready_matched_blocks == 10
    expected = np.repeat(
        (seq.block_hashes.astype(np.uint64) & 255)[:, None], 16, axis=1
    )
    np.testing.assert_array_equal(env.data[match.physical_blocks], expected)
    assert env.cpu.mempool.num_used_blocks == 10


@pytest.mark.parametrize(
    "bitmap,missing,failed,expected",
    [
        ((True, False), False, False, 4),
        ((False, True), False, False, 0),
        ((True,), False, False, 0),
        ((True, True), True, False, 0),
        ((True, True), False, True, 0),
    ],
)
def test_partial_result_and_out_of_order_tail_recycle(
    env, bitmap, missing, failed, expected
):
    h = start(env)
    env.c.tick()
    finish(env, env.sent[1])
    finish(env, env.sent[0], bitmap, failed, missing)
    env.c.tick()
    s = env.c.snapshot(h)
    assert s.terminal and s.loaded_tokens == expected
    assert env.cpu.mempool.num_used_blocks == expected // 4
    assert (
        env.cpu.match(SequenceMeta(env.tokens, 4)).num_ready_matched_blocks
        == expected // 4
    )


@pytest.mark.parametrize("policy", ["timeout", "best_effort"])
def test_stop_drains_claimed_chunks_without_issuing_next(env, policy):
    h = start(env, policy=policy, timeout_budget_s=10)
    env.c.tick()
    if policy == "timeout":
        env.c.clock = lambda: env.c.sessions[h].deadline + 1
    else:
        env.c.demand([h])
    env.c.tick()
    assert not env.c.snapshot(h).terminal
    assert len(env.sent) == 2
    finish(env, env.sent[0])
    env.c.tick()
    assert not env.c.snapshot(h).terminal
    finish(env, env.sent[1])
    env.c.tick()
    assert env.c.snapshot(h).terminal
    assert env.c.snapshot(h).loaded_tokens == 16
    assert len(env.sent) == 2


def test_initial_local_prefix_and_concurrent_publication(env):
    seq = SequenceMeta(env.tokens, 4)
    env.cpu.insert(seq, env.cpu.take(2), num_insert_blocks=2)
    h = start(env)
    assert next(iter(env.sent[0]._op_map.values())).src_block_ids.tolist() == [2, 3]
    # A foreground writer publishes a longer prefix while REMOTE2H is inflight.
    env.cpu.insert(seq, env.cpu.take(8), num_insert_blocks=10)
    finish(env, env.sent[0])
    env.c.stop(h, "demand")
    env.c.tick()
    assert env.c.snapshot(h).l3_loaded_spans == ((8, 16),)
    assert env.cpu.mempool.num_used_blocks == 10  # duplicate detached slots recycled
    assert env.planner.pinned_bytes == 10 * env.planner.block_bytes
    env.c.release(h)
    assert env.planner.pinned_bytes == 0


def test_initial_prefix_pin_capacity_stops_without_poisoning_next_session(env):
    seq = SequenceMeta(env.tokens, 4)
    env.cpu.insert(seq, env.cpu.take(10), num_insert_blocks=10)
    env.planner.max_pinned_bytes = 4 * env.planner.block_bytes
    handle = start(env)
    snapshot = env.c.snapshot(handle)
    assert snapshot.terminal and snapshot.stop_reason == "capacity"
    assert snapshot.error is None and snapshot.loaded_tokens == 0
    assert not env.sent and env.planner.pinned_bytes == 0
    assert env.cpu.mempool.num_used_blocks == 10

    env.c.release(handle)
    env.planner.max_pinned_bytes = 10 * env.planner.block_bytes
    retry = start(env)
    assert env.c.snapshot(retry).terminal
    assert env.c.snapshot(retry).reusable_prefix_end_token == len(env.tokens)
    assert env.planner.pinned_bytes == 10 * env.planner.block_bytes


def test_concurrent_long_node_does_not_overrun_pin_budget(env):
    h = start(env)
    env.planner.max_pinned_bytes = 4 * env.planner.block_bytes
    seq = SequenceMeta(env.tokens, 4)
    env.cpu.insert(seq, env.cpu.take(10), num_insert_blocks=10)
    finish(env, env.sent[0])
    env.c.tick()
    assert env.c.snapshot(h).terminal
    assert env.c.snapshot(h).loaded_tokens == 0
    assert env.planner.pinned_bytes <= env.planner.max_pinned_bytes
    assert env.cpu.mempool.num_used_blocks == 10


def test_reset_discards_inflight_without_publishing(env):
    h = start(env)
    env.c.stop_all("reset")
    assert env.cpu.mempool.num_used_blocks == 2
    finish(env, env.sent[0])
    env.c.tick()
    assert env.cpu.mempool.num_used_blocks == 0
    assert env.c.snapshot(h).outcome == "aborted"


def test_namespace_changes_full_chain(env):
    h1 = env.c.start(env.tokens, PrefetchOptions(), namespace=["tenant-A"])
    h2 = env.c.start(env.tokens, PrefetchOptions(), namespace=["tenant-B"])
    a = env.c.sessions[h1].context.future.result(timeout=2)[0]
    b = env.c.sessions[h2].context.future.result(timeout=2)[0]
    assert not np.any(a.block_hashes == b.block_hashes)


def test_metadata_query_is_outside_cache_lock_and_stops_between_batches(env):
    entered, proceed = threading.Event(), threading.Event()
    calls = []

    def query(keys):
        assert not env.engine.cache_engine._cache_tree_lock._is_owned()
        calls.append(len(keys))
        entered.set()
        assert proceed.wait(2)
        return len(keys)

    env.planner._query_client = NS(batch_exists=query)
    h = env.c.start(
        np.arange(3000, dtype=np.int64), PrefetchOptions(policy="best_effort")
    )
    assert entered.wait(2)
    env.c.demand([h])
    env.c.tick()
    assert env.c.snapshot(h).terminal
    proceed.set()
    env.c.sessions[h].context.future.result(timeout=2)
    assert calls == [256]
    assert env.sent == []


def _cpu_pipe_worker(command, result, shared, allow):
    """Independent CPU byte worker using the unchanged graph/result wire format."""
    data = np.frombuffer(shared, dtype=np.uint8).reshape(64, 16)
    while True:
        message = command.recv()
        if message["type"] == "shutdown":
            break
        assert message["type"] == "submit_batch"
        for graph in message["transfer_graphs"]:
            assert allow.wait(10)
            op = next(iter(graph._op_map.values()))
            for h, dst in zip(op.mooncake_store_block_hashes, op.dst_block_ids):
                data[dst] = int(h) & 255
            result.send(
                [
                    CompletedOp(
                        graph.graph_id,
                        op.op_id,
                        num_blocks=len(op.dst_block_ids),
                        block_results=(True,) * len(op.dst_block_ids),
                    ),
                    CompletedOp.completed_graph(graph.graph_id),
                ]
            )
    command.close()
    result.close()


def test_actual_graph_ipc_runtime_drain_and_cpu_bytes_without_polling(env):
    import multiprocessing as mp
    from flexkv.kvtask import KVTaskEngine
    from flexkv.transfer_manager import (
        TransferManagerInterProcessHandle,
        TransferManagerHandle,
    )
    from flexkv.prefetch.runtime import TaskRuntime, QueuedTransferHandle

    ctx = mp.get_context("spawn")
    commands, worker_commands = ctx.Pipe()
    results, worker_results = ctx.Pipe()
    shared = ctx.RawArray("B", 64 * 16)
    allow = ctx.Event()
    worker = ctx.Process(
        target=_cpu_pipe_worker, args=(worker_commands, worker_results, shared, allow)
    )
    worker.start()
    ipc = TransferManagerInterProcessHandle.__new__(TransferManagerInterProcessHandle)
    ipc.command_parent_conn, ipc.result_parent_conn, ipc.process = (
        commands,
        results,
        worker,
    )
    handle = TransferManagerHandle.__new__(TransferManagerHandle)
    handle._handle = ipc
    queued = QueuedTransferHandle(handle)
    engine = KVTaskEngine.__new__(KVTaskEngine)
    engine.model_config, engine.cache_config = (
        env.engine.model_config,
        env.engine.cache_config,
    )
    engine.cache_engine, engine._prefetch = env.engine.cache_engine, env.c
    engine._prefetch_backend, engine._prefetch_options = env.planner, PrefetchOptions()
    engine.tasks, engine.graph_to_task = {}, {}
    engine.uncompleted_ops, engine.uncompleted_op_results, engine.uncompleted_graphs = (
        {},
        {},
        {},
    )
    engine.required_completed_count, engine.transfer_handles = 1, [queued]
    env.planner.engine = engine
    engine._runtime = TaskRuntime(engine)
    engine._runtime.start()
    try:
        h = engine.start_prefetch(
            env.tokens, PrefetchOptions(policy="best_effort", chunk_max_blocks=2)
        )
        deadline = time.monotonic() + 10
        while engine.progress_prefetch([h])[h].submitted_chunks < 2:
            assert time.monotonic() < deadline
            time.sleep(0.002)
        stopped = engine.progress_prefetch([h], demand_handles=[h])[h]
        assert not stopped.terminal and stopped.inflight_chunks == 2
        assert env.cpu.mempool.num_used_blocks == 4
        allow.set()
        # No request-side progress calls while bytes and completion messages flow.
        # Use only the runtime's condition and a diagnostic owner snapshot.
        while not engine._runtime.call(lambda: env.c.snapshot(h).terminal):
            assert time.monotonic() < deadline
            with engine._runtime.changed:
                engine._runtime.changed.wait(0.02)
        snapshot = engine.progress_prefetch([h])[h]
        assert snapshot.loaded_tokens == 16 and snapshot.submitted_chunks == 2
        sequence = SequenceMeta(env.tokens, 4)
        match = engine._runtime.call(env.cpu.match, sequence)
        actual = np.frombuffer(shared, dtype=np.uint8).reshape(64, 16)[
            match.physical_blocks
        ]
        expected = np.repeat(
            (sequence.block_hashes[:4].astype(np.uint64) & 255)[:, None], 16, axis=1
        )
        np.testing.assert_array_equal(actual, expected)
        engine.release_prefetch(h)
        assert env.planner.pinned_bytes == 0
    finally:
        allow.set()
        engine.shutdown()
    assert worker.exitcode == 0
    assert not engine.transfer_handles[0].thread.is_alive()


def test_compressed_group_reservations_use_exact_block_bytes(env):
    import torch

    env.engine.model_config = ModelConfig(
        num_layers=1,
        num_kv_heads=2,
        kv_dim=1,
        dtype=torch.uint8,
        layer_groups=[
            LayerGroupSpec(1, 1, 3, [0], dtype=torch.uint8, compress_ratio=4)
        ],
    )
    planner = MooncakeChunkPlanner(env.engine, 1024)
    try:
        # 3 bytes per 4-token block. Rounded per-token metrics are zero.
        assert env.engine.model_config.token_size_in_bytes == 0
        assert planner.block_bytes == 3
    finally:
        planner.shutdown()


@pytest.fixture
def swa_env(env):
    from flexkv.cache.swa_cache_engine import SWAOpConstructor
    from flexkv.common.config import SWAPoolConfig

    if not isinstance(env.cpu, CacheEngineAccel):
        pytest.skip("SWA node-mounted snapshots require the C++ radix engine")
    config = SWAPoolConfig(
        enabled=True,
        num_slots=8,
        num_swa_layers=1,
        bytes_per_token_per_layer=16,
        pin_memory=False,
    )
    env.cpu.init_swa(config)
    env.engine.cache_config.swa = config
    env.engine.cache_config.enable_swa_transfer = True
    env.engine.cache_engine.cache_config = env.engine.cache_config
    env.engine.cache_engine.swa_op_constructor = SWAOpConstructor(
        env.engine.cache_engine
    )
    env.planner.swa_slot_bytes = 64
    env.engine.cache_engine.remote_cache_engine._build_pool_key = (
        lambda h, kind: f"{kind.name}:{h}"
    )
    seq = SequenceMeta(env.tokens, 4)
    env.checkpoints = {str(seq.block_hashes[i]) for i in (3, 7)}
    env.planner._query_client = NS(
        batch_exists_impl=lambda keys: [
            int(key.startswith("KV:") or key.split(":")[1] in env.checkpoints)
            for key in keys
        ]
    )
    return env


def finish_swa(
    env, graph, *, full=None, snapshot=True, failed=False, omit_snapshot=False
):
    for op in graph._op_map.values():
        if op.is_swa and omit_snapshot:
            continue
        bitmap = (
            (snapshot,)
            if op.is_swa
            else full if full is not None else (True,) * len(op.dst_block_ids)
        )
        env.c.on_completion(
            CompletedOp(
                graph.graph_id, op.op_id, num_blocks=len(bitmap), block_results=bitmap
            )
        )
    env.c.on_completion(CompletedOp(graph.graph_id, -1, failed=failed))


def test_swa_requires_exact_resident_checkpoint(swa_env):
    env = swa_env
    h = start(env, swa_aware=True)
    env.c.tick()
    assert env.c.sessions[h].target == 32  # Full KV exists to 40; last SWA is at 32.
    finish_swa(env, env.sent[0])
    env.c.tick()
    assert env.c.sessions[h].committed == 8
    assert env.c.snapshot(h).loaded_tokens == 0
    finish_swa(env, env.sent[1])
    env.c.tick()
    assert env.c.snapshot(h).loaded_tokens == 16
    while not env.c.snapshot(h).terminal:
        for graph in list(env.sent):
            if (
                graph.graph_id in env.c.graphs
                and not env.c.sessions[h].chunks[graph.graph_id].done
            ):
                finish_swa(env, graph)
        env.c.tick()
    assert env.c.snapshot(h).loaded_tokens == 32
    assert env.cpu.match(SequenceMeta(env.tokens, 4)).swa_hit_blocks == 8
    assert env.cpu.swa_pool.num_free == 6


@pytest.mark.parametrize(
    "snapshot,omit,failed",
    [(False, False, False), (True, True, False), (True, False, True)],
)
def test_swa_failed_snapshot_keeps_previous_checkpoint(swa_env, snapshot, omit, failed):
    env = swa_env
    h = start(env, swa_aware=True)
    env.c.tick()
    finish_swa(env, env.sent[0])
    finish_swa(env, env.sent[1])
    env.c.tick()
    env.c.tick()
    env.c.stop(h, "demand")
    for graph in list(env.sent)[2:]:
        finish_swa(env, graph, snapshot=snapshot, omit_snapshot=omit, failed=failed)
    env.c.tick()
    s = env.c.snapshot(h)
    assert s.terminal and s.loaded_tokens == 16
    assert s.l3_loaded_spans == ((0, 16),)
    assert env.cpu.swa_pool.num_free == 7


def test_swa_stop_between_checkpoints_drains_full_but_reports_zero(swa_env):
    env = swa_env
    h = env.c.start(
        env.tokens,
        PrefetchOptions(swa_aware=True, chunk_max_blocks=2, max_inflight_chunks=1),
    )
    env.c.sessions[h].context.future.result(timeout=2)
    env.c.tick()
    env.c.stop(h, "demand")
    assert not env.c.snapshot(h).terminal
    finish_swa(env, env.sent[0])
    env.c.tick()
    assert env.c.snapshot(h).terminal and env.c.snapshot(h).loaded_tokens == 0
    assert env.cpu.mempool.num_used_blocks == 2
    assert env.cpu.swa_pool.num_free == 8


def test_swa_abort_releases_snapshot_pins_after_drain(swa_env):
    env = swa_env
    h = start(env, swa_aware=True)
    env.c.tick()
    env.c.release(h)
    for graph in env.sent:
        finish_swa(env, graph)
    env.c.tick()
    assert env.c.snapshot(h).terminal
    assert env.planner.pinned_bytes == 0
    # Abort may retain completed cache data, but it must be evictable after drain.
    env.cpu._evict_swa_slots(8)
    assert env.cpu.swa_pool.num_free == 8


def test_swa_reset_discards_all_inflight_staging(swa_env):
    env = swa_env
    h = start(env, swa_aware=True)
    env.c.tick()
    env.c.stop_all("reset")
    for graph in env.sent:
        finish_swa(env, graph)
    env.c.tick()
    assert env.c.snapshot(h).terminal
    assert env.cpu.swa_pool.num_free == 8
    assert env.cpu.mempool.num_used_blocks == 0
    assert env.planner.pinned_bytes == 0


@pytest.mark.parametrize("bitmap", [(), (True, True)])
def test_swa_malformed_completion_cannot_publish_checkpoint(swa_env, bitmap):
    env = swa_env
    h = start(env, swa_aware=True)
    env.c.tick()
    env.c.stop(h, "demand")
    finish_swa(env, env.sent[0])
    finish_swa(env, env.sent[1], snapshot=False)
    swa_op = next(op for op in env.sent[1]._op_map.values() if op.is_swa)
    env.c.on_completion(
        CompletedOp(
            swa_op.graph_id, swa_op.op_id, num_blocks=len(bitmap), block_results=bitmap
        )
    )
    env.c.tick()
    assert env.c.snapshot(h).terminal
    assert env.c.snapshot(h).loaded_tokens == 0
    assert env.cpu.swa_pool.num_free == 8
