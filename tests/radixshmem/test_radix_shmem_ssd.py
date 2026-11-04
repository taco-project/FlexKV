"""The radixshmem SSD tier: an index-only radix-server whose slot ids are
FlexKV's SSD blocks, next to the data-mode CPU server.

Drives `RadixShmemCacheEngine` on real in-process servers (standalone, so the
SSD matches are local; a peer's SSD run is forced through a synthetic match):
PUT writes through (D2H -> H2DISK, each tier published by its own writers),
GET falls back to the local SSD, PREFETCH stages the SSD tail into the CPU
tree, SWA rides along, a stale index is cleared at attach; plus the
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
