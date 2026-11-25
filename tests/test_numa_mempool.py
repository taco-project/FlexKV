from dataclasses import replace

import numpy as np
import pytest

from flexkv.cache.cache_engine import DEFAULT_CACHE_STRATEGY, GlobalCacheEngine
from flexkv.cache.mempool import Mempool
from flexkv.common import numa
from flexkv.common.block import SequenceMeta
from flexkv.common.config import CacheConfig, GLOBAL_CONFIG_FROM_ENV, ModelConfig
from flexkv.common.storage import KVCacheLayoutType
from flexkv.common.transfer import TransferType

pytestmark = pytest.mark.unit


def test_single_block_requests_interleave_below_half_occupancy():
    pool = Mempool(64, [(0, 32, 0), (32, 64, 1)])
    allocated = [pool.allocate_blocks(1).item() for _ in range(12)]
    assert allocated == [value for pair in zip(range(6), range(32, 38), strict=True) for value in pair]
    assert pool.num_used_blocks == 12


def test_balanced_batches_redistribute_a_region_shortfall():
    pool = Mempool(31, [(0, 10, 0), (10, 20, 1), (20, 30, 2), (30, 31, 3)])
    first = pool.allocate_blocks(10, preferred_node=99)
    assert np.bincount(np.minimum(first // 10, 3)).tolist() == [3, 3, 3, 1]
    second = pool.allocate_blocks(4)
    assert np.bincount(second // 10).tolist() == [1, 2, 1]


@pytest.mark.parametrize("preferred_node", [0, 1])
def test_preferred_node_spills_and_reuses_recycled_ids(preferred_node):
    pool = Mempool(16, [(0, 8, 0), (8, 16, 1)])
    local = pool.allocate_blocks(7, preferred_node=preferred_node)
    assert (local // 8 == preferred_node).all()
    spill = pool.allocate_blocks(4, preferred_node=preferred_node)
    assert (spill // 8).tolist() == [preferred_node] + [1 - preferred_node] * 3
    recycled = np.r_[local[1], local[4], spill[-1], local[1]]
    pool.recycle_blocks(recycled)
    assert pool.num_free_blocks == 8
    with pytest.raises(ValueError, match="already free"):
        pool.recycle_blocks(recycled)
    assert pool.allocate_blocks(2, preferred_node=preferred_node).tolist() == local[[1, 4]].tolist()
    assert set(pool.allocate_blocks(6, preferred_node=1 - preferred_node)) == (
        set(range((1 - preferred_node) * 8, (2 - preferred_node) * 8)) - set(spill[1:-1])
    )
    assert pool.num_free_blocks == 0
    pool.reset()
    assert pool.num_free_blocks == 16
    assert pool.allocate_blocks(1).item() == 0
    assert pool.allocate_blocks(1).item() == 8


def test_multiple_ranges_for_one_node_precede_spill():
    pool = Mempool(10, [(0, 2, 0), (2, 6, 1), (6, 8, 0), (8, 10, 2)])
    assert pool.allocate_blocks(5, preferred_node=0).tolist() == [0, 1, 6, 7, 2]
    assert pool.allocate_blocks(4, preferred_node=0).tolist() == [3, 4, 5, 8]
    assert pool.allocate_blocks(1, preferred_node=0).item() == 9


@pytest.mark.parametrize("ranges", [None, [], [(0, 8, 1)]])
def test_legacy_pool_keeps_ascending_allocation(ranges):
    pool = Mempool(8, ranges)
    assert pool.allocate_blocks(2, preferred_node=1).tolist() == [0, 1]
    pool.recycle_blocks(np.array([0], dtype=np.int64))
    assert pool.allocate_blocks(2).tolist() == [2, 3]
    assert pool.allocate_blocks(5).tolist() == [0, 4, 5, 6, 7]


@pytest.mark.parametrize("ranges", [[(0, 2, 0), (3, 8, 1)], [(0, 5, 0), (4, 8, 1)],
                                     [(0, 4, 0), (4, 7, 1)], [(0, 4, 0), (4, 9, 1)]])
def test_range_validation(ranges):
    with pytest.raises(ValueError, match="partition"):
        Mempool(8, ranges)


def test_block_ranges_follow_byte_plan_and_exclude_layerfirst():
    assert numa.plan_cpu_pool_numa_block_ranges(129, 4096, 8192, nodes=(0, 1)) == [
        (0, 64, 0), (64, 129, 1)]
    assert numa.plan_cpu_pool_numa_block_ranges(
        129, 4096, 8192, is_blockfirst=False, nodes=(0, 1)) == []
    assert numa.plan_cpu_pool_numa_block_ranges(0, 4096, 4096, nodes=(0, 1)) == []
    assert numa.plan_cpu_pool_numa_block_ranges(129, 0, 4096, nodes=(0, 1)) == []


@pytest.fixture
def gpu_env(monkeypatch):
    for name in ("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES",
                 "CUDA_DEVICE_ORDER", numa._ENV_CLIENT_CUDA_VISIBLE_DEVICES,
                 numa.ENV_CLIENT_NUMA_MAP):
        monkeypatch.delenv(name, raising=False)
    numa._gpu_numa_nodes.cache_clear()
    yield
    numa._gpu_numa_nodes.cache_clear()


@pytest.mark.parametrize("visibility", [
    "CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
    numa._ENV_CLIENT_CUDA_VISIBLE_DEVICES,
])
def test_gpu_visibility_and_tp_group(monkeypatch, gpu_env, visibility):
    monkeypatch.setattr(numa, "query_gpu_numa_node",
                        lambda gpu: gpu // 4 if 0 <= gpu < 8 else None)
    monkeypatch.setenv(visibility, "4,5,0,1")
    assert numa.resolve_client_numa_node(0, 2) == 1
    assert numa.resolve_client_numa_node(1, 2) == 0
    assert numa.resolve_client_numa_node(0, 4) is None
    assert numa.resolve_client_numa_node(2, 2) is None
    monkeypatch.setenv(visibility, "GPU-unsupported")
    assert numa.resolve_client_numa_node(0, 1) is None
    monkeypatch.setenv(numa.ENV_CLIENT_NUMA_MAP, "0:1,1:0")
    assert numa.resolve_client_numa_node(0, 1) == 1
    assert numa.resolve_client_numa_node(1, 1) == 0


def test_combined_rocr_and_hip_visibility(monkeypatch, gpu_env):
    monkeypatch.setattr(numa, "query_gpu_numa_node", {1: 0, 4: 1, 7: 1}.get)
    monkeypatch.setenv("ROCR_VISIBLE_DEVICES", "4,1,7")
    monkeypatch.setenv("HIP_VISIBLE_DEVICES", "2,0")
    assert numa.resolve_client_numa_node(0, 2) == 1


def test_gpu_numa_from_drm_and_unknown_node(tmp_path, monkeypatch, gpu_env):
    monkeypatch.setattr(numa, "_DRM_CLASS", tmp_path / "drm")
    monkeypatch.setattr(numa, "_NVIDIA_GPUS", tmp_path / "nvidia")
    for render_id, node in [(128, 1), (130, 0), (132, -1)]:
        device = tmp_path / "drm" / f"renderD{render_id}" / "device"
        device.mkdir(parents=True)
        (device / "vendor").write_text("0x1002")
        (device / "numa_node").write_text(str(node))
    assert numa.query_gpu_numa_node(0) == 1
    assert numa.query_gpu_numa_node(1) == 0
    assert numa.query_gpu_numa_node(2) is None
    assert numa.resolve_client_numa_node(0, 2) is None


def test_nvidia_numa_from_proc_pci_bus_id(tmp_path, monkeypatch, gpu_env):
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    for name, attr in [("drm", "_DRM_CLASS"), ("nvidia", "_NVIDIA_GPUS"), ("pci", "_PCI_DEVICES")]:
        monkeypatch.setattr(numa, attr, tmp_path / name)
    for bus, minor, node in [("0000:41:00.0", 0, 1), ("0000:01:00.0", 1, 0)]:
        info = tmp_path / "nvidia" / bus / "information"
        info.parent.mkdir(parents=True)
        info.write_text(f"Model: Fake GPU\nDevice Minor: {minor}\n")
        device = tmp_path / "pci" / bus
        device.mkdir(parents=True)
        (device / "numa_node").write_text(str(node))
    assert numa.query_gpu_numa_node(0) == 0
    assert numa.query_gpu_numa_node(1) == 1


@pytest.fixture
def numa_engine(monkeypatch):
    monkeypatch.setenv(numa.ENV_ENABLE_NUMA_SPREAD, "1")
    monkeypatch.setenv(numa.ENV_CLIENT_NUMA_MAP, "0:0,1:0,2:0,3:0,4:1,5:1,6:1,7:1")
    monkeypatch.setattr(numa, "get_spread_nodes", lambda: (0, 1))
    monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "cpu_layout_type", KVCacheLayoutType.BLOCKFIRST)

    def build(index_accel, **config):
        monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "index_accel", index_accel)
        return GlobalCacheEngine(
            CacheConfig(tokens_per_block=16, num_cpu_blocks=256, **config),
            ModelConfig(num_layers=1, head_size=64, dp_size=8),
        )

    return build


@pytest.mark.parametrize("index_accel", [False, True])
def test_global_put_places_dp8_blocks_locally(numa_engine, index_accel):
    engine = numa_engine(index_accel)
    for client in range(8):
        tokens = np.arange(16, dtype=np.int64) + 16 * client
        graph, *_ = engine.put(
            request_id=client, token_ids=tokens, token_mask=np.ones_like(tokens),
            slot_mapping=np.arange(16, dtype=np.int64), dp_client_id=client,
        )
        writes = [op for op in graph._op_map.values() if op.transfer_type == TransferType.D2H]
        assert len(writes) == 1
        assert (writes[0].dst_block_ids // 128).tolist() == [client // 4]


@pytest.mark.parametrize("index_accel", [False, True])
@pytest.mark.parametrize("prefetch", [False, True])
def test_global_get_and_prefetch_stage_on_requesting_node(numa_engine, index_accel, prefetch):
    engine = numa_engine(index_accel, enable_ssd=True, num_ssd_blocks=32)
    tokens = np.arange(16, dtype=np.int64)
    seq = SequenceMeta(token_ids=tokens, tokens_per_block=16)
    engine.ssd_cache_engine.insert(seq, engine.ssd_cache_engine.take(1))
    graph, *_ = engine.get(
        request_id=1, token_ids=tokens, token_mask=np.ones_like(tokens),
        slot_mapping=np.arange(16, dtype=np.int64), dp_client_id=5,
        temp_cache_strategy=replace(DEFAULT_CACHE_STRATEGY, ignore_gpu=prefetch),
    )
    reads = [op for op in graph._op_map.values() if op.transfer_type == TransferType.DISK2H]
    assert reads
    assert all((op.dst_block_ids >= 128).all() for op in reads)


def test_pool_confirmation_can_disable_planned_numa_after_allocator_fallback(numa_engine):
    engine = numa_engine(False)
    engine.confirm_cpu_numa_block_ranges(())
    assert engine._take_cpu_blocks(7, num_required_blocks=2).tolist() == [0, 1]


def test_transfer_manager_ranges_reach_cache_engine_before_ready(numa_engine):
    from flexkv.kvtask import KVTaskManager
    from flexkv.transfer_manager import TransferManagerInterProcessHandle

    engine = numa_engine(False)
    handle = TransferManagerInterProcessHandle(
        engine.model_config, engine.cache_config, "/tmp/unused")
    manager = object.__new__(KVTaskManager)
    manager.cache_engine = engine
    manager.transfer_handles = [handle]
    try:
        assert not manager.is_ready()
        # Simulate the worker reporting an adjusted split after mapping the pool.
        actual_ranges = ((0, 96, 0), (96, 256, 1))
        handle._numa_ranges_writer.send(actual_ranges)
        handle.ready_event.set()
        assert manager.is_ready()
        assert engine._take_cpu_blocks(5, num_required_blocks=1).item() == 96
        assert manager.is_ready()
    finally:
        # Only the readiness IPC is exercised. The transfer process is absent.
        handle.shutdown()
        manager.transfer_handles = []
