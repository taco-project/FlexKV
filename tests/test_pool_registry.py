"""A pool is a *value* keying the storage registry, the op and the worker.

Before the refactor SWA was a parallel set of maps and branches selected by an
``is_swa`` bool. These cases pin the three places where collapsing that into a
``PoolId`` key can go wrong *silently* -- each one would write the right bytes
into the wrong pool rather than raise:

* the storage key must separate two pools sharing one physical ``device_id``,
  which is exactly what ``(DeviceType, device_id)`` could not express;
* the pool must survive the worker pipe as the interned member, because the
  worker spells the selection ``op.pool_id is PoolId.SWA``;
* on the worker, region indices and per-layer milestones are integers that
  cross into cpp, and an SWA op arriving at a worker with no SWA pool must not
  fall through to the full-KV regions -- the block ids index a different slot
  space, so it would corrupt main KV.
"""
import pickle

import numpy as np
import pytest
import torch

from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.common.pool import PoolEndpoint, PoolId
from flexkv.common.transfer import DeviceType, TransferOp, TransferType
from flexkv.storage.storage_engine import StorageEngine
from flexkv.transfer.worker_op import WorkerTransferOp
from flexkv.transfer.workers.gpu_cpu import GPUCPUTransferWorker, _Pool

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# The storage registry key
# --------------------------------------------------------------------------

def _engine():
    """A StorageEngine with no tiers enabled: __init__ allocates nothing, so
    every key in the registry below was put there by the test."""
    model_config = ModelConfig(
        num_layers=2, num_kv_heads=1, head_size=8, kv_dim=1,
        dtype=torch.uint8, tp_size=1, pp_size=1, dp_size=1, cp_size=1,
    )
    cache_config = CacheConfig(
        tokens_per_block=16,
        enable_cpu=False, enable_ssd=False, enable_3rd_remote=False,
    )
    return StorageEngine(model_config, cache_config, num_layers_per_pp_stage=2)


def test_two_pools_coexist_on_one_device_id():
    """The case that forced a second dict, expressed as one key each.

    GPU 0 holds an independent main-KV and SWA handle; a key without the pool
    collides here and hands out whichever was written last.
    """
    engine = _engine()
    main, swa = object(), object()
    engine._handles[(PoolEndpoint(PoolId.FULL_KV, DeviceType.GPU), 0)] = main
    engine._handles[(PoolEndpoint(PoolId.SWA, DeviceType.GPU), 0)] = swa

    assert engine.get_storage_handle(DeviceType.GPU, 0) is main
    assert engine.get_storage_handle(
        DeviceType.GPU, 0, pool_id=PoolId.SWA) is swa
    assert engine.get_swa_storage_handle(0) is swa
    assert not engine.has_storage_handle(DeviceType.GPU, 1, pool_id=PoolId.SWA)


# --------------------------------------------------------------------------
# The pool crosses the worker pipe
# --------------------------------------------------------------------------

def test_worker_op_carries_the_pool_across_the_pipe_as_the_member():
    """``op.pool_id is PoolId.SWA`` is how the worker selects its pool, so the
    hand-rolled int encoding must decode back to the interned member.

    The bare int on the wire is not decoration: an IntEnum member costs ~6x a
    bare int to pickle because it inherits ``Enum.__reduce_ex__``.
    """
    ids = np.array([0], dtype=np.int64)
    for pid in PoolId:
        op = TransferOp(graph_id=1, transfer_type=TransferType.D2H,
                        src_block_ids=ids, dst_block_ids=ids, pool_id=pid)
        wop = WorkerTransferOp(op)
        assert type(wop.__getstate__()["pool_id"]) is int

        round_tripped = pickle.loads(pickle.dumps(wop))
        assert round_tripped.pool_id is pid
        assert round_tripped.is_swa is pid.is_swa


# --------------------------------------------------------------------------
# The worker's pools, as a registry keyed by PoolId
# --------------------------------------------------------------------------

def _worker(*pool_ids, num_layers=2, regions_per_pool=1):
    """A worker holding only the fields the methods under test read.

    Built with ``object.__new__``: everything here is index bookkeeping over
    ``_Pool`` records, and a real one needs CUDA IPC handles from a live
    producer process.
    """
    w = object.__new__(GPUCPUTransferWorker)
    w.worker_id = 0
    w._num_original_layers = num_layers
    w._pools = {}
    for pid in pool_ids:
        w._pools[pid] = _Pool(
            pool_id=pid,
            name=pid.name.lower(),
            regions=[object() for _ in range(regions_per_pool)],
            # Every original layer lives in region ordinal 0 at local layer L.
            layer_members=[[(0, layer)] for layer in range(num_layers)],
            bytes_per_block=1024,
        )
    return w


class _Op:
    def __init__(self, pool_id):
        self.pool_id = pool_id
        self.is_swa = pool_id is PoolId.SWA


def test_an_unserved_pool_raises_rather_than_falling_through():
    """The dangerous failure mode: an SWA op reaching a worker with no SWA
    pool must not quietly transfer against the full-KV regions."""
    w = _worker(PoolId.FULL_KV)
    assert w._pool_for(_Op(PoolId.FULL_KV)).pool_id is PoolId.FULL_KV
    with pytest.raises(RuntimeError, match="no SWA pool registered"):
        w._pool_for(_Op(PoolId.SWA))


def test_region_indices_partition_one_index_space_in_pool_order():
    """What ``_build_region_batch`` does, isolated from the cpp build.

    Each pool gets a contiguous slice; the slices are disjoint and cover
    ``range(total)``, with FULL_KV first regardless of insertion order --
    these integers cross into cpp, so insertion order drifting would silently
    renumber every region. Registered SWA-first here, so a walk over the dict
    itself would yield the wrong order.
    """
    w = _worker(PoolId.SWA, PoolId.FULL_KV, regions_per_pool=3)
    specs = []
    for pool in w._ordered_pools():
        pool.region_indices = list(
            range(len(specs), len(specs) + len(pool.regions)))
        specs.extend(pool.regions)

    assert list(w._pools) == [PoolId.SWA, PoolId.FULL_KV]
    assert w._pools[PoolId.FULL_KV].region_indices == [0, 1, 2]
    assert w._pools[PoolId.SWA].region_indices == [3, 4, 5]
    assert len(specs) == 6


def test_milestones_carry_the_pool_and_the_global_region_index():
    """The two pools' block ids come from different tensors, so a milestone
    that named the wrong pool would have the request read the wrong one."""
    w = _worker(PoolId.FULL_KV, PoolId.SWA, num_layers=2)
    w._pools[PoolId.FULL_KV].region_indices = [0]
    w._pools[PoolId.SWA].region_indices = [1]

    milestones = w._build_layer_milestones()

    assert len(milestones) == 2
    for layer, members in enumerate(milestones):
        # One entry per pool per layer, full KV first (PoolId order), each
        # naming the pool's *global* region index.
        assert members == [(PoolId.FULL_KV, 0, layer),
                           (PoolId.SWA, 1, layer)]
