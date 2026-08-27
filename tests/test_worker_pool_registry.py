"""GPUCPUTransferWorker's pools, as a registry keyed by PoolId.

The refactor's central claim on the worker side is that SWA is a *pool on this
worker*, not a second worker: same GPUs, same direction, same regions batch,
with only the slot-id space differing. That makes three things load-bearing,
and each is pinned here:

* an op selects its pool by ``op.pool_id``, and a pool this worker does not
  serve is an error rather than a silent fall-through to full KV -- falling
  through would write SWA block ids into main-KV slots;
* the pools' region indices partition one ``RegionBatchGroup``'s index space in
  ``PoolId`` order, and those integers cross into cpp;
* per-layer milestones name the pool, because the two pools' block ids come
  from different tensors and the request has to read the right one.

The worker is built with ``object.__new__``: everything under test is index
bookkeeping over ``_Pool`` records, and constructing a real one needs CUDA IPC
handles from a live producer process.
"""
import pytest

from flexkv.common.pool import PoolId
from flexkv.transfer.workers.gpu_cpu import GPUCPUTransferWorker, _Pool

pytestmark = pytest.mark.unit


def _worker(*pool_ids, num_layers=2, regions_per_pool=1):
    """A worker holding only the fields the methods under test read."""
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


# --------------------------------------------------------------------------
# _pool_for: an op names its pool
# --------------------------------------------------------------------------

def test_op_selects_its_own_pool():
    w = _worker(PoolId.FULL_KV, PoolId.SWA)
    assert w._pool_for(_Op(PoolId.FULL_KV)).pool_id is PoolId.FULL_KV
    assert w._pool_for(_Op(PoolId.SWA)).pool_id is PoolId.SWA


def test_an_unserved_pool_raises_rather_than_falling_through():
    """The dangerous failure mode. An SWA op reaching a worker with no SWA
    pool must not quietly transfer against the full-KV regions -- the block
    ids index a different slot space, so it would corrupt main KV."""
    w = _worker(PoolId.FULL_KV)
    with pytest.raises(RuntimeError, match="no SWA pool registered"):
        w._pool_for(_Op(PoolId.SWA))


# --------------------------------------------------------------------------
# _ordered_pools / region index assignment
# --------------------------------------------------------------------------

def test_pools_are_ordered_by_pool_id_not_insertion():
    """Region indices are assigned in this order and cross into cpp, so
    insertion order drifting would silently renumber every region."""
    # Registered SWA-first, so a walk over the dict itself would yield the
    # wrong order.
    w = _worker(PoolId.SWA, PoolId.FULL_KV)
    assert list(w._pools) == [PoolId.SWA, PoolId.FULL_KV]
    assert [p.pool_id for p in w._ordered_pools()] == [
        PoolId.FULL_KV, PoolId.SWA]


def test_region_indices_partition_one_index_space_in_pool_order():
    """What ``_build_region_batch`` does, isolated from the cpp build.

    Each pool gets a contiguous slice; the slices are disjoint and cover
    ``range(total)``, with FULL_KV first regardless of insertion order.
    """
    w = _worker(PoolId.SWA, PoolId.FULL_KV, regions_per_pool=3)
    specs = []
    for pool in w._ordered_pools():
        pool.region_indices = list(
            range(len(specs), len(specs) + len(pool.regions)))
        specs.extend(pool.regions)

    assert w._pools[PoolId.FULL_KV].region_indices == [0, 1, 2]
    assert w._pools[PoolId.SWA].region_indices == [3, 4, 5]
    assert len(specs) == 6


# --------------------------------------------------------------------------
# _build_layer_milestones: the milestone names its pool
# --------------------------------------------------------------------------

def test_milestones_carry_the_pool_and_the_global_region_index():
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


def test_a_pool_declaring_more_layers_than_the_stage_is_an_error():
    w = _worker(PoolId.FULL_KV, num_layers=2)
    w._pools[PoolId.FULL_KV].region_indices = [0]
    w._pools[PoolId.FULL_KV].layer_members = [[(0, 0)], [(0, 1)], [(0, 2)]]
    with pytest.raises(ValueError, match="declares layer 2"):
        w._build_layer_milestones()
