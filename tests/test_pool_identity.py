"""The pool abstraction itself: PoolId, PoolEndpoint, and the seams they key.

The refactor's claim is that a pool is a *value* -- a key on the op, on the
storage registry and on the worker -- rather than a parallel set of classes,
maps and branches. These tests pin the properties that claim depends on:

* the value survives every boundary it crosses (pickle, dataclass, kwargs bag)
  as the interned member, because readers compare with ``is``;
* the endpoint distinguishes two pools on the *same* physical device id, which
  is the thing a ``(DeviceType, device_id)`` key could not express and the
  reason there used to be a second dict;
* a caller that names both selectors gets an error, not a precedence rule.
"""
import pickle

import pytest

from flexkv.common.pool import PoolEndpoint, PoolId
from flexkv.common.transfer import DeviceType, TransferOp, TransferType
from flexkv.storage.storage_engine import _pool_from_kwargs, _resolve_pool

pytestmark = pytest.mark.unit


# --------------------------------------------------------------------------
# PoolId
# --------------------------------------------------------------------------

def test_full_kv_is_zero_so_the_is_swa_false_default_maps_onto_it():
    # Wire-visible: WorkerTransferOp carries the raw int, so this is a
    # compatibility constraint, not a style choice.
    assert int(PoolId.FULL_KV) == 0
    assert int(PoolId.SWA) == 1


def test_pool_id_round_trips_as_the_interned_member():
    """Readers spell it ``op.pool_id is PoolId.SWA``; ``is`` must survive."""
    for pid in PoolId:
        assert PoolId(int(pid)) is pid
        assert pickle.loads(pickle.dumps(pid)) is pid


def test_is_swa_alias_agrees_with_the_pool_in_both_directions():
    for is_swa in (False, True):
        assert PoolId.from_is_swa(is_swa).is_swa is is_swa
    assert PoolId.FULL_KV.is_swa is False
    assert PoolId.SWA.is_swa is True


# --------------------------------------------------------------------------
# PoolEndpoint
# --------------------------------------------------------------------------

def test_endpoint_separates_two_pools_on_one_device_id():
    """The whole reason the pool is in the storage key.

    The same physical GPU 0 holds an independent main-KV and SWA handle. A
    (DeviceType, device_id) key collides here; this one does not.
    """
    main = PoolEndpoint(PoolId.FULL_KV, DeviceType.GPU)
    swa = PoolEndpoint(PoolId.SWA, DeviceType.GPU)
    assert main != swa
    registry = {(main, 0): "main-handle", (swa, 0): "swa-handle"}
    assert len(registry) == 2
    assert registry[(swa, 0)] == "swa-handle"


# --------------------------------------------------------------------------
# _resolve_pool / _pool_from_kwargs: the two selectors
# --------------------------------------------------------------------------


def test_resolve_pool_rejects_a_contradiction_rather_than_picking_a_winner():
    """A caller saying ``pool_id=FULL_KV, is_swa=True`` has a bug.

    Honouring either silently would hide it, and the two possible precedence
    rules send the data to two different pools.
    """
    with pytest.raises(ValueError, match="conflicting pool selectors"):
        _resolve_pool(PoolId.FULL_KV, True)


# --------------------------------------------------------------------------
# TransferOp: pool_id is the field, is_swa the derived alias
# --------------------------------------------------------------------------

def _op(**kwargs):
    # op_id is not a constructor argument -- __post_init__ draws it from the
    # class counter.
    import numpy as np
    base = dict(
        graph_id=1,
        transfer_type=TransferType.D2H,
        src_block_ids=np.array([0], dtype=np.int64),
        dst_block_ids=np.array([0], dtype=np.int64),
    )
    base.update(kwargs)
    return TransferOp(**base)


def test_transfer_op_rejects_a_contradiction():
    with pytest.raises(ValueError, match="conflicting pool selectors"):
        _op(pool_id=PoolId.SWA, is_swa=False)


# --------------------------------------------------------------------------
# WorkerTransferOp: the pool crosses the process boundary
# --------------------------------------------------------------------------

def test_worker_op_carries_the_pool_across_the_pipe_as_the_member():
    """``op.pool_id is PoolId.SWA`` in the worker is how the pool is selected,
    so the hand-rolled int encoding must decode back to the interned member."""
    from flexkv.transfer.worker_op import WorkerTransferOp

    for pid in PoolId:
        wop = WorkerTransferOp(_op(pool_id=pid))
        assert wop.pool_id is pid
        round_tripped = pickle.loads(pickle.dumps(wop))
        assert round_tripped.pool_id is pid
        assert round_tripped.is_swa is pid.is_swa


def test_worker_op_encodes_the_pool_as_a_bare_int_on_the_wire():
    """Not decoration: an IntEnum member costs ~6x a bare int to pickle
    because it inherits Enum.__reduce_ex__. The encoding is why."""
    from flexkv.transfer.worker_op import WorkerTransferOp

    state = WorkerTransferOp(_op(pool_id=PoolId.SWA)).__getstate__()
    assert type(state["pool_id"]) is int
