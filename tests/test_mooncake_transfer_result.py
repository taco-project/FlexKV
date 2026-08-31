"""Focused unit tests for Mooncake per-block transfer completion results.

Upstream wrote these against ``MooncakeStoreTransferWorker.launch_transfer``.
That class is gone: mooncake-store is a ``StorageBackend`` now, and the
CPU<->Remote *edge* it plugs into is ``CPURemoteTransferWorker``.  The
behaviour under test did not move, though -- it just spans two objects instead
of one, so each case drives the same pair the runtime does:
``TransferWorkerBase._run_backend`` -> ``MooncakeStoreBackend.transfer_blocks``.

The contract these pin down is the reason the split is safe:

* a partial result stays partial (non-contiguous misses are preserved),
* every failure -- in key building, in the store call, in the result count,
  even in perf logging -- still *completes* the op with all-False rather than
  raising, because an op that never reports hangs its graph and leaks every
  cache block the plan holds.
"""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

pytest.importorskip("flexkv.c_ext")

from flexkv.common.transfer import TransferType
from flexkv.external.mooncake_store_keys import PoolKind
from flexkv.transfer.backends import MooncakeStoreBackend
from flexkv.transfer.workers import CPURemoteTransferWorker
from flexkv.transfer.workers.runtime import TransferWorkerBase
from flexkv.transfer.worker_op import WorkerTransferResult


pytestmark = pytest.mark.unit


def _backend(block_results=None, error=None):
    """A backend with the store client and key building stubbed out.

    ``object.__new__`` skips ``__init__`` deliberately: constructing the real
    thing dials a mooncake master and registers RDMA memory regions, neither of
    which says anything about how a result is reported.
    """
    backend = object.__new__(MooncakeStoreBackend)
    backend.pool_kind = PoolKind.KV
    backend._keys_and_ptrs = lambda _op, _src, _dst: (
        [100, 200, 300],
        [16, 16, 16],
        ["k0", "k1", "k2"],
    )

    def batch_get(_keys, _ptrs, _sizes):
        if error is not None:
            raise error
        return block_results

    backend.mooncake_client = SimpleNamespace(batch_get=batch_get)
    return backend


def _worker(backend):
    """A worker shell holding just what ``_run_backend`` reads."""
    worker = object.__new__(CPURemoteTransferWorker)
    worker.worker_id = 0
    worker._backend = backend
    worker._log_transfer_performance = lambda *_args, **_kwargs: None
    return worker


def _remote2h_op(op_id=17):
    block_ids = np.array([0, 1, 2], dtype=np.int64)
    return SimpleNamespace(
        transfer_op_id=op_id,
        transfer_graph_id=0,
        transfer_type=TransferType.REMOTE2H,
        # -1 slots: block ids ride on the op itself rather than the shared op
        # buffer, which is what every mooncake op does (see WorkerTransferOp).
        src_slot_id=-1,
        dst_slot_id=-1,
        valid_block_num=3,
        src_block_ids=block_ids,
        dst_block_ids=block_ids,
    )


def _launch(backend, op):
    return TransferWorkerBase._run_backend(_worker(backend), op)


def test_mooncake_preserves_non_contiguous_block_results():
    result = _launch(_backend([True, False, True]), _remote2h_op())

    assert result == WorkerTransferResult(
        transfer_op_id=17,
        block_results=(True, False, True),
    )


def test_mooncake_exception_still_returns_completed_failure():
    backend = _backend(error=RuntimeError("injected get failure"))

    result = _launch(backend, _remote2h_op(op_id=23))

    assert result == WorkerTransferResult(
        transfer_op_id=23,
        block_results=(False, False, False),
    )


def test_mooncake_preprocess_exception_still_returns_completed_failure():
    backend = _backend()

    def fail_preprocess(_op, _src, _dst):
        raise RuntimeError("injected preprocess failure")

    backend._keys_and_ptrs = fail_preprocess

    result = _launch(backend, _remote2h_op(op_id=29))

    assert result == WorkerTransferResult(
        transfer_op_id=29,
        block_results=(False, False, False),
    )


@pytest.mark.parametrize("block_results", [None, [True, False]])
def test_mooncake_invalid_result_count_fails_closed(block_results):
    result = _launch(_backend(block_results), _remote2h_op(op_id=31))

    assert result.block_results == (False, False, False)


def test_mooncake_perf_logging_exception_still_completes():
    backend = _backend([True, True, True])
    worker = _worker(backend)
    worker._log_transfer_performance = Mock(side_effect=RuntimeError("log failure"))

    result = TransferWorkerBase._run_backend(worker, _remote2h_op(op_id=37))

    # Completed, not raised -- and failed closed, because a worker that cannot
    # record what it did is not one whose success we should believe.
    assert result == WorkerTransferResult(
        transfer_op_id=37,
        block_results=(False, False, False),
    )
