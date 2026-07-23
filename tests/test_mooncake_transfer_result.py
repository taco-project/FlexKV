"""Focused unit tests for Mooncake per-block transfer completion results."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

pytest.importorskip("flexkv.c_ext")

from flexkv.common.transfer import TransferType
from flexkv.external.mooncake_store_keys import PoolKind
from flexkv.transfer.worker import MooncakeStoreTransferWorker
from flexkv.transfer.worker_op import WorkerTransferResult


pytestmark = pytest.mark.unit


def _worker(block_results=None, error=None):
    """Build a worker without CUDA registration or a real Mooncake client."""
    worker = object.__new__(MooncakeStoreTransferWorker)
    worker.pool_kind = PoolKind.KV
    worker._preprocess_kv = lambda _op: (
        [100, 200, 300],
        [16, 16, 16],
        ["k0", "k1", "k2"],
    )
    worker._log_transfer_performance = lambda *_args: None

    def batch_get(_keys, _ptrs, _sizes):
        if error is not None:
            raise error
        return block_results

    worker.mooncake_client = SimpleNamespace(batch_get=batch_get)
    return worker


def _remote2h_op(op_id=17):
    return SimpleNamespace(
        transfer_op_id=op_id,
        transfer_type=TransferType.REMOTE2H,
        src_block_ids=[0, 1, 2],
    )


def test_mooncake_preserves_non_contiguous_block_results():
    worker = _worker([True, False, True])

    result = worker.launch_transfer(_remote2h_op())

    assert result == WorkerTransferResult(
        transfer_op_id=17,
        block_results=(True, False, True),
    )


def test_mooncake_exception_still_returns_completed_failure():
    worker = _worker(error=RuntimeError("injected get failure"))

    result = worker.launch_transfer(_remote2h_op(op_id=23))

    assert result == WorkerTransferResult(
        transfer_op_id=23,
        block_results=(False, False, False),
    )


def test_mooncake_preprocess_exception_still_returns_completed_failure():
    worker = _worker()

    def fail_preprocess(_op):
        raise RuntimeError("injected preprocess failure")

    worker._preprocess_kv = fail_preprocess

    result = worker.launch_transfer(_remote2h_op(op_id=29))

    assert result == WorkerTransferResult(
        transfer_op_id=29,
        block_results=(False, False, False),
    )


@pytest.mark.parametrize("block_results", [None, [True, False]])
def test_mooncake_invalid_result_count_fails_closed(block_results):
    worker = _worker(block_results)

    result = worker.launch_transfer(_remote2h_op(op_id=31))

    assert result.block_results == (False, False, False)


def test_mooncake_perf_logging_exception_still_completes():
    worker = _worker([True, True, True])
    worker._log_transfer_performance = Mock(side_effect=RuntimeError("log failure"))

    result = worker.launch_transfer(_remote2h_op(op_id=37))

    assert result == WorkerTransferResult(
        transfer_op_id=37,
        block_results=(False, False, False),
    )
