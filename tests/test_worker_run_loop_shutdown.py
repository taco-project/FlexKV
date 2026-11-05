"""Worker run-loop shutdown/EOF regression tests.

The first-op batching regression is already fixed on upstream main. These tests
cover the two remaining edges: an EOF while draining must not discard the
already-received batch, and a drained shutdown sentinel must not call subclass
shutdown before ``_worker_process`` performs its single cleanup in ``finally``.
"""

import types

import pytest


pytestmark = pytest.mark.unit
pytest.importorskip("flexkv.c_ext", reason="requires compiled c_ext")


class _FakeQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class _FakeOp:
    def __init__(self, op_id):
        from flexkv.common.transfer import TransferType

        self.transfer_op_id = op_id
        self.transfer_graph_id = 0
        self.transfer_type = TransferType.H2D
        self.valid_block_num = 1


class _EOFAfterDrainPipe:
    def __init__(self, items):
        self.items = list(items)

    def poll(self, timeout=0):
        if self.items:
            return True
        if timeout == 0:
            return True
        raise EOFError

    def recv(self):
        if self.items:
            return self.items.pop(0)
        raise EOFError


class _SentinelPipe:
    def __init__(self, items):
        self.items = list(items)

    def poll(self, timeout=0):
        if self.items:
            return True
        if timeout and timeout > 0:
            raise EOFError
        return False

    def recv(self):
        return self.items.pop(0)


def _worker(pipe, monkeypatch):
    # ``TransferWorkerBase`` and its run loop live in workers/runtime.py now;
    # flexkv.transfer.worker is a re-export façade, so it has no ``trace``
    # module attribute to patch. The loop under test is unchanged.
    from flexkv.transfer.workers import runtime as worker_module

    monkeypatch.setattr(
        worker_module.trace,
        "build_worker_metrics",
        lambda *_args, **_kwargs: {"test": True},
    )
    launched = []
    shutdowns = []
    instance = types.SimpleNamespace(
        worker_id=0,
        transfer_conn=pipe,
        finished_ops_queue=_FakeQueue(),
        kv_dim=2,
        _bytes_per_block=1,
        launch_transfer=lambda op: launched.append(op.transfer_op_id) or True,
        shutdown=lambda: shutdowns.append(True),
    )
    instance.run = types.MethodType(
        worker_module.TransferWorkerBase.run, instance)
    return instance, launched, shutdowns


def test_eof_during_drain_runs_received_batch(monkeypatch):
    worker, launched, shutdowns = _worker(
        _EOFAfterDrainPipe([_FakeOp(1), _FakeOp(2)]), monkeypatch)

    worker.run()

    assert launched == [1, 2]
    assert [item[0] for item in worker.finished_ops_queue.items] == [1, 2]
    assert shutdowns == []


def test_drained_sentinel_defers_shutdown_to_worker_process(monkeypatch):
    worker, launched, shutdowns = _worker(
        _SentinelPipe([_FakeOp(1), _FakeOp(2), None]), monkeypatch)

    worker.run()

    assert launched == [1, 2]
    assert [item[0] for item in worker.finished_ops_queue.items] == [1, 2]
    assert shutdowns == []
