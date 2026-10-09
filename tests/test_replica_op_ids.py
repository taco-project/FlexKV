"""Receiver-local GPU replica IDs must not alias incoming parent IDs.

Parents cross pickle IPC without advancing the receiver's TransferOp counter.
The regression recreates sender op 5 (30-block remote) and receiver replica 5
(127-block GPU) from the real failure. Workers/buffer allocation are faked;
the scheduler loop, queues, dispatch, completion and failure routing are real.
"""
import itertools
import multiprocessing as mp
import os
import pickle
import queue
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from flexkv.common.pool import PoolId
from flexkv.common.transfer import (
    LayerwiseTransferOp,
    TransferOp,
    TransferOpGraph,
    TransferType,
    WorkerKey,
)
from flexkv.transfer import transfer_engine as module
from flexkv.transfer.scheduler import TransferScheduler
from flexkv.transfer.worker_op import (
    WorkerLayerwiseTransferOp,
    WorkerTransferOp,
    WorkerTransferResult,
)

pytestmark = pytest.mark.unit


def make_op(graph, kind, blocks, pool=PoolId.FULL_KV):
    ids = np.arange(blocks, dtype=np.int64)
    if kind == TransferType.LAYERWISE:
        op = LayerwiseTransferOp(graph.graph_id, ids, ids.copy())
    else:
        op = TransferOp(graph.graph_id, kind, ids, ids.copy(), pool_id=pool)
    graph.add_transfer_op(op)
    return op


@pytest.fixture
def engine(monkeypatch):
    e = module.TransferEngine.__new__(module.TransferEngine)
    e.scheduler = TransferScheduler()
    ctx = mp.get_context("spawn")
    e.task_queue = ctx.Queue()
    e.finished_ops_queue = ctx.Queue()
    e.completed_queue = queue.Queue()
    e.shutdown_read_fd, e.shutdown_write_fd = os.pipe()
    e.op_id_to_op = {}
    e.op_id_to_nvtx_range = {}
    e._child_id_to_child = {}
    e._child_to_parent_op_id = {}
    e._failed_graph_ids = set()
    e._failed_parent_op_ids = set()
    e.pin_buffer = None
    e.model_config = NS(token_size_in_bytes=16, num_layers=1)
    e.cache_config = NS(tokens_per_block=16)
    e._num_layers_for_local_pp_stage = 1
    e._running = False
    e._test_thread = None
    e._test_freed = []
    monkeypatch.setattr(e, "shutdown", lambda: None)
    monkeypatch.setattr(e, "_emit_xfer_trace", lambda *args: None)
    monkeypatch.setattr(module, "register_op_to_buffer", lambda *args: None)
    monkeypatch.setattr(
        module, "free_op_from_buffer", lambda op, pool: e._test_freed.append(op.op_id)
    )
    yield e
    if e._test_thread is not None:
        e._running = False
        os.write(e.shutdown_write_fd, b"x")
        e._test_thread.join(timeout=3)
        assert not e._test_thread.is_alive()
    os.close(e.shutdown_read_fd)
    os.close(e.shutdown_write_fd)
    for q in (e.task_queue, e.finished_ops_queue):
        q.close()
        q.join_thread()


@pytest.mark.parametrize("siblings", [1, 2])
@pytest.mark.parametrize(
    "kind,pool",
    [
        (TransferType.H2D, PoolId.FULL_KV),
        (TransferType.D2H, PoolId.FULL_KV),
        (TransferType.H2D, PoolId.SWA),
        (TransferType.D2H, PoolId.SWA),
        (TransferType.LAYERWISE, PoolId.FULL_KV),
    ],
)
def test_replica_namespace_is_disjoint_before_worker_serialization(
    engine, monkeypatch, kind, pool, siblings
):
    # Independent sender/receiver counters can have the same next value. A
    # replica must not collide even with a parent that arrives in a FUTURE graph.
    monkeypatch.setattr(TransferOp, "_op_id_counter", itertools.count(5))
    remote = make_op(TransferOpGraph(), TransferType.H2REMOTE, 30)
    parent = make_op(TransferOpGraph(), kind, 127, pool)
    remote, parent = pickle.loads(pickle.dumps((remote, parent)))
    monkeypatch.setattr(TransferOp, "_op_id_counter", itertools.count(5))
    submitted = []

    def send(replica):
        wire_type = (
            WorkerLayerwiseTransferOp
            if kind == TransferType.LAYERWISE
            else WorkerTransferOp
        )
        wire = pickle.loads(pickle.dumps(wire_type(replica)))
        assert wire.transfer_op_id == replica.op_id
        submitted.append(wire)

    engine._workers.setdefault(pool, {})[kind] = {
        WorkerKey(0, pp): NS(submit_transfer=send) for pp in range(siblings)
    }
    for _ in range(2):
        engine._assign_op_to_worker(parent)
    ids = [op.transfer_op_id for op in submitted]
    assert len(ids) == len(set(ids)) == 2 * siblings
    assert all(op_id < -1 for op_id in ids)
    assert remote.op_id == 5 and parent.op_id == 6
    assert remote.op_id not in engine._child_to_parent_op_id
    assert parent.op_id not in engine._child_to_parent_op_id
    assert all(engine._child_to_parent_op_id[op_id] == parent.op_id for op_id in ids)
    assert parent.pending_count == 2 * siblings


@pytest.mark.parametrize("first", ["remote", "gpu"])
@pytest.mark.parametrize("failed", [None, "remote", "gpu"])
def test_completion_and_failure_cannot_cross_parent_replica_namespaces(
    engine, monkeypatch, first, failed
):
    monkeypatch.setattr(TransferOp, "_op_id_counter", itertools.count(5))
    remote_graph, gpu_graph = TransferOpGraph(), TransferOpGraph()
    remote = make_op(remote_graph, TransferType.H2REMOTE, 30)
    gpu = make_op(gpu_graph, TransferType.H2D, 127)
    assert remote.op_id == 5 and gpu.op_id == 6
    # The task queue pickles the graphs. Resetting before receiver dispatch
    # models its process-local counter after five previous GPU replicas.
    monkeypatch.setattr(TransferOp, "_op_id_counter", itertools.count(5))
    remote_submitted, gpu_submitted = queue.Queue(), queue.Queue()
    engine._workers[PoolId.FULL_KV] = {
        TransferType.H2REMOTE: NS(
            submit_transfer=lambda op: remote_submitted.put(WorkerTransferOp(op))
        ),
        TransferType.H2D: {
            WorkerKey(): NS(
                submit_transfer=lambda op: gpu_submitted.put(WorkerTransferOp(op))
            )
        },
    }
    engine._running = True
    engine._test_thread = threading.Thread(target=engine._scheduler_loop, daemon=True)
    engine._test_thread.start()
    engine.task_queue.put([remote_graph, gpu_graph])
    submitted = {
        "remote": remote_submitted.get(timeout=3),
        "gpu": gpu_submitted.get(timeout=3),
    }
    parents = {"remote": remote, "gpu": gpu}
    order = (first, "gpu" if first == "remote" else "remote")
    for count, side in enumerate(order, 1):
        wire = submitted[side]
        parent = parents[side]
        bitmap = (True,) * len(parent.src_block_ids)
        payload = (
            (wire.transfer_op_id, False)
            if failed == side
            else WorkerTransferResult(wire.transfer_op_id, bitmap)
        )
        engine.finished_ops_queue.put(pickle.loads(pickle.dumps(payload)))
        messages = [engine.completed_queue.get(timeout=3)]
        if failed == side:
            assert messages[0].is_graph_failed()
        else:
            messages.append(engine.completed_queue.get(timeout=3))
            completion, terminal = messages
            assert completion.op_id == parent.op_id >= 0
            assert completion.block_results == bitmap
            assert terminal.is_graph_completed()
        assert all(msg.graph_id == parent.graph_id for msg in messages)
        # No completion/free from the other graph before its own worker reports.
        assert engine._test_freed == [
            submitted[s].transfer_op_id for s in order[:count]
        ]
        assert engine.completed_queue.empty()
    assert len(engine._test_freed) == 2
    assert not engine.op_id_to_op
    assert not engine._child_id_to_child and not engine._child_to_parent_op_id
