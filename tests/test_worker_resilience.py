"""Regression contracts for worker death, dispatch isolation, fencing and recovery."""
from queue import Queue
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from flexkv.common.pool import PoolId
from flexkv.common.transfer import TransferOp, TransferOpGraph, TransferType, WorkerKey
from flexkv.transfer.scheduler import TransferScheduler
from flexkv.transfer.transfer_engine import TransferEngine

pytestmark = pytest.mark.unit


def worker(*, alive=True, drained=True, send_error=None):
    ack = Event()
    if drained:
        ack.set()
    ready = Event()
    ready.set()
    w = SimpleNamespace(worker_id=17, process=SimpleNamespace(
        is_alive=Mock(return_value=alive), exitcode=0), quiesced_event=ack,
        ready_event=ready, submit_transfer=Mock(side_effect=send_error))
    w.shutdown = Mock(side_effect=lambda: w.process.is_alive.configure_mock(return_value=False))
    w.abort = w.shutdown
    return w


def engine():
    e = TransferEngine.__new__(TransferEngine)
    e._init_resilience()
    e.scheduler = TransferScheduler()
    e.completed_queue = Queue()
    e.op_id_to_op = {}
    e.op_id_to_nvtx_range = {}
    e._child_id_to_child = {}
    e._child_to_parent_op_id = {}
    e._failed_graph_ids = set()
    e._failed_parent_op_ids = set()
    e._parked_skip_io = {}
    e._completed_op_keys = set()
    e.pin_buffer = SimpleNamespace(allocate_slot=Mock(return_value=-1), free_slot=Mock())
    e.h2d_workers = {WorkerKey(0, 0): worker(), WorkerKey(1, 0): worker()}
    e.d2h_workers = {WorkerKey(0, 0): worker(), WorkerKey(1, 0): worker()}
    for pool in (PoolId.FULL_KV, PoolId.SWA):
        e._register_worker(pool, TransferType.H2D, e.h2d_workers)
        e._register_worker(pool, TransferType.D2H, e.d2h_workers)
    return e


def graph(dp=0, count=1, pool=PoolId.FULL_KV):
    g = TransferOpGraph.create_empty_graph()
    for _ in range(count):
        g.add_transfer_op(TransferOp(graph_id=g.graph_id, transfer_type=TransferType.H2D,
            src_block_ids=np.array([1], dtype=np.int64), dst_block_ids=np.array([2], dtype=np.int64),
            dp_client_id=dp, pool_id=pool))
    return g


def drain(e):
    e._emit_drained_graph_failures()
    result = []
    while not e.completed_queue.empty():
        result.append(e.completed_queue.get_nowait())
    return result


def test_invalid_route_does_not_drop_healthy_graph_in_same_batch():
    e = engine()
    bad, good = graph(999), graph(1)
    for g in (bad, good):
        e.scheduler.add_transfer_graph(g)
    e._dispatch_scheduled_ops([])
    assert [x.graph_id for x in drain(e) if x.is_graph_failed()] == [bad.graph_id]
    assert e.h2d_workers[WorkerKey(1, 0)].submit_transfer.call_count == 1
    assert all(op.graph_id != bad.graph_id for op in e.op_id_to_op.values())


@pytest.mark.parametrize("drained", [True, False])
@pytest.mark.parametrize("pool", [PoolId.FULL_KV, PoolId.SWA])
def test_lost_worker_requires_explicit_drain_ack(drained, pool):
    e = engine()
    dead = e.h2d_workers[WorkerKey(0, 0)] = worker(drained=drained)
    g = graph(pool=pool)
    e.scheduler.add_transfer_graph(g)
    e._dispatch_scheduled_ops([])
    dead.process.is_alive.return_value = False
    e._check_exited_workers()
    failures = drain(e)
    assert bool(failures) == drained
    assert bool(e.op_id_to_op) != drained
    assert bool(e._op_workers) != drained
    assert 0 in e._disabled_routes and 1 not in e._disabled_routes
    if not drained:
        e.pin_buffer.free_slot.assert_not_called()


def test_fence_stops_both_directions_and_preserves_other_instance():
    e = engine()
    assert e.fence_gpu_route(0) == {"safe": True, "generation": 0}
    for mapping in (e.h2d_workers, e.d2h_workers):
        mapping[WorkerKey(0, 0)].shutdown.assert_called_once()
        mapping[WorkerKey(1, 0)].shutdown.assert_not_called()
    assert 0 in e._disabled_routes
    doomed, healthy = graph(0), graph(1)
    for g in (doomed, healthy):
        e.scheduler.add_transfer_graph(g)
    e._dispatch_scheduled_ops([])
    assert [x.graph_id for x in drain(e)] == [doomed.graph_id]
    assert e.h2d_workers[WorkerKey(1, 0)].submit_transfer.call_count == 1


def test_broken_pipe_pp_fanout_drains_parent_once():
    e = engine()
    e.h2d_workers = {WorkerKey(0, 0): worker(send_error=BrokenPipeError()),
                     WorkerKey(0, 1): worker()}
    e._register_worker(PoolId.FULL_KV, TransferType.H2D, e.h2d_workers)
    g = graph()
    e.scheduler.add_transfer_graph(g)
    e._dispatch_scheduled_ops([])
    e._check_exited_workers()
    assert not drain(e)  # PP sibling still owns a replica
    parent = next(iter(e.op_id_to_op.values()))
    assert parent.pending_count == 1
    sibling = e.h2d_workers[WorkerKey(0, 1)]
    sibling.process.is_alive.return_value = False
    e._check_exited_workers()
    assert len(drain(e)) == 1
    assert not e.op_id_to_op and not e._child_to_parent_op_id
    assert not e._child_id_to_child and not e._op_workers
    assert not drain(e)


def test_recovery_reimports_both_pools_and_advances_generation():
    e = engine()
    e.gpu_handle_groups = {WorkerKey(0, 0): [object()]}
    e._create_gpu_cpu_worker = Mock(side_effect=[worker(), worker()])
    e.fence_gpu_route(0)
    assert e.recover_gpu_route(0, 0) == {"safe": True, "generation": 1}
    assert 0 not in e._disabled_routes
    assert e._workers[PoolId.SWA][TransferType.H2D] is e.h2d_workers
    assert e._create_gpu_cpu_worker.call_count == 2
    e.fence_gpu_route(0)
    with pytest.raises(ValueError, match="Stale"):
        e.recover_gpu_route(0, 0)


def test_recovery_failure_keeps_route_fenced_and_stops_new_worker():
    e = engine()
    e.gpu_handle_groups = {WorkerKey(0, 0): [object()]}
    replacement = worker()
    e._create_gpu_cpu_worker = Mock(side_effect=[replacement, RuntimeError("OOM")])
    e.fence_gpu_route(0)
    with pytest.raises(RuntimeError, match="OOM"):
        e.recover_gpu_route(0, 0)
    replacement.shutdown.assert_called_once()
    assert 0 in e._disabled_routes and not e._route_generation


def test_quarantine_prevents_recovery_and_preserves_buffers():
    e = engine()
    w = e.h2d_workers[WorkerKey(0, 0)] = worker(drained=False)
    g = graph()
    e.scheduler.add_transfer_graph(g)
    e._dispatch_scheduled_ops([])
    w.process.is_alive.return_value = False
    assert not e.fence_gpu_route(0)["safe"]
    with pytest.raises(RuntimeError, match="quarantined"):
        e.recover_gpu_route(0, 0)
    assert e.op_id_to_op and e._child_id_to_child
    assert not drain(e)


def test_shared_worker_that_already_died_rejects_new_ops_without_leaks():
    e = engine()
    e.finished_ops_queue = Queue()
    dead = worker(alive=False)
    e._register_worker(PoolId.FULL_KV, TransferType.H2REMOTE, dead)
    e._check_exited_workers()
    g = graph()
    op = next(iter(g._op_map.values()))
    op.transfer_type = TransferType.H2REMOTE
    e.scheduler.add_transfer_graph(g)
    e._dispatch_scheduled_ops([])
    assert not e._op_workers
    op_id, success, _ = e.finished_ops_queue.get_nowait()
    assert not success
    e._handle_failed_op(op_id)
    assert [msg.graph_id for msg in drain(e)] == [g.graph_id]
    assert not e.op_id_to_op
    dead.submit_transfer.assert_not_called()


def test_partial_pin_allocation_is_rolled_back_once():
    from flexkv.transfer.transfer_engine import register_op_to_buffer, free_op_from_buffer

    op = next(iter(graph()._op_map.values()))
    pool = SimpleNamespace(allocate_slot=Mock(side_effect=[5, RuntimeError("pool unavailable")]),
                           free_slot=Mock())
    with pytest.raises(RuntimeError, match="pool unavailable"):
        register_op_to_buffer(op, pool)
    free_op_from_buffer(op, pool)
    pool.free_slot.assert_called_once_with(5)
    assert op.src_slot_id == -1 and op.dst_slot_id == -1


def test_recovery_retry_after_lost_ack_does_not_spawn_again():
    e = engine()
    e.gpu_handle_groups = {WorkerKey(0, 0): [object()]}
    e._create_gpu_cpu_worker = Mock(side_effect=[worker(), worker()])
    e.fence_gpu_route(0)
    assert e.recover_gpu_route(0, 0)["generation"] == 1
    assert e.recover_gpu_route(0, 0) == {"safe": True, "generation": 1}
    assert e._create_gpu_cpu_worker.call_count == 2


def test_fault_abort_is_bounded_and_never_sends_into_full_pipe():
    from flexkv.transfer.workers.handle import WorkerHandle

    process = Mock(pid=123)
    process.is_alive.side_effect = [True, True, False]
    pipe = Mock()
    handle = WorkerHandle(0, pipe, process, Event(), Event())
    handle.abort(timeout=.01)
    process.terminate.assert_called_once()
    process.kill.assert_called_once()
    assert process.join.call_count == 2
    assert all(call.kwargs["timeout"] == .01 for call in process.join.call_args_list)
    pipe.send.assert_not_called()
    pipe.close.assert_called_once()
    handle.abort(timeout=.01)
    assert process.join.call_count == 2


def test_uninterruptible_worker_is_not_acknowledged_or_closed():
    from flexkv.transfer.workers.handle import WorkerHandle

    process = Mock(pid=123)
    process.is_alive.return_value = True
    pipe = Mock()
    handle = WorkerHandle(0, pipe, process, Event(), Event())
    try:
        with pytest.raises(TimeoutError, match="SIGKILL"):
            handle.abort(timeout=.01)
        assert not handle._shutdown_done
        assert not handle.quiesced_event.is_set()
        pipe.close.assert_not_called()
        pipe.send.assert_not_called()
    finally:
        process.is_alive.return_value = False
        handle.abort(timeout=.01)


def test_dispatch_exception_skips_same_graph_siblings_but_not_other_graphs():
    e = engine()
    failed = graph(999)
    sibling = next(iter(graph(0)._op_map.values()))
    sibling.graph_id = failed.graph_id
    failed.add_transfer_op(sibling)
    healthy = graph(1)
    e.scheduler.add_transfer_graph(failed)
    e.scheduler.add_transfer_graph(healthy)
    e._dispatch_scheduled_ops([])
    e.h2d_workers[WorkerKey(0, 0)].submit_transfer.assert_not_called()
    e.h2d_workers[WorkerKey(1, 0)].submit_transfer.assert_called_once()
    assert [msg.graph_id for msg in drain(e)] == [failed.graph_id]
