"""CPU task ownership regressions; no DMA or worker process is launched."""

from types import SimpleNamespace as NS
from collections import OrderedDict
from concurrent.futures import Future
import threading
from unittest.mock import Mock, patch

import numpy as np
import pytest

from flexkv.kvtask import KVTask, KVTaskEngine, KVTaskManager, TaskStatus, TaskType
from flexkv.common.transfer import CompletedOp

pytestmark = pytest.mark.unit


def engine_with_task(status=TaskStatus.RUNNING, kind=TaskType.GET, request_returned=False):
    engine = KVTaskEngine.__new__(KVTaskEngine)
    plan = Mock()
    task = KVTask(
        7,
        kind,
        11,
        False,
        status,
        np.arange(4),
        None,
        None,
        NS(graph_id=13, num_ops=2, _op_map={}),
        np.ones(4, dtype=bool),
        plan,
        {},
    )
    task.request_returned = request_returned
    task.prefetch_key = 19
    engine.tasks = {7: task}
    engine._terminal_tasks = OrderedDict()
    engine.prefetch_tasks = {19: 7}
    engine.graph_to_task = {13: 7}
    engine.uncompleted_ops = {}
    engine.uncompleted_op_results = {}
    engine.uncompleted_graphs = {}
    engine._log_task_terminal = Mock()
    engine._finalize_prefetch_return_mask = Mock()
    engine.tracer = Mock()
    engine.cache_engine = Mock()
    return engine, task, plan


@pytest.mark.parametrize("kind", list(TaskType))
@pytest.mark.parametrize("failed", [False, True])
def test_running_cancel_keeps_ownership_until_graph_drains(kind, failed):
    engine, task, plan = engine_with_task(kind=kind)
    engine.cancel_tasks([7])
    engine.cancel_tasks([7])  # idempotent while draining
    assert task.cancel_requested and task.status == TaskStatus.RUNNING
    assert engine.tasks[7] is task and engine.graph_to_task == {13: 7}
    assert engine.prefetch_tasks == {19: 7}
    assert not plan.mock_calls
    task.task_end_op_finished = True
    assert not engine.check_completed(7)  # early data-path completion is insufficient
    if failed:
        engine._fail_task(7)
    else:
        engine._mark_completed(7)
    plan.abort.assert_called_once_with()
    plan.assert_not_called()
    assert task.status == TaskStatus.CANCELLED
    assert engine.tasks == engine.graph_to_task == engine.prefetch_tasks == {}
    assert task.op_callback_dict == {} and task.graph is None


@pytest.mark.parametrize("status", [TaskStatus.UNREADY, TaskStatus.READY])
def test_unlaunched_cancel_releases_immediately(status):
    engine, task, plan = engine_with_task(status=status)
    engine._cancel_task(7)
    plan.abort.assert_called_once_with()
    assert engine.tasks == engine.prefetch_tasks == engine.graph_to_task == {}


def test_terminal_response_wait_recycles_all_indexes():
    engine, task, plan = engine_with_task()
    engine._mark_completed(7)
    assert engine.tasks[7] is task
    engine._update_tasks = Mock()
    result = engine.try_wait([7])
    assert result[7].status.value == "success"
    plan.assert_called_once_with()
    assert engine.tasks == engine.prefetch_tasks == engine.graph_to_task == {}


def test_releasing_old_prefetch_does_not_remove_new_owner():
    engine, task, plan = engine_with_task(status=TaskStatus.READY)
    engine.prefetch_tasks[19] = 8
    engine._cancel_task(7)
    assert engine.prefetch_tasks == {19: 8}


@pytest.mark.parametrize("status", [TaskStatus.RUNNING, TaskStatus.READY, TaskStatus.UNREADY])
def test_busy_reset_has_no_side_effects(status):
    engine, task, plan = engine_with_task(status=status)
    with pytest.raises(RuntimeError, match="requires drained transfers"):
        engine.reset_cache()
    assert engine.tasks[7] is task
    assert engine.graph_to_task == {13: 7}
    assert not engine.cache_engine.mock_calls and not plan.mock_calls


def test_drained_reset_clears_terminal_metadata():
    engine, task, plan = engine_with_task()
    engine._mark_completed(7)
    engine.reset_cache()
    engine.cache_engine.reset.assert_called_once_with()
    assert engine.tasks == engine.prefetch_tasks == engine.graph_to_task == {}
    assert not engine._terminal_tasks


def test_constructor_uses_explicit_task_ownership():
    import flexkv.kvtask as module

    config = NS(enable_cpu=True, enable_remote=True, enable_gds=False, enable_kv_sharing=False, enable_nixl=False)
    model = NS(use_trtllm_subprocess=False, nnodes=1, local_dp_size=None)
    with patch.object(module, "GlobalCacheEngine"), patch.object(module, "TransferManagerHandle"):
        manager = KVTaskManager(model, config)
        assert type(manager.tasks) is dict and type(manager.prefetch_tasks) is dict
        assert isinstance(manager._terminal_tasks, OrderedDict)


def test_releasing_missing_task_removes_terminal_entry():
    engine, _, _ = engine_with_task()
    engine._terminal_tasks[404] = 0
    engine._release_task(404)
    assert not engine._terminal_tasks


@pytest.mark.parametrize("status", [TaskStatus.RUNNING, TaskStatus.COMPLETED])
def test_legacy_completion_poll_reaps_only_expired_terminal_tasks(monkeypatch, status):
    engine, task, _ = engine_with_task()
    engine.transfer_handles = []
    if status == TaskStatus.COMPLETED:
        engine._mark_completed(7)
        engine._terminal_tasks[7] = 0
    monkeypatch.setattr("flexkv.kvtask.time.monotonic", lambda: 1801)
    engine._update_tasks(timeout=0)
    if status == TaskStatus.COMPLETED:
        assert not engine.tasks and not engine.prefetch_tasks and not engine._terminal_tasks
    else:
        assert engine.tasks[7] is task and engine.graph_to_task == {13: 7}


def test_job_only_prefetch_cancellation_preserves_radixshmem_ownership():
    engine, task, plan = engine_with_task(kind=TaskType.PREFETCH)
    job = NS(cancel=Mock(), done=Mock(return_value=False))
    task.prefetch_job = job
    task.graph.num_ops = 0
    engine.prefetch_jobs = {7: job}
    engine.cancel_tasks(7)
    job.cancel.assert_called_once_with()
    plan.abort.assert_called_once_with()
    assert task.status == TaskStatus.CANCELLED and task.prefetch_job is None
    assert not engine.prefetch_jobs and not engine.tasks and not engine.prefetch_tasks
    job.done.return_value = True
    engine._poll_prefetch_jobs()
    job.done.assert_not_called()


def test_chunked_reset_retains_stop_and_drain_until_transfers_finish():
    engine, task, plan = engine_with_task()
    coordinator = NS(
        stop_all=Mock(), drained=False, sessions={}, expired={}, accepting=False, epoch="old"
    )
    engine._prefetch = coordinator
    with pytest.raises(RuntimeError, match="cache reset is draining"):
        engine.reset_cache()
    coordinator.stop_all.assert_called_once_with("reset")
    assert engine.tasks[7] is task and engine.graph_to_task == {13: 7}
    engine.cache_engine.reset.assert_not_called()
    plan.abort.assert_not_called()
    engine.cancel_tasks(7)
    engine._mark_completed(7)
    coordinator.drained = True
    engine.reset_cache()
    assert coordinator.accepting and coordinator.epoch != "old"
    engine.cache_engine.reset.assert_called_once_with()
    assert not engine.tasks and not engine._terminal_tasks


@pytest.mark.parametrize("failed", [False, True])
def test_cancel_waits_for_every_transfer_handle_before_abort(failed):
    engine, task, plan = engine_with_task()
    writer_commit = Mock()
    task.op_callback_dict[11] = writer_commit
    engine.required_completed_count = 2
    first = [CompletedOp(13, 11), CompletedOp(13, -1)]
    second = [CompletedOp(13, 11), CompletedOp(13, -1, failed=failed)]
    engine.transfer_handles = [
        NS(wait=Mock(side_effect=[first, []]), shutdown=Mock()),
        NS(wait=Mock(side_effect=[[], second]), shutdown=Mock()),
    ]
    engine.cancel_tasks([7])
    engine._update_tasks(timeout=0)
    assert engine.tasks[7] is task
    plan.abort.assert_not_called()
    writer_commit.assert_not_called()
    engine._update_tasks(timeout=0)
    writer_commit.assert_called_once()  # completed writers can safely publish
    plan.abort.assert_called_once_with()
    plan.assert_not_called()
    assert engine.tasks == engine.graph_to_task == engine.prefetch_tasks == {}
    assert engine.uncompleted_graphs == engine.uncompleted_ops == {}


def test_put_match_batch_rolls_back_earlier_reservation_on_later_failure():
    engine, task, plan = engine_with_task(status=TaskStatus.UNREADY, kind=TaskType.PUT)
    engine.tasks.clear()
    engine.graph_to_task.clear()
    engine.prefetch_tasks.clear()
    task.prefetch_key = None
    engine._update_tasks = Mock()

    def reserve(tokens, slots, **kwargs):
        if kwargs["task_id"] == 8:
            raise RuntimeError("second metadata match failed")
        task.task_id = kwargs["task_id"]
        engine.tasks[task.task_id] = task
        engine.graph_to_task[task.graph.graph_id] = task.task_id
        return task.task_id, np.ones(len(tokens), dtype=bool)

    engine._put_match_impl = reserve
    with pytest.raises(RuntimeError, match="second metadata"):
        engine.put_match_batch([
            dict(task_id=7, token_ids=np.arange(4), namespace=["a"]),
            dict(task_id=8, token_ids=np.arange(4), namespace=["b"]),
        ])
    plan.abort.assert_called_once_with()
    assert engine.tasks == {} and engine.graph_to_task == {}
    engine._update_tasks.assert_called_once_with(timeout=0)


def test_put_match_batch_duplicate_id_does_not_cancel_existing_foreign_owner():
    engine, task, plan = engine_with_task(status=TaskStatus.UNREADY, kind=TaskType.PUT)
    engine._update_tasks = Mock()
    with pytest.raises(ValueError, match="already exists"):
        engine.put_match_batch([dict(task_id=7, token_ids=np.arange(4))])
    assert engine.tasks[7] is task
    plan.abort.assert_not_called()


def test_async_put_match_failure_rolls_back_partial_reservation_on_runtime_owner():
    from flexkv.kvmanager import KVManager
    from flexkv.prefetch.runtime import TaskRuntime

    engine, task, plan = engine_with_task(status=TaskStatus.UNREADY, kind=TaskType.PUT)
    engine.tasks.clear()
    engine.graph_to_task.clear()
    engine.prefetch_tasks.clear()
    task.prefetch_key = None
    engine.transfer_handles = []
    engine._update_tasks = Mock()
    engine._prefetch = NS(
        sessions={}, tick=lambda: None, next_wakeup=lambda poll_s: None,
    )
    engine._runtime = TaskRuntime(engine)

    def reserve(tokens, slots, **kwargs):
        assert engine._runtime.is_owner
        if kwargs["task_id"] == 8:
            raise RuntimeError("second queued match failed")
        task.task_id = kwargs["task_id"]
        engine.tasks[task.task_id] = task
        engine.graph_to_task[task.graph.graph_id] = task.task_id
        return task.task_id, np.ones(len(tokens), dtype=bool)

    engine._put_match_impl = reserve
    manager = KVManager.__new__(KVManager)
    manager.server_client_mode = False
    manager.kv_task_engine = engine
    manager.dp_client_id = 0
    engine._runtime.start()
    try:
        future = manager.put_match_batch_async([
            dict(task_id=7, token_ids=np.arange(4), namespace=["a"]),
            dict(task_id=8, token_ids=np.arange(4), namespace=["b"]),
        ])
        with pytest.raises(RuntimeError, match="second queued match"):
            future.result(timeout=2)
        plan.abort.assert_called_once_with()
        assert not engine.tasks and not engine.graph_to_task
    finally:
        engine._runtime.stop()
        engine._runtime = None


def test_match_rollback_attempts_every_owned_task_and_reports_remaining_ids():
    from flexkv.common.request import StoreReservationError

    engine, _, _ = engine_with_task(status=TaskStatus.UNREADY, kind=TaskType.PUT)
    engine.tasks.clear()
    engine._update_tasks = Mock()
    cancelled = []

    def reserve(tokens, slots, **kwargs):
        tid = kwargs["task_id"]
        if tid == 9:
            raise ValueError("last reservation failed")
        engine.tasks[tid] = NS(graph=NS(num_ops=1))
        return tid, np.ones(len(tokens), dtype=bool)

    def cancel(tid):
        cancelled.append(tid)
        if tid == 7:
            raise RuntimeError("abort temporarily failed")
        engine.tasks.pop(tid, None)

    engine._put_match_impl = reserve
    engine._cancel_task = cancel
    with pytest.raises(StoreReservationError, match="rollback failed") as error:
        engine.put_match_batch([
            dict(task_id=tid, token_ids=np.arange(4)) for tid in (7, 8, 9)
        ])
    assert cancelled == [7, 8, 9]
    assert error.value.task_ids == [7]
    assert set(engine.tasks) == {7}


@pytest.fixture
def held_launch():
    """Real FULL/SWA graph and task lifecycle, with DMA submission held in memory."""
    from flexkv.cache.cache_engine import GlobalCacheEngine
    from flexkv.common.transfer import TransferOp, TransferOpGraph, TransferType
    from flexkv.kvmanager import KVManager

    engine, task, plan = engine_with_task(status=TaskStatus.UNREADY, kind=TaskType.PUT)
    graph = TransferOpGraph()
    full = TransferOp(
        graph_id=graph.graph_id, transfer_type=TransferType.D2H,
        src_block_ids=np.array([0, 1]), dst_block_ids=np.array([10, 11]),
    )
    swa = TransferOp(
        graph_id=graph.graph_id, transfer_type=TransferType.D2H,
        src_block_ids=np.array([1]), dst_block_ids=np.array([20]), is_swa=True,
    )
    graph.add_transfer_op(full)
    graph.add_transfer_op(swa)
    task.graph = graph
    engine.graph_to_task = {graph.graph_id: task.task_id}
    engine.cache_config = NS(tokens_per_block=2)
    engine.cache_engine.slot_mapping_to_block_ids = GlobalCacheEngine.slot_mapping_to_block_ids
    engine._runtime = None
    engine._update_tasks = Mock()
    submitted = []
    engine.transfer_handles = [NS(submit_batch=submitted.extend, error=None)]
    manager = KVManager.__new__(KVManager)
    manager.server_client_mode = False
    manager.kv_task_engine = engine
    manager.dp_client_id = 0
    yield NS(
        engine=engine, manager=manager, task=task, plan=plan,
        graph=graph, full=full, swa=swa, submitted=submitted,
    )
    if engine._runtime is not None:
        engine._runtime.stop()
        engine._runtime = None
    engine.transfer_handles = []


@pytest.mark.parametrize("runtime_owner", [False, True])
def test_async_launch_acceptance_precedes_transfer_completion(held_launch, runtime_owner):
    import torch
    from flexkv.prefetch.runtime import TaskRuntime

    env = held_launch
    engine, manager = env.engine, env.manager
    entered, release = threading.Event(), threading.Event()
    blocked = None
    if runtime_owner:
        engine._prefetch = NS(
            sessions={}, expired={}, tick=lambda: None, next_wakeup=lambda poll_s: None,
        )
        engine._runtime = TaskRuntime(engine)
        engine._runtime.start()

        def hold_owner():
            entered.set()
            assert release.wait(3)

        blocked = engine._runtime.submit(hold_owner)
        assert entered.wait(2)
    try:
        future = manager.launch_async(
            7, torch.tensor([100, 101, 102, 103]),
            swa_slot_mappings=torch.tensor([200, 201, 202, 203]),
        )
        assert isinstance(future, Future)
        if runtime_owner:
            # The owner remains blocked until this submitting caller releases it.
            assert not future.done()
            assert env.task.status == TaskStatus.UNREADY and not env.submitted
        else:
            assert future.done()
    finally:
        release.set()
    if blocked is not None:
        blocked.result(timeout=2)
    assert future.result(timeout=2) == [7]
    assert env.task.status == TaskStatus.RUNNING
    assert engine.tasks[7] is env.task
    assert env.submitted == [env.graph]
    assert isinstance(env.task.slot_mapping, np.ndarray)
    np.testing.assert_array_equal(env.full.src_block_ids, [50, 51])
    np.testing.assert_array_equal(env.swa.src_block_ids, [101])
    assert env.graph.stage_local_task_id == 7
    env.plan.assert_not_called()
    env.plan.abort.assert_not_called()
    assert not engine.check_completed(7, completely=True)

    manager.cancel_async(7).result(timeout=2)
    assert env.task.cancel_requested and env.task.status == TaskStatus.RUNNING
    assert engine.tasks[7] is env.task
    env.plan.abort.assert_not_called()
    if runtime_owner:
        engine._runtime.call(engine._mark_completed, 7)
    else:
        engine._mark_completed(7)
    env.plan.abort.assert_called_once_with()
    assert not engine.tasks and not engine.graph_to_task and not engine.prefetch_tasks


@pytest.mark.parametrize("foreign_owner", [(1, "owner"), (0, "different-endpoint")])
def test_control_launch_rejects_entire_mixed_owner_batch(held_launch, foreign_owner):
    from flexkv.server.server import KVServer

    env = held_launch
    server = KVServer.__new__(KVServer)
    server.kv_task_engine = env.engine
    server._control_store_tasks = {7: (0, "owner"), 8: foreign_owner}
    request = NS(dp_client_id=0, reply_port="owner")
    payload = dict(task_ids=[7, 8], slot_mappings=[np.arange(4), np.arange(4)])
    with pytest.raises(ValueError, match="another control client"):
        server._dispatch_control(request, "launch", payload)
    assert server._control_store_tasks == {7: (0, "owner"), 8: foreign_owner}
    assert env.task.status == TaskStatus.UNREADY and env.task.slot_mapping is None
    assert not env.submitted
    env.plan.abort.assert_not_called()

    # The rejected request cannot prevent the genuine owner from retiring its pin.
    server._dispatch_control(request, "cancel", {"task_ids": [7]})
    env.plan.abort.assert_called_once_with()
    assert server._control_store_tasks == {8: foreign_owner}
    assert not env.engine.tasks


@pytest.mark.parametrize("failure_stage", ["mapping", "submission"])
def test_control_launch_failure_keeps_owner_until_cancel_and_reset(held_launch, failure_stage):
    from flexkv.server.server import KVServer

    env = held_launch
    server = KVServer.__new__(KVServer)
    server.kv_task_engine = env.engine
    server._control_store_tasks = {7: (0, "owner")}
    request = NS(dp_client_id=0, reply_port="owner")

    def fail(*args, **kwargs):
        raise RuntimeError("launch acceptance uncertain")

    if failure_stage == "mapping":
        env.engine.cache_engine.slot_mapping_to_block_ids = fail
    else:
        env.engine.transfer_handles[0].submit_batch = fail
    with pytest.raises(RuntimeError, match="launch acceptance uncertain"):
        server._dispatch_control(
            request, "launch", {"task_ids": [7], "slot_mappings": [np.arange(4)]},
        )
    assert server._control_store_tasks == {7: (0, "owner")}
    assert env.engine.tasks[7] is env.task
    env.plan.abort.assert_not_called()
    with pytest.raises(RuntimeError, match="requires drained transfers"):
        env.engine.reset_cache()
    env.engine.cache_engine.reset.assert_not_called()

    server._dispatch_control(request, "cancel", {"task_ids": [7]})
    assert not server._control_store_tasks
    if failure_stage == "submission":
        assert env.task.status == TaskStatus.RUNNING and env.task.cancel_requested
        assert env.engine.tasks[7] is env.task
        env.plan.abort.assert_not_called()
        with pytest.raises(RuntimeError, match="requires drained transfers"):
            env.engine.reset_cache()
        env.engine._mark_completed(7)
    env.plan.abort.assert_called_once_with()
    assert not env.engine.tasks and not env.engine.graph_to_task
    env.engine.reset_cache()
    env.engine.cache_engine.reset.assert_called_once_with()


@pytest.mark.parametrize("reject_second_handle", [False, True])
def test_control_launch_keeps_multiple_request_ids_through_partial_submission(
    held_launch, reject_second_handle,
):
    from dataclasses import replace
    from flexkv.common.transfer import TransferOp, TransferOpGraph, TransferType
    from flexkv.server.server import KVServer

    env = held_launch
    second_graph = TransferOpGraph()
    second_graph.add_transfer_op(TransferOp(
        graph_id=second_graph.graph_id, transfer_type=TransferType.D2H,
        src_block_ids=np.array([0, 1]), dst_block_ids=np.array([30, 31]),
    ))
    second_plan = Mock()
    second_task = replace(
        env.task, task_id=8, graph=second_graph, callback=second_plan,
        op_callback_dict={}, prefetch_key=None,
    )
    env.engine.tasks[8] = second_task
    env.engine.graph_to_task[second_graph.graph_id] = 8
    server = KVServer.__new__(KVServer)
    server.kv_task_engine = env.engine
    owners = {7: (0, "owner"), 8: (0, "owner")}
    server._control_store_tasks = owners.copy()
    request = NS(dp_client_id=0, reply_port="owner")
    payload = dict(
        task_ids=[7, 8], slot_mappings=[np.arange(4), np.arange(4, 8)],
    )

    def reject(graphs):
        # The first transfer handle has already accepted both independent graphs.
        assert env.submitted == [env.graph, second_graph]
        raise RuntimeError("second transfer handle rejected launch")

    if reject_second_handle:
        env.engine.transfer_handles.append(NS(submit_batch=reject))
        with pytest.raises(RuntimeError, match="second transfer handle"):
            server._dispatch_control(request, "launch", payload)
        assert server._control_store_tasks == owners
        server._dispatch_control(request, "cancel", {"task_ids": [7, 8]})
        assert env.task.cancel_requested and second_task.cancel_requested
    else:
        assert server._dispatch_control(request, "launch", payload) == [7, 8]
    assert not server._control_store_tasks
    assert set(env.engine.tasks) == {7, 8}
    assert env.task.status == second_task.status == TaskStatus.RUNNING
    assert env.engine.graph_to_task == {
        env.graph.graph_id: 7, second_graph.graph_id: 8,
    }
    assert env.submitted == [env.graph, second_graph]
    assert [graph.stage_local_task_id for graph in env.submitted] == [7, 8]
    env.plan.assert_not_called()
    second_plan.assert_not_called()
    env.plan.abort.assert_not_called()
    second_plan.abort.assert_not_called()
    with pytest.raises(RuntimeError, match="requires drained transfers"):
        env.engine.reset_cache()
    env.engine._mark_completed(7)
    assert env.engine.tasks[8] is second_task
    second_plan.abort.assert_not_called()
    env.engine._mark_completed(8)
    for plan in (env.plan, second_plan):
        if reject_second_handle:
            plan.abort.assert_called_once_with()
            plan.assert_not_called()
        else:
            plan.assert_called_once_with()
            plan.abort.assert_not_called()
    env.engine.reset_cache()
    assert not env.engine.tasks and not env.engine.graph_to_task


def test_control_owner_callback_submits_launch_without_waiting_for_server(held_launch, tmp_path):
    import torch
    import zmq
    from flexkv.kvmanager import KVManager
    from flexkv.server.client import KVDPClient
    from flexkv.server.server import KVServer

    env = held_launch
    address = "ipc://" + str(tmp_path / "launch-control")
    ready, entered, release, stop = (threading.Event() for _ in range(4))
    errors, endpoints = [], set()
    server = KVServer.__new__(KVServer)
    server.kv_task_engine = env.engine
    server._prefetch_clients = {}
    server.client_manager = NS(client_dict={0: None})
    # The native held graph is already reserved; the real control path records
    # its endpoint owner when returning the match, then launches that same graph.
    def match_reserved(**kwargs):
        assert callback_gate.wait(3)
        return 7, np.ones(4, dtype=bool)

    env.engine.get_match = match_reserved

    def serve():
        context = zmq.Context()
        server.context = context
        recv = context.socket(zmq.PULL)
        recv.bind(address)
        ready.set()
        try:
            while not stop.is_set():
                if not recv.poll(2):
                    continue
                request = recv.recv_pyobj()
                endpoints.add(request.reply_port)
                if any(action == "launch" for action, _ in request.payload["calls"]):
                    entered.set()
                    assert release.wait(3)
                server._handle_control(request)
        except BaseException as exc:
            errors.append(exc)
        finally:
            for reply in getattr(server, "_control_replies", {}).values():
                reply.close(0)
            recv.close(0)
            context.term()

    worker = threading.Thread(target=serve)
    client = KVDPClient.__new__(KVDPClient)
    client.server_recv_port, client.dp_client_id = address, 0
    client._task_id_lock = threading.Lock()
    client._task_id_counter, client._task_id_range = 0, (0, 10000000)
    manager = KVManager.__new__(KVManager)
    manager.server_client_mode, manager.dp_client = True, client
    callback_result = Future()
    callback_gate = threading.Event()

    def launch_from_owner(completed):
        try:
            assert threading.current_thread() is client._control_owner.thread
            callback_result.set_result(manager.launch_async(
                completed.result()[0], torch.tensor([100, 101, 102, 103]),
                swa_slot_mappings=torch.tensor([200, 201, 202, 203]),
            ))
        except BaseException as exc:
            callback_result.set_exception(exc)

    worker.start()
    try:
        assert ready.wait(2)
        # Hold the match reply until callback attachment, so the callback must
        # execute on the persistent owner rather than this submitting thread.
        match = manager.get_match_async(np.arange(4))
        owner = client._control_owner
        match.add_done_callback(launch_from_owner)
        callback_gate.set()
        launch = callback_result.result(timeout=2)
        assert entered.wait(2)
        assert not launch.done()
        assert env.task.status == TaskStatus.UNREADY and not env.submitted
        assert server._control_store_tasks[7] == (0, next(iter(endpoints)))
        release.set()
        assert launch.result(timeout=2) == [7]
        assert client._control_owner is owner and len(endpoints) == 1
        assert not server._control_store_tasks
        assert env.task.status == TaskStatus.RUNNING
        assert env.submitted == [env.graph]
        np.testing.assert_array_equal(env.full.src_block_ids, [50, 51])
        np.testing.assert_array_equal(env.swa.src_block_ids, [101])
        env.plan.assert_not_called()
        env.plan.abort.assert_not_called()
        env.engine._mark_completed(7)
        env.plan.assert_called_once_with()
    finally:
        callback_gate.set()
        release.set()
        try:
            client.close_control()
        finally:
            stop.set()
            worker.join(2)
    assert not worker.is_alive() and not errors
