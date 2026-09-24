"""CPU task ownership regressions; no DMA or worker process is launched."""

from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np
import pytest

from flexkv.kvtask import KVTask, KVTaskEngine, KVTaskManager, TaskStatus, TaskType
from flexkv.common.transfer import CompletedOp


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


def test_constructor_uses_explicit_task_ownership():
    import flexkv.kvtask as module

    config = NS(enable_cpu=True, enable_remote=True, enable_gds=False, enable_kv_sharing=False, enable_nixl=False)
    model = NS(use_trtllm_subprocess=False, nnodes=1)
    with patch.object(module, "GlobalCacheEngine"), patch.object(module, "TransferManagerHandle"):
        manager = KVTaskManager(model, config)
        assert type(manager.tasks) is dict and type(manager.prefetch_tasks) is dict


@pytest.mark.parametrize("failed", [False, True])
def test_cancel_waits_for_every_transfer_handle_before_abort(failed):
    engine, task, plan = engine_with_task()
    writer_commit = Mock()
    task.op_callback_dict[11] = writer_commit
    engine.required_completed_count = 2
    first = [CompletedOp(13, 11), CompletedOp(13, -1)]
    second = [CompletedOp(13, 11), CompletedOp(13, -1, failed=failed)]
    engine.transfer_handles = [
        NS(wait=Mock(side_effect=[first, []])),
        NS(wait=Mock(side_effect=[[], second])),
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
