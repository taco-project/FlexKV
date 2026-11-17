"""PUT source fences and storage-tail ownership without GPU or network I/O."""

from collections import OrderedDict
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from flexkv.common.transfer import (
    CompletedOp,
    TransferOp,
    TransferOpGraph,
    TransferType,
    add_virtual_op_for_multiple_finished_ops,
)
from flexkv.integration.sglang.connector import FlexKVConnector
from flexkv.kvtask import KVTask, KVTaskEngine, TaskStatus, TaskType
from flexkv.transfer.scheduler import TransferScheduler

pytestmark = pytest.mark.unit


class _Handle:
    def __init__(self):
        self.completed = []

    def wait(self, _timeout):
        completed, self.completed = self.completed, []
        return completed

    def shutdown(self):
        pass


def _case(swa=True, handles=2, storage=True):
    graph = TransferOpGraph()

    def op(kind, is_swa=False):
        result = TransferOp(
            graph_id=graph.graph_id,
            transfer_type=kind,
            src_block_ids=np.array([1], dtype=np.int64),
            dst_block_ids=np.array([2], dtype=np.int64),
            is_swa=is_swa,
        )
        graph.add_transfer_op(result)
        return result

    full = op(TransferType.D2H)
    if storage:
        remote = op(TransferType.H2REMOTE)
        graph.add_dependency(remote.op_id, full.op_id)
    sources = [full]
    if swa:
        sources.append(op(TransferType.D2H, is_swa=True))
        if storage:
            swa_remote = op(TransferType.H2REMOTE, is_swa=True)
            graph.add_dependency(swa_remote.op_id, sources[-1].op_id)
    graph, end_id = add_virtual_op_for_multiple_finished_ops(
        graph, [source.op_id for source in sources], 0
    )
    plan = Mock()
    task = KVTask(
        task_id=17, task_type=TaskType.PUT, task_end_op_id=end_id,
        task_end_op_finished=False, status=TaskStatus.RUNNING,
        token_ids=np.array([1], dtype=np.int64),
        slot_mapping=np.array([1], dtype=np.int64), token_mask=None,
        graph=graph, return_mask=np.array([True]), callback=plan,
        op_callback_dict={},
    )
    engine = KVTaskEngine.__new__(KVTaskEngine)
    engine.tasks = {17: task}
    engine.graph_to_task = {graph.graph_id: 17}
    engine.transfer_handles = [_Handle() for _ in range(handles)]
    engine.required_completed_count = handles
    engine.uncompleted_ops = {}
    engine.uncompleted_op_results = {}
    engine.uncompleted_graphs = {}
    engine._terminal_tasks = OrderedDict()
    engine._log_task_terminal = Mock()
    engine.max_inflight_prefetches = 0

    class Manager:
        def __init__(self):
            self.waits = []

        def wait(self, task_ids, *, timeout, completely):
            self.waits.append((list(task_ids), timeout, completely))
            return engine._wait_impl(task_ids, timeout=0, completely=completely)

    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector.kv_manager = Manager()
    connector._sync_ctx = SimpleNamespace(is_sync_leader=True, needs_sync=False)
    connector._inflight_stores = {"request": 17}
    connector._inflight_store_contexts = {"request": SimpleNamespace(task_id=17)}
    connector._store_tail_tasks = {}
    connector._log_cache_op = Mock()
    connector._new_op_context = Mock()
    return connector, engine, task, plan, sources


def _finish(engine, op_id, handle=None, failed=False):
    task = next(iter(engine.tasks.values()))
    completion = CompletedOp(task.graph.graph_id, op_id, failed=failed)
    targets = engine.transfer_handles if handle is None else [engine.transfer_handles[handle]]
    for target in targets:
        target.completed.append(completion)


@pytest.mark.parametrize("swa", [False, True])
@pytest.mark.parametrize("handles", [1, 2])
@pytest.mark.parametrize("tail_fails", [False, True])
def test_source_release_precedes_storage_tail_but_preserves_its_buffers(swa, handles, tail_fails):
    connector, engine, task, plan, sources = _case(swa, handles)
    scheduler = TransferScheduler()
    scheduler.add_transfer_graph(task.graph)
    scheduler.schedule([])
    assert connector.check_completed_stores() == []

    # A Full completion cannot release the source while SWA is still reading.
    _, ready = scheduler.schedule([sources[0]])
    if swa:
        assert task.task_end_op_id not in [item.op_id for item in ready]
        assert connector.check_completed_stores() == []
        _, ready = scheduler.schedule([sources[1]])
        assert task.task_end_op_id in [item.op_id for item in ready]

    # The task-end must be acknowledged by every TP/CP/PP transfer handle.
    _finish(engine, task.task_end_op_id, handle=0)
    if handles > 1:
        assert connector.check_completed_stores() == []
        _finish(engine, task.task_end_op_id, handle=1)
    assert connector.check_completed_stores() == ["request"]
    assert not connector._inflight_stores
    assert 17 in connector._store_tail_tasks
    assert engine.tasks[17] is task
    assert task.status == TaskStatus.RUNNING
    assert task.callback is plan and task.graph is not None
    assert not plan.called and not plan.abort.called

    # Idle sweeps still advance the tail after the last request has returned.
    assert connector.check_completed_stores() == []
    assert 17 in connector._store_tail_tasks
    _finish(engine, -1, failed=tail_fails)
    assert connector.check_completed_stores() == []
    assert not connector._store_tail_tasks
    assert not engine.tasks and not engine.graph_to_task
    if tail_fails:
        plan.abort.assert_called_once_with()
        assert not plan.called
    else:
        plan.assert_called_once_with()
        assert not plan.abort.called
    assert any(not completely for _, _, completely in connector.kv_manager.waits)
    assert any(completely for _, _, completely in connector.kv_manager.waits)


def _reset_state(connector):
    for name in [
        "_pending_lookups", "_pending_lookup_contexts", "_prefetch_contexts",
        "_ongoing_prefetches", "_inflight_loads", "_completed_layerwise",
        "_launched_load_contexts",
    ]:
        setattr(connector, name, {})
    connector._launched_load_tids = []
    connector.layer_done_counter = None


def test_reset_fences_released_source_tails_and_retains_them_on_timeout():
    connector, engine, task, plan, _ = _case()
    _finish(engine, task.task_end_op_id)
    assert connector.check_completed_stores() == ["request"]
    _reset_state(connector)

    with pytest.raises(RuntimeError, match="not safely drained"):
        connector.reset()
    assert 17 in connector._store_tail_tasks
    assert engine.tasks[17] is task
    assert not plan.called and not plan.abort.called
    assert connector.kv_manager.waits[-1] == ([17], 30.0, True)

    _finish(engine, -1)
    connector.reset()
    assert not connector._store_tail_tasks and not engine.tasks
    plan.assert_called_once_with()


def test_old_storage_tail_cannot_release_a_reused_request_id():
    connector, engine, task, _, _ = _case()
    _finish(engine, task.task_end_op_id)
    assert connector.check_completed_stores() == ["request"]
    connector._inflight_stores["request"] = 18
    connector._inflight_store_contexts["request"] = SimpleNamespace(task_id=18)
    original_wait = connector.kv_manager.wait

    def wait(task_ids, *, timeout, completely):
        if task_ids == [18]:
            return {}
        return original_wait(task_ids, timeout=timeout, completely=completely)

    connector.kv_manager.wait = wait
    _finish(engine, -1)
    assert connector.check_completed_stores() == []
    assert not connector._store_tail_tasks
    assert connector._inflight_stores == {"request": 18}


def test_wait_store_keeps_follower_source_ownership_until_collective_sweep():
    connector, _, _, _, _ = _case()
    connector._sync_ctx.is_sync_leader = False
    connector.kv_manager = None
    assert not connector.wait_store("request", timeout=0)
    assert connector._inflight_stores == {"request": 17}


def test_source_failure_waits_for_every_handle_to_drain():
    connector, engine, task, plan, _ = _case()
    _finish(engine, -1, handle=0, failed=True)
    assert connector.check_completed_stores() == []
    assert connector._inflight_stores == {"request": 17}
    assert not plan.abort.called
    _finish(engine, -1, handle=1)
    assert connector.check_completed_stores() == ["request"]
    assert not connector._inflight_stores and not connector._store_tail_tasks
    plan.abort.assert_called_once_with()


def test_tail_poll_error_preserves_engine_and_connector_ownership():
    connector, engine, task, plan, _ = _case()
    _finish(engine, task.task_end_op_id)
    assert connector.check_completed_stores() == ["request"]
    connector.kv_manager.wait = Mock(side_effect=OSError("temporary control failure"))
    assert connector.check_completed_stores() == []
    assert 17 in connector._store_tail_tasks and engine.tasks[17] is task
    assert not plan.called and not plan.abort.called


def test_cpu_only_l2_store_does_not_add_a_tail_poll():
    connector, engine, task, _, _ = _case(storage=False)
    connector.cache_config = SimpleNamespace(enable_ssd=False, enable_remote=False)
    _finish(engine, task.task_end_op_id)
    assert connector.check_completed_stores() == ["request"]
    assert not connector._store_tail_tasks
    before = len(connector.kv_manager.waits)
    assert connector.check_completed_stores() == []
    assert len(connector.kv_manager.waits) == before
