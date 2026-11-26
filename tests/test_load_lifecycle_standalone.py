"""CPU-only lifecycle contracts executing production methods with fake transports."""

import __future__
import ast
import contextlib
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONNECTOR = "flexkv/integration/sglang/connector.py"


def load_method(path, owner, name, **values):
    tree = ast.parse((ROOT / path).read_text())
    nodes = tree.body if owner is None else next(
        node.body for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    method = next(node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name)
    method.decorator_list = []
    namespace = dict(time=time, replace=replace, np=np, logger=logging.getLogger(__name__),
                     flexkv_logger=logging.getLogger(__name__), _SGLANG_REQ_ID_UNSET=object())
    namespace.update(values)
    exec(compile(ast.Module(body=[method], type_ignores=[]), path, "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


comm_tree = ast.parse((ROOT / "flexkv/integration/sglang/comm.py").read_text())
channel_nodes = [node for node in comm_tree.body if
                 isinstance(node, ast.ClassDef) and node.name == "FlexKVScatterChannel" or
                 isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                 {"_SCATTER_CHANNEL_OFFSETS", "_SCATTER_CHANNEL_TYPES"} for t in node.targets)]
channel_ns = dict(Enum=Enum)
exec(compile(ast.Module(body=channel_nodes, type_ignores=[]), "comm.py", "exec"), channel_ns)
Channel = channel_ns["FlexKVScatterChannel"]
status_value = load_method(CONNECTOR, None, "_status_value")


@dataclass
class Context:
    task_id: int = 11
    start_ns: int = 0
    operation: str = "load"


@dataclass
class Prepared:
    task_id: int
    producer_id: int
    slot_mapping: object
    swa_slot_mapping: object
    context: Context
    start_ns: int


values = dict(_status_value=status_value, FlexKVScatterChannel=Channel,
              _PreparedLayerwiseLoad=Prepared, CMD_LAYERWISE=3)
retrieve = load_method(CONNECTOR, "FlexKVConnector", "retrieve_kv", **values)
start = load_method(CONNECTOR, "FlexKVConnector", "start_load_kv_layerwise", **values)
drain = load_method(CONNECTOR, "FlexKVConnector", "drain_launched_loads", **values)
lookup = load_method(CONNECTOR, "FlexKVConnector", "lookup_kv", **values)


def response(status):
    return SimpleNamespace(status=status)


def engine(leader=True, payload=None):
    context = Context()
    obj = SimpleNamespace(
        _sync_ctx=SimpleNamespace(is_sync_leader=leader, needs_sync=True,
                                  should_send_slot_mapping_to_remote=False, world_rank=0,
                                  all_reduce_min=lambda x: x,
                                  scatter=Mock(side_effect=lambda value, **kw: value if leader else payload)),
        kv_manager=SimpleNamespace(launch=Mock(return_value=[71]),
                                   wait=Mock(return_value={71: response("success")}), cancel=Mock()),
        _pending_lookups={"r": 11}, _pending_lookup_contexts={"r": context},
        _launched_load_tids=[], _launched_load_contexts={}, _prepared_layerwise_loads={},
        _layerwise_batch_launch_error=None,
        _pop_context=lambda *args: context, _log_cache_op=Mock(),
        _to_cpu_int64=lambda x: x, _build_swa_slot_mapping=lambda x: None,
        enable_layerwise=True,
        layer_done_counter=Mock(consumer_index=0, num_counters=2,
                                events=[Mock(pending_transfers=lambda: 1)]),
    )
    obj.layer_done_counter.update_producer.return_value = 0
    obj.layer_done_counter.expected_transfer_count.return_value = 1
    return obj


def slots():
    return SimpleNamespace(numel=lambda: 8)


def test_sync_load_tracks_actual_ids_and_waits_whole_completion():
    obj = engine()
    assert retrieve(obj, "r", slots()) == 8
    assert obj.kv_manager.launch.call_args.kwargs["as_batch"] is False
    obj.kv_manager.wait.assert_called_once_with([71], timeout=30.0, completely=True)
    assert obj._launched_load_tids == [] and obj._launched_load_contexts == {}
    assert obj._sync_ctx.scatter.call_args.kwargs["channel"] == Channel.LOAD_COMPLETION


@pytest.mark.parametrize("status,retained", [("failed", False), ("timeout", True),
                                             ("not_found", True), (None, True)])
def test_sync_load_failure_is_broadcast_and_unknown_writer_retained(status, retained):
    obj = engine()
    obj.kv_manager.wait.return_value = {71: response(status)}
    with pytest.raises(RuntimeError, match="synchronous restore failed"):
        retrieve(obj, "r", slots())
    assert (71 in obj._launched_load_tids) is retained
    payload = obj._sync_ctx.scatter.call_args.args[0]
    follower = engine(False, payload)
    with pytest.raises(RuntimeError, match="synchronous restore failed"):
        retrieve(follower, "r", slots())
    follower.kv_manager.launch.assert_not_called()
    assert payload["loaded"] == 0


@pytest.mark.parametrize("actual_ids", [None, [72, 73]])
def test_sync_partial_launch_preserves_drain_ids_and_blocks_unknown_reset(actual_ids):
    obj = engine()
    exc = RuntimeError("partial launch")
    if actual_ids:
        exc.launched_task_ids = actual_ids
    obj.kv_manager.launch.side_effect = exc
    with pytest.raises(RuntimeError):
        retrieve(obj, "r", slots())
    assert obj._launched_load_tids == (actual_ids or [11])
    payload = obj._sync_ctx.scatter.call_args.args[0]
    follower = engine(False, payload)
    with pytest.raises(RuntimeError):
        retrieve(follower, "r", slots())
    assert bool(obj._layerwise_batch_launch_error) is (actual_ids is None)
    assert bool(follower._layerwise_batch_launch_error) is (actual_ids is None)


def test_poll_exception_retains_launched_writer():
    obj = engine()
    obj.kv_manager.wait.side_effect = RuntimeError("connection lost")
    with pytest.raises(RuntimeError):
        retrieve(obj, "r", slots())
    assert obj._launched_load_tids == [71]


@pytest.mark.parametrize("result", [None, [], [71]])
def test_immediate_layerwise_launch_keeps_main_preflight_and_tracks_result(result):
    obj = engine()
    obj.kv_manager.launch.return_value = result
    if result:
        assert start(obj, "r", slots()) == (8, 0)
        assert not obj._prepared_layerwise_loads
        assert obj._layerwise_batch_launch_error is None
    else:
        with pytest.raises(RuntimeError, match="drainable batch ID"):
            start(obj, "r", slots())
        assert "r" in obj._prepared_layerwise_loads
        assert obj._layerwise_batch_launch_error
    obj.layer_done_counter.register_task.assert_called_once_with(11, 0)
    obj.layer_done_counter.rollback_prepared_transfer.assert_not_called()
    assert obj._launched_load_tids == (result or [])


@pytest.mark.parametrize("actual_ids", [None, [72]])
def test_immediate_layerwise_partial_launch_is_retained_on_all_ranks(actual_ids):
    obj = engine()
    exc = RuntimeError("partial layerwise launch")
    if actual_ids:
        exc.launched_task_ids = actual_ids
    obj.kv_manager.launch.side_effect = exc
    with pytest.raises(RuntimeError, match="partial layerwise"):
        start(obj, "r", slots())
    messages = [call.args[0] for call in obj._sync_ctx.scatter.call_args_list]
    follower = engine(False)
    follower._sync_ctx.scatter.side_effect = messages
    with pytest.raises(RuntimeError, match="partial layerwise"):
        start(follower, "r", slots())
    for item in (obj, follower):
        assert item._prepared_layerwise_loads["r"].task_id == 11
        assert item._layerwise_batch_launch_error
        item.layer_done_counter.rollback_prepared_transfer.assert_not_called()
    assert obj._launched_load_tids == (actual_ids or [])
    follower.kv_manager.launch.assert_not_called()


def test_deferred_layerwise_still_uses_main_prepared_path():
    obj = engine()
    assert start(obj, "r", slots(), defer_launch=True) == (8, 0)
    obj.kv_manager.launch.assert_not_called()
    assert obj._prepared_layerwise_loads["r"].task_id == 11
    assert obj._layerwise_batch_launch_error is None


@pytest.mark.parametrize("status", ["not_found", "timeout", "cancelled", None])
def test_drain_never_accepts_unknown_completion(status):
    obj = engine()
    obj._launched_load_tids = [71]
    obj.kv_manager.wait.return_value = {71: response(status)}
    drain(obj)
    assert obj._launched_load_tids == [71]


def test_drain_retires_known_terminals_but_reports_failure():
    obj = engine()
    obj._launched_load_tids = [71, 72, 73]
    obj.kv_manager.wait.return_value = {71: response("success"), 72: response("failed")}
    with pytest.raises(RuntimeError, match="72"):
        drain(obj)
    assert obj._launched_load_tids == [73]


@pytest.mark.parametrize("hit,page_size", [(0, 1), (3, 4)])
def test_unused_lookup_retires_terminal_metadata(hit, page_size):
    obj = engine()
    obj._new_op_context = lambda *args: Context()
    obj._as_numpy_mask = lambda x: x
    obj._release_prefetch_result = Mock()
    obj._swa_kv_pool = None
    obj.page_size = page_size
    obj.kv_manager.get_match = Mock(return_value=(11, np.arange(8) < hit))
    assert lookup(obj, list(range(8)), None, rid="new") == (11, 0)
    obj.kv_manager.cancel.assert_called_once_with([11])
    assert "new" not in obj._pending_lookups


def test_completion_channel_does_not_renumber_existing_protocol():
    offsets = channel_ns["_SCATTER_CHANNEL_OFFSETS"]
    assert offsets[Channel.LOAD_COMPLETION] == 11
    assert sorted(offsets.values()) == list(range(1, 12))
    assert channel_ns["_SCATTER_CHANNEL_TYPES"][Channel.LOAD_COMPLETION] is dict
    assert offsets[Channel.LAYERWISE_BATCH] == 10
    assert offsets[Channel.STORE_COMPLETION] == 3


Status = Enum("Status", "UNREADY READY RUNNING SUCCESS FAILED")
reject = load_method("flexkv/kvtask.py", "KVTaskEngine", "reject_launch", TaskStatus=Status)
server_launch = load_method("flexkv/server/server.py", "KVServer", "_handle_launch_task_request")


@pytest.mark.parametrize("status", list(Status))
@pytest.mark.parametrize("graph", [None, object()])
def test_rejection_never_short_circuits_running_or_terminal_tasks(status, graph):
    task = SimpleNamespace(status=status, graph=graph, transfer_failed=False, shed_heavy_resources=Mock())
    obj = SimpleNamespace(tasks={11: task}, _terminal_tasks={}, _fail_task=Mock())
    reject(obj, [11, 99], -1, "rejected")
    if status not in (Status.UNREADY, Status.READY):
        assert task.status is status and not task.transfer_failed
        obj._fail_task.assert_not_called()
        task.shed_heavy_resources.assert_not_called()
    elif graph is None:
        assert task.status is Status.FAILED
        task.shed_heavy_resources.assert_called_once()
        assert 11 in obj._terminal_tasks
    else:
        assert task.transfer_failed
        obj._fail_task.assert_called_once_with(11)


def test_rejection_targets_existing_merged_batch():
    task = SimpleNamespace(status=Status.READY, graph=object(), transfer_failed=False)
    obj = SimpleNamespace(tasks={71: task}, _fail_task=Mock())
    reject(obj, [11], 71, "rejected")
    obj._fail_task.assert_called_once_with(71)
    assert set(obj.tasks) == {71}


def test_server_rejection_preserves_original_exception():
    original = RuntimeError("submission rejected")
    manager = SimpleNamespace(launch_tasks=Mock(side_effect=original), reject_launch=Mock())
    req = SimpleNamespace(task_ids=[11], slot_mappings=[], swa_slot_mappings=[], as_batch=False,
                          batch_id=-1, layerwise_transfer=False, counter_id=0)
    with pytest.raises(RuntimeError) as result:
        server_launch(SimpleNamespace(kv_task_engine=manager), req)
    assert result.value is original
    manager.reject_launch.assert_called_once_with([11], -1, "submission rejected")


@pytest.mark.parametrize("phase", ["constructor", "initialize", "selector", "signal"])
def test_worker_startup_cleanup_is_explicit_and_keeps_main_exit_status(phase, monkeypatch):
    manager = Mock()
    selector = Mock()
    factory = Mock(return_value=manager)
    failure = SystemExit(0) if phase == "signal" else RuntimeError(phase)
    if phase == "constructor":
        factory.side_effect = failure
    elif phase in ("initialize", "signal"):
        manager.initialize_transfer_engine.side_effect = failure
    selectors = SimpleNamespace(DefaultSelector=Mock(side_effect=failure))
    method = load_method("flexkv/transfer_manager.py", "TransferManagerInterProcessHandle", "_process_worker",
                         reset_inherited_numa_binding=Mock(), signal=Mock(), os=os, sys=sys,
                         traceback=traceback, contextlib=contextlib, TransferManager=factory,
                         selectors=selectors)
    monkeypatch.setenv("MPI4PY_RC_INITIALIZE", "false")
    command, result, errors = Mock(), Mock(), Mock()
    with pytest.raises(SystemExit) as exited:
        method(None, None, None, command, result, "unused", Mock(), Mock(), errors)
    assert exited.value.code == (0 if phase == "signal" else 1)
    command.close.assert_called_once()
    result.close.assert_called_once()
    errors.close.assert_called_once()
    if phase == "constructor":
        manager.shutdown.assert_not_called()
    else:
        manager.shutdown.assert_called_once()
    source = (ROOT / "flexkv/transfer_manager.py").read_text()
    assert "if 'sel' in locals()" not in source
    assert "if 'transfer_manager' in locals()" not in source


def real_task_engine():
    from dataclasses import field
    from types import MethodType
    tree = ast.parse((ROOT / "flexkv/kvtask.py").read_text())
    declarations = [node for node in tree.body if isinstance(node, ast.ClassDef)
                    and node.name in {"KVTask", "TaskStatus", "TaskType"}]
    namespace = dict(__name__=__name__, dataclass=dataclass, field=field, Enum=Enum, time=time)
    exec(compile(ast.Module(body=declarations, type_ignores=[]), "kvtask.py", "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    task_status, task_type, task_cls = (namespace[key] for key in ("TaskStatus", "TaskType", "KVTask"))
    logger = Mock(is_enabled_for=lambda level: False)
    response_status = Enum("ResponseStatus", "NOTFOUND UNREADY SUCCESS FAILED TIMEOUT")
    deps = dict(TaskStatus=task_status, TaskType=task_type, KVTask=task_cls,
                nvtx=Mock(), flexkv_logger=logger, logging=logging,
                GLOBAL_CONFIG_FROM_ENV=SimpleNamespace(enable_layerwise_transfer=True),
                KVResponse=SimpleNamespace, KVResponseStatus=response_status,
                convert_to_response_status=lambda value: response_status.FAILED
                if value is task_status.FAILED else response_status.SUCCESS)
    obj = SimpleNamespace(tasks={}, graph_to_task={}, _terminal_tasks={}, prefetch_tasks={},
                          transfer_handles=[Mock()], tracer=Mock(),
                          cache_config=SimpleNamespace(tokens_per_block=64),
                          cache_engine=Mock(), _abort_task_plans=Mock(), _log_task_terminal=Mock(),
                          _update_tasks=Mock())
    manager_methods = ("set_slot_mappings", "_set_slot_mapping_impl", "check_task_ready", "_fail_task",
                       "_release_task")
    engine_methods = ("reject_launch", "launch_tasks", "_wait_impl")
    for owner, names in (("KVTaskManager", manager_methods), ("KVTaskEngine", engine_methods)):
        for name in names:
            setattr(obj, name, MethodType(load_method("flexkv/kvtask.py", owner, name, **deps), obj))
    obj.check_completed = lambda tid, **kw: obj.tasks[tid].is_completed()
    for tid in (11, 12):
        graph = Mock(graph_id=tid, num_ops=1)
        task = task_cls(tid, task_type.GET, tid, False, task_status.UNREADY,
                        np.arange(8), None, None, graph, np.ones(8, dtype=bool), None, {})
        obj.tasks[tid] = task
        obj.graph_to_task[tid] = tid
    return obj, task_status, response_status


@pytest.mark.parametrize("batch", [False, True])
def test_real_launch_rejection_does_not_leave_unsubmitted_running_task(batch):
    obj, status, responses = real_task_engine()
    req = SimpleNamespace(task_ids=[11, 12], slot_mappings=[np.arange(8)], swa_slot_mappings=None,
                          as_batch=batch, batch_id=71 if batch else -1,
                          layerwise_transfer=batch, counter_id=0)
    with pytest.raises(ValueError, match="corresponding slot mapping"):
        server_launch(SimpleNamespace(kv_task_engine=obj), req)
    obj.transfer_handles[0].submit_batch.assert_not_called()
    assert obj.tasks[11].status is status.FAILED
    assert obj.tasks[12].status is status.FAILED
    waited = [71] if batch else [11, 12]
    result = obj._wait_impl(waited, timeout=0, completely=True)
    assert all(item.status is responses.FAILED for item in result.values())
    assert all(tid not in obj.tasks for tid in waited)


def test_real_merge_failure_restores_only_pre_submit_readiness():
    obj, status, responses = real_task_engine()

    def merge(*args):
        obj.check_task_ready(11)
        obj.check_task_ready(12)
        raise RuntimeError("graph planning failed")

    obj.merge_to_batch_kvtask = merge
    req = SimpleNamespace(task_ids=[11, 12], slot_mappings=[np.arange(8), np.arange(8)],
                          swa_slot_mappings=None, as_batch=True, batch_id=71,
                          layerwise_transfer=True, counter_id=0)
    with pytest.raises(RuntimeError, match="graph planning failed"):
        server_launch(SimpleNamespace(kv_task_engine=obj), req)
    obj.transfer_handles[0].submit_batch.assert_not_called()
    assert obj._abort_task_plans.call_count == 2
    result = obj._wait_impl([71], timeout=0, completely=True)
    assert result[71].status is responses.FAILED


def test_real_partial_submission_keeps_main_writer_ownership():
    obj, status, responses = real_task_engine()
    obj.transfer_handles.append(Mock())
    obj.transfer_handles[1].submit_batch.side_effect = RuntimeError("second handle failed")
    req = SimpleNamespace(task_ids=[11, 12], slot_mappings=[np.arange(8), np.arange(8)],
                          swa_slot_mappings=None, as_batch=False, batch_id=-1,
                          layerwise_transfer=False, counter_id=0)
    with pytest.raises(RuntimeError, match="second handle failed"):
        server_launch(SimpleNamespace(kv_task_engine=obj), req)
    obj.transfer_handles[0].submit_batch.assert_called_once()
    obj._abort_task_plans.assert_not_called()
    assert all(task.status is status.RUNNING and task.graph is not None for task in obj.tasks.values())
    result = obj._wait_impl([11, 12], timeout=0, completely=True)
    assert all(item.status is responses.TIMEOUT for item in result.values())


def test_rejected_batch_does_not_claim_running_children_are_drained():
    obj, status, responses = real_task_engine()
    obj.tasks[11].status = status.RUNNING
    obj.reject_launch([11, 12], 71, "duplicate request")
    assert obj.tasks[11].status is status.RUNNING
    assert 71 not in obj.tasks


@pytest.mark.parametrize("mask,success,expected", [
    ([0, 0, 1, 1, 1, 1, 1, 1, 0, 0], 1, [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]),
    ([0, 0, 1, 1, 1, 1, 1, 0, 0, 0], 9, [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]),
    ([1, 1, 1, 1], 0, [0, 0, 0, 0]),
    ([0, 0, 0, 0], 3, [0, 0, 0, 0]),
])
def test_return_mask_narrowing_keeps_last_true_bound(mask, success, expected):
    narrow = load_method("flexkv/kvtask.py", "KVTaskManager", "_narrow_return_mask_to_prefix_blocks")
    task = SimpleNamespace(return_mask=np.array(mask, dtype=np.bool_))
    engine = SimpleNamespace(cache_config=SimpleNamespace(tokens_per_block=2))
    narrow(engine, task, success)
    assert task.return_mask.tolist() == [bool(v) for v in expected]
