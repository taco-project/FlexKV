"""Native imports, real task-engine dispatch, ZMQ control and rank agreement."""

from array import array
from concurrent.futures import Future, ThreadPoolExecutor
from collections import OrderedDict
from types import SimpleNamespace as NS
from unittest.mock import Mock
import threading
import time
import uuid

import numpy as np
import pytest
import torch

pytest.importorskip("flexkv.c_ext")
import zmq

from flexkv.common.config import ModelConfig, CacheConfig, SWAPoolConfig
from flexkv.integration.sglang.connector import FlexKVConnector
from flexkv.kvtask import KVTaskEngine, KVTaskManager, TaskStatus
from flexkv.kvmanager import KVManager
from flexkv.prefetch.coordinator import PrefetchCoordinator
from flexkv.prefetch.runtime import TaskRuntime
from flexkv.prefetch.types import PrefetchOptions, PrefetchCapabilities
from flexkv.server.client import KVDPClient
from flexkv.server.server import KVServer
from flexkv.server.request import ControlRequest, WaitRequest
from test_coordinator import Backend, Clock


@pytest.mark.parametrize(
    "enabled,explicit,server_policy,expected",
    [
        (True, "wait_complete", "timeout", False),
        (True, None, "wait_complete", False),
        (True, "timeout", "wait_complete", True),
        (True, None, "timeout", True),
        (True, "best_effort", "wait_complete", True),
        (False, "timeout", "timeout", False),
    ],
)
def test_connector_routes_policy_before_runtime_creation(
    monkeypatch, enabled, explicit, server_policy, expected
):
    import flexkv.integration.sglang.connector as module

    cfg = CacheConfig(
        enable_chunked_prefetch=enabled,
        prefetch_options={"policy": explicit} if explicit is not None else {},
    )
    config = NS(
        cache_config=cfg,
        model_config=ModelConfig(),
        post_init_from_sglang_config=Mock(return_value=NS()),
    )
    monkeypatch.setattr(module.FlexKVConfig, "from_env", lambda: config)

    class ConfigResolved(Exception):
        pass

    # Stop before distributed/GPU setup; routing must already be finalized
    # before the downstream KVManager sees its cache configuration.
    monkeypatch.setattr(module.FlexKVComm, "__init__", Mock(side_effect=ConfigResolved))
    connector = FlexKVConnector.__new__(FlexKVConnector)
    with pytest.raises(ConfigResolved):
        connector.__init__(
            sgl_model_config=NS(),
            server_args=NS(hicache_storage_prefetch_policy=server_policy),
            page_size=64,
            kvcache=None,
            tp_rank=0,
            dp_rank=0,
            pp_rank=0,
            attn_cp_rank=0,
        )
    assert connector._chunked_prefetch is expected
    assert cfg.enable_chunked_prefetch is expected
    if enabled:
        assert cfg.prefetch_options["policy"] == (explicit or server_policy)


@pytest.mark.parametrize("dedup", [False, True])
def test_connector_keeps_snapshot_budget_when_resolving_indexer_group(monkeypatch, dedup):
    import flexkv.integration.sglang.connector as module

    monkeypatch.setenv("FLEXKV_DEDUP_INDEXER_GROUP", "1" if dedup else "0")
    cfg = CacheConfig(swa=SWAPoolConfig(enabled=True))
    config = NS(
        cache_config=cfg,
        model_config=ModelConfig(),
        post_init_from_sglang_config=Mock(return_value=NS()),
    )
    monkeypatch.setattr(module.FlexKVConfig, "from_env", lambda: config)
    monkeypatch.setattr(
        module, "FlexKVComm", lambda **kwargs: NS(all_reduce_min=lambda value: value)
    )
    kvcache = NS(
        kv_buffer=[torch.empty((8, 1, 4))],
        index_k_with_scale_buffer=[
            torch.empty((8, 4), dtype=torch.uint8),
            torch.empty((0, 4), dtype=torch.uint8),
            torch.empty((8, 4), dtype=torch.uint8),
        ],
        skip_topk_layers=[False, True, False],
        swa_kv_pool=NS(kv_buffer=[
            torch.empty((8, 4, 3), dtype=torch.float16),
            torch.empty((8, 4, 5), dtype=torch.uint8),
        ]),
    )

    class GeometryResolved(Exception):
        pass

    # Both geometries must survive constructor setup before capacity sizing
    # and KVManager creation; no GPU registration is needed for this check.
    def check_geometry(self, buffers, indexer_group):
        assert buffers[0] is kvcache.kv_buffer[0]
        assert indexer_group.layer_indices == ((0, 2) if dedup else (0, 1, 2))
        assert cfg.swa.snapshot_bytes == 44
        raise GeometryResolved

    monkeypatch.setattr(
        FlexKVConnector, "_apply_layer_groups_for_cache_sizing", check_geometry
    )
    with pytest.raises(GeometryResolved):
        FlexKVConnector(
            sgl_model_config=NS(), server_args=NS(), page_size=64, kvcache=kvcache,
            tp_rank=0, dp_rank=0, pp_rank=0, attn_cp_rank=0,
        )


@pytest.fixture
def engine():
    e = KVTaskEngine.__new__(KVTaskEngine)
    e.transfer_handles = []
    e.tasks = {}
    e._terminal_tasks = OrderedDict()
    e.graph_to_task = {}
    e._update_tasks = lambda timeout=0: None
    e._wait_impl = lambda *args, **kwargs: {}  # a legacy request remains running
    e._prefetch_options = PrefetchOptions()
    e._prefetch = PrefetchCoordinator(Backend(), clock=Clock())
    e._runtime = TaskRuntime(e)
    e._runtime.start()
    yield e
    e._runtime.stop()
    e._runtime = None  # __del__ has no owned transfer process


def test_runtime_polls_running_store_tail_but_not_held_plan(engine):
    def check():
        engine.tasks[7] = NS(status=TaskStatus.RUNNING)
        engine.graph_to_task[107] = 7
        assert engine._next_runtime_wakeup(0.002) == 0.002
        engine.tasks[7].status = TaskStatus.UNREADY
        assert engine._next_runtime_wakeup(0.002) is None
        engine.tasks.clear()
        engine.graph_to_task.clear()

    engine._runtime.call(check)


def test_retained_result_expires_without_foreground_polling(engine):
    def start():
        engine._prefetch = PrefetchCoordinator(
            Backend(target=0), result_ttl_s=0.03)
        return engine._prefetch.start([1], PrefetchOptions())

    handle = engine._runtime.call(start)
    with engine._runtime.changed:
        assert engine._runtime.changed.wait_for(
            lambda: handle in engine._prefetch.expired, timeout=2)
    assert len(engine._prefetch.backend.releases) == 1


def test_capability_gate_runs_before_any_resource_allocation(monkeypatch):
    init = Mock(side_effect=AssertionError("allocated before validating"))
    monkeypatch.setattr(KVTaskManager, "__init__", init)
    with pytest.raises(ValueError, match="Mooncake"):
        KVTaskEngine(ModelConfig(), CacheConfig(enable_chunked_prefetch=True))
    init.assert_not_called()


@pytest.mark.parametrize(
    "field,value",
    [
        ("prefetch_max_sessions", 0),
        ("prefetch_max_pinned_bytes", -1),
        ("prefetch_result_ttl_s", float("nan")),
    ],
)
def test_invalid_runtime_limit_rejected_before_allocation(monkeypatch, field, value):
    init = Mock(side_effect=AssertionError("allocated before validating"))
    monkeypatch.setattr(KVTaskManager, "__init__", init)
    cfg = CacheConfig(
        enable_chunked_prefetch=True,
        enable_ssd=False,
        use_mooncake_store_backend=True,
        mooncake_store_config_path="unused.json",
    )
    setattr(cfg, field, value)
    with pytest.raises(ValueError, match="limits"):
        KVTaskEngine(ModelConfig(), cfg)
    init.assert_not_called()


def test_real_zmq_control_is_independent_of_pending_legacy_wait(engine, tmp_path):
    address = "ipc://" + str(tmp_path / "server")
    engine._runtime.call(lambda: setattr(engine._prefetch.backend, "target", 0))
    ready, stop = threading.Event(), threading.Event()
    errors = []
    legacy_responses = []
    server = KVServer.__new__(KVServer)
    server.kv_task_engine = engine
    server._prefetch_clients = {}
    server._pending_waits = []
    server.client_manager = NS(
        client_dict={0: None, 1: None},
        get_zmq=lambda cid: NS(send_pyobj=legacy_responses.append),
    )

    def serve():
        ctx = zmq.Context()
        server.context = ctx
        sock = ctx.socket(zmq.PULL)
        sock.bind(address)
        ready.set()
        try:
            while not stop.is_set():
                if sock.poll(2):
                    req = sock.recv_pyobj()
                    if isinstance(req, ControlRequest):
                        server._handle_control(req)
                    else:
                        server._handle_wait_request(req)
                server._drain_pending_waits()
        except BaseException as exc:
            errors.append(exc)
        finally:
            sock.close(0)
            ctx.term()

    worker = threading.Thread(target=serve)
    worker.start()
    assert ready.wait(2)
    client = KVDPClient.__new__(KVDPClient)
    client.server_recv_port, client.dp_client_id = address, 0
    other = KVDPClient.__new__(KVDPClient)
    other.server_recv_port, other.dp_client_id = address, 1
    ctx = zmq.Context()
    send = ctx.socket(zmq.PUSH)
    send.connect(address)
    try:
        send.send_pyobj(
            WaitRequest(dp_client_id=0, wait_task_ids=[99], wait_timeout=10)
        )
        before = time.monotonic()
        assert client.control("capabilities") == PrefetchCapabilities(
            partial_swa_checkpoints=True
        )
        assert time.monotonic() - before < 2
        assert server._pending_waits and not legacy_responses
        with ThreadPoolExecutor(max_workers=8) as pool:
            caps = list(
                pool.map(lambda _: client.control("capabilities"), range(32))
            )
        assert all(c.protocol_version == 1 for c in caps)
        assert len(server._control_replies) == 1
        assert client._control_owner.thread.is_alive()
        h = client.control(
            "start",
            token_ids=np.arange(32, dtype=np.int64),
            options=PrefetchOptions(policy="best_effort"),
        )
        with pytest.raises(RuntimeError, match="another client"):
            other.control("progress", handles=[h])
        snapshot = client.control("progress", handles=[h], demand_handles=[h])[
            h
        ]
        assert snapshot.stop_reason in ("demand", "complete")
        client.control("release", handle=h)
        client.control("release", handle=h)
        assert not legacy_responses
    finally:
        client.close_control()
        other.close_control()
        assert not client._control_owner.thread.is_alive()
        assert not other._control_owner.thread.is_alive()
        assert not server._control_replies
        stop.set()
        worker.join(2)
        send.close(0)
        ctx.term()
    server.kv_task_engine = NS(shutdown=lambda: None)  # fixture owns runtime teardown
    assert not worker.is_alive() and not errors


def test_failed_runtime_cannot_starve_control_receiver(engine):
    server = KVServer.__new__(KVServer)
    replies = []
    server.kv_task_engine = NS(
        _runtime=NS(call=Mock(side_effect=RuntimeError("unhealthy"))),
        _wait_impl=Mock(),
        shutdown=lambda: None,
    )
    server._pending_waits = [(WaitRequest(0, [7], 10), time.monotonic() + 10, {})]
    server.client_manager = NS(
        client_dict={0: None}, get_zmq=lambda cid: NS(send_pyobj=replies.append)
    )
    server._drain_pending_waits()
    assert not server._pending_waits
    assert len(replies) == 1 and replies[0].error_msg == "unhealthy"


def test_running_legacy_cancel_retains_callbacks_until_terminal(engine):
    from flexkv.kvtask import TaskStatus
    from flexkv.common.transfer import CompletedOp
    from test_kvtask_lifecycle import _make_task, _FakeGraph

    callback = Mock()
    task = _make_task(
        task_id=7, graph=_FakeGraph(107), status=TaskStatus.RUNNING, callback=callback
    )
    engine._runtime.call(lambda: engine.tasks.update({7: task}))
    engine._runtime.call(lambda: engine.graph_to_task.update({107: 7}))
    engine.cancel_tasks([7])
    assert engine._runtime.call(lambda: engine.tasks[7]) is task
    assert task.request_returned and task.status == TaskStatus.RUNNING
    callback.assert_not_called()
    engine._get_completed_ops = lambda timeout: [CompletedOp.completed_graph(107)]
    engine._runtime.call(KVTaskManager._update_tasks, engine, 0)
    callback.assert_not_called()
    callback.abort.assert_called_once_with()
    assert 7 not in engine.tasks and 107 not in engine.graph_to_task


def test_terminal_legacy_reaper_preserves_active_ownership(engine):
    from collections import OrderedDict
    from flexkv.kvtask import TaskStatus
    from test_kvtask_lifecycle import _make_task, _FakeGraph

    def install():
        engine.tasks = {
            1: _make_task(
                task_id=1, graph=_FakeGraph(101), status=TaskStatus.COMPLETED
            ),
            2: _make_task(task_id=2, graph=_FakeGraph(102), status=TaskStatus.RUNNING),
        }
        engine.graph_to_task = {102: 2}
        engine._terminal_tasks = OrderedDict({1: time.monotonic() - 2000})
        engine._reap_completed_tasks()
        return set(engine.tasks)

    assert engine._runtime.call(install) == {2}


def test_existing_positional_cache_config_is_preserved():
    from flexkv.common.config import UserConfig

    assert CacheConfig(32).tokens_per_block == 32
    assert UserConfig(8).cpu_cache_gb == 8


def test_user_configuration_propagates_prefetch_limits(monkeypatch):
    from flexkv.common.config import (
        load_user_config_from_env,
        update_default_config_from_user_config,
        RankInfo,
    )

    monkeypatch.setenv("FLEXKV_ENABLE_CHUNKED_PREFETCH", "1")
    monkeypatch.setenv(
        "FLEXKV_PREFETCH_OPTIONS", '{"policy":"timeout","timeout_budget_s":0.02}'
    )
    monkeypatch.setenv("FLEXKV_PREFETCH_MAX_SESSIONS", "17")
    user = load_user_config_from_env()
    cfg = CacheConfig()
    update_default_config_from_user_config(RankInfo(ModelConfig()), cfg, user)
    assert cfg.enable_chunked_prefetch and cfg.prefetch_max_sessions == 17
    assert cfg.prefetch_options["policy"] == "timeout"


@pytest.mark.parametrize("namespace", [None, ["tenant-a"], ["tenant-b", "adapter"]])
def test_connector_forwards_namespace_to_chunked_session(namespace):
    c = FlexKVConnector.__new__(FlexKVConnector)
    c._prefetch_enabled = c._chunked_prefetch = True
    c._chunked_prefetch_options = PrefetchOptions(policy="best_effort")
    c._prefetch_sessions, c._prefetch_result_sessions = {}, {}
    c._prefetch_loaded_tokens, c._prefetch_loaded_spans = {}, {}
    c._swa_kv_pool = None
    c.kv_manager = Mock()
    c.kv_manager.prefetch_control_async.return_value = Future()
    c._sync_ctx = NS(is_sync_leader=True, needs_sync=False)
    assert c.supports_cache_namespace is True
    assert c.prefetch_async("r", array("q", [1, 2, 3, 4]), namespace=namespace, candidate_start_token=2) == 0
    call = c.kv_manager.prefetch_control_async.call_args
    assert call.args == ("start",)
    assert call.kwargs["token_ids"].tolist() == [1, 2, 3, 4]
    assert call.kwargs["options"].candidate_start_token == 2
    assert call.kwargs["namespace"] == namespace


def queued_connector(manager):
    c = FlexKVConnector.__new__(FlexKVConnector)
    c._prefetch_enabled = c._chunked_prefetch = c._queued_prefetch = True
    c._chunked_prefetch_options = PrefetchOptions(policy="timeout")
    c._prefetch_sessions, c._prefetch_result_sessions = {}, {}
    c._prepared_lookups = {}
    c._prefetch_loaded_tokens, c._prefetch_loaded_spans = {}, {}
    c._pending_lookups, c._pending_lookup_contexts = {}, {}
    c._swa_kv_pool = None
    c._sync_ctx = NS(is_sync_leader=True, needs_sync=False)
    c.kv_manager = manager
    return c


def native_manager(engine):
    manager = KVManager.__new__(KVManager)
    manager.server_client_mode = False
    manager.kv_task_engine = engine
    return manager


def test_pending_native_admission_is_not_miss_and_late_cancel_releases(engine):
    engine._runtime.call(lambda: setattr(engine._prefetch.backend, "target", 0))
    gate, entered = threading.Event(), threading.Event()

    def hold_owner():
        entered.set()
        assert gate.wait(3)

    blocked = engine._runtime.submit(hold_owner)
    assert entered.wait(2)
    c = queued_connector(native_manager(engine))
    try:
        assert c.prefetch_async("old", np.arange(4), namespace=["a"]) >= 0
        assert c.check_prefetch_progress("old") is False
        c.cancel_prefetch("old")
        # Reusing the request ID must not publish the old admission's result.
        assert c.prefetch_async("old", np.arange(8), namespace=["b"]) >= 0
        assert c.check_prefetch_progress("old") is False
        c.cancel_prefetch("old")
    finally:
        gate.set()
    blocked.result(timeout=2)
    c._drain_queued_prefetch()
    assert c._prefetch_sessions == {}
    assert not engine._runtime.call(lambda: bool(engine._prefetch.sessions))
    assert len(engine._prefetch.backend.releases) == 2


def test_progress_batches_admitted_requests_and_preserves_loaded_spans(engine):
    from test_coordinator import finish

    c = queued_connector(native_manager(engine))
    matches = [Future(), Future()]
    c.kv_manager.get_match_async = Mock(side_effect=matches)
    c.prefetch_async("a", np.arange(32), namespace=["a"])
    c.prefetch_async("b", np.arange(32), namespace=["b"])
    admissions = [entry["admission"] for entry in c._prefetch_sessions.values()]
    handles = [future.result(timeout=2) for future in admissions]
    # Complete actual coordinator chunks rather than manufacture snapshots.
    def complete():
        engine._prefetch.tick()
        for chunk in list(engine._prefetch.backend.sent):
            finish(engine._prefetch, chunk)
        engine._prefetch.tick()

    engine._runtime.call(complete)
    assert c.check_prefetch_progress("a") is False
    future, entries = c._prefetch_progress_batch
    assert {entry["handle"] for entry in entries} == set(handles)
    future.result(timeout=2)
    assert c.check_prefetch_progress("a") is True
    assert c.check_prefetch_progress("b") is True
    assert c.kv_manager.get_match_async.call_count == 2
    for match in matches:
        match.set_result((-1, np.zeros(32, dtype=np.bool_)))
    assert c.pop_prefetch_loaded_span("a") == (20, 0)
    assert c.pop_prefetch_loaded_span("b") == (20, 0)
    c._release_prefetch_result("a")
    c._release_prefetch_result("b")
    c._drain_queued_prefetch()
    assert not engine._prefetch.backend.allocations
    assert len(engine._prefetch.backend.releases) == 2


def test_release_and_reset_drain_retains_allocations_until_completion(engine):
    from test_coordinator import finish

    c = queued_connector(native_manager(engine))
    c.prefetch_async("a", np.arange(32))
    handle = c._prefetch_sessions["a"]["admission"].result(timeout=2)
    engine._runtime.call(engine._prefetch.tick)
    c.cancel_prefetch("a")
    for future in c._prefetch_retirements:
        future.result(timeout=2)
    assert engine._runtime.call(lambda: bool(engine._prefetch.backend.allocations))
    with ThreadPoolExecutor(max_workers=1) as pool:
        drain = pool.submit(c._drain_queued_prefetch)
        assert not drain.done()
        def complete():
            for chunk in list(engine._prefetch.backend.sent):
                finish(engine._prefetch, chunk)
            engine._prefetch.tick()
        engine._runtime.call(complete)
        drain.result(timeout=2)
    assert not engine._prefetch.backend.allocations
    assert engine._runtime.call(lambda: engine._prefetch.snapshot(handle).terminal)
    assert len(engine._prefetch.backend.releases) == 1


def test_delayed_ipc_admission_batches_queue_and_releases_cancelled_late_sessions(engine, tmp_path):
    engine._runtime.call(lambda: setattr(engine._prefetch.backend, "target", 0))
    address = "ipc://" + str(tmp_path / "delayed-control")
    ready, entered, gate, stop = (threading.Event() for _ in range(4))
    errors, batch_sizes, endpoints = [], [], set()
    server = KVServer.__new__(KVServer)
    server.kv_task_engine = engine
    server._prefetch_clients = {}
    server.client_manager = NS(client_dict={0: None})

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
                req = recv.recv_pyobj()
                endpoints.add(req.reply_port)
                batch_sizes.append(len(req.payload["calls"]))
                if not entered.is_set():
                    entered.set()
                    assert gate.wait(3)
                server._handle_control(req)
        except BaseException as exc:
            errors.append(exc)
        finally:
            for reply in getattr(server, "_control_replies", {}).values():
                reply.close(0)
            recv.close(0)
            context.term()

    server_thread = threading.Thread(target=serve)
    server_thread.start()
    assert ready.wait(2)
    client = KVDPClient.__new__(KVDPClient)
    client.server_recv_port, client.dp_client_id = address, 0
    manager = KVManager.__new__(KVManager)
    manager.server_client_mode, manager.dp_client = True, client
    c = queued_connector(manager)
    try:
        # The server cannot reply until gate opens. These scheduler operations
        # must return while admission is still blocked.
        assert c.prefetch_async("a", np.arange(8), namespace=["a"]) == 0
        assert entered.wait(2)
        assert c.check_prefetch_progress("a") is False
        c.cancel_prefetch("a")
        for rid in ("b", "c", "d"):
            assert c.prefetch_async(rid, np.arange(8), namespace=[rid]) >= 0
            assert c.check_prefetch_progress(rid) is False
            c.cancel_prefetch(rid)
        gate.set()
        c._drain_queued_prefetch()
        assert not engine._runtime.call(lambda: bool(engine._prefetch.sessions))
        assert len(engine._prefetch.backend.releases) == 4
        assert len(endpoints) == 1
        assert max(batch_sizes) >= 3
    finally:
        gate.set()
        client.close_control()
        stop.set()
        server_thread.join(2)
    assert not server_thread.is_alive() and not errors
    assert not client._control_owner.thread.is_alive()
    server.kv_task_engine = NS(shutdown=lambda: None)


def test_native_async_store_reserves_only_on_existing_task_runtime(engine):
    manager = native_manager(engine)
    manager.dp_client_id = 3
    gate, entered = threading.Event(), threading.Event()
    observed = []

    def hold_owner():
        entered.set()
        assert gate.wait(3)

    def reserve(requests, dp_client_id):
        assert engine._runtime.is_owner
        observed.append((requests, dp_client_id))
        return [(17, np.ones(4, dtype=bool), True)]

    engine.put_match_batch = reserve
    blocked = engine._runtime.submit(hold_owner)
    assert entered.wait(2)
    try:
        future = manager.put_match_batch_async([
            dict(token_ids=torch.arange(4), namespace=["tenant"]),
        ])
        assert not future.done()
        assert not observed
    finally:
        gate.set()
    blocked.result(timeout=2)
    assert future.result(timeout=2)[0][0] == 17
    requests, client_id = observed[0]
    assert client_id == 3
    assert isinstance(requests[0]["token_ids"], np.ndarray)
    assert requests[0]["namespace"] == ["tenant"]


def test_legacy_native_store_future_preserves_scheduler_thread_ownership():
    owner = threading.get_ident()

    def reserve(requests, dp_client_id):
        assert threading.get_ident() == owner
        assert dp_client_id == 4 and requests[0]["namespace"] == ["tenant"]
        return [(17, np.ones(4, dtype=bool), True)]

    manager = native_manager(NS(_runtime=None, put_match_batch=reserve))
    manager.dp_client_id = 4
    future = manager.put_match_batch_async([
        dict(token_ids=np.arange(4), namespace=["tenant"]),
    ])
    assert future.done() and future.result()[0][0] == 17


def test_delayed_store_rpc_shares_control_owner_and_close_cancels_late_reservations(tmp_path):
    from flexkv.common.request import KVResponse, KVResponseStatus, StoreReservationError

    address = "ipc://" + str(tmp_path / "store-control")
    ready, entered, gate, stop = (threading.Event() for _ in range(4))
    errors, endpoints, batches, cancelled = [], set(), [], []
    tasks = {}
    server = KVServer.__new__(KVServer)
    server._prefetch_clients = {}
    server.client_manager = NS(client_dict={0: None})

    def reserve(requests, dp_client_id):
        assert threading.get_ident() == server_thread.ident
        assert dp_client_id == 0
        if any(request["namespace"] == ["fail"] for request in requests):
            raise RuntimeError("queued store match failed")
        results = []
        for request in requests:
            tid = request["task_id"]
            tasks[tid] = request
            if request["namespace"] == ["cleanup"]:
                raise StoreReservationError("rollback incomplete", [tid])
            results.append((tid, np.ones(len(request["token_ids"]), dtype=bool), True))
        return results

    def cancel(task_ids):
        cancelled.extend(task_ids)
        for tid in task_ids:
            tasks.pop(tid, None)

    server.kv_task_engine = NS(
        _runtime=None, tasks=tasks, put_match_batch=reserve,
        cancel_tasks=cancel, try_wait=lambda task_ids: {},
        wait=lambda task_ids, **kwargs: {
            tid: KVResponse(KVResponseStatus.NOTFOUND, tid, None) for tid in task_ids
        },
        shutdown=lambda: None,
    )

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
                batches.append(len(request.payload["calls"]))
                if not entered.is_set():
                    entered.set()
                    assert gate.wait(3)
                server._handle_control(request)
        except BaseException as exc:
            errors.append(exc)
        finally:
            for reply in getattr(server, "_control_replies", {}).values():
                reply.close(0)
            recv.close(0)
            context.term()

    server_thread = threading.Thread(target=serve)
    server_thread.start()
    assert ready.wait(2)
    client = KVDPClient.__new__(KVDPClient)
    client.server_recv_port, client.dp_client_id = address, 0
    client._task_id_lock = threading.Lock()
    client._task_id_counter, client._task_id_range = 0, (0, 10000000)
    manager = KVManager.__new__(KVManager)
    manager.server_client_mode, manager.dp_client = True, client
    request = dict(token_ids=np.arange(4, dtype=np.int64), namespace=["tenant"])
    try:
        first = manager.put_match_batch_async([request, request])
        assert entered.wait(2)
        assert not first.done() and not tasks
        failed = manager.put_match_batch_async([dict(request, namespace=["fail"])])
        cleanup = manager.put_match_batch_async([dict(request, namespace=["cleanup"])])
        later = manager.put_match_batch_async([request])
        assert not failed.done() and not later.done()
        owner = client._control_owner
        gate.set()
        first_matches = first.result(timeout=2)
        with pytest.raises(RuntimeError, match="queued store match failed"):
            failed.result(timeout=2)
        with pytest.raises(StoreReservationError, match="rollback incomplete") as error:
            cleanup.result(timeout=2)
        later_matches = later.result(timeout=2)
        task_ids = [match[0] for match in first_matches + later_matches]
        task_ids.extend(error.value.task_ids)
        assert set(tasks) == set(task_ids)
        assert client._control_owner is owner
        client.close_control()
        assert set(cancelled) == set(task_ids)
        assert not tasks and not server._control_store_tasks
        assert len(endpoints) == 1 and max(batches) >= 2
        assert not owner.thread.is_alive()
    finally:
        gate.set()
        client.close_control()
        stop.set()
        server_thread.join(2)
    assert not server_thread.is_alive() and not errors


def test_control_transport_failure_completes_current_and_queued_store_futures(monkeypatch):
    from flexkv.server import control

    entered, gate = threading.Event(), threading.Event()
    context, send, reply = Mock(), Mock(), Mock()

    def receive():
        entered.set()
        assert gate.wait(3)
        return NS(request_id=-1, result=[], error=None)

    reply.recv_pyobj.side_effect = receive
    monkeypatch.setattr(control.zmq, "Context", lambda: context)
    monkeypatch.setattr(
        control, "get_zmq_socket",
        lambda context, kind, port, bind: reply if bind else send,
    )
    owner = control.ControlClient("unused", 0)
    try:
        first = owner.submit("put_match_batch", requests=[])
        assert entered.wait(2)
        second = owner.submit("put_match_batch", requests=[])
        third = owner.submit("put_match_batch", requests=[])
        assert not first.done() and not second.done() and not third.done()
        gate.set()
        for future in (first, second, third):
            with pytest.raises(control.ControlTransportError, match="sequence mismatch") as error:
                future.result(timeout=2)
            assert error.value.retain_slots
        with pytest.raises(control.ControlTransportError, match="sequence mismatch"):
            owner.close()
    finally:
        gate.set()
        owner.thread.join(2)
    assert not owner.thread.is_alive()
    send.close.assert_called_once_with(linger=0)
    reply.close.assert_called_once_with(linger=0)
    context.term.assert_called_once_with()


@pytest.mark.parametrize("partial_tail", [0, 3])
def test_async_get_holds_real_swa_plan_until_callback_cancel(engine, partial_tail):
    from flexkv.common.block import SequenceMeta
    from test_kvtask_lifecycle import _engine, _put, _tokens, TPB

    tokens = _tokens(31)
    query_tokens = np.concatenate((tokens, np.arange(partial_tail, dtype=np.int64)))
    query_tokens.setflags(write=False)
    query_mask = np.ones(query_tokens.size, dtype=np.bool_)
    query_mask.setflags(write=False)
    engine.cache_engine = _engine()
    engine.cache_config = CacheConfig(tokens_per_block=TPB)
    engine.task_id_lock = threading.Lock()
    engine.task_id_counter = 100
    engine.tracer = NS(trace_request=lambda **kwargs: None)
    engine._runtime.call(_put, engine.cache_engine, tokens)
    manager = native_manager(engine)
    manager.dp_client_id = 0
    entered, gate = threading.Event(), threading.Event()

    def hold_owner():
        entered.set()
        assert gate.wait(3)

    blocked = engine._runtime.submit(hold_owner)
    assert entered.wait(2)
    try:
        future = manager.get_match_async(
            query_tokens, token_mask=query_mask, cpu_only=True, swa_aware=True,
        )
        assert not future.done()
    finally:
        gate.set()
    blocked.result(timeout=2)
    task_id, mask = future.result(timeout=2)
    np.testing.assert_array_equal(
        mask, np.arange(query_tokens.size) < tokens.size,
    )
    assert query_mask.all()

    def pinned():
        seq = SequenceMeta(tokens, TPB)
        seq.gen_hashes()
        node = engine.cache_engine.cpu_cache_engine.match(seq).last_swa_node
        return node.swa_lock_ref

    assert engine._runtime.call(pinned) > 0
    assert engine._runtime.call(lambda: engine.tasks[task_id].status) == TaskStatus.UNREADY
    plan = engine._runtime.call(lambda: engine.tasks[task_id].callback)

    def fail_abort():
        raise RuntimeError("plan abort rejected")

    engine._runtime.call(
        lambda: setattr(engine.tasks[task_id], "callback", NS(abort=fail_abort))
    )
    with pytest.raises(RuntimeError, match="plan abort rejected"):
        manager.cancel_async(task_id).result(timeout=2)
    assert engine._runtime.call(pinned) > 0
    assert engine._runtime.call(lambda: engine.tasks[task_id].status) == TaskStatus.UNREADY
    engine._runtime.call(lambda: setattr(engine.tasks[task_id], "callback", plan))
    # A completion callback runs on TaskRuntime itself. It must enqueue rather
    # than wait on that same thread, while the GET pin survives until cleanup.
    cancellation = Future()
    entered.clear()
    gate.clear()
    blocked = engine._runtime.submit(hold_owner)
    assert entered.wait(2)
    trigger = engine._runtime.submit(lambda: task_id)

    def retire(completed):
        assert engine._runtime.is_owner
        assert pinned() > 0
        cancellation.set_result(manager.cancel_async(completed.result()))

    trigger.add_done_callback(retire)
    gate.set()
    blocked.result(timeout=2)
    cancellation.result(timeout=2).result(timeout=2)
    assert engine._runtime.call(pinned) == 0
    assert engine._runtime.call(lambda: task_id not in engine.tasks)


def test_async_get_and_cancel_failures_remain_future_errors(engine):
    manager = native_manager(engine)
    manager.dp_client_id = 3
    observed = []

    def fail_match(**kwargs):
        assert engine._runtime.is_owner
        observed.append(kwargs)
        raise ValueError("reservation rejected")

    def fail_cancel(task_ids):
        assert engine._runtime.is_owner
        assert task_ids == [17]
        raise RuntimeError("cleanup rejected")

    engine.get_match = fail_match
    engine.cancel_tasks = fail_cancel
    with pytest.raises(ValueError, match="reservation rejected"):
        manager.get_match_async(
            torch.arange(4), torch.tensor([False, False, True, True]),
            cpu_only=True, swa_aware=True, namespace=["tenant", "adapter"],
        ).result(timeout=2)
    kwargs = observed[0]
    assert kwargs["dp_client_id"] == 3
    assert kwargs["cpu_only"] and kwargs["swa_aware"]
    assert kwargs["namespace"] == ["tenant", "adapter"]
    np.testing.assert_array_equal(kwargs["token_ids"], np.arange(4))
    np.testing.assert_array_equal(kwargs["token_mask"], [False, False, True, True])
    with pytest.raises(RuntimeError, match="cleanup rejected"):
        manager.cancel_async(17).result(timeout=2)
    engine._runtime.stop()
    with pytest.raises(RuntimeError, match="stopped"):
        manager.get_match_async(np.arange(4)).result(timeout=2)
    with pytest.raises(RuntimeError, match="stopped"):
        manager.cancel_async(17).result(timeout=2)


def test_legacy_native_get_and_cancel_futures_preserve_owner():
    owner = threading.get_ident()
    calls = []

    def match(**kwargs):
        assert threading.get_ident() == owner
        assert kwargs["dp_client_id"] == 4
        calls.append("get")
        return 17, np.ones(4, dtype=bool)

    def cancel(task_ids):
        assert threading.get_ident() == owner
        assert task_ids == [17]
        calls.append("cancel")

    manager = native_manager(NS(_runtime=None, get_match=match, cancel_tasks=cancel))
    manager.dp_client_id = 4
    future = manager.get_match_async(np.arange(4))
    assert future.done() and future.result()[0] == 17
    cleanup = manager.cancel_async(17)
    assert cleanup.done() and cleanup.result() is None
    assert calls == ["get", "cancel"]


def test_delayed_get_rpc_callback_cancel_errors_and_close_ownership(tmp_path):
    from flexkv.common.request import KVResponse, KVResponseStatus

    address = "ipc://" + str(tmp_path / "get-control")
    ready, entered, gate, stop = (threading.Event() for _ in range(4))
    errors, tasks, cancelled = [], {}, []
    fail_cancel = set()
    server = KVServer.__new__(KVServer)
    server._prefetch_clients = {}
    server.client_manager = NS(client_dict={0: None, 1: None})

    def match(**kwargs):
        assert threading.get_ident() == server_thread.ident
        assert kwargs["dp_client_id"] == 0
        assert kwargs["cpu_only"] and kwargs["swa_aware"]
        np.testing.assert_array_equal(kwargs["token_ids"], np.arange(4))
        np.testing.assert_array_equal(kwargs["token_mask"], [False, False, True, True])
        if kwargs["namespace"] == ["fail"]:
            raise ValueError("GET reservation rejected")
        mask = np.array([False, False, True, True])
        if kwargs["namespace"] == ["miss"]:
            mask[:] = False
        else:
            assert kwargs["namespace"] == ["tenant"]
            tasks[kwargs["task_id"]] = kwargs
        return kwargs["task_id"], mask

    def cancel(task_ids):
        assert threading.get_ident() == server_thread.ident
        if any(tid in fail_cancel for tid in task_ids):
            raise RuntimeError("GET cleanup rejected")
        cancelled.extend(task_ids)
        for tid in task_ids:
            tasks.pop(tid, None)

    server.kv_task_engine = NS(
        _runtime=None, tasks=tasks, get_match=match, cancel_tasks=cancel,
        try_wait=lambda task_ids: {},
        wait=lambda task_ids, **kwargs: {
            tid: KVResponse(KVResponseStatus.NOTFOUND, tid, None) for tid in task_ids
        },
        shutdown=lambda: None,
    )

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
                if not entered.is_set():
                    entered.set()
                    assert gate.wait(3)
                server._handle_control(request)
        except BaseException as exc:
            errors.append(exc)
        finally:
            for reply in getattr(server, "_control_replies", {}).values():
                reply.close(0)
            recv.close(0)
            context.term()

    server_thread = threading.Thread(target=serve)
    server_thread.start()
    assert ready.wait(2)
    client = KVDPClient.__new__(KVDPClient)
    client.server_recv_port, client.dp_client_id = address, 0
    client._task_id_lock = threading.Lock()
    client._task_id_counter, client._task_id_range = 0, (0, 10000000)
    other = KVDPClient.__new__(KVDPClient)
    other.server_recv_port, other.dp_client_id = address, 1
    manager = KVManager.__new__(KVManager)
    manager.server_client_mode, manager.dp_client = True, client
    request = dict(
        token_ids=np.arange(4), token_mask=np.array([False, False, True, True]),
        cpu_only=True, swa_aware=True, namespace=["tenant"],
    )
    callback_cleanup = Future()

    def retire(completed):
        assert threading.current_thread() is client._control_owner.thread
        callback_cleanup.set_result(manager.cancel_async(completed.result()[0]))

    try:
        first = manager.get_match_async(**request)
        assert entered.wait(2)
        assert not first.done() and not tasks
        first.add_done_callback(retire)
        failed = manager.get_match_async(**dict(request, namespace=["fail"]))
        miss = manager.get_match_async(**dict(request, namespace=["miss"]))
        held = manager.get_match_async(**request)
        gate.set()
        callback_cleanup.result(timeout=2).result(timeout=2)
        assert first.result()[0] in cancelled
        with pytest.raises(RuntimeError, match="GET reservation rejected"):
            failed.result(timeout=2)
        miss_id, miss_mask = miss.result(timeout=2)
        assert not miss_mask.any()
        task_id, mask = held.result(timeout=2)
        assert mask.sum() == 2 and task_id in tasks
        assert miss_id not in server._control_store_tasks
        with pytest.raises(RuntimeError, match="another control client"):
            other.cancel_async([task_id]).result(timeout=2)
        fail_cancel.add(task_id)
        with pytest.raises(RuntimeError, match="GET cleanup rejected"):
            manager.cancel_async(task_id).result(timeout=2)
        assert task_id in tasks and task_id in server._control_store_tasks
        fail_cancel.clear()
        client.close_control()
        assert task_id in cancelled and not tasks and not server._control_store_tasks
        with pytest.raises(RuntimeError, match="closed"):
            manager.get_match_async(**request).result(timeout=2)
        with pytest.raises(RuntimeError, match="closed"):
            manager.cancel_async(task_id).result(timeout=2)
    finally:
        gate.set()
        fail_cancel.clear()
        client.close_control()
        other.close_control()
        stop.set()
        server_thread.join(2)
    assert not server_thread.is_alive() and not errors
