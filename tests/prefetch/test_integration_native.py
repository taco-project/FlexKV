"""Native imports, real task-engine dispatch, ZMQ control and rank agreement."""

from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from types import SimpleNamespace as NS
from unittest.mock import Mock
import threading
import time
import uuid

import numpy as np
import pytest

pytest.importorskip("flexkv.c_ext")
import zmq

from flexkv.common.config import ModelConfig, CacheConfig
from flexkv.integration.sglang.connector import FlexKVConnector
from flexkv.kvtask import KVTaskEngine, KVTaskManager, TaskStatus
from flexkv.prefetch.coordinator import PrefetchCoordinator
from flexkv.prefetch.runtime import TaskRuntime
from flexkv.prefetch.types import PrefetchOptions, PrefetchCapabilities
from flexkv.server.client import KVDPClient
from flexkv.server.server import KVServer
from flexkv.server.request import PrefetchControlRequest, WaitRequest
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
                    if isinstance(req, PrefetchControlRequest):
                        server._handle_prefetch_control(req)
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
        assert client.prefetch_control("capabilities") == PrefetchCapabilities(
            partial_swa_checkpoints=True
        )
        assert time.monotonic() - before < 2
        assert server._pending_waits and not legacy_responses
        with ThreadPoolExecutor(max_workers=8) as pool:
            caps = list(
                pool.map(lambda _: client.prefetch_control("capabilities"), range(32))
            )
        assert all(c.protocol_version == 1 for c in caps)
        h = client.prefetch_control(
            "start",
            token_ids=np.arange(32, dtype=np.int64),
            options=PrefetchOptions(policy="best_effort"),
        )
        with pytest.raises(RuntimeError, match="another client"):
            other.prefetch_control("progress", handles=[h])
        snapshot = client.prefetch_control("progress", handles=[h], demand_handles=[h])[
            h
        ]
        assert snapshot.stop_reason in ("demand", "complete")
        client.prefetch_control("release", handle=h)
        client.prefetch_control("release", handle=h)
        assert not legacy_responses
    finally:
        stop.set()
        worker.join(2)
        send.close(0)
        ctx.term()
    server.kv_task_engine = NS(shutdown=lambda: None)  # fixture owns runtime teardown
    assert not worker.is_alive() and not errors


def test_follower_publishes_same_terminal_span_and_does_no_rpc():
    from flexkv.prefetch.types import PrefetchHandle, PrefetchSnapshot

    h = PrefetchHandle("epoch", 7)
    snapshot = PrefetchSnapshot(
        h,
        5,
        "terminal",
        True,
        "partial",
        "demand",
        40,
        16,
        ((8, 16),),
        0,
        0,
        1,
        1,
        True,
    )
    c = FlexKVConnector.__new__(FlexKVConnector)
    c._prefetch_enabled = c._chunked_prefetch = True
    c._chunked_prefetch_options = PrefetchOptions(policy="best_effort")
    c._prefetch_sessions, c._prefetch_result_sessions = {}, {}
    c._prefetch_loaded_tokens, c._prefetch_loaded_spans = {}, {}
    c.kv_manager = Mock()
    c._sync_ctx = NS(
        is_sync_leader=False,
        needs_sync=True,
        scatter=Mock(
            side_effect=[
                {"handle": h, "error": None},
                {"snapshot": snapshot, "error": None},
            ]
        ),
    )
    assert c.prefetch_async("r", [1, 2, 3, 4]) == 7
    assert c.check_prefetch_progress("r")
    assert c.pop_prefetch_loaded_span("r") == (8, 8)
    assert c.pop_prefetch_loaded_span("r") == (0, None)
    c._release_prefetch_result("r")
    assert not c.kv_manager.mock_calls


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
    callback.assert_called_once()
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
