from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from flexkv.integration.sglang.prefetch_window import PrefetchWindow, parse_prefetch_max_blocks


CHANNELS = SimpleNamespace(PREFETCH_START="start", PREFETCH_PROGRESS="progress", RESET="reset")


def response(status="success", loaded=0):
    return SimpleNamespace(status=SimpleNamespace(value=status), return_mask=np.ones(loaded, dtype=bool))


@pytest.fixture
def window(monkeypatch):
    monkeypatch.setenv("FLEXKV_L3_PREFETCH_MAX_INFLIGHT", "2")
    monkeypatch.setenv("FLEXKV_L3_PREFETCH_MAX_STAGED", "4")
    monkeypatch.setenv("FLEXKV_L3_PREFETCH_MAX_QUEUED_BLOCKS", "4")
    monkeypatch.setenv("FLEXKV_L3_PREFETCH_POLL_INTERVAL_S", "0.01")
    monkeypatch.setenv("FLEXKV_L3_PREFETCH_WAIT_S", "1")
    monkeypatch.setenv("FLEXKV_L3_PREFETCH_UNCONSUMED_TTL_S", "60")
    now = [0.0]
    manager = SimpleNamespace(prefetch_async=Mock(side_effect=iter(range(10, 100))),
                              wait=Mock(return_value={}), cancel=Mock())
    connector = SimpleNamespace(page_size=4, _prefetch_enabled=True, _chunked_prefetch=False,
                                cache_config=SimpleNamespace(num_cpu_blocks=16),
                                kv_manager=manager, _swa_kv_pool=None,
                                _sync_ctx=SimpleNamespace(is_sync_leader=True, needs_sync=False))
    value = PrefetchWindow(connector, 4, scatter_channels=CHANNELS, clock=lambda: now[0])
    value.test_now = now
    return value


def test_default_off_does_not_inspect_new_options_or_touch_manager(window, monkeypatch):
    monkeypatch.delenv("FLEXKV_L3_PREFETCH_MAX_BLOCKS", raising=False)
    monkeypatch.setenv("FLEXKV_L3_PREFETCH_MAX_INFLIGHT", "invalid")
    assert PrefetchWindow.from_connector(window.connector, scatter_channels=CHANNELS) is None
    window.connector.kv_manager.prefetch_async.assert_not_called()
    window.connector.kv_manager.wait.assert_not_called()


def test_budget_parser_and_chunk_runtime_conflict(window, monkeypatch):
    assert parse_prefetch_max_blocks(None) == 0
    assert parse_prefetch_max_blocks("auto", 16) == 8
    with pytest.raises(ValueError):
        parse_prefetch_max_blocks("auto", 0)
    with pytest.raises(ValueError):
        parse_prefetch_max_blocks("-1")
    monkeypatch.setenv("FLEXKV_L3_PREFETCH_MAX_BLOCKS", "4")
    window.connector._chunked_prefetch = True
    with pytest.raises(ValueError, match="chunked"):
        PrefetchWindow.from_connector(window.connector, scatter_channels=CHANNELS)


def test_window_requires_pp1_and_collective_sync(window, monkeypatch):
    monkeypatch.setenv("FLEXKV_L3_PREFETCH_MAX_BLOCKS", "4")
    window.connector.model_config = SimpleNamespace(pp_size=2, effective_tp_size=1)
    with pytest.raises(NotImplementedError, match="PP=1"):
        PrefetchWindow.from_connector(window.connector, scatter_channels=CHANNELS)
    window.connector.model_config = SimpleNamespace(pp_size=1, effective_tp_size=8)
    with pytest.raises(ValueError, match="collective sync"):
        PrefetchWindow.from_connector(window.connector, scatter_channels=CHANNELS)
    window.connector._sync_ctx.needs_sync = True
    assert PrefetchWindow.from_connector(window.connector, scatter_channels=CHANNELS) is not None


def test_oversized_request_rejected_before_materializing(window):
    producer = Mock(return_value=list(range(17)))
    assert window.start("large", producer, token_count=17) == -1
    producer.assert_not_called()
    assert window.held_blocks == 0 and not window.entries
    window.connector.kv_manager.prefetch_async.assert_not_called()


def test_block_queue_bounded_and_fifo_materialization(window):
    assert window.start("one", list(range(12))) == 10
    producer = Mock(return_value=list(range(8)))
    assert window.start("two", producer, token_count=8) >= 1 << 62
    assert window.start("three", list(range(12))) == -1
    producer.assert_not_called()
    assert window.held_blocks == 3
    window.connector.kv_manager.wait.return_value = {10: response(loaded=12)}
    assert window.check("one")
    assert window.held_blocks == 3
    assert window.pop_loaded("one") == 12
    window.test_now[0] += 0.01
    window.connector.kv_manager.wait.return_value = {}
    window.poll()
    producer.assert_called_once()
    assert window.entries["two"].handle == 11
    assert window.held_blocks == 2


def test_poll_once_per_interval_and_whole_graph_only(window):
    window.start("one", list(range(4)))
    window.start("two", list(range(4)))
    assert not window.check("one")
    assert not window.check("two")
    manager = window.connector.kv_manager
    manager.wait.assert_called_once_with([10, 11], timeout=0, completely=True)
    window.test_now[0] = 0.009
    window.poll()
    assert manager.wait.call_count == 1
    window.test_now[0] = 0.01
    window.poll()
    assert manager.wait.call_count == 2


def test_cancel_retains_block_credit_until_certified_drain(window):
    window.start("one", list(range(12)))
    window.cancel("one")
    assert window.held_blocks == 3
    assert window.check("one")
    window.connector.kv_manager.cancel.assert_called_once_with([10])
    window.connector.kv_manager.wait.return_value = {10: response("cancelled")}
    window.test_now[0] = 0.01
    window.poll()
    assert window.held_blocks == 0 and "one" not in window.entries


@pytest.mark.parametrize("status", ["timeout", "unready", "not_found"])
def test_non_certificate_status_never_releases_ownership(window, status):
    window.start("one", list(range(8)))
    window.cancel("one")
    window.connector.kv_manager.wait.return_value = {10: response(status)}
    window.poll()
    assert window.held_blocks == 2 and window.entries["one"].task_ids == {10}
    with pytest.raises(RuntimeError, match="drain incomplete"):
        window.drain(timeout=0)
    assert window.held_blocks == 2


def test_expiry_detaches_but_does_not_give_back_blocks(window):
    window.start("one", list(range(8)))
    window.test_now[0] = 1.1
    assert window.check("one")
    assert window.entries["one"].detached
    assert window.held_blocks == 2
    window.connector.kv_manager.cancel.assert_called_once_with([10])


def test_partial_launch_tracks_every_task_until_all_drain(window):
    error = RuntimeError("second node submission failed")
    error.launched_task_ids = [21, 22]
    window.connector.kv_manager.prefetch_async.side_effect = error
    assert window.start("partial", list(range(8))) == -1
    assert window.entries["partial"].task_ids == {21, 22}
    assert window.held_blocks == 2
    window.connector.kv_manager.wait.return_value = {21: response("failed")}
    window.poll()
    assert window.held_blocks == 2 and window.entries["partial"].task_ids == {22}
    window.test_now[0] = 0.01
    window.connector.kv_manager.wait.return_value = {22: response("cancelled")}
    window.poll()
    assert window.held_blocks == 0 and not window.entries


def test_unknown_submission_blocks_reset_and_retains_budget(window):
    window.connector.kv_manager.prefetch_async.side_effect = RuntimeError("connection lost after submit")
    with pytest.raises(RuntimeError, match="Unacknowledged"):
        window.start("unknown", list(range(8)))
    assert window.held_blocks == 2 and "unknown" in window.entries
    with pytest.raises(RuntimeError, match="refusing to release"):
        window.drain()
    assert window.held_blocks == 2


def test_known_rejected_submission_is_not_unknown_dma(window):
    error = RuntimeError("rejected before submit")
    error.launched_task_ids = []
    window.connector.kv_manager.prefetch_async.side_effect = error
    assert window.start("rejected", list(range(8))) == -1
    assert window.held_blocks == 0 and not window.entries
    window.drain()


def test_namespace_is_forwarded_once_without_reencoding(window):
    namespace = ["tenant", "cp_schema_v1", "draft_indexer"]
    window.start("one", list(range(4)), namespace=namespace)
    kwargs = window.connector.kv_manager.prefetch_async.call_args.kwargs
    assert kwargs["namespace"] == namespace
    assert kwargs["swa_aware"] is False
    assert set(kwargs) == {"token_ids", "namespace", "swa_aware"}


def test_drain_failure_keeps_pending_only_and_can_retry(window):
    window.start("active", list(range(12)))
    window.start("queued", list(range(8)))
    window.connector.kv_manager.wait.return_value = {10: response("timeout")}
    with pytest.raises(RuntimeError, match="drain incomplete"):
        window.drain(timeout=0)
    assert list(window.entries) == ["active"] and window.held_blocks == 3
    window.connector.kv_manager.wait.return_value = {10: response("cancelled")}
    window.drain(timeout=0)
    assert not window.entries and window.held_blocks == 0


def test_completed_results_ttl_releases_only_drained_credit(window):
    window.start("one", list(range(8)))
    window.connector.kv_manager.wait.return_value = {10: response(loaded=8)}
    window.poll()
    assert window.held_blocks == 2
    window.test_now[0] = 60.1
    window.poll()
    assert window.held_blocks == 0 and not window.entries


def test_follower_applies_one_batched_progress_payload_without_rpc(window):
    leader = window
    messages = []

    def send(payload, **kwargs):
        messages.append((payload, kwargs))
        return payload

    leader.connector._sync_ctx = SimpleNamespace(is_sync_leader=True, needs_sync=True, scatter=send)
    follower_connector = SimpleNamespace(**vars(leader.connector))
    follower_connector.kv_manager = SimpleNamespace(prefetch_async=Mock(), wait=Mock(), cancel=Mock())
    follower_connector._sync_ctx = SimpleNamespace(
        is_sync_leader=False, needs_sync=True,
        scatter=lambda payload, **kwargs: messages.pop(0)[0])
    follower = PrefetchWindow(follower_connector, 4, scatter_channels=CHANNELS,
                              clock=lambda: leader.test_now[0] + 100)
    assert leader.start("one", list(range(8))) == follower.start("one", list(range(8))) == 10
    leader.connector.kv_manager.wait.return_value = {10: response(loaded=8)}
    leader.poll()
    assert len(messages) == 1 and messages[0][1]["channel"] == "progress"
    follower.poll()
    assert leader.pop_loaded("one") == follower.pop_loaded("one") == 8
    assert leader.held_blocks == follower.held_blocks == 0
    follower_connector.kv_manager.prefetch_async.assert_not_called()
    follower_connector.kv_manager.wait.assert_not_called()


def test_token_supplier_failure_never_submits(window):
    producer = Mock(side_effect=ValueError("bad tokens"))
    assert window.start("bad", producer, token_count=4) == -1
    assert not window.entries and window.held_blocks == 0
    window.connector.kv_manager.prefetch_async.assert_not_called()


def test_fanout_failure_preserves_already_submitted_graph(window):
    window.connector._sync_ctx = SimpleNamespace(
        is_sync_leader=True, needs_sync=True, scatter=Mock(side_effect=RuntimeError("fanout failed")))
    with pytest.raises(RuntimeError, match="fanout failed"):
        window.start("one", list(range(8)))
    assert window.entries["one"].task_ids == {10}
    assert window.held_blocks == 2
    with pytest.raises(RuntimeError, match="ownership is retained"):
        window.drain()


def test_shared_manager_task_id_is_polled_once(window):
    window.connector.kv_manager.prefetch_async.side_effect = [10, 10]
    window.start("one", list(range(4)))
    window.start("two", list(range(4)))
    window.connector.kv_manager.wait.return_value = {10: response(loaded=4)}
    window.poll()
    window.connector.kv_manager.wait.assert_called_once_with([10], timeout=0, completely=True)
    assert window.pop_loaded("one") == window.pop_loaded("two") == 4
    assert window.held_blocks == 0


def test_staged_cancel_does_not_materialize_or_cancel_manager(window):
    window.start("active", list(range(16)))
    supplier = Mock(return_value=list(range(8)))
    window.start("queued", supplier, token_count=8)
    window.cancel("queued")
    supplier.assert_not_called()
    window.connector.kv_manager.cancel.assert_not_called()
    assert window.held_blocks == 4 and list(window.entries) == ["active"]


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_invalid_poll_interval_rejected_when_enabled(window, monkeypatch, value):
    monkeypatch.setenv("FLEXKV_L3_PREFETCH_POLL_INTERVAL_S", value)
    with pytest.raises(ValueError, match="finite and positive"):
        PrefetchWindow(window.connector, 4, scatter_channels=CHANNELS)


def test_idle_poll_caches_completion_without_fanout_or_ledger_mutation(window):
    window.start("one", list(range(8)))
    scatter = Mock(side_effect=lambda payload, **kwargs: payload)
    window.connector._sync_ctx = SimpleNamespace(is_sync_leader=True, needs_sync=True, scatter=scatter)
    manager = window.connector.kv_manager
    manager.wait.return_value = {10: response(loaded=8)}
    window.poll_idle()
    window.poll_idle()
    scatter.assert_not_called()
    manager.wait.assert_called_once_with([10], timeout=0, completely=True)
    assert window.entries["one"].state == "active"
    assert window.entries["one"].task_ids == {10}
    assert window.held_blocks == 2
    manager.wait.side_effect = AssertionError("terminal task was already consumed by idle poll")
    window.poll()
    assert scatter.call_count == 1
    assert window.pop_loaded("one") == 8
    assert window.held_blocks == 0


def test_follower_idle_poll_never_calls_manager_or_scatter(window):
    window.start("one", list(range(8)))
    scatter = Mock()
    window.connector._sync_ctx = SimpleNamespace(is_sync_leader=False, needs_sync=True, scatter=scatter)
    window.poll_idle()
    scatter.assert_not_called()
    window.connector.kv_manager.wait.assert_not_called()
    assert window.entries["one"].task_ids == {10}


def test_shutdown_drain_is_leader_local_and_uses_idle_certificate(window):
    window.start("one", list(range(8)))
    manager = window.connector.kv_manager
    manager.wait.return_value = {10: response(loaded=8)}
    scatter = Mock(side_effect=AssertionError("atexit must not wait for other ranks"))
    window.connector._sync_ctx = SimpleNamespace(is_sync_leader=True, needs_sync=True, scatter=scatter)
    window.poll_idle()
    manager.wait.side_effect = AssertionError("drain must reuse the idle completion certificate")
    window.drain(timeout=0, collective=False)
    scatter.assert_not_called()
    assert not window.entries and window.held_blocks == 0


def test_follower_shutdown_drain_does_not_wait_or_drop_ledger(window):
    window.start("one", list(range(8)))
    scatter = Mock()
    window.connector._sync_ctx = SimpleNamespace(is_sync_leader=False, needs_sync=True, scatter=scatter)
    window.drain(collective=False)
    scatter.assert_not_called()
    window.connector.kv_manager.wait.assert_not_called()
    window.connector.kv_manager.cancel.assert_not_called()
    assert window.entries["one"].task_ids == {10} and window.held_blocks == 2
