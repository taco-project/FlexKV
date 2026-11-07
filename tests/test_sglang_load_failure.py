"""Leader/follower load-result agreement and safe bypass contracts."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from flexkv.integration.sglang.comm import FlexKVScatterChannel
from flexkv.integration.sglang.connector import FlexKVConnector

pytestmark = pytest.mark.unit


def connector(leader=True, *, status="success", safe=True):
    c = FlexKVConnector.__new__(FlexKVConnector)
    c._sync_ctx = SimpleNamespace(is_sync_leader=leader, needs_sync=False,
        should_send_slot_mapping_to_remote=False, barrier=Mock(), scatter=Mock())
    c.kv_manager = Mock()
    c.kv_manager.wait.return_value = {17: SimpleNamespace(status=status)}
    c.tp_client = Mock()
    c.tp_client.gpu_route_control.return_value = {"safe": safe, "generation": 3}
    c._pending_lookups = {"r": 17}
    c._pending_lookup_contexts = {}
    c._inflight_stores = {}
    c._pop_context = Mock(return_value=SimpleNamespace(task_id=17))
    c._log_cache_op = Mock()
    c._to_cpu_int64 = lambda x: x
    c._build_swa_slot_mapping = lambda x: None
    return c


@pytest.mark.parametrize("status", ["failed", "cancelled", "not_found", "timeout", "missing"])
def test_leader_failure_is_broadcast_and_follower_returns_same_count(status):
    c = connector(status=status)
    if status in ("timeout", "missing"):
        c.kv_manager.wait.side_effect = [
            {17: SimpleNamespace(status=status)}, {17: SimpleNamespace(status="failed")}]
    c._sync_ctx.needs_sync = True
    c._sync_ctx.scatter.side_effect = lambda result, **_: result
    assert c.retrieve_kv("r", torch.arange(4)) == 0
    payload = c._sync_ctx.scatter.call_args.args[0]
    assert c._sync_ctx.scatter.call_args.kwargs["channel"] == FlexKVScatterChannel.LOAD_RESULT
    assert payload == {"loaded": 0, "bypass": True, "error": None, "generation": 3}
    c._sync_ctx.barrier.assert_not_called()
    follower = connector(False)
    follower._sync_ctx.needs_sync = True
    follower._sync_ctx.scatter.return_value = payload
    assert follower.retrieve_kv("r", torch.arange(4)) == 0
    assert follower._cache_bypassed and follower._route_generation == 3
    follower.kv_manager.launch.assert_not_called()


@pytest.mark.parametrize("stage", ["launch", "wait"])
def test_leader_exception_reaches_all_ranks_and_can_safely_bypass(stage):
    c = connector()
    getattr(c.kv_manager, stage).side_effect = RuntimeError("transport lost")
    if stage == "wait":
        c.kv_manager.wait.side_effect = [RuntimeError("transport lost"),
                                       {17: SimpleNamespace(status="cancelled")}]
    else:
        c.kv_manager.wait.return_value = {17: SimpleNamespace(status="not_found")}
    assert c.retrieve_kv("r", torch.arange(4)) == 0
    assert c._cache_bypassed


@pytest.mark.parametrize("fence_error", [False, True])
def test_unsafe_timeout_raises_identical_error_on_leader_and_follower(fence_error):
    c = connector(status="timeout", safe=False)
    if fence_error:
        c.tp_client.gpu_route_control.side_effect = RuntimeError("server gone")
    c._sync_ctx.needs_sync = True
    c._sync_ctx.scatter.side_effect = lambda result, **_: result
    with pytest.raises(RuntimeError, match="cannot safely bypass") as error:
        c.retrieve_kv("r", torch.arange(4))
    payload = c._sync_ctx.scatter.call_args.args[0]
    follower = connector(False)
    follower._sync_ctx.needs_sync = True
    follower._sync_ctx.scatter.return_value = payload
    with pytest.raises(RuntimeError) as other:
        follower.retrieve_kv("r", torch.arange(4))
    assert str(error.value) == str(other.value)


def test_success_broadcasts_actual_loaded_count():
    c = connector()
    c._sync_ctx.needs_sync = True
    c._sync_ctx.scatter.side_effect = lambda result, **_: result
    assert c.retrieve_kv("r", torch.arange(7)) == 7
    assert not c._cache_bypassed
    c.tp_client.gpu_route_control.assert_not_called()


def test_bypass_blocks_new_lookup_store_and_prefetch():
    c = connector(status="failed")
    assert c.retrieve_kv("r", torch.arange(4)) == 0
    c.kv_manager.reset_mock()
    assert c.lookup_kv([1], torch.ones(1, dtype=torch.bool), "new") == (-1, 0)
    assert c.store_kv("new", [1], torch.arange(1)) == -1
    assert c.prefetch_async("new", [1]) == -1
    assert not c.kv_manager.mock_calls


def test_recover_cache_is_collective_and_advances_generation():
    c = connector()
    c._cache_bypassed = True
    c._route_generation = 2
    c.tp_client.gpu_route_control.return_value = {"safe": True, "generation": 3}
    c._sync_ctx.needs_sync = True
    c._sync_ctx.scatter.side_effect = lambda result, **_: result
    assert c.recover_cache()
    assert not c._cache_bypassed and c._route_generation == 3
    c.tp_client.gpu_route_control.assert_called_once_with("recover", 2)
    assert c._sync_ctx.scatter.call_args.kwargs["channel"] == FlexKVScatterChannel.RECOVERY


def test_recovery_rejection_keeps_bypass():
    c = connector()
    c._cache_bypassed = True
    c._inflight_stores = {"old": 10}
    assert not c.recover_cache()
    assert c._cache_bypassed
    c.tp_client.gpu_route_control.assert_not_called()


def test_failed_store_trips_collective_bypass():
    c = connector(status="failed")
    c._inflight_stores = {"store": 17}
    c._inflight_store_contexts = {}
    c._new_op_context = Mock(return_value=SimpleNamespace(task_id=17))
    c._sync_ctx.needs_sync = True
    c._sync_ctx.scatter.side_effect = lambda result, **_: result
    assert c.check_completed_stores() == ["store"]
    assert c._cache_bypassed and not c._inflight_stores
    payload = c._sync_ctx.scatter.call_args.args[0]
    assert payload["bypass"] and payload["error"] is None
    assert c._sync_ctx.scatter.call_args.kwargs["channel"] == FlexKVScatterChannel.STORE_RESULT


def test_store_timeout_never_releases_unacknowledged_dma():
    c = connector(status="timeout", safe=False)
    c._inflight_stores = {"store": 17}
    c._inflight_store_contexts = {"store": SimpleNamespace(task_id=17, start_ns=0)}
    with pytest.raises(RuntimeError, match="DMA drain"):
        c.check_completed_stores()
    assert c._inflight_stores == {"store": 17}
    with pytest.raises(RuntimeError, match="DMA drain"):
        c.lookup_kv([1], torch.ones(1, dtype=torch.bool), "new")


def test_store_poll_does_not_reset_an_existing_circuit_breaker():
    c = connector(status="failed")
    c.retrieve_kv("r", torch.arange(4))
    assert c.check_completed_stores() == []
    assert c._cache_bypassed


def test_late_success_poll_cannot_clear_bypass_or_poison_new_generation():
    c = connector(status="failed")
    c.retrieve_kv("r", torch.arange(4))
    c._apply_cache_result({"bypass": False, "error": None, "generation": 3})
    assert c._cache_bypassed
    c._route_generation = 4
    c._cache_bypassed = False
    c._apply_cache_result({"bypass": True, "error": "old route", "generation": 3})
    assert not c._cache_bypassed and c._cache_failure is None


def test_optional_recovery_retry_obeys_cooldown():
    c = connector()
    c._cache_bypassed = True
    c._next_recovery_at = float("inf")
    assert not c.recover_cache(force=False)
    c.tp_client.gpu_route_control.assert_not_called()
    assert c._cache_bypassed
    assert c.recover_cache(force=True)
    assert not c._cache_bypassed


def test_aged_store_with_missing_response_is_fenced():
    c = connector()
    c.kv_manager.wait.return_value = {}
    c._inflight_stores = {"store": 17}
    c._inflight_store_contexts = {"store": SimpleNamespace(task_id=17, start_ns=0)}
    assert c.check_completed_stores() == []
    assert c._cache_bypassed
    # Pending source slots remain held until a later terminal response.
    assert c._inflight_stores == {"store": 17}
