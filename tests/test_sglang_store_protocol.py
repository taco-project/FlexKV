from array import array
from concurrent.futures import Future, ThreadPoolExecutor
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from flexkv.common.request import KVResponse, KVResponseStatus
from flexkv.integration.sglang.comm import FlexKVScatterChannel
from flexkv.integration.sglang.connector import FlexKVConnector


def _follower_connector(payload):
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector.page_size = 1
    connector.kv_manager = None
    connector._sync_ctx = SimpleNamespace(
        is_sync_leader=False,
        needs_sync=True,
        should_send_slot_mapping_to_remote=False,
        scatter=MagicMock(return_value=payload),
        all_reduce_min=lambda value: value,
    )
    connector._inflight_stores = {}
    connector._inflight_store_contexts = {}
    connector._new_op_context = MagicMock(return_value=SimpleNamespace(task_id=-1))
    connector._log_cache_op = MagicMock()
    return connector


def _completed(value=None):
    future = Future()
    future.set_result(value)
    return future


def _observe_result(future, entered):
    result = future.result

    def observed(timeout=None):
        entered.set()
        return result(timeout)

    future.result = observed


def test_store_start_tracks_task_on_nonleader_rank():
    payload = {
        "rid": "request",
        "task_id": 17,
        "active": True,
        "slot_count": 4,
        "unmatched_count": 2,
        "unmatched_mask": [],
        "error": "",
    }
    payload.pop("error")
    connector = _follower_connector({"stores": [payload], "error": ""})

    task_id = connector.store_kv("request", [1, 2, 3, 4], torch.arange(4, dtype=torch.int64))

    assert task_id == 17
    assert connector._inflight_stores == {"request": 17}
    connector._sync_ctx.scatter.assert_called_once_with(
        {"stores": [{
            "rid": "request", "task_id": -1, "active": False,
            "slot_count": 4, "unmatched_count": 0, "unmatched_mask": [],
        }], "error": ""},
        channel=FlexKVScatterChannel.STORE_START,
    )


def test_store_leader_accepts_pinned_sideband_cpu_mapping():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector.page_size = 1
    connector._swa_kv_pool = None
    connector._profile_store_stages = False
    connector.kv_manager = MagicMock()
    connector.kv_manager.put_match_batch.return_value = [(
        17, np.asarray([True, False, True, False], dtype=np.bool_), True,
    )]
    connector._sync_ctx = SimpleNamespace(
        is_sync_leader=True,
        needs_sync=False,
        should_send_slot_mapping_to_remote=False,
        is_cross_node_pp=False,
    )
    connector._inflight_stores = {}
    connector._inflight_store_contexts = {}
    connector._new_op_context = MagicMock(return_value=SimpleNamespace(task_id=-1))
    connector._log_cache_op = MagicMock()
    cpu_mapping = torch.tensor([4, 5, 8, 9], dtype=torch.int64)

    assert connector.store_kv("request", [1, 2, 3, 4], cpu_mapping) == 17
    launch = connector.kv_manager.launch.call_args.kwargs
    assert launch["slot_mappings"][0].tolist() == [4, 8]
    assert launch["swa_slot_mappings"] == [None]


def test_lookup_accepts_sglang_array_token_ids():
    connector = _follower_connector({"task_id": -1, "hit": 0})
    connector._pending_lookups = {}
    connector._pending_lookup_contexts = {}
    token_ids = array("q", [1, 2, 3, 4])

    assert connector.lookup_kv(
        token_ids,
        torch.ones(len(token_ids), dtype=torch.bool),
        rid="request",
    ) == (-1, 0)
    connector._log_cache_op.assert_called_once()


def test_prefetch_start_accepts_legacy_manager_result_with_planned_tokens():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector._prefetch_enabled = True
    connector._swa_kv_pool = None
    connector.kv_manager = MagicMock()
    connector.kv_manager.prefetch_async.return_value = (23, 256)
    connector._sync_ctx = SimpleNamespace(
        is_sync_leader=True,
        needs_sync=False,
    )
    connector._ongoing_prefetches = {}
    connector._prefetch_contexts = {}
    connector._prefetch_planned_tokens = {}
    connector._prefetch_loaded_tokens = {}
    context = SimpleNamespace(task_id=-1)
    connector._new_op_context = MagicMock(return_value=context)
    connector._log_cache_op = MagicMock()

    task_id = connector.prefetch_async("request", [1, 2, 3])

    assert task_id == 23
    assert context.task_id == 23
    assert connector._ongoing_prefetches == {"request": 23}
    assert connector._prefetch_contexts == {"request": context}
    assert connector._prefetch_planned_tokens == {"request": 256}


def test_prefetch_completion_reports_loaded_tokens_once():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector._prefetch_enabled = True
    connector.kv_manager = MagicMock()
    connector.kv_manager.try_wait.return_value = {
        23: SimpleNamespace(
            status=KVResponseStatus.SUCCESS,
            return_mask=np.ones(256, dtype=np.bool_),
        )
    }
    connector._sync_ctx = SimpleNamespace(
        is_sync_leader=True,
        needs_sync=False,
    )
    connector._ongoing_prefetches = {"request": 23}
    connector._prefetch_contexts = {"request": SimpleNamespace(task_id=23)}
    connector._prefetch_planned_tokens = {"request": 0}
    connector._prefetch_loaded_tokens = {}
    connector._new_op_context = MagicMock()
    connector._pop_context = MagicMock(return_value=SimpleNamespace(task_id=23))
    connector._log_cache_op = MagicMock()

    assert connector.check_prefetch_progress("request") is True
    assert connector.pop_prefetch_loaded_tokens("request") == 256
    assert connector.pop_prefetch_loaded_tokens("request") == 0


def test_prefetch_completion_reports_partial_remote_prefix():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector._prefetch_enabled = True
    connector.kv_manager = MagicMock()
    connector.kv_manager.try_wait.return_value = {
        23: SimpleNamespace(
            status=KVResponseStatus.SUCCESS,
            return_mask=np.asarray([True] * 192 + [False] * 64),
        )
    }
    connector._sync_ctx = SimpleNamespace(
        is_sync_leader=True,
        needs_sync=False,
    )
    connector._ongoing_prefetches = {"request": 23}
    connector._prefetch_contexts = {"request": SimpleNamespace(task_id=23)}
    connector._prefetch_planned_tokens = {"request": 0}
    connector._prefetch_loaded_tokens = {}
    connector._new_op_context = MagicMock()
    connector._pop_context = MagicMock(return_value=SimpleNamespace(task_id=23))
    connector._log_cache_op = MagicMock()

    assert connector.check_prefetch_progress("request") is True
    assert connector.pop_prefetch_loaded_tokens("request") == 192


def test_duplicate_lookup_cancels_replaced_held_task():
    """A repeated SGLang match for one rid must not orphan the first lock."""
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector.page_size = 1
    connector._swa_kv_pool = None
    connector.kv_manager = MagicMock()
    connector.kv_manager.get_match.side_effect = [
        (41, np.ones(4, dtype=np.bool_)),
        (42, np.ones(4, dtype=np.bool_)),
    ]
    connector._sync_ctx = SimpleNamespace(
        is_sync_leader=True,
        needs_sync=False,
    )
    connector._pending_lookups = {}
    connector._pending_lookup_contexts = {}
    contexts = [
        SimpleNamespace(task_id=-1, operation="lookup"),
        SimpleNamespace(task_id=-1, operation="lookup"),
    ]
    connector._new_op_context = MagicMock(side_effect=contexts)
    connector._log_cache_op = MagicMock()

    mask = torch.ones(4, dtype=torch.bool)
    assert connector.lookup_kv([1, 2, 3, 4], mask, rid="same") == (41, 4)
    assert connector.lookup_kv([1, 2, 3, 4], mask, rid="same") == (42, 4)

    connector.kv_manager.cancel.assert_called_once_with([41])
    assert connector._pending_lookups == {"same": 42}
    assert connector._pending_lookup_contexts == {"same": contexts[1]}
    assert contexts[0].operation == "load"
    assert any(
        call.args[1:3] == ("complete", "cancelled") and call.kwargs.get("reason") == "lookup_replaced"
        for call in connector._log_cache_op.call_args_list
    )

    connector.release_pending("same")
    assert connector.kv_manager.cancel.call_args_list == [
        (([41],), {}),
        (([42],), {}),
    ]
    assert connector._pending_lookups == {}
    assert connector._pending_lookup_contexts == {}


def test_store_completion_clears_nonleader_tracking():
    connector = _follower_connector(["request"])
    connector._sync_ctx.is_sync_leader = False
    connector._inflight_stores = {"request": 17}
    connector._inflight_store_contexts = {"request": object()}

    assert connector.check_completed_stores() == ["request"]
    assert connector._inflight_stores == {}
    assert connector._inflight_store_contexts == {}
    connector._sync_ctx.scatter.assert_called_once_with([], channel=FlexKVScatterChannel.STORE_COMPLETION)


def test_store_ready_uses_leader_payload_and_dedicated_channel():
    connector = _follower_connector(["request"])

    assert connector.sync_ready_store_rids(["ignored-on-follower"]) == ["request"]
    connector._sync_ctx.scatter.assert_called_once_with([], channel=FlexKVScatterChannel.STORE_READY)


def test_store_completion_waits_for_fully_drained_graph():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector.kv_manager = MagicMock()
    connector._sync_ctx = SimpleNamespace(
        is_sync_leader=True,
        needs_sync=False,
    )
    connector._inflight_stores = {"request": 17}
    connector._inflight_store_contexts = {
        "request": SimpleNamespace(task_id=17)
    }
    connector._new_op_context = MagicMock(
        return_value=SimpleNamespace(task_id=17)
    )
    connector._log_cache_op = MagicMock()
    connector.kv_manager.wait.side_effect = [
        {17: KVResponse(KVResponseStatus.TIMEOUT, 17, None)},
        {17: KVResponse(KVResponseStatus.SUCCESS, 17, None)},
    ]

    assert connector.check_completed_stores() == []
    assert connector._inflight_stores == {"request": 17}

    assert connector.check_completed_stores() == ["request"]
    assert connector._inflight_stores == {}
    assert connector.kv_manager.wait.call_args_list == [
        (([17],), {"timeout": 0.0, "completely": True}),
        (([17],), {"timeout": 0.0, "completely": True}),
    ]


def test_wait_store_keeps_tracking_until_fully_drained():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector.kv_manager = MagicMock()
    connector._sync_ctx = SimpleNamespace(is_sync_leader=True)
    connector._inflight_stores = {"request": 17}
    connector._inflight_store_contexts = {
        "request": SimpleNamespace(task_id=17)
    }
    connector._new_op_context = MagicMock(
        return_value=SimpleNamespace(task_id=17)
    )
    connector._log_cache_op = MagicMock()
    connector.kv_manager.wait.side_effect = [
        {17: KVResponse(KVResponseStatus.TIMEOUT, 17, None)},
        {17: KVResponse(KVResponseStatus.SUCCESS, 17, None)},
    ]

    assert not connector.wait_store("request", timeout=5.0)
    assert connector._inflight_stores == {"request": 17}

    assert connector.wait_store("request", timeout=5.0)
    assert connector._inflight_stores == {}
    assert connector.kv_manager.wait.call_args_list == [
        (([17],), {"timeout": 5.0, "completely": True}),
        (([17],), {"timeout": 5.0, "completely": True}),
    ]


def test_store_reset_waits_for_leader_drain_on_follower():
    connector = _follower_connector(None)
    connector._sync_ctx.scatter.side_effect = [
        {"ok": True, "error": ""},
        {"ok": True, "error": ""},
    ]
    connector._launched_load_tids = []
    connector._pending_lookups = {}
    connector._pending_lookup_contexts = {}
    connector._prefetch_contexts = {}
    connector._ongoing_prefetches = {}
    connector._inflight_loads = {}
    connector._completed_layerwise = {}
    connector._launched_load_contexts = {}
    connector._inflight_stores = {"request": 17}
    connector._inflight_store_contexts = {"request": object()}
    connector.layer_done_counter = None

    connector.reset()

    assert connector._sync_ctx.scatter.call_args_list[1].kwargs == {
        "channel": FlexKVScatterChannel.STORE_RESET,
        "blocking": True,
    }
    assert connector._inflight_stores == {}


def test_namespace_capability_is_explicit_for_all_active_paths():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    assert connector.supports_cache_namespace is True
    # A chunked-prefetch implementation needs its own full-chain opt-in.
    connector._chunked_prefetch = True
    connector._chunked_namespace_supported = False
    assert connector.supports_cache_namespace is False
    connector._chunked_namespace_supported = True
    assert connector.supports_cache_namespace is True


def test_namespace_is_forwarded_to_lookup_store_and_prefetch():
    connector = _follower_connector(None)
    connector._sync_ctx.is_sync_leader = True
    connector._sync_ctx.needs_sync = False
    connector._sync_ctx.is_pp_active = False
    connector._swa_kv_pool = None
    connector._pending_lookups = {}
    connector._pending_lookup_contexts = {}
    connector._prefetch_enabled = True
    connector._ongoing_prefetches = {}
    connector._prefetch_contexts = {}
    connector._prefetch_planned_tokens = {}
    connector.kv_manager = MagicMock()
    connector.kv_manager.get_match.return_value = (-1, np.zeros(4, dtype=bool))
    connector.kv_manager.put_match_batch.return_value = [(17, np.zeros(4, dtype=bool), False)]
    connector.kv_manager.prefetch_async.return_value = (23, 4)
    namespace = ['["sglang-cache-v1","adapter","salt"]']
    connector.lookup_kv([1, 2, 3, 4], torch.ones(4, dtype=torch.bool), rid="r", namespace=namespace)
    connector.store_kv("r", [1, 2, 3, 4], torch.arange(4), namespace=namespace)
    assert connector.prefetch_async("r", [1, 2, 3, 4], namespace=namespace) == 23
    for method in (connector.kv_manager.get_match, connector.kv_manager.prefetch_async):
        assert method.call_args.kwargs["namespace"] == namespace
        assert method.call_args.kwargs["token_ids"].tolist() == [1, 2, 3, 4]
    request = connector.kv_manager.put_match_batch.call_args.args[0][0]
    assert request["namespace"] == namespace
    assert request["token_ids"].tolist() == [1, 2, 3, 4]


def _batch_connector(matches):
    connector = _follower_connector(None)
    connector._sync_ctx.is_sync_leader = True
    connector._sync_ctx.needs_sync = False
    connector._sync_ctx.is_cross_node_pp = False
    connector._swa_kv_pool = object()
    connector._kvcache = SimpleNamespace(
        translate_loc_from_full_to_swa=MagicMock(
            side_effect=AssertionError("CPU snapshots must not be translated again")
        ),
    )
    connector.kv_manager = MagicMock()
    connector.kv_manager.put_match_batch.return_value = matches
    connector.kv_manager.launch_async.return_value = _completed([])
    connector.kv_manager.cancel_async.return_value = _completed()
    return connector


def test_batch_store_filters_full_and_swa_then_retains_each_owner_until_drain():
    connector = _batch_connector([
        (17, np.array([False, True, True, True]), True),
        (18, np.array([False, False, True, True]), True),
        (19, np.zeros(4, dtype=bool), False),
    ])
    tokens = np.arange(4, dtype=np.int64)
    stores = [
        dict(rid="a", token_ids=tokens, namespace=["a"],
             kv_indices=torch.tensor([4, 5, 8, 9]),
             swa_indices=torch.tensor([10, 0, 18, 19])),
        dict(rid="b", token_ids=array("q", [5, 6, 7, 8]), namespace=["b"],
             kv_indices=torch.tensor([20, 21, 30, 31]),
             swa_indices=torch.tensor([0, 0, 40, 41])),
        dict(rid="cached", token_ids=tokens, kv_indices=torch.arange(4)),
    ]
    assert connector.store_kv_batch(stores) == [17, 18, -1]
    metadata = connector.kv_manager.put_match_batch.call_args.args[0]
    assert np.shares_memory(metadata[0]["token_ids"], tokens)
    assert [item["namespace"] for item in metadata] == [["a"], ["b"], None]
    launch = connector.kv_manager.launch.call_args.kwargs
    assert launch["task_ids"] == [17, 18]
    assert [x.tolist() for x in launch["slot_mappings"]] == [[5, 8, 9], [30, 31]]
    assert [x.tolist() for x in launch["swa_slot_mappings"]] == [[18, 19], [40, 41]]
    assert launch["confirm_launch"]
    connector.kv_manager.cancel.assert_called_once_with([19])
    assert connector._inflight_stores == {"a": 17, "b": 18}
    connector.kv_manager.wait.side_effect = [
        {17: KVResponse(KVResponseStatus.SUCCESS, 17, None),
         18: KVResponse(KVResponseStatus.TIMEOUT, 18, None)},
        {18: KVResponse(KVResponseStatus.SUCCESS, 18, None)},
    ]
    assert connector.check_completed_stores() == ["a"]
    assert connector._inflight_stores == {"b": 18}
    assert connector.check_completed_stores() == ["b"]
    assert connector._inflight_stores == {}


def test_zero_full_mask_does_not_discard_l2_to_l3_or_swa_graph():
    connector = _batch_connector([(17, np.zeros(4, dtype=bool), True)])
    assert connector.store_kv_batch([dict(
        rid="l3", token_ids=array("q", [1, 2, 3, 4]), kv_indices=torch.arange(4),
        swa_indices=torch.tensor([0, 0, 8, 9]),
    )]) == [17]
    launch = connector.kv_manager.launch.call_args.kwargs
    assert launch["slot_mappings"][0].tolist() == []
    assert launch["swa_slot_mappings"][0].tolist() == [8, 9]
    connector.kv_manager.cancel.assert_not_called()


def test_batch_mask_failure_cancels_every_reservation_before_launch():
    connector = _batch_connector([
        (17, np.ones(4, dtype=bool), True),
        (18, np.ones(3, dtype=bool), True),
    ])
    stores = [
        dict(rid=rid, token_ids=range(4), kv_indices=torch.arange(4))
        for rid in ("a", "b")
    ]
    with pytest.raises(RuntimeError, match="mask length mismatch"):
        connector.store_kv_batch(stores)
    connector.kv_manager.cancel.assert_called_once_with([17, 18])
    connector.kv_manager.launch.assert_not_called()
    assert connector._inflight_stores == {}


def test_partial_launch_failure_keeps_uncertain_gpu_ownership():
    connector = _batch_connector([(17, np.ones(4, dtype=bool), True)])
    connector.kv_manager.launch.side_effect = RuntimeError("partial submit")
    connector.kv_manager.wait.return_value = {
        17: KVResponse(KVResponseStatus.TIMEOUT, 17, None),
    }
    with pytest.raises(RuntimeError, match="retain GPU slots"):
        connector.store_kv_batch([
            dict(rid="a", token_ids=range(4), kv_indices=torch.arange(4)),
        ])
    assert connector._inflight_stores == {"a": 17}
    assert connector.check_completed_stores() == []
    connector.kv_manager.wait.return_value = {
        17: KVResponse(KVResponseStatus.CANCELLED, 17, None),
    }
    assert connector.check_completed_stores() == ["a"]


def test_common_ready_prefix_requires_each_rank_and_identical_order():
    connector = _follower_connector(["a", "b", "c"])
    connector._sync_ctx.all_reduce_min = MagicMock(return_value=1)
    assert connector.sync_ready_store_rids(
        ["a", "other"], require_all_ranks=True,
    ) == ["a"]
    connector._sync_ctx.all_reduce_min.assert_called_once_with(1)


def test_all_rank_validation_failure_precedes_metadata_allocation():
    connector = _batch_connector([])
    connector._sync_ctx.needs_sync = True
    connector._sync_ctx.all_reduce_min = MagicMock(return_value=0)
    with pytest.raises(ValueError, match="bad SWA tail"):
        connector.store_kv_batch([dict(
            rid="a", token_ids=range(4), kv_indices=torch.arange(4),
            validation_error="bad SWA tail",
        )])
    connector.kv_manager.put_match_batch.assert_not_called()


def _resettable_store_connector():
    connector = _batch_connector([])
    connector._launched_load_tids = []
    connector._pending_lookups = {}
    connector._pending_lookup_contexts = {}
    connector._prefetch_contexts = {}
    connector._ongoing_prefetches = {}
    connector._inflight_loads = {}
    connector._completed_layerwise = {}
    connector._launched_load_contexts = {}
    connector.layer_done_counter = None
    return connector


def test_async_store_pending_then_ordered_tasks_preserves_live_zero_mask_graph():
    connector = _batch_connector([])
    connector.page_size = 2
    future = Future()
    connector.kv_manager.put_match_batch_async.return_value = future
    cancellation, launch_ack = Future(), Future()
    connector.kv_manager.cancel_async.return_value = cancellation
    connector.kv_manager.launch_async.return_value = launch_ack
    tokens = array("q", [1, 2, 3, 4])
    stores = [
        dict(rid="full", token_ids=tokens, namespace=["tenant"],
             kv_indices=torch.tensor([4, 5, 8, 9]),
             swa_indices=torch.tensor([0, 0, 18, 19])),
        dict(rid="partial", token_ids=[1], kv_indices=torch.tensor([4])),
        dict(rid="cached", token_ids=tokens, kv_indices=torch.arange(4)),
        dict(rid="l3", token_ids=tokens, kv_indices=torch.arange(4),
             swa_indices=torch.tensor([0, 0, 20, 21])),
    ]
    admission = connector.submit_store_kv_batch(stores)
    assert connector.poll_store_kv_batch(admission) is None
    assert not connector._inflight_stores
    connector.kv_manager.launch.assert_not_called()
    connector.kv_manager.put_match_batch.assert_not_called()
    future.set_result([
        (17, np.array([False, False, True, True]), True),
        (18, np.zeros(4, dtype=bool), False),
        (19, np.zeros(4, dtype=bool), True),
    ])
    assert connector.poll_store_kv_batch(admission) is None
    connector.kv_manager.launch_async.assert_not_called()
    cancellation.set_result(None)
    assert connector.poll_store_kv_batch(admission) is None
    assert connector._pending_store_batch is admission
    launch_ack.set_result([17, 19])
    assert connector.poll_store_kv_batch(admission) == [17, -1, -1, 19]
    assert connector.poll_store_kv_batch(admission) == [17, -1, -1, 19]
    connector.kv_manager.launch_async.assert_called_once()
    launch = connector.kv_manager.launch_async.call_args.kwargs
    assert launch["task_ids"] == [17, 19]
    assert [mapping.tolist() for mapping in launch["slot_mappings"]] == [[8, 9], []]
    assert [mapping.tolist() for mapping in launch["swa_slot_mappings"]] == [[18, 19], [20, 21]]
    connector.kv_manager.cancel_async.assert_called_once_with([18])
    assert connector._pending_store_batch is None


def test_async_store_admission_rejects_second_batch_before_reserving():
    connector = _batch_connector([])
    connector.kv_manager.put_match_batch_async.return_value = Future()
    store = dict(rid="one", token_ids=range(4), kv_indices=torch.arange(4))
    admission = connector.submit_store_kv_batch([store])
    with pytest.raises(ValueError, match="already pending"):
        connector.submit_store_kv_batch([dict(store, rid="two")])
    assert connector._pending_store_batch is admission
    connector.kv_manager.put_match_batch_async.assert_called_once()


def test_async_store_leader_failure_is_reported_on_follower_and_cannot_relaunch():
    leader = _batch_connector([])
    leader._sync_ctx.needs_sync = True
    payloads = []
    leader._sync_ctx.scatter = lambda payload, **kwargs: payloads.append(payload) or payload
    follower = _follower_connector(None)
    follower._sync_ctx.scatter = lambda payload, **kwargs: payloads.pop(0)
    ready = []
    leader._sync_ctx.all_reduce_min = lambda value: ready.append(value) or value
    follower._sync_ctx.all_reduce_min = lambda value: ready.pop(0)
    future = Future()
    leader.kv_manager.put_match_batch_async.return_value = future
    stores = [dict(rid="a", token_ids=range(4), kv_indices=torch.arange(4))]
    leader_batch = leader.submit_store_kv_batch(stores)
    follower_batch = follower.submit_store_kv_batch(stores)
    assert leader.poll_store_kv_batch(leader_batch) is None
    assert follower.poll_store_kv_batch(follower_batch) is None
    future.set_exception(RuntimeError("queued match failed"))
    for connector, batch in ((leader, leader_batch), (follower, follower_batch)):
        with pytest.raises(RuntimeError, match="queued match failed"):
            connector.poll_store_kv_batch(batch)
        assert connector._pending_store_batch is batch
    for connector, batch in ((leader, leader_batch), (follower, follower_batch)):
        with pytest.raises(RuntimeError, match="queued match failed"):
            connector.poll_store_kv_batch(batch)
    leader.kv_manager.launch.assert_not_called()
    assert not payloads and not ready


def test_reset_waits_for_late_store_reservations_then_cancels_and_drains():
    connector = _resettable_store_connector()
    entered = threading.Event()

    future = Future()
    connector.kv_manager.put_match_batch_async.return_value = future
    connector.kv_manager.wait.return_value = {
        17: KVResponse(KVResponseStatus.CANCELLED, 17, None),
    }
    admission = connector.submit_store_kv_batch([
        dict(rid="a", token_ids=range(4), kv_indices=torch.arange(4)),
    ])
    _observe_result(admission.future, entered)
    with ThreadPoolExecutor(max_workers=1) as pool:
        reset = pool.submit(connector.reset)
        try:
            assert entered.wait(2)
            assert not reset.done()
            assert connector._pending_store_batch is admission
            connector.kv_manager.cancel.assert_not_called()
        finally:
            future.set_result([(17, np.ones(4, dtype=bool), True)])
        reset.result(timeout=2)
    connector.kv_manager.launch_async.assert_not_called()
    connector.kv_manager.cancel.assert_called_once_with([17])
    connector.kv_manager.wait.assert_called_once_with([17], timeout=30.0, completely=True)
    assert connector._pending_store_batch is None
    with pytest.raises(RuntimeError, match="was reset"):
        connector.poll_store_kv_batch(admission)


def test_reset_retains_store_sources_until_pending_launch_acknowledges():
    connector = _resettable_store_connector()
    connector.kv_manager.put_match_batch_async.return_value = _completed([
        (17, np.ones(4, dtype=bool), True),
    ])
    launch_ack = Future()
    connector.kv_manager.launch_async.return_value = launch_ack
    connector.kv_manager.wait.return_value = {
        17: KVResponse(KVResponseStatus.CANCELLED, 17, None),
    }
    admission = connector.submit_store_kv_batch([
        dict(rid="a", token_ids=range(4), kv_indices=torch.arange(4)),
    ])
    entered = threading.Event()
    _observe_result(admission.future, entered)
    with ThreadPoolExecutor(max_workers=1) as pool:
        reset = pool.submit(connector.reset)
        try:
            assert entered.wait(2)
            assert not reset.done()
            assert connector._pending_store_batch is admission
            connector.kv_manager.cancel.assert_not_called()
        finally:
            launch_ack.set_result([17])
        reset.result(timeout=2)
    connector.kv_manager.cancel.assert_called_once_with([17])
    assert admission.reset and connector._pending_store_batch is None


def test_pending_reset_failure_retains_snapshots_until_every_task_drains():
    connector = _resettable_store_connector()
    future = Future()
    future.set_result([(17, np.ones(4, dtype=bool), True)])
    connector.kv_manager.put_match_batch_async.return_value = future
    stores = [dict(rid="a", token_ids=range(4), kv_indices=torch.arange(4))]
    admission = connector.submit_store_kv_batch(stores)
    connector.kv_manager.wait.return_value = {
        17: KVResponse(KVResponseStatus.TIMEOUT, 17, None),
    }
    with pytest.raises(RuntimeError, match="retain GPU slots"):
        connector.reset()
    assert connector._pending_store_batch is admission
    assert not admission.reset and admission.stores[0] is stores[0]
    connector.kv_manager.wait.return_value = {
        17: KVResponse(KVResponseStatus.CANCELLED, 17, None),
    }
    connector.reset()
    assert connector._pending_store_batch is None
    connector.kv_manager.launch.assert_not_called()


def test_async_launch_failure_retains_admission_and_does_not_retry_launch():
    connector = _resettable_store_connector()
    future = Future()
    future.set_result([(17, np.ones(4, dtype=bool), True)])
    connector.kv_manager.put_match_batch_async.return_value = future
    connector.kv_manager.launch_async.side_effect = RuntimeError("partial submit")
    connector.kv_manager.wait.return_value = {
        17: KVResponse(KVResponseStatus.TIMEOUT, 17, None),
    }
    admission = connector.submit_store_kv_batch([
        dict(rid="a", token_ids=range(4), kv_indices=torch.arange(4)),
    ])
    for _ in range(2):
        with pytest.raises(RuntimeError, match="retain GPU slots"):
            connector.poll_store_kv_batch(admission)
    connector.kv_manager.launch_async.assert_called_once()
    assert connector._pending_store_batch is admission
    connector.kv_manager.wait.return_value = {
        17: KVResponse(KVResponseStatus.CANCELLED, 17, None),
    }
    connector.reset()
    assert not connector._inflight_stores


def test_control_transport_failure_retains_pending_admission_during_reset():
    from flexkv.server.control import ControlTransportError

    connector = _resettable_store_connector()
    future = Future()
    future.set_exception(ControlTransportError("reply sequence mismatch"))
    connector.kv_manager.put_match_batch_async.return_value = future
    admission = connector.submit_store_kv_batch([
        dict(rid="a", token_ids=range(4), kv_indices=torch.arange(4)),
    ])
    with pytest.raises(RuntimeError, match="sequence mismatch"):
        connector.poll_store_kv_batch(admission)
    with pytest.raises(RuntimeError, match="sequence mismatch"):
        connector.reset()
    assert connector._pending_store_batch is admission
    assert not admission.reset


def test_ordinary_async_slot_mapping_capability_stays_disabled_for_swa_and_pp():
    connector = _batch_connector([])
    assert not connector.supports_async_store_slot_mapping
    connector._swa_kv_pool = None
    connector._sync_ctx.is_pp_active = True
    assert not connector.supports_async_store_slot_mapping
    connector._sync_ctx.is_pp_active = False
    assert connector.supports_async_store_slot_mapping


def test_reset_recovers_partial_rollback_ids_without_polling_or_launching():
    from flexkv.common.request import StoreReservationError

    connector = _resettable_store_connector()
    future = Future()
    future.set_exception(StoreReservationError("rollback incomplete", [17, 18]))
    connector.kv_manager.put_match_batch_async.return_value = future
    connector.kv_manager.wait.return_value = {
        tid: KVResponse(KVResponseStatus.CANCELLED, tid, None) for tid in (17, 18)
    }
    admission = connector.submit_store_kv_batch([
        dict(rid="a", token_ids=range(4), kv_indices=torch.arange(4)),
    ])
    connector.reset()
    connector.kv_manager.cancel.assert_called_once_with([17, 18])
    connector.kv_manager.wait.assert_called_once_with([17, 18], timeout=30.0, completely=True)
    connector.kv_manager.launch.assert_not_called()
    assert admission.reset and connector._pending_store_batch is None
