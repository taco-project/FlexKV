from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

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
    )
    connector._inflight_stores = {}
    connector._inflight_store_contexts = {}
    connector._new_op_context = MagicMock(
        return_value=SimpleNamespace(task_id=-1)
    )
    connector._log_cache_op = MagicMock()
    return connector


def test_store_start_tracks_task_on_nonleader_rank():
    payload = {
        "rid": "request",
        "task_id": 17,
        "active": True,
        "unmatched_mask": [True, False, True, False],
        "error": "",
    }
    connector = _follower_connector(payload)

    task_id = connector.store_kv(
        "request", [1, 2, 3, 4], torch.arange(4, dtype=torch.int64)
    )

    assert task_id == 17
    assert connector._inflight_stores == {"request": 17}
    connector._sync_ctx.scatter.assert_called_once_with(
        {
            "rid": "request",
            "task_id": -1,
            "active": False,
            "unmatched_mask": [],
            "error": "",
        },
        channel=FlexKVScatterChannel.STORE_START,
    )


def test_store_completion_clears_nonleader_tracking():
    connector = _follower_connector(["request"])
    connector._sync_ctx.is_sync_leader = False
    connector._inflight_stores = {"request": 17}
    connector._inflight_store_contexts = {"request": object()}

    assert connector.check_completed_stores() == ["request"]
    assert connector._inflight_stores == {}
    assert connector._inflight_store_contexts == {}
    connector._sync_ctx.scatter.assert_called_once_with(
        [], channel=FlexKVScatterChannel.STORE_COMPLETION
    )


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
