from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from flexkv.integration.sglang.comm import FlexKVScatterChannel
from flexkv.integration.sglang.connector import FlexKVConnector


def test_lookup_accepts_sglang_array_token_ids():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector.page_size = 1
    connector.kv_manager = None
    connector._sync_ctx = SimpleNamespace(
        is_sync_leader=False,
        needs_sync=True,
        scatter=MagicMock(return_value={"task_id": -1, "hit": 0}),
    )
    connector._pending_lookups = {}
    connector._pending_lookup_contexts = {}
    connector._new_op_context = MagicMock(
        return_value=SimpleNamespace(task_id=-1)
    )
    connector._log_cache_op = MagicMock()
    token_ids = array("q", [1, 2, 3, 4])

    assert connector.lookup_kv(
        token_ids,
        torch.ones(len(token_ids), dtype=torch.bool),
        rid="request",
    ) == (-1, 0)
    connector._log_cache_op.assert_called_once()


def test_registration_distinguishes_physical_and_global_kv_heads():
    for physical_heads, global_heads in ((8, 32), (1, 4), (1, 1)):
        connector = FlexKVConnector.__new__(FlexKVConnector)
        connector.model_config = SimpleNamespace(
            kv_dim=2, num_kv_heads=global_heads
        )
        connector.rank_info = SimpleNamespace(num_layers_per_pp_stage=2)
        connector.page_size = 16
        connector.tp_client = MagicMock()
        connector._label = "test"
        kv_caches = [torch.empty(160, physical_heads, 128)] * 4

        connector._register_standard_to_server(kv_caches)

        layout = connector.tp_client.register_to_server.call_args.kwargs[
            "kv_layout"
        ]
        assert layout.num_head == physical_heads
        assert layout.num_kv_heads == global_heads


def _push_connector(*, leader: bool = True, enabled: bool = True):
    import numpy as np  # noqa: F401  (the connector builds the int64 array)

    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector._push_enabled = enabled
    connector._sync_ctx = SimpleNamespace(is_sync_leader=leader, needs_sync=False)
    connector.kv_manager = MagicMock() if leader else None
    connector._swa_kv_pool = None
    connector._ongoing_pushes = {}
    connector._log_cache_op = MagicMock()
    return connector


def test_push_async_is_leader_only_and_tracks_the_task():
    """The leader hands the push to KVManager (radixshmem) and keeps the task
    for the drain; followers and a non-clustered connector do nothing."""
    import numpy as np

    connector = _push_connector()
    connector.kv_manager.push_async.return_value = 7
    assert connector.push_async("req", [1, 2, 3, 4], target_rank=3, evict_local=True) == 7
    kwargs = connector.kv_manager.push_async.call_args.kwargs
    assert kwargs["token_ids"].dtype == np.int64 and kwargs["token_ids"].tolist() == [1, 2, 3, 4]
    assert (kwargs["target_rank"], kwargs["evict_local"], kwargs["swa_aware"]) == (3, True, False)
    assert list(connector._ongoing_pushes) == [7]
    assert connector._ongoing_pushes[7].operation == "push"

    follower = _push_connector(leader=False)
    assert follower.push_async("req", [1, 2, 3, 4]) == -1
    standalone = _push_connector(enabled=False)
    assert standalone.push_async("req", [1, 2, 3, 4]) == -1
    standalone.kv_manager.push_async.assert_not_called()


def test_check_completed_pushes_reaps_terminal_tasks():
    """try_wait on the pushes in flight: a finished one is logged with the
    tokens the peer took and dropped, a running one stays."""
    import numpy as np
    from flexkv.common.request import KVResponseStatus

    connector = _push_connector()
    connector.kv_manager.push_async.side_effect = [7, 8]
    connector.push_async("a", list(range(48)))
    connector.push_async("b", list(range(48)))
    mask = np.zeros(48, dtype=np.bool_)
    mask[16:48] = True                      # the peer already held the first block
    connector.kv_manager.try_wait.return_value = {
        7: SimpleNamespace(status=KVResponseStatus.SUCCESS, return_mask=mask),
        8: SimpleNamespace(status=KVResponseStatus.TIMEOUT, return_mask=None),
    }
    assert connector.check_completed_pushes() == 1
    assert list(connector._ongoing_pushes) == [8]
    complete = [c for c in connector._log_cache_op.call_args_list if c.args[1] == "complete"]
    assert len(complete) == 1 and complete[0].kwargs["tokens"] == 32
    assert connector.kv_manager.try_wait.call_args.kwargs["task_ids"] == [7, 8]
