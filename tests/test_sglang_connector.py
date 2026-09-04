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
