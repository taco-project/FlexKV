from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import flexkv.integration.sglang.connector as connector_module
from flexkv.common.config import UserConfig
from flexkv.integration.config import FlexKVConfig
from flexkv.integration.sglang.comm import FlexKVScatterChannel
from flexkv.integration.sglang.connector import FlexKVConnector

pytestmark = pytest.mark.unit


@pytest.mark.parametrize('architecture,legacy', [
    ('DeepseekV4ForCausalLM', False),
    ('DeepseekV4ForCausalLMNextN', False),
    ('DeepseekV4ForCausalLMDSpark', False),
    (None, True),
])
def test_dsv4_config_without_removed_sglang_arch_flag(monkeypatch, architecture, legacy):
    """Modern SGLang's ModelConfig has no legacy V4 boolean. Its indexed
    compressed KV must still get a SWA pool and must not use NSA's byte width.
    """
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    model = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=[architecture] if architecture else []),
        num_hidden_layers=4, qk_nope_head_dim=448, qk_rope_head_dim=64,
        index_head_dim=128, dtype=torch.bfloat16,
    )
    if legacy:
        model.is_deepseek_v4_arch = True
    args = SimpleNamespace(tp_size=4, pp_size=1, dp_size=1, attn_dp_size=1,
        nnodes=1, node_rank=0, attn_cp_size=1, kv_cache_dtype='fp8_e4m3')
    config = FlexKVConfig(user_config=UserConfig(cpu_cache_gb=1))
    config.post_init_from_sglang_config(model, args, page_size=256)
    assert config.model_config.head_size == 512
    assert config.model_config.kv_dim == 1
    assert config.cache_config.swa is not None
    assert config.cache_config.swa.bytes_per_token_per_layer == 585


def test_non_dsv4_nsa_keeps_packed_kv_width(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'current_device', lambda: 0)
    model = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=['DeepseekV32ForCausalLM']),
        num_hidden_layers=4, qk_nope_head_dim=128, qk_rope_head_dim=64,
        kv_lora_rank=512, index_head_dim=128, dtype=torch.bfloat16,
    )
    args = SimpleNamespace(tp_size=4, pp_size=1, dp_size=1, nnodes=1,
        node_rank=0, kv_cache_dtype='fp8_e4m3')
    config = FlexKVConfig(user_config=UserConfig(cpu_cache_gb=1))
    config.post_init_from_sglang_config(model, args, page_size=256)
    assert config.model_config.head_size == 656
    assert config.cache_config.swa is None


def test_resolving_buffers_does_not_format_kv_tensor_contents():
    class PrivateBuffer:
        def __repr__(self):
            raise AssertionError('KV contents must not be formatted')
    buffer = PrivateBuffer()
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector._is_dsv4 = False
    cache = SimpleNamespace(k_buffer=[buffer], v_buffer=[buffer], private_state=buffer)
    buffers, indexer = connector._resolve_kv_buffers(cache)
    assert buffers[0] is buffer and buffers[1] is buffer
    assert indexer is None


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


def _idle_connector(chunked_prefetch):
    return SimpleNamespace(
        _sync_ctx=SimpleNamespace(is_sync_leader=True),
        kv_manager=MagicMock(),
        _launched_load_tids=[],
        _chunked_prefetch=chunked_prefetch,
    )


def test_idle_connector_polls_detached_tasks():
    connector = _idle_connector(chunked_prefetch=False)
    FlexKVConnector.drain_launched_loads(connector)
    connector.kv_manager.try_wait.assert_called_once_with(task_ids=[])
    connector.kv_manager.wait.assert_not_called()


def test_idle_connector_rate_limits_engine_nudges(monkeypatch):
    connector = _idle_connector(chunked_prefetch=False)
    now = [100.0]
    monkeypatch.setattr(connector_module.time, "monotonic", lambda: now[0])

    for _ in range(1000):
        FlexKVConnector.drain_launched_loads(connector)
    assert connector.kv_manager.try_wait.call_count == 1

    now[0] += connector_module._IDLE_DRIVE_INTERVAL_S * 0.5
    FlexKVConnector.drain_launched_loads(connector)
    assert connector.kv_manager.try_wait.call_count == 1

    now[0] += connector_module._IDLE_DRIVE_INTERVAL_S
    FlexKVConnector.drain_launched_loads(connector)
    assert connector.kv_manager.try_wait.call_count == 2


def test_idle_chunked_connector_leaves_draining_to_task_runtime():
    connector = _idle_connector(chunked_prefetch=True)
    for _ in range(10):
        FlexKVConnector.drain_launched_loads(connector)
    connector.kv_manager.try_wait.assert_not_called()
    connector.kv_manager.wait.assert_not_called()


def test_single_launched_load_is_polled_until_fully_complete():
    manager = MagicMock()
    manager.wait.return_value = {}
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector._sync_ctx = SimpleNamespace(is_sync_leader=True)
    connector.kv_manager = manager
    connector._launched_load_tids = [7]

    connector.drain_launched_loads()

    manager.wait.assert_called_once_with([7], timeout=0.0, completely=True)
    assert connector._launched_load_tids == [7]


def test_nonleader_does_not_poll_idle_or_launched_loads():
    for task_ids in ([], [7]):
        manager = MagicMock()
        connector = SimpleNamespace(
            _sync_ctx=SimpleNamespace(is_sync_leader=False),
            kv_manager=manager,
            _launched_load_tids=task_ids,
        )

        FlexKVConnector.drain_launched_loads(connector)

        assert not manager.mock_calls


def test_legacy_cancel_prefetch_cancels_lookup_and_prefetch_tasks():
    connector = FlexKVConnector.__new__(FlexKVConnector)
    connector._chunked_prefetch = False
    connector.kv_manager = MagicMock()
    connector._sync_ctx = SimpleNamespace(is_sync_leader=True)
    connector._pending_lookups = {"r": 11}
    connector._pending_lookup_contexts = {}
    connector._ongoing_prefetches = {"r": 22}
    connector._prefetch_planned_tokens = {"r": 64}
    connector._prefetch_loaded_tokens = {}
    connector._prefetch_contexts = {}
    connector._log_cache_op = MagicMock()

    connector.cancel_prefetch("r")
    connector.cancel_prefetch("r")  # Repeated cancellation must be harmless.

    assert [c.args[0] for c in connector.kv_manager.cancel.call_args_list] == [
        [11],
        [22],
    ]
    assert not connector._pending_lookups and not connector._ongoing_prefetches
