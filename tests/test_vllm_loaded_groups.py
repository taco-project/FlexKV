"""vLLM learns which KV cache group a FlexKV load restores."""
from unittest.mock import MagicMock

import pytest

pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

from flexkv.common.config import RankInfo  # noqa: E402
from flexkv.integration.config import FlexKVConfig, VllmKVGroup  # noqa: E402
from flexkv.integration.vllm import vllm_v1_adapter as adapter  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("group, loaded", [
    pytest.param(None, (0,), id="one-group"),
    pytest.param(VllmKVGroup(1, ("layers.0.attn",)), (1,), id="hybrid"),
])
def test_a_load_restores_the_group_flexkv_stores(monkeypatch, group, loaded):
    manager = MagicMock()
    manager.is_ready.return_value = True
    monkeypatch.setattr(adapter, "KVManager", MagicMock(return_value=manager))
    config = FlexKVConfig()
    config.vllm_kv_group = group
    impl = object.__new__(adapter.FlexKVConnectorV1Impl)
    impl.role = adapter.KVConnectorRole.SCHEDULER
    impl.connector = adapter.FlexKVSchedulerConnector(
        config, RankInfo(model_config=config.model_config), enable_layerwise_transfer=False)

    assert impl.get_loaded_kv_cache_group_ids(MagicMock()) == loaded
