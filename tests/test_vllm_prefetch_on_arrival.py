"""A request's FlexKV prefetch starts when it reaches vLLM, not when it is first scheduled."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

from flexkv.common.config import RankInfo  # noqa: E402
from flexkv.integration.config import FlexKVConfig  # noqa: E402
from flexkv.integration.vllm import vllm_v1_adapter as adapter  # noqa: E402

pytestmark = pytest.mark.unit

PREFETCH = 5


@pytest.fixture
def manager(monkeypatch):
    manager = MagicMock()
    manager.is_ready.return_value = True
    manager.prefetch_async.return_value = PREFETCH
    manager.try_wait.return_value = {}
    manager.get_match.return_value = (9, np.zeros(32, dtype=bool))
    monkeypatch.setattr(adapter, "KVManager", MagicMock(return_value=manager))
    return manager


@pytest.fixture
def scheduler(manager):
    config = FlexKVConfig()
    connector = adapter.FlexKVSchedulerConnector(
        config, RankInfo(model_config=config.model_config), enable_layerwise_transfer=False)
    connector.block_size = 16
    connector.enable_prefetch = True
    impl = object.__new__(adapter.FlexKVConnectorV1Impl)
    impl.role = adapter.KVConnectorRole.SCHEDULER
    impl.connector = connector
    return impl


def _request(num_tokens=40):
    return SimpleNamespace(request_id="req-0", num_tokens=num_tokens,
                           all_token_ids=list(range(num_tokens)))


def test_prefetch_starts_on_arrival_and_gates_the_match(scheduler, manager):
    request = _request()
    scheduler.on_new_request(request)

    manager.prefetch_async.assert_called_once()
    assert list(manager.prefetch_async.call_args.kwargs["token_ids"]) == list(range(32))

    connector = scheduler.connector
    assert connector.get_num_new_matched_tokens(request, 16) == (None, False)
    manager.get_match.assert_not_called()

    manager.try_wait.return_value = {PREFETCH: SimpleNamespace()}
    assert connector.get_num_new_matched_tokens(request, 16) == (0, False)
    manager.prefetch_async.assert_called_once()
    manager.get_match.assert_called_once()


def test_request_without_a_full_block_has_nothing_to_wait_for(scheduler, manager):
    request = _request(num_tokens=10)
    scheduler.on_new_request(request)

    manager.prefetch_async.assert_not_called()
    assert scheduler.connector.get_num_new_matched_tokens(request, 0) == (0, False)


def test_request_ended_before_scheduling_cancels_its_prefetch(scheduler, manager):
    request = _request()
    request.is_finished = lambda: True
    request.get_finished_reason = lambda: adapter.FinishReason.ABORT
    scheduler.on_new_request(request)

    scheduler.connector.request_finished(request, [])

    manager.cancel.assert_called_once_with(task_ids=[PREFETCH])
    assert not scheduler.connector.prefetch_tasks


def test_no_prefetch_without_a_tier_to_prefetch_from(scheduler, manager):
    scheduler.connector.enable_prefetch = False
    scheduler.on_new_request(_request())

    manager.prefetch_async.assert_not_called()
