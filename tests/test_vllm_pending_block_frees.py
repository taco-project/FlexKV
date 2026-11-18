"""vLLM waits for a FlexKV PUT to free blocks instead of preempting, but not on one that hangs."""
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

from flexkv.common.config import RankInfo  # noqa: E402
from flexkv.integration.config import FlexKVConfig  # noqa: E402
from flexkv.integration.vllm import vllm_v1_adapter as adapter  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture
def scheduler(monkeypatch):
    manager = MagicMock()
    manager.is_ready.return_value = True
    monkeypatch.setattr(adapter, "KVManager", MagicMock(return_value=manager))
    config = FlexKVConfig()
    impl = object.__new__(adapter.FlexKVConnectorV1Impl)
    impl.role = adapter.KVConnectorRole.SCHEDULER
    impl.connector = adapter.FlexKVSchedulerConnector(
        config, RankInfo(model_config=config.model_config), enable_layerwise_transfer=False)
    impl.connector.pending_free_wait_s = 1.0
    return impl


def _task(cls, task_id, launched_ago=None, **fields):
    task = cls(task_id=task_id, request=SimpleNamespace(request_id=f"req-{task_id}"), **fields)
    if launched_ago is not None:
        task.task_launch_time = time.perf_counter() - launched_ago
    return task


def _put(task_id, launched_ago=None):
    return _task(adapter.FlexKVPutTask, task_id, launched_ago,
                 num_matched_tokens=0, num_unmatched_tokens=16)


def test_nothing_to_wait_for_without_a_put(scheduler):
    scheduler.connector.tasks_to_launch[1] = _task(
        adapter.FlexKVGetTask, 1, num_computed_tokens=0, num_new_matched_tokens=16)

    assert not scheduler.has_pending_block_frees()


def test_a_put_about_to_launch_frees_blocks_soon(scheduler):
    scheduler.connector.tasks_to_launch[1] = _put(1)

    assert scheduler.has_pending_block_frees()


def test_vllm_waits_for_puts_until_the_oldest_is_overdue(scheduler):
    connector = scheduler.connector
    connector.put_tasks[1] = [_put(1, launched_ago=0.5)]
    connector.put_tasks[2] = [_put(2, launched_ago=0.0)]
    assert scheduler.has_pending_block_frees()

    connector.put_tasks[1][0].task_launch_time -= 1.0
    assert not scheduler.has_pending_block_frees()


def test_a_zero_wait_never_waits(scheduler):
    scheduler.connector.pending_free_wait_s = 0
    scheduler.connector.tasks_to_launch[1] = _put(1)

    assert not scheduler.has_pending_block_frees()


def test_the_worker_side_never_waits():
    impl = object.__new__(adapter.FlexKVConnectorV1Impl)
    impl.role = adapter.KVConnectorRole.WORKER

    assert not impl.has_pending_block_frees()
