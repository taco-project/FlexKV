"""FlexKV's vLLM connector on the scheduler side, KVManager faked: the stats
it reports, the group a load restores, its prefetches, when vLLM waits for
its PUTs to free blocks, and how a failed load reaches vLLM."""
import importlib.util
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

from vllm.v1.engine import FinishReason  # noqa: E402
from vllm.v1.outputs import KVConnectorOutput  # noqa: E402

from flexkv.common.config import RankInfo  # noqa: E402
from flexkv.common.request import KVResponseStatus  # noqa: E402
from flexkv.integration.config import FlexKVConfig, VllmKVGroup  # noqa: E402
from flexkv.integration.stats import FlexKVStats  # noqa: E402
from flexkv.integration.vllm import vllm_v1_adapter as adapter  # noqa: E402

pytestmark = pytest.mark.unit

needs_kv_metrics = pytest.mark.skipif(
    importlib.util.find_spec("vllm.distributed.kv_transfer.kv_connector.v1.metrics") is None,
    reason="vLLM without KV connector metrics")
needs_transfer_results = pytest.mark.skipif(
    adapter.KVConnectorTransferResults is None,
    reason="vLLM without KVConnectorTransferResults")

Impl = adapter.FlexKVConnectorV1Impl
REQ = "req-0"
TASK = 7
PREFETCH = 5


@pytest.fixture
def manager(monkeypatch):
    manager = MagicMock()
    manager.is_ready.return_value = True
    manager.has_running_graphs.return_value = False
    manager.prefetch_async.return_value = PREFETCH
    manager.try_wait.return_value = {}
    manager.get_match.return_value = (9, np.zeros(32, dtype=bool))
    monkeypatch.setattr(adapter, "KVManager", MagicMock(return_value=manager))
    return manager


def _impl(role, connector=None):
    impl = object.__new__(Impl)
    impl.role = role
    impl.connector = connector
    impl.previous_scheduled_req_ids = set()
    impl._bound_metadata = None
    return impl


def _scheduler(config=None):
    config = config or FlexKVConfig()
    return _impl(adapter.KVConnectorRole.SCHEDULER, adapter.FlexKVSchedulerConnector(
        config, RankInfo(model_config=config.model_config), enable_layerwise_transfer=False))


@pytest.fixture
def scheduler(manager):
    return _scheduler()


@pytest.fixture
def worker():
    return _impl(adapter.KVConnectorRole.WORKER)


@needs_kv_metrics
def test_take_counts_hands_each_count_out_once_across_log_windows():
    stats = FlexKVStats(2)
    stats.record_get(num_prompt_tokens=64, num_gpu_matched_tokens=16,
                     num_flexkv_matched_tokens=32)
    # The second GET closes FlexKV's own log window, which clears its counters.
    stats.record_get(num_prompt_tokens=64, num_gpu_matched_tokens=0,
                     num_flexkv_matched_tokens=48)
    stats.record_put(num_all_tokens=64, num_unmatched_tokens=16)

    assert stats.num_get_requests == 0
    assert stats.take_counts() == {
        "num_get_requests": 2, "num_get_query_tokens": 128,
        "num_gpu_matched_tokens": 16, "num_flexkv_matched_tokens": 80,
        "num_put_requests": 1, "num_put_query_tokens": 64,
        "num_put_unmatched_tokens": 16,
    }
    assert stats.take_counts() == {}


@needs_kv_metrics
def test_scheduler_reports_the_counts_of_each_step():
    impl = _impl(adapter.KVConnectorRole.SCHEDULER,
                 SimpleNamespace(flexkv_stats=FlexKVStats(200)))
    impl.connector.flexkv_stats.record_get(
        num_prompt_tokens=64, num_gpu_matched_tokens=0, num_flexkv_matched_tokens=64)

    assert impl.get_kv_connector_stats().data["num_get_requests"] == 1
    assert impl.get_kv_connector_stats().is_empty()


@needs_kv_metrics
def test_logged_stats_sum_the_step_reports():
    # What vLLM's KVConnectorLogging does with the payloads it receives.
    total = Impl.build_kv_connector_stats(
        {"num_get_query_tokens": 64, "num_flexkv_matched_tokens": 64})
    total.aggregate(Impl.build_kv_connector_stats(
        {"num_get_query_tokens": 64, "num_gpu_matched_tokens": 32}))

    metrics = total.reduce()
    assert metrics["num_get_query_tokens"] == 128
    assert metrics["get_flexkv_match_ratio"] == 0.5
    assert metrics["get_gpu_match_ratio"] == 0.25
    assert Impl.build_kv_connector_stats(None).is_empty()


@pytest.mark.parametrize("group, loaded", [
    pytest.param(None, (0,), id="one-group"),
    pytest.param(VllmKVGroup(1, ("layers.0.attn",)), (1,), id="hybrid"),
])
def test_a_load_restores_the_group_flexkv_stores(manager, group, loaded):
    config = FlexKVConfig()
    config.vllm_kv_group = group

    assert _scheduler(config).get_loaded_kv_cache_group_ids(MagicMock()) == loaded


@pytest.fixture
def prefetching(scheduler):
    scheduler.connector.enable_prefetch = True
    return scheduler


def _arriving(num_tokens=40):
    return SimpleNamespace(request_id=REQ, num_tokens=num_tokens,
                           all_token_ids=list(range(num_tokens)))


def test_prefetch_starts_on_arrival_and_gates_the_match(prefetching, manager):
    request = _arriving()
    prefetching.on_new_request(request)

    manager.prefetch_async.assert_called_once()
    assert list(manager.prefetch_async.call_args.kwargs["token_ids"]) == list(range(32))

    connector = prefetching.connector
    assert connector.get_num_new_matched_tokens(request, 16) == (None, False)
    manager.get_match.assert_not_called()

    manager.try_wait.return_value = {PREFETCH: SimpleNamespace()}
    assert connector.get_num_new_matched_tokens(request, 16) == (0, False)
    manager.prefetch_async.assert_called_once()
    manager.get_match.assert_called_once()


def test_request_without_a_full_block_has_nothing_to_wait_for(prefetching, manager):
    request = _arriving(num_tokens=10)
    prefetching.on_new_request(request)

    manager.prefetch_async.assert_not_called()
    assert prefetching.connector.get_num_new_matched_tokens(request, 0) == (0, False)


def test_scheduling_does_not_start_a_missing_prefetch(prefetching, manager):
    # A vLLM that does not call on_new_request gets no prefetch.
    assert prefetching.connector.get_num_new_matched_tokens(_arriving(), 0) == (0, False)

    manager.prefetch_async.assert_not_called()
    manager.get_match.assert_called_once()


def test_request_ended_before_scheduling_cancels_its_prefetch(prefetching, manager):
    request = _arriving()
    request.is_finished = lambda: True
    request.get_finished_reason = lambda: FinishReason.ABORT
    prefetching.on_new_request(request)

    prefetching.connector.request_finished(request, [])

    manager.cancel.assert_called_once_with(task_ids=[PREFETCH])
    assert not prefetching.connector.prefetch_tasks


def test_no_prefetch_without_a_tier_to_prefetch_from(scheduler, manager):
    scheduler.connector.enable_prefetch = False
    scheduler.on_new_request(_arriving())

    manager.prefetch_async.assert_not_called()


@pytest.fixture
def waiting(scheduler):
    scheduler.connector.pending_free_wait_s = 1.0
    return scheduler


def _task(cls, task_id, launched_ago=None, **fields):
    task = cls(task_id=task_id, request=SimpleNamespace(request_id=f"req-{task_id}"), **fields)
    if launched_ago is not None:
        task.task_launch_time = time.perf_counter() - launched_ago
    return task


def _put(task_id, launched_ago=None):
    return _task(adapter.FlexKVPutTask, task_id, launched_ago,
                 num_matched_tokens=0, num_unmatched_tokens=16)


def test_nothing_to_wait_for_without_a_put(waiting):
    waiting.connector.tasks_to_launch[1] = _task(
        adapter.FlexKVGetTask, 1, num_computed_tokens=0, num_new_matched_tokens=16)

    assert not waiting.has_pending_block_frees()


def test_a_put_about_to_launch_frees_blocks_soon(waiting):
    waiting.connector.tasks_to_launch[1] = _put(1)

    assert waiting.has_pending_block_frees()


def test_vllm_waits_for_puts_until_the_oldest_is_overdue(waiting):
    connector = waiting.connector
    connector.put_tasks[1] = [_put(1, launched_ago=0.5)]
    connector.put_tasks[2] = [_put(2, launched_ago=0.0)]
    assert waiting.has_pending_block_frees()

    connector.put_tasks[1][0].task_launch_time -= 1.0
    assert not waiting.has_pending_block_frees()


def test_a_zero_wait_never_waits(waiting):
    waiting.connector.pending_free_wait_s = 0
    waiting.connector.tasks_to_launch[1] = _put(1)

    assert not waiting.has_pending_block_frees()


def test_the_worker_side_never_waits(worker):
    assert not worker.has_pending_block_frees()


def _finished(reason=FinishReason.ABORT, offload=False):
    return SimpleNamespace(
        request_id=REQ, num_tokens=33, is_finished=lambda: True,
        get_finished_reason=lambda: reason, offload_kv_on_finish=offload)


def _finish_get(scheduler, manager, status):
    """End an async GET of REQ with ``status`` and let vLLM poll it."""
    request = _finished()
    connector = scheduler.connector
    connector.req_id_to_task_dict[REQ] = TASK
    connector.get_tasks[TASK] = [adapter.FlexKVGetTask(
        task_id=TASK, request=request, num_computed_tokens=0, num_new_matched_tokens=16)]
    manager.try_wait.return_value = {TASK: SimpleNamespace(status=status)}
    output = KVConnectorOutput()
    scheduler.update_connector_output(output)
    manager.try_wait.return_value = {}
    return request, output


def _step(scheduler):
    return scheduler.build_connector_meta(
        SimpleNamespace(preempted_req_ids=set(), new_block_ids_to_zero=None))


@needs_transfer_results
def test_failed_get_is_reported_by_the_workers_next_step(scheduler, worker, manager):
    request, output = _finish_get(scheduler, manager, KVResponseStatus.FAILED)
    assert not output.finished_recving

    metadata = _step(scheduler)
    assert metadata.failed_recving == {REQ}
    assert not _step(scheduler).failed_recving

    worker.bind_connector_metadata(metadata)
    results = worker.get_transfer_results(set())
    assert results.finished_recving == results.failed_recving == {REQ}
    assert worker.get_finished(set()) == (None, {REQ})

    output = KVConnectorOutput(finished_recving={REQ}, failed_recving={REQ})
    scheduler.update_connector_output(output)
    assert output.finished_recving == {REQ}

    # vLLM recomputes the request: it is not loaded again.
    assert scheduler.connector.get_num_new_matched_tokens(request, 0) == (0, False)
    manager.get_match.assert_not_called()
    scheduler.connector.request_finished(request, [0, 1])
    assert REQ not in scheduler.connector.failed_load_req_ids


@needs_transfer_results
def test_request_ended_before_its_failed_load_is_reported_stores_nothing(scheduler, manager):
    _finish_get(scheduler, manager, KVResponseStatus.FAILED)

    delay = scheduler.connector.request_finished(_finished(offload=True), [0, 1])

    assert delay is False
    manager.put_match.assert_not_called()
    # vLLM still waits for the report to free the blocks.
    assert _step(scheduler).failed_recving == {REQ}


@needs_transfer_results
def test_worker_reads_metadata_bound_on_the_vllm_wrapper(worker, monkeypatch):
    import vllm.distributed.kv_transfer as kv_transfer

    metadata = adapter.FlexKVConnectorMetadata(failed_recving=frozenset({REQ}))
    outer = SimpleNamespace(has_connector_metadata=lambda: True,
                            _get_connector_metadata=lambda: metadata)
    monkeypatch.setattr(kv_transfer, "get_kv_transfer_group", lambda: outer)

    assert worker.get_finished(set()) == (None, {REQ})


@needs_transfer_results
def test_successful_get_is_reported_finished_at_once(scheduler, manager):
    _, output = _finish_get(scheduler, manager, KVResponseStatus.SUCCESS)

    assert output.finished_recving == {REQ}
    assert not _step(scheduler).failed_recving


@needs_transfer_results
def test_vllm_without_failed_receives_takes_a_failed_get_as_finished(scheduler, manager):
    scheduler.connector.report_failed_loads = False

    _, output = _finish_get(scheduler, manager, KVResponseStatus.FAILED)

    assert output.finished_recving == {REQ}
    assert not _step(scheduler).failed_recving
