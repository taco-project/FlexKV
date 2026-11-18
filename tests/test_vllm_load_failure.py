"""A failed async GET reaches vLLM as a failed receive, reported by the workers."""
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.base")

from vllm.v1.engine import FinishReason  # noqa: E402
from vllm.v1.outputs import KVConnectorOutput  # noqa: E402

from flexkv.common.config import RankInfo  # noqa: E402
from flexkv.common.request import KVResponseStatus  # noqa: E402
from flexkv.integration.config import FlexKVConfig  # noqa: E402
from flexkv.integration.vllm import vllm_v1_adapter as adapter  # noqa: E402

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(adapter.KVConnectorTransferResults is None,
                       reason="vLLM without KVConnectorTransferResults"),
]

REQ = "req-0"
TASK = 7


@pytest.fixture
def manager(monkeypatch):
    manager = MagicMock()
    manager.is_ready.return_value = True
    manager.has_running_graphs.return_value = False
    monkeypatch.setattr(adapter, "KVManager", MagicMock(return_value=manager))
    return manager


@pytest.fixture
def scheduler(manager):
    config = FlexKVConfig()
    connector = adapter.FlexKVSchedulerConnector(
        config, RankInfo(model_config=config.model_config), enable_layerwise_transfer=False)
    impl = object.__new__(adapter.FlexKVConnectorV1Impl)
    impl.role = adapter.KVConnectorRole.SCHEDULER
    impl.connector = connector
    impl.previous_scheduled_req_ids = set()
    impl._bound_metadata = None
    return impl


@pytest.fixture
def worker():
    impl = object.__new__(adapter.FlexKVConnectorV1Impl)
    impl.role = adapter.KVConnectorRole.WORKER
    impl._bound_metadata = None
    return impl


def _request(reason=FinishReason.ABORT, offload=False):
    return SimpleNamespace(
        request_id=REQ, num_tokens=33, is_finished=lambda: True,
        get_finished_reason=lambda: reason, offload_kv_on_finish=offload)


def _finish_get(scheduler, manager, status):
    """End an async GET of REQ with ``status`` and let vLLM poll it."""
    request = _request()
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


def test_request_ended_before_its_failed_load_is_reported_stores_nothing(scheduler, manager):
    _finish_get(scheduler, manager, KVResponseStatus.FAILED)

    delay = scheduler.connector.request_finished(_request(offload=True), [0, 1])

    assert delay is False
    manager.put_match.assert_not_called()
    # vLLM still waits for the report to free the blocks.
    assert _step(scheduler).failed_recving == {REQ}


def test_worker_reads_metadata_bound_on_the_vllm_wrapper(worker, monkeypatch):
    import vllm.distributed.kv_transfer as kv_transfer

    metadata = adapter.FlexKVConnectorMetadata(failed_recving=frozenset({REQ}))
    outer = SimpleNamespace(has_connector_metadata=lambda: True,
                            _get_connector_metadata=lambda: metadata)
    monkeypatch.setattr(kv_transfer, "get_kv_transfer_group", lambda: outer)

    assert worker.get_finished(set()) == (None, {REQ})


def test_successful_get_is_reported_finished_at_once(scheduler, manager):
    _, output = _finish_get(scheduler, manager, KVResponseStatus.SUCCESS)

    assert output.finished_recving == {REQ}
    assert not _step(scheduler).failed_recving


def test_vllm_without_failed_receives_takes_a_failed_get_as_finished(scheduler, manager):
    scheduler.connector.report_failed_loads = False

    _, output = _finish_get(scheduler, manager, KVResponseStatus.FAILED)

    assert output.finished_recving == {REQ}
    assert not _step(scheduler).failed_recving
