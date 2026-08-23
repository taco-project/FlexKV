from types import SimpleNamespace
from unittest.mock import Mock

from vllm.v1.engine import FinishReason

from flexkv.integration.vllm.vllm_v1_adapter import FlexKVSchedulerConnector


def _connector() -> FlexKVSchedulerConnector:
    connector = object.__new__(FlexKVSchedulerConnector)
    connector.req_id_to_task_dict = {}
    connector.maybe_skip_put = False
    connector.block_size = 16
    connector.flexkv_stats = Mock()
    connector._put_match = Mock(return_value=(-1, 0, 0))
    return connector


def _request(reason: FinishReason, offload: bool):
    return SimpleNamespace(
        request_id="request-0",
        is_finished=lambda: True,
        get_finished_reason=lambda: reason,
        offload_kv_on_finish=offload,
        num_tokens=32,
    )


def test_abort_without_vllm_directive_does_not_enter_put_path():
    connector = _connector()

    delayed = connector.request_finished(
        _request(FinishReason.ABORT, offload=False), [0, 1]
    )

    assert delayed is False
    connector._put_match.assert_not_called()


def test_abort_with_vllm_directive_enters_existing_put_path():
    connector = _connector()

    request = _request(FinishReason.ABORT, offload=True)
    delayed = connector.request_finished(request, [0, 1])

    assert delayed is False
    connector._put_match.assert_called_once_with(
        request=request,
        num_tokens_to_put=16,
    )
    connector.flexkv_stats.record_put.assert_called_once_with(
        num_all_tokens=16,
        num_unmatched_tokens=0,
    )


def test_normal_finish_remains_eligible_for_put():
    connector = _connector()

    delayed = connector.request_finished(
        _request(FinishReason.STOP, offload=False), [0, 1]
    )

    assert delayed is False
    connector._put_match.assert_called_once()
    connector.flexkv_stats.record_put.assert_called_once()


def test_abort_with_no_computed_blocks_skips_put():
    connector = _connector()

    delayed = connector.request_finished(
        _request(FinishReason.ABORT, offload=True), []
    )

    assert delayed is False
    connector._put_match.assert_not_called()
    connector.flexkv_stats.record_put.assert_not_called()


def test_abort_limits_put_to_computed_blocks():
    connector = _connector()
    request = _request(FinishReason.ABORT, offload=True)
    request.num_tokens = 80

    connector.request_finished(request, [4, 5])

    connector._put_match.assert_called_once_with(
        request=request,
        num_tokens_to_put=32,
    )
    connector.flexkv_stats.record_put.assert_called_once_with(
        num_all_tokens=32,
        num_unmatched_tokens=0,
    )
