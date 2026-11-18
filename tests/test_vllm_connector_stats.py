"""FlexKV's counts reach vLLM's KV transfer metrics, each once."""
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.metrics")

from flexkv.integration.stats import FlexKVStats  # noqa: E402
from flexkv.integration.vllm import vllm_v1_adapter as adapter  # noqa: E402

pytestmark = pytest.mark.unit

Impl = adapter.FlexKVConnectorV1Impl


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


def test_scheduler_reports_the_counts_of_each_step():
    impl = object.__new__(Impl)
    impl.role = adapter.KVConnectorRole.SCHEDULER
    impl.connector = SimpleNamespace(flexkv_stats=FlexKVStats(200))
    impl.connector.flexkv_stats.record_get(
        num_prompt_tokens=64, num_gpu_matched_tokens=0, num_flexkv_matched_tokens=64)

    assert impl.get_kv_connector_stats().data["num_get_requests"] == 1
    assert impl.get_kv_connector_stats().is_empty()


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
