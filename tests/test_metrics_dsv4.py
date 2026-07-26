"""Tests for DSV4 metrics: SWA, cache query/readiness, layerwise.

Covers:
- enabled path: metric values are recorded correctly (incl. cache-engine SWA hooks)
- disabled path (default): all record_* are no-ops and never raise
"""
import numpy as np
import pytest

from prometheus_client import CollectorRegistry

from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
from flexkv.common.transfer import DeviceType
from flexkv.metrics.collector import FlexKVMetricsCollector
from flexkv.swa.swa_host_pool import SWAHostPool


@pytest.fixture()
def enabled_collector(monkeypatch):
    """Fresh registry per test: full isolation from the process-global default
    REGISTRY (which other tests populate via GlobalCacheEngine)."""
    monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "enable_metrics", True)
    registry = CollectorRegistry()
    collector = FlexKVMetricsCollector(role="worker", registry=registry)
    assert collector.enabled
    collector.test_registry = registry
    return collector


def sample(collector, name, labels=None):
    return collector.test_registry.get_sample_value(name, labels or {}) or 0.0


class TestSWAMetrics:
    def test_swa_query_and_hit_blocks(self, enabled_collector):
        c = enabled_collector
        before_hit = sample(c, "flexkv_py_swa_query_total", {"result": "hit"})
        before_miss = sample(c, "flexkv_py_swa_query_total", {"result": "miss"})
        before_blocks = sample(c, "flexkv_py_swa_hit_blocks_total")

        c.record_swa_query("hit")
        c.record_swa_hit_blocks(7)
        c.record_swa_query("miss")
        c.record_swa_hit_blocks(0)   # ignored
        c.record_swa_query("bogus")  # unknown label value, ignored

        assert sample(c, "flexkv_py_swa_query_total", {"result": "hit"}) == before_hit + 1
        assert sample(c, "flexkv_py_swa_query_total", {"result": "miss"}) == before_miss + 1
        assert sample(c, "flexkv_py_swa_hit_blocks_total") == before_blocks + 7

    def test_swa_slot_stats_and_failures(self, enabled_collector):
        c = enabled_collector
        c.update_swa_slot_stats("cpu", 3, 100)
        assert sample(c, "flexkv_py_swa_slot_used", {"device": "cpu"}) == 3
        assert sample(c, "flexkv_py_swa_slot_total", {"device": "cpu"}) == 100

        before = sample(c, "flexkv_py_swa_slot_alloc_failed_total", {"device": "cpu"})
        c.record_swa_slot_alloc_failed("cpu")
        assert sample(c, "flexkv_py_swa_slot_alloc_failed_total", {"device": "cpu"}) == before + 1

    def test_swa_evicted_by_reason(self, enabled_collector):
        c = enabled_collector
        before_pf = sample(c, "flexkv_py_swa_evicted_slots_total",
                           {"device": "cpu", "reason": "pool_full"})
        before_ca = sample(c, "flexkv_py_swa_evicted_slots_total",
                           {"device": "cpu", "reason": "cascade"})
        c.record_swa_evicted("cpu", "pool_full", 2)
        c.record_swa_evicted("cpu", "cascade", 5)
        c.record_swa_evicted("cpu", "cascade", 0)  # ignored
        assert sample(c, "flexkv_py_swa_evicted_slots_total",
                      {"device": "cpu", "reason": "pool_full"}) == before_pf + 2
        assert sample(c, "flexkv_py_swa_evicted_slots_total",
                      {"device": "cpu", "reason": "cascade"}) == before_ca + 5

    def test_swa_transfer_bytes(self, enabled_collector):
        c = enabled_collector
        before = sample(c, "flexkv_py_swa_transfer_bytes_total", {"operation": "get"})
        c.record_transfer_completed("H2D", 4, 2048, "get", is_swa=True)
        c.record_transfer_completed("H2D", 4, 1024, "get", is_swa=False)
        assert sample(c, "flexkv_py_swa_transfer_bytes_total", {"operation": "get"}) == before + 2048


class _FakeIndex:
    """Minimal stand-in for the radix index used by SWA hook paths."""
    def __init__(self, evict_freed=0, drain_slots=()):
        self._evict_freed = evict_freed
        self._drain_slots = list(drain_slots)

    def evict_swa(self, num):
        return np.array([], dtype=np.int64), self._evict_freed

    def drain_freed_swa_slots(self):
        slots, self._drain_slots = self._drain_slots, []
        return slots


class _FakeMempool:
    @staticmethod
    def recycle_blocks(blocks):
        pass


def _make_engine_stub(collector, num_slots, index):
    """CacheEngine (Python variant) with only the SWA hook dependencies set."""
    from types import SimpleNamespace
    from flexkv.cache.cache_engine import CacheEngine

    engine = CacheEngine.__new__(CacheEngine)
    engine._metrics_collector = collector
    engine.device_type = DeviceType.CPU
    engine.swa_pool = SWAHostPool(SimpleNamespace(num_slots=num_slots, pin_memory=False))
    engine.index = index
    engine.mempool = _FakeMempool()
    return engine


class TestCacheEngineSWAHooks:
    def test_slot_gauge_follows_alloc_and_free(self, enabled_collector):
        c = enabled_collector
        engine = _make_engine_stub(enabled_collector, num_slots=4, index=_FakeIndex())
        engine._record_swa_slot_stats()
        assert sample(c, "flexkv_py_swa_slot_used", {"device": "cpu"}) == 0
        assert sample(c, "flexkv_py_swa_slot_total", {"device": "cpu"}) == 4

        slot = engine._alloc_swa_slot()
        assert slot >= 0
        assert sample(c, "flexkv_py_swa_slot_used", {"device": "cpu"}) == 1

        engine._free_swa_slot(slot)
        assert sample(c, "flexkv_py_swa_slot_used", {"device": "cpu"}) == 0

    def test_alloc_failure_counts_after_eviction_retry(self, enabled_collector):
        c = enabled_collector
        # Pool of 1, already exhausted; eviction frees nothing.
        engine = _make_engine_stub(enabled_collector, num_slots=1, index=_FakeIndex(evict_freed=0))
        assert engine._alloc_swa_slot() >= 0
        before = sample(c, "flexkv_py_swa_slot_alloc_failed_total", {"device": "cpu"})
        assert engine._alloc_swa_slot() == -1
        assert sample(c, "flexkv_py_swa_slot_alloc_failed_total", {"device": "cpu"}) == before + 1

    def test_evict_and_cascade_counted(self, enabled_collector):
        c = enabled_collector
        engine = _make_engine_stub(enabled_collector, num_slots=2,
                                   index=_FakeIndex(evict_freed=1, drain_slots=[0]))
        before_pf = sample(c, "flexkv_py_swa_evicted_slots_total",
                           {"device": "cpu", "reason": "pool_full"})
        before_ca = sample(c, "flexkv_py_swa_evicted_slots_total",
                           {"device": "cpu", "reason": "cascade"})
        freed = engine._evict_swa_slots(1)
        assert freed == 1
        assert sample(c, "flexkv_py_swa_evicted_slots_total",
                      {"device": "cpu", "reason": "pool_full"}) == before_pf + 1
        # drain happened inside _evict_swa_slots
        assert sample(c, "flexkv_py_swa_evicted_slots_total",
                      {"device": "cpu", "reason": "cascade"}) == before_ca + 1


class TestCacheQueryMetrics:
    def test_query_results(self, enabled_collector):
        c = enabled_collector
        for result in ("full", "partial", "miss"):
            before = sample(c, "flexkv_py_cache_query_total", {"result": result})
            c.record_cache_query(result)
            assert sample(c, "flexkv_py_cache_query_total", {"result": result}) == before + 1

    def test_match_blocks_ready_split(self, enabled_collector):
        c = enabled_collector
        before_r = sample(c, "flexkv_py_cache_match_blocks_total", {"device": "cpu", "ready": "true"})
        before_n = sample(c, "flexkv_py_cache_match_blocks_total", {"device": "cpu", "ready": "false"})
        c.record_cache_match_blocks("cpu", ready_blocks=6, not_ready_blocks=2)
        c.record_cache_match_blocks("cpu", ready_blocks=0, not_ready_blocks=0)  # ignored
        assert sample(c, "flexkv_py_cache_match_blocks_total",
                      {"device": "cpu", "ready": "true"}) == before_r + 6
        assert sample(c, "flexkv_py_cache_match_blocks_total",
                      {"device": "cpu", "ready": "false"}) == before_n + 2


class TestLayerwiseMetrics:
    def test_workers_progress(self, enabled_collector):
        c = enabled_collector
        c.update_layerwise_workers(2, 4)
        assert sample(c, "flexkv_py_layerwise_workers_ready") == 2
        assert sample(c, "flexkv_py_layerwise_workers_expected") == 4

    def test_submit_count_and_latency(self, enabled_collector):
        c = enabled_collector
        before_ok = sample(c, "flexkv_py_layerwise_submit_total", {"status": "ok"})
        before_err = sample(c, "flexkv_py_layerwise_submit_total", {"status": "error"})
        before_cnt = sample(c, "flexkv_py_layerwise_submit_seconds_count")
        c.record_layerwise_submit("ok", 0.25)
        c.record_layerwise_submit("error", 0.0)  # counted, but no latency sample
        assert sample(c, "flexkv_py_layerwise_submit_total", {"status": "ok"}) == before_ok + 1
        assert sample(c, "flexkv_py_layerwise_submit_total", {"status": "error"}) == before_err + 1
        assert sample(c, "flexkv_py_layerwise_submit_seconds_count") == before_cnt + 1


class TestDisabledPath:
    """Metrics off: everything must be a safe no-op."""

    @pytest.fixture()
    def disabled_collector(self, monkeypatch):
        monkeypatch.setattr(GLOBAL_CONFIG_FROM_ENV, "enable_metrics", False)
        c = FlexKVMetricsCollector(role="worker")
        assert not c.enabled
        return c

    def test_all_record_methods_noop(self, disabled_collector):
        c = disabled_collector
        c.record_swa_query("hit")
        c.record_swa_hit_blocks(3)
        c.update_swa_slot_stats("cpu", 1, 10)
        c.record_swa_slot_alloc_failed("cpu")
        c.record_swa_evicted("cpu", "pool_full", 1)
        c.record_cache_query("full")
        c.record_cache_match_blocks("cpu", 1, 1)
        c.update_layerwise_workers(1, 4)
        c.record_layerwise_submit("ok", 0.1)  # exercises DummyMetric.observe
        c.record_transfer_completed("H2D", 1, 8, "get", is_swa=True)

    def test_engine_hooks_noop(self, disabled_collector):
        engine = _make_engine_stub(disabled_collector,
                                   num_slots=2, index=_FakeIndex(evict_freed=1, drain_slots=[0]))
        engine._record_swa_slot_stats()
        assert engine._alloc_swa_slot() >= 0
        engine._free_swa_slot(0)
        assert engine._evict_swa_slots(1) == 1
        engine._drain_unmounted_swa_slots()
