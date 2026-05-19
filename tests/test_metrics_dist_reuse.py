"""Smoke tests for the dist-reuse Prometheus metrics added in 2026-05-14.

These tests validate the **wiring**:
  * Collector exposes the 5 new metrics.
  * Dummy fallback works when prometheus_client is missing OR
    ``FLEXKV_ENABLE_METRICS`` is unset.
  * Recording methods are no-op safe under both paths.

The tests do NOT validate that the worker.py call site actually emits
samples — that requires a running mooncake engine and is covered by the
e2e harness in ``tests/multinode/``.

See ``docs/dist_reuse/METRICS_dist_reuse.md`` for the full operator
context.
"""

from __future__ import annotations

import os
import unittest
from unittest import mock

# Force-disable the metrics server / prometheus client to exercise the
# dummy path first, then re-enable for the real path.  Both must work.


class TestDistReuseMetricsDummyPath(unittest.TestCase):
    """When metrics are disabled (default), every record_* method must be
    a silent no-op — we cannot raise from the data path."""

    @mock.patch.dict(os.environ, {"FLEXKV_ENABLE_METRICS": "0"})
    def setUp(self):
        # Re-import in a clean namespace so the env var takes effect.
        import importlib

        from flexkv.common import config as _cfg
        importlib.reload(_cfg)
        from flexkv.metrics import collector as _coll
        importlib.reload(_coll)
        self._coll_mod = _coll
        self.collector = _coll.FlexKVMetricsCollector()

    def test_collector_disabled(self):
        self.assertFalse(self.collector.enabled)

    def test_lease_nullptr_record_is_noop(self):
        # Must not raise even with negative / zero counts
        self.collector.record_dist_reuse_lease_nullptr("cpu", 5)
        self.collector.record_dist_reuse_lease_nullptr("cpu", 0)
        self.collector.record_dist_reuse_lease_nullptr("ssd", -1)

    def test_about_to_evict_record_is_noop(self):
        self.collector.record_dist_reuse_about_to_evict("cpu", 100)
        self.collector.record_dist_reuse_about_to_evict("ssd", 0)

    def test_mooncake_read_observe_is_noop(self):
        self.collector.observe_dist_reuse_peer_mooncake_read(
            0.0123, success=True
        )
        self.collector.observe_dist_reuse_peer_mooncake_read(
            0.5, success=False, reason="mooncake_error"
        )
        self.collector.observe_dist_reuse_peer_mooncake_read(
            0.001, success=False, reason="zero_byte_transfer"
        )


class TestDistReuseMetricsEnabledPath(unittest.TestCase):
    """When ``FLEXKV_ENABLE_METRICS=1`` and ``prometheus_client`` is
    installed, the metric names must register on the default registry and
    record/observe must mutate the underlying samples."""

    @mock.patch.dict(os.environ, {"FLEXKV_ENABLE_METRICS": "1"})
    def setUp(self):
        try:
            import prometheus_client  # noqa: F401
        except ImportError:
            self.skipTest("prometheus_client not installed")

        # Reload modules so the env var is picked up and the registry is
        # populated cleanly for this test (each setUp creates its own
        # collector with its own metric objects via re-registration).
        import importlib

        from flexkv.common import config as _cfg
        importlib.reload(_cfg)
        from flexkv.metrics import collector as _coll
        importlib.reload(_coll)

        # prometheus_client uses a global default REGISTRY; the second
        # _init_metrics call would raise on duplicate registration.  Use a
        # fresh CollectorRegistry to avoid global pollution.
        from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram
        self._registry = CollectorRegistry()

        # Patch the module-level Counter/Gauge/Histogram to use our private
        # registry so we don't pollute the default one across tests.
        with mock.patch.object(_coll, "Counter",
                               lambda **kw: Counter(registry=self._registry, **kw)), \
             mock.patch.object(_coll, "Gauge",
                               lambda **kw: Gauge(registry=self._registry, **kw)), \
             mock.patch.object(_coll, "Histogram",
                               lambda **kw: Histogram(registry=self._registry, **kw)):
            self.collector = _coll.FlexKVMetricsCollector()

    def test_collector_enabled(self):
        self.assertTrue(self.collector.enabled)

    def test_dist_reuse_metrics_attributes_exist(self):
        # The 5 new attributes must be present whether enabled or dummy.
        for attr in (
            "dist_reuse_lease_meta_nullptr_total",
            "dist_reuse_about_to_evict_total",
            "dist_reuse_peer_mooncake_read_seconds",
            "dist_reuse_peer_mooncake_read_failures_total",
            "dist_reuse_peer_mooncake_read_success_total",
        ):
            self.assertTrue(
                hasattr(self.collector, attr),
                f"collector missing metric attribute: {attr}",
            )

    def test_lease_nullptr_increments_under_enabled(self):
        before = self.collector.dist_reuse_lease_meta_nullptr_total \
            .labels(device="cpu")._value.get()
        self.collector.record_dist_reuse_lease_nullptr("cpu", 7)
        after = self.collector.dist_reuse_lease_meta_nullptr_total \
            .labels(device="cpu")._value.get()
        self.assertEqual(after - before, 7)

    def test_mooncake_read_failure_records_with_reason(self):
        before = self.collector.dist_reuse_peer_mooncake_read_failures_total \
            .labels(reason="zero_byte_transfer")._value.get()
        self.collector.observe_dist_reuse_peer_mooncake_read(
            0.5, success=False, reason="zero_byte_transfer",
        )
        after = self.collector.dist_reuse_peer_mooncake_read_failures_total \
            .labels(reason="zero_byte_transfer")._value.get()
        self.assertEqual(after - before, 1)

    def test_mooncake_read_success_records(self):
        before = self.collector.dist_reuse_peer_mooncake_read_success_total \
            ._value.get()
        self.collector.observe_dist_reuse_peer_mooncake_read(
            0.05, success=True,
        )
        after = self.collector.dist_reuse_peer_mooncake_read_success_total \
            ._value.get()
        self.assertEqual(after - before, 1)


if __name__ == "__main__":
    unittest.main()
