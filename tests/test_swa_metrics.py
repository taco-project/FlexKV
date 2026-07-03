# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SWA hit/miss telemetry on FlexKVMetricsCollector.

Pure-logic coverage (no torch / GPU / prometheus_client needed): asserts the
record_swa_hit / record_swa_miss methods gate on ``enabled`` and non-positive
counts exactly like their Full-KV peers (record_cache_hit / record_cache_miss),
and route to the right counter with the right device label.

Why this matters: without a SWA reuse-rate signal, an SWA-LRU eviction regression
(e.g. the list degrading toward FIFO because a hit never re-promotes the node to
MRU) is invisible in production — no gauge would ever move. These counters are the
denominator/numerator of that signal, so the recording contract is worth pinning.
The swa_align emission path itself needs a GlobalCacheEngine (CUDA) and is covered
by the e2e control-plane test; here we pin the collector contract.
"""
import pytest

from flexkv.metrics.collector import FlexKVMetricsCollector

pytestmark = pytest.mark.unit


class _FakeCounter:
    """Minimal Counter stand-in: records .labels()/.inc() calls.

    Mirrors the prometheus_client Counter surface the collector uses so the test
    is independent of whether prometheus_client is installed.
    """

    def __init__(self):
        self.total = 0.0
        self.by_label = {}
        self._pending_label = None

    def labels(self, **kwargs):
        # single-label counters here: device=...
        self._pending_label = tuple(sorted(kwargs.items()))
        return self

    def inc(self, amount=1):
        if self._pending_label is not None:
            self.by_label[self._pending_label] = (
                self.by_label.get(self._pending_label, 0.0) + amount)
            self._pending_label = None
        else:
            self.total += amount


def _make_collector(enabled: bool) -> FlexKVMetricsCollector:
    """Build a collector without touching the metrics server / prometheus, then
    swap in fake counters so we can inspect calls. Bypasses __init__ (which would
    try to auto-start the metrics server)."""
    c = FlexKVMetricsCollector.__new__(FlexKVMetricsCollector)
    c.enabled = enabled
    c.swa_hit_blocks_total = _FakeCounter()
    c.swa_miss_blocks_total = _FakeCounter()
    return c


def test_record_swa_hit_increments_by_device():
    c = _make_collector(enabled=True)
    c.record_swa_hit("cpu", 3)
    c.record_swa_hit("cpu", 2)
    c.record_swa_hit("ssd", 4)
    assert c.swa_hit_blocks_total.by_label[(("device", "cpu"),)] == 5
    assert c.swa_hit_blocks_total.by_label[(("device", "ssd"),)] == 4


def test_record_swa_miss_increments_total():
    c = _make_collector(enabled=True)
    c.record_swa_miss(7)
    c.record_swa_miss(1)
    assert c.swa_miss_blocks_total.total == 8


@pytest.mark.parametrize("num_blocks", [0, -1, -100])
def test_non_positive_counts_are_noops(num_blocks):
    c = _make_collector(enabled=True)
    c.record_swa_hit("cpu", num_blocks)
    c.record_swa_miss(num_blocks)
    assert c.swa_hit_blocks_total.by_label == {}
    assert c.swa_miss_blocks_total.total == 0.0


def test_disabled_collector_records_nothing():
    c = _make_collector(enabled=False)
    c.record_swa_hit("cpu", 5)
    c.record_swa_miss(5)
    assert c.swa_hit_blocks_total.by_label == {}
    assert c.swa_miss_blocks_total.total == 0.0


def test_dummy_metrics_have_swa_attributes():
    """When prometheus is unavailable the collector installs dummy metrics; the
    SWA counters must be among them so record_swa_* never AttributeErrors."""
    c = FlexKVMetricsCollector.__new__(FlexKVMetricsCollector)
    c._init_dummy_metrics()
    assert hasattr(c, "swa_hit_blocks_total")
    assert hasattr(c, "swa_miss_blocks_total")
    # dummy .labels().inc() must be a no-op that does not raise
    c.swa_hit_blocks_total.labels(device="cpu").inc(3)
    c.swa_miss_blocks_total.inc(3)
