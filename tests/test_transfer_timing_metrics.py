"""Unit tests for the transfer timing path added alongside the duration histograms.

Covers the four seams where a per-op duration can be silently lost or faked:
  * trace.configure_timing(): metrics export turns timestamp collection on
    without turning the [XFER] log lines on.
  * TransferEngine._accumulate_timing(): a PP-fan-out parent keeps its slowest
    replica, not whichever one was dequeued last.
  * KVTaskManager._merge_completed_op(): multi-handle completions keep the
    slowest handle instead of the first one to arrive.
  * FlexKVMetricsCollector.record_transfer_duration(): an untimed op
    (e2e_ms <= 0) is skipped rather than observed as a real zero.

Run with: pytest tests/test_transfer_timing_metrics.py -q
"""
import importlib

import numpy as np
import pytest

from flexkv.common.transfer import CompletedOp, TransferOp, TransferType
from flexkv.kvtask import KVTaskManager
from flexkv.metrics.collector import FlexKVMetricsCollector
from flexkv.transfer.transfer_engine import TransferEngine

pytestmark = pytest.mark.unit


def _reload_trace():
    import flexkv.transfer.trace as trace
    return importlib.reload(trace)


# --------------------------------------------------------------------------
# trace: timing is a superset of tracing
# --------------------------------------------------------------------------

def test_timing_off_by_default():
    trace = _reload_trace()
    assert trace.timing_enabled() is False
    # No collection means no entry is retained, so no unbounded dict growth.
    trace.set_submit_ns(1, 123)
    assert trace.consume_submit_ns(1) == 0.0


def test_configure_timing_enables_collection_without_trace_lines():
    trace = _reload_trace()
    trace.configure_timing(True)
    assert trace.timing_enabled() is True
    assert trace._TRACE_ON is False
    trace.set_submit_ns(7, 1)
    assert trace.consume_submit_ns(7) > 0.0


def test_configure_trace_also_enables_timing():
    trace = _reload_trace()
    trace.configure(True)
    assert trace._TRACE_ON is True
    assert trace.timing_enabled() is True


def test_configure_timing_false_does_not_disable_tracing():
    # Ordering independence: whichever consumer asks for timing first wins, and
    # the other one's "off" must not take it away again.
    trace = _reload_trace()
    trace.configure(True)
    trace.configure_timing(False)
    assert trace.timing_enabled() is True


# --------------------------------------------------------------------------
# TransferEngine._accumulate_timing: slowest replica wins
# --------------------------------------------------------------------------

def _op() -> TransferOp:
    empty = np.empty(0, dtype=np.int64)
    return TransferOp(
        graph_id=0,
        transfer_type=TransferType.H2D,
        src_block_ids=empty,
        dst_block_ids=empty,
    )


def test_accumulate_timing_first_value_is_assigned():
    op = _op()
    assert op.timing_ms is None
    TransferEngine._accumulate_timing(op, (1.0, 2.0, 3.0))
    assert op.timing_ms == (1.0, 2.0, 3.0)


def test_accumulate_timing_keeps_max_per_component():
    op = _op()
    TransferEngine._accumulate_timing(op, (5.0, 1.0, 9.0))
    TransferEngine._accumulate_timing(op, (2.0, 8.0, 4.0))
    assert op.timing_ms == (5.0, 8.0, 9.0)


def test_accumulate_timing_slower_replica_last_still_wins():
    # The arrival order of PP replicas is not controlled; both orders must give
    # the same parent duration.
    fast_last, slow_last = _op(), _op()
    TransferEngine._accumulate_timing(fast_last, (0.0, 0.0, 100.0))
    TransferEngine._accumulate_timing(fast_last, (0.0, 0.0, 1.0))
    TransferEngine._accumulate_timing(slow_last, (0.0, 0.0, 1.0))
    TransferEngine._accumulate_timing(slow_last, (0.0, 0.0, 100.0))
    assert fast_last.timing_ms == slow_last.timing_ms == (0.0, 0.0, 100.0)


def test_accumulate_timing_none_is_a_noop():
    op = _op()
    TransferEngine._accumulate_timing(op, None)
    assert op.timing_ms is None
    TransferEngine._accumulate_timing(op, (1.0, 2.0, 3.0))
    TransferEngine._accumulate_timing(op, None)
    assert op.timing_ms == (1.0, 2.0, 3.0)


# --------------------------------------------------------------------------
# KVTaskManager._merge_completed_op: slowest handle wins
# --------------------------------------------------------------------------

def _completed(**kwargs) -> CompletedOp:
    base = dict(graph_id=0, op_id=1, transfer_type="H2D", num_blocks=2, num_bytes=64)
    base.update(kwargs)
    return CompletedOp(**base)


def test_merge_completed_op_keeps_slowest_handle():
    first = _completed(wait_ms=1.0, xfer_ms=10.0, e2e_ms=12.0)
    second = _completed(wait_ms=5.0, xfer_ms=2.0, e2e_ms=30.0)
    merged = KVTaskManager._merge_completed_op(first, second)
    assert (merged.wait_ms, merged.xfer_ms, merged.e2e_ms) == (5.0, 10.0, 30.0)


def test_merge_completed_op_duration_is_arrival_order_independent():
    a = _completed(wait_ms=1.0, xfer_ms=10.0, e2e_ms=12.0)
    b = _completed(wait_ms=5.0, xfer_ms=2.0, e2e_ms=30.0)
    forward = KVTaskManager._merge_completed_op(a, b)
    reverse = KVTaskManager._merge_completed_op(b, a)
    assert (forward.wait_ms, forward.xfer_ms, forward.e2e_ms) == \
           (reverse.wait_ms, reverse.xfer_ms, reverse.e2e_ms)


def test_merge_completed_op_single_handle_passes_durations_through():
    only = _completed(wait_ms=1.5, xfer_ms=2.5, e2e_ms=4.0)
    merged = KVTaskManager._merge_completed_op(None, only)
    assert (merged.wait_ms, merged.xfer_ms, merged.e2e_ms) == (1.5, 2.5, 4.0)


# --------------------------------------------------------------------------
# collector.record_transfer_duration
# --------------------------------------------------------------------------

class _FakeHistogram:
    def __init__(self):
        self.observations = []
        self.label_calls = []

    def labels(self, **kwargs):
        self.label_calls.append(kwargs)
        return self

    def observe(self, value):
        self.observations.append(value)


class _FakeCollector:
    # Stands in for FlexKVMetricsCollector without running its __init__, which
    # would register global prometheus metrics and may start the HTTP server.
    def __init__(self, enabled=True):
        self.enabled = enabled
        self._duration_children = {}
        self.transfer_wait_duration_seconds = _FakeHistogram()
        self.transfer_xfer_duration_seconds = _FakeHistogram()
        self.transfer_e2e_duration_seconds = _FakeHistogram()

    record_transfer_duration = FlexKVMetricsCollector.record_transfer_duration


def test_record_transfer_duration_converts_ms_to_seconds():
    c = _FakeCollector()
    c.record_transfer_duration("H2D", wait_ms=1.0, xfer_ms=20.0, e2e_ms=500.0)
    assert c.transfer_wait_duration_seconds.observations == [0.001]
    assert c.transfer_xfer_duration_seconds.observations == [0.02]
    assert c.transfer_e2e_duration_seconds.observations == [0.5]


def test_record_transfer_duration_skips_untimed_ops():
    # e2e_ms == 0 means "never timed", not "took no time": recording it would
    # pull every quantile toward zero.
    c = _FakeCollector()
    c.record_transfer_duration("H2D", 0.0, 0.0, 0.0)
    c.record_transfer_duration("H2D", 1.0, 1.0, -1.0)
    assert c.transfer_e2e_duration_seconds.observations == []


def test_record_transfer_duration_noop_when_disabled():
    c = _FakeCollector(enabled=False)
    c.record_transfer_duration("H2D", 1.0, 1.0, 1.0)
    assert c.transfer_e2e_duration_seconds.observations == []


def test_record_transfer_duration_caches_label_children():
    # labels() is not free; the per-type children are resolved once.
    c = _FakeCollector()
    for _ in range(3):
        c.record_transfer_duration("H2D", 1.0, 1.0, 1.0)
    c.record_transfer_duration("D2H", 1.0, 1.0, 1.0)
    assert c.transfer_e2e_duration_seconds.label_calls == [
        {"transfer_type": "H2D"},
        {"transfer_type": "D2H"},
    ]
    assert len(c.transfer_e2e_duration_seconds.observations) == 4
