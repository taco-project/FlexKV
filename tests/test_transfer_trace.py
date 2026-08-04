"""Unit tests for flexkv.transfer.trace (transfer diagnosis tracing).

Run with: pytest tests/test_transfer_trace.py -s
"""
import importlib
import logging
import os

import pytest


class _CaptureHandler(logging.Handler):
    # Collect formatted messages from the FLEXKV logger (propagate is off,
    # so caplog on the root logger never sees them).
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _reload_with_env(monkeypatch, value):
    if value is None:
        monkeypatch.delenv("FLEXKV_TRANSFER_TRACE", raising=False)
    else:
        monkeypatch.setenv("FLEXKV_TRANSFER_TRACE", value)
    import flexkv.transfer.trace as trace
    trace = importlib.reload(trace)
    trace.configure(os.getenv("FLEXKV_TRANSFER_TRACE") is not None)
    return trace


class _FakeType:
    def __init__(self, name):
        self.name = name


class _FakeOp:
    def __init__(self, valid_block_num=10, graph_id=7, type_name="H2D"):
        self.prof_submitted_ns = 0
        self.valid_block_num = valid_block_num
        self.transfer_graph_id = graph_id
        self.transfer_type = _FakeType(type_name)


def test_trace_off_returns_none(monkeypatch):
    trace = _reload_with_env(monkeypatch, None)
    assert trace._TRACE_ON is False
    op = _FakeOp()
    assert trace.build_worker_metrics(op, 1_000_000, 2_000_000,
                                       3_000_000, 4_000_000, 0, 0, False,
                                       True) is None


def test_build_worker_metrics_timing(monkeypatch):
    trace = _reload_with_env(monkeypatch, "1")
    assert trace._TRACE_ON is True
    # 1e6 ns == 1 ms
    op = _FakeOp()
    m = trace.build_worker_metrics(
        op,
        submitted_ns=0,        # submit at 0
        received_ns=1_000_000,    # +1 ms ipc
        launch_ns=3_000_000,    # +2 ms wait
        launched_ns=8_000_000,    # +5 ms xfer
        worker_id=2,
        bytes_per_block=4096,
        is_mla=True,
        is_h2d=True,
    )
    assert m["ipc_ms"] == pytest.approx(1.0)
    assert m["wait_ms"] == pytest.approx(2.0)
    assert m["xfer_ms"] == pytest.approx(5.0)
    assert m["type"] == "H2D"
    assert m["nb"] == 10
    assert m["bytes"] == 4096 * 10
    assert m["is_h2d"] is True
    assert m["is_mla"] is True
    assert m["graph_id"] == 7
    assert m["worker_id"] == 2


def test_record_xfer_and_summary(monkeypatch):
    trace = _reload_with_env(monkeypatch, "1")
    # Force the summary window stale so the flush triggers within this test.
    with trace._sum_lock:
        trace._win_start = trace._win_start - trace._SUMMARY_INTERVAL_S - 1
        trace._wait_ms.clear()
        trace._xfer_ms.clear()
        trace._type_counts.clear()
        trace._inflight_max_window = 0

    cap = _CaptureHandler()
    target = trace.flexkv_logger.logger  # logging.getLogger("FLEXKV")
    prev_level = target.level
    target.setLevel(logging.INFO)
    target.addHandler(cap)
    try:
        trace.set_submit_ns(100, 0)
        trace.inc_inflight()  # inflight == 1 before record_xfer prints backlog
        metrics = trace.build_worker_metrics(
            _FakeOp(valid_block_num=4, type_name="D2H"),
            submitted_ns=0, received_ns=500_000, launch_ns=1_500_000,
            launched_ns=6_500_000,
            worker_id=0, bytes_per_block=1024, is_mla=False, is_h2d=False)
        trace.record_xfer(100, metrics, e2e_ms=7.0)
    finally:
        target.removeHandler(cap)
        target.setLevel(prev_level)

    lines = "\n".join(cap.messages)
    assert "[XFER] op=100" in lines
    assert "type=D2H" in lines
    assert "bytes=4096" in lines
    assert "backlog=1" in lines   # inflight not yet decremented at print time
    # Summary should have flushed (window forced stale).
    assert "[XFER-SUMMARY]" in lines
    assert "inflight_max=" in lines
    assert "type: D2H=1" in lines


def test_pct(monkeypatch):
    trace = _reload_with_env(monkeypatch, "1")
    vals = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert trace._pct(vals, 50) == 3.0
    assert trace._pct(vals, 99) == 5.0
    assert trace._pct([], 50) == 0.0
