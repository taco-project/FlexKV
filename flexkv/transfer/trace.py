"""Transfer diagnosis tracing (single switch, raw-data only).

One switch: FLEXKV_TRANSFER_TRACE. When on, every transfer prints one
[XFER] line (lifecycle timing + backlog) plus a periodic [XFER-SUMMARY].
No verdict; raw data for the operator to grep. Uses flexkv_logger.info,
so it also obeys FLEXKV_LOG_LEVEL (default INFO => on).
"""
import time
import threading
from collections import defaultdict
from typing import Dict, List, Optional

from flexkv.common.debug import flexkv_logger

# Set by configure(); caller reads GLOBAL_CONFIG_FROM_ENV.
_TRACE_ON = False

# Summary window in seconds (hardcoded).
_SUMMARY_INTERVAL_S = 10.0

# Scheduler-side state (parent process).
_submit_lock = threading.Lock()
_submit_ns: Dict[int, int] = {}

_inflight_lock = threading.Lock()
_inflight = 0
_inflight_max_window = 0

# Summary accumulators (reset each window).
_sum_lock = threading.Lock()
_win_start = time.monotonic()
_wait_ms: List[float] = []
_xfer_ms: List[float] = []
_type_counts: Dict[str, int] = defaultdict(int)


def configure(enabled: bool) -> None:
    """Enable/disable tracing. Call once at startup from GLOBAL_CONFIG_FROM_ENV."""
    global _TRACE_ON
    _TRACE_ON = bool(enabled)


def set_submit_ns(op_id: int, submitted_ns: int) -> None:
    """Record submit time (scheduler side)."""
    if not _TRACE_ON:
        return
    with _submit_lock:
        _submit_ns[op_id] = submitted_ns


def consume_submit_ns(op_id: int) -> float:
    """e2e ms from submit to now; 0.0 if unknown."""
    if not _TRACE_ON:
        return 0.0
    with _submit_lock:
        t = _submit_ns.pop(op_id, None)
    if t is None:
        return 0.0
    return (time.perf_counter_ns() - t) / 1e6


def inc_inflight() -> None:
    """One op submitted to a worker."""
    if not _TRACE_ON:
        return
    global _inflight, _inflight_max_window
    with _inflight_lock:
        _inflight += 1
        if _inflight > _inflight_max_window:
            _inflight_max_window = _inflight


def dec_inflight() -> int:
    """One op finished; return current inflight."""
    if not _TRACE_ON:
        return 0
    global _inflight
    with _inflight_lock:
        _inflight = max(0, _inflight - 1)
        return _inflight


def inflight_value() -> int:
    if not _TRACE_ON:
        return 0
    with _inflight_lock:
        return _inflight


def build_worker_metrics(
    op,
    submitted_ns: int,
    received_ns: int,
    launch_ns: int,
    launched_ns: int,
    worker_id: int,
    bytes_per_block: int,
    single_kv_region: bool,
    is_h2d: bool,
) -> Optional[dict]:
    """Worker-side: ipc / wait / xfer + op metadata. None when off.

    bytes = bytes_per_block * num_blocks; backends without a block byte
    size pass 0 (bytes=0, bandwidth omitted).
    """
    if not _TRACE_ON:
        return None
    transfer_type = getattr(op, "transfer_type", None)
    type_name = transfer_type.name if transfer_type is not None else "UNKNOWN"

    nb = getattr(op, "valid_block_num", 0)
    if not nb:
        sb = getattr(op, "src_block_ids_h2d", None)
        nb = int(sb.size) if sb is not None else 0

    ipc_ms = max(0.0, (received_ns - submitted_ns) / 1e6)
    wait_ms = max(0.0, (launch_ns - received_ns) / 1e6)
    xfer_ms = max(0.0, (launched_ns - launch_ns) / 1e6)

    return {
        "type": type_name,
        "nb": int(nb),
        "graph_id": getattr(op, "transfer_graph_id", -1),
        "worker_id": worker_id,
        "is_h2d": bool(is_h2d),
        "single_kv_region": bool(single_kv_region),
        "bytes": int(bytes_per_block) * int(nb),
        "ipc_ms": ipc_ms,
        "wait_ms": wait_ms,
        "xfer_ms": xfer_ms,
    }


def record_xfer(op_id: int, metrics: Optional[dict], e2e_ms: float) -> None:
    """Engine-side: print one [XFER] line + update the summary window."""
    if not _TRACE_ON or metrics is None:
        return
    backlog = inflight_value()
    bytes_ = metrics.get("bytes", 0)
    xfer_ms = metrics.get("xfer_ms", 0.0)
    bw = (bytes_ / (xfer_ms / 1000.0) / 1e6) if (bytes_ > 0 and xfer_ms > 0) else 0.0

    flexkv_logger.info(
        "[XFER] op=%d type=%s nb=%d bytes=%d h2d=%d single_kv_region=%d "
        "ipc=%.3fms wait=%.3fms xfer=%.3fms e2e=%.3fms bw=%.0fMB/s "
        "worker=%d graph=%d backlog=%d",
        op_id,
        metrics["type"],
        metrics["nb"],
        bytes_,
        int(metrics["is_h2d"]),
        int(metrics["single_kv_region"]),
        metrics["ipc_ms"],
        metrics["wait_ms"],
        xfer_ms,
        e2e_ms,
        bw,
        metrics["worker_id"],
        metrics["graph_id"],
        backlog,
    )
    _accumulate(metrics)


def _accumulate(metrics: dict) -> None:
    global _win_start
    with _sum_lock:
        _wait_ms.append(metrics["wait_ms"])
        _xfer_ms.append(metrics["xfer_ms"])
        _type_counts[metrics["type"]] += 1
        now = time.monotonic()
        if now - _win_start >= _SUMMARY_INTERVAL_S:
            _flush_locked(now)


def _flush_locked(now: float) -> None:
    global _win_start, _inflight_max_window
    w_p50, w_p99 = _pct(_wait_ms), _pct(_wait_ms, 99)
    x_p50, x_p99 = _pct(_xfer_ms), _pct(_xfer_ms, 99)
    types = " ".join(f"{k}={v}" for k, v in sorted(_type_counts.items()))
    flexkv_logger.info(
        "[XFER-SUMMARY] window=%.0fs ops=%d inflight_max=%d "
        "wait_p50=%.3fms wait_p99=%.3fms xfer_p50=%.3fms xfer_p99=%.3fms "
        "type: %s",
        now - _win_start,
        len(_wait_ms),
        _inflight_max_window,
        w_p50,
        w_p99,
        x_p50,
        x_p99,
        types,
    )
    _win_start = now
    _wait_ms.clear()
    _xfer_ms.clear()
    _type_counts.clear()
    _inflight_max_window = inflight_value()


def _pct(vals: List[float], p: float = 50.0) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    k = max(0, min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1)))))
    return s[k]
