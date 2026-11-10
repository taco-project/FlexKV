"""Opt-in GET/PUT/PREFETCH timelines. No CUDA calls, polling, counters or new IPC.

Worker stage stamps are buffered and emitted once per logical op. Events are
JSONL in the configured file; scripts/report_put_trace.py reconstructs queue occupancy
from complete intervals across all request operations.
"""
import atexit
import fcntl
import json
import os
import re
import socket
import threading
import time
import uuid
import warnings
from contextlib import contextmanager, nullcontext
from functools import wraps

ENABLED = False
CACHE_DETAILS = False
RUN = ""
HOST = ""
OUTPUT_FILE = ""
_fd = None
_fd_pid = None
_broken = False
_io_lock = threading.Lock()
_local = threading.local()
_forward_stats = None

# Polling is a scheduler hot path. Aggregate in memory and write at most one
# process/rank summary per second instead of emitting one record per poll.
POLL_SUMMARY_INTERVAL_NS = 1_000_000_000
# Per-store diagnostics scan the inflight map. Above this limit retain the
# constant-cost process summary and suppress details so tracing does not turn
# a high-concurrency poll into O(number of stores) bookkeeping.
POLL_DETAIL_MAX_INFLIGHT = 32


def parse_config(value=None):
    config = {
        "enabled": False,
        "output_file": "put-trace.jsonl",
        "rotate_on_start": False,
        "cache_details": False,
    }
    if value is not None:
        if not isinstance(value, dict):
            raise ValueError("put_trace must be a mapping")
        unknown = value.keys() - config.keys()
        if unknown:
            raise ValueError(f"Unknown put_trace keys: {sorted(unknown)}")
        config.update(value)
    if type(config["enabled"]) is not bool:
        raise ValueError("put_trace.enabled must be a boolean")
    if not isinstance(config["output_file"], str) or not config["output_file"].strip():
        raise ValueError("put_trace.output_file must be a nonempty path")
    if type(config["rotate_on_start"]) is not bool:
        raise ValueError("put_trace.rotate_on_start must be a boolean")
    if type(config["cache_details"]) is not bool:
        raise ValueError("put_trace.cache_details must be a boolean")
    return config


def _flexkv_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def _resolve_output_file(output_file):
    def expand_variable(match):
        name = match.group(1) or match.group(2)
        value = os.environ.get(name)
        if not value:
            raise ValueError(f"put_trace.output_file requires nonempty environment variable {name}")
        return value

    path = re.sub(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))",
                  expand_variable, output_file.strip())
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(_flexkv_root(), path)
    return os.path.abspath(path)


def _run_stamp():
    """MMDDHHMMSS local-time stamp identifying one launch's trace file."""
    return time.strftime("%m%d%H%M%S", time.localtime())


def _trace_mode():
    """Transfer-mode tag (ce / gdr) for the trace filename.

    Explicit FLEXKV_PUT_TRACE_MODE wins; otherwise it is derived from the
    published GDR config (FLEXKV_GDR_CONFIG) so toggling gdr.enabled or
    gdr.h2d/d2h.path in the YAML is enough to tag the file. Falls back to 'na'.
    """
    override = os.environ.get("FLEXKV_PUT_TRACE_MODE", "").strip().lower()
    if override:
        return override
    raw = os.environ.get("FLEXKV_GDR_CONFIG")
    if not raw:
        return "na"
    try:
        gdr = json.loads(raw)
    except (ValueError, TypeError):
        return "na"
    if not gdr.get("enabled"):
        return "ce"
    ce_paths = {"ce", "original"}
    paths = {(gdr.get("h2d") or {}).get("path"), (gdr.get("d2h") or {}).get("path")}
    if paths <= ce_paths:
        return "ce"
    if "auto" in paths:
        return "auto"
    return "gdr"


def _rotated_output_file(path):
    """One file per SGLang launch: ``put-trace-<mode>-<MMDDHHMMSS>.jsonl``.

    ``mode`` is derived from the GDR config (ce/gdr), the timestamp from the
    launch time. Ranks/workers that share a parent inherit the resolved path
    and append to the same file.
    """
    directory, name = os.path.split(path)
    stem, ext = os.path.splitext(name)
    if not ext:
        ext = ".jsonl"
    return os.path.join(directory, f"{stem}-{_trace_mode()}-{_run_stamp()}{ext}")


def _close_file():
    global _fd, _fd_pid
    if _fd is not None:
        os.close(_fd)
    _fd = _fd_pid = None


def _after_fork():
    global _io_lock, _local, _forward_stats
    _close_file()
    _io_lock = threading.Lock()
    _local = threading.local()
    _forward_stats = None


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)
atexit.register(_close_file)


def _open_file():
    global _fd, _fd_pid
    if _fd is None or _fd_pid != os.getpid():
        _close_file()
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        _fd = os.open(OUTPUT_FILE, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        _fd_pid = os.getpid()


def configure(value, run_id=None):
    """Apply YAML in an already-imported process; propagate it through spawn.

    Called before workers start. Invalid/unwritable output fails at setup, not
    silently after the first PUT. Worker file descriptors are process-local.
    """
    global ENABLED, CACHE_DETAILS, RUN, HOST, OUTPUT_FILE, _broken, _forward_stats
    if os.environ.get("FLEXKV_CACHE_PROBE") == "1":
        from flexkv.transfer.cache_probe import output_dir, announce
        announce()
        value = dict(value or {}, enabled=True, cache_details=True,
                     output_file=os.path.join(output_dir(), f"trace-{socket.gethostname()}.jsonl"),
                     rotate_on_start=False)
    config = parse_config(value)
    path = _resolve_output_file(config["output_file"])
    if config["enabled"] and config["rotate_on_start"]:
        path = _rotated_output_file(path)
    with _io_lock:
        if path != OUTPUT_FILE or not config["enabled"]:
            _close_file()
        OUTPUT_FILE = path
        ENABLED = config["enabled"]
        CACHE_DETAILS = ENABLED and config["cache_details"]
        _broken = False
        _forward_stats = None
        if ENABLED:
            RUN = run_id or os.environ.get("FLEXKV_PUT_TRACE_RUN_ID") or uuid.uuid4().hex[:12]
            HOST = socket.gethostname()
            _open_file()
            os.environ["FLEXKV_PUT_TRACE_RUN_ID"] = RUN
        config["output_file"] = path
        # Children inherit the resolved file and must not rotate again.
        config["rotate_on_start"] = False
        os.environ["FLEXKV_PUT_TRACE_CONFIG"] = json.dumps(config)
    return config


# New children inherit the validated file configuration. Legacy opt-in also
# writes a file now; YAML is authoritative once configure() is called.
_environment_config = os.getenv("FLEXKV_PUT_TRACE_CONFIG")
if _environment_config:
    configure(json.loads(_environment_config))
elif os.getenv("FLEXKV_PUT_TRACE", "0") == "1":
    configure({"enabled": True,
               "output_file": os.getenv("FLEXKV_PUT_TRACE_FILE", "put-trace.jsonl")})


def now():
    return time.perf_counter_ns() if ENABLED else 0


def cache_event(event, hashes=None, **fields):
    """Reuse existing CPU block hashes; no lookup, token hashing or GPU work."""
    if not CACHE_DETAILS:
        return
    if hashes is not None:
        fields["block_hashes"] = [f"{int(value) & ((1 << 64) - 1):016x}" for value in hashes]
    emit(event, **fields)


def emit(event, at=None, **fields):
    global _broken
    if not ENABLED or _broken:
        return
    # One buffered worker timeline per record, not one write per stage.
    # flock protects whole JSON records across processes, including short writes.
    # No fsync: a record is visible immediately but not crash-durable storage.
    try:
        mono = time.perf_counter_ns()
        stamp = mono if at is None else at
        record = dict(v=1, run=RUN, host=HOST, pid=os.getpid(), event=event,
                      ns=stamp, wall_ns=time.time_ns() - mono + stamp, **fields)
        data = (json.dumps(record, separators=(",", ":")) + "\n").encode("utf-8")
        with _io_lock:
            _open_file()
            fcntl.flock(_fd, fcntl.LOCK_EX)
            try:
                remaining = memoryview(data)
                while remaining:
                    written = os.write(_fd, remaining)
                    if written <= 0:
                        raise OSError("zero-length PUT trace write")
                    remaining = remaining[written:]
            finally:
                fcntl.flock(_fd, fcntl.LOCK_UN)
    except Exception as exc:
        # Disk errors must not fail transfers or spam stderr per operation.
        _broken = True
        try:
            warnings.warn(f"PUT trace disabled in pid {os.getpid()}: {exc}", RuntimeWarning)
        except Exception:
            pass


@contextmanager
def _poll_stage(event):
    stages = _local.poll_stages
    start = now()
    status = "error"
    try:
        yield
        status = "ok"
    finally:
        end = now()
        stages[event] = dict(begin_ns=start, end_ns=end,
                             duration_ns=end-start, status=status)


def poll_stage(event):
    if ENABLED and getattr(_local, "poll_stages", None) is not None:
        return _poll_stage(event)
    return nullcontext()


def record_forward_span(begin_ns, end_ns, *, rank, pp_rank, dp_rank, mode,
                        batch_size, status):
    """Accumulate model-forward spans without writing per-batch records."""
    global _forward_stats
    if not ENABLED or begin_ns <= 0 or end_ns < begin_ns:
        return
    stats = _forward_stats
    if stats is None or stats["run"] != RUN:
        stats = _forward_stats = dict(
            run=RUN,
            count=0,
            total_ns=0,
            max_ns=0,
            error_count=0,
        )
    duration = end_ns - begin_ns
    stats["count"] += 1
    stats["total_ns"] += duration
    if status != "ok":
        stats["error_count"] += 1
    if duration > stats["max_ns"]:
        stats.update(
            max_ns=duration,
            max_begin_ns=begin_ns,
            max_end_ns=end_ns,
            max_rank=rank,
            max_pp_rank=pp_rank,
            max_dp_rank=dp_rank,
            max_mode=mode,
            max_batch_size=batch_size,
            max_status=status,
        )


def _take_forward_stats():
    """Return and reset process-local forward aggregates for a poll summary."""
    global _forward_stats
    stats = _forward_stats
    _forward_stats = None
    if stats is None:
        return dict(
            forward_count=0,
            total_forward_ns=0,
            max_forward_ns=0,
            forward_error_count=0,
        )
    result = dict(
        forward_count=stats["count"],
        total_forward_ns=stats["total_ns"],
        max_forward_ns=stats["max_ns"],
        forward_error_count=stats["error_count"],
    )
    if stats["max_ns"]:
        result.update(
            max_forward_begin_ns=stats["max_begin_ns"],
            max_forward_end_ns=stats["max_end_ns"],
            max_forward_rank=stats["max_rank"],
            max_forward_pp_rank=stats["max_pp_rank"],
            max_forward_dp_rank=stats["max_dp_rank"],
            max_forward_mode=stats["max_mode"],
            max_forward_batch_size=stats["max_batch_size"],
            max_forward_status=stats["max_status"],
        )
    return result


def _new_poll_summary(start, rank):
    return dict(
        run=RUN,
        rank=rank,
        window_begin_ns=start,
        next_emit_ns=start + POLL_SUMMARY_INTERVAL_NS,
        last_poll_ns=0,
        poll_count=0,
        active_poll_count=0,
        empty_poll_count=0,
        error_count=0,
        total_poll_ns=0,
        max_poll_ns=0,
        max_gap_ns=0,
        max_inflight_stores=0,
        detail_suppressed_poll_count=0,
        total_try_wait_ns=0,
        max_try_wait_ns=0,
        total_sync_ns=0,
        max_sync_ns=0,
    )


def trace_store_poll(fn):
    """Per-task milestones plus one aggregate record per second, never per poll."""
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        if not ENABLED:
            return fn(self, *args, **kwargs)
        inflight_count = len(self._inflight_stores)
        detail_enabled = inflight_count <= POLL_DETAIL_MAX_INFLIGHT
        pending = (
            dict(self._inflight_stores)
            if detail_enabled and self._inflight_stores
            else {}
        )
        if not pending and hasattr(self, "_put_poll_stats"):
            self._put_poll_stats.clear()
        stats = getattr(self, "_put_poll_stats", None)
        if stats is None:
            stats = self._put_poll_stats = {}
        # Cancellation/reset may remove entries outside this method.
        for rid in list(stats):
            if rid not in pending or stats[rid]["task"] != pending[rid]:
                del stats[rid]
        start = now()
        rank = self._sync_ctx.world_rank
        summary = getattr(self, "_put_poll_summary", None)
        if summary is None or summary["run"] != RUN or summary["rank"] != rank:
            summary = self._put_poll_summary = _new_poll_summary(start, rank)
        for rid, task in pending.items():
            if rid not in stats:
                stats[rid] = dict(task=int(task), first_poll_ns=start, last_poll_ns=start,
                                 poll_count=0, error_count=0, max_gap_ns=0,
                                 max_poll_ns=0, stages={})
                emit("store_poll_first", at=start, task=int(task), rid=str(rid), rank=rank)
        previous = getattr(_local, "poll_stages", None)
        stages = {}
        _local.poll_stages = stages
        failed = True
        try:
            result = fn(self, *args, **kwargs)
            failed = False
            return result
        finally:
            end = now()
            _local.poll_stages = previous
            error = failed or any(v["status"] == "error" for v in stages.values())
            duration = end - start

            gap = start - summary["last_poll_ns"] if summary["last_poll_ns"] else 0
            summary["last_poll_ns"] = start
            summary["poll_count"] += 1
            summary[
                "active_poll_count" if inflight_count else "empty_poll_count"
            ] += 1
            summary["error_count"] += int(error)
            summary["total_poll_ns"] += duration
            summary["max_poll_ns"] = max(summary["max_poll_ns"], duration)
            summary["max_gap_ns"] = max(summary["max_gap_ns"], gap)
            summary["max_inflight_stores"] = max(
                summary["max_inflight_stores"], inflight_count
            )
            summary["detail_suppressed_poll_count"] += int(not detail_enabled)
            for stage_name, total_name, max_name in (
                ("store_try_wait", "total_try_wait_ns", "max_try_wait_ns"),
                ("store_completion_sync", "total_sync_ns", "max_sync_ns"),
            ):
                stage = stages.get(stage_name)
                if stage is not None:
                    stage_duration = stage["duration_ns"]
                    summary[total_name] += stage_duration
                    summary[max_name] = max(summary[max_name], stage_duration)

            for rid, task in pending.items():
                item = stats[rid]
                gap = start - item["last_poll_ns"]
                if gap > item["max_gap_ns"]:
                    item.update(max_gap_ns=gap, gap_begin_ns=item["last_poll_ns"], gap_end_ns=start)
                item["last_poll_ns"] = start
                item["poll_count"] += 1
                if end-start > item["max_poll_ns"]:
                    item.update(max_poll_ns=end-start, max_poll_begin_ns=start, max_poll_end_ns=end)
                for name, value in stages.items():
                    old = item["stages"].get(name)
                    if old is None or value["duration_ns"] > old["duration_ns"]:
                        item["stages"][name] = value
                if error:
                    item["error_count"] += 1
                    if item["error_count"] == 1:
                        emit("store_poll_error", task=int(task), rid=str(rid), rank=rank,
                             begin_ns=start, end_ns=end, stages=stages)
                if rid not in self._inflight_stores:
                    emit("store_poll_complete", rid=str(rid), rank=rank, **item)
                    del stats[rid]
            # Once adaptive polling becomes idle there may be no later call to
            # flush a sub-second window. Emit once on the active->idle edge;
            # this is per completion batch, never per poll.
            became_idle = inflight_count > 0 and not self._inflight_stores
            if end >= summary["next_emit_ns"] or became_idle:
                fields = {
                    key: value
                    for key, value in summary.items()
                    if key not in ("run", "next_emit_ns", "last_poll_ns")
                }
                fields.update(
                    window_end_ns=end,
                    inflight_at_emit=len(self._inflight_stores),
                    **_take_forward_stats(),
                )
                emit("store_poll_summary", at=end, **fields)
                next_summary = _new_poll_summary(end, rank)
                # Gaps are measured between poll starts. Preserve this start
                # across summary windows so a boundary cannot hide the next
                # scheduling gap.
                next_summary["last_poll_ns"] = start
                self._put_poll_summary = next_summary
    return wrapped


def request_created(task_id, operation="put"):
    stamp = now()
    if ENABLED:
        emit("request_created", at=stamp, request=f"{RUN}/{os.getpid()}/{task_id}",
             task=task_id, operation=operation)
    return stamp


def bind_graph(graph, task_id, created, children=(), operation="put"):
    if not ENABLED:
        return
    context = dict(request=f"{RUN}/{os.getpid()}/{task_id}", task=task_id,
                   graph=graph.graph_id, operation=operation)
    graph._put_trace = context
    for op in graph._op_map.values():
        op._put_trace = context
    emit("planned", created_ns=created, children=list(children),
         ops=[dict(op=o.op_id, direction=o.transfer_type.name,
                   predecessors=sorted(o.predecessors)) for o in graph._op_map.values()],
         **context)


def graph_event(event, graph, **fields):
    if ENABLED and getattr(graph, "_put_trace", None):
        emit(event, **graph._put_trace, **fields)


def task_event(event, task, **fields):
    if ENABLED and getattr(task, "graph", None) is not None:
        graph_event(event, task.graph, **fields)


def replica(parent, child):
    if ENABLED and getattr(parent, "_put_trace", None):
        child._put_trace = parent._put_trace
        child._put_parent = parent.op_id


def submitted(op, worker_op, worker_id):
    if not ENABLED:
        return
    stamp = now()
    queue = f"{HOST}/{RUN}/{os.getpid()}/worker-{worker_id}"
    context = dict(getattr(op, "_put_trace", {}))
    context.update(ticket=f"{queue}/op-{op.op_id}", queue=queue,
                   op=op.op_id, parent_op=getattr(op, "_put_parent", op.op_id),
                   graph=op.graph_id, worker=worker_id, direction=op.transfer_type.name,
                   pool="swa" if getattr(op, "is_swa", False) else "main",
                   submitted_ns=stamp)
    worker_op._put_diag = context
    emit("worker_submit", at=stamp, **context)


def batch_prepared(ops, start):
    if not ENABLED:
        return
    end = now()
    for op in ops:
        if not isinstance(op, dict):
            op._put_batch_prepare = (start, end)


def worker_begin(op, start):
    if not ENABLED:
        return
    # Record all directions: a GET in the same worker can block a PUT.
    preparation = getattr(op, "_put_batch_prepare", None)
    _local.points = ([("batch_prepare_begin", preparation[0]),
                      ("batch_prepare_end", preparation[1])] if preparation else [])
    _local.points.append(("worker_start", start))
    _local.volume = {}



def volume(**fields):
    """Payload metadata for this physical execution, buffered with its stamps."""
    if ENABLED and getattr(_local, "points", None) is not None:
        _local.volume.update(fields)


def mark(stage):
    if ENABLED:
        points = getattr(_local, "points", None)
        if points is not None:
            points.append((stage, time.perf_counter_ns()))


def launch_mark(stage):
    if ENABLED:
        points = getattr(_local, "launch_points", None)
        if points is not None:
            points.append((stage, time.perf_counter_ns()))


def trace_layerwise_launch(fn):
    """Reuse the existing SGLang launch record: buffer, do not write here."""
    @wraps(fn)
    def wrapped(self, *args, **kwargs):
        if not ENABLED:
            return fn(self, *args, **kwargs)
        previous = getattr(_local, "launch_points", None)
        points = [("launch_begin", time.perf_counter_ns())]
        _local.launch_points = points
        try:
            return fn(self, *args, **kwargs)
        finally:
            points.append(("launch_return", time.perf_counter_ns()))
            self._trace_layerwise_launch_points = points
            _local.launch_points = previous
    return wrapped


def worker_end(op, end, status):
    if not ENABLED:
        return
    try:
        points = getattr(_local, "points", None) or []
        amount = dict(getattr(_local, "volume", {}))
        shared = getattr(op, "_gdr_batch", None)
        physical = getattr(op, "_put_diag", {}).get("ticket", "")
        if shared is not None and getattr(shared, "done", False):
            if not hasattr(shared, "_put_points"):
                shared._put_volume = amount
                shared._put_points = list(points)
                shared._put_physical = physical
            amount = shared._put_volume
            points = list(shared._put_points)
            physical = shared._put_physical
            end = shared.finished_ns
        points = list(points) + [("worker_end", end)]
        emit("worker_done", at=end, **getattr(op, "_put_diag", {}),
             received_ns=getattr(op, "_received_ns", 0), physical=physical,
             status=status, points=points, volume=amount,
             logical_blocks=getattr(op, "valid_block_num", None))
    finally:
        _local.points = None
        _local.volume = None
