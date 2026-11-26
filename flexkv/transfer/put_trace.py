"""Opt-in request/graph/worker JSONL timelines, independent of transfer tracing."""

import atexit
import fcntl
import json
import os
import re
import socket
import stat
import threading
import time
import uuid
import warnings

ENABLED = False
RUN = ""
HOST = ""
OUTPUT_FILE = ""
_fd = None
_fd_pid = None
_broken = False
_io_lock = threading.Lock()
_local = threading.local()


def parse_config(value=None):
    config = {"enabled": False, "output_file": "put-trace.jsonl"}
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
    return config


def _resolve_output_file(value):
    def expand(match):
        name = match.group(1) or match.group(2)
        if not os.environ.get(name):
            raise ValueError(f"put_trace.output_file requires environment variable {name}")
        return os.environ[name]

    path = re.sub(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))", expand, value)
    path = os.path.expanduser(path)
    if not os.path.isabs(path):
        path = os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, path)
    return os.path.abspath(path)


def _close_file():
    global _fd, _fd_pid
    fd = _fd
    _fd = _fd_pid = None
    if fd is not None:
        os.close(fd)


def _after_fork():
    global _io_lock, _local
    _close_file()
    _io_lock = threading.Lock()
    _local = threading.local()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_after_fork)
atexit.register(_close_file)


def _open_file():
    global _fd, _fd_pid
    if _fd is None or _fd_pid != os.getpid():
        _close_file()
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        flags = os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NONBLOCK
        flags |= getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        fd = os.open(OUTPUT_FILE, flags, 0o600)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise ValueError("put_trace.output_file must be a regular file")
        _fd, _fd_pid = fd, os.getpid()


def configure(value=None, run_id=None):
    """Validate/open at setup and publish the resolved configuration for spawn."""
    global ENABLED, RUN, HOST, OUTPUT_FILE, _broken
    config = parse_config(value)
    path = _resolve_output_file(config["output_file"]) if config["enabled"] else ""
    with _io_lock:
        if path != OUTPUT_FILE or not config["enabled"]:
            _close_file()
        ENABLED = False
        OUTPUT_FILE = path
        _broken = False
        RUN = ((run_id or os.environ.get("FLEXKV_PUT_TRACE_RUN_ID") or uuid.uuid4().hex[:12])
               if config["enabled"] else "")
        HOST = socket.gethostname() if config["enabled"] else ""
        if config["enabled"]:
            _open_file()
            ENABLED = True
            config["output_file"] = path
            os.environ["FLEXKV_PUT_TRACE_RUN_ID"] = RUN
        os.environ["FLEXKV_PUT_TRACE_CONFIG"] = json.dumps(config)
    return config


def now():
    return time.perf_counter_ns() if ENABLED else 0


def emit(event, at=None, **fields):
    global _broken
    if not ENABLED or _broken:
        return
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
        _broken = True
        try:
            warnings.warn(f"PUT trace disabled in pid {os.getpid()}: {exc}", RuntimeWarning)
        except Exception:
            pass


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
         ops=[dict(op=op.op_id, direction=op.transfer_type.name,
                   predecessors=sorted(op.predecessors)) for op in graph._op_map.values()], **context)


def graph_event(event, graph, **fields):
    if ENABLED and getattr(graph, "_put_trace", None):
        emit(event, **graph._put_trace, **fields)


def task_event(event, task, **fields):
    if ENABLED:
        context = getattr(task, "_put_trace", None) or getattr(getattr(task, "graph", None), "_put_trace", None)
        if context:
            emit(event, **context, **fields)


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
    context.update(ticket=f"{queue}/op-{op.op_id}", queue=queue, op=op.op_id,
                   parent_op=getattr(op, "_put_parent", op.op_id), graph=op.graph_id,
                   worker=worker_id, direction=op.transfer_type.name,
                   pool="swa" if getattr(op, "is_swa", False) else "main", submitted_ns=stamp)
    worker_op._put_diag = context
    emit("worker_submit", at=stamp, **context)


def batch_prepared(ops, start):
    if ENABLED:
        end = now()
        for op in ops:
            if not isinstance(op, dict):
                op._put_batch_prepare = (start, end)


def worker_begin(op, start):
    if ENABLED:
        preparation = getattr(op, "_put_batch_prepare", None)
        _local.points = ([("batch_prepare_begin", preparation[0]), ("batch_prepare_end", preparation[1])]
                         if preparation else [])
        _local.points.append(("worker_start", start))
        _local.volume = {}


def volume(**fields):
    if ENABLED and getattr(_local, "points", None) is not None:
        _local.volume.update(fields)


def mark(stage):
    if ENABLED and getattr(_local, "points", None) is not None:
        _local.points.append((stage, time.perf_counter_ns()))


def worker_end(op, end, status):
    if not ENABLED:
        return
    try:
        points = list(getattr(_local, "points", None) or [])
        amount = dict(getattr(_local, "volume", None) or {})
        shared = getattr(op, "_gdr_batch", None)
        physical = getattr(op, "_put_diag", {}).get("ticket", "")
        if shared is not None and getattr(shared, "done", False):
            if not hasattr(shared, "_put_points"):
                shared._put_volume = amount
                shared._put_points = points
                shared._put_physical = physical
            amount = shared._put_volume
            points = list(shared._put_points)
            physical = shared._put_physical
            end = shared.finished_ns
        logical_blocks = getattr(op, "valid_block_num", None)
        if logical_blocks is None:
            logical_blocks = len(getattr(op, "src_block_ids_h2d", ()))
        emit("worker_done", at=end, **getattr(op, "_put_diag", {}),
             received_ns=getattr(op, "_received_ns", 0), physical=physical,
             status=status, points=points + [("worker_end", end)], volume=amount,
             logical_blocks=int(logical_blocks))
    finally:
        _local.points = _local.volume = None


_environment_config = os.getenv("FLEXKV_PUT_TRACE_CONFIG")
if _environment_config:
    configure(json.loads(_environment_config))
elif os.getenv("FLEXKV_PUT_TRACE", "0") == "1":
    configure({"enabled": True, "output_file": os.getenv("FLEXKV_PUT_TRACE_FILE", "put-trace.jsonl")})
