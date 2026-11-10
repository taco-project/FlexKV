"""Slow, opt-in byte-integrity diagnostics. Imported ONLY inside the probe gate.

FLEXKV_CACHE_PROBE=1 is a startup-only switch, identical on every serving rank.
No tensors/streams/leases are owned by this module. Callers keep transfer leases
until verification returns. Hashes prove byte identity, not model accuracy.
"""
import ctypes
import hashlib
import json
import os
import socket
import time
import uuid
import fcntl
import threading
import sqlite3

_instance = uuid.uuid4().hex
_io_failed = False
_pd = threading.local()
_announced = set()


def output_dir():
    return os.path.abspath(os.environ.get("FLEXKV_CACHE_PROBE_DIR", "/tmp/flexkv-cache-probe"))


def announce():
    if os.getpid() not in _announced:
        _announced.add(os.getpid())
        print(f"[CACHE-PROBE] enabled pid={os.getpid()} directory={output_dir()} (slow byte verification)", flush=True)
        emit("probe_enabled", directory=output_dir())


def emit(event, **fields):
    global _io_failed
    record = dict(v=1, event=event, host=socket.gethostname(), pid=os.getpid(),
                  instance=_instance, wall_ns=time.time_ns(), mono_ns=time.monotonic_ns(),
                  **fields)
    try:
        os.makedirs(output_dir(), exist_ok=True)
        path = os.path.join(output_dir(), f"probe-{socket.gethostname()}-{os.getpid()}.jsonl")
        _append_record(path, record)
        if event in ("transfer_check", "l3_check", "coverage_gap"):
            _append_record(os.path.join(output_dir(),
                           f"checks-{socket.gethostname()}-{os.getpid()}.jsonl"), record)
    except Exception as exc:
        if not _io_failed:
            print(f"[CACHE-PROBE] LOGGING FAILED: {exc!r}; coverage is incomplete", flush=True)
            _io_failed = True


def _append_record(path, record):
    data = (json.dumps(record, separators=(",", ":"), default=str) + "\n").encode()
    fd = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        while data:
            written = os.write(fd, data)
            if written <= 0:
                raise OSError("short probe write")
            data = data[written:]
    finally:
        os.close(fd)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def op_context(op):
    return dict(transfer_op=int(op.transfer_op_id), graph=int(op.transfer_graph_id),
                direction=op.transfer_type.name, trace=getattr(op, "_put_diag", {}))


def register_regions(worker, specs, device_ids, layout):
    worker._cache_probe_regions = list(specs)
    worker._cache_probe_devices = list(device_ids)
    groups = getattr(worker, "layer_groups", None)
    worker._cache_probe_layers = [list(g.layer_indices) for g in groups] if groups else None
    emit("region_schema", worker=worker.worker_id, worker_type=type(worker).__name__,
         tokens_per_page=layout.tokens_per_block,
         regions=[dict(vars(s), original_layers=(worker._cache_probe_layers[i]
                       if worker._cache_probe_layers and i < len(worker._cache_probe_layers)
                       else list(range(s.num_layers)))) for i, s in enumerate(specs)],
         groups=[dict(vars(g)) for g in groups] if groups else [],
         devices=list(device_ids), h2d_holders=getattr(worker, "h2d_layer_holders", None))


def spans(spec, gpu_ids, cpu_ids, *, direction, mode, owners=None):
    """Byte geometry matching gtensor_handler.cuh and region_batch.cpp.

    Yields complete logical-page member spans (or rank slices for sharded D2H).
    Sparse group layers and CP owners are supplied by the worker, not inferred
    from layer counts. No unowned CP placeholder is read.
    """
    if len(gpu_ids) != len(cpu_ids):
        raise ValueError("GPU/Host page count mismatch")
    ranks = len(spec.gpu_chunk_sizes)
    nt = spec.num_tensors_per_gpu
    if nt not in (1, spec.num_layers, 2 * spec.num_layers):
        raise ValueError("unsupported GPU pointer layout")
    if mode == "rank_rotate":
        raise ValueError("rank_rotate requires a per-op designated rank")
    for page, (gpu_page, cpu_page) in enumerate(zip(gpu_ids, cpu_ids)):
        if int(gpu_page) < 0 or int(cpu_page) < 0:
            raise ValueError("negative page id")
        for layer in range(spec.num_layers):
            for rank in range(ranks):
                size = int(spec.gpu_chunk_sizes[rank])
                goff = coff = 0
                if owners is not None:
                    if rank != owners[layer]:
                        continue
                elif spec.num_kv_heads > 1 or mode == "all_write":
                    coff = rank * spec.cpu_tp_stride
                elif direction == "D2H" and ranks > 1:
                    if mode == "sharded":
                        if size % ranks:
                            raise ValueError("chunk is not divisible by rank count")
                        size //= ranks
                        goff = coff = rank * size
                    elif mode == "layer_parallel":
                        q, rem = divmod(spec.num_layers, ranks)
                        start = rank * q + min(rank, rem)
                        if not start <= layer < start + q + (rank < rem):
                            continue
                    elif mode in ("rank0_only", "cp_owner_only"):
                        if mode == "cp_owner_only":
                            raise ValueError("CP owner table missing")
                        if rank != 0:
                            continue
                    else:
                        raise ValueError(f"unsupported rank mode {mode}")
                for kv in range(spec.kv_dim):
                    if nt == 1:
                        tensor, offset = 0, layer * spec.gpu_layer_strides[rank] + kv * spec.gpu_kv_strides[rank]
                    elif nt == spec.num_layers:
                        tensor, offset = layer, kv * spec.gpu_kv_strides[rank]
                    else:
                        tensor, offset = kv * spec.num_layers + layer, 0
                    gpu = (spec.gpu_block_ptrs_flat[rank * nt + tensor] + offset
                           + int(gpu_page) * spec.gpu_block_strides[rank] + goff)
                    cpu = (spec.cpu_ptr + int(cpu_page) * spec.cpu_block_stride
                           + layer * spec.cpu_layer_stride + kv * spec.cpu_kv_stride + coff)
                    yield dict(page=page, gpu_page=int(gpu_page), host_page=int(cpu_page),
                               layer=layer, rank=rank, kv=kv, bytes=size,
                               gpu_ptr=gpu, host_ptr=cpu, byte_offset=goff)


class DeviceReader:
    """Use the SAME resolved cudart as Host registration, including XMLIR."""
    def __init__(self):
        from flexkv.transfer.host_buffer import _get_cudart
        self.rt = _get_cudart()
        self.rt.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
        self.rt.cudaMemcpy.restype = ctypes.c_int
        self.rt.cudaSetDevice.argtypes = [ctypes.c_int]
        self.rt.cudaGetDevice.argtypes = [ctypes.POINTER(ctypes.c_int)]
        current = ctypes.c_int()
        self.check(self.rt.cudaGetDevice(ctypes.byref(current)), "cudaGetDevice")
        self.original = current.value

    @staticmethod
    def check(code, operation):
        if code:
            raise RuntimeError(f"{operation}: CUDA error {code}")

    def read(self, ptr, size, device):
        self.check(self.rt.cudaSetDevice(int(device)), "cudaSetDevice")
        buf = ctypes.create_string_buffer(size)
        self.check(self.rt.cudaMemcpy(buf, ctypes.c_void_p(ptr), size, 2), "cudaMemcpy D2H")
        return buf.raw

    def close(self):
        self.check(self.rt.cudaSetDevice(self.original), "restore cudaSetDevice")


def compare_span(span, reader, device):
    host_hash = sha(ctypes.string_at(span["host_ptr"], span["bytes"]))
    gpu_hash = sha(reader.read(span["gpu_ptr"], span["bytes"], device))
    return dict(host_sha256=host_hash, gpu_sha256=gpu_hash,
                status="pass" if host_hash == gpu_hash else "mismatch")


def verify_transfer(worker, op, result):
    context = op_context(op)
    direction = context["direction"]
    if direction not in ("D2H", "H2D", "LAYERWISE"):
        return
    reader = None
    checked = mismatches = 0
    try:
        if not result:
            emit("transfer_check", **context, status="transfer_failed", checked=0)
            return
        specs = getattr(worker, "_cache_probe_regions", None)
        if not specs:
            raise ValueError("no physical region schema; byte verification unavailable")
        if getattr(getattr(worker, "_compressor", None), "needs_gpu_cpu_thread_group", False):
            raise ValueError("compressed GPU/Host layout requires a decompressed comparison")
        if direction == "LAYERWISE":
            cpu_ids, gpu_ids = op.src_block_ids_h2d, op.dst_block_ids_h2d
            if len(op.swa_src_block_ids_h2d):
                emit("coverage_gap", **context, status="unsupported", detail="SWA pool has separate page IDs")
        else:
            src, dst = worker.get_transfer_block_ids(op, pinned=False)
            gpu_ids, cpu_ids = (src.tolist(), dst.tolist()) if direction == "D2H" else (dst.tolist(), src.tolist())
        emit("transfer_pages", **context, gpu_pages=list(map(int, gpu_ids)),
             host_pages=list(map(int, cpu_ids)), page_count=len(gpu_ids))
        reader = DeviceReader()
        for gi, spec in enumerate(specs):
            if spec.name == "swa":
                continue
            owners = None
            holders = getattr(worker, "h2d_layer_holders", None)
            # D2H uses the CP table only when its real transfer selected sharding.
            use_owners = holders is not None and (direction == "LAYERWISE" or
                          getattr(worker, "_layer_shard_ranges_per_region", None) is not None)
            layers = (worker._cache_probe_layers[gi] if worker._cache_probe_layers
                      else list(range(spec.num_layers)))
            if use_owners:
                owners = [holders[i] for i in layers]
            for span in spans(spec, gpu_ids, cpu_ids,
                              direction="D2H" if direction == "D2H" else "H2D",
                              mode=getattr(worker, "kv_shared_across_ranks_mode", "sharded"), owners=owners):
                verdict = compare_span(span, reader, worker._cache_probe_devices[span["rank"]])
                checked += 1
                mismatches += verdict["status"] == "mismatch"
                emit("page_bytes", **context, region=spec.name, original_layer=layers[span["layer"]],
                     **span, **verdict)
        emit("transfer_check", **context, checked=checked, mismatches=mismatches,
             status=("mismatch" if mismatches else "pass" if checked else "empty"))
    except Exception as exc:
        emit("transfer_check", **context, status="error", checked=checked, mismatches=mismatches, error=repr(exc))
    finally:
        if reader is not None:
            try:
                reader.close()
            except Exception as exc:
                emit("coverage_gap", **context, status="error", error=repr(exc))


def l3_snapshot(op, ptrs, sizes, keys, results, stage):
    rows = []
    for page, (ptr, size, key) in enumerate(zip(ptrs, sizes, keys)):
        ok = results is None or (page < len(results) and bool(results[page]))
        row = dict(**op_context(op), stage=stage, page=page, key=str(key), bytes=int(size),
                   io_success=(None if results is None else ok),
                   sha256=sha(ctypes.string_at(int(ptr), int(size))) if ok else None,
                   status="snapshot" if ok else "io_failed")
        emit("l3_page", **row)
        rows.append(dict(row, snapshot_ns=time.time_ns()))
    check_l3_rows(rows)


def check_l3_rows(rows):
    """Compare L3 bytes on the server, sharing PUT baselines across workers.

    Keep every digest for a key: overwriting an ambiguous baseline could hide
    corruption. Baselines are scoped to the trace run and host/log directory;
    absent/ambiguous baselines never pass. No KV bytes are stored in SQLite.
    """
    db = None
    verdicts = []
    try:
        os.makedirs(output_dir(), exist_ok=True)
        db = sqlite3.connect(os.path.join(output_dir(),
                             f"l3-baselines-{socket.gethostname()}.sqlite3"), timeout=30)
        with db:
            db.execute("CREATE TABLE IF NOT EXISTS baselines (run TEXT, key TEXT, size INTEGER, "
                       "digest TEXT, first_ns INTEGER, PRIMARY KEY (run,key,size,digest))")
            for row in rows:
                trace = row.get("trace") or {}
                queue = trace.get("queue", "").split("/")
                run = (trace.get("request", "").split("/")[0]
                       or (queue[1] if len(queue) == 4 else ""))
                identity = (run, row["key"], row["bytes"])
                status, expected = "unverified_missing_trace", []
                if not row["io_success"]:
                    status = "io_failed"
                elif run and row["stage"] == "put_complete":
                    db.execute("INSERT INTO baselines VALUES (?,?,?,?,?) "
                               "ON CONFLICT(run,key,size,digest) DO UPDATE SET "
                               "first_ns=min(first_ns,excluded.first_ns)",
                               (*identity, row["sha256"], row["snapshot_ns"]))
                    continue
                elif run and row["stage"] == "get_complete":
                    expected = [r[0] for r in db.execute(
                        "SELECT digest FROM baselines WHERE run=? AND key=? AND size=? AND first_ns<=?",
                        (*identity, row["snapshot_ns"]))]
                    status = ("unverified_no_prior_put" if not expected else
                              "unverified_multiple_versions" if len(expected) != 1 else
                              "pass" if row["sha256"] == expected[0] else "mismatch")
                verdicts.append(dict(row, status=status, expected_sha256=sorted(expected)))
        # Do not publish PASS until the baseline transaction has committed.
        for verdict in verdicts:
            emit("l3_check", **verdict)
    except Exception as exc:
        emit("coverage_gap", path="l3_server_check", status="error", error=repr(exc))
    finally:
        if db is not None:
            db.close()


def wait_layerwise(connector):
    """Probe-only barrier: keep scheduler leases until worker hashes complete.

    Native eventfds can signal earlier. Without this barrier the scheduler could
    consume/evict a restored page while the worker is still reading its bytes.
    This intentionally disables load/compute overlap ONLY with the probe on.
    """
    ok, error = 1, None
    if connector._sync_ctx.is_sync_leader and connector.kv_manager is not None:
        tids = list(dict.fromkeys(connector._launched_load_tids))
        try:
            replies = connector.kv_manager.wait(tids, timeout=600.0, completely=True) or {}
            for tid in tids:
                response = replies.get(tid)
                status = getattr(response, "status", None)
                if getattr(status, "name", None) != "SUCCESS":
                    raise RuntimeError(f"probe restore wait incomplete: task={tid}, status={status}")
                connector._launched_load_tids = [value for value in connector._launched_load_tids if value != tid]
                context = connector._launched_load_contexts.pop(tid, None)
                log = getattr(connector, "_log_cache_op", None)
                if context is not None and callable(log):
                    log(context, "complete", "success", direction="H2D", transfer_mode="layerwise", probe_barrier=True)
            emit("restore_probe_barrier", tasks=tids, status="complete")
        except Exception as exc:
            ok, error = 0, repr(exc)
            emit("restore_probe_barrier", tasks=tids, status="error", error=error)
    if connector._sync_ctx.needs_sync:
        ok = connector._sync_ctx.all_reduce_min(ok)
    if not ok:
        raise RuntimeError(f"CACHE-PROBE restore verification did not drain: {error or 'leader failure'}")


def batch_event(scheduler, batch):
    emit("batch", batch=int(scheduler.forward_ct), mode=str(batch.forward_mode),
         deployment=str(scheduler.server_args.disaggregation_mode),
         rank=getattr(scheduler, "tp_rank", None), pp_rank=getattr(scheduler, "pp_rank", None),
         requests=[dict(rid=req.rid, bootstrap_room=getattr(req, "bootstrap_room", None),
                        sender=type(getattr(req, "disagg_kv_sender", None)).__name__,
                        prefix_tokens=(len(req.prefix_indices) if getattr(req, "prefix_indices", None) is not None else None),
                        extend_tokens=getattr(req, "extend_input_len", None),
                        cached_tokens=getattr(req, "cached_tokens", None)) for req in batch.reqs])


def pd_room(room):
    _pd.room = room


def pd_send(manager, session, layers, src_ids, dst_ids, *, state=False):
    """Direct Mooncake path only: snapshot before notifying D of completion."""
    reader = None
    try:
        if getattr(manager.kv_args, "kv_schema_version", 0) or getattr(manager.kv_args, "enable_hisparse", False):
            raise ValueError("mixed/compressed address spaces need a schema-specific PD probe")
        if len(src_ids) != len(dst_ids):
            raise ValueError("PD source/destination page counts differ")
        reader = DeviceReader()
        for buffer, (src, dst, size) in enumerate(layers):
            for logical_page, (s, d) in enumerate(zip(src_ids, dst_ids)):
                data = reader.read(int(src) + int(s) * int(size), int(size), manager.kv_args.gpu_id)
                emit("pd_page", stage="send", room=getattr(_pd, "room", None), session=session,
                     rank=manager.kv_args.engine_rank, buffer=buffer, state=state,
                     logical_page=logical_page, src_page=int(s), dst_page=int(d),
                     dst_ptr=int(dst) + int(d) * int(size), bytes=int(size), sha256=sha(data))
    except Exception as exc:
        emit("coverage_gap", path="pd_send", room=getattr(_pd, "room", None), error=repr(exc))
    finally:
        if reader is not None:
            try:
                reader.close()
            except Exception as exc:
                emit("coverage_gap", path="pd_send_restore_device", error=repr(exc))


def pd_receive(receiver):
    reader = None
    try:
        mgr = receiver.kv_mgr
        args = mgr.kv_args
        if getattr(args, "kv_schema_version", 0) or getattr(args, "enable_hisparse", False):
            raise ValueError("mixed/compressed PD address spaces are not byte-verified")
        ids, state_ids, draft_ids = receiver._cache_probe_page_ids
        if draft_ids is not None:
            raise ValueError("separate draft indices require target/draft schema")
        reader = DeviceReader()
        groups = [(False, args.kv_data_ptrs, args.kv_item_lens, ids)]
        if getattr(args, "state_type", "none") == "nsa" and state_ids is not None:
            groups.append((True, args.state_data_ptrs, args.state_item_lens, state_ids))
        elif getattr(args, "state_type", "none") != "none":
            emit("coverage_gap", path="pd_state", room=receiver.bootstrap_room,
                 state_type=args.state_type)
        for state, ptrs, sizes, pages in groups:
            for buffer, (ptr, size) in enumerate(zip(ptrs, sizes)):
                if not ptr or not size:
                    continue
                for logical_page, page in enumerate(pages):
                    address = int(ptr) + int(page) * int(size)
                    data = reader.read(address, int(size), args.gpu_id)
                    emit("pd_page", stage="receive", room=receiver.bootstrap_room,
                         session=mgr.get_session_id(), rank=args.engine_rank,
                         buffer=buffer, state=state, logical_page=logical_page,
                         dst_page=int(page), dst_ptr=address, bytes=int(size), sha256=sha(data))
    except Exception as exc:
        emit("coverage_gap", path="pd_receive", room=receiver.bootstrap_room, error=repr(exc))
    finally:
        if reader is not None:
            try:
                reader.close()
            except Exception as exc:
                emit("coverage_gap", path="pd_receive_restore_device", error=repr(exc))
