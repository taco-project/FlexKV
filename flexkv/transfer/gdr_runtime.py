"""GDR attachment on main's region backend, without a separate worker class."""

import os
import time
from copy import deepcopy

from flexkv.common.debug import flexkv_logger
from flexkv.transfer import put_trace
from flexkv.transfer.gdr_owner import attach_remote_targets, close_leases, wants_gdr


def configure_worker_gdr(worker):
    config = worker._gdr_config
    if not config["enabled"]:
        return
    for direction in ("h2d", "d2h"):
        path = config[direction]["path"]
        if path in ("ce", "kernel"):
            setattr(worker, f"use_ce_transfer_{direction}", path == "ce")
    from flexkv import c_ext
    compiled = bool(getattr(c_ext, "gdr_compiled", lambda: False)())
    flexkv_logger.info(f"GDR runtime requested=True compiled={compiled}")
    if config["h2d"]["batch_requests"] > 1 and config["h2d"]["path"] in ("gdr", "auto"):
        flexkv_logger.info(
            "gdr.h2d.batch_requests is not applied: H2D keeps main's layerwise "
            "batch/counter protocol; only D2H requests are coalesced")
    group = worker.region_batch
    compressed = getattr(getattr(worker, "_compressor", None), "needs_gpu_cpu_thread_group", False)
    missing_group = group is None or not hasattr(group, "configure_gdr")
    if compressed or missing_group or (wants_gdr(config) and not compiled):
        if compressed:
            reason = "compression requires the original GPU/CPU thread group"
        elif missing_group:
            reason = "extension has no GDR region backend"
        else:
            reason = "extension was not built with GDR"
        config = deepcopy(config)
        for direction in ("h2d", "d2h"):
            entry = config[direction]
            if entry["path"] in ("gdr", "auto"):
                if entry["on_init_failure"] == "error":
                    raise RuntimeError(f"GDR {direction} initialization failed: {reason}")
                flexkv_logger.warning(f"GDR {direction} fallback=original reason={reason}")
                entry["path"] = "original"
        worker._gdr_config = config
        if compressed or missing_group:
            return
    group.configure_gdr(config)
    if not wants_gdr(config):
        return
    unique = {}
    for tensor in worker._gdr_tensors:
        key = (tensor.device.index, tensor.data_ptr(), tensor.numel() * tensor.element_size())
        unique.setdefault(key, tensor)
    hosts = {}
    for tensor, _ in getattr(worker, "_host_registered", []):
        hosts.setdefault(tensor.data_ptr(), tensor)
    worker._gdr_leases = attach_remote_targets(
        group, list(unique.values()), worker._device_ids, config, list(hosts.values()))
    worker._gdr_producer_error = None
    worker._gdr_need_producer_fence = True
    # Fresh endpoints: nothing produced through the previous leases counts.
    worker._gdr_fence_ns = 0
    flexkv_logger.info(
        f"GDR worker={worker.worker_id} enabled={config['enabled']} "
        f"h2d={config['h2d']['path']} d2h={config['d2h']['path']} "
        f"owner_leases={len(worker._gdr_leases)}")


def fence_worker_producers(worker, transfer_op):
    maybe_fence_d2h(worker, transfer_op)


def maybe_fence_d2h(worker, transfer_op) -> None:
    """Make the allocation owner finish producing before any RDMA READ starts.

    One fence serves a whole received batch. That is a happens-before argument,
    not a shortcut: the worker drains the pipe completely before launching
    anything, so every op in the batch had already been sent when the fence
    runs, so the kernels writing its KV had already been launched, and the
    owner's device-wide synchronization waits for exactly those launches.

    ``_gdr_fence_ns`` enforces that premise instead of trusting it. An op that
    reached the worker after the fence began -- which is what a save issued
    from the forward path rather than from SGLang's result processing would
    look like -- pays for a fresh fence. The timestamp is taken before ``R`` is
    sent, so an op that raced the round trip counts as uncovered.
    """
    from flexkv.common.transfer import TransferType
    cfg = getattr(worker, "_gdr_config", {})
    batch_fence = os.environ.get("FLEXKV_GDR_BATCH_FENCE", "1").strip().lower() not in (
        "0", "false", "no", "off")
    uncovered = (not batch_fence
                 or getattr(transfer_op, "_received_ns", 0)
                 > getattr(worker, "_gdr_fence_ns", 0))
    if (cfg.get("enabled") and cfg.get("d2h", {}).get("path") in ("auto", "gdr")
            and getattr(transfer_op, "transfer_type", None) == TransferType.D2H
            and (getattr(worker, "_gdr_need_producer_fence", True) or uncovered)
            and not getattr(worker, "_gdr_producer_error", None)):
        from flexkv.transfer.gdr_owner import wait_remote_producers
        started = time.perf_counter_ns()
        put_trace.mark("producer_fence_begin")
        try:
            wait_remote_producers(
                getattr(worker, "_gdr_leases", []),
                timing=cfg.get("diagnostics", {}).get("timing", False),
            )
            worker._gdr_need_producer_fence = False
            worker._gdr_fence_ns = started
        except Exception as exc:
            worker._gdr_producer_error = exc
        finally:
            put_trace.mark("producer_fence_end")
            if cfg.get("diagnostics", {}).get("probe"):
                probe = worker.__dict__.setdefault("_gdr_probe", {})
                probe["producer_rpc_ns"] = probe.get("producer_rpc_ns", 0) + time.perf_counter_ns() - started
                probe["producer_rpc_calls"] = probe.get("producer_rpc_calls", 0) + 1
    if getattr(worker, "_gdr_producer_error", None):
        raise RuntimeError(
            "GDR producer synchronization failed; worker must be restarted"
        ) from worker._gdr_producer_error


def emit_gdr_probe(worker, *, force=False):
    """Called only after native completion; cumulative, at most once/s.

    Reading statistics while a native lane runs would race. Never call this
    from a scheduler/monitor thread, or add a fence just to measure.
    """
    cfg = getattr(worker, "_gdr_config", {})
    if not cfg.get("enabled") or not cfg.get("diagnostics", {}).get("probe"):
        return
    stamp = time.perf_counter_ns()
    previous = getattr(worker, "_gdr_probe_snapshot_ns", 0)
    if not force and previous and stamp - previous < 1_000_000_000:
        return
    group = getattr(worker, "region_batch", None)
    if group is None:
        return
    worker._gdr_probe_snapshot_ns = stamp
    try:
        native = dict(group.gdr_statistics())
        fields = dict(worker_id=worker.worker_id, previous_snapshot_ns=previous,
                      python=dict(getattr(worker, "_gdr_probe", {})), native=native,
                      h2d_completion_supported=any(k.startswith("h2d_completion.") for k in native),
                      timing_scope="cumulative_host_wall_ns_parallel_ranks",
                      collection_ns=time.perf_counter_ns() - stamp)
        if put_trace.ENABLED:
            put_trace.emit("gdr_probe_snapshot", at=stamp, **fields)
        else:
            import json
            flexkv_logger.info("[GDR-PROBE] %s", json.dumps(fields))
    except Exception as exc:
        flexkv_logger.warning("GDR diagnostic snapshot unavailable: %s", exc)


def run_gdr_transfer(worker, op, launch):
    """Run a coalesced batch once, or one op with optional D2H probe accounting."""
    batch = getattr(op, "_gdr_batch", None)
    if batch is not None:
        return batch.run(lambda merged: run_gdr_transfer(worker, merged, launch))
    from flexkv.common.transfer import TransferType
    cfg = getattr(worker, "_gdr_config", {})
    if (not cfg.get("enabled") or not cfg.get("diagnostics", {}).get("probe")
            or op.transfer_type != TransferType.D2H):
        return launch(op)
    probe = worker.__dict__.setdefault("_gdr_probe", {})
    started = time.perf_counter_ns()
    producer_before = probe.get("producer_rpc_ns", 0)
    completed = False
    try:
        result = launch(op)
        completed = True
        return result
    finally:
        probe["physical_calls"] = probe.get("physical_calls", 0) + 1
        probe["call_excluding_producer_ns"] = (
            probe.get("call_excluding_producer_ns", 0) + time.perf_counter_ns() - started
            - (probe.get("producer_rpc_ns", 0) - producer_before))
        if completed:
            emit_gdr_probe(worker)


def close_worker_gdr(worker):
    group = getattr(worker, "region_batch", None)
    config = getattr(worker, "_gdr_config", {})
    if group is not None and config.get("enabled") and hasattr(group, "close_gdr"):
        group.wait_all_streams()
        emit_gdr_probe(worker, force=True)
        group.close_gdr()
    close_leases(getattr(worker, "_gdr_leases", []))
