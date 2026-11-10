#!/usr/bin/env python3
"""Benchmark real FlexKV request planning, scheduling, and PUT/GET transfers."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import statistics
import time
from typing import Iterable

import numpy as np


def _drop_unsupported_xmlir_allocator_options() -> None:
    """Keep inherited allocator settings except XMLIR-unsupported expansion."""
    if importlib.util.find_spec("torch_xmlir") is None:
        return
    for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        value = os.environ.get(name)
        if not value:
            continue
        options = [
            option for option in value.split(",")
            if option.strip().split(":", 1)[0] != "expandable_segments"
        ]
        if options:
            os.environ[name] = ",".join(options)
        else:
            os.environ.pop(name, None)


_drop_unsupported_xmlir_allocator_options()

import torch

from flexkv.common.config import (
    CacheConfig,
    ModelConfig,
    RankInfo,
    UserConfig,
    update_default_config_from_user_config,
)
from flexkv.common.memory_handle import _IS_XMLIR
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.kvmanager import KVManager
from flexkv.kvtask import KVResponseStatus
from flexkv.server.client import KVTPClient
from flexkv.transfer import put_trace


def parse_int_list(value: str) -> list[int]:
    result = [int(item) for item in value.split(",") if item]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def tp_process(conn, rank: int, port: str, model: ModelConfig,
               cache: CacheConfig, num_gpu_blocks: int) -> None:
    device = rank
    torch.cuda.set_device(device)
    layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=model.num_layers,
        num_block=num_gpu_blocks,
        tokens_per_block=cache.tokens_per_block,
        num_head=model.num_kv_heads,
        head_size=model.head_size,
        kv_dim=model.kv_dim,
        num_kv_heads=model.num_kv_heads,
    )
    # Heads are sharded across TP ranks for MHA; num_kv_heads == 1 is the MLA
    # case where every rank holds the same head. Mirrors create_gpu_kv_layout.
    if model.num_kv_heads > 1:
        layout = layout.div_head(model.tp_size)
    pools = [
        torch.empty(tuple(layout.kv_shape[1:]), dtype=model.dtype, device=device)
        for _ in range(model.num_layers)
    ]
    client = KVTPClient(
        port, dp_client_id=0, device_id=device, intra_client_id=rank)
    client.register_to_server(pools, layout)
    conn.send(("ready", rank))
    try:
        while True:
            command, payload = conn.recv()
            if command == "stop":
                conn.send(("stopped", rank))
                # Real SGLang TP ranks are long-lived. This synthetic owner is
                # disposable after every consumer has joined; XMLIR's allocator
                # has a known atexit failure for tensors formerly exported by
                # torch IPC, so skip only that broken teardown on XMLIR.
                if _IS_XMLIR:
                    conn.close()
                    os._exit(0)
                break
            if command == "seed":
                start_block = int(payload["start_block"])
                values = payload["values"]
                for layer, tensor in enumerate(pools):
                    for offset, value in enumerate(values):
                        tensor[:, start_block + offset].fill_(value + layer)
                torch.cuda.synchronize(device)
                conn.send(("seeded", rank))
            elif command == "zero":
                start_block = int(payload["start_block"])
                count = int(payload["count"])
                for tensor in pools:
                    tensor[:, start_block:start_block + count].zero_()
                torch.cuda.synchronize(device)
                conn.send(("zeroed", rank))
            elif command == "sync":
                torch.cuda.synchronize(device)
                conn.send(("synced", rank))
            elif command == "verify":
                start_block = int(payload["start_block"])
                values = payload["values"]
                detail = None
                for layer, tensor in enumerate(pools):
                    for offset, value in enumerate(values):
                        block_id = start_block + offset
                        block = tensor[:, block_id]
                        expected = torch.tensor(
                            value + layer,
                            dtype=block.dtype,
                            device=block.device,
                        )
                        if bool(torch.all(block == expected)):
                            continue
                        detail = {
                            "layer": layer,
                            "block": block_id,
                            "expected": float(expected),
                            "min": float(block.min()),
                            "max": float(block.max()),
                            "zero_fraction": float(
                                (block == 0).to(torch.float32).mean()),
                        }
                        break
                    if detail is not None:
                        break
                conn.send(("verified", rank, detail is None, detail))
            else:
                raise ValueError(f"unknown TP command: {command}")
    finally:
        conn.close()
        for owner in getattr(client, "_gdr_owners", []):
            owner.close()
        client._gdr_owners = []
        for handle in getattr(client, "_exported_handles", []):
            handle.release_exported_vmm_handle()
        client._exported_handles = []
        pools.clear()


def broadcast(conns, command: str, payload, case: str = "") -> list:
    for conn in conns:
        conn.send((command, payload))
    replies = [conn.recv() for conn in conns]
    expected = {
        "seed": "seeded",
        "zero": "zeroed",
        "sync": "synced",
        "verify": "verified",
    }[command]
    if any(reply[0] != expected for reply in replies):
        raise RuntimeError(f"{command} failed{case}: {replies}")
    if command == "verify" and not all(reply[2] for reply in replies):
        details = {reply[1]: reply[3] for reply in replies if not reply[2]}
        raise RuntimeError(f"GET byte verification failed{case}: {details}")
    return replies


def require_success(result, task_ids: Iterable[int], operation: str) -> None:
    failures = {
        task_id: result.get(task_id).status.value if result.get(task_id) else "missing"
        for task_id in task_ids
        if result.get(task_id) is None
        or result[task_id].status != KVResponseStatus.SUCCESS
    }
    if failures:
        raise RuntimeError(f"{operation} failed: {failures}")


def make_configs(args) -> tuple[ModelConfig, CacheConfig]:
    dtype = getattr(torch, args.dtype)
    model = ModelConfig(
        num_layers=args.layers,
        num_kv_heads=args.kv_heads,
        head_size=args.head_size,
        kv_dim=args.kv_dim,
        dtype=dtype,
        tp_size=args.tp,
        pp_size=1,
        dp_size=1,
        cp_size=1,
    )
    gdr = {
        "enabled": args.arm != "original",
        "qp_count": args.qps,
        "use_dedicated_gdr_thread": bool(args.dedicated),
        "h2d": {
            "path": args.arm,
            "on_init_failure": args.on_init_failure,
            "batch_requests": args.h2d_batch_requests,
        },
        "d2h": {
            "path": args.arm,
            "on_init_failure": args.on_init_failure,
            "read_depth": args.read_depth,
            "batch_requests": args.d2h_batch_requests,
        },
        "diagnostics": {
            "summary": bool(args.gdr_diagnostics),
            "timing": bool(args.gdr_diagnostics),
            "probe": bool(args.gdr_diagnostics),
        },
    }
    user = UserConfig(
        cpu_cache_gb=args.cpu_cache_gb,
        ssd_cache_gb=0,
        ssd_cache_dir=["./ssd_cache"],
        use_mooncake_store_backend=args.storage == "mooncake",
        mooncake_store_config_path=args.mooncake_config,
        gdr=gdr,
        put_trace={"enabled": True, "output_file": str(Path(args.trace_file).resolve())},
    )
    cache = CacheConfig(tokens_per_block=args.tokens_per_block)
    update_default_config_from_user_config(RankInfo(model_config=model), cache, user)
    cache.num_remote_blocks = cache.num_cpu_blocks
    return model, cache


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def run_case(manager: KVManager, conns, args, blocks: int,
             concurrency: int, iteration: int, warmup: bool,
             gpu_region_blocks: int) -> dict:
    prepare_start = time.perf_counter_ns()
    # Every round derives its own token range from ``iteration``, so each PUT is a
    # cold write and each GET matches only what this round just stored. That keeps
    # rounds independent without calling KVManager.reset() between them.
    tokens_per_request = blocks * args.tokens_per_block
    total_blocks = blocks * concurrency
    case = (f" [blocks={blocks} concurrency={concurrency} iteration={iteration} "
            f"tokens_per_request={tokens_per_request} warmup={warmup}]")
    values = [
        1 + ((args.seed + iteration * total_blocks + block) % 1024)
        for block in range(total_blocks)
    ]
    # Use disjoint source and destination regions. Reusing the PUT source as
    # the GET destination requires overwriting it immediately after PUT and
    # turns this throughput benchmark into a cross-process CUDA-IPC source
    # recycling stress test. It can corrupt an otherwise in-flight CE D2H on
    # some special machine types and scenarios. Keeping the regions separate measures the actual PUT and GET paths
    # without the verifier mutating their source data.
    source_start_block = 0
    destination_start_block = gpu_region_blocks
    broadcast(conns, "seed", {
        "start_block": source_start_block,
        "values": values,
    }, case)
    broadcast(conns, "zero", {
        "start_block": destination_start_block,
        "count": total_blocks,
    }, case)

    token_batches, put_slot_mappings, get_slot_mappings = [], [], []
    for request_index in range(concurrency):
        token_base = (
            args.seed * 10_000_000
            + iteration * concurrency * tokens_per_request
            + request_index * tokens_per_request
        )
        token_batches.append(np.arange(
            token_base, token_base + tokens_per_request, dtype=np.int64))
        source_slot_base = (
            source_start_block + request_index * blocks
        ) * args.tokens_per_block
        destination_slot_base = (
            destination_start_block + request_index * blocks
        ) * args.tokens_per_block
        put_slot_mappings.append(np.arange(
            source_slot_base, source_slot_base + tokens_per_request,
            dtype=np.int64))
        get_slot_mappings.append(np.arange(
            destination_slot_base, destination_slot_base + tokens_per_request,
            dtype=np.int64))

    sample_start = time.perf_counter_ns()
    put_plan_start = time.perf_counter_ns()
    put_pairs = [manager.put_match(tokens) for tokens in token_batches]
    put_ids = [pair[0] for pair in put_pairs]
    put_plan_end = time.perf_counter_ns()
    put_planned = [int(np.asarray(mask).sum()) for _, mask in put_pairs]
    if any(planned == 0 for planned in put_planned):
        raise RuntimeError(
            f"PUT planned {put_planned} tokens per request but this round's tokens "
            f"should all be new; nothing would be written to the cache{case}")
    put_launch_start = time.perf_counter_ns()
    put_wait_ids = manager.launch(
        put_ids, put_slot_mappings, as_batch=False)
    put_result = manager.wait(put_wait_ids, timeout=args.timeout, completely=True)
    put_end = time.perf_counter_ns()
    require_success(put_result, put_wait_ids, "PUT")
    post_put_sync_ms = 0.0
    if args.owner_sync_after_put:
        sync_start = time.perf_counter_ns()
        broadcast(conns, "sync", None, case)
        post_put_sync_ms = (time.perf_counter_ns() - sync_start) / 1e6

    prefetch_ms = 0.0
    prefetch_ids: list[int] = []
    if args.storage == "mooncake":
        manager._clear_cpu_cache()
        prefetch_start = time.perf_counter_ns()
        prefetch_ids = [manager.prefetch_async(tokens) for tokens in token_batches]
        prefetch_result = manager.wait(
            prefetch_ids, timeout=args.timeout, completely=True)
        prefetch_end = time.perf_counter_ns()
        require_success(prefetch_result, prefetch_ids, "PREFETCH")
        prefetch_ms = (prefetch_end - prefetch_start) / 1e6

    get_plan_start = time.perf_counter_ns()
    get_pairs = [manager.get_match(tokens) for tokens in token_batches]
    get_ids = [pair[0] for pair in get_pairs]
    for _, mask in get_pairs:
        if int(np.asarray(mask).sum()) != tokens_per_request:
            raise RuntimeError(
                f"GET matched {int(np.asarray(mask).sum())}, "
                f"expected {tokens_per_request} tokens")
    get_plan_end = time.perf_counter_ns()
    get_launch_start = time.perf_counter_ns()
    get_wait_ids = manager.launch(
        get_ids, get_slot_mappings, as_batch=False)
    get_result = manager.wait(get_wait_ids, timeout=args.timeout, completely=True)
    get_end = time.perf_counter_ns()
    require_success(get_result, get_wait_ids, "GET")
    broadcast(conns, "verify", {
        "start_block": destination_start_block,
        "values": values,
    }, case)

    payload_bytes = (
        tokens_per_request * concurrency * args.layers * args.kv_dim
        * args.kv_heads * args.head_size * getattr(torch, args.dtype).itemsize
    )
    put_e2e_ns = put_end - put_plan_start
    get_e2e_ns = get_end - get_plan_start
    put_transfer_ns = put_end - put_launch_start
    get_transfer_ns = get_end - get_launch_start
    row = {
        "arm": args.arm,
        "storage": args.storage,
        "blocks_per_request": blocks,
        "tokens_per_request": tokens_per_request,
        "concurrency": concurrency,
        "iteration": iteration,
        "warmup": warmup,
        "payload_bytes": payload_bytes,
        "source_block_range": [
            source_start_block, source_start_block + total_blocks],
        "destination_block_range": [
            destination_start_block,
            destination_start_block + total_blocks,
        ],
        "put_planned_tokens": put_planned,
        "data_prepare_ms": (sample_start - prepare_start) / 1e6,
        "put_plan_ms": (put_plan_end - put_plan_start) / 1e6,
        "put_transfer_ms": put_transfer_ns / 1e6,
        "put_transfer_GB_s": payload_bytes / max(put_transfer_ns, 1),
        "put_e2e_ms": put_e2e_ns / 1e6,
        "put_effective_GB_s": payload_bytes / max(put_e2e_ns, 1),
        "post_put_owner_sync_ms": post_put_sync_ms,
        "prefetch_e2e_ms": prefetch_ms,
        "get_plan_ms": (get_plan_end - get_plan_start) / 1e6,
        "get_transfer_ms": get_transfer_ns / 1e6,
        "get_transfer_GB_s": payload_bytes / max(get_transfer_ns, 1),
        "get_e2e_ms": get_e2e_ns / 1e6,
        "get_effective_GB_s": payload_bytes / max(get_e2e_ns, 1),
        "roundtrip_ms": (get_end - sample_start) / 1e6,
        "active_roundtrip_ms": (
            put_e2e_ns / 1e6 + prefetch_ms + get_e2e_ns / 1e6),
        "run_label": args.run_label,
        "qps": args.qps,
        "read_depth": args.read_depth,
        "dedicated": args.dedicated,
        "h2d_batch_requests": args.h2d_batch_requests,
        "d2h_batch_requests": args.d2h_batch_requests,
        "gdr_diagnostics": args.gdr_diagnostics,
        "owner_sync_after_put": args.owner_sync_after_put,
        "round_gap_ms": args.round_gap_ms,
        "kv_shared_mode": os.environ.get(
            "FLEXKV_KV_SHARED_ACROSS_RANKS_MODE", "sharded"),
        "put_task_ids": put_ids,
        "prefetch_task_ids": prefetch_ids,
        "get_task_ids": get_ids,
        "verified": True,
    }
    put_trace.emit("benchmark_sample", **row)
    return row


def write_outputs(output: Path, rows: list[dict]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "raw.jsonl").open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
    measured = [row for row in rows if not row["warmup"]]
    fields = [key for key in rows[0] if not key.endswith("_task_ids")]
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(measured)
    if not measured:
        return
    summary = {}
    for key in (
            "put_e2e_ms", "prefetch_e2e_ms", "get_e2e_ms", "roundtrip_ms",
            "put_transfer_GB_s", "get_transfer_GB_s"):
        values = [float(row[key]) for row in measured]
        summary[key] = {
            "median": statistics.median(values),
            "p95": percentile(values, 0.95),
        }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace-file", required=True)
    parser.add_argument("--storage", choices=("host", "mooncake"), default="host")
    parser.add_argument("--mooncake-config")
    parser.add_argument("--arm", choices=("ce", "gdr", "auto", "original"), default="ce")
    parser.add_argument("--on-init-failure", choices=("error", "original"), default="error")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--kv-dim", type=int, default=2)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--tokens-per-block", type=int, default=256)
    parser.add_argument("--cpu-cache-gb", type=int, default=16)
    parser.add_argument("--blocks", type=parse_int_list, default=parse_int_list("4,16,64"))
    parser.add_argument("--concurrency", type=parse_int_list, default=parse_int_list("1,4"))
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--qps", type=int, default=4)
    parser.add_argument("--read-depth", type=int, default=16)
    parser.add_argument("--dedicated", type=int, choices=(0, 1), default=1)
    parser.add_argument("--h2d-batch-requests", type=int, default=4)
    parser.add_argument("--d2h-batch-requests", type=int, default=1)
    parser.add_argument("--gdr-diagnostics", type=int, choices=(0, 1), default=0)
    parser.add_argument("--owner-sync-after-put", type=int, choices=(0, 1), default=0)
    parser.add_argument("--round-gap-ms", type=float, default=0.0)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=20260921)
    args = parser.parse_args()
    if args.storage == "mooncake" and not args.mooncake_config:
        parser.error("--mooncake-config is required for --storage mooncake")
    if not 1 <= args.qps <= 16:
        parser.error("--qps must be in 1..16")
    if (args.read_depth < 0 or args.read_depth > 128
            or (args.read_depth and args.read_depth & (args.read_depth - 1))):
        parser.error("--read-depth must be 0 or a power of two up to 128")
    if not 1 <= args.h2d_batch_requests <= 64:
        parser.error("--h2d-batch-requests must be in 1..64")
    if not 1 <= args.d2h_batch_requests <= 64:
        parser.error("--d2h-batch-requests must be in 1..64")
    if args.round_gap_ms < 0:
        parser.error("--round-gap-ms must be nonnegative")
    return args


def main() -> int:
    args = parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    Path(args.trace_file).resolve().unlink(missing_ok=True)
    model, cache = make_configs(args)
    gpu_region_blocks = max(args.blocks) * max(args.concurrency)
    max_gpu_blocks = 2 * gpu_region_blocks
    if model.tp_size > torch.cuda.device_count():
        raise RuntimeError(
            f"TP={model.tp_size}, but only {torch.cuda.device_count()} GPUs are visible")

    manager = KVManager(model, cache)
    manager.start()
    ctx = mp.get_context("spawn")
    processes, conns = [], []
    rows: list[dict] = []
    try:
        for rank in range(model.tp_size):
            parent, child = ctx.Pipe()
            process = ctx.Process(
                target=tp_process,
                args=(child, rank, manager.gpu_register_port, model, cache,
                      max_gpu_blocks),
                daemon=True,
            )
            process.start()
            child.close()
            processes.append(process)
            conns.append(parent)
        ready = [conn.recv() for conn in conns]
        if any(item[0] != "ready" for item in ready):
            raise RuntimeError(f"GPU registration failed: {ready}")
        deadline = time.monotonic() + args.timeout
        while not manager.is_ready():
            if time.monotonic() >= deadline:
                raise TimeoutError("FlexKV did not become ready")
            time.sleep(0.2)

        iteration = 0
        for blocks in args.blocks:
            for concurrency in args.concurrency:
                for local_round in range(args.warmup + args.rounds):
                    rows.append(run_case(
                        manager, conns, args, blocks, concurrency, iteration,
                        warmup=local_round < args.warmup,
                        gpu_region_blocks=gpu_region_blocks))
                    print(json.dumps(rows[-1], separators=(",", ":")), flush=True)
                    iteration += 1
                    if args.round_gap_ms:
                        time.sleep(args.round_gap_ms / 1000.0)
    finally:
        if rows:
            write_outputs(output, rows)
        # Real serving has one long-lived GPU owner per TP rank. The synthetic
        # owners follow that topology through the timed run; only teardown is
        # benchmark-specific. Consumers must join before owners acknowledge
        # shutdown and bypass XMLIR's broken exported-tensor atexit.
        manager.shutdown()
        for conn in conns:
            try:
                conn.send(("stop", None))
            except (BrokenPipeError, EOFError):
                pass
        for conn in conns:
            try:
                if conn.poll(10):
                    reply = conn.recv()
                    if reply[0] != "stopped":
                        print(
                            f"warning: unexpected TP shutdown reply: {reply}",
                            flush=True,
                        )
            except (BrokenPipeError, EOFError):
                pass
        for process in processes:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
    print(f"done: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
