#!/usr/bin/env python3
"""Benchmark one mixed-length PUT/GET batch through the real FlexKV request path."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

import benchmark_put_get_e2e as base


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trace-file", required=True)
    parser.add_argument("--request-blocks", type=base.parse_int_list, required=True)
    parser.add_argument("--storage", choices=("host", "mooncake"), default="host")
    parser.add_argument("--mooncake-config")
    parser.add_argument("--arm", choices=("ce", "gdr", "auto", "original"),
                        default="ce")
    parser.add_argument("--on-init-failure", choices=("error", "original"),
                        default="error")
    parser.add_argument("--tp", type=int, default=8)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--kv-heads", type=int, default=1)
    parser.add_argument("--head-size", type=int, default=128)
    parser.add_argument("--kv-dim", type=int, default=2)
    parser.add_argument(
        "--dtype", choices=("float16", "bfloat16", "float32"),
        default="bfloat16")
    parser.add_argument("--tokens-per-block", type=int, default=256)
    parser.add_argument("--cpu-cache-gb", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--qps", type=int, default=4)
    parser.add_argument("--read-depth", type=int, default=16)
    parser.add_argument("--dedicated", type=int, choices=(0, 1), default=1)
    parser.add_argument("--h2d-batch-requests", type=int, default=4)
    parser.add_argument("--d2h-batch-requests", type=int, default=1)
    parser.add_argument("--gdr-diagnostics", type=int, choices=(0, 1), default=0)
    parser.add_argument("--owner-sync-after-put", type=int, choices=(0, 1),
                        default=0)
    parser.add_argument("--round-gap-ms", type=float, default=0.0)
    parser.add_argument("--run-label", default="")
    parser.add_argument("--timeout", type=float, default=1800.0)
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
    if args.rounds <= 0 or args.warmup < 0:
        parser.error("--rounds must be positive and --warmup nonnegative")
    if args.round_gap_ms < 0:
        parser.error("--round-gap-ms must be nonnegative")
    return args


def length_histogram(request_blocks: list[int],
                     tokens_per_block: int) -> dict[str, int]:
    result: dict[str, int] = {}
    for blocks in request_blocks:
        tokens = blocks * tokens_per_block
        key = str(tokens)
        result[key] = result.get(key, 0) + 1
    return result


def run_case(manager, conns, args, iteration: int, warmup: bool,
             total_blocks: int) -> dict:
    prepare_start = time.perf_counter_ns()
    request_blocks = args.request_blocks
    request_tokens = [
        blocks * args.tokens_per_block for blocks in request_blocks]
    total_tokens = sum(request_tokens)
    concurrency = len(request_blocks)
    case = (
        f" [profile={args.run_label} concurrency={concurrency} "
        f"total_blocks={total_blocks} iteration={iteration} warmup={warmup}]"
    )

    values = [
        1 + ((args.seed + iteration * total_blocks + block) % 1024)
        for block in range(total_blocks)
    ]
    source_start_block = 0
    destination_start_block = total_blocks
    base.broadcast(conns, "seed", {
        "start_block": source_start_block,
        "values": values,
    }, case)
    base.broadcast(conns, "zero", {
        "start_block": destination_start_block,
        "count": total_blocks,
    }, case)

    token_batches, put_slot_mappings, get_slot_mappings = [], [], []
    block_offset = 0
    token_offset = 0
    iteration_token_base = args.seed * 10_000_000 + iteration * total_tokens
    for blocks, tokens in zip(request_blocks, request_tokens):
        token_base = iteration_token_base + token_offset
        token_batches.append(np.arange(
            token_base, token_base + tokens, dtype=np.int64))
        source_slot_base = (
            source_start_block + block_offset) * args.tokens_per_block
        destination_slot_base = (
            destination_start_block + block_offset) * args.tokens_per_block
        put_slot_mappings.append(np.arange(
            source_slot_base, source_slot_base + tokens, dtype=np.int64))
        get_slot_mappings.append(np.arange(
            destination_slot_base,
            destination_slot_base + tokens,
            dtype=np.int64))
        block_offset += blocks
        token_offset += tokens

    sample_start = time.perf_counter_ns()
    put_plan_start = time.perf_counter_ns()
    put_pairs = [manager.put_match(tokens) for tokens in token_batches]
    put_ids = [pair[0] for pair in put_pairs]
    put_plan_end = time.perf_counter_ns()
    put_planned = [int(np.asarray(mask).sum()) for _, mask in put_pairs]
    if any(planned != expected
           for planned, expected in zip(put_planned, request_tokens)):
        raise RuntimeError(
            f"PUT planned {put_planned}, expected {request_tokens}{case}")

    put_launch_start = time.perf_counter_ns()
    put_wait_ids = manager.launch(
        put_ids, put_slot_mappings, as_batch=False)
    put_result = manager.wait(
        put_wait_ids, timeout=args.timeout, completely=True)
    put_end = time.perf_counter_ns()
    base.require_success(put_result, put_wait_ids, "PUT")

    post_put_sync_ms = 0.0
    if args.owner_sync_after_put:
        sync_start = time.perf_counter_ns()
        base.broadcast(conns, "sync", None, case)
        post_put_sync_ms = (time.perf_counter_ns() - sync_start) / 1e6

    prefetch_ms = 0.0
    prefetch_ids: list[int] = []
    if args.storage == "mooncake":
        manager._clear_cpu_cache()
        prefetch_start = time.perf_counter_ns()
        prefetch_ids = [
            manager.prefetch_async(tokens) for tokens in token_batches]
        prefetch_result = manager.wait(
            prefetch_ids, timeout=args.timeout, completely=True)
        prefetch_end = time.perf_counter_ns()
        base.require_success(prefetch_result, prefetch_ids, "PREFETCH")
        prefetch_ms = (prefetch_end - prefetch_start) / 1e6

    get_plan_start = time.perf_counter_ns()
    get_pairs = [manager.get_match(tokens) for tokens in token_batches]
    get_ids = [pair[0] for pair in get_pairs]
    get_matched = [int(np.asarray(mask).sum()) for _, mask in get_pairs]
    if get_matched != request_tokens:
        raise RuntimeError(
            f"GET matched {get_matched}, expected {request_tokens}{case}")
    get_plan_end = time.perf_counter_ns()

    get_launch_start = time.perf_counter_ns()
    get_wait_ids = manager.launch(
        get_ids, get_slot_mappings, as_batch=False)
    get_result = manager.wait(
        get_wait_ids, timeout=args.timeout, completely=True)
    get_end = time.perf_counter_ns()
    base.require_success(get_result, get_wait_ids, "GET")
    base.broadcast(conns, "verify", {
        "start_block": destination_start_block,
        "values": values,
    }, case)

    payload_bytes = (
        total_tokens * args.layers * args.kv_dim * args.kv_heads
        * args.head_size * getattr(base.torch, args.dtype).itemsize
    )
    put_e2e_ns = put_end - put_plan_start
    get_e2e_ns = get_end - get_plan_start
    put_transfer_ns = put_end - put_launch_start
    get_transfer_ns = get_end - get_launch_start
    row = {
        "arm": args.arm,
        "storage": args.storage,
        "blocks_per_request": "mixed",
        "request_blocks": request_blocks,
        "request_tokens": request_tokens,
        "length_histogram": length_histogram(
            request_blocks, args.tokens_per_block),
        "total_blocks": total_blocks,
        "total_tokens": total_tokens,
        "concurrency": concurrency,
        "iteration": iteration,
        "warmup": warmup,
        "payload_bytes": payload_bytes,
        "source_block_range": [0, total_blocks],
        "destination_block_range": [total_blocks, 2 * total_blocks],
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
    base.put_trace.emit("benchmark_sample", **row)
    return row


def main() -> int:
    args = parse_args()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    Path(args.trace_file).resolve().unlink(missing_ok=True)
    model, cache = base.make_configs(args)
    total_blocks = sum(args.request_blocks)
    max_gpu_blocks = 2 * total_blocks
    if total_blocks > cache.num_cpu_blocks:
        raise RuntimeError(
            f"mixed batch needs {total_blocks} CPU blocks, but cache has "
            f"{cache.num_cpu_blocks}; increase --cpu-cache-gb")
    if model.tp_size > base.torch.cuda.device_count():
        raise RuntimeError(
            f"TP={model.tp_size}, but only "
            f"{base.torch.cuda.device_count()} GPUs are visible")

    manager = base.KVManager(model, cache)
    manager.start()
    ctx = base.mp.get_context("spawn")
    processes, conns = [], []
    rows: list[dict] = []
    try:
        for rank in range(model.tp_size):
            parent, child = ctx.Pipe()
            process = ctx.Process(
                target=base.tp_process,
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

        for iteration in range(args.warmup + args.rounds):
            row = run_case(
                manager, conns, args, iteration,
                warmup=iteration < args.warmup,
                total_blocks=total_blocks)
            rows.append(row)
            print(json.dumps(row, separators=(",", ":")), flush=True)
            if args.round_gap_ms:
                time.sleep(args.round_gap_ms / 1000.0)
    finally:
        if rows:
            base.write_outputs(output, rows)
        manager.shutdown()
        for conn in conns:
            try:
                conn.send(("stop", None))
            except (BrokenPipeError, EOFError):
                pass
        for conn in conns:
            try:
                if conn.poll(10):
                    conn.recv()
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
