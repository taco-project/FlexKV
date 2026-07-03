"""
Microbenchmark: CE path per-call overhead in transfer_kv_blocks.

The CE path uses a triple-nested loop (layers × kv_dim × blocks), issuing one
cudaMemcpyAsync per chunk. This benchmark quantifies the API-call overhead by:

 1. Sweeping num_blocks at fixed layer count and head_dim.
 2. Comparing CE vs CUDA kernel path at each point.
 3. Reporting effective bandwidth and the number of cudaMemcpyAsync calls.

The hypothesis: as num_blocks grows, CE latency scales worse than kernel due
to O(num_layers × kv_dim × num_blocks) cudaMemcpyAsync calls.

Usage:
    python benchmarks/microbenchmark_ce_overhead.py
    python benchmarks/microbenchmark_ce_overhead.py --num-gpus 1 --iters 20
"""

import argparse
import sys
import time
from collections import defaultdict

import numpy as np

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    print("ERROR: PyTorch not available")
    sys.exit(1)

try:
    from flexkv.c_ext import TPTransferThreadGroup
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
    FLEXKV_AVAILABLE = True
except ImportError as e:
    print("ERROR: FlexKV not available ({})".format(e))
    sys.exit(1)

DTYPE = torch.float16
ES = DTYPE.itemsize
WARMUP_ITERS = 3

# ── Test matrix ──────────────────────────────────────────────────────────────

# Fixed: 32 layers, non-MLA (kv_dim=2), head_dim=128 (Llama-3-8B style)
NUM_LAYERS = 32
HEAD_DIM = 128
KV_DIM = 2  # non-MLA

# Sweep: vary block count to expose per-call overhead
BLOCK_COUNTS = [64, 128, 256, 512, 1024, 2048, 4096]


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_layouts(num_layers, num_blocks, head_dim, num_gpus):
    """GPU (LAYERFIRST) and CPU (LAYERFIRST) layouts for non-MLA."""
    num_head = num_gpus
    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=1, num_head=1,    # 1 head per GPU
        head_size=head_dim, is_mla=False)
    cpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=1, num_head=num_head,
        head_size=head_dim, is_mla=False)
    return gpu_layout, cpu_layout


def make_gpu_tensors(num_layers, num_blocks, head_dim, device):
    """[num_layers, 2, num_blocks, 1, 1, head_dim] per GPU."""
    full = torch.empty(
        (num_layers, KV_DIM, num_blocks, 1, 1, head_dim),
        dtype=DTYPE, device="cuda:{}".format(device))
    return [full[i] for i in range(num_layers)]


def make_cpu_tensor(cpu_layout):
    return torch.empty(tuple(cpu_layout.kv_shape), dtype=DTYPE, pin_memory=True)


def make_tp_group(cpu_ptr, all_gpu, num_gpus, gpu_layout, num_layers):
    gpu_ptrs = []
    for g in range(num_gpus):
        for l in range(num_layers):
            gpu_ptrs.append(all_gpu[g][l].data_ptr())
    # Positional args matching current source:
    # (num_gpus, gpu_block_ptrs_flat, num_tensors_per_gpu, cpu_blocks_ptr,
    #  num_layers, gpu_kv_strides, gpu_block_strides, gpu_layer_strides,
    #  gpu_chunk_sizes, gpu_device_ids, enable_nvcomp=False)
    return TPTransferThreadGroup(
        num_gpus=num_gpus, gpu_block_ptrs_flat=gpu_ptrs,
        num_tensors_per_gpu=num_layers, cpu_blocks_ptr=cpu_ptr,
        num_layers=num_layers,
        gpu_kv_strides_in_bytes=[gpu_layout.get_kv_stride() * ES] * num_gpus,
        gpu_block_strides_in_bytes=[gpu_layout.get_block_stride() * ES] * num_gpus,
        gpu_layer_strides_in_bytes=[gpu_layout.get_layer_stride() * ES] * num_gpus,
        gpu_chunk_sizes_in_bytes=[gpu_layout.get_chunk_size() * ES] * num_gpus,
        gpu_device_ids=list(range(num_gpus)),
        enable_nvcomp=False)


def block_ids(n):
    return torch.arange(n, dtype=torch.int64).pin_memory()


# ── Benchmark core ──────────────────────────────────────────────────────────

def bench_transfer(tp, gpu_ids, cpu_ids, cpu_kv_sb, cpu_ly_sb, cpu_bl_sb, cpu_tp_sb,
                   num_layers, is_host_to_device, use_ce, num_gpus, iters):
    """Time one direction (H2D or D2H), returns (avg_ms, p99_ms)."""

    def do_transfer():
        tp.tp_group_transfer(
            gpu_block_id_tensor=gpu_ids, cpu_block_id_tensor=cpu_ids,
            cpu_kv_stride_in_bytes=cpu_kv_sb, cpu_layer_stride_in_bytes=cpu_ly_sb,
            cpu_block_stride_in_bytes=cpu_bl_sb, cpu_tp_stride_in_bytes=cpu_tp_sb,
            transfer_num_cta=4, is_host_to_device=is_host_to_device,
            use_ce_transfer=use_ce, layer_id=0, layer_granularity=num_layers,
            is_mla=False, mla_d2h_mode="sharded")

    # Warmup
    for _ in range(WARMUP_ITERS):
        do_transfer()

    # Use CUDA events for GPU-side timing
    start_ev = [torch.cuda.Event(enable_timing=True) for _ in range(num_gpus)]
    end_ev = [torch.cuda.Event(enable_timing=True) for _ in range(num_gpus)]

    gpu_times_ms = []
    wall_times_ms = []

    for _ in range(iters):
        t0 = time.perf_counter()
        for g in range(num_gpus):
            start_ev[g].record()
        do_transfer()
        for g in range(num_gpus):
            end_ev[g].record()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000

        # Take max GPU time across all devices
        max_gpu_ms = 0.0
        for g in range(num_gpus):
            gpu_ms = start_ev[g].elapsed_time(end_ev[g])
            if gpu_ms > max_gpu_ms:
                max_gpu_ms = gpu_ms
        gpu_times_ms.append(max_gpu_ms)
        wall_times_ms.append(wall_ms)

    return {
        "gpu_avg_ms": float(np.mean(gpu_times_ms)),
        "gpu_p99_ms": float(np.percentile(gpu_times_ms, 99)),
        "wall_avg_ms": float(np.mean(wall_times_ms)),
        "wall_p99_ms": float(np.percentile(wall_times_ms, 99)),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Microbenchmark CE transfer per-call overhead")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs (default: 1)")
    parser.add_argument("--iters", type=int, default=20,
                        help="Timing iterations per config")
    args = parser.parse_args()

    num_gpus = args.num_gpus
    if num_gpus < 1:
        print("ERROR: need at least 1 GPU")
        sys.exit(1)

    print("=" * 90)
    print("  CE Transfer Overhead Microbenchmark")
    print("=" * 90)
    print("  GPUs:        {}".format(num_gpus))
    print("  Layers:      {}".format(NUM_LAYERS))
    print("  KV dim:      {} (non-MLA)".format(KV_DIM))
    print("  Head dim:    {}".format(HEAD_DIM))
    print("  Dtype:       {}".format(DTYPE))
    print("  Block sweep: {}".format(BLOCK_COUNTS))
    print("  Iters:       {}".format(args.iters))
    print("=" * 90)

    results = []

    for num_blocks in BLOCK_COUNTS:
        # Total data per direction (bytes)
        total_bytes = NUM_LAYERS * KV_DIM * num_blocks * 1 * 1 * HEAD_DIM * ES

        gpu_layout, cpu_layout = make_layouts(
            NUM_LAYERS, num_blocks, HEAD_DIM, num_gpus)
        cpu_kv_sb = cpu_layout.get_kv_stride() * ES
        cpu_ly_sb = cpu_layout.get_layer_stride() * ES
        cpu_bl_sb = cpu_layout.get_block_stride() * ES
        cpu_tp_sb = cpu_bl_sb  # tp_stride = block_stride for single GPU

        all_gpu = [make_gpu_tensors(NUM_LAYERS, num_blocks, HEAD_DIM, g)
                   for g in range(num_gpus)]
        cpu_kv = make_cpu_tensor(cpu_layout)
        tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus,
                           gpu_layout, NUM_LAYERS)

        gpu_ids = block_ids(num_blocks)
        cpu_ids = block_ids(num_blocks)

        n_calls = NUM_LAYERS * KV_DIM * num_blocks  # cudaMemcpyAsync calls per transfer

        print("\n── Blocks={} | Total={:.1f} MB | CE API calls={:,} ──".format(
            num_blocks, total_bytes / (1024**2), n_calls))

        for use_ce, engine_name in [(False, "Kernel"), (True, "CE")]:
            for is_h2d, dir_name in [(True, "H2D"), (False, "D2H")]:
                label = "{} | {}".format(engine_name, dir_name)
                print("  {} ...".format(label), end=" ", flush=True)

                try:
                    r = bench_transfer(
                        tp, gpu_ids, cpu_ids, cpu_kv_sb, cpu_ly_sb, cpu_bl_sb,
                        cpu_tp_sb, NUM_LAYERS, is_h2d, use_ce, num_gpus,
                        args.iters)

                    bw = total_bytes / (r["gpu_avg_ms"] / 1000) / 1e9  # GB/s

                    r.update({
                        "num_blocks": num_blocks,
                        "total_mb": total_bytes / (1024**2),
                        "engine": engine_name,
                        "direction": dir_name,
                        "n_calls": n_calls,
                        "bw_gbps": bw,
                    })
                    results.append(r)
                    print("GPU={:.3f}ms  Wall={:.3f}ms  BW={:.2f} GB/s".format(
                        r["gpu_avg_ms"], r["wall_avg_ms"], bw))

                except Exception as e:
                    print("FAILED: {}".format(e))

        del tp, all_gpu, cpu_kv

    # ── Summary ──────────────────────────────────────────────────────────────

    print("\n" + "=" * 90)
    print("  Summary: CE vs Kernel bandwidth (GB/s)")
    print("=" * 90)

    # Group by direction
    for dir_name in ["H2D", "D2H"]:
        print("\n  Direction: {}".format(dir_name))
        hdr = "{:>8s}  {:>10s}  {:>10s}  {:>12s}  {:>10s}".format(
            "Blocks", "API Calls", "CE GB/s", "Kernel GB/s", "CE/Kernel")
        print("  " + hdr)
        print("  " + "-" * len(hdr))

        for num_blocks in BLOCK_COUNTS:
            ce = [r for r in results
                  if r["num_blocks"] == num_blocks and r["engine"] == "CE"
                  and r["direction"] == dir_name]
            kw = [r for r in results
                  if r["num_blocks"] == num_blocks and r["engine"] == "Kernel"
                  and r["direction"] == dir_name]
            if ce and kw:
                ce_bw = ce[0]["bw_gbps"]
                kw_bw = kw[0]["bw_gbps"]
                ratio = ce_bw / kw_bw if kw_bw > 0 else 0
                n = ce[0]["n_calls"]
                print("  {:>8d}  {:>10,}  {:>10.2f}  {:>10.2f}  {:>9.2f}x".format(
                    num_blocks, n, ce_bw, kw_bw, ratio))

    # ── Overhead analysis ────────────────────────────────────────────────────

    print("\n" + "=" * 90)
    print("  Overhead Analysis: CE per-call cost")
    print("=" * 90)
    print("  (difference between CE and Kernel wall-clock time, divided by")
    print("   number of cudaMemcpyAsync calls)")

    for dir_name in ["H2D", "D2H"]:
        print("\n  Direction: {}".format(dir_name))
        hdr = "{:>8s}  {:>12s}  {:>12s}  {:>14s}".format(
            "Blocks", "CE Wall ms", "Kernel Wall ms", "Overhead/call")
        print("  " + hdr)
        print("  " + "-" * len(hdr))

        for num_blocks in BLOCK_COUNTS:
            ce = [r for r in results
                  if r["num_blocks"] == num_blocks and r["engine"] == "CE"
                  and r["direction"] == dir_name]
            kw = [r for r in results
                  if r["num_blocks"] == num_blocks and r["engine"] == "Kernel"
                  and r["direction"] == dir_name]
            if ce and kw:
                delta_ms = ce[0]["wall_avg_ms"] - kw[0]["wall_avg_ms"]
                n = ce[0]["n_calls"]
                overhead_us = (delta_ms * 1000) / n  # microseconds per call
                print("  {:>8d}  {:>12.3f}  {:>12.3f}  {:>12.3f} µs".format(
                    num_blocks, ce[0]["wall_avg_ms"], kw[0]["wall_avg_ms"],
                    overhead_us))

    print("\n  Note: CE path calls cudaMemcpyAsync once per (layer, kv, block).")
    print("        Higher blocks → more calls → larger wall-clock gap.")
    print("=" * 90)


if __name__ == "__main__":
    main()
