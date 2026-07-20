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

# Fixed: 32 layers, head_dim=128 (Llama-3-8B style)
NUM_LAYERS = 32
HEAD_DIM = 128

# Available KV configs. Selected via --config on the CLI.
#   non-MLA (kv_dim=2): standard multi-head attention layout.
#   MLA     (kv_dim=1): DeepSeek-style MLA layout.
# Use --config all to sweep both.
KV_CONFIGS_ALL = [
    # (label, kv_dim, is_mla)
    ("non-MLA", 2, False),
    ("MLA",     1, True),
]
KV_CONFIGS_BY_NAME = {label: (label, kv_dim, is_mla)
                      for label, kv_dim, is_mla in KV_CONFIGS_ALL}

# Sweep: vary block count to expose per-call overhead
BLOCK_COUNTS = [64, 128, 256, 512, 1024, 2048, 4096]


# ── Helpers ──────────────────────────────────────────────────────────────────

def make_layouts(num_layers, num_blocks, head_dim, num_gpus, is_mla, num_heads):
    """GPU (LAYERFIRST) and CPU (LAYERFIRST) layouts."""
    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=1, num_head=num_heads,
        head_size=head_dim, is_mla=is_mla)
    cpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=1, num_head=num_heads,
        head_size=head_dim, is_mla=is_mla)
    return gpu_layout, cpu_layout


def make_gpu_tensors(num_layers, num_blocks, head_dim, kv_dim, device, num_heads):
    """[num_layers, kv_dim, num_blocks, num_heads, 1, head_dim] per GPU."""
    full = torch.empty(
        (num_layers, kv_dim, num_blocks, num_heads, 1, head_dim),
        dtype=DTYPE, device="cuda:{}".format(device))
    return [full[i] for i in range(num_layers)]


def make_cpu_tensor(cpu_layout):
    return torch.empty(tuple(cpu_layout.kv_shape), dtype=DTYPE, pin_memory=True)


def make_tp_group(cpu_ptr, all_gpu, num_gpus, gpu_layout, num_layers,
                  is_mla=False, is_blockfirst=False):
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
        enable_nvcomp=False,
        is_mla=is_mla,
        is_blockfirst=is_blockfirst)


def block_ids(n):
    return torch.arange(n, dtype=torch.int64).pin_memory()


# ── Benchmark core ──────────────────────────────────────────────────────────

def bench_transfer(tp, gpu_ids, cpu_ids, cpu_kv_sb, cpu_ly_sb, cpu_bl_sb, cpu_tp_sb,
                   num_layers, is_host_to_device, use_ce, num_gpus, iters,
                   is_mla, transfer_num_cta=4):
    """Time one direction (H2D or D2H), returns (avg_ms, p99_ms)."""

    def do_transfer():
        tp.tp_group_transfer(
            gpu_block_id_tensor=gpu_ids, cpu_block_id_tensor=cpu_ids,
            cpu_kv_stride_in_bytes=cpu_kv_sb, cpu_layer_stride_in_bytes=cpu_ly_sb,
            cpu_block_stride_in_bytes=cpu_bl_sb, cpu_tp_stride_in_bytes=cpu_tp_sb,
            transfer_num_cta=transfer_num_cta, is_host_to_device=is_host_to_device,
            use_ce_transfer=use_ce, layer_id=0, layer_granularity=num_layers,
            is_mla=is_mla, mla_d2h_mode="sharded")

    # Warmup
    for _ in range(WARMUP_ITERS):
        do_transfer()

    # Wall-clock timing via time.perf_counter(). torch.cuda.Event.elapsed_time
    # returns 0 on non-NVIDIA backends, so it cannot be used for portable
    # benchmarking; wall-clock around the synchronous transfer + synchronize
    # captures the full host-observed transfer time.
    wall_times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        do_transfer()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000
        wall_times_ms.append(wall_ms)

    return {
        "avg_ms": float(np.mean(wall_times_ms)),
        "p99_ms": float(np.percentile(wall_times_ms, 99)),
    }


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Microbenchmark CE transfer per-call overhead")
    parser.add_argument("--num-gpus", type=int, default=1,
                        help="Number of GPUs (default: 1)")
    parser.add_argument("--iters", type=int, default=20,
                        help="Timing iterations per config")
    parser.add_argument("--config", choices=["non-MLA", "MLA", "all"],
                        default="non-MLA",
                        help="Which KV config to run: non-MLA (default), MLA, "
                             "or all (runs both configs sequentially)")
    parser.add_argument("--cta", type=int, default=4,
                        help="transfer_num_cta (default: 4)")
    parser.add_argument("--num-heads", type=int, default=1,
                        help="Number of KV heads per GPU (default: 1)")
    args = parser.parse_args()

    num_gpus = args.num_gpus
    if num_gpus < 1:
        print("ERROR: need at least 1 GPU")
        sys.exit(1)

    if args.config == "all":
        kv_configs = KV_CONFIGS_ALL
    else:
        kv_configs = [KV_CONFIGS_BY_NAME[args.config]]

    print("=" * 90)
    print("  CE Transfer Overhead Microbenchmark")
    print("=" * 90)
    print("  GPUs:        {}".format(num_gpus))
    print("  Layers:      {}".format(NUM_LAYERS))
    print("  Configs:     {}".format(
        ", ".join("{}(kv_dim={})".format(l, k) for l, k, _ in kv_configs)))
    print("  Head dim:    {}".format(HEAD_DIM))
    print("  Dtype:       {}".format(DTYPE))
    print("  Block sweep: {}".format(BLOCK_COUNTS))
    print("  Iters:       {}".format(args.iters))
    print("  CTA count:   {}".format(args.cta))
    print("  Num heads:   {}".format(args.num_heads))
    print("=" * 90)

    results = []

    for cfg_label, kv_dim, is_mla in kv_configs:
        print("\n" + "#" * 90)
        print("#  Config: {} (kv_dim={}, is_mla={})".format(
            cfg_label, kv_dim, is_mla))
        print("#" * 90)

        for num_blocks in BLOCK_COUNTS:
            # Total data per direction (bytes)
            total_bytes = (NUM_LAYERS * kv_dim * num_blocks *
                           args.num_heads * 1 * HEAD_DIM * ES)

            gpu_layout, cpu_layout = make_layouts(
                NUM_LAYERS, num_blocks, HEAD_DIM, num_gpus, is_mla,
                args.num_heads)
            cpu_kv_sb = cpu_layout.get_kv_stride() * ES
            cpu_ly_sb = cpu_layout.get_layer_stride() * ES
            cpu_bl_sb = cpu_layout.get_block_stride() * ES
            cpu_tp_sb = cpu_bl_sb  # tp_stride = block_stride for single GPU

            all_gpu = [make_gpu_tensors(NUM_LAYERS, num_blocks, HEAD_DIM,
                                        kv_dim, g, args.num_heads)
                       for g in range(num_gpus)]
            cpu_kv = make_cpu_tensor(cpu_layout)
            tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus,
                               gpu_layout, NUM_LAYERS)

            gpu_ids = block_ids(num_blocks)
            cpu_ids = block_ids(num_blocks)

            n_calls = NUM_LAYERS * kv_dim * num_blocks  # cudaMemcpyAsync calls

            print("\n── [{}] Blocks={} | Total={:.1f} MB | CE API calls={:,} ──".format(
                cfg_label, num_blocks, total_bytes / (1024**2), n_calls))

            for use_ce, engine_name in [(False, "Kernel"), (True, "CE")]:
                for is_h2d, dir_name in [(True, "H2D"), (False, "D2H")]:
                    label = "{} | {}".format(engine_name, dir_name)
                    print("  {} ...".format(label), end=" ", flush=True)

                    try:
                        r = bench_transfer(
                            tp, gpu_ids, cpu_ids, cpu_kv_sb, cpu_ly_sb,
                            cpu_bl_sb, cpu_tp_sb, NUM_LAYERS, is_h2d, use_ce,
                            num_gpus, args.iters, is_mla,
                            transfer_num_cta=args.cta)

                        bw = total_bytes / (r["avg_ms"] / 1000) / 1e9  # GB/s

                        r.update({
                            "config": cfg_label,
                            "kv_dim": kv_dim,
                            "num_blocks": num_blocks,
                            "total_mb": total_bytes / (1024**2),
                            "engine": engine_name,
                            "direction": dir_name,
                            "n_calls": n_calls,
                            "bw_gbps": bw,
                        })
                        results.append(r)
                        print("Avg={:.3f}ms  P99={:.3f}ms  BW={:.2f} GB/s".format(
                            r["avg_ms"], r["p99_ms"], bw))

                    except Exception as e:
                        print("FAILED: {}".format(e))

            del tp, all_gpu, cpu_kv

    # ── Summary ──────────────────────────────────────────────────────────────

    print("\n" + "=" * 90)
    print("  Summary: CE vs Kernel bandwidth (GB/s), per config")
    print("=" * 90)

    for cfg_label, kv_dim, is_mla in kv_configs:
        for dir_name in ["H2D", "D2H"]:
            print("\n  Config: {} | Direction: {}".format(cfg_label, dir_name))
            hdr = "{:>8s}  {:>10s}  {:>10s}  {:>12s}  {:>10s}".format(
                "Blocks", "API Calls", "CE GB/s", "Kernel GB/s", "CE/Kernel")
            print("  " + hdr)
            print("  " + "-" * len(hdr))

            for num_blocks in BLOCK_COUNTS:
                ce = [r for r in results
                      if r["config"] == cfg_label
                      and r["num_blocks"] == num_blocks
                      and r["engine"] == "CE"
                      and r["direction"] == dir_name]
                kw = [r for r in results
                      if r["config"] == cfg_label
                      and r["num_blocks"] == num_blocks
                      and r["engine"] == "Kernel"
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
    print("  Overhead Analysis: CE per-call cost, per config")
    print("=" * 90)
    print("  (difference between CE and Kernel wall-clock time, divided by")
    print("   number of cudaMemcpyAsync calls)")

    for cfg_label, kv_dim, is_mla in kv_configs:
        for dir_name in ["H2D", "D2H"]:
            print("\n  Config: {} | Direction: {}".format(cfg_label, dir_name))
            hdr = "{:>8s}  {:>12s}  {:>12s}  {:>14s}".format(
                "Blocks", "CE Wall ms", "Kernel Wall ms", "Overhead/call")
            print("  " + hdr)
            print("  " + "-" * len(hdr))

            for num_blocks in BLOCK_COUNTS:
                ce = [r for r in results
                      if r["config"] == cfg_label
                      and r["num_blocks"] == num_blocks
                      and r["engine"] == "CE"
                      and r["direction"] == dir_name]
                kw = [r for r in results
                      if r["config"] == cfg_label
                      and r["num_blocks"] == num_blocks
                      and r["engine"] == "Kernel"
                      and r["direction"] == dir_name]
                if ce and kw:
                    delta_ms = ce[0]["avg_ms"] - kw[0]["avg_ms"]
                    n = ce[0]["n_calls"]
                    overhead_us = (delta_ms * 1000) / n  # microseconds per call
                    print("  {:>8d}  {:>12.3f}  {:>12.3f}  {:>12.3f} µs".format(
                        num_blocks, ce[0]["avg_ms"], kw[0]["avg_ms"],
                        overhead_us))

    print("\n  Note: CE path calls cudaMemcpyAsync once per (layer, kv, block).")
    print("        Higher blocks → more calls → larger wall-clock gap.")
    print("=" * 90)


if __name__ == "__main__":
    main()
