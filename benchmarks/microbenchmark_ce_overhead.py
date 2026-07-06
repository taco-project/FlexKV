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
    from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
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


def make_tp_group(cpu_ptr, all_gpu, num_gpus, gpu_layout, num_layers,
                  ce_segment_threshold=None,
                  ce_use_pingpong=None,
                  ce_path_opt=None):
    gpu_ptrs = []
    for g in range(num_gpus):
        for l in range(num_layers):
            gpu_ptrs.append(all_gpu[g][l].data_ptr())
    if ce_segment_threshold is None:
        ce_segment_threshold = GLOBAL_CONFIG_FROM_ENV.transfer_segment_threshold
    if ce_use_pingpong is None:
        ce_use_pingpong = GLOBAL_CONFIG_FROM_ENV.transfer_pingpong
    if ce_path_opt is None:
        ce_path_opt = GLOBAL_CONFIG_FROM_ENV.transfer_path_opt
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
        ce_segment_threshold=ce_segment_threshold,
        ce_use_pingpong=ce_use_pingpong)


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


# ── Path comparison helpers ──────────────────────────────────────────────────

# Sizes for path comparison:
#   (num_layers, num_blocks, tpb, head_dim, num_heads, is_mla)
# Path selection depends only on block-id contiguity and cpu_block_stride vs
# chunk_size, not on head count — but kv_dim (1 MLA / 2 MHA) affects the
# per-layer loop count, so we benchmark both.
PATH_SIZES = {
    "mla-small":  (8, 16, 16, 512, 1, True),
    "mla-medium": (32, 64, 16, 512, 1, True),
    "mla-large":  (80, 256, 16, 512, 1, True),
    "mha-small":  (8, 16, 16, 128, 8, False),
    "mha-medium": (32, 64, 16, 128, 8, False),
    "mha-large":  (80, 256, 16, 128, 8, False),
}

PATH_PATTERNS = ["contiguous", "few_seg", "scattered"]
PATH_LAYOUTS = ["LAYERFIRST", "BLOCKFIRST"]
MLA_MODES_BENCH = ["sharded", "all_write", "rank0_only"]


def make_block_id_pattern_bench(pattern_name, num_blocks):
    """Block-id permutation yielding a specific segment count.

    contiguous → [0,1,...,N-1]          (1 segment → path 0 LAYERFIRST)
    few_seg    → interleaved 4 segments  (4 segments → path 1)
    scattered  → random permutation      (N segments → path 2)
    """
    if pattern_name == "contiguous":
        ids = torch.arange(num_blocks, dtype=torch.int64)
    elif pattern_name == "few_seg":
        q = num_blocks // 4
        base = torch.arange(num_blocks, dtype=torch.int64)
        ids = torch.cat([base[0:q], base[2 * q:3 * q],
                         base[q:2 * q], base[3 * q:4 * q]])
    elif pattern_name == "scattered":
        gen = torch.Generator().manual_seed(42)
        ids = torch.randperm(num_blocks, generator=gen, dtype=torch.int64)
    else:
        raise ValueError("unknown pattern: " + pattern_name)
    return ids.pin_memory()


def make_gpu_tensors_path(num_layers, num_blocks, tpb, head_dim,
                          kv_dim, heads_per_rank, device):
    """[num_layers, kv_dim, num_blocks, tpb, heads_per_rank, head_dim] per GPU."""
    full = torch.empty(
        (num_layers, kv_dim, num_blocks, tpb, heads_per_rank, head_dim),
        dtype=DTYPE, device="cuda:{}".format(device))
    return [full[i] for i in range(num_layers)]


def make_layouts_path(num_layers, num_blocks, tpb, head_dim, cpu_layout_name,
                      num_gpus, is_mla=True, num_heads=1):
    """GPU (LAYERFIRST) + CPU (given layout).

    Returns (gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank).
    For non-MLA + BLOCKFIRST, cpu_layout_tp = cpu_layout.div_head(num_gpus).
    """
    kv_dim = 1 if is_mla else 2
    heads_per_rank = 1 if is_mla else num_heads // num_gpus

    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=tpb, num_head=heads_per_rank,
        head_size=head_dim, is_mla=is_mla)
    cpu_layout = KVCacheLayout(
        type=KVCacheLayoutType[cpu_layout_name],
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=tpb, num_head=num_heads,
        head_size=head_dim, is_mla=is_mla)

    if not is_mla and cpu_layout.type == KVCacheLayoutType.BLOCKFIRST:
        cpu_layout_tp = cpu_layout.div_head(num_gpus)
    else:
        cpu_layout_tp = cpu_layout
    return gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank


def bench_one_transfer(tp, gpu_ids, cpu_ids, cpu_kv_sb, cpu_ly_sb, cpu_bl_sb,
                       cpu_tp_sb, num_layers, is_h2d, num_gpus, iters,
                       is_mla=True, mode="sharded"):
    """Time a single CE transfer config, return median GPU ms."""
    def do():
        tp.tp_group_transfer(
            gpu_block_id_tensor=gpu_ids, cpu_block_id_tensor=cpu_ids,
            cpu_kv_stride_in_bytes=cpu_kv_sb, cpu_layer_stride_in_bytes=cpu_ly_sb,
            cpu_block_stride_in_bytes=cpu_bl_sb, cpu_tp_stride_in_bytes=cpu_tp_sb,
            transfer_num_cta=4, is_host_to_device=is_h2d, use_ce_transfer=True,
            layer_id=0, layer_granularity=num_layers, is_mla=is_mla,
            mla_d2h_mode=mode)

    for _ in range(WARMUP_ITERS):
        do()

    start_ev = [torch.cuda.Event(enable_timing=True) for _ in range(num_gpus)]
    end_ev = [torch.cuda.Event(enable_timing=True) for _ in range(num_gpus)]
    times = []
    for _ in range(iters):
        for g in range(num_gpus):
            start_ev[g].record()
        do()
        for g in range(num_gpus):
            end_ev[g].record()
        torch.cuda.synchronize()
        max_ms = max(start_ev[g].elapsed_time(end_ev[g]) for g in range(num_gpus))
        times.append(max_ms)
    return float(np.median(times))


# ── Path comparison main ────────────────────────────────────────────────────

def run_paths(args):
    if args.baseline:
        os.environ["FLEXKV_TRANSFER_PATH_OPT"] = "0"
        print("  *** BASELINE: per-block memcpy, no path optimization ***")
    else:
        os.environ["FLEXKV_TRANSFER_PATH_OPT"] = "1"
        print("  *** OPTIMIZED: Path 0/1/2 auto-select ***")
    """CE Path 0/1/2 performance comparison via block-id patterns.

    Tests all MLA modes (sharded, all_write, rank0_only) × layouts × patterns.
    Shows which path is auto-selected and its performance for each combo.
    """
    num_gpus = args.num_gpus
    iters = args.iters
    if num_gpus < 1:
        print("ERROR: need at least 1 GPU")
        sys.exit(1)

    print("=" * 90)
    print("  CE Transfer Path Comparison (Path 0/1/2)")
    print("=" * 90)
    print("  GPUs:   {}".format(num_gpus))
    print("  Engine: CE (cudaMemcpyAsync)")
    print("  Models: MLA (kv_dim=1) + MHA (kv_dim=2)")
    print("  Dtype:  {}".format(DTYPE))
    print("  Iters:  {} (median of {})".format(iters, iters))
    print("  Patterns: {} ".format(PATH_PATTERNS))
    print("  Layouts:  {}".format(PATH_LAYOUTS))
    print("=" * 90)

    results = []

    for mode in MLA_MODES_BENCH:
      for size_name, (nl, nb, tpb, hd, nh, is_mla) in PATH_SIZES.items():
        if not is_mla and mode != "sharded":
            continue
        if not is_mla and nh % num_gpus != 0:
            print("\n[{}] skipped: MHA num_heads={} not divisible by num_gpus={}".format(
                size_name, nh, num_gpus))
            continue
        kv_dim = 1 if is_mla else 2
        total_bytes = nl * kv_dim * nb * tpb * (nh if is_mla else nh // num_gpus) * hd * ES
        model_tag = "MLA" if is_mla else "MHA"
        print("\n[{}] {} mode={} layers={} blocks={} tpb={} hd={} heads={} data={:.2f} MB".format(
            size_name, model_tag, mode, nl, nb, tpb, hd, nh,
            total_bytes / (1024**2)))

        for layout_name in PATH_LAYOUTS:
            gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = \
                make_layouts_path(nl, nb, tpb, hd, layout_name, num_gpus,
                                  is_mla=is_mla, num_heads=nh)

            all_gpu = [make_gpu_tensors_path(nl, nb, tpb, hd, kv_dim,
                                             heads_per_rank, g)
                       for g in range(num_gpus)]
            cpu_kv = make_cpu_tensor(cpu_layout)
            tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus,
                               gpu_layout, nl)

            cpu_kv_sb = cpu_layout_tp.get_kv_stride() * ES
            cpu_ly_sb = cpu_layout_tp.get_layer_stride() * ES
            cpu_bl_sb = cpu_layout.get_block_stride() * ES
            cpu_tp_sb = cpu_bl_sb // num_gpus

            print("  {:<12s}".format(layout_name))

            for pattern in PATH_PATTERNS:
                ids = make_block_id_pattern_bench(pattern, nb)

                try:
                    d2h_ms = bench_one_transfer(
                        tp, ids, ids, cpu_kv_sb, cpu_ly_sb, cpu_bl_sb,
                        cpu_tp_sb, nl, False, num_gpus, iters, is_mla=is_mla)
                    h2d_ms = bench_one_transfer(
                        tp, ids, ids, cpu_kv_sb, cpu_ly_sb, cpu_bl_sb,
                        cpu_tp_sb, nl, True, num_gpus, iters, is_mla=is_mla)
                except Exception as e:
                    d2h_ms = h2d_ms = float("nan")

                results.append({
                    "size": size_name, "pattern": pattern,
                    "layout": layout_name, "model": model_tag, "mode": mode,
                    "ce_d2h_ms": d2h_ms, "ce_h2d_ms": h2d_ms,
                    "total_mb": total_bytes / (1024**2),
                })
                print("    {:<12s}  CE D2H={:.4f}ms  H2D={:.4f}ms".format(
                    pattern, d2h_ms, h2d_ms))

            del tp, all_gpu, cpu_kv

    # ── Summary table ─────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  Summary: CE path times (ms) by pattern x layout x model")
    print("=" * 90)
    hdr = "{:<14s}  {:<5s}  {:<12s}  {:<12s}  {:>10s}  {:>10s}".format(
        "size", "model", "pattern", "layout", "CE D2H", "CE H2D")
    print("  " + hdr)
    print("  " + "-" * (len(hdr) + 2))
    for r in results:
        print("  {:<14s}  {:<5s}  {:<12s}  {:<12s}  {:>10.4f}  {:>10.4f}".format(
            r["size"], r["model"], r["pattern"], r["layout"],
            r["ce_d2h_ms"], r["ce_h2d_ms"]))

    # ── Pattern comparison (fixed layout) ────────────────────────────────
    print("\n" + "=" * 90)
    print("  Pattern comparison: contiguous vs few_seg vs scattered")
    print("=" * 90)
    for size_name in PATH_SIZES:
        for layout_name in PATH_LAYOUTS:
            subset = [r for r in results
                      if r["size"] == size_name and r["layout"] == layout_name]
            if not subset:
                continue
            print("\n  {} / {}:".format(size_name, layout_name))
            for r in subset:
                d2h_bw = r["total_mb"] / r["ce_d2h_ms"] if r["ce_d2h_ms"] > 0 else 0
                print("    {:<5s} {:<12s}  D2H={:.4f}ms ({:.1f} MB/s)  H2D={:.4f}ms".format(
                    r["model"], r["pattern"], r["ce_d2h_ms"],
                    d2h_bw * 1024, r["ce_h2d_ms"]))

    print("\n  Note: contiguous → Path 0 (LAYERFIRST) / Path 1 (BLOCKFIRST)")
    print("        few_seg    → Path 1 (segment memcpy or staging+scatter)")
    print("        scattered  → Path 2 (gather/scatter pipeline)")
    print("=" * 90)

    if getattr(args, "deep", False):
        _run_pingpong_comparison(args)


# ── Ping-pong on/off comparison (subprocess) ────────────────────────────────

def _run_single_pp_bench(pp_val, num_gpus, iters):
    """Run a single BLOCKFIRST + few_seg transfer, print timing.

    Executed in a subprocess so FLEXKV_TRANSFER_PINGPONG (static-cached in
    C++) is read fresh from the environment.
    """
    nl, nb, tpb, hd, nh, is_mla = PATH_SIZES["mla-medium"]  # 32 layers, 64 blocks
    gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = \
        make_layouts_path(nl, nb, tpb, hd, "BLOCKFIRST", num_gpus,
                          is_mla=is_mla, num_heads=nh)

    all_gpu = [make_gpu_tensors_path(nl, nb, tpb, hd, kv_dim, heads_per_rank, g)
               for g in range(num_gpus)]
    cpu_kv = make_cpu_tensor(cpu_layout)
    tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus, gpu_layout, nl)

    cpu_kv_sb = cpu_layout_tp.get_kv_stride() * ES
    cpu_ly_sb = cpu_layout_tp.get_layer_stride() * ES
    cpu_bl_sb = cpu_layout.get_block_stride() * ES
    cpu_tp_sb = cpu_bl_sb // num_gpus

    ids = make_block_id_pattern_bench("few_seg", nb)

    d2h_ms = bench_one_transfer(tp, ids, ids, cpu_kv_sb, cpu_ly_sb,
                                cpu_bl_sb, cpu_tp_sb, nl, False, num_gpus,
                                iters, is_mla=is_mla)
    h2d_ms = bench_one_transfer(tp, ids, ids, cpu_kv_sb, cpu_ly_sb,
                                cpu_bl_sb, cpu_tp_sb, nl, True, num_gpus,
                                iters, is_mla=is_mla)
    del tp
    # Machine-readable output for the parent process
    print("PP_RESULT {} {:.6f} {:.6f}".format(pp_val, d2h_ms, h2d_ms))


def _run_pingpong_comparison(args):
    """Compare ping-pong on vs off via subprocesses."""
    import subprocess
    import os
    import sys

    this_file = os.path.abspath(__file__)
    base_env = os.environ.copy()

    print("\n" + "=" * 90)
    print("  Ping-pong on/off comparison (MLA medium, BLOCKFIRST, few_seg)")
    print("=" * 90)

    results = {}
    for pp in ["0", "1"]:
        env = base_env.copy()
        env["FLEXKV_TRANSFER_PINGPONG"] = pp
        proc = subprocess.run(
            [sys.executable, this_file, "--pp-bench", pp,
             "--num-gpus", str(args.num_gpus),
             "--iters", str(args.iters)],
            env=env, capture_output=True, text=True, timeout=300)
        if proc.returncode != 0:
            print("  ping-pong={} FAILED: {}".format(
                pp, proc.stderr[-500:] if proc.stderr else "unknown"))
            continue
        # Parse "PP_RESULT <val> <d2h> <h2d>"
        for line in proc.stdout.strip().split("\n"):
            if line.startswith("PP_RESULT"):
                parts = line.split()
                results[parts[1]] = (float(parts[2]), float(parts[3]))

    if "0" in results and "1" in results:
        print("  {:<10s}  {:>10s}  {:>10s}".format("ping-pong", "D2H ms", "H2D ms"))
        print("  " + "-" * 34)
        print("  {:<10s}  {:>10.4f}  {:>10.4f}".format(
            "off (0)", results["0"][0], results["0"][1]))
        print("  {:<10s}  {:>10.4f}  {:>10.4f}".format(
            "on (1)", results["1"][0], results["1"][1]))
        d2h_speedup = results["0"][0] / results["1"][0] if results["1"][0] > 0 else 0
        h2d_speedup = results["0"][1] / results["1"][1] if results["1"][1] > 0 else 0
        print("\n  Speedup (on vs off): D2H {:.2f}x  H2D {:.2f}x".format(
            d2h_speedup, h2d_speedup))
    else:
        print("  (insufficient data for comparison)")

    print("=" * 90)


def run_overhead(args):
    """Original CE per-call overhead sweep (num_blocks)."""
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


# ── Threshold sweep ─────────────────────────────────────────────────────────

THRESHOLD_VALUES = [2, 4, 8, 16, 32, 64]
SWEEP_SEGMENT_COUNTS = [1, 2, 4, 8, 16, 32, 64]


def make_pattern_with_segments(num_blocks, num_segments):
    """Construct a block-id permutation with exactly num_segments segments.

    Strategy: split range into num_segments equal groups, then reverse group
    order so adjacent groups in the result are not value-adjacent.

    E.g. num_blocks=64, num_segments=4 → [48..63, 32..47, 16..31, 0..15]
    → 4 segments (each group is internally contiguous, groups are disjoint).
    """
    seg = num_blocks // num_segments
    groups = [torch.arange(i * seg, (i + 1) * seg, dtype=torch.int64)
              for i in range(num_segments)]
    result = torch.cat(list(reversed(groups)))
    return result.pin_memory()


def run_threshold(args):
    """Sweep segment_threshold to find optimal Path 1/2 crossover point.

    For each (segment_count, threshold) combination, constructs a block-id
    pattern with exactly segment_count segments, then runs CE D2H+H2D with
    the given threshold. When threshold >= segment_count → Path 1, else
    Path 2. The goal is to find the threshold that minimizes total time
    across realistic segment distributions.
    """
    num_gpus = args.num_gpus
    iters = args.iters
    if num_gpus < 1:
        print("ERROR: need at least 1 GPU")
        sys.exit(1)

    # Use MLA medium (32 layers, 64 blocks) — enough segments to be meaningful
    nl, nb, tpb, hd, nh, is_mla = PATH_SIZES["mla-medium"]

    print("=" * 90)
    print("  CE Segment Threshold Sweep")
    print("=" * 90)
    print("  GPUs:       {}".format(num_gpus))
    print("  Model:      MLA (layers={}, blocks={}, tpb={}, hd={})".format(
        nl, nb, tpb, hd))
    print("  Thresholds: {}".format(THRESHOLD_VALUES))
    print("  Segments:   {}".format(SWEEP_SEGMENT_COUNTS))
    print("  Iters:      {} (median of {})".format(iters, iters))
    print("=" * 90)

    # results[seg_count][threshold] = (d2h_ms, h2d_ms, path)
    results = {}

    for seg_count in SWEEP_SEGMENT_COUNTS:
        if seg_count > nb:
            continue
        ids = make_pattern_with_segments(nb, seg_count)
        results[seg_count] = {}

        for threshold in THRESHOLD_VALUES:
            # Path prediction: threshold >= seg_count → Path 1, else Path 2
            # (contiguous + LAYERFIRST → Path 0, but seg_count>1 means not contiguous)
            path = 1 if threshold >= seg_count else 2
            if seg_count == 1:
                path = 0  # contiguous → Path 0

            # Use LAYERFIRST (most common production layout)
            gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = \
                make_layouts_path(nl, nb, tpb, hd, "LAYERFIRST", num_gpus,
                                  is_mla=is_mla, num_heads=nh)

            all_gpu = [make_gpu_tensors_path(nl, nb, tpb, hd, kv_dim,
                                             heads_per_rank, g)
                       for g in range(num_gpus)]
            cpu_kv = make_cpu_tensor(cpu_layout)
            tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus,
                               gpu_layout, nl,
                               ce_segment_threshold=threshold)

            cpu_kv_sb = cpu_layout_tp.get_kv_stride() * ES
            cpu_ly_sb = cpu_layout_tp.get_layer_stride() * ES
            cpu_bl_sb = cpu_layout.get_block_stride() * ES
            cpu_tp_sb = cpu_bl_sb // num_gpus

            try:
                d2h_ms = bench_one_transfer(
                    tp, ids, ids, cpu_kv_sb, cpu_ly_sb, cpu_bl_sb,
                    cpu_tp_sb, nl, False, num_gpus, iters, is_mla=is_mla)
                h2d_ms = bench_one_transfer(
                    tp, ids, ids, cpu_kv_sb, cpu_ly_sb, cpu_bl_sb,
                    cpu_tp_sb, nl, True, num_gpus, iters, is_mla=is_mla)
            except Exception as e:
                d2h_ms = h2d_ms = float("nan")

            results[seg_count][threshold] = (d2h_ms, h2d_ms, path)
            del tp, all_gpu, cpu_kv

    # ── D2H matrix ───────────────────────────────────────────────────────
    for dir_label, idx in [("D2H", 0), ("H2D", 1)]:
        print("\n" + "=" * 90)
        print("  {} time (ms) by segments x threshold".format(dir_label))
        print("=" * 90)
        hdr = "{:>10s}".format("seg\\thresh")
        for t in THRESHOLD_VALUES:
            hdr += "  {:>8d}".format(t)
        print("  " + hdr)
        print("  " + "-" * len(hdr))

        for seg_count in SWEEP_SEGMENT_COUNTS:
            if seg_count > nb or seg_count not in results:
                continue
            row = "  {:>10d}".format(seg_count)
            for t in THRESHOLD_VALUES:
                if t in results[seg_count]:
                    val = results[seg_count][t][idx]
                    path = results[seg_count][t][2]
                    marker = "P{}".format(path)
                    row += "  {:>6.3f}{}".format(val, marker if path > 0 else "  ")
                else:
                    row += "  {:>8s}".format("-")
            print(row)

    # ── Path map ─────────────────────────────────────────────────────────
    print("\n  Path map (P0=single memcpy, P1=segment memcpy, P2=gather/scatter):")
    print("  {:>10s}".format("seg\\thresh"), end="")
    for t in THRESHOLD_VALUES:
        print("  {:>8d}".format(t), end="")
    print()
    for seg_count in SWEEP_SEGMENT_COUNTS:
        if seg_count > nb or seg_count not in results:
            continue
        print("  {:>10d}".format(seg_count), end="")
        for t in THRESHOLD_VALUES:
            if t in results[seg_count]:
                print("  {:>7s} ".format("P{}".format(results[seg_count][t][2])), end="")
            else:
                print("  {:>8s}".format("-"), end="")
        print()

    # ── Optimal threshold per segment count ──────────────────────────────
    print("\n" + "=" * 90)
    print("  Optimal threshold per segment count (minimizing D2H+H2D total)")
    print("=" * 90)
    print("  {:>10s}  {:>10s}  {:>12s}  {:>10s}".format(
        "segments", "threshold", "total ms", "path"))
    print("  " + "-" * 46)
    for seg_count in SWEEP_SEGMENT_COUNTS:
        if seg_count > nb or seg_count not in results:
            continue
        best_t = None
        best_total = float("inf")
        best_path = 0
        for t in THRESHOLD_VALUES:
            if t in results[seg_count]:
                d2h, h2d, path = results[seg_count][t]
                total = d2h + h2d
                if total < best_total:
                    best_total = total
                    best_t = t
                    best_path = path
        if best_t is not None:
            print("  {:>10d}  {:>10d}  {:>12.4f}  {:>10s}".format(
                seg_count, best_t, best_total, "Path {}".format(best_path)))

    print("\n  Note: threshold determines Path 1 vs Path 2 crossover.")
    print("        threshold >= num_segments → Path 1 (segment memcpy)")
    print("        threshold <  num_segments → Path 2 (gather/scatter pipeline)")
    print("        The default threshold=8 is optimal when most transfers have")
    print("        ≤8 segments. For highly scattered access patterns, a larger")
    print("        threshold (16-32) may be better if Path 1 staging is cheaper")
    print("        than Path 2 gather/scatter overhead.")
    print("=" * 90)


# ── Main entry point ────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Microbenchmark CE transfer overhead and path selection")
    parser.add_argument("--baseline", action="store_true",
                        help="Run baseline (per-block memcpy, no Path 0/1/2 optimization). Default is optimized.")
    parser.add_argument("--mode", choices=["overhead", "paths", "threshold", "all"],
                        default="overhead",
                        help="Benchmark mode: overhead (block sweep), "
                             "paths (Path 0/1/2 comparison), "
                             "threshold (segment threshold sweep), or all")
    parser.add_argument("--num-gpus", type=int, default=8,
                        help="Number of GPUs (default: 8)")
    parser.add_argument("--iters", type=int, default=20,
                        help="Timing iterations per config")
    parser.add_argument("--deep", action="store_true",
                        help="Also run ping-pong on/off comparison (subprocess)")
    parser.add_argument("--pp-bench", type=str, default=None,
                        help="Internal: run single ping-pong bench for subprocess")
    args = parser.parse_args()

    # Subprocess mode: run a single ping-pong bench config
    if args.pp_bench is not None:
        _run_single_pp_bench(args.pp_bench, args.num_gpus, args.iters)
        return

    if args.mode in ("overhead", "all"):
        run_overhead(args)
    if args.mode in ("paths", "all"):
        run_paths(args)
    if args.mode in ("threshold", "all"):
        run_threshold(args)


if __name__ == "__main__":
    main()
