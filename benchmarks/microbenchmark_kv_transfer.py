"""
Benchmark: KV transfer modes — REAL FlexKV API measurement.

Compares 4 transfer strategies:
  - MHA (non-MLA):    each TP rank owns different head partition (cpu_tp_stride)
  - MLA sharded:     each GPU writes 1/N shard, H2D all read full KV
  - MLA all_write:   each GPU writes full KV, H2D each reads own copy
  - MLA rank0_only:  only rank0 writes, H2D all read from rank0's slot

Matrix:
  - Strategy: MHA / sharded / all_write / rank0_only  (4 types)
  - Engine:    CUDA kernel / CE  (probed, unsupported skipped)
  - Direction: D2H / H2D / H2D-layerwise (layer_granularity=1, looped)
  - Layout:    LAYERFIRST / BLOCKFIRST  (CPU side)
  - Size:      small / medium / large  (each gets its own summary)

Uses KVCacheLayout for stride computation (same as production worker.py).

Usage:
    python benchmarks/benchmark_mla_d2h_modes.py --num-gpus 4 --iters 10
    python benchmarks/benchmark_mla_d2h_modes.py --sizes small large
    python benchmarks/benchmark_mla_d2h_modes.py --no-ce --no-layerwise
"""

import argparse
import sys
import time
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# GPU / FlexKV detection
# ---------------------------------------------------------------------------
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    NUM_GPUS = torch.cuda.device_count() if CUDA_AVAILABLE else 0
except ImportError:
    CUDA_AVAILABLE = False
    NUM_GPUS = 0
    print("ERROR: PyTorch not available")
    sys.exit(1)

try:
    from flexkv.c_ext import TPTransferThreadGroup
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
    FLEXKV_AVAILABLE = True
except ImportError as e:
    FLEXKV_AVAILABLE = False
    print("ERROR: FlexKV not available ({})".format(e))
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configurations
# ---------------------------------------------------------------------------

DTYPE = torch.float16
ES = DTYPE.itemsize

# Model-representative configs: (num_layers, num_blocks, head_dim)
#   num_blocks = max_batch * max_seq_len / tokens_per_block (tpb=1 for MLA)
#   head_dim   = MLA latent_dim (the per-head KV size)
#
# MLA models (DeepSeek-V2/V3, Kimi-K2, etc.):
#   DeepSeek-V3:  61 layers, kv_heads=1, latent_dim=512, fp8/bf16
#   Kimi-K2:      61 layers, kv_heads=1, latent_dim=512, bf16
# MHA models (Llama-3, Qwen2, etc.):
#   Llama-3-8B:   32 layers, kv_heads=8, head_dim=128, bf16
#   Llama-3-70B:  80 layers, kv_heads=8, head_dim=128, bf16
#   Qwen2-72B:    80 layers, kv_heads=8, head_dim=128, bf16
#
# For MHA, num_heads is set to num_gpus at runtime (heads_per_rank=1).
SIZES = {
    # Small: ~Llama-3-8B scale
    "small":  (32,   512,  128),
    # Medium: ~DeepSeek-V3 / Kimi-K2 (MLA) or Llama-3-70B (MHA)
    "medium": (61,  2048,  512),
    # Large: 80-layer model with long context
    "large":  (80,  8192,  512),
}

LAYOUTS = {
    "lfirst": KVCacheLayoutType.LAYERFIRST,
    "bfirst": KVCacheLayoutType.BLOCKFIRST,
}

# 4 strategies. MLA modes use is_mla=True; MHA uses is_mla=False.
STRATEGIES = [
    ("MHA",         False, "sharded"),     # non-MLA, mode ignored
    ("MLA-sharded", True,  "sharded"),
    ("MLA-all_write", True, "all_write"),
    ("MLA-rank0_only", True, "rank0_only"),
]

WARMUP_ITERS = 3


# ---------------------------------------------------------------------------
# Engine probe
# ---------------------------------------------------------------------------

_probe_cache = {}

def probe_engine(use_ce):
    key = use_ce
    if key in _probe_cache:
        return _probe_cache[key]
    try:
        layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=1, num_block=1, tokens_per_block=1,
            num_head=1, head_size=16, is_mla=True)
        g = torch.zeros((1, 1, 1, 1, 1, 16), dtype=DTYPE, device="cuda:0")
        c = torch.zeros(tuple(layout.kv_shape), dtype=DTYPE, pin_memory=True)
        ids = torch.arange(1, dtype=torch.int64).pin_memory()
        tp = TPTransferThreadGroup(
            num_gpus=1, gpu_block_ptrs_flat=[g[0].data_ptr()],
            num_tensors_per_gpu=1, cpu_blocks_ptr=c.data_ptr(), num_layers=1,
            gpu_kv_strides_in_bytes=[layout.get_kv_stride() * ES],
            gpu_block_strides_in_bytes=[layout.get_block_stride() * ES],
            gpu_layer_strides_in_bytes=[layout.get_layer_stride() * ES],
            gpu_chunk_sizes_in_bytes=[layout.get_chunk_size() * ES],
            gpu_device_ids=[0], enable_nvcomp=False)
        tp.tp_group_transfer(
            gpu_block_id_tensor=ids, cpu_block_id_tensor=ids,
            cpu_kv_stride_in_bytes=layout.get_kv_stride() * ES,
            cpu_layer_stride_in_bytes=layout.get_layer_stride() * ES,
            cpu_block_stride_in_bytes=layout.get_block_stride() * ES,
            cpu_tp_stride_in_bytes=layout.get_block_stride() * ES,
            transfer_num_cta=4, is_host_to_device=False, use_ce_transfer=use_ce,
            layer_id=0, layer_granularity=1, is_mla=True, mla_d2h_mode="sharded")
        torch.cuda.synchronize()
        del tp
        _probe_cache[key] = True
        return True
    except Exception:
        _probe_cache[key] = False
        return False


# ---------------------------------------------------------------------------
# Helpers (matching production worker.py)
# ---------------------------------------------------------------------------

def make_layouts(num_layers, num_blocks, head_dim, cpu_layout_type, is_mla, num_gpus):
    """Create GPU (LAYERFIRST) and CPU layouts.

    MLA: kv_dim=1, head=1 (all ranks identical).
    MHA: kv_dim=2, head=num_gpus (each rank gets 1 head).
    GPU uses heads_per_rank (per-rank), CPU uses full num_head.
    """
    num_head = 1 if is_mla else num_gpus
    heads_per_rank = 1  # MLA: 1 shared head; MHA: num_gpus/num_gpus = 1
    # GPU layout uses per-rank heads (tensor only has heads_per_rank)
    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=1, num_head=heads_per_rank,
        head_size=head_dim, is_mla=is_mla)
    # CPU layout uses full heads (all ranks' data on one buffer)
    cpu_layout = KVCacheLayout(
        type=cpu_layout_type,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=1, num_head=num_head,
        head_size=head_dim, is_mla=is_mla)
    return gpu_layout, cpu_layout


def cpu_strides_for_strategy(cpu_layout, num_layers, num_blocks, head_dim,
                             is_mla, mode, num_gpus):
    """Return (cpu_kv_sb, cpu_layer_sb, cpu_block_sb, cpu_tp_sb, total_blocks).

    For MLA all_write: CPU holds N copies, total = num_blocks * num_gpus.
    For MHA: div_head(tp_size) on BLOCKFIRST for per-rank CPU strides.
    """
    total = num_blocks * num_gpus if (is_mla and mode == "all_write") else num_blocks
    num_head = 1 if is_mla else num_gpus

    layout_for_kv_stride = KVCacheLayout(
        type=cpu_layout.type,
        num_layer=num_layers, num_block=total,
        tokens_per_block=1, num_head=num_head,
        head_size=head_dim, is_mla=is_mla)

    # For non-MLA BLOCKFIRST, div_head to get per-rank strides
    if not is_mla and cpu_layout.type == KVCacheLayoutType.BLOCKFIRST:
        layout_for_kv_stride = layout_for_kv_stride.div_head(num_gpus)

    kv_sb = layout_for_kv_stride.get_kv_stride() * ES
    layer_sb = layout_for_kv_stride.get_layer_stride() * ES
    block_sb = cpu_layout.get_block_stride() * ES
    tp_sb = block_sb // num_gpus
    return kv_sb, layer_sb, block_sb, tp_sb, total


def make_gpu_tensors(num_layers, kv_dim, num_blocks, heads_per_rank, head_dim, device):
    """Contiguous [num_layers, kv_dim, num_blocks, 1, heads_per_rank, head_dim]."""
    full = torch.empty(
        (num_layers, kv_dim, num_blocks, 1, heads_per_rank, head_dim),
        dtype=DTYPE, device="cuda:{}".format(device))
    return [full[i] for i in range(num_layers)]


def make_cpu_tensor(cpu_layout, num_layers, total_blocks, head_dim, is_mla, num_gpus):
    num_head = 1 if is_mla else num_gpus
    layout = KVCacheLayout(
        type=cpu_layout.type,
        num_layer=num_layers, num_block=total_blocks,
        tokens_per_block=1, num_head=num_head,
        head_size=head_dim, is_mla=is_mla)
    return torch.empty(tuple(layout.kv_shape), dtype=DTYPE, pin_memory=True)


def fill_gpu(all_gpu, gpu_id, num_layers, num_blocks, head_dim):
    for layer in range(num_layers):
        torch.manual_seed(gpu_id * 10000 + layer)
        all_gpu[gpu_id][layer].uniform_()


def block_ids(n):
    return torch.arange(n, dtype=torch.int64).pin_memory()


def make_tp_group(cpu_ptr, all_gpu, num_gpus, gpu_layout, num_layers):
    gpu_ptrs = []
    for g in range(num_gpus):
        for l in range(num_layers):
            gpu_ptrs.append(all_gpu[g][l].data_ptr())
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


def sync_all(num_gpus):
    for g in range(num_gpus):
        torch.cuda.synchronize(g)


# ---------------------------------------------------------------------------
# Benchmark core
# ---------------------------------------------------------------------------

def bench_one(strategy_label, is_mla, mode, use_ce, cpu_layout_type,
              num_gpus, num_layers, num_blocks, head_dim, iters):
    """Run one benchmark configuration: full D2H+H2D round-trip.

    Measures the entire offload+reload path as a single timing:
      D2H (GPU->CPU) + H2D (CPU->GPU) = one complete round-trip.
    This matches the real production scenario where a block is offloaded
    and later reloaded — the total latency is what matters, not D2H alone.

    Returns dict with round-trip timing results.
    """
    kv_dim = 1 if is_mla else 2
    heads_per_rank = 1  # MLA: 1 head shared; MHA: num_gpus heads / num_gpus = 1 per rank

    gpu_layout, cpu_layout = make_layouts(
        num_layers, num_blocks, head_dim, cpu_layout_type, is_mla, num_gpus)
    cpu_kv_sb, cpu_ly_sb, cpu_bl_sb, cpu_tp_sb, total_blocks = \
        cpu_strides_for_strategy(cpu_layout, num_layers, num_blocks, head_dim,
                                  is_mla, mode, num_gpus)

    all_gpu = [make_gpu_tensors(num_layers, kv_dim, num_blocks, heads_per_rank,
                                head_dim, g) for g in range(num_gpus)]
    cpu_kv = make_cpu_tensor(cpu_layout, num_layers, total_blocks, head_dim,
                             is_mla, num_gpus)
    tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus, gpu_layout, num_layers)

    gpu_ids = block_ids(num_blocks)
    cpu_ids = block_ids(num_blocks)

    def do_d2h():
        tp.tp_group_transfer(
            gpu_block_id_tensor=gpu_ids, cpu_block_id_tensor=cpu_ids,
            cpu_kv_stride_in_bytes=cpu_kv_sb, cpu_layer_stride_in_bytes=cpu_ly_sb,
            cpu_block_stride_in_bytes=cpu_bl_sb, cpu_tp_stride_in_bytes=cpu_tp_sb,
            transfer_num_cta=4, is_host_to_device=False, use_ce_transfer=use_ce,
            layer_id=0, layer_granularity=num_layers, is_mla=is_mla, mla_d2h_mode=mode)

    def do_h2d():
        tp.tp_group_transfer(
            gpu_block_id_tensor=gpu_ids, cpu_block_id_tensor=cpu_ids,
            cpu_kv_stride_in_bytes=cpu_kv_sb, cpu_layer_stride_in_bytes=cpu_ly_sb,
            cpu_block_stride_in_bytes=cpu_bl_sb, cpu_tp_stride_in_bytes=cpu_tp_sb,
            transfer_num_cta=4, is_host_to_device=True, use_ce_transfer=use_ce,
            layer_id=0, layer_granularity=num_layers, is_mla=is_mla, mla_d2h_mode=mode)

    # Warmup: full D2H + H2D round-trip
    for _ in range(WARMUP_ITERS):
        for g in range(num_gpus):
            fill_gpu(all_gpu, g, num_layers, num_blocks, head_dim)
        sync_all(num_gpus)
        do_d2h()
        for g in range(num_gpus):
            for l in range(num_layers):
                all_gpu[g][l].zero_()
        sync_all(num_gpus)
        do_h2d()
        sync_all(num_gpus)

    # Timing: full D2H + H2D round-trip per iteration
    times = []
    for _ in range(iters):
        for g in range(num_gpus):
            fill_gpu(all_gpu, g, num_layers, num_blocks, head_dim)
        sync_all(num_gpus)
        t0 = time.perf_counter()
        do_d2h()
        for g in range(num_gpus):
            for l in range(num_layers):
                all_gpu[g][l].zero_()
        sync_all(num_gpus)
        do_h2d()
        sync_all(num_gpus)
        times.append((time.perf_counter() - t0) * 1000)

    del tp
    return {
        "avg_ms": float(np.mean(times)),
        "p99_ms": float(np.percentile(times, 99)),
        "min_ms": float(np.min(times)),
    }


# ---------------------------------------------------------------------------
# Print helpers
# ---------------------------------------------------------------------------

def _col(text, width):
    return str(text).ljust(width)


def print_results_table(results):
    w = [10, 10, 18, 8, 10, 10, 10]
    hdr = (_col("Size", w[0]) + _col("Layout", w[1]) + _col("Strategy", w[2])
           + _col("Engine", w[3])
           + _col("Avg ms", w[4]) + _col("P99 ms", w[5]) + _col("Min ms", w[6]))
    sep = "-" * len(hdr)
    print("")
    print(hdr)
    print(sep)
    for r in results:
        line = (_col(r["size"], w[0]) + _col(r["layout"], w[1])
                + _col(r["strategy"], w[2]) + _col(r["engine"], w[3])
                + _col("{:.3f}".format(r["avg_ms"]), w[4])
                + _col("{:.3f}".format(r["p99_ms"]), w[5])
                + _col("{:.3f}".format(r["min_ms"]), w[6]))
        print(line)
    print(sep)


def print_analysis(results):
    """Per-size analysis: group by engine -> layout -> strategy.
    For each engine, find best layout for MHA and best layout+mode for MLA."""
    print("\n" + "=" * 90)
    print("Benchmark Results (D2H+H2D round-trip)")
    print("=" * 90)

    sizes_present = sorted(set(r["size"] for r in results))

    for size in sizes_present:
        size_results = [r for r in results if r["size"] == size]
        num_layers, num_blocks, head_dim = SIZES[size]
        kv_bytes = num_layers * 1 * num_blocks * 1 * 1 * head_dim * ES

        print("\n  === Size: {} ({}L / {}B / hd={} / {:.1f}MB) ===".format(
            size, num_layers, num_blocks, head_dim, kv_bytes / (1024**2)))

        engines_present = sorted(set(r["engine"] for r in size_results))

        for engine in engines_present:
            eng_results = [r for r in size_results if r["engine"] == engine]

            print("\n  [{}] Engine: {}".format(engines_present.index(engine) + 1, engine))

            # Group by layout, then list all strategies within each layout
            layouts_present = sorted(set(r["layout"] for r in eng_results))
            best_all = min(r["avg_ms"] for r in eng_results) if eng_results else 1

            for layout in layouts_present:
                lay_results = [r for r in eng_results if r["layout"] == layout]

                print("\n      Layout: {}".format(layout))
                print("        {:>18} {:>12} {:>10}".format(
                    "Strategy", "Round-trip ms", "vs best"))
                print("        " + "-" * 42)

                for r in sorted(lay_results, key=lambda x: x["avg_ms"]):
                    ratio = r["avg_ms"] / best_all if best_all > 0 else 0
                    marker = " *" if r["avg_ms"] == best_all else ""
                    print("        {:>18} {:>12.3f} {:>9.2f}x{}".format(
                        r["strategy"], r["avg_ms"], ratio, marker))

        # --- Recommendation ---
        print("\n  [{}] Recommendation for size={}".format(len(engines_present) + 1, size))
        print("      " + "-" * 60)

        for engine in engines_present:
            eng_results = [r for r in size_results if r["engine"] == engine]

            # MHA best layout
            mha = [r for r in eng_results if r["strategy"] == "MHA"]
            if mha:
                best_mha = min(mha, key=lambda r: r["avg_ms"])
                print("      {} MHA:      best layout = {} ({:.3f} ms)".format(
                    engine, best_mha["layout"], best_mha["avg_ms"]))

            # MLA best layout + mode
            mla = [r for r in eng_results if r["strategy"] != "MHA"]
            if mla:
                best_mla = min(mla, key=lambda r: r["avg_ms"])
                print("      {} MLA:      best layout = {}, best mode = {} ({:.3f} ms)".format(
                    engine, best_mla["layout"], best_mla["strategy"], best_mla["avg_ms"]))

    print("\n  Note: Performance depends on hardware (NUMA topology, PCIe/NVLink")
    print("        bandwidth, GPU model). Always benchmark on your target machine.")
    print("=" * 90)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark KV transfer strategies (REAL FlexKV API)")
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Number of GPUs (default: 0 = all available)")
    parser.add_argument("--iters", type=int, default=10,
                        help="Timing iterations per config (default: 10)")
    parser.add_argument("--strategies", type=str, nargs="+",
                        default=[s[0] for s in STRATEGIES],
                        help="Strategies to test")
    parser.add_argument("--sizes", type=str, nargs="+",
                        default=list(SIZES.keys()),
                        choices=list(SIZES.keys()),
                        help="Data sizes to test (default: all)")
    parser.add_argument("--layouts", type=str, nargs="+",
                        default=list(LAYOUTS.keys()),
                        choices=list(LAYOUTS.keys()),
                        help="CPU layouts to test (default: both)")
    parser.add_argument("--no-kernel", action="store_true",
                        help="Skip CUDA Kernel tests")
    parser.add_argument("--no-ce", action="store_true",
                        help="Skip CE tests")
    args = parser.parse_args()

    num_gpus = NUM_GPUS if args.num_gpus <= 0 else min(args.num_gpus, NUM_GPUS)
    if num_gpus < 2:
        print("ERROR: need at least 2 GPUs, found {}".format(NUM_GPUS))
        sys.exit(1)

    # Engine probe
    engines = []
    if not args.no_kernel:
        if probe_engine(use_ce=False):
            engines.append(("CUDA", False))
        else:
            print("WARNING: CUDA kernel probe failed, skipping kernel tests")
    if not args.no_ce:
        if probe_engine(use_ce=True):
            engines.append(("CE", True))
        else:
            print("WARNING: CE probe failed, skipping CE tests")
    if not engines:
        print("ERROR: no transfer engine available")
        sys.exit(1)

    # Filter strategies
    active_strategies = [s for s in STRATEGIES if s[0] in args.strategies]

    print("=" * 90)
    print("  FlexKV KV Transfer Benchmark")
    print("=" * 90)
    print("  GPUs:        {}".format(num_gpus))
    print("  Strategies:  {}".format([s[0] for s in active_strategies]))
    print("  Sizes:       {}".format(args.sizes))
    print("  Layouts:     {}".format(args.layouts))
    print("  Engines:     {}".format([e[0] for e in engines]))
    print("  Iters:       {}".format(args.iters))
    print("  Dtype:       {}".format(DTYPE))
    print("=" * 90)

    results = []
    for size_name in args.sizes:
        num_layers, num_blocks, head_dim = SIZES[size_name]
        kv_bytes = num_layers * 1 * num_blocks * 1 * 1 * head_dim * ES
        print("\n--- Size: {} ({} layers, {} blocks, hd={}, {:.1f} MB) ---".format(
            size_name, num_layers, num_blocks, head_dim, kv_bytes / (1024**2)))

        for engine_name, use_ce in engines:
            for layout_name in args.layouts:
                for strat_label, is_mla, mode in active_strategies:
                    cpu_layout_type = LAYOUTS[layout_name]
                    label = "{} | {} | {} | {}".format(
                        size_name, engine_name, layout_name, strat_label)
                    print("  Running: {} ...".format(label), end=" ", flush=True)
                    try:
                        r = bench_one(
                            strat_label, is_mla, mode, use_ce,
                            cpu_layout_type, num_gpus, num_layers, num_blocks,
                            head_dim, args.iters)
                        r.update({
                            "size": size_name,
                            "layout": layout_name,
                            "strategy": strat_label,
                            "engine": engine_name,
                        })
                        results.append(r)
                        print("avg={:.3f}ms".format(r["avg_ms"]))
                    except Exception as e:
                        print("FAILED: {}".format(e))

    print_results_table(results)
    print_analysis(results)


if __name__ == "__main__":
    main()
