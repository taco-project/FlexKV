"""
Microbenchmark: CE transfer Monte Carlo simulation under random access patterns.

Instead of fixed block-id patterns (contiguous/few_seg/scattered), this runs
many rounds with RANDOM block-id permutations. Each round generates a random
permutation of range(num_blocks) — the segment count naturally varies from 1
(fully contiguous) to N (fully scattered). This simulates real inference
workloads where KV cache block IDs have unpredictable fragmentation.

For each round, 3 configs are timed SEPARATELY for D2H and H2D:
  - kernel   : CUDA kernel transfer (use_ce=False)
  - baseline : CE with path_opt off (PER_BLOCK)
  - opt      : CE with path_opt on (choose_path auto-select)

D2H and H2D are timed independently with CUDA events, and the round-trip
(D2H + H2D) total is also reported. The output groups results by (size,
layout, mode) combination and shows median times across all random rounds.

Usage:
    python benchmarks/microbenchmark_ce_simulation.py --num-gpus 8 --rounds 50
    python benchmarks/microbenchmark_ce_simulation.py --sizes small medium --layouts bfirst
    python benchmarks/microbenchmark_ce_simulation.py --skip-kernel   # non-NVIDIA: omit CUDA kernel config
"""

import argparse
import random
import sys
import time
from collections import defaultdict

import numpy as np

try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    NUM_GPUS = torch.cuda.device_count() if CUDA_AVAILABLE else 0
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
ITERS_PER_ROUND = 5  # median of 5 timed iterations per (round, config, dir)


# Fixed parameters (sweep matrix)

SIZES = {
    "small":  (32,   512,  128),
    "medium": (61,  2048,  512),
    "large":  (80,  8192,  512),
}

# Each round transfers a RANDOM SUBSET of blocks (simulates variable batch
# size in real evict/load). The subset size is drawn from a distribution
# that favors small-to-medium batches (typical LRU evict / prefix load).
# Format: (weight, min_frac, max_frac) as fraction of max_num_blocks.
BATCH_FRACTIONS = [
    (0.30, 0.05, 0.15),   # 30%: tiny batch (5-15% of pool, e.g. 26-77 of 512)
    (0.40, 0.15, 0.40),   # 40%: small batch (15-40%, e.g. 77-205 of 512)
    (0.20, 0.40, 0.70),   # 20%: medium batch (40-70%, e.g. 205-358 of 512)
    (0.10, 0.70, 1.00),   # 10%: large batch (70-100%, e.g. 358-512 of 512)
]

LAYOUTS = {
    "lfirst": KVCacheLayoutType.LAYERFIRST,
    "bfirst": KVCacheLayoutType.BLOCKFIRST,
}

# (label, is_mla, mode)
# MHA: non-MLA, mode ignored.
# MLA: all 4 D2H modes (sharded / rank0_only / layer_parallel / rank_rotate).
# CPU tensor total blocks = num_blocks * num_gpus ONLY for all_write (handled in
# cpu_strides_for_strategy); the 4 modes below all use num_blocks, correct as-is.
STRATEGIES = [
    ("MHA",               False, "sharded"),
    ("MLA-sharded",        True, "sharded"),
    ("MLA-rank0_only",     True, "rank0_only"),
    ("MLA-layer_parallel", True, "layer_parallel"),
    ("MLA-rank_rotate",    True, "rank_rotate"),
]

CE_CONFIGS = [
    ("kernel",   False, False),  # use_ce, path_opt
    ("baseline", True,  False),
    ("opt",      True,  True),
]


# Helpers (mirrors microbenchmark_ce_strategy.py)

def make_layouts(num_layers, num_blocks, head_dim, cpu_layout_type, is_mla, num_gpus):
    num_head = 1 if is_mla else num_gpus
    heads_per_rank = 1
    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=1, num_head=heads_per_rank,
        head_size=head_dim, is_mla=is_mla)
    cpu_layout = KVCacheLayout(
        type=cpu_layout_type,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=1, num_head=num_head,
        head_size=head_dim, is_mla=is_mla)
    return gpu_layout, cpu_layout


def cpu_strides_for_strategy(cpu_layout, num_layers, num_blocks, head_dim,
                             is_mla, mode, num_gpus):
    total = num_blocks * num_gpus if (is_mla and mode == "all_write") else num_blocks
    num_head = 1 if is_mla else num_gpus
    layout_for_kv_stride = KVCacheLayout(
        type=cpu_layout.type,
        num_layer=num_layers, num_block=total,
        tokens_per_block=1, num_head=num_head,
        head_size=head_dim, is_mla=is_mla)
    if not is_mla and cpu_layout.type == KVCacheLayoutType.BLOCKFIRST:
        layout_for_kv_stride = layout_for_kv_stride.div_head(num_gpus)
    kv_sb = layout_for_kv_stride.get_kv_stride() * ES
    layer_sb = layout_for_kv_stride.get_layer_stride() * ES
    block_sb = cpu_layout.get_block_stride() * ES
    tp_sb = block_sb // num_gpus
    return kv_sb, layer_sb, block_sb, tp_sb, total


def make_gpu_tensors(num_layers, kv_dim, num_blocks, heads_per_rank, head_dim, device):
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


def make_tp_group(cpu_ptr, all_gpu, num_gpus, gpu_layout, num_layers,
                  ce_path_opt=True, ce_segment_threshold=8,
                  is_mla=False, is_blockfirst=False,
                  ce_enable_memcpy2d=None):
    # Default to the global CE memcpy2d setting (FLEXKV_ENABLE_CE_MEMCPY2D,
    # default ON since the env rename). Pass explicitly to override, e.g.
    # ce_enable_memcpy2d=False to match the pre-rename default-off behavior.
    if ce_enable_memcpy2d is None:
        ce_enable_memcpy2d = GLOBAL_CONFIG_FROM_ENV.enable_ce_memcpy2d
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
        enable_nvcomp=False,
        ce_segment_threshold=ce_segment_threshold,
        ce_path_opt=ce_path_opt,
        is_mla=is_mla,
        is_blockfirst=is_blockfirst,
        ce_enable_memcpy2d=ce_enable_memcpy2d)


def fill_gpu(all_gpu, gpu_id, num_layers, num_blocks, head_dim):
    for layer in range(num_layers):
        torch.manual_seed(gpu_id * 10000 + layer)
        all_gpu[gpu_id][layer].uniform_()


def sync_all(num_gpus):
    for g in range(num_gpus):
        torch.cuda.synchronize(g)


def count_segments(ids):
    """Count contiguous ascending runs in an int64 tensor."""
    if ids.numel() <= 1:
        return 1
    diffs = ids[1:] - ids[:-1]
    breaks = (diffs != 1).sum().item()
    return int(breaks) + 1


# Random block-id generation

def random_batch_frac(rng):
    """Random batch fraction (0-1) — models variable evict/load batch size.

    Small batches dominate (typical LRU evict / prefix cache load).
    """
    r = rng.py_rng.random()
    cum = 0.0
    for weight, min_frac, max_frac in BATCH_FRACTIONS:
        cum += weight
        if r < cum:
            return rng.py_rng.uniform(min_frac, max_frac)
    return 1.0


def random_target_seg(rng):
    """Random target segment count (weighted toward low fragmentation).

    40%: seg 1-4, 30%: seg 5-16, 20%: seg 17-64, 10%: seg 65+
    """
    r = rng.py_rng.random()
    if r < 0.40:
        return rng.py_rng.randint(1, 4)
    elif r < 0.70:
        return rng.py_rng.randint(5, 16)
    elif r < 0.90:
        return rng.py_rng.randint(17, 64)
    else:
        return rng.py_rng.randint(65, 256)


def generate_block_ids(batch_size, pool_size, target_seg, rng):
    """Generate block IDs with exactly target_seg contiguous runs.

    Picks `batch_size` blocks from [0, pool_size), organized into
    `target_seg` contiguous runs scattered across the pool.

    Returns (ids_tensor, actual_segment_count).
    """
    batch_size = min(batch_size, pool_size)
    target_seg = max(1, min(batch_size, target_seg))

    if target_seg <= 1:
        start = rng.py_rng.randint(0, pool_size - batch_size)
        ids = list(range(start, start + batch_size))
    else:
        base = batch_size // target_seg
        rem = batch_size % target_seg
        chunk_sizes = [base + (1 if i < rem else 0) for i in range(target_seg)]

        chunks = []
        placed = []
        attempts = 0
        while len(chunks) < target_seg and attempts < 100:
            sz = chunk_sizes[len(chunks)]
            start = rng.py_rng.randint(0, pool_size - sz)
            ok = True
            for (s, e) in placed:
                if start < e and start + sz > s:
                    ok = False
                    break
            if ok:
                chunks.append(list(range(start, start + sz)))
                placed.append((start, start + sz))
            attempts += 1
        idx = 0
        while len(chunks) < target_seg:
            sz = chunk_sizes[len(chunks)]
            chunks.append(list(range(idx, idx + sz)))
            idx += sz
        rng.py_rng.shuffle(chunks)
        ids = []
        for chunk in chunks:
            ids.extend(chunk)

    ids = torch.tensor(ids, dtype=torch.int64).pin_memory()
    actual_seg = count_segments(ids)
    return ids, actual_seg


class TorchRNG:
    """Wrap torch.Generator so random.shuffle and torch.randperm share state."""
    def __init__(self, seed):
        self.torch_gen = torch.Generator()
        self.torch_gen.manual_seed(seed)
        self.py_rng = random.Random(seed)

    def shuffle(self, lst):
        self.py_rng.shuffle(lst)


# Benchmark core: time one direction with wall-clock timer

def bench_one_dir(tp, ids, cpu_kv_sb, cpu_ly_sb, cpu_bl_sb, cpu_tp_sb,
                  num_layers, is_h2d, num_gpus, iters, is_mla, mode,
                  use_ce, transfer_num_cta=16):
    """Time ONE direction over `iters`, return median wall-clock ms."""

    def do_transfer():
        tp.tp_group_transfer(
            gpu_block_id_tensor=ids, cpu_block_id_tensor=ids,
            cpu_kv_stride_in_bytes=cpu_kv_sb, cpu_layer_stride_in_bytes=cpu_ly_sb,
            cpu_block_stride_in_bytes=cpu_bl_sb, cpu_tp_stride_in_bytes=cpu_tp_sb,
            transfer_num_cta=transfer_num_cta, is_host_to_device=is_h2d,
            use_ce_transfer=use_ce, layer_id=0, layer_granularity=num_layers,
            is_mla=is_mla, mla_d2h_mode=mode)

    for _ in range(WARMUP_ITERS):
        do_transfer()
    sync_all(num_gpus)

    # Wall-clock timing via time.perf_counter(). torch.cuda.Event.elapsed_time
    # returns 0 on non-NVIDIA backends, so it cannot be used for portable
    # benchmarking; wall-clock captures the full host-observed transfer time
    # (critical path across all GPUs).
    times_ms = []
    for _ in range(iters):
        t0 = time.perf_counter()
        do_transfer()
        torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000
        times_ms.append(wall_ms)

    return float(np.median(times_ms))


# Main simulation

def run_simulation(args):
    num_gpus = args.num_gpus
    threshold = 8
    cta = 16

    # Effective config list: drop the 'kernel' (CUDA kernel) config when the
    # --skip-kernel flag is set (non-NVIDIA platforms where the custom kernel
    # cannot build/run). baseline/opt (CE paths) are always kept.
    configs = [c for c in CE_CONFIGS if not (args.skip_kernel and c[0] == "kernel")]
    if args.skip_kernel:
        print("[note] --skip-kernel set: omitting 'kernel' (CUDA kernel) config")

    print("=" * 96)
    print("  CE Transfer Monte Carlo Simulation (controlled random fragmentation)")
    print("=" * 96)
    print("  GPUs:        {}".format(num_gpus))
    print("  Sizes:       {}".format(args.sizes))
    print("  Layouts:     {}".format(args.layouts))
    print("  Strategies:  {}".format([s[0] for s in STRATEGIES]))
    print("  Rounds:      {}".format(args.rounds))
    print("  Iters/round: {} (median)".format(ITERS_PER_ROUND))
    print("  Threshold:   {}".format(threshold))
    print("  Seed:        {}".format(args.seed))
    print("  Configs:     {}".format([c[0] for c in configs]))
    print("  Design:      Fixed (batch_frac, target_seg) pairs shared across all combos")
    print("=" * 96)

    rng = TorchRNG(args.seed)

    # ── Pre-generate a FIXED set of (batch_frac, target_seg) pairs ──────────
    # These are dimensionless parameters shared across all sizes/layouts/strategies.
    # This ensures a FAIR controlled comparison: every config sees the exact same
    # fragmentation pattern for each round.
    round_params = []
    for _ in range(args.rounds):
        batch_frac = random_batch_frac(rng)
        target_seg = random_target_seg(rng)
        round_params.append((batch_frac, target_seg))

    print("\n  Pre-generated {} rounds of (batch_frac, target_seg) pairs:".format(len(round_params)))
    for i, (bf, ts) in enumerate(round_params):
        print("    round {:>3d}: batch_frac={:.2f} target_seg={}".format(i, bf, ts))

    # results[combo][config] = {"d2h": [ms...], "h2d": [ms...], "rt": [ms...]}
    all_results = defaultdict(lambda: defaultdict(lambda: {"d2h": [], "h2d": [], "rt": []}))
    round_meta = {}  # combo -> [(batch_size, actual_seg), ...]

    for size_name in args.sizes:
        num_layers, pool_size, head_dim = SIZES[size_name]
        kv_bytes = num_layers * 1 * pool_size * 1 * 1 * head_dim * ES
        print("\n--- Size: {} ({} layers, pool={} blocks, hd={}, {:.1f} MB) ---".format(
            size_name, num_layers, pool_size, head_dim, kv_bytes / (1024**2)))

        # ── Generate block IDs for this size (shared across all combos) ──────
        # Each round's IDs are generated ONCE from the fixed (batch_frac, target_seg)
        # params, then reused for every layout × strategy combination.
        round_ids = []  # [(batch_size, actual_seg, ids_tensor), ...]
        for (batch_frac, target_seg) in round_params:
            batch_size = max(1, int(pool_size * batch_frac))
            batch_size = min(batch_size, pool_size)
            ids, actual_seg = generate_block_ids(batch_size, pool_size, target_seg, rng)
            round_ids.append((batch_size, actual_seg, ids))

        print("  Generated {} block-id sets (batch: min={} med={} max={}, seg: min={} med={} max={})".format(
            len(round_ids),
            min(r[0] for r in round_ids),
            sorted(r[0] for r in round_ids)[len(round_ids) // 2],
            max(r[0] for r in round_ids),
            min(r[1] for r in round_ids),
            sorted(r[1] for r in round_ids)[len(round_ids) // 2],
            max(r[1] for r in round_ids),
        ))

        for layout_name in args.layouts:
            for strat_label, is_mla, mode in STRATEGIES:
                combo = "{}|{}|{}".format(size_name, layout_name, strat_label)
                cpu_layout_type = LAYOUTS[layout_name]
                kv_dim = 1 if is_mla else 2
                heads_per_rank = 1

                print("\n  === {} ===".format(combo))

                # Pre-allocate GPU/CPU tensors at pool_size (max batch).
                gpu_layout, cpu_layout = make_layouts(
                    num_layers, pool_size, head_dim, cpu_layout_type, is_mla, num_gpus)
                cpu_kv_sb, cpu_ly_sb, cpu_bl_sb, cpu_tp_sb, total_blocks = \
                    cpu_strides_for_strategy(cpu_layout, num_layers, pool_size,
                                             head_dim, is_mla, mode, num_gpus)

                all_gpu = [make_gpu_tensors(num_layers, kv_dim, pool_size,
                                            heads_per_rank, head_dim, g)
                           for g in range(num_gpus)]
                cpu_kv = make_cpu_tensor(cpu_layout, num_layers, total_blocks,
                                        head_dim, is_mla, num_gpus)

                round_meta[combo] = []

                for rnd_idx, (batch_size, actual_seg, ids) in enumerate(round_ids):
                    round_meta[combo].append((batch_size, actual_seg))

                    # Run all 4 configs for this round's ids
                    round_times = {}
                    for cfg_label, use_ce, path_opt in configs:
                        tp = make_tp_group(
                            cpu_kv.data_ptr(), all_gpu, num_gpus, gpu_layout,
                            num_layers, ce_path_opt=path_opt,
                            ce_segment_threshold=threshold,
                            is_mla=is_mla,
                            is_blockfirst=(cpu_layout_type == KVCacheLayoutType.BLOCKFIRST))

                        try:
                            d2h_ms = bench_one_dir(
                                tp, ids, cpu_kv_sb, cpu_ly_sb, cpu_bl_sb, cpu_tp_sb,
                                num_layers, False, num_gpus, ITERS_PER_ROUND,
                                is_mla, mode, use_ce=use_ce, transfer_num_cta=cta)
                            h2d_ms = bench_one_dir(
                                tp, ids, cpu_kv_sb, cpu_ly_sb, cpu_bl_sb, cpu_tp_sb,
                                num_layers, True, num_gpus, ITERS_PER_ROUND,
                                is_mla, mode, use_ce=use_ce, transfer_num_cta=cta)
                            round_times[cfg_label] = {"d2h": d2h_ms, "h2d": h2d_ms,
                                                      "rt": d2h_ms + h2d_ms}
                        except Exception as e:
                            round_times[cfg_label] = {"d2h": None, "h2d": None,
                                                      "rt": None}
                            print("    FAILED {}: {}".format(cfg_label, e))
                        del tp

                    # Store
                    for cfg_label in round_times:
                        for dir_key in ("d2h", "h2d", "rt"):
                            v = round_times[cfg_label][dir_key]
                            if v is not None:
                                all_results[combo][cfg_label][dir_key].append(v)

                    # Print this round
                    seg_str = "blk={:>5d} seg={:>5d}".format(batch_size, actual_seg)
                    parts = []
                    for cfg_label, _, _ in configs:
                        t = round_times.get(cfg_label, {})
                        d2h = t.get("d2h")
                        h2d = t.get("h2d")
                        rt = t.get("rt")
                        parts.append("{}: D2H={} H2D={} RT={}".format(
                            cfg_label,
                            "{:.2f}".format(d2h) if d2h else "-",
                            "{:.2f}".format(h2d) if h2d else "-",
                            "{:.2f}".format(rt) if rt else "-"))
                    print("    {}  {}".format(seg_str, "  |  ".join(parts)))

                del all_gpu, cpu_kv

    # ── Summary: per-combo median times ─────────────────────────────────────
    print("\n" + "=" * 110)
    print("  Simulation Summary: median times across all rounds")
    print("  (same block-id sets used for every combo = fair controlled comparison)")
    print("=" * 110)

    for combo in sorted(all_results.keys()):
        meta = round_meta.get(combo, [])
        batches = [m[0] for m in meta]
        segs = [m[1] for m in meta]
        med_seg = sorted(segs)[len(segs) // 2] if segs else 0
        min_seg = min(segs) if segs else 0
        max_seg = max(segs) if segs else 0
        med_batch = sorted(batches)[len(batches) // 2] if batches else 0
        min_batch = min(batches) if batches else 0
        max_batch = max(batches) if batches else 0
        print("\n  {} (rounds={}, batch: min={} med={} max={}, seg: min={} med={} max={})".format(
            combo, len(meta), min_batch, med_batch, max_batch,
            min_seg, med_seg, max_seg))
        print("  {:<14s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format(
            "config", "D2H(ms)", "H2D(ms)", "RT(ms)", "D2H spd", "H2D spd", "RT spd"))
        print("  " + "-" * 90)

        # Get baseline for speedup calc
        base_d2h = _median(all_results[combo].get("baseline", {}).get("d2h", []))
        base_h2d = _median(all_results[combo].get("baseline", {}).get("h2d", []))
        base_rt = _median(all_results[combo].get("baseline", {}).get("rt", []))

        for cfg_label, _, _ in configs:
            d2h = _median(all_results[combo][cfg_label]["d2h"])
            h2d = _median(all_results[combo][cfg_label]["h2d"])
            rt = _median(all_results[combo][cfg_label]["rt"])
            d2h_spd = "{:.2f}x".format(base_d2h / d2h) if (base_d2h and d2h) else "-"
            h2d_spd = "{:.2f}x".format(base_h2d / h2d) if (base_h2d and h2d) else "-"
            rt_spd = "{:.2f}x".format(base_rt / rt) if (base_rt and rt) else "-"
            print("  {:<14s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}  {:>10s}".format(
                cfg_label,
                "{:.3f}".format(d2h) if d2h else "-",
                "{:.3f}".format(h2d) if h2d else "-",
                "{:.3f}".format(rt) if rt else "-",
                d2h_spd, h2d_spd, rt_spd))

    print("\n  Note: spd = baseline / config (higher = faster).")
    print("        RT = D2H + H2D (complete round-trip).")
    print("        All combos use the SAME (batch_size, segment_count) pairs per round")
    print("        — fair controlled comparison across configs.")
    print("=" * 110)


def _median(vals):
    """Return median of a list, or None if empty."""
    if not vals:
        return None
    s = sorted(vals)
    return s[len(s) // 2]


# Main

def main():
    parser = argparse.ArgumentParser(
        description="CE transfer Monte Carlo simulation (random fragmentation)")
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Number of GPUs (0 = all available)")
    parser.add_argument("--rounds", type=int, default=50,
                        help="Random rounds per combination (default: 50)")
    parser.add_argument("--sizes", nargs="+", default=list(SIZES.keys()),
                        choices=list(SIZES.keys()),
                        help="Data sizes to test (default: all)")
    parser.add_argument("--layouts", nargs="+", default=list(LAYOUTS.keys()),
                        choices=list(LAYOUTS.keys()),
                        help="CPU layouts (default: both)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--skip-kernel", action="store_true",
                        help="Omit the 'kernel' (CUDA kernel) config. Use on "
                             "non-NVIDIA platforms where the custom CUDA kernel "
                             "cannot build/run; baseline+opt (CE paths) are kept.")
    args = parser.parse_args()

    num_gpus = NUM_GPUS if args.num_gpus <= 0 else min(args.num_gpus, NUM_GPUS)
    if num_gpus < 2:
        print("ERROR: need at least 2 GPUs, found {}".format(NUM_GPUS))
        sys.exit(1)
    args.num_gpus = num_gpus

    run_simulation(args)


if __name__ == "__main__":
    main()
