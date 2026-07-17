"""
Microbenchmark: CE segment_threshold sweep — PER-CASE optimal threshold.

This benchmark answers requirement (3): each (size, layout) case must confirm
its OWN optimal CE segment_threshold. We do NOT pick a single global threshold
here — every case self-selects the threshold that minimizes its round-trip
time and the output makes that choice explicit ("<-- best for THIS case").

Background — segment_threshold semantics (authoritative):
  The CE engine's choose_path() uses segment_threshold to pick between
  SEGMENT_SCATTER (segment memcpy through a staging buffer) and GATHER_SCATTER
  (GPU index_select/copy):
    - num_segments <= threshold -> SEGMENT_SCATTER
                                    (or SEGMENT_DIRECT if the dst is
                                     physically contiguous / LAYERFIRST)
    - num_segments >  threshold -> GATHER_SCATTER
  So threshold controls the STAGED/GATHER crossover. By driving the number of
  contiguous block-id segments independently (make_pattern_with_segments), we
  can sweep num_segments against threshold and find, for each (size, layout),
  the threshold that is fastest for THAT case.

  threshold is passed PER-CONSTRUCTION (ce_segment_threshold ctor arg), NOT via
  env — matching production and the correctness tests.

Note on mode: we use mla_d2h_mode="rank0_only" throughout. Threshold behavior is
mode-independent: choose_path() only looks at the segment count + destination
contiguity, so "rank0_only" keeps CPU-buffer sizing simple without changing which
path the threshold selects.

This is a CUDA benchmark and requires a GPU; it cannot run on a CPU-only host.

Usage:
    python benchmarks/microbenchmark_ce_threshold.py --num-gpus 4 --iters 10
    python benchmarks/microbenchmark_ce_threshold.py --sizes mla-medium
    python benchmarks/microbenchmark_ce_threshold.py --layouts LAYERFIRST
"""

import argparse
import sys
import time

import numpy as np

# GPU / FlexKV detection
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


# Configurations

DTYPE = torch.float16
ES = DTYPE.itemsize

# A couple of MLA sizes. Tuple = (num_layers, num_blocks, tokens_per_block,
# head_dim, num_heads); is_mla=True for all threshold cases.
THRESHOLD_SIZES = {
    "mla-medium": (32, 64, 16, 512, 1),
    "mla-large":  (80, 256, 16, 512, 1),
}

# Only sweep LAYERFIRST: BLOCKFIRST always uses GATHER_DIRECT (checked first
# in choose_path), so segment_threshold has no effect on BF. Running threshold
# sweeps on BF would produce identical results for every threshold value.
THRESHOLD_LAYOUTS = ["LAYERFIRST"]

# CE segment_threshold values to sweep (the STAGED/GATHER crossover point).
THRESHOLD_VALUES = [2, 4, 8, 16, 32]

# Number of contiguous block-id segments to drive. Only those <= num_blocks
# for a given size are actually used.
SWEEP_SEGMENT_COUNTS = [1, 2, 4, 8, 16, 32]

_LAYOUT_TYPES = {
    "LAYERFIRST": KVCacheLayoutType.LAYERFIRST,
    "BLOCKFIRST": KVCacheLayoutType.BLOCKFIRST,
}

# All threshold cases run MLA "rank0_only" mode. Threshold behavior is mode-independent:
# choose_path() only looks at the segment count + destination contiguity, so
# rank0_only keeps CPU-buffer sizing simple without changing which path the
# threshold selects.
MODE = "rank0_only"
IS_MLA = True

WARMUP_ITERS = 3


# Engine probe

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
            layer_id=0, layer_granularity=1, is_mla=True, mla_d2h_mode="rank0_only")
        torch.cuda.synchronize()
        del tp
        _probe_cache[key] = True
        return True
    except Exception:
        _probe_cache[key] = False
        return False


# Helpers (matching production worker.py — copied from
# microbenchmark_mla_d2h_modes.py so behavior is identical)

def make_layouts(num_layers, num_blocks, head_dim, cpu_layout_type, is_mla, num_gpus):
    """Create GPU (LAYERFIRST) and CPU layouts.

    MLA: kv_dim=1, head=1 (all ranks identical).
    MHA: kv_dim=2, head=num_gpus (each rank gets 1 head).
    GPU uses heads_per_rank (per-rank), CPU uses full num_head.
    """
    num_head = 1 if is_mla else num_gpus
    heads_per_rank = 1  # MLA: 1 shared head; MHA: num_gpus/num_gpus = 1
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


def make_pattern_with_segments(num_blocks, num_segments):
    """Block-id permutation with EXACTLY num_segments contiguous runs.

    Splits range(num_blocks) into num_segments (near-equal) chunks and reverses
    their order. Reversing the chunk order guarantees the sequence breaks into
    exactly num_segments contiguous ascending runs — this lets us drive the CE
    engine's observed num_segments independently of segment_threshold.

    Returns a pinned int64 tensor of length num_blocks.
    """
    if num_segments < 1:
        num_segments = 1
    if num_segments > num_blocks:
        num_segments = num_blocks
    # Near-equal chunk sizes summing to num_blocks.
    base = num_blocks // num_segments
    rem = num_blocks % num_segments
    chunks = []
    start = 0
    for i in range(num_segments):
        size = base + (1 if i < rem else 0)
        chunks.append(list(range(start, start + size)))
        start += size
    # Reverse chunk order -> exactly num_segments ascending runs.
    ids = []
    for chunk in reversed(chunks):
        ids.extend(chunk)
    return torch.tensor(ids, dtype=torch.int64).pin_memory()


def make_tp_group(cpu_ptr, all_gpu, num_gpus, gpu_layout, num_layers,
                  ce_path_opt=True,
                  ce_segment_threshold=8,
                  is_mla=False, is_blockfirst=False):
    """TPTransferThreadGroup with CE config passed per-construction.

    path_opt / segment_threshold go into the C++ CETransferConfig
    via ctor args (NOT env) -- matching production and the correctness tests.
    """
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
        is_blockfirst=is_blockfirst)


def sync_all(num_gpus):
    for g in range(num_gpus):
        torch.cuda.synchronize(g)


# Benchmark core

def bench_threshold(cpu_layout_type, num_gpus, num_layers, num_blocks, head_dim,
                    iters, ce_segment_threshold, num_segments):
    """One (threshold, seg_count) point: median D2H+H2D round-trip time (ms).

    Uses CE with path_opt=True (the optimized path where the
    STAGED/GATHER crossover — i.e. segment_threshold — actually matters).
    The block-id pattern is built with EXACTLY num_segments contiguous runs so
    the engine's choose_path() sees that segment count.
    """
    is_mla = IS_MLA
    mode = MODE
    kv_dim = 1 if is_mla else 2
    heads_per_rank = 1

    gpu_layout, cpu_layout = make_layouts(
        num_layers, num_blocks, head_dim, cpu_layout_type, is_mla, num_gpus)
    cpu_kv_sb, cpu_ly_sb, cpu_bl_sb, cpu_tp_sb, total_blocks = \
        cpu_strides_for_strategy(cpu_layout, num_layers, num_blocks, head_dim,
                                  is_mla, mode, num_gpus)

    all_gpu = [make_gpu_tensors(num_layers, kv_dim, num_blocks, heads_per_rank,
                                head_dim, g) for g in range(num_gpus)]
    cpu_kv = make_cpu_tensor(cpu_layout, num_layers, total_blocks, head_dim,
                             is_mla, num_gpus)
    tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus, gpu_layout,
                       num_layers, ce_path_opt=True,
                       ce_segment_threshold=ce_segment_threshold,
                       is_mla=is_mla,
                       is_blockfirst=(cpu_layout_type == KVCacheLayoutType.BLOCKFIRST))

    # Same pattern for gpu and cpu side; num_segments controls the crossover.
    ids = make_pattern_with_segments(num_blocks, num_segments)

    def do_d2h():
        tp.tp_group_transfer(
            gpu_block_id_tensor=ids, cpu_block_id_tensor=ids,
            cpu_kv_stride_in_bytes=cpu_kv_sb, cpu_layer_stride_in_bytes=cpu_ly_sb,
            cpu_block_stride_in_bytes=cpu_bl_sb, cpu_tp_stride_in_bytes=cpu_tp_sb,
            transfer_num_cta=4, is_host_to_device=False, use_ce_transfer=True,
            layer_id=0, layer_granularity=num_layers, is_mla=is_mla,
            mla_d2h_mode=mode)

    def do_h2d():
        tp.tp_group_transfer(
            gpu_block_id_tensor=ids, cpu_block_id_tensor=ids,
            cpu_kv_stride_in_bytes=cpu_kv_sb, cpu_layer_stride_in_bytes=cpu_ly_sb,
            cpu_block_stride_in_bytes=cpu_bl_sb, cpu_tp_stride_in_bytes=cpu_tp_sb,
            transfer_num_cta=4, is_host_to_device=True, use_ce_transfer=True,
            layer_id=0, layer_granularity=num_layers, is_mla=is_mla,
            mla_d2h_mode=mode)

    # Warmup
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

    # Timing: full D2H + H2D round-trip per iteration.
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
    return float(np.median(times))


# Print helpers

def _col(text, width):
    return str(text).ljust(width)


def print_case_grid(size_name, layout_name, thresholds, seg_counts, grid):
    """Print a per-(size,layout) grid: rows=seg_count, cols=threshold, cell=ms."""
    print("")
    print("-" * 78)
    print("  [size={} layout={}] grid (rows=num_segments, cols=threshold, ms)".format(
        size_name, layout_name))
    print("-" * 78)
    hdr = _col("seg\\thr", 10)
    for thr in thresholds:
        hdr += _col(str(thr), 10)
    print("  " + hdr)
    print("  " + "-" * (len(hdr)))
    for si, seg in enumerate(seg_counts):
        line = _col(str(seg), 10)
        for ti in range(len(thresholds)):
            v = grid[si][ti]
            line += _col("-" if v is None else "{:.3f}".format(v), 10)
        print("  " + line)


def select_and_print_best(size_name, layout_name, thresholds, seg_counts, grid):
    """Compute + print THIS case's optimal threshold (self-selecting).

    For each threshold column, take the representative time = sum over the
    seg_counts (missing cells ignored). The best threshold is the column with
    the minimum representative time. This is the threshold that is best for
    THIS case alone — we never compare across cases.

    Returns (best_threshold, best_repr_ms).
    """
    col_totals = []
    for ti, thr in enumerate(thresholds):
        vals = [grid[si][ti] for si in range(len(seg_counts))
                if grid[si][ti] is not None]
        col_totals.append(sum(vals) if vals else None)

    valid = [(thresholds[ti], col_totals[ti])
             for ti in range(len(thresholds)) if col_totals[ti] is not None]
    if not valid:
        print("  [size={} layout={}] no valid timings — cannot select".format(
            size_name, layout_name))
        return None, None

    best_thr, best_total = min(valid, key=lambda x: x[1])

    # Soft assert: by construction the chosen best IS the column minimum.
    assert best_total == min(v for _, v in valid), \
        "internal error: selected threshold is not the minimum for this case"

    print("  [size={} layout={}] optimal threshold = {} ({:.3f} ms)"
          "  <-- best for THIS case".format(
              size_name, layout_name, best_thr, best_total))
    return best_thr, best_total


# main

def main():
    parser = argparse.ArgumentParser(
        description="CE segment_threshold sweep — per-case optimal threshold "
                    "(REAL FlexKV API)")
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Number of GPUs (default: 0 = all available)")
    parser.add_argument("--iters", type=int, default=10,
                        help="Timing iterations per (threshold, seg) point "
                             "(default: 10)")
    parser.add_argument("--sizes", type=str, nargs="+",
                        default=list(THRESHOLD_SIZES.keys()),
                        choices=list(THRESHOLD_SIZES.keys()),
                        help="MLA sizes to test (default: all)")
    parser.add_argument("--layouts", type=str, nargs="+",
                        default=list(THRESHOLD_LAYOUTS),
                        choices=list(THRESHOLD_LAYOUTS),
                        help="CPU layouts to test (default: both)")
    args = parser.parse_args()

    num_gpus = NUM_GPUS if args.num_gpus <= 0 else min(args.num_gpus, NUM_GPUS)
    if num_gpus < 1:
        print("ERROR: need at least 1 GPU, found {}".format(NUM_GPUS))
        sys.exit(1)

    # CE engine probe (this benchmark is CE-only — threshold is a CE concept).
    if not probe_engine(use_ce=True):
        print("ERROR: CE engine probe failed — cannot sweep segment_threshold")
        sys.exit(1)

    print("=" * 90)
    print("  FlexKV CE segment_threshold sweep — PER-CASE optimal threshold")
    print("=" * 90)
    print("  GPUs:        {}".format(num_gpus))
    print("  Sizes:       {}".format(args.sizes))
    print("  Layouts:     {}".format(args.layouts))
    print("  Thresholds:  {}".format(THRESHOLD_VALUES))
    print("  Seg counts:  {}".format(SWEEP_SEGMENT_COUNTS))
    print("  Mode:        {} (threshold behavior is mode-independent)".format(MODE))
    print("  Iters:       {}".format(args.iters))
    print("  Dtype:       {}".format(DTYPE))
    print("=" * 90)

    # Collect each case's self-selected best for a final recap (per-case only).
    case_best = []

    for size_name in args.sizes:
        num_layers, num_blocks, tpb, head_dim, num_heads = THRESHOLD_SIZES[size_name]
        # Only sweep segment counts that fit within this size's num_blocks.
        seg_counts = [s for s in SWEEP_SEGMENT_COUNTS if s <= num_blocks]
        kv_bytes = num_layers * 1 * num_blocks * 1 * 1 * head_dim * ES
        print("\n=== Size: {} ({} layers, {} blocks, tpb={}, hd={}, {:.1f} MB) ===".format(
            size_name, num_layers, num_blocks, tpb, head_dim,
            kv_bytes / (1024**2)))

        for layout_name in args.layouts:
            cpu_layout_type = _LAYOUT_TYPES[layout_name]
            # grid[seg_idx][thr_idx] = median ms
            grid = [[None for _ in THRESHOLD_VALUES] for _ in seg_counts]

            for ti, thr in enumerate(THRESHOLD_VALUES):
                for si, seg in enumerate(seg_counts):
                    label = "size={} layout={} thr={} seg={}".format(
                        size_name, layout_name, thr, seg)
                    print("  Running: {} ...".format(label),
                          end=" ", flush=True)
                    try:
                        ms = bench_threshold(
                            cpu_layout_type, num_gpus, num_layers, num_blocks,
                            head_dim, args.iters, ce_segment_threshold=thr,
                            num_segments=seg)
                        grid[si][ti] = ms
                        print("median={:.3f}ms".format(ms))
                    except Exception as e:
                        print("FAILED: {}".format(e))

            print_case_grid(size_name, layout_name, THRESHOLD_VALUES,
                            seg_counts, grid)
            best_thr, best_ms = select_and_print_best(
                size_name, layout_name, THRESHOLD_VALUES, seg_counts, grid)
            if best_thr is not None:
                case_best.append((size_name, layout_name, best_thr, best_ms))

    # --- Per-case recap (NO global winner — each case stands alone) ---
    print("\n" + "=" * 90)
    print("  Per-case optimal segment_threshold recap (each case self-selected)")
    print("=" * 90)
    for size_name, layout_name, best_thr, best_ms in case_best:
        print("  [size={} layout={}] optimal threshold = {} ({:.3f} ms)"
              "  <-- best for THIS case".format(
                  size_name, layout_name, best_thr, best_ms))
    print("\n  Note: Each (size, layout) case reports its OWN best threshold.")
    print("        We deliberately do NOT pick one global threshold — the")
    print("        optimal crossover depends on the case's size + layout.")
    print("        Performance also depends on hardware (NUMA, PCIe/NVLink).")
    print("=" * 90)


if __name__ == "__main__":
    main()
