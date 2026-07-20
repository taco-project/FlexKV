"""
Microbenchmark: CE transfer strategy comparison and auto-selection.

Runs all viable CEPath strategies for each (pattern, layout, mode, dir) form,
then automatically selects the fastest strategy as the recommendation.

Five strategies + baseline (PER_BLOCK):
  Naming: <source>_<dest> - how data is read from GPU -> how data is written to CPU
  - baseline           PER_BLOCK: per-block memcpy, no optimization (cached)
  - CONTIG_DIRECT      contiguous source -> direct memcpy (1 segment, no staging)
  - SEGMENT_DIRECT     segmented source -> direct per-segment memcpy (no staging)
  - SEGMENT_SCATTER    segmented source -> staging buffer + CPU scatter
  - GATHER_SCATTER     GPU index_select_out gather -> staging + CPU scatter
                       (many segments OR sharded D2H via strided from_blob)
  - GATHER_DIRECT      GPU index_select_out gather into 3D staging (BLOCKFIRST)
                       -> direct per-segment memcpy (no staging, no CPU scatter)
                       BF only: D2D layout transform needed for direct match

For each form, all viable strategies are force-tested head-to-head. The
fastest is marked as 'recommended'. If choose_path's auto-pick matches the
recommended, choose_path is optimal for that form; otherwise it should be
investigated.

memcpy2d (FLEXKV_ENABLE_CE_MEMCPY2D, default ON) affects path 2 SEGMENT_SCATTER,
path 3 GATHER_SCATTER, and path 4 GATHER_DIRECT (D2H + H2D). With it ON, paths
2/3/4 bypass the pinned host staging buffer and use cudaMemcpy2DAsync for the
strided transfer directly (SEGMENT_SCATTER returns before allocating any
staging buffer; GATHER_SCATTER/GATHER_DIRECT keep only the device staging
buffer needed for the GPU index_select / D2D transpose). On platforms where
cudaMemcpy2DAsync is slow or unsupported, set FLEXKV_ENABLE_CE_MEMCPY2D=0 to
use the portable staging + CPU scatter path. Default is ON (NVIDIA fast);
pass --memcpy2d off to disable.

Usage:
    python benchmarks/microbenchmark_ce_strategy.py --num-gpus 4 --iters 20
    python benchmarks/microbenchmark_ce_strategy.py --num-gpus 4 --iters 20 --memcpy2d off
"""

import argparse
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
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
    FLEXKV_AVAILABLE = True
except ImportError as e:
    print("ERROR: FlexKV not available ({})".format(e))
    sys.exit(1)

DTYPE = torch.float16
ES = DTYPE.itemsize
WARMUP_ITERS = 3

SIZES = {
    "small":  (32,   512,  128),
    "medium": (61,  2048,  512),
    "large":  (80,  8192,  512),
}


# -- Layout / stride helpers (MLA + BLOCKFIRST capable) -----------------------

STRAT_LAYOUTS = {
    "lfirst": KVCacheLayoutType.LAYERFIRST,
    "bfirst": KVCacheLayoutType.BLOCKFIRST,
}


def make_layouts_strat(num_layers, num_blocks, head_dim, cpu_layout_type,
                       is_mla, num_gpus):
    """GPU (LAYERFIRST, per-rank heads) and CPU (cpu_layout_type) layouts.

    MLA: kv_dim=1, head=1 (all ranks identical).
    MHA: kv_dim=2, head=num_gpus (each rank gets 1 head).
    """
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
    """Return (cpu_kv_sb, cpu_layer_sb, cpu_block_sb, cpu_tp_sb, total_blocks)."""
    total = num_blocks
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


def make_gpu_tensors_strat(num_layers, kv_dim, num_blocks, heads_per_rank,
                           head_dim, device):
    """Contiguous [num_layers, kv_dim, num_blocks, 1, heads_per_rank, head_dim]."""
    full = torch.empty(
        (num_layers, kv_dim, num_blocks, 1, heads_per_rank, head_dim),
        dtype=DTYPE, device="cuda:{}".format(device))
    return [full[i] for i in range(num_layers)]


def make_cpu_tensor_strat(cpu_layout, num_layers, total_blocks, head_dim,
                          is_mla, num_gpus):
    num_head = 1 if is_mla else num_gpus
    layout = KVCacheLayout(
        type=cpu_layout.type,
        num_layer=num_layers, num_block=total_blocks,
        tokens_per_block=1, num_head=num_head,
        head_size=head_dim, is_mla=is_mla)
    return torch.empty(tuple(layout.kv_shape), dtype=DTYPE, pin_memory=True)


def make_tp_group(cpu_ptr, all_gpu, num_gpus, gpu_layout, num_layers,
                  ce_path_opt=True,
                  ce_segment_threshold=8, ce_force_path=-1,
                  is_mla=False, is_blockfirst=False,
                  ce_enable_memcpy2d=False):
    """TPTransferThreadGroup with CE config passed per-construction."""
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
        ce_force_path=ce_force_path,
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


def make_block_id_pattern(kind, num_blocks):
    """Build a pinned int64 block-id tensor selecting a CE execution form."""
    if kind == "contiguous":
        ids = torch.arange(num_blocks, dtype=torch.int64)
    elif kind == "few_seg":
        q = num_blocks // 4
        parts = [torch.arange(i * q, (i + 1) * q, dtype=torch.int64)
                 for i in range(4)]
        ids = torch.cat([parts[0], parts[2], parts[1], parts[3]])[:num_blocks]
    elif kind == "scattered":
        gen = torch.Generator()
        gen.manual_seed(42)
        ids = torch.randperm(num_blocks, generator=gen).to(torch.int64)
    else:
        raise ValueError("unknown pattern kind: {}".format(kind))
    return ids.pin_memory()


# -- Benchmark core -----------------------------------------------------------

def bench_one_dir(tp, ids, cpu_kv_sb, cpu_ly_sb, cpu_bl_sb, cpu_tp_sb,
                  num_layers, is_h2d, num_gpus, iters, is_mla, mode,
                  transfer_num_cta=16):
    """Time ONE direction (H2D or D2H) over `iters`, return median wall-clock ms."""

    def do_transfer():
        tp.tp_group_transfer(
            gpu_block_id_tensor=ids, cpu_block_id_tensor=ids,
            cpu_kv_stride_in_bytes=cpu_kv_sb, cpu_layer_stride_in_bytes=cpu_ly_sb,
            cpu_block_stride_in_bytes=cpu_bl_sb, cpu_tp_stride_in_bytes=cpu_tp_sb,
            transfer_num_cta=transfer_num_cta, is_host_to_device=is_h2d,
            use_ce_transfer=True, layer_id=0, layer_granularity=num_layers,
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


# Five strategies
# CEPath enum: 0=CONTIG_DIRECT, 1=SEGMENT_DIRECT, 2=SEGMENT_SCATTER,
#              3=GATHER_SCATTER, 4=GATHER_DIRECT

STRATEGIES = [
    (0, "CONTIG_DIRECT"),
    (1, "SEGMENT_DIRECT"),
    (2, "SEGMENT_SCATTER"),
    (3, "GATHER_SCATTER"),
    (4, "GATHER_DIRECT"),
]

# Abbreviations for table columns: <source>_<dest> pattern
# Source: C=CONTIG, S=SEGMENT, G=GATHER | Dest: DIR=DIRECT, SCT=SCATTER
STR_ABBR = {
    "CONTIG_DIRECT": "C_DIR",
    "SEGMENT_DIRECT": "S_DIR",
    "SEGMENT_SCATTER": "S_SCT",
    "GATHER_SCATTER": "G_SCT",
    "GATHER_DIRECT": "G_DIR",
}

# Paths that consume FLEXKV_ENABLE_CE_MEMCPY2D (cudaMemcpy2DAsync). These three
# are affected by the flag; the others ignore it. Used by --memcpy2d on.
# NOTE: must use string names (not int IDs) — checked via `fp_name in AFFECTED_PATHS`.
AFFECTED_PATHS = {"SEGMENT_SCATTER", "GATHER_SCATTER", "GATHER_DIRECT"}  # path 2, 3, 4
# Column abbreviation for the memcpy2d=1 variant of an affected path.
STR_ABBR_2D = {
    "SEGMENT_SCATTER": "S_SCT2",
    "GATHER_SCATTER": "G_SCT2",
    "GATHER_DIRECT": "G_DIR2",
}

# -- Viable force paths per form ---------------------------------------------
# STRICT compatibility (data correct + no segfault).
#
# - CONTIG_DIRECT: contiguous + LF + non-sharded
# - SEGMENT_DIRECT: LF + non-sharded (cpu_phys_contig)
# - SEGMENT_SCATTER: all scenarios (ptr_at + staging + scatter)
# - GATHER_SCATTER: LF + MLA (non-sharded OR sharded via strided from_blob)
#   (BF segfault; MHA segfault)
# - GATHER_DIRECT: BF only (LF segfault — from_blob with BF stride assumption)
def correct_paths_for(layout_key, is_mla, pattern, mode, is_h2d, threshold):
    """Return the (path_id, name) list of strategies that are DATA-CORRECT for
    a form. Mirrors the hard constraints each CEPath imposes (see csrc/ce_transfer.h):

      CONTIG_DIRECT(0):   cpu_phys_contig & gpu_phys_contig & contiguous (1 seg)
      SEGMENT_DIRECT(1):  cpu_phys_contig & gpu_phys_contig & <=threshold segs
      SEGMENT_SCATTER(2): gpu_phys_contig (non-sharded); CPU scatter handles any layout
      GATHER_SCATTER(3):  ALWAYS correct — GPU gather + CPU scatter, any layout / sharded
      GATHER_DIRECT(4):   BF (!cpu_phys_contig) & gpu_phys_contig (non-sharded)

    cpu_phys_contig = (layout==lfirst) and is_mla.
    gpu_phys_contig = !(mode==sharded and direction==D2H); sharded only breaks D2H.
    Computed rather than hand-listed so it can never drift from the C++ rules.
    """
    cpu_phys_contig = (layout_key == "lfirst") and is_mla
    gpu_phys_contig = not (mode == "sharded" and not is_h2d)
    if pattern == "contiguous":
        num_segments = 1
    elif pattern == "few_seg":
        num_segments = 4
    else:
        num_segments = threshold + 1
    out = []
    if cpu_phys_contig and gpu_phys_contig and num_segments == 1:
        out.append((0, "CONTIG_DIRECT"))
    if cpu_phys_contig and gpu_phys_contig and num_segments <= threshold:
        out.append((1, "SEGMENT_DIRECT"))
    if gpu_phys_contig:
        out.append((2, "SEGMENT_SCATTER"))
    out.append((3, "GATHER_SCATTER"))  # universal fallback — always correct
    # GATHER_DIRECT is BF-only (BLOCKFIRST). On LF it assumes the wrong layout,
    # so gate on is_blockfirst — not just `not cpu_phys_contig` (which is also
    # true for MHA+LF and would wrongly include GATHER_DIRECT there).
    if (layout_key == "bfirst") and gpu_phys_contig:
        out.append((4, "GATHER_DIRECT"))
    return out


# Full matrix ordered by: model → mode → layout → continuity.
# This groups H2D+D2H for the same (model × mode × layout × continuity) together.
PATH_FORMS = [
    # --- MLA + rank0_only ---
    ("contiguous", "lfirst", True,  "rank0_only",     [True, False]),
    ("contiguous", "bfirst", True,  "rank0_only",     [True, False]),
    ("few_seg",    "lfirst", True,  "rank0_only",     [True, False]),
    ("few_seg",    "bfirst", True,  "rank0_only",     [True, False]),
    ("scattered",  "lfirst", True,  "rank0_only",     [True, False]),
    ("scattered",  "bfirst", True,  "rank0_only",     [True, False]),
    # --- MLA + layer_parallel ---
    ("contiguous", "lfirst", True,  "layer_parallel", [True, False]),
    ("contiguous", "bfirst", True,  "layer_parallel", [True, False]),
    ("few_seg",    "lfirst", True,  "layer_parallel", [True, False]),
    ("few_seg",    "bfirst", True,  "layer_parallel", [True, False]),
    ("scattered",  "lfirst", True,  "layer_parallel", [True, False]),
    ("scattered",  "bfirst", True,  "layer_parallel", [True, False]),
    # --- MLA + sharded (D2H only) ---
    ("contiguous", "lfirst", True,  "sharded",        [False]),
    ("contiguous", "bfirst", True,  "sharded",        [False]),
    ("few_seg",    "lfirst", True,  "sharded",        [False]),
    ("few_seg",    "bfirst", True,  "sharded",        [False]),
    ("scattered",  "lfirst", True,  "sharded",        [False]),
    ("scattered",  "bfirst", True,  "sharded",        [False]),
    # --- MHA (H2D + D2H, mode is don't-care) ---
    ("contiguous", "lfirst", False, "rank0_only",     [True, False]),
    ("contiguous", "bfirst", False, "rank0_only",     [True, False]),
    ("few_seg",    "lfirst", False, "rank0_only",     [True, False]),
    ("few_seg",    "bfirst", False, "rank0_only",     [True, False]),
    ("scattered",  "lfirst", False, "rank0_only",     [True, False]),
    ("scattered",  "bfirst", False, "rank0_only",     [True, False]),
]


def python_choose_path(pattern, layout_key, mode, is_h2d, threshold,
                       chunk_size_bytes, is_mla=True):
    """Mirror of C++ choose_path (ce_transfer.cu). Returns the strategy name
    that choose_path would pick for the given (pattern, layout, mode, dir).
    """
    cpu_phys_contig = (layout_key == "lfirst") and is_mla
    # MHA + LF: cpu_block_stride = num_gpus * head_dim != chunk_size = head_dim
    # (per-rank), so cpu_phys_contig = false even for LAYERFIRST.
    is_blockfirst = (layout_key == "bfirst")
    gpu_phys_contig = not (mode == "sharded" and not is_h2d)
    if pattern == "contiguous":
        num_segments = 1
    elif pattern == "few_seg":
        num_segments = 4
    else:
        num_segments = threshold + 1

    # GATHER_DIRECT: BF + !cpu_phys_contig + GPU physically contiguous
    # (non-sharded). Sharded D2H breaks gpu_phys_contig -> GATHER_SCATTER.
    # Exception: bfirst + MLA + D2H + !full_block (layer_parallel) ->
    # SEGMENT_SCATTER is 30%-8.9x faster than GATHER_DIRECT. Mirrors
    # ce_transfer.cu:120-128 (is_full_block = mode in (rank0_only, all_write);
    # !is_host_to_device = D2H).
    if is_blockfirst and not cpu_phys_contig and gpu_phys_contig:
        if not is_h2d and is_mla and mode not in ("rank0_only", "all_write"):
            return "SEGMENT_SCATTER"
        return "GATHER_DIRECT"
    if cpu_phys_contig and gpu_phys_contig and num_segments == 1:
        return "CONTIG_DIRECT"
    if not gpu_phys_contig:
        return "GATHER_SCATTER"
    if num_segments <= threshold:
        return "SEGMENT_DIRECT" if cpu_phys_contig else "SEGMENT_SCATTER"
    if chunk_size_bytes > 0 and chunk_size_bytes % 8 != 0:
        return "SEGMENT_SCATTER"
    return "GATHER_SCATTER"


# -- Main benchmark -----------------------------------------------------------

def run_strategy_compare(args):
    """Run all viable strategies per form, select fastest as recommendation."""
    num_gpus = args.num_gpus
    threshold = 8
    cta = 16
    # PER_BLOCK baseline is pattern-independent. Cache by (size, layout, mode,
    # dir) so we only time it once per unique combo.
    baseline_cache = {}

    print("=" * 100)
    print("  CE Strategy Auto-Selection: 5 strategies head-to-head + recommended")
    print("=" * 100)
    print("  GPUs:       {}".format(num_gpus))
    print("  Sizes:      {}".format(args.sizes))
    print("  Threshold:  {}".format(threshold))
    print("  Iters:      {}".format(args.iters))
    print("  Strategies: {}".format(", ".join(s[1] for s in STRATEGIES)))
    print("  memcpy2d:   {}".format(
        "on (default) — SEGMENT_SCATTER/GATHER_SCATTER/GATHER_DIRECT also timed "
        "with cudaMemcpy2DAsync"
        if args.memcpy2d == "on" else "off"))
    print("=" * 100)

    # results[size][(form_name, dir_name)][strategy_name] = median_ms
    all_results = defaultdict(lambda: defaultdict(dict))

    for size_name in args.sizes:
        num_layers, num_blocks, head_dim = SIZES[size_name]
        kv_bytes = num_layers * 1 * num_blocks * 1 * 1 * head_dim * ES
        print("\n--- Size: {} ({} layers, {} blocks, hd={}, {:.1f} MB) ---".format(
            size_name, num_layers, num_blocks, head_dim, kv_bytes / (1024**2)))

        results = all_results[size_name]
        heads_per_rank = 1

        for pattern, layout_key, is_mla, mode, dirs in PATH_FORMS:
            mla_tag = "mla" if is_mla else "mha"
            # MHA mode is don't-care — don't show it in the form name.
            if is_mla:
                form_name = "{}/{}/{}/{}".format(pattern, layout_key, mla_tag, mode)
            else:
                form_name = "{}/{}/{}".format(pattern, layout_key, mla_tag)
            if pattern == "scattered" and num_blocks <= threshold:
                print("  SKIP {} (num_blocks={} <= threshold={})".format(
                    form_name, num_blocks, threshold))
                continue
            kv_dim = 1 if is_mla else 2
            cpu_layout_type = STRAT_LAYOUTS[layout_key]

            gpu_layout, cpu_layout = make_layouts_strat(
                num_layers, num_blocks, head_dim, cpu_layout_type, is_mla, num_gpus)
            cpu_kv_sb, cpu_ly_sb, cpu_bl_sb, cpu_tp_sb, total_blocks = \
                cpu_strides_for_strategy(cpu_layout, num_layers, num_blocks,
                                         head_dim, is_mla, mode, num_gpus)

            all_gpu = [make_gpu_tensors_strat(num_layers, kv_dim, num_blocks,
                                              heads_per_rank, head_dim, g)
                       for g in range(num_gpus)]
            cpu_kv = make_cpu_tensor_strat(cpu_layout, num_layers, total_blocks,
                                           head_dim, is_mla, num_gpus)
            ids = make_block_id_pattern(pattern, num_blocks)

            for is_h2d in dirs:
                dir_name = "H2D" if is_h2d else "D2H"
                key = (form_name, dir_name)
                auto_path = python_choose_path(
                    pattern, layout_key, mode, is_h2d, threshold,
                    head_dim * ES, is_mla=is_mla)
                results[key]["auto_path"] = auto_path
                if is_mla:
                    form_display = "{} | {} | mla-{}".format(pattern, layout_key, mode)
                else:
                    form_display = "{} | {} | mha".format(pattern, layout_key)
                print("\n-- {} | {} | auto={} --".format(
                    form_display, dir_name, auto_path))

                # Run baseline (PER_BLOCK, path_opt=false) — cached
                bk = (size_name, layout_key, mode, dir_name)
                cached_bs = baseline_cache.get(bk)
                if cached_bs is not None:
                    results[key]["baseline"] = cached_bs
                    print("  baseline (cached) {:.3f} ms".format(cached_bs))
                else:
                    print("  baseline ...", end=" ", flush=True)
                    try:
                        tp = make_tp_group(
                            cpu_kv.data_ptr(), all_gpu, num_gpus, gpu_layout,
                            num_layers, ce_path_opt=False,
                            ce_segment_threshold=threshold,
                            is_mla=is_mla,
                            is_blockfirst=(layout_key == "bfirst"))
                        med = bench_one_dir(
                            tp, ids, cpu_kv_sb, cpu_ly_sb, cpu_bl_sb, cpu_tp_sb,
                            num_layers, is_h2d, num_gpus, args.iters, is_mla, mode,
                            transfer_num_cta=cta)
                        results[key]["baseline"] = med
                        baseline_cache[bk] = med
                        print("{:.3f} ms".format(med))
                        del tp
                    except Exception as e:
                        print("FAILED: {}".format(e))

                # Run auto (force_path=-1, choose_path decides)
                print("  auto [{}] ...".format(auto_path), end=" ", flush=True)
                try:
                    tp = make_tp_group(
                        cpu_kv.data_ptr(), all_gpu, num_gpus, gpu_layout,
                        num_layers, ce_path_opt=True,
                        ce_segment_threshold=threshold,
                        ce_force_path=-1,
                        is_mla=is_mla,
                        is_blockfirst=(layout_key == "bfirst"))
                    med = bench_one_dir(
                        tp, ids, cpu_kv_sb, cpu_ly_sb, cpu_bl_sb, cpu_tp_sb,
                        num_layers, is_h2d, num_gpus, args.iters, is_mla, mode,
                        transfer_num_cta=cta)
                    results[key]["auto"] = med
                    print("{:.3f} ms".format(med))
                    del tp
                except Exception as e:
                    print("FAILED: {}".format(e))

                # Run each viable strategy (skip auto-pick — redundant).
                # When --memcpy2d on, the two affected paths (SEGMENT_SCATTER,
                # GATHER_DIRECT) are also timed with cudaMemcpy2DAsync so the
                # benefit of FLEXKV_ENABLE_CE_MEMCPY2D can be measured head-to-head.
                # The off variant of the auto-picked path is already timed by
                # the 'auto' run, so we skip that one to avoid redundancy.
                viable = correct_paths_for(layout_key, is_mla, pattern, mode,
                                            is_h2d, threshold)
                for fp_id, fp_name in viable:
                    m2d_settings = [False]
                    if args.memcpy2d == "on" and fp_name in AFFECTED_PATHS:
                        m2d_settings = [False, True]
                    for m2d in m2d_settings:
                        if m2d is False and fp_name == auto_path:
                            continue
                        tag = " [memcpy2d=1]" if m2d else ""
                        label = fp_name + (" [2d]" if m2d else "")
                        print("  {}{} ...".format(fp_name, tag), end=" ", flush=True)
                        try:
                            tp = make_tp_group(
                                cpu_kv.data_ptr(), all_gpu, num_gpus, gpu_layout,
                                num_layers, ce_path_opt=True,
                                ce_segment_threshold=threshold,
                                ce_force_path=fp_id,
                                is_mla=is_mla,
                                is_blockfirst=(layout_key == "bfirst"),
                                ce_enable_memcpy2d=m2d)
                            med = bench_one_dir(
                                tp, ids, cpu_kv_sb, cpu_ly_sb, cpu_bl_sb, cpu_tp_sb,
                                num_layers, is_h2d, num_gpus, args.iters, is_mla, mode,
                                transfer_num_cta=cta)
                            results[key][label] = med
                            print("{:.3f} ms".format(med))
                            del tp
                        except Exception as e:
                            results[key][label] = None
                            print("FAILED: {}".format(e))

            del all_gpu, cpu_kv

    # -- Print results per size ------------------------------------------------
    for size_name in args.sizes:
        results = all_results[size_name]
        num_layers, num_blocks, head_dim = SIZES[size_name]
        print("\n" + "=" * 100)
        print("  Results for size={} ({}L / {}B / hd={})".format(
            size_name, num_layers, num_blocks, head_dim))
        print("=" * 100)

        # Build the list of (form_name, dir_name) rows actually run.
        run_rows = []
        for pattern, layout_key, is_mla, mode, dirs in PATH_FORMS:
            if pattern == "scattered" and num_blocks <= threshold:
                continue
            mla_tag = "mla" if is_mla else "mha"
            if is_mla:
                form_name = "{}/{}/{}/{}".format(pattern, layout_key, mla_tag, mode)
            else:
                form_name = "{}/{}/{}".format(pattern, layout_key, mla_tag)
            for is_h2d in dirs:
                viable = correct_paths_for(layout_key, is_mla, pattern, mode,
                                            is_h2d, threshold)
                run_rows.append((form_name, "H2D" if is_h2d else "D2H", viable))

        col_w = 9
        # Header
        hdr = "  {:>32s}  {:>4s}  {:>{w}s}".format("Form", "Dir", "base", w=col_w)
        for _, pname in STRATEGIES:
            hdr += "  {:>{w}s}".format(STR_ABBR[pname], w=col_w)
        if args.memcpy2d == "on":
            for _, pname in STRATEGIES:
                if pname in AFFECTED_PATHS:
                    hdr += "  {:>{w}s}".format(STR_ABBR_2D[pname], w=col_w)
        hdr += "  {:>{w}s}".format("auto", w=col_w)
        hdr += "  {:>16s}".format("recommended")
        hdr += " {:>2s}".format("=")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))

        auto_wins = 0
        auto_total = 0

        for form_name, dir_name, viable in run_rows:
            cfgs = results.get((form_name, dir_name), {})
            auto = cfgs.get("auto")
            baseline = cfgs.get("baseline")
            auto_path = cfgs.get("auto_path", "")
            viable_names = {pn for _, pn in viable}

            # Collect all strategy timings
            strategy_times = {}
            for _, pname in STRATEGIES:
                if pname == auto_path:
                    strategy_times[pname] = auto
                elif pname in viable_names:
                    strategy_times[pname] = cfgs.get(pname)
                else:
                    strategy_times[pname] = None

            # Find fastest
            all_vals = {k: v for k, v in strategy_times.items() if v is not None}
            if all_vals:
                recommended = min(all_vals, key=all_vals.get)
                fastest_val = all_vals[recommended]
            else:
                recommended = "-"
                fastest_val = None

            # Check if auto matches recommended
            auto_optimal = (auto is not None and fastest_val is not None
                            and auto == fastest_val)

            if auto is not None:
                auto_total += 1
                if auto_optimal:
                    auto_wins += 1

            # Format row
            def fmt_val(v, is_fastest):
                if v is None:
                    return "{:>{w}s}".format("-", w=col_w)
                star = "*" if is_fastest else " "
                return "{:>{w}.3f}{}".format(v, star, w=col_w - 1)

            line = "  {:>32s}  {:>4s}".format(form_name, dir_name)
            # baseline column
            if baseline is None:
                line += "  {:>{w}s}".format("-", w=col_w)
            else:
                line += "  {:>{w}.3f} ".format(baseline, w=col_w - 1)
            for _, pname in STRATEGIES:
                v = strategy_times.get(pname)
                is_fast = (v is not None and v == fastest_val)
                line += "  {}".format(fmt_val(v, is_fast))

            if args.memcpy2d == "on":
                for _, pname in STRATEGIES:
                    if pname in AFFECTED_PATHS:
                        v = cfgs.get(pname + " [2d]")
                        off_v = strategy_times.get(pname)
                        # '*' here means the memcpy2d=1 variant is FASTER than
                        # its off sibling for the same path (i.e. benefit).
                        is_fast = (v is not None and off_v is not None and v < off_v)
                        line += "  {}".format(fmt_val(v, is_fast))

            # auto column
            if auto is None:
                line += "  {:>{w}s}".format("-", w=col_w)
            else:
                star = "*" if auto_optimal else " "
                line += "  {:>{w}.3f}{}".format(auto, star, w=col_w - 1)

            # recommended + match
            match_sym = "=" if auto_optimal else "!" if auto is not None else "?"
            line += "  {:>16s}".format(recommended)
            line += " {:>2s}".format(match_sym)
            print(line)

        print("  " + "-" * (len(hdr) - 2))
        if auto_total:
            print("  choose_path auto pick optimal in {}/{} rows.".format(
                auto_wins, auto_total))
            if auto_wins == auto_total:
                print("  => choose_path is OPTIMAL for this size.")
            else:
                print("  => inspect rows marked '!' (auto not fastest).")
        print("  '*' = fastest strategy. '=' = auto matches recommended. '!' = auto NOT optimal.")
        if args.memcpy2d == "on":
            print_memcpy2d_benefit(results, run_rows)
        print("=" * 100)

    # -- Print recommendation summary across all sizes --------------------------
    print_recommendation_summary(all_results, args, threshold)


def print_memcpy2d_benefit(results, run_rows):
    """Focused block: for the two memcpy2d-affected paths (SEGMENT_SCATTER,
    GATHER_DIRECT), show off vs on (FLEXKV_ENABLE_CE_MEMCPY2D=1) timing and the
    speedup (off / on). Surfaces whether memcpy2d has any benefit per form.
    """
    print("\n" + "=" * 100)
    print("  memcpy2d benefit (FLEXKV_ENABLE_CE_MEMCPY2D=1) — affected paths only")
    print("=" * 100)
    print("  speedup = off / on  (>1: memcpy2d FASTER, <1: SLOWER)")
    affected_order = [p for p in ["SEGMENT_SCATTER", "GATHER_SCATTER", "GATHER_DIRECT"]
                      if p in AFFECTED_PATHS]
    hdr = "  {:>32s}  {:>4s}".format("Form", "Dir")
    for pname in affected_order:
        hdr += "  {:>11s}  {:>11s}  {:>7s}".format(
            STR_ABBR[pname] + "(off)", STR_ABBR_2D[pname], "speed")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    any_row = False
    for form_name, dir_name, viable in run_rows:
        cfgs = results.get((form_name, dir_name), {})
        auto_path = cfgs.get("auto_path", "")
        viable_names = {pn for _, pn in viable}
        if not any(p in viable_names for p in affected_order):
            continue
        any_row = True
        line = "  {:>32s}  {:>4s}".format(form_name, dir_name)
        for pname in affected_order:
            if pname not in viable_names:
                line += "  {:>11s}  {:>11s}  {:>7s}".format("-", "-", "-")
                continue
            off_v = cfgs.get(pname)
            if off_v is None and auto_path == pname:
                off_v = cfgs.get("auto")
            on_v = cfgs.get(pname + " [2d]")
            if off_v is not None and on_v is not None and on_v > 0:
                sp = off_v / on_v
                sp_str = "{:.2f}x".format(sp)
            else:
                sp_str = "-"
            line += "  {:>11.3f}  {:>11.3f}  {:>7s}".format(
                off_v if off_v is not None else 0.0,
                on_v if on_v is not None else 0.0, sp_str)
        print(line)
    if not any_row:
        print("  (no affected-path forms in this size)")
    print("  " + "-" * (len(hdr) - 2))
    print("  Non-NVIDIA platforms: memcpy2d=1 is slow/unsupported (keep off).")
    print("  NVIDIA: memcpy2d=1 is fast — enable it there.")
    print("=" * 100)


def print_recommendation_summary(all_results, args, threshold):
    """Print layout/mode recommendation per (size × model).

    When --memcpy2d on: outputs BOTH off and on recommendations (data already
    collected in a single run — auto=off, S_SCT2/G_SCT2/G_DIR2=on).
    When --memcpy2d off: only off recommendation.
    """
    mla_modes = ["rank0_only", "layer_parallel", "sharded"]
    show_on = (args.memcpy2d == "on")

    for size_name in args.sizes:
        results = all_results[size_name]
        num_layers, num_blocks, head_dim = SIZES[size_name]

        # For each (form, dir), compute best_off and best_on timing.
        # best_off = min of all off-variant viable path timings (auto or forced).
        # best_on  = min of all timings including [2d] variants (scattered guard:
        #            [2d] variants excluded for scattered since guard falls through).
        best_per_formdir = {}  # (form_name, dir_name) → (best_off, best_on)
        for pattern, layout_key, is_mla, mode, dirs in PATH_FORMS:
            if pattern == "scattered" and num_blocks <= threshold:
                continue
            mla_tag = "mla" if is_mla else "mha"
            if is_mla:
                form_name = "{}/{}/{}/{}".format(pattern, layout_key, mla_tag, mode)
            else:
                form_name = "{}/{}/{}".format(pattern, layout_key, mla_tag)
            viable = correct_paths_for(layout_key, is_mla, pattern, mode,
                                        True, threshold)  # H2D viable (superset)
            viable_names = {pn for _, pn in viable}
            for is_h2d in dirs:
                dir_name = "H2D" if is_h2d else "D2H"
                cfgs = results.get((form_name, dir_name), {})
                auto_path = cfgs.get("auto_path", "")
                # Collect off timings
                off_vals = []
                for pname in viable_names:
                    v = cfgs.get(pname)
                    if v is None and pname == auto_path:
                        v = cfgs.get("auto")
                    if v is not None:
                        off_vals.append(v)
                best_off = min(off_vals) if off_vals else None
                # Collect on timings (off + [2d] variants, with scattered guard)
                on_vals = list(off_vals)  # off variants still available
                if show_on:
                    is_scattered = (pattern == "scattered")
                    for pname in viable_names:
                        if pname in AFFECTED_PATHS and not is_scattered:
                            v = cfgs.get(pname + " [2d]")
                            if v is not None:
                                on_vals.append(v)
                best_on = min(on_vals) if on_vals else None
                best_per_formdir[(form_name, dir_name)] = (best_off, best_on)

        # Print recommendation tables
        for memcpy_label, use_on in [("memcpy2d=off", False), ("memcpy2d=on", True)]:
            if use_on and not show_on:
                continue  # skip on-table if --memcpy2d off

            print("\n" + "=" * 100)
            print("  Recommendation for size={} ({}L / {}B / hd={})  {}".format(
                size_name, num_layers, num_blocks, head_dim, memcpy_label))
            print("=" * 100)

            # Group by (is_mla, mode, layout) → list of best timings.
            # Sharded H2D == rank0_only H2D (C++ uses cpu_startoff=0 for both),
            # so use rank0_only's H2D data to fill in sharded's H2D for fair
            # comparison (all modes get 6 data points: 3 patterns × 2 dirs).
            groups = {}
            for pattern, layout_key, is_mla, mode, dirs in PATH_FORMS:
                if pattern == "scattered" and num_blocks <= threshold:
                    continue
                mla_tag = "mla" if is_mla else "mha"
                if is_mla:
                    form_name = "{}/{}/{}/{}".format(pattern, layout_key, mla_tag, mode)
                else:
                    form_name = "{}/{}/{}".format(pattern, layout_key, mla_tag)
                for is_h2d in dirs:
                    dir_name = "H2D" if is_h2d else "D2H"
                    vals = best_per_formdir.get((form_name, dir_name))
                    if vals is None and is_mla and mode == "sharded" and is_h2d:
                        # Sharded H2D: use rank0_only H2D (same C++ path)
                        h2d_form = "{}/{}/{}/{}".format(pattern, layout_key, mla_tag, "rank0_only")
                        vals = best_per_formdir.get((h2d_form, dir_name))
                    if vals is not None:
                        v = vals[1] if use_on else vals[0]
                        if v is not None:
                            key = (is_mla, mode, layout_key)
                            groups.setdefault(key, []).append(v)

            avgs = {}
            for key, times in groups.items():
                avgs[key] = sum(times) / len(times)

            # --- MLA: table of mode × layout ---
            print("\n  MLA — avg best timing (ms) per mode × layout:")
            print("  {:>16s}  {:>12s}  {:>12s}".format("mode", "lfirst", "bfirst"))
            print("  " + "-" * 44)
            mla_best = None
            mla_best_val = float('inf')
            for mode in mla_modes:
                for layout_key in ["lfirst", "bfirst"]:
                    v = avgs.get((True, mode, layout_key))
                    if v is not None and v < mla_best_val:
                        mla_best_val = v
                        mla_best = (mode, layout_key)
            for mode in mla_modes:
                row = "  {:>16s}".format(mode)
                for layout_key in ["lfirst", "bfirst"]:
                    v = avgs.get((True, mode, layout_key))
                    if v is not None:
                        star = " *" if (mode, layout_key) == mla_best else "  "
                        row += "  {:>9.3f}{}".format(v, star)
                    else:
                        row += "  {:>10s}  ".format("-")
                print(row)
            if mla_best:
                print("  => Recommended: MLA {} + {}".format(mla_best[0], mla_best[1]))

            # --- MHA: lfirst vs bfirst ---
            print("\n  MHA — avg best timing (ms) per layout:")
            mha_best = None
            mha_best_val = float('inf')
            for layout_key in ["lfirst", "bfirst"]:
                v = avgs.get((False, "rank0_only", layout_key))
                if v is not None and v < mha_best_val:
                    mha_best_val = v
                    mha_best = layout_key
            for layout_key in ["lfirst", "bfirst"]:
                v = avgs.get((False, "rank0_only", layout_key))
                if v is not None:
                    star = " *" if layout_key == mha_best else ""
                    print("    {:>8s}  avg={:.3f} ms{}".format(layout_key, v, star))
            if mha_best:
                print("  => Recommended: MHA {}".format(mha_best))

    print("\n" + "=" * 100)
    print("  '*' = best (lowest average across all continuity × direction).")
    print("  best = fastest viable path for each (form, dir), including [2d] variants when memcpy2d=on.")
    print("=" * 100)


# -- Main --------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Microbenchmark CE transfer strategy auto-selection "
                    "(5 strategies head-to-head + recommended)")
    parser.add_argument("--num-gpus", type=int, default=0,
                        help="Number of GPUs (0 = all available, default: 0)")
    parser.add_argument("--iters", type=int, default=20,
                        help="Timing iterations per strategy (default: 20)")
    parser.add_argument("--sizes", nargs="+", default=list(SIZES.keys()),
                        choices=list(SIZES.keys()),
                        help="Data sizes to test (default: all)")
    parser.add_argument("--memcpy2d", choices=["off", "on"], default="on",
                        help="When 'on' (default), also time SEGMENT_SCATTER (path 2), "
                             "GATHER_SCATTER (path 3), and GATHER_DIRECT (path 4) "
                             "with cudaMemcpy2DAsync (FLEXKV_ENABLE_CE_MEMCPY2D=1) "
                             "and print a benefit block. "
                             "On by default (NVIDIA fast); set 'off' on non-NVIDIA "
                             "platforms where cudaMemcpy2DAsync is slow/unsupported.")
    args = parser.parse_args()

    num_gpus = NUM_GPUS if args.num_gpus <= 0 else min(args.num_gpus, NUM_GPUS)
    if num_gpus < 2:
        print("ERROR: need at least 2 GPUs, found {}".format(NUM_GPUS))
        sys.exit(1)
    args.num_gpus = num_gpus

    run_strategy_compare(args)


if __name__ == "__main__":
    main()
