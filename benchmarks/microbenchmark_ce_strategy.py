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
                       kv_dim, num_kv_heads, num_gpus):
    """GPU (LAYERFIRST, per-rank heads) and CPU (cpu_layout_type) layouts.

    MLA (kv_dim=1, num_kv_heads=1): head=1 (all ranks identical).
    MHA (kv_dim=2, num_kv_heads>1): head=num_gpus (each rank gets 1 head).
    """
    num_head = 1 if num_kv_heads == 1 else num_gpus
    heads_per_rank = 1
    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=1, num_head=heads_per_rank,
        head_size=head_dim, kv_dim=kv_dim)
    cpu_layout = KVCacheLayout(
        type=cpu_layout_type,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=1, num_head=num_head,
        head_size=head_dim, kv_dim=kv_dim)
    return gpu_layout, cpu_layout


def cpu_strides_for_strategy(cpu_layout, num_layers, num_blocks, head_dim,
                             kv_dim, num_kv_heads, mode, num_gpus):
    """Return (cpu_kv_sb, cpu_layer_sb, cpu_block_sb, cpu_tp_sb, total_blocks)."""
    total = num_blocks
    num_head = 1 if num_kv_heads == 1 else num_gpus

    layout_for_kv_stride = KVCacheLayout(
        type=cpu_layout.type,
        num_layer=num_layers, num_block=total,
        tokens_per_block=1, num_head=num_head,
        head_size=head_dim, kv_dim=kv_dim)

    if num_kv_heads > 1 and cpu_layout.type == KVCacheLayoutType.BLOCKFIRST:
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
                          kv_dim, num_kv_heads, num_gpus):
    num_head = 1 if num_kv_heads == 1 else num_gpus
    layout = KVCacheLayout(
        type=cpu_layout.type,
        num_layer=num_layers, num_block=total_blocks,
        tokens_per_block=1, num_head=num_head,
        head_size=head_dim, kv_dim=kv_dim)
    return torch.empty(tuple(layout.kv_shape), dtype=DTYPE, pin_memory=True)


def make_tp_group(cpu_ptr, all_gpu, num_gpus, gpu_layout, num_layers,
                  ce_path_opt=True,
                  ce_segment_threshold=8, ce_force_path=-1,
                  kv_dim=2, is_blockfirst=False,
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
        kv_dim=kv_dim,
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
                  num_layers, is_h2d, num_gpus, iters, kv_dim, num_kv_heads,
                  mode, transfer_num_cta=16):
    """Time ONE direction (H2D or D2H) over `iters`, return median wall-clock ms."""

    def do_transfer():
        tp.tp_group_transfer(
            gpu_block_id_tensor=ids, cpu_block_id_tensor=ids,
            cpu_kv_stride_in_bytes=cpu_kv_sb, cpu_layer_stride_in_bytes=cpu_ly_sb,
            cpu_block_stride_in_bytes=cpu_bl_sb, cpu_tp_stride_in_bytes=cpu_tp_sb,
            transfer_num_cta=transfer_num_cta, is_host_to_device=is_h2d,
            use_ce_transfer=True, layer_id=0, layer_granularity=num_layers,
            kv_dim=kv_dim, num_kv_heads=num_kv_heads,
            kv_shared_across_ranks_mode=mode)

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
def correct_paths_for(layout_key, kv_dim, pattern, mode, is_h2d, threshold):
    """Return the (path_id, name) list of strategies that are DATA-CORRECT for
    a form. Mirrors the hard constraints each CEPath imposes (see csrc/ce_transfer.h):

      CONTIG_DIRECT(0):   cpu_phys_contig & gpu_phys_contig & contiguous (1 seg)
      SEGMENT_DIRECT(1):  cpu_phys_contig & gpu_phys_contig & <=threshold segs
      SEGMENT_SCATTER(2): gpu_phys_contig (non-sharded); CPU scatter handles any layout
      GATHER_SCATTER(3):  ALWAYS correct — GPU gather + CPU scatter, any layout / sharded
      GATHER_DIRECT(4):   BF (!cpu_phys_contig) & gpu_phys_contig (non-sharded)

    cpu_phys_contig = (layout==lfirst) and (kv_dim == 1).
    gpu_phys_contig = !(mode==sharded and direction==D2H); sharded only breaks D2H.
    Computed rather than hand-listed so it can never drift from the C++ rules.
    """
    cpu_phys_contig = (layout_key == "lfirst") and (kv_dim == 1)
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
# Third element is kv_dim: 1=MLA (K/V combined, single shared head),
#                          2=MHA  (K/V separate, per-rank heads).
PATH_FORMS = [
    # --- MLA (kv_dim=1) + rank0_only ---
    ("contiguous", "lfirst", 1,  "rank0_only",     [True, False]),
    ("contiguous", "bfirst", 1,  "rank0_only",     [True, False]),
    ("few_seg",    "lfirst", 1,  "rank0_only",     [True, False]),
    ("few_seg",    "bfirst", 1,  "rank0_only",     [True, False]),
    ("scattered",  "lfirst", 1,  "rank0_only",     [True, False]),
    ("scattered",  "bfirst", 1,  "rank0_only",     [True, False]),
    # --- MLA (kv_dim=1) + layer_parallel ---
    ("contiguous", "lfirst", 1,  "layer_parallel", [True, False]),
    ("contiguous", "bfirst", 1,  "layer_parallel", [True, False]),
    ("few_seg",    "lfirst", 1,  "layer_parallel", [True, False]),
    ("few_seg",    "bfirst", 1,  "layer_parallel", [True, False]),
    ("scattered",  "lfirst", 1,  "layer_parallel", [True, False]),
    ("scattered",  "bfirst", 1,  "layer_parallel", [True, False]),
    # --- MLA (kv_dim=1) + sharded (D2H only) ---
    ("contiguous", "lfirst", 1,  "sharded",        [False]),
    ("contiguous", "bfirst", 1,  "sharded",        [False]),
    ("few_seg",    "lfirst", 1,  "sharded",        [False]),
    ("few_seg",    "bfirst", 1,  "sharded",        [False]),
    ("scattered",  "lfirst", 1,  "sharded",        [False]),
    ("scattered",  "bfirst", 1,  "sharded",        [False]),
    # --- MHA (kv_dim=2, H2D + D2H, mode is don't-care) ---
    ("contiguous", "lfirst", 2, "rank0_only",     [True, False]),
    ("contiguous", "bfirst", 2, "rank0_only",     [True, False]),
    ("few_seg",    "lfirst", 2, "rank0_only",     [True, False]),
    ("few_seg",    "bfirst", 2, "rank0_only",     [True, False]),
    ("scattered",  "lfirst", 2, "rank0_only",     [True, False]),
    ("scattered",  "bfirst", 2, "rank0_only",     [True, False]),
]


# CEPath enum mapping for predict_ce_path output.
CE_PATH_NAMES = {
    -1: "PER_BLOCK",
    0: "CONTIG_DIRECT",
    1: "SEGMENT_DIRECT",
    2: "SEGMENT_SCATTER",
    3: "GATHER_SCATTER",
    4: "GATHER_DIRECT",
}


_SIZEOF_INT64 = 8


def _analyze_ce_transfer(gpu_block_ids, cpu_block_ids, num_blocks,
                         cpu_block_stride_in_bytes, chunk_size_in_bytes,
                         gpu_block_stride_in_bytes):
    gpu_log_contig = True
    cpu_log_contig = True
    cpu_phys_contig = (cpu_block_stride_in_bytes == chunk_size_in_bytes)
    gpu_phys_contig = (gpu_block_stride_in_bytes == 0 or
                       gpu_block_stride_in_bytes == chunk_size_in_bytes)
    num_segments = 1 if num_blocks > 0 else 0
    seg_start = 0
    for k in range(1, num_blocks):
        src_step = (gpu_block_ids[k] == gpu_block_ids[k - 1] + 1)
        dst_step = (cpu_block_ids[k] == cpu_block_ids[k - 1] + 1)
        if not src_step:
            gpu_log_contig = False
        if not dst_step:
            cpu_log_contig = False
        if not src_step or not dst_step:
            num_segments += 1
            seg_start = k
    return {
        "gpu_log_contig": gpu_log_contig,
        "cpu_log_contig": cpu_log_contig,
        "cpu_phys_contig": cpu_phys_contig,
        "gpu_phys_contig": gpu_phys_contig,
        "num_segments": num_segments,
    }


def _choose_ce_path(ce, is_blockfirst, kv_dim, segment_threshold,
                    chunk_size_in_bytes, is_host_to_device, is_full_block):
    # BF + !cpu_phys_contig + GPU contig: MLA D2H path refinement.
    if is_blockfirst and not ce["cpu_phys_contig"] and ce["gpu_phys_contig"]:
        if not is_host_to_device and kv_dim == 1:
            if ce["num_segments"] > segment_threshold:
                return (2 if (chunk_size_in_bytes > 0 and
                              chunk_size_in_bytes % _SIZEOF_INT64 != 0)
                        else 3)  # SEGMENT_SCATTER : GATHER_SCATTER
            if not is_full_block:
                return 2  # SEGMENT_SCATTER
        return 4  # GATHER_DIRECT
    # CONTIG_DIRECT: both sides contig -> one big memcpy.
    if (ce["gpu_log_contig"] and ce["cpu_log_contig"] and
            ce["cpu_phys_contig"] and ce["gpu_phys_contig"]):
        return 0  # CONTIG_DIRECT
    # Sharded D2H -> GATHER_SCATTER (SEGMENT_SCATTER misplaces shards).
    if not ce["gpu_phys_contig"]:
        return 3  # GATHER_SCATTER
    # LAYERFIRST: pick by segment count.
    if ce["num_segments"] <= segment_threshold:
        return 1 if ce["cpu_phys_contig"] else 2  # SEGMENT_DIRECT : SEGMENT_SCATTER
    if chunk_size_in_bytes > 0 and chunk_size_in_bytes % _SIZEOF_INT64 != 0:
        return 2  # SEGMENT_SCATTER
    return 3  # GATHER_SCATTER


def predict_ce_path(gpu_block_id_tensor, cpu_block_id_tensor,
                    cpu_block_stride_in_bytes, chunk_size_in_bytes,
                    num_layers, is_host_to_device, kv_dim, is_blockfirst,
                    ce_segment_threshold=8, gpu_block_stride_in_bytes=0):
    """Mirror of C++ predict_ce_path_binding — returns the CEPath int."""
    gpu_ids = gpu_block_id_tensor.tolist()
    cpu_ids = cpu_block_id_tensor.tolist()
    num_blocks = len(gpu_ids)
    ce = _analyze_ce_transfer(gpu_ids, cpu_ids, num_blocks,
                              cpu_block_stride_in_bytes, chunk_size_in_bytes,
                              gpu_block_stride_in_bytes)
    is_full_block = (num_layers * kv_dim * chunk_size_in_bytes ==
                     cpu_block_stride_in_bytes)
    return _choose_ce_path(ce, is_blockfirst, kv_dim, ce_segment_threshold,
                           chunk_size_in_bytes, is_host_to_device, is_full_block)


# -- Main benchmark -----------------------------------------------------------

def run_strategy_compare(args):
    """Run all viable strategies per form, select fastest as recommendation."""
    num_gpus = args.num_gpus
    threshold = 8
    cta = 16
    # PER_BLOCK baseline is pattern-independent. Cache by (size, layout, mode,
    # dir) so we only time it once per unique combo.
    baseline_cache = {}
    # MLA H2D is mode-independent — cache first mode's timings for reuse.
    h2d_mla_cache = {}

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

        for pattern, layout_key, kv_dim, mode, dirs in PATH_FORMS:
            mla_tag = "mla" if kv_dim == 1 else "mha"
            # MHA mode is don't-care — don't show it in the form name.
            if kv_dim == 1:
                form_name = "{}/{}/{}/{}".format(pattern, layout_key, mla_tag, mode)
            else:
                form_name = "{}/{}/{}".format(pattern, layout_key, mla_tag)
            if pattern == "scattered" and num_blocks <= threshold:
                print("  SKIP {} (num_blocks={} <= threshold={})".format(
                    form_name, num_blocks, threshold))
                continue
            num_kv_heads = 1 if kv_dim == 1 else num_gpus
            cpu_layout_type = STRAT_LAYOUTS[layout_key]

            gpu_layout, cpu_layout = make_layouts_strat(
                num_layers, num_blocks, head_dim, cpu_layout_type, kv_dim, num_kv_heads, num_gpus)
            cpu_kv_sb, cpu_ly_sb, cpu_bl_sb, cpu_tp_sb, total_blocks = \
                cpu_strides_for_strategy(cpu_layout, num_layers, num_blocks,
                                         head_dim, kv_dim, num_kv_heads, mode, num_gpus)

            all_gpu = [make_gpu_tensors_strat(num_layers, kv_dim, num_blocks,
                                              heads_per_rank, head_dim, g)
                       for g in range(num_gpus)]
            cpu_kv = make_cpu_tensor_strat(cpu_layout, num_layers, total_blocks,
                                           head_dim, kv_dim, num_kv_heads, num_gpus)
            ids = make_block_id_pattern(pattern, num_blocks)

            for is_h2d in dirs:
                dir_name = "H2D" if is_h2d else "D2H"
                # For H2D, MLA mode is mode-independent — label it just "mla".
                if is_h2d and kv_dim == 1:
                    row_name = "{}/{}/{}".format(pattern, layout_key, mla_tag)
                else:
                    row_name = form_name
                key = (row_name, dir_name)
                if kv_dim == 1:
                    if is_h2d:
                        form_display = "{} | {} | mla".format(pattern, layout_key)
                    else:
                        form_display = "{} | {} | mla-{}".format(pattern, layout_key, mode)
                else:
                    form_display = "{} | {} | mha".format(pattern, layout_key)

                # MLA H2D is mode-independent — skip duplicate modes.
                if is_h2d and kv_dim == 1:
                    if h2d_mla_cache.get((pattern, layout_key)) is not None:
                        continue

                # Query C++ choose_path directly via predict_ce_path.
                chunk_size = gpu_layout.get_chunk_size() * ES
                # sharded D2H: strided GPU view breaks phys contiguity
                gpu_bl_sb = (chunk_size * num_gpus
                             if (mode == "sharded" and not is_h2d) else 0)
                auto_path_id = predict_ce_path(
                    gpu_block_id_tensor=ids, cpu_block_id_tensor=ids,
                    cpu_block_stride_in_bytes=cpu_bl_sb,
                    chunk_size_in_bytes=chunk_size,
                    num_layers=num_layers, is_host_to_device=is_h2d,
                    kv_dim=kv_dim,
                    is_blockfirst=(layout_key == "bfirst"),
                    ce_segment_threshold=threshold,
                    gpu_block_stride_in_bytes=gpu_bl_sb)
                auto_path = CE_PATH_NAMES.get(auto_path_id, "PER_BLOCK")
                results[key]["auto_path"] = auto_path
                print("\n-- {} | {} | auto={} --".format(
                    form_display, dir_name, auto_path))

                # Viable force-run paths for this form/dir (also used by warmup).
                viable = correct_paths_for(layout_key, kv_dim, pattern, mode,
                                            is_h2d, threshold)

                # Warmup each path to absorb first-run overhead.
                warm_paths = [(-1, auto_path)]
                _seen = {auto_path}
                for fp_id, fp_name in viable:
                    if fp_name not in _seen:
                        _seen.add(fp_name)
                        warm_paths.append((fp_id, fp_name))
                for wp_id, wp_name in warm_paths:
                    # Match memcpy2d setting used in timed runs.
                    wp_m2d = [(args.memcpy2d == "on")] if wp_id == -1 else [False]
                    if wp_id != -1 and args.memcpy2d == "on" and wp_name in AFFECTED_PATHS:
                        wp_m2d = [False, True]
                    for wp_m in wp_m2d:
                        try:
                            wp_tp = make_tp_group(
                                cpu_kv.data_ptr(), all_gpu, num_gpus, gpu_layout,
                                num_layers, ce_path_opt=True,
                                ce_segment_threshold=threshold,
                                ce_force_path=wp_id,
                                kv_dim=kv_dim,
                                is_blockfirst=(layout_key == "bfirst"),
                                ce_enable_memcpy2d=wp_m)
                            bench_one_dir(
                                wp_tp, ids, cpu_kv_sb, cpu_ly_sb, cpu_bl_sb, cpu_tp_sb,
                                num_layers, is_h2d, num_gpus,
                                max(1, args.iters // 5), kv_dim, num_kv_heads, mode,
                                transfer_num_cta=cta)
                            del wp_tp
                        except Exception as e:
                            # Timed runs will surface real failures
                            print("  warmup [{}] skipped: {}".format(wp_name, e))

                # Run baseline (PER_BLOCK, path_opt=false) — cached
                bk = (size_name, layout_key, mode, dir_name)
                cached_bs = baseline_cache.get(bk)
                if cached_bs is not None:
                    results[key]["baseline"] = cached_bs
                    print("  baseline {:.3f} ms".format(cached_bs))
                else:
                    print("  baseline ...", end=" ", flush=True)
                    try:
                        tp = make_tp_group(
                            cpu_kv.data_ptr(), all_gpu, num_gpus, gpu_layout,
                            num_layers, ce_path_opt=False,
                            ce_segment_threshold=threshold,
                            kv_dim=kv_dim,
                            is_blockfirst=(layout_key == "bfirst"))
                        med = bench_one_dir(
                            tp, ids, cpu_kv_sb, cpu_ly_sb, cpu_bl_sb, cpu_tp_sb,
                            num_layers, is_h2d, num_gpus, args.iters, kv_dim, num_kv_heads, mode,
                            transfer_num_cta=cta)
                        results[key]["baseline"] = med
                        baseline_cache[bk] = med
                        print("{:.3f} ms".format(med))
                        del tp
                    except Exception as e:
                        print("FAILED: {}".format(e))

                # Time every viable strategy (auto measured separately above).
                for fp_id, fp_name in viable:
                    m2d_settings = [False]
                    if args.memcpy2d == "on" and fp_name in AFFECTED_PATHS:
                        m2d_settings = [False, True]
                    for m2d in m2d_settings:
                        tag = " [memcpy2d=1]" if m2d else ""
                        label = fp_name + (" [2d]" if m2d else "")
                        print("  {}{} ...".format(fp_name, tag), end=" ", flush=True)
                        try:
                            tp = make_tp_group(
                                cpu_kv.data_ptr(), all_gpu, num_gpus, gpu_layout,
                                num_layers, ce_path_opt=True,
                                ce_segment_threshold=threshold,
                                ce_force_path=fp_id,
                                kv_dim=kv_dim,
                                is_blockfirst=(layout_key == "bfirst"),
                                ce_enable_memcpy2d=m2d)
                            med = bench_one_dir(
                                tp, ids, cpu_kv_sb, cpu_ly_sb, cpu_bl_sb, cpu_tp_sb,
                                num_layers, is_h2d, num_gpus, args.iters, kv_dim, num_kv_heads, mode,
                                transfer_num_cta=cta)
                            results[key][label] = med
                            print("{:.3f} ms".format(med))
                            del tp
                        except Exception as e:
                            results[key][label] = None
                            print("FAILED: {}".format(e))

                # Run auto after force-runs to reuse warmup state.
                print("  auto [{}] ...".format(auto_path), end=" ", flush=True)
                try:
                    tp = make_tp_group(
                        cpu_kv.data_ptr(), all_gpu, num_gpus, gpu_layout,
                        num_layers, ce_path_opt=True,
                        ce_segment_threshold=threshold,
                        ce_force_path=-1,
                        kv_dim=kv_dim,
                        is_blockfirst=(layout_key == "bfirst"),
                        ce_enable_memcpy2d=(args.memcpy2d == "on"))
                    med = bench_one_dir(
                        tp, ids, cpu_kv_sb, cpu_ly_sb, cpu_bl_sb, cpu_tp_sb,
                        num_layers, is_h2d, num_gpus, args.iters, kv_dim, num_kv_heads, mode,
                        transfer_num_cta=cta)
                    results[key]["auto"] = med
                    print("{:.3f} ms".format(med))
                    del tp
                except Exception as e:
                    print("FAILED: {}".format(e))

                # Cache H2D results for MLA reuse (mode-independent, see above).
                if is_h2d and kv_dim == 1:
                    h2d_mla_cache[(pattern, layout_key)] = dict(results[key])

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
        for pattern, layout_key, kv_dim, mode, dirs in PATH_FORMS:
            if pattern == "scattered" and num_blocks <= threshold:
                continue
            mla_tag = "mla" if kv_dim == 1 else "mha"
            if kv_dim == 1:
                form_name = "{}/{}/{}/{}".format(pattern, layout_key, mla_tag, mode)
            else:
                form_name = "{}/{}/{}".format(pattern, layout_key, mla_tag)
            for is_h2d in dirs:
                # MLA H2D is mode-independent — show only rank0_only.
                if is_h2d and kv_dim == 1 and mode != "rank0_only":
                    continue
                if is_h2d and kv_dim == 1:
                    form_name = "{}/{}/{}".format(pattern, layout_key, mla_tag)
                elif kv_dim == 1:
                    form_name = "{}/{}/{}/{}".format(pattern, layout_key, mla_tag, mode)
                else:
                    form_name = "{}/{}/{}".format(pattern, layout_key, mla_tag)
                viable = correct_paths_for(layout_key, kv_dim, pattern, mode,
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
        overhead_rows = []

        for form_name, dir_name, viable in run_rows:
            cfgs = results.get((form_name, dir_name), {})
            auto = cfgs.get("auto")
            baseline = cfgs.get("baseline")
            auto_path = cfgs.get("auto_path", "")
            viable_names = {pn for _, pn in viable}

            # Display: off-variant timings; [2d] columns show ON variant.
            strategy_times = {}
            for _, pname in STRATEGIES:
                strategy_times[pname] = cfgs.get(pname) if pname in viable_names else None

            # Comparison set: match auto's memcpy2d setting for fair '=' check.
            use_on = (args.memcpy2d == "on")
            cmp_times = {}
            for _, pname in STRATEGIES:
                if pname not in viable_names:
                    cmp_times[pname] = None
                elif use_on and pname in AFFECTED_PATHS:
                    on_v = cfgs.get(pname + " [2d]")
                    cmp_times[pname] = on_v if on_v is not None else cfgs.get(pname)
                else:
                    cmp_times[pname] = cfgs.get(pname)

            # Find fastest (at auto's memcpy2d setting)
            all_vals = {k: v for k, v in cmp_times.items() if v is not None}
            if all_vals:
                recommended = min(all_vals, key=all_vals.get)
                fastest_val = all_vals[recommended]
            else:
                recommended = "-"
                fastest_val = None

            # Correctness: did choose_path pick the optimal path?
            path_optimal = (auto_path != "" and recommended != "-"
                            and auto_path == recommended)

            # Timing sanity: auto within 8% of fastest (informational).
            if auto is not None and fastest_val is not None:
                tol = 0.08 * fastest_val
                time_ok = auto <= fastest_val + tol
            else:
                time_ok = False

            if auto is not None:
                auto_total += 1
                if path_optimal:
                    auto_wins += 1
                    if fastest_val:
                        overhead = (auto - fastest_val) / fastest_val
                        if overhead > 0.15:
                            overhead_rows.append(
                                (form_name, dir_name, auto_path,
                                 auto, fastest_val, overhead))

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
            # Main columns: '*' marks the fastest variant.
            for _, pname in STRATEGIES:
                v = strategy_times.get(pname)
                is_fast = (pname == recommended and v is not None
                           and fastest_val is not None and v == fastest_val)
                line += "  {}".format(fmt_val(v, is_fast))

            if args.memcpy2d == "on":
                for _, pname in STRATEGIES:
                    if pname in AFFECTED_PATHS:
                        v = cfgs.get(pname + " [2d]")
                        # '*' = fastest or benefits vs off variant.
                        is_fast = ((pname == recommended and v is not None
                                    and fastest_val is not None and v == fastest_val)
                                   or (v is not None and strategy_times.get(pname) is not None
                                       and v < strategy_times.get(pname)))
                        line += "  {}".format(fmt_val(v, is_fast))

            # auto column
            if auto is None:
                line += "  {:>{w}s}".format("-", w=col_w)
            else:
                star = "*" if time_ok else " "
                line += "  {:>{w}.3f}{}".format(auto, star, w=col_w - 1)

            # recommended + match
            match_sym = "=" if path_optimal else "!" if auto is not None else "?"
            line += "  {:>16s}".format(recommended)
            line += " {:>2s}".format(match_sym)
            print(line)

        print("  " + "-" * (len(hdr) - 2))
        if auto_total:
            print("  choose_path picked optimal path in {}/{} rows.".format(
                auto_wins, auto_total))
            if auto_wins == auto_total:
                print("  => choose_path is OPTIMAL for this size.")
            else:
                print("  => inspect rows marked '!' (auto picked a SUBOPTIMAL path).")
        print("  '*' (path cols) = fastest variant at current memcpy2d setting.")
        print("  '*' (auto col)  = auto time within 8% of fastest (timing sanity).")
        print("  '=' = choose_path picked the optimal path; '!' = chose a SUBOPTIMAL path (real bug).")
        if overhead_rows:
            print("  note: {} path-optimal row(s) show >15% auto-time overhead vs the "
                  "isolated fastest (full-API harness overhead / run variance) — "
                  "NOT choose_path bugs:".format(len(overhead_rows)))
            for fn, dn, ap, au, fv, oh in overhead_rows:
                print("    - {} {} [{}]: auto={:.3f} vs fastest={:.3f} ({:.0%} over)".format(
                    fn, dn, ap, au, fv, oh))
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
        for pattern, layout_key, kv_dim, mode, dirs in PATH_FORMS:
            if pattern == "scattered" and num_blocks <= threshold:
                continue
            mla_tag = "mla" if kv_dim == 1 else "mha"
            if kv_dim == 1:
                form_name = "{}/{}/{}/{}".format(pattern, layout_key, mla_tag, mode)
            else:
                form_name = "{}/{}/{}".format(pattern, layout_key, mla_tag)
            viable = correct_paths_for(layout_key, kv_dim, pattern, mode,
                                        True, threshold)  # H2D viable (superset)
            viable_names = {pn for _, pn in viable}
            for is_h2d in dirs:
                dir_name = "H2D" if is_h2d else "D2H"
                # MLA H2D is mode-independent: label it just "mla", and reuse
                # rank0_only's H2D best so every mode still contributes its
                # full 6 data points (3 patterns × 2 dirs) to the average.
                if is_h2d and kv_dim == 1:
                    h2d_name = "{}/{}/{}".format(pattern, layout_key, mla_tag)
                else:
                    h2d_name = form_name
                if is_h2d and kv_dim == 1 and mode != "rank0_only":
                    best_per_formdir[(h2d_name, dir_name)] = best_per_formdir.get(
                        (h2d_name, dir_name), (None, None))
                    continue
                cfgs = results.get((h2d_name, dir_name), {})
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

            # Group by (kv_dim, mode, layout). Sharded H2D == rank0_only H2D.
            groups = {}
            for pattern, layout_key, kv_dim, mode, dirs in PATH_FORMS:
                if pattern == "scattered" and num_blocks <= threshold:
                    continue
                mla_tag = "mla" if kv_dim == 1 else "mha"
                if kv_dim == 1:
                    form_name = "{}/{}/{}/{}".format(pattern, layout_key, mla_tag, mode)
                else:
                    form_name = "{}/{}/{}".format(pattern, layout_key, mla_tag)
                for is_h2d in dirs:
                    dir_name = "H2D" if is_h2d else "D2H"
                    # MLA H2D stored under "mla" (no mode suffix).
                    if is_h2d and kv_dim == 1:
                        lookup = "{}/{}/{}".format(pattern, layout_key, mla_tag)
                    else:
                        lookup = form_name
                    vals = best_per_formdir.get((lookup, dir_name))
                    if vals is None and kv_dim == 1 and mode == "sharded" and is_h2d:
                        # Sharded H2D: same as rank0_only H2D (mode-independent)
                        vals = best_per_formdir.get((lookup, dir_name))
                    if vals is not None:
                        v = vals[1] if use_on else vals[0]
                        if v is not None:
                            key = (kv_dim, mode, layout_key)
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
                    v = avgs.get((1, mode, layout_key))
                    if v is not None and v < mla_best_val:
                        mla_best_val = v
                        mla_best = (mode, layout_key)
            for mode in mla_modes:
                row = "  {:>16s}".format(mode)
                for layout_key in ["lfirst", "bfirst"]:
                    v = avgs.get((1, mode, layout_key))
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
                v = avgs.get((2, "rank0_only", layout_key))
                if v is not None and v < mha_best_val:
                    mha_best_val = v
                    mha_best = layout_key
            for layout_key in ["lfirst", "bfirst"]:
                v = avgs.get((2, "rank0_only", layout_key))
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
