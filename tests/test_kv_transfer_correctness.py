"""
Comprehensive KV transfer correctness tests for FlexKV.

Tests D2H (GPU->CPU) and H2D (CPU->GPU) data integrity across:
  - Layout:   LAYERFIRST, BLOCKFIRST (CPU side)
  - Model:    MLA (kv_heads=1, kv_dim=1), MHA (kv_heads>1, kv_dim=2)
  - Mode:     sharded, all_write, rank0_only (MLA only)
  - Engine:   CUDA kernel, CE (cudaMemcpyAsync)
  - Direction: D2H, H2D, Round-trip

Uses KVCacheLayout for stride computation (same as production code in worker.py).
GPU is always LAYERFIRST; CPU can be LAYERFIRST or BLOCKFIRST.

Run:
    pytest tests/test_kv_transfer_correctness.py -v
    pytest tests/test_kv_transfer_correctness.py -v -k "mla and sharded"
    FLEXKV_TEST_SKIP_KERNEL=1 pytest tests/test_kv_transfer_correctness.py -v   # non-NVIDIA: skip CUDA kernel engine
"""

import os
import pytest
import torch
import gc

from flexkv.c_ext import TPTransferThreadGroup, LayerwiseTransferGroup
from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType


# Skip conditions

NUM_GPUS = min(4, torch.cuda.device_count()) if torch.cuda.is_available() else 0

pytestmark = pytest.mark.skipif(
    NUM_GPUS < 2,
    reason=f"Need at least 2 GPUs, found {NUM_GPUS}"
)


@pytest.fixture(autouse=True)
def _cleanup_gpu_mem():
    """Force GC + empty_cache after each test to prevent GPU memory
    fragmentation from accumulated PyTorch tensors and C++ thread_local
    cached buffers (get_cached_device_buffer / get_cached_host_buffer)
    that are only freed when their owning thread exits."""
    yield
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _probe_engine(use_ce):
    """Probe whether a single tiny transfer succeeds with the given engine.

    The custom CUDA kernel requires chunk_size divisible by 16 bytes (float4).
    CE (cudaMemcpyAsync) works for any size.  We run a throwaway transfer to
    detect at session start whether each engine is usable; results are cached.
    """
    cache_key = f"__probe_{use_ce}"
    cached = globals().get(cache_key)
    if cached is not None:
        return cached
    try:
        layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=1, num_block=1, tokens_per_block=1,
            num_head=1, head_size=16, is_mla=True)
        g = torch.zeros((1, 1, 1, 1, 1, 16), dtype=torch.float16, device="cuda:0")
        c = torch.zeros(tuple(layout.kv_shape), dtype=torch.float16, pin_memory=True)
        ids = torch.arange(1, dtype=torch.int64).pin_memory()
        tp = TPTransferThreadGroup(
            num_gpus=1, gpu_block_ptrs_flat=[g[0].data_ptr()],
            num_tensors_per_gpu=1, cpu_blocks_ptr=c.data_ptr(),
            num_layers=1,
            gpu_kv_strides_in_bytes=[layout.get_kv_stride() * 2],
            gpu_block_strides_in_bytes=[layout.get_block_stride() * 2],
            gpu_layer_strides_in_bytes=[layout.get_layer_stride() * 2],
            gpu_chunk_sizes_in_bytes=[layout.get_chunk_size() * 2],
            gpu_device_ids=[0], enable_nvcomp=False)
        tp.tp_group_transfer(
            gpu_block_id_tensor=ids, cpu_block_id_tensor=ids,
            cpu_kv_stride_in_bytes=layout.get_kv_stride() * 2,
            cpu_layer_stride_in_bytes=layout.get_layer_stride() * 2,
            cpu_block_stride_in_bytes=layout.get_block_stride() * 2,
            cpu_tp_stride_in_bytes=layout.get_block_stride() * 2,
            transfer_num_cta=4, is_host_to_device=False, use_ce_transfer=use_ce,
            layer_id=0, layer_granularity=1, is_mla=True,
            mla_d2h_mode="sharded")
        torch.cuda.synchronize()
        del tp
        globals()[cache_key] = True
        return True
    except Exception as _e:
        # Surface the real probe failure (was swallowed -> silent mass-skip).
        import traceback as _tb
        print("\n[engine probe {} FAILED] {}: {}".format(
            use_ce, type(_e).__name__, _e))
        _tb.print_exc()
        globals()[cache_key] = False
        return False


def skip_if_engine_unsupported(use_ce):
    """Skip test if the engine probe failed (kernel needs float4 alignment, CE
    needs CUDA runtime, etc.), or if CUDA-kernel tests are disabled via the
    FLEXKV_TEST_SKIP_KERNEL env var (set on non-NVIDIA platforms where the
    custom kernel cannot build/run)."""
    if not use_ce and os.environ.get("FLEXKV_TEST_SKIP_KERNEL"):
        pytest.skip("CUDA kernel test disabled via FLEXKV_TEST_SKIP_KERNEL")
    if not _probe_engine(use_ce):
        kind = "CE (cudaMemcpyAsync)" if use_ce else "CUDA kernel"
        pytest.skip(f"{kind} engine not available on this platform")


# Test configurations

DTYPE = torch.float16
ES = DTYPE.itemsize

# (num_layers, num_blocks, tokens_per_block, num_heads, head_dim)
# num_blocks must be divisible by NUM_GPUS (default 4) for sharded mode.
#
# MLA models (DeepSeek-V3, Kimi-K2):
#   kv_heads=1, latent_dim=512, 61 layers, bf16/fp8
# MHA models (Llama-3, Qwen2):
#   kv_heads=8, head_dim=128, 32-80 layers, bf16
#   num_heads must be divisible by NUM_GPUS for non-MLA TP sharding.
MLA_SIZES = [
    # DeepSeek-V2/V3 scale: 61 layers, latent_dim=512
    pytest.param((4, 8, 16, 1, 512), id="ds3-mini"),      # quick smoke test
    pytest.param((32, 64, 16, 1, 512), id="llama3-8b"),   # 32 layers like Llama-3-8B
    pytest.param((61, 256, 16, 1, 512), id="ds3"),         # DeepSeek-V3: 61 layers
    pytest.param((80, 512, 16, 1, 512), id="llama3-70b"), # 80 layers like Llama-3-70B
    pytest.param((2, 4, 1, 1, 512), id="edge"),           # tpb=1 edge case
]
MHA_SIZES = [
    # Llama-3 scale: 32 layers, kv_heads=8, head_dim=128
    pytest.param((4, 8, 16, 8, 128), id="llama3-mini"),
    pytest.param((32, 64, 16, 8, 128), id="llama3-8b"),
    pytest.param((80, 256, 16, 8, 128), id="llama3-70b"),
    pytest.param((2, 4, 1, 8, 128), id="edge"),            # tpb=1 edge case
    pytest.param((4, 4, 16, 16, 128), id="16head"),       # 16 heads variant
]

CPU_LAYOUTS = [
    pytest.param("LAYERFIRST", id="lfirst"),
    pytest.param("BLOCKFIRST", id="bfirst"),
]

ENGINES = [
    pytest.param("cuda", False, id="cuda"),
    pytest.param("ce", True, id="ce"),
]

MLA_MODES = ["sharded", "all_write", "rank0_only", "layer_parallel", "rank_rotate"]

CE_MEMCPY2D_CONFIGS = [False, True]


# Helpers (matching production code in worker.py / layerwise.py)

def make_layouts(num_layers, num_blocks, tpb, num_heads, head_dim,
                 cpu_layout_name, is_mla, tp_size):
    """Create GPU and CPU KVCacheLayout objects matching production conventions.

    GPU: LAYERFIRST, per-rank heads for non-MLA.
    CPU: specified layout, full heads.
    For non-MLA + BLOCKFIRST: CPU strides use div_head(tp_size).

    Returns (gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank).
    """
    kv_dim = 1 if is_mla else 2

    # Non-MLA shards heads across TP ranks, so num_heads must be divisible by
    # tp_size. This is guaranteed by MHA_SIZES (num_heads=8, tp_size=4) but we
    # keep the assertion as a safety net against future config changes.
    if not is_mla:
        assert num_heads % tp_size == 0, \
            f"non-MLA requires num_heads % tp_size == 0, got {num_heads} % {tp_size}"

    heads_per_rank = num_heads if is_mla else num_heads // tp_size

    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=tpb, num_head=heads_per_rank,
        head_size=head_dim, is_mla=is_mla)

    cpu_layout = KVCacheLayout(
        type=KVCacheLayoutType[cpu_layout_name.upper()],
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=tpb, num_head=num_heads,
        head_size=head_dim, is_mla=is_mla)

    if not is_mla and cpu_layout.type == KVCacheLayoutType.BLOCKFIRST:
        cpu_layout_tp = cpu_layout.div_head(tp_size)
    else:
        cpu_layout_tp = cpu_layout

    return gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank


def make_gpu_tensors(num_layers, num_blocks, tpb, heads_per_rank, head_dim, kv_dim, device):
    """Create contiguous GPU buffer: [num_layers, kv_dim, num_blocks, tpb, heads_per_rank, head_dim].

    Returns list of per-layer tensor views (matching vLLM convention).
    """
    full = torch.zeros(
        (num_layers, kv_dim, num_blocks, tpb, heads_per_rank, head_dim),
        dtype=DTYPE, device=f"cuda:{device}")
    return [full[i] for i in range(num_layers)]


def make_cpu_tensor(cpu_layout, num_layers, total_blocks):
    """Create pinned CPU tensor with the given total_blocks.

    Rebuilds the layout with num_block=total_blocks so strides match the
    actual allocation (needed for all_write where total = num_gpus * blocks).
    """
    layout = KVCacheLayout(
        type=cpu_layout.type,
        num_layer=num_layers, num_block=total_blocks,
        tokens_per_block=cpu_layout.tokens_per_block,
        num_head=cpu_layout.num_head, head_size=cpu_layout.head_size,
        is_mla=cpu_layout.is_mla)
    return torch.zeros(tuple(layout.kv_shape), dtype=DTYPE, pin_memory=True)


def cpu_layout_for_mode(cpu_layout, cpu_layout_tp, num_layers, num_blocks,
                        num_heads, head_dim, tpb, is_mla, mode, num_gpus):
    """Resolve CPU buffer size + kv/layer/block/tp strides for an MLA D2H mode.

    Matches the authoritative PR #192 semantics (tp_transfer offset logic):
      - sharded / rank0_only: CPU holds one rank's KV -> num_blocks, TP strides.
      - all_write: each rank writes its full KV into its own slot at offset
        i * num_blocks * cpu_block_stride, so the CPU buffer spans
        total = num_gpus * num_blocks. kv/layer strides must be recomputed from
        a layout with num_block=total (NOT the single-rank TP strides), which
        naturally scales LAYERFIRST strides by num_gpus while leaving
        BLOCKFIRST strides unchanged (block_stride already includes layers).
    Returns (total_cpu_blocks, cpu_stride_kv, cpu_stride_layer,
             cpu_stride_block, cpu_stride_tp).
    """
    if is_mla and mode == "all_write":
        total_cpu_blocks = num_blocks * num_gpus
        layout_for_strides = KVCacheLayout(
            type=cpu_layout.type,
            num_layer=num_layers, num_block=total_cpu_blocks,
            tokens_per_block=tpb, num_head=num_heads,
            head_size=head_dim, is_mla=is_mla)
        cpu_stride_kv = layout_for_strides.get_kv_stride() * ES
        cpu_stride_layer = layout_for_strides.get_layer_stride() * ES
    else:
        total_cpu_blocks = num_blocks
        cpu_stride_kv = cpu_layout_tp.get_kv_stride() * ES
        cpu_stride_layer = cpu_layout_tp.get_layer_stride() * ES
    # block_stride never depends on num_block; tp_stride derived from it.
    cpu_stride_block = cpu_layout.get_block_stride() * ES
    cpu_stride_tp = cpu_stride_block // num_gpus
    return (total_cpu_blocks, cpu_stride_kv, cpu_stride_layer,
            cpu_stride_block, cpu_stride_tp)


def fill_gpu(gpu_tensors, gpu_id, num_layers, num_blocks, tpb, heads, hd, kv_dim):
    """Fill GPU tensors with deterministic per-GPU pattern. K and V differ."""
    for layer in range(num_layers):
        dev = gpu_tensors[layer].device
        kv = torch.arange(kv_dim, device=dev).view(kv_dim, 1, 1, 1, 1)
        blk = torch.arange(num_blocks, device=dev).view(1, num_blocks, 1, 1, 1)
        tok = torch.arange(tpb, device=dev).view(1, 1, tpb, 1, 1)
        h = torch.arange(hd, device=dev).view(1, 1, 1, 1, hd)
        vals = (gpu_id * 100000 + kv * 500000 + layer * 10000 +
                blk * 1000 + tok * 10 + h) % 997
        gpu_tensors[layer][:] = (vals / 997.0).to(DTYPE)


def expected_val(gpu_id, layer, block, token, hd, kv_dim_idx=0):
    """Expected value for (gpu_id, layer, block, token, hd, kv_dim_idx)."""
    return float(((gpu_id * 100000 + kv_dim_idx * 500000 + layer * 10000 +
                    block * 1000 + token * 10 + hd) % 997) / 997.0)


def make_tp_group(cpu_ptr, all_gpu, num_gpus, gpu_layout, num_layers,
                  ce_segment_threshold=None,
                  ce_path_opt=None,
                  ce_enable_memcpy2d=None,
                  ce_gather_threads=None,
                  ce_gather_nt=None,
                  is_blockfirst=None,
                  is_mla=None):
    """Create TPTransferThreadGroup with strides from KVCacheLayout.

    Matches production worker.py:472 exactly -- chunk_size does NOT include kv_dim.
    The C++ kernel iterates num_chunks = num_layers * kv_dim * num_blocks and
    copies chunk_size bytes per chunk, so kv_dim is a separate iteration axis.

    CE config defaults from GLOBAL_CONFIG_FROM_ENV (same as production).
    """
    if ce_segment_threshold is None:
        ce_segment_threshold = GLOBAL_CONFIG_FROM_ENV.ce_segment_threshold
    if ce_path_opt is None:
        ce_path_opt = GLOBAL_CONFIG_FROM_ENV.ce_path_opt
    if ce_enable_memcpy2d is None:
        ce_enable_memcpy2d = GLOBAL_CONFIG_FROM_ENV.enable_ce_memcpy2d
    if ce_gather_threads is None:
        ce_gather_threads = GLOBAL_CONFIG_FROM_ENV.ce_gather_threads
    if ce_gather_nt is None:
        ce_gather_nt = GLOBAL_CONFIG_FROM_ENV.ce_gather_nt
    if is_blockfirst is None:
        is_blockfirst = (GLOBAL_CONFIG_FROM_ENV.cpu_layout_type == KVCacheLayoutType.BLOCKFIRST)
    if is_mla is None:
        is_mla = gpu_layout.is_mla
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
        # Pass all trailing defaulted params explicitly so pybind11 never has
        # to synthesize a default (see pybind11-construct-debug).
        enable_nvcomp=False,
        nvcomp_batch_size=0,
        nvcomp_data_type=0,
        ce_segment_threshold=ce_segment_threshold,
        ce_path_opt=ce_path_opt,
        ce_enable_memcpy2d=ce_enable_memcpy2d,
        ce_gather_threads=ce_gather_threads,
        ce_gather_nt=ce_gather_nt,
        is_blockfirst=is_blockfirst,
        is_mla=is_mla,
    )


def make_layerwise_group(cpu_tensor, all_gpu, num_gpus, gpu_layout, num_layers,
                         ce_segment_threshold=None,
                         ce_path_opt=None,
                         ce_enable_memcpy2d=None,
                         ce_gather_threads=None,
                         ce_gather_nt=None,
                         is_blockfirst=None,
                         is_mla=None,
                         layer_eventfds_tensor=None):
    """Create LayerwiseTransferGroup for H2D-only testing (no SSD).

    Mirrors LayerwiseTransferWorker construction in layerwise.py. GPU chunk_size
    does NOT include kv_dim (same as tp_group). SSD disabled via empty ssd_files.
    eventfd disabled by default (empty layer_eventfds_tensor); pass a non-empty
    tensor to exercise the notify-mode path (upstream #199).

    CE config defaults from GLOBAL_CONFIG_FROM_ENV (same as production).
    """
    if ce_segment_threshold is None:
        ce_segment_threshold = GLOBAL_CONFIG_FROM_ENV.ce_segment_threshold
    if ce_path_opt is None:
        ce_path_opt = GLOBAL_CONFIG_FROM_ENV.ce_path_opt
    if ce_enable_memcpy2d is None:
        ce_enable_memcpy2d = GLOBAL_CONFIG_FROM_ENV.enable_ce_memcpy2d
    if ce_gather_threads is None:
        ce_gather_threads = GLOBAL_CONFIG_FROM_ENV.ce_gather_threads
    if ce_gather_nt is None:
        ce_gather_nt = GLOBAL_CONFIG_FROM_ENV.ce_gather_nt
    if is_blockfirst is None:
        is_blockfirst = (GLOBAL_CONFIG_FROM_ENV.cpu_layout_type == KVCacheLayoutType.BLOCKFIRST)
    if is_mla is None:
        is_mla = gpu_layout.is_mla
    if layer_eventfds_tensor is None:
        layer_eventfds_tensor = torch.empty(0, dtype=torch.int32)
    def strides_tensor(getter):
        return torch.tensor([getter() * ES] * num_gpus, dtype=torch.int64)

    # Pass every trailing optional parameter explicitly. This exercises the
    # same SWA-capable + adaptive-CE constructor used by production.
    empty_tensor = torch.empty(0)
    return LayerwiseTransferGroup(
        num_gpus=num_gpus,
        gpu_blocks=all_gpu,
        cpu_blocks=cpu_tensor,
        ssd_files={},
        num_layers=num_layers,
        gpu_kv_strides_tensor=strides_tensor(gpu_layout.get_kv_stride),
        gpu_block_strides_tensor=strides_tensor(gpu_layout.get_block_stride),
        gpu_layer_strides_tensor=strides_tensor(gpu_layout.get_layer_stride),
        gpu_chunk_sizes_tensor=strides_tensor(gpu_layout.get_chunk_size),
        iouring_entries=0,
        iouring_flags=0,
        layer_eventfds_tensor=layer_eventfds_tensor,
        tp_size=num_gpus,
        has_swa=False,
        swa_gpu_blocks=[],
        swa_cpu_blocks=empty_tensor,
        swa_ssd_files={},
        swa_gpu_kv_strides_tensor=empty_tensor,
        swa_gpu_block_strides_tensor=empty_tensor,
        swa_gpu_layer_strides_tensor=empty_tensor,
        swa_gpu_chunk_sizes_tensor=empty_tensor,
        ce_segment_threshold=ce_segment_threshold,
        ce_path_opt=ce_path_opt,
        ce_enable_memcpy2d=ce_enable_memcpy2d,
        ce_gather_threads=ce_gather_threads,
        ce_gather_nt=ce_gather_nt,
        is_blockfirst=is_blockfirst,
        is_mla=is_mla,
    )


def layerwise_h2d_readback(all_gpu, cpu_kv, num_gpus, gpu_layout, num_layers,
                           ids, cpu_stride_kv, cpu_stride_layer,
                           cpu_stride_block, cpu_stride_tp, chunk_size,
                           is_mla, mode, ce_path_opt=None,
                           ce_segment_threshold=None,
                           notify_mode="hostfunc", layer_granularity=None,
                           is_blockfirst=None,
                           enable_memcpy2d=None,
                           ce_gather_threads=None,
                           ce_gather_nt=None):
    """Run a single CE H2D via LayerwiseTransferGroup, reading `cpu_kv` back
    into `all_gpu` with block-id list `ids`.

    LayerwiseTransferGroup is H2D-only and CE-only; this wraps the
    SWA-capable layerwise_transfer() call. Shared by the CE-path layerwise
    test and the roundtrip layerwise twins.

    notify_mode: "hostfunc" (default, uses CUDA hostfunc callback) or
    "polling" (uses a CPU polling thread that queries cudaEventQuery per
    batch).  Polling mode exercises the async GATHER_SCATTER/SEGMENT_SCATTER
    path (sync=false) that was previously deadlocked.
    """
    lw_group = make_layerwise_group(cpu_kv, all_gpu, num_gpus,
                                    gpu_layout, num_layers,
                                    ce_path_opt=ce_path_opt,
                                    ce_segment_threshold=ce_segment_threshold,
                                    ce_gather_threads=ce_gather_threads,
                                    ce_gather_nt=ce_gather_nt,
                                    is_blockfirst=is_blockfirst,
                                    ce_enable_memcpy2d=enable_memcpy2d)
    empty_ids = torch.empty(0, dtype=torch.int64).pin_memory()
    lw_group.layerwise_transfer(
        ssd_block_ids=empty_ids,
        cpu_block_ids_d2h=empty_ids,
        ssd_layer_stride_in_bytes=0,
        ssd_kv_stride_in_bytes=0,
        num_blocks_per_file=0, round_robin=0, num_threads_per_device=0,
        gpu_block_id_tensor=ids, cpu_block_id_tensor=ids,
        cpu_kv_stride_in_bytes=cpu_stride_kv,
        cpu_layer_stride_in_bytes=cpu_stride_layer,
        cpu_block_stride_in_bytes=cpu_stride_block,
        cpu_chunk_size_in_bytes=chunk_size,
        h2d_cpu_kv_stride_in_bytes=cpu_stride_kv,
        h2d_cpu_layer_stride_in_bytes=cpu_stride_layer,
        cpu_tp_stride_in_bytes=cpu_stride_tp,
        transfer_cta_num=4, use_ce_transfer=True,
        num_layers=num_layers, layer_granularity=num_layers if layer_granularity is None else layer_granularity,
        is_mla=is_mla,
        counter_id=0,
        swa_h2d_src=empty_ids,
        swa_h2d_dst=empty_ids,
        swa_disk2h_src=empty_ids,
        swa_disk2h_dst=empty_ids,
        mla_d2h_mode=mode,
        notify_mode=notify_mode,
    )
    sync_all(num_gpus)
    del lw_group


def block_ids(n):
    """Create block-id tensor in PINNED host memory.

    The CUDA kernel dereferences block-id arrays on the DEVICE (via UVA), so
    pageable CPU tensors trigger 'illegal memory access'. Production uses
    .pin_memory() (worker.py:173). Must match here.
    """
    return torch.arange(n, dtype=torch.int64).pin_memory()


def sync_all(num_gpus):
    for g in range(num_gpus):
        torch.cuda.synchronize(g)


def spot_check_gpu(all_gpu, expected_gpu_id, num_gpus, num_layers, num_blocks,
                   tpb, hd, kv_dim, label=""):
    """Spot-check a few GPU values for both K and V."""
    for g in range(num_gpus):
        for layer in [0, num_layers - 1]:
            for block in [0, num_blocks - 1]:
                for kv in range(kv_dim):
                    for hd_idx in [0, hd - 1]:
                        exp = expected_val(expected_gpu_id, layer, block, 0, hd_idx, kv)
                        act = all_gpu[g][layer][kv, block, 0, 0, hd_idx].item()
                        assert abs(act - exp) < 1e-3, \
                            f"{label} mismatch: gpu={g} layer={layer} block={block} " \
                            f"kv={kv} hd={hd_idx}: expected={exp:.6f} got={act:.6f}"


# Round-trip tests (D2H -> clear GPU -> H2D -> verify)

@pytest.mark.parametrize("data_config", MHA_SIZES)
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("engine_name,use_ce", ENGINES)
@pytest.mark.parametrize("enable_memcpy2d", CE_MEMCPY2D_CONFIGS, ids=["no_memcpy2d", "memcpy2d"])
def test_non_mla_roundtrip(data_config, cpu_layout_name, engine_name, use_ce, enable_memcpy2d):
    """Non-MLA round-trip: D2H -> clear GPU -> H2D -> verify per-rank data.

    Non-MLA does NOT use mla_d2h_mode — the C++ else-branch uses
    cpu_tp_stride to place each rank's head partition at a different
    CPU offset. This tests the default TP-shard path.
    """
    skip_if_engine_unsupported(use_ce)
    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    num_gpus = NUM_GPUS
    is_mla = False

    gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim,
        cpu_layout_name, is_mla, num_gpus)

    all_gpu = [make_gpu_tensors(num_layers, num_blocks, tpb, heads_per_rank, head_dim, kv_dim, g)
               for g in range(num_gpus)]

    # Each rank owns a different head partition — fill with its own pattern.
    for g in range(num_gpus):
        fill_gpu(all_gpu[g], g, num_layers, num_blocks, tpb, heads_per_rank, head_dim, kv_dim)
    sync_all(num_gpus)

    cpu_kv = make_cpu_tensor(cpu_layout, num_layers, num_blocks)

    tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus, gpu_layout, num_layers,
                       is_blockfirst=(cpu_layout_name == "BLOCKFIRST"),
                       ce_enable_memcpy2d=enable_memcpy2d)
    gpu_block_ids = block_ids(num_blocks)
    cpu_block_ids = block_ids(num_blocks)

    # D2H
    tp.tp_group_transfer(
        gpu_block_id_tensor=gpu_block_ids, cpu_block_id_tensor=cpu_block_ids,
        cpu_kv_stride_in_bytes=cpu_layout_tp.get_kv_stride() * ES,
        cpu_layer_stride_in_bytes=cpu_layout_tp.get_layer_stride() * ES,
        cpu_block_stride_in_bytes=cpu_layout.get_block_stride() * ES,
        cpu_tp_stride_in_bytes=cpu_layout.get_block_stride() * ES // num_gpus,
        transfer_num_cta=4, is_host_to_device=False, use_ce_transfer=use_ce,
        layer_id=0, layer_granularity=num_layers, is_mla=False,
        mla_d2h_mode="sharded",  # ignored for non-MLA
    )
    sync_all(num_gpus)

    # Clear GPUs
    for g in range(num_gpus):
        for l in range(num_layers):
            all_gpu[g][l].zero_()
    sync_all(num_gpus)

    # H2D
    tp.tp_group_transfer(
        gpu_block_id_tensor=gpu_block_ids, cpu_block_id_tensor=cpu_block_ids,
        cpu_kv_stride_in_bytes=cpu_layout_tp.get_kv_stride() * ES,
        cpu_layer_stride_in_bytes=cpu_layout_tp.get_layer_stride() * ES,
        cpu_block_stride_in_bytes=cpu_layout.get_block_stride() * ES,
        cpu_tp_stride_in_bytes=cpu_layout.get_block_stride() * ES // num_gpus,
        transfer_num_cta=4, is_host_to_device=True, use_ce_transfer=use_ce,
        layer_id=0, layer_granularity=num_layers, is_mla=False,
        mla_d2h_mode="sharded",  # ignored for non-MLA
    )
    sync_all(num_gpus)

    # Verify: each rank should have its own original data back.
    for g in range(num_gpus):
        for layer in [0, num_layers - 1]:
            for block in [0, num_blocks - 1]:
                for kv in range(kv_dim):
                    for hd_idx in [0, head_dim - 1]:
                        exp = expected_val(g, layer, block, 0, hd_idx, kv)
                        act = all_gpu[g][layer][kv, block, 0, 0, hd_idx].item()
                        assert abs(act - exp) < 1e-3, \
                            f"Non-MLA round-trip mismatch: layout={cpu_layout_name} " \
                            f"gpu={g} layer={layer} block={block} kv={kv} hd={hd_idx}: " \
                            f"expected={exp:.6f} got={act:.6f}"

    del tp


# MLA mode tests (sharded / all_write / rank0_only)

@pytest.mark.parametrize("data_config", MLA_SIZES)
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("engine_name,use_ce", ENGINES)
@pytest.mark.parametrize("mode", MLA_MODES)
@pytest.mark.parametrize("enable_memcpy2d", CE_MEMCPY2D_CONFIGS, ids=["no_memcpy2d", "memcpy2d"])
def test_mla_roundtrip_modes(data_config, cpu_layout_name, engine_name, use_ce, mode, enable_memcpy2d):
    """MLA round-trip with each D2H mode. Verifies K and V."""
    skip_if_engine_unsupported(use_ce)
    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    assert num_heads == 1, "MLA_SIZES must only contain num_heads=1 configs"

    num_gpus = NUM_GPUS
    is_mla = True

    gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim,
        cpu_layout_name, is_mla, num_gpus)

    total_cpu_blocks = num_blocks * num_gpus if mode == "all_write" else num_blocks

    all_gpu = [make_gpu_tensors(num_layers, num_blocks, tpb, heads_per_rank, head_dim, kv_dim, g)
               for g in range(num_gpus)]

    # MLA: all GPUs have identical data
    fill_gpu(all_gpu[0], 0, num_layers, num_blocks, tpb, heads_per_rank, head_dim, kv_dim)
    for g in range(1, num_gpus):
        for l in range(num_layers):
            all_gpu[g][l].copy_(all_gpu[0][l])
    sync_all(num_gpus)

    cpu_kv = make_cpu_tensor(cpu_layout, num_layers, total_cpu_blocks)

    # For all_write the CPU holds N ranks' KV, so kv_stride/layer_stride must
    # be computed from a layout with num_block=total_cpu_blocks (the per-block
    # chunk is N*chunk wide). Rebuild a TP-stride layout accordingly.
    if mode == "all_write":
        cpu_layout_for_strides = KVCacheLayout(
            type=cpu_layout.type,
            num_layer=num_layers, num_block=total_cpu_blocks,
            tokens_per_block=tpb, num_head=num_heads,
            head_size=head_dim, is_mla=is_mla)
        cpu_stride_kv = cpu_layout_for_strides.get_kv_stride() * ES
        cpu_stride_layer = cpu_layout_for_strides.get_layer_stride() * ES
    else:
        cpu_stride_kv = cpu_layout_tp.get_kv_stride() * ES
        cpu_stride_layer = cpu_layout_tp.get_layer_stride() * ES
    # block_stride never depends on num_block; tp_stride derived from it.
    cpu_stride_block = cpu_layout.get_block_stride() * ES
    cpu_stride_tp = cpu_stride_block // num_gpus

    tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus, gpu_layout, num_layers,
                       is_blockfirst=(cpu_layout_name == "BLOCKFIRST"),
                       ce_enable_memcpy2d=enable_memcpy2d)
    gpu_block_ids = block_ids(num_blocks)
    cpu_block_ids = block_ids(num_blocks)

    # D2H
    tp.tp_group_transfer(
        gpu_block_id_tensor=gpu_block_ids, cpu_block_id_tensor=cpu_block_ids,
        cpu_kv_stride_in_bytes=cpu_stride_kv,
        cpu_layer_stride_in_bytes=cpu_stride_layer,
        cpu_block_stride_in_bytes=cpu_stride_block,
        cpu_tp_stride_in_bytes=cpu_stride_tp,
        transfer_num_cta=4, is_host_to_device=False, use_ce_transfer=use_ce,
        layer_id=0, layer_granularity=num_layers, is_mla=True,
        mla_d2h_mode=mode,
    )
    sync_all(num_gpus)

    # Clear GPUs
    for g in range(num_gpus):
        for l in range(num_layers):
            all_gpu[g][l].zero_()
    sync_all(num_gpus)

    # H2D
    tp.tp_group_transfer(
        gpu_block_id_tensor=gpu_block_ids, cpu_block_id_tensor=cpu_block_ids,
        cpu_kv_stride_in_bytes=cpu_stride_kv,
        cpu_layer_stride_in_bytes=cpu_stride_layer,
        cpu_block_stride_in_bytes=cpu_stride_block,
        cpu_tp_stride_in_bytes=cpu_stride_tp,
        transfer_num_cta=4, is_host_to_device=True, use_ce_transfer=use_ce,
        layer_id=0, layer_granularity=num_layers, is_mla=True,
        mla_d2h_mode=mode,
    )
    sync_all(num_gpus)

    # Verify: all GPUs should have GPU 0's original data
    spot_check_gpu(all_gpu, 0, num_gpus, num_layers, num_blocks,
                   tpb, head_dim, kv_dim, label=f"mode={mode}")

    del tp


# Layerwise H2D test with notify modes (hostfunc / polling)

LAYERWISE_NOTIFY_MODES = ["hostfunc", "polling"]


@pytest.mark.parametrize("data_config", [pytest.param((4, 8, 16, 1, 512), id="ds3-mini")])
@pytest.mark.parametrize("engine_name,use_ce", ENGINES)
@pytest.mark.parametrize("notify_mode", LAYERWISE_NOTIFY_MODES)
def test_layerwise_h2d_notify_modes(data_config, engine_name, use_ce, notify_mode):
    """Layerwise H2D round-trip under hostfunc / polling notify modes.

    Verifies data correctness under both notification modes.
    """
    skip_if_engine_unsupported(use_ce)

    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    num_gpus = NUM_GPUS

    gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim,
        "BLOCKFIRST", True, num_gpus)

    all_gpu = [make_gpu_tensors(num_layers, num_blocks, tpb, heads_per_rank,
                                head_dim, kv_dim, g) for g in range(num_gpus)]
    fill_gpu(all_gpu[0], 0, num_layers, num_blocks, tpb, heads_per_rank,
             head_dim, kv_dim)
    for g in range(1, num_gpus):
        for l in range(num_layers):
            all_gpu[g][l].copy_(all_gpu[0][l])
    sync_all(num_gpus)

    cpu_kv = make_cpu_tensor(cpu_layout, num_layers, num_blocks)
    cpu_stride_kv = cpu_layout_tp.get_kv_stride() * ES
    cpu_stride_layer = cpu_layout_tp.get_layer_stride() * ES
    cpu_stride_block = cpu_layout.get_block_stride() * ES
    cpu_stride_tp = cpu_stride_block // num_gpus

    # D2H via TP group to populate CPU
    tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus, gpu_layout, num_layers,
                       is_blockfirst=True,
                       is_mla=True)
    gpu_block_ids = block_ids(num_blocks)
    cpu_block_ids = block_ids(num_blocks)
    tp.tp_group_transfer(
        gpu_block_id_tensor=gpu_block_ids, cpu_block_id_tensor=cpu_block_ids,
        cpu_kv_stride_in_bytes=cpu_stride_kv,
        cpu_layer_stride_in_bytes=cpu_stride_layer,
        cpu_block_stride_in_bytes=cpu_stride_block,
        cpu_tp_stride_in_bytes=cpu_stride_tp,
        transfer_num_cta=4, is_host_to_device=False, use_ce_transfer=use_ce,
        layer_id=0, layer_granularity=num_layers, is_mla=True,
        mla_d2h_mode="sharded",
    )
    sync_all(num_gpus)
    del tp

    for g in range(num_gpus):
        for l in range(num_layers):
            all_gpu[g][l].zero_()
    sync_all(num_gpus)

    # H2D via layerwise with the requested notify mode
    lw = make_layerwise_group(cpu_kv, all_gpu, num_gpus, gpu_layout, num_layers,
                                is_blockfirst=True,
                                is_mla=True)
    lw.layerwise_transfer(
        torch.empty(0, dtype=torch.int64), torch.empty(0, dtype=torch.int64),
        0, 0, 0, 0, 0,
        gpu_block_ids, cpu_block_ids,
        cpu_stride_kv, cpu_stride_layer, cpu_stride_block,
        gpu_layout.get_chunk_size() * ES,
        cpu_stride_kv, cpu_stride_layer, cpu_stride_tp,
        4, use_ce, num_layers, 1, True,
        counter_id=0,
        # pybind cannot fill the SWA tensor defaults (torch::Tensor() -> None
        # can't cast back to a non-optional `torch.Tensor` param), so the four
        # swa_* tensors must be passed explicitly even for this non-SWA test.
        # Mirrors production LayerwiseTransfer._swa_transfer_kwargs().
        swa_h2d_src=torch.empty(0, dtype=torch.int64),
        swa_h2d_dst=torch.empty(0, dtype=torch.int64),
        swa_disk2h_src=torch.empty(0, dtype=torch.int64),
        swa_disk2h_dst=torch.empty(0, dtype=torch.int64),
        mla_d2h_mode="sharded",
        notify_mode=notify_mode,
    )
    sync_all(num_gpus)

    spot_check_gpu(all_gpu, 0, num_gpus, num_layers, num_blocks,
                   tpb, head_dim, kv_dim, label=f"notify={notify_mode}")
    del lw


# Round-trip tests via LayerwiseTransferGroup H2D
#
# LayerwiseTransferGroup is H2D-only and CE-only (no cuda-kernel engine, no
# independent D2H), so these twins prepare the CPU reference with a verified
# TPTransferThreadGroup CE D2H, then read it back with layerwise H2D and check
# correctness. Same size matrix / modes / layouts as the TP-group round-trips
# above, so layerwise H2D is exercised across the full production shape space.
# Uses contiguous block ids (identity) like the TP round-trips; the CE-path
# tests separately sweep few_seg/scattered patterns for both groups.

@pytest.mark.parametrize("data_config", MHA_SIZES)
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
def test_non_mla_roundtrip_layerwise(data_config, cpu_layout_name):
    """Non-MLA: TP-group CE D2H prepares CPU, LayerwiseTransferGroup CE H2D
    reads it back; verify each rank recovers its own data."""
    skip_if_engine_unsupported(use_ce=True)
    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    num_gpus = NUM_GPUS
    is_mla = False
    mode = "sharded"  # ignored for non-MLA

    gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim,
        cpu_layout_name, is_mla, num_gpus)

    all_gpu = [make_gpu_tensors(num_layers, num_blocks, tpb,
                                heads_per_rank, head_dim, kv_dim, g)
               for g in range(num_gpus)]
    for g in range(num_gpus):
        fill_gpu(all_gpu[g], g, num_layers, num_blocks, tpb,
                 heads_per_rank, head_dim, kv_dim)
    sync_all(num_gpus)

    (total_blocks, cpu_stride_kv, cpu_stride_layer,
     cpu_stride_block, cpu_stride_tp) = cpu_layout_for_mode(
        cpu_layout, cpu_layout_tp, num_layers, num_blocks,
        num_heads, head_dim, tpb, is_mla, mode, num_gpus)
    cpu_kv = make_cpu_tensor(cpu_layout, num_layers, total_blocks)
    ids = block_ids(num_blocks)
    chunk_size = gpu_layout.get_chunk_size() * ES

    # D2H prepare via TP-group (CE), verified correct elsewhere.
    tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus,
                       gpu_layout, num_layers,
                       is_blockfirst=(cpu_layout_name == "BLOCKFIRST"))
    tp.tp_group_transfer(
        gpu_block_id_tensor=ids, cpu_block_id_tensor=ids,
        cpu_kv_stride_in_bytes=cpu_stride_kv,
        cpu_layer_stride_in_bytes=cpu_stride_layer,
        cpu_block_stride_in_bytes=cpu_stride_block,
        cpu_tp_stride_in_bytes=cpu_stride_tp,
        transfer_num_cta=4, is_host_to_device=False, use_ce_transfer=True,
        layer_id=0, layer_granularity=num_layers, is_mla=is_mla,
        mla_d2h_mode=mode,
    )
    sync_all(num_gpus)
    del tp

    for g in range(num_gpus):
        for l in range(num_layers):
            all_gpu[g][l].zero_()
    sync_all(num_gpus)

    # H2D readback via layerwise (test target).
    # is_blockfirst MUST match the D2H path (make_tp_group above) — if it
    # defaults to GLOBAL_CONFIG_FROM_ENV.cpu_layout_type (BLOCKFIRST), the H2D
    # CE path selector sees is_blockfirst=True and picks GATHER_DIRECT instead
    # of SEGMENT_SCATTER, breaking the round-trip for LAYERFIRST non-MLA TP.
    layerwise_h2d_readback(
        all_gpu, cpu_kv, num_gpus, gpu_layout, num_layers, ids,
        cpu_stride_kv, cpu_stride_layer, cpu_stride_block, cpu_stride_tp,
        chunk_size, is_mla, mode,
        is_blockfirst=(cpu_layout_name == "BLOCKFIRST"))

    for g in range(num_gpus):
        for layer in [0, num_layers - 1]:
            for block in [0, num_blocks - 1]:
                for kv in range(kv_dim):
                    for hd_idx in [0, head_dim - 1]:
                        exp = expected_val(g, layer, block, 0, hd_idx, kv)
                        act = all_gpu[g][layer][kv, block, 0, 0, hd_idx].item()
                        assert abs(act - exp) < 1e-3, \
                            "Non-MLA layerwise round-trip mismatch: " \
                            "layout={} gpu={} layer={} block={} kv={} hd={}: " \
                            "expected={:.6f} got={:.6f}".format(
                                cpu_layout_name, g, layer, block, kv, hd_idx,
                                exp, act)


@pytest.mark.parametrize("data_config", MLA_SIZES)
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("mode", MLA_MODES)
def test_mla_roundtrip_modes_layerwise(data_config, cpu_layout_name, mode):
    """MLA: TP-group CE D2H prepares CPU, LayerwiseTransferGroup CE H2D reads
    it back; verify all ranks recover GPU 0's data. Covers all D2H modes."""
    skip_if_engine_unsupported(use_ce=True)
    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    assert num_heads == 1, "MLA_SIZES must only contain num_heads=1 configs"
    num_gpus = NUM_GPUS
    is_mla = True

    gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim,
        cpu_layout_name, is_mla, num_gpus)

    all_gpu = [make_gpu_tensors(num_layers, num_blocks, tpb,
                                heads_per_rank, head_dim, kv_dim, g)
               for g in range(num_gpus)]
    fill_gpu(all_gpu[0], 0, num_layers, num_blocks, tpb,
             heads_per_rank, head_dim, kv_dim)
    for g in range(1, num_gpus):
        for l in range(num_layers):
            all_gpu[g][l].copy_(all_gpu[0][l])
    sync_all(num_gpus)

    (total_blocks, cpu_stride_kv, cpu_stride_layer,
     cpu_stride_block, cpu_stride_tp) = cpu_layout_for_mode(
        cpu_layout, cpu_layout_tp, num_layers, num_blocks,
        num_heads, head_dim, tpb, is_mla, mode, num_gpus)
    cpu_kv = make_cpu_tensor(cpu_layout, num_layers, total_blocks)
    ids = block_ids(num_blocks)
    chunk_size = gpu_layout.get_chunk_size() * ES

    tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus,
                       gpu_layout, num_layers,
                       is_blockfirst=(cpu_layout_name == "BLOCKFIRST"))
    tp.tp_group_transfer(
        gpu_block_id_tensor=ids, cpu_block_id_tensor=ids,
        cpu_kv_stride_in_bytes=cpu_stride_kv,
        cpu_layer_stride_in_bytes=cpu_stride_layer,
        cpu_block_stride_in_bytes=cpu_stride_block,
        cpu_tp_stride_in_bytes=cpu_stride_tp,
        transfer_num_cta=4, is_host_to_device=False, use_ce_transfer=True,
        layer_id=0, layer_granularity=num_layers, is_mla=is_mla,
        mla_d2h_mode=mode,
    )
    sync_all(num_gpus)
    del tp

    for g in range(num_gpus):
        for l in range(num_layers):
            all_gpu[g][l].zero_()
    sync_all(num_gpus)

    layerwise_h2d_readback(
        all_gpu, cpu_kv, num_gpus, gpu_layout, num_layers, ids,
        cpu_stride_kv, cpu_stride_layer, cpu_stride_block, cpu_stride_tp,
        chunk_size, is_mla, mode,
        is_blockfirst=(cpu_layout_name == "BLOCKFIRST"))

    # All ranks should recover GPU 0's data (MLA replicates).
    spot_check_gpu(all_gpu, 0, num_gpus, num_layers, num_blocks,
                   tpb, head_dim, kv_dim, label="layerwise mode={}".format(mode))


# Invalid mode fallback test

def test_invalid_mode_fallback():
    """Invalid mla_d2h_mode falls back to 'sharded' without crash."""
    skip_if_engine_unsupported(use_ce=False)
    num_layers, num_blocks, tpb, num_heads, head_dim = 4, 8, 16, 1, 128
    num_gpus = NUM_GPUS

    gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim,
        "LAYERFIRST", True, num_gpus)

    all_gpu = [make_gpu_tensors(num_layers, num_blocks, tpb, 1, head_dim, kv_dim, g)
               for g in range(num_gpus)]
    fill_gpu(all_gpu[0], 0, num_layers, num_blocks, tpb, 1, head_dim, kv_dim)
    for g in range(1, num_gpus):
        for l in range(num_layers):
            all_gpu[g][l].copy_(all_gpu[0][l])
    sync_all(num_gpus)

    cpu_kv = make_cpu_tensor(cpu_layout, num_layers, num_blocks)
    tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus, gpu_layout, num_layers,
                       is_blockfirst=False)

    gpu_block_ids = block_ids(num_blocks)
    cpu_block_ids = block_ids(num_blocks)

    tp.tp_group_transfer(
        gpu_block_id_tensor=gpu_block_ids, cpu_block_id_tensor=cpu_block_ids,
        cpu_kv_stride_in_bytes=cpu_layout_tp.get_kv_stride() * ES,
        cpu_layer_stride_in_bytes=cpu_layout_tp.get_layer_stride() * ES,
        cpu_block_stride_in_bytes=cpu_layout.get_block_stride() * ES,
        cpu_tp_stride_in_bytes=cpu_layout.get_block_stride() * ES // num_gpus,
        transfer_num_cta=4, is_host_to_device=False, use_ce_transfer=False,
        layer_id=0, layer_granularity=num_layers, is_mla=True,
        mla_d2h_mode="invalid_xyz",
    )
    sync_all(num_gpus)

    # Should behave like sharded — just verify no crash
    del tp


# CE adaptive strategy tests
#
# The C++ CE engine selects among five execution strategies (see the CEPath
# taxonomy in csrc/ce_transfer.h). path_opt_enabled
# picks PER_BLOCK (baseline) vs the five optimized strategies; choose_path()
# picks among the optimized ones by block-id contiguity + CPU/GPU layout:
#   PER_BLOCK        — one memcpy per block (baseline, path_opt=False)
#   CONTIG_DIRECT      — single large memcpy (contiguous ids + dst phys contig)
#   SEGMENT_DIRECT — per-run memcpy, dst phys contig (LAYERFIRST), no staging
#   SEGMENT_SCATTER     — staging buffer + CPU scatter (dst strided / BLOCKFIRST),
#                      GPU contiguous (non-sharded) -> merged segment memcpy
#   GATHER_SCATTER     — staging buffer + CPU scatter (sharded D2H),
#                      GPU non-contiguous -> per-block memcpy
#   GATHER_SCATTER   — GPU index_select/index_copy_ (many segments > threshold)
# GATHER_DIRECT is CEPath(4), checked before !gpu_phys_contig in choose_path:
#   BF (BLOCKFIRST) + !cpu_phys_contig + GPU physically contiguous (non-sharded)
#   — covers both MLA and MHA. Sharded D2H breaks gpu_phys_contig and routes to
#   GATHER_SCATTER instead.
#
# We trigger each strategy by constructing block-id *permutations* of [0..N-1]
# so that every block is still transferred (round-trip correctness preserved):
#   contiguous — identity permutation        → 1 segment
#   few_seg    — interleaved 4-segment perm  → 4 segments
#   scattered  — random permutation           → N segments (>8)
#
# Strategy is chosen automatically from strides; there is no force override.
# Coverage of the five optimized strategies is asserted by
# test_ce_strategy_coverage below (via _expected_strategy).
#
# ce_path_opt (baseline vs optimized) is a per-construction CETransferConfig
# field (bindings.cpp sets cfg.path_opt_enabled from the ctor arg), NOT env-
# cached static -- so we sweep it as an ordinary orthogonal parametrize
# dimension in-process.

# Combined (data_config, is_mla, mode) parametrization.
#
# The C++ transfer branches at the top level on is_mla:
#   - is_mla == False (MHA): a single code path (heads sharded across TP,
#     cpu_startoff = i * cpu_tp_stride). The mla_d2h_mode argument is IGNORED.
#   - is_mla == True  (MLA): the mode selects sharded / all_write / rank0_only.
#
# So mode is only meaningful for MLA. We therefore emit exactly ONE combo per
# non-MLA size (mode is a don't-care placeholder), and THREE combos per MLA
# size. This avoids the previous mode x is_mla cross-product that produced
# nonsensical all_write-mha / rank0_only-mha combos handled only via skip().
#
# Sizes cover the SAME production shape matrix as MLA_SIZES / MHA_SIZES (the
# TP-group round-trips), so CE strategy selection is exercised across every
# real config -- including large (ds3 / llama3-70b) and small (edge, 16head)
# ones. "scattered" needs num_blocks > segment_threshold to form more segments
# than the threshold; the tests skip scattered only when num_blocks <= the
# swept threshold, so with threshold=2 even the small sizes run it.
#
# id suffix must be unique per size (CE_MODE_CONFIGS ids use it verbatim).
_MLA_SIZES = [
    ((4, 8, 16, 1, 512), "ds3-mini"),
    ((32, 64, 16, 1, 512), "llama8b"),
    ((61, 256, 16, 1, 512), "ds3"),
    ((80, 512, 16, 1, 512), "llama70b"),
    ((2, 4, 1, 1, 512), "edge"),
]
_MHA_SIZES = [
    ((4, 8, 16, 8, 128), "mha-mini"),
    ((32, 64, 16, 8, 128), "mha-llama8b"),
    ((80, 256, 16, 8, 128), "mha-llama70b"),
    ((2, 4, 1, 8, 128), "mha-edge"),
    ((4, 4, 16, 16, 128), "mha-16head"),
]
CE_MODE_CONFIGS = (
    [pytest.param(cfg, True, mode, id=f"mla_{mode}-{sid}")
     for (cfg, sid) in _MLA_SIZES
     for mode in ("sharded", "all_write", "rank0_only", "layer_parallel", "rank_rotate")]
    + [pytest.param(cfg, False, "sharded", id=f"non_mla-{sid}")
       for (cfg, sid) in _MHA_SIZES]
)

CE_PATTERNS = ["contiguous", "few_seg", "scattered"]

# segment_threshold is swept as an orthogonal dimension. threshold=8 is the
# production default; threshold=2 is small enough that "scattered" (~N segments)
# exceeds it for every size with num_blocks > 2, so even the small sizes
# (nb=4/8) exercise GATHER_SCATTER instead of skipping.
# It also tests the threshold config itself. scattered still skips only when
# num_blocks <= threshold (i.e. it cannot form more than `threshold` segments).
CE_SEGMENT_THRESHOLDS = [8, 2]

# enable_memcpy2d is swept as an orthogonal dimension. When True, the C++ engine
# uses cudaMemcpy2DAsync for strided direct transfer (both D2H and H2D) instead of
# staging + scatter/gather. It applies to path 2 SEGMENT_SCATTER, path 3
# GATHER_SCATTER, and path 4 GATHER_DIRECT; other paths (CONTIG_DIRECT /
# SEGMENT_DIRECT) ignore it. For GATHER_SCATTER, memcpy2d replaces the CPU-side
# scatter/gather (GPU index_select/index_copy_ still runs).
# CE_MEMCPY2D_CONFIGS defined near top of file (before first use).


def make_block_id_pattern(pattern_name, num_blocks):
    """Construct a block-id permutation that yields a specific segment count.

    All patterns are permutations of range(num_blocks), so every block is
    transferred exactly once — round-trip data integrity is preserved.

    contiguous → [0,1,...,N-1]                      (1 segment)
    few_seg    → [0..N/4-1, N/2..3N/4-1,            (4 segments)
                   N/4..N/2-1, 3N/4..N-1]
    scattered  → random permutation (fixed seed)    (N segments, >8)
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
        raise ValueError("unknown pattern: {}".format(pattern_name))
    return ids.pin_memory()


def _expected_strategy(pattern_name, cpu_layout_name, is_mla, mode,
                       is_host_to_device, num_blocks=64, threshold=8):
    """Predict which CE strategy auto-selection should pick, mirroring
    csrc/ce_transfer.cu choose_path().

    Returns (strategy, variant) where strategy is one of
    CONTIG_DIRECT / SEGMENT_DIRECT / SEGMENT_SCATTER / GATHER_SCATTER /
    GATHER_DIRECT and variant is always "".

    Key stride facts (see cpu_layout_for_mode / tp_transfer_thread_group.cpp):
      dst_phys_contig == (cpu_block_stride == chunk_size).
        MLA + LF: true (num_head=1, block_stride = chunk_size).
        MHA + LF: false (num_head=num_gpus, block_stride = num_gpus*chunk_size).
        BF: always false.
      src_phys_contig == (gpu_block_stride == chunk_size).
        Non-sharded: always contiguous.
        sharded D2H: chunk shrinks to shard -> NOT contiguous.
        sharded H2D: full chunk -> contiguous.
    GATHER_DIRECT is selected when !dst_phys && BLOCKFIRST && GPU physically
      contiguous (non-sharded). Sharded BLOCKFIRST routes to GATHER_SCATTER.
      It is CEPath enum value 4, checked before !gpu_phys_contig in choose_path.
    segment_threshold decides the SEGMENT/GATHER crossover: with a small
      threshold even few_seg (4 segments) can exceed it and route to
      GATHER_SCATTER, exactly as choose_path() does.
    """
    # MHA + LF: cpu_block_stride = num_gpus * head_dim != chunk_size = head_dim
    dst_phys = (cpu_layout_name == "LAYERFIRST") and is_mla
    is_blockfirst = (cpu_layout_name == "BLOCKFIRST")
    sharded_d2h = (is_mla and mode == "sharded" and not is_host_to_device)
    src_phys = not sharded_d2h  # only sharded D2H breaks GPU-side contiguity

    if pattern_name == "contiguous":
        num_segments = 1
    elif pattern_name == "few_seg":
        num_segments = 4  # make_block_id_pattern builds exactly 4 runs
    else:  # scattered: (near-)full permutation -> ~num_blocks segments
        num_segments = num_blocks

    # choose_path() replica -----------------------------------------------
    # GATHER_DIRECT: BLOCKFIRST + !cpu_phys_contig + GPU physically contiguous
    # (non-sharded). Sharded D2H breaks gpu_phys_contig, so the compact-staging
    # direct memcpy is invalid; those route to GATHER_SCATTER (which CPU-scatters
    # each shard to its exact offset). Covers rank0_only/all_write/layer_parallel.
    # Exception (commit eab52a2a3): bfirst + MLA + D2H + !full_block
    # (layer_parallel / rank_rotate) -> SEGMENT_SCATTER, which is 30%-8.9x
    # faster than GATHER_DIRECT for the lighter staging+scatter path.
    # is_full_block == (all layers*kv_dim transferred in one call);
    # rank0_only / all_write store the full CPU block per rank -> full_block;
    # layer_parallel / rank_rotate / sharded store a shard -> !full_block.
    # (Sharded is handled by the `not src_phys` branch below and never reaches
    # here because its gpu_phys_contig is false.)
    is_full_block = mode in ("rank0_only", "all_write")
    if not dst_phys and is_blockfirst and src_phys:
        if not is_host_to_device and is_mla and not is_full_block:
            return ("SEGMENT_SCATTER", "")
        return ("GATHER_DIRECT", "")
    # CONTIG_DIRECT: logical + physical contiguity on both sides.
    if pattern_name == "contiguous" and dst_phys and src_phys:
        return ("CONTIG_DIRECT", "")
    # Sharded D2H (LF + MLA sharded, !gpu_phys_contig) -> GATHER_SCATTER.
    if not src_phys:
        return ("GATHER_SCATTER", "")
    # LAYERFIRST or BF MHA: few segments -> SEGMENT_DIRECT or SEGMENT_SCATTER.
    if num_segments <= threshold:
        if dst_phys:
            return ("SEGMENT_DIRECT", "")
        return ("SEGMENT_SCATTER", "")
    # many segments, src contiguous -> GATHER_SCATTER
    return ("GATHER_SCATTER", "")


@pytest.mark.parametrize("data_config,is_mla,mode", CE_MODE_CONFIGS)
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("pattern", CE_PATTERNS)
@pytest.mark.parametrize("segment_threshold", CE_SEGMENT_THRESHOLDS,
                         ids=lambda t: "thr{}".format(t))
@pytest.mark.parametrize("path_opt", [False, True], ids=["baseline", "optimized"])
@pytest.mark.parametrize("enable_memcpy2d", CE_MEMCPY2D_CONFIGS, ids=["no_memcpy2d", "memcpy2d"])
def test_ce_paths_roundtrip(data_config, is_mla, cpu_layout_name, pattern,
                            path_opt, mode, segment_threshold, enable_memcpy2d):
    """CE strategy round-trip correctness via block-id patterns.

    Combos come from CE_MODE_CONFIGS: MLA sizes x {sharded, all_write,
    rank0_only}, plus non-MLA sizes once (mode is a don't-care for MHA).
    Each pattern triggers a different auto-selected CE strategy:
      contiguous -> CONTIG_DIRECT (LF) / SEGMENT_SCATTER (BF)
      few_seg    -> SEGMENT_DIRECT (LF) / SEGMENT_SCATTER (BF)
      scattered  -> GATHER_SCATTER (LF/BF non-sharded) /
                    GATHER_SCATTER per-block (sharded D2H)
    (SEGMENT_SCATTER vs GATHER_SCATTER is chosen by choose_path based on
    gpu_phys_contig; see _expected_strategy and test_ce_strategy_coverage.)
    """
    skip_if_engine_unsupported(use_ce=True)
    # BF always uses GATHER_DIRECT (checked first in choose_path), so
    # segment_threshold has no effect — skip redundant threshold sweeps.
    if cpu_layout_name == "BLOCKFIRST" and segment_threshold != 8:
        pytest.skip("BF always uses GATHER_DIRECT, threshold has no effect")
    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    if pattern == "scattered" and num_blocks <= segment_threshold:
        pytest.skip("scattered needs num_blocks > segment_threshold ({}) "
                    "to exceed it".format(segment_threshold))

    num_gpus = NUM_GPUS
    gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim,
        cpu_layout_name, is_mla, num_gpus)

    all_gpu = [make_gpu_tensors(num_layers, num_blocks, tpb,
                               heads_per_rank, head_dim, kv_dim, g)
               for g in range(num_gpus)]

    if is_mla:
        fill_gpu(all_gpu[0], 0, num_layers, num_blocks, tpb,
                 heads_per_rank, head_dim, kv_dim)
        for g in range(1, num_gpus):
            for l in range(num_layers):
                all_gpu[g][l].copy_(all_gpu[0][l])
    else:
        for g in range(num_gpus):
            fill_gpu(all_gpu[g], g, num_layers, num_blocks, tpb,
                     heads_per_rank, head_dim, kv_dim)
    sync_all(num_gpus)

    (total_cpu_blocks, cpu_stride_kv, cpu_stride_layer,
     cpu_stride_block, cpu_stride_tp) = cpu_layout_for_mode(
        cpu_layout, cpu_layout_tp, num_layers, num_blocks,
        num_heads, head_dim, tpb, is_mla, mode, num_gpus)
    cpu_kv = make_cpu_tensor(cpu_layout, num_layers, total_cpu_blocks)
    tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus,
                       gpu_layout, num_layers, ce_path_opt=path_opt,
                       ce_segment_threshold=segment_threshold,
                       is_blockfirst=(cpu_layout_name == "BLOCKFIRST"),
                       ce_enable_memcpy2d=enable_memcpy2d)

    ids = make_block_id_pattern(pattern, num_blocks)
    gpu_block_ids = ids
    cpu_block_ids = ids

    # D2H
    tp.tp_group_transfer(
        gpu_block_id_tensor=gpu_block_ids, cpu_block_id_tensor=cpu_block_ids,
        cpu_kv_stride_in_bytes=cpu_stride_kv,
        cpu_layer_stride_in_bytes=cpu_stride_layer,
        cpu_block_stride_in_bytes=cpu_stride_block,
        cpu_tp_stride_in_bytes=cpu_stride_tp,
        transfer_num_cta=4, is_host_to_device=False, use_ce_transfer=True,
        layer_id=0, layer_granularity=num_layers, is_mla=is_mla,
        mla_d2h_mode=mode,
    )
    sync_all(num_gpus)

    # Clear GPUs
    for g in range(num_gpus):
        for l in range(num_layers):
            all_gpu[g][l].zero_()
    sync_all(num_gpus)

    # H2D
    tp.tp_group_transfer(
        gpu_block_id_tensor=gpu_block_ids, cpu_block_id_tensor=cpu_block_ids,
        cpu_kv_stride_in_bytes=cpu_stride_kv,
        cpu_layer_stride_in_bytes=cpu_stride_layer,
        cpu_block_stride_in_bytes=cpu_stride_block,
        cpu_tp_stride_in_bytes=cpu_stride_tp,
        transfer_num_cta=4, is_host_to_device=True, use_ce_transfer=True,
        layer_id=0, layer_granularity=num_layers, is_mla=is_mla,
        mla_d2h_mode=mode,
    )
    sync_all(num_gpus)

    # Verify round-trip data integrity
    expected_gpu = 0 if is_mla else None
    for g in range(num_gpus):
        src_g = expected_gpu if expected_gpu is not None else g
        for layer in [0, num_layers // 2, num_layers - 1]:
            for block in [0, num_blocks // 2, num_blocks - 1]:
                for kv in range(kv_dim):
                    for hd_idx in [0, head_dim - 1]:
                        exp = expected_val(src_g, layer, block, 0, hd_idx, kv)
                        act = all_gpu[g][layer][kv, block, 0, 0, hd_idx].item()
                        assert abs(act - exp) < 1e-3, \
                            "CE path round-trip mismatch: pattern={} layout={} " \
                            "gpu={} layer={} block={} kv={} hd={}: " \
                            "expected={:.6f} got={:.6f}".format(
                                pattern, cpu_layout_name, g, layer, block,
                                kv, hd_idx, exp, act)

    del tp


@pytest.mark.parametrize("data_config,is_mla,mode", CE_MODE_CONFIGS)
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("pattern", CE_PATTERNS)
@pytest.mark.parametrize("segment_threshold", CE_SEGMENT_THRESHOLDS,
                         ids=lambda t: "thr{}".format(t))
@pytest.mark.parametrize("path_opt", [False, True], ids=["baseline", "optimized"])
@pytest.mark.parametrize("notify_mode", ["polling"], ids=["polling"])
@pytest.mark.parametrize("layer_granularity", [1, None], ids=["lg1", "lg_all"])
@pytest.mark.parametrize("enable_memcpy2d", CE_MEMCPY2D_CONFIGS, ids=["no_memcpy2d", "memcpy2d"])
def test_ce_paths_layerwise_h2d(data_config, is_mla, cpu_layout_name, pattern,
                                path_opt, mode, segment_threshold,
                                notify_mode, layer_granularity, enable_memcpy2d):
    """CE strategy correctness for LayerwiseTransferGroup H2D.

    Uses TPTransferThreadGroup D2H (already verified correct) to prepare
    CPU data, then LayerwiseTransferGroup H2D to read it back with the
    same block-id pattern.  Verifies that the layerwise CE strategy produces
    identical results to the TP-group CE strategy.

    Combos come from CE_MODE_CONFIGS: MLA sizes x {sharded, all_write,
    rank0_only} plus non-MLA sizes once (mode is a don't-care for MHA).

    notify_mode="polling" exercises the async GATHER_SCATTER/SEGMENT_SCATTER
    path (sync=false), which was previously deadlocked by internal
    cudaStreamSynchronize. hostfunc mode is already covered by the default
    in other layerwise tests, so we only sweep polling here to avoid doubling
    the test count.
    """
    skip_if_engine_unsupported(use_ce=True)
    # BF always uses GATHER_DIRECT (checked first in choose_path), so
    # segment_threshold has no effect — skip redundant threshold sweeps.
    if cpu_layout_name == "BLOCKFIRST" and segment_threshold != 8:
        pytest.skip("BF always uses GATHER_DIRECT, threshold has no effect")
    # memcpy2d applies to H2D as well (symmetric to D2H): when the selected
    # path is SEGMENT_SCATTER, GATHER_SCATTER, or GATHER_DIRECT and
    # enable_memcpy2d=True, H2D goes through the cudaMemcpy2DAsync branch.
    # Other paths (CONTIG_DIRECT / SEGMENT_DIRECT) do not consult
    # enable_memcpy2d, so their behavior is unchanged.
    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    if pattern == "scattered" and num_blocks <= segment_threshold:
        pytest.skip("scattered needs num_blocks > segment_threshold ({}) "
                    "to exceed it".format(segment_threshold))

    num_gpus = NUM_GPUS
    gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim,
        cpu_layout_name, is_mla, num_gpus)

    # Fill GPU with deterministic data
    all_gpu = [make_gpu_tensors(num_layers, num_blocks, tpb,
                               heads_per_rank, head_dim, kv_dim, g)
               for g in range(num_gpus)]
    if is_mla:
        fill_gpu(all_gpu[0], 0, num_layers, num_blocks, tpb,
                 heads_per_rank, head_dim, kv_dim)
        for g in range(1, num_gpus):
            for l in range(num_layers):
                all_gpu[g][l].copy_(all_gpu[0][l])
    else:
        for g in range(num_gpus):
            fill_gpu(all_gpu[g], g, num_layers, num_blocks, tpb,
                     heads_per_rank, head_dim, kv_dim)
    sync_all(num_gpus)

    (total_blocks, cpu_stride_kv, cpu_stride_layer,
     cpu_stride_block, cpu_stride_tp) = cpu_layout_for_mode(
        cpu_layout, cpu_layout_tp, num_layers, num_blocks,
        num_heads, head_dim, tpb, is_mla, mode, num_gpus)
    cpu_kv = make_cpu_tensor(cpu_layout, num_layers, total_blocks)
    ids = make_block_id_pattern(pattern, num_blocks)
    chunk_size = gpu_layout.get_chunk_size() * ES

    # Step 1: D2H via TPTransferThreadGroup (prepare CPU data)
    tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus,
                       gpu_layout, num_layers,
                       is_blockfirst=(cpu_layout_name == "BLOCKFIRST"),
                       ce_enable_memcpy2d=enable_memcpy2d)
    tp.tp_group_transfer(
        gpu_block_id_tensor=ids, cpu_block_id_tensor=ids,
        cpu_kv_stride_in_bytes=cpu_stride_kv,
        cpu_layer_stride_in_bytes=cpu_stride_layer,
        cpu_block_stride_in_bytes=cpu_stride_block,
        cpu_tp_stride_in_bytes=cpu_stride_tp,
        transfer_num_cta=4, is_host_to_device=False, use_ce_transfer=True,
        layer_id=0, layer_granularity=num_layers, is_mla=is_mla,
        mla_d2h_mode=mode,
    )
    sync_all(num_gpus)
    del tp

    # Clear GPUs
    for g in range(num_gpus):
        for l in range(num_layers):
            all_gpu[g][l].zero_()
    sync_all(num_gpus)

    # Step 2: H2D via LayerwiseTransferGroup (test target). path_opt selects
    # baseline (PER_BLOCK) vs optimized (CONTIG_DIRECT / SEGMENT_DIRECT /
    # SEGMENT_SCATTER / GATHER_SCATTER) on the H2D path.
    # (Step 1 above intentionally keeps default config -- it only prepares the
    # reference CPU data, the swept dims apply to this H2D test target.)
    layerwise_h2d_readback(
        all_gpu, cpu_kv, num_gpus, gpu_layout, num_layers, ids,
        cpu_stride_kv, cpu_stride_layer, cpu_stride_block, cpu_stride_tp,
        chunk_size, is_mla, mode,
        ce_path_opt=path_opt,
        ce_segment_threshold=segment_threshold, notify_mode=notify_mode,
        layer_granularity=layer_granularity,
        is_blockfirst=(cpu_layout_name == "BLOCKFIRST"),
        enable_memcpy2d=enable_memcpy2d)

    # Verify GPU data == original
    expected_gpu = 0 if is_mla else None
    for g in range(num_gpus):
        src_g = expected_gpu if expected_gpu is not None else g
        for layer in [0, num_layers // 2, num_layers - 1]:
            for block in [0, num_blocks // 2, num_blocks - 1]:
                for kv in range(kv_dim):
                    for hd_idx in [0, head_dim - 1]:
                        exp = expected_val(src_g, layer, block, 0, hd_idx, kv)
                        act = all_gpu[g][layer][kv, block, 0, 0, hd_idx].item()
                        assert abs(act - exp) < 1e-3, \
                            "CE layerwise H2D mismatch: pattern={} layout={} " \
                            "notify={} gpu={} layer={} block={} kv={} hd={}: " \
                            "expected={:.6f} got={:.6f}".format(
                                pattern, cpu_layout_name, notify_mode, g, layer,
                                block, kv, hd_idx, exp, act)


def _strategy_matrix():
    """Enumerate (threshold, pattern, layout, is_mla, mode, direction, size) ->
    (strategy, variant) over exactly the swept parametrize space, so this
    matches what test_ce_paths_roundtrip / _layerwise_h2d actually exercise
    (including the scattered-skip-when-num_blocks<=threshold rule).

    Returns a list of (label, strategy, variant) rows.
    """
    rows = []
    layouts = ["LAYERFIRST", "BLOCKFIRST"]
    # (is_mla, mode) combos as produced by CE_MODE_CONFIGS.
    mode_combos = [(True, "sharded"), (True, "all_write"),
                   (True, "rank0_only"), (True, "layer_parallel"),
                   (True, "rank_rotate"), (False, "sharded")]
    # Representative block counts from the size matrix: a small one (skips
    # scattered at threshold=8) and a large one.
    block_counts = [4, 64]
    for threshold in CE_SEGMENT_THRESHOLDS:
        for num_blocks in block_counts:
            for pattern in CE_PATTERNS:
                # Mirror the runtime skip: scattered needs > threshold segments.
                if pattern == "scattered" and num_blocks <= threshold:
                    continue
                for layout in layouts:
                    for is_mla, mode in mode_combos:
                        for is_h2d in (False, True):
                            strat, variant = _expected_strategy(
                                pattern, layout, is_mla, mode, is_h2d,
                                num_blocks=num_blocks, threshold=threshold)
                            tag = "mla_{}".format(mode) if is_mla else "non_mla"
                            label = ("thr{:<2d} nb{:<3d} {:<10s} {:<10s} "
                                     "{:<12s} {}").format(
                                threshold, num_blocks, pattern, layout, tag,
                                "h2d" if is_h2d else "d2h")
                            rows.append((label, strat, variant))
    return rows


# designated_rank test: rank0_only with a non-zero designated rank

@pytest.mark.parametrize("data_config", [pytest.param((4, 8, 16, 1, 512), id="ds3-mini")])
@pytest.mark.parametrize("designated_rank", list(range(NUM_GPUS)))
def test_mla_designated_rank_d2h(data_config, designated_rank):
    """D2H with rank0_only + designated_rank=X -> H2D -> verify.

    Exercises the designated_rank parameter: only the designated GPU
    performs D2H, then all GPUs read back via layerwise H2D.
    """
    skip_if_engine_unsupported(use_ce=True)
    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    num_gpus = NUM_GPUS
    gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim, "BLOCKFIRST", True, num_gpus)
    all_gpu = [make_gpu_tensors(num_layers, num_blocks, tpb, heads_per_rank, head_dim, kv_dim, g) for g in range(num_gpus)]
    fill_gpu(all_gpu[0], 0, num_layers, num_blocks, tpb, heads_per_rank, head_dim, kv_dim)
    for g in range(1, num_gpus):
        for l in range(num_layers):
            all_gpu[g][l].copy_(all_gpu[0][l])
    sync_all(num_gpus)
    cpu_stride_block = cpu_layout.get_block_stride() * ES
    cpu_stride_tp = cpu_stride_block // num_gpus
    cpu_kv = make_cpu_tensor(cpu_layout, num_layers, num_blocks)
    tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus, gpu_layout, num_layers, is_blockfirst=True, is_mla=True)
    ids = block_ids(num_blocks)
    tp.tp_group_transfer(
        gpu_block_id_tensor=ids, cpu_block_id_tensor=ids,
        cpu_kv_stride_in_bytes=cpu_layout_tp.get_kv_stride() * ES,
        cpu_layer_stride_in_bytes=cpu_layout_tp.get_layer_stride() * ES,
        cpu_block_stride_in_bytes=cpu_stride_block,
        cpu_tp_stride_in_bytes=cpu_stride_tp,
        transfer_num_cta=4, is_host_to_device=False, use_ce_transfer=True,
        layer_id=0, layer_granularity=num_layers, is_mla=True,
        mla_d2h_mode="rank0_only", designated_rank=designated_rank)
    sync_all(num_gpus)
    del tp
    for g in range(num_gpus):
        for l in range(num_layers):
            all_gpu[g][l].zero_()
    sync_all(num_gpus)
    lw = make_layerwise_group(cpu_kv, all_gpu, num_gpus, gpu_layout, num_layers, is_blockfirst=True, is_mla=True)
    empty_ids = torch.empty(0, dtype=torch.int64)
    lw.layerwise_transfer(
        ssd_block_ids=empty_ids,
        cpu_block_ids_d2h=empty_ids,
        ssd_layer_stride_in_bytes=0,
        ssd_kv_stride_in_bytes=0,
        num_blocks_per_file=0,
        round_robin=0,
        num_threads_per_device=0,
        gpu_block_id_tensor=ids,
        cpu_block_id_tensor=ids,
        cpu_kv_stride_in_bytes=cpu_layout_tp.get_kv_stride() * ES,
        cpu_layer_stride_in_bytes=cpu_layout_tp.get_layer_stride() * ES,
        cpu_block_stride_in_bytes=cpu_stride_block,
        cpu_chunk_size_in_bytes=gpu_layout.get_chunk_size() * ES,
        h2d_cpu_kv_stride_in_bytes=cpu_layout_tp.get_kv_stride() * ES,
        h2d_cpu_layer_stride_in_bytes=cpu_layout_tp.get_layer_stride() * ES,
        cpu_tp_stride_in_bytes=cpu_stride_tp,
        transfer_cta_num=4,
        use_ce_transfer=True,
        num_layers=num_layers,
        layer_granularity=1,
        is_mla=True,
        counter_id=0,
        swa_h2d_src=empty_ids,
        swa_h2d_dst=empty_ids,
        swa_disk2h_src=empty_ids,
        swa_disk2h_dst=empty_ids,
        mla_d2h_mode="rank0_only",
        notify_mode="hostfunc",
    )
    sync_all(num_gpus)
    spot_check_gpu(all_gpu, 0, num_gpus, num_layers, num_blocks, tpb, head_dim, kv_dim, label=f"designated={designated_rank}")
    del lw


def test_ce_strategy_coverage():
    """Assert the swept parametrize space covers every optimized strategy,
    and print the selection matrix.

    This is still an analytical mapping (mirrors choose_path); it does not
    introspect the C++ choice at runtime — that would need the engine to
    expose a path counter. But it guarantees the test suite is not
    silently skipping a whole strategy.
    """
    skip_if_engine_unsupported(use_ce=True)
    rows = _strategy_matrix()

    print("\n  CE strategy selection matrix "
          "(pattern / layout / mode / dir -> strategy):")
    print("  " + "-" * 72)
    for label, strat, variant in rows:
        shown = strat + (":" + variant if variant else "")
        print("  {}  ->  {}".format(label, shown))

    strategies = {s for _, s, _ in rows}

    for required in ("CONTIG_DIRECT", "SEGMENT_DIRECT",
                     "SEGMENT_SCATTER", "GATHER_SCATTER"):
        assert required in strategies, \
            "no swept case exercises strategy {} (covered: {})".format(
                required, sorted(strategies))

    # GATHER_DIRECT (CEPath=4) is checked first in choose_path.
    # Verify it is exercised by the swept space.
    assert "GATHER_DIRECT" in strategies, \
        "no swept case exercises GATHER_DIRECT strategy"


# NT store and gather multi-thread correctness tests
#
# gather_threads controls parallel CPU gather/scatter (CopyPool thread_local):
#   0 = disable (single-thread fallback), 1/4/8 = thread count
# gather_nt controls non-temporal (streaming) stores (AVX-512/AVX2):
#   True = use NT stores, False = regular stores
# Both are correctness-transparent optimizations on the CE path.

GATHER_NT_MODE_CONFIGS = [
    pytest.param((4, 8, 16, 1, 512), True, "sharded", id="mla_sharded"),
    pytest.param((4, 8, 16, 1, 512), True, "all_write", id="mla_all_write"),
    pytest.param((4, 8, 16, 1, 512), True, "rank0_only", id="mla_rank0_only"),
    pytest.param((4, 8, 16, 8, 128), False, "sharded", id="mha"),
]

GATHER_THREADS_VALUES = [0, 1, 4, 8]
GATHER_NT_VALUES = [True, False]


@pytest.mark.parametrize("data_config,is_mla,mode", GATHER_NT_MODE_CONFIGS)
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("gather_threads", GATHER_THREADS_VALUES,
                         ids=["gt0", "gt1", "gt4", "gt8"])
@pytest.mark.parametrize("gather_nt", GATHER_NT_VALUES,
                         ids=["nt_on", "nt_off"])
def test_gather_nt_roundtrip(data_config, is_mla, cpu_layout_name, mode,
                             gather_threads, gather_nt):
    """CE gather threads and NT store round-trip correctness.

    Sweeps gather_threads (0=disable, 1, 4, 8) and gather_nt (on/off)
    to verify parallel CPU gather/scatter and non-temporal stores produce
    correct results across MLA modes and CPU layouts.
    """
    skip_if_engine_unsupported(use_ce=True)
    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    num_gpus = NUM_GPUS

    gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim,
        cpu_layout_name, is_mla, num_gpus)

    all_gpu = [make_gpu_tensors(num_layers, num_blocks, tpb,
                               heads_per_rank, head_dim, kv_dim, g)
               for g in range(num_gpus)]

    if is_mla:
        fill_gpu(all_gpu[0], 0, num_layers, num_blocks, tpb,
                 heads_per_rank, head_dim, kv_dim)
        for g in range(1, num_gpus):
            for l in range(num_layers):
                all_gpu[g][l].copy_(all_gpu[0][l])
    else:
        for g in range(num_gpus):
            fill_gpu(all_gpu[g], g, num_layers, num_blocks, tpb,
                     heads_per_rank, head_dim, kv_dim)
    sync_all(num_gpus)

    (total_cpu_blocks, cpu_stride_kv, cpu_stride_layer,
     cpu_stride_block, cpu_stride_tp) = cpu_layout_for_mode(
        cpu_layout, cpu_layout_tp, num_layers, num_blocks,
        num_heads, head_dim, tpb, is_mla, mode, num_gpus)
    cpu_kv = make_cpu_tensor(cpu_layout, num_layers, total_cpu_blocks)
    tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus,
                       gpu_layout, num_layers,
                       is_blockfirst=(cpu_layout_name == "BLOCKFIRST"),
                       ce_gather_threads=gather_threads,
                       ce_gather_nt=gather_nt)

    ids = block_ids(num_blocks)
    gpu_block_ids = ids
    cpu_block_ids = ids

    # D2H
    tp.tp_group_transfer(
        gpu_block_id_tensor=gpu_block_ids, cpu_block_id_tensor=cpu_block_ids,
        cpu_kv_stride_in_bytes=cpu_stride_kv,
        cpu_layer_stride_in_bytes=cpu_stride_layer,
        cpu_block_stride_in_bytes=cpu_stride_block,
        cpu_tp_stride_in_bytes=cpu_stride_tp,
        transfer_num_cta=4, is_host_to_device=False, use_ce_transfer=True,
        layer_id=0, layer_granularity=num_layers, is_mla=is_mla,
        mla_d2h_mode=mode,
    )
    sync_all(num_gpus)

    # Clear GPUs
    for g in range(num_gpus):
        for l in range(num_layers):
            all_gpu[g][l].zero_()
    sync_all(num_gpus)

    # H2D
    tp.tp_group_transfer(
        gpu_block_id_tensor=gpu_block_ids, cpu_block_id_tensor=cpu_block_ids,
        cpu_kv_stride_in_bytes=cpu_stride_kv,
        cpu_layer_stride_in_bytes=cpu_stride_layer,
        cpu_block_stride_in_bytes=cpu_stride_block,
        cpu_tp_stride_in_bytes=cpu_stride_tp,
        transfer_num_cta=4, is_host_to_device=True, use_ce_transfer=True,
        layer_id=0, layer_granularity=num_layers, is_mla=is_mla,
        mla_d2h_mode=mode,
    )
    sync_all(num_gpus)

    # Verify round-trip data integrity
    expected_gpu = 0 if is_mla else None
    for g in range(num_gpus):
        src_g = expected_gpu if expected_gpu is not None else g
        for layer in [0, num_layers - 1]:
            for block in [0, num_blocks - 1]:
                for kv in range(kv_dim):
                    for hd_idx in [0, head_dim - 1]:
                        exp = expected_val(src_g, layer, block, 0, hd_idx, kv)
                        act = all_gpu[g][layer][kv, block, 0, 0, hd_idx].item()
                        assert abs(act - exp) < 1e-3, \
                            "gather/NT round-trip mismatch: gt={} nt={} " \
                            "layout={} gpu={} layer={} block={} kv={} hd={}: " \
                            "expected={:.6f} got={:.6f}".format(
                                gather_threads, gather_nt, cpu_layout_name,
                                g, layer, block, kv, hd_idx, exp, act)

    del tp


@pytest.mark.parametrize("data_config,is_mla,mode", GATHER_NT_MODE_CONFIGS)
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("gather_threads", GATHER_THREADS_VALUES,
                         ids=["gt0", "gt1", "gt4", "gt8"])
@pytest.mark.parametrize("gather_nt", GATHER_NT_VALUES,
                         ids=["nt_on", "nt_off"])
def test_gather_nt_layerwise_h2d(data_config, is_mla, cpu_layout_name, mode,
                                 gather_threads, gather_nt):
    """CE gather threads and NT store correctness for layerwise H2D.

    TP-group CE D2H prepares CPU, LayerwiseTransferGroup CE H2D reads back
    with swept gather_threads and gather_nt settings.
    """
    skip_if_engine_unsupported(use_ce=True)
    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    num_gpus = NUM_GPUS

    gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim,
        cpu_layout_name, is_mla, num_gpus)

    all_gpu = [make_gpu_tensors(num_layers, num_blocks, tpb,
                               heads_per_rank, head_dim, kv_dim, g)
               for g in range(num_gpus)]
    if is_mla:
        fill_gpu(all_gpu[0], 0, num_layers, num_blocks, tpb,
                 heads_per_rank, head_dim, kv_dim)
        for g in range(1, num_gpus):
            for l in range(num_layers):
                all_gpu[g][l].copy_(all_gpu[0][l])
    else:
        for g in range(num_gpus):
            fill_gpu(all_gpu[g], g, num_layers, num_blocks, tpb,
                     heads_per_rank, head_dim, kv_dim)
    sync_all(num_gpus)

    (total_blocks, cpu_stride_kv, cpu_stride_layer,
     cpu_stride_block, cpu_stride_tp) = cpu_layout_for_mode(
        cpu_layout, cpu_layout_tp, num_layers, num_blocks,
        num_heads, head_dim, tpb, is_mla, mode, num_gpus)
    cpu_kv = make_cpu_tensor(cpu_layout, num_layers, total_blocks)
    ids = block_ids(num_blocks)
    chunk_size = gpu_layout.get_chunk_size() * ES

    # D2H via TP-group (default config)
    tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus,
                       gpu_layout, num_layers,
                       is_blockfirst=(cpu_layout_name == "BLOCKFIRST"))
    tp.tp_group_transfer(
        gpu_block_id_tensor=ids, cpu_block_id_tensor=ids,
        cpu_kv_stride_in_bytes=cpu_stride_kv,
        cpu_layer_stride_in_bytes=cpu_stride_layer,
        cpu_block_stride_in_bytes=cpu_stride_block,
        cpu_tp_stride_in_bytes=cpu_stride_tp,
        transfer_num_cta=4, is_host_to_device=False, use_ce_transfer=True,
        layer_id=0, layer_granularity=num_layers, is_mla=is_mla,
        mla_d2h_mode=mode,
    )
    sync_all(num_gpus)
    del tp

    for g in range(num_gpus):
        for l in range(num_layers):
            all_gpu[g][l].zero_()
    sync_all(num_gpus)

    # H2D via layerwise with swept gather_threads and gather_nt
    layerwise_h2d_readback(
        all_gpu, cpu_kv, num_gpus, gpu_layout, num_layers, ids,
        cpu_stride_kv, cpu_stride_layer, cpu_stride_block, cpu_stride_tp,
        chunk_size, is_mla, mode,
        is_blockfirst=(cpu_layout_name == "BLOCKFIRST"),
        ce_gather_threads=gather_threads,
        ce_gather_nt=gather_nt)

    expected_gpu = 0 if is_mla else None
    for g in range(num_gpus):
        src_g = expected_gpu if expected_gpu is not None else g
        for layer in [0, num_layers - 1]:
            for block in [0, num_blocks - 1]:
                for kv in range(kv_dim):
                    for hd_idx in [0, head_dim - 1]:
                        exp = expected_val(src_g, layer, block, 0, hd_idx, kv)
                        act = all_gpu[g][layer][kv, block, 0, 0, hd_idx].item()
                        assert abs(act - exp) < 1e-3, \
                            "gather/NT layerwise H2D mismatch: gt={} nt={} " \
                            "layout={} gpu={} layer={} block={} kv={} hd={}: " \
                            "expected={:.6f} got={:.6f}".format(
                                gather_threads, gather_nt, cpu_layout_name,
                                g, layer, block, kv, hd_idx, exp, act)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])