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

from flexkv.c_ext import TPTransferThreadGroup
# GDS is a build option (ENABLE_GDS), not a runtime one: a c_ext built without
# cuFile has no such symbol, and importing it at module scope takes the whole
# file's collection down with it -- including the ~40 tests that need no GDS.
try:
    from flexkv.c_ext import TPGDSTransferThreadGroup
except ImportError:  # pragma: no cover - depends on how c_ext was built
    TPGDSTransferThreadGroup = None
from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.storage.allocator import SSDAllocator
from flexkv.transfer.region_batch import (
    RegionSpec,
    build_region_batch,
    make_requests,
    rank_share_mode,
)

from eventfd_probe import Fds


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
            num_head=1, head_size=16, kv_dim=1, num_kv_heads=1)
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
            layer_id=0, layer_granularity=1, kv_dim=1, num_kv_heads=1,
            kv_shared_across_ranks_mode="sharded")
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
    # DeepSeek-V3.2 / GLM-5.2 cache the *whole* MLA latent: kv_lora_rank 512 +
    # qk_rope_head_dim 64 = 576. Worth its own entries rather than reusing 512:
    # every other size here is a power of two, so a chunk size that is only
    # correct under 128B alignment passes all of them and fails these.
    pytest.param((61, 256, 16, 1, 576), id="dsv32"),      # DeepSeek-V3.2-Exp
    pytest.param((78, 256, 16, 1, 576), id="glm52"),      # GLM-5.2 (GlmMoeDsa)
]
MHA_SIZES = [
    # Llama-3 scale: 32 layers, kv_heads=8, head_dim=128
    pytest.param((4, 8, 16, 8, 128), id="llama3-mini"),
    pytest.param((32, 64, 16, 8, 128), id="llama3-8b"),
    pytest.param((80, 256, 16, 8, 128), id="llama3-70b"),
    pytest.param((2, 4, 1, 8, 128), id="edge"),            # tpb=1 edge case
    pytest.param((4, 4, 16, 16, 128), id="16head"),       # 16 heads variant
    # Qwen3-8B: 36 layers, 8 kv heads, head_dim 128 (GQA, full attention).
    pytest.param((36, 64, 16, 8, 128), id="qwen3-8b"),
    # Qwen3.5-397B-A17B: only the 15 full_attention layers cache; head_dim 256
    # is the widest head here, so it is the only case where one (layer, kv)
    # chunk exceeds 8 KiB per rank.
    pytest.param((15, 64, 16, 4, 256), id="qwen3.5-397b"),
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
                 cpu_layout_name, kv_dim, num_kv_heads, tp_size):
    """Create GPU and CPU KVCacheLayout objects matching production conventions.

    GPU: LAYERFIRST, per-rank heads for multi-head (num_kv_heads > 1).
    CPU: specified layout, full heads.
    For multi-head + BLOCKFIRST: CPU strides use div_head(tp_size).

    Returns (gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank).
    """
    # Multi-head models (num_kv_heads > 1) shard heads across TP ranks, so
    # num_heads must be divisible by tp_size. Single-head models (num_kv_heads
    # == 1, MLA) share the single head across ranks without splitting.
    if num_kv_heads > 1:
        assert num_heads % tp_size == 0, \
            f"multi-head requires num_heads % tp_size == 0, got {num_heads} % {tp_size}"

    heads_per_rank = num_heads if num_kv_heads == 1 else num_heads // tp_size

    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=tpb, num_head=heads_per_rank,
        head_size=head_dim, kv_dim=kv_dim, num_kv_heads=num_kv_heads)

    cpu_layout = KVCacheLayout(
        type=KVCacheLayoutType[cpu_layout_name.upper()],
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=tpb, num_head=num_heads,
        head_size=head_dim, kv_dim=kv_dim, num_kv_heads=num_kv_heads)

    if num_kv_heads > 1 and cpu_layout.type == KVCacheLayoutType.BLOCKFIRST:
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
        kv_dim=cpu_layout.kv_dim, num_kv_heads=cpu_layout.num_kv_heads)
    return torch.zeros(tuple(layout.kv_shape), dtype=DTYPE, pin_memory=True)


def cpu_layout_for_mode(cpu_layout, cpu_layout_tp, num_layers, num_blocks,
                        num_heads, head_dim, tpb, kv_dim, num_kv_heads, mode, num_gpus):
    """Resolve CPU buffer size + kv/layer/block/tp strides for a shared-across-ranks D2H mode.

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
    if num_kv_heads == 1 and mode == "all_write":
        total_cpu_blocks = num_blocks * num_gpus
        layout_for_strides = KVCacheLayout(
            type=cpu_layout.type,
            num_layer=num_layers, num_block=total_cpu_blocks,
            tokens_per_block=tpb, num_head=num_heads,
            head_size=head_dim, kv_dim=kv_dim, num_kv_heads=num_kv_heads)
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
                  kv_dim=None,
                  num_kv_heads=None):
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
    if kv_dim is None:
        kv_dim = gpu_layout.kv_dim
    if num_kv_heads is None:
        num_kv_heads = gpu_layout.num_kv_heads
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
        num_kv_heads=num_kv_heads,
    )


def make_layerwise_group(cpu_tensor, all_gpu, num_gpus, gpu_layout, num_layers,
                         ce_segment_threshold=None,
                         ce_path_opt=None,
                         ce_enable_memcpy2d=None,
                         ce_gather_threads=None,
                         ce_gather_nt=None,
                         is_blockfirst=None,
                         kv_dim=None,
                         num_kv_heads=None,
                         cpu_kv_stride=None,
                         cpu_layer_stride=None,
                         cpu_block_stride=None,
                         cpu_tp_stride=None):
    """A one-region ``RegionBatchGroup``, standing in for the old layerwise one.

    Per-layer completion is no longer a group of its own: it is
    ``RegionBatchGroup::submit_layerwise``, so the group these tests build is
    the same one the whole-block tests build.  The old ``LayerwiseTransferGroup``
    carried the CPU strides in the *call*; ``RegionDesc`` carries them in the
    region, which is why they are constructor arguments here.  They default to
    ``gpu_layout``-derived values only so callers that never varied them stay
    short.

    GPU chunk_size does NOT include kv_dim (same as tp_group). The group is used
    CPU->GPU only here -- the SSD read is a separate DISK2H op the engine
    schedules before it.  CE config defaults from GLOBAL_CONFIG_FROM_ENV (same
    as production).
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
    if kv_dim is None:
        kv_dim = gpu_layout.kv_dim
    if num_kv_heads is None:
        num_kv_heads = gpu_layout.num_kv_heads

    spec = RegionSpec(
        name="kv",
        cpu_ptr=cpu_tensor.data_ptr(),
        cpu_kv_stride=cpu_kv_stride,
        cpu_layer_stride=cpu_layer_stride,
        cpu_block_stride=cpu_block_stride,
        cpu_tp_stride=cpu_tp_stride,
        gpu_block_ptrs_flat=[all_gpu[g][l].data_ptr()
                             for g in range(num_gpus)
                             for l in range(num_layers)],
        num_tensors_per_gpu=num_layers,
        gpu_kv_strides=[gpu_layout.get_kv_stride() * ES] * num_gpus,
        gpu_block_strides=[gpu_layout.get_block_stride() * ES] * num_gpus,
        gpu_layer_strides=[gpu_layout.get_layer_stride() * ES] * num_gpus,
        gpu_chunk_sizes=[gpu_layout.get_chunk_size() * ES] * num_gpus,
        num_layers=num_layers,
        kv_dim=kv_dim,
        num_kv_heads=num_kv_heads,
    )
    return build_region_batch(
        [spec], list(range(num_gpus)),
        ce_segment_threshold=ce_segment_threshold,
        ce_path_opt=ce_path_opt,
        ce_enable_memcpy2d=ce_enable_memcpy2d,
        is_blockfirst=is_blockfirst,
        num_kv_heads=num_kv_heads,
        ce_gather_threads=ce_gather_threads,
        ce_gather_nt=ce_gather_nt,
    )


def layerwise_requests(num_layers, ids, use_ce, mode, layer_granularity=1,
                       designated_rank=0):
    """One request per layer chunk, each tagged with the milestone it closes.

    ``layer_granularity`` is what the old call took: layers go out in chunks of
    that size, and the chunk's first layer is the milestone, so a consumer is
    told about a chunk only once every layer in it is on the stream.  The
    production contract pins this to 1 (``CompletionContract.PER_LAYER``);
    larger values exist here only because the CE sweep varies them to exercise
    different launch sizes.
    """
    requests = []
    for start in range(0, num_layers, layer_granularity):
        count = min(layer_granularity, num_layers - start)
        req = make_requests(
            1, ids, ids, True,
            transfer_num_cta=4, use_ce_transfer=use_ce,
            layer_id=start, layer_granularity=count,
            share_mode=rank_share_mode(mode),
            designated_rank=designated_rank)[0]
        req.milestone_layer = start
        requests.append(req)
    return requests


def layerwise_h2d_readback(all_gpu, cpu_kv, num_gpus, gpu_layout, num_layers,
                           ids, cpu_stride_kv, cpu_stride_layer,
                           cpu_stride_block, cpu_stride_tp, chunk_size,
                           kv_dim, num_kv_heads, mode, ce_path_opt=None,
                           ce_segment_threshold=None,
                           notify_mode="hostfunc", layer_granularity=None,
                           is_blockfirst=None,
                           enable_memcpy2d=None,
                           ce_gather_threads=None,
                           ce_gather_nt=None,
                           use_ce=True):
    """Run a single per-layer H2D, reading `cpu_kv` back into `all_gpu`.

    use_ce selects the CE adaptive path (True) or the baseline PER_BLOCK cuda
    kernel (False). Shared by the CE/cuda layerwise roundtrip twins and the
    notify-mode test.

    notify_mode: "hostfunc" (default, uses CUDA hostfunc callback) or
    "polling" (uses a CPU polling thread that queries cudaEventQuery per
    batch).  Polling mode exercises the async GATHER_SCATTER/SEGMENT_SCATTER
    path (sync=false) that was previously deadlocked.

    Real eventfds are registered on every call, not just the notify-mode test.
    ``LayerNotifier::record`` returns immediately when no table is registered,
    so an empty table would mean neither mode is actually exercised anywhere --
    the marker/callback machinery would be compiled out of every one of these
    round-trips, and only the copies would be under test.
    """
    group = make_layerwise_group(cpu_kv, all_gpu, num_gpus,
                                 gpu_layout, num_layers,
                                 ce_path_opt=ce_path_opt,
                                 ce_segment_threshold=ce_segment_threshold,
                                 ce_gather_threads=ce_gather_threads,
                                 ce_gather_nt=ce_gather_nt,
                                 is_blockfirst=is_blockfirst,
                                 ce_enable_memcpy2d=enable_memcpy2d,
                                 kv_dim=kv_dim, num_kv_heads=num_kv_heads,
                                 cpu_kv_stride=cpu_stride_kv,
                                 cpu_layer_stride=cpu_stride_layer,
                                 cpu_block_stride=cpu_stride_block,
                                 cpu_tp_stride=cpu_stride_tp)
    requests = layerwise_requests(
        num_layers, ids, use_ce, mode,
        layer_granularity=1 if layer_granularity is None else layer_granularity)
    milestones = [r.milestone_layer for r in requests]
    with Fds(num_counters=1, tp_size=num_gpus, num_layers=num_layers) as fds:
        group.set_layer_eventfds(fds.tensor(), num_gpus, num_layers,
                                 notify_mode)
        group.submit_layerwise(requests, [], 0)
        ok, err = group.wait_layer_completion(120.0)
        assert ok, f"layerwise H2D did not complete: {err}"
        sync_all(num_gpus)
        for layer in milestones:
            assert fds.units(layer) == [1] * num_gpus, (
                f"notify={notify_mode}: layer {layer} did not get exactly one "
                f"semaphore unit on every rank")
        del group


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
@pytest.mark.parametrize("kv_dim", [1, 2], ids=["packed", "plain"])
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("engine_name,use_ce", ENGINES)
@pytest.mark.parametrize("enable_memcpy2d", CE_MEMCPY2D_CONFIGS, ids=["no_memcpy2d", "memcpy2d"])
def test_non_mla_roundtrip(data_config, kv_dim, cpu_layout_name, engine_name, use_ce, enable_memcpy2d):
    """Multi-head round-trip: D2H -> clear GPU -> H2D -> verify per-rank data.

    Covers packed MHA (kv_dim=1, K/V combined) and plain MHA (kv_dim=2,
    K/V separate). Both have num_kv_heads > 1, so the C++ else-branch uses
    cpu_tp_stride to place each rank's head partition at a different CPU
    offset. kv_shared_across_ranks_mode is ignored for multi-head.
    """
    skip_if_engine_unsupported(use_ce)
    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    num_gpus = NUM_GPUS
    num_kv_heads = num_heads  # data_config num_heads IS num_kv_heads

    gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim,
        cpu_layout_name, kv_dim, num_kv_heads, num_gpus)

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
        layer_id=0, layer_granularity=num_layers, kv_dim=kv_dim, num_kv_heads=num_kv_heads,
        kv_shared_across_ranks_mode="sharded",  # ignored for multi-head
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
        layer_id=0, layer_granularity=num_layers, kv_dim=kv_dim, num_kv_heads=num_kv_heads,
        kv_shared_across_ranks_mode="sharded",  # ignored for multi-head
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
                            f"Multi-head round-trip mismatch: layout={cpu_layout_name} " \
                            f"gpu={g} layer={layer} block={block} kv={kv} hd={hd_idx}: " \
                            f"expected={exp:.6f} got={act:.6f}"

    del tp


# Shared-across-ranks mode tests (sharded / all_write / rank0_only)
# Covers MLA (kv_dim=1, num_kv_heads=1) and plain MHA, single head (kv_dim=2, num_kv_heads=1).

@pytest.mark.parametrize("data_config", MLA_SIZES)
@pytest.mark.parametrize("kv_dim", [1, 2], ids=["mla", "kv_sep_1h"])
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("engine_name,use_ce", ENGINES)
@pytest.mark.parametrize("mode", MLA_MODES)
@pytest.mark.parametrize("enable_memcpy2d", CE_MEMCPY2D_CONFIGS, ids=["no_memcpy2d", "memcpy2d"])
def test_mla_roundtrip_modes(data_config, kv_dim, cpu_layout_name, engine_name, use_ce, mode, enable_memcpy2d):
    """Shared-across-ranks round-trip with each D2H mode. Verifies K and V.

    Covers MLA (kv_dim=1, K/V combined) and plain MHA, single head (kv_dim=2, K/V separate).
    Both have num_kv_heads=1, so KV is shared across TP ranks and the
    kv_shared_across_ranks_mode selects the D2H placement strategy.
    """
    skip_if_engine_unsupported(use_ce)
    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    assert num_heads == 1, "MLA_SIZES must only contain num_heads=1 configs"

    num_gpus = NUM_GPUS
    num_kv_heads = num_heads  # == 1 for MLA

    gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim,
        cpu_layout_name, kv_dim, num_kv_heads, num_gpus)

    total_cpu_blocks = num_blocks * num_gpus if mode == "all_write" else num_blocks

    all_gpu = [make_gpu_tensors(num_layers, num_blocks, tpb, heads_per_rank, head_dim, kv_dim, g)
               for g in range(num_gpus)]

    # Shared across ranks: all GPUs have identical data
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
            head_size=head_dim, kv_dim=kv_dim, num_kv_heads=num_kv_heads)
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
        layer_id=0, layer_granularity=num_layers, kv_dim=kv_dim, num_kv_heads=num_kv_heads,
        kv_shared_across_ranks_mode=mode,
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
        layer_id=0, layer_granularity=num_layers, kv_dim=kv_dim, num_kv_heads=num_kv_heads,
        kv_shared_across_ranks_mode=mode,
    )
    sync_all(num_gpus)

    # Verify: all GPUs should have GPU 0's original data
    spot_check_gpu(all_gpu, 0, num_gpus, num_layers, num_blocks,
                   tpb, head_dim, kv_dim, label=f"mode={mode}")

    del tp


# tp1 (single-GPU) round-trip test
#
# All four-scenario tests above use num_gpus >= 2 (module-level skip at line 34
# requires NUM_GPUS >= 2). tp1 is an independent code path: with a single GPU
# the TP sharing logic (kv_shared_across_ranks_mode, per-rank head partitioning)
# does not engage, and cpu_tp_stride equals the full block_stride (not divided
# by num_gpus). This test exercises that path across the four kv_dim x
# num_kv_heads scenarios x cuda/ce engines. mode is "sharded" (placeholder --
# no other rank to share with).

TP1_CONFIGS = [
    # (num_layers, num_blocks, tpb, num_heads, head_dim, kv_dim, num_kv_heads)
    pytest.param((4, 8, 16, 1, 512, 1, 1), id="mla_packed"),    # MLA, kv_dim=1
    pytest.param((4, 8, 16, 1, 512, 2, 1), id="mla_kv_sep"),    # MLA, kv_dim=2
    pytest.param((4, 8, 16, 8, 128, 1, 8), id="mha_packed"),    # MHA, kv_dim=1
    pytest.param((4, 8, 16, 8, 128, 2, 8), id="mha_plain"),     # MHA, kv_dim=2
]


@pytest.mark.parametrize("data_config", TP1_CONFIGS)
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("engine_name,use_ce", ENGINES)
def test_tp1_roundtrip(data_config, cpu_layout_name, engine_name, use_ce):
    """tp1 (single-GPU) round-trip: D2H -> clear GPU -> H2D -> verify.

    Covers the four kv_dim x num_kv_heads scenarios x cuda/ce. With num_gpus=1
    the TP sharing logic is inert: cpu_tp_stride == block_stride (cpu_layout_for_mode
    divides by num_gpus=1), heads_per_rank == num_heads, and mode is a placeholder.
    Structured after test_non_mla_roundtrip / test_mla_roundtrip_modes but with
    num_gpus=1.
    """
    skip_if_engine_unsupported(use_ce)
    (num_layers, num_blocks, tpb, num_heads, head_dim,
     kv_dim, num_kv_heads) = data_config
    num_gpus = 1
    mode = "sharded"  # placeholder; no other rank to share with

    gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim,
        cpu_layout_name, kv_dim, num_kv_heads, num_gpus)

    all_gpu = [make_gpu_tensors(num_layers, num_blocks, tpb,
                                heads_per_rank, head_dim, kv_dim, 0)]
    fill_gpu(all_gpu[0], 0, num_layers, num_blocks, tpb,
             heads_per_rank, head_dim, kv_dim)
    sync_all(num_gpus)

    (total_blocks, cpu_stride_kv, cpu_stride_layer,
     cpu_stride_block, cpu_stride_tp) = cpu_layout_for_mode(
        cpu_layout, cpu_layout_tp, num_layers, num_blocks,
        num_heads, head_dim, tpb, kv_dim, num_kv_heads, mode, num_gpus)
    # tp1: cpu_tp_stride == cpu_stride_block (not divided).
    assert cpu_stride_tp == cpu_stride_block, \
        "tp1 cpu_tp_stride must equal block_stride"
    cpu_kv = make_cpu_tensor(cpu_layout, num_layers, total_blocks)
    ids = block_ids(num_blocks)

    tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus, gpu_layout, num_layers,
                       is_blockfirst=(cpu_layout_name == "BLOCKFIRST"))

    # D2H
    tp.tp_group_transfer(
        gpu_block_id_tensor=ids, cpu_block_id_tensor=ids,
        cpu_kv_stride_in_bytes=cpu_stride_kv,
        cpu_layer_stride_in_bytes=cpu_stride_layer,
        cpu_block_stride_in_bytes=cpu_stride_block,
        cpu_tp_stride_in_bytes=cpu_stride_tp,
        transfer_num_cta=4, is_host_to_device=False, use_ce_transfer=use_ce,
        layer_id=0, layer_granularity=num_layers, kv_dim=kv_dim, num_kv_heads=num_kv_heads,
        kv_shared_across_ranks_mode=mode,
    )
    sync_all(num_gpus)

    # Clear GPU
    for l in range(num_layers):
        all_gpu[0][l].zero_()
    sync_all(num_gpus)

    # H2D
    tp.tp_group_transfer(
        gpu_block_id_tensor=ids, cpu_block_id_tensor=ids,
        cpu_kv_stride_in_bytes=cpu_stride_kv,
        cpu_layer_stride_in_bytes=cpu_stride_layer,
        cpu_block_stride_in_bytes=cpu_stride_block,
        cpu_tp_stride_in_bytes=cpu_stride_tp,
        transfer_num_cta=4, is_host_to_device=True, use_ce_transfer=use_ce,
        layer_id=0, layer_granularity=num_layers, kv_dim=kv_dim, num_kv_heads=num_kv_heads,
        kv_shared_across_ranks_mode=mode,
    )
    sync_all(num_gpus)

    # Verify the single GPU recovers its own data.
    spot_check_gpu(all_gpu, 0, num_gpus, num_layers, num_blocks,
                   tpb, head_dim, kv_dim, label="tp1")

    del tp


# Layerwise H2D test with notify modes (hostfunc / polling)

LAYERWISE_NOTIFY_MODES = ["hostfunc", "polling"]


@pytest.mark.parametrize("data_config", [pytest.param((4, 8, 16, 1, 512), id="ds3-mini")])
@pytest.mark.parametrize("kv_dim", [1, 2], ids=["mla", "kv_sep_1h"])
@pytest.mark.parametrize("engine_name,use_ce", ENGINES)
@pytest.mark.parametrize("notify_mode", LAYERWISE_NOTIFY_MODES)
def test_layerwise_h2d_notify_modes(data_config, kv_dim, engine_name, use_ce, notify_mode):
    """Layerwise H2D round-trip under hostfunc / polling notify modes.

    Verifies data correctness under both notification modes.
    Covers MLA (kv_dim=1) and plain MHA, single head (kv_dim=2), both with num_kv_heads=1.
    """
    skip_if_engine_unsupported(use_ce)

    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    num_gpus = NUM_GPUS
    num_kv_heads = num_heads  # == 1

    gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim,
        "BLOCKFIRST", kv_dim, num_kv_heads, num_gpus)

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
                       kv_dim=kv_dim)
    gpu_block_ids = block_ids(num_blocks)
    cpu_block_ids = block_ids(num_blocks)
    tp.tp_group_transfer(
        gpu_block_id_tensor=gpu_block_ids, cpu_block_id_tensor=cpu_block_ids,
        cpu_kv_stride_in_bytes=cpu_stride_kv,
        cpu_layer_stride_in_bytes=cpu_stride_layer,
        cpu_block_stride_in_bytes=cpu_stride_block,
        cpu_tp_stride_in_bytes=cpu_stride_tp,
        transfer_num_cta=4, is_host_to_device=False, use_ce_transfer=use_ce,
        layer_id=0, layer_granularity=num_layers, kv_dim=kv_dim, num_kv_heads=num_kv_heads,
        kv_shared_across_ranks_mode="sharded",
    )
    sync_all(num_gpus)
    del tp

    for g in range(num_gpus):
        for l in range(num_layers):
            all_gpu[g][l].zero_()
    sync_all(num_gpus)

    # H2D via layerwise with the requested notify mode
    layerwise_h2d_readback(
        all_gpu, cpu_kv, num_gpus, gpu_layout, num_layers, gpu_block_ids,
        cpu_stride_kv, cpu_stride_layer, cpu_stride_block, cpu_stride_tp,
        gpu_layout.get_chunk_size() * ES, kv_dim, num_kv_heads, "sharded",
        notify_mode=notify_mode, is_blockfirst=True, use_ce=use_ce)

    spot_check_gpu(all_gpu, 0, num_gpus, num_layers, num_blocks,
                   tpb, head_dim, kv_dim, label=f"notify={notify_mode}")


# Round-trip tests via per-layer (submit_layerwise) H2D
#
# The layerwise path is H2D-only (no independent D2H), so these twins
# prepare the CPU reference with a verified TPTransferThreadGroup D2H, then
# read it back with layerwise H2D and check correctness. Same size matrix /
# modes / layouts as the TP-group round-trips above, so layerwise H2D is
# exercised across the full production shape space. Both cuda and ce engines
# are parametrized (layerwise_transfer accepts use_ce_transfer).
# Uses contiguous block ids (identity) like the TP round-trips; the CE-path
# tests separately sweep few_seg/scattered patterns for both groups.

@pytest.mark.parametrize("data_config", MHA_SIZES)
@pytest.mark.parametrize("kv_dim", [1, 2], ids=["packed", "plain"])
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("engine_name,use_ce", ENGINES)
def test_non_mla_roundtrip_layerwise(data_config, kv_dim, cpu_layout_name, engine_name, use_ce):
    """Multi-head: TP-group D2H prepares CPU, per-layer H2D
    reads it back; verify each rank recovers its own data.
    Covers packed MHA (kv_dim=1) and plain MHA (kv_dim=2), cuda + ce."""
    skip_if_engine_unsupported(use_ce)
    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    num_gpus = NUM_GPUS
    num_kv_heads = num_heads  # data_config num_heads IS num_kv_heads
    mode = "sharded"  # ignored for multi-head

    gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim,
        cpu_layout_name, kv_dim, num_kv_heads, num_gpus)

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
        num_heads, head_dim, tpb, kv_dim, num_kv_heads, mode, num_gpus)
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
        transfer_num_cta=4, is_host_to_device=False, use_ce_transfer=use_ce,
        layer_id=0, layer_granularity=num_layers, kv_dim=kv_dim, num_kv_heads=num_kv_heads,
        kv_shared_across_ranks_mode=mode,
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
    # of SEGMENT_SCATTER, breaking the round-trip for LAYERFIRST multi-head TP.
    layerwise_h2d_readback(
        all_gpu, cpu_kv, num_gpus, gpu_layout, num_layers, ids,
        cpu_stride_kv, cpu_stride_layer, cpu_stride_block, cpu_stride_tp,
        chunk_size, kv_dim, num_kv_heads, mode,
        is_blockfirst=(cpu_layout_name == "BLOCKFIRST"),
        use_ce=use_ce)

    for g in range(num_gpus):
        for layer in [0, num_layers - 1]:
            for block in [0, num_blocks - 1]:
                for kv in range(kv_dim):
                    for hd_idx in [0, head_dim - 1]:
                        exp = expected_val(g, layer, block, 0, hd_idx, kv)
                        act = all_gpu[g][layer][kv, block, 0, 0, hd_idx].item()
                        assert abs(act - exp) < 1e-3, \
                            "Multi-head layerwise round-trip mismatch: " \
                            "layout={} gpu={} layer={} block={} kv={} hd={}: " \
                            "expected={:.6f} got={:.6f}".format(
                                cpu_layout_name, g, layer, block, kv, hd_idx,
                                exp, act)


@pytest.mark.parametrize("data_config", MLA_SIZES)
@pytest.mark.parametrize("kv_dim", [1, 2], ids=["mla", "kv_sep_1h"])
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("mode", MLA_MODES)
@pytest.mark.parametrize("engine_name,use_ce", ENGINES)
def test_mla_roundtrip_modes_layerwise(data_config, kv_dim, cpu_layout_name, mode, engine_name, use_ce):
    """Shared-across-ranks: TP-group D2H prepares CPU, per-layer
    H2D reads it back; verify all ranks recover GPU 0's data. Covers all D2H
    modes. Covers MLA (kv_dim=1) and plain MHA, single head (kv_dim=2), cuda + ce."""
    skip_if_engine_unsupported(use_ce)
    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    assert num_heads == 1, "MLA_SIZES must only contain num_heads=1 configs"
    num_gpus = NUM_GPUS
    num_kv_heads = num_heads  # == 1

    gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim,
        cpu_layout_name, kv_dim, num_kv_heads, num_gpus)

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
        num_heads, head_dim, tpb, kv_dim, num_kv_heads, mode, num_gpus)
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
        transfer_num_cta=4, is_host_to_device=False, use_ce_transfer=use_ce,
        layer_id=0, layer_granularity=num_layers, kv_dim=kv_dim, num_kv_heads=num_kv_heads,
        kv_shared_across_ranks_mode=mode,
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
        chunk_size, kv_dim, num_kv_heads, mode,
        is_blockfirst=(cpu_layout_name == "BLOCKFIRST"),
        use_ce=use_ce)

    # All ranks should recover GPU 0's data (shared across ranks replicates).
    spot_check_gpu(all_gpu, 0, num_gpus, num_layers, num_blocks,
                   tpb, head_dim, kv_dim, label="layerwise mode={}".format(mode))


# Invalid mode fallback test

def test_invalid_mode_fallback():
    """Invalid kv_shared_across_ranks_mode falls back to 'sharded' without crash."""
    skip_if_engine_unsupported(use_ce=False)
    num_layers, num_blocks, tpb, num_heads, head_dim = 4, 8, 16, 1, 128
    num_gpus = NUM_GPUS

    gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim,
        "LAYERFIRST", 1, 1, num_gpus)

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
        layer_id=0, layer_granularity=num_layers, kv_dim=1, num_kv_heads=1,
        kv_shared_across_ranks_mode="invalid_xyz",
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

# Combined (data_config, kv_dim, num_kv_heads, mode) parametrization.
#
# The C++ transfer branches at the top level on num_kv_heads:
#   - num_kv_heads > 1 (MHA): a single code path (heads sharded across TP,
#     cpu_startoff = i * cpu_tp_stride). The kv_shared_across_ranks_mode
#     argument is IGNORED.
#   - num_kv_heads == 1 (MLA): the mode selects sharded / all_write /
#     rank0_only.
#
# So mode is only meaningful when num_kv_heads == 1. We therefore emit exactly
# ONE combo per multi-head size (mode is a don't-care placeholder), and all
# modes per single-head size. This avoids nonsensical all_write-mha /
# rank0_only-mha combos.
#
# Four-quadrant coverage (kv_dim x num_kv_heads):
#   MLA        (kv_dim=1, num_kv_heads=1) — K/V combined + single head, shared
#   kv_sep_1h  (kv_dim=2, num_kv_heads=1) — K/V separate + single head, shared
#   packed MHA (kv_dim=1, num_kv_heads=H) — K/V combined + multi head, sharded
#   plain MHA  (kv_dim=2, num_kv_heads=H) — K/V separate + multi head, sharded
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
    ((61, 256, 16, 1, 576), "dsv32"),    # 512 lora + 64 rope, not a power of 2
    ((78, 256, 16, 1, 576), "glm52"),
]
_MHA_SIZES = [
    ((4, 8, 16, 8, 128), "mha-mini"),
    ((32, 64, 16, 8, 128), "mha-llama8b"),
    ((80, 256, 16, 8, 128), "mha-llama70b"),
    ((2, 4, 1, 8, 128), "mha-edge"),
    ((4, 4, 16, 16, 128), "mha-16head"),
    # Qwen3.5-397B-A17B: only its 15 full_attention layers cache, 2 kv heads,
    # head_dim 256. The head_dim/head-count pair differs from every entry
    # above, and 2 kv heads is the narrowest multi-head TP split that still
    # shards (NUM_GPUS=4 would not divide it, so it stays a shared-mode case
    # in MLA_SIZES terms; here it exercises head_dim=256 packing).
    ((15, 64, 16, 4, 256), "mha-qwen35"),
]
# Four-quadrant: MLA (kv_dim=1, nkh=1), kv_sep_1h (kv_dim=2, nkh=1),
# packed MHA (kv_dim=1, nkh=H), plain MHA (kv_dim=2, nkh=H).
# mode only matters for num_kv_heads==1 (shared across ranks).
_SHARED_MODES = ("sharded", "all_write", "rank0_only", "layer_parallel", "rank_rotate")
CE_MODE_CONFIGS = (
    # MLA: kv_dim=1, num_kv_heads=1 (from _MLA_SIZES where num_heads=1)
    [pytest.param(cfg, 1, 1, mode, id=f"mla_{mode}-{sid}")
     for (cfg, sid) in _MLA_SIZES
     for mode in _SHARED_MODES]
    # plain MHA, single head: kv_dim=2, num_kv_heads=1 (from _MLA_SIZES where num_heads=1)
    + [pytest.param(cfg, 2, 1, mode, id=f"kv_sep_1h_{mode}-{sid}")
       for (cfg, sid) in _MLA_SIZES
       for mode in _SHARED_MODES]
    # packed MHA: kv_dim=1, num_kv_heads=H (from _MHA_SIZES where num_heads=H)
    + [pytest.param(cfg, 1, cfg[3], "sharded", id=f"packed_mha-{sid}")
       for (cfg, sid) in _MHA_SIZES]
    # plain MHA: kv_dim=2, num_kv_heads=H (from _MHA_SIZES where num_heads=H)
    + [pytest.param(cfg, 2, cfg[3], "sharded", id=f"plain_mha-{sid}")
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


def _expected_strategy(pattern_name, cpu_layout_name, kv_dim, num_kv_heads, mode,
                       is_host_to_device, num_blocks=64, threshold=8):
    """Predict which CE strategy auto-selection should pick, mirroring
    csrc/ce_transfer.cu choose_path().

    Returns (strategy, variant) where strategy is one of
    CONTIG_DIRECT / SEGMENT_DIRECT / SEGMENT_SCATTER / GATHER_SCATTER /
    GATHER_DIRECT and variant is always "".

    Key stride facts (see cpu_layout_for_mode / tp_transfer_thread_group.cpp):
      dst_phys_contig == (cpu_block_stride == chunk_size).
        block_stride = kv_dim * chunk_size, so dst_phys iff kv_dim == 1
        and LAYERFIRST (MLA/packed MHA). kv_sep_1h/plain MHA (kv_dim=2) -> false.
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
    # dst_phys_contig: block_stride == kv_dim * chunk_size, so contiguous
    # only when kv_dim == 1 (K/V combined: MLA or packed MHA).
    dst_phys = (cpu_layout_name == "LAYERFIRST") and kv_dim == 1
    is_blockfirst = (cpu_layout_name == "BLOCKFIRST")
    # Sharded D2H only applies when KV is shared across ranks (num_kv_heads == 1).
    sharded_d2h = (num_kv_heads == 1 and mode == "sharded" and not is_host_to_device)
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
    # Exception (commit eab52a2a3): bfirst + shared + D2H + !full_block
    # (layer_parallel / rank_rotate) -> SEGMENT_SCATTER, which is 30%-8.9x
    # faster than GATHER_DIRECT for the lighter staging+scatter path.
    # is_full_block == (all layers*kv_dim transferred in one call);
    # rank0_only / all_write store the full CPU block per rank -> full_block;
    # layer_parallel / rank_rotate / sharded store a shard -> !full_block.
    # (Sharded is handled by the `not src_phys` branch below and never reaches
    # here because its gpu_phys_contig is false.)
    is_full_block = mode in ("rank0_only", "all_write")
    if not dst_phys and is_blockfirst and src_phys:
        if not is_host_to_device and num_kv_heads == 1 and not is_full_block:
            return ("SEGMENT_SCATTER", "")
        return ("GATHER_DIRECT", "")
    # CONTIG_DIRECT: logical + physical contiguity on both sides.
    if pattern_name == "contiguous" and dst_phys and src_phys:
        return ("CONTIG_DIRECT", "")
    # Sharded D2H (LF + shared sharded, !gpu_phys_contig) -> GATHER_SCATTER.
    if not src_phys:
        return ("GATHER_SCATTER", "")
    # LAYERFIRST or BF multi-head: few segments -> SEGMENT_DIRECT or SEGMENT_SCATTER.
    if num_segments <= threshold:
        if dst_phys:
            return ("SEGMENT_DIRECT", "")
        return ("SEGMENT_SCATTER", "")
    # many segments, src contiguous -> GATHER_SCATTER
    return ("GATHER_SCATTER", "")


@pytest.mark.parametrize("data_config,kv_dim,num_kv_heads,mode", CE_MODE_CONFIGS)
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("pattern", CE_PATTERNS)
@pytest.mark.parametrize("segment_threshold", CE_SEGMENT_THRESHOLDS,
                         ids=lambda t: "thr{}".format(t))
@pytest.mark.parametrize("path_opt", [False, True], ids=["baseline", "optimized"])
@pytest.mark.parametrize("enable_memcpy2d", CE_MEMCPY2D_CONFIGS, ids=["no_memcpy2d", "memcpy2d"])
def test_ce_paths_roundtrip(data_config, kv_dim, num_kv_heads, cpu_layout_name, pattern,
                            path_opt, mode, segment_threshold, enable_memcpy2d):
    """CE strategy round-trip correctness via block-id patterns.

    Combos come from CE_MODE_CONFIGS: four-quadrant coverage
    (MLA/kv_sep_1h/packed MHA/plain MHA).  Each pattern triggers a different
    auto-selected CE strategy:
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
        cpu_layout_name, kv_dim, num_kv_heads, num_gpus)

    all_gpu = [make_gpu_tensors(num_layers, num_blocks, tpb,
                               heads_per_rank, head_dim, kv_dim, g)
               for g in range(num_gpus)]

    if num_kv_heads == 1:
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
        num_heads, head_dim, tpb, kv_dim, num_kv_heads, mode, num_gpus)
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
        layer_id=0, layer_granularity=num_layers, kv_dim=kv_dim, num_kv_heads=num_kv_heads,
        kv_shared_across_ranks_mode=mode,
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
        layer_id=0, layer_granularity=num_layers, kv_dim=kv_dim, num_kv_heads=num_kv_heads,
        kv_shared_across_ranks_mode=mode,
    )
    sync_all(num_gpus)

    # Verify round-trip data integrity
    expected_gpu = 0 if num_kv_heads == 1 else None
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


@pytest.mark.parametrize("data_config,kv_dim,num_kv_heads,mode", CE_MODE_CONFIGS)
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("pattern", CE_PATTERNS)
@pytest.mark.parametrize("segment_threshold", CE_SEGMENT_THRESHOLDS,
                         ids=lambda t: "thr{}".format(t))
@pytest.mark.parametrize("path_opt", [False, True], ids=["baseline", "optimized"])
@pytest.mark.parametrize("notify_mode", ["polling"], ids=["polling"])
@pytest.mark.parametrize("layer_granularity", [1, None], ids=["lg1", "lg_all"])
@pytest.mark.parametrize("enable_memcpy2d", CE_MEMCPY2D_CONFIGS, ids=["no_memcpy2d", "memcpy2d"])
def test_ce_paths_layerwise_h2d(data_config, kv_dim, num_kv_heads, cpu_layout_name, pattern,
                                path_opt, mode, segment_threshold,
                                notify_mode, layer_granularity, enable_memcpy2d):
    """CE strategy correctness for the per-layer H2D path.

    Uses TPTransferThreadGroup D2H (already verified correct) to prepare
    CPU data, then per-layer H2D to read it back with the
    same block-id pattern.  Verifies that the layerwise CE strategy produces
    identical results to the TP-group CE strategy.

    Combos come from CE_MODE_CONFIGS: four-quadrant coverage
    (MLA/kv_sep_1h/packed MHA/plain MHA).

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
        cpu_layout_name, kv_dim, num_kv_heads, num_gpus)

    # Fill GPU with deterministic data
    all_gpu = [make_gpu_tensors(num_layers, num_blocks, tpb,
                               heads_per_rank, head_dim, kv_dim, g)
               for g in range(num_gpus)]
    if num_kv_heads == 1:
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
        num_heads, head_dim, tpb, kv_dim, num_kv_heads, mode, num_gpus)
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
        layer_id=0, layer_granularity=num_layers, kv_dim=kv_dim, num_kv_heads=num_kv_heads,
        kv_shared_across_ranks_mode=mode,
    )
    sync_all(num_gpus)
    del tp

    # Clear GPUs
    for g in range(num_gpus):
        for l in range(num_layers):
            all_gpu[g][l].zero_()
    sync_all(num_gpus)

    # Step 2: H2D via submit_layerwise (test target). path_opt selects
    # baseline (PER_BLOCK) vs optimized (CONTIG_DIRECT / SEGMENT_DIRECT /
    # SEGMENT_SCATTER / GATHER_SCATTER) on the H2D path.
    # (Step 1 above intentionally keeps default config -- it only prepares the
    # reference CPU data, the swept dims apply to this H2D test target.)
    layerwise_h2d_readback(
        all_gpu, cpu_kv, num_gpus, gpu_layout, num_layers, ids,
        cpu_stride_kv, cpu_stride_layer, cpu_stride_block, cpu_stride_tp,
        chunk_size, kv_dim, num_kv_heads, mode,
        ce_path_opt=path_opt,
        ce_segment_threshold=segment_threshold, notify_mode=notify_mode,
        layer_granularity=layer_granularity,
        is_blockfirst=(cpu_layout_name == "BLOCKFIRST"),
        enable_memcpy2d=enable_memcpy2d)

    # Verify GPU data == original
    expected_gpu = 0 if num_kv_heads == 1 else None
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
    """Enumerate (threshold, pattern, layout, kv_dim, num_kv_heads, mode,
    direction, size) -> (strategy, variant) over exactly the swept parametrize
    space, so this matches what test_ce_paths_roundtrip / _layerwise_h2d
    actually exercise (including the scattered-skip-when-num_blocks<=threshold
    rule).

    Returns a list of (label, strategy, variant) rows.
    """
    rows = []
    layouts = ["LAYERFIRST", "BLOCKFIRST"]
    # (kv_dim, num_kv_heads, mode) combos as produced by CE_MODE_CONFIGS.
    mode_combos = [
        (1, 1, "sharded"), (1, 1, "all_write"),
        (1, 1, "rank0_only"), (1, 1, "layer_parallel"),
        (1, 1, "rank_rotate"),
        (2, 1, "sharded"), (2, 1, "all_write"),
        (2, 1, "rank0_only"), (2, 1, "layer_parallel"),
        (2, 1, "rank_rotate"),
        (1, 8, "sharded"), (2, 8, "sharded"),
    ]
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
                    for kv_dim, num_kv_heads, mode in mode_combos:
                        for is_h2d in (False, True):
                            strat, variant = _expected_strategy(
                                pattern, layout, kv_dim, num_kv_heads, mode,
                                is_h2d, num_blocks=num_blocks,
                                threshold=threshold)
                            if num_kv_heads == 1:
                                tag = "mla_{}".format(mode) if kv_dim == 1 else "kv_sep_1h_{}".format(mode)
                            else:
                                tag = "packed_mha" if kv_dim == 1 else "plain_mha"
                            label = ("thr{:<2d} nb{:<3d} {:<10s} {:<10s} "
                                     "{:<16s} {}").format(
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
    num_kv_heads = num_heads  # == 1
    gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim, "BLOCKFIRST", 1, num_kv_heads, num_gpus)
    all_gpu = [make_gpu_tensors(num_layers, num_blocks, tpb, heads_per_rank, head_dim, kv_dim, g) for g in range(num_gpus)]
    fill_gpu(all_gpu[0], 0, num_layers, num_blocks, tpb, heads_per_rank, head_dim, kv_dim)
    for g in range(1, num_gpus):
        for l in range(num_layers):
            all_gpu[g][l].copy_(all_gpu[0][l])
    sync_all(num_gpus)
    cpu_stride_block = cpu_layout.get_block_stride() * ES
    cpu_stride_tp = cpu_stride_block // num_gpus
    cpu_kv = make_cpu_tensor(cpu_layout, num_layers, num_blocks)
    tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus, gpu_layout, num_layers, is_blockfirst=True, kv_dim=kv_dim)
    ids = block_ids(num_blocks)
    tp.tp_group_transfer(
        gpu_block_id_tensor=ids, cpu_block_id_tensor=ids,
        cpu_kv_stride_in_bytes=cpu_layout_tp.get_kv_stride() * ES,
        cpu_layer_stride_in_bytes=cpu_layout_tp.get_layer_stride() * ES,
        cpu_block_stride_in_bytes=cpu_stride_block,
        cpu_tp_stride_in_bytes=cpu_stride_tp,
        transfer_num_cta=4, is_host_to_device=False, use_ce_transfer=True,
        layer_id=0, layer_granularity=num_layers, kv_dim=kv_dim, num_kv_heads=num_kv_heads,
        kv_shared_across_ranks_mode="rank0_only", designated_rank=designated_rank)
    sync_all(num_gpus)
    del tp
    for g in range(num_gpus):
        for l in range(num_layers):
            all_gpu[g][l].zero_()
    sync_all(num_gpus)
    # H2D is direction-symmetric on rank share modes: every rank needs the whole
    # thing regardless of who wrote it, so this reads back with the default
    # sharded mode even though the D2H above was rank0_only.
    layerwise_h2d_readback(
        all_gpu, cpu_kv, num_gpus, gpu_layout, num_layers, ids,
        cpu_layout_tp.get_kv_stride() * ES,
        cpu_layout_tp.get_layer_stride() * ES,
        cpu_stride_block, cpu_stride_tp,
        gpu_layout.get_chunk_size() * ES, kv_dim, num_kv_heads, "rank0_only",
        is_blockfirst=True, use_ce=True)
    spot_check_gpu(all_gpu, 0, num_gpus, num_layers, num_blocks, tpb, head_dim, kv_dim, label=f"designated={designated_rank}")


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

# Four-quadrant: (cfg, kv_dim, num_kv_heads, mode)
# MLA (kv_dim=1, nkh=1), kv_sep_1h (kv_dim=2, nkh=1), packed MHA (kv_dim=1, nkh=8),
# plain MHA (kv_dim=2, nkh=8). Shared modes only for num_kv_heads==1.
GATHER_NT_MODE_CONFIGS = [
    pytest.param((4, 8, 16, 1, 512), 1, 1, "sharded", id="mla_sharded"),
    pytest.param((4, 8, 16, 1, 512), 1, 1, "all_write", id="mla_all_write"),
    pytest.param((4, 8, 16, 1, 512), 1, 1, "rank0_only", id="mla_rank0_only"),
    pytest.param((4, 8, 16, 1, 512), 2, 1, "sharded", id="kv_sep_1h_sharded"),
    pytest.param((4, 8, 16, 8, 128), 1, 8, "sharded", id="packed_mha"),
    pytest.param((4, 8, 16, 8, 128), 2, 8, "sharded", id="plain_mha"),
]

GATHER_THREADS_VALUES = [0, 1, 4, 8]
GATHER_NT_VALUES = [True, False]


@pytest.mark.parametrize("data_config,kv_dim,num_kv_heads,mode", GATHER_NT_MODE_CONFIGS)
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("gather_threads", GATHER_THREADS_VALUES,
                         ids=["gt0", "gt1", "gt4", "gt8"])
@pytest.mark.parametrize("gather_nt", GATHER_NT_VALUES,
                         ids=["nt_on", "nt_off"])
def test_gather_nt_roundtrip(data_config, kv_dim, num_kv_heads, cpu_layout_name, mode,
                             gather_threads, gather_nt):
    """CE gather threads and NT store round-trip correctness.

    Sweeps gather_threads (0=disable, 1, 4, 8) and gather_nt (on/off)
    to verify parallel CPU gather/scatter and non-temporal stores produce
    correct results across four-quadrant configs and CPU layouts.
    """
    skip_if_engine_unsupported(use_ce=True)
    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    num_gpus = NUM_GPUS

    gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim,
        cpu_layout_name, kv_dim, num_kv_heads, num_gpus)

    all_gpu = [make_gpu_tensors(num_layers, num_blocks, tpb,
                               heads_per_rank, head_dim, kv_dim, g)
               for g in range(num_gpus)]

    if num_kv_heads == 1:
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
        num_heads, head_dim, tpb, kv_dim, num_kv_heads, mode, num_gpus)
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
        layer_id=0, layer_granularity=num_layers, kv_dim=kv_dim, num_kv_heads=num_kv_heads,
        kv_shared_across_ranks_mode=mode,
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
        layer_id=0, layer_granularity=num_layers, kv_dim=kv_dim, num_kv_heads=num_kv_heads,
        kv_shared_across_ranks_mode=mode,
    )
    sync_all(num_gpus)

    # Verify round-trip data integrity
    expected_gpu = 0 if num_kv_heads == 1 else None
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


@pytest.mark.parametrize("data_config,kv_dim,num_kv_heads,mode", GATHER_NT_MODE_CONFIGS)
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("gather_threads", GATHER_THREADS_VALUES,
                         ids=["gt0", "gt1", "gt4", "gt8"])
@pytest.mark.parametrize("gather_nt", GATHER_NT_VALUES,
                         ids=["nt_on", "nt_off"])
def test_gather_nt_layerwise_h2d(data_config, kv_dim, num_kv_heads, cpu_layout_name, mode,
                                 gather_threads, gather_nt):
    """CE gather threads and NT store correctness for layerwise H2D.

    TP-group CE D2H prepares CPU, per-layer CE H2D reads back
    with swept gather_threads and gather_nt settings.
    """
    skip_if_engine_unsupported(use_ce=True)
    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    num_gpus = NUM_GPUS

    gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim,
        cpu_layout_name, kv_dim, num_kv_heads, num_gpus)

    all_gpu = [make_gpu_tensors(num_layers, num_blocks, tpb,
                               heads_per_rank, head_dim, kv_dim, g)
               for g in range(num_gpus)]
    if num_kv_heads == 1:
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
        num_heads, head_dim, tpb, kv_dim, num_kv_heads, mode, num_gpus)
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
        layer_id=0, layer_granularity=num_layers, kv_dim=kv_dim, num_kv_heads=num_kv_heads,
        kv_shared_across_ranks_mode=mode,
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
        chunk_size, kv_dim, num_kv_heads, mode,
        is_blockfirst=(cpu_layout_name == "BLOCKFIRST"),
        ce_gather_threads=gather_threads,
        ce_gather_nt=gather_nt)

    expected_gpu = 0 if num_kv_heads == 1 else None
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


# ---------------------------------------------------------------------------
# SSD transfer correctness tests (non-compressed path: transfer_kv_blocks_ssd)
# ---------------------------------------------------------------------------
# CPU → SSD → CPU roundtrip: fill, write, clear, read, verify byte equality.
# Covers LAYERFIRST/BLOCKFIRST × MLA/MHA × multi-layer × 1/4 threads.

try:
    from flexkv.c_ext import SSDIOCTX, transfer_kv_blocks_ssd as _ssd_xfer_fn
    _SSD_AVAILABLE = True
except ImportError:
    _SSD_AVAILABLE = False


SSD_SKIP_REASON = "c_ext not built or SSD support disabled"

# Small sizes: SSD I/O is slow, only verifying correctness.
SSD_SIZES = [
    pytest.param((4, 8, 16, 1, 512), id="mla-mini"),
    pytest.param((16, 16, 16, 1, 512), id="mla-multi-layer"),
    pytest.param((4, 8, 16, 8, 128), id="mha-mini"),
    pytest.param((16, 16, 16, 8, 128), id="mha-multi-layer"),
    pytest.param((2, 4, 1, 1, 512), id="mla-edge"),
    pytest.param((2, 4, 1, 8, 128), id="mha-edge"),
]

SSD_THREADS = [1, 4]


def _ssd_file_path(tmpdir, dev_id=0, file_id=0):
    """Create a temp SSD file and return its path."""
    import os
    dev_dir = os.path.join(tmpdir, f"dev{dev_id}")
    os.makedirs(dev_dir, exist_ok=True)
    return os.path.join(dev_dir, f"ssd_cache_{dev_id}_{file_id}.bin")


def _setup_ssd_file(layout, num_blocks_per_file, tmpdir):
    """Pre-allocate SSD file sized for the layout."""
    import os
    file_size = layout.get_total_elements() * ES

    fpath = _ssd_file_path(tmpdir, 0, 0)
    with open(fpath, "wb+") as f:
        os.truncate(f.fileno(), file_size)
        os.fsync(f.fileno())
    return {0: [fpath]}

def _setup_ssd_files(layout, dtype, cache_dirs):
    ssd_handle = SSDAllocator.allocate(
        layout=layout,
        dtype=dtype,
        cache_dir=cache_dirs,
        max_file_size_gb=GLOBAL_CONFIG_FROM_ENV.max_file_size_gb,
    )
    return ssd_handle.get_file_list(), ssd_handle.num_blocks_per_file

def _fill_cpu_kv(cpu_tensor, layout, num_layers, num_blocks, tpb, num_heads,
                 head_dim, kv_dim, seed=42):
    """Fill CPU KV buffer with layout-independent deterministic pattern."""
    total_elems = cpu_tensor.numel()
    flat = torch.arange(total_elems, dtype=torch.int32) % 9973
    cpu_tensor[:] = flat.view(cpu_tensor.shape).to(DTYPE)


def _ssd_roundtrip_verify(cpu_orig, cpu_readback, layout, num_layers,
                          num_blocks, tpb, num_heads, head_dim, kv_dim,
                          label=""):
    """Verify byte-level equality after SSD roundtrip."""
    orig_bytes = cpu_orig.view(torch.uint8).reshape(-1)
    read_bytes = cpu_readback.view(torch.uint8).reshape(-1)
    assert torch.equal(orig_bytes, read_bytes), \
        f"SSD roundtrip data mismatch ({label}): " \
        f"{(orig_bytes != read_bytes).sum().item()} / {orig_bytes.numel()} bytes differ"


@pytest.mark.parametrize("data_config", SSD_SIZES)
@pytest.mark.parametrize("kv_dim", [1, 2], ids=["kv1", "kv2"])
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("num_threads", SSD_THREADS, ids=["t1", "t4"])
@pytest.mark.parametrize("ssd_io_opt", [False, True], ids=["baseline", "optimized"])
@pytest.mark.parametrize("iouring_entries", [0, 512], ids=["thread", "iouring"])
def test_ssd_transfer_roundtrip(data_config, kv_dim, cpu_layout_name, num_threads,
                                ssd_io_opt, iouring_entries, tmp_path):
    """SSD roundtrip: CPU → SSD → CPU with byte-level verification. Covers thread/io_uring engines (iouring_entries=0/512).

    Covers LAYERFIRST/BLOCKFIRST × four-quadrant (MLA/kv_sep_1h/packed MHA/plain
    MHA) × multi-layer × 1/4 threads × FLEXKV_SSD_IO_OPT (baseline=off=basic
    fragmented I/O, optimized=on=bf/lf opt).
    SSD layout matches CPU layout. Both modes must be byte-identical to source.
    """
    if not _SSD_AVAILABLE:
        pytest.skip(SSD_SKIP_REASON)

    import shutil
    import tempfile

    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    num_kv_heads = num_heads  # data_config num_heads IS num_kv_heads

    # Build CPU/SSD layout (same type, same as production)
    layout = KVCacheLayout(
        type=KVCacheLayoutType[cpu_layout_name.upper()],
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=tpb, num_head=num_heads,
        head_size=head_dim, kv_dim=kv_dim, num_kv_heads=num_kv_heads)

    chunk_size = layout.get_chunk_size()
    block_stride = layout.get_block_stride()
    kv_stride = layout.get_kv_stride()
    layer_stride = layout.get_layer_stride()

    # Create CPU buffer
    cpu_kv = torch.zeros(tuple(layout.kv_shape), dtype=DTYPE).pin_memory()
    _fill_cpu_kv(cpu_kv, layout, num_layers, num_blocks, tpb,
                 num_heads, head_dim, kv_dim)

    # Snapshot original for comparison
    cpu_orig = cpu_kv.clone()

    # Setup SSD file
    tmpdir = str(tmp_path)
    ssd_files = _setup_ssd_file(layout, num_blocks, tmpdir)

    # Block IDs (identity mapping)
    block_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()
    layer_ids = torch.arange(num_layers, dtype=torch.int32).pin_memory()

    chunk_bytes = chunk_size * ES
    block_stride_bytes = block_stride * ES
    kv_stride_bytes = kv_stride * ES
    layer_stride_bytes = layer_stride * ES

    ioctx = SSDIOCTX(ssd_files, 1, iouring_entries, 0)  # 1 device, no io_uring

    # Write: CPU → SSD
    _ssd_xfer_fn(
        ioctx=ioctx,
        cpu_layer_id_list=layer_ids,
        cpu_tensor_ptr=cpu_kv.data_ptr(),
        ssd_block_ids=block_ids,
        cpu_block_ids=block_ids,
        cpu_layer_stride_in_bytes=layer_stride_bytes,
        cpu_kv_stride_in_bytes=kv_stride_bytes,
        ssd_layer_stride_in_bytes=layer_stride_bytes,
        ssd_kv_stride_in_bytes=kv_stride_bytes,
        chunk_size_in_bytes=chunk_bytes,
        block_stride_in_bytes=block_stride_bytes,
        is_read=False,
        num_blocks_per_file=num_blocks,
        round_robin=1,
        num_threads_per_device=num_threads,
        kv_dim=kv_dim,
        ssd_io_opt=ssd_io_opt,
    )

    # Clear CPU buffer
    cpu_kv.zero_()

    # Read: SSD → CPU
    _ssd_xfer_fn(
        ioctx=ioctx,
        cpu_layer_id_list=layer_ids,
        cpu_tensor_ptr=cpu_kv.data_ptr(),
        ssd_block_ids=block_ids,
        cpu_block_ids=block_ids,
        cpu_layer_stride_in_bytes=layer_stride_bytes,
        cpu_kv_stride_in_bytes=kv_stride_bytes,
        ssd_layer_stride_in_bytes=layer_stride_bytes,
        ssd_kv_stride_in_bytes=kv_stride_bytes,
        chunk_size_in_bytes=chunk_bytes,
        block_stride_in_bytes=block_stride_bytes,
        is_read=True,
        num_blocks_per_file=num_blocks,
        round_robin=1,
        num_threads_per_device=num_threads,
        kv_dim=kv_dim,
        ssd_io_opt=ssd_io_opt,
    )

    # Verify
    label = (f"layout={cpu_layout_name} kv_dim={kv_dim} layers={num_layers} "
             f"blocks={num_blocks} threads={num_threads}")
    _ssd_roundtrip_verify(cpu_orig, cpu_kv, layout, num_layers, num_blocks,
                          tpb, num_heads, head_dim, kv_dim, label=label)

    del ioctx
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.mark.parametrize("data_config", [pytest.param((4, 8, 16, 1, 512), id="mla-mini"),
                                         pytest.param((4, 8, 16, 8, 128), id="mha-mini")])
@pytest.mark.parametrize("kv_dim", [1, 2], ids=["kv1", "kv2"])
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("num_threads", [1, 4], ids=["t1", "t4"])
@pytest.mark.parametrize("ssd_io_opt", [False, True], ids=["baseline", "optimized"])
@pytest.mark.parametrize("iouring_entries", [0, 512], ids=["thread", "iouring"])
def test_ssd_transfer_partial_blocks(data_config, kv_dim, cpu_layout_name, num_threads,
                                     ssd_io_opt, iouring_entries, tmp_path):
    """SSD roundtrip with non-contiguous block IDs: write [0,2,4,6] → read into [1,3,5,7]. Covers thread/io_uring engines (iouring_entries=0/512).

    Exercises block-id remapping and fragmented layer-first I/O path.
    Covers FLEXKV_SSD_IO_OPT (baseline=off=basic fragmented I/O, optimized=on=bf/lf opt).
    """
    if not _SSD_AVAILABLE:
        pytest.skip(SSD_SKIP_REASON)

    import shutil

    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    num_kv_heads = num_heads  # data_config num_heads IS num_kv_heads

    layout = KVCacheLayout(
        type=KVCacheLayoutType[cpu_layout_name.upper()],
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=tpb, num_head=num_heads,
        head_size=head_dim, kv_dim=kv_dim, num_kv_heads=num_kv_heads)

    chunk_size = layout.get_chunk_size()
    block_stride = layout.get_block_stride()
    kv_stride = layout.get_kv_stride()
    layer_stride = layout.get_layer_stride()

    cpu_kv = torch.zeros(tuple(layout.kv_shape), dtype=DTYPE).pin_memory()
    _fill_cpu_kv(cpu_kv, layout, num_layers, num_blocks, tpb,
                 num_heads, head_dim, kv_dim)
    cpu_orig = cpu_kv.clone()

    tmpdir = str(tmp_path)
    ssd_files = _setup_ssd_file(layout, num_blocks, tmpdir)

    # Write even blocks to SSD
    write_ssd_ids = torch.tensor([0, 2, 4, 6], dtype=torch.int64).pin_memory()
    write_cpu_ids = torch.tensor([0, 2, 4, 6], dtype=torch.int64).pin_memory()
    layer_ids = torch.arange(num_layers, dtype=torch.int32).pin_memory()

    chunk_bytes = chunk_size * ES
    block_stride_bytes = block_stride * ES
    kv_stride_bytes = kv_stride * ES
    layer_stride_bytes = layer_stride * ES

    ioctx = SSDIOCTX(ssd_files, 1, iouring_entries, 0)

    _ssd_xfer_fn(
        ioctx=ioctx,
        cpu_layer_id_list=layer_ids,
        cpu_tensor_ptr=cpu_kv.data_ptr(),
        ssd_block_ids=write_ssd_ids,
        cpu_block_ids=write_cpu_ids,
        cpu_layer_stride_in_bytes=layer_stride_bytes,
        cpu_kv_stride_in_bytes=kv_stride_bytes,
        ssd_layer_stride_in_bytes=layer_stride_bytes,
        ssd_kv_stride_in_bytes=kv_stride_bytes,
        chunk_size_in_bytes=chunk_bytes,
        block_stride_in_bytes=block_stride_bytes,
        is_read=False,
        num_blocks_per_file=num_blocks,
        round_robin=1,
        num_threads_per_device=num_threads,
        kv_dim=kv_dim,
        ssd_io_opt=ssd_io_opt,
    )

    # Clear the written CPU blocks
    for bid in [0, 2, 4, 6]:
        # Zero out the block in the CPU buffer
        if layout.type == KVCacheLayoutType.LAYERFIRST:
            cpu_kv[:, :, bid] = 0
        else:
            cpu_kv[bid] = 0

    # Read back into ODD CPU slots
    read_ssd_ids = torch.tensor([0, 2, 4, 6], dtype=torch.int64).pin_memory()
    read_cpu_ids = torch.tensor([1, 3, 5, 7], dtype=torch.int64).pin_memory()

    _ssd_xfer_fn(
        ioctx=ioctx,
        cpu_layer_id_list=layer_ids,
        cpu_tensor_ptr=cpu_kv.data_ptr(),
        ssd_block_ids=read_ssd_ids,
        cpu_block_ids=read_cpu_ids,
        cpu_layer_stride_in_bytes=layer_stride_bytes,
        cpu_kv_stride_in_bytes=kv_stride_bytes,
        ssd_layer_stride_in_bytes=layer_stride_bytes,
        ssd_kv_stride_in_bytes=kv_stride_bytes,
        chunk_size_in_bytes=chunk_bytes,
        block_stride_in_bytes=block_stride_bytes,
        is_read=True,
        num_blocks_per_file=num_blocks,
        round_robin=1,
        num_threads_per_device=num_threads,
        kv_dim=kv_dim,
        ssd_io_opt=ssd_io_opt,
    )

    # Verify: blocks 1,3,5,7 should now contain the original data from
    # blocks 0,2,4,6 respectively.
    for src_bid, dst_bid in zip([0, 2, 4, 6], [1, 3, 5, 7]):
        if layout.type == KVCacheLayoutType.LAYERFIRST:
            for layer in range(num_layers):
                for kv in range(kv_dim):
                    orig_block = cpu_orig[layer, kv, src_bid]
                    read_block = cpu_kv[layer, kv, dst_bid]
                    assert torch.equal(orig_block, read_block), \
                        f"SSD partial roundtrip mismatch: src={src_bid} dst={dst_bid} " \
                        f"layer={layer} kv={kv} layout={cpu_layout_name}"
        else:
            orig_block = cpu_orig[src_bid]
            read_block = cpu_kv[dst_bid]
            assert torch.equal(orig_block, read_block), \
                f"SSD partial roundtrip mismatch: src={src_bid} dst={dst_bid} " \
                f"layout={cpu_layout_name}"

    del ioctx
    shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Incremental (partial layer range) transfers: path validation
# ---------------------------------------------------------------------------

def _ssd_layer_view(cpu_kv, layout, layer_id):
    """All blocks of one layer, for either CPU layout."""
    if layout.type == KVCacheLayoutType.LAYERFIRST:
        # kv_shape = [layer, kv, block, tpb, head, head_size]
        return cpu_kv[layer_id]
    # BLOCKFIRST kv_shape = [block, layer, kv, tpb, head, head_size]
    return cpu_kv[:, layer_id]


def _ssd_block_layer_view(cpu_kv, layout, block_id, layer_id):
    """One (block, layer) cell (all K/V halves), for either CPU layout."""
    if layout.type == KVCacheLayoutType.LAYERFIRST:
        return cpu_kv[layer_id, :, block_id]
    return cpu_kv[block_id, layer_id]


# ---------------------------------------------------------------------------
# FLEXKV_SSD_IO_OPT: opt/baseline equivalence + layerwise SSD disk2h coverage
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("data_config", [pytest.param((4, 8, 16, 1, 512), id="mla-mini"),
                                         pytest.param((16, 16, 16, 1, 512), id="mla-multi"),
                                         pytest.param((4, 8, 16, 8, 128), id="mha-mini"),
                                         pytest.param((16, 16, 16, 8, 128), id="mha-multi")])
@pytest.mark.parametrize("kv_dim", [1, 2], ids=["kv1", "kv2"])
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("num_threads", [1, 4], ids=["t1", "t4"])
@pytest.mark.parametrize("iouring_entries", [0, 512], ids=["thread", "iouring"])
def test_ssd_io_opt_on_off_byte_identical(data_config, kv_dim, cpu_layout_name, num_threads,
                                          iouring_entries, tmp_path):
    """FLEXKV_SSD_IO_OPT ON vs OFF must produce byte-identical SSD roundtrips. Covers thread/io_uring engines (iouring_entries=0/512).

    The optimization only changes the I/O strategy (bf one-shot vs lf
    layer-major vs basic fragmented), never the bytes transferred. This guards
    against the opt path silently diverging from the baseline.
    """
    if not _SSD_AVAILABLE:
        pytest.skip(SSD_SKIP_REASON)
    import shutil

    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    num_kv_heads = num_heads  # data_config num_heads IS num_kv_heads

    layout = KVCacheLayout(
        type=KVCacheLayoutType[cpu_layout_name.upper()],
        num_layer=num_layers, num_block=num_blocks,
        tokens_per_block=tpb, num_head=num_heads,
        head_size=head_dim, kv_dim=kv_dim, num_kv_heads=num_kv_heads)
    chunk_bytes = layout.get_chunk_size() * ES
    block_stride_bytes = layout.get_block_stride() * ES
    kv_stride_bytes = layout.get_kv_stride() * ES
    layer_stride_bytes = layout.get_layer_stride() * ES

    block_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()
    layer_ids = torch.arange(num_layers, dtype=torch.int32).pin_memory()

    def _run(opt):
        cpu_kv = torch.zeros(tuple(layout.kv_shape), dtype=DTYPE).pin_memory()
        _fill_cpu_kv(cpu_kv, layout, num_layers, num_blocks, tpb,
                     num_heads, head_dim, kv_dim)
        orig = cpu_kv.clone()
        sub = str(tmp_path / ("opt_on" if opt else "opt_off"))
        ssd_files = _setup_ssd_file(layout, num_blocks, sub)
        ioctx = SSDIOCTX(ssd_files, 1, iouring_entries, 0)
        try:
            _ssd_xfer_fn(ioctx=ioctx, cpu_layer_id_list=layer_ids,
                         cpu_tensor_ptr=cpu_kv.data_ptr(),
                         ssd_block_ids=block_ids, cpu_block_ids=block_ids,
                         cpu_layer_stride_in_bytes=layer_stride_bytes,
                         cpu_kv_stride_in_bytes=kv_stride_bytes,
                         ssd_layer_stride_in_bytes=layer_stride_bytes,
                         ssd_kv_stride_in_bytes=kv_stride_bytes,
                         chunk_size_in_bytes=chunk_bytes,
                         block_stride_in_bytes=block_stride_bytes,
                         is_read=False, num_blocks_per_file=num_blocks,
                         round_robin=1, num_threads_per_device=num_threads,
                         kv_dim=kv_dim, ssd_io_opt=opt)
            cpu_kv.zero_()
            _ssd_xfer_fn(ioctx=ioctx, cpu_layer_id_list=layer_ids,
                         cpu_tensor_ptr=cpu_kv.data_ptr(),
                         ssd_block_ids=block_ids, cpu_block_ids=block_ids,
                         cpu_layer_stride_in_bytes=layer_stride_bytes,
                         cpu_kv_stride_in_bytes=kv_stride_bytes,
                         ssd_layer_stride_in_bytes=layer_stride_bytes,
                         ssd_kv_stride_in_bytes=kv_stride_bytes,
                         chunk_size_in_bytes=chunk_bytes,
                         block_stride_in_bytes=block_stride_bytes,
                         is_read=True, num_blocks_per_file=num_blocks,
                         round_robin=1, num_threads_per_device=num_threads,
                         kv_dim=kv_dim, ssd_io_opt=opt)
        finally:
            del ioctx
            shutil.rmtree(sub, ignore_errors=True)
        return orig, cpu_kv

    orig_on, read_on = _run(True)
    orig_off, read_off = _run(False)
    _ssd_roundtrip_verify(orig_on, read_on, layout, num_layers, num_blocks,
                          tpb, num_heads, head_dim, kv_dim, label="on")
    _ssd_roundtrip_verify(orig_off, read_off, layout, num_layers, num_blocks,
                          tpb, num_heads, head_dim, kv_dim, label="off")
    on_bytes = read_on.view(torch.uint8).reshape(-1)
    off_bytes = read_off.view(torch.uint8).reshape(-1)
    assert torch.equal(on_bytes, off_bytes), \
        f"SSD opt ON vs OFF diverge: {(on_bytes != off_bytes).sum().item()} " \
        f"bytes differ (layout={cpu_layout_name} kv_dim={kv_dim} threads={num_threads})"


@pytest.mark.parametrize("data_config", [pytest.param((4, 8, 16, 1, 512), id="mla-mini"),
                                         pytest.param((16, 16, 16, 1, 512), id="mla-multi")])
@pytest.mark.parametrize("kv_dim", [1, 2], ids=["mla", "kv_sep_1h"])
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("ssd_io_opt", [False, True], ids=["baseline", "optimized"])
@pytest.mark.parametrize("iouring_entries", [0, 512], ids=["thread", "iouring"])
def test_ssd_transfer_layerwise_disk2h(data_config, kv_dim, cpu_layout_name, ssd_io_opt,
                                       iouring_entries, tmp_path):
    """Layerwise SSD disk2h: SSD -> CPU (standalone read) -> GPU (layerwise).

    Covers thread/io_uring engines (iouring_entries=0/512).

    Mirrors production wiring after the SSD read was hoisted out of the fused
    op: the engine schedules a standalone DISK2H (transfer_kv_blocks_ssd,
    is_read=True) that fills the CPU staging buffer, and only then does the
    LAYERWISE op run its CE H2D loop CPU -> GPU. Verifies the CPU staging
    buffer recovers the original source for both opt/baseline (the direct
    output of the SSD path), which is exactly the precondition the dependency
    edge in merge_to_batch_graph guarantees.

    Shared-across-ranks + sharded: all ranks hold identical data, so the SSD
    read's full CPU layout is consistent with the H2D sharded offset (0).
    Covers MLA (kv_dim=1) and plain MHA, single head (kv_dim=2), both with num_kv_heads=1.
    """
    if not _SSD_AVAILABLE:
        pytest.skip(SSD_SKIP_REASON)
    import shutil

    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    assert num_heads == 1, "layerwise SSD disk2h test is shared-across-ranks only (num_kv_heads=1)"
    num_gpus = NUM_GPUS
    num_kv_heads = num_heads  # == 1

    gpu_layout, cpu_layout, cpu_layout_tp, _, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim,
        cpu_layout_name, kv_dim, num_kv_heads, num_gpus)

    (total_blocks, cpu_stride_kv, cpu_stride_layer,
     cpu_stride_block, cpu_stride_tp) = cpu_layout_for_mode(
        cpu_layout, cpu_layout_tp, num_layers, num_blocks,
        num_heads, head_dim, tpb, kv_dim, num_kv_heads, "sharded", num_gpus)
    cpu_chunk_bytes = cpu_layout.get_chunk_size() * ES

    # CPU staging buffer (source). Fill with deterministic pattern.
    cpu_kv = torch.zeros(tuple(cpu_layout.kv_shape), dtype=DTYPE).pin_memory()
    _fill_cpu_kv(cpu_kv, cpu_layout, num_layers, num_blocks, tpb,
                 num_heads, head_dim, kv_dim)
    cpu_orig = cpu_kv.clone()

    # GPU tensors (zeroed; H2D writes into them).
    all_gpu = [make_gpu_tensors(num_layers, num_blocks, tpb,
                                heads_per_rank, head_dim, kv_dim, g)
               for g in range(num_gpus)]
    sync_all(num_gpus)

    # SSD layout mirrors CPU layout (1 file holds all blocks).
    ssd_layer_bytes = cpu_layout.get_layer_stride() * ES
    ssd_kv_bytes = cpu_layout.get_kv_stride() * ES

    tmpdir = str(tmp_path)
    ssd_files = _setup_ssd_file(cpu_layout, num_blocks, tmpdir)
    ioctx = SSDIOCTX(ssd_files, 1, iouring_entries, 0)

    # Populate SSD from the CPU source (CPU -> SSD) using the same strides the
    # layerwise SSD read will use to pull it back.
    block_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()
    layer_ids = torch.arange(num_layers, dtype=torch.int32).pin_memory()
    _ssd_xfer_fn(ioctx=ioctx, cpu_layer_id_list=layer_ids,
                 cpu_tensor_ptr=cpu_kv.data_ptr(),
                 ssd_block_ids=block_ids, cpu_block_ids=block_ids,
                 cpu_layer_stride_in_bytes=cpu_stride_layer,
                 cpu_kv_stride_in_bytes=cpu_stride_kv,
                 ssd_layer_stride_in_bytes=ssd_layer_bytes,
                 ssd_kv_stride_in_bytes=ssd_kv_bytes,
                 chunk_size_in_bytes=cpu_chunk_bytes,
                 block_stride_in_bytes=cpu_stride_block,
                 is_read=False, num_blocks_per_file=num_blocks,
                 round_robin=1, num_threads_per_device=32,
                 kv_dim=kv_dim, ssd_io_opt=ssd_io_opt)
    cpu_kv.zero_()

    # Step 1 -- the hoisted DISK2H op: SSD -> CPU staging.  In production this
    # is a separate op the engine runs first; the LAYERWISE op below depends on
    # it, so by the time it starts every CPU block it reads is already filled.
    _ssd_xfer_fn(ioctx=ioctx, cpu_layer_id_list=layer_ids,
                 cpu_tensor_ptr=cpu_kv.data_ptr(),
                 ssd_block_ids=block_ids, cpu_block_ids=block_ids,
                 cpu_layer_stride_in_bytes=cpu_stride_layer,
                 cpu_kv_stride_in_bytes=cpu_stride_kv,
                 ssd_layer_stride_in_bytes=ssd_layer_bytes,
                 ssd_kv_stride_in_bytes=ssd_kv_bytes,
                 chunk_size_in_bytes=cpu_chunk_bytes,
                 block_stride_in_bytes=cpu_stride_block,
                 is_read=True, num_blocks_per_file=num_blocks,
                 round_robin=1, num_threads_per_device=32,
                 kv_dim=kv_dim, ssd_io_opt=ssd_io_opt)

    # Step 2 -- the LAYERWISE op: CPU -> GPU only.
    layerwise_h2d_readback(
        all_gpu, cpu_kv, num_gpus, gpu_layout, num_layers, block_ids,
        cpu_stride_kv, cpu_stride_layer, cpu_stride_block, cpu_stride_tp,
        cpu_chunk_bytes, kv_dim, num_kv_heads, "sharded",
        is_blockfirst=(cpu_layout_name == "BLOCKFIRST"), use_ce=True)

    # The SSD read wrote the staging buffer; H2D only read from it. So the CPU
    # staging buffer must equal the original source for both opt/baseline.
    orig_bytes = cpu_orig.view(torch.uint8).reshape(-1)
    read_bytes = cpu_kv.view(torch.uint8).reshape(-1)
    assert torch.equal(orig_bytes, read_bytes), \
        f"layerwise SSD disk2h mismatch (layout={cpu_layout_name} " \
        f"ssd_io_opt={ssd_io_opt}): " \
        f"{(orig_bytes != read_bytes).sum().item()} / {orig_bytes.numel()} bytes differ"

    del ioctx
    shutil.rmtree(tmpdir, ignore_errors=True)

@pytest.fixture
def ssd_cache_dirs(tmp_path):
    raw = os.getenv("FLEXKV_TEST_SSD_CACHE_DIRS")
    return [p.strip() for p in raw.split(";") if p.strip()]

def make_tp_gds_group(gpu_blocks,
                      ssd_files,
                      num_layers,
                      gpu_kv_layout,
                      num_gpus):
    gpu_block_ptrs_flat = []
    for gpu_tensors in gpu_blocks:
        for t in gpu_tensors:
            gpu_block_ptrs_flat.append(t.data_ptr())

    tp_gds = TPGDSTransferThreadGroup(num_gpus,
                                      gpu_block_ptrs_flat,
                                      num_layers, # num_tensors_per_gpu
                                      ssd_files,
                                      num_layers,
                                      [gpu_kv_layout.get_kv_stride() * ES] * num_gpus,
                                      [gpu_kv_layout.get_block_stride() * ES] * num_gpus,
                                      [gpu_kv_layout.get_layer_stride() * ES] * num_gpus,
                                      [gpu_kv_layout.get_chunk_size() * ES] * num_gpus,
                                      list(range(num_gpus))
                                      )

    return tp_gds

@pytest.mark.skipif(TPGDSTransferThreadGroup is None,
                    reason="c_ext built with ENABLE_GDS=false")
@pytest.mark.parametrize("data_config", MHA_SIZES)
def test_mha_gds(data_config, ssd_cache_dirs):
    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    is_mla = (num_heads == 1)
    assert is_mla is False
    kv_dim = 1 if is_mla else 2
    num_gpus = NUM_GPUS

    ssd_layout_name = "BLOCKFIRST"

    gpu_layout, ssd_layout, _, kv_dim, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim,
        ssd_layout_name, is_mla, num_gpus)

    all_gpu = [make_gpu_tensors(num_layers, num_blocks, tpb, heads_per_rank, head_dim, kv_dim, g)
               for g in range(num_gpus)]

    files, num_blocks_per_file = _setup_ssd_files(ssd_layout, DTYPE, ssd_cache_dirs)

    ssd_block_stride_in_bytes = ssd_layout.get_block_stride() * ES
    ssd_tp_stride_in_bytes = (ssd_layout.get_block_stride() * ES // num_gpus
                              if not is_mla else ssd_layout.get_block_stride() * ES)
    if not is_mla:
        ssd_layout = ssd_layout.div_head(num_gpus)

    tp_gds = make_tp_gds_group(all_gpu, files, num_layers, gpu_layout, num_gpus)

    gpu_block_ids = block_ids(num_blocks)
    ssd_block_ids = block_ids(num_blocks)
    ssd_block_ids = ssd_block_ids[torch.randperm(ssd_block_ids.size(0))]

    for g in range(num_gpus):
        fill_gpu(all_gpu[g], g, num_layers, num_blocks, tpb, heads_per_rank, head_dim, kv_dim)

    sync_all(num_gpus)

    tp_gds.tp_group_transfer(gpu_block_ids,
                             ssd_block_ids,
                             ssd_layout.get_layer_stride() * ES,
                             ssd_layout.get_kv_stride() * ES,
                             ssd_block_stride_in_bytes,
                             ssd_tp_stride_in_bytes,
                             num_blocks_per_file,
                             False,
                             0, num_layers,
                             is_mla)


    for g in range(num_gpus):
        for l in range(num_layers):
            all_gpu[g][l].zero_()
    sync_all(num_gpus)

    tp_gds.tp_group_transfer(gpu_block_ids,
                             ssd_block_ids,
                             ssd_layout.get_layer_stride() * ES,
                             ssd_layout.get_kv_stride() * ES,
                             ssd_block_stride_in_bytes,
                             ssd_tp_stride_in_bytes,
                             num_blocks_per_file,
                             True,
                             0, num_layers,
                             is_mla)

    expected_gpu = 0 if is_mla else None
    for g in range(num_gpus):
        src_g = expected_gpu if expected_gpu is not None else g
        for layer in [0, num_layers - 1]:
            for block in range(num_blocks):
                for kv in range(kv_dim):
                    for hd_idx in [0, head_dim - 1]:
                        exp = expected_val(src_g, layer, block, 0, hd_idx, kv)
                        act = all_gpu[g][layer][kv, block, 0, 0, hd_idx].item()
                        assert abs(act - exp) < 1e-3, \
                            f"Non-MLA round-trip mismatch: layout={ssd_layout_name} " \
                            f"gpu={g} layer={layer} block={block} kv={kv} hd={hd_idx}: " \
                            f"expected={exp:.6f} got={act:.6f}"

    del tp_gds

@pytest.mark.parametrize("data_config", [pytest.param((4, 8, 16, 1, 512), id="mla-mini"),
                                         pytest.param((16, 16, 16, 1, 512), id="mla-multi")])
@pytest.mark.parametrize("kv_dim", [1, 2], ids=["mla", "kv_sep_1h"])
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("ssd_io_opt", [False, True], ids=["baseline", "optimized"])
@pytest.mark.parametrize("iouring_entries", [0, 512], ids=["thread", "iouring"])
def test_ssd_transfer_layerwise_h2disk(data_config, kv_dim, cpu_layout_name, ssd_io_opt,
                                       iouring_entries, tmp_path):
    """Layerwise SSD h2disk: CPU -> SSD via transfer_kv_blocks_ssd(is_read=False).

    Mirrors test_ssd_transfer_layerwise_disk2h (same MLA-only shape matrix, same
    SSD file setup / stride computation via cpu_layout_for_mode) but isolates the
    H2Disk write direction as the test target. The disk2h twin only verifies the
    SSD->CPU readback (treating the CPU->SSD write as setup); this test verifies
    the write itself by reading the SSD file back into a fresh CPU buffer and
    checking byte-level equality.

    NOTE: RegionBatchGroup::submit_layerwise touches no SSD at all
    -- it is CPU->GPU only. Both directions of SSD I/O go through the standalone
    transfer_kv_blocks_ssd: is_read=True as the hoisted DISK2H op, is_read=False
    via CPUSSDDiskTransferWorker (flexkv/transfer/transfer_engine.py), so this
    test exercises that same path with the layerwise-style strides/config to
    match the disk2h twin's wiring.

    Shared-across-ranks + sharded: num_kv_heads=1 (layerwise SSD is
    shared-across-ranks only). Covers MLA (kv_dim=1) and plain MHA, single head
    (kv_dim=2).
    """
    if not _SSD_AVAILABLE:
        pytest.skip(SSD_SKIP_REASON)
    import shutil

    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    assert num_heads == 1, "layerwise SSD h2disk test is shared-across-ranks only (num_kv_heads=1)"
    num_gpus = NUM_GPUS
    num_kv_heads = num_heads  # == 1

    gpu_layout, cpu_layout, cpu_layout_tp, _, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim,
        cpu_layout_name, kv_dim, num_kv_heads, num_gpus)

    (total_blocks, cpu_stride_kv, cpu_stride_layer,
     cpu_stride_block, cpu_stride_tp) = cpu_layout_for_mode(
        cpu_layout, cpu_layout_tp, num_layers, num_blocks,
        num_heads, head_dim, tpb, kv_dim, num_kv_heads, "sharded", num_gpus)
    cpu_chunk_bytes = cpu_layout.get_chunk_size() * ES

    # CPU source buffer (the H2Disk source). Fill with deterministic pattern.
    cpu_kv = torch.zeros(tuple(cpu_layout.kv_shape), dtype=DTYPE).pin_memory()
    _fill_cpu_kv(cpu_kv, cpu_layout, num_layers, num_blocks, tpb,
                 num_heads, head_dim, kv_dim)
    cpu_orig = cpu_kv.clone()

    # SSD layout mirrors CPU layout (1 file holds all blocks).
    ssd_layer_bytes = cpu_layout.get_layer_stride() * ES
    ssd_kv_bytes = cpu_layout.get_kv_stride() * ES

    tmpdir = str(tmp_path)
    ssd_files = _setup_ssd_file(cpu_layout, num_blocks, tmpdir)
    ioctx = SSDIOCTX(ssd_files, 1, iouring_entries, 0)

    block_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()
    layer_ids = torch.arange(num_layers, dtype=torch.int32).pin_memory()

    # H2Disk: CPU -> SSD (the test target).
    _ssd_xfer_fn(ioctx=ioctx, cpu_layer_id_list=layer_ids,
                 cpu_tensor_ptr=cpu_kv.data_ptr(),
                 ssd_block_ids=block_ids, cpu_block_ids=block_ids,
                 cpu_layer_stride_in_bytes=cpu_stride_layer,
                 cpu_kv_stride_in_bytes=cpu_stride_kv,
                 ssd_layer_stride_in_bytes=ssd_layer_bytes,
                 ssd_kv_stride_in_bytes=ssd_kv_bytes,
                 chunk_size_in_bytes=cpu_chunk_bytes,
                 block_stride_in_bytes=cpu_stride_block,
                 is_read=False, num_blocks_per_file=num_blocks,
                 round_robin=1, num_threads_per_device=32,
                 kv_dim=kv_dim, ssd_io_opt=ssd_io_opt)
    cpu_kv.zero_()

    # Read back SSD -> CPU (verification, mirrors disk2h setup write direction).
    _ssd_xfer_fn(ioctx=ioctx, cpu_layer_id_list=layer_ids,
                 cpu_tensor_ptr=cpu_kv.data_ptr(),
                 ssd_block_ids=block_ids, cpu_block_ids=block_ids,
                 cpu_layer_stride_in_bytes=cpu_stride_layer,
                 cpu_kv_stride_in_bytes=cpu_stride_kv,
                 ssd_layer_stride_in_bytes=ssd_layer_bytes,
                 ssd_kv_stride_in_bytes=ssd_kv_bytes,
                 chunk_size_in_bytes=cpu_chunk_bytes,
                 block_stride_in_bytes=cpu_stride_block,
                 is_read=True, num_blocks_per_file=num_blocks,
                 round_robin=1, num_threads_per_device=32,
                 kv_dim=kv_dim, ssd_io_opt=ssd_io_opt)

    # The CPU readback must equal the original source for both opt/baseline.
    orig_bytes = cpu_orig.view(torch.uint8).reshape(-1)
    read_bytes = cpu_kv.view(torch.uint8).reshape(-1)
    assert torch.equal(orig_bytes, read_bytes), \
        f"layerwise SSD h2disk mismatch (layout={cpu_layout_name} " \
        f"ssd_io_opt={ssd_io_opt}): " \
        f"{(orig_bytes != read_bytes).sum().item()} / {orig_bytes.numel()} bytes differ"

    del ioctx
    shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
