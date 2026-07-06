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
"""

import pytest
import torch

from flexkv.c_ext import TPTransferThreadGroup, LayerwiseTransferGroup
from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType


# ---------------------------------------------------------------------------
# Skip conditions
# ---------------------------------------------------------------------------

NUM_GPUS = min(4, torch.cuda.device_count()) if torch.cuda.is_available() else 0

pytestmark = pytest.mark.skipif(
    NUM_GPUS < 2,
    reason=f"Need at least 2 GPUs, found {NUM_GPUS}"
)


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
    except Exception:
        globals()[cache_key] = False
        return False


def skip_if_engine_unsupported(use_ce):
    """Skip test if the engine probe failed (kernel needs float4 alignment, CE
    needs CUDA runtime, etc.)."""
    if not _probe_engine(use_ce):
        kind = "CE (cudaMemcpyAsync)" if use_ce else "CUDA kernel"
        pytest.skip(f"{kind} engine not available on this platform")


# ---------------------------------------------------------------------------
# Test configurations
# ---------------------------------------------------------------------------

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

MLA_MODES = ["sharded", "all_write", "rank0_only"]


# ---------------------------------------------------------------------------
# Helpers (matching production code in worker.py / layerwise.py)
# ---------------------------------------------------------------------------

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
                  ce_use_pingpong=None,
                  ce_path_opt=None):
    """Create TPTransferThreadGroup with strides from KVCacheLayout.

    Matches production worker.py:472 exactly — chunk_size does NOT include kv_dim.
    The C++ kernel iterates num_chunks = num_layers * kv_dim * num_blocks and
    copies chunk_size bytes per chunk, so kv_dim is a separate iteration axis.

    CE config defaults from GLOBAL_CONFIG_FROM_ENV (same as production).
    """
    if ce_segment_threshold is None:
        ce_segment_threshold = GLOBAL_CONFIG_FROM_ENV.transfer_segment_threshold
    if ce_use_pingpong is None:
        ce_use_pingpong = GLOBAL_CONFIG_FROM_ENV.transfer_pingpong
    if ce_path_opt is None:
        ce_path_opt = GLOBAL_CONFIG_FROM_ENV.transfer_path_opt
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
        ce_use_pingpong=ce_use_pingpong,
    )


def make_layerwise_group(cpu_ptr_unused, all_gpu, num_gpus, gpu_layout, num_layers,
                         ce_segment_threshold=None,
                         ce_use_pingpong=None,
                         ce_path_opt=None):
    """Create LayerwiseTransferGroup for H2D-only testing (no SSD, no eventfd).

    Mirrors LayerwiseTransferWorker construction in layerwise.py. GPU chunk_size
    does NOT include kv_dim (same as tp_group). SSD disabled via empty ssd_files;
    eventfd disabled via empty layer_eventfds_tensor.

    CE config defaults from GLOBAL_CONFIG_FROM_ENV (same as production).
    """
    if ce_segment_threshold is None:
        ce_segment_threshold = GLOBAL_CONFIG_FROM_ENV.transfer_segment_threshold
    if ce_use_pingpong is None:
        ce_use_pingpong = GLOBAL_CONFIG_FROM_ENV.transfer_pingpong
    if ce_path_opt is None:
        ce_path_opt = GLOBAL_CONFIG_FROM_ENV.transfer_path_opt
    def strides_tensor(getter):
        return torch.tensor([getter() * ES] * num_gpus, dtype=torch.int64)

    return LayerwiseTransferGroup(
        num_gpus=num_gpus,
        gpu_blocks=all_gpu,
        cpu_blocks=cpu_ptr_unused,  # actual pinned CPU tensor
        ssd_files={},
        num_layers=num_layers,
        gpu_kv_strides_tensor=strides_tensor(gpu_layout.get_kv_stride),
        gpu_block_strides_tensor=strides_tensor(gpu_layout.get_block_stride),
        gpu_layer_strides_tensor=strides_tensor(gpu_layout.get_layer_stride),
        gpu_chunk_sizes_tensor=strides_tensor(gpu_layout.get_chunk_size),
        iouring_entries=0,
        iouring_flags=0,
        layer_eventfds_tensor=torch.empty(0, dtype=torch.int32),
        tp_size=num_gpus,
        ce_segment_threshold=ce_segment_threshold,
        ce_use_pingpong=ce_use_pingpong,
    )


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


# ---------------------------------------------------------------------------
# Round-trip tests (D2H -> clear GPU -> H2D -> verify)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("data_config", MHA_SIZES)
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("engine_name,use_ce", ENGINES)
def test_non_mla_roundtrip(data_config, cpu_layout_name, engine_name, use_ce):
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

    tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus, gpu_layout, num_layers)
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


# ---------------------------------------------------------------------------
# MLA mode tests (sharded / all_write / rank0_only)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("data_config", MLA_SIZES)
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("engine_name,use_ce", ENGINES)
@pytest.mark.parametrize("mode", MLA_MODES)
def test_mla_roundtrip_modes(data_config, cpu_layout_name, engine_name, use_ce, mode):
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

    tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus, gpu_layout, num_layers)
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


# ---------------------------------------------------------------------------
# Invalid mode fallback test
# ---------------------------------------------------------------------------

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
    tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus, gpu_layout, num_layers)

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


# ---------------------------------------------------------------------------
# CE adaptive path tests (Path 0/1/2)
#
# The C++ CE engine selects among three code paths based on block-id
# contiguity and CPU layout:
#   Path 0 — single large memcpy (needs contiguous ids + LAYERFIRST)
#   Path 1 — per-segment memcpy (few segments ≤ threshold=8)
#            BLOCKFIRST uses staging buffer + CPU scatter + ping-pong
#   Path 2 — GPU gather/scatter pipeline (many segments > threshold)
#
# We trigger each path by constructing block-id *permutations* of [0..N-1]
# so that every block is still transferred (round-trip correctness preserved):
#   contiguous — identity permutation        → 1 segment
#   few_seg    — interleaved 4-segment perm  → 4 segments
#   scattered  — random permutation           → N segments (>8)
#
# NOTE: FLEXKV_TRANSFER_FORCE_PATH / FLEXKV_TRANSFER_PINGPONG are read once
# and cached as C++ static locals, so they CANNOT be toggled within a single
# process.  We rely on block-id patterns to exercise auto path selection, and
# use subprocesses for ping-pong on/off verification.
# ---------------------------------------------------------------------------

CE_PATTERN_SIZES = [
    # (num_layers, num_blocks, tpb, num_heads, head_dim, is_mla, id)
    pytest.param((4, 16, 16, 1, 512), True, id="mla-16blk"),
    pytest.param((32, 64, 16, 1, 512), True, id="mla-llama8b"),
    pytest.param((4, 16, 16, 8, 128), False, id="mha-16blk"),
    pytest.param((32, 64, 16, 8, 128), False, id="mha-llama8b"),
]

CE_PATTERNS = ["contiguous", "few_seg", "scattered"]


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


def _expected_path(pattern_name, cpu_layout_name):
    """Predict which CE path auto-selection should pick.

    Path 0 needs src_log_contig && dst_log_contig && dst_phys_contig.
    dst_phys_contig is true only for LAYERFIRST.
    Threshold defaults to 8; scattered yields N segments (>8 for N>=16).
    """
    if pattern_name == "contiguous" and cpu_layout_name == "LAYERFIRST":
        return 0
    if pattern_name == "contiguous":
        return 1  # BLOCKFIRST: 1 segment but not phys-contig
    if pattern_name == "few_seg":
        return 1  # 4 segments ≤ 8
    return 2  # scattered: N > 8 segments


@pytest.mark.parametrize("data_config,is_mla", CE_PATTERN_SIZES)
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("pattern", CE_PATTERNS)
@pytest.mark.parametrize("path_opt", [False, True], ids=["baseline", "optimized"])
@pytest.mark.parametrize("mode", ["sharded", "all_write", "rank0_only"])
def test_ce_paths_roundtrip(data_config, is_mla, cpu_layout_name, pattern, mode, path_opt):
    """CE Path 0/1/2 round-trip correctness via block-id patterns.

    Tests all MLA modes (sharded, all_write, rank0_only) × layouts × patterns.
    Each pattern triggers a different auto-selected CE path:
      contiguous -> Path 0 (LF) / Path 1 (BF)
      few_seg    -> Path 1
      scattered  -> Path 2 (non-sharded) / Path 1 per-block (sharded D2H)
    """
    skip_if_engine_unsupported(use_ce=True)
    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    if pattern == "scattered":
        assert num_blocks > 8, "scattered needs >8 blocks for path 2"
    if not is_mla and mode != "sharded":
        pytest.skip("MLA modes only apply to MLA models")
    # Non-MLA doesn't have MLA modes
    if not is_mla and mode != "sharded":
        pytest.skip("MLA modes only apply to MLA models")

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

    cpu_kv = make_cpu_tensor(cpu_layout, num_layers, num_blocks)
    tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus,
                       gpu_layout, num_layers)

    ids = make_block_id_pattern(pattern, num_blocks)
    gpu_block_ids = ids
    cpu_block_ids = ids

    cpu_stride_kv = cpu_layout_tp.get_kv_stride() * ES
    cpu_stride_layer = cpu_layout_tp.get_layer_stride() * ES
    cpu_stride_block = cpu_layout.get_block_stride() * ES
    cpu_stride_tp = cpu_stride_block // num_gpus

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


@pytest.mark.parametrize("data_config,is_mla", CE_PATTERN_SIZES)
@pytest.mark.parametrize("cpu_layout_name", CPU_LAYOUTS)
@pytest.mark.parametrize("pattern", CE_PATTERNS)
@pytest.mark.parametrize("path_opt", [False, True], ids=["baseline", "optimized"])
@pytest.mark.parametrize("mode", ["sharded", "all_write", "rank0_only"])
def test_ce_paths_layerwise_h2d(data_config, is_mla, cpu_layout_name, pattern, mode, path_opt):
    """CE Path 0/1/2 correctness for LayerwiseTransferGroup H2D.

    Uses TPTransferThreadGroup D2H (already verified correct) to prepare
    CPU data, then LayerwiseTransferGroup H2D to read it back with the
    same block-id pattern.  Verifies that the layerwise CE path produces
    identical results to the TP-group CE path.
    """
    skip_if_engine_unsupported(use_ce=True)
    num_layers, num_blocks, tpb, num_heads, head_dim = data_config
    if pattern == "scattered":
        assert num_blocks > 8, "scattered needs >8 blocks for path 2"
    if not is_mla and mode != "sharded":
        pytest.skip("MLA modes only apply to MLA models")

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

    # all_write needs num_gpus * num_blocks CPU blocks
    total_blocks = num_blocks * num_gpus if (is_mla and mode == "all_write") else num_blocks
    cpu_kv = make_cpu_tensor(cpu_layout, num_layers, total_blocks)
    ids = make_block_id_pattern(pattern, num_blocks)

    cpu_stride_kv = cpu_layout_tp.get_kv_stride() * ES
    cpu_stride_layer = cpu_layout_tp.get_layer_stride() * ES
    cpu_stride_block = cpu_layout.get_block_stride() * ES
    cpu_stride_tp = cpu_stride_block // num_gpus
    # all_write expands CPU buffer to total_blocks = num_gpus * num_blocks.
    # LAYERFIRST strides depend on num_blocks, so multiply by num_gpus.
    # BLOCKFIRST strides don't depend on num_blocks (block_stride includes layers).
    if is_mla and mode == "all_write" and cpu_layout_name.upper() == "LAYERFIRST":
        cpu_stride_kv *= num_gpus
        cpu_stride_layer *= num_gpus
    chunk_size = gpu_layout.get_chunk_size() * ES

    # Step 1: D2H via TPTransferThreadGroup (prepare CPU data)
    tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus,
                       gpu_layout, num_layers)
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

    # Step 2: H2D via LayerwiseTransferGroup (test target)
    lw_group = make_layerwise_group(cpu_kv, all_gpu, num_gpus,
                                    gpu_layout, num_layers)
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
        num_layers=num_layers, layer_granularity=num_layers,
        is_mla=is_mla, mla_d2h_mode=mode,
    )
    sync_all(num_gpus)
    del lw_group

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
                            "gpu={} layer={} block={} kv={} hd={}: " \
                            "expected={:.6f} got={:.6f}".format(
                                pattern, cpu_layout_name, g, layer, block,
                                kv, hd_idx, exp, act)


def test_ce_path_selection_documentation():
    """Document which path each (pattern, layout) combination triggers.

    This is a lightweight smoke test — it does not introspect the C++ path
    choice (that would require exposing an internal API), but it documents
    the expected mapping and ensures all combos run without error.
    """
    skip_if_engine_unsupported(use_ce=True)
    print("\n  CE auto path selection matrix:")
    print("  {:<12s}  {:<12s}  {}".format("pattern", "layout", "path"))
    print("  " + "-" * 40)
    for pattern in CE_PATTERNS:
        for layout in ["LAYERFIRST", "BLOCKFIRST"]:
            path = _expected_path(pattern, layout)
            print("  {:<12s}  {:<12s}  {}".format(pattern, layout, path))


# ---------------------------------------------------------------------------
# Ping-pong consistency test (subprocess — static cache requires separate procs)
# ---------------------------------------------------------------------------

def _run_pingpong_roundtrip(pingpong_val):
    """Run a single CE round-trip with the given ping-pong setting.

    Executed in a subprocess where FLEXKV_TRANSFER_PINGPONG env var is set
    before Python starts, so GLOBAL_CONFIG_FROM_ENV picks it up at import
    time and passes it to TPTransferThreadGroup via ce_use_pingpong.
    """
    num_layers, num_blocks, tpb = 4, 16, 16
    num_heads, head_dim = 1, 512
    is_mla = True
    num_gpus = NUM_GPUS

    gpu_layout, cpu_layout, cpu_layout_tp, kv_dim, heads_per_rank = make_layouts(
        num_layers, num_blocks, tpb, num_heads, head_dim,
        "BLOCKFIRST", is_mla, num_gpus)

    all_gpu = [make_gpu_tensors(num_layers, num_blocks, tpb,
                               heads_per_rank, head_dim, kv_dim, g)
               for g in range(num_gpus)]
    fill_gpu(all_gpu[0], 0, num_layers, num_blocks, tpb,
             heads_per_rank, head_dim, kv_dim)
    for g in range(1, num_gpus):
        for l in range(num_layers):
            all_gpu[g][l].copy_(all_gpu[0][l])
    sync_all(num_gpus)

    cpu_kv = make_cpu_tensor(cpu_layout, num_layers, num_blocks)
    tp = make_tp_group(cpu_kv.data_ptr(), all_gpu, num_gpus,
                       gpu_layout, num_layers)

    # few_seg pattern → path 1 with BLOCKFIRST staging + ping-pong
    ids = make_block_id_pattern("few_seg", num_blocks)

    cpu_stride_kv = cpu_layout_tp.get_kv_stride() * ES
    cpu_stride_layer = cpu_layout_tp.get_layer_stride() * ES
    cpu_stride_block = cpu_layout.get_block_stride() * ES
    cpu_stride_tp = cpu_stride_block // num_gpus
    # all_write expands CPU buffer to total_blocks = num_gpus * num_blocks.
    # LAYERFIRST strides depend on num_blocks, so multiply by num_gpus.
    # BLOCKFIRST strides don't depend on num_blocks (block_stride includes layers).
    if is_mla and mode == "all_write" and cpu_layout_name.upper() == "LAYERFIRST":
        cpu_stride_kv *= num_gpus
        cpu_stride_layer *= num_gpus

    tp.tp_group_transfer(
        gpu_block_id_tensor=ids, cpu_block_id_tensor=ids,
        cpu_kv_stride_in_bytes=cpu_stride_kv,
        cpu_layer_stride_in_bytes=cpu_stride_layer,
        cpu_block_stride_in_bytes=cpu_stride_block,
        cpu_tp_stride_in_bytes=cpu_stride_tp,
        transfer_num_cta=4, is_host_to_device=False, use_ce_transfer=True,
        layer_id=0, layer_granularity=num_layers, is_mla=True,
        mla_d2h_mode=mode,
    )
    sync_all(num_gpus)
    del tp
    return cpu_kv.clone()


def test_ce_pingpong_consistency():
    """Ping-pong on/off produce identical D2H data (BLOCKFIRST staging).

    Uses subprocesses because FLEXKV_TRANSFER_PINGPONG is a C++ static-cached
    env var — toggling it in-process has no effect after first read.
    """
    import subprocess
    import sys
    import os

    if NUM_GPUS < 2:
        pytest.skip("Need at least 2 GPUs for ping-pong consistency test")
    skip_if_engine_unsupported(use_ce=True)

    this_file = os.path.abspath(__file__)
    env = os.environ.copy()

    results = {}
    for pp in ["0", "1"]:
        env_pp = env.copy()
        env_pp["FLEXKV_TRANSFER_PINGPONG"] = pp
        env_pp["FLEXKV_PINGPONG_CHECK"] = pp
        proc = subprocess.run(
            [sys.executable, this_file, "--pingpong-check", pp],
            env=env_pp, capture_output=True, text=True, timeout=120)
        if proc.returncode != 0:
            pytest.fail(
                "ping-pong={} subprocess failed: {}".format(
                    pp, proc.stderr[-2000:] if proc.stderr else proc.stdout))
        results[pp] = proc.stdout.strip()

    # Both subprocesses should report success and identical checksums
    assert results["0"] == results["1"], \
        "ping-pong on/off produced different data:\n" \
        "  on:  {}\n  off: {}".format(results["1"], results["0"])


# ---------------------------------------------------------------------------
# CLI entry for subprocess-based ping-pong check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    if "--pingpong-check" in sys.argv:
        pp = sys.argv[sys.argv.index("--pingpong-check") + 1]
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            print("SKIP: need >=2 GPUs", file=sys.stderr)
            sys.exit(0)
        try:
            cpu_kv = _run_pingpong_roundtrip(pp)
            # Print a checksum so the parent can compare on/off
            print("{:.6f}".format(float(cpu_kv.float().sum().item())))
            sys.exit(0)
        except Exception as e:
            print("ERROR: {}".format(e), file=sys.stderr)
            sys.exit(1)
    pytest.main([__file__, "-v", "-s"])
