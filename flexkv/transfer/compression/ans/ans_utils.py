import os
import uuid
from typing import Tuple, List, Dict, Optional, Any

import torch

from flexkv.common.debug import flexkv_logger
from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV

try:
    from flexkv.c_ext import ANSTransferContext, transfer_kv_blocks_ans_comp
    _NVCOMP_AVAILABLE = True
except ImportError:
    ANSTransferContext = None
    transfer_kv_blocks_ans_comp = None
    _NVCOMP_AVAILABLE = False


AVAILABLE = _NVCOMP_AVAILABLE
SUPPORTED_DTYPES = (
    torch.bfloat16, torch.float16,
    torch.float8_e4m3fn, torch.float8_e5m2,
)

MIN_CHUNK_BYTES = 4096
# Below this per-device chunk the compression ratio drops noticeably
# (it plateaus around 16KB). Advisory threshold used only for a warning.
RATIO_PLATEAU_BYTES = 16 * 1024
SSD_PACKED_IO_ALIGN = 512


def check_dtype(dtype) -> None:
    if dtype not in SUPPORTED_DTYPES:
        raise RuntimeError(
            f"nvcomp only supports bf16/fp16/fp8, got {dtype}")


def data_type(dtype) -> int:
    """nvcomp ANS data type: 0 = FLOAT16 (bf16/fp16), 1 = UCHAR (fp8)."""
    check_dtype(dtype)
    return 0 if dtype.itemsize == 2 else 1


def check_engine_nvcomp_enable(
    gpu_handle_groups: Dict[Any, List[Any]],
    *,
    layerwise_enabled: bool,
    cpu_handle: Optional[Any],
) -> bool:
    """Engine-level enable/fallback policy for nvcomp ANS."""
    if os.environ.get("FLEXKV_ENABLE_NVCOMP", "0") != "1":
        return False
    if not AVAILABLE:
        flexkv_logger.warning(
            "[nvcomp-fallback] FLEXKV_ENABLE_NVCOMP=1 but the extension "
            "was not compiled with nvcomp support; disabling nvcomp.")
        return False

    gpu_handles = [
        gpu_handle
        for tp_gpu_handles in gpu_handle_groups.values()
        for gpu_handle in tp_gpu_handles
    ]

    # TODO(nvcomp-guard): layerwise transfer is not supported yet
    if layerwise_enabled:
        flexkv_logger.warning(
            "[nvcomp-fallback] FLEXKV_ENABLE_NVCOMP=1 but layerwise "
            "transfer is enabled; disabling nvcomp because layerwise "
            "H2D currently consumes uncompressed CPU cache blocks.")
        return False

    chunk_sizes = [
        gpu_handle.kv_layout.get_chunk_size() * gpu_handle.dtype.itemsize
        for gpu_handle in gpu_handles
    ]
    if not chunk_sizes:
        flexkv_logger.warning(
            "[nvcomp-fallback] FLEXKV_ENABLE_NVCOMP=1 but no GPU chunk sizes "
            "were available; disabling nvcomp.")
        return False

    unsupported_dtypes = sorted(
        {str(gpu_handle.dtype) for gpu_handle in gpu_handles
         if gpu_handle.dtype not in SUPPORTED_DTYPES})
    if unsupported_dtypes:
        flexkv_logger.warning(
            "[nvcomp-fallback] FLEXKV_ENABLE_NVCOMP=1 but unsupported "
            f"GPU cache dtypes were found: {unsupported_dtypes}; "
            "disabling nvcomp.")
        return False

    if min(chunk_sizes) < MIN_CHUNK_BYTES:
        flexkv_logger.warning(
            f"[nvcomp-fallback] min_chunk_size={min(chunk_sizes)}B "
            f"is below the {MIN_CHUNK_BYTES}B ANS minimum. "
            "Disabling nvcomp for GPU<->CPU and CPU<->SSD so all "
            "paths use uncompressed transfers consistently.")
        return False
    return True


def check_worker_nvcomp_enable(
    flag: Optional[bool],
    *,
    path: str,
    cpu_size_table: Optional[torch.Tensor] = None,
    ssd_size_table: Optional[torch.Tensor] = None,
    tp_size: int = 1,
    gpu_chunk_sizes_in_bytes: Optional[List[int]] = None,
) -> bool:
    if flag is None:
        flag = os.environ.get("FLEXKV_ENABLE_NVCOMP", "0") == "1"
    enabled = AVAILABLE and bool(flag)
    if not enabled:
        return False

    if path not in ("gpu_cpu", "cpu_ssd"):
        raise RuntimeError(f"Unknown nvcomp worker path: {path}")

    if cpu_size_table is None:
        raise RuntimeError(
            "nvcomp is enabled but the corresponding CPU size table "
            "was not supplied.")

    if path == "cpu_ssd" and ssd_size_table is None:
        raise RuntimeError(
            "nvcomp+SSD is enabled but the corresponding SSD size table "
            "was not supplied.")

    if tp_size > 1:
        if (gpu_chunk_sizes_in_bytes is not None and
                len(set(gpu_chunk_sizes_in_bytes)) > 1):
            raise RuntimeError(
                f"nvcomp TP requires all ranks to have identical chunk "
                f"sizes, got {gpu_chunk_sizes_in_bytes}. "
                f"Per-rank GPU layouts must be head-equally-partitioned.")
    return True


def size_table_shape(
    num_blocks: int, num_layers: int, kv_dim: int, tp_size: int,
    *, canonical: bool,
) -> Tuple[int, ...]:
    """uint32 compressed-size-table shape. Same interface for the CPU and
    SSD tables -- only num_blocks differs. canonical (tp==1 or MLA, where KV
    is replicated / not head-sharded) -> [blocks, layers, kv]; otherwise a
    per-rank stack -> [tp, blocks, layers, kv]."""
    if canonical:
        return (num_blocks, num_layers, kv_dim)
    return (tp_size, num_blocks, num_layers, kv_dim)


def allocate_size_table(
    *,
    num_blocks: int,
    num_layers: int,
    kv_dim: int,
    tp_size: int,
    canonical: bool,
    name: str,
) -> torch.Tensor:
    shape = size_table_shape(
        num_blocks, num_layers, kv_dim, tp_size, canonical=canonical)
    table = torch.zeros(shape, dtype=torch.uint32)
    flexkv_logger.info(
        f"[nvcomp-size-table] {name} allocated: "
        f"shape={tuple(table.shape)} dtype=uint32 "
        f"size={table.numel() * 4 / (1024**2):.1f} MB")
    return table


def allocate_engine_size_tables(
    *,
    cpu_handle: Any,
    ssd_handle: Optional[Any],
    cache_config: Any,
    model_config: Any,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    layout = cpu_handle.kv_layout
    num_layers = layout.num_layer
    kv_dim = 1 if layout.is_mla else 2
    tp_size = model_config.effective_tp_size_per_node
    canonical = (tp_size == 1 or layout.is_mla)

    cpu_table = allocate_size_table(
        num_blocks=cache_config.num_cpu_blocks,
        num_layers=num_layers,
        kv_dim=kv_dim,
        tp_size=tp_size,
        canonical=canonical,
        name="cpu_size_table" if canonical else "cpu_size_table_tp",
    ).share_memory_()

    # SSD table keeps compressed sizes by SSD block, so DISK2H can restore
    # them even when data lands in a different CPU slot:
    #   H2DISK: ssd_size_table[s] = cpu_size_table[c]
    #   DISK2H: cpu_size_table[c'] = ssd_size_table[s]
    # TODO(persistence): persist this table if packed SSD cache must survive restart.
    ssd_table = (
        allocate_size_table(
            num_blocks=ssd_handle.kv_layout.num_block,
            num_layers=num_layers,
            kv_dim=kv_dim,
            tp_size=tp_size,
            canonical=canonical,
            name="ssd_size_table" if canonical else "ssd_size_table_tp",
        ).share_memory_()
        if ssd_handle is not None else None
    )

    if canonical:
        return cpu_table, None, ssd_table, None
    return None, cpu_table, None, ssd_table


def ssd_packed_threads(is_mla: bool) -> Tuple[int, int]:
    """(write_threads, read_threads) for the nvcomp SSD packed path.

    Per-direction env (FLEXKV_NVCOMP_SSD_PACKED_WRITE_THREADS / _READ_THREADS)
    overrides the is_mla default. MLA reads/writes the replicated canonical
    chunk, so it benefits from more threads than head-sharded MHA.
    """
    default_write = 16
    default_read = 8
    write = int(os.environ.get(
        "FLEXKV_NVCOMP_SSD_PACKED_WRITE_THREADS", default_write))
    read = int(os.environ.get(
        "FLEXKV_NVCOMP_SSD_PACKED_READ_THREADS", default_read))
    return write, read


def get_nvcomp_batch_size(
    chunk_size_bytes: int, data_type: int = 0
) -> Tuple[int, str]:
    """Resolve the nvcomp batch_size. The env var FLEXKV_NVCOMP_BATCH_SIZE
    (> 0) overrides; otherwise auto-calibrate. Returns (batch_size, source)
    where source is "env" or "auto"."""
    if GLOBAL_CONFIG_FROM_ENV.nvcomp_batch_size > 0:
        return GLOBAL_CONFIG_FROM_ENV.nvcomp_batch_size, "env"
    return optimal_batch_size(
        chunk_size_bytes, data_type=data_type), "auto"


def create_ans_context(
    *,
    chunk_size_bytes: int,
    dtype: torch.dtype,
    is_mla: bool,
    log_prefix: str,
) -> Any:
    if chunk_size_bytes < RATIO_PLATEAU_BYTES:
        flexkv_logger.warning(
            f"{log_prefix} per-device chunk_size={chunk_size_bytes}B "
            f"< {RATIO_PLATEAU_BYTES // 1024}KB plateau; "
            "compression ratio may be low. Consider increasing "
            "tokens_per_block.")

    kv_dim = 1 if is_mla else 2
    ans_data_type = data_type(dtype)
    batch_size, batch_size_source = get_nvcomp_batch_size(
        chunk_size_bytes, data_type=ans_data_type)
    ctx = ANSTransferContext(batch_size, chunk_size_bytes, ans_data_type)
    flexkv_logger.info(
        f"{log_prefix} ANSTransferContext created: "
        f"max_chunks={ctx.max_num_chunks}, "
        f"chunk_size={ctx.max_chunk_size}, "
        f"max_comp_chunk={ctx.max_comp_chunk_bytes}, "
        f"batch_size={batch_size} ({batch_size_source}), "
        f"dtype={dtype}, kv_dim={kv_dim}")
    return ctx


def tp_worker_config(
    *,
    gpu_chunk_sizes_in_bytes: List[int],
    dtype: torch.dtype,
    tp_size: int,
) -> Tuple[int, int]:
    chunk_size = gpu_chunk_sizes_in_bytes[0]
    ans_data_type = data_type(dtype)
    if chunk_size < RATIO_PLATEAU_BYTES:
        flexkv_logger.warning(
            f"[nvcomp-tp] per-device chunk={chunk_size}B "
            f"< {RATIO_PLATEAU_BYTES // 1024}KB plateau; "
            "compression ratio may be low. Consider increasing "
            f"tokens_per_block or lowering tp_size (current tp={tp_size}).")

    batch_size, batch_size_source = get_nvcomp_batch_size(
        chunk_size, data_type=ans_data_type)
    flexkv_logger.info(
        f"[nvcomp-tp] Enabled: tp={tp_size} "
        f"chunk={chunk_size}B batch_size={batch_size} "
        f"({batch_size_source}) data_type={ans_data_type}")
    return batch_size, ans_data_type


def optimal_batch_size(
    chunk_size_bytes: int,
    data_type: int = 0,
    calibration_chunks: int = 2048,
    calibration_iters: int = 3,
) -> int:
    """Find the optimal nvcomp batch size (chunks per batch) via calibration."""
    if not _NVCOMP_AVAILABLE:
        return 4096

    NUM_WARPS_PER_CTA = 4
    MAX_SUB_CHUNKS = 64
    MIN_SUB_CHUNK = 2048
    NUM_WAVES_PER_SM = 1

    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    num_warps = props.multi_processor_count * (
        props.max_threads_per_multi_processor // 32)

    def _nvcomp_sub_chunk_size(bsz: int) -> int:
        target = (num_warps * NUM_WAVES_PER_SM + bsz - 1) // bsz
        target = min(target, MAX_SUB_CHUNKS)
        target = max(target, NUM_WARPS_PER_CTA)
        unrounded = (chunk_size_bytes + target - 1) // target
        p = 0
        v = unrounded
        while v >> 1:
            p += 1
            v >>= 1
        lower = 1 << p
        if (chunk_size_bytes + lower - 1) // lower <= MAX_SUB_CHUNKS:
            sub_chunk_size = lower
        else:
            sub_chunk_size = 1 << (p + 1)
        return max(MIN_SUB_CHUNK, sub_chunk_size)

    seen_configs: dict[int, int] = {}
    for batch_size in range(64, num_warps + 1, 8):
        sub_chunk_size = _nvcomp_sub_chunk_size(batch_size)
        if sub_chunk_size not in seen_configs:
            seen_configs[sub_chunk_size] = batch_size

    if not seen_configs:
        return 4096

    candidates: list[tuple[int, int]] = []
    sorted_sub_chunks = sorted(seen_configs.keys())
    for idx, sub_chunk_size in enumerate(sorted_sub_chunks):
        first_batch_size = seen_configs[sub_chunk_size]
        if idx + 1 < len(sorted_sub_chunks):
            last_batch_size = seen_configs[sorted_sub_chunks[idx + 1]] - 1
        else:
            last_batch_size = num_warps
        midpoint = ((first_batch_size + last_batch_size) // 2) & ~63
        midpoint = max(midpoint, first_batch_size)
        midpoint = min(midpoint, last_batch_size)
        candidates.append((midpoint, sub_chunk_size))

    if len(candidates) == 1:
        flexkv_logger.warning(
            f"nvcomp calibration: only 1 sub-chunk config found for "
            f"chunk_size={chunk_size_bytes} on this GPU; selection is trivial")
        return candidates[0][0]

    num_layers = 4
    num_blocks = max(1, calibration_chunks // (num_layers * 2))
    elem = 2 if data_type == 0 else 1
    tpb = chunk_size_bytes // elem
    nh = 1
    hs = 1
    dtype = torch.bfloat16 if data_type == 0 else torch.float8_e4m3fn

    shape_per_layer = (2, num_blocks, tpb, nh, hs)
    gpu_cache = torch.randn(
        (num_layers,) + shape_per_layer,
        dtype=torch.bfloat16,
        device=f"cuda:{dev}")
    if dtype != torch.bfloat16:
        gpu_cache = gpu_cache.to(dtype)
    gpu_blocks = [gpu_cache[i] for i in range(num_layers)]
    cpu_shape = (num_blocks, num_layers, 2, tpb, nh, hs)
    cpu_data = torch.zeros(cpu_shape, dtype=dtype, device="cpu").pin_memory()
    gpu_ptrs = torch.tensor(
        [block.data_ptr() for block in gpu_blocks],
        dtype=torch.int64).pin_memory()

    chunk_size = chunk_size_bytes
    gpu_kv_stride = num_blocks * tpb * nh * hs * elem
    gpu_block_stride = chunk_size
    gpu_layer_stride = 2 * gpu_kv_stride
    cpu_kv_stride = chunk_size
    cpu_layer_stride = 2 * chunk_size
    cpu_block_stride = num_layers * 2 * chunk_size

    gpu_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()
    cpu_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()
    stream = torch.cuda.Stream(device=dev)

    size_table = torch.zeros(
        (num_blocks, num_layers, 2), dtype=torch.uint32).pin_memory()
    size_table_ptr = size_table.data_ptr()
    size_table_block_stride = size_table.stride(0)
    size_table_layer_stride = size_table.stride(1)

    best_batch_size = candidates[0][0]
    best_time = float("inf")
    results = []

    start_evt = torch.cuda.Event(enable_timing=True)
    end_evt = torch.cuda.Event(enable_timing=True)

    assert ANSTransferContext is not None
    assert transfer_kv_blocks_ans_comp is not None
    for batch_size, sub_chunk_size in candidates:
        ctx = ANSTransferContext(batch_size, chunk_size, data_type)
        for _ in range(2):
            with torch.cuda.stream(stream):
                transfer_kv_blocks_ans_comp(
                    ctx, gpu_ids, gpu_ptrs, gpu_kv_stride, gpu_block_stride,
                    gpu_layer_stride, cpu_ids, cpu_data, cpu_kv_stride,
                    cpu_layer_stride, cpu_block_stride, chunk_size, 0,
                    num_layers, False, 0, size_table_ptr,
                    size_table_block_stride, size_table_layer_stride)
            stream.synchronize()

        start_evt.record(stream)
        for _ in range(calibration_iters):
            with torch.cuda.stream(stream):
                transfer_kv_blocks_ans_comp(
                    ctx, gpu_ids, gpu_ptrs, gpu_kv_stride, gpu_block_stride,
                    gpu_layer_stride, cpu_ids, cpu_data, cpu_kv_stride,
                    cpu_layer_stride, cpu_block_stride, chunk_size, 0,
                    num_layers, False, 0, size_table_ptr,
                    size_table_block_stride, size_table_layer_stride)
        end_evt.record(stream)
        stream.synchronize()
        elapsed = start_evt.elapsed_time(end_evt)
        ctx.destroy()
        results.append((batch_size, sub_chunk_size, elapsed))

        if elapsed < best_time:
            best_time = elapsed
            best_batch_size = batch_size

    del gpu_cache, gpu_blocks, cpu_data, gpu_ptrs, gpu_ids, cpu_ids, size_table
    torch.cuda.empty_cache()

    detail = ", ".join(
        f"bsz={batch_size} sc={sub_chunk_size} {elapsed:.2f}ms"
        for batch_size, sub_chunk_size, elapsed in results)
    flexkv_logger.info(
        f"nvcomp batch_size calibration: [{detail}], "
        f"selected bsz={best_batch_size}")

    return best_batch_size
