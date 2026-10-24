import os
import uuid
from typing import Tuple, List, Dict, NamedTuple, Optional, Any, Sequence, Union

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
    torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8,
)

MIN_CHUNK_BYTES = 4096
# Below this per-device chunk the compression ratio drops noticeably
# (it plateaus around 16KB). Advisory threshold used only for a warning.
RATIO_PLATEAU_BYTES = 16 * 1024
SSD_PACKED_IO_ALIGN = 512

# Which layer groups of a heterogeneous-KV model ANS compresses.
#   "auto" (default) -- every group the kernels can address AND that ANS has
#                       headroom to shrink; see ``select_nvcomp_groups``
#   "all"            -- every addressable group, headroom or not
#   "0" / "0,1"      -- exactly those ordinals; a named group that is
#                       ineligible is an error, not a silent drop
#   "none"           -- nothing, which disables nvcomp
# The list is resolved once, in the engine process, and handed to the worker
# strategy; it is never re-read inside a worker. See ``select_nvcomp_groups``.
NVCOMP_GROUPS_ENV = "FLEXKV_NVCOMP_GROUPS"

# Element types ANS can address but has no headroom on: already-quantized KV
# is close to maximum entropy, so the compressed chunk routinely comes out
# LARGER than the slot the CPU tier reserved for it -- which is not a silent
# corruption but a hard failure ("compressed payload exceeded the CPU chunk
# slot") that fails the whole transfer graph and the request with it.
#
# Measured on GLM-5.3: group 0 (bf16 MLA latent) compresses 1.42x, while
# group 1 (the uint8 DSA indexer, fp8 scales packed in) overflowed on every
# single store. So "auto" skips these, and ``FLEXKV_NVCOMP_GROUPS=all`` or an
# explicit ordinal is how you ask for one anyway.
LOW_HEADROOM_DTYPES = (
    torch.float8_e4m3fn, torch.float8_e5m2, torch.uint8,
)

# Sentinel for "read the env"; ``None`` already means "auto" on the wire.
_FROM_ENV = object()

# Sentinel for "all": every addressable group, skipping the headroom
# heuristic. Distinct from an explicit ordinal list, which treats an
# unaddressable group as a configuration error rather than dropping it.
_ALL_GROUPS = object()


def check_dtype(dtype) -> None:
    if dtype not in SUPPORTED_DTYPES:
        raise RuntimeError(
            f"nvcomp only supports bf16/fp16/fp8/uint8, got {dtype}")


def data_type(dtype) -> int:
    """Map torch dtype to nvCOMP ANS symbol type.

    0 = FLOAT16, 1 = UCHAR, 2 = native FLOAT8_E4M3 (nvCOMP >= 5.3).
    """
    check_dtype(dtype)
    if dtype.itemsize == 2:
        return 0
    if dtype == torch.float8_e4m3fn:
        return 2
    return 1


def check_engine_nvcomp_enable(
    gpu_handle_groups: Dict[Any, List[Any]],
    *,
    layerwise_enabled: bool,
    cpu_handle: Optional[Any],
    model_config: Optional[Any] = None,
    ssd_handle: Optional[Any] = None,
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

    # TODO(nvcomp-guard): the packed 4D KV layout is not supported yet. The ANS
    # kernels take a single kv_dim that means both "one KV region" and "the
    # per-rank size table is canonical", and a packed layout wants only the
    # former, so neither value addresses it correctly.
    if cpu_handle is not None and cpu_handle.kv_layout.kv_dim == 1 and cpu_handle.kv_layout.num_kv_heads > 1:
        flexkv_logger.warning(
            "[nvcomp-fallback] FLEXKV_ENABLE_NVCOMP=1 but the KV cache uses "
            "the packed 4D layout, which the ANS kernels cannot address yet; "
            "disabling nvcomp.")
        return False

    # Heterogeneous (multi-group) KV: a DSA model (DeepSeek-V4, GLM-5.3) splits
    # its layers into groups with different geometry -- group 0 is the MLA
    # latent, group 1 the indexer. Which of them get compressed is decided by
    # ``select_nvcomp_groups``; an empty selection means nvcomp has nothing to
    # do and the whole feature is declined.
    #
    # This has to be decided here rather than in the strategy, which is where
    # the refusal used to live: by the time a strategy attaches it is already
    # inside a spawned worker process, so raising there kills the worker and
    # the engine waits on a `ready` that can never arrive. Declining at the
    # engine level degrades to uncompressed transfers instead, matching every
    # other guard in this function. `layer_groups` is the same condition the
    # worker keys on -- it is what makes `tp_group_transfer_groups` non-None.
    layer_groups = getattr(model_config, "layer_groups", None)
    if layer_groups and not check_multi_group_nvcomp(
            layer_groups,
            tokens_per_block=getattr(cpu_handle.kv_layout, "tokens_per_block",
                                     None) if cpu_handle is not None else None,
            default_dtype=getattr(model_config, "dtype", None),
            has_ssd=ssd_handle is not None):
        return False

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


class GroupPlan(NamedTuple):
    """One compressed layer group, as the engine hands it to a worker.

    ``table_layer_offset`` is the group's first row in the size table's layer
    dimension. Every group -- compressed or not -- owns a private stretch of
    ``num_layers`` rows at the running prefix sum, so two compressed groups can
    never alias each other's lengths. Offsets are computed over *all* groups,
    not just the selected ones, so a table allocated under one selection is
    still addressed correctly under another.
    """
    group_index: int
    table_layer_offset: int
    num_layers: int


def group_table_layer_offsets(layer_groups: Sequence[Any]) -> List[int]:
    """Per-group first row in the size table's layer dimension (prefix sum)."""
    offsets: List[int] = []
    running = 0
    for group in layer_groups:
        offsets.append(running)
        running += int(group.num_layers)
    return offsets


def group_low_headroom_reason(
    group: Any, *, default_dtype: Optional[torch.dtype],
) -> Optional[str]:
    """Why ``auto`` should not pick this group even though ANS can address it.

    Separate from ``group_nvcomp_ineligible_reason`` because the two answer
    different questions: that one is "would this even run", this one is "is it
    worth running". Only ``auto`` consults this; an explicitly named group is
    compressed regardless, which is how you override the heuristic.
    """
    dtype = getattr(group, "dtype", None) or default_dtype
    if dtype in LOW_HEADROOM_DTYPES:
        return (f"dtype {dtype} is already quantized, so ANS is likely to "
                "overflow the CPU chunk slot rather than shrink it")
    return None


def group_nvcomp_ineligible_reason(
    group: Any,
    *,
    tokens_per_block: Optional[int],
    default_dtype: Optional[torch.dtype],
) -> Optional[str]:
    """Why ANS cannot compress this layer group, or ``None`` if it can.

    Every check here is about whether the *kernels* can address the group, not
    about whether compressing it is a good idea. Whether it is worthwhile is
    ``group_low_headroom_reason``, which only ``auto`` consults -- so an
    operator naming a group explicitly still gets it.
    """
    dtype = getattr(group, "dtype", None) or default_dtype
    if dtype not in SUPPORTED_DTYPES:
        return f"dtype {dtype} is not one ANS supports"

    if tokens_per_block is not None:
        chunk = group_chunk_bytes(
            group, tokens_per_block=tokens_per_block, default_dtype=default_dtype)
        if chunk % (getattr(group, "num_kv_heads", 1) or 1):
            # Unreachable arithmetically; kept so a future geometry change
            # trips here rather than in the kernel.
            return f"chunk={chunk}B does not divide by the head count"
        if chunk < MIN_CHUNK_BYTES:
            return f"chunk={chunk}B is below the {MIN_CHUNK_BYTES}B ANS minimum"
    return None


def group_chunk_bytes(
    group: Any, *, tokens_per_block: int,
    default_dtype: Optional[torch.dtype],
) -> int:
    """Bytes of one (layer, kv) slot of one block of this group.

    Same formula as ``flexkv.transfer.template.group_chunk_bytes``; duplicated
    rather than imported because that module pulls in the storage layer and
    this one is imported from the engine before any layout exists.
    ``compress_ratio`` shrinks the token dimension on both sides of the
    transfer, so it is a smaller chunk rather than a ratio between the two.
    """
    dtype = getattr(group, "dtype", None) or default_dtype
    tokens = tokens_per_block // getattr(group, "compress_ratio", 1)
    return (tokens * group.num_kv_heads * group.head_size * dtype.itemsize)


def _parse_groups_env(
    raw: str, num_groups: int,
) -> Union[None, List[int], object]:
    """``FLEXKV_NVCOMP_GROUPS`` -> requested ordinals.

    Returns ``None`` for "auto" and the ``_ALL_GROUPS`` sentinel for "all".
    "all" is not just ``range(num_groups)``: naming an ordinal the kernels
    cannot address is a configuration error and declines nvcomp, whereas "all"
    means "every group you can address" and drops the ones you cannot -- it
    overrides the headroom heuristic, not the addressability check.
    """
    value = raw.strip().lower()
    if value in ("", "auto"):
        return None
    if value == "all":
        return _ALL_GROUPS
    if value == "none":
        return []
    requested: List[int] = []
    for piece in value.replace(" ", "").split(","):
        if not piece:
            continue
        try:
            idx = int(piece)
        except ValueError:
            raise ValueError(
                f"{NVCOMP_GROUPS_ENV}={raw!r}: {piece!r} is not an integer, "
                "'auto', 'all', or 'none'.")
        if not 0 <= idx < num_groups:
            raise ValueError(
                f"{NVCOMP_GROUPS_ENV}={raw!r}: group {idx} is out of range for "
                f"a model with {num_groups} layer group(s).")
        if idx not in requested:
            requested.append(idx)
    return requested


def select_nvcomp_groups(
    layer_groups: Sequence[Any],
    *,
    tokens_per_block: Optional[int],
    default_dtype: Optional[torch.dtype],
    has_ssd: bool = False,
    env: Union[str, None, Any] = _FROM_ENV,
) -> List[GroupPlan]:
    """Which layer groups ANS compresses, and where each one's table rows are.

    Resolved once in the engine process and handed to the worker strategy, so
    the engine's accept decision and the worker's binding can never disagree --
    they are the same list, not two copies of a constant.

    ``FLEXKV_NVCOMP_GROUPS`` selects:

    * ``auto`` (default) -- every group the kernels can address *and* that ANS
      has headroom to shrink. A group whose data is already quantized is
      skipped: it does not merely compress badly, it overflows the CPU chunk
      slot and fails the whole transfer graph (see ``LOW_HEADROOM_DTYPES``).
      That failure is loud rather than corrupting, but a default that makes
      every store fail is still the wrong default.
    * ``all`` -- every addressable group, headroom or not. This is the escape
      hatch for measuring the heuristic rather than trusting it.
    * ``0`` / ``0,1`` -- exactly those ordinals, headroom or not. Naming a
      group the kernels cannot address is a configuration error, so it
      declines nvcomp outright rather than quietly compressing a subset of
      what was asked for.
    * ``none`` -- nothing, which disables nvcomp.

    Returns an empty list when nvcomp should not run.
    """
    # The SSD tier cannot carry a compressed CPU block on this layout.
    # CPUSSDDiskTransferWorker moves a multi-group block as one opaque blob
    # (``_init_multi_group_ssd``: CPU and SSD share a byte-identical BLOCKFIRST
    # block, so there is no per-chunk addressing to hang a table off), which
    # means the compressed *lengths* never travel with the bytes. The uniform
    # path avoids this by packing through NvcompCpuSsdStrategy, which keeps
    # ssd_size_table[s] alongside and restores cpu_size_table[c'] on DISK2H.
    # Blob-copying instead would restore a CPU block into a slot whose table
    # row still holds some earlier block's lengths -- decompressed at the wrong
    # length, with no error. Decline rather than half-support it.
    if has_ssd:
        flexkv_logger.warning(
            "[nvcomp-fallback] multi-group KV with an SSD tier: CPU<->SSD "
            "moves these blocks as opaque blobs and cannot carry the "
            "compressed-size table, so a DISK2H restore would decompress at a "
            "stale length. Disabling nvcomp.")
        return []

    raw = os.environ.get(NVCOMP_GROUPS_ENV, "auto") if env is _FROM_ENV else (
        "auto" if env is None else str(env))
    try:
        requested = _parse_groups_env(raw, len(layer_groups))
    except ValueError as exc:
        flexkv_logger.warning(f"[nvcomp-fallback] {exc} Disabling nvcomp.")
        return []

    if requested == []:
        flexkv_logger.info(
            f"[nvcomp-multi-group] {NVCOMP_GROUPS_ENV}={raw!r} selects no "
            "layer groups; disabling nvcomp.")
        return []

    offsets = group_table_layer_offsets(layer_groups)
    reasons = [
        group_nvcomp_ineligible_reason(
            group, tokens_per_block=tokens_per_block,
            default_dtype=default_dtype)
        for group in layer_groups
    ]

    if requested is None:
        # "auto" answers two questions per group, and they are deliberately
        # separate: can the kernels address it, and is compressing it worth
        # doing. Only auto asks the second one -- "all" and an explicit
        # ordinal take the group regardless, so the heuristic is overridable.
        skips = [
            reason or group_low_headroom_reason(
                group, default_dtype=default_dtype)
            for reason, group in zip(reasons, layer_groups)
        ]
        chosen = [gi for gi, reason in enumerate(skips) if reason is None]
        for gi, reason in enumerate(skips):
            if reason is not None:
                flexkv_logger.info(
                    f"[nvcomp-multi-group] layer group {gi} transfers "
                    f"uncompressed: {reason}.")
    elif requested is _ALL_GROUPS:
        # "all" overrides the headroom heuristic but not addressability: a
        # group the kernels cannot reach is dropped, same as under auto.
        chosen = [gi for gi, reason in enumerate(reasons) if reason is None]
        for gi, reason in enumerate(reasons):
            if reason is not None:
                flexkv_logger.info(
                    f"[nvcomp-multi-group] layer group {gi} transfers "
                    f"uncompressed: {reason}.")
    else:
        # An explicitly named group that cannot be compressed is a config
        # error. Silently dropping it would leave the operator believing a
        # measurement covered a group it never touched.
        rejected = [(gi, reasons[gi]) for gi in requested
                    if reasons[gi] is not None]
        if rejected:
            detail = "; ".join(f"group {gi}: {r}" for gi, r in rejected)
            flexkv_logger.warning(
                f"[nvcomp-fallback] {NVCOMP_GROUPS_ENV}={raw!r} names "
                f"group(s) ANS cannot compress ({detail}). Disabling nvcomp "
                "rather than compressing a subset of what was requested.")
            return []
        chosen = list(requested)

    if not chosen:
        flexkv_logger.warning(
            "[nvcomp-fallback] multi-group KV: no layer group can be "
            "compressed; disabling nvcomp.")
        return []

    plans = [
        GroupPlan(group_index=gi,
                  table_layer_offset=offsets[gi],
                  num_layers=int(layer_groups[gi].num_layers))
        for gi in sorted(chosen)
    ]
    described = ", ".join(
        f"{p.group_index}({p.num_layers}L, "
        f"dtype={getattr(layer_groups[p.group_index], 'dtype', None) or default_dtype}, "
        f"rows {p.table_layer_offset}..{p.table_layer_offset + p.num_layers - 1})"
        for p in plans)
    skipped = len(layer_groups) - len(plans)
    flexkv_logger.info(
        f"[nvcomp-multi-group] {NVCOMP_GROUPS_ENV}={raw!r} -> compressing "
        f"layer group(s) {described}; "
        f"{skipped} other group(s) transfer uncompressed.")
    return plans


def check_multi_group_nvcomp(
    layer_groups: List[Any],
    *,
    tokens_per_block: Optional[int],
    default_dtype: Optional[torch.dtype],
    has_ssd: bool = False,
) -> bool:
    """Whether nvcomp can run on a heterogeneous-KV model at all.

    Thin wrapper so the engine's accept/decline is literally
    ``select_nvcomp_groups`` finding something to do, rather than a second
    predicate that could drift from it.
    """
    return bool(select_nvcomp_groups(
        layer_groups,
        tokens_per_block=tokens_per_block,
        default_dtype=default_dtype,
        has_ssd=has_ssd,
    ))


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
    # On a heterogeneous-KV model every group gets a private stretch of the
    # table's layer dimension, at the prefix sum of the groups' layer counts
    # (``group_table_layer_offsets``). Sizing from ``layout.num_layer`` instead
    # would give one group's worth of rows for all of them: two groups whose
    # local layer ids both start at 0 would write each other's lengths, and the
    # restore would decompress at the wrong length with no error. Offsets cover
    # every group, not just the compressed ones, so the table stays valid when
    # FLEXKV_NVCOMP_GROUPS changes without reallocating.
    layer_groups = getattr(model_config, "layer_groups", None)
    if layer_groups:
        num_layers = sum(int(g.num_layers) for g in layer_groups)
    else:
        num_layers = layout.num_layer
    # Read the region count off the layout rather than recomputing it from
    # the legacy flag, so the table always matches what the kernels stride over.
    kv_dim = layout.kv_dim
    tp_size = model_config.effective_tp_size_per_node
    canonical = (tp_size == 1 or layout.num_kv_heads == 1)

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


def ssd_packed_threads(kv_shared: bool) -> Tuple[int, int]:
    """(write_threads, read_threads) for the nvcomp SSD packed path.

    Per-direction env (FLEXKV_NVCOMP_SSD_PACKED_WRITE_THREADS / _READ_THREADS)
    overrides the default. KV shared across ranks reads/writes the replicated
    canonical chunk, so it benefits from more threads than head-sharded MHA.
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
    kv_dim: int,
    log_prefix: str,
) -> Any:
    if chunk_size_bytes < RATIO_PLATEAU_BYTES:
        flexkv_logger.warning(
            f"{log_prefix} per-device chunk_size={chunk_size_bytes}B "
            f"< {RATIO_PLATEAU_BYTES // 1024}KB plateau; "
            "compression ratio may be low. Consider increasing "
            "tokens_per_block.")

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

    # The calibration buffer is built K+V, so every "2" below is this kv_dim.
    # It must also be what the kernel is told: passing a bool here used to
    # collapse to 0, and total_chunks = layers*kv_dim*blocks went to 0 too --
    # the calibration then timed a kernel that transferred nothing and picked
    # a batch size from that.
    CALIBRATION_KV_DIM = 2
    shape_per_layer = (CALIBRATION_KV_DIM, num_blocks, tpb, nh, hs)
    gpu_cache = torch.randn(
        (num_layers,) + shape_per_layer,
        dtype=torch.bfloat16,
        device=f"cuda:{dev}")
    if dtype != torch.bfloat16:
        gpu_cache = gpu_cache.to(dtype)
    gpu_blocks = [gpu_cache[i] for i in range(num_layers)]
    cpu_shape = (num_blocks, num_layers, CALIBRATION_KV_DIM, tpb, nh, hs)
    cpu_data = torch.zeros(cpu_shape, dtype=dtype, device="cpu").pin_memory()
    gpu_ptrs = torch.tensor(
        [block.data_ptr() for block in gpu_blocks],
        dtype=torch.int64).pin_memory()

    chunk_size = chunk_size_bytes
    gpu_kv_stride = num_blocks * tpb * nh * hs * elem
    gpu_block_stride = chunk_size
    gpu_layer_stride = CALIBRATION_KV_DIM * gpu_kv_stride
    cpu_kv_stride = chunk_size
    cpu_layer_stride = CALIBRATION_KV_DIM * chunk_size
    cpu_block_stride = num_layers * CALIBRATION_KV_DIM * chunk_size

    gpu_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()
    cpu_ids = torch.arange(num_blocks, dtype=torch.int64).pin_memory()
    stream = torch.cuda.Stream(device=dev)

    size_table = torch.zeros(
        (num_blocks, num_layers, CALIBRATION_KV_DIM),
        dtype=torch.uint32).pin_memory()
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
                    num_layers, CALIBRATION_KV_DIM, 0, size_table_ptr,
                    size_table_block_stride, size_table_layer_stride)
            stream.synchronize()

        start_evt.record(stream)
        for _ in range(calibration_iters):
            with torch.cuda.stream(stream):
                transfer_kv_blocks_ans_comp(
                    ctx, gpu_ids, gpu_ptrs, gpu_kv_stride, gpu_block_stride,
                    gpu_layer_stride, cpu_ids, cpu_data, cpu_kv_stride,
                    cpu_layer_stride, cpu_block_stride, chunk_size, 0,
                    num_layers, CALIBRATION_KV_DIM, 0, size_table_ptr,
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
