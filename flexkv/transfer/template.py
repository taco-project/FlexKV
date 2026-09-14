"""Compile model KV geometry into a backend-agnostic transfer template.

Three places in the transfer layer independently rebuild the same per-group
stride table (six before the two CPU<->GPU workers were merged, the two GDS
workers were merged, and the layerwise worker was folded into the per-layer
completion contract):

  worker.py  GPUCPUTransferWorker._init_tp_multi_group
  worker.py  CPUSSDDiskTransferWorker._init_multi_group_ssd
  worker.py  GDSTransferWorker._init_multi_group_gds

They differ in what they *do* with the numbers (build a dict, construct a
TPTransferThreadGroup, fill parallel C++ arrays) but the numbers themselves
are one formula.  Each copy also has to independently get the compression
divisor, the per-group dtype, the LAYERFIRST/BLOCKFIRST fork and the running
byte offset right; that is three chances to introduce a silent mismatch
between, say, what the D2H worker writes and what the SSD worker later reads
back from the same CPU block.

This module owns the formula once.  ``compile_host_regions`` returns the
host-side (CPU or SSD) geometry, ``compile_gpu_regions`` the device side, and
``compile_template`` pairs them into a ``TransferTemplate`` describing a whole
edge.  Nothing here touches CUDA, a tensor handle, or a worker: it is pure
arithmetic over ``LayerGroupSpec`` + ``KVCacheLayout``, so it is testable
without a GPU and reusable by any backend.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from flexkv.common.config import LayerGroupSpec, build_layer_member_map
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType


@dataclass(frozen=True)
class HostRegion:
    """Geometry of one layer group inside a CPU/SSD block.

    All strides are bytes.  ``base_offset`` is where this group's data starts
    within one block (BLOCKFIRST) or within the whole buffer (LAYERFIRST);
    groups are packed back to back in ``layer_groups`` order.
    """
    group_index: int
    num_layers: int
    chunk_bytes: int
    kv_stride: int
    layer_stride: int
    block_stride: int
    base_offset: int
    # Bytes this group occupies in total. base_offset of group i+1 is
    # base_offset of group i plus this. Kept explicit so callers that only
    # need "where does the next group start" do not re-derive it.
    span_bytes: int


@dataclass(frozen=True)
class GpuRegion:
    """Geometry of one layer group in one device's KV tensor.  Bytes."""
    group_index: int
    device_index: int
    num_layers: int
    chunk_bytes: int
    kv_stride: int
    block_stride: int
    layer_stride: int


@dataclass(frozen=True)
class TransferTemplate:
    """One compiled edge: host geometry, device geometry, layer milestones.

    ``layer_milestones[i]`` lists the ``(group_index, local_layer_id)`` members
    that must land before original layer ``i`` is complete.  An original layer
    with no cached state has an empty tuple, which a completion check must
    treat as already satisfied rather than as never-satisfied.
    """
    host_regions: Tuple[HostRegion, ...]
    gpu_regions: Tuple[GpuRegion, ...]
    layer_milestones: Tuple[Tuple[Tuple[int, int], ...], ...]
    # Total bytes one logical block occupies across every group. For
    # BLOCKFIRST this equals the layout's block stride; kept here so callers
    # can assert the two agree instead of trusting one of them.
    total_span_bytes: int

    def gpu_regions_for_device(self, device_index: int) -> Tuple[GpuRegion, ...]:
        return tuple(r for r in self.gpu_regions if r.device_index == device_index)


def group_dtype_size(group: LayerGroupSpec, default_dtype: torch.dtype) -> int:
    """Element size for a group, honouring ``dtype=None`` = inherit."""
    return (group.dtype or default_dtype).itemsize


def group_chunk_bytes(
    group: LayerGroupSpec,
    tokens_per_block: int,
    default_dtype: torch.dtype,
) -> int:
    """Bytes of one (layer, kv) slot for one block of this group.

    ``compress_ratio`` shrinks the token dimension: the GPU tensor the
    framework allocates is already the compressed shape, so the divisor
    applies on both sides of the transfer and never appears as a ratio
    between them.
    """
    tokens = tokens_per_block // group.compress_ratio
    return tokens * group.num_kv_heads * group.head_size * group_dtype_size(
        group, default_dtype)


def compile_host_regions(
    layer_groups: Sequence[LayerGroupSpec],
    host_layout: KVCacheLayout,
    kv_dim: int,
    default_dtype: torch.dtype,
    *,
    block_stride_bytes: Optional[int] = None,
) -> Tuple[HostRegion, ...]:
    """Per-group CPU/SSD geometry, in ``layer_groups`` order.

    ``block_stride_bytes`` overrides the layout's block stride; SSD paths pass
    the SSD layout's own stride while sharing the CPU layout's token count.
    Only used for BLOCKFIRST, where block stride is a whole-block constant.
    """
    tokens_per_block = host_layout.tokens_per_block
    layout_type = host_layout.type
    if layout_type == KVCacheLayoutType.BLOCKFIRST:
        # get_block_stride() already returns bytes_per_block for multi-group
        # BLOCKFIRST (see KVCacheLayout._compute_kv_shape), which is why it is
        # not multiplied by an element size here the way the strides below are.
        whole_block_bytes = (
            host_layout.get_block_stride()
            if block_stride_bytes is None else block_stride_bytes
        )
    else:
        whole_block_bytes = None
    num_blocks = host_layout.num_block

    regions: List[HostRegion] = []
    offset = 0
    for gi, g in enumerate(layer_groups):
        chunk = group_chunk_bytes(g, tokens_per_block, default_dtype)
        if layout_type == KVCacheLayoutType.BLOCKFIRST:
            # [num_block, bytes_per_block]; inside a block the groups sit side
            # by side, each holding layer0_k, layer0_v, layer1_k, ...
            kv_stride = chunk
            layer_stride = kv_dim * chunk
            block_stride = whole_block_bytes
        else:
            # [all_layers, kv_dim, num_block, tokens, heads, head_dim]
            kv_stride = num_blocks * chunk
            layer_stride = kv_dim * num_blocks * chunk
            block_stride = chunk
        span = g.num_layers * layer_stride
        regions.append(HostRegion(
            group_index=gi,
            num_layers=g.num_layers,
            chunk_bytes=chunk,
            kv_stride=kv_stride,
            layer_stride=layer_stride,
            block_stride=block_stride,
            base_offset=offset,
            span_bytes=span,
        ))
        offset += span
    return tuple(regions)


def gpu_strides_from_tensor(
    tensor: torch.Tensor,
    tokens_per_block: int,
    dtype_size: int,
    kv_dim: int,
) -> Optional[Tuple[int, int, int]]:
    """(kv_stride, block_stride, layer_stride) in bytes from a real tensor.

    Attention backends disagree on the dim order of the 5D KV tensor:
      flash_attn:        [2, num_blocks, block_size, num_kv_heads, head_size]
      triton/flashinfer: [num_blocks, 2, block_size, num_kv_heads, head_size]
    so the order is recovered from the sizes rather than assumed.  Returns
    ``None`` when it cannot be recovered unambiguously (kv_dim=1, wrong rank,
    or a coincidence of sizes); callers fall back to the declared layout.
    """
    if kv_dim == 1 or tensor.ndim != 5:
        return None

    dim_sizes = [tensor.shape[i] for i in range(3)]
    kv_dim_idx = None
    block_size_idx = None
    block_dim_idx = None

    for i in range(3):
        if dim_sizes[i] == 2 and kv_dim_idx is None:
            kv_dim_idx = i
    for i in range(3):
        if i != kv_dim_idx and dim_sizes[i] == tokens_per_block and block_size_idx is None:
            block_size_idx = i
    for i in range(3):
        if i != kv_dim_idx and i != block_size_idx:
            block_dim_idx = i
            break

    if kv_dim_idx is None or block_dim_idx is None:
        return None

    return (
        tensor.stride(kv_dim_idx) * dtype_size,
        tensor.stride(block_dim_idx) * dtype_size,
        tensor.numel() * dtype_size,
    )


def compile_gpu_regions(
    layer_groups: Sequence[LayerGroupSpec],
    gpu_layouts_per_group: Sequence[Sequence[KVCacheLayout]],
    tokens_per_block: int,
    kv_dim: int,
    default_dtype: torch.dtype,
    *,
    tensors_per_group_device: Optional[Sequence[Sequence[Optional[torch.Tensor]]]] = None,
) -> Tuple[GpuRegion, ...]:
    """Per-(group, device) device-side geometry.

    ``gpu_layouts_per_group[gi][d]`` is group ``gi``'s layout on device ``d``;
    non-TP callers pass a single-element device list.  When
    ``tensors_per_group_device`` supplies a real tensor its measured strides
    win over the declared layout, because the layout records the *logical*
    shape while the tensor records what the attention backend actually
    allocated.
    """
    regions: List[GpuRegion] = []
    for gi, g in enumerate(layer_groups):
        dtype_size = group_dtype_size(g, default_dtype)
        tokens = tokens_per_block // g.compress_ratio
        chunk = group_chunk_bytes(g, tokens_per_block, default_dtype)
        for d, layout in enumerate(gpu_layouts_per_group[gi]):
            measured = None
            if tensors_per_group_device is not None:
                t = tensors_per_group_device[gi][d]
                if t is not None:
                    measured = gpu_strides_from_tensor(t, tokens, dtype_size, kv_dim)
            if measured is not None:
                kv_s, blk_s, layer_s = measured
            else:
                kv_s = layout.get_kv_stride() * dtype_size
                blk_s = layout.get_block_stride() * dtype_size
                layer_s = layout.get_layer_stride() * dtype_size
            regions.append(GpuRegion(
                group_index=gi,
                device_index=d,
                num_layers=g.num_layers,
                chunk_bytes=chunk,
                kv_stride=kv_s,
                block_stride=blk_s,
                layer_stride=layer_s,
            ))
    return tuple(regions)


def compile_template(
    layer_groups: Sequence[LayerGroupSpec],
    host_layout: KVCacheLayout,
    kv_dim: int,
    default_dtype: torch.dtype,
    num_original_layers: int,
    *,
    gpu_layouts_per_group: Optional[Sequence[Sequence[KVCacheLayout]]] = None,
    tensors_per_group_device: Optional[Sequence[Sequence[Optional[torch.Tensor]]]] = None,
    block_stride_bytes: Optional[int] = None,
) -> TransferTemplate:
    """Compile one edge's geometry.  Pure arithmetic; no CUDA, no handles."""
    host_regions = compile_host_regions(
        layer_groups, host_layout, kv_dim, default_dtype,
        block_stride_bytes=block_stride_bytes,
    )
    if gpu_layouts_per_group is None:
        gpu_regions: Tuple[GpuRegion, ...] = ()
    else:
        gpu_regions = compile_gpu_regions(
            layer_groups, gpu_layouts_per_group, host_layout.tokens_per_block,
            kv_dim, default_dtype,
            tensors_per_group_device=tensors_per_group_device,
        )
    member_map = build_layer_member_map(list(layer_groups), num_original_layers)
    return TransferTemplate(
        host_regions=host_regions,
        gpu_regions=gpu_regions,
        layer_milestones=member_map.members,
        total_span_bytes=sum(r.span_bytes for r in host_regions),
    )


def host_regions_as_group_params(
    regions: Sequence[HostRegion],
    key_prefix: str,
) -> List[Dict[str, int]]:
    """Adapter: HostRegion -> the ``{prefix}_kv_stride`` dicts workers use.

    Exists so a worker can adopt the compiler without also rewriting the dict
    keys its ``_transfer_impl`` already reads.
    """
    return [
        {
            f"{key_prefix}_kv_stride": r.kv_stride,
            f"{key_prefix}_layer_stride": r.layer_stride,
            f"{key_prefix}_block_stride": r.block_stride,
            f"{key_prefix}_offset_bytes": r.base_offset,
            "chunk_size": r.chunk_bytes,
            "num_layers": r.num_layers,
        }
        for r in regions
    ]
