"""The template compiler must reproduce all six hand-rolled stride builders.

``flexkv/transfer/template.py`` replaces six independent copies of the same
per-group stride arithmetic (see its module docstring for the list).  A
refactor that merely *looks* equivalent is not enough here: these numbers are
raw byte offsets handed to a C++ memcpy, so a one-group-off base_offset does
not raise, it silently reads a neighbouring group's bytes and the KV cache
returns plausible garbage.

So the tests below re-implement each original builder inline, verbatim from
the pre-refactor source, and assert the compiler agrees field by field.  When
one of the originals is deleted its inline copy here becomes the spec.
"""
import pytest
import torch

from flexkv.common.config import LayerGroupSpec
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.transfer.template import (
    compile_gpu_regions,
    compile_host_regions,
    gpu_strides_from_tensor,
)

pytestmark = pytest.mark.unit

TPB = 16


def _dsv4_groups():
    """Main KV (bf16, 8 heads) + indexer (uint8, 1 head, 4x compressed).

    This is the shape that motivated multi-group in the first place, and the
    only one that exercises all three of: per-group dtype, per-group head
    count, and compression.
    """
    return [
        LayerGroupSpec(num_layers=4, num_kv_heads=8, head_size=64,
                       layer_indices=[0, 1, 2, 3], dtype=torch.bfloat16),
        LayerGroupSpec(num_layers=4, num_kv_heads=1, head_size=32,
                       layer_indices=[0, 1, 2, 3], dtype=torch.uint8,
                       compress_ratio=4),
    ]


def _uniform_groups():
    return [
        LayerGroupSpec(num_layers=6, num_kv_heads=4, head_size=64,
                       layer_indices=list(range(6)), dtype=torch.bfloat16),
    ]


def _partitioned_groups():
    """1 full-attention layer + 3 linear-attention layers, alternating.

    Each original layer belongs to exactly one group, so layer_milestones has
    one member per layer rather than two -- the case where a completion check
    keyed on "every group reported" would deadlock.
    """
    return [
        LayerGroupSpec(num_layers=2, num_kv_heads=8, head_size=64,
                       layer_indices=[0, 4], dtype=torch.bfloat16),
        LayerGroupSpec(num_layers=6, num_kv_heads=2, head_size=64,
                       layer_indices=[1, 2, 3, 5, 6, 7], dtype=torch.bfloat16),
    ]


def _blockfirst_layout(groups, tp_size=1, num_block=64):
    return KVCacheLayout(
        type=KVCacheLayoutType.BLOCKFIRST,
        num_layer=sum(g.num_layers for g in groups),
        num_block=num_block,
        tokens_per_block=TPB,
        num_head=groups[0].num_kv_heads,
        head_size=groups[0].head_size,
        kv_dim=2,
        layer_groups=groups,
        tp_size=tp_size,
    )


# ---------------------------------------------------------------------------
# Reference implementations, transcribed from the pre-refactor sources.
# ---------------------------------------------------------------------------

def _ref_init_multi_group(groups, cpu_kv_layout, kv_dim, default_dtype):
    """worker.py GPUCPUTransferWorker._init_multi_group, CPU-stride part."""
    tpb = cpu_kv_layout.tokens_per_block
    cpu_layout_type = cpu_kv_layout.type
    num_cpu_blocks = cpu_kv_layout.num_block
    total_block_bytes = (
        cpu_kv_layout.get_block_stride()
        if cpu_layout_type == KVCacheLayoutType.BLOCKFIRST else None
    )
    out = []
    cpu_offset_bytes = 0
    for g in groups:
        dtype_size_g = (g.dtype or default_dtype).itemsize
        tpb_g = tpb // g.compress_ratio
        chunk_elements = tpb_g * g.num_kv_heads * g.head_size
        if cpu_layout_type == KVCacheLayoutType.BLOCKFIRST:
            cpu_layer_stride = kv_dim * chunk_elements * dtype_size_g
            cpu_block_stride = total_block_bytes
            cpu_kv_stride = chunk_elements * dtype_size_g
        else:
            cpu_layer_stride = kv_dim * num_cpu_blocks * chunk_elements * dtype_size_g
            cpu_block_stride = chunk_elements * dtype_size_g
            cpu_kv_stride = num_cpu_blocks * chunk_elements * dtype_size_g
        out.append(dict(
            cpu_layer_stride=cpu_layer_stride,
            cpu_block_stride=cpu_block_stride,
            cpu_kv_stride=cpu_kv_stride,
            cpu_offset_bytes=cpu_offset_bytes,
            num_layers=g.num_layers,
            chunk_size=chunk_elements * dtype_size_g,
        ))
        if cpu_layout_type == KVCacheLayoutType.BLOCKFIRST:
            cpu_offset_bytes += g.num_layers * kv_dim * chunk_elements * dtype_size_g
        else:
            cpu_offset_bytes += (
                g.num_layers * kv_dim * num_cpu_blocks * chunk_elements * dtype_size_g
            )
    return out


def _ref_compute_multi_group_tables(groups, cpu_kv_layout, kv_dim,
                                    default_dtype, tp_group_size):
    """The stride table the multi-group path used to compute inline.

    Kept as an independent reference implementation: ``compile_host_regions``
    has to agree with it byte for byte, and a shared helper would let both
    drift together.
    """
    tpb = cpu_kv_layout.tokens_per_block
    cpu_block_stride = cpu_kv_layout.get_block_stride()
    cpu_tp_stride = cpu_block_stride // tp_group_size
    group_cpu_offset_bytes = []
    group_cpu_layer_strides = []
    group_cpu_kv_strides = []
    group_chunk_sizes = []
    offset_bytes = 0
    for g in groups:
        dtype_size_g = (g.dtype or default_dtype).itemsize
        tpb_g = tpb // g.compress_ratio
        chunk_elements = tpb_g * g.num_kv_heads * g.head_size
        chunk_size_bytes = chunk_elements * dtype_size_g
        layer_stride_bytes = kv_dim * chunk_size_bytes
        group_cpu_offset_bytes.append(offset_bytes)
        group_cpu_layer_strides.append(layer_stride_bytes)
        group_cpu_kv_strides.append(chunk_size_bytes)
        group_chunk_sizes.append(chunk_size_bytes)
        offset_bytes += g.num_layers * layer_stride_bytes
    return dict(
        group_cpu_offset_bytes=group_cpu_offset_bytes,
        group_cpu_layer_strides=group_cpu_layer_strides,
        group_cpu_kv_strides=group_cpu_kv_strides,
        group_chunk_sizes=group_chunk_sizes,
        cpu_block_stride=cpu_block_stride,
        cpu_tp_stride=cpu_tp_stride,
    )


# ---------------------------------------------------------------------------
# Differential tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("groups_fn", [_uniform_groups, _dsv4_groups,
                                       _partitioned_groups])
def test_matches_init_multi_group_blockfirst(groups_fn):
    groups = groups_fn()
    layout = _blockfirst_layout(groups)
    ref = _ref_init_multi_group(groups, layout, 2, torch.bfloat16)
    got = compile_host_regions(groups, layout, 2, torch.bfloat16)

    assert len(got) == len(ref)
    for r, region in zip(ref, got):
        assert region.kv_stride == r["cpu_kv_stride"]
        assert region.layer_stride == r["cpu_layer_stride"]
        assert region.block_stride == r["cpu_block_stride"]
        assert region.base_offset == r["cpu_offset_bytes"]
        assert region.chunk_bytes == r["chunk_size"]
        assert region.num_layers == r["num_layers"]


@pytest.mark.parametrize("tp_group_size", [1, 2, 4])
@pytest.mark.parametrize("groups_fn", [_uniform_groups, _dsv4_groups])
def test_matches_layerwise_compute_multi_group_tables(groups_fn, tp_group_size):
    groups = groups_fn()
    layout = _blockfirst_layout(groups, tp_size=tp_group_size)
    ref = _ref_compute_multi_group_tables(groups, layout, 2, torch.bfloat16,
                                          tp_group_size)
    got = compile_host_regions(groups, layout, 2, torch.bfloat16)

    assert [r.base_offset for r in got] == ref["group_cpu_offset_bytes"]
    assert [r.layer_stride for r in got] == ref["group_cpu_layer_strides"]
    assert [r.kv_stride for r in got] == ref["group_cpu_kv_strides"]
    assert [r.chunk_bytes for r in got] == ref["group_chunk_sizes"]
    # cpu_tp_stride is a property of the layout, not of a group; the compiler
    # deliberately does not own it. Assert the block stride it *does* own is
    # the one that tp stride is derived from.
    assert got[0].block_stride == ref["cpu_block_stride"]
    assert ref["cpu_block_stride"] // tp_group_size == ref["cpu_tp_stride"]


# ---------------------------------------------------------------------------
# GPU-side geometry
# ---------------------------------------------------------------------------

def _gpu_layout(num_layer, num_head, head_size, num_block=64):
    return KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=num_layer,
        num_block=num_block,
        tokens_per_block=TPB,
        num_head=num_head,
        head_size=head_size,
        kv_dim=2,
    )


def test_gpu_regions_prefer_measured_strides_over_the_layout():
    """The layout records the logical shape; the tensor records what the
    attention backend actually allocated. When they disagree the tensor is
    right, because that is the memory the copy will touch."""
    groups = _uniform_groups()
    layouts = [[_gpu_layout(6, 4, 64)]]
    # flash_attn order: [2, num_blocks, block_size, heads, head_size]
    t = torch.empty(2, 8, TPB, 4, 64, dtype=torch.bfloat16)
    regions = compile_gpu_regions(
        groups, layouts, TPB, 2, torch.bfloat16,
        tensors_per_group_device=[[t]])
    expected = gpu_strides_from_tensor(t, TPB, 2, 2)
    assert expected is not None
    assert (regions[0].kv_stride, regions[0].block_stride,
            regions[0].layer_stride) == expected
    assert regions[0].kv_stride != layouts[0][0].get_kv_stride() * 2


