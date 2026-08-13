import pytest
import torch

from flexkv.transfer.nixlutil import gpu_chunk_u8_view


def test_gpu_chunk_u8_view_aliases_noncontiguous_physical_storage():
    num_blocks = 3
    block_size = 4
    num_heads = 2
    content_size = 6
    physical = torch.arange(
        num_blocks * block_size * num_heads * content_size,
        dtype=torch.int16,
    ).view(num_blocks, block_size, num_heads, content_size)
    logical = physical.permute(0, 2, 1, 3)
    assert not logical.is_contiguous()

    block_stride_b = block_size * num_heads * content_size * logical.element_size()
    original_first_block = physical[0].clone()
    view = gpu_chunk_u8_view(
        gpu_blocks=[logical],
        gpu_block_type=0,
        num_layers=1,
        gpu_block_id=1,
        layer_id=0,
        kv_idx=0,
        gpu_kv_stride_b=block_stride_b,
        gpu_block_stride_b=block_stride_b,
        gpu_layer_stride_b=logical.numel() * logical.element_size(),
        chunk_size_b=block_stride_b,
        kv_dim=1,
    )

    assert view.untyped_storage().data_ptr() == logical.untyped_storage().data_ptr()
    assert view.data_ptr() == logical.data_ptr() + block_stride_b
    view.zero_()

    assert torch.equal(physical[0], original_first_block)
    assert torch.count_nonzero(physical[1]).item() == 0


def test_gpu_chunk_u8_view_rejects_storage_overrun():
    tensor = torch.zeros((1, 2, 3, 4), dtype=torch.float16)

    with pytest.raises(ValueError, match="out of bounds"):
        gpu_chunk_u8_view(
            gpu_blocks=[tensor],
            gpu_block_type=0,
            num_layers=1,
            gpu_block_id=1,
            layer_id=0,
            kv_idx=0,
            gpu_kv_stride_b=1,
            gpu_block_stride_b=tensor.numel() * tensor.element_size(),
            gpu_layer_stride_b=tensor.numel() * tensor.element_size(),
            chunk_size_b=1,
            kv_dim=1,
        )
