from __future__ import annotations

import ctypes
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.storage.allocator import (
    DEFAULT_HUGE_PAGE_SIZE,
    free_hugepage_tensor,
)
from tests.hugepage.conftest import (
    alloc_hugepage_or_skip,
    cuda_ops_or_skip,
    unregister_suppress,
)

PAGE = DEFAULT_HUGE_PAGE_SIZE
_NUM_LAYERS = 1
_NUM_BLOCKS = 4
_TOKENS_PER_BLOCK = 16
_NUM_HEADS = 8
_HEAD_SIZE = 128
_DTYPE = torch.bfloat16
_ELEM_SIZE = _DTYPE.itemsize

_CPU_LAYOUT = KVCacheLayout(
    type=KVCacheLayoutType.LAYERFIRST,
    num_layer=_NUM_LAYERS,
    num_block=_NUM_BLOCKS,
    tokens_per_block=_TOKENS_PER_BLOCK,
    num_head=_NUM_HEADS,
    head_size=_HEAD_SIZE,
    is_mla=False,
)

_CHUNK = _CPU_LAYOUT.get_chunk_size()
_BLOCK_STRIDE = _CPU_LAYOUT.get_block_stride()
_KV_STRIDE = _CPU_LAYOUT.get_kv_stride()
_LAYER_STRIDE = _CPU_LAYOUT.get_layer_stride()


def _ensure_c_ext():
    try:
        from flexkv.c_ext import SSDIOCTX, transfer_kv_blocks_ssd
    except ImportError:
        pytest.skip("c_ext not built or SSD support disabled")
    return SSDIOCTX, transfer_kv_blocks_ssd


def _ssd_layout_for(num_blocks_per_file: int) -> KVCacheLayout:
    return KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
        num_layer=_NUM_LAYERS,
        num_block=num_blocks_per_file,
        tokens_per_block=_TOKENS_PER_BLOCK,
        num_head=_NUM_HEADS,
        head_size=_HEAD_SIZE,
        is_mla=False,
    )


def _verify_ssd_read(
    buf: np.ndarray,
    pattern: np.ndarray,
    kv_stride_bytes: int,
    layer_stride_bytes: int,
    chunk_bytes: int,
    num_blocks: int,
    kv_dim: int,
) -> None:
    pattern_u8 = pattern.view(np.uint8)
    lid = 0
    for bid in range(num_blocks):
        for kv in range(kv_dim):
            buf_start = lid * layer_stride_bytes + kv * kv_stride_bytes + bid * chunk_bytes
            pat_start = buf_start
            actual = buf[buf_start:buf_start + chunk_bytes]
            expected = pattern_u8[pat_start:pat_start + chunk_bytes]
            assert np.array_equal(actual, expected), (
                f"block {bid} {'K' if kv == 0 else 'V'} mismatch "
                f"at offset={buf_start}, size={chunk_bytes}"
            )


def test_hugepage_ssd_to_gpu_roundtrip() -> None:
    SSDIOCTX, transfer_kv_blocks_ssd = _ensure_c_ext()
    cudaHostRegister, _ = cuda_ops_or_skip()

    num_blocks_per_file = _NUM_BLOCKS
    kv_dim = 2
    _ssd_layout_for(num_blocks_per_file)

    chunk_bytes = _CHUNK * _ELEM_SIZE
    block_stride_bytes = _BLOCK_STRIDE * _ELEM_SIZE
    kv_stride_bytes = _KV_STRIDE * _ELEM_SIZE
    layer_stride_bytes = _LAYER_STRIDE * _ELEM_SIZE
    cpu_chunk_bytes = chunk_bytes
    cpu_kv_stride_bytes = kv_stride_bytes
    cpu_layer_stride_bytes = layer_stride_bytes
    ssd_kv_stride_bytes = kv_stride_bytes
    ssd_layer_stride_bytes = layer_stride_bytes
    ssd_chunk_bytes = chunk_bytes
    ssd_block_stride_bytes = block_stride_bytes
    file_size = kv_dim * num_blocks_per_file * chunk_bytes

    pattern = np.arange(file_size // 2, dtype=np.int16)
    pattern_bytes = pattern.view(np.uint8)

    tmpdir = Path(tempfile.mkdtemp(prefix="flexkv_e2e_"))
    ssd_path = tmpdir / "ssd_0.bin"
    pattern_bytes.tofile(ssd_path)

    hugepage_tensor = alloc_hugepage_or_skip(
        _CPU_LAYOUT.get_total_elements(),
        _DTYPE,
        PAGE,
    )
    ptr = hugepage_tensor.data_ptr()
    needs_unpin = False

    try:
        cudaHostRegister(hugepage_tensor)
        needs_unpin = True

        ioctx = SSDIOCTX({0: [str(ssd_path)]}, 1, 0, 0)
        layer_ids = torch.arange(0, _NUM_LAYERS, dtype=torch.int32)
        ssd_block_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64)
        cpu_block_ids = torch.tensor([0, 1, 2, 3], dtype=torch.int64)

        transfer_kv_blocks_ssd(
            ioctx=ioctx,
            cpu_layer_id_list=layer_ids,
            cpu_tensor_ptr=hugepage_tensor.data_ptr(),
            ssd_block_ids=ssd_block_ids,
            cpu_block_ids=cpu_block_ids,
            cpu_layer_stride_in_bytes=cpu_layer_stride_bytes,
            cpu_kv_stride_in_bytes=cpu_kv_stride_bytes,
            ssd_layer_stride_in_bytes=ssd_layer_stride_bytes,
            ssd_kv_stride_in_bytes=ssd_kv_stride_bytes,
            chunk_size_in_bytes=ssd_chunk_bytes,
            block_stride_in_bytes=ssd_block_stride_bytes,
            is_read=True,
            num_blocks_per_file=num_blocks_per_file,
            round_robin=1,
            num_threads_per_device=1,
            is_mla=False,
        )

        buf_np = np.frombuffer(
            (ctypes.c_uint8 * (hugepage_tensor.numel() * _ELEM_SIZE)).from_address(ptr),
            dtype=np.uint8,
        )
        _verify_ssd_read(
            buf_np,
            pattern,
            cpu_kv_stride_bytes,
            cpu_layer_stride_bytes,
            cpu_chunk_bytes,
            num_blocks_per_file,
            kv_dim,
        )

        gpu_tensor = torch.empty_like(hugepage_tensor, device="cuda")
        gpu_tensor.copy_(hugepage_tensor, non_blocking=True)
        torch.cuda.synchronize()

        roundtrip = torch.empty_like(hugepage_tensor)
        roundtrip.copy_(gpu_tensor, non_blocking=True)
        torch.cuda.synchronize()

        assert torch.equal(
            hugepage_tensor.view(torch.int16),
            roundtrip.view(torch.int16),
        )
    finally:
        if needs_unpin:
            unregister_suppress(hugepage_tensor)
        free_hugepage_tensor(hugepage_tensor)
        shutil.rmtree(tmpdir)
