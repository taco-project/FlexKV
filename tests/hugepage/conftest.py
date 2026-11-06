from __future__ import annotations

from contextlib import suppress

import pytest
import torch

from flexkv.storage.allocator import alloc_hugepage_tensor
from flexkv.transfer import host_buffer


def alloc_hugepage_or_skip(
    num_elements: int,
    dtype: torch.dtype,
    page_size_bytes: int,
) -> torch.Tensor:
    try:
        return alloc_hugepage_tensor(
            num_elements=num_elements,
            dtype=dtype,
            page_size_bytes=page_size_bytes,
        )
    except Exception as e:
        pytest.skip(f"hugepage allocation failed: {e}")


def cuda_ops_or_skip() -> tuple:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return host_buffer.cudaHostRegister, host_buffer.cudaHostUnregister


def unregister_suppress(tensor: torch.Tensor) -> None:
    with suppress(Exception):
        host_buffer.cudaHostUnregister(tensor)
