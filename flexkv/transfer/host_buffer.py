from __future__ import annotations

import ctypes
import os
import time
import weakref
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import torch

from flexkv.common.debug import flexkv_logger
from flexkv.storage.allocator import (
    alloc_hugepage_tensor,
    free_hugepage_tensor,
)

_cudart = None
_cudart_load_error: Optional[OSError] = None


def _get_cudart():
    global _cudart
    global _cudart_load_error

    if _cudart is None and _cudart_load_error is None:
        try:
            _cudart = ctypes.CDLL("libcudart.so")
        except OSError as e:
            _cudart_load_error = e

    if _cudart is None:
        raise RuntimeError(f"libcudart.so is unavailable: {_cudart_load_error}")
    return _cudart


def cuda_host_registration_available() -> bool:
    try:
        _get_cudart()
    except RuntimeError:
        return False
    return True


# Portable + Mapped: required for custom D2H kernels that store into host pointers.
CUDA_HOST_REGISTER_PORTABLE = 0x01
CUDA_HOST_REGISTER_MAPPED = 0x02
CUDA_HOST_ALLOC_PORTABLE = 0x01
CUDA_HOST_ALLOC_MAPPED = 0x02


_FLEXKV_CUDAHOST_REGISTER_PORTABLE = 1  # cudaHostRegisterPortable
_FLEXKV_CUDAHOST_CHUNK_SAFETY_MARGIN_BYTES = 4 * 1024  # 4 KiB cushion for CUDA internal alignment


def _get_cudahost_chunk_bytes() -> int:
    chunk_gb = float(os.getenv("FLEXKV_CUDAHOST_CHUNK_SIZE_GB", "0"))
    if chunk_gb <= 0:
        return 0  # no chunking
    chunk = int(chunk_gb * (1024 ** 3)) - _FLEXKV_CUDAHOST_CHUNK_SAFETY_MARGIN_BYTES
    # Round down to 4 KiB page boundary
    return chunk & ~0xFFF


def cudaHostRegister(tensor: torch.Tensor) -> None:
    """Register a CPU tensor with CUDA, optionally in chunks of FLEXKV_CUDAHOST_CHUNK_SIZE_GB.

    Chunked registration keeps each cudaHostRegister call under the configured size,
    which avoids failures on very large buffers. When the chunk size is 0 (default)
    the whole tensor is registered in a single call.
    """
    cudart = _get_cudart()
    ptr_base = tensor.data_ptr()
    total_size = tensor.numel() * tensor.element_size()
    flags = CUDA_HOST_REGISTER_PORTABLE | CUDA_HOST_REGISTER_MAPPED
    chunk_size = _get_cudahost_chunk_bytes()

    if chunk_size == 0 or total_size <= chunk_size:
        ret = cudart.cudaHostRegister(
            ctypes.c_void_p(ptr_base), ctypes.c_size_t(total_size), ctypes.c_uint(flags)
        )
        if ret != 0:
            raise RuntimeError(f"cudaHostRegister failed with error code {ret}")
        return

    for offset in range(0, total_size, chunk_size):
        size = min(chunk_size, total_size - offset)
        ret = cudart.cudaHostRegister(
            ctypes.c_void_p(ptr_base + offset), ctypes.c_size_t(size), ctypes.c_uint(flags)
        )
        if ret != 0:
            # Roll back already-registered chunks so we don't leak registrations.
            for done in range(0, offset, chunk_size):
                cudart.cudaHostUnregister(ctypes.c_void_p(ptr_base + done))
            raise RuntimeError(
                f"cudaHostRegister failed at offset={offset} bytes with error code {ret}"
            )


def cudaHostUnregister(tensor: torch.Tensor) -> None:
    """Unregister a CPU tensor from CUDA, matching the chunked register layout."""
    cudart = _get_cudart()
    ptr_base = tensor.data_ptr()
    total_size = tensor.numel() * tensor.element_size()
    chunk_size = _get_cudahost_chunk_bytes()

    if chunk_size == 0 or total_size <= chunk_size:
        cudart.cudaHostUnregister(ctypes.c_void_p(ptr_base))
        return

    for offset in range(0, total_size, chunk_size):
        ret = cudart.cudaHostUnregister(ctypes.c_void_p(ptr_base + offset))
        if ret != 0:
            flexkv_logger.warning(
                f"cudaHostUnregister failed at offset={offset} bytes, "
                f"error code {ret} (ignored)"
            )

def safe_cuda_host_unregister(tensor: torch.Tensor, label: str = "") -> None:
    """Best-effort unregister; never raises (idempotent shutdown helper)."""
    suffix = f" ({label})" if label else ""
    try:
        size_gb = tensor.numel() * tensor.element_size() / (1024 ** 3)
        ptr = int(tensor.data_ptr())
        # Print before/after so test.log still sees progress even if logger
        # handlers are torn down mid-unpin.
        flexkv_logger.info(
            f"[host_buffer] cudaHostUnregister begin {suffix}: "
            f"ptr=0x{ptr:x} size={size_gb:.3f} GiB",
        )
        cudaHostUnregister(tensor)
        msg = (
            f"[host_buffer] cudaHostUnregister ok {suffix}: "
            f"ptr=0x{ptr:x} size={size_gb:.3f} GiB"
        )
        flexkv_logger.info(msg)
    except Exception as e:
        msg = f"[host_buffer] cudaHostUnregister failed{suffix}: {e}"
        flexkv_logger.warning(msg)



@dataclass
class HostBufferHandle:
    tensor: torch.Tensor
    is_hugepage: bool = False
    is_cuda_registered: bool = False

    def __post_init__(self) -> None:
        if self.is_cuda_registered and not self.is_hugepage:
            raise ValueError("CUDA-registered host buffer must be HugePage-backed")

    @classmethod
    def pinned(cls, tensor: torch.Tensor) -> HostBufferHandle:
        return cls(tensor=tensor)

    @classmethod
    def hugepage(cls, tensor: torch.Tensor) -> HostBufferHandle:
        return cls(tensor=tensor, is_hugepage=True, is_cuda_registered=True)

    def release(self) -> None:
        if not self.is_hugepage:
            return

        if self.is_cuda_registered:
            try:
                cudaHostUnregister(self.tensor)
            except Exception as e:
                flexkv_logger.warning(
                    f"[host_buffer] release hugepage host buffer: cuda unregister failed ({e})"
                )
            self.is_cuda_registered = False

        free_hugepage_tensor(self.tensor)
        flexkv_logger.info("[host_buffer] release hugepage host buffer")
        self.is_hugepage = False


def alloc_mapped_host_tensor(num_elements: int, dtype: torch.dtype) -> torch.Tensor:
    """cudaHostAlloc(PORTABLE|MAPPED) buffer writable from device kernels."""
    cudart = _get_cudart()
    num_bytes = num_elements * dtype.itemsize
    host_ptr = ctypes.c_void_p()
    flags = CUDA_HOST_ALLOC_PORTABLE | CUDA_HOST_ALLOC_MAPPED
    err = cudart.cudaHostAlloc(
        ctypes.byref(host_ptr),
        ctypes.c_size_t(num_bytes),
        ctypes.c_uint(flags),
    )
    if err != 0:
        raise RuntimeError(f"cudaHostAlloc(mapped) failed with error code {err}")

    buf_type = ctypes.c_uint8 * num_bytes
    raw = buf_type.from_address(host_ptr.value)
    np_arr = np.frombuffer(raw, dtype=np.uint8, count=num_bytes)
    tensor = (
        torch.frombuffer(np_arr, dtype=torch.uint8, count=num_bytes)
        .view(dtype)[:num_elements]
    )
    weakref.finalize(tensor, lambda p=host_ptr: cudart.cudaFreeHost(p))
    return tensor


def _allocate_pinned_cpu_tensor(num_elements: int, dtype: torch.dtype) -> HostBufferHandle:
    return HostBufferHandle.pinned(alloc_mapped_host_tensor(num_elements, dtype))


def _fallback_to_pinned(
    num_elements: int,
    dtype: torch.dtype,
    reason: Exception,
) -> HostBufferHandle:
    flexkv_logger.warning(
        f"[host_buffer] fallback to pinned host buffer ({reason})"
    )
    return _allocate_pinned_cpu_tensor(num_elements, dtype)


def allocate_host_buffer(
    num_elements: int,
    dtype: torch.dtype,
    use_hugepage: bool,
    hugepage_size_bytes: int,
) -> HostBufferHandle:
    if not use_hugepage:
        return _allocate_pinned_cpu_tensor(num_elements, dtype)

    flexkv_logger.info("[host_buffer] attempt hugepage host buffer")

    hugepage_buf = None
    try:
        hugepage_buf = alloc_hugepage_tensor(
            num_elements=num_elements,
            dtype=dtype,
            page_size_bytes=hugepage_size_bytes,
        )
        cudaHostRegister(hugepage_buf)
    except Exception as e:
        if hugepage_buf is not None:
            free_hugepage_tensor(hugepage_buf)
        return _fallback_to_pinned(num_elements, dtype, e)

    flexkv_logger.info(
        f"[host_buffer] hugepage host buffer ready: "
        f"{hugepage_buf.numel() * hugepage_buf.element_size() / (1024 ** 3):.3f} GB"
    )
    return HostBufferHandle.hugepage(hugepage_buf)
