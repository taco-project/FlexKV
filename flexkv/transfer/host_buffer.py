from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from flexkv.common.debug import flexkv_logger
from flexkv.storage.allocator import alloc_hugepage_tensor, free_hugepage_tensor


def cuda_host_registration_available() -> bool:
    try:
        torch.cuda.cudart()
        return True
    except Exception:
        return False


def cudaHostRegister(tensor: torch.Tensor) -> None:
    ptr = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()
    err = torch.cuda.cudart().cudaHostRegister(ptr, size, 1)
    if isinstance(err, tuple):
        err = err[0]
    if err != 0:
        raise RuntimeError(f"cudaHostRegister failed with error code {err}")


def cudaHostUnregister(tensor: torch.Tensor) -> None:
    ptr = tensor.data_ptr()
    err = torch.cuda.cudart().cudaHostUnregister(ptr)
    if isinstance(err, tuple):
        err = err[0]
    if err != 0:
        raise RuntimeError(f"cudaHostUnregister failed with error code {err}")


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


def _allocate_pinned_cpu_tensor(num_elements: int, dtype: torch.dtype) -> HostBufferHandle:
    return HostBufferHandle.pinned(
        torch.empty(
            num_elements,
            dtype=dtype,
            device="cpu",
            pin_memory=True,
        )
    )


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
