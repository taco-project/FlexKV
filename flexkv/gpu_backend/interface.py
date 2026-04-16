"""
GpuBackend ABC — the core interface of the FlexKV multi-vendor GPU abstraction layer.
All application code depends only on this module; no vendor-specific APIs
(cuda*, musa*, acl*, etc.) should appear outside backend implementations.
"""
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import torch


class GpuBackend(ABC):

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True if the hardware/SDK for this backend is present."""
        ...

    @classmethod
    @abstractmethod
    def device_name(cls) -> str:
        """Human-readable device name."""
        ...

    @abstractmethod
    def set_device(self, device_id: int) -> None: ...

    @abstractmethod
    def current_device(self) -> int: ...

    @abstractmethod
    def device_count(self) -> int: ...

    @abstractmethod
    def synchronize(self, stream=None) -> None: ...

    def make_device(self, index: Optional[int] = None) -> torch.device:
        raise NotImplementedError

    @abstractmethod
    def alloc_pinned(self, size: int):
        """Allocate page-locked (pinned) host memory of the given byte size."""
        ...

    @abstractmethod
    def free_pinned(self, ptr) -> None: ...

    @abstractmethod
    def register_host_tensor(self, tensor: torch.Tensor) -> None:
        """Register an existing CPU tensor as page-locked memory."""
        ...

    @abstractmethod
    def unregister_host_tensor(self, tensor: torch.Tensor) -> None: ...

    @abstractmethod
    def create_stream(self): ...

    @abstractmethod
    def destroy_stream(self, stream) -> None: ...

    def stream_context(self, stream):
        raise NotImplementedError

    def get_current_stream(self):
        raise NotImplementedError

    @abstractmethod
    def transfer_kv_blocks(
        self,
        src: List[torch.Tensor],
        dst: List[torch.Tensor],
        block_ids: List[int],
        stream=None,
    ) -> None:
        """Transfer KV cache blocks — hot path, calls vendor C++ kernel directly."""
        ...

    @abstractmethod
    def layout_transform(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        stream=None,
    ) -> None:
        """Re-layout KV tensors for GDS (Direct Storage) path."""
        ...

    def get_ipc_handle(self, tensor: torch.Tensor) -> bytes:
        raise NotImplementedError(f"{self.__class__.__name__} does not support IPC handles")

    def open_ipc_handle(self, handle: bytes, shape: Tuple[int, ...],
                        dtype: torch.dtype, device_id: int) -> torch.Tensor:
        raise NotImplementedError

    def close_ipc_handle(self, ptr: int) -> None:
        raise NotImplementedError

    def supports_direct_storage(self) -> bool:
        return False

    def read_direct(self, fd: int, tensor: torch.Tensor, offset: int) -> int:
        raise NotImplementedError

    def write_direct(self, fd: int, tensor: torch.Tensor, offset: int) -> int:
        raise NotImplementedError

    def empty_cache(self) -> None:
        pass

    def is_initialized(self) -> bool:
        return False

    def init_runtime(self) -> None:
        pass

    def is_gpu_tensor(self, tensor: torch.Tensor) -> bool:
        raise NotImplementedError
