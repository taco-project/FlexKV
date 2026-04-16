"""GenericBackend — pure PyTorch fallback with no vendor-specific GPU kernels."""
import ctypes
from typing import List, Optional
import torch

from ..interface import GpuBackend


class GenericBackend(GpuBackend):

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def device_name(cls) -> str:
        return "Generic (PyTorch fallback)"

    def set_device(self, device_id: int) -> None:
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)

    def current_device(self) -> int:
        return torch.cuda.current_device() if torch.cuda.is_available() else 0

    def device_count(self) -> int:
        return max(1, torch.cuda.device_count())

    def synchronize(self, stream=None) -> None:
        pass

    def make_device(self, index: Optional[int] = None) -> torch.device:
        if torch.cuda.is_available():
            return torch.device(f"cuda:{index}" if index is not None else "cuda")
        return torch.device("cpu")

    def create_stream(self):
        return None

    def destroy_stream(self, stream) -> None:
        pass

    def register_host_tensor(self, tensor: torch.Tensor) -> None:
        pass

    def unregister_host_tensor(self, tensor: torch.Tensor) -> None:
        pass

    def alloc_pinned(self, size: int):
        return ctypes.create_string_buffer(size)

    def free_pinned(self, ptr) -> None:
        pass

    def transfer_kv_blocks(self, src: List[torch.Tensor], dst: List[torch.Tensor],
                           block_ids: List[int], stream=None) -> None:
        for s, d in zip(src, dst):
            d.copy_(s)

    def layout_transform(self, src: torch.Tensor, dst: torch.Tensor,
                         stream=None) -> None:
        dst.copy_(src)

    def is_gpu_tensor(self, tensor: torch.Tensor) -> bool:
        return tensor.is_cuda
