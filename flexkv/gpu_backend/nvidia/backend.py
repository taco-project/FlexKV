"""NvidiaBackend — NVIDIA CUDA GPU implementation."""

import ctypes
from typing import List, Optional, Tuple

import torch

from ..interface import GpuBackend


class NvidiaBackend(GpuBackend):
    """
    NVIDIA CUDA backend — production default.
    Hot paths call flexkv.c_ext directly with no extra Python-layer overhead.
    """

    try:
        import flexkv.c_ext as _ext  # type: ignore
        _HAS_CUDA_EXT = True
    except ImportError:
        _ext = None
        _HAS_CUDA_EXT = False

    @classmethod
    def is_available(cls) -> bool:
        return torch.cuda.is_available()

    @classmethod
    def device_name(cls) -> str:
        if torch.cuda.is_available():
            return torch.cuda.get_device_name(0)
        return "NVIDIA (unavailable)"

    def set_device(self, device_id: int) -> None:
        torch.cuda.set_device(device_id)

    def current_device(self) -> int:
        return torch.cuda.current_device()

    def device_count(self) -> int:
        return torch.cuda.device_count()

    def synchronize(self, stream=None) -> None:
        torch.cuda.synchronize()

    def make_device(self, index: Optional[int] = None) -> torch.device:
        return torch.device(f"cuda:{index}" if index is not None else "cuda")

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def is_initialized(self) -> bool:
        return torch.cuda.is_initialized()

    def init_runtime(self) -> None:
        torch.cuda.init()

    def is_gpu_tensor(self, tensor: torch.Tensor) -> bool:
        return tensor.is_cuda

    def get_current_stream(self):
        return torch.cuda.current_stream()

    def create_stream(self):
        return torch.cuda.Stream()

    def destroy_stream(self, stream) -> None:
        pass  # Python GC handles CUDA stream lifetime

    def stream_context(self, stream):
        return torch.cuda.stream(stream)

    @property
    def _cudart(self):
        if not hasattr(self, '_cudart_lib'):
            self._cudart_lib = ctypes.CDLL("libcudart.so")
        return self._cudart_lib

    def alloc_pinned(self, size: int):
        ptr = ctypes.create_string_buffer(size)
        self._cudart.cudaHostRegister(
            ctypes.c_void_p(ctypes.addressof(ptr)), ctypes.c_size_t(size), 1)
        return ptr

    def free_pinned(self, ptr) -> None:
        self._cudart.cudaHostUnregister(ctypes.c_void_p(ctypes.addressof(ptr)))

    def register_host_tensor(self, tensor: torch.Tensor) -> None:
        ret = self._cudart.cudaHostRegister(
            ctypes.c_void_p(tensor.data_ptr()),
            ctypes.c_size_t(tensor.numel() * tensor.element_size()), 1)
        if ret != 0:
            raise RuntimeError(f"cudaHostRegister failed with error code {ret}")

    def unregister_host_tensor(self, tensor: torch.Tensor) -> None:
        self._cudart.cudaHostUnregister(ctypes.c_void_p(tensor.data_ptr()))

    def transfer_kv_blocks(self, src: List[torch.Tensor], dst: List[torch.Tensor],
                           block_ids: List[int], stream=None) -> None:
        if self._HAS_CUDA_EXT:
            self._ext.transfer_kv_blocks(src, dst, block_ids,
                                         stream.cuda_stream if stream else 0)
        else:
            for s, d in zip(src, dst):
                d.copy_(s, non_blocking=True)

    def layout_transform(self, src: torch.Tensor, dst: torch.Tensor,
                         stream=None) -> None:
        if self._HAS_CUDA_EXT:
            self._ext.layout_transform(src, dst, stream.cuda_stream if stream else 0)
        else:
            dst.copy_(src)

    def get_ipc_handle(self, tensor: torch.Tensor) -> bytes:
        handle = (ctypes.c_byte * 64)()
        ret = self._cudart.cudaIpcGetMemHandle(
            handle, ctypes.c_void_p(tensor.data_ptr()))
        if ret != 0:
            raise RuntimeError(f"cudaIpcGetMemHandle failed with error code {ret}")
        return bytes(handle)

    def open_ipc_handle(self, handle: bytes, shape: Tuple[int, ...],
                        dtype: torch.dtype, device_id: int) -> torch.Tensor:
        handle_struct = (ctypes.c_byte * 64)(*handle)
        dev_ptr = ctypes.c_void_p()
        ret = self._cudart.cudaIpcOpenMemHandle(
            ctypes.byref(dev_ptr), handle_struct, ctypes.c_int(1))
        if ret != 0:
            raise RuntimeError(f"cudaIpcOpenMemHandle failed with error code {ret}")
        return torch.frombuffer(
            (ctypes.c_byte * 1).from_address(dev_ptr.value), dtype=dtype
        ).reshape(shape).to(f"cuda:{device_id}")

    def close_ipc_handle(self, ptr: int) -> None:
        self._cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr))

    def supports_direct_storage(self) -> bool:
        return self._HAS_CUDA_EXT and hasattr(self._ext, 'cufile_read')

    def read_direct(self, fd: int, tensor: torch.Tensor, offset: int) -> int:
        return self._ext.cufile_read(fd, tensor, offset)

    def write_direct(self, fd: int, tensor: torch.Tensor, offset: int) -> int:
        return self._ext.cufile_write(fd, tensor, offset)
