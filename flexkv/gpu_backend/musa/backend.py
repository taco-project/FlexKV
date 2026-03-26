"""MusaBackend — MUSA GPU implementation (experimental)."""

import ctypes
from typing import List, Optional, Tuple

import torch

from ..interface import GpuBackend


class MusaBackend(GpuBackend):

    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch_musa  # noqa: F401
            return torch.musa.is_available()
        except (ImportError, AttributeError):
            return False

    @classmethod
    def device_name(cls) -> str:
        try:
            import torch_musa  # noqa: F401
            return torch.musa.get_device_name(0)
        except Exception:
            return "MUSA (unavailable)"

    def set_device(self, device_id: int) -> None:
        import torch_musa; torch.musa.set_device(device_id)  # noqa: E702

    def current_device(self) -> int:
        import torch_musa; return torch.musa.current_device()  # noqa: E702

    def device_count(self) -> int:
        import torch_musa; return torch.musa.device_count()  # noqa: E702

    def synchronize(self, stream=None) -> None:
        import torch_musa; torch.musa.synchronize()  # noqa: E702

    def make_device(self, index: Optional[int] = None) -> torch.device:
        return torch.device(f"musa:{index}" if index is not None else "musa")

    def empty_cache(self) -> None:
        import torch_musa; torch.musa.empty_cache()  # noqa: E702

    def is_initialized(self) -> bool:
        try:
            return getattr(torch.musa, "is_initialized", lambda: False)()
        except Exception:
            return False

    def init_runtime(self) -> None:
        import torch_musa; torch.musa.init()  # noqa: E702

    def is_gpu_tensor(self, tensor: torch.Tensor) -> bool:
        # tensor.is_cuda always returns False on MUSA devices
        return str(tensor.device).startswith("musa")

    def get_current_stream(self):
        import torch_musa; return torch.musa.current_stream()  # noqa: E702

    def create_stream(self):
        import torch_musa; return torch.musa.Stream()  # noqa: E702

    def destroy_stream(self, stream) -> None:
        pass  # Python GC handles stream lifetime

    def stream_context(self, stream):
        import torch_musa; return torch.musa.stream(stream)  # noqa: E702

    @property
    def _musart(self):
        if not hasattr(self, '_musart_lib'):
            self._musart_lib = ctypes.CDLL("libmusart.so")
        return self._musart_lib

    def register_host_tensor(self, tensor: torch.Tensor) -> None:
        ret = self._musart.musaHostRegister(
            ctypes.c_void_p(tensor.data_ptr()),
            ctypes.c_size_t(tensor.numel() * tensor.element_size()), 1)
        if ret != 0:
            raise RuntimeError(f"musaHostRegister failed with error code {ret}")

    def unregister_host_tensor(self, tensor: torch.Tensor) -> None:
        self._musart.musaHostUnregister(ctypes.c_void_p(tensor.data_ptr()))

    def alloc_pinned(self, size: int):
        buf = ctypes.create_string_buffer(size)
        self._musart.musaHostRegister(
            ctypes.c_void_p(ctypes.addressof(buf)), ctypes.c_size_t(size), 1)
        return buf

    def free_pinned(self, ptr) -> None:
        self._musart.musaHostUnregister(ctypes.c_void_p(ctypes.addressof(ptr)))

    def transfer_kv_blocks(self, src: List[torch.Tensor], dst: List[torch.Tensor],
                           block_ids: List[int], stream=None) -> None:
        try:
            import flexkv.c_ext as ext  # type: ignore  # unified module name
            ext.transfer_kv_blocks(src, dst, block_ids,
                                   stream.musa_stream if stream else 0)
        except (ImportError, AttributeError):
            for s, d in zip(src, dst):
                d.copy_(s, non_blocking=True)

    def layout_transform(self, src: torch.Tensor, dst: torch.Tensor,
                         stream=None) -> None:
        try:
            import flexkv.c_ext as ext  # type: ignore
            ext.layout_transform(src, dst, stream.musa_stream if stream else 0)
        except (ImportError, AttributeError):
            dst.copy_(src)

    def get_ipc_handle(self, tensor: torch.Tensor) -> bytes:
        handle = (ctypes.c_byte * 64)()
        ret = self._musart.musaIpcGetMemHandle(
            handle, ctypes.c_void_p(tensor.data_ptr()))
        if ret != 0:
            raise RuntimeError(f"musaIpcGetMemHandle failed with error code {ret}")
        return bytes(handle)

    def open_ipc_handle(self, handle: bytes, shape: Tuple[int, ...],
                        dtype: torch.dtype, device_id: int) -> torch.Tensor:
        handle_struct = (ctypes.c_byte * 64)(*handle)
        dev_ptr = ctypes.c_void_p()
        ret = self._musart.musaIpcOpenMemHandle(
            ctypes.byref(dev_ptr), handle_struct, ctypes.c_int(1))
        if ret != 0:
            raise RuntimeError(f"musaIpcOpenMemHandle failed with error code {ret}")
        return torch.frombuffer(
            (ctypes.c_byte * 1).from_address(dev_ptr.value), dtype=dtype
        ).reshape(shape).to(f"musa:{device_id}")

    def close_ipc_handle(self, ptr: int) -> None:
        self._musart.musaIpcCloseMemHandle(ctypes.c_void_p(ptr))

    def supports_direct_storage(self) -> bool:
        try:
            import flexkv.c_ext as ext  # type: ignore
            return hasattr(ext, 'mufile_read')
        except ImportError:
            return False

    def read_direct(self, fd: int, tensor: torch.Tensor, offset: int) -> int:
        import flexkv.c_ext as ext  # type: ignore
        return ext.mufile_read(fd, tensor, offset)

    def write_direct(self, fd: int, tensor: torch.Tensor, offset: int) -> int:
        import flexkv.c_ext as ext  # type: ignore
        return ext.mufile_write(fd, tensor, offset)
