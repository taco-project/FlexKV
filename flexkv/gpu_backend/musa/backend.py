"""MusaBackend: Moore Threads GPU backend (skeleton).

This skeleton mirrors NvidiaBackend / RocmBackend but defers C-extension
calls to ``flexkv.c_ext_musa`` (compiled via ``mcc``). Most methods are
fully wired; the C-extension parts raise ``NotImplementedError`` until a
MUSA build is actually produced.
"""
from __future__ import annotations

import ctypes
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Tuple

import torch

from ..interface import GpuBackend, GpuVendor


MUSA_IPC_HANDLE_SIZE = 64
_MUSA_SUCCESS = 0
_MUSA_HOST_REGISTER_PORTABLE = 1
_MUSA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1


class _MusaIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * MUSA_IPC_HANDLE_SIZE)]


def _try_load_torch_musa() -> bool:
    try:
        import torch_musa  # noqa: F401
        return True
    except Exception:
        return False


def _load_musart() -> ctypes.CDLL:
    last_err: Optional[Exception] = None
    for name in ("libmusa.so", "libmusart.so", "libmusart.so.1"):
        try:
            return ctypes.CDLL(name)
        except OSError as e:
            last_err = e
            continue
    raise RuntimeError(f"Failed to load libmusa.so: {last_err}")


class MusaBackend(GpuBackend):
    vendor = GpuVendor.MUSA

    _musart: Optional[ctypes.CDLL] = None

    # ============================================================
    # 1. Meta
    # ============================================================
    @classmethod
    def is_available(cls) -> bool:
        if not _try_load_torch_musa():
            return False
        try:
            return torch.musa.is_available()  # type: ignore[attr-defined]
        except Exception:
            return False

    @classmethod
    def device_name(cls) -> str:
        return "Moore Threads MUSA"

    @classmethod
    def torch_device_type(cls) -> str:
        return "musa"

    @classmethod
    def visible_devices_env_vars(cls) -> Tuple[str, ...]:
        return ("MUSA_VISIBLE_DEVICES",)

    @property
    def musart(self) -> ctypes.CDLL:
        if MusaBackend._musart is None:
            MusaBackend._musart = _load_musart()
        return MusaBackend._musart

    # ============================================================
    # 2. Device management
    # ============================================================
    def _t_musa(self):
        return torch.musa  # type: ignore[attr-defined]

    def set_device(self, device_id: int) -> None:
        self._t_musa().set_device(device_id)

    def current_device(self) -> int:
        return self._t_musa().current_device()

    def device_count(self) -> int:
        return self._t_musa().device_count()

    def synchronize(self, stream: Any = None) -> None:
        if stream is None:
            self._t_musa().synchronize()
        else:
            stream.synchronize()

    def is_initialized(self) -> bool:
        try:
            return self._t_musa().is_initialized()
        except AttributeError:
            return True

    def init_runtime(self) -> None:
        try:
            self._t_musa().init()
        except AttributeError:
            pass

    def empty_cache(self) -> None:
        try:
            self._t_musa().empty_cache()
        except AttributeError:
            pass

    def is_gpu_tensor(self, tensor: torch.Tensor) -> bool:
        return getattr(tensor, "is_musa", False) or tensor.device.type == "musa"

    def get_device_capability(self, device_id: int = 0) -> Tuple[int, int]:
        try:
            return self._t_musa().get_device_capability(device_id)
        except AttributeError:
            return (0, 0)

    # ============================================================
    # 3. Stream management
    # ============================================================
    def create_stream(self, device_id: Optional[int] = None) -> Any:
        if device_id is None:
            return self._t_musa().Stream()
        return self._t_musa().Stream(device=device_id)

    def destroy_stream(self, stream: Any) -> None:
        del stream

    def get_current_stream(self, device_id: Optional[int] = None) -> Any:
        return self._t_musa().current_stream(device_id)

    def stream_handle(self, stream: Any) -> int:
        # torch_musa exposes the raw stream pointer here too.
        return int(getattr(stream, "musa_stream", getattr(stream, "cuda_stream", 0)))

    @contextmanager
    def stream_context(self, stream: Any) -> Iterator[None]:
        with self._t_musa().stream(stream):
            yield

    # ============================================================
    # 4. Pinned host memory
    # ============================================================
    def register_host_tensor(self, tensor: torch.Tensor) -> None:
        ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        ret = self.musart.musaHostRegister(
            ctypes.c_void_p(ptr),
            ctypes.c_size_t(size),
            _MUSA_HOST_REGISTER_PORTABLE,
        )
        if ret != _MUSA_SUCCESS:
            raise RuntimeError(f"musaHostRegister failed with error code {ret}")

    def unregister_host_tensor(self, tensor: torch.Tensor) -> None:
        ptr = tensor.data_ptr()
        self.musart.musaHostUnregister(ctypes.c_void_p(ptr))

    # ============================================================
    # 5. Hot path
    # ============================================================
    def transfer_kv_blocks(self, *args, **kwargs) -> Any:
        try:
            from flexkv import c_ext_musa  # type: ignore
        except ImportError as e:
            raise NotImplementedError(
                "flexkv.c_ext_musa is not built; install MUSA wheel."
            ) from e
        return c_ext_musa.transfer_kv_blocks(*args, **kwargs)

    def transfer_kv_blocks_ssd(self, *args, **kwargs) -> Any:
        # SSD path is GPU-agnostic; reuse the common C extension if present.
        from flexkv import c_ext
        return c_ext.transfer_kv_blocks_ssd(*args, **kwargs)

    def layout_transform(self, *args, **kwargs) -> Any:
        try:
            from flexkv import c_ext_musa  # type: ignore
        except ImportError as e:
            raise NotImplementedError("flexkv.c_ext_musa is not built") from e
        fn = getattr(c_ext_musa, "layout_transform", None)
        if fn is None:
            raise NotImplementedError(
                "layout_transform not exposed by flexkv.c_ext_musa"
            )
        return fn(*args, **kwargs)

    # ============================================================
    # 6. IPC
    # ============================================================
    def supports_ipc(self) -> bool:
        return True

    def get_ipc_handle(self, tensor: torch.Tensor) -> bytes:
        if not self.is_gpu_tensor(tensor):
            raise ValueError("get_ipc_handle requires a MUSA tensor")
        self.set_device(tensor.device.index or 0)
        handle = _MusaIpcMemHandle()
        ret = self.musart.musaIpcGetMemHandle(
            ctypes.byref(handle), ctypes.c_void_p(tensor.data_ptr())
        )
        if ret != _MUSA_SUCCESS:
            raise RuntimeError(f"musaIpcGetMemHandle failed: {ret}")
        return ctypes.string_at(ctypes.byref(handle), MUSA_IPC_HANDLE_SIZE)

    def open_ipc_handle(
        self,
        ipc_handle: bytes,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device_id: int,
        offset: int = 0,
    ) -> torch.Tensor:
        self.init_runtime()
        self.set_device(device_id)

        if len(ipc_handle) != MUSA_IPC_HANDLE_SIZE:
            raise ValueError("Invalid MUSA IPC handle size")
        handle = _MusaIpcMemHandle()
        ctypes.memmove(ctypes.byref(handle), ipc_handle, MUSA_IPC_HANDLE_SIZE)

        base_ptr = ctypes.c_void_p()
        ret = self.musart.musaIpcOpenMemHandle(
            ctypes.byref(base_ptr),
            handle,
            ctypes.c_int(_MUSA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS),
        )
        if ret != _MUSA_SUCCESS:
            raise RuntimeError(f"musaIpcOpenMemHandle failed: {ret}")
        data_ptr = (base_ptr.value or 0) + int(offset)

        from flexkv.common.memory_handle import TensorSharedHandle
        return TensorSharedHandle._create_tensor_from_cuda_ptr(
            data_ptr, shape, dtype, torch.device(f"musa:{device_id}")
        )

    def close_ipc_handle(self, ptr: int) -> None:
        self.musart.musaIpcCloseMemHandle(ctypes.c_void_p(ptr))

    # ============================================================
    # 7. Direct storage
    # ============================================================
    def supports_direct_storage(self) -> bool:
        try:
            from flexkv import c_ext_musa  # type: ignore
        except ImportError:
            return False
        return hasattr(c_ext_musa, "MUFileManager")
