"""NvidiaBackend: thin wrapper over ``torch.cuda`` and ``flexkv.c_ext``.

This module also owns the loading of ``libcudart.so`` (replacing the
duplicated ``ctypes.CDLL`` blocks that previously lived in
``flexkv/transfer/worker.py`` and ``flexkv/common/memory_handle.py``).
"""
from __future__ import annotations

import ctypes
import os
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Tuple

import torch

from ..interface import GpuBackend, GpuVendor


# CUDA IPC handle is 64 bytes on Linux.
CUDA_IPC_HANDLE_SIZE = 64
_CUDA_SUCCESS = 0
_CUDA_HOST_REGISTER_PORTABLE = 1
_CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1


class _CudaIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * CUDA_IPC_HANDLE_SIZE)]


def _load_cudart() -> ctypes.CDLL:
    """Try a few common SONAMEs of libcudart on Linux."""
    last_err: Optional[Exception] = None
    for name in ("libcudart.so", "libcudart.so.12", "libcudart.so.11"):
        try:
            return ctypes.CDLL(name)
        except OSError as e:  # pragma: no cover - environment dependent
            last_err = e
            continue
    raise RuntimeError(
        f"Failed to load libcudart.so (tried .so / .so.12 / .so.11): {last_err}"
    )


class NvidiaBackend(GpuBackend):
    vendor = GpuVendor.NVIDIA

    # Lazily initialized to avoid loading libcudart in subprocesses that do
    # not actually need it.
    _cudart: Optional[ctypes.CDLL] = None

    # ============================================================
    # 1. Meta
    # ============================================================
    @classmethod
    def is_available(cls) -> bool:
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            # ROCm PyTorch also reports cuda available; route those to RocmBackend.
            return getattr(torch.version, "hip", None) is None
        except Exception:
            return False

    @classmethod
    def device_name(cls) -> str:
        return "NVIDIA CUDA"

    @classmethod
    def torch_device_type(cls) -> str:
        return "cuda"

    @classmethod
    def visible_devices_env_vars(cls) -> Tuple[str, ...]:
        return ("CUDA_VISIBLE_DEVICES",)

    @property
    def cudart(self) -> ctypes.CDLL:
        if NvidiaBackend._cudart is None:
            NvidiaBackend._cudart = _load_cudart()
        return NvidiaBackend._cudart

    # ============================================================
    # 2. Device management
    # ============================================================
    def set_device(self, device_id: int) -> None:
        torch.cuda.set_device(device_id)

    def current_device(self) -> int:
        return torch.cuda.current_device()

    def device_count(self) -> int:
        return torch.cuda.device_count()

    def synchronize(self, stream: Any = None) -> None:
        if stream is None:
            torch.cuda.synchronize()
        else:
            stream.synchronize()

    def is_initialized(self) -> bool:
        return torch.cuda.is_initialized()

    def init_runtime(self) -> None:
        if not torch.cuda.is_initialized():
            torch.cuda.init()

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def is_gpu_tensor(self, tensor: torch.Tensor) -> bool:
        return tensor.is_cuda

    def get_device_capability(self, device_id: int = 0) -> Tuple[int, int]:
        return torch.cuda.get_device_capability(device_id)

    @classmethod
    def detect_arch_list(cls) -> list:
        """Return the SM list visible to torch.cuda, e.g. ['8.0', '9.0'].

        Used by ``build_backends/cuda_builder.py`` to populate
        ``TORCH_CUDA_ARCH_LIST`` when the user did not set it explicitly.
        """
        try:
            if not torch.cuda.is_available():
                return []
            return sorted({
                "{}.{}".format(*torch.cuda.get_device_capability(i))
                for i in range(torch.cuda.device_count())
            })
        except Exception:
            return []

    # ============================================================
    # 3. Stream management
    # ============================================================
    def create_stream(self, device_id: Optional[int] = None) -> Any:
        if device_id is None:
            return torch.cuda.Stream()
        return torch.cuda.Stream(device=device_id)

    def destroy_stream(self, stream: Any) -> None:
        # PyTorch streams are GC-managed; nothing to do here.
        del stream

    def get_current_stream(self, device_id: Optional[int] = None) -> Any:
        return torch.cuda.current_stream(device_id)

    def stream_handle(self, stream: Any) -> int:
        return int(stream.cuda_stream)

    @contextmanager
    def stream_context(self, stream: Any) -> Iterator[None]:
        with torch.cuda.stream(stream):
            yield

    # ============================================================
    # 4. Pinned host memory (replaces the cudaHostRegister wrappers
    # previously inlined in flexkv/transfer/worker.py)
    # ============================================================
    def register_host_tensor(self, tensor: torch.Tensor) -> None:
        ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        ret = self.cudart.cudaHostRegister(
            ctypes.c_void_p(ptr),
            ctypes.c_size_t(size),
            _CUDA_HOST_REGISTER_PORTABLE,
        )
        if ret != _CUDA_SUCCESS:
            raise RuntimeError(f"cudaHostRegister failed with error code {ret}")

    def unregister_host_tensor(self, tensor: torch.Tensor) -> None:
        ptr = tensor.data_ptr()
        ret = self.cudart.cudaHostUnregister(ctypes.c_void_p(ptr))
        if ret != _CUDA_SUCCESS:
            # Best-effort: the original code silently swallowed errors here.
            pass

    # ============================================================
    # 5. Hot path
    # ============================================================
    def transfer_kv_blocks(self, *args, **kwargs) -> Any:
        from flexkv import c_ext
        return c_ext.transfer_kv_blocks(*args, **kwargs)

    def transfer_kv_blocks_ssd(self, *args, **kwargs) -> Any:
        from flexkv import c_ext
        return c_ext.transfer_kv_blocks_ssd(*args, **kwargs)

    def layout_transform(self, *args, **kwargs) -> Any:
        from flexkv import c_ext
        # Optional symbol; kept lazy so non-GDS builds don't break import.
        fn = getattr(c_ext, "layout_transform", None)
        if fn is None:
            raise NotImplementedError(
                "layout_transform is not exposed by flexkv.c_ext "
                "(GDS not enabled at build time?)"
            )
        return fn(*args, **kwargs)

    # ============================================================
    # 6. IPC
    # ============================================================
    def supports_ipc(self) -> bool:
        return True

    def get_ipc_handle(self, tensor: torch.Tensor) -> bytes:
        if not tensor.is_cuda:
            raise ValueError("get_ipc_handle requires a CUDA tensor")

        torch.cuda.set_device(tensor.device)

        handle = _CudaIpcMemHandle()
        ret = self.cudart.cudaIpcGetMemHandle(
            ctypes.byref(handle), ctypes.c_void_p(tensor.data_ptr())
        )
        if ret != _CUDA_SUCCESS:
            raise RuntimeError(
                f"cudaIpcGetMemHandle failed with error code {ret} "
                f"on device {tensor.device}, ptr=0x{tensor.data_ptr():x}"
            )
        return ctypes.string_at(ctypes.byref(handle), CUDA_IPC_HANDLE_SIZE)

    def open_ipc_handle(
        self,
        ipc_handle: bytes,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device_id: int,
        offset: int = 0,
    ) -> torch.Tensor:
        # Make sure CUDA is initialized in the current process.
        if not torch.cuda.is_initialized():
            torch.cuda.init()
        torch.cuda.set_device(device_id)
        # Force context creation
        _ = torch.zeros(1, device=f"cuda:{device_id}")

        if len(ipc_handle) != CUDA_IPC_HANDLE_SIZE:
            raise ValueError(
                f"Invalid CUDA IPC handle size: {len(ipc_handle)} "
                f"(expected {CUDA_IPC_HANDLE_SIZE})"
            )
        handle = _CudaIpcMemHandle()
        ctypes.memmove(ctypes.byref(handle), ipc_handle, CUDA_IPC_HANDLE_SIZE)

        base_ptr = ctypes.c_void_p()
        ret = self.cudart.cudaIpcOpenMemHandle(
            ctypes.byref(base_ptr),
            handle,
            ctypes.c_int(_CUDA_IPC_MEM_LAZY_ENABLE_PEER_ACCESS),
        )
        if ret != _CUDA_SUCCESS:
            raise RuntimeError(
                f"cudaIpcOpenMemHandle failed with error code {ret} "
                f"on device {device_id}; handle={ipc_handle.hex()}"
            )
        data_ptr = (base_ptr.value or 0) + int(offset)

        # Build a zero-copy torch.Tensor from the device pointer.
        # We delegate the type-erased reconstruction to the helper that
        # already exists in flexkv.common.memory_handle to avoid duplication.
        from flexkv.common.memory_handle import TensorSharedHandle
        return TensorSharedHandle._create_tensor_from_cuda_ptr(
            data_ptr, shape, dtype, torch.device(f"cuda:{device_id}")
        )

    def close_ipc_handle(self, ptr: int) -> None:
        self.cudart.cudaIpcCloseMemHandle(ctypes.c_void_p(ptr))

    # ============================================================
    # 7. Direct storage (GDS)
    # ============================================================
    def supports_direct_storage(self) -> bool:
        try:
            from flexkv import c_ext
        except ImportError:
            return False
        return hasattr(c_ext, "GDSManager") or hasattr(c_ext, "TPGDSTransferThreadGroup")

    def gds_create_manager(self, *args, **kwargs) -> Any:
        from flexkv import c_ext
        if not hasattr(c_ext, "GDSManager"):
            raise NotImplementedError(
                "GDS not enabled at build time (FLEXKV_ENABLE_GDS=0)"
            )
        return c_ext.GDSManager(*args, **kwargs)
