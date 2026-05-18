"""RocmBackend: AMD ROCm / HIP backend.

Strategy notes
--------------
1. ROCm PyTorch reuses ``torch.cuda.*`` and the ``"cuda"`` device type.
2. The HIP IPC handle ABI is byte-compatible with CUDA's 64-byte handle.
3. ROCm has no GDS counterpart; ``supports_direct_storage`` returns ``False``
   and callers fall back to the POSIX/io_uring path in
   ``csrc/transfer_ssd.cpp`` (already GPU-agnostic).
"""
from __future__ import annotations

import ctypes
from contextlib import contextmanager
from typing import Any, Iterator, Optional, Tuple

import torch

from ..interface import GpuBackend, GpuVendor


HIP_IPC_HANDLE_SIZE = 64
_HIP_SUCCESS = 0
_HIP_HOST_REGISTER_PORTABLE = 1
_HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS = 1


class _HipIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_byte * HIP_IPC_HANDLE_SIZE)]


def _load_hipart() -> ctypes.CDLL:
    last_err: Optional[Exception] = None
    for name in (
        "libamdhip64.so",
        "libamdhip64.so.5",
        "libamdhip64.so.6",
    ):
        try:
            return ctypes.CDLL(name)
        except OSError as e:
            last_err = e
            continue
    raise RuntimeError(
        f"Failed to load libamdhip64.so: {last_err}. "
        "Is ROCm installed and on LD_LIBRARY_PATH?"
    )


class RocmBackend(GpuBackend):
    vendor = GpuVendor.ROCM

    _hipart: Optional[ctypes.CDLL] = None

    # ============================================================
    # 1. Meta
    # ============================================================
    @classmethod
    def is_available(cls) -> bool:
        try:
            return (
                torch.cuda.is_available()
                and getattr(torch.version, "hip", None) is not None
            )
        except Exception:
            return False

    @classmethod
    def device_name(cls) -> str:
        return "AMD ROCm/HIP"

    @classmethod
    def torch_device_type(cls) -> str:
        # ROCm PyTorch keeps "cuda" as the device type.
        return "cuda"

    @classmethod
    def visible_devices_env_vars(cls) -> Tuple[str, ...]:
        # HIP_VISIBLE_DEVICES is the canonical one. ROCR_VISIBLE_DEVICES is
        # the lower-level ROCr runtime mask. CUDA_VISIBLE_DEVICES is honored
        # by ROCm PyTorch for backwards compatibility, so we strip it too.
        return (
            "HIP_VISIBLE_DEVICES",
            "ROCR_VISIBLE_DEVICES",
            "CUDA_VISIBLE_DEVICES",
        )

    @property
    def hipart(self) -> ctypes.CDLL:
        if RocmBackend._hipart is None:
            RocmBackend._hipart = _load_hipart()
        return RocmBackend._hipart

    # ============================================================
    # 2. Device management (transparently routes via torch.cuda which
    #    on ROCm PyTorch is wired to HIP)
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
        """Return ``gfx*`` strings for visible AMD GPUs (e.g. ['gfx90a','gfx942']).

        Used by ``build_backends/rocm_builder.py`` to populate
        ``PYTORCH_ROCM_ARCH``. PyTorch ROCm exposes ``device.gcnArchName``
        on each device; we strip any ``:xnack`` suffixes for compatibility
        with hipcc's expected format.
        """
        try:
            if not torch.cuda.is_available():
                return []
            out = set()
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                name = getattr(props, "gcnArchName", None)
                if name:
                    out.add(name.split(":")[0])
            return sorted(out)
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
        del stream

    def get_current_stream(self, device_id: Optional[int] = None) -> Any:
        return torch.cuda.current_stream(device_id)

    def stream_handle(self, stream: Any) -> int:
        # On ROCm PyTorch the same attribute returns hipStream_t (uintptr_t).
        return int(stream.cuda_stream)

    @contextmanager
    def stream_context(self, stream: Any) -> Iterator[None]:
        with torch.cuda.stream(stream):
            yield

    # ============================================================
    # 4. Pinned host memory
    # ============================================================
    def register_host_tensor(self, tensor: torch.Tensor) -> None:
        ptr = tensor.data_ptr()
        size = tensor.numel() * tensor.element_size()
        ret = self.hipart.hipHostRegister(
            ctypes.c_void_p(ptr),
            ctypes.c_size_t(size),
            _HIP_HOST_REGISTER_PORTABLE,
        )
        if ret != _HIP_SUCCESS:
            raise RuntimeError(f"hipHostRegister failed with error code {ret}")

    def unregister_host_tensor(self, tensor: torch.Tensor) -> None:
        ptr = tensor.data_ptr()
        ret = self.hipart.hipHostUnregister(ctypes.c_void_p(ptr))
        if ret != _HIP_SUCCESS:
            pass  # best-effort, mirrors the original CUDA path

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
        fn = getattr(c_ext, "layout_transform", None)
        if fn is None:
            raise NotImplementedError(
                "layout_transform not exposed by flexkv.c_ext on ROCm build"
            )
        return fn(*args, **kwargs)

    # ============================================================
    # 6. IPC (HIP IPC handle is byte-compatible with CUDA's)
    # ============================================================
    def supports_ipc(self) -> bool:
        return True

    def get_ipc_handle(self, tensor: torch.Tensor) -> bytes:
        if not tensor.is_cuda:
            raise ValueError("get_ipc_handle requires a GPU tensor")

        torch.cuda.set_device(tensor.device)
        handle = _HipIpcMemHandle()
        ret = self.hipart.hipIpcGetMemHandle(
            ctypes.byref(handle), ctypes.c_void_p(tensor.data_ptr())
        )
        if ret != _HIP_SUCCESS:
            raise RuntimeError(
                f"hipIpcGetMemHandle failed with error code {ret} "
                f"on device {tensor.device}, ptr=0x{tensor.data_ptr():x}"
            )
        return ctypes.string_at(ctypes.byref(handle), HIP_IPC_HANDLE_SIZE)

    def open_ipc_handle(
        self,
        ipc_handle: bytes,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device_id: int,
        offset: int = 0,
    ) -> torch.Tensor:
        if not torch.cuda.is_initialized():
            torch.cuda.init()
        torch.cuda.set_device(device_id)
        _ = torch.zeros(1, device=f"cuda:{device_id}")

        if len(ipc_handle) != HIP_IPC_HANDLE_SIZE:
            raise ValueError(
                f"Invalid HIP IPC handle size: {len(ipc_handle)} "
                f"(expected {HIP_IPC_HANDLE_SIZE})"
            )
        handle = _HipIpcMemHandle()
        ctypes.memmove(ctypes.byref(handle), ipc_handle, HIP_IPC_HANDLE_SIZE)

        base_ptr = ctypes.c_void_p()
        ret = self.hipart.hipIpcOpenMemHandle(
            ctypes.byref(base_ptr),
            handle,
            ctypes.c_int(_HIP_IPC_MEM_LAZY_ENABLE_PEER_ACCESS),
        )
        if ret != _HIP_SUCCESS:
            raise RuntimeError(
                f"hipIpcOpenMemHandle failed with error code {ret} "
                f"on device {device_id}"
            )
        data_ptr = (base_ptr.value or 0) + int(offset)

        from flexkv.common.memory_handle import TensorSharedHandle
        return TensorSharedHandle._create_tensor_from_cuda_ptr(
            data_ptr, shape, dtype, torch.device(f"cuda:{device_id}")
        )

    def close_ipc_handle(self, ptr: int) -> None:
        self.hipart.hipIpcCloseMemHandle(ctypes.c_void_p(ptr))

    # ============================================================
    # 7. Direct storage (none on ROCm today)
    # ============================================================
    def supports_direct_storage(self) -> bool:
        return False
