"""
GPU runtime abstraction for FlexKV — CUDA and MUSA.

Provides backend-agnostic wrappers for host memory registration, stream
acquisition, device management, IPC handles, and cache control so callers
do not need to know which GPU backend is active.

MUSA support uses the torch_musa project:
  https://github.com/MooreThreads/torch_musa
"""
import contextlib
import ctypes
from typing import Any, Generator, Optional, Tuple

import torch

from flexkv.common.gpu_backend import get_gpu_backend

_runtime_lib = None
_backend: str = ""


def _load_runtime():
    global _runtime_lib, _backend
    if _runtime_lib is not None:
        return
    _backend = get_gpu_backend()
    if _backend == "musa":
        _runtime_lib = ctypes.CDLL("libmusart.so")
    else:
        _runtime_lib = ctypes.CDLL("libcudart.so")


def _require_musa_module():
    """Return ``torch.musa`` or raise if the MUSA backend was selected but is unavailable."""
    musa_mod = getattr(torch, "musa", None)
    if musa_mod is None:
        raise RuntimeError(
            "GPU backend is 'musa' but torch.musa is not available. "
            "Install torch_musa (https://github.com/MooreThreads/torch_musa) "
            "or set FLEXKV_GPU_BACKEND to 'cuda'."
        )
    return musa_mod


# ---------------------------------------------------------------------------
# Host memory registration
# ---------------------------------------------------------------------------

def host_register(tensor: torch.Tensor) -> None:
    """Register a CPU tensor for pinned-memory access on the active backend."""
    _load_runtime()
    ptr = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()
    if _backend == "musa":
        ret = _runtime_lib.musaHostRegister(
            ctypes.c_void_p(ptr), ctypes.c_size_t(size), 1
        )
        if ret != 0:
            raise RuntimeError(f"musaHostRegister failed with error code {ret}")
    else:
        ret = _runtime_lib.cudaHostRegister(
            ctypes.c_void_p(ptr), ctypes.c_size_t(size), 1
        )
        if ret != 0:
            raise RuntimeError(f"cudaHostRegister failed with error code {ret}")


def host_unregister(tensor: torch.Tensor) -> None:
    """Unregister a previously registered CPU tensor."""
    _load_runtime()
    ptr = tensor.data_ptr()
    if _backend == "musa":
        _runtime_lib.musaHostUnregister(ctypes.c_void_p(ptr))
    else:
        _runtime_lib.cudaHostUnregister(ctypes.c_void_p(ptr))


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------

def set_device(device_id: int) -> None:
    """Set the active GPU device."""
    if get_gpu_backend() == "musa":
        _require_musa_module().set_device(device_id)
    else:
        torch.cuda.set_device(device_id)


def current_device() -> int:
    """Return the index of the currently selected GPU device."""
    if get_gpu_backend() == "musa":
        return _require_musa_module().current_device()
    return torch.cuda.current_device()


def device_count() -> int:
    """Return the number of available GPU devices."""
    if get_gpu_backend() == "musa":
        return _require_musa_module().device_count()
    return torch.cuda.device_count()


def get_device_string(device_id: int) -> str:
    """Return the device string for the given device id, e.g. ``'cuda:0'`` or ``'musa:0'``."""
    backend = get_gpu_backend()
    return f"{backend}:{device_id}"


def get_device(device_id: int) -> torch.device:
    """Return a ``torch.device`` for the given device id on the active backend."""
    return torch.device(get_device_string(device_id))


# ---------------------------------------------------------------------------
# Stream management
# ---------------------------------------------------------------------------

def get_current_stream() -> Any:
    """Return the current GPU stream for the active backend."""
    if get_gpu_backend() == "musa":
        return _require_musa_module().current_stream()
    return torch.cuda.current_stream()


def create_stream() -> Any:
    """Create a new GPU stream on the active backend."""
    if get_gpu_backend() == "musa":
        return _require_musa_module().Stream()
    return torch.cuda.Stream()


@contextlib.contextmanager
def stream_context(stream: Any) -> Generator[None, None, None]:
    """Context manager to execute on the given stream (works for both CUDA and MUSA)."""
    if get_gpu_backend() == "musa":
        with _require_musa_module().stream(stream):
            yield
    else:
        with torch.cuda.stream(stream):
            yield


# ---------------------------------------------------------------------------
# Synchronisation & cache
# ---------------------------------------------------------------------------

def synchronize(device_id: int | None = None) -> None:
    """Synchronize the GPU device."""
    if get_gpu_backend() == "musa":
        musa = _require_musa_module()
        if device_id is not None:
            musa.set_device(device_id)
        musa.synchronize()
    else:
        if device_id is not None:
            torch.cuda.set_device(device_id)
        torch.cuda.synchronize()


def empty_cache() -> None:
    """Release all unoccupied cached memory held by the GPU allocator."""
    if get_gpu_backend() == "musa":
        _require_musa_module().empty_cache()
    else:
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Initialisation helpers
# ---------------------------------------------------------------------------

def is_initialized() -> bool:
    """Return whether the GPU runtime has been initialised."""
    if get_gpu_backend() == "musa":
        return getattr(_require_musa_module(), "is_initialized", lambda: False)()
    return torch.cuda.is_initialized()


def init_runtime() -> None:
    """Explicitly initialise the GPU runtime context."""
    if get_gpu_backend() == "musa":
        _require_musa_module().init()
    else:
        torch.cuda.init()


# ---------------------------------------------------------------------------
# Tensor device helpers
# ---------------------------------------------------------------------------

def is_gpu_tensor(tensor: torch.Tensor) -> bool:
    """Return True if *tensor* lives on the active GPU backend (CUDA or MUSA)."""
    if tensor.is_cuda:
        return True
    if get_gpu_backend() == "musa":
        return str(tensor.device).startswith("musa")
    return False


# ---------------------------------------------------------------------------
# IPC handle helpers (ctypes)
# ---------------------------------------------------------------------------

IPC_HANDLE_SIZE = 64


class _IpcMemHandle(ctypes.Structure):
    """64-byte IPC memory handle — identical layout for both CUDA and MUSA."""
    _fields_ = [("reserved", ctypes.c_byte * IPC_HANDLE_SIZE)]


def _get_runtime_lib() -> ctypes.CDLL:
    """Return the loaded runtime library (loads lazily)."""
    _load_runtime()
    return _runtime_lib  # type: ignore[return-value]


def ipc_get_mem_handle(data_ptr: int) -> bytes:
    """Call cudaIpcGetMemHandle / musaIpcGetMemHandle and return 64-byte handle."""
    lib = _get_runtime_lib()
    handle = _IpcMemHandle()
    if get_gpu_backend() == "musa":
        result = lib.musaIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(data_ptr))
        if result != 0:
            raise RuntimeError(f"musaIpcGetMemHandle failed with error code {result}")
    else:
        result = lib.cudaIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(data_ptr))
        if result != 0:
            raise RuntimeError(f"cudaIpcGetMemHandle failed with error code {result}")
    return ctypes.string_at(ctypes.byref(handle), IPC_HANDLE_SIZE)


def ipc_open_mem_handle(handle_bytes: bytes) -> int:
    """Call cudaIpcOpenMemHandle / musaIpcOpenMemHandle and return the device pointer."""
    lib = _get_runtime_lib()
    handle = _IpcMemHandle()
    ctypes.memmove(ctypes.byref(handle), handle_bytes, IPC_HANDLE_SIZE)
    dev_ptr = ctypes.c_void_p()
    if get_gpu_backend() == "musa":
        result = lib.musaIpcOpenMemHandle(
            ctypes.byref(dev_ptr), handle, ctypes.c_int(1)
        )
        if result != 0:
            raise RuntimeError(f"musaIpcOpenMemHandle failed with error code {result}")
    else:
        result = lib.cudaIpcOpenMemHandle(
            ctypes.byref(dev_ptr), handle, ctypes.c_int(1)
        )
        if result != 0:
            raise RuntimeError(f"cudaIpcOpenMemHandle failed with error code {result}")
    return dev_ptr.value
