"""GpuBackend abstract base class.

The methods listed here form the contract every concrete backend must
satisfy. They are organized in 7 groups (meta, devices, streams, pinned
host memory, hot path, IPC, direct storage). See the design doc for the
rationale of each group.
"""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple

import torch


class GpuVendor(str, Enum):
    """GPU hardware vendor enum.

    Note: this is *intentionally* different from ``flexkv.c_ext`` /
    ``csrc/gtensor_handler.cuh``'s ``BackendType``, which describes the
    LLM framework backend (vLLM / TRT-LLM / SGLang).
    """
    NVIDIA = "nvidia"
    ROCM = "rocm"
    MUSA = "musa"
    GENERIC = "generic"


class GpuBackend(ABC):
    """Abstract GPU backend.

    Implementations live under ``flexkv/gpu_backend/<vendor>/backend.py``.
    The hot path (``transfer_kv_blocks`` / ``layout_transform``) is expected
    to forward into a compiled C extension with negligible overhead.
    """

    # =================================================================
    # 1. Meta information
    # =================================================================
    vendor: GpuVendor

    @classmethod
    @abstractmethod
    def is_available(cls) -> bool:
        """Return True iff this backend can run in the current process."""

    @classmethod
    @abstractmethod
    def device_name(cls) -> str:
        """Human-readable backend / device family name (e.g. 'NVIDIA CUDA')."""

    @classmethod
    def torch_device_type(cls) -> str:
        """Return the ``torch.device`` type string ('cuda', 'musa', 'cpu', ...).

        For ROCm PyTorch this is still ``'cuda'`` (the ROCm port reuses the
        ``cuda`` device type).
        """
        return "cuda"

    # =================================================================
    # 2. Device management
    # =================================================================
    @abstractmethod
    def set_device(self, device_id: int) -> None: ...

    @abstractmethod
    def current_device(self) -> int: ...

    @abstractmethod
    def device_count(self) -> int: ...

    @abstractmethod
    def synchronize(self, stream: Any = None) -> None: ...

    @abstractmethod
    def is_initialized(self) -> bool: ...

    @abstractmethod
    def init_runtime(self) -> None: ...

    @abstractmethod
    def empty_cache(self) -> None: ...

    @abstractmethod
    def is_gpu_tensor(self, tensor: torch.Tensor) -> bool: ...

    @abstractmethod
    def get_device_capability(self, device_id: int = 0) -> Tuple[int, int]: ...

    def make_device(self, index: Optional[int] = None) -> torch.device:
        """Construct a ``torch.device`` for this backend's device family."""
        if index is None:
            return torch.device(self.torch_device_type())
        return torch.device(f"{self.torch_device_type()}:{index}")

    @classmethod
    def detect_arch_list(cls) -> list:
        """Return the list of architecture strings for *all* visible devices.

        Used by the build backends (``cuda_builder.py`` etc.) to populate the
        equivalent of ``TORCH_CUDA_ARCH_LIST`` / ``PYTORCH_ROCM_ARCH``.

        The default implementation returns an empty list; vendor backends
        override this to query the actual hardware via their respective
        runtime APIs.
        """
        return []

    # ----- Device visibility env (CUDA_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES /
    #       MUSA_VISIBLE_DEVICES / ...) ---------------------------------
    @classmethod
    def visible_devices_env_vars(cls) -> Tuple[str, ...]:
        """Vendor-specific env var names that mask GPU visibility.

        The first entry is the canonical / preferred one (used when *writing*
        a mask). All entries are considered when *reading* / *stripping*.

        Default returns an empty tuple, which makes the helpers below behave
        as no-ops; vendor backends override this:

        * NVIDIA -> ``("CUDA_VISIBLE_DEVICES",)``
        * ROCm   -> ``("HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                       "CUDA_VISIBLE_DEVICES")``  (HIP also reads the CUDA
          name for backwards compatibility)
        * MUSA   -> ``("MUSA_VISIBLE_DEVICES",)``
        """
        return ()

    def get_visible_device_map(self) -> Optional[List[int]]:
        """Return the logical->physical device mapping derived from the
        current visibility env vars, or ``None`` if no mask is set.

        For example, with ``CUDA_VISIBLE_DEVICES=4,5`` this returns
        ``[4, 5]``: logical device 0 is physical 4, logical 1 is physical 5.

        Vendor-agnostic: the env var name(s) are pulled from
        :py:meth:`visible_devices_env_vars`.
        """
        for name in self.visible_devices_env_vars():
            raw = os.environ.get(name)
            if raw:
                try:
                    return [int(x.strip()) for x in raw.split(",") if x.strip()]
                except ValueError:
                    # Mask contains UUIDs (e.g. "GPU-xxxx") -- can't map to
                    # integers; treat as if no mask were set.
                    return None
        return None

    def strip_visible_devices(self, env: Dict[str, str]) -> Dict[str, str]:
        """Remove all vendor visibility masks from ``env`` (in-place) and
        return it. Used by parent processes that need to spawn children
        which must see *all* physical GPUs (e.g. the FlexKV server / the
        cross-process TransferManager).
        """
        for name in self.visible_devices_env_vars():
            env.pop(name, None)
        return env

    # =================================================================
    # 3. Stream management
    # =================================================================
    @abstractmethod
    def create_stream(self, device_id: Optional[int] = None) -> Any: ...

    @abstractmethod
    def destroy_stream(self, stream: Any) -> None: ...

    @abstractmethod
    def get_current_stream(self, device_id: Optional[int] = None) -> Any: ...

    @abstractmethod
    def stream_handle(self, stream: Any) -> int:
        """Return the underlying stream pointer (uintptr_t) usable from C extensions."""

    @contextmanager
    def stream_context(self, stream: Any) -> Iterator[None]:
        """Default context manager: subclasses may override for performance."""
        try:
            yield
        finally:
            pass

    # =================================================================
    # 4. Pinned host memory
    # =================================================================
    @abstractmethod
    def register_host_tensor(self, tensor: torch.Tensor) -> None: ...

    @abstractmethod
    def unregister_host_tensor(self, tensor: torch.Tensor) -> None: ...

    def alloc_pinned(self, size_bytes: int) -> Any:
        """Allocate ``size_bytes`` of page-locked host memory. Optional."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement alloc_pinned()"
        )

    def free_pinned(self, ptr: Any) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement free_pinned()"
        )

    # =================================================================
    # 5. KV block transfer / layout transform (hot path)
    # =================================================================
    @abstractmethod
    def transfer_kv_blocks(self, *args, **kwargs) -> Any:
        """Forward to the compiled C extension. Must be a single-line wrapper."""

    def transfer_kv_blocks_ssd(self, *args, **kwargs) -> Any:
        """SSD-side transfer (POSIX/io_uring). Default forwards to ``flexkv.c_ext``
        because SSD code is GPU-agnostic and shared by NVIDIA / ROCm wheels."""
        from flexkv import c_ext
        return c_ext.transfer_kv_blocks_ssd(*args, **kwargs)

    def layout_transform(self, *args, **kwargs) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement layout_transform()"
        )

    # =================================================================
    # 6. Cross-process IPC (optional)
    # =================================================================
    def supports_ipc(self) -> bool:
        return False

    def get_ipc_handle(self, tensor: torch.Tensor) -> bytes:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support IPC"
        )

    def open_ipc_handle(
        self,
        ipc_handle: bytes,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
        device_id: int,
        offset: int = 0,
    ) -> torch.Tensor:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support IPC"
        )

    def close_ipc_handle(self, ptr: int) -> None:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support IPC"
        )

    # =================================================================
    # 7. Direct storage (GDS / muFile / equivalent)
    # =================================================================
    def supports_direct_storage(self) -> bool:
        return False

    def gds_create_manager(self, *args, **kwargs) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support direct storage"
        )

    # =================================================================
    # Misc
    # =================================================================
    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return f"<{self.__class__.__name__} vendor={self.vendor.value}>"
