"""GenericBackend: pure-PyTorch CPU fallback.

Used by:
- CI environments without any GPU
- Smoke tests for the abstraction layer itself
- Bring-up of new hardware before the C extension exists
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Iterator, Optional, Tuple

import torch

from ..interface import GpuBackend, GpuVendor


class GenericBackend(GpuBackend):
    vendor = GpuVendor.GENERIC

    @classmethod
    def is_available(cls) -> bool:
        return True

    @classmethod
    def device_name(cls) -> str:
        return "Generic CPU/PyTorch"

    @classmethod
    def torch_device_type(cls) -> str:
        return "cpu"

    # ============================================================
    # 2. Device management
    # ============================================================
    def set_device(self, device_id: int) -> None:  # pragma: no cover - no-op
        pass

    def current_device(self) -> int:
        return 0

    def device_count(self) -> int:
        return 1

    def synchronize(self, stream: Any = None) -> None:
        return

    def is_initialized(self) -> bool:
        return True

    def init_runtime(self) -> None:
        return

    def empty_cache(self) -> None:
        return

    def is_gpu_tensor(self, tensor: torch.Tensor) -> bool:
        return False

    def get_device_capability(self, device_id: int = 0) -> Tuple[int, int]:
        return (0, 0)

    # ============================================================
    # 3. Stream management (no-op streams)
    # ============================================================
    def create_stream(self, device_id: Optional[int] = None) -> Any:
        return None

    def destroy_stream(self, stream: Any) -> None:
        return

    def get_current_stream(self, device_id: Optional[int] = None) -> Any:
        return None

    def stream_handle(self, stream: Any) -> int:
        return 0

    @contextmanager
    def stream_context(self, stream: Any) -> Iterator[None]:
        yield

    # ============================================================
    # 4. Pinned host memory (no-op)
    # ============================================================
    def register_host_tensor(self, tensor: torch.Tensor) -> None:
        return

    def unregister_host_tensor(self, tensor: torch.Tensor) -> None:
        return

    # ============================================================
    # 5. Hot path: pure-PyTorch fallback
    # ============================================================
    def transfer_kv_blocks(self, *args, **kwargs) -> Any:
        raise NotImplementedError(
            "GenericBackend.transfer_kv_blocks is a stub. "
            "It is intended for tests and bring-up; production builds "
            "must use a real GPU backend (NVIDIA / ROCm / MUSA)."
        )

    def layout_transform(self, *args, **kwargs) -> Any:
        raise NotImplementedError("GenericBackend has no layout_transform")
