from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DeviceBackend:
    name: str
    is_rocm: bool

    @classmethod
    def detect(cls, torch_module: Any = None) -> "DeviceBackend":
        if torch_module is None:
            import torch as torch_module
        is_rocm = getattr(torch_module.version, "hip", None) is not None
        return cls(name="rocm" if is_rocm else "cuda", is_rocm=is_rocm)

    def device(self, device_id: int):
        import torch
        return torch.device(f"cuda:{device_id}")

    def set_device(self, device_id: int) -> None:
        import torch
        torch.cuda.set_device(device_id)

    def empty(self, shape, dtype, device_id: int, *, fill_zero: bool = False):
        import torch
        factory = torch.zeros if fill_zero else torch.empty
        return factory(shape, dtype=dtype, device=self.device(device_id))

    def synchronize(self, device_id: int | None = None) -> None:
        import torch
        torch.cuda.synchronize(device_id)

    def memory(self, device_id: int) -> dict[str, int]:
        import torch
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated(device_id)),
            "reserved_bytes": int(torch.cuda.memory_reserved(device_id)),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device_id)),
        }


def torch_dtype(name: str):
    import torch
    aliases = {
        "fp16": "float16", "bf16": "bfloat16", "fp32": "float32",
        "fp8": "float8_e4m3fn", "e4m3": "float8_e4m3fn",
    }
    attr = aliases.get(name.lower(), name.lower())
    if not hasattr(torch, attr):
        raise ValueError(f"PyTorch does not provide dtype {name!r}")
    return getattr(torch, attr)
