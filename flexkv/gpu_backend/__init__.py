"""
flexkv.gpu_backend — multi-vendor GPU abstraction layer.

Exports:
    current_backend  GpuBackend instance (process-level singleton)
    GpuBackend       abstract base class

Backend selection priority:
    1. Environment variable FLEXKV_GPU_BACKEND (nvidia/cuda/musa/rocm/kunlun/generic)
    2. entry_points plugins (group: flexkv.gpu_backends)
    3. Built-in backends probed in priority order
    4. Default: NvidiaBackend
"""
import os
import importlib
import importlib.metadata

from .interface import GpuBackend

_ENV_MAP = {
    "nvidia": "flexkv.gpu_backend.nvidia.backend.NvidiaBackend",
    "cuda":   "flexkv.gpu_backend.nvidia.backend.NvidiaBackend",
    "kunlun": "flexkv.gpu_backend.nvidia.backend.NvidiaBackend",
    "musa":   "flexkv.gpu_backend.musa.backend.MusaBackend",
    "generic":"flexkv.gpu_backend.generic.backend.GenericBackend",
}

_BUILTINS = [
    ("flexkv.gpu_backend.nvidia.backend", "NvidiaBackend"),
    ("flexkv.gpu_backend.musa.backend",   "MusaBackend"),
]


def _load_cls(dotted_path: str):
    mod_path, cls_name = dotted_path.rsplit(".", 1)
    return getattr(importlib.import_module(mod_path), cls_name)


def _detect_backend() -> GpuBackend:
    forced = os.environ.get("FLEXKV_GPU_BACKEND", "").strip().lower()
    if forced and forced in _ENV_MAP:
        cls = _load_cls(_ENV_MAP[forced])
        return cls()

    try:
        for ep in importlib.metadata.entry_points(group="flexkv.gpu_backends"):
            try:
                cls = ep.load()
                if cls.is_available():
                    return cls()
            except Exception:
                pass
    except Exception:
        pass

    for mod_path, cls_name in _BUILTINS:
        try:
            cls = getattr(importlib.import_module(mod_path), cls_name)
            if cls.is_available():
                return cls()
        except ImportError:
            pass

    from .nvidia.backend import NvidiaBackend
    return NvidiaBackend()


current_backend: GpuBackend = _detect_backend()

__all__ = ["GpuBackend", "current_backend"]
