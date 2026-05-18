"""FlexKV GPU Backend abstraction layer.

This package centralizes all GPU vendor-specific code paths
(NVIDIA / ROCm / MUSA / Generic) behind a single ``GpuBackend`` ABC.
Upper-layer modules should only call ``current_backend.xxx()`` and
must avoid any literal ``cuda*`` / ``hip*`` / ``musa*`` symbol.

See ``docs/gpu_backend/README_zh.md`` for the full design.
"""
from __future__ import annotations

import importlib
import os
from typing import List, Tuple

from .interface import GpuBackend, GpuVendor

__all__ = [
    "GpuBackend",
    "GpuVendor",
    "current_backend",
    "select_backend",
    "available_backends",
]

# (module_path, class_name) tuples for auto-detection in priority order.
# NVIDIA is preferred when both NVIDIA and ROCm appear available
# (e.g. CUDA-compatible third-party hardware that masquerades as CUDA).
_BUILTIN_BACKENDS: List[Tuple[str, str]] = [
    ("flexkv.gpu_backend.nvidia.backend", "NvidiaBackend"),
    ("flexkv.gpu_backend.rocm.backend", "RocmBackend"),
    ("flexkv.gpu_backend.musa.backend", "MusaBackend"),
    ("flexkv.gpu_backend.generic.backend", "GenericBackend"),
]

# User-facing aliases for ``FLEXKV_GPU_BACKEND``.
_FORCE_MAP = {
    "nvidia":  ("flexkv.gpu_backend.nvidia.backend",  "NvidiaBackend"),
    "cuda":    ("flexkv.gpu_backend.nvidia.backend",  "NvidiaBackend"),
    "kunlun":  ("flexkv.gpu_backend.nvidia.backend",  "NvidiaBackend"),
    "rocm":    ("flexkv.gpu_backend.rocm.backend",    "RocmBackend"),
    "hip":     ("flexkv.gpu_backend.rocm.backend",    "RocmBackend"),
    "amd":     ("flexkv.gpu_backend.rocm.backend",    "RocmBackend"),
    "musa":    ("flexkv.gpu_backend.musa.backend",    "MusaBackend"),
    "moore":   ("flexkv.gpu_backend.musa.backend",    "MusaBackend"),
    "generic": ("flexkv.gpu_backend.generic.backend", "GenericBackend"),
    "cpu":     ("flexkv.gpu_backend.generic.backend", "GenericBackend"),
}


def _instantiate(module_path: str, cls_name: str) -> GpuBackend:
    mod = importlib.import_module(module_path)
    cls = getattr(mod, cls_name)
    return cls()


def select_backend(name: str) -> GpuBackend:
    """Explicitly construct a backend instance by alias name."""
    key = name.lower().strip()
    if key not in _FORCE_MAP:
        raise ValueError(
            f"Unknown gpu backend '{name}'. "
            f"Available aliases: {sorted(_FORCE_MAP.keys())}"
        )
    return _instantiate(*_FORCE_MAP[key])


def available_backends() -> List[GpuBackend]:
    """Return every backend whose ``is_available()`` returned True (best-effort)."""
    out: List[GpuBackend] = []
    for module_path, cls_name in _BUILTIN_BACKENDS:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, cls_name)
            if cls.is_available():
                out.append(cls())
        except Exception:
            continue
    return out


def _detect_backend() -> GpuBackend:
    # 1) Explicit override via env var
    forced = os.environ.get("FLEXKV_GPU_BACKEND", "").lower().strip()
    if forced:
        if forced not in _FORCE_MAP:
            raise ValueError(
                f"FLEXKV_GPU_BACKEND='{forced}' is not recognized. "
                f"Available aliases: {sorted(_FORCE_MAP.keys())}"
            )
        return _instantiate(*_FORCE_MAP[forced])

    # 2) third-party plugins via entry_points (group = ``flexkv.gpu_backends``)
    try:
        try:
            from importlib.metadata import entry_points
        except ImportError:  # pragma: no cover - py<3.8
            from importlib_metadata import entry_points  # type: ignore

        try:
            eps = entry_points(group="flexkv.gpu_backends")
        except TypeError:
            # py<3.10 returns a dict-like
            eps = entry_points().get("flexkv.gpu_backends", [])
        for ep in eps:
            try:
                cls = ep.load()
                if cls.is_available():
                    return cls()
            except Exception:
                continue
    except Exception:
        pass

    # 3) Built-in detection by priority
    for module_path, cls_name in _BUILTIN_BACKENDS:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, cls_name)
            if cls.is_available():
                return cls()
        except ImportError:
            continue
        except Exception:
            # A misbehaving backend should not break detection
            continue

    # 4) Last resort fallback
    from .generic.backend import GenericBackend
    return GenericBackend()


# Eagerly resolve once at import time. Deterministic across the process.
current_backend: GpuBackend = _detect_backend()
