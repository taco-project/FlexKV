"""Build-side GPU vendor dispatch used by ``setup.py``.

Each concrete builder produces a ``setuptools.Extension``-compatible
description for one GPU vendor. ``setup.py`` selects exactly one
builder based on ``FLEXKV_GPU_BACKEND``.
"""
from .base import GPUBuilder

__all__ = ["GPUBuilder", "load_builder"]


def load_builder(name: str) -> GPUBuilder:
    """Resolve a builder by alias name.

    Aliases mirror those in ``flexkv/gpu_backend/__init__.py``::

        nvidia / cuda / kunlun  -> NvidiaBuilder
        rocm   / hip / amd      -> RocmBuilder
        musa   / moore          -> MusaBuilder
        generic / cpu           -> GenericBuilder
    """
    import importlib

    key = name.lower().strip()
    table = {
        "nvidia":  ("build_backends.cuda_builder",   "NvidiaBuilder"),
        "cuda":    ("build_backends.cuda_builder",   "NvidiaBuilder"),
        "kunlun":  ("build_backends.cuda_builder",   "NvidiaBuilder"),
        "rocm":    ("build_backends.rocm_builder",   "RocmBuilder"),
        "hip":     ("build_backends.rocm_builder",   "RocmBuilder"),
        "amd":     ("build_backends.rocm_builder",   "RocmBuilder"),
        "musa":    ("build_backends.musa_builder",   "MusaBuilder"),
        "moore":   ("build_backends.musa_builder",   "MusaBuilder"),
        "generic": ("build_backends.generic_builder", "GenericBuilder"),
        "cpu":     ("build_backends.generic_builder", "GenericBuilder"),
    }
    if key not in table:
        raise ValueError(
            f"Unknown FLEXKV_GPU_BACKEND='{name}'. "
            f"Known values: {sorted(table.keys())}"
        )
    mod_path, cls = table[key]
    return getattr(importlib.import_module(mod_path), cls)()
