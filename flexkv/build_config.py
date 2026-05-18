"""Build-time config introspection for FlexKV.

Tells the runtime which compiled C extensions are expected to be present
based on the ``FLEXKV_GPU_BACKEND`` env var that ``setup.py`` saw at
build time. This avoids hard-coding ``try: import flexkv.c_ext`` blocks
all over the codebase.
"""
from __future__ import annotations

import os
from typing import List


def get_active_backend_name() -> str:
    """Return the GPU backend name selected at build time.

    Falls back to "nvidia" (the historical default) when unset.
    """
    return os.environ.get("FLEXKV_GPU_BACKEND", "nvidia").lower().strip() or "nvidia"


def get_cpp_extension_names() -> List[str]:
    """Return the C-extension module names compiled for this build."""
    backend = get_active_backend_name()
    return {
        "nvidia":  ["flexkv.c_ext"],
        "cuda":    ["flexkv.c_ext"],
        "kunlun":  ["flexkv.c_ext"],
        "rocm":    ["flexkv.c_ext"],
        "hip":     ["flexkv.c_ext"],
        "amd":     ["flexkv.c_ext"],
        "musa":    ["flexkv.c_ext_musa"],
        "moore":   ["flexkv.c_ext_musa"],
        "generic": [],
        "cpu":     [],
    }.get(backend, ["flexkv.c_ext"])


def has_native_extension() -> bool:
    """True if the current build is expected to ship a compiled C extension."""
    return bool(get_cpp_extension_names())
