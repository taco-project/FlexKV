"""Smoke tests for the GPU backend dispatch layer.

These tests intentionally do *not* require a real GPU. They cover:
1. ``current_backend`` is resolvable on import
2. ``select_backend('generic')`` always works
3. ``GpuBackend`` ABC matches the documented contract
4. ``flexkv.build_config.get_cpp_extension_names`` returns expected
   names for each known FLEXKV_GPU_BACKEND value.
"""
from __future__ import annotations

import os
import importlib

import pytest


def test_interface_module_importable():
    from flexkv.gpu_backend.interface import GpuBackend, GpuVendor
    assert hasattr(GpuVendor, "NVIDIA")
    assert hasattr(GpuVendor, "ROCM")
    assert hasattr(GpuVendor, "MUSA")
    assert hasattr(GpuVendor, "GENERIC")

    expected_methods = {
        "set_device", "current_device", "device_count", "synchronize",
        "is_initialized", "init_runtime", "empty_cache",
        "is_gpu_tensor", "get_device_capability",
        "create_stream", "destroy_stream", "get_current_stream",
        "stream_handle", "stream_context",
        "register_host_tensor", "unregister_host_tensor",
        "transfer_kv_blocks",
    }
    for name in expected_methods:
        assert callable(getattr(GpuBackend, name, None)), \
            f"GpuBackend is missing method '{name}'"


def test_current_backend_resolved():
    import flexkv.gpu_backend as gb
    assert gb.current_backend is not None
    # vendor is one of the four enum values
    assert gb.current_backend.vendor in (
        gb.GpuVendor.NVIDIA,
        gb.GpuVendor.ROCM,
        gb.GpuVendor.MUSA,
        gb.GpuVendor.GENERIC,
    )
    # device_name returns a non-empty string
    assert isinstance(gb.current_backend.device_name(), str)
    assert gb.current_backend.device_name()


def test_select_generic_backend():
    from flexkv.gpu_backend import select_backend, GpuVendor
    backend = select_backend("generic")
    assert backend.vendor is GpuVendor.GENERIC
    assert backend.is_available()
    assert backend.device_count() == 1
    assert backend.is_gpu_tensor.__qualname__.endswith(".is_gpu_tensor")
    # Generic backend is a no-op on host registration
    backend.synchronize()
    backend.empty_cache()
    backend.init_runtime()
    assert backend.stream_handle(None) == 0


def test_detect_arch_list_returns_list():
    """detect_arch_list must always return a list (possibly empty)
    on every backend, including in environments without a GPU."""
    from flexkv.gpu_backend.nvidia.backend import NvidiaBackend
    from flexkv.gpu_backend.rocm.backend import RocmBackend
    from flexkv.gpu_backend.generic.backend import GenericBackend
    from flexkv.gpu_backend.musa.backend import MusaBackend

    for cls in (NvidiaBackend, RocmBackend, GenericBackend, MusaBackend):
        out = cls.detect_arch_list()
        assert isinstance(out, list), f"{cls.__name__}.detect_arch_list() must return list"


def test_select_unknown_alias_raises():
    from flexkv.gpu_backend import select_backend
    with pytest.raises(ValueError):
        select_backend("not-a-real-backend")


def test_make_device():
    from flexkv.gpu_backend import select_backend
    backend = select_backend("generic")
    dev = backend.make_device(0)
    assert dev.type == "cpu"


def test_build_config_known_backends(monkeypatch):
    from flexkv import build_config

    monkeypatch.setenv("FLEXKV_GPU_BACKEND", "nvidia")
    importlib.reload(build_config)
    assert build_config.get_cpp_extension_names() == ["flexkv.c_ext"]

    monkeypatch.setenv("FLEXKV_GPU_BACKEND", "rocm")
    importlib.reload(build_config)
    assert build_config.get_cpp_extension_names() == ["flexkv.c_ext"]

    monkeypatch.setenv("FLEXKV_GPU_BACKEND", "musa")
    importlib.reload(build_config)
    assert build_config.get_cpp_extension_names() == ["flexkv.c_ext_musa"]

    monkeypatch.setenv("FLEXKV_GPU_BACKEND", "generic")
    importlib.reload(build_config)
    assert build_config.get_cpp_extension_names() == []


def test_force_generic_via_env(monkeypatch):
    """Re-import ``flexkv.gpu_backend`` with FLEXKV_GPU_BACKEND=generic and
    confirm ``current_backend`` becomes the GenericBackend."""
    monkeypatch.setenv("FLEXKV_GPU_BACKEND", "generic")
    import flexkv.gpu_backend as gb
    importlib.reload(gb)
    try:
        assert gb.current_backend.vendor is gb.GpuVendor.GENERIC
    finally:
        # Reset to default detection for downstream tests in the same session.
        monkeypatch.delenv("FLEXKV_GPU_BACKEND", raising=False)
        importlib.reload(gb)


def test_builder_registry():
    from build_backends import load_builder

    nv = load_builder("nvidia")
    assert nv.name == "nvidia"
    assert nv.get_extension_name() == "flexkv.c_ext"
    assert nv.vendor_macro() == "-DFLEXKV_BACKEND_NVIDIA"

    rocm = load_builder("rocm")
    assert rocm.name == "rocm"
    assert rocm.vendor_macro() == "-DFLEXKV_BACKEND_ROCM"

    generic = load_builder("generic")
    assert generic.name == "generic"
    assert generic.get_extension_name() == ""
    assert generic.get_sources() == []
