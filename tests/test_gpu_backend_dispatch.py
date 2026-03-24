"""
Tests for GPU backend selection (CUDA vs MUSA).
Ensures the correct extension module is used and CUDA path is unchanged when only CUDA is available.
"""
import os
import pytest


def test_get_gpu_backend_returns_cuda_or_musa():
    """get_gpu_backend() returns either 'cuda' or 'musa'."""
    from flexkv.common.gpu_backend import get_gpu_backend
    backend = get_gpu_backend()
    assert backend in ("cuda", "musa")


def test_cuda_available_uses_c_ext_by_default():
    """When CUDA is available and no override, get_transfer_kv_blocks_module returns c_ext."""
    import torch
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    env_orig = os.environ.get("FLEXKV_GPU_BACKEND")
    try:
        if "FLEXKV_GPU_BACKEND" in os.environ:
            del os.environ["FLEXKV_GPU_BACKEND"]
        from flexkv.common.gpu_backend import get_gpu_backend, get_transfer_kv_blocks_module
        backend = get_gpu_backend()
        mod = get_transfer_kv_blocks_module()
        assert backend == "cuda"
        assert mod.__name__ == "flexkv.c_ext"
        assert hasattr(mod, "transfer_kv_blocks")
    finally:
        if env_orig is not None:
            os.environ["FLEXKV_GPU_BACKEND"] = env_orig
        elif "FLEXKV_GPU_BACKEND" in os.environ:
            del os.environ["FLEXKV_GPU_BACKEND"]


def test_musa_override_selects_c_ext_musa_when_built():
    """When FLEXKV_GPU_BACKEND=musa and c_ext_musa is built, that module is used."""
    env_orig = os.environ.get("FLEXKV_GPU_BACKEND")
    try:
        os.environ["FLEXKV_GPU_BACKEND"] = "musa"
        from flexkv.common.gpu_backend import get_gpu_backend, get_transfer_kv_blocks_module
        backend = get_gpu_backend()
        assert backend == "musa"
        try:
            mod = get_transfer_kv_blocks_module()
            # Module may be c_ext_musa if built, or c_ext if fallback
            assert mod.__name__ in ("flexkv.c_ext_musa", "flexkv.c_ext")
            assert hasattr(mod, "transfer_kv_blocks")
        except ImportError:
            # Both extensions not built, skip assertion about module
            pytest.skip("Neither flexkv.c_ext_musa nor flexkv.c_ext is available")
    finally:
        if env_orig is not None:
            os.environ["FLEXKV_GPU_BACKEND"] = env_orig
        elif "FLEXKV_GPU_BACKEND" in os.environ:
            del os.environ["FLEXKV_GPU_BACKEND"]


def test_c_ext_musa_importable_when_built():
    """If flexkv.c_ext_musa is built, it can be imported and has transfer_kv_blocks."""
    try:
        import flexkv.c_ext_musa as musa_ext
        assert hasattr(musa_ext, "transfer_kv_blocks")
    except ImportError:
        pytest.skip("flexkv.c_ext_musa not built (FLEXKV_USE_MUSA=1 and build completed)")
