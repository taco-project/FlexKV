"""
Build configuration helpers for FlexKV.
Used by setup.py and tests to get the list of C++ extensions to build.
"""
import os
import shutil


def _has_cuda_toolkit():
    """Return True if a CUDA toolkit (nvcc) is available on this system."""
    return (
        bool(os.environ.get("CUDA_HOME"))
        or bool(os.environ.get("CUDA_PATH"))
        or shutil.which("nvcc") is not None
    )


def get_cpp_extension_names():
    """
    Return list of C++ extension module names that would be built
    under the current environment (FLEXKV_USE_MUSA, etc.).
    """
    enable_musa = os.environ.get("FLEXKV_USE_MUSA", "0").strip() == "1"
    has_cuda = _has_cuda_toolkit()

    names = []
    if has_cuda or not enable_musa:
        names.append("flexkv.c_ext")
    if enable_musa:
        names.append("flexkv.c_ext_musa")
    return names
