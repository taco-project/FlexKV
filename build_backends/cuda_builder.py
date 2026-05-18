"""NvidiaBuilder: produces ``flexkv.c_ext`` via ``nvcc``.

Faithfully preserves the existing ``setup.py`` flags / env knobs:
- ``FLEXKV_ENABLE_GDS``    : compile GDS / cuFile path
- ``FLEXKV_ENABLE_P2P``    : compile distributed Redis path
- ``FLEXKV_ENABLE_METRICS``: compile prometheus-cpp monitoring
- ``FLEXKV_ENABLE_CFS``    : compile pcfs
- ``FLEXKV_ENABLE_CPUTEST``: build without -lcuda for CPU-only test envs
- ``FLEXKV_DEBUG``         : skip cythonize, used by setup.py directly
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Type

from .base import GPUBuilder


# Common (vendor-agnostic) sources. These compile identically on NVIDIA / ROCm.
_COMMON_SOURCES = [
    "csrc/bindings.cpp",
    "csrc/hash.cpp",
    "csrc/transfer_ssd.cpp",
    "csrc/radix_tree.cpp",
    "csrc/eviction_strategy.cpp",
    "csrc/monitoring/metrics_manager.cpp",
]


def _exists(path: str) -> bool:
    return os.path.isfile(os.path.join(os.path.dirname(__file__), "..", path))


def _pick(*candidates: str) -> str:
    """Return the first existing path among ``candidates``.

    Used to make the builder tolerant to the P3 source-tree migration:
    while ``csrc/gpu_backend/nvidia/transfer.cu`` is being introduced,
    the legacy path ``csrc/transfer.cu`` may still be the real one.
    """
    for c in candidates:
        if _exists(c):
            return c
    # Fall back to the first; setuptools will surface a clear error.
    return candidates[0]


class NvidiaBuilder(GPUBuilder):
    name = "nvidia"

    # ------------------------------------------------------------
    def is_available(self) -> bool:
        try:
            import torch
            return getattr(torch.version, "hip", None) is None
        except Exception:
            # Even without torch.cuda we should still be able to build
            # in a CPU-only test container, controlled via FLEXKV_ENABLE_CPUTEST.
            return os.environ.get("FLEXKV_ENABLE_CPUTEST", "0") == "1"

    def get_extension_class(self) -> Type:
        from torch.utils.cpp_extension import CUDAExtension
        return CUDAExtension

    def get_extension_name(self) -> str:
        return "flexkv.c_ext"

    # ------------------------------------------------------------
    def configure_env(self) -> None:
        # Auto-detect CUDA arch list when not explicitly set.
        if os.environ.get("FLEXKV_ENABLE_CPUTEST", "0") == "1":
            os.environ.setdefault(
                "TORCH_CUDA_ARCH_LIST", "7.0;7.5;8.0;8.6;9.0"
            )
            return
        if os.environ.get("TORCH_CUDA_ARCH_LIST"):
            return
        # Delegate hardware detection to the GPU-backend abstraction so the
        # builder stays free of literal ``torch.cuda.*`` calls. Falling back
        # to a sensible default if detection fails.
        try:
            from flexkv.gpu_backend.nvidia.backend import NvidiaBackend
            archs = NvidiaBackend.detect_arch_list()
            if archs:
                os.environ["TORCH_CUDA_ARCH_LIST"] = ";".join(archs)
                return
        except Exception:
            pass
        os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;9.0"

    # ------------------------------------------------------------
    def get_sources(self, **opts: Any) -> List[str]:
        sources: List[str] = list(_COMMON_SOURCES)

        # GPU kernels: prefer the new vendor-scoped path, fall back to legacy.
        sources.append(_pick(
            "csrc/gpu_backend/nvidia/transfer.cu",
            "csrc/transfer.cu",
        ))
        sources.append(_pick(
            "csrc/gpu_backend/nvidia/tp_transfer_thread_group.cpp",
            "csrc/tp_transfer_thread_group.cpp",
        ))
        # NVIDIA-only pybind bindings (transfer_kv_blocks, GDSManager, ...).
        # csrc/bindings.cpp itself is GPU-agnostic and only forwards into
        # this translation unit via register_active_backend_bindings(m).
        sources.append("csrc/gpu_backend/nvidia/nvidia_bindings.cpp")

        if opts.get("enable_cfs"):
            sources.append("csrc/pcfs/pcfs.cpp")

        if opts.get("enable_gds"):
            sources.extend([
                _pick(
                    "csrc/gpu_backend/nvidia/gds/gds_manager.cpp",
                    "csrc/gds/gds_manager.cpp",
                ),
                _pick(
                    "csrc/gpu_backend/nvidia/gds/tp_gds_transfer_thread_group.cpp",
                    "csrc/gds/tp_gds_transfer_thread_group.cpp",
                ),
                _pick(
                    "csrc/gpu_backend/nvidia/gds/layout_transform.cu",
                    "csrc/gds/layout_transform.cu",
                ),
            ])

        if opts.get("enable_p2p"):
            sources.extend([
                "csrc/dist/distributed_radix_tree.cpp",
                "csrc/dist/local_radix_tree.cpp",
                "csrc/dist/redis_meta_channel.cpp",
                "csrc/dist/lease_meta_mempool.cpp",
            ])

        return sources

    # ------------------------------------------------------------
    def get_compile_args(self, **opts: Any) -> Dict[str, List[str]]:
        cxx = ["-std=c++17", "-O3", self.vendor_macro(), "-DCUDA_AVAILABLE"]
        nvcc = ["-O3", self.vendor_macro(), "-DCUDA_AVAILABLE"]

        if opts.get("enable_metrics"):
            cxx.append("-DFLEXKV_ENABLE_MONITORING")
            nvcc.append("-DFLEXKV_ENABLE_MONITORING")
        if opts.get("enable_gds"):
            cxx.append("-DFLEXKV_ENABLE_GDS")
            nvcc.append("-DFLEXKV_ENABLE_GDS")
        if opts.get("enable_cfs"):
            cxx.append("-DFLEXKV_ENABLE_CFS")
        if opts.get("enable_p2p"):
            cxx.append("-DFLEXKV_ENABLE_P2P")
            nvcc.append("-DFLEXKV_ENABLE_P2P")

        return {"cxx": cxx, "nvcc": nvcc}

    # ------------------------------------------------------------
    def get_link_args(self, **opts: Any) -> List[str]:
        link: List[str] = ["-lxxhash", "-lpthread", "-lrt", "-luring"]

        if not opts.get("enable_cputest"):
            link.append("-lcuda")

        if opts.get("enable_p2p"):
            link.append("-lhiredis")
        if opts.get("enable_gds"):
            link.append("-lcufile")
        if opts.get("enable_cfs"):
            link.append("-lhifs_client_sdk")
        if opts.get("enable_metrics"):
            link.extend(["-lprometheus-cpp-pull", "-lprometheus-cpp-core"])

        # rpath for build/lib + package-local lib
        build_dir = opts.get("build_dir", "build")
        lib_dir = os.path.join(build_dir, "lib")
        if os.path.exists(lib_dir):
            link.extend([f"-Wl,-rpath,{lib_dir}", "-Wl,-rpath,$ORIGIN"])
            link.append("-Wl,-rpath,$ORIGIN/../lib")

        return link

    def get_include_dirs(self, **opts: Any) -> List[str]:
        build_dir = opts.get("build_dir", "build")
        return [
            os.path.abspath(os.path.join(build_dir, "include")),
            os.path.abspath("csrc"),
            os.path.abspath("csrc/gpu_backend"),
        ]
