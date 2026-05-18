"""RocmBuilder: produces ``flexkv.c_ext`` via ``hipcc`` (P4).

Strategy
--------
1. ``configure_env()`` runs hipify-perl on every ``.cu/.cuh`` source under
   ``csrc/gpu_backend/nvidia/`` (or the legacy path while P3 is in
   progress) and writes the converted files into
   ``csrc/gpu_backend/rocm/.hipified/``.
2. ROCm PyTorch's ``CUDAExtension`` automatically dispatches to ``hipcc``
   when ``torch.version.hip`` is set. We therefore just feed the
   hipified ``.cu`` files in as sources.
3. GDS / cuFile have no ROCm counterpart, so ``enable_gds`` is silently
   ignored. SSD path uses the GPU-agnostic ``csrc/transfer_ssd.cpp``.
"""
from __future__ import annotations

import os
import pathlib
import shutil
import subprocess
from typing import Any, Dict, List, Type

from .base import GPUBuilder
from .cuda_builder import _COMMON_SOURCES, _pick


_HIPIFY_TARGETS = [
    # (source under csrc/gpu_backend/nvidia or csrc/, dest under .hipified/)
    ("csrc/gpu_backend/nvidia/transfer.cu",
     "csrc/transfer.cu"),
    ("csrc/gpu_backend/nvidia/transfer.cuh",
     "csrc/transfer.cuh"),
    ("csrc/gpu_backend/nvidia/gtensor_handler.cuh",
     "csrc/gtensor_handler.cuh"),
]
_HIPIFY_DST_DIR = "csrc/gpu_backend/rocm/.hipified"


class RocmBuilder(GPUBuilder):
    name = "rocm"

    # ------------------------------------------------------------
    def is_available(self) -> bool:
        try:
            import torch
            return getattr(torch.version, "hip", None) is not None
        except Exception:
            return False

    def get_extension_class(self) -> Type:
        # ROCm PyTorch reuses CUDAExtension and routes through hipcc.
        from torch.utils.cpp_extension import CUDAExtension
        return CUDAExtension

    def get_extension_name(self) -> str:
        return "flexkv.c_ext"

    # ------------------------------------------------------------
    def configure_env(self) -> None:
        # Auto-detect PYTORCH_ROCM_ARCH via the GPU-backend abstraction.
        # If detection fails (no GPU visible during build), fall back to
        # MI200 (gfx90a) + MI300 (gfx942) which covers the common cases.
        if not os.environ.get("PYTORCH_ROCM_ARCH"):
            archs: List[str] = []
            try:
                from flexkv.gpu_backend.rocm.backend import RocmBackend
                archs = RocmBackend.detect_arch_list()
            except Exception:
                archs = []
            os.environ["PYTORCH_ROCM_ARCH"] = (
                ";".join(archs) if archs else "gfx90a;gfx942"
            )
        # Run hipify ahead of compilation.
        self._run_hipify()

    def _run_hipify(self) -> None:
        repo_root = pathlib.Path(__file__).resolve().parents[1]
        dst_root = repo_root / _HIPIFY_DST_DIR
        dst_root.mkdir(parents=True, exist_ok=True)

        hipify = shutil.which("hipify-perl")
        if hipify is None:
            raise RuntimeError(
                "hipify-perl is required for FLEXKV_GPU_BACKEND=rocm but "
                "was not found on PATH. Make sure ROCm dev tools are "
                "installed (typically /opt/rocm/bin)."
            )

        for primary, fallback in _HIPIFY_TARGETS:
            src_path = repo_root / primary
            if not src_path.exists():
                src_path = repo_root / fallback
            if not src_path.exists():
                continue
            rel = src_path.name
            dst_path = dst_root / rel
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            subprocess.check_call(
                [hipify, str(src_path), "-o", str(dst_path)]
            )

    # ------------------------------------------------------------
    def get_sources(self, **opts: Any) -> List[str]:
        sources: List[str] = list(_COMMON_SOURCES)

        # tp_transfer_thread_group.cpp uses cross-vendor gpu_* macros from
        # csrc/gpu_backend/gpu_types.h, so we can reuse the NVIDIA-tree copy
        # under -DFLEXKV_BACKEND_ROCM (no second source file needed).
        sources.append("csrc/gpu_backend/nvidia/tp_transfer_thread_group.cpp")

        # Hipified GPU kernels.
        sources.append(os.path.join(_HIPIFY_DST_DIR, "transfer.cu"))

        # ROCm-only pybind bindings (transfer_kv_blocks). No GDS counterpart.
        sources.append("csrc/gpu_backend/rocm/rocm_bindings.cpp")

        if opts.get("enable_cfs"):
            sources.append("csrc/pcfs/pcfs.cpp")

        if opts.get("enable_gds"):
            # ROCm has no GDS today; emit a clear warning rather than fail.
            print(
                "[flexkv] FLEXKV_ENABLE_GDS=1 ignored on ROCm "
                "(no equivalent of cuFile)."
            )

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
        cxx = ["-std=c++17", "-O3", self.vendor_macro()]
        nvcc = ["-O3", self.vendor_macro()]

        if opts.get("enable_metrics"):
            cxx.append("-DFLEXKV_ENABLE_MONITORING")
            nvcc.append("-DFLEXKV_ENABLE_MONITORING")
        if opts.get("enable_cfs"):
            cxx.append("-DFLEXKV_ENABLE_CFS")
        if opts.get("enable_p2p"):
            cxx.append("-DFLEXKV_ENABLE_P2P")

        return {"cxx": cxx, "nvcc": nvcc}

    # ------------------------------------------------------------
    def get_link_args(self, **opts: Any) -> List[str]:
        # No -lcuda / -lcufile on ROCm. ROCm PyTorch provides hip libs.
        link: List[str] = ["-lxxhash", "-lpthread", "-lrt", "-luring"]

        if opts.get("enable_p2p"):
            link.append("-lhiredis")
        if opts.get("enable_cfs"):
            link.append("-lhifs_client_sdk")
        if opts.get("enable_metrics"):
            link.extend(["-lprometheus-cpp-pull", "-lprometheus-cpp-core"])

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
            # tp_transfer_thread_group.h is cross-vendor (gpu_* macros) but
            # physically lives under the nvidia tree; we need it on the
            # include path for rocm_bindings.cpp.
            os.path.abspath("csrc/gpu_backend/nvidia"),
            os.path.abspath(_HIPIFY_DST_DIR),
        ]
