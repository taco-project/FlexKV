"""MusaBuilder: skeleton, mirrors NvidiaBuilder but compiles into
``flexkv.c_ext_musa`` using ``mcc``.

This file does not implement a full MUSA build today; it raises a clear
error when invoked, so users understand what to install.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Type

from .base import GPUBuilder


class MusaBuilder(GPUBuilder):
    name = "musa"

    def is_available(self) -> bool:
        try:
            import torch_musa  # noqa: F401
            return True
        except Exception:
            return False

    def get_extension_class(self) -> Type:
        try:
            from torch_musa.utils.musa_extension import MUSAExtension  # type: ignore
            return MUSAExtension
        except Exception as e:
            raise RuntimeError(
                "MusaBuilder requires torch_musa.utils.musa_extension.MUSAExtension "
                "to be importable. Install torch_musa first."
            ) from e

    def get_extension_name(self) -> str:
        return "flexkv.c_ext_musa"

    def configure_env(self) -> None:
        if "MUSA_HOME" not in os.environ:
            raise RuntimeError(
                "MUSA_HOME must be set when building with FLEXKV_GPU_BACKEND=musa"
            )

    def get_sources(self, **opts: Any) -> List[str]:
        return [
            "csrc/bindings.cpp",
            "csrc/hash.cpp",
            "csrc/transfer_ssd.cpp",
            "csrc/radix_tree.cpp",
            "csrc/eviction_strategy.cpp",
            "csrc/monitoring/metrics_manager.cpp",
            "csrc/gpu_backend/nvidia/tp_transfer_thread_group.cpp",
            "csrc/gpu_backend/musa/transfer.mu",
        ]

    def get_compile_args(self, **opts: Any) -> Dict[str, List[str]]:
        cxx = ["-std=c++17", "-O3", self.vendor_macro()]
        return {"cxx": cxx, "mcc": ["-O3", self.vendor_macro()]}

    def get_link_args(self, **opts: Any) -> List[str]:
        return ["-lxxhash", "-lpthread", "-lrt", "-luring"]

    def get_include_dirs(self, **opts: Any) -> List[str]:
        build_dir = opts.get("build_dir", "build")
        return [
            os.path.abspath(os.path.join(build_dir, "include")),
            os.path.abspath("csrc"),
            os.path.abspath("csrc/gpu_backend"),
        ]
