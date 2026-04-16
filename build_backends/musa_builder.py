"""MUSABuilder — MUSA compiler (mcc) + MUSAExtension build backend."""
import os
from .base import GPUBuilder

_MUSA_SOURCES = [
    "csrc/bindings.cpp",          # unified entry point — GPU backend selected via FLEXKV_BACKEND_MUSA macro
    "csrc/hash.cpp",
    "csrc/transfer_ssd.cpp",
    "csrc/radix_tree.cpp",
    "csrc/monitoring/metrics_manager.cpp",
    # MUSA-specific sources
    "csrc/gpu_backend/musa/tp_transfer_thread_group_musa.cpp",
]

_MUSA_KERNEL_SOURCES = [
    "csrc/gpu_backend/musa/transfer_musa.mu",
]

_MUSA_GDS_SOURCES = [
    "csrc/gpu_backend/musa/gds/gds_manager_musa.cpp",
    "csrc/gpu_backend/musa/gds/tp_gds_transfer_thread_group_musa.cpp",
    "csrc/gpu_backend/musa/gds/layout_transform_musa.mu",
]

_CFS_SOURCES = ["csrc/pcfs/pcfs.cpp"]


class MUSABuilder(GPUBuilder):

    def get_extension_class(self):
        try:
            from torch_musa.utils.musa_extension import MUSAExtension
            return MUSAExtension
        except ImportError:
            # Fall back to CppExtension (no GPU kernels, stub only)
            from torch.utils.cpp_extension import CppExtension
            return CppExtension

    def get_sources(self, *, storage_backend="posix", enable_p2p=False,
                    enable_cfs=False, enable_metrics=False, **kw) -> list:
        musa_home = os.environ.get("MUSA_HOME", "")
        has_mcc = bool(musa_home) and os.path.isfile(os.path.join(musa_home, "bin", "mcc"))

        srcs = list(_MUSA_SOURCES)
        if has_mcc:
            srcs.extend(_MUSA_KERNEL_SOURCES)
            # MUSA backend always enables GDS (muFile)
            srcs.extend(_MUSA_GDS_SOURCES)
        else:
            print("WARNING: mcc not found — building MUSA stub (no GPU kernels)")

        if enable_cfs:
            srcs.extend(_CFS_SOURCES)
        return srcs

    def get_compile_args(self, *, storage_backend="posix", enable_metrics=False,
                         enable_cfs=False, **kw) -> dict:
        flags = ["-DFLEXKV_BACKEND_MUSA", "-DMUSA_AVAILABLE", "-DFLEXKV_STORAGE_MUFILE",
                 "-DFLEXKV_ENABLE_GDS"]
        if enable_metrics:
            flags.append("-DFLEXKV_ENABLE_MONITORING")
        if enable_cfs:
            flags.append("-DFLEXKV_ENABLE_CFS")
        return {
            "mcc": ["-O3"] + flags,
            "cxx": ["-std=c++17", "-O3"] + flags,
        }

    def get_link_args(self, *, enable_p2p=False, enable_metrics=False,
                      enable_cfs=False, **kw) -> list:
        musa_home = os.environ.get("MUSA_HOME", "")
        args = ["-lxxhash", "-lpthread", "-lrt", "-luring", "-lmusart", "-lmufile"]
        if musa_home:
            args += [f"-L{os.path.join(musa_home, 'lib')}",
                     f"-Wl,-rpath,{os.path.join(musa_home, 'lib')}"]
        if enable_p2p:
            args.append("-lhiredis")
        if enable_metrics:
            args += ["-lprometheus-cpp-pull", "-lprometheus-cpp-core"]
        if enable_cfs:
            args.append("-lhifs_client_sdk")
        return args

    def get_build_ext_class(self):
        try:
            from torch_musa.utils.musa_extension import BuildExtension
            return BuildExtension
        except ImportError:
            from torch.utils.cpp_extension import BuildExtension
            return BuildExtension

    def configure_env(self):
        musa_home = os.environ.get("MUSA_HOME", "")
        if not musa_home:
            raise EnvironmentError(
                "MUSA_HOME must be set when FLEXKV_GPU_BACKEND=musa. "
                "Please set MUSA_HOME to your MUSA SDK installation path."
            )
        if musa_home:
            include = os.path.join(musa_home, "include")
            existing = os.environ.get("CPATH", "")
            os.environ["CPATH"] = f"{include}:{existing}" if existing else include
