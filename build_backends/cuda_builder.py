"""CUDABuilder — NVIDIA nvcc + CUDAExtension."""
import os
from .base import GPUBuilder

# Shared CPU-only source files required by all backends
_COMMON_SOURCES = [
    "csrc/bindings.cpp",
    "csrc/hash.cpp",
    "csrc/transfer_ssd.cpp",
    "csrc/radix_tree.cpp",
    "csrc/monitoring/metrics_manager.cpp",
]

# NVIDIA-specific kernel and runtime source files
_NVIDIA_SOURCES = [
    "csrc/transfer.cu",
    "csrc/tp_transfer_thread_group.cpp",
]

_GDS_SOURCES = [
    "csrc/gds/gds_manager.cpp",
    "csrc/gds/tp_gds_transfer_thread_group.cpp",
    "csrc/gds/layout_transform.cu",
]

_P2P_SOURCES = [
    "csrc/dist/distributed_radix_tree.cpp",
    "csrc/dist/local_radix_tree.cpp",
    "csrc/dist/redis_meta_channel.cpp",
    "csrc/dist/lease_meta_mempool.cpp",
]

_CFS_SOURCES = [
    "csrc/pcfs/pcfs.cpp",
]


class CUDABuilder(GPUBuilder):

    def get_extension_class(self):
        from torch.utils.cpp_extension import CUDAExtension
        return CUDAExtension

    def get_sources(self, *, storage_backend="posix", enable_p2p=False,
                    enable_cfs=False, enable_metrics=False, **kw) -> list:
        srcs = list(_COMMON_SOURCES) + list(_NVIDIA_SOURCES)
        if storage_backend in ("cufile", "gds"):
            srcs.extend(_GDS_SOURCES)
        if enable_p2p:
            srcs.extend(_P2P_SOURCES)
        if enable_cfs:
            srcs.extend(_CFS_SOURCES)
        return srcs

    def get_compile_args(self, *, storage_backend="posix", enable_metrics=False,
                         enable_p2p=False, enable_cfs=False, **kw) -> dict:
        cxx = ["-std=c++17", "-O3", "-DCUDA_AVAILABLE", "-DFLEXKV_BACKEND_NVIDIA"]
        nvcc = ["-O3", "-DCUDA_AVAILABLE", "-DFLEXKV_BACKEND_NVIDIA"]
        if storage_backend in ("cufile", "gds"):
            cxx.append("-DFLEXKV_ENABLE_GDS")
            nvcc.append("-DFLEXKV_ENABLE_GDS")
        if enable_metrics:
            cxx.append("-DFLEXKV_ENABLE_MONITORING")
            nvcc.append("-DFLEXKV_ENABLE_MONITORING")
        if enable_cfs:
            cxx.append("-DFLEXKV_ENABLE_CFS")
        if enable_p2p:
            cxx.append("-DFLEXKV_ENABLE_P2P")
        return {"nvcc": nvcc, "cxx": cxx}

    def get_link_args(self, *, storage_backend="posix", enable_p2p=False,
                      enable_metrics=False, enable_cfs=False,
                      enable_cputest=False, **kw) -> list:
        args = ["-lxxhash", "-lpthread", "-lrt", "-luring"]
        if not enable_cputest:
            args.insert(0, "-lcuda")
        if storage_backend in ("cufile", "gds"):
            args.append("-lcufile")
        if enable_p2p:
            args.append("-lhiredis")
        if enable_metrics:
            args += ["-lprometheus-cpp-pull", "-lprometheus-cpp-core"]
        if enable_cfs:
            args.append("-lhifs_client_sdk")
        return args

    def get_build_ext_class(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def configure_env(self):
        if not os.environ.get("TORCH_CUDA_ARCH_LIST"):
            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0;8.6;9.0"
