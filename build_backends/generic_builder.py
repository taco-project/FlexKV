"""GenericBuilder — pure CppExtension with no GPU kernels, for CI or GPU-less environments."""
from .base import GPUBuilder

_GENERIC_SOURCES = [
    "csrc/bindings.cpp",
    "csrc/hash.cpp",
    "csrc/transfer_ssd.cpp",
    "csrc/radix_tree.cpp",
    "csrc/monitoring/metrics_manager.cpp",
    "csrc/tp_transfer_thread_group.cpp",
]


class GenericBuilder(GPUBuilder):

    def get_extension_class(self):
        from torch.utils.cpp_extension import CppExtension
        return CppExtension

    def get_sources(self, **kw) -> list:
        return list(_GENERIC_SOURCES)

    def get_compile_args(self, *, enable_metrics=False, **kw) -> dict:
        flags = ["-std=c++17", "-O2", "-DFLEXKV_BACKEND_GENERIC"]
        if enable_metrics:
            flags.append("-DFLEXKV_ENABLE_MONITORING")
        return {"cxx": flags}

    def get_link_args(self, *, enable_metrics=False, **kw) -> list:
        args = ["-lxxhash", "-lpthread", "-lrt", "-luring"]
        if enable_metrics:
            args += ["-lprometheus-cpp-pull", "-lprometheus-cpp-core"]
        return args

    def get_build_ext_class(self):
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension
