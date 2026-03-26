"""GPUBuilder abstract base class."""
import os
from abc import ABC, abstractmethod


class GPUBuilder(ABC):

    @abstractmethod
    def get_extension_class(self):
        """Return CUDAExtension / MUSAExtension / CppExtension class."""

    @abstractmethod
    def get_sources(self, *, storage_backend="posix", enable_p2p=False,
                    enable_cfs=False, enable_metrics=False, **kw) -> list:
        """Return list of source files to compile."""

    @abstractmethod
    def get_compile_args(self, **opts) -> dict:
        """Return extra_compile_args dict (keys: nvcc/mcc/cxx)."""

    @abstractmethod
    def get_link_args(self, **opts) -> list:
        """Return extra_link_args list."""

    @abstractmethod
    def get_build_ext_class(self):
        """Return BuildExtension class."""

    def configure_env(self):
        """Pre-build environment setup (e.g. set TORCH_CUDA_ARCH_LIST)."""
        pass
