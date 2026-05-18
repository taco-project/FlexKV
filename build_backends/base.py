"""GPUBuilder ABC."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Type


class GPUBuilder(ABC):
    """Vendor-specific build description for ``setup.py``.

    A builder centralizes:

    - the list of source files to compile;
    - the C++/NVCC/HIPCC compile flags;
    - the link flags;
    - environment-variable side effects (e.g. ``TORCH_CUDA_ARCH_LIST`` /
      ``PYTORCH_ROCM_ARCH``);
    - which Extension subclass to use (``CUDAExtension`` for both
      NVIDIA and ROCm under PyTorch, plain ``Extension`` for MUSA/generic).
    """

    name: str

    @abstractmethod
    def is_available(self) -> bool: ...

    @abstractmethod
    def get_extension_class(self) -> Type: ...

    def get_extension_name(self) -> str:
        return "flexkv.c_ext"

    @abstractmethod
    def get_sources(self, **opts: Any) -> List[str]: ...

    @abstractmethod
    def get_compile_args(self, **opts: Any) -> Dict[str, List[str]]: ...

    @abstractmethod
    def get_link_args(self, **opts: Any) -> List[str]: ...

    @abstractmethod
    def get_include_dirs(self, **opts: Any) -> List[str]: ...

    def get_library_dirs(self, **opts: Any) -> List[str]:
        return [os.path.join(opts.get("build_dir", "build"), "lib")]

    def get_build_ext_class(self) -> Type:
        from torch.utils.cpp_extension import BuildExtension
        return BuildExtension

    def configure_env(self) -> None:
        """Mutate environment variables ahead of compilation."""
        return None

    def vendor_macro(self) -> str:
        """Return ``-DFLEXKV_BACKEND_<VENDOR>`` for the C ``gpu_types.h``."""
        return f"-DFLEXKV_BACKEND_{self.name.upper()}"
