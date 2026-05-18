"""GenericBuilder: produces NO C extension.

Used to install FlexKV in CPU-only / dev environments. ``setup.py`` will
detect this builder and skip the Extension altogether.
"""
from __future__ import annotations

from typing import Any, Dict, List, Type

from .base import GPUBuilder


class GenericBuilder(GPUBuilder):
    name = "generic"

    def is_available(self) -> bool:
        return True

    def get_extension_class(self) -> Type:  # pragma: no cover - never called
        from setuptools import Extension
        return Extension

    def get_extension_name(self) -> str:
        return ""

    def get_sources(self, **opts: Any) -> List[str]:
        return []

    def get_compile_args(self, **opts: Any) -> Dict[str, List[str]]:
        return {"cxx": []}

    def get_link_args(self, **opts: Any) -> List[str]:
        return []

    def get_include_dirs(self, **opts: Any) -> List[str]:
        return []
