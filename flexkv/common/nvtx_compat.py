"""NVTX compatibility layer for FlexKV's Python sources.

Why this exists
---------------
FlexKV annotates hot Python paths with NVTX ranges (``nvtx.start_range`` /
``end_range`` / ``push_range`` / ``pop_range`` / ``mark`` / ``@annotate``) so
they line up with the C++ ranges in Nsight Systems timelines. Those calls come
from the ``nvtx`` PyPI package, which is NVIDIA-specific: it links against the
CUDA toolkit's NVTX library and needs an NVIDIA profiler to consume the events.

On CUDA-like accelerators that are not NVIDIA GPUs -- Baidu Kunlun P800 being
the case FlexKV cares about -- the ``nvtx`` wheel is typically not installable
and there is no profiler listening. Historically the options were to either
install a dummy wheel or to strip every annotation from the sources; the latter
forks the code and makes upstream merges painful.

This module removes the dilemma. It re-exports the real ``nvtx`` API when the
package is importable, and otherwise provides zero-overhead no-op stubs with
identical signatures. Call sites change only their import statement::

    -import nvtx
    +from flexkv.common import nvtx_compat as nvtx

Everything downstream (``nvtx.start_range(...)``, ``@nvtx.annotate(...)``,
context-manager usage) keeps working unchanged.

Override
--------
Set ``FLEXKV_ENABLE_NVTX=0`` to force the stubs even on a machine where the
real package is installed. This is useful for A/B measuring the (small)
annotation overhead, and for reproducing P800 behaviour on an NVIDIA host.
This module is imported by FlexKV's lowest-level hot paths, so it deliberately
avoids ``from __future__ import annotations`` and any syntax newer than what
``setup.py``'s ``python_requires=">=3.6"`` promises.
"""

import importlib.util
import os
from typing import Any, Callable, Optional, TypeVar

__all__ = [
    "HAS_NVTX",
    "annotate",
    "end_range",
    "mark",
    "pop_range",
    "push_range",
    "start_range",
]

_F = TypeVar("_F", bound=Callable[..., Any])


# Python modules whose presence identifies a vendor accelerator that emulates
# CUDA rather than being one. Such runtimes deliberately look like CUDA -- they
# ship a CUDA toolkit, report a ``torch.version.cuda``, and may even provide the
# ``nvtx`` wheel as a transitive dependency -- so importability of ``nvtx``
# alone is not a reliable signal.
#
#   torch_xmlir : Baidu Kunlun (P800 and friends)
#   torch_npu   : Huawei Ascend
#
# NVTX events are only ever consumed by Nsight Systems talking to the NVIDIA
# driver, so on these platforms the annotations cost time and produce nothing.
_EMULATED_CUDA_MODULES = ("torch_xmlir", "torch_npu")


def _detect_emulated_cuda_backend():
    # type: () -> Optional[str]
    """Return the name of a detected non-NVIDIA CUDA-like backend, or None.

    Uses ``find_spec`` rather than a real import so that merely asking the
    question does not pull in a heavyweight vendor runtime (importing
    ``torch_xmlir`` loads libbkcl and rewrites torch symbols).
    """
    for name in _EMULATED_CUDA_MODULES:
        try:
            if importlib.util.find_spec(name) is not None:
                return name
        except (ImportError, ValueError):
            # find_spec can raise for half-installed packages; treat as absent.
            continue
    return None


def _nvtx_requested():
    # type: () -> bool
    """Whether real NVTX should be used, honouring FLEXKV_ENABLE_NVTX."""
    flag = os.getenv("FLEXKV_ENABLE_NVTX")
    if flag is not None:
        return flag.strip().lower() not in ("0", "false", "no", "off")
    # Auto: real NVTX only on genuine NVIDIA platforms.
    return _detect_emulated_cuda_backend() is None


_real_nvtx = None  # type: Optional[Any]
if _nvtx_requested():
    try:
        import nvtx as _real_nvtx  # type: ignore[no-redef]
    except ImportError:
        _real_nvtx = None

HAS_NVTX = _real_nvtx is not None  # type: bool

if HAS_NVTX:
    # Re-export the genuine implementations. Kept as direct references (rather
    # than wrappers) so there is no added call overhead on NVIDIA platforms.
    annotate = _real_nvtx.annotate
    end_range = _real_nvtx.end_range
    mark = _real_nvtx.mark
    pop_range = _real_nvtx.pop_range
    push_range = _real_nvtx.push_range
    start_range = _real_nvtx.start_range

else:

    class _NoopAnnotate:
        """Stand-in for ``nvtx.annotate``.

        The real object doubles as a decorator and a context manager, and
        accepts both ``annotate("name")`` and ``annotate(message="name",
        color="red")``. All of those forms must keep working.
        """

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        # --- context-manager protocol: `with nvtx.annotate("x"):` ---
        def __enter__(self):
            # type: () -> "_NoopAnnotate"
            return self

        def __exit__(self, exc_type, exc, tb):
            # type: (Any, Any, Any) -> bool
            # Never swallow exceptions.
            return False

        # --- decorator protocol: `@nvtx.annotate("x")` ---
        def __call__(self, func):
            # type: (_F) -> _F
            return func

    def annotate(*args, **kwargs):
        # type: (*Any, **Any) -> _NoopAnnotate
        """No-op replacement for ``nvtx.annotate``."""
        return _NoopAnnotate(*args, **kwargs)

    def start_range(*args, **kwargs):
        # type: (*Any, **Any) -> int
        """No-op ``nvtx.start_range``; returns an opaque handle.

        A non-zero constant is returned so that call sites guarding on
        truthiness (``if nvtx_range:``) behave as they do with real NVTX,
        where a successful start never yields 0.
        """
        del args, kwargs
        return 1

    def end_range(range_id=None):
        # type: (Any) -> None
        """No-op ``nvtx.end_range``."""
        del range_id

    def push_range(*args, **kwargs):
        # type: (*Any, **Any) -> None
        """No-op ``nvtx.push_range``."""
        del args, kwargs

    def pop_range(*args, **kwargs):
        # type: (*Any, **Any) -> None
        """No-op ``nvtx.pop_range``."""
        del args, kwargs

    def mark(*args, **kwargs):
        # type: (*Any, **Any) -> None
        """No-op ``nvtx.mark``."""
        del args, kwargs
