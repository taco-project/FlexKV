"""NVTX API with an opt-in no-op implementation for CUDA compatibility."""

import os

from flexkv.common.platform import non_kernel_stack_mode

_flag = os.environ.get("FLEXKV_ENABLE_NVTX")
_requested = (_flag.strip().lower() not in ("0", "false", "no", "off")
              if _flag is not None else not non_kernel_stack_mode())
_real_nvtx = None
if _requested:
    import nvtx as _real_nvtx

HAS_NVTX = _real_nvtx is not None

if HAS_NVTX:
    annotate = _real_nvtx.annotate
    start_range = _real_nvtx.start_range
    end_range = _real_nvtx.end_range
    push_range = _real_nvtx.push_range
    pop_range = _real_nvtx.pop_range
    mark = _real_nvtx.mark
else:
    class _NoopAnnotate:
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def __call__(self, func):
            return func

    annotate = _NoopAnnotate

    def start_range(*args, **kwargs):
        return 1

    def end_range(*args, **kwargs):
        pass

    def push_range(*args, **kwargs):
        pass

    def pop_range(*args, **kwargs):
        pass

    def mark(*args, **kwargs):
        pass
