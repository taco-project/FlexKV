"""
GPU backend detection and dispatch for FlexKV.
Selects CUDA (flexkv.c_ext) or MUSA (flexkv.c_ext_musa) based on availability.

MUSA uses the torch_musa project (PyTorch for Moore Threads MUSA GPUs).
When torch.musa is available, backend is 'musa' and transfer uses c_ext_musa.
See: https://github.com/MooreThreads/torch_musa
"""
from typing import Optional, Any


def get_gpu_backend() -> str:
    """
    Return the active GPU backend: 'cuda' or 'musa'.
    Prefers CUDA if both are available unless explicitly overridden.
    """
    import torch
    # Allow override for testing or explicit MUSA selection
    import os
    if os.environ.get("FLEXKV_GPU_BACKEND", "").lower() == "musa":
        return "musa"
    if torch.cuda.is_available():
        return "cuda"
    if _musa_available():
        return "musa"
    return "cuda"  # default name for extension; may not be available


def _musa_available() -> bool:
    """Return True if torch.musa (torch_musa project) is available and has devices."""
    try:
        import torch
        musa = getattr(torch, "musa", None)
        if musa is None:
            return False
        return getattr(musa, "is_available", lambda: False)()
    except Exception:
        return False


def get_transfer_kv_blocks_module() -> Any:
    """
    Return the module that provides transfer_kv_blocks for the active backend.
    Either flexkv.c_ext (CUDA) or flexkv.c_ext_musa (MUSA).
    """
    backend = get_gpu_backend()
    if backend == "musa":
        try:
            return __import__("flexkv.c_ext_musa", fromlist=["transfer_kv_blocks"])
        except ImportError:
            # MUSA extension not built; fall back to c_ext (will fail if no CUDA)
            try:
                return __import__("flexkv.c_ext", fromlist=["transfer_kv_blocks"])
            except ImportError:
                raise ImportError(
                    "Neither flexkv.c_ext_musa nor flexkv.c_ext is available. "
                    "Please build FlexKV with CUDA support or MUSA support."
                )
    try:
        return __import__("flexkv.c_ext", fromlist=["transfer_kv_blocks"])
    except ImportError:
        # Try MUSA as fallback
        try:
            return __import__("flexkv.c_ext_musa", fromlist=["transfer_kv_blocks"])
        except ImportError:
            raise ImportError(
                "Neither flexkv.c_ext nor flexkv.c_ext_musa is available. "
                "Please build FlexKV with CUDA support or MUSA support."
            )
