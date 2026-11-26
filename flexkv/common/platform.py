"""Runtime policy for host-only builds; GDR backend configuration is independent."""

import importlib.util
import logging
import os
from typing import Dict, List, Optional

_logger = logging.getLogger("flexkv.platform")
_MODE = "host-only build (FLEXKV_ENABLE_KERNEL_TRANSFER=0)"
# Written by setup.py; absent in unbuilt source trees, which default to kernel builds.
_BUILD_INFO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "_build_info.py")

# Contradict a host-only build (kernel transfers, mapped host memory); rejected.
NON_KERNEL_STACK_REQUIRED_ENV: Dict[str, str] = {
    "FLEXKV_USE_CE_TRANSFER_H2D": "1",
    "FLEXKV_USE_CE_TRANSFER_D2H": "1",
    "FLEXKV_CUDAHOST_REGISTER_FLAGS": "0",
}

# Compatibility defaults for non-kernel / mapped runtimes; explicit values win with a warning.
NON_KERNEL_STACK_RECOMMENDED_ENV: Dict[str, str] = {
    "FLEXKV_ENABLE_CE_MEMCPY2D": "0",
    "FLEXKV_CE_GATHER_BACKEND": "memcpy",
    "FLEXKV_DISABLE_VMM_PROBE": "1",
    "FLEXKV_FORCE_DIRECT_IPC": "0",
    "FLEXKV_STRICT_IPC_MODE": "1",
    "FLEXKV_LAYERWISE_NOTIFY_MODE": "polling",
    "FLEXKV_ENABLE_GDS": "0",
    "FLEXKV_ENABLE_NVCOMP": "0",
}

NON_KERNEL_STACK_ENV_PRESETS: Dict[str, str] = {
    **NON_KERNEL_STACK_REQUIRED_ENV, **NON_KERNEL_STACK_RECOMMENDED_ENV}

_warned_overrides = set()
_kernel_transfer_built: Optional[bool] = None


class NonKernelStackConfigError(RuntimeError):
    """An explicit setting conflicts with a host-only build."""


def kernel_transfer_built() -> bool:
    global _kernel_transfer_built
    if _kernel_transfer_built is None:
        if os.path.exists(_BUILD_INFO):
            spec = importlib.util.spec_from_file_location("flexkv._build_info", _BUILD_INFO)
            info = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(info)
            _kernel_transfer_built = bool(info.KERNEL_TRANSFER)
        else:
            _kernel_transfer_built = True
    return _kernel_transfer_built


def non_kernel_stack_mode() -> bool:
    """Host-only builds always run the non-kernel stack; there is no runtime switch."""
    return not kernel_transfer_built()


def _required_conflicts() -> List[str]:
    problems: List[str] = []
    for key, expected in NON_KERNEL_STACK_REQUIRED_ENV.items():
        if key not in os.environ:
            continue
        raw = os.environ[key]
        if key == "FLEXKV_CUDAHOST_REGISTER_FLAGS":
            try:
                valid = int(raw, 0) == 0
            except ValueError:
                valid = False
        else:
            valid = raw == expected
        if not valid:
            problems.append(f"{key}={raw!r} (requires {expected!r})")
    return problems


def _warn_once(key: str, message: str, *args) -> None:
    if key not in _warned_overrides:
        _warned_overrides.add(key)
        _logger.warning(message, *args)


def apply_non_kernel_stack_env_presets() -> Dict[str, str]:
    """Fill unset keys only; reject kernel-path settings before CUDA use."""
    if not non_kernel_stack_mode():
        return {}
    problems = _required_conflicts()
    if problems:
        raise NonKernelStackConfigError(
            f"{_MODE} conflicts with explicit settings: " + "; ".join(problems))

    applied: Dict[str, str] = {}
    for key, value in NON_KERNEL_STACK_ENV_PRESETS.items():
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value
        elif key in NON_KERNEL_STACK_RECOMMENDED_ENV and os.environ[key] != value:
            _warn_once(key, "%s: %s=%r overrides the recommended %r",
                       _MODE, key, os.environ[key], value)

    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        conf = os.environ.get(key, "").replace(" ", "").lower()
        if "expandable_segments:true" in conf:
            _warn_once(key, "%s: %s enables expandable_segments; "
                       "the recommended setting leaves it off", _MODE, key)

    if applied:
        _logger.info("%s: applied presets %s", _MODE, applied)
    return applied
