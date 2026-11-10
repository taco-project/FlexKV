"""Opt-in runtime presets for some special machine types and scenarios; GDR/XDR configuration is independent."""

import importlib.util
import logging
import os
import re
import sys
from typing import Dict, List, Optional

_logger = logging.getLogger("flexkv.platform")
ENV_ENABLE_CE_ONLY = "FLEXKV_ENABLE_CE_ONLY"
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"", "0", "false", "no", "off"}

CE_ONLY_ENV_PRESETS: Dict[str, str] = {
    "FLEXKV_USE_CE_TRANSFER_H2D": "1",
    "FLEXKV_USE_CE_TRANSFER_D2H": "1",
    "FLEXKV_ENABLE_CE_MEMCPY2D": "0",
    "FLEXKV_CE_GATHER_BACKEND": "memcpy",
    "FLEXKV_DISABLE_VMM_PROBE": "1",
    "FLEXKV_FORCE_DIRECT_IPC": "0",
    "FLEXKV_LAYERWISE_NOTIFY_MODE": "polling",
    "FLEXKV_ENABLE_MPS": "0",
    "FLEXKV_CUDAHOST_REGISTER_FLAGS": "0",
}


class CeOnlyConfigError(RuntimeError):
    """An explicit setting conflicts with FLEXKV_ENABLE_CE_ONLY=1."""


def _parse_bool(name: str, raw: Optional[str]) -> bool:
    value = (raw or "").strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(
        f"{name} must be one of 0/1/true/false/yes/no/on/off (got {raw!r})")


def ce_only_enabled() -> bool:
    return _parse_bool(ENV_ENABLE_CE_ONLY, os.environ.get(ENV_ENABLE_CE_ONLY))


def ce_only_mode() -> bool:
    """Emulated CUDA detection does not silently enable the runtime switch."""
    return ce_only_enabled()


def is_emulated_cuda_backend() -> bool:
    """Probe without importing a backend or initializing a device."""
    if "torch_xmlir" in sys.modules:
        return True
    try:
        return importlib.util.find_spec("torch_xmlir") is not None
    except (ImportError, ValueError):
        return False


def _validate_explicit_values() -> List[str]:
    problems: List[str] = []
    for key, expected in CE_ONLY_ENV_PRESETS.items():
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
    for key in ("FLEXKV_ENABLE_GDS", "FLEXKV_ENABLE_NVCOMP"):
        if key in os.environ and os.environ[key] != "0":
            problems.append(f"{key}={os.environ[key]!r} (requires '0')")
    return problems


def strip_expandable_segments(conf: str) -> str:
    """Remove an allocator option without splitting bracketed option values."""
    cleaned = re.sub(
        r"(^|[,;])\s*expandable_segments\s*:\s*(?:true|false)\s*(?=[,;]|$)",
        "", conf, flags=re.IGNORECASE)
    return cleaned.strip(",; ") if cleaned != conf else conf


def apply_ce_only_env_presets() -> Dict[str, str]:
    """Fill unset keys only; reject incompatible explicit settings before CUDA use."""
    if not ce_only_enabled():
        return {}
    problems = _validate_explicit_values()
    if problems:
        raise CeOnlyConfigError(
            f"{ENV_ENABLE_CE_ONLY}=1 conflicts with explicit settings: "
            + "; ".join(problems))

    applied: Dict[str, str] = {}
    for key, value in CE_ONLY_ENV_PRESETS.items():
        if key not in os.environ:
            os.environ[key] = value
            applied[key] = value

    for key in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        conf = os.environ.get(key, "")
        cleaned = strip_expandable_segments(conf)
        if cleaned == conf:
            continue
        if cleaned:
            os.environ[key] = cleaned
        else:
            os.environ.pop(key, None)
        applied[key] = cleaned
        _logger.warning(
            "%s=1: removed expandable_segments from %s; start the process "
            "without this option if the CUDA allocator was already initialized",
            ENV_ENABLE_CE_ONLY, key)

    if applied:
        _logger.info("%s=1: applied presets %s", ENV_ENABLE_CE_ONLY, applied)
    return applied
