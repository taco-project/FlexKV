"""Opt-in Mooncake transfer fault injection (off by default).

Purpose
-------
Exercise the "publish what actually transferred" pipeline
(``DeferredCacheInsert`` joint guard + ``_finalize_prefetch_return_mask``)
end-to-end without waiting for real network flakes. Call this at the only
point where per-block booleans are produced (``batch_get`` / ``batch_put``);
every downstream stage — ``CompletedOp.block_results``, ``MooncakeLoadResult``,
commit-time prefix math, return_mask narrowing (including the joint Full+SWA
gate that reports 0 unless both succeed), outcome metric — sees the same
injected bitmap.

Environment variables (all optional)
------------------------------------
``FLEXKV_MOONCAKE_FAULT_MODE``
    ``none`` (default / empty) — passthrough.
    ``tail_fail`` — first ``N - K`` True, last ``K`` False;
    ``K`` from ``FLEXKV_MOONCAKE_FAULT_TAIL_N`` (default 1).
    ``first_ok`` — exactly one True at the head.
    ``random`` — each block fails with probability
    ``FLEXKV_MOONCAKE_FAULT_RANDOM_RATIO`` (default 0.3).
    ``full_fail`` — all False.

``FLEXKV_MOONCAKE_FAULT_PROB``
    Per-call probability the fault applies (default 1.0).

``FLEXKV_MOONCAKE_FAULT_OP``
    ``both`` (default) / ``get`` / ``put`` — restrict to one direction.

``FLEXKV_MOONCAKE_FAULT_SEED``
    Optional int; seeds the fault RNG for reproducible runs.

A bare production env (mode unset / ``none``) is a byte-for-byte no-op.
"""

from __future__ import annotations

import os
from typing import List, Optional, Union

import numpy as np

from flexkv.common.debug import flexkv_logger
from flexkv.common.transfer import TransferType

_FAULT_RNG: Optional[np.random.Generator] = None


def is_mooncake_fault_inject_enabled() -> bool:
    """True when ``FLEXKV_MOONCAKE_FAULT_MODE`` is set to an active mode."""
    mode = os.environ.get("FLEXKV_MOONCAKE_FAULT_MODE", "").strip().lower()
    return bool(mode) and mode != "none"


def _get_fault_rng() -> np.random.Generator:
    global _FAULT_RNG
    if _FAULT_RNG is None:
        seed_env = os.environ.get("FLEXKV_MOONCAKE_FAULT_SEED", "")
        seed = int(seed_env) if seed_env else None
        _FAULT_RNG = np.random.default_rng(seed)
    return _FAULT_RNG


def reset_mooncake_fault_rng() -> None:
    """Drop the cached RNG so the next call re-reads ``FAULT_SEED``.

    Intended for unit tests; production callers do not need this.
    """
    global _FAULT_RNG
    _FAULT_RNG = None


def inject_mooncake_fault(
    results: List[bool],
    transfer_type: Union[TransferType, str],
) -> List[bool]:
    """Optionally replace ``results`` with a partial-failure pattern.

    ``transfer_type`` may be a ``TransferType`` (``REMOTE2H`` / ``H2REMOTE``)
    or a string name / ``\"get\"`` / ``\"put\"``. No-op unless
    ``FLEXKV_MOONCAKE_FAULT_MODE`` is set to something other than ``none``.
    """
    mode = os.environ.get("FLEXKV_MOONCAKE_FAULT_MODE", "").strip().lower()
    if not mode or mode == "none":
        return results

    op_name = _normalize_op_name(transfer_type)
    op_filter = os.environ.get("FLEXKV_MOONCAKE_FAULT_OP", "both").strip().lower()
    if op_filter == "get" and op_name != "get":
        return results
    if op_filter == "put" and op_name != "put":
        return results

    n = len(results)
    if n == 0:
        return results

    try:
        prob = float(os.environ.get("FLEXKV_MOONCAKE_FAULT_PROB", "1.0"))
    except ValueError:
        prob = 1.0
    if prob < 1.0 and _get_fault_rng().random() >= prob:
        return results

    if mode == "full_fail":
        injected = [False] * n
    elif mode == "first_ok":
        injected = [True] + [False] * (n - 1)
    elif mode == "tail_fail":
        try:
            tail_n = int(os.environ.get("FLEXKV_MOONCAKE_FAULT_TAIL_N", "1"))
        except ValueError:
            tail_n = 1
        tail_n = max(1, min(n, tail_n))
        injected = [True] * (n - tail_n) + [False] * tail_n
    elif mode == "random":
        try:
            ratio = float(os.environ.get("FLEXKV_MOONCAKE_FAULT_RANDOM_RATIO", "0.3"))
        except ValueError:
            ratio = 0.3
        ratio = max(0.0, min(1.0, ratio))
        rng = _get_fault_rng()
        injected = [rng.random() >= ratio for _ in range(n)]
    else:
        flexkv_logger.warning(
            f"[Mooncake-Fault] unknown mode '{mode}'; passthrough")
        return results

    flexkv_logger.warning(
        f"[Mooncake-Fault] mode={mode} op={op_name} "
        f"injected={sum(injected)}/{n} true "
        f"(orig={sum(bool(r) for r in results)}/{n})")
    return injected


def _normalize_op_name(transfer_type: Union[TransferType, str]) -> str:
    if isinstance(transfer_type, TransferType):
        if transfer_type == TransferType.REMOTE2H:
            return "get"
        if transfer_type == TransferType.H2REMOTE:
            return "put"
        return transfer_type.name.lower()
    name = str(transfer_type).strip().lower()
    if name in ("get", "remote2h", "transfertype.remote2h"):
        return "get"
    if name in ("put", "h2remote", "transfertype.h2remote"):
        return "put"
    return name
