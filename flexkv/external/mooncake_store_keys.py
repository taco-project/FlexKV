"""
mooncake_store_keys.py
----------------------
Single source of truth for Mooncake-store key suffixes used across the
FlexKV ↔ mooncake-store integration.

Two pool kinds are defined:

* ``KV``  – full KV-cache blocks (default FlexKV traffic).
* ``SWA`` – sliding-window attention snapshot (one tail block per window).

Indexer data is stored inside the KV block payload, not as a separate key.
"""

from __future__ import annotations

from enum import Enum
from typing import Union


class PoolKind(str, Enum):
    KV = "FlexKV"
    SWA = "FlexKV_swa"


def build_key(
    block_hash: Union[int, str],
    kind: PoolKind,
    pp_rank: int = 0,
    pp_size: int = 1,
    node_layer_start: int = 0,
    node_layer_end: int = 0,
    total_layers: int = 0,
) -> str:
    """Build a Mooncake-store key from a block hash and a pool kind."""
    base = f"{block_hash}_{kind.value}"
    if total_layers > 0 and (node_layer_end - node_layer_start) == total_layers:
        return base
    if pp_size > 1:
        return f"{base}_pp_rank_{pp_rank}_of_{pp_size}"
    return base
