"""
mooncake_store_keys.py
----------------------
Single source of truth for Mooncake-store key suffixes used across the
FlexKV ↔ mooncake-store integration.

Two suffixes are defined:

* ``KV_SUFFIX``       – full KV-cache blocks (default FlexKV traffic).
* ``INDEXER_SUFFIX``  – DSA indexer side-car blocks (one per main KV
  block, identical block-id namespace, separate Mooncake worker).

A helper ``build_key(block_hash, suffix)`` is provided to make the
key-construction explicit at call sites.
"""

from __future__ import annotations

from typing import Union
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Dict


class PoolKind(str, Enum):
    KV = "FlexKV"
    INDEXER = "FlexKV_indexer"
    SWA = "FlexKV_swa"

@dataclass(frozen=True)
class PoolSpec:
    kind: PoolKind
    required_for_hit: bool = True

def build_key(
    block_hash: Union[int, str],
    kind: PoolKind,
    pp_rank: int = 0,
    pp_size: int = 1,
    node_layer_start: int = 0,
    node_layer_end: int = 0,
    total_layers: int = 0,
) -> str:
    """Build a Mooncake-store key from a block hash and a pool kind.

    Key suffix policy (driven by the *node-local* CPU pool layer range):

    * Single-node deployment (any ``pp_size``): the per-node CPU pool covers
      the *full* model layer set, so the on-the-wire block is bitwise
      identical to a PP=1 deployment's block. No suffix is appended,
      letting single-node PP=1 / PP=2 / PP=4 instances share keys and
      hit each other's cache.

    * Cross-node PP (node only holds part of the model): suffix is
      ``_pp_rank_<i>_of_<N>`` so that different PP topologies (e.g.
      2-node PP=2 vs 4-node PP=4) do NOT alias each other ��� their
      per-node layer slices have different lengths and would yield
      bitwise-incompatible blocks under the same key.

    Determination of "single-node" is done from the *layer range* the
    node's CPU pool covers, not from ``pp_size`` directly, because
    single-node PP>1 deployments still get a full-model CPU pool
    (see PR #171 / ``num_layers_on_node`` in ``transfer_manager.py``).
    """
    base = f"{block_hash}_{kind.value}"
    # Case 1: node CPU pool covers the entire model -> no suffix.
    # ``total_layers == 0`` defends pre-PR call sites that didn't pass
    # the new fields; in that case fall through to the legacy logic.
    if total_layers > 0 and (node_layer_end - node_layer_start) == total_layers:
        return base
    # Case 2: cross-node PP -> per-stage suffix carrying the topology.
    if pp_size > 1:
        return f"{base}_pp_rank_{pp_rank}_of_{pp_size}"
    return base
