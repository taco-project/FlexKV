"""SWA Production Manager — node-attached SWA pool manager for C++ radix trees.

Works with the C++ LocalRadixTree (via flexkv.c_ext) whose CRadixNode now
carries SWA state directly (swa_host_slot / swa_tombstone / swa_lock_ref).
This manager is therefore a thin layer over the SWA host pool plus an LRU
ordering used only for *SWA-only* eviction (tombstoning). It does NOT do any
prefix matching of its own — the caller resolves the node via the radix tree
and passes it in.

Design (mirrors the Python SWARadixManager and sglang's swa_radix_cache):
    - SWA state lives on the node: node.swa_host_slot (-1 = none),
      node.swa_tombstone (True = no SWA data), node.swa_lock_ref.
    - put/get/has take a node, not token_ids.
    - SWA-only eviction: when the pool is full, pick the LRU unlocked node,
      free its slot, tombstone it (the node and its full KV stay alive).
    - Cascade eviction (full KV evicted -> SWA slot released) is handled by the
      C++ tree recording freed slots; the cache engine drains them and calls
      free_slots(). This manager only owns the pool and the LRU ordering.

Invariant: SWA is a subset of full. A node may have full KV without SWA
(tombstone); it must never have SWA without full. The single source of truth is
node.swa_host_slot — the LRU dict may lag (stale entries) but is never trusted
for correctness, only for eviction ordering.

Thread-safety: all public methods are protected by a lock for safe use from
the FlexKV task engine's callback threads.
"""
import threading
from typing import Optional, Iterable, Union
from collections import OrderedDict

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from flexkv.common.config import SWAPoolConfig
from flexkv.common.debug import flexkv_logger
from flexkv.swa.swa_host_pool import SWAHostPool


class SWAProductionManager:
    """Node-attached SWA pool manager for the C++ radix tree path.

    SWA state is stored on the CRadixNode (swa_host_slot / swa_tombstone /
    swa_lock_ref). This class owns only the host pool and an LRU ordering for
    SWA-only eviction.
    """

    def __init__(self, config: SWAPoolConfig, tokens_per_block: int):
        """Initialize the production SWA manager.

        Args:
            config: SWA pool configuration.
            tokens_per_block: Number of tokens per radix tree block (kept for
                API parity; not used for keying anymore).
        """
        self._config = config
        self._tokens_per_block = tokens_per_block
        self._pool = SWAHostPool(config)
        self._lock = threading.Lock()

        # LRU ordering for SWA-only eviction. Key = id(node), Value = node.
        # Only nodes that currently hold a slot are kept here. id(node) is used
        # as the key to avoid relying on CRadixNode being hashable; the node
        # object is stored as the value so we can read/write its SWA fields.
        self._lru: "OrderedDict[int, object]" = OrderedDict()

        # Statistics
        self._stats_puts = 0
        self._stats_hits = 0
        self._stats_misses = 0
        self._stats_evictions = 0
        self._stats_cascade_evictions = 0

    # --- Core Operations -----------------------------------------------------

    def put(self, node, swa_data: Union['torch.Tensor', np.ndarray, bytes]) -> bool:
        """Store SWA data on a radix-tree node (write-through on request finish).

        The node must already exist in the tree (full KV inserted), which the
        caller guarantees by resolving it via match_prefix / insert. If the pool
        is full, a SWA-only eviction is triggered to make space.

        Args:
            node: The radix tree node (CRadixNode) to annotate.
            swa_data: Raw SWA snapshot data.

        Returns:
            True if stored, False if the pool is full and every entry is locked.
        """
        if node is None:
            return False

        with self._lock:
            # Case A: node already holds a slot -> overwrite in place (no new
            # allocation, so the old slot is never leaked).
            if node.swa_host_slot != -1:
                self._pool.write(node.swa_host_slot, swa_data)
                node.swa_tombstone = False
                self._lru[id(node)] = node
                self._lru.move_to_end(id(node))
                self._stats_puts += 1
                return True

            # Case B: allocate a new slot, evicting (tombstoning) if needed.
            slot_id = self._pool.allocate()
            if slot_id is None:
                slot_id = self._evict_lru_and_alloc()
                if slot_id is None:
                    return False  # everything locked, cannot make space

            # Write data first, then bind to the node (write-before-bind: a node
            # never points at a slot whose data is not yet valid).
            self._pool.write(slot_id, swa_data)
            node.swa_host_slot = slot_id
            node.swa_tombstone = False
            self._lru[id(node)] = node
            self._lru.move_to_end(id(node))
            self._stats_puts += 1
            return True

    def get(self, node) -> Optional[Union['torch.Tensor', np.ndarray]]:
        """Retrieve SWA data for a node, or None if not available.

        Args:
            node: The radix tree node to read.

        Returns:
            SWA data buffer (CPU tensor/array copy) or None.
        """
        if node is None:
            return None

        with self._lock:
            if node.swa_tombstone or node.swa_host_slot == -1:
                self._stats_misses += 1
                return None
            data = self._pool.read_copy(node.swa_host_slot)
            # promote to MRU
            self._lru[id(node)] = node
            self._lru.move_to_end(id(node))
            self._stats_hits += 1
            return data

    def has(self, node) -> bool:
        """Check whether SWA data is available for a node.

        Args:
            node: The radix tree node to check.

        Returns:
            True if the node has a live SWA slot (not a tombstone).
        """
        if node is None:
            return False
        return (not node.swa_tombstone) and node.swa_host_slot != -1

    def lock(self, node) -> None:
        """Lock a node's SWA entry from eviction."""
        if node is None:
            return
        with self._lock:
            if node.swa_host_slot != -1:
                node.inc_swa_lock_ref()

    def unlock(self, node) -> None:
        """Unlock a node's SWA entry."""
        if node is None:
            return
        with self._lock:
            if node.swa_lock_ref > 0:
                node.dec_swa_lock_ref()

    def free_slots(self, slot_ids: Iterable[int]) -> int:
        """Return slots to the pool (cascade eviction from the radix tree).

        Called by the cache engine after draining freed_swa_slots from the C++
        tree, i.e. when full-KV eviction / split deleted or invalidated nodes
        that held SWA slots. Enforces the SWA-subset-of-full invariant.

        Note: the node's swa_host_slot was already cleared to -1 by the C++ side
        (record_freed_swa_slot), so the stale LRU entry will be skipped lazily
        on the next eviction scan.

        Args:
            slot_ids: Slot ids to release.

        Returns:
            Number of slots freed.
        """
        freed = 0
        with self._lock:
            for s in slot_ids:
                if s is not None and int(s) >= 0:
                    self._pool.free(int(s))
                    freed += 1
                    self._stats_cascade_evictions += 1
        return freed

    # --- Internal Helpers ----------------------------------------------------

    def _evict_lru_and_alloc(self) -> Optional[int]:
        """SWA-only eviction: free the LRU unlocked node's slot, then allocate.

        Walks the LRU from oldest to newest. Stale entries (whose node was
        already released to -1 by a cascade) are dropped. Locked entries
        (swa_lock_ref > 0) are skipped. The victim is tombstoned — its node and
        full KV remain; only the SWA snapshot is dropped.

        MUST be called while holding self._lock.

        Returns:
            A freshly allocated slot id, or None if nothing was evictable.
        """
        for nid in list(self._lru):
            node = self._lru[nid]
            # Stale: the tree already released this node's slot (cascade). Drop.
            if node.swa_host_slot == -1:
                del self._lru[nid]
                continue
            # Locked: actively in use, cannot evict.
            if node.swa_lock_ref > 0:
                continue
            # Evict (tombstone) this victim.
            self._pool.free(node.swa_host_slot)
            node.swa_host_slot = -1
            node.swa_tombstone = True
            del self._lru[nid]
            self._stats_evictions += 1
            return self._pool.allocate()
        return None

    # --- Properties & Stats --------------------------------------------------

    @property
    def pool(self) -> SWAHostPool:
        """Access the underlying host pool."""
        return self._pool

    @property
    def config(self) -> SWAPoolConfig:
        """Access the SWA configuration."""
        return self._config

    @property
    def num_entries(self) -> int:
        """Number of nodes currently tracked as holding a slot (approximate)."""
        with self._lock:
            return len(self._lru)

    @property
    def stats(self) -> dict:
        """Return SWA statistics."""
        with self._lock:
            return {
                "puts": self._stats_puts,
                "hits": self._stats_hits,
                "misses": self._stats_misses,
                "evictions": self._stats_evictions,
                "cascade_evictions": self._stats_cascade_evictions,
                "pool_used": self._pool.num_used,
                "pool_free": self._pool.num_free,
                "lru_size": len(self._lru),
            }
