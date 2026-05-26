"""SWA Radix Manager — Coordinates SWA pool operations with radix tree annotations.

This module provides the SWARadixManager class which:
1. Manages the SWA CPU host pool (put/load_back)
2. Maintains the SWA LRU list for eviction ordering
3. Handles cascading eviction (radix tree leaf evict -> SWA release)
4. Validates SWA trailing-window availability for prefix matches
5. Manages SWA lock/unlock for active requests
"""
import time
from typing import Optional, List, Union, TYPE_CHECKING

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from flexkv.common.config import SWAPoolConfig
from flexkv.swa.swa_host_pool import SWAHostPool
from flexkv.swa.swa_pool_lru import SWAPoolLRU

if TYPE_CHECKING:
    from flexkv.cache.radixtree import RadixNode


class SWARadixManager:
    """Coordinates SWA storage with radix tree node annotations.

    Usage (write-through on finish):
        1. Request finishes -> call swa_put(leaf_node, swa_data)
        2. Prefix match -> call check_swa_trailing() then swa_load_back()
        3. Radix tree evicts leaf -> call on_leaf_evict(node)
    """

    def __init__(self, config: SWAPoolConfig, tokens_per_block: int = 16):
        self._config = config
        self._tokens_per_block = tokens_per_block
        self._pool = SWAHostPool(config)
        self._lru = SWAPoolLRU()

        # Statistics
        self._stats_puts = 0
        self._stats_hits = 0
        self._stats_misses = 0
        self._stats_evictions = 0

    # --- Core Operations ---------------------------------------------------

    def swa_put(self, node: 'RadixNode', swa_data: Union['torch.Tensor', np.ndarray, bytes]) -> bool:
        """Store SWA data for a radix tree node (write-through on request finish).

        If the pool is full, triggers LRU eviction to make space.

        Args:
            node: The radix tree node to annotate (typically the last node covering
                  the trailing window_size tokens).
            swa_data: Raw SWA snapshot bytes.

        Returns:
            True if successfully stored, False if eviction couldn't free space.
        """
        # If node already has SWA, update in place
        if node.swa_host_slot is not None:
            self._pool.write(node.swa_host_slot, swa_data)
            node.swa_last_access_time = time.time()
            self._lru.promote_mru(node)
            self._stats_puts += 1
            return True

        # Allocate new slot
        slot_id = self._pool.allocate()
        if slot_id is None:
            # Pool full — evict to make space
            num_to_evict = max(1, int(self._config.num_slots * self._config.evict_ratio))
            evicted = self.swa_evict_for_space(num_to_evict)
            if evicted == 0:
                return False  # All locked, can't evict
            slot_id = self._pool.allocate()
            if slot_id is None:
                return False

        # Write data and annotate node
        self._pool.write(slot_id, swa_data)
        node.swa_host_slot = slot_id
        node.swa_tombstone = False
        node.swa_last_access_time = time.time()
        self._lru.insert_mru(node)
        self._stats_puts += 1
        return True

    def swa_load_back(self, node: 'RadixNode') -> Optional[Union['torch.Tensor', np.ndarray]]:
        """Read SWA data from CPU pool for load-back to GPU.

        Args:
            node: Node with swa_host_slot set (not tombstone).

        Returns:
            SWA data buffer (CPU tensor/array) or None if not available.
        """
        if node.swa_tombstone or node.swa_host_slot is None:
            self._stats_misses += 1
            return None

        data = self._pool.read_copy(node.swa_host_slot)
        node.swa_last_access_time = time.time()
        self._lru.promote_mru(node)
        self._stats_hits += 1
        return data

    def swa_evict_for_space(self, num_to_evict: int = 1) -> int:
        """Evict LRU SWA entries to free pool slots.

        Evicts the oldest unlocked SWA entries, marks their nodes as tombstone.
        Does NOT delete radix tree nodes — only releases SWA data.

        Returns:
            Number of slots actually freed.
        """
        evicted = 0
        while evicted < num_to_evict:
            node = self._lru.get_lru_evictable()
            if node is None:
                break  # All remaining are locked

            # Release slot
            self._pool.free(node.swa_host_slot)
            # Mark node as tombstone
            node.swa_host_slot = None
            node.swa_tombstone = True
            # Remove from LRU
            self._lru.remove(node)

            evicted += 1
            self._stats_evictions += 1

        return evicted

    # --- Radix Tree Integration --------------------------------------------

    def on_leaf_evict(self, node: 'RadixNode') -> None:
        """Called when a radix tree leaf node is being evicted/deleted.

        Cascades: releases the SWA slot if present.
        """
        if node.swa_host_slot is not None:
            self._pool.free(node.swa_host_slot)
            self._lru.remove(node)
            node.swa_host_slot = None
            node.swa_tombstone = True

    def check_swa_trailing(self, path: List['RadixNode'],
                           window_tokens: Optional[int] = None) -> bool:
        """Check if the trailing `window_tokens` tokens along path have SWA.

        The path is ordered root->leaf. We sum token counts from the leaf end
        backward and verify each node has SWA (not tombstone).

        Args:
            path: List of RadixNodes from root to leaf (inclusive).
            window_tokens: Number of trailing tokens required (default: config.window_size).

        Returns:
            True if at least window_tokens of trailing tokens have SWA available.
        """
        if window_tokens is None:
            window_tokens = self._config.window_size

        trailing_available = 0
        # Walk from leaf to root
        for node in reversed(path):
            if node.swa_tombstone:
                break  # Gap found
            node_tokens = node.size() * self._tokens_per_block
            trailing_available += node_tokens
            if trailing_available >= window_tokens:
                return True

        return trailing_available >= window_tokens

    # --- Locking -----------------------------------------------------------

    def swa_lock(self, leaf_node: 'RadixNode',
                 window_tokens: Optional[int] = None) -> None:
        """Lock SWA for trailing window_tokens from leaf toward root.

        Locked nodes won't be evicted by swa_evict_for_space().
        """
        if window_tokens is None:
            window_tokens = self._config.window_size

        locked_tokens = 0
        node = leaf_node
        while node is not None and not node.is_root():
            if node.swa_tombstone:
                node = node.parent
                continue
            node.swa_lock_ref += 1
            locked_tokens += node.size() * self._tokens_per_block
            if locked_tokens >= window_tokens:
                break
            node = node.parent

    def swa_unlock(self, leaf_node: 'RadixNode',
                   window_tokens: Optional[int] = None) -> None:
        """Unlock SWA for trailing window_tokens from leaf toward root."""
        if window_tokens is None:
            window_tokens = self._config.window_size

        unlocked_tokens = 0
        node = leaf_node
        while node is not None and not node.is_root():
            if node.swa_tombstone:
                node = node.parent
                continue
            assert node.swa_lock_ref > 0, \
                f"swa_unlock on node with swa_lock_ref={node.swa_lock_ref}"
            node.swa_lock_ref -= 1
            unlocked_tokens += node.size() * self._tokens_per_block
            if unlocked_tokens >= window_tokens:
                break
            node = node.parent

    # --- Properties & Stats ------------------------------------------------

    @property
    def pool(self) -> SWAHostPool:
        return self._pool

    @property
    def lru(self) -> SWAPoolLRU:
        return self._lru

    @property
    def config(self) -> SWAPoolConfig:
        return self._config

    @property
    def stats(self) -> dict:
        return {
            "puts": self._stats_puts,
            "hits": self._stats_hits,
            "misses": self._stats_misses,
            "evictions": self._stats_evictions,
            "pool_used": self._pool.num_used,
            "pool_free": self._pool.num_free,
            "lru_size": len(self._lru),
        }
