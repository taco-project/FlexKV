"""SWA Production Manager — Production-ready SWA manager for C++ radix trees.

Works with the C++ LocalRadixTree (via flexkv.c_ext) where nodes are opaque
CRadixNode pointers that cannot have Python fields attached. Instead, SWA
metadata is tracked in Python-side dictionaries keyed by an endpoint hash
derived from the token sequence.

Design:
    - Uses hash(token_ids[-tokens_per_block:].tobytes()) as the endpoint key
    - Maintains a Dict[int, int] mapping: endpoint_hash -> swa_pool_slot_id
    - Has its own LRU tracking (simple ordered list, no intrusive pointers)
    - Integrates with HierarchyLRCacheEngine via on_blocks_evicted() callback
    - No dependency on Python RadixNode objects at all
"""
import time
import threading
from typing import Optional, Dict, List, Union
from collections import OrderedDict

import numpy as np

try:
    import torch
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from flexkv.common.config import SWAPoolConfig
from flexkv.swa.swa_host_pool import SWAHostPool


class SWAProductionManager:
    """Production SWA manager that works with C++ radix trees.

    Uses endpoint hashing (hash of last block's token_ids) to key SWA metadata,
    avoiding any dependency on Python RadixNode objects.

    Thread-safety: All public methods are protected by a lock for safe use
    from the FlexKV task engine's callback threads.
    """

    def __init__(self, config: SWAPoolConfig, tokens_per_block: int):
        """Initialize the production SWA manager.

        Args:
            config: SWA pool configuration.
            tokens_per_block: Number of tokens per radix tree block.
        """
        self._config = config
        self._tokens_per_block = tokens_per_block
        self._pool = SWAHostPool(config)
        self._lock = threading.Lock()

        # SWA slot tracking: endpoint_hash -> slot_id
        self._hash_to_slot: Dict[int, int] = {}

        # LRU ordering: OrderedDict for efficient move-to-end (MRU) and pop-first (LRU)
        # Key = endpoint_hash, Value = insertion/access timestamp
        self._lru_order: 'OrderedDict[int, float]' = OrderedDict()

        # Lock tracking: endpoint_hash -> lock_count
        self._locks: Dict[int, int] = {}

        # Reverse mapping for eviction cascade: block_id -> endpoint_hash
        # Tracks the LAST block ID of each stored sequence so that when the
        # radix tree evicts blocks, we can identify which SWA entries to invalidate.
        self._block_to_hash: Dict[int, int] = {}

        # Forward mapping: endpoint_hash -> last_block_id (for cleanup)
        self._hash_to_block: Dict[int, int] = {}

        # Statistics
        self._stats_puts = 0
        self._stats_hits = 0
        self._stats_misses = 0
        self._stats_evictions = 0
        self._stats_cascade_evictions = 0

    # --- Endpoint Hash Computation -------------------------------------------

    def _compute_endpoint_hash(self, token_ids: np.ndarray) -> int:
        """Compute the endpoint hash for a token sequence.

        Uses the last tokens_per_block tokens as the key.
        This is stable: same token sequence always produces same hash.

        Args:
            token_ids: Token sequence (at least tokens_per_block long).

        Returns:
            Integer hash value.
        """
        if len(token_ids) < self._tokens_per_block:
            # For short sequences, use all tokens
            return hash(token_ids.tobytes())
        # Use the last block's worth of tokens
        last_block = token_ids[-self._tokens_per_block:]
        return hash(last_block.tobytes())

    def _compute_last_block_id(self, token_ids: np.ndarray,
                                physical_block_ids: Optional[np.ndarray] = None) -> Optional[int]:
        """Compute the last physical block ID for eviction tracking.

        If physical_block_ids is provided, uses the last one.
        Otherwise returns None (eviction tracking disabled for this entry).
        """
        if physical_block_ids is not None and len(physical_block_ids) > 0:
            return int(physical_block_ids[-1])
        return None

    # --- Core Operations -----------------------------------------------------

    def put(self, token_ids: np.ndarray,
            swa_data: Union['torch.Tensor', np.ndarray, bytes],
            physical_block_ids: Optional[np.ndarray] = None) -> bool:
        """Store SWA data keyed by endpoint hash of token_ids.

        Args:
            token_ids: Token sequence for this SWA entry.
            swa_data: Raw SWA snapshot data.
            physical_block_ids: Optional physical block IDs from the radix tree
                insert. Used for eviction cascade tracking.

        Returns:
            True if successfully stored, False if pool is full and all locked.
        """
        if _TORCH_AVAILABLE and isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        token_ids = np.asarray(token_ids, dtype=np.int64)

        endpoint_hash = self._compute_endpoint_hash(token_ids)
        last_block_id = self._compute_last_block_id(token_ids, physical_block_ids)

        with self._lock:
            # If already stored, update in place
            if endpoint_hash in self._hash_to_slot:
                slot_id = self._hash_to_slot[endpoint_hash]
                self._pool.write(slot_id, swa_data)
                # Promote to MRU
                self._lru_order.move_to_end(endpoint_hash)
                self._lru_order[endpoint_hash] = time.time()
                # Update block tracking if changed
                if last_block_id is not None:
                    old_block = self._hash_to_block.get(endpoint_hash)
                    if old_block is not None and old_block != last_block_id:
                        self._block_to_hash.pop(old_block, None)
                    self._hash_to_block[endpoint_hash] = last_block_id
                    self._block_to_hash[last_block_id] = endpoint_hash
                self._stats_puts += 1
                return True

            # Allocate new slot
            slot_id = self._pool.allocate()
            if slot_id is None:
                # Pool full — evict to make space
                num_to_evict = max(1, int(self._config.num_slots * self._config.evict_ratio))
                evicted = self._evict_lru(num_to_evict)
                if evicted == 0:
                    return False  # All locked, can't evict
                slot_id = self._pool.allocate()
                if slot_id is None:
                    return False

            # Write data
            self._pool.write(slot_id, swa_data)
            self._hash_to_slot[endpoint_hash] = slot_id
            self._lru_order[endpoint_hash] = time.time()

            # Set up block tracking
            if last_block_id is not None:
                self._hash_to_block[endpoint_hash] = last_block_id
                self._block_to_hash[last_block_id] = endpoint_hash

            self._stats_puts += 1
            return True

    def get(self, token_ids: np.ndarray) -> Optional[Union['torch.Tensor', np.ndarray]]:
        """Retrieve SWA data by token_ids.

        Args:
            token_ids: Token sequence to look up.

        Returns:
            SWA data buffer (CPU tensor/array copy) or None if not available.
        """
        if _TORCH_AVAILABLE and isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        token_ids = np.asarray(token_ids, dtype=np.int64)

        endpoint_hash = self._compute_endpoint_hash(token_ids)

        with self._lock:
            slot_id = self._hash_to_slot.get(endpoint_hash)
            if slot_id is None:
                self._stats_misses += 1
                return None

            # Read data and promote to MRU
            data = self._pool.read_copy(slot_id)
            self._lru_order.move_to_end(endpoint_hash)
            self._lru_order[endpoint_hash] = time.time()
            self._stats_hits += 1
            return data

    def has(self, token_ids: np.ndarray) -> bool:
        """Check if SWA is available for given token_ids.

        Args:
            token_ids: Token sequence to check.

        Returns:
            True if SWA data is stored for this sequence.
        """
        if _TORCH_AVAILABLE and isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        token_ids = np.asarray(token_ids, dtype=np.int64)

        endpoint_hash = self._compute_endpoint_hash(token_ids)

        with self._lock:
            return endpoint_hash in self._hash_to_slot

    def on_blocks_evicted(self, evicted_block_ids: np.ndarray) -> int:
        """Called when blocks are evicted from the radix tree.

        Checks if any of the evicted blocks is an "endpoint block" for one of
        our tracked SWA entries. If so, invalidates that SWA entry (cascading eviction).

        Args:
            evicted_block_ids: Array of physical block IDs that were evicted.

        Returns:
            Number of SWA entries invalidated.
        """
        if len(evicted_block_ids) == 0:
            return 0

        invalidated = 0
        with self._lock:
            for block_id in evicted_block_ids:
                block_id_int = int(block_id)
                endpoint_hash = self._block_to_hash.get(block_id_int)
                if endpoint_hash is not None:
                    # Check if locked
                    if self._locks.get(endpoint_hash, 0) > 0:
                        # Cannot evict locked entry - skip
                        continue
                    self._release_entry(endpoint_hash)
                    invalidated += 1
                    self._stats_cascade_evictions += 1

        return invalidated

    def lock(self, token_ids: np.ndarray) -> None:
        """Lock SWA entry from eviction.

        Args:
            token_ids: Token sequence identifying the SWA entry to lock.
        """
        if _TORCH_AVAILABLE and isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        token_ids = np.asarray(token_ids, dtype=np.int64)

        endpoint_hash = self._compute_endpoint_hash(token_ids)

        with self._lock:
            if endpoint_hash in self._hash_to_slot:
                self._locks[endpoint_hash] = self._locks.get(endpoint_hash, 0) + 1

    def unlock(self, token_ids: np.ndarray) -> None:
        """Unlock SWA entry.

        Args:
            token_ids: Token sequence identifying the SWA entry to unlock.
        """
        if _TORCH_AVAILABLE and isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        token_ids = np.asarray(token_ids, dtype=np.int64)

        endpoint_hash = self._compute_endpoint_hash(token_ids)

        with self._lock:
            if endpoint_hash in self._locks:
                self._locks[endpoint_hash] -= 1
                if self._locks[endpoint_hash] <= 0:
                    del self._locks[endpoint_hash]

    # --- Internal Helpers ----------------------------------------------------

    def _evict_lru(self, num_to_evict: int) -> int:
        """Evict LRU (oldest) SWA entries to free pool slots.

        Skips locked entries.

        Args:
            num_to_evict: Target number of entries to evict.

        Returns:
            Actual number of entries evicted.
        """
        evicted = 0
        # Iterate from LRU (front of OrderedDict) toward MRU (end)
        # Collect hashes to evict first to avoid modifying dict during iteration
        to_evict: List[int] = []
        for endpoint_hash in self._lru_order:
            if evicted + len(to_evict) >= num_to_evict:
                break
            if self._locks.get(endpoint_hash, 0) == 0:
                to_evict.append(endpoint_hash)

        for endpoint_hash in to_evict:
            self._release_entry(endpoint_hash)
            evicted += 1
            self._stats_evictions += 1

        return evicted

    def _release_entry(self, endpoint_hash: int) -> None:
        """Release an SWA entry: free slot, remove from all tracking dicts.

        MUST be called while holding self._lock.
        """
        slot_id = self._hash_to_slot.pop(endpoint_hash, None)
        if slot_id is not None:
            self._pool.free(slot_id)

        # Remove from LRU
        self._lru_order.pop(endpoint_hash, None)

        # Remove block tracking
        block_id = self._hash_to_block.pop(endpoint_hash, None)
        if block_id is not None:
            self._block_to_hash.pop(block_id, None)

        # Remove locks (if any)
        self._locks.pop(endpoint_hash, None)

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
        """Number of active SWA entries."""
        with self._lock:
            return len(self._hash_to_slot)

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
                "num_entries": len(self._hash_to_slot),
                "num_locked": len(self._locks),
            }
