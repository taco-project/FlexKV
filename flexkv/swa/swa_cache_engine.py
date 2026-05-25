# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SWA (Sliding Window Attention) Cache Engine.

Manages a fixed-size pool of SWA pages.  Each page stores a complete
ring-buffer snapshot (window_size tokens x all SWA layers).

Unlike the main KV CacheEngine which uses a RadixTree (prefix matching),
the SWA pool uses a flat hash-map keyed by the sequence endpoint hash.
This implements the TRAILING_PAGES hit policy: only the latest page matters.

Design rationale:
  - Each sequence has exactly ONE SWA page (fixed 128-token window)
  - No prefix-sharing hierarchy => O(1) hash-map lookup is optimal
  - Endpoint hash = hash of the trailing block from the main KV tree
  - Eviction: LRU (oldest-accessed pages evicted first)
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

# Avoid importing from flexkv.cache (triggers c_ext load via __init__.py).
# Inline a minimal block allocator instead.

# HashType is NewType('HashType', int) from flexkv.common.hash_utils.
# We use int directly here to avoid the c_ext dependency at import time.
HashType = int


class _SlotAllocator:
    """Minimal free-list allocator for SWA slots (avoids Mempool c_ext dep)."""

    def __init__(self, num_slots: int):
        self._free = list(range(num_slots - 1, -1, -1))  # stack
        self._num_total = num_slots

    def allocate(self) -> Optional[int]:
        if not self._free:
            return None
        return self._free.pop()

    def recycle(self, slot_id: int) -> None:
        self._free.append(slot_id)

    def reset(self) -> None:
        self._free = list(range(self._num_total - 1, -1, -1))

    @property
    def num_free(self) -> int:
        return len(self._free)


@dataclass
class SWAMatchResult:
    """Result of an SWA match query."""
    hit: bool = False
    physical_block: int = -1
    endpoint_hash: HashType = field(default_factory=lambda: HashType(0))


@dataclass
class SWAEntry:
    """Metadata for one cached SWA page."""
    physical_block: int
    endpoint_hash: HashType
    last_access_time: float = 0.0
    lock_cnt: int = 0
    is_ready: bool = False

    def in_use(self) -> bool:
        """Returns True if entry is locked or not yet ready (transfer in-flight)."""
        return self.lock_cnt > 0 or not self.is_ready


class SWACacheEngine:
    """Cache engine for SWA pages with TRAILING_PAGES hit policy.

    Args:
        num_slots: Maximum number of SWA pages that can be cached.
        evict_ratio: Fraction of slots to evict when pool is full.
    """

    def __init__(self, num_slots: int, evict_ratio: float = 0.1):
        if num_slots <= 0:
            raise ValueError(f"num_slots must be > 0, got {num_slots}")
        if not (0.0 < evict_ratio <= 1.0):
            raise ValueError(f"evict_ratio must be in (0, 1], got {evict_ratio}")

        self.num_slots = num_slots
        self.evict_ratio = evict_ratio
        self._allocator = _SlotAllocator(num_slots)

        # Hash-map: endpoint_hash -> SWAEntry
        self._index: Dict[HashType, SWAEntry] = {}

    def reset(self) -> None:
        """Clear all cached SWA pages."""
        self._index.clear()
        self._allocator.reset()

    def match(self, endpoint_hash: HashType) -> SWAMatchResult:
        """TRAILING_PAGES match: look up SWA page by sequence endpoint hash.

        Args:
            endpoint_hash: Hash of the trailing block (sequence endpoint).

        Returns:
            SWAMatchResult with hit=True if page exists and is ready.
        """
        entry = self._index.get(endpoint_hash)
        if entry is not None and entry.is_ready:
            entry.last_access_time = time.time()
            return SWAMatchResult(
                hit=True,
                physical_block=entry.physical_block,
                endpoint_hash=endpoint_hash,
            )
        return SWAMatchResult(hit=False, endpoint_hash=endpoint_hash)

    def allocate(self, endpoint_hash: HashType) -> Optional[int]:
        """Allocate an SWA slot for a new page.  Evicts if needed.

        If a slot with the same endpoint_hash already exists, returns
        its existing physical_block (idempotent).

        Args:
            endpoint_hash: Hash key for this SWA page.

        Returns:
            Physical block ID or None if allocation failed.
        """
        existing = self._index.get(endpoint_hash)
        if existing is not None:
            return existing.physical_block

        # Evict if pool is full
        if self._allocator.num_free == 0:
            self._evict()

        if self._allocator.num_free == 0:
            return None

        physical_block = self._allocator.allocate()
        if physical_block is None:
            return None

        self._index[endpoint_hash] = SWAEntry(
            physical_block=physical_block,
            endpoint_hash=endpoint_hash,
            last_access_time=time.time(),
            lock_cnt=0,
            is_ready=False,
        )
        return physical_block

    def set_ready(self, endpoint_hash: HashType, ready: bool = True) -> None:
        """Mark an SWA entry as ready (data transfer complete)."""
        entry = self._index.get(endpoint_hash)
        if entry is not None:
            entry.is_ready = ready

    def lock(self, endpoint_hash: HashType) -> None:
        """Lock an SWA entry to prevent eviction during transfer."""
        entry = self._index.get(endpoint_hash)
        if entry is not None:
            entry.lock_cnt += 1

    def unlock(self, endpoint_hash: HashType) -> None:
        """Unlock an SWA entry."""
        entry = self._index.get(endpoint_hash)
        if entry is not None:
            entry.lock_cnt = max(0, entry.lock_cnt - 1)

    def remove(self, endpoint_hash: HashType) -> None:
        """Explicitly remove an SWA entry and recycle its block."""
        entry = self._index.pop(endpoint_hash, None)
        if entry is not None:
            self._allocator.recycle(entry.physical_block)

    def _evict(self) -> int:
        """Evict LRU entries to free slots.  Returns number evicted."""
        num_to_evict = max(1, int(self.num_slots * self.evict_ratio))

        # Collect evictable entries sorted by last_access_time (oldest first)
        candidates = [
            (entry.last_access_time, h, entry)
            for h, entry in self._index.items()
            if not entry.in_use()
        ]
        candidates.sort(key=lambda x: x[0])

        evicted = 0
        for _, h, entry in candidates[:num_to_evict]:
            self._allocator.recycle(entry.physical_block)
            del self._index[h]
            evicted += 1

        return evicted

    @property
    def num_cached(self) -> int:
        """Number of SWA pages currently cached."""
        return len(self._index)

    @property
    def num_free_slots(self) -> int:
        """Number of free SWA slots."""
        return self._allocator.num_free
