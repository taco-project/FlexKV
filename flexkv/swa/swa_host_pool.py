"""SWA Host Pool — CPU-side slot-id allocator for SWA pages.

SWA is managed at PAGE granularity: each slot denotes exactly one cache page of
SWA KV. Cache/storage code derives the page size from ``tokens_per_block`` and
all SWA IO addresses a whole slot (= one page) at a time.

This pool is purely a slot-id allocator / free-list (stack): it hands out and
reclaims integer slot ids and keeps the used/free accounting. It does NOT hold
the SWA KV bytes — those live in the ``StorageEngine`` buffer allocated with
``is_swa=True`` (sized from the cache page geometry) and are read/written
by the transfer worker via the storage handle, addressed by slot id. When the
pool is full, the caller (cache engine) triggers SWA-LRU eviction before retrying.
"""
from typing import Optional

from flexkv.common.config import SWAPoolConfig


class SWAHostPool:
    """Fixed-size SWA slot-id allocator (free-list); holds no KV bytes."""

    def __init__(self, config: SWAPoolConfig):
        self._config = config
        self._num_slots = config.num_slots

        # Free-list (stack-based)
        self._free_slots = list(range(self._num_slots - 1, -1, -1))

    # --- Allocation --------------------------------------------------------

    def allocate(self) -> Optional[int]:
        """Allocate a slot. Returns slot_id or None if pool is full."""
        if not self._free_slots:
            return None
        return self._free_slots.pop()

    def free(self, slot_id: int) -> None:
        """Return a slot to the free list."""
        slot_id = int(slot_id)
        if slot_id < 0 or slot_id >= self._num_slots:
            raise ValueError(f"Invalid SWA slot id: {slot_id}")
        if slot_id in self._free_slots:
            return
        self._free_slots.append(slot_id)

    def reset(self) -> None:
        """Return every slot to the free list (all SWA state dropped).

        Called when the owning radix tree is reset: the tree bulk-deletes all
        nodes without buffering their slots, so the pool must be re-armed as
        fully free to avoid permanently leaking those slots.
        """
        self._free_slots = list(range(self._num_slots - 1, -1, -1))

    # --- Properties --------------------------------------------------------

    @property
    def num_free(self) -> int:
        return len(self._free_slots)

    @property
    def num_used(self) -> int:
        return self._num_slots - self.num_free

    @property
    def num_slots(self) -> int:
        return self._num_slots

    @property
    def config(self) -> SWAPoolConfig:
        return self._config
