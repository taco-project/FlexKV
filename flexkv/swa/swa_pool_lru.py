"""SWA Pool LRU — Doubly-linked list for SWA eviction ordering.

Tracks RadixNodes that have SWA data in the CPU host pool.
Uses intrusive pointers (_swa_lru_prev, _swa_lru_next) on RadixNode to avoid
extra allocation. Head = MRU (most recently used), Tail = LRU (least recently used).
"""
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from flexkv.cache.radixtree import RadixNode


class _SentinelNode:
    """Lightweight sentinel for list head/tail (avoids importing RadixNode)."""
    __slots__ = ('_swa_lru_prev', '_swa_lru_next')

    def __init__(self):
        self._swa_lru_prev = None
        self._swa_lru_next = None


class SWAPoolLRU:
    """Intrusive doubly-linked LRU list for SWA pool eviction.

    Invariants:
    - Only nodes with swa_host_slot is not None are in the list.
    - Head sentinel's _next points to MRU node.
    - Tail sentinel's _prev points to LRU node.
    - Nodes with swa_lock_ref > 0 are skipped during eviction scan.
    """

    def __init__(self):
        self._head = _SentinelNode()  # MRU side
        self._tail = _SentinelNode()  # LRU side
        self._head._swa_lru_next = self._tail
        self._tail._swa_lru_prev = self._head
        self._size = 0

    def insert_mru(self, node: 'RadixNode') -> None:
        """Insert node at MRU position (head)."""
        assert node._swa_lru_prev is None and node._swa_lru_next is None, \
            "Node already in LRU list"
        after = self._head
        before = self._head._swa_lru_next
        node._swa_lru_prev = after
        node._swa_lru_next = before
        after._swa_lru_next = node
        before._swa_lru_prev = node
        self._size += 1

    def remove(self, node: 'RadixNode') -> None:
        """Remove node from the list."""
        if node._swa_lru_prev is None and node._swa_lru_next is None:
            return  # Not in list (idempotent)
        prev_node = node._swa_lru_prev
        next_node = node._swa_lru_next
        prev_node._swa_lru_next = next_node
        next_node._swa_lru_prev = prev_node
        node._swa_lru_prev = None
        node._swa_lru_next = None
        self._size -= 1

    def promote_mru(self, node: 'RadixNode') -> None:
        """Move an existing node to MRU position."""
        self.remove(node)
        self.insert_mru(node)

    def get_lru_evictable(self) -> Optional['RadixNode']:
        """Get the LRU node that can be evicted (swa_lock_ref == 0).

        Scans from tail (LRU) toward head (MRU), skipping locked nodes.
        Returns None if all nodes are locked.
        """
        current = self._tail._swa_lru_prev
        while current is not self._head:
            # current is a RadixNode (not sentinel)
            if current.swa_lock_ref == 0:
                return current
            current = current._swa_lru_prev
        return None

    def __len__(self) -> int:
        return self._size

    def __bool__(self) -> bool:
        return self._size > 0
