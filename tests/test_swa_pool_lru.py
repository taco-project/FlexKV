"""Tests for SWA Pool LRU — doubly-linked eviction list."""
import sys
from unittest.mock import MagicMock

# Mock the C extension before importing flexkv.cache
if 'flexkv.c_ext' not in sys.modules:
    sys.modules['flexkv.c_ext'] = MagicMock()

import numpy as np
import pytest

from flexkv.cache.radixtree import RadixNode
from flexkv.swa.swa_pool_lru import SWAPoolLRU


def _make_node(slot_id=0, lock_ref=0):
    """Create a minimal RadixNode for LRU testing."""
    node = RadixNode(
        block_hashes=np.array([slot_id], dtype=np.int64),
        physical_blocks=np.array([slot_id], dtype=np.int64),
        is_ready=True,
        lock_cnt=0,
        grace_time=0.0,
        swa_tombstone=False,
        swa_host_slot=slot_id,
        swa_lock_ref=lock_ref,
    )
    return node


class TestSWAPoolLRU:
    def test_empty(self):
        lru = SWAPoolLRU()
        assert len(lru) == 0
        assert not lru
        assert lru.get_lru_evictable() is None

    def test_insert_one(self):
        lru = SWAPoolLRU()
        node = _make_node(0)
        lru.insert_mru(node)
        assert len(lru) == 1
        assert lru.get_lru_evictable() is node

    def test_insert_order(self):
        lru = SWAPoolLRU()
        n1 = _make_node(1)
        n2 = _make_node(2)
        n3 = _make_node(3)
        lru.insert_mru(n1)
        lru.insert_mru(n2)
        lru.insert_mru(n3)
        # LRU should be n1 (inserted first = oldest)
        assert lru.get_lru_evictable() is n1

    def test_remove(self):
        lru = SWAPoolLRU()
        n1 = _make_node(1)
        n2 = _make_node(2)
        lru.insert_mru(n1)
        lru.insert_mru(n2)
        lru.remove(n1)
        assert len(lru) == 1
        assert lru.get_lru_evictable() is n2

    def test_remove_idempotent(self):
        lru = SWAPoolLRU()
        node = _make_node(0)
        lru.insert_mru(node)
        lru.remove(node)
        lru.remove(node)  # Should not crash
        assert len(lru) == 0

    def test_promote_mru(self):
        lru = SWAPoolLRU()
        n1 = _make_node(1)
        n2 = _make_node(2)
        n3 = _make_node(3)
        lru.insert_mru(n1)
        lru.insert_mru(n2)
        lru.insert_mru(n3)
        # Promote n1 (currently LRU) to MRU
        lru.promote_mru(n1)
        # Now LRU should be n2
        assert lru.get_lru_evictable() is n2

    def test_skip_locked(self):
        lru = SWAPoolLRU()
        n1 = _make_node(1, lock_ref=1)  # locked
        n2 = _make_node(2, lock_ref=0)  # unlocked
        lru.insert_mru(n1)
        lru.insert_mru(n2)
        # n1 is LRU but locked, should skip to n2
        assert lru.get_lru_evictable() is n2

    def test_all_locked(self):
        lru = SWAPoolLRU()
        n1 = _make_node(1, lock_ref=1)
        n2 = _make_node(2, lock_ref=1)
        lru.insert_mru(n1)
        lru.insert_mru(n2)
        assert lru.get_lru_evictable() is None
