"""Tests for SWA Radix Manager — full lifecycle integration."""
import sys
from unittest.mock import MagicMock

# Mock the C extension before importing flexkv.cache
if 'flexkv.c_ext' not in sys.modules:
    sys.modules['flexkv.c_ext'] = MagicMock()

import numpy as np
import pytest
import time

from flexkv.common.config import SWAPoolConfig
from flexkv.cache.radixtree import RadixNode
from flexkv.swa.swa_radix_manager import SWARadixManager


@pytest.fixture
def swa_config():
    return SWAPoolConfig(
        enabled=True,
        num_slots=4,
        window_size=4,
        num_swa_layers=2,
        bytes_per_token_per_layer=8,
        evict_ratio=0.5,  # evict 50% when full
        pin_memory=False,
    )


@pytest.fixture
def manager(swa_config):
    return SWARadixManager(swa_config, tokens_per_block=4)


def _make_tree_node(block_id, parent=None):
    """Create a RadixNode with given block id."""
    node = RadixNode(
        block_hashes=np.array([block_id], dtype=np.int64),
        physical_blocks=np.array([block_id], dtype=np.int64),
        is_ready=True,
        lock_cnt=0,
        grace_time=time.time(),
    )
    node.parent = parent
    return node


def _make_data(config, value=42):
    """Create test SWA data of correct size."""
    return np.full(config.slot_size_bytes, value, dtype=np.uint8)


class TestSWAPut:
    def test_basic_put(self, manager, swa_config):
        node = _make_tree_node(1)
        data = _make_data(swa_config, value=7)
        assert manager.swa_put(node, data)
        assert node.swa_host_slot is not None
        assert node.swa_tombstone is False
        assert manager.pool.num_used == 1

    def test_put_updates_existing(self, manager, swa_config):
        node = _make_tree_node(1)
        data1 = _make_data(swa_config, value=7)
        data2 = _make_data(swa_config, value=99)
        manager.swa_put(node, data1)
        slot = node.swa_host_slot
        manager.swa_put(node, data2)
        # Same slot, updated data
        assert node.swa_host_slot == slot
        result = np.asarray(manager.pool.read(slot))
        assert result[0] == 99

    def test_put_pool_full_triggers_eviction(self, manager, swa_config):
        # Fill all 4 slots
        nodes = []
        for i in range(4):
            n = _make_tree_node(i)
            manager.swa_put(n, _make_data(swa_config, value=i))
            nodes.append(n)

        # 5th put should trigger eviction
        new_node = _make_tree_node(99)
        assert manager.swa_put(new_node, _make_data(swa_config, value=99))
        # Oldest nodes should be tombstoned (evict_ratio=0.5 -> evict 2)
        tombstoned = sum(1 for n in nodes if n.swa_tombstone)
        assert tombstoned >= 1  # At least 1 evicted

    def test_put_all_locked_fails(self, manager, swa_config):
        # Fill pool with locked nodes
        for i in range(4):
            n = _make_tree_node(i)
            n.swa_lock_ref = 1
            manager.swa_put(n, _make_data(swa_config, value=i))

        # New put should fail (can't evict locked)
        new_node = _make_tree_node(99)
        assert not manager.swa_put(new_node, _make_data(swa_config, value=99))


class TestSWALoadBack:
    def test_load_back_success(self, manager, swa_config):
        node = _make_tree_node(1)
        data = _make_data(swa_config, value=42)
        manager.swa_put(node, data)

        result = manager.swa_load_back(node)
        assert result is not None
        assert np.asarray(result)[0] == 42

    def test_load_back_tombstone_returns_none(self, manager):
        node = _make_tree_node(1)
        # Node is tombstone by default
        assert node.swa_tombstone is True
        result = manager.swa_load_back(node)
        assert result is None

    def test_load_back_promotes_lru(self, manager, swa_config):
        n1 = _make_tree_node(1)
        n2 = _make_tree_node(2)
        manager.swa_put(n1, _make_data(swa_config, value=1))
        manager.swa_put(n2, _make_data(swa_config, value=2))
        # n1 is LRU, n2 is MRU
        # Load back n1 -> should promote to MRU
        manager.swa_load_back(n1)
        # Now n2 is LRU
        assert manager.lru.get_lru_evictable() is n2


class TestSWAEviction:
    def test_evict_releases_slot(self, manager, swa_config):
        node = _make_tree_node(1)
        manager.swa_put(node, _make_data(swa_config))
        assert manager.pool.num_used == 1

        evicted = manager.swa_evict_for_space(1)
        assert evicted == 1
        assert manager.pool.num_used == 0
        assert node.swa_tombstone is True
        assert node.swa_host_slot is None

    def test_evict_skips_locked(self, manager, swa_config):
        n1 = _make_tree_node(1)
        n1.swa_lock_ref = 1
        n2 = _make_tree_node(2)
        manager.swa_put(n1, _make_data(swa_config))
        manager.swa_put(n2, _make_data(swa_config))

        evicted = manager.swa_evict_for_space(1)
        assert evicted == 1
        # n1 is locked, so n2 gets evicted
        assert n1.swa_tombstone is False
        assert n2.swa_tombstone is True


class TestOnLeafEvict:
    def test_cascade_release(self, manager, swa_config):
        node = _make_tree_node(1)
        manager.swa_put(node, _make_data(swa_config))
        assert len(manager.lru) == 1

        manager.on_leaf_evict(node)
        assert node.swa_host_slot is None
        assert node.swa_tombstone is True
        assert len(manager.lru) == 0
        assert manager.pool.num_used == 0

    def test_cascade_noop_if_no_swa(self, manager):
        node = _make_tree_node(1)
        # No SWA data, should be a no-op
        manager.on_leaf_evict(node)
        assert node.swa_tombstone is True


class TestCheckSWATrailing:
    def test_all_available(self, manager, swa_config):
        # Build path: root -> n1 -> n2 (leaf)
        root = _make_tree_node(0)
        root.parent = None  # make it root-like
        n1 = _make_tree_node(1, parent=root)
        n2 = _make_tree_node(2, parent=n1)

        # Put SWA on both
        manager.swa_put(n1, _make_data(swa_config))
        manager.swa_put(n2, _make_data(swa_config))

        # Each node has 1 block x 4 tokens_per_block = 4 tokens
        # Total trailing available = 8 tokens from n2+n1
        # window = 4 tokens -> should pass
        path = [root, n1, n2]
        assert manager.check_swa_trailing(path, window_tokens=4)

    def test_tombstone_breaks_trailing(self, manager, swa_config):
        root = _make_tree_node(0)
        root.parent = None
        n1 = _make_tree_node(1, parent=root)
        n2 = _make_tree_node(2, parent=n1)

        # Only n2 has SWA, n1 is tombstone
        manager.swa_put(n2, _make_data(swa_config))
        # n1 remains tombstone (default)

        # n2 provides 4 tokens, but window=8 needs more -> n1 is tombstone -> fail
        path = [root, n1, n2]
        assert not manager.check_swa_trailing(path, window_tokens=8)

    def test_leaf_alone_sufficient(self, manager, swa_config):
        root = _make_tree_node(0)
        root.parent = None
        n1 = _make_tree_node(1, parent=root)

        manager.swa_put(n1, _make_data(swa_config))
        # n1 has 4 tokens, window=4 -> pass
        path = [root, n1]
        assert manager.check_swa_trailing(path, window_tokens=4)


class TestSWALock:
    def test_lock_unlock(self, manager, swa_config):
        root = _make_tree_node(0)
        root.parent = None
        n1 = _make_tree_node(1, parent=root)
        manager.swa_put(n1, _make_data(swa_config))

        manager.swa_lock(n1, window_tokens=4)
        assert n1.swa_lock_ref == 1

        manager.swa_unlock(n1, window_tokens=4)
        assert n1.swa_lock_ref == 0

    def test_lock_protects_from_eviction(self, manager, swa_config):
        n1 = _make_tree_node(1)
        n1.parent = RadixNode(
            block_hashes=np.array([], dtype=np.int64),
            physical_blocks=np.array([], dtype=np.int64),
            is_ready=True, lock_cnt=0, grace_time=0.0,
        )  # Fake root
        manager.swa_put(n1, _make_data(swa_config))
        manager.swa_lock(n1, window_tokens=4)

        # Try to evict — should fail (locked)
        evicted = manager.swa_evict_for_space(1)
        assert evicted == 0
        assert n1.swa_tombstone is False


class TestStats:
    def test_stats_tracking(self, manager, swa_config):
        node = _make_tree_node(1)
        manager.swa_put(node, _make_data(swa_config))
        manager.swa_load_back(node)

        stats = manager.stats
        assert stats["puts"] == 1
        assert stats["hits"] == 1
        assert stats["pool_used"] == 1
