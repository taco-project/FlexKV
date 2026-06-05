"""Tests for the node-attached SWAProductionManager (Step 3).

Uses a lightweight FakeNode that mimics the CRadixNode SWA interface
(swa_host_slot / swa_tombstone / swa_lock_ref + inc/dec). No C extension or
radix tree is needed, so these run fast and isolate the manager's logic:
put (overwrite vs allocate), SWA-only eviction (tombstoning), lock skipping,
stale-entry handling, and cascade free_slots.
"""
import sys
from unittest.mock import MagicMock

# Mock the C extension before importing flexkv.swa
if 'flexkv.c_ext' not in sys.modules:
    sys.modules['flexkv.c_ext'] = MagicMock()

import numpy as np
import pytest

from flexkv.common.config import SWAPoolConfig
from flexkv.swa.swa_production_manager import SWAProductionManager


class FakeNode:
    """Mimics the CRadixNode SWA interface used by SWAProductionManager."""

    def __init__(self):
        self.swa_host_slot = -1
        self.swa_tombstone = True
        self.swa_lock_ref = 0

    def inc_swa_lock_ref(self):
        self.swa_lock_ref += 1

    def dec_swa_lock_ref(self):
        assert self.swa_lock_ref > 0
        self.swa_lock_ref -= 1


@pytest.fixture
def swa_config():
    return SWAPoolConfig(
        enabled=True,
        num_slots=4,
        window_size=4,
        num_swa_layers=2,
        bytes_per_token_per_layer=8,
        evict_ratio=0.25,
        pin_memory=False,
    )


@pytest.fixture
def manager(swa_config):
    return SWAProductionManager(config=swa_config, tokens_per_block=4)


def _data(config, value=42):
    return np.full(config.slot_size_bytes, value, dtype=np.uint8)


# --------------------------------------------------------------------------- #
# put / get / has                                                             #
# --------------------------------------------------------------------------- #

class TestPutGetHas:
    def test_put_sets_node_state(self, manager, swa_config):
        n = FakeNode()
        assert manager.put(n, _data(swa_config)) is True
        assert n.swa_host_slot != -1
        assert n.swa_tombstone is False
        assert manager.has(n) is True

    def test_get_after_put(self, manager, swa_config):
        n = FakeNode()
        manager.put(n, _data(swa_config, value=123))
        out = manager.get(n)
        assert out is not None
        assert int(out[0]) == 123

    def test_get_miss_returns_none(self, manager):
        assert manager.get(FakeNode()) is None
        assert manager.get(None) is None

    def test_has_false_for_tombstone(self, manager, swa_config):
        n = FakeNode()
        manager.put(n, _data(swa_config))
        # simulate tombstone
        n.swa_tombstone = True
        assert manager.has(n) is False

    def test_put_none_node(self, manager, swa_config):
        assert manager.put(None, _data(swa_config)) is False


class TestPutCaseAReuse:
    def test_overwrite_reuses_slot(self, manager, swa_config):
        n = FakeNode()
        manager.put(n, _data(swa_config, value=1))
        slot = n.swa_host_slot
        used_after_first = manager.pool.num_used
        manager.put(n, _data(swa_config, value=2))
        # same slot, no new allocation
        assert n.swa_host_slot == slot
        assert manager.pool.num_used == used_after_first
        out = manager.get(n)
        assert int(out[0]) == 2

    def test_overwrite_clears_tombstone(self, manager, swa_config):
        n = FakeNode()
        manager.put(n, _data(swa_config))
        n.swa_tombstone = True  # pretend it got tombstoned
        # re-put on same node still holding slot -> revive
        manager.put(n, _data(swa_config))
        assert n.swa_tombstone is False


# --------------------------------------------------------------------------- #
# SWA-only eviction (tombstoning)                                             #
# --------------------------------------------------------------------------- #

class TestSwaOnlyEviction:
    def test_pool_full_tombstones_lru_victim(self, manager, swa_config):
        nodes = [FakeNode() for _ in range(swa_config.num_slots)]
        for n in nodes:
            assert manager.put(n, _data(swa_config)) is True
        assert manager.pool.num_free == 0

        victim = nodes[0]  # LRU end (oldest put, never touched)
        newn = FakeNode()
        assert manager.put(newn, _data(swa_config)) is True

        # victim got tombstoned, but the node object itself survives
        assert victim.swa_tombstone is True
        assert victim.swa_host_slot == -1
        # new node took the freed slot
        assert newn.swa_host_slot != -1
        assert newn.swa_tombstone is False

    def test_get_promotes_mru_changes_victim(self, manager, swa_config):
        nodes = [FakeNode() for _ in range(swa_config.num_slots)]
        for n in nodes:
            manager.put(n, _data(swa_config))
        # touch nodes[0] so it is no longer LRU
        manager.get(nodes[0])
        newn = FakeNode()
        manager.put(newn, _data(swa_config))
        # nodes[0] should survive; nodes[1] (now LRU) should be the victim
        assert nodes[0].swa_tombstone is False
        assert nodes[1].swa_tombstone is True

    def test_all_locked_put_fails(self, manager, swa_config):
        nodes = [FakeNode() for _ in range(swa_config.num_slots)]
        for n in nodes:
            manager.put(n, _data(swa_config))
            n.inc_swa_lock_ref()  # lock every entry
        newn = FakeNode()
        assert manager.put(newn, _data(swa_config)) is False
        assert newn.swa_host_slot == -1

    def test_locked_victim_skipped(self, manager, swa_config):
        nodes = [FakeNode() for _ in range(swa_config.num_slots)]
        for n in nodes:
            manager.put(n, _data(swa_config))
        nodes[0].inc_swa_lock_ref()  # lock the LRU one
        newn = FakeNode()
        assert manager.put(newn, _data(swa_config)) is True
        # locked node survives; next unlocked LRU (nodes[1]) is evicted
        assert nodes[0].swa_tombstone is False
        assert nodes[1].swa_tombstone is True


# --------------------------------------------------------------------------- #
# Stale-entry handling + cascade free_slots                                   #
# --------------------------------------------------------------------------- #

class TestStaleAndCascade:
    def test_stale_lru_entry_skipped(self, manager, swa_config):
        nodes = [FakeNode() for _ in range(swa_config.num_slots)]
        for n in nodes:
            manager.put(n, _data(swa_config))
        # Simulate a cascade that already released nodes[0]'s slot to the pool:
        # node.swa_host_slot cleared, slot returned, but LRU still has a stale entry.
        freed_slot = nodes[0].swa_host_slot
        nodes[0].swa_host_slot = -1
        nodes[0].swa_tombstone = True
        manager.free_slots([freed_slot])
        # Now a put should consume the cascade-freed slot via allocate (no need
        # to evict), and the stale LRU entry must not cause a double free / crash.
        newn = FakeNode()
        assert manager.put(newn, _data(swa_config)) is True
        assert newn.swa_host_slot != -1

    def test_free_slots_returns_to_pool(self, manager, swa_config):
        n = FakeNode()
        manager.put(n, _data(swa_config))
        slot = n.swa_host_slot
        used_before = manager.pool.num_used
        # mimic C++ cascade clearing the node, then engine returning the slot
        n.swa_host_slot = -1
        freed = manager.free_slots([slot])
        assert freed == 1
        assert manager.pool.num_used == used_before - 1

    def test_free_slots_ignores_negative(self, manager):
        assert manager.free_slots([-1, None]) == 0


# --------------------------------------------------------------------------- #
# lock / unlock                                                               #
# --------------------------------------------------------------------------- #

class TestLockUnlock:
    def test_lock_unlock(self, manager, swa_config):
        n = FakeNode()
        manager.put(n, _data(swa_config))
        manager.lock(n)
        assert n.swa_lock_ref == 1
        manager.unlock(n)
        assert n.swa_lock_ref == 0

    def test_lock_noop_without_slot(self, manager):
        n = FakeNode()
        manager.lock(n)  # no slot -> no lock
        assert n.swa_lock_ref == 0
