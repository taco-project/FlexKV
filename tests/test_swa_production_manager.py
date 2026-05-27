"""Tests for SWA Production Manager — production-ready SWA with endpoint hashing.

Tests the full lifecycle: put → get → evict → cascade, all without dependency
on Python RadixNode objects. Uses only token_ids and physical_block_ids.
"""
import threading
import time
from collections import OrderedDict

import numpy as np
import pytest

from flexkv.common.config import SWAPoolConfig
from flexkv.swa.swa_production_manager import SWAProductionManager


@pytest.fixture
def swa_config():
    return SWAPoolConfig(
        enabled=True,
        num_slots=8,
        window_size=4,
        num_swa_layers=2,
        bytes_per_token_per_layer=8,
        evict_ratio=0.25,  # evict 25% (= 2 slots) when full
        pin_memory=False,
    )


@pytest.fixture
def tokens_per_block():
    return 4


@pytest.fixture
def manager(swa_config, tokens_per_block):
    return SWAProductionManager(config=swa_config, tokens_per_block=tokens_per_block)


def _make_swa_data(config: SWAPoolConfig, value: int = 42) -> np.ndarray:
    """Create test SWA data of the correct size."""
    return np.full(config.slot_size_bytes, value, dtype=np.uint8)


def _make_token_ids(base: int, num_tokens: int = 8) -> np.ndarray:
    """Create a deterministic token_ids array."""
    return np.arange(base, base + num_tokens, dtype=np.int64)


def _make_block_ids(base: int, num_blocks: int = 2) -> np.ndarray:
    """Create physical block IDs."""
    return np.arange(base, base + num_blocks, dtype=np.int64)


class TestSWAProductionManagerBasicPut:
    """Test basic put operations."""

    def test_put_single(self, manager, swa_config):
        token_ids = _make_token_ids(100)
        data = _make_swa_data(swa_config, value=7)
        assert manager.put(token_ids, data) is True
        assert manager.num_entries == 1
        assert manager.pool.num_used == 1

    def test_put_with_block_ids(self, manager, swa_config):
        token_ids = _make_token_ids(100)
        block_ids = _make_block_ids(500, 2)
        data = _make_swa_data(swa_config, value=7)
        assert manager.put(token_ids, data, physical_block_ids=block_ids) is True
        assert manager.num_entries == 1

    def test_put_updates_existing(self, manager, swa_config):
        token_ids = _make_token_ids(100)
        data1 = _make_swa_data(swa_config, value=7)
        data2 = _make_swa_data(swa_config, value=99)

        manager.put(token_ids, data1)
        manager.put(token_ids, data2)

        # Should still be 1 entry
        assert manager.num_entries == 1
        assert manager.pool.num_used == 1

        # Data should be updated
        loaded = manager.get(token_ids)
        assert loaded is not None
        assert np.asarray(loaded)[0] == 99

    def test_put_multiple_distinct(self, manager, swa_config):
        for i in range(5):
            token_ids = _make_token_ids(i * 100)
            data = _make_swa_data(swa_config, value=i + 1)
            assert manager.put(token_ids, data) is True

        assert manager.num_entries == 5
        assert manager.pool.num_used == 5


class TestSWAProductionManagerGet:
    """Test get (retrieve) operations."""

    def test_get_after_put(self, manager, swa_config):
        token_ids = _make_token_ids(100)
        data = _make_swa_data(swa_config, value=42)
        manager.put(token_ids, data)

        result = manager.get(token_ids)
        assert result is not None
        assert np.asarray(result)[0] == 42

    def test_get_nonexistent_returns_none(self, manager):
        token_ids = _make_token_ids(999)
        result = manager.get(token_ids)
        assert result is None

    def test_get_returns_copy(self, manager, swa_config):
        """Verify get returns a copy (modifications don't affect stored data)."""
        token_ids = _make_token_ids(100)
        data = _make_swa_data(swa_config, value=42)
        manager.put(token_ids, data)

        result1 = manager.get(token_ids)
        np.asarray(result1)[0] = 99  # Modify the copy

        result2 = manager.get(token_ids)
        assert np.asarray(result2)[0] == 42  # Original should be unchanged

    def test_get_promotes_to_mru(self, manager, swa_config):
        """Verify that get promotes the entry to MRU position."""
        # Fill with 3 entries
        for i in range(3):
            token_ids = _make_token_ids(i * 100)
            manager.put(token_ids, _make_swa_data(swa_config, value=i))

        # Access the first entry (oldest/LRU) to promote it
        manager.get(_make_token_ids(0))

        # Now if we trigger eviction, the first entry should be protected (it's MRU)
        # The second entry (i=1) should be LRU now
        # Fill remaining slots + 1 to trigger eviction
        for i in range(3, 9):  # Fill 5 more (pool has 8 slots, 3 used → need 6 to overflow)
            token_ids = _make_token_ids(i * 100)
            manager.put(token_ids, _make_swa_data(swa_config, value=i))

        # The first entry (promoted to MRU) should still be available
        result = manager.get(_make_token_ids(0))
        assert result is not None
        assert np.asarray(result)[0] == 0


class TestSWAProductionManagerHas:
    """Test has (availability check) operations."""

    def test_has_true_after_put(self, manager, swa_config):
        token_ids = _make_token_ids(100)
        manager.put(token_ids, _make_swa_data(swa_config))
        assert manager.has(token_ids) is True

    def test_has_false_no_entry(self, manager):
        token_ids = _make_token_ids(999)
        assert manager.has(token_ids) is False

    def test_has_false_after_eviction(self, manager, swa_config):
        """After all slots used and oldest evicted, has() returns False for evicted entry."""
        # Fill all 8 slots
        for i in range(8):
            token_ids = _make_token_ids(i * 100)
            manager.put(token_ids, _make_swa_data(swa_config, value=i))

        # Add one more — triggers eviction of LRU entries
        extra_tokens = _make_token_ids(900)
        manager.put(extra_tokens, _make_swa_data(swa_config, value=99))

        # The oldest entry should have been evicted
        assert manager.has(_make_token_ids(0)) is False
        # The newest entry should be available
        assert manager.has(extra_tokens) is True


class TestSWAProductionManagerEviction:
    """Test LRU eviction behavior."""

    def test_eviction_on_full_pool(self, manager, swa_config):
        """Pool full → put triggers LRU eviction."""
        # Fill all 8 slots
        for i in range(8):
            token_ids = _make_token_ids(i * 100)
            manager.put(token_ids, _make_swa_data(swa_config, value=i))

        assert manager.pool.num_free == 0
        assert manager.num_entries == 8

        # 9th put should trigger eviction (evict_ratio=0.25 → evict 2)
        token_ids_9 = _make_token_ids(900)
        assert manager.put(token_ids_9, _make_swa_data(swa_config, value=9)) is True

        # At least 2 evicted (evict_ratio=0.25 of 8 = 2)
        assert manager.num_entries <= 7  # 8 - 2 + 1 = 7

    def test_eviction_skips_locked(self, manager, swa_config):
        """Locked entries are not evicted."""
        # Fill all 8 slots, lock the first one
        for i in range(8):
            token_ids = _make_token_ids(i * 100)
            manager.put(token_ids, _make_swa_data(swa_config, value=i))

        # Lock the first (LRU) entry
        manager.lock(_make_token_ids(0))

        # Trigger eviction
        token_ids_9 = _make_token_ids(900)
        manager.put(token_ids_9, _make_swa_data(swa_config, value=9))

        # The locked entry should still be available
        assert manager.has(_make_token_ids(0)) is True

    def test_eviction_all_locked_fails(self, swa_config, tokens_per_block):
        """When all entries are locked, put fails gracefully."""
        # Small pool: 2 slots
        small_config = SWAPoolConfig(
            enabled=True,
            num_slots=2,
            window_size=4,
            num_swa_layers=2,
            bytes_per_token_per_layer=8,
            evict_ratio=1.0,
            pin_memory=False,
        )
        mgr = SWAProductionManager(config=small_config, tokens_per_block=tokens_per_block)

        # Fill and lock both
        for i in range(2):
            token_ids = _make_token_ids(i * 100)
            mgr.put(token_ids, _make_swa_data(small_config, value=i))
            mgr.lock(token_ids)

        # 3rd put should fail
        token_ids_3 = _make_token_ids(300)
        assert mgr.put(token_ids_3, _make_swa_data(small_config, value=3)) is False


class TestSWAProductionManagerCascadeEviction:
    """Test cascade eviction via on_blocks_evicted()."""

    def test_cascade_eviction_basic(self, manager, swa_config):
        """When endpoint block is evicted from radix tree, SWA entry is invalidated."""
        token_ids = _make_token_ids(100)
        block_ids = _make_block_ids(500, 2)  # blocks [500, 501]
        manager.put(token_ids, _make_swa_data(swa_config, value=42),
                    physical_block_ids=block_ids)

        assert manager.has(token_ids) is True

        # Evict the endpoint block (last block = 501)
        evicted = manager.on_blocks_evicted(np.array([501], dtype=np.int64))
        assert evicted == 1
        assert manager.has(token_ids) is False
        assert manager.pool.num_free == 8  # Slot freed

    def test_cascade_eviction_non_endpoint_block(self, manager, swa_config):
        """Evicting a non-endpoint block does not invalidate the SWA entry."""
        token_ids = _make_token_ids(100)
        block_ids = _make_block_ids(500, 2)  # blocks [500, 501]
        manager.put(token_ids, _make_swa_data(swa_config, value=42),
                    physical_block_ids=block_ids)

        # Evict block 500 (not the endpoint)
        evicted = manager.on_blocks_evicted(np.array([500], dtype=np.int64))
        assert evicted == 0
        assert manager.has(token_ids) is True  # Still available

    def test_cascade_eviction_multiple_entries(self, manager, swa_config):
        """Multiple entries evicted in one call."""
        # Store 3 entries with different endpoint blocks
        for i in range(3):
            token_ids = _make_token_ids(i * 100)
            block_ids = _make_block_ids(i * 10, 2)  # endpoints: 1, 11, 21
            manager.put(token_ids, _make_swa_data(swa_config, value=i),
                        physical_block_ids=block_ids)

        # Evict endpoint blocks for entries 0 and 2
        evicted = manager.on_blocks_evicted(np.array([1, 21], dtype=np.int64))
        assert evicted == 2
        assert manager.has(_make_token_ids(0)) is False
        assert manager.has(_make_token_ids(100)) is True  # Entry 1 unaffected
        assert manager.has(_make_token_ids(200)) is False

    def test_cascade_eviction_locked_entry_skipped(self, manager, swa_config):
        """Locked entries are not invalidated even if their endpoint block is evicted."""
        token_ids = _make_token_ids(100)
        block_ids = _make_block_ids(500, 2)
        manager.put(token_ids, _make_swa_data(swa_config, value=42),
                    physical_block_ids=block_ids)
        manager.lock(token_ids)

        # Try to cascade-evict
        evicted = manager.on_blocks_evicted(np.array([501], dtype=np.int64))
        assert evicted == 0
        assert manager.has(token_ids) is True

    def test_cascade_eviction_empty_array(self, manager, swa_config):
        """Empty eviction array is a no-op."""
        token_ids = _make_token_ids(100)
        manager.put(token_ids, _make_swa_data(swa_config, value=42))

        evicted = manager.on_blocks_evicted(np.array([], dtype=np.int64))
        assert evicted == 0
        assert manager.has(token_ids) is True


class TestSWAProductionManagerLocking:
    """Test lock/unlock operations."""

    def test_lock_unlock_basic(self, manager, swa_config):
        token_ids = _make_token_ids(100)
        manager.put(token_ids, _make_swa_data(swa_config))

        manager.lock(token_ids)
        assert manager.stats["num_locked"] == 1

        manager.unlock(token_ids)
        assert manager.stats["num_locked"] == 0

    def test_lock_reentrant(self, manager, swa_config):
        """Multiple locks require multiple unlocks."""
        token_ids = _make_token_ids(100)
        manager.put(token_ids, _make_swa_data(swa_config))

        manager.lock(token_ids)
        manager.lock(token_ids)
        assert manager.stats["num_locked"] == 1  # Still one entry locked

        manager.unlock(token_ids)
        assert manager.stats["num_locked"] == 1  # Still locked (ref count > 0)

        manager.unlock(token_ids)
        assert manager.stats["num_locked"] == 0  # Now fully unlocked

    def test_lock_nonexistent_is_noop(self, manager):
        """Locking a non-existent entry is a no-op."""
        token_ids = _make_token_ids(999)
        manager.lock(token_ids)  # Should not raise
        assert manager.stats["num_locked"] == 0


class TestSWAProductionManagerStats:
    """Test statistics tracking."""

    def test_stats_puts(self, manager, swa_config):
        for i in range(3):
            manager.put(_make_token_ids(i * 100), _make_swa_data(swa_config, value=i))
        assert manager.stats["puts"] == 3

    def test_stats_hits_and_misses(self, manager, swa_config):
        token_ids = _make_token_ids(100)
        manager.put(token_ids, _make_swa_data(swa_config))

        # Hit
        manager.get(token_ids)
        assert manager.stats["hits"] == 1
        assert manager.stats["misses"] == 0

        # Miss
        manager.get(_make_token_ids(999))
        assert manager.stats["hits"] == 1
        assert manager.stats["misses"] == 1

    def test_stats_evictions(self, manager, swa_config):
        # Fill all 8 slots
        for i in range(8):
            manager.put(_make_token_ids(i * 100), _make_swa_data(swa_config, value=i))

        # Trigger eviction
        manager.put(_make_token_ids(900), _make_swa_data(swa_config, value=9))
        assert manager.stats["evictions"] >= 1

    def test_stats_cascade_evictions(self, manager, swa_config):
        token_ids = _make_token_ids(100)
        block_ids = _make_block_ids(500, 2)
        manager.put(token_ids, _make_swa_data(swa_config), physical_block_ids=block_ids)

        manager.on_blocks_evicted(np.array([501], dtype=np.int64))
        assert manager.stats["cascade_evictions"] == 1


class TestSWAProductionManagerEndpointHash:
    """Test endpoint hash computation behavior."""

    def test_same_tokens_same_hash(self, manager, swa_config):
        """Same token sequence should always hit the same entry."""
        token_ids = _make_token_ids(100)
        manager.put(token_ids, _make_swa_data(swa_config, value=42))

        # Same token_ids → same hash → same data
        result = manager.get(token_ids)
        assert result is not None
        assert np.asarray(result)[0] == 42

    def test_different_last_block_different_hash(self, manager, swa_config, tokens_per_block):
        """Different last blocks should produce different hashes."""
        # Two sequences that share a prefix but differ in the last block
        tokens_a = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        tokens_b = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=np.int64)  # Last token differs

        manager.put(tokens_a, _make_swa_data(swa_config, value=10))
        manager.put(tokens_b, _make_swa_data(swa_config, value=20))

        assert manager.num_entries == 2
        result_a = manager.get(tokens_a)
        result_b = manager.get(tokens_b)
        assert np.asarray(result_a)[0] == 10
        assert np.asarray(result_b)[0] == 20

    def test_short_sequence_supported(self, manager, swa_config, tokens_per_block):
        """Sequences shorter than tokens_per_block still work."""
        short_tokens = np.array([1, 2], dtype=np.int64)  # Less than tokens_per_block=4
        manager.put(short_tokens, _make_swa_data(swa_config, value=55))
        result = manager.get(short_tokens)
        assert result is not None
        assert np.asarray(result)[0] == 55


class TestSWAProductionManagerThreadSafety:
    """Test thread safety of the production manager."""

    def test_concurrent_puts(self, swa_config, tokens_per_block):
        """Multiple threads putting concurrently should not corrupt state."""
        mgr = SWAProductionManager(config=swa_config, tokens_per_block=tokens_per_block)
        num_threads = 4
        puts_per_thread = 50
        errors = []

        def worker(thread_id):
            try:
                for i in range(puts_per_thread):
                    token_ids = _make_token_ids(thread_id * 1000 + i * 10)
                    data = _make_swa_data(swa_config, value=(thread_id * 10 + i) % 256)
                    mgr.put(token_ids, data)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        # Total entries should be <= num_threads * puts_per_thread (some may have been evicted)
        assert mgr.num_entries <= num_threads * puts_per_thread
        assert mgr.num_entries > 0

    def test_concurrent_put_get(self, swa_config, tokens_per_block):
        """Concurrent puts and gets should not crash."""
        mgr = SWAProductionManager(config=swa_config, tokens_per_block=tokens_per_block)
        errors = []

        # Pre-populate some entries
        for i in range(4):
            token_ids = _make_token_ids(i * 100)
            mgr.put(token_ids, _make_swa_data(swa_config, value=i))

        def writer():
            try:
                for i in range(50):
                    token_ids = _make_token_ids(1000 + i * 10)
                    mgr.put(token_ids, _make_swa_data(swa_config, value=i % 256))
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for i in range(50):
                    token_ids = _make_token_ids((i % 4) * 100)
                    mgr.get(token_ids)  # May return None if evicted — that's fine
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestSWAProductionManagerIntegration:
    """Integration-level tests simulating the production flow."""

    def test_full_lifecycle(self, manager, swa_config):
        """Simulate: put → get → evict → cascade."""
        # 1. Put
        token_ids = _make_token_ids(100)
        block_ids = _make_block_ids(500, 2)
        data = _make_swa_data(swa_config, value=77)
        assert manager.put(token_ids, data, physical_block_ids=block_ids) is True

        # 2. Get
        loaded = manager.get(token_ids)
        assert loaded is not None
        assert np.asarray(loaded)[0] == 77

        # 3. has
        assert manager.has(token_ids) is True

        # 4. Cascade eviction
        evicted = manager.on_blocks_evicted(np.array([501], dtype=np.int64))
        assert evicted == 1

        # 5. Verify gone
        assert manager.has(token_ids) is False
        assert manager.get(token_ids) is None

    def test_serve_cycle_simulation(self, manager, swa_config):
        """Simulate a serving cycle: req1 finishes → req2 arrives with shared prefix."""
        # Request 1 finishes: store main KV + SWA
        req1_tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int64)
        req1_blocks = np.array([100, 101, 102], dtype=np.int64)
        swa_data = _make_swa_data(swa_config, value=42)
        assert manager.put(req1_tokens, swa_data, physical_block_ids=req1_blocks) is True

        # Request 2 arrives: shares prefix (same last block of the matched part)
        # The matched prefix would be [1,2,3,4,5,6,7,8] (first 8 tokens)
        # But the SWA key is based on the FULL sequence that was stored
        # So req2 would need to look up by req1's full token_ids
        assert manager.has(req1_tokens) is True
        loaded = manager.get(req1_tokens)
        assert loaded is not None
        assert np.asarray(loaded)[0] == 42

    def test_block_id_update_on_reput(self, manager, swa_config):
        """Re-putting with different block IDs updates the block tracking."""
        token_ids = _make_token_ids(100)
        block_ids_v1 = _make_block_ids(500, 2)  # endpoint = 501
        block_ids_v2 = _make_block_ids(700, 2)  # endpoint = 701

        manager.put(token_ids, _make_swa_data(swa_config, value=1),
                    physical_block_ids=block_ids_v1)

        # Update with new block IDs
        manager.put(token_ids, _make_swa_data(swa_config, value=2),
                    physical_block_ids=block_ids_v2)

        # Old endpoint block eviction should NOT affect (old mapping removed)
        evicted = manager.on_blocks_evicted(np.array([501], dtype=np.int64))
        assert evicted == 0
        assert manager.has(token_ids) is True

        # New endpoint block eviction SHOULD affect
        evicted = manager.on_blocks_evicted(np.array([701], dtype=np.int64))
        assert evicted == 1
        assert manager.has(token_ids) is False

    def test_torch_tensor_input(self, manager, swa_config):
        """Verify torch.Tensor inputs are handled correctly."""
        try:
            import torch
        except ImportError:
            pytest.skip("torch not available")

        token_ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8], dtype=torch.int64)
        swa_data = torch.full((swa_config.slot_size_bytes,), 55, dtype=torch.uint8)

        assert manager.put(token_ids, swa_data) is True
        assert manager.has(token_ids) is True

        loaded = manager.get(token_ids)
        assert loaded is not None
        assert np.asarray(loaded)[0] == 55


class TestSWAProductionManagerLoadBackFlow:
    """Tests simulating the FlexKV connector's SWA load-back path.

    These tests verify the production manager operations used by the connector
    during the get_new_hit_length → start_load_kv flow:
    1. has() for backward search (checking SWA availability at different prefix lengths)
    2. get() for loading SWA data to restore to GPU
    """

    def test_backward_search_finds_available_prefix(self, manager, swa_config, tokens_per_block):
        """Simulate backward search: full prefix has no SWA, shorter prefix does.

        Connector logic: If swa_available(full_match) is False, try progressively
        shorter prefixes (page-aligned) until one has SWA available.
        """
        page_size = tokens_per_block  # page_size == tokens_per_block == 4

        # Store SWA for a shorter prefix (8 tokens = 2 pages)
        short_prefix = _make_token_ids(100, num_tokens=8)
        manager.put(short_prefix, _make_swa_data(swa_config, value=42))

        # Full prefix is 16 tokens (4 pages) — SWA not stored for this
        full_prefix = _make_token_ids(100, num_tokens=16)
        assert manager.has(full_prefix) is False

        # Backward search: try hit_length - page_size each step
        hit_length = 16
        found_hit_length = 0
        while hit_length > 0:
            hit_length -= page_size
            if hit_length <= 0:
                break
            prefix = _make_token_ids(100, num_tokens=hit_length)
            if manager.has(prefix):
                found_hit_length = hit_length
                break

        # Should find at hit_length=8 (the shorter prefix we stored)
        assert found_hit_length == 8

        # Verify we can get the data for the found prefix
        loaded = manager.get(_make_token_ids(100, num_tokens=8))
        assert loaded is not None
        assert np.asarray(loaded)[0] == 42

    def test_backward_search_no_swa_returns_zero(self, manager, swa_config, tokens_per_block):
        """When no prefix has SWA available, backward search returns hit_length=0."""
        page_size = tokens_per_block

        # Don't store any SWA data
        full_prefix = _make_token_ids(200, num_tokens=16)

        # Backward search
        hit_length = 16
        found_hit_length = 0
        while hit_length > 0:
            hit_length -= page_size
            if hit_length <= 0:
                break
            prefix = _make_token_ids(200, num_tokens=hit_length)
            if manager.has(prefix):
                found_hit_length = hit_length
                break

        assert found_hit_length == 0

    def test_swa_available_at_full_hit(self, manager, swa_config):
        """When SWA is available at full hit_length, no backward search needed."""
        full_prefix = _make_token_ids(300, num_tokens=12)
        manager.put(full_prefix, _make_swa_data(swa_config, value=77))

        # SWA available at full hit → no reduction needed
        assert manager.has(full_prefix) is True

        # get() returns the data for load-back
        loaded = manager.get(full_prefix)
        assert loaded is not None
        assert np.asarray(loaded)[0] == 77

    def test_load_back_data_integrity(self, manager, swa_config):
        """Verify data integrity through the store → check → load cycle.

        Simulates the full connector flow:
        1. start_store_kv: swa_put (store)
        2. get_new_hit_length: swa_available (check)
        3. start_load_kv: swa_get (load for GPU restore)
        """
        token_ids = _make_token_ids(400, num_tokens=8)
        # Simulate distinct data per-byte to verify no corruption
        data = np.arange(swa_config.slot_size_bytes, dtype=np.uint8)
        manager.put(token_ids, data)

        # Check availability (connector's get_new_hit_length)
        assert manager.has(token_ids) is True

        # Load data (connector's start_load_kv → _do_swa_restore_for_op)
        loaded = manager.get(token_ids)
        assert loaded is not None
        loaded_arr = np.asarray(loaded)
        assert np.array_equal(loaded_arr, data)

    def test_load_back_after_eviction_returns_none(self, manager, swa_config):
        """If SWA data was evicted between check and load, get returns None gracefully."""
        token_ids = _make_token_ids(500, num_tokens=8)
        block_ids = _make_block_ids(900, 2)
        manager.put(token_ids, _make_swa_data(swa_config, value=33),
                    physical_block_ids=block_ids)

        # Check: available
        assert manager.has(token_ids) is True

        # Cascade eviction happens between check and load
        manager.on_blocks_evicted(np.array([901], dtype=np.int64))

        # Load: should return None (evicted)
        loaded = manager.get(token_ids)
        assert loaded is None

    def test_multiple_requests_independent_swa_state(self, manager, swa_config):
        """Multiple concurrent requests have independent SWA load state."""
        # Request A
        tokens_a = _make_token_ids(600, num_tokens=8)
        data_a = _make_swa_data(swa_config, value=11)
        manager.put(tokens_a, data_a)

        # Request B
        tokens_b = _make_token_ids(700, num_tokens=8)
        data_b = _make_swa_data(swa_config, value=22)
        manager.put(tokens_b, data_b)

        # Both available independently
        assert manager.has(tokens_a) is True
        assert manager.has(tokens_b) is True

        # Load each independently (simulates start_load_kv for different ops)
        loaded_a = manager.get(tokens_a)
        loaded_b = manager.get(tokens_b)
        assert np.asarray(loaded_a)[0] == 11
        assert np.asarray(loaded_b)[0] == 22

    def test_backward_search_page_alignment(self, swa_config):
        """Backward search respects page_size alignment."""
        # Use larger page_size to verify alignment
        page_size = 4
        mgr = SWAProductionManager(config=swa_config, tokens_per_block=page_size)

        # Store SWA for exactly 2 pages (8 tokens)
        short_prefix = _make_token_ids(800, num_tokens=8)
        mgr.put(short_prefix, _make_swa_data(swa_config, value=55))

        # Full hit is 5 pages (20 tokens), no SWA
        # Backward search: 20→16→12→8 (found!)
        hit_length = 20
        steps = 0
        found_hit_length = 0
        while hit_length > 0:
            hit_length -= page_size
            steps += 1
            if hit_length <= 0:
                break
            prefix = _make_token_ids(800, num_tokens=hit_length)
            if mgr.has(prefix):
                found_hit_length = hit_length
                break

        assert found_hit_length == 8
        assert steps == 3  # 20→16, 16→12, 12→8
