# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for SWA (Sliding Window Attention) pool support in FlexKV."""

import time

import numpy as np
import pytest

from flexkv.common.config import SWAPoolConfig
from flexkv.swa.swa_cache_engine import SWACacheEngine, SWAMatchResult, SWAEntry
from flexkv.swa.swa_storage import SWAStorage, SWAStorageConfig

# HashType is just int (NewType alias). Use plain int in tests.
HashType = int


# ===========================================================================
# SWAPoolConfig Tests
# ===========================================================================

class TestSWAPoolConfig:
    def test_default_config(self):
        cfg = SWAPoolConfig()
        assert cfg.enabled is False
        assert cfg.window_size == 128
        assert cfg.num_swa_layers == 61
        assert cfg.bytes_per_token_per_layer == 584
        assert cfg.num_slots == 1000
        assert cfg.evict_ratio == 0.1

    def test_page_size_calculation(self):
        cfg = SWAPoolConfig(
            window_size=128, bytes_per_token_per_layer=584, num_swa_layers=61
        )
        expected = 128 * 584 * 61  # = 4,553,472 bytes (~4.3 MB)
        assert cfg.page_size_bytes == expected

    def test_custom_config(self):
        cfg = SWAPoolConfig(
            enabled=True, window_size=64, num_slots=500, num_swa_layers=30
        )
        assert cfg.enabled is True
        assert cfg.window_size == 64
        assert cfg.num_slots == 500
        assert cfg.num_swa_layers == 30
        assert cfg.page_size_bytes == 64 * 584 * 30


# ===========================================================================
# SWACacheEngine Tests
# ===========================================================================

class TestSWACacheEngine:
    @pytest.fixture
    def engine(self):
        return SWACacheEngine(num_slots=16, evict_ratio=0.25)

    def test_init_valid(self, engine):
        assert engine.num_slots == 16
        assert engine.num_cached == 0
        assert engine.num_free_slots == 16

    def test_init_invalid_slots(self):
        with pytest.raises(ValueError, match="num_slots must be > 0"):
            SWACacheEngine(num_slots=0)

    def test_init_invalid_evict_ratio(self):
        with pytest.raises(ValueError, match="evict_ratio"):
            SWACacheEngine(num_slots=10, evict_ratio=0.0)
        with pytest.raises(ValueError, match="evict_ratio"):
            SWACacheEngine(num_slots=10, evict_ratio=1.5)

    def test_allocate_basic(self, engine):
        h = HashType(12345)
        block = engine.allocate(h)
        assert block is not None
        assert block >= 0
        assert engine.num_cached == 1
        assert engine.num_free_slots == 15

    def test_allocate_multiple(self, engine):
        for i in range(8):
            block = engine.allocate(HashType(i * 100))
            assert block is not None
        assert engine.num_cached == 8
        assert engine.num_free_slots == 8

    def test_allocate_duplicate_returns_same_block(self, engine):
        h = HashType(12345)
        block1 = engine.allocate(h)
        block2 = engine.allocate(h)
        assert block1 == block2
        assert engine.num_cached == 1  # not double-counted

    def test_match_miss_on_empty(self, engine):
        result = engine.match(HashType(999))
        assert result.hit is False
        assert result.physical_block == -1

    def test_match_miss_when_not_ready(self, engine):
        h = HashType(100)
        engine.allocate(h)  # allocated but is_ready=False
        result = engine.match(h)
        assert result.hit is False

    def test_match_hit_when_ready(self, engine):
        h = HashType(100)
        block = engine.allocate(h)
        engine.set_ready(h, True)
        result = engine.match(h)
        assert result.hit is True
        assert result.physical_block == block

    def test_trailing_pages_semantics(self, engine):
        """Only the endpoint hash matters for SWA match (TRAILING_PAGES)."""
        h_endpoint = HashType(222)
        engine.allocate(h_endpoint)
        engine.set_ready(h_endpoint, True)

        # Match at endpoint succeeds
        assert engine.match(h_endpoint).hit is True
        # Match at a different hash fails
        assert engine.match(HashType(111)).hit is False
        assert engine.match(HashType(333)).hit is False

    def test_set_ready_false(self, engine):
        h = HashType(50)
        engine.allocate(h)
        engine.set_ready(h, True)
        assert engine.match(h).hit is True
        # Mark not ready again
        engine.set_ready(h, False)
        assert engine.match(h).hit is False

    def test_lock_prevents_eviction(self, engine):
        # Fill all 16 slots
        hashes = [HashType(i) for i in range(16)]
        for h in hashes:
            engine.allocate(h)
            engine.set_ready(h, True)
        assert engine.num_free_slots == 0

        # Lock the first entry
        engine.lock(hashes[0])

        # Allocate new entry — should evict something but NOT hashes[0]
        new_h = HashType(9999)
        block = engine.allocate(new_h)
        assert block is not None
        # hashes[0] should still be there (locked)
        assert engine.match(hashes[0]).hit is True

    def test_unlock_allows_eviction(self, engine):
        h = HashType(1)
        engine.allocate(h)
        engine.set_ready(h, True)
        engine.lock(h)
        engine.unlock(h)
        entry = engine._index[h]
        assert not entry.in_use()

    def test_eviction_lru_order(self, engine):
        """Oldest-accessed entries are evicted first."""
        hashes = [HashType(i) for i in range(16)]
        for i, h in enumerate(hashes):
            engine.allocate(h)
            engine.set_ready(h, True)
            # Stagger access times slightly
            engine._index[h].last_access_time = float(i)

        # Access hash[15] to make it the most recently used
        engine.match(hashes[15])

        # Allocate one more — should evict hash[0] (oldest)
        new_h = HashType(9999)
        engine.allocate(new_h)
        assert engine.match(hashes[0]).hit is False  # evicted
        assert engine.match(hashes[15]).hit is True  # survived

    def test_remove_explicit(self, engine):
        h = HashType(42)
        engine.allocate(h)
        engine.set_ready(h, True)
        assert engine.num_cached == 1

        engine.remove(h)
        assert engine.num_cached == 0
        assert engine.num_free_slots == 16
        assert engine.match(h).hit is False

    def test_remove_nonexistent(self, engine):
        # Should not raise
        engine.remove(HashType(99999))

    def test_reset(self, engine):
        for i in range(10):
            engine.allocate(HashType(i))
            engine.set_ready(HashType(i), True)
        assert engine.num_cached == 10

        engine.reset()
        assert engine.num_cached == 0
        assert engine.num_free_slots == 16

    def test_full_capacity_with_eviction(self):
        engine = SWACacheEngine(num_slots=4, evict_ratio=0.5)
        for i in range(4):
            engine.allocate(HashType(i))
            engine.set_ready(HashType(i), True)
        assert engine.num_free_slots == 0

        # Allocating one more should trigger eviction of 2 (50%)
        block = engine.allocate(HashType(99))
        assert block is not None
        assert engine.num_cached <= 4

    def test_all_locked_prevents_eviction(self):
        """If all entries are locked, eviction fails and allocate returns None."""
        engine = SWACacheEngine(num_slots=2, evict_ratio=0.5)
        h1, h2 = HashType(1), HashType(2)
        engine.allocate(h1)
        engine.set_ready(h1, True)
        engine.lock(h1)
        engine.allocate(h2)
        engine.set_ready(h2, True)
        engine.lock(h2)

        # All locked, can't evict => allocate fails
        result = engine.allocate(HashType(999))
        assert result is None


# ===========================================================================
# SWAEntry Tests
# ===========================================================================

class TestSWAEntry:
    def test_in_use_when_locked(self):
        entry = SWAEntry(
            physical_block=0, endpoint_hash=HashType(0),
            lock_cnt=1, is_ready=True
        )
        assert entry.in_use() is True

    def test_in_use_when_not_ready(self):
        entry = SWAEntry(
            physical_block=0, endpoint_hash=HashType(0),
            lock_cnt=0, is_ready=False
        )
        assert entry.in_use() is True

    def test_not_in_use(self):
        entry = SWAEntry(
            physical_block=0, endpoint_hash=HashType(0),
            lock_cnt=0, is_ready=True
        )
        assert entry.in_use() is False


# ===========================================================================
# SWAStorageConfig Tests
# ===========================================================================

class TestSWAStorageConfig:
    def test_page_size(self):
        cfg = SWAStorageConfig(
            num_slots=8, window_size=128,
            bytes_per_token_per_layer=584, num_swa_layers=61,
        )
        assert cfg.page_size_bytes == 128 * 584 * 61

    def test_total_size(self):
        cfg = SWAStorageConfig(
            num_slots=8, window_size=128,
            bytes_per_token_per_layer=584, num_swa_layers=61,
        )
        assert cfg.total_size_bytes == 8 * 128 * 584 * 61

    def test_from_pool_config(self):
        pool_cfg = SWAPoolConfig(
            enabled=True, window_size=64,
            bytes_per_token_per_layer=256, num_swa_layers=30, num_slots=100
        )
        storage_cfg = SWAStorageConfig.from_pool_config(pool_cfg)
        assert storage_cfg.num_slots == 100
        assert storage_cfg.window_size == 64
        assert storage_cfg.bytes_per_token_per_layer == 256
        assert storage_cfg.num_swa_layers == 30


# ===========================================================================
# SWAStorage Tests
# ===========================================================================

class TestSWAStorage:
    @pytest.fixture
    def storage_config(self):
        # Use small values for fast tests
        return SWAStorageConfig(
            num_slots=8,
            window_size=4,
            bytes_per_token_per_layer=16,
            num_swa_layers=2,
        )

    @pytest.fixture
    def storage(self, storage_config):
        return SWAStorage(storage_config, pin_memory=False)

    def test_buffer_shape(self, storage, storage_config):
        buf = storage.buffer
        expected_page_size = 4 * 16 * 2  # = 128
        assert buf.shape == (8, expected_page_size)

    def test_page_size_bytes(self, storage):
        assert storage.page_size_bytes == 4 * 16 * 2  # = 128

    def test_num_slots(self, storage):
        assert storage.num_slots == 8

    def test_get_slot_view(self, storage):
        view = storage.get_slot_view(0)
        assert len(view) == storage.page_size_bytes

    def test_get_slot_view_out_of_range(self, storage):
        with pytest.raises(IndexError):
            storage.get_slot_view(-1)
        with pytest.raises(IndexError):
            storage.get_slot_view(8)

    def test_write_and_read_slot(self, storage):
        data = np.arange(storage.page_size_bytes, dtype=np.uint8)
        storage.write_slot(0, data)
        result = storage.read_slot(0)
        if hasattr(result, 'numpy'):
            result = result.numpy()
        np.testing.assert_array_equal(result, data)

    def test_slot_isolation(self, storage):
        """Writing to one slot doesn't affect others."""
        data0 = np.full(storage.page_size_bytes, 0xFF, dtype=np.uint8)
        data1 = np.full(storage.page_size_bytes, 0x00, dtype=np.uint8)
        storage.write_slot(0, data0)
        storage.write_slot(1, data1)

        r0 = storage.read_slot(0)
        r1 = storage.read_slot(1)
        if hasattr(r0, 'numpy'):
            r0 = r0.numpy()
        if hasattr(r1, 'numpy'):
            r1 = r1.numpy()

        assert r0.sum() > 0
        assert r1.sum() == 0

    def test_slot_address(self, storage):
        page_size = storage.page_size_bytes
        assert storage.get_slot_address(0) == 0
        assert storage.get_slot_address(1) == page_size
        assert storage.get_slot_address(7) == 7 * page_size

    def test_data_ptr(self, storage):
        ptr = storage.data_ptr
        assert ptr > 0


# ===========================================================================
# Integration Tests: SWA Put/Get Flow
# ===========================================================================

class TestSWAPutGetFlow:
    """End-to-end test simulating SWA put (offload) and get (restore)."""

    @pytest.fixture
    def swa_system(self):
        """Create a small SWA cache engine + storage system."""
        config = SWAStorageConfig(
            num_slots=8, window_size=4,
            bytes_per_token_per_layer=16, num_swa_layers=2,
        )
        engine = SWACacheEngine(num_slots=8, evict_ratio=0.25)
        storage = SWAStorage(config, pin_memory=False)
        return engine, storage

    def test_put_then_get(self, swa_system):
        """Simulate: PUT SWA page to CPU, then GET it back."""
        engine, storage = swa_system
        endpoint_hash = HashType(12345)
        page_size = storage.page_size_bytes

        # PUT: allocate slot, write data, mark ready
        slot = engine.allocate(endpoint_hash)
        assert slot is not None

        fake_data = np.random.randint(0, 255, size=page_size, dtype=np.uint8)
        storage.write_slot(slot, fake_data)
        engine.set_ready(endpoint_hash, True)

        # GET: match, read data
        result = engine.match(endpoint_hash)
        assert result.hit is True
        retrieved = storage.read_slot(result.physical_block)
        if hasattr(retrieved, 'numpy'):
            retrieved = retrieved.numpy()
        np.testing.assert_array_equal(retrieved, fake_data)

    def test_put_multiple_sequences(self, swa_system):
        """Multiple sequences each get their own SWA page."""
        engine, storage = swa_system
        page_size = storage.page_size_bytes

        for i in range(5):
            h = HashType(i * 1000)
            slot = engine.allocate(h)
            data = np.full(page_size, i, dtype=np.uint8)
            storage.write_slot(slot, data)
            engine.set_ready(h, True)

        # Verify each can be retrieved
        for i in range(5):
            h = HashType(i * 1000)
            result = engine.match(h)
            assert result.hit is True
            retrieved = storage.read_slot(result.physical_block)
            if hasattr(retrieved, 'numpy'):
                retrieved = retrieved.numpy()
            assert retrieved[0] == i

    def test_overwrite_same_endpoint(self, swa_system):
        """Re-PUT for same endpoint updates the data in-place."""
        engine, storage = swa_system
        page_size = storage.page_size_bytes
        h = HashType(42)

        # First put
        slot = engine.allocate(h)
        data_v1 = np.full(page_size, 0x11, dtype=np.uint8)
        storage.write_slot(slot, data_v1)
        engine.set_ready(h, True)

        # Second put (same endpoint, same slot)
        slot2 = engine.allocate(h)
        assert slot2 == slot  # idempotent
        data_v2 = np.full(page_size, 0x22, dtype=np.uint8)
        storage.write_slot(slot2, data_v2)

        # GET should return v2
        result = engine.match(h)
        retrieved = storage.read_slot(result.physical_block)
        if hasattr(retrieved, 'numpy'):
            retrieved = retrieved.numpy()
        np.testing.assert_array_equal(retrieved, data_v2)

    def test_eviction_under_pressure(self, swa_system):
        """When pool is full, oldest entries get evicted."""
        engine, storage = swa_system

        # Fill all 8 slots
        for i in range(8):
            h = HashType(i)
            slot = engine.allocate(h)
            engine.set_ready(h, True)
            # Stagger access times
            engine._index[h].last_access_time = float(i)

        # Allocate 9th — triggers eviction of oldest (hash=0)
        h_new = HashType(999)
        slot = engine.allocate(h_new)
        assert slot is not None
        engine.set_ready(h_new, True)

        assert engine.match(h_new).hit is True
        assert engine.match(HashType(0)).hit is False  # evicted

    def test_locked_page_survives_eviction(self, swa_system):
        """Locked pages are not evicted even under pressure."""
        engine, storage = swa_system

        # Fill slots and lock one
        h_important = HashType(42)
        engine.allocate(h_important)
        engine.set_ready(h_important, True)
        engine.lock(h_important)
        # Set it as oldest
        engine._index[h_important].last_access_time = 0.0

        for i in range(7):
            h = HashType(100 + i)
            engine.allocate(h)
            engine.set_ready(h, True)
            engine._index[h].last_access_time = float(i + 1)

        # Force eviction
        engine.allocate(HashType(999))

        # Locked page survives
        assert engine.match(h_important).hit is True
        engine.unlock(h_important)

    def test_swa_page_size_matches_deepseek_v4(self):
        """Verify page size matches DeepSeek V4 spec: ~4.3 MB per request."""
        cfg = SWAPoolConfig(
            enabled=True,
            window_size=128,
            bytes_per_token_per_layer=584,
            num_swa_layers=61,
        )
        # 128 * 584 * 61 = 4,559,872 bytes ≈ 4.35 MB
        assert cfg.page_size_bytes == 4_559_872
        assert cfg.page_size_bytes < 5 * 1024 * 1024  # less than 5 MB
        assert cfg.page_size_bytes > 4 * 1024 * 1024  # more than 4 MB
