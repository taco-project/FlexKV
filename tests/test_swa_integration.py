# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end integration tests for SWA Pool Manager.

Tests the full SWA lifecycle:
  - SWAPoolManager initialization from config
  - Endpoint hash computation
  - Put/Get/Match operations
  - Eviction under pressure
  - Multi-sequence concurrent access
  - Statistics tracking
  - Simulated request eviction/restoration flow
"""

import time

import numpy as np
import pytest

from flexkv.common.config import SWAPoolConfig, CacheConfig
from flexkv.swa.swa_pool_manager import SWAPoolManager, SWATransferRequest
from flexkv.swa.swa_cache_engine import SWACacheEngine, SWAMatchResult


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture
def small_config():
    """Small SWA config for fast tests."""
    return SWAPoolConfig(
        enabled=True,
        window_size=4,
        bytes_per_token_per_layer=16,
        num_swa_layers=2,
        num_slots=8,
        evict_ratio=0.25,
    )


@pytest.fixture
def manager(small_config):
    return SWAPoolManager(small_config, tokens_per_block=4)


@pytest.fixture
def deepseekv4_config():
    """DeepSeek V4 realistic config."""
    return SWAPoolConfig(
        enabled=True,
        window_size=128,
        bytes_per_token_per_layer=584,
        num_swa_layers=61,
        num_slots=100,
        evict_ratio=0.1,
    )


# ===========================================================================
# Initialization Tests
# ===========================================================================

class TestSWAPoolManagerInit:
    def test_init_from_config(self, manager, small_config):
        assert manager.config == small_config
        assert manager.get_num_cached() == 0
        assert manager.get_num_free_slots() == 8

    def test_init_disabled_raises(self):
        cfg = SWAPoolConfig(enabled=False)
        with pytest.raises(ValueError, match="disabled"):
            SWAPoolManager(cfg)

    def test_stats_initial(self, manager):
        stats = manager.stats
        assert stats["puts"] == 0
        assert stats["gets"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_page_size_deepseekv4(self, deepseekv4_config):
        mgr = SWAPoolManager(deepseekv4_config, tokens_per_block=16)
        page_size = deepseekv4_config.page_size_bytes
        assert page_size == 128 * 584 * 61  # 4,559,872 bytes
        assert mgr.storage.page_size_bytes == page_size


# ===========================================================================
# Endpoint Hash Tests
# ===========================================================================

class TestEndpointHash:
    def test_same_sequence_same_hash(self, manager):
        tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        h1 = manager.compute_endpoint_hash(tokens)
        h2 = manager.compute_endpoint_hash(tokens)
        assert h1 == h2

    def test_different_sequences_different_hash(self, manager):
        t1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        t2 = np.array([9, 10, 11, 12, 13, 14, 15, 16], dtype=np.int64)
        assert manager.compute_endpoint_hash(t1) != manager.compute_endpoint_hash(t2)

    def test_endpoint_uses_last_block(self, manager):
        """Endpoint hash depends on the trailing block only."""
        # Two sequences with same last block should have same hash
        t1 = np.array([99, 99, 99, 99, 1, 2, 3, 4], dtype=np.int64)
        t2 = np.array([88, 88, 88, 88, 1, 2, 3, 4], dtype=np.int64)
        assert manager.compute_endpoint_hash(t1) == manager.compute_endpoint_hash(t2)

    def test_growing_sequence_changes_hash(self, manager):
        """As sequence grows past block boundary, hash changes."""
        t_short = np.array([1, 2, 3, 4], dtype=np.int64)
        t_longer = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        # Different trailing block → different hash
        assert manager.compute_endpoint_hash(t_short) != manager.compute_endpoint_hash(t_longer)

    def test_empty_sequence(self, manager):
        tokens = np.array([], dtype=np.int64)
        h = manager.compute_endpoint_hash(tokens)
        assert h == 0


# ===========================================================================
# Put/Get Operations
# ===========================================================================

class TestPutGet:
    def test_put_then_get(self, manager, small_config):
        page_size = small_config.page_size_bytes
        h = 12345
        data = np.arange(page_size, dtype=np.uint8)

        assert manager.put(h, data) is True
        result = manager.get(h)
        assert result is not None
        np.testing.assert_array_equal(result, data)

    def test_get_nonexistent(self, manager):
        result = manager.get(99999)
        assert result is None

    def test_put_idempotent(self, manager, small_config):
        page_size = small_config.page_size_bytes
        h = 42

        data1 = np.full(page_size, 0x11, dtype=np.uint8)
        data2 = np.full(page_size, 0x22, dtype=np.uint8)

        # First put
        assert manager.put(h, data1) is True
        # Second put (same hash) — overwrites data
        assert manager.put(h, data2) is True
        # Get returns latest data
        result = manager.get(h)
        np.testing.assert_array_equal(result, data2)
        # Only 1 slot used
        assert manager.get_num_cached() == 1

    def test_put_multiple(self, manager, small_config):
        page_size = small_config.page_size_bytes
        for i in range(5):
            data = np.full(page_size, i, dtype=np.uint8)
            assert manager.put(i * 1000, data) is True

        assert manager.get_num_cached() == 5
        for i in range(5):
            result = manager.get(i * 1000)
            assert result[0] == i

    def test_remove(self, manager, small_config):
        page_size = small_config.page_size_bytes
        h = 100
        data = np.full(page_size, 0xFF, dtype=np.uint8)
        manager.put(h, data)
        assert manager.get(h) is not None

        manager.remove(h)
        assert manager.get(h) is None
        assert manager.get_num_cached() == 0


# ===========================================================================
# Match (TRAILING_PAGES) Tests
# ===========================================================================

class TestMatch:
    def test_match_hit(self, manager, small_config):
        h = 777
        data = np.zeros(small_config.page_size_bytes, dtype=np.uint8)
        manager.put(h, data)

        result = manager.match(h)
        assert result.hit is True
        assert result.physical_block >= 0

    def test_match_miss(self, manager):
        result = manager.match(999)
        assert result.hit is False

    def test_trailing_pages_semantics(self, manager, small_config):
        """TRAILING_PAGES: only the current endpoint matters.
        Different history, same endpoint hash → hit."""
        page_size = small_config.page_size_bytes

        # Store SWA for endpoint hash 500
        manager.put(500, np.full(page_size, 0xAA, dtype=np.uint8))

        # Match at endpoint 500 succeeds (regardless of prefix)
        assert manager.match(500).hit is True
        # Match at different endpoint fails
        assert manager.match(501).hit is False


# ===========================================================================
# Eviction Tests
# ===========================================================================

class TestEviction:
    def test_eviction_on_full(self, manager, small_config):
        page_size = small_config.page_size_bytes
        # Fill all 8 slots
        for i in range(8):
            data = np.full(page_size, i, dtype=np.uint8)
            manager.put(i, data)
            # Stagger access times for deterministic LRU
            manager.engine._index[i].last_access_time = float(i)

        assert manager.get_num_free_slots() == 0

        # 9th put triggers eviction
        data = np.full(page_size, 0xFF, dtype=np.uint8)
        assert manager.put(999, data) is True
        # Oldest (hash=0) should be evicted
        assert manager.match(0).hit is False
        # Newest should exist
        assert manager.match(999).hit is True

    def test_locked_survives_eviction(self, manager, small_config):
        page_size = small_config.page_size_bytes
        # Fill all slots
        for i in range(8):
            manager.put(i, np.full(page_size, i, dtype=np.uint8))
            manager.engine._index[i].last_access_time = float(i)

        # Lock the oldest
        manager.lock(0)

        # Trigger eviction
        manager.put(999, np.full(page_size, 0xFF, dtype=np.uint8))

        # hash=0 survives (locked), hash=1 evicted instead
        assert manager.match(0).hit is True
        assert manager.match(1).hit is False

        manager.unlock(0)


# ===========================================================================
# Statistics Tests
# ===========================================================================

class TestStatistics:
    def test_put_increments(self, manager, small_config):
        page_size = small_config.page_size_bytes
        manager.put(1, np.zeros(page_size, dtype=np.uint8))
        manager.put(2, np.zeros(page_size, dtype=np.uint8))
        assert manager.stats["puts"] == 2

    def test_get_hit_increments(self, manager, small_config):
        page_size = small_config.page_size_bytes
        manager.put(1, np.zeros(page_size, dtype=np.uint8))
        manager.get(1)
        assert manager.stats["hits"] == 1
        assert manager.stats["gets"] == 1

    def test_get_miss_increments(self, manager):
        manager.get(999)
        assert manager.stats["misses"] == 1
        assert manager.stats["gets"] == 0  # miss doesn't count as "get"

    def test_match_hit_miss(self, manager, small_config):
        page_size = small_config.page_size_bytes
        manager.put(1, np.zeros(page_size, dtype=np.uint8))
        manager.match(1)  # hit
        manager.match(2)  # miss
        assert manager.stats["hits"] == 1
        assert manager.stats["misses"] == 1

    def test_reset_clears_stats(self, manager, small_config):
        page_size = small_config.page_size_bytes
        manager.put(1, np.zeros(page_size, dtype=np.uint8))
        manager.get(1)
        manager.reset()
        assert all(v == 0 for v in manager.stats.values())


# ===========================================================================
# End-to-End Scenario: Request Eviction and Restoration
# ===========================================================================

class TestE2ERequestLifecycle:
    """Simulates the full lifecycle of SWA pages during request scheduling."""

    def test_single_request_evict_restore(self, manager, small_config):
        """
        Scenario:
        1. Request A is decoding (has SWA data on GPU)
        2. Scheduler decides to evict Request A
        3. SWA page is offloaded to CPU via put()
        4. Later, Request A is re-scheduled
        5. SWA page is restored from CPU via get()
        """
        page_size = small_config.page_size_bytes
        tokens_a = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int64)
        endpoint_a = manager.compute_endpoint_hash(tokens_a)

        # Step 1-2: Simulate GPU SWA data
        gpu_swa_snapshot = np.random.randint(0, 255, size=page_size, dtype=np.uint8)

        # Step 3: Evict → put to CPU
        assert manager.put(endpoint_a, gpu_swa_snapshot) is True
        assert manager.get_num_cached() == 1

        # Step 4-5: Restore → get from CPU
        restored_data = manager.get(endpoint_a)
        assert restored_data is not None
        np.testing.assert_array_equal(restored_data, gpu_swa_snapshot)

    def test_multiple_requests_concurrent(self, manager, small_config):
        """
        Multiple requests being evicted/restored interleaved.
        """
        page_size = small_config.page_size_bytes
        num_requests = 5

        # Evict all 5 requests
        snapshots = {}
        for i in range(num_requests):
            tokens = np.arange(i * 8, (i + 1) * 8, dtype=np.int64)
            h = manager.compute_endpoint_hash(tokens)
            data = np.full(page_size, i + 1, dtype=np.uint8)
            manager.put(h, data)
            snapshots[h] = data

        assert manager.get_num_cached() == num_requests

        # Restore them in reverse order
        for i in reversed(range(num_requests)):
            tokens = np.arange(i * 8, (i + 1) * 8, dtype=np.int64)
            h = manager.compute_endpoint_hash(tokens)
            result = manager.get(h)
            assert result is not None
            np.testing.assert_array_equal(result, snapshots[h])

    def test_request_finishes_normally(self, manager, small_config):
        """
        Request finishes normally (all tokens generated).
        SWA page should be explicitly removed (not needed anymore).
        """
        page_size = small_config.page_size_bytes
        tokens = np.array([1, 2, 3, 4], dtype=np.int64)
        h = manager.compute_endpoint_hash(tokens)

        # Put during eviction
        manager.put(h, np.zeros(page_size, dtype=np.uint8))
        assert manager.match(h).hit is True

        # Request finishes — remove SWA page
        manager.remove(h)
        assert manager.match(h).hit is False
        assert manager.get_num_cached() == 0

    def test_sequence_grows_new_swa(self, manager, small_config):
        """
        A restored request continues decoding, creating a new SWA snapshot.
        The old endpoint is obsolete; new endpoint gets the updated data.
        """
        page_size = small_config.page_size_bytes

        # Initial state: 8 tokens
        tokens_v1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        h_v1 = manager.compute_endpoint_hash(tokens_v1)
        data_v1 = np.full(page_size, 0x11, dtype=np.uint8)
        manager.put(h_v1, data_v1)

        # Sequence grows to 12 tokens (new trailing block → new hash)
        tokens_v2 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int64)
        h_v2 = manager.compute_endpoint_hash(tokens_v2)
        data_v2 = np.full(page_size, 0x22, dtype=np.uint8)

        # Remove old, put new
        manager.remove(h_v1)
        manager.put(h_v2, data_v2)

        # Old endpoint gone
        assert manager.match(h_v1).hit is False
        # New endpoint available
        result = manager.get(h_v2)
        np.testing.assert_array_equal(result, data_v2)

    def test_eviction_pressure_realistic(self, manager, small_config):
        """
        Simulate high-load scenario: many requests with limited SWA slots.
        """
        page_size = small_config.page_size_bytes
        num_requests = 20  # much more than 8 slots

        # Continuously put requests (older ones get evicted)
        hashes = []
        for i in range(num_requests):
            h = i * 7919  # prime hash spread
            data = np.full(page_size, i % 256, dtype=np.uint8)
            manager.put(h, data)
            hashes.append(h)

        # Most recent requests should still be cached
        # Older ones may have been evicted
        cached_count = sum(1 for h in hashes if manager.match(h).hit)
        assert cached_count <= 8  # can't exceed num_slots
        assert cached_count >= 1  # at least the most recent ones

        # Most recent 5 should definitely be there
        for h in hashes[-5:]:
            assert manager.match(h).hit is True


# ===========================================================================
# Integration with main KV matching
# ===========================================================================

class TestMainKVIntegration:
    """Tests how SWA pool integrates with the main KV prefix matching."""

    def test_combined_match_kv_plus_swa(self, manager, small_config):
        """
        Simulates the scenario where:
        - Main KV radix tree reports prefix hit (N blocks)
        - SWA pool reports page hit for the endpoint
        - Both must be present for full cache hit
        """
        page_size = small_config.page_size_bytes
        tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        h = manager.compute_endpoint_hash(tokens)

        # Main KV matched (simulated: N blocks of prefix data available)
        # SWA also matched
        manager.put(h, np.zeros(page_size, dtype=np.uint8))

        swa_result = manager.match(h)
        assert swa_result.hit is True
        # In the real system, the connector would combine:
        # main_kv_hit AND swa_hit → full prefix available

    def test_main_kv_hit_but_swa_miss(self, manager):
        """
        Main KV has prefix but SWA page was evicted.
        → Partial hit: main KV available but SWA needs recompute.
        """
        tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        h = manager.compute_endpoint_hash(tokens)

        # No SWA page stored
        swa_result = manager.match(h)
        assert swa_result.hit is False
        # In the real system: main KV is used but SWA ring buffer
        # is rebuilt from scratch during prefill
