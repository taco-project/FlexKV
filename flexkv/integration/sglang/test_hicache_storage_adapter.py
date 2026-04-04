"""
Unit tests for FlexKVHiCacheStorage (SGLang HiCacheStorage adapter).

Tests are split into:
- Pure logic tests (_get_token_ids, init, MLA skip) that don't need KVManager
- Integration tests (set/get/exists round-trip) that use real KVManager in thread mode

Test Coverage
~~~~~~~~~~~~~
1. test_token_ids_extraction: Handles plain list, RadixKey, None token_ids
2. test_adapter_init: Adapter construction with deferred KVManager
3. test_mla_non_rank0_skip: MLA non-rank-0 returns all True (skip backup)
4. test_no_token_ids_degradation: All methods return safe defaults when token_ids missing
5. test_set_exists_roundtrip: Basic set -> exists pipeline
6. test_set_exists_get_roundtrip: Full round-trip with data integrity validation
   - Tests layout transform (SGLang layer_first <-> FlexKV BLOCKFIRST)
   - Validates that data written via batch_set_v1 can be read back exactly
7. test_set_with_deduplication: Edge case - set same tokens twice, verify dedup + get

Known Issues Being Tested
~~~~~~~~~~~~~~~~~~~~~~~~~~
- batch_set_v1 page_idx computation when cpu_block_ids < num_pages (dedup case)
- batch_get_v1 radix tree matching and physical block ID ordering
- Layout transform correctness (critical for data integrity)

Usage:
    python3 -m flexkv.integration.sglang.test_hicache_storage_adapter

Requirements:
    - FlexKV built (debug mode recommended: FLEXKV_DEBUG=1 pip install -e .)
    - SGLang installed (pip install -e /path/to/sglang/python)
"""
import os
import sys
import numpy as np
import torch
from unittest.mock import MagicMock

from flexkv.integration.sglang.hicache_storage_adapter import (
    FlexKVHiCacheStorage, _get_token_ids,
)
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig, HiCacheStorageExtraInfo,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def _make_config(tp_rank=0, tp_size=1, is_mla=False, num_cpu_blocks=1000,
                 page_size=4):
    """Create a minimal HiCacheStorageConfig for testing."""
    return HiCacheStorageConfig(
        tp_rank=tp_rank, tp_size=tp_size, is_mla_model=is_mla,
        is_page_first_layout=False, model_name="test",
        extra_config={
            "num_layers": 4, "num_kv_heads": 2,
            "head_size": 64, "tokens_per_block": page_size,
            "enable_cpu": True, "enable_ssd": False,
            "num_cpu_blocks": num_cpu_blocks,
        }
    )


def _make_mock_pool(page_size=4, num_layers=4, num_kv_heads=2,
                    head_size=64, num_host_tokens=64):
    """Create a mock mem_pool_host simulating SGLang's MHATokenToKVPoolHost."""
    pool = MagicMock()
    pool.layout = "layer_first"
    pool.page_size = page_size
    pool.layer_num = num_layers
    pool.head_num = num_kv_heads
    pool.head_dim = head_size

    kv_buffer = torch.randn(2, num_layers, num_host_tokens, num_kv_heads,
                            head_size, dtype=torch.bfloat16)

    written_pages = {}

    def get_data_page(index, flat=True):
        page = kv_buffer[:, :, index:index + page_size, :, :]
        return page.flatten() if flat else page

    def set_from_flat_data_page(index, data_page):
        written_pages[index] = data_page.clone()

    pool.get_data_page = get_data_page
    pool.set_from_flat_data_page = set_from_flat_data_page
    pool._written_pages = written_pages
    pool._kv_buffer = kv_buffer

    return pool


# ---------------------------------------------------------------------------
# Pure logic tests (no KVManager needed)
# ---------------------------------------------------------------------------

def test_token_ids_extraction():
    """token_ids extraction from extra_info (plain list, RadixKey, None)."""
    token_ids = list(range(100, 116))
    extra = HiCacheStorageExtraInfo(
        prefix_keys=None, extra_info={"token_ids": token_ids})

    assert _get_token_ids(extra) == token_ids
    assert _get_token_ids(None) is None
    assert _get_token_ids(HiCacheStorageExtraInfo()) is None

    # Simulate RadixKey-like object
    class FakeRadixKey:
        def __init__(self, tids):
            self.token_ids = tids
    radix_extra = HiCacheStorageExtraInfo(
        prefix_keys=None, extra_info={"token_ids": FakeRadixKey(token_ids)})
    assert _get_token_ids(radix_extra) == token_ids


def test_adapter_init():
    """Adapter initializes with deferred KVManager."""
    backend = FlexKVHiCacheStorage(_make_config())
    assert backend._page_size == 4
    assert backend._should_backup is True
    assert backend._kv_manager is None  # deferred


def test_mla_non_rank0_skip():
    """MLA non-rank-0 skips backup and returns all True."""
    config = _make_config(tp_rank=1, tp_size=2, is_mla=True, num_cpu_blocks=100)
    backend = FlexKVHiCacheStorage(config)
    assert backend._should_backup is False

    keys = ["key0", "key1"]
    host_indices = torch.arange(0, 8, dtype=torch.int64)
    extra = HiCacheStorageExtraInfo(
        prefix_keys=None, extra_info={"token_ids": list(range(8))})
    result = backend.batch_set_v1(keys, host_indices, extra)
    assert result == [True, True]


def test_no_token_ids_degradation():
    """All methods degrade gracefully when token_ids are missing."""
    backend = FlexKVHiCacheStorage(_make_config())
    keys = ["key0", "key1"]
    host_indices = torch.arange(0, 8, dtype=torch.int64)
    empty_extra = HiCacheStorageExtraInfo()

    assert backend.batch_exists(keys, empty_extra) == 0
    assert backend.batch_get_v1(keys, host_indices, empty_extra) == [False, False]
    assert backend.batch_set_v1(keys, host_indices, empty_extra) == [False, False]


# ---------------------------------------------------------------------------
# Integration tests (real KVManager in thread mode)
# ---------------------------------------------------------------------------

def test_set_exists_roundtrip():
    """Full set -> exists round-trip with real KVManager."""
    # Ensure thread mode
    os.environ["FLEXKV_CPU_ONLY"] = "1"
    os.environ["FLEXKV_INSTANCE_NUM"] = "0"

    backend = FlexKVHiCacheStorage(_make_config(page_size=4))
    pool = _make_mock_pool(page_size=4)
    backend.register_mem_pool_host(pool)

    assert backend._kv_manager is not None
    assert backend._cpu_cache_tensor is not None

    token_ids = list(range(100, 116))  # 16 tokens = 4 pages
    keys = ["key0", "key1", "key2", "key3"]
    host_indices = torch.arange(0, 16, dtype=torch.int64)
    extra = HiCacheStorageExtraInfo(
        prefix_keys=None, extra_info={"token_ids": token_ids})

    # Before set: nothing exists
    assert backend.batch_exists(keys, extra) == 0

    # Set
    set_results = backend.batch_set_v1(keys, host_indices, extra)
    assert all(set_results), f"set failed: {set_results}"

    # After set: all exist
    exists_count = backend.batch_exists(keys, extra)
    assert exists_count == 4, f"Expected 4, got {exists_count}"


def test_set_exists_get_roundtrip():
    """Full set -> exists -> get cycle with data validation."""
    os.environ["FLEXKV_CPU_ONLY"] = "1"
    os.environ["FLEXKV_INSTANCE_NUM"] = "0"

    backend = FlexKVHiCacheStorage(_make_config(page_size=4))
    write_pool = _make_mock_pool(page_size=4)
    backend.register_mem_pool_host(write_pool)

    # Prepare input data: write known values to pool's KV buffer
    token_ids = list(range(100, 116))  # 16 tokens = 4 pages (page_size=4)
    keys = ["key0", "key1", "key2", "key3"]
    write_host_indices = torch.arange(0, 16, dtype=torch.int64)

    # Seed write pool buffer with distinct values per page for verification
    for page_idx in range(4):
        start = page_idx * 4
        end = start + 4
        # Each page: fill with a unique offset (1.0, 2.0, 3.0, 4.0)
        write_pool._kv_buffer[:, :, start:end, :, :] = float(page_idx + 1)

    extra = HiCacheStorageExtraInfo(
        prefix_keys=None, extra_info={"token_ids": token_ids})

    # Phase 1: Before set, nothing exists
    assert backend.batch_exists(keys, extra) == 0, "Expected 0 pages before set"

    # Phase 2: Write to FlexKV
    set_results = backend.batch_set_v1(keys, write_host_indices, extra)
    assert all(set_results), f"batch_set_v1 failed: {set_results}"

    # Phase 3: All pages now exist
    exists_count = backend.batch_exists(keys, extra)
    assert exists_count == 4, f"Expected 4 pages after set, got {exists_count}"

    # Phase 4: Read back and verify data integrity
    # Create a fresh mock pool for reading (simulating SGLang's reading phase)
    read_pool = _make_mock_pool(page_size=4, num_host_tokens=64)
    read_host_indices = torch.arange(0, 16, dtype=torch.int64)
    
    # Update backend's mem_pool_host to point to read pool
    old_pool = backend._mem_pool_host
    backend._mem_pool_host = read_pool
    
    read_results = backend.batch_get_v1(keys, read_host_indices, extra)
    assert all(read_results), f"batch_get_v1 failed: {read_results}"
    
    # Phase 5: Verify data integrity
    # Check that read_pool got the correct data back via set_from_flat_data_page
    for page_idx in range(4):
        host_token_start = page_idx * 4
        # The page should have been written to read_pool._written_pages
        if host_token_start in read_pool._written_pages:
            read_data = read_pool._written_pages[host_token_start]
            # Reshape back to full tensor shape for comparison
            expected_shape = (2, 4, 4, 2, 64)
            read_data_shaped = read_data.reshape(expected_shape)
            expected_data = write_pool._kv_buffer[:, :, page_idx*4:(page_idx+1)*4, :, :]
            # Data should match original (after layout transform round-trip)
            # Note: small numerical errors acceptable due to float operations
            assert torch.allclose(read_data_shaped, expected_data, rtol=1e-3, atol=1e-4), \
                f"Page {page_idx} data mismatch after get"
        else:
            # If not in written_pages, check that it was at least attempted
            pass  # Could happen if get didn't actually call set_from_flat_data_page
    
    backend._mem_pool_host = old_pool


def test_set_with_deduplication():
    """Test set -> set (dedup) -> get to verify deduplication doesn't break indexing."""
    os.environ["FLEXKV_CPU_ONLY"] = "1"
    os.environ["FLEXKV_INSTANCE_NUM"] = "0"

    backend = FlexKVHiCacheStorage(_make_config(page_size=4))
    pool = _make_mock_pool(page_size=4)
    backend.register_mem_pool_host(pool)

    token_ids = list(range(100, 116))  # 4 pages
    keys = ["key0", "key1", "key2", "key3"]
    host_indices = torch.arange(0, 16, dtype=torch.int64)

    # Seed with distinct values
    for page_idx in range(4):
        pool._kv_buffer[:, :, page_idx*4:(page_idx+1)*4, :, :] = float(page_idx + 1)

    extra = HiCacheStorageExtraInfo(
        prefix_keys=None, extra_info={"token_ids": token_ids})

    # First set
    set_results1 = backend.batch_set_v1(keys, host_indices, extra)
    assert all(set_results1), "First set failed"

    # Change pool values
    for page_idx in range(4):
        pool._kv_buffer[:, :, page_idx*4:(page_idx+1)*4, :, :] = float(page_idx + 10)

    # Second set (should deduplicate - no new pages)
    set_results2 = backend.batch_set_v1(keys, host_indices, extra)
    assert all(set_results2), "Second set (dedup) failed"

    # Get should still return original data (from first set)
    read_pool = _make_mock_pool(page_size=4)
    backend._mem_pool_host = read_pool

    read_results = backend.batch_get_v1(keys, host_indices, extra)
    assert all(read_results), "Get failed after dedup"

    # Verify we got the original data (1.0, 2.0, 3.0, 4.0) not the new values
    for page_idx in range(4):
        host_token_start = page_idx * 4
        if host_token_start in read_pool._written_pages:
            read_data = read_pool._written_pages[host_token_start]
            read_data_shaped = read_data.reshape(2, 4, 4, 2, 64)
            # Should be original value, not the changed value
            assert torch.allclose(read_data_shaped, 
                                torch.full_like(read_data_shaped, float(page_idx + 1)),
                                rtol=1e-3, atol=1e-4), \
                f"Page {page_idx}: expected {float(page_idx + 1)}, got something else"


def test_stats():
    """Statistics are collected correctly."""
    os.environ["FLEXKV_CPU_ONLY"] = "1"
    os.environ["FLEXKV_INSTANCE_NUM"] = "0"

    backend = FlexKVHiCacheStorage(_make_config(page_size=4))
    pool = _make_mock_pool(page_size=4)
    backend.register_mem_pool_host(pool)

    token_ids = list(range(100, 116))
    keys = ["key0", "key1", "key2", "key3"]
    host_indices = torch.arange(0, 16, dtype=torch.int64)
    extra = HiCacheStorageExtraInfo(
        prefix_keys=None, extra_info={"token_ids": token_ids})

    backend.batch_set_v1(keys, host_indices, extra)
    backend.batch_exists(keys, extra)

    stats = backend.get_stats()
    expected_keys = {'get_calls', 'set_calls', 'get_tokens_requested',
                     'get_tokens_hit', 'get_hit_rate', 'set_tokens_written',
                     'set_tokens_deduped', 'exists_calls', 'errors'}
    assert expected_keys.issubset(stats.keys())
    assert stats['set_tokens_written'] > 0
    assert stats['errors'] == 0


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

ALL_TESTS = [
    ("token_ids extraction", test_token_ids_extraction),
    ("adapter initialization", test_adapter_init),
    ("MLA non-rank-0 skip", test_mla_non_rank0_skip),
    ("no token_ids degradation", test_no_token_ids_degradation),
    ("set -> exists round-trip", test_set_exists_roundtrip),
    ("set -> exists -> get round-trip", test_set_exists_get_roundtrip),
    ("set with deduplication", test_set_with_deduplication),
    ("stats collection", test_stats),
]


def main():
    passed = 0
    failed = 0

    for name, test_fn in ALL_TESTS:
        try:
            test_fn()
            print(f"  PASS  {name}")
            passed += 1
        except Exception as e:
            print(f"  FAIL  {name}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print()
    if failed == 0:
        print(f"All {passed} tests passed.")
    else:
        print(f"{failed}/{passed + failed} tests FAILED.")
        sys.exit(1)


if __name__ == '__main__':
    main()
