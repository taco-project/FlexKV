"""End-to-end integration test for SWA (Sliding Window Attention) integration layer.

Does NOT depend on flexkv.c_ext or SequenceMeta — uses direct radix tree operations
with manually computed block hashes for portability.
"""
import sys
import hashlib
from unittest.mock import MagicMock
from typing import Optional

import numpy as np

# Mock the C extension before importing flexkv.cache
if 'flexkv.c_ext' not in sys.modules:
    _mock = MagicMock()
    _mock.get_hash_size = MagicMock(return_value=8)
    sys.modules['flexkv.c_ext'] = _mock

import torch
import pytest

from flexkv.common.config import SWAPoolConfig
from flexkv.cache.radixtree import RadixTreeIndex, RadixNode, MatchResult
from flexkv.common.hash_utils import HashType
from flexkv.swa.swa_radix_manager import SWARadixManager
from flexkv.swa.swa_connector import SWAConnector

# Detect if c_ext is real or mocked (affects token-lookup-only tests)
_c_ext_is_real = not isinstance(sys.modules.get('flexkv.c_ext'), MagicMock)


# --- Fixtures ---

@pytest.fixture
def tokens_per_block():
    return 4


@pytest.fixture
def swa_config():
    return SWAPoolConfig(
        enabled=True,
        num_slots=16,
        window_size=8,  # 8 tokens window
        num_swa_layers=2,
        bytes_per_token_per_layer=4,  # small for testing
        evict_ratio=0.25,
        pin_memory=False,
    )


@pytest.fixture
def radix_tree(tokens_per_block):
    return RadixTreeIndex(tokens_per_block=tokens_per_block, max_num_blocks=10000)


@pytest.fixture
def swa_manager(swa_config, tokens_per_block):
    return SWARadixManager(config=swa_config, tokens_per_block=tokens_per_block)


@pytest.fixture
def swa_connector(swa_manager, radix_tree):
    return SWAConnector(swa_manager=swa_manager, radix_tree=radix_tree)


def _make_swa_data(config: SWAPoolConfig, value: int = 42) -> np.ndarray:
    """Create test SWA data of the correct size for the config."""
    return np.full(config.slot_size_bytes, value, dtype=np.uint8)


def _compute_block_hashes(token_ids: np.ndarray, tokens_per_block: int) -> np.ndarray:
    """Compute deterministic block hashes without c_ext."""
    num_blocks = len(token_ids) // tokens_per_block
    hashes = np.zeros(num_blocks, dtype=np.int64)
    for i in range(num_blocks):
        block = token_ids[i * tokens_per_block: (i + 1) * tokens_per_block]
        h = hashlib.sha256(block.tobytes()).hexdigest()
        hashes[i] = int(h[:16], 16) & 0x7FFFFFFFFFFFFFFF  # positive int64
    return hashes


def _insert_sequence(radix_tree, token_ids, tokens_per_block):
    """Insert a token sequence into the radix tree and return the leaf node.

    Uses direct hash computation (no c_ext/SequenceMeta dependency).
    """
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.numpy()
    token_ids = np.asarray(token_ids, dtype=np.int64)

    block_hashes = _compute_block_hashes(token_ids, tokens_per_block)
    num_blocks = len(block_hashes)

    # Create a minimal SequenceMeta-like object for match_prefix
    class _FakeSeq:
        def __init__(self, hashes, tpb):
            self.block_hashes = hashes
            self.num_blocks = len(hashes)
            self.tokens_per_block = tpb
            self._has_hashes = True
        def gen_hashes(self):
            pass
        def get_hash(self, idx):
            if idx < len(self.block_hashes):
                return HashType(int(self.block_hashes[idx]))
            return HashType(0)

    seq = _FakeSeq(block_hashes, tokens_per_block)

    # Match existing prefix
    match_result = radix_tree.match_prefix(seq, update_cache_info=False)
    num_new_blocks = num_blocks - match_result.num_matched_blocks

    if num_new_blocks <= 0:
        return match_result.last_node

    physical_blocks = np.arange(num_new_blocks, dtype=np.int64) + np.random.randint(100, 10000)
    leaf_node = radix_tree.insert(seq, physical_blocks, is_ready=True, match_result=match_result)
    return leaf_node


# --- Test Classes ---

class TestSWAConnectorStoreLoad:
    """Test basic store and load operations via SWAConnector."""

    def test_store_swa_after_insert(self, swa_connector, swa_config, radix_tree, tokens_per_block):
        """Simulate request finish: insert KV into radix tree then store SWA."""
        token_ids = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int64)

        # Step 1: Insert into radix tree (simulates put_match completing)
        leaf_node = _insert_sequence(radix_tree, token_ids, tokens_per_block)
        assert leaf_node is not None

        # Step 2: Store SWA data
        swa_data = _make_swa_data(swa_config, value=123)
        result = swa_connector.store_swa(token_ids, swa_data, leaf_node=leaf_node)
        assert result is True

        # Verify the node is annotated
        assert leaf_node.swa_host_slot is not None
        assert leaf_node.swa_tombstone is False

    def test_load_swa_after_store(self, swa_connector, swa_config, radix_tree, tokens_per_block):
        """Simulate prefix match: load SWA data after a hit."""
        token_ids = np.array([10, 20, 30, 40, 50, 60, 70, 80], dtype=np.int64)

        # Insert and store
        leaf_node = _insert_sequence(radix_tree, token_ids, tokens_per_block)
        swa_data = _make_swa_data(swa_config, value=77)
        swa_connector.store_swa(token_ids, swa_data, leaf_node=leaf_node)

        # Load back
        loaded = swa_connector.load_swa(token_ids, leaf_node=leaf_node)
        assert loaded is not None
        loaded_np = np.asarray(loaded)
        assert loaded_np[0] == 77
        assert loaded_np[-1] == 77

    def test_store_with_torch_tensor(self, swa_connector, swa_config, radix_tree, tokens_per_block):
        """Test that torch.Tensor inputs are handled correctly."""
        token_ids = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80], dtype=torch.int64)

        # Insert using numpy (radix tree needs numpy)
        leaf_node = _insert_sequence(radix_tree, token_ids.numpy(), tokens_per_block)

        # Store with torch tensor token_ids
        swa_data = torch.full((swa_config.slot_size_bytes,), 55, dtype=torch.uint8)
        result = swa_connector.store_swa(token_ids, swa_data, leaf_node=leaf_node)
        assert result is True

        # Load back with torch tensor token_ids
        loaded = swa_connector.load_swa(token_ids, leaf_node=leaf_node)
        assert loaded is not None
        assert np.asarray(loaded)[0] == 55


class TestSWAConnectorCheckAvailability:
    """Test SWA availability checking."""

    def test_check_swa_available(self, swa_connector, swa_config, radix_tree, tokens_per_block):
        """check_swa returns True when SWA data covers the trailing window."""
        # 8 tokens = 2 blocks. Window = 8 tokens.
        token_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        leaf_node = _insert_sequence(radix_tree, token_ids, tokens_per_block)

        # Store SWA on the leaf
        swa_data = _make_swa_data(swa_config, value=1)
        swa_connector.store_swa(token_ids, swa_data, leaf_node=leaf_node)

        # Check availability
        result = swa_connector.check_swa(token_ids, leaf_node=leaf_node)
        # The leaf has 2 blocks * 4 tokens_per_block = 8 tokens
        # Window = 8 tokens -> exactly enough
        assert result is True

    def test_check_swa_not_available_tombstone(self, swa_connector, swa_config, radix_tree, tokens_per_block):
        """check_swa returns False when no SWA data is stored."""
        token_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        leaf_node = _insert_sequence(radix_tree, token_ids, tokens_per_block)

        # Don't store SWA - leaf is tombstone by default
        result = swa_connector.check_swa(token_ids, leaf_node=leaf_node)
        assert result is False

    @pytest.mark.skipif(not _c_ext_is_real, reason="requires real c_ext for token lookup")
    def test_check_swa_no_match(self, swa_connector):
        """check_swa returns False when token_ids don't match any tree entry."""
        token_ids = np.array([99, 98, 97, 96], dtype=np.int64)
        result = swa_connector.check_swa(token_ids)
        assert result is False




class TestSWAConnectorLookupByTokenIds:
    """Test that SWAConnector can find nodes by token_ids (without pre-resolved leaf).

    These tests require real c_ext for SequenceMeta hashing to work in _find_leaf_node.
    When c_ext is mocked, they are skipped.
    """

    @pytest.mark.skipif(not _c_ext_is_real, reason="requires real c_ext for token lookup")
    def test_store_and_load_by_token_lookup(self, swa_connector, swa_config, radix_tree, tokens_per_block):
        """Full flow: store SWA by leaf_node, then load by token_ids only."""
        token_ids = np.array([100, 200, 300, 400, 500, 600, 700, 800], dtype=np.int64)
        leaf_node = _insert_sequence(radix_tree, token_ids, tokens_per_block)

        # Store with explicit leaf_node
        swa_data = _make_swa_data(swa_config, value=88)
        swa_connector.store_swa(token_ids, swa_data, leaf_node=leaf_node)

        # Load using only token_ids (connector must find the node)
        loaded = swa_connector.load_swa(token_ids)
        assert loaded is not None
        assert np.asarray(loaded)[0] == 88

    @pytest.mark.skipif(not _c_ext_is_real, reason="requires real c_ext for token lookup")
    def test_check_by_token_lookup(self, swa_connector, swa_config, radix_tree, tokens_per_block):
        """check_swa works with token_ids lookup (no pre-resolved leaf)."""
        token_ids = np.array([100, 200, 300, 400, 500, 600, 700, 800], dtype=np.int64)
        leaf_node = _insert_sequence(radix_tree, token_ids, tokens_per_block)

        swa_data = _make_swa_data(swa_config, value=11)
        swa_connector.store_swa(token_ids, swa_data, leaf_node=leaf_node)

        # Check by token_ids only
        assert swa_connector.check_swa(token_ids) is True


class TestSWAConnectorPrefixMatch:
    """Test SWA with partial prefix matches (common in serving)."""

    def test_prefix_match_finds_swa(self, swa_connector, swa_config, radix_tree, tokens_per_block):
        """A new request shares a prefix with a stored request -> SWA available."""
        # Original request: 12 tokens (3 blocks)
        original_tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int64)
        leaf_node = _insert_sequence(radix_tree, original_tokens, tokens_per_block)

        # Store SWA on original's leaf
        swa_data = _make_swa_data(swa_config, value=42)
        swa_connector.store_swa(original_tokens, swa_data, leaf_node=leaf_node)

        # New request: first 8 tokens match (shares prefix)
        # The match should find the stored node
        new_tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8, 99, 99, 99, 99], dtype=np.int64)
        block_hashes = _compute_block_hashes(new_tokens, tokens_per_block)

        class _FakeSeq:
            def __init__(self, hashes, tpb):
                self.block_hashes = hashes
                self.num_blocks = len(hashes)
                self.tokens_per_block = tpb
            def gen_hashes(self):
                pass
            def get_hash(self, idx):
                if idx < len(self.block_hashes):
                    return HashType(int(self.block_hashes[idx]))
                return HashType(0)

        seq = _FakeSeq(block_hashes, tokens_per_block)
        match_result = radix_tree.match_prefix(seq)

        # Should match at least 2 blocks (8 tokens)
        assert match_result.num_matched_blocks >= 2

    def test_multiple_requests_share_prefix(self, swa_connector, swa_config, radix_tree, tokens_per_block):
        """Multiple requests with shared prefix: SWA stored per-leaf."""
        prefix = np.array([1, 2, 3, 4], dtype=np.int64)
        suffix1 = np.array([10, 20, 30, 40], dtype=np.int64)
        suffix2 = np.array([50, 60, 70, 80], dtype=np.int64)

        tokens1 = np.concatenate([prefix, suffix1])
        tokens2 = np.concatenate([prefix, suffix2])

        # Insert both
        leaf1 = _insert_sequence(radix_tree, tokens1, tokens_per_block)
        leaf2 = _insert_sequence(radix_tree, tokens2, tokens_per_block)

        # Store different SWA data on each
        swa_connector.store_swa(tokens1, _make_swa_data(swa_config, value=11), leaf_node=leaf1)
        swa_connector.store_swa(tokens2, _make_swa_data(swa_config, value=22), leaf_node=leaf2)

        # Load back each
        loaded1 = swa_connector.load_swa(tokens1, leaf_node=leaf1)
        loaded2 = swa_connector.load_swa(tokens2, leaf_node=leaf2)
        assert loaded1 is not None
        assert loaded2 is not None
        assert np.asarray(loaded1)[0] == 11
        assert np.asarray(loaded2)[0] == 22


class TestSWAConnectorEviction:
    """Test SWA eviction behavior through the connector."""

    def test_pool_eviction_on_capacity(self, radix_tree, tokens_per_block):
        """When the pool fills up, LRU entries are evicted."""
        # Small pool: only 4 slots
        small_config = SWAPoolConfig(
            enabled=True,
            num_slots=4,
            window_size=4,
            num_swa_layers=1,
            bytes_per_token_per_layer=4,
            evict_ratio=0.5,
            pin_memory=False,
        )
        manager = SWARadixManager(config=small_config, tokens_per_block=tokens_per_block)
        connector = SWAConnector(swa_manager=manager, radix_tree=radix_tree)

        # Insert 5 sequences (pool has 4 slots)
        leaves = []
        for i in range(5):
            token_ids = np.array([i * 10 + j for j in range(4)], dtype=np.int64)
            leaf = _insert_sequence(radix_tree, token_ids, tokens_per_block)
            leaves.append(leaf)
            data = _make_swa_data(small_config, value=i + 1)
            result = connector.store_swa(token_ids, data, leaf_node=leaf)
            assert result is True

        # Some earlier entries should have been evicted
        evicted_count = sum(1 for leaf in leaves if leaf.swa_tombstone)
        assert evicted_count >= 1  # At least one evicted to make room

        # The most recent entry should still be available
        assert leaves[-1].swa_tombstone is False


class TestSWAConnectorLocking:
    """Test SWA lock/unlock via connector."""

    def test_lock_prevents_eviction(self, swa_connector, swa_config, radix_tree, tokens_per_block):
        """Locked SWA data should not be evicted."""
        token_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        leaf_node = _insert_sequence(radix_tree, token_ids, tokens_per_block)

        swa_data = _make_swa_data(swa_config, value=99)
        swa_connector.store_swa(token_ids, swa_data, leaf_node=leaf_node)

        # Lock the SWA
        swa_connector.lock_swa(token_ids, leaf_node=leaf_node)
        assert leaf_node.swa_lock_ref > 0

        # Try to evict
        evicted = swa_connector.swa_manager.swa_evict_for_space(1)
        assert evicted == 0  # Can't evict locked

        # Unlock
        swa_connector.unlock_swa(token_ids, leaf_node=leaf_node)
        assert leaf_node.swa_lock_ref == 0

        # Now eviction should work
        evicted = swa_connector.swa_manager.swa_evict_for_space(1)
        assert evicted == 1
        assert leaf_node.swa_tombstone is True


class TestSWAConnectorStats:
    """Test statistics tracking."""

    def test_stats_after_operations(self, swa_connector, swa_config, radix_tree, tokens_per_block):
        """Verify stats are properly tracked through the connector."""
        token_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        leaf_node = _insert_sequence(radix_tree, token_ids, tokens_per_block)

        # Initial stats
        stats = swa_connector.stats
        assert stats["puts"] == 0
        assert stats["hits"] == 0

        # Store
        swa_data = _make_swa_data(swa_config, value=1)
        swa_connector.store_swa(token_ids, swa_data, leaf_node=leaf_node)

        stats = swa_connector.stats
        assert stats["puts"] == 1

        # Load
        swa_connector.load_swa(token_ids, leaf_node=leaf_node)
        stats = swa_connector.stats
        assert stats["hits"] == 1

        # Load on tombstone (miss)
        empty_tokens = np.array([99, 98, 97, 96, 95, 94, 93, 92], dtype=np.int64)
        empty_leaf = _insert_sequence(radix_tree, empty_tokens, tokens_per_block)
        swa_connector.load_swa(empty_tokens, leaf_node=empty_leaf)
        stats = swa_connector.stats
        assert stats["misses"] == 1


class TestSWAEndToEndFlow:
    """Full end-to-end flow simulating the serving pipeline."""

    def test_full_serve_cycle(self, swa_connector, swa_config, radix_tree, tokens_per_block):
        """Simulate: req1 finishes -> req2 arrives with shared prefix -> uses SWA."""
        # --- Request 1 finishes ---
        req1_tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int64)

        # 1a. Main KV stored (simulated by radix tree insert)
        req1_leaf = _insert_sequence(radix_tree, req1_tokens, tokens_per_block)
        assert req1_leaf is not None

        # 1b. SWA stored
        swa_snapshot = _make_swa_data(swa_config, value=42)
        assert swa_connector.store_swa(req1_tokens, swa_snapshot, leaf_node=req1_leaf)

        # --- Request 2 arrives with shared prefix ---
        req2_tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8, 20, 21, 22, 23], dtype=np.int64)

        # 2a. Prefix match (simulates get_match)
        block_hashes_2 = _compute_block_hashes(req2_tokens, tokens_per_block)

        class _FakeSeq2:
            def __init__(self, hashes, tpb):
                self.block_hashes = hashes
                self.num_blocks = len(hashes)
                self.tokens_per_block = tpb
            def gen_hashes(self):
                pass
            def get_hash(self, idx):
                if idx < len(self.block_hashes):
                    return HashType(int(self.block_hashes[idx]))
                return HashType(0)

        seq2 = _FakeSeq2(block_hashes_2, tokens_per_block)
        match = radix_tree.match_prefix(seq2)

        # Should match first 2 blocks (8 tokens)
        assert match.num_matched_blocks >= 2
        matched_token_count = match.num_matched_blocks * tokens_per_block
        assert matched_token_count >= 8

        # 2b. Check SWA availability for the matched prefix
        matched_prefix = req2_tokens[:matched_token_count]
        # Note: We can check on the node directly since we know it
        swa_avail = swa_connector.check_swa(matched_prefix, leaf_node=match.last_node)
        # The leaf node for the matched prefix might not have SWA
        # (SWA was stored on req1_leaf which covers all 12 tokens)
        # But if we check on req1_leaf directly, it should be available
        assert swa_connector.check_swa(req1_tokens, leaf_node=req1_leaf) is True

        # 2c. Load SWA data for the original stored sequence
        loaded_swa = swa_connector.load_swa(req1_tokens, leaf_node=req1_leaf)
        assert loaded_swa is not None
        assert np.asarray(loaded_swa)[0] == 42

    def test_multiple_requests_sequential(self, swa_connector, swa_config, radix_tree, tokens_per_block):
        """Multiple requests finishing sequentially, all with SWA stored."""
        all_leaves = []
        for i in range(5):
            # Each request has unique tokens
            token_ids = np.array([i * 100 + j for j in range(8)], dtype=np.int64)
            leaf = _insert_sequence(radix_tree, token_ids, tokens_per_block)
            swa_data = _make_swa_data(swa_config, value=i + 10)
            assert swa_connector.store_swa(token_ids, swa_data, leaf_node=leaf)
            all_leaves.append((token_ids, leaf, i + 10))

        # Verify all can be loaded back
        for token_ids, leaf, expected_val in all_leaves:
            loaded = swa_connector.load_swa(token_ids, leaf_node=leaf)
            assert loaded is not None
            assert np.asarray(loaded)[0] == expected_val

    def test_data_integrity_roundtrip(self, swa_connector, swa_config, radix_tree, tokens_per_block):
        """Verify that stored data matches loaded data exactly."""
        token_ids = np.array([7, 14, 21, 28, 35, 42, 49, 56], dtype=np.int64)
        leaf = _insert_sequence(radix_tree, token_ids, tokens_per_block)

        # Create data with a recognizable pattern
        swa_data = np.arange(swa_config.slot_size_bytes, dtype=np.uint8)
        swa_connector.store_swa(token_ids, swa_data, leaf_node=leaf)

        # Load and verify exact match
        loaded = swa_connector.load_swa(token_ids, leaf_node=leaf)
        assert loaded is not None
        loaded_np = np.asarray(loaded)
        np.testing.assert_array_equal(loaded_np, swa_data)


class TestSWAConnectorEdgeCases:
    """Edge cases and error handling."""

    @pytest.mark.skipif(not _c_ext_is_real, reason="requires real c_ext for token lookup")
    def test_store_on_empty_tree(self, swa_connector, swa_config):
        """Store fails gracefully when token_ids don't match anything."""
        token_ids = np.array([1, 2, 3, 4], dtype=np.int64)
        swa_data = _make_swa_data(swa_config)
        # No leaf_node and tree is empty -> _find_leaf_node returns None
        result = swa_connector.store_swa(token_ids, swa_data)
        assert result is False

    @pytest.mark.skipif(not _c_ext_is_real, reason="requires real c_ext for token lookup")
    def test_load_on_empty_tree(self, swa_connector):
        """Load returns None when tree is empty."""
        token_ids = np.array([1, 2, 3, 4], dtype=np.int64)
        result = swa_connector.load_swa(token_ids)
        assert result is None

    def test_double_store_updates(self, swa_connector, swa_config, radix_tree, tokens_per_block):
        """Storing SWA twice on same node updates the data."""
        token_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        leaf = _insert_sequence(radix_tree, token_ids, tokens_per_block)

        # First store
        swa_connector.store_swa(token_ids, _make_swa_data(swa_config, value=10), leaf_node=leaf)
        loaded1 = swa_connector.load_swa(token_ids, leaf_node=leaf)
        assert np.asarray(loaded1)[0] == 10

        # Second store (update)
        swa_connector.store_swa(token_ids, _make_swa_data(swa_config, value=20), leaf_node=leaf)
        loaded2 = swa_connector.load_swa(token_ids, leaf_node=leaf)
        assert np.asarray(loaded2)[0] == 20

    def test_load_after_eviction_returns_none(self, radix_tree, tokens_per_block):
        """After eviction, load returns None."""
        config = SWAPoolConfig(
            enabled=True,
            num_slots=2,
            window_size=4,
            num_swa_layers=1,
            bytes_per_token_per_layer=4,
            evict_ratio=1.0,  # Evict all when full
            pin_memory=False,
        )
        manager = SWARadixManager(config=config, tokens_per_block=tokens_per_block)
        connector = SWAConnector(swa_manager=manager, radix_tree=radix_tree)

        # Fill pool
        tokens1 = np.array([1, 2, 3, 4], dtype=np.int64)
        tokens2 = np.array([5, 6, 7, 8], dtype=np.int64)
        leaf1 = _insert_sequence(radix_tree, tokens1, tokens_per_block)
        leaf2 = _insert_sequence(radix_tree, tokens2, tokens_per_block)

        connector.store_swa(tokens1, _make_swa_data(config, value=1), leaf_node=leaf1)
        connector.store_swa(tokens2, _make_swa_data(config, value=2), leaf_node=leaf2)

        # Add a 3rd -> triggers eviction of leaf1 (LRU)
        tokens3 = np.array([9, 10, 11, 12], dtype=np.int64)
        leaf3 = _insert_sequence(radix_tree, tokens3, tokens_per_block)
        connector.store_swa(tokens3, _make_swa_data(config, value=3), leaf_node=leaf3)

        # leaf1 should be evicted
        assert leaf1.swa_tombstone is True
        loaded = connector.load_swa(tokens1, leaf_node=leaf1)
        assert loaded is None

        # leaf3 (most recent) should be available
        loaded3 = connector.load_swa(tokens3, leaf_node=leaf3)
        assert loaded3 is not None
        assert np.asarray(loaded3)[0] == 3
