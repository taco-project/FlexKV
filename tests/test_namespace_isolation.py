import pytest
import numpy as np

from flexkv.common.block import SequenceMeta


class TestNamespace:
    def test_different_namespace_different_hash(self):
        tokens_per_block = 4
        token_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        
        # Create sequences with different namespace
        seq_a = SequenceMeta(
            token_ids=token_ids,
            tokens_per_block=tokens_per_block,
            namespace=["namespace_a"]
        )
        seq_b = SequenceMeta(
            token_ids=token_ids,
            tokens_per_block=tokens_per_block,
            namespace=["namespace_b"]
        )
        seq_none = SequenceMeta(
            token_ids=token_ids,
            tokens_per_block=tokens_per_block,
            namespace=None
        )

        assert not np.array_equal(seq_a.block_hashes, seq_b.block_hashes), \
            "Different namespace should produce different hashes"
        assert not np.array_equal(seq_a.block_hashes, seq_none.block_hashes), \
            "Namespace should affect hash calculation"
        assert not np.array_equal(seq_b.block_hashes, seq_none.block_hashes), \
            "Namespace should affect hash calculation"
    
    def test_same_namespace_same_hash(self):
        """Test that same namespace produces same hashes."""
        tokens_per_block = 4
        token_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        namespace = ["test_namespace"]
        
        # Create two sequences with same tokens and namespace
        seq1 = SequenceMeta(
            token_ids=token_ids,
            tokens_per_block=tokens_per_block,
            namespace=namespace
        )
        seq2 = SequenceMeta(
            token_ids=token_ids,
            tokens_per_block=tokens_per_block,
            namespace=namespace
        )

        assert np.array_equal(seq1.block_hashes, seq2.block_hashes), \
            "Same tokens with same namespace should produce same hashes"

        assert seq1.namespace_id == seq2.namespace_id, \
            "Same namespace should produce same namespace_id"
    
    def test_empty_namespace(self):
        """Test that empty namespace is treated the same as None."""
        tokens_per_block = 4
        token_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        
        seq_empty = SequenceMeta(
            token_ids=token_ids,
            tokens_per_block=tokens_per_block,
            namespace=[]
        )
        seq_none = SequenceMeta(
            token_ids=token_ids,
            tokens_per_block=tokens_per_block,
            namespace=None
        )
        seq_named = SequenceMeta(
            token_ids=token_ids,
            tokens_per_block=tokens_per_block,
            namespace=["namespace"]
        )

        assert np.array_equal(seq_empty.block_hashes, seq_none.block_hashes), \
            "Empty namespace should be the same as None"
        assert seq_empty.namespace_id is None and seq_none.namespace_id is None, \
            "Empty/None namespace should have None namespace_id"

        assert not np.array_equal(seq_empty.block_hashes, seq_named.block_hashes), \
            "Empty/None namespace should be different from named namespace"
        assert seq_named.namespace_id is not None, \
            "Named namespace should have non-None namespace_id"
    
    def test_prefix_matching_with_namespace(self):
        """Test that prefix matching works correctly with namespace."""
        tokens_per_block = 4
        prefix_tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
        full_tokens = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=np.int64)
        namespace = ["test_namespace"]
        
        seq_prefix = SequenceMeta(
            token_ids=prefix_tokens,
            tokens_per_block=tokens_per_block,
            namespace=namespace
        )
        seq_full = SequenceMeta(
            token_ids=full_tokens,
            tokens_per_block=tokens_per_block,
            namespace=namespace
        )

        assert np.array_equal(
            seq_prefix.block_hashes,
            seq_full.block_hashes[:len(seq_prefix.block_hashes)]
        ), "Prefix hashes should match when namespace is the same"
    
    def test_multi_element_namespace(self):
        """Test namespace with multiple elements (e.g., lora_id, cache_salt, namespace_info)."""
        tokens_per_block = 4
        token_ids = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)

        ns_full = ["lora1", "salt1", "user_ns1"]
        ns_diff_lora = ["lora2", "salt1", "user_ns1"]
        ns_diff_salt = ["lora1", "salt2", "user_ns1"]
        ns_diff_user = ["lora1", "salt1", "user_ns2"]
        
        seq_full = SequenceMeta(token_ids=token_ids, tokens_per_block=tokens_per_block, namespace=ns_full)
        seq_diff_lora = SequenceMeta(token_ids=token_ids, tokens_per_block=tokens_per_block, namespace=ns_diff_lora)
        seq_diff_salt = SequenceMeta(token_ids=token_ids, tokens_per_block=tokens_per_block, namespace=ns_diff_salt)
        seq_diff_user = SequenceMeta(token_ids=token_ids, tokens_per_block=tokens_per_block, namespace=ns_diff_user)

        assert not np.array_equal(seq_full.block_hashes, seq_diff_lora.block_hashes), \
            "Different lora_id should produce different hashes"
        assert not np.array_equal(seq_full.block_hashes, seq_diff_salt.block_hashes), \
            "Different cache_salt should produce different hashes"
        assert not np.array_equal(seq_full.block_hashes, seq_diff_user.block_hashes), \
            "Different user namespace should produce different hashes"
    
    def test_namespace_id_generation(self):
        """Test that namespace_id is correctly generated from namespace."""
        tokens_per_block = 4
        token_ids = np.array([1, 2, 3, 4], dtype=np.int64)

        seq_with_ns = SequenceMeta(
            token_ids=token_ids,
            tokens_per_block=tokens_per_block,
            namespace=["test"]
        )
        assert seq_with_ns.namespace_id is not None, \
            "Sequence with namespace should have namespace_id"

        seq_without_ns = SequenceMeta(
            token_ids=token_ids,
            tokens_per_block=tokens_per_block,
            namespace=None
        )
        assert seq_without_ns.namespace_id is None, \
            "Sequence without namespace should have None namespace_id"

        seq_same_ns = SequenceMeta(
            token_ids=token_ids,
            tokens_per_block=tokens_per_block,
            namespace=["test"]
        )
        assert seq_with_ns.namespace_id == seq_same_ns.namespace_id, \
            "Same namespace should produce same namespace_id"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
