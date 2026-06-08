"""Step 1 + Step 2 verification: node-attached SWA state on the C++ radix tree.

Covers:
  - CRadixNode SWA fields are exposed and read/write round-trips (Step 1).
  - split / evict release the node's SWA slot into freed_swa_slots so the Python
    side can drain it, enforcing the "SWA subset of full" invariant (Step 2).

Requires a real flexkv.c_ext built with FLEXKV_ENABLE_P2P=1 (LocalRadixTree).
Skips cleanly if the C extension or LocalRadixTree is unavailable.
"""
import numpy as np
import pytest
import torch

c_ext = pytest.importorskip("flexkv.c_ext")

if not hasattr(c_ext, "LocalRadixTree"):
    pytest.skip("LocalRadixTree not built (needs FLEXKV_ENABLE_P2P=1)",
                allow_module_level=True)

if "swa_host_slot" not in dir(c_ext.CRadixNode):
    pytest.skip("CRadixNode SWA bindings not present (rebuild c_ext)",
                allow_module_level=True)

from flexkv.cache.radix_remote import LocalRadixTree
from flexkv.common.block import SequenceMeta


TPB = 4


def _block_hashes(token_ids: np.ndarray) -> torch.Tensor:
    """Compute per-block hashes via the canonical SequenceMeta path."""
    seq = SequenceMeta(token_ids=token_ids.astype(np.int64), tokens_per_block=TPB)
    return torch.from_numpy(seq.block_hashes.astype(np.int64))


def _make_tree() -> LocalRadixTree:
    return LocalRadixTree(tokens_per_block=TPB, max_num_blocks=1024)


def _insert(tree: LocalRadixTree, token_ids: np.ndarray, base_block: int):
    """Insert a fresh sequence; return the new leaf CRadixNode."""
    num_blocks = len(token_ids) // TPB
    block_hashes = _block_hashes(token_ids)
    phys = torch.arange(base_block, base_block + num_blocks, dtype=torch.int64)
    node = tree.insert(phys, block_hashes, num_blocks, -1, True)
    return node, block_hashes


# --------------------------------------------------------------------------- #
# Step 1: field round-trip                                                     #
# --------------------------------------------------------------------------- #

class TestStep1Fields:
    def test_defaults(self):
        tree = _make_tree()
        tokens = np.arange(0, TPB * 3, dtype=np.int64)
        node, _ = _insert(tree, tokens, 0)
        assert node is not None
        assert node.swa_host_slot == -1
        assert node.swa_tombstone is True
        assert node.swa_lock_ref == 0

    def test_setters_roundtrip(self):
        tree = _make_tree()
        tokens = np.arange(0, TPB * 3, dtype=np.int64)
        node, _ = _insert(tree, tokens, 0)
        node.swa_host_slot = 7
        node.swa_tombstone = False
        assert node.swa_host_slot == 7
        assert node.swa_tombstone is False

    def test_lock_ref_inc_dec(self):
        tree = _make_tree()
        tokens = np.arange(0, TPB * 3, dtype=np.int64)
        node, _ = _insert(tree, tokens, 0)
        node.inc_swa_lock_ref()
        node.inc_swa_lock_ref()
        assert node.swa_lock_ref == 2
        node.dec_swa_lock_ref()
        assert node.swa_lock_ref == 1


# --------------------------------------------------------------------------- #
# Step 2: cascade release on evict / split                                     #
# --------------------------------------------------------------------------- #

class TestStep2Cascade:
    def test_evict_drains_slot(self):
        tree = _make_tree()
        tokens = np.arange(0, TPB * 2, dtype=np.int64)
        node, _ = _insert(tree, tokens, 100)
        node.swa_host_slot = 7
        node.swa_tombstone = False

        # Evict everything; the node should be deleted and its slot drained.
        buf = torch.zeros(64, dtype=torch.int64)
        tree.evict(buf, 64)
        freed = tree.drain_freed_swa_slots()
        assert 7 in freed

    def test_no_slot_no_drain(self):
        tree = _make_tree()
        tokens = np.arange(0, TPB * 2, dtype=np.int64)
        node, _ = _insert(tree, tokens, 100)
        # node keeps default slot=-1
        buf = torch.zeros(64, dtype=torch.int64)
        tree.evict(buf, 64)
        assert tree.drain_freed_swa_slots() == []

    def test_split_drains_slot(self):
        tree = _make_tree()
        # Long sequence, then a sequence sharing a prefix to force a split.
        long_tokens = np.arange(0, TPB * 4, dtype=np.int64)
        node, _ = _insert(tree, long_tokens, 200)
        node.swa_host_slot = 9
        node.swa_tombstone = False

        # Shared first 2 blocks, divergent tail -> split of the existing node.
        shared = np.arange(0, TPB * 2, dtype=np.int64)
        divergent = np.arange(900, 900 + TPB * 2, dtype=np.int64)
        seq2 = np.concatenate([shared, divergent])
        _insert(tree, seq2, 300)

        freed = tree.drain_freed_swa_slots()
        assert 9 in freed

    def test_drain_is_idempotent(self):
        tree = _make_tree()
        tokens = np.arange(0, TPB * 2, dtype=np.int64)
        node, _ = _insert(tree, tokens, 100)
        node.swa_host_slot = 5
        node.swa_tombstone = False
        buf = torch.zeros(64, dtype=torch.int64)
        tree.evict(buf, 64)
        assert 5 in tree.drain_freed_swa_slots()
        # second drain returns nothing
        assert tree.drain_freed_swa_slots() == []
