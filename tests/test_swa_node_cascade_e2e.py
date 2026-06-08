"""End-to-end cascade test (Step 4 + Step 5): real C++ LocalRadixTree + real
node-attached SWAProductionManager, wired the way the cache engine wires them.

Proves the SWA-subset-of-full invariant end to end:
  - put a snapshot on a tree node, get it back (node-attached round trip);
  - SWA-only eviction tombstones an LRU node but leaves its full KV in the tree;
  - full-KV eviction of a node drains its SWA slot back to the pool (no leak).

Requires flexkv.c_ext built with FLEXKV_ENABLE_P2P=1.
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
from flexkv.common.config import SWAPoolConfig
from flexkv.swa.swa_production_manager import SWAProductionManager


TPB = 4


def _swa_config(num_slots):
    return SWAPoolConfig(
        enabled=True, num_slots=num_slots, window_size=4, num_swa_layers=2,
        bytes_per_token_per_layer=8, evict_ratio=0.25, pin_memory=False,
    )


def _hashes(token_ids):
    seq = SequenceMeta(token_ids=token_ids.astype(np.int64), tokens_per_block=TPB)
    return torch.from_numpy(seq.block_hashes.astype(np.int64))


def _insert(tree, token_ids, base_block):
    nb = len(token_ids) // TPB
    phys = torch.arange(base_block, base_block + nb, dtype=torch.int64)
    return tree.insert(phys, _hashes(token_ids), nb, -1, True)


def _data(cfg, value):
    return np.full(cfg.slot_size_bytes, value, dtype=np.uint8)


def _match_last_node(tree, token_ids):
    seq = SequenceMeta(token_ids=token_ids.astype(np.int64), tokens_per_block=TPB)
    bh = torch.from_numpy(seq.block_hashes.astype(np.int64))
    mr = tree.match_prefix(bh, int(seq.num_blocks), False)
    if mr is None or int(mr.num_matched_blocks) == 0:
        return None
    return mr.last_node


def test_put_get_roundtrip_via_node():
    tree = LocalRadixTree(tokens_per_block=TPB, max_num_blocks=1024)
    mgr = SWAProductionManager(_swa_config(8), TPB)

    tokens = np.arange(0, TPB * 3, dtype=np.int64)
    _insert(tree, tokens, 0)
    node = _match_last_node(tree, tokens)
    assert node is not None

    assert mgr.put(node, _data(mgr.config, 99)) is True
    assert mgr.has(node) is True
    out = mgr.get(node)
    assert out is not None and int(out[0]) == 99


def test_full_evict_frees_swa_slot_no_leak():
    tree = LocalRadixTree(tokens_per_block=TPB, max_num_blocks=1024)
    mgr = SWAProductionManager(_swa_config(8), TPB)

    # Insert a few distinct sequences and attach SWA to each leaf.
    nodes = []
    for i in range(3):
        tokens = np.arange(1000 * (i + 1), 1000 * (i + 1) + TPB * 2, dtype=np.int64)
        _insert(tree, tokens, 100 * (i + 1))
        n = _match_last_node(tree, tokens)
        assert mgr.put(n, _data(mgr.config, i)) is True
        nodes.append(n)

    used_before = mgr.pool.num_used
    assert used_before == 3

    # Evict everything from the tree; nodes get deleted and their slots recorded.
    buf = torch.zeros(256, dtype=torch.int64)
    tree.evict(buf, 256)
    freed = tree.drain_freed_swa_slots()
    assert len(freed) == 3  # all three SWA slots recorded by the cascade

    mgr.free_slots(freed)
    assert mgr.pool.num_used == 0  # no leak: pool fully reclaimed


def test_swa_only_evict_keeps_full_kv():
    # Pool smaller than the number of nodes -> SWA-only eviction must kick in.
    tree = LocalRadixTree(tokens_per_block=TPB, max_num_blocks=1024)
    mgr = SWAProductionManager(_swa_config(2), TPB)

    seqs = []
    nodes = []
    for i in range(3):
        tokens = np.arange(2000 * (i + 1), 2000 * (i + 1) + TPB * 2, dtype=np.int64)
        _insert(tree, tokens, 100 * (i + 1))
        n = _match_last_node(tree, tokens)
        assert mgr.put(n, _data(mgr.config, i)) is True
        seqs.append(tokens)
        nodes.append(n)

    # Pool had 2 slots, 3 puts -> the first (LRU) node was tombstoned.
    assert nodes[0].swa_tombstone is True
    assert nodes[0].swa_host_slot == -1
    # But its full KV is still in the tree: it still matches.
    assert _match_last_node(tree, seqs[0]) is not None
    # SWA is reported unavailable for it (tombstone), available for the newest.
    assert mgr.has(nodes[0]) is False
    assert mgr.has(nodes[2]) is True
