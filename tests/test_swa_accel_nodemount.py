# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Node-mounted SWA on the PRODUCTION C++ radix tree (CRadixTreeIndex).

DSv4 runs SWA through ``CRadixTreeIndex`` (via ``CacheEngineAccel``,
FLEXKV_INDEX_ACCEL=1), NOT through ``LocalRadixTree`` (which needs
FLEXKV_ENABLE_P2P=1 and is skipped in the default build). The existing
``test_swa_cnode_cascade.py`` targets ``LocalRadixTree`` and therefore skips
entirely in the default build, leaving the production node-mount SWA path with
no coverage. This module fills that gap: it drives ``CRadixTreeIndex`` directly
and asserts the node-mount invariants (design doc 08_节点挂载SWA架构.md §2):

  I0  each node holds ≤1 SWA at its trailing page; split preserves it on the
      suffix half (the half that still owns the original last page).
  I1  SWA ⊆ Full: freeing a node's Full KV frees its SWA slot (drained to pool).
  I2  a leaf without SWA and without a full lock is meaningless → deleted;
      SWA-only eviction prefers interior nodes (multi-turn).
  I3  full_lock_ref (lock_cnt) ≥ swa_lock_ref, with symmetric inc/dec.
  I4  match returns the deepest fully-matched ready node carrying a live SWA
      (single forward pass); a partial node match must not expose its tail SWA.

Requires a real ``flexkv.c_ext`` with the CRadixNode SWA bindings.
"""
import numpy as np
import pytest
import torch

c_ext = pytest.importorskip("flexkv.c_ext")

if "swa_host_slot" not in dir(c_ext.CRadixNode):
    pytest.skip("CRadixNode SWA bindings not present (rebuild c_ext)",
                allow_module_level=True)
# The invariant assertions read structural accessors added to the CRadixNode
# binding for the non-P2P build. Skip cleanly on an older extension.
for _need in ("is_leaf", "get_lock_cnt", "has_swa"):
    if _need not in dir(c_ext.CRadixNode):
        pytest.skip(f"CRadixNode.{_need} not bound (rebuild c_ext)",
                    allow_module_level=True)

from flexkv.common.block import SequenceMeta

TPB = 2


def _hashes(ids):
    seq = SequenceMeta(token_ids=np.array(ids, dtype=np.int64), tokens_per_block=TPB)
    return torch.from_numpy(seq.block_hashes.astype(np.int64))


def _tree():
    return c_ext.CRadixTreeIndex(TPB, 4096, 0, "lru")


def _insert(tree, ids, base, ready=True, match=None):
    n = len(ids) // TPB
    bh = _hashes(ids)
    phys = torch.arange(base, base + n, dtype=torch.int64)
    if match is None:
        return tree.insert(phys, bh, n, -1, ready)
    return tree.insert(phys, bh, n, -1, ready,
                       match.last_node, match.num_matched_blocks,
                       match.last_node_matched_length)


def _evict_full(tree, k):
    buf = torch.zeros(k, dtype=torch.int64)
    got = tree.evict(buf, k)
    return buf.numpy()[:got]


def _evict_swa(tree, k):
    buf = torch.zeros(0, dtype=torch.int64)
    freed = tree.evict_swa(buf, k)
    return buf.numpy(), freed


# --------------------------------------------------------------------------- #
# I4 — match returns the last SWA-bearing node in one pass                     #
# --------------------------------------------------------------------------- #

def test_match_reports_last_swa_node():
    t = _tree()
    n = _insert(t, [1, 2, 3, 4, 5, 6, 7, 8], 0)
    t.set_swa(n, 100)
    assert n.has_swa() and n.swa_host_slot == 100
    mr = t.match_prefix(_hashes([1, 2, 3, 4, 5, 6, 7, 8]), 4, False)
    assert mr.num_matched_blocks == 4
    assert mr.last_swa_node is not None
    assert mr.last_swa_node.swa_host_slot == 100
    assert mr.swa_hit_blocks == 4


def test_partial_node_match_hides_tail_swa():
    """I4: SWA sits on the node's last page; a partial match cannot claim it."""
    t = _tree()
    n = _insert(t, [1, 2, 3, 4, 5, 6, 7, 8], 0)
    t.set_swa(n, 800)
    mr = t.match_prefix(_hashes([1, 2, 3, 4, 5, 6, 99, 98]), 4, False)
    assert mr.num_matched_blocks == 3
    assert mr.last_swa_node is None
    assert mr.swa_hit_blocks == 0


# --------------------------------------------------------------------------- #
# I0 — split preserves SWA on the suffix half                                  #
# --------------------------------------------------------------------------- #

def test_split_preserves_swa_on_suffix():
    t = _tree()
    n = _insert(t, [1, 2, 3, 4, 5, 6, 7, 8], 0)
    t.set_swa(n, 200)
    # Diverge after 2 blocks -> split the 4-block node into prefix(2)+suffix(2).
    s2 = [1, 2, 3, 4, 55, 66, 77, 88]
    m = t.match_prefix(_hashes(s2), 4, False)
    assert m.num_matched_blocks == 2
    _insert(t, s2, 8, match=m)
    # original node is now the suffix half and KEEPS its slot; nothing freed.
    assert n.swa_host_slot == 200 and not n.swa_tombstone and n.size() == 2
    parent = n.parent
    assert parent is not None and parent.swa_host_slot == -1 and parent.swa_tombstone
    assert t.drain_freed_swa_slots() == []
    mr = t.match_prefix(_hashes([1, 2, 3, 4, 5, 6, 7, 8]), 4, False)
    assert mr.last_swa_node is not None and mr.swa_hit_blocks == 4


# --------------------------------------------------------------------------- #
# I1 — full eviction connect-frees the SWA slot                                #
# --------------------------------------------------------------------------- #

def test_full_evict_frees_swa_slot():
    t = _tree()
    n = _insert(t, [1, 2, 3, 4], 0)
    t.set_swa(n, 300)
    ev = _evict_full(t, 2)
    assert len(ev) == 2
    assert 300 in t.drain_freed_swa_slots()
    assert t.is_empty()


def test_full_evict_no_swa_no_drain():
    t = _tree()
    _insert(t, [1, 2, 3, 4], 0)
    _evict_full(t, 2)
    assert t.drain_freed_swa_slots() == []


# --------------------------------------------------------------------------- #
# I2 — SWA-only eviction: internal-first, leaf deletes/keeps by full lock      #
# --------------------------------------------------------------------------- #

def test_evict_swa_internal_first_keeps_full():
    """Multi-turn: interior-prefix SWA dropped first, its Full KV kept."""
    t = _tree()
    nfull = _insert(t, [1, 2, 3, 4, 5, 6, 7, 8], 0)
    t.set_swa(nfull, 400)
    sd = [1, 2, 3, 4, 99, 98, 97, 96]
    m = t.match_prefix(_hashes(sd), 4, False)
    _insert(t, sd, 8, match=m)
    A = nfull.parent          # internal prefix node
    assert A is not None and not A.is_leaf()
    t.set_swa(A, 401)         # give the interior node an SWA
    # Touch the leaf so the interior node A is the LRU victim.
    t.set_swa(nfull, 400)     # re-mount == MRU bump on nfull
    evf, nfreed = _evict_swa(t, 1)
    assert nfreed == 1
    assert 401 in t.drain_freed_swa_slots()
    assert A.swa_tombstone and A.swa_host_slot == -1
    assert A.size() == 2 and not A.is_leaf()   # Full KV kept
    assert nfull.has_swa()
    assert evf.size == 0                        # interior evict frees no full


def test_evict_swa_leaf_unlocked_deletes_node():
    t = _tree()
    nl = _insert(t, [1, 2, 3, 4], 0)
    t.set_swa(nl, 500)
    evf, nfreed = _evict_swa(t, 1)
    assert nfreed == 1
    assert evf.size == 2                 # whole node deleted, full freed
    assert t.is_empty()
    assert 500 in t.drain_freed_swa_slots()


def test_evict_swa_leaf_locked_keeps_full():
    t = _tree()
    nl = _insert(t, [1, 2, 3, 4], 0)
    t.set_swa(nl, 600)
    t.lock(nl)                           # full lock
    evf, nfreed = _evict_swa(t, 1)
    assert nfreed == 1 and evf.size == 0
    assert nl.swa_tombstone and nl.size() == 2
    assert not t.is_empty()


# --------------------------------------------------------------------------- #
# I3 — dual lock: lock_cnt >= swa_lock_ref, symmetric inc/dec                  #
# --------------------------------------------------------------------------- #

def test_dual_lock_symmetric():
    t = _tree()
    n = _insert(t, [1, 2, 3, 4], 0)
    t.set_swa(n, 700)
    b = t.inc_lock_ref(n)
    assert b is not None
    assert n.get_lock_cnt() == 1 and n.swa_lock_ref == 1
    assert n.get_lock_cnt() >= n.swa_lock_ref     # I3
    t.dec_lock_ref(n, b, False)
    assert n.get_lock_cnt() == 0 and n.swa_lock_ref == 0


def test_only_deepest_swa_node_locked():
    t = _tree()
    a = _insert(t, [1, 2, 3, 4, 5, 6, 7, 8], 0)
    t.set_swa(a, 10)
    sd = [1, 2, 3, 4, 55, 66, 77, 88]
    m = t.match_prefix(_hashes(sd), 4, False)
    _insert(t, sd, 8, match=m)
    A = a.parent
    t.set_swa(A, 11)             # both interior A and leaf a carry SWA
    b = t.inc_lock_ref(a)
    assert b is not None
    # deepest SWA node (the leaf a) is the boundary; only it is SWA-locked.
    assert a.swa_lock_ref == 1 and A.swa_lock_ref == 0
    assert a.get_lock_cnt() == 1 and A.get_lock_cnt() == 1   # full on both
    t.dec_lock_ref(a, b, False)
    assert a.swa_lock_ref == 0 and A.swa_lock_ref == 0
    assert a.get_lock_cnt() == 0 and A.get_lock_cnt() == 0


def test_dec_swa_lock_only_early_release():
    t = _tree()
    n = _insert(t, [1, 2, 3, 4], 0)
    t.set_swa(n, 701)
    b = t.inc_lock_ref(n)
    t.dec_swa_lock_only(b)
    assert n.swa_lock_ref == 0 and n.swa_tombstone   # leaf SWA freed early
    assert n.get_lock_cnt() == 1                      # full lock still held
    assert 701 in t.drain_freed_swa_slots()
    t.dec_lock_ref(n, b, True)                         # skip_swa
    assert n.get_lock_cnt() == 0


def test_swa_locked_node_not_full_evictable():
    """in_use() includes swa_lock_ref: SWA-locked node survives full eviction."""
    t = _tree()
    n = _insert(t, [1, 2, 3, 4], 0)
    t.set_swa(n, 5)
    n.inc_swa_lock_ref()
    ev = _evict_full(t, 2)
    assert len(ev) == 0            # locked -> not evicted
    assert not t.is_empty()
    n.dec_swa_lock_ref()


# --------------------------------------------------------------------------- #
# I0 — merge moves SWA to the merged node (triggered via cascade)              #
# --------------------------------------------------------------------------- #

def test_reset_clears_tree():
    t = _tree()
    n = _insert(t, [1, 2, 3, 4], 0)
    t.set_swa(n, 900)
    t.reset()
    assert t.is_empty()


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
