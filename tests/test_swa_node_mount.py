# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Node-mounted SWA logic tests for the pure-Python RadixTreeIndex mirror.

These validate the node-mount re-architecture (deployments/swa_design/
08_节点挂载SWA架构.md): SWA mounted on radix nodes, split-preserve, unified
match, dual eviction (full-by-leaf connects SWA; SWA-only LRU), dual lock.

The pure-Python RadixTreeIndex (flexkv/cache/radixtree.py) is the executable
spec for the C++ CRadixTreeIndex; DSv4 runs the C++ path (FLEXKV_INDEX_ACCEL=1).
Runs without torch/GPU (numpy only), so the invariants are checkable locally.
"""
import numpy as np
import pytest

from flexkv.common.block import SequenceMeta
from flexkv.cache.radixtree import RadixTreeIndex

TPB = 2


def _seq(ids):
    return SequenceMeta(np.array(ids, dtype=np.int64), tokens_per_block=TPB)


def _phys(*xs):
    return np.array(xs, dtype=np.int64)


def test_match_returns_last_swa_node():
    idx = RadixTreeIndex(tokens_per_block=TPB)
    n1 = idx.insert(_seq([1, 2, 3, 4, 5, 6, 7, 8]), _phys(0, 1, 2, 3), is_ready=True)
    assert n1 is not None and n1.size() == 4
    idx.set_swa(n1, slot=100)
    assert n1.has_swa() and n1.swa_host_slot == 100 and n1.on_swa_lru

    mr = idx.match_prefix(_seq([1, 2, 3, 4, 5, 6, 7, 8]))
    assert mr.num_matched_blocks == 4
    assert mr.last_swa_node is n1
    assert mr.swa_hit_blocks == 4


def test_split_preserves_swa_on_suffix_half():
    """I0/I4: split keeps the SWA on the half that owns the original last page."""
    idx = RadixTreeIndex(tokens_per_block=TPB)
    n1 = idx.insert(_seq([1, 2, 3, 4, 5, 6, 7, 8]), _phys(0, 1, 2, 3), is_ready=True)
    idx.set_swa(n1, slot=200)

    s2 = _seq([1, 2, 3, 4, 55, 66, 77, 88])
    assert idx.match_prefix(s2).num_matched_blocks == 2
    idx.insert(s2, _phys(8, 9), is_ready=True, match_result=idx.match_prefix(s2))

    # original n1 is now the suffix half (last 2 pages) and KEEPS its SWA.
    assert n1.has_swa() and n1.swa_host_slot == 200 and n1.size() == 2
    parent = n1.parent
    assert parent is not None and not parent.is_root()
    assert parent.swa_host_slot == -1 and parent.swa_tombstone  # prefix half: no SWA
    assert idx.drain_freed_swa_slots() == []  # split freed nothing

    mr = idx.match_prefix(_seq([1, 2, 3, 4, 5, 6, 7, 8]))
    assert mr.last_swa_node is n1 and mr.swa_hit_blocks == 4


def test_full_evict_frees_swa_slot():
    """I1: evicting a node's Full KV frees its SWA slot (drained to pool)."""
    idx = RadixTreeIndex(tokens_per_block=TPB)
    na = idx.insert(_seq([1, 2, 3, 4]), _phys(0, 1), is_ready=True)
    idx.set_swa(na, slot=300)
    ev_blocks, _ = idx.evict(2)
    assert len(ev_blocks) == 2
    assert idx.drain_freed_swa_slots() == [300]
    assert not na.on_swa_lru
    assert idx.total_swa_slots() == 0


def test_evict_swa_prefers_internal_node():
    """Multi-turn: SWA-only eviction drops interior-prefix SWA first, keeps Full."""
    idx = RadixTreeIndex(tokens_per_block=TPB)
    nfull = idx.insert(_seq([1, 2, 3, 4, 5, 6, 7, 8]), _phys(0, 1, 2, 3), is_ready=True)
    idx.set_swa(nfull, slot=400)
    sd = _seq([1, 2, 3, 4, 99, 98, 97, 96])
    idx.insert(sd, _phys(8, 9), is_ready=True, match_result=idx.match_prefix(sd))
    A = nfull.parent
    assert A is not None and not A.is_leaf()
    idx.set_swa(A, slot=401)
    # make the leaf MRU so the internal node A is the LRU victim
    idx._swa_lru_add_mru(nfull)

    evf, nfreed = idx.evict_swa(1)
    assert nfreed == 1
    assert idx.drain_freed_swa_slots() == [401]
    assert A.swa_tombstone and A.swa_host_slot == -1
    assert A.size() == 2  # Full KV kept
    assert nfull.has_swa()
    assert evf.size == 0  # internal SWA evict frees no full blocks


def test_evict_swa_leaf_without_lock_deletes_node():
    """I2: a leaf that would lose its SWA and has no full lock is deleted whole."""
    idx = RadixTreeIndex(tokens_per_block=TPB)
    nl = idx.insert(_seq([1, 2, 3, 4]), _phys(0, 1), is_ready=True)
    idx.set_swa(nl, slot=500)
    evf, nfreed = idx.evict_swa(1)
    assert nfreed == 1
    assert evf.size == 2  # whole node deleted, full blocks freed
    assert idx.is_empty()
    assert idx.drain_freed_swa_slots() == [500]


def test_evict_swa_leaf_with_full_lock_keeps_full():
    idx = RadixTreeIndex(tokens_per_block=TPB)
    nl = idx.insert(_seq([1, 2, 3, 4]), _phys(0, 1), is_ready=True)
    idx.set_swa(nl, slot=600)
    nl.lock_cnt = 1
    evf, nfreed = idx.evict_swa(1)
    assert nfreed == 1 and evf.size == 0
    assert nl.swa_tombstone and nl.size() == 2
    assert not idx.is_empty()


def test_dual_lock_invariant():
    """I3: full_lock_ref (lock_cnt) >= swa_lock_ref, with paired inc/dec."""
    idx = RadixTreeIndex(tokens_per_block=TPB)
    n1 = idx.insert(_seq([1, 2, 3, 4]), _phys(0, 1), is_ready=True)
    idx.set_swa(n1, slot=700)
    b = idx.inc_lock_ref(n1)
    assert n1.lock_cnt == 1 and n1.swa_lock_ref == 1 and b is n1
    assert n1.lock_cnt >= n1.swa_lock_ref
    idx.dec_lock_ref(n1, swa_boundary=b)
    assert n1.lock_cnt == 0 and n1.swa_lock_ref == 0


def test_dec_swa_lock_only_early_release():
    idx = RadixTreeIndex(tokens_per_block=TPB)
    n1 = idx.insert(_seq([1, 2, 3, 4]), _phys(0, 1), is_ready=True)
    idx.set_swa(n1, slot=701)
    b = idx.inc_lock_ref(n1)
    idx.dec_swa_lock_only(b)
    assert n1.swa_lock_ref == 0 and n1.swa_tombstone  # leaf SWA freed early
    assert n1.lock_cnt == 1  # full lock still held
    assert idx.drain_freed_swa_slots() == [701]
    idx.dec_lock_ref(n1, swa_boundary=b, skip_swa=True)
    assert n1.lock_cnt == 0


def test_dual_lock_only_deepest_swa_node_locked():
    """I3 + scope: inc_lock_ref locks full on [node,root) but SWA only on the
    single deepest node with SWA; dec is symmetric (no underflow)."""
    idx = RadixTreeIndex(tokens_per_block=TPB)
    a = idx.insert(_seq([1, 2, 3, 4, 5, 6, 7, 8]), _phys(0, 1, 2, 3), is_ready=True)
    idx.set_swa(a, slot=10)
    sd = _seq([1, 2, 3, 4, 55, 66, 77, 88])
    idx.insert(sd, _phys(8, 9), is_ready=True, match_result=idx.match_prefix(sd))
    A = a.parent
    idx.set_swa(A, slot=11)  # both internal A and leaf a carry SWA
    b = idx.inc_lock_ref(a)
    assert b is a  # deepest SWA node
    assert a.swa_lock_ref == 1 and A.swa_lock_ref == 0  # only deepest SWA locked
    assert a.lock_cnt == 1 and A.lock_cnt == 1  # full locked on both
    idx.dec_lock_ref(a, swa_boundary=b)
    assert a.swa_lock_ref == 0 and A.swa_lock_ref == 0
    assert a.lock_cnt == 0 and A.lock_cnt == 0


def test_swa_locked_node_not_full_evictable():
    """in_use() includes swa_lock_ref: a SWA-locked node is not full-evictable."""
    idx = RadixTreeIndex(tokens_per_block=TPB)
    n = idx.insert(_seq([1, 2, 3, 4]), _phys(0, 1), is_ready=True)
    idx.set_swa(n, slot=5)
    n.swa_lock_ref = 1
    assert n.in_use()
    assert not n.evictable()


def test_reset_rearms_swa_pool_via_host_pool():
    """SWAHostPool.reset re-arms every slot free (tree reset drops all nodes)."""
    from flexkv.swa.swa_host_pool import SWAHostPool
    from flexkv.common.config import SWAPoolConfig
    cfg = SWAPoolConfig(enabled=True, num_slots=4, window_size=TPB,
                        num_swa_layers=1, bytes_per_token_per_layer=2)
    pool = SWAHostPool(cfg)
    a, b = pool.allocate(), pool.allocate()
    assert a is not None and b is not None and pool.num_free == 2
    pool.reset()
    assert pool.num_free == 4  # all slots reclaimed


def test_partial_node_match_does_not_report_swa():
    """§5.2: a partially-matched node does not expose its trailing-page SWA."""
    idx = RadixTreeIndex(tokens_per_block=TPB)
    n1 = idx.insert(_seq([1, 2, 3, 4, 5, 6, 7, 8]), _phys(0, 1, 2, 3), is_ready=True)
    idx.set_swa(n1, slot=800)
    # query shares only the first 3 blocks (partial match of the 4-block node)
    mr = idx.match_prefix(_seq([1, 2, 3, 4, 5, 6, 99, 98]))
    assert mr.num_matched_blocks == 3
    # SWA sits on block-4's page; a partial match must not claim it
    assert mr.last_swa_node is None and mr.swa_hit_blocks == 0


def test_merge_child_moves_swa_to_merged_node():
    """I0: merging a single child moves the SWA to follow the child's last page,
    freeing the parent's stale SWA."""
    idx = RadixTreeIndex(tokens_per_block=TPB)
    # Build parent(2 blk, SWA=900) -> child(2 blk, SWA=901) by split.
    n = idx.insert(_seq([1, 2, 3, 4, 5, 6, 7, 8]), _phys(0, 1, 2, 3), is_ready=True)
    idx.set_swa(n, slot=901)  # this node is the 4-block leaf
    sd = _seq([1, 2, 3, 4, 55, 66, 77, 88])
    idx.insert(sd, _phys(8, 9), is_ready=True, match_result=idx.match_prefix(sd))
    parent = n.parent  # prefix half (2 blk), give it a stale SWA
    idx.set_swa(parent, slot=900)
    assert parent.num_children() == 2  # n + the divergent node -> can't merge yet

    # Fresh 2-level chain we can merge: root -> A(2blk,SWA) -> B(2blk,SWA), A has 1 child.
    idx2 = RadixTreeIndex(tokens_per_block=TPB)
    a = idx2.insert(_seq([1, 2, 3, 4, 5, 6, 7, 8]), _phys(0, 1, 2, 3), is_ready=True)
    idx2.set_swa(a, slot=901)
    sd2 = _seq([1, 2, 3, 4, 55, 66, 77, 88])
    idx2.insert(sd2, _phys(8, 9), is_ready=True, match_result=idx2.match_prefix(sd2))
    A = a.parent
    idx2.set_swa(A, slot=900)  # stale SWA on the internal prefix node
    # Remove the divergent sibling so A has exactly one child (a), enabling merge.
    sibling = [c for c in A.children.values() if c is not a][0]
    A.children.pop(sibling.head_hash())
    assert A.num_children() == 1
    idx2.merge_child(A)
    # A absorbed a; A's last page is a's last page, so A now carries a's SWA (901),
    # and A's stale 900 was freed.
    assert A.swa_host_slot == 901 and not A.swa_tombstone
    assert 900 in idx2.drain_freed_swa_slots()
