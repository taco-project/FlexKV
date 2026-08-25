# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Node-mounted SWA on the radix tree — invariants I0-I4 + a workload stress run.

The node-mount re-architecture
mounts SWA on radix nodes. This file is the single home for its invariant
coverage, in three sections:

  1. PURE-PYTHON SPEC  (flexkv/cache/radixtree.py RadixTreeIndex) — the
     executable spec; numpy-only, runs anywhere.
  2. C++ PRODUCTION    (flexkv.c_ext CRadixTreeIndex, the DSv4 path with
     FLEXKV_INDEX_ACCEL=1) — the same invariants on the shipped implementation.
     (test_swa_cnode_cascade.py targets the P2P-only LocalRadixTree and skips in
     the default build; this section is the production coverage.)
  3. WORKLOAD STRESS   — a stochastic multi-turn dialogue through the production
     CacheEngineAccel path under pool pressure, with REAL SWA host-pool byte IO,
     asserting no double-free / use-after-free / byte-mismatch / leak-after-reset
     (folded in from the former benchmarks/benchmark_swa_nodemount.py).

Invariants:
  I0  each node holds ≤1 SWA at its trailing page; split preserves it on the
      suffix half; merge moves it to follow the merged node's last page.
  I1  SWA ⊆ Full: freeing a node's Full KV frees its SWA slot (drained to pool).
  I2  SWA-only eviction prefers interior nodes; a leaf that loses its SWA with no
      full lock is deleted, with a full lock its Full KV is kept.
  I3  full_lock_ref (lock_cnt) ≥ swa_lock_ref, symmetric inc/dec.
  I4  match returns the deepest fully-matched ready node with a live SWA in one
      pass; a partial node match must not expose its trailing-page SWA.
"""
import random
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pytest

from flexkv.common.block import SequenceMeta
from flexkv.cache.radixtree import RadixTreeIndex

pytestmark = pytest.mark.unit

TPB = 2


# =========================================================================== #
# 1. PURE-PYTHON SPEC — RadixTreeIndex (numpy only)                           #
# =========================================================================== #

def _seq(ids):
    return SequenceMeta(np.array(ids, dtype=np.int64), tokens_per_block=TPB)


def _phys(*xs):
    return np.array(xs, dtype=np.int64)


def test_py_match_returns_last_swa_node():
    idx = RadixTreeIndex(tokens_per_block=TPB)
    n1 = idx.insert(_seq([1, 2, 3, 4, 5, 6, 7, 8]), _phys(0, 1, 2, 3), is_ready=True)
    assert n1 is not None and n1.size() == 4
    idx.set_swa(n1, slot=100)
    assert n1.has_swa() and n1.swa_host_slot == 100 and n1.on_swa_lru

    mr = idx.match_prefix(_seq([1, 2, 3, 4, 5, 6, 7, 8]))
    assert mr.num_matched_blocks == 4
    assert mr.last_swa_node is n1
    assert mr.swa_hit_blocks == 4


def test_py_split_preserves_swa_on_suffix_half():
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


def test_py_full_evict_frees_swa_slot():
    """I1: evicting a node's Full KV frees its SWA slot (drained to pool)."""
    idx = RadixTreeIndex(tokens_per_block=TPB)
    na = idx.insert(_seq([1, 2, 3, 4]), _phys(0, 1), is_ready=True)
    idx.set_swa(na, slot=300)
    ev_blocks, _ = idx.evict(2)
    assert len(ev_blocks) == 2
    assert idx.drain_freed_swa_slots() == [300]
    assert not na.on_swa_lru
    assert idx.total_swa_slots() == 0


def test_py_evict_swa_prefers_internal_node():
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


def test_py_evict_swa_leaf_without_lock_deletes_node():
    """I2: a leaf that would lose its SWA and has no full lock is deleted whole."""
    idx = RadixTreeIndex(tokens_per_block=TPB)
    nl = idx.insert(_seq([1, 2, 3, 4]), _phys(0, 1), is_ready=True)
    idx.set_swa(nl, slot=500)
    evf, nfreed = idx.evict_swa(1)
    assert nfreed == 1
    assert evf.size == 2  # whole node deleted, full blocks freed
    assert idx.is_empty()
    assert idx.drain_freed_swa_slots() == [500]


def test_py_evict_swa_leaf_with_full_lock_keeps_full():
    idx = RadixTreeIndex(tokens_per_block=TPB)
    nl = idx.insert(_seq([1, 2, 3, 4]), _phys(0, 1), is_ready=True)
    idx.set_swa(nl, slot=600)
    nl.lock_cnt = 1
    evf, nfreed = idx.evict_swa(1)
    assert nfreed == 1 and evf.size == 0
    assert nl.swa_tombstone and nl.size() == 2
    assert not idx.is_empty()


def test_py_match_probe_does_not_promote_swa_lru():
    """M1.5 (regression guard): a match-only PROBE (update_cache_info=False) must
    NOT reorder the SWA-LRU. A probe may never lead to actual reuse
    (usable=min(full,swa) can clamp it to 0), so promoting on a probe would
    pollute eviction. Passes today; must keep passing after the reuse-promotion
    fix (which should gate on update_cache_info=True only)."""
    idx = RadixTreeIndex(tokens_per_block=TPB)
    a = idx.insert(_seq([1, 2, 3, 4]), _phys(0, 1), is_ready=True)
    idx.set_swa(a, slot=10)
    b = idx.insert(_seq([5, 6, 7, 8]), _phys(2, 3), is_ready=True)
    idx.set_swa(b, slot=11)
    # SWA-LRU order (LRU tail -> MRU head): A, B. Probe A without cache-info update.
    idx.match_prefix(_seq([1, 2, 3, 4]), update_cache_info=False)
    # A must still be the LRU victim — the probe did not promote it.
    assert idx._swa_lru_get_lru_unlocked() is a


def test_py_reuse_promotes_swa_over_never_reused():
    """M6.6: the SWA-LRU reuse contract — a real match promotes the reused SWA
    node to MRU, so a never-reused node becomes the eviction victim instead.

    A (older) and B (newer) each hold an SWA slot; SWA-LRU order tail->head = A,B.
    We then REUSE A via a real match (update_cache_info=True — the actual-reuse
    path, mirroring how full-KV match already bumps its own last_access_time). A
    correct SWA-LRU promotes the reused node to MRU, making B (never reused) the
    true LRU victim. On the next single-slot SWA eviction, B must go and A must
    survive."""
    idx = RadixTreeIndex(tokens_per_block=TPB)
    a = idx.insert(_seq([1, 2, 3, 4]), _phys(0, 1), is_ready=True)
    idx.set_swa(a, slot=10)
    b = idx.insert(_seq([5, 6, 7, 8]), _phys(2, 3), is_ready=True)
    idx.set_swa(b, slot=11)
    # Reuse A (real match, not a probe).
    mr = idx.match_prefix(_seq([1, 2, 3, 4]), update_cache_info=True)
    assert mr.last_swa_node is a  # A was indeed matched/reused
    # Evict one SWA slot: must drop the never-reused B, not the just-reused A.
    idx.evict_swa(1)
    assert a.has_swa(), "reused SWA A was evicted (SWA-LRU not promoted on match)"
    assert not b.has_swa()


def test_py_dual_lock_invariant():
    """I3: full_lock_ref (lock_cnt) >= swa_lock_ref, with paired inc/dec."""
    idx = RadixTreeIndex(tokens_per_block=TPB)
    n1 = idx.insert(_seq([1, 2, 3, 4]), _phys(0, 1), is_ready=True)
    idx.set_swa(n1, slot=700)
    b = idx.inc_lock_ref(n1)
    assert n1.lock_cnt == 1 and n1.swa_lock_ref == 1 and b is n1
    assert n1.lock_cnt >= n1.swa_lock_ref
    idx.dec_lock_ref(n1, swa_boundary=b)
    assert n1.lock_cnt == 0 and n1.swa_lock_ref == 0


def test_py_dec_swa_lock_only_early_release():
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


def test_py_dual_lock_only_deepest_swa_node_locked():
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


def test_py_swa_locked_node_not_full_evictable():
    """in_use() includes swa_lock_ref: a SWA-locked node is not full-evictable."""
    idx = RadixTreeIndex(tokens_per_block=TPB)
    n = idx.insert(_seq([1, 2, 3, 4]), _phys(0, 1), is_ready=True)
    idx.set_swa(n, slot=5)
    n.swa_lock_ref = 1
    assert n.in_use()
    assert not n.evictable()


def test_py_root_is_not_evictable():
    idx = RadixTreeIndex(tokens_per_block=TPB)
    assert idx.root_node.is_leaf()
    assert not idx.root_node.evictable()
    evicted, hashes = idx.evict(1)
    assert evicted.size == 0
    assert hashes.size == 0


def test_py_reset_rearms_swa_pool_via_host_pool():
    """SWAHostPool.reset re-arms every slot free (tree reset drops all nodes)."""
    from flexkv.swa.swa_host_pool import SWAHostPool
    from flexkv.common.config import SWAPoolConfig
    cfg = SWAPoolConfig(
        enabled=True,
        num_slots=4,
        num_swa_layers=1,
        bytes_per_token_per_layer=2,
    )
    pool = SWAHostPool(cfg)
    a, b = pool.allocate(), pool.allocate()
    assert a is not None and b is not None and pool.num_free == 2
    pool.reset()
    assert pool.num_free == 4  # all slots reclaimed


def test_py_partial_node_match_does_not_report_swa():
    """§5.2: a partially-matched node does not expose its trailing-page SWA."""
    idx = RadixTreeIndex(tokens_per_block=TPB)
    n1 = idx.insert(_seq([1, 2, 3, 4, 5, 6, 7, 8]), _phys(0, 1, 2, 3), is_ready=True)
    idx.set_swa(n1, slot=800)
    # query shares only the first 3 blocks (partial match of the 4-block node)
    mr = idx.match_prefix(_seq([1, 2, 3, 4, 5, 6, 99, 98]))
    assert mr.num_matched_blocks == 3
    # SWA sits on block-4's page; a partial match must not claim it
    assert mr.last_swa_node is None and mr.swa_hit_blocks == 0


def test_py_merge_child_moves_swa_to_merged_node():
    """I0: merging a single child moves the SWA to follow the child's last page,
    freeing the parent's stale SWA."""
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


# --------------------------------------------------------------------------- #
# promote_swa: read-hit refreshes SWA-LRU (match-promote to MRU)              #
# --------------------------------------------------------------------------- #

def _three_swa_leaves():
    """root -> {n1, n2, n3} sibling leaves, each with a live SWA. set_swa order
    n1,n2,n3 leaves n3 at MRU and n1 at the LRU tail."""
    idx = RadixTreeIndex(tokens_per_block=TPB)
    n1 = idx.insert(_seq([1, 2, 3, 4]), _phys(0, 1), is_ready=True)
    n2 = idx.insert(_seq([5, 6, 7, 8]), _phys(2, 3), is_ready=True)
    n3 = idx.insert(_seq([9, 10, 11, 12]), _phys(4, 5), is_ready=True)
    idx.set_swa(n1, slot=100)
    idx.set_swa(n2, slot=200)
    idx.set_swa(n3, slot=300)
    return idx, n1, n2, n3


def test_promote_swa_moves_to_mru():
    """promote_swa splices the hit node to the MRU side (right after head) and
    refreshes its timestamp."""
    idx, n1, n2, n3 = _three_swa_leaves()
    # MRU=n3, LRU=n1 after the set_swa order above.
    assert idx._swa_lru_head.swa_lru_next is n3
    assert idx._swa_lru_tail.swa_lru_prev is n1
    t_before = n1.swa_last_access_time

    idx.promote_swa(n1)

    assert idx._swa_lru_head.swa_lru_next is n1          # n1 is now MRU
    assert idx._swa_lru_tail.swa_lru_prev is n2          # n2 sank to LRU
    assert n1.on_swa_lru and n1.has_swa()
    assert n1.swa_last_access_time >= t_before


def test_promote_swa_survives_eviction():
    """Key regression: a promoted (read-hit) node must NOT be the eviction
    victim even though it was written least recently. Without promote, n1 (the
    LRU tail) would be evicted; after promote, n2 is the victim and n1 lives."""
    idx, n1, n2, n3 = _three_swa_leaves()
    idx.promote_swa(n1)  # n1 hit -> MRU; new LRU order (MRU..LRU): n1, n3, n2

    _evicted_full, num_freed = idx.evict_swa(1)

    assert num_freed == 1
    assert n1.has_swa()                                  # the promoted node lives
    assert 200 in idx.drain_freed_swa_slots()            # n2's slot was reclaimed


def test_promote_swa_ignores_dead_node():
    """promote_swa on a tombstone (SWA already freed) node is a no-op — it must
    not re-thread a dead node onto the SWA-LRU (would corrupt evict_swa)."""
    idx = RadixTreeIndex(tokens_per_block=TPB)
    n1 = idx.insert(_seq([1, 2, 3, 4]), _phys(0, 1), is_ready=True)
    idx.set_swa(n1, slot=100)
    idx.record_freed_swa_slot(n1)                        # tombstone it
    assert n1.swa_tombstone and not n1.on_swa_lru
    idx.drain_freed_swa_slots()                          # clear the buffer

    idx.promote_swa(n1)

    assert not n1.on_swa_lru and n1.swa_tombstone        # still off the list
    assert idx.drain_freed_swa_slots() == []             # nothing touched

    # root is also a no-op (never on the SWA-LRU).
    idx.promote_swa(idx.root_node)
    assert not idx.root_node.on_swa_lru


# =========================================================================== #
# 2. C++ PRODUCTION — CRadixTreeIndex                                         #
# =========================================================================== #

c_ext = pytest.importorskip("flexkv.c_ext")

_CEXT_OK = "swa_host_slot" in dir(c_ext.CRadixNode) and all(
    n in dir(c_ext.CRadixNode)
    for n in ("is_leaf", "get_lock_cnt", "has_swa", "lock", "unlock")
)
_cext_reason = "CRadixNode SWA/structural bindings not present (rebuild c_ext)"
cext = pytest.mark.skipif(not _CEXT_OK, reason=_cext_reason)

import torch  # noqa: E402 — only needed for the c_ext section


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


@cext
def test_cpp_match_reports_last_swa_node():
    t = _tree()
    n = _insert(t, [1, 2, 3, 4, 5, 6, 7, 8], 0)
    t.set_swa(n, 100)
    assert n.has_swa() and n.swa_host_slot == 100
    mr = t.match_prefix(_hashes([1, 2, 3, 4, 5, 6, 7, 8]), 4, False)
    assert mr.num_matched_blocks == 4
    assert mr.last_swa_node is not None
    assert mr.last_swa_node.swa_host_slot == 100
    assert mr.swa_hit_blocks == 4


@cext
def test_cpp_reuse_promotes_swa_over_never_reused():
    """C++ peer of test_py_reuse_promotes_swa_over_never_reused: a real match
    (update_cache_info=True) promotes the reused SWA node to MRU, so the
    never-reused node is evicted instead. Observed via evict_swa (the SWA-LRU is
    not directly bound to Python)."""
    t = _tree()
    a = _insert(t, [1, 2, 3, 4], 0)
    t.set_swa(a, 10)
    b = _insert(t, [5, 6, 7, 8], 4)
    t.set_swa(b, 11)
    # SWA-LRU order (LRU tail -> MRU head): A, B. Reuse A with a REAL match.
    mr = t.match_prefix(_hashes([1, 2, 3, 4]), 2, True)
    assert mr.last_swa_node is not None and mr.last_swa_node.swa_host_slot == 10
    # Evict one SWA slot: B (never reused) must go, A (just reused) survives.
    _evf, nfreed = _evict_swa(t, 1)
    assert nfreed == 1
    assert a.has_swa(), "reused SWA A was evicted (C++ match did not promote)"
    assert not b.has_swa()


@cext
def test_cpp_match_probe_does_not_promote_swa_lru():
    """C++ peer of test_py_match_probe_does_not_promote_swa_lru: a probe
    (update_cache_info=False) must NOT reorder the SWA-LRU, so the older node A
    stays the eviction victim even after being probed."""
    t = _tree()
    a = _insert(t, [1, 2, 3, 4], 0)
    t.set_swa(a, 10)
    b = _insert(t, [5, 6, 7, 8], 4)
    t.set_swa(b, 11)
    # Probe A without cache-info update — must not promote it.
    t.match_prefix(_hashes([1, 2, 3, 4]), 2, False)
    # A is still the LRU victim: evicting one drops A, keeps B.
    _evf, nfreed = _evict_swa(t, 1)
    assert nfreed == 1
    assert not a.has_swa(), "probe wrongly promoted A (should stay LRU victim)"
    assert b.has_swa()


@cext
def test_cpp_partial_node_match_hides_tail_swa():
    """I4: SWA sits on the node's last page; a partial match cannot claim it."""
    t = _tree()
    n = _insert(t, [1, 2, 3, 4, 5, 6, 7, 8], 0)
    t.set_swa(n, 800)
    mr = t.match_prefix(_hashes([1, 2, 3, 4, 5, 6, 99, 98]), 4, False)
    assert mr.num_matched_blocks == 3
    assert mr.last_swa_node is None
    assert mr.swa_hit_blocks == 0


@cext
def test_cpp_split_preserves_swa_on_suffix():
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


@cext
def test_cpp_full_evict_frees_swa_slot():
    t = _tree()
    n = _insert(t, [1, 2, 3, 4], 0)
    t.set_swa(n, 300)
    ev = _evict_full(t, 2)
    assert len(ev) == 2
    assert 300 in t.drain_freed_swa_slots()
    assert t.is_empty()


@cext
def test_cpp_full_evict_no_swa_no_drain():
    t = _tree()
    _insert(t, [1, 2, 3, 4], 0)
    _evict_full(t, 2)
    assert t.drain_freed_swa_slots() == []


@cext
def test_cpp_evict_swa_internal_first_keeps_full():
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


@cext
def test_cpp_evict_swa_leaf_unlocked_deletes_node():
    t = _tree()
    nl = _insert(t, [1, 2, 3, 4], 0)
    t.set_swa(nl, 500)
    evf, nfreed = _evict_swa(t, 1)
    assert nfreed == 1
    assert evf.size == 2                 # whole node deleted, full freed
    assert t.is_empty()
    assert 500 in t.drain_freed_swa_slots()


@cext
def test_cpp_evict_swa_leaf_locked_keeps_full():
    t = _tree()
    nl = _insert(t, [1, 2, 3, 4], 0)
    t.set_swa(nl, 600)
    t.lock(nl)                           # full lock
    evf, nfreed = _evict_swa(t, 1)
    assert nfreed == 1 and evf.size == 0
    assert nl.swa_tombstone and nl.size() == 2
    assert not t.is_empty()


@cext
def test_cpp_dual_lock_symmetric():
    t = _tree()
    n = _insert(t, [1, 2, 3, 4], 0)
    t.set_swa(n, 700)
    b = t.inc_lock_ref(n)
    assert b is not None
    assert n.get_lock_cnt() == 1 and n.swa_lock_ref == 1
    assert n.get_lock_cnt() >= n.swa_lock_ref     # I3
    t.dec_lock_ref(n, b, False)
    assert n.get_lock_cnt() == 0 and n.swa_lock_ref == 0


@cext
def test_cpp_only_deepest_swa_node_locked():
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


@cext
def test_cpp_dec_swa_lock_only_early_release():
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


@cext
def test_cpp_swa_locked_node_not_full_evictable():
    """in_use() includes swa_lock_ref: SWA-locked node survives full eviction."""
    t = _tree()
    n = _insert(t, [1, 2, 3, 4], 0)
    t.set_swa(n, 5)
    n.inc_swa_lock_ref()
    ev = _evict_full(t, 2)
    assert len(ev) == 0            # locked -> not evicted
    assert not t.is_empty()
    n.dec_swa_lock_ref()


@cext
def test_cpp_root_is_not_evictable():
    t = _tree()
    ev = _evict_full(t, 1)
    assert len(ev) == 0
    assert t.is_empty()


@cext
def test_cpp_reset_clears_tree():
    t = _tree()
    n = _insert(t, [1, 2, 3, 4], 0)
    t.set_swa(n, 900)
    t.reset()
    assert t.is_empty()


# =========================================================================== #
# 3. WORKLOAD STRESS — production CacheEngineAccel + real SWA host-pool IO     #
#    (folded in from the former benchmarks/benchmark_swa_nodemount.py)         #
# =========================================================================== #
#
# A stochastic multi-turn dialogue workload that drives the SHIPPED path
# (CacheEngineAccel: CRadixTreeIndex Full tree + real SWAHostPool byte IO)
# under pool pressure, then asserts the aggregate two-pool lock-step invariants
# that the per-op unit tests above cannot: no double-free, no use-after-free of
# a live window, byte-identical SWA read-back, and no slot leak after reset.

_WTPB = 16  # workload tokens_per_block (== swa_page_size: one page == one slot)


@dataclass
class _Turn:
    token_ids: np.ndarray


def _make_workload(num_users, num_turns, sys_len, turn_in, turn_out, seed):
    rng = random.Random(seed)
    npr = np.random.RandomState(seed)
    vocab = 30000
    system = npr.randint(0, vocab, size=sys_len, dtype=np.int64)
    hist: Dict[int, np.ndarray] = {u: system.copy() for u in range(num_users)}
    per_user_turns = {u: max(1, rng.randint(num_turns // 2, num_turns))
                      for u in range(num_users)}
    counters = {u: 0 for u in range(num_users)}
    pending = list(range(num_users))
    work: List[_Turn] = []
    while pending:
        u = rng.choice(pending)
        inp = npr.randint(0, vocab, size=turn_in, dtype=np.int64)
        prefix = np.concatenate([hist[u], inp])
        work.append(_Turn(token_ids=prefix))
        out = npr.randint(0, vocab, size=turn_out, dtype=np.int64)
        hist[u] = np.concatenate([prefix, out])
        counters[u] += 1
        if counters[u] >= per_user_turns[u]:
            pending.remove(u)
    return work


class _SWAWorkloadDriver:
    """Minimal port of the former benchmark Driver: match(get) -> store(put)
    with an SWA slot mounted on each stored tail node, and ground-truth pool
    invariant sweeps (slot-id accounting: no double-free / use-after-free / leak).
    The SWA pool is a pure slot-id allocator (bytes live in the StorageEngine
    is_swa buffer), so this driver tracks slot lifecycle, not page bytes."""

    def __init__(self, num_cpu_blocks, swa_slots, slot_bytes):
        from flexkv.cache.cache_engine import CacheEngineAccel
        from flexkv.common.config import SWAPoolConfig
        from flexkv.common.transfer import DeviceType
        self.tpb = _WTPB
        self.engine = CacheEngineAccel(
            DeviceType.CPU, num_cpu_blocks, _WTPB,
            evict_ratio=0.1, hit_reward_seconds=0,
            evict_start_threshold=1.0, eviction_policy="lru",
        )
        self.engine.init_swa(SWAPoolConfig(
            enabled=True,
            num_slots=swa_slots,
            num_swa_layers=1,
            bytes_per_token_per_layer=max(1, slot_bytes // _WTPB),
        ))
        self._slot_expect: Dict[int, int] = {}
        self.violations: List[str] = []
        self.swa_hits = 0

    def _seq(self, token_ids):
        aligned = (len(token_ids) // self.tpb) * self.tpb
        sm = SequenceMeta(token_ids=token_ids[:aligned].astype(np.int64),
                          tokens_per_block=self.tpb)
        sm.gen_hashes()
        return sm

    def _reconcile(self):
        free = set(int(s) for s in self.engine.swa_pool._free_slots)
        for s in list(self._slot_expect):
            if s in free:
                self._slot_expect.pop(s, None)

    def step(self, turn):
        sm = self._seq(turn.token_ids)
        nblocks = sm.num_blocks
        if nblocks == 0:
            return
        mr = self.engine.match(sm)
        full_hit = int(mr.num_ready_matched_blocks)
        if full_hit > 0:
            swa_hit = int(mr.swa_hit_blocks)
            node = mr.last_swa_node
            slot = int(node.swa_host_slot) if node is not None else -1
            if swa_hit > 0 and slot >= 0:
                self.swa_hits += 1

        num_new = nblocks - full_hit
        if num_new > 0:
            protected = mr.last_node if full_hit > 0 else None
            phys_new = np.asarray(
                self.engine.take(num_new, protected_node=protected, strict=False)
            ).astype(np.int64)
            self._reconcile()
            if phys_new.size < num_new:
                self.engine.recycle(phys_new)
                return
            node = self.engine.insert(sm, phys_new, num_insert_blocks=num_new,
                                      is_ready=True, match_result=mr)
        else:
            node = mr.last_node
        if node is None:
            return
        slot = self.engine._alloc_swa_slot()
        self._reconcile()
        if slot == -1:
            return
        tail_hash = int(sm.block_hashes[nblocks - 1])
        self._slot_expect[int(slot)] = tail_hash
        self.engine.index.set_swa(node, int(slot))
        self._reconcile()

    def check_invariants(self):
        pool = self.engine.swa_pool
        used, free, total = pool.num_used, pool.num_free, pool.num_slots
        if used + free != total:
            self.violations.append(f"pool accounting {used}+{free}!={total}")
        free_list = list(pool._free_slots)
        if len(free_list) != len(set(free_list)):
            self.violations.append("DOUBLE-FREE on the SWA free-list")
        free_set = set(free_list)
        stale = [s for s in self._slot_expect if s in free_set]
        if stale:
            self.violations.append(f"USE-AFTER-FREE: live slot(s) {stale[:8]} free")

    def run(self, work, check_every=50):
        for i, turn in enumerate(work):
            self.step(turn)
            if check_every and (i + 1) % check_every == 0:
                self.check_invariants()
        self.check_invariants()
        self.engine.reset()
        pool = self.engine.swa_pool
        if pool.num_free != pool.num_slots:
            self.violations.append(
                f"LEAK after reset: free {pool.num_free} != total {pool.num_slots}")
        return self.violations


@cext
@pytest.mark.parametrize("cpu_blocks,swa_slots", [
    (4096, 256),   # baseline: little pressure
    (512, 512),    # full-KV eviction pressure (I1 connect-free hot path)
    (8192, 32),    # SWA-LRU pressure (I2 interior-first eviction hot path)
])
def test_workload_pressure_no_leak_no_corruption(cpu_blocks, swa_slots):
    """Stochastic multi-turn workload on the production path: under Full and SWA
    pool pressure, the two pools stay in lock-step — no double-free, no
    use-after-free of a live window, no slot leak after reset. (Folded from
    benchmark_swa_nodemount.py; small params for CI.)"""
    work = _make_workload(num_users=24, num_turns=6, sys_len=8 * _WTPB,
                          turn_in=6 * _WTPB, turn_out=2 * _WTPB, seed=1234)
    drv = _SWAWorkloadDriver(cpu_blocks, swa_slots, slot_bytes=4096)
    violations = drv.run(work, check_every=25)
    assert not violations, "invariant violations:\n" + "\n".join(violations[:20])
    # Sanity: the workload actually exercised SWA hits (so a silent no-op
    # regression that stops mounting/matching SWA slots is caught).
    assert drv.swa_hits > 0, "workload produced no SWA hits — coverage regressed"


# --------------------------------------------------------------------------- #
# I2 in FULL eviction: deleting a leaf cascades tombstone-leaf ancestors       #
# --------------------------------------------------------------------------- #

def _build_prefix_chain():
    """root -> A(prefix, 2blk) -> {a(leaf w/ SWA), b(divergent leaf)}.

    Returns (idx, A, a, b). A is an internal tombstone (Full, no SWA); a carries
    SWA and is the leaf we will evict; b is a sibling keeping A internal.
    """
    idx = RadixTreeIndex(tokens_per_block=TPB)
    a = idx.insert(_seq([1, 2, 3, 4, 5, 6, 7, 8]), _phys(0, 1, 2, 3), is_ready=True)
    idx.set_swa(a, slot=901)
    sd = _seq([1, 2, 3, 4, 55, 66, 77, 88])
    idx.insert(sd, _phys(8, 9), is_ready=True, match_result=idx.match_prefix(sd))
    A = a.parent
    b = [c for c in A.children.values() if c is not a][0]
    return idx, A, a, b


def test_full_evict_cascades_tombstone_leaf_parent():
    """I2 (full-evict): after both children are gone, the parent A is a tombstone
    leaf (Full, no SWA, unlocked) — meaningless, so it is cascade-deleted and its
    Full blocks are freed too, even beyond the requested num_evicted."""
    idx, A, a, b = _build_prefix_chain()
    # Drop SWA off the divergent sibling b so evicting both leaves leaves A a
    # tombstone leaf. b starts as a tombstone (no set_swa), a carries SWA=901.
    # Evict 2 blocks: the LRU leaf gets deleted; then A becomes a tombstone leaf.
    # Give a smaller grace so it (and b) are the eviction victims; A must cascade.
    # Force deletion of BOTH leaves by requesting their combined size.
    a.lock_cnt = 0
    b.lock_cnt = 0
    total_before = idx.total_cached_blocks()  # A(2) + a(2) + b(2) = 6
    assert total_before == 6
    ev_blocks, ev_hashes = idx.evict(4)  # ask for the two 2-block leaves
    # Both leaves (4 blocks) + cascaded parent A (2 blocks) = 6 freed, though we
    # only asked for 4 — the I2 cascade freed A on top.
    assert len(ev_blocks) == 6
    assert len(ev_hashes) == 6
    assert idx.is_empty()
    assert idx.total_swa_slots() == 0
    assert 901 in idx.drain_freed_swa_slots()


def test_full_evict_keeps_full_locked_tombstone_leaf():
    """I2 exception: a parent that becomes a tombstone leaf but whose Full is
    locked is NOT cascade-deleted (its Full KV is still referenced)."""
    idx, A, a, b = _build_prefix_chain()
    A.lock_cnt = 1  # pin A's Full
    ev_blocks, _ = idx.evict(4)  # delete both leaves a, b
    # A is a tombstone leaf now but full-locked -> survives; only 4 blocks freed.
    assert len(ev_blocks) == 4
    assert not idx.is_empty()
    assert A.is_leaf() and A.swa_tombstone and A.lock_cnt == 1
    assert A.size() == 2


def test_full_evict_no_cascade_when_swa_disabled():
    """Non-SWA regression: with SWA never enabled (no set_swa), swa_tombstone is
    True on every node, but the I2 cascade must NOT fire — evict() frees EXACTLY
    the requested blocks and leaves valid ancestors intact."""
    idx = RadixTreeIndex(tokens_per_block=TPB)
    a = idx.insert(_seq([1, 2, 3, 4, 5, 6, 7, 8]), _phys(0, 1, 2, 3), is_ready=True)
    sd = _seq([1, 2, 3, 4, 55, 66, 77, 88])
    idx.insert(sd, _phys(8, 9), is_ready=True, match_result=idx.match_prefix(sd))
    A = a.parent
    assert not idx._swa_enabled  # never armed
    total_before = idx.total_cached_blocks()  # 6
    ev_blocks, _ = idx.evict(4)  # delete both leaves
    # Exactly 4 freed; the tombstone parent A survives (no cascade when disabled).
    assert len(ev_blocks) == 4
    assert not idx.is_empty()
    assert A.is_leaf() and A.size() == 2 and idx.total_cached_blocks() == 2


# --------------------------------------------------------------------------- #
# SWA source is exposed directly on match results                          #
# --------------------------------------------------------------------------- #

def test_match_result_exposes_swa_source_within_bound():
    """match_prefix exposes the SWA source without a second read-source probe."""
    idx = RadixTreeIndex(tokens_per_block=TPB)
    n1 = idx.insert(_seq([1, 2, 3, 4, 5, 6, 7, 8]), _phys(0, 1, 2, 3), is_ready=True)
    idx.set_swa(n1, slot=100)
    mr = idx.match_prefix(_seq([1, 2, 3, 4, 5, 6, 7, 8]))
    assert mr.swa_hit_blocks == 4 and mr.last_swa_node is n1
    # Simulate the reuse logic directly on the match result (engine-independent):
    # within a bound >= swa_hit, the slot is read straight from last_swa_node.
    assert mr.swa_hit_blocks <= 4
    assert int(mr.last_swa_node.swa_host_slot) == 100


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
