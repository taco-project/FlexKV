"""§2.2 — Eviction refcount guard (RadixTreeIndex layer).

These tests exercise the ``is_evictable_fn`` predicate that
:meth:`RadixTreeIndex.evict` now accepts.  The guard is how
``GlobalCacheEngine`` plumbs ``MasterCoordinator.is_evictable`` down
to the eviction path so that blocks pinned by an in-flight coord GET
(refcount > 0) are never recycled.

See docs/dist_reuse/implementation_gap_2026-05-11.md §2.2.

Kept dependency-free on purpose — only ``numpy`` + stdlib.  The
Python ``RadixTreeIndex`` is not coupled to torch / c_ext, so we can
import it directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Importing radixtree pulls in ``torch`` via the module header.  We
# skip the whole file when torch isn't available; the real test
# environment (`flexkv_distreuse` container) always has it.
torch = pytest.importorskip("torch")

from flexkv.cache.radixtree import RadixTreeIndex  # noqa: E402
from flexkv.common.block import SequenceMeta  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TOKENS_PER_BLOCK = 16


def _seq(tokens: int, salt: int = 0) -> SequenceMeta:
    """Build a minimal SequenceMeta with deterministic but distinct
    token ids (salted so that different inserts don't collide in the
    radix tree)."""
    token_ids = np.arange(tokens, dtype=np.int64) + salt * 1_000_000
    return SequenceMeta(
        token_ids=token_ids,
        tokens_per_block=TOKENS_PER_BLOCK,
    )


def _insert(idx: RadixTreeIndex, token_count: int, phys_start: int) -> np.ndarray:
    """Insert ``token_count`` tokens (multiple of TOKENS_PER_BLOCK)
    mapping to consecutive physical block ids starting at
    ``phys_start``.  Returns the physical block array for later
    cross-checking.
    """
    assert token_count % TOKENS_PER_BLOCK == 0
    num_blocks = token_count // TOKENS_PER_BLOCK
    phys = np.arange(phys_start, phys_start + num_blocks, dtype=np.int64)
    sm = _seq(token_count, salt=phys_start)
    sm.gen_hashes()
    idx.insert(sm, phys, num_insert_blocks=num_blocks, is_ready=True)
    return phys


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_evict_without_guard_is_legacy_behaviour():
    """Default call ``evict(n)`` (no predicate) must behave exactly as
    before — zero regression for all existing callers."""
    idx = RadixTreeIndex(tokens_per_block=TOKENS_PER_BLOCK)
    p1 = _insert(idx, TOKENS_PER_BLOCK * 3, phys_start=0)     # blocks 0..2
    p2 = _insert(idx, TOKENS_PER_BLOCK * 2, phys_start=100)   # blocks 100..101

    evicted, _ = idx.evict(4)
    assert len(evicted) == 4
    # The evicted set must be a subset of the blocks we inserted.
    assert set(evicted.tolist()).issubset(set(p1.tolist()) | set(p2.tolist()))


def test_evict_guard_pins_blocks_with_refcount_gt_zero():
    """When the predicate returns False for a subset of block ids,
    those ids must NOT appear in the returned ``evicted_blocks``."""
    idx = RadixTreeIndex(tokens_per_block=TOKENS_PER_BLOCK)
    p1 = _insert(idx, TOKENS_PER_BLOCK * 3, phys_start=0)      # 0,1,2
    p2 = _insert(idx, TOKENS_PER_BLOCK * 3, phys_start=10)     # 10,11,12

    pinned = {1, 11}

    def guard(block_id: int) -> bool:
        return block_id not in pinned

    evicted, _ = idx.evict(6, is_evictable_fn=guard)
    # None of the pinned blocks should ever be evicted.
    assert pinned.isdisjoint(set(evicted.tolist()))
    # The guard does not increase capacity; at most we can return
    # ``total_blocks - len(pinned)`` = 4 blocks.
    assert len(evicted) <= 4


def test_evict_guard_all_pinned_returns_empty():
    """If every candidate is pinned, ``evict`` must return an empty
    array (no crash, no partial exception)."""
    idx = RadixTreeIndex(tokens_per_block=TOKENS_PER_BLOCK)
    _insert(idx, TOKENS_PER_BLOCK * 2, phys_start=0)

    def guard(_block_id: int) -> bool:
        return False

    evicted, block_hashes = idx.evict(10, is_evictable_fn=guard)
    assert evicted.size == 0
    assert block_hashes.size == 0


def test_evict_guard_exception_falls_back_to_allow():
    """A buggy guard must not wedge eviction.  Design intent: defensive
    fallback logs + treats the block as evictable (safer than dead-
    locking the allocator).
    """
    idx = RadixTreeIndex(tokens_per_block=TOKENS_PER_BLOCK)
    _insert(idx, TOKENS_PER_BLOCK * 2, phys_start=0)

    def bad_guard(_block_id: int) -> bool:
        raise RuntimeError("boom")

    evicted, _ = idx.evict(1, is_evictable_fn=bad_guard)
    # Exception-fallback treats blocks as evictable, so we still get a
    # non-empty eviction set.
    assert len(evicted) >= 1


def test_evict_guard_shrink_path_respects_pins():
    """Regression: the "node.size() > remaining" branch (``shrink``)
    must also drop pinned ids from its partial output."""
    idx = RadixTreeIndex(tokens_per_block=TOKENS_PER_BLOCK)
    # One long leaf — eviction will hit the shrink branch on it.
    p = _insert(idx, TOKENS_PER_BLOCK * 5, phys_start=0)  # 0..4

    # Pin block id 2 (in the middle).
    pinned = {2}

    def guard(block_id: int) -> bool:
        return block_id not in pinned

    evicted, _ = idx.evict(3, is_evictable_fn=guard)
    assert pinned.isdisjoint(set(evicted.tolist()))
    # And we should still get up to 3 of the remaining 4 evictable
    # blocks (whichever ones shrink picks).
    assert len(evicted) <= 3


def test_evict_guard_kwarg_is_optional_for_callers():
    """Call sites that haven't been ported yet must keep working —
    ``is_evictable_fn`` is a keyword-only optional parameter."""
    idx = RadixTreeIndex(tokens_per_block=TOKENS_PER_BLOCK)
    _insert(idx, TOKENS_PER_BLOCK, phys_start=0)
    # No predicate kw at all.
    evicted, _ = idx.evict(1)
    assert len(evicted) == 1
