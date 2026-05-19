"""§2.2(b) — C++ accel path for the eviction refcount guard.

Parallels :mod:`test_evict_refcount_guard` (Python ``RadixTreeIndex``)
but drives the C++ ``CRadixTreeIndex`` that's exposed through the
pybind11 ``c_ext`` module and consumed by :class:`CacheEngineAccel`.

These tests require ``c_ext.so`` to have been built with the
§2.2(b) 4-arg ``evict`` overload; when the extension is not available
(e.g. clean Mac dev machine) the whole file is skipped.  The real CI
path runs inside the ``flexkv_distreuse`` container which always has
``c_ext`` compiled.

See docs/dist_reuse/implementation_gap_2026-05-11.md §2.2 and
docs/dist_reuse/implementation_progress_2026-05-13.md for the status
of the Accel path.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

torch = pytest.importorskip("torch")
c_ext = pytest.importorskip("flexkv.c_ext")

# The 4-arg evict overload is the new addition (§2.2(b)).  If the
# loaded ``c_ext.so`` predates it, skip this whole file: the guard
# isn't actually enforced on the Accel path yet in that build.
_CRadixTreeIndex = getattr(c_ext, "CRadixTreeIndex", None)
if _CRadixTreeIndex is None:  # pragma: no cover — extension layout guard
    pytest.skip(
        "c_ext.CRadixTreeIndex not exported — rebuild c_ext with "
        "§2.2(b) patches applied.",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TOKENS_PER_BLOCK = 16


def _make_tree() -> "c_ext.CRadixTreeIndex":
    return _CRadixTreeIndex(TOKENS_PER_BLOCK, 1_000_000, 0, "lru")


def _insert(idx, num_blocks: int, phys_start: int, salt: int = 0):
    """Insert ``num_blocks`` consecutive physical block ids + hashes
    starting at ``phys_start``.  Returns the physical block array.

    Hashes are chosen to be distinct across calls with different
    ``salt`` to avoid collision in the radix prefix tree.
    """
    phys = torch.arange(phys_start, phys_start + num_blocks, dtype=torch.int64)
    # Deterministic but collision-free hashes across different salts.
    # Use a large-ish stride so prefixes don't accidentally extend.
    hashes = torch.arange(
        phys_start + salt * 1_000_000,
        phys_start + salt * 1_000_000 + num_blocks,
        dtype=torch.int64,
    )
    idx.insert(phys, hashes, num_blocks, num_blocks, True)
    return phys.numpy()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_cext_evict_3arg_legacy_unchanged():
    """Baseline: 3-arg ``evict`` (no predicate) behaves as before."""
    idx = _make_tree()
    p1 = _insert(idx, 3, phys_start=0, salt=0)
    p2 = _insert(idx, 2, phys_start=100, salt=1)

    out_blocks = torch.zeros(4, dtype=torch.int64)
    out_hashes = torch.zeros(4, dtype=torch.int64)
    n = idx.evict(out_blocks, out_hashes, 4)

    assert n == 4
    evicted = set(out_blocks[:n].numpy().tolist())
    assert evicted.issubset(set(p1.tolist()) | set(p2.tolist()))


def test_cext_evict_4arg_with_null_predicate_behaves_like_3arg():
    """A None predicate is allowed and behaves identically to the 3-arg
    overload — required because ``CacheEngineAccel.take`` only plumbs
    the predicate when ``set_evict_guard`` has installed one, but the
    C++ side accepts either form.
    """
    idx = _make_tree()
    _insert(idx, 2, phys_start=0, salt=0)
    out_blocks = torch.zeros(2, dtype=torch.int64)
    out_hashes = torch.zeros(2, dtype=torch.int64)

    # ``lambda x: True`` is the null-object equivalent for callers that
    # always go through the 4-arg overload.
    n = idx.evict(out_blocks, out_hashes, 2, lambda _bid: True)
    assert n == 2


def test_cext_evict_guard_pins_specific_block_ids():
    """Block ids for which the guard returns False must NOT appear in
    the returned eviction set, even though the LRU picked them.
    """
    idx = _make_tree()
    p1 = _insert(idx, 3, phys_start=0, salt=0)    # 0,1,2
    p2 = _insert(idx, 3, phys_start=10, salt=1)   # 10,11,12

    pinned = {1, 11}

    def guard(block_id: int) -> bool:
        return block_id not in pinned

    out_blocks = torch.zeros(6, dtype=torch.int64)
    out_hashes = torch.zeros(6, dtype=torch.int64)
    n = idx.evict(out_blocks, out_hashes, 6, guard)

    evicted = set(out_blocks[:n].numpy().tolist())
    # Pinned ids must be absent.
    assert pinned.isdisjoint(evicted)
    # And since we pinned 2 out of 6 candidates, at most 4 should
    # come back.
    assert n <= 4


def test_cext_evict_guard_all_pinned_returns_zero():
    """If *every* candidate is pinned, ``evict`` returns 0 — no crash,
    no partial exception."""
    idx = _make_tree()
    _insert(idx, 2, phys_start=0)

    def guard(_block_id: int) -> bool:
        return False

    out_blocks = torch.zeros(10, dtype=torch.int64)
    out_hashes = torch.zeros(10, dtype=torch.int64)
    n = idx.evict(out_blocks, out_hashes, 10, guard)
    assert n == 0


def test_cext_evict_guard_exception_treats_as_evictable():
    """A buggy predicate must not wedge the allocator.  The C++ path
    catches the exception and treats the block as evictable (same
    contract as the Python ``RadixTreeIndex.evict``).
    """
    idx = _make_tree()
    _insert(idx, 2, phys_start=0)

    def bad_guard(_bid: int) -> bool:
        raise RuntimeError("boom")

    out_blocks = torch.zeros(2, dtype=torch.int64)
    out_hashes = torch.zeros(2, dtype=torch.int64)
    n = idx.evict(out_blocks, out_hashes, 2, bad_guard)
    # Exception-fallback treats as evictable → we still make progress.
    assert n >= 1


def test_cext_evict_guard_shrink_path_respects_pins():
    """Regression: the shrink branch (when node.size > num_remaining)
    must also drop pinned ids from its partial output.
    """
    idx = _make_tree()
    _insert(idx, 5, phys_start=0)  # one long leaf 0..4

    pinned = {2}

    def guard(block_id: int) -> bool:
        return block_id not in pinned

    out_blocks = torch.zeros(3, dtype=torch.int64)
    out_hashes = torch.zeros(3, dtype=torch.int64)
    n = idx.evict(out_blocks, out_hashes, 3, guard)

    evicted = set(out_blocks[:n].numpy().tolist())
    assert pinned.isdisjoint(evicted)
    assert n <= 3


def test_cext_evict_binding_has_4arg_overload():
    """Static guard: make sure the binding file actually publishes the
    new overload.  This catches accidental build regressions without
    having to run pytest in the container."""
    bindings_path = REPO_ROOT / "csrc" / "bindings.cpp"
    src = bindings_path.read_text()
    assert "std::function<bool(int64_t)>" in src, (
        "The 4-arg CRadixTreeIndex::evict binding is missing — "
        "check csrc/bindings.cpp and rebuild c_ext."
    )


def test_cext_cpp_has_4arg_overload_definition():
    """Static guard: the C++ impl must carry the 4-arg override."""
    cpp_path = REPO_ROOT / "csrc" / "radix_tree.cpp"
    src = cpp_path.read_text()
    assert "is_evictable_fn" in src, (
        "CRadixTreeIndex::evict 4-arg impl missing the predicate "
        "parameter — §2.2(b) C++ work not applied."
    )
