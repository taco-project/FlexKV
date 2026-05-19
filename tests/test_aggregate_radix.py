"""Unit tests for ``flexkv.cache.aggregate_radix`` (Phase 0 task 0-H)."""
from __future__ import annotations

from typing import List

import pytest

from flexkv.common.dist_reuse.aggregate_radix import (
    AggregateRadixTree,
    BlockNotTrackedError,
)


# ---------------------------------------------------------------------------
# Manual clock for deterministic time-based assertions
# ---------------------------------------------------------------------------
class _ManualClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


# ---------------------------------------------------------------------------
# mark_sd_ready / fully-ready transition
# ---------------------------------------------------------------------------
class TestMarkSdReady:
    def test_single_sd_immediately_ready(self):
        agg = AggregateRadixTree(total_sd_count=1)
        became = agg.mark_sd_ready(prefix_hash=0xAA, sd_key="sd0", block_ids=[1, 2])
        assert became is True
        result = agg.match_fully_ready(0xAA)
        assert result is not None
        assert result.block_ids == (1, 2)
        # ready_sds is now a Dict[str, int] (sd_key -> contributing node_id).
        # Default node_id is -1 when caller doesn't pass one.
        assert set(result.ready_sds) == {"sd0"}
        assert result.ready_sds == {"sd0": -1}

    def test_multi_sd_progressive(self):
        agg = AggregateRadixTree(total_sd_count=3)
        for sd in ("sd0", "sd1"):
            assert agg.mark_sd_ready(0xBB, sd, [10, 20]) is False
        # Not yet fully ready.
        assert agg.match_fully_ready(0xBB) is None
        # Last SD acks → transitions to fully ready.
        became = agg.mark_sd_ready(0xBB, "sd2", [10, 20])
        assert became is True
        result = agg.match_fully_ready(0xBB)
        assert result is not None
        assert set(result.ready_sds) == {"sd0", "sd1", "sd2"}

    def test_idempotent_per_sd(self):
        agg = AggregateRadixTree(total_sd_count=2)
        agg.mark_sd_ready(1, "sd0", [1])
        # Re-acking sd0 must NOT count as a transition.
        assert agg.mark_sd_ready(1, "sd0", [1]) is False
        assert agg.match_fully_ready(1) is None
        # sd1 transitions.
        assert agg.mark_sd_ready(1, "sd1", [1]) is True

    def test_block_id_mismatch_raises(self):
        agg = AggregateRadixTree(total_sd_count=2)
        agg.mark_sd_ready(1, "sd0", [1, 2])
        with pytest.raises(ValueError):
            agg.mark_sd_ready(1, "sd1", [1, 99])  # different block layout

    def test_invalid_sd_key(self):
        agg = AggregateRadixTree(total_sd_count=1)
        with pytest.raises(ValueError):
            agg.mark_sd_ready(1, "", [1])

    def test_contributing_peer_recorded(self):
        agg = AggregateRadixTree(total_sd_count=2)
        agg.mark_sd_ready(1, "sd0", [1], contributing_peer="peer-A")
        agg.mark_sd_ready(1, "sd1", [1], contributing_peer="peer-A")
        result = agg.match_fully_ready(1)
        assert result is not None
        assert result.contributing_peers == {"peer-A"}

    def test_node_id_recorded_per_sd(self):
        """Per-SD node_id schema (multi-SD GET-path support).

        Each SD ack carries the contributing peer's node_id so the
        GET-path knows which Mooncake server to RDMA-READ from for
        that SD's slice.  Master SD typically passes its own node_id;
        peer SDs pass the node_id from
        ``CompletedOp.contributing_node_id`` (Phase D-2 graph-dispatch
        PUT path).
        """
        agg = AggregateRadixTree(total_sd_count=3)
        agg.mark_sd_ready(1, "sd0", [10, 20], node_id=100)
        agg.mark_sd_ready(1, "sd1", [10, 20], node_id=200)
        agg.mark_sd_ready(1, "sd2", [10, 20], node_id=300)
        result = agg.match_fully_ready(1)
        assert result is not None
        assert result.ready_sds == {"sd0": 100, "sd1": 200, "sd2": 300}
        # node_id_for_sd helper returns the per-SD node_id.
        assert result.node_id_for_sd("sd0") == 100
        assert result.node_id_for_sd("sd2") == 300
        assert result.node_id_for_sd("nonexistent") is None

    def test_node_id_late_binding(self):
        """Idempotent re-acks with a real node_id must overwrite the
        sentinel ``-1`` (so callers can defer node_id resolution)."""
        agg = AggregateRadixTree(total_sd_count=2)
        # First ack — node_id unknown.
        agg.mark_sd_ready(1, "sd0", [10], node_id=-1)
        # Second ack — same SD, real node_id.  Idempotent transition,
        # but node_id should patch.
        assert agg.mark_sd_ready(1, "sd0", [10], node_id=42) is False
        # Other SD acks; prefix becomes fully ready.
        agg.mark_sd_ready(1, "sd1", [10], node_id=99)
        result = agg.match_fully_ready(1)
        assert result is not None
        assert result.node_id_for_sd("sd0") == 42  # patched
        assert result.node_id_for_sd("sd1") == 99

    def test_node_id_real_value_not_overwritten_by_sentinel(self):
        """Once a real node_id is recorded, a subsequent ``-1`` ack
        must NOT clobber it."""
        agg = AggregateRadixTree(total_sd_count=1)
        agg.mark_sd_ready(1, "sd0", [10], node_id=42)
        agg.mark_sd_ready(1, "sd0", [10], node_id=-1)
        result = agg.match_fully_ready(1)
        assert result is not None
        assert result.node_id_for_sd("sd0") == 42


# ---------------------------------------------------------------------------
# mark_sd_evicted
# ---------------------------------------------------------------------------
class TestMarkSdEvicted:
    def test_evict_single_sd_drops_to_partial(self):
        agg = AggregateRadixTree(total_sd_count=2)
        agg.mark_sd_ready(1, "sd0", [10])
        agg.mark_sd_ready(1, "sd1", [10])
        assert agg.match_fully_ready(1) is not None
        agg.mark_sd_evicted(1, "sd0")
        assert agg.match_fully_ready(1) is None  # no longer fully ready

    def test_evict_last_sd_drops_entry(self):
        agg = AggregateRadixTree(total_sd_count=2)
        agg.mark_sd_ready(1, "sd0", [10])
        agg.mark_sd_evicted(1, "sd0")
        # No SDs left → entry is gone, not just "partial"
        assert 1 not in agg.known_prefixes()

    def test_evict_unknown_prefix_is_noop(self):
        agg = AggregateRadixTree(total_sd_count=1)
        agg.mark_sd_evicted(99, "sd0")  # must not raise


# ---------------------------------------------------------------------------
# Refcount lifecycle
# ---------------------------------------------------------------------------
class TestRefcount:
    def test_acquire_release_cycle(self):
        agg = AggregateRadixTree(total_sd_count=1)
        assert agg.is_evictable(7) is True
        agg.acquire([7])
        assert agg.is_evictable(7) is False
        assert agg.get_refcount(7) == 1
        agg.release([7])
        assert agg.is_evictable(7) is True
        assert agg.get_refcount(7) == 0

    def test_acquire_increments(self):
        agg = AggregateRadixTree(total_sd_count=1)
        agg.acquire([1, 1, 1])
        assert agg.get_refcount(1) == 3
        agg.release([1, 1])
        assert agg.get_refcount(1) == 1
        assert agg.is_evictable(1) is False

    def test_double_release_raises(self):
        agg = AggregateRadixTree(total_sd_count=1)
        agg.acquire([1])
        agg.release([1])
        with pytest.raises(BlockNotTrackedError):
            agg.release([1])

    def test_release_untracked_raises(self):
        agg = AggregateRadixTree(total_sd_count=1)
        with pytest.raises(BlockNotTrackedError):
            agg.release([42])


# ---------------------------------------------------------------------------
# Leak scanner
# ---------------------------------------------------------------------------
class TestLeakScanner:
    def test_finds_leaked_blocks(self):
        clock = _ManualClock()
        agg = AggregateRadixTree(total_sd_count=1, time_fn=clock)
        agg.acquire([1])
        clock.advance(40.0)
        leaked = agg.scan_leaked_refcount(timeout_seconds=30.0)
        assert leaked == [1]

    def test_fresh_acquires_not_leaked(self):
        clock = _ManualClock()
        agg = AggregateRadixTree(total_sd_count=1, time_fn=clock)
        agg.acquire([1])
        leaked = agg.scan_leaked_refcount(timeout_seconds=30.0)
        assert leaked == []

    def test_force_release_clears_refcount(self):
        clock = _ManualClock()
        agg = AggregateRadixTree(total_sd_count=1, time_fn=clock)
        agg.acquire([1, 1, 1])
        prev = agg.force_release(1)
        assert prev == 3
        assert agg.is_evictable(1) is True

    def test_force_release_unknown_returns_zero(self):
        agg = AggregateRadixTree(total_sd_count=1)
        assert agg.force_release(999) == 0

    def test_negative_timeout_raises(self):
        agg = AggregateRadixTree(total_sd_count=1)
        with pytest.raises(ValueError):
            agg.scan_leaked_refcount(-1.0)


# ---------------------------------------------------------------------------
# Invalidation
# ---------------------------------------------------------------------------
class TestInvalidation:
    def test_invalidate_prefix(self):
        agg = AggregateRadixTree(total_sd_count=1)
        agg.mark_sd_ready(1, "sd0", [10])
        assert agg.invalidate_prefix(1) is True
        assert agg.match_fully_ready(1) is None
        # Idempotent
        assert agg.invalidate_prefix(1) is False

    def test_invalidate_by_peer(self):
        agg = AggregateRadixTree(total_sd_count=2)
        # Two prefixes from peer-A
        agg.mark_sd_ready(1, "sd0", [10], contributing_peer="peer-A")
        agg.mark_sd_ready(1, "sd1", [10], contributing_peer="peer-A")
        agg.mark_sd_ready(2, "sd0", [20], contributing_peer="peer-A")
        agg.mark_sd_ready(2, "sd1", [20], contributing_peer="peer-A")
        # One prefix from peer-B
        agg.mark_sd_ready(3, "sd0", [30], contributing_peer="peer-B")
        agg.mark_sd_ready(3, "sd1", [30], contributing_peer="peer-B")

        n = agg.invalidate_by_peer_instance("peer-A")
        assert n == 2
        assert agg.match_fully_ready(1) is None
        assert agg.match_fully_ready(2) is None
        assert agg.match_fully_ready(3) is not None  # peer-B prefix survives

    def test_invalidate_by_unknown_peer(self):
        agg = AggregateRadixTree(total_sd_count=1)
        agg.mark_sd_ready(1, "sd0", [10], contributing_peer="peer-A")
        assert agg.invalidate_by_peer_instance("peer-Z") == 0

    def test_invalidate_by_empty_peer_raises(self):
        agg = AggregateRadixTree(total_sd_count=1)
        with pytest.raises(ValueError):
            agg.invalidate_by_peer_instance("")


# ---------------------------------------------------------------------------
# Constructor
# ---------------------------------------------------------------------------
class TestConstructor:
    @pytest.mark.parametrize("bad", [0, -1, "1", 1.5])
    def test_bad_total_sd_count(self, bad):
        with pytest.raises(ValueError):
            AggregateRadixTree(total_sd_count=bad)  # type: ignore[arg-type]

    def test_total_sd_count_property(self):
        agg = AggregateRadixTree(total_sd_count=5)
        assert agg.total_sd_count == 5


# ---------------------------------------------------------------------------
# Concurrency smoke (light — full stress would slow CI)
# ---------------------------------------------------------------------------
def test_concurrent_acquire_release_does_not_corrupt():
    import threading

    agg = AggregateRadixTree(total_sd_count=1)
    block_ids: List[int] = list(range(100))

    def worker():
        for _ in range(50):
            agg.acquire(block_ids)
            agg.release(block_ids)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All refcounts should be back to zero.
    for b in block_ids:
        assert agg.is_evictable(b) is True
