"""Aggregate-layer radix tree for cross-SD KV reuse consistency.

Design doc §4.3 (option B, "fully-ready aggregate radix") and §4.3.1
(refcount-protected eviction).  This module is the **central truth** about
which token prefixes have been confirmed by every SD in the instance and
are therefore safe to expose as remote-hit candidates.

Implementation notes
--------------------

* The aggregate radix is a *flat hash map keyed by leaf block hash*, not a
  full radix tree — design doc §4.3 only requires per-prefix readiness +
  per-block refcount, both of which collapse to a flat map once the per-SD
  ack count hits ``total_sd_count``.  A genuine tree adds no semantic value
  here; the *legacy* radix in ``RefRadixTree`` already handles longest-prefix
  match.
* All public methods are thread-safe via a single ``RLock``.  Concurrency
  pressure is low (a few hundred ack/release events per second), so a
  finer-grained scheme would be premature optimization.
* The only Phase-0 consumers are unit tests — the actual wiring into
  ``GlobalCacheEngine`` is task 0-H "integration" and lands in Batch C.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, FrozenSet, Iterable, List, Optional, Set, Tuple


__all__ = [
    "ReadyEntry",
    "AggregateMatchResult",
    "AggregateRadixTree",
    "BlockNotTrackedError",
]


class BlockNotTrackedError(KeyError):
    """Raised when ``release()`` / ``mark_sd_ready()`` is called for a block
    that the aggregate radix has never seen."""


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------
@dataclass
class ReadyEntry:
    """Per-prefix bookkeeping.

    ``prefix_hash`` identifies a *leaf block hash* — i.e. the hash of the
    last block in some token prefix.  Two distinct prefixes that happen to
    share a leaf block hash collapse to a single entry, which is exactly
    what we want (the on-wire effect is identical).

    ``ready_sds`` was historically a ``Set[str]`` of acked SD keys.  As of
    the multi-SD GET-path work it is now a ``Dict[str, int]`` mapping each
    acked SD key → the *distributed_node_id* of the FlexKV instance that
    contributed that SD's slice.  For the Master's own SD the value is the
    Master's own node_id; for peer SDs it is whichever peer instance's
    Remote completed the per-SD D2H clone op and shipped back a
    ``CompletedOp(sd_key, contributing_node_id, success=True)`` through
    the graph-dispatch completion sink (Phase D-2 PUT path).  Knowing
    the per-SD node_id at GET time lets the cross-instance reuse path
    target each peer SD's Mooncake server independently — required
    when PP>1 splits the layers across machines (design doc §4.5 / §5.1).

    ``contributing_peers`` continues to track the **set** of peer-instance
    IDs whose blocks we pulled in to fill this prefix; the failure
    detector uses this to do O(N_prefix) batch invalidation when a peer
    dies (design doc §4.3.2 bullet 3).  This stays as a Set because
    invalidation is keyed by *instance* not SD.
    """

    prefix_hash: int
    block_ids: Tuple[int, ...]
    # sd_key_str -> contributing peer's distributed_node_id.  ``-1`` means
    # the Master's own node (we may not know our own node_id at the time
    # the self-SD ack lands — callers can update later via ``mark_sd_ready``
    # which is idempotent).
    ready_sds: Dict[str, int] = field(default_factory=dict)
    contributing_peers: Set[str] = field(default_factory=set)
    # Wall-clock time (seconds since epoch) when this prefix first became
    # fully-ready.  Used by leak detection to age out abandoned entries.
    first_ready_at: Optional[float] = None

    def is_fully_ready(self, total_sd_count: int) -> bool:
        return len(self.ready_sds) >= total_sd_count

    def node_id_for_sd(self, sd_key: str) -> Optional[int]:
        """Return the contributing peer's node_id for a given SD, or None
        if that SD has not acked yet."""
        return self.ready_sds.get(sd_key)


@dataclass
class AggregateMatchResult:
    """Result of :meth:`AggregateRadixTree.match_fully_ready`."""

    matched_block_ids: Tuple[int, ...]
    contributing_peers: FrozenSet[str]
    # Always a single value (matches §4.7.1.4 single-Node match constraint
    # already enforced by the C++ ``RefRadixTree``).  ``None`` when the
    # match length is zero.
    matched_node_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Refcount entry
# ---------------------------------------------------------------------------
@dataclass
class _RefCountEntry:
    count: int = 0
    # Wall-clock time (seconds since epoch) of the most recent acquire.
    # Used by ``scan_leaked_refcount`` to identify stuck refcounts.
    last_acquired_at: float = 0.0


# ---------------------------------------------------------------------------
# Tree
# ---------------------------------------------------------------------------
class AggregateRadixTree:
    """Cross-SD aggregate radix + block-level refcount manager.

    Public API surface mirrors design doc §4.3 / §4.3.1:

    * :meth:`mark_sd_ready` / :meth:`mark_sd_evicted` — per-SD ack tracker
    * :meth:`match_fully_ready` — query the longest fully-ready prefix
    * :meth:`acquire` / :meth:`release` / :meth:`is_evictable` — refcount
    * :meth:`scan_leaked_refcount` — leak detector
    * :meth:`invalidate_by_peer_instance` / :meth:`invalidate_prefix` —
      reactions to failure-detector events
    """

    def __init__(
        self,
        total_sd_count: int,
        *,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(total_sd_count, int) or total_sd_count < 1:
            raise ValueError(f"total_sd_count must be int>=1, got {total_sd_count!r}")
        self._total_sd_count: int = total_sd_count
        self._time_fn: Callable[[], float] = time_fn
        self._lock = threading.RLock()

        # prefix_hash -> ReadyEntry
        self._prefixes: Dict[int, ReadyEntry] = {}
        # Reverse index: block_id -> set of prefix_hash that contain it.
        # Lets us O(1) look up "which prefixes own this block" when
        # invalidating or releasing.
        self._block_to_prefixes: Dict[int, Set[int]] = {}
        # block_id -> _RefCountEntry
        self._refcounts: Dict[int, _RefCountEntry] = {}

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    @property
    def total_sd_count(self) -> int:
        return self._total_sd_count

    def __len__(self) -> int:
        with self._lock:
            return len(self._prefixes)

    def known_prefixes(self) -> List[int]:
        """Snapshot of prefix hashes currently tracked (not necessarily ready)."""
        with self._lock:
            return list(self._prefixes.keys())

    # ------------------------------------------------------------------
    # Per-SD ack tracker
    # ------------------------------------------------------------------
    def mark_sd_ready(
        self,
        prefix_hash: int,
        sd_key: str,
        block_ids: Iterable[int],
        *,
        contributing_peer: Optional[str] = None,
        node_id: int = -1,
    ) -> bool:
        """Record that ``sd_key`` finished its share of ``prefix_hash``.

        Returns True iff this call transitioned the prefix to *fully ready*
        (i.e. all SDs are now accounted for).  Subsequent calls for the
        same SD are idempotent and return False — but they **do** update
        the recorded ``node_id`` if the previous call passed -1 (sentinel).
        That lets callers fill in the node_id after the fact when it
        wasn't available at first-ack time.

        Args:
            prefix_hash: leaf-block hash that identifies the prefix.
            sd_key: serialized SD key of the SD that just acked.
            block_ids: physical block IDs in the Master's CPU pool.
                Must agree across acks for the same prefix.
            contributing_peer: instance_id of the peer FlexKV instance
                whose data we pulled to fill this SD's slot (None when
                this is a self-SD ack).
            node_id: distributed_node_id of the FlexKV node that holds
                the data for this SD.  ``-1`` is the sentinel "unknown
                yet"; callers can re-issue ``mark_sd_ready`` later with
                the real node_id and the entry will be patched.
        """
        if not isinstance(sd_key, str) or not sd_key:
            raise ValueError(f"sd_key must be a non-empty str, got {sd_key!r}")

        block_tuple = tuple(int(b) for b in block_ids)

        with self._lock:
            entry = self._prefixes.get(prefix_hash)
            if entry is None:
                entry = ReadyEntry(prefix_hash=int(prefix_hash), block_ids=block_tuple)
                self._prefixes[entry.prefix_hash] = entry
                for b in block_tuple:
                    self._block_to_prefixes.setdefault(b, set()).add(entry.prefix_hash)
            else:
                # Validate block_ids stay consistent across acks.  Mismatched
                # block_ids would mean two SDs disagree on the physical
                # placement, which is an upstream bug; fail loudly.
                if entry.block_ids != block_tuple and block_tuple:
                    raise ValueError(
                        f"AggregateRadixTree: prefix_hash={prefix_hash} block_ids "
                        f"mismatch (existing={entry.block_ids}, new={block_tuple})"
                    )

            was_ready = entry.is_fully_ready(self._total_sd_count)
            existing_nid = entry.ready_sds.get(sd_key, None)
            # If the SD has acked before and we now have a real node_id
            # to fill in, update — otherwise leave it alone.  This makes
            # mark_sd_ready idempotent w.r.t. transition flag but still
            # useful for late-binding the node_id.
            if existing_nid is None or (existing_nid == -1 and int(node_id) != -1):
                entry.ready_sds[sd_key] = int(node_id)
            if contributing_peer:
                entry.contributing_peers.add(contributing_peer)
            became_ready = (not was_ready) and entry.is_fully_ready(self._total_sd_count)
            if became_ready:
                entry.first_ready_at = self._time_fn()
            return became_ready

    def mark_sd_evicted(self, prefix_hash: int, sd_key: str) -> None:
        """Remove ``sd_key`` from a prefix's ready-set.

        If the set becomes empty the prefix is dropped entirely.  Silently
        no-op if the prefix is unknown (matches the "Master single-handedly
        evicts" semantics in design doc §4.3.1 — Remotes do not need to
        observe an eviction)."""
        with self._lock:
            entry = self._prefixes.get(prefix_hash)
            if entry is None:
                return
            entry.ready_sds.pop(sd_key, None)
            if not entry.ready_sds:
                self._drop_prefix_locked(entry)

    # ------------------------------------------------------------------
    # Match
    # ------------------------------------------------------------------
    def match_fully_ready(self, prefix_hash: int) -> Optional[ReadyEntry]:
        """Return the ``ReadyEntry`` iff ``prefix_hash`` is fully ready, else None.

        Note: this is *not* a longest-prefix matcher.  The legacy
        ``RefRadixTree`` does prefix matching; the aggregate layer just
        gates whether each candidate from the legacy match is allowed
        through.  Callers iterate over candidate prefixes and ask us for a
        gate decision.
        """
        with self._lock:
            entry = self._prefixes.get(prefix_hash)
            if entry is None:
                return None
            if not entry.is_fully_ready(self._total_sd_count):
                return None
            # Defensive copy — keep the entry immutable from the caller's pov.
            return ReadyEntry(
                prefix_hash=entry.prefix_hash,
                block_ids=entry.block_ids,
                ready_sds=dict(entry.ready_sds),
                contributing_peers=set(entry.contributing_peers),
                first_ready_at=entry.first_ready_at,
            )

    # ------------------------------------------------------------------
    # Refcount
    # ------------------------------------------------------------------
    def acquire(self, block_ids: Iterable[int]) -> None:
        """Increment the refcount of every block in ``block_ids``.

        Refcounts are recorded *per block*, regardless of which prefix(es)
        the block participates in.  This matches design doc §4.3.1
        prerequisite B which says "the block has refcount > 0 if it is in
        flight on any SD".
        """
        now = self._time_fn()
        with self._lock:
            for raw in block_ids:
                b = int(raw)
                ent = self._refcounts.get(b)
                if ent is None:
                    ent = _RefCountEntry()
                    self._refcounts[b] = ent
                ent.count += 1
                ent.last_acquired_at = now

    def release(self, block_ids: Iterable[int]) -> None:
        """Decrement refcounts.  Raises :class:`BlockNotTrackedError` if a
        block was never acquired (catches double-release bugs early)."""
        with self._lock:
            for raw in block_ids:
                b = int(raw)
                ent = self._refcounts.get(b)
                if ent is None or ent.count <= 0:
                    raise BlockNotTrackedError(
                        f"AggregateRadixTree.release: block_id={b} not tracked or refcount already zero"
                    )
                ent.count -= 1
                if ent.count == 0:
                    # Reclaim memory eagerly — leaked entries can pile up
                    # in long-running processes otherwise.
                    del self._refcounts[b]

    def is_evictable(self, block_id: int) -> bool:
        """Return True iff ``block_id`` has zero in-flight uses."""
        with self._lock:
            ent = self._refcounts.get(int(block_id))
            return ent is None or ent.count <= 0

    def get_refcount(self, block_id: int) -> int:
        """Helper for tests / observability.  Never raises."""
        with self._lock:
            ent = self._refcounts.get(int(block_id))
            return ent.count if ent is not None else 0

    def scan_leaked_refcount(self, timeout_seconds: float) -> List[int]:
        """Return all block_ids whose refcount has been > 0 longer than
        ``timeout_seconds``.

        The Master is expected to call this periodically (design doc §4.3.1
        prerequisite C "refcount timeout safety net") and then forcibly
        zero each leaked refcount + invalidate the owning prefix(es).
        """
        if timeout_seconds < 0:
            raise ValueError(f"timeout_seconds must be >= 0, got {timeout_seconds!r}")
        cutoff = self._time_fn() - timeout_seconds
        with self._lock:
            return [b for b, ent in self._refcounts.items() if ent.last_acquired_at <= cutoff]

    def force_release(self, block_id: int) -> int:
        """Hard-reset a block's refcount to zero.

        Returns the previous refcount (0 if block was untracked).  Designed
        to be the second half of the leak-recovery sequence — call it for
        every entry returned by :meth:`scan_leaked_refcount`, then call
        :meth:`invalidate_prefix` to drop the prefix from the radix.
        """
        with self._lock:
            ent = self._refcounts.pop(int(block_id), None)
            return ent.count if ent is not None else 0

    # ------------------------------------------------------------------
    # Invalidation
    # ------------------------------------------------------------------
    def invalidate_prefix(self, prefix_hash: int) -> bool:
        """Drop a single prefix.  Returns True if it existed."""
        with self._lock:
            entry = self._prefixes.get(prefix_hash)
            if entry is None:
                return False
            self._drop_prefix_locked(entry)
            return True

    def invalidate_by_peer_instance(self, peer_instance_id: str) -> int:
        """Batch-drop every prefix that lists ``peer_instance_id`` as a
        contributing peer.  Returns the number of prefixes invalidated.

        Design doc §4.3.2 Layer-1: when the failure detector observes
        ``peer_instance_id`` go away (TTL expiry / epoch bump), we tear down
        any "fully-ready" claim that depended on it."""
        if not peer_instance_id:
            raise ValueError("peer_instance_id must be a non-empty str")
        with self._lock:
            victims = [
                e for e in self._prefixes.values()
                if peer_instance_id in e.contributing_peers
            ]
            for e in victims:
                self._drop_prefix_locked(e)
            return len(victims)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _drop_prefix_locked(self, entry: ReadyEntry) -> None:
        """Remove a prefix from both the prefix map and the reverse index.

        Caller must hold ``self._lock``.
        """
        self._prefixes.pop(entry.prefix_hash, None)
        for b in entry.block_ids:
            owners = self._block_to_prefixes.get(b)
            if owners is None:
                continue
            owners.discard(entry.prefix_hash)
            if not owners:
                self._block_to_prefixes.pop(b, None)
