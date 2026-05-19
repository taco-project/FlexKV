"""Master-side sharing-domain orchestration.

Phase 0 task 0-G / 0-H-integration / 0-K:

* :func:`build_sharing_domain_handles` — given a ``(ModelConfig,
  CacheConfig, SharingDomainKey)`` triple, enumerate the full set of SDs
  in the instance and construct the correct ``TransferManagerHandle``
  list (1 × "process" for the Master SD + N-1 × "remote" for each peer SD).

* :class:`MasterCoordinator` — owns the instance's
  :class:`AggregateRadixTree`, glues Remote ``RemoteReadyMsg`` arrivals
  to instance-level Redis registration, and exposes small helpers
  (``acquire_blocks`` / ``release_blocks`` / ``mark_sd_ready``) that
  ``GlobalCacheEngine`` calls from its match / put / evict paths.

* :func:`graph_needs_gpu_clear` — tiny predicate used by
  ``KVTaskManager._launch_task`` to decide whether a remote-submitted
  :class:`TransferOpGraph` needs its GPU block IDs cleared (PP / CP
  Remotes need a fresh slot_mapping; TP Remotes share the Master's one).

This module is **Python-only** and free of GPU/C++ imports, so it can be
unit-tested on CPU-only machines.  Its callers in
``flexkv.transfer_manager`` / ``flexkv.kvtask`` / ``flexkv.cache.cache_engine``
import it conditionally — disabling ``CacheConfig.enable_sharing_domain``
keeps the legacy paths untouched.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .aggregate_radix import AggregateRadixTree
from .coordination_protocol import RemoteReadyMsg
from .failure_detector import FailureDetector, RedisSessionClient, make_session_epoch
from .sharing_domain import SharingDomainKey
from .sharing_domain_namespace import SharingDomainNamespace


__all__ = [
    "MasterCoordinator",
    "SharingDomainHandleSpec",
    "build_sharing_domain_handles",
    "graph_needs_gpu_clear",
]


_LOG = logging.getLogger("flexkv.dist_reuse.master")


# ---------------------------------------------------------------------------
# Handle spec (used by KVTaskManager)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SharingDomainHandleSpec:
    """Static description of **one** TransferManagerHandle the Master has
    to create.  ``KVTaskManager`` consumes a list of these.

    * ``sd_key`` — which SD this handle serves.
    * ``mode`` — ``"process"`` for the Master's own SD (in-proc
      TransferManager) or ``"remote"`` for every other SD.
    * ``endpoint`` — populated for ``mode="remote"`` only; the Master
      reads this from ``CacheConfig.remote_endpoints_by_sd`` (task 0-J).
      ``None`` for the Master's own SD.
    """

    sd_key: SharingDomainKey
    mode: str
    endpoint: Optional[Any] = None  # RemoteEndpoint; typed Any to dodge config-import cycle


def build_sharing_domain_handles(
    *,
    self_sd: SharingDomainKey,
    remote_endpoints_by_sd: Optional[Dict[str, Any]] = None,
) -> List[SharingDomainHandleSpec]:
    """Enumerate every SD in the instance and produce a handle spec.

    ``self_sd`` is the SD the Master owns (produced by
    :meth:`SharingDomainKey.from_model_config` on the Master node).  The
    function returns one :class:`SharingDomainHandleSpec` per SD,
    ordered so the Master's own SD is first (index 0).  This preserves
    the legacy invariant that ``self.transfer_handles[0]`` is the local
    in-process handle.

    ``remote_endpoints_by_sd`` maps serialized SD key strings to
    ``RemoteEndpoint`` instances from ``CacheConfig``.  A ``KeyError`` is
    raised eagerly for any non-master SD without an endpoint so the
    Master fails fast at startup instead of silently running with
    fewer handles than it thinks.
    """
    remote_endpoints_by_sd = remote_endpoints_by_sd or {}

    specs: List[SharingDomainHandleSpec] = []

    # Master's own SD first.
    specs.append(SharingDomainHandleSpec(
        sd_key=self_sd, mode="process", endpoint=None,
    ))

    for peer_sd in self_sd.enumerate_peers():
        if peer_sd == self_sd:
            continue
        key_str = peer_sd.serialize()
        endpoint = remote_endpoints_by_sd.get(key_str)
        if endpoint is None:
            raise KeyError(
                f"build_sharing_domain_handles: missing endpoint for peer SD "
                f"{key_str!r}. Populate CacheConfig.remote_endpoints_by_sd "
                f"from the framework launcher before constructing KVManager."
            )
        specs.append(SharingDomainHandleSpec(
            sd_key=peer_sd, mode="remote", endpoint=endpoint,
        ))
    return specs


# ---------------------------------------------------------------------------
# GPU-clear predicate (task 0-K)
# ---------------------------------------------------------------------------
def graph_needs_gpu_clear(self_sd: SharingDomainKey, peer_sd: SharingDomainKey) -> bool:
    """Return True iff a TransferOpGraph submitted to ``peer_sd`` needs
    its GPU blocks reset before sending.

    Rule of thumb (design doc §4.12.2 (4), simplified):

    * **Different pp_rank** → True — the peer's scheduler sees a different
      slot_mapping for the same task because it owns a different layer range.
    * **Only tp_node_idx differs** → False — cross-node TP shares the
      same slot_mapping across all TP ranks (just with different head shards).
    * **Same SD as self** → False — it's the local graph.

    CP is intentionally absent from the SD key (see simplified design §4.5),
    so any CP-level differences are handled at the connector layer (sync_leader
    scatter) and do not show up here.
    """
    if peer_sd == self_sd:
        return False
    return peer_sd.pp_rank != self_sd.pp_rank


# ---------------------------------------------------------------------------
# Master coordinator
# ---------------------------------------------------------------------------
class MasterCoordinator:
    """Owns the per-instance AggregateRadixTree + failure detector + Remote
    ready-wait handshake.

    The ``GlobalCacheEngine`` holds exactly one of these when
    ``CacheConfig.enable_sharing_domain=True``.  The class is deliberately
    framework-agnostic: it doesn't import ``torch``, ``zmq``, or
    ``transfer_manager``; it only exposes pure-Python hooks that the
    relevant modules call from within their own flow.

    Lifecycle:

    1. Construct with the Master's own ``SharingDomainKey``.
    2. Call :meth:`expect_remotes` once you've built the handle list —
       tells the coordinator how many Remote ready acks to wait for.
    3. For each incoming :class:`RemoteReadyMsg`, call
       :meth:`on_remote_ready`.  Returns True when the last Remote has
       reported, at which point ``register_instance_discoverables``
       is safe to call.
    4. ``get_match`` / ``put_match`` / ``_transfer_callback`` in
       ``GlobalCacheEngine`` call ``acquire_blocks`` / ``release_blocks`` /
       ``mark_sd_ready`` to update the aggregate radix.
    5. ``evict`` in the hierarchical cache engine calls
       :meth:`is_evictable` before actually dropping a block.
    6. The failure detector fires :meth:`on_peer_lost` which invalidates
       every aggregate entry contributed by the dead peer.
    """

    def __init__(
        self,
        *,
        self_sd: SharingDomainKey,
        instance_id: str,
        session_epoch: Optional[str] = None,
        refcount_leak_timeout_seconds: float = 30.0,
        failure_escalation_threshold: int = 3,
    ) -> None:
        self._self_sd = self_sd
        self._namespace = SharingDomainNamespace(self_sd)
        self._instance_id = str(instance_id)
        self._session_epoch = session_epoch or make_session_epoch()

        # The Master's own distributed_node_id.  Filled in by
        # ``register_instance_discoverables`` once the Master knows what
        # it was assigned in Redis.  Defaults to ``-1`` (sentinel for
        # "not yet known"); ``GlobalCacheEngine._notify_master_sd_ready``
        # threads this value into ``mark_sd_ready`` so the per-SD
        # node_id map starts populated for the master SD on the very
        # first ack.
        self._self_node_id: int = -1

        self._aggregate = AggregateRadixTree(total_sd_count=self_sd.total_sd_count())
        self._refcount_leak_timeout = float(refcount_leak_timeout_seconds)

        # Phase D-4: Layer-2 closed loop bookkeeping (migrated from
        # the now-deleted CoordinationCoordinator).
        self._failure_escalation_threshold = int(failure_escalation_threshold)
        self._peer_failure_counts: Dict[str, int] = {}

        self._lock = threading.RLock()

        # Filled in by on_remote_ready()
        self._expected_remote_count: Optional[int] = None
        self._ready_remotes: Dict[str, RemoteReadyMsg] = {}

        # Populated on register_instance_discoverables()
        self._session_client: Optional[RedisSessionClient] = None
        self._failure_detector: Optional[FailureDetector] = None

        # Optional external callback invoked from ``_on_peer_lost`` after
        # the internal aggregate invalidation.  Set via
        # :meth:`set_peer_lost_hook`.
        self._extra_peer_lost_hook: Optional[Any] = None

    # ---------------------------------------------------------------- state
    @property
    def self_sd(self) -> SharingDomainKey:
        return self._self_sd

    @property
    def namespace(self) -> SharingDomainNamespace:
        return self._namespace

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def session_epoch(self) -> str:
        return self._session_epoch

    @property
    def aggregate_radix(self) -> AggregateRadixTree:
        return self._aggregate

    @property
    def self_node_id(self) -> int:
        """Distributed node_id of the Master itself.  ``-1`` until
        :meth:`register_instance_discoverables` is called.
        """
        return int(self._self_node_id)

    @property
    def total_sd_count(self) -> int:
        """Total number of SDs in this instance.  == ``self_sd.total_sd_count()``.

        Used by the GET main path to decide whether a coord GET
        barrier is needed (``> 1``) or we can short-circuit
        (``== 1``, single-SD degenerate case).
        """
        return int(self._self_sd.total_sd_count())

    def expect_remotes(self, count: int) -> None:
        """Tell the coordinator how many ``RemoteReadyMsg`` to wait for.

        Must be called before :meth:`on_remote_ready`.  ``count`` is the
        length of ``transfer_handles`` minus 1 (the Master's own handle).
        """
        if count < 0:
            raise ValueError(f"count must be >= 0, got {count}")
        with self._lock:
            if self._expected_remote_count is not None:
                raise RuntimeError("expect_remotes() has already been called")
            self._expected_remote_count = int(count)

    # ---------------------------------------------------------------- remotes
    def on_remote_ready(self, msg: RemoteReadyMsg) -> bool:
        """Record a Remote's ready-ack.  Returns True when the last Remote
        has reported (i.e. ``len(ready) == expected``).  Further calls
        return True too (idempotent)."""
        if self._expected_remote_count is None:
            raise RuntimeError(
                "on_remote_ready called before expect_remotes(); wire up the "
                "handle list first"
            )
        with self._lock:
            # Ack comes straight off the wire — keep the canonical SD-key
            # as the dict key so tests can assert on it.
            self._ready_remotes[msg.sd_key] = msg
            return len(self._ready_remotes) >= self._expected_remote_count

    def all_remotes_ready(self) -> bool:
        with self._lock:
            return (
                self._expected_remote_count is not None
                and len(self._ready_remotes) >= self._expected_remote_count
            )

    def ready_remote_infos(self) -> Dict[str, RemoteReadyMsg]:
        with self._lock:
            return dict(self._ready_remotes)

    def build_sd_to_nid_map(self, self_node_id: int) -> Dict[str, int]:
        """Produce the ``sd_key → node_id`` mapping for
        ``RedisMeta.register_instance_sd_nodes``."""
        with self._lock:
            out: Dict[str, int] = {self._self_sd.serialize(): int(self_node_id)}
            for sd_key_str, msg in self._ready_remotes.items():
                out[sd_key_str] = int(msg.distributed_node_id)
            return out

    def get_sd_to_nid_map(self) -> Dict[str, int]:
        """Public read-only accessor — Phase D-2 (proposal §6.3): the
        Master's PUT-path graph builder needs to attach
        ``target_node_ids=[peer_node_id]`` to each peer-SD D2H op so that
        each Remote's ``_handle_submit`` filter picks up the right slice.

        Returns ``{sd_key_str: distributed_node_id}`` for every SD known
        to this MasterCoordinator (master's own SD + every Remote that
        has finished its ready handshake).  Returns an empty dict if the
        Master's own ``self_node_id`` hasn't been set yet (i.e.
        ``register_instance_discoverables`` hasn't been called).
        """
        with self._lock:
            if int(self._self_node_id) < 0:
                # Bootstrap not finished — no canonical mapping yet.
                return {}
            out: Dict[str, int] = {
                self._self_sd.serialize(): int(self._self_node_id),
            }
            for sd_key_str, msg in self._ready_remotes.items():
                out[sd_key_str] = int(msg.distributed_node_id)
            return out

    # ------------------------------------------------------------ lookup
    def lookup_peer_by_node_id(self, node_id: int) -> Optional[Dict[str, str]]:
        """Reverse-lookup: given a (global) distributed_node_id, return
        the peer SD it belongs to plus the peer instance_id that owns
        that SD.

        Returns a dict with keys ``sd_key_str`` and ``instance_id``,
        or ``None`` if the node_id isn't one of our ready peers (i.e.
        it's either this instance's own node, unknown, or belongs to
        an instance that hasn't finished its ready handshake).

        Used by the GET main-path glue
        (``GlobalCacheEngine._maybe_attach_multi_sd_peerh2h_ops``,
        Phase D-3) and by ``handle_failure_report`` to map an
        offending node_id back to the peer SD + instance it sits on.
        """
        if node_id is None:
            return None
        try:
            nid = int(node_id)
        except Exception:
            return None
        if nid < 0:
            return None
        with self._lock:
            for sd_key_str, msg in self._ready_remotes.items():
                if int(getattr(msg, "distributed_node_id", -1)) == nid:
                    return {
                        "sd_key_str": sd_key_str,
                        "instance_id": str(getattr(msg, "sender_instance_id", "") or ""),
                    }
        return None

    # -------------------------------------------------------- pin helpers
    def pin_blocks_for_coord_get(self, block_ids: Iterable[int]) -> None:
        """Refcount-pin ``block_ids`` against Master-side eviction while
        an in-flight coord GET is expected to land on them.

        Thin alias for :meth:`acquire_blocks` — kept so the GET-path
        glue reads as intent rather than the lower-level primitive.
        Used in conjunction with the multi-SD PEERH2H fan-out
        (``GlobalCacheEngine._maybe_attach_multi_sd_peerh2h_ops``,
        Phase D-3).
        """
        self._aggregate.acquire(block_ids)

    def unpin_blocks_for_coord_get(self, block_ids: Iterable[int]) -> None:
        """Release the refcount pin set by
        :meth:`pin_blocks_for_coord_get`.  Must be called on both the
        success and failure paths of the coord GET.
        """
        self._aggregate.release(block_ids)

    # ------------------------------------------------------------- discovery
    def register_instance_discoverables(
        self,
        *,
        redis_meta: Any,
        self_node_id: int,
        master_zmq_addr: str,
        mooncake_addrs_by_sd: Optional[Dict[str, str]] = None,
        ttl_seconds: int = 8,
    ) -> None:
        """Write the two instance-level Redis keys (design doc §4.6.1 /
        §4.7.1.6) and start the heartbeat session.  Safe to call once all
        Remotes have acked; raises RuntimeError otherwise.
        """
        with self._lock:
            if not self.all_remotes_ready():
                raise RuntimeError(
                    "register_instance_discoverables requires all Remotes to be ready; "
                    f"got {len(self._ready_remotes)} / {self._expected_remote_count}"
                )
            sd_to_nid = self.build_sd_to_nid_map(self_node_id)
            # Cache the master's own node_id so PUT-path callers can
            # populate AggregateRadixTree entries without re-deriving it.
            self._self_node_id = int(self_node_id)

        redis_meta.register_instance_sd_nodes(self._instance_id, sd_to_nid)

        # The failure detector's RedisSessionClient owns its own redis client
        # — it does NOT share ``redis_meta``'s client connection because
        # the heartbeat runs on a background thread.
        mooncake_addrs = dict(mooncake_addrs_by_sd or {})
        redis_client = redis_meta._client()  # private but stable — see redis_meta.py
        self._session_client = RedisSessionClient(
            redis_client,
            instance_id=self._instance_id,
            epoch=self._session_epoch,
            ttl_seconds=int(ttl_seconds),
            master_zmq_addr=master_zmq_addr,
            node_ids=list(sd_to_nid.values()),
            mooncake_addrs_by_sd=mooncake_addrs,
        )
        self._session_client.register()

    def start_failure_detector(
        self,
        redis_meta: Any,
        *,
        poll_interval_seconds: float = 2.0,
    ) -> FailureDetector:
        """Spawn the background :class:`FailureDetector` that scans peer
        instances and invalidates aggregate-radix entries on peer loss.

        Returns the detector so the caller can stop it on shutdown.
        """
        fd = FailureDetector(
            redis_meta._client(),
            self_instance_id=self._instance_id,
            poll_interval_seconds=poll_interval_seconds,
            on_peer_lost=self._on_peer_lost,
        )
        fd.start()
        self._failure_detector = fd
        return fd

    def shutdown(self) -> None:
        """Graceful teardown — stop the detector, unregister session."""
        if self._failure_detector is not None:
            try:
                self._failure_detector.stop()
            except Exception:  # pragma: no cover
                _LOG.exception("FailureDetector.stop() raised")
            self._failure_detector = None
        if self._session_client is not None:
            try:
                self._session_client.unregister()
            except Exception:  # pragma: no cover
                _LOG.exception("RedisSessionClient.unregister() raised")
            self._session_client = None

    # ------------------------------------------------------------- hooks
    def acquire_blocks(self, block_ids: Iterable[int]) -> None:
        self._aggregate.acquire(block_ids)

    def release_blocks(self, block_ids: Iterable[int]) -> None:
        self._aggregate.release(block_ids)

    def is_evictable(self, block_id: int) -> bool:
        return self._aggregate.is_evictable(block_id)

    def mark_sd_ready(
        self,
        prefix_hash: int,
        sd_key_str: str,
        block_ids: Iterable[int],
        *,
        contributing_peer: Optional[str] = None,
        node_id: int = -1,
    ) -> bool:
        return self._aggregate.mark_sd_ready(
            prefix_hash, sd_key_str, block_ids,
            contributing_peer=contributing_peer,
            node_id=int(node_id),
        )

    def mark_sd_evicted(self, prefix_hash: int, sd_key_str: str) -> None:
        self._aggregate.mark_sd_evicted(prefix_hash, sd_key_str)

    def match_fully_ready(self, prefix_hash: int) -> Any:
        return self._aggregate.match_fully_ready(prefix_hash)

    def invalidate_prefix(self, prefix_hash: int) -> bool:
        return self._aggregate.invalidate_prefix(prefix_hash)

    # --------------------------------------------------- periodic scans
    def scan_leaked_refcount(self) -> List[int]:
        """Called periodically by the KVTaskManager's background thread.

        For every block that has been in-flight too long (design doc
        §4.3.1 prerequisite C), force-release its refcount and
        invalidate any prefix that owns it.
        """
        leaked = self._aggregate.scan_leaked_refcount(self._refcount_leak_timeout)
        for block_id in leaked:
            self._aggregate.force_release(block_id)
        return leaked

    # --------------------------------------------------- failure callbacks
    def set_peer_lost_hook(self, cb: Optional[Any]) -> None:
        """Register an **additional** callback to fire when a peer
        instance is lost (Layer-1 session TTL expiry or epoch bump).

        The coordinator already invalidates the aggregate radix
        internally; this hook lets upstream consumers (e.g.
        :class:`GlobalCacheEngine`) react too (e.g. flush in-flight
        coord requests targeting the dead peer).

        Passing ``None`` clears the hook.
        """
        with self._lock:
            self._extra_peer_lost_hook = cb

    def invalidate_by_peer_instance(self, peer_instance_id: str) -> int:
        """Public delegate to the aggregate radix; returns the number
        of invalidated prefixes.  Safe to call from any thread."""
        return self._aggregate.invalidate_by_peer_instance(peer_instance_id)

    # ---------------------------------------------- Layer-2 failure handling
    # Phase D-4 (proposal_unify_with_graph_dispatch_2026-05-15.md §附录 A):
    # ``_handle_failure_report`` migrated here from the now-deleted
    # ``CoordinationCoordinator``.  Layer-2 closed loop: a Remote
    # reports a Mooncake transfer failure; Master invalidates the
    # affected prefix and escalates to a full-peer invalidation after
    # ``failure_escalation_threshold`` repeat reports.
    def handle_failure_report(self, report) -> None:
        """Invalidate the reported prefix and escalate on repeated
        failures from the same peer instance.

        ``report`` must duck-type as a ``FailureReportMsg`` —
        ``peer_instance_id`` (str) and ``failed_block_hashes``
        (Iterable[int]).  Anything else is silently ignored to keep
        callers (the master polling worker, fault-injection tests)
        defensive.
        """
        peer = getattr(report, "peer_instance_id", "") or ""
        if not peer:
            return
        for h in (getattr(report, "failed_block_hashes", []) or []):
            try:
                self._aggregate.invalidate_prefix(int(h))
            except Exception:  # pragma: no cover
                pass

        with self._lock:
            self._peer_failure_counts[peer] = self._peer_failure_counts.get(peer, 0) + 1
            escalate = self._peer_failure_counts[peer] >= int(self._failure_escalation_threshold)

        if escalate:
            self._aggregate.invalidate_by_peer_instance(peer)
            with self._lock:
                self._peer_failure_counts[peer] = 0
            _LOG.warning(
                "[MasterCoordinator:%s] Layer-2 escalated to full-peer "
                "invalidation for %s",
                self._instance_id, peer,
            )

    def peer_failure_count(self, peer_instance_id: str) -> int:
        """Number of unescalated failures from ``peer_instance_id`` —
        used by tests and ops dashboards."""
        with self._lock:
            return self._peer_failure_counts.get(peer_instance_id, 0)

    def _on_peer_lost(self, peer_instance_id: str) -> None:
        """Invoked by the FailureDetector on peer disappearance / epoch bump.

        Batch-invalidates every aggregate entry that listed the lost peer
        as a contributor (design doc §4.3.2 Layer-1), then runs the
        optional user-registered hook so higher layers can react too.
        """
        n = self._aggregate.invalidate_by_peer_instance(peer_instance_id)
        if n:
            _LOG.info(
                "[MasterCoordinator:%s] peer %s lost; invalidated %d prefixes",
                self._instance_id, peer_instance_id, n,
            )
        extra = getattr(self, "_extra_peer_lost_hook", None)
        if extra is not None:
            try:
                extra(peer_instance_id)
            except Exception:  # pragma: no cover — defensive
                _LOG.exception("peer_lost_hook raised")


# ---------------------------------------------------------------------------
# Utility — Remote endpoint lookup (used by KVTaskManager)
# ---------------------------------------------------------------------------
def find_endpoint_for_sd(
    cache_config: Any, sd_key: SharingDomainKey,
) -> Any:
    """Return ``cache_config.remote_endpoints_by_sd[sd_key.serialize()]``
    or raise ``KeyError`` with a diagnostic message."""
    mapping = getattr(cache_config, "remote_endpoints_by_sd", {}) or {}
    key_str = sd_key.serialize()
    if key_str not in mapping:
        raise KeyError(
            f"CacheConfig.remote_endpoints_by_sd is missing an entry for {key_str!r}. "
            f"Populate it from the launcher (e.g. sglang connector) before "
            f"constructing KVTaskManager."
        )
    return mapping[key_str]
