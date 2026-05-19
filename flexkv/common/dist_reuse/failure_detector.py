"""Layer-1 failure detector — Redis session + epoch heartbeat.

Design doc §4.3.2 Layer-1.  Each FlexKV instance writes
``flexkv:instance:<id>:session`` with a TTL of a few seconds and renews it
every TTL/3.  Peer instances poll the ``flexkv:instance:*:session`` keyspace
periodically and react to two events:

* **session key disappears** → peer was killed / lost network (TTL expiry).
* **epoch field changes** → peer was restarted (cold boot) since we last
  looked.

Either case triggers ``on_peer_lost(peer_instance_id)``, which the
:class:`AggregateRadixTree` consumer turns into a batch invalidation.

This module is **transport-agnostic**: it works against any object that
exposes the small subset of redis-py operations we need (``set/get/expire/
scan_iter``).  Tests use an in-memory fake to avoid bringing up a real Redis
server.

Layer-2 (data-plane closed loop on Mooncake P2P read/write failure) is
implemented in :mod:`flexkv.cache.coordination_protocol`'s
:class:`FailureReportMsg` handler — not here.

Phase 0 task 0-L.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional, Protocol, Set

from .sharing_domain_namespace import SharingDomainNamespace


__all__ = [
    "InstanceSession",
    "RedisSessionClient",
    "FailureDetector",
    "make_session_epoch",
    "make_redis_client_from_cache_config",
]


_LOG = logging.getLogger("flexkv.failure_detector")


def make_session_epoch() -> str:
    """Generate a fresh, monotonic-ish session epoch string.

    Combines the current monotonic-ms timestamp (so two epochs from the
    same process can be totally ordered) and a uuid4 suffix (so two
    distinct processes never collide).  Format::

        <12-hex-of-monotonic-ms>-<8-hex-of-uuid4>
    """
    ms = int(time.monotonic() * 1000) & 0xFFFFFFFFFFFF
    rand = uuid.uuid4().hex[:8]
    return f"{ms:012x}-{rand}"


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class InstanceSession:
    """Decoded ``flexkv:instance:<id>:session`` payload."""

    instance_id: str
    epoch: str
    master_zmq_addr: str
    node_ids: tuple = ()
    mooncake_addrs_by_sd: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Redis abstraction
# ---------------------------------------------------------------------------
class _RedisLike(Protocol):
    """Subset of ``redis.Redis`` actually used here.

    Defined as a Protocol so tests can plug in an in-memory fake.
    """

    def set(self, name: str, value: str, ex: Optional[int] = None) -> Any: ...
    def get(self, name: str) -> Any: ...
    def expire(self, name: str, ex: int) -> Any: ...
    def delete(self, *names: str) -> Any: ...
    def scan_iter(self, match: Optional[str] = None, count: Optional[int] = None) -> Iterable[Any]: ...


class RedisSessionClient:
    """Thin wrapper that owns the *write* side of an instance session.

    Wires up:
      - :meth:`register` — initial ``SET ... EX ttl`` of the session payload.
      - :meth:`renew` — periodic ``EXPIRE`` (or full re-set if missing).
      - :meth:`unregister` — clean shutdown.

    The class is **not** responsible for running the renewal loop; that is
    :class:`FailureDetector`'s job (which combines write + read sides).
    """

    def __init__(
        self,
        redis_client: _RedisLike,
        *,
        instance_id: str,
        epoch: str,
        ttl_seconds: int,
        master_zmq_addr: str = "",
        node_ids: Optional[Iterable[int]] = None,
        mooncake_addrs_by_sd: Optional[Dict[str, str]] = None,
    ) -> None:
        if ttl_seconds < 1:
            raise ValueError(f"ttl_seconds must be >= 1, got {ttl_seconds!r}")
        self._client = redis_client
        self._instance_id = instance_id
        self._epoch = epoch
        self._ttl = ttl_seconds
        self._key = SharingDomainNamespace.instance_session_key(instance_id)
        # Snapshot the static portion of the payload once.
        self._payload: Dict[str, Any] = {
            "instance_id": instance_id,
            "epoch": epoch,
            "master_zmq_addr": master_zmq_addr,
            "node_ids": list(node_ids or []),
            "mooncake_addrs_by_sd": dict(mooncake_addrs_by_sd or {}),
        }

    @property
    def instance_id(self) -> str:
        return self._instance_id

    @property
    def epoch(self) -> str:
        return self._epoch

    @property
    def key(self) -> str:
        return self._key

    def register(self) -> None:
        """Write the session payload with a TTL.  Overwrites any existing
        key — by design, a restarted instance must replace its old record."""
        self._client.set(self._key, json.dumps(self._payload), ex=self._ttl)

    def renew(self) -> None:
        """Refresh the TTL.  If the key has expired since last renewal we
        re-write the full payload to avoid the "ghost peer" scenario where
        a watchdog observes the gap as a restart."""
        ok = bool(self._client.expire(self._key, self._ttl))
        if not ok:
            self.register()

    def unregister(self) -> None:
        try:
            self._client.delete(self._key)
        except Exception as e:  # pragma: no cover — best-effort cleanup
            _LOG.warning("unregister(%s) failed: %s", self._key, e)


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
class FailureDetector:
    """Polls Redis for peer instance liveness and fires user callbacks.

    Lifecycle:

    >>> detector = FailureDetector(client, "self-instance", on_peer_lost=cb)
    >>> detector.start()
    ... # ... run normally ...
    >>> detector.stop()

    The callbacks must be **thread-safe and fast** — they execute on the
    detector's polling thread.  The recommended pattern is to enqueue work
    into the aggregate radix's internal lock-protected state and return
    immediately.
    """

    def __init__(
        self,
        redis_client: _RedisLike,
        self_instance_id: str,
        *,
        poll_interval_seconds: float = 2.0,
        on_peer_lost: Optional[Callable[[str], None]] = None,
        on_peer_seen: Optional[Callable[[str, InstanceSession], None]] = None,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if poll_interval_seconds <= 0:
            raise ValueError(f"poll_interval_seconds must be > 0, got {poll_interval_seconds!r}")
        if not self_instance_id:
            raise ValueError("self_instance_id must be a non-empty str")
        self._client = redis_client
        self._self_instance_id = self_instance_id
        self._poll_interval = poll_interval_seconds
        self._on_peer_lost = on_peer_lost or (lambda _pid: None)
        self._on_peer_seen = on_peer_seen or (lambda _pid, _s: None)
        self._time_fn = time_fn

        # peer_instance_id -> last observed epoch
        self._known_peers: Dict[str, str] = {}
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        # Used by tests to drive a single iteration deterministically.
        self._iteration_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("FailureDetector already started")
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="flexkv-failure-detector",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: Optional[float] = 2.0) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        self._thread = None

    # ------------------------------------------------------------------
    # Polling loop
    # ------------------------------------------------------------------
    def _run(self) -> None:  # pragma: no cover — exercised through tests via poll_once()
        while not self._stop_event.is_set():
            try:
                self.poll_once()
            except Exception as e:
                _LOG.exception("FailureDetector poll error: %s", e)
            self._stop_event.wait(self._poll_interval)

    def poll_once(self) -> None:
        """Run a single polling cycle.  Public to let tests drive the
        detector deterministically without bringing up the thread."""
        with self._iteration_lock:
            current = self._scan_peers()

            # Handle disappeared peers first (TTL expiry).
            disappeared = set(self._known_peers) - set(current) - {self._self_instance_id}
            for pid in disappeared:
                _LOG.info("FailureDetector: peer %s disappeared (TTL expiry)", pid)
                self._invoke_lost(pid)
                self._known_peers.pop(pid, None)

            # Handle new + epoch-changed peers.
            for pid, session in current.items():
                if pid == self._self_instance_id:
                    continue
                prev = self._known_peers.get(pid)
                if prev is None:
                    _LOG.info("FailureDetector: peer %s appeared (epoch=%s)", pid, session.epoch)
                    self._invoke_seen(pid, session)
                elif prev != session.epoch:
                    _LOG.info("FailureDetector: peer %s restarted (epoch %s -> %s)", pid, prev, session.epoch)
                    # Treat epoch change as "lost then seen" — the lost
                    # callback invalidates stale state and the seen
                    # callback re-registers.
                    self._invoke_lost(pid)
                    self._invoke_seen(pid, session)
                self._known_peers[pid] = session.epoch

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _scan_peers(self) -> Dict[str, InstanceSession]:
        out: Dict[str, InstanceSession] = {}
        pattern = SharingDomainNamespace.instance_session_key_pattern()
        for raw_key in self._client.scan_iter(match=pattern, count=100):
            key = raw_key.decode("utf-8") if isinstance(raw_key, (bytes, bytearray)) else str(raw_key)
            try:
                pid = SharingDomainNamespace.parse_instance_session_key(key)
            except ValueError:
                continue
            raw_value = self._client.get(key)
            if raw_value is None:
                continue
            value = (
                raw_value.decode("utf-8")
                if isinstance(raw_value, (bytes, bytearray))
                else str(raw_value)
            )
            try:
                payload = json.loads(value)
            except (TypeError, ValueError):
                _LOG.warning("FailureDetector: malformed session payload at %s", key)
                continue
            try:
                session = InstanceSession(
                    instance_id=str(payload["instance_id"]),
                    epoch=str(payload["epoch"]),
                    master_zmq_addr=str(payload.get("master_zmq_addr", "")),
                    node_ids=tuple(payload.get("node_ids", ())),
                    mooncake_addrs_by_sd=dict(payload.get("mooncake_addrs_by_sd", {})),
                )
            except KeyError as e:
                _LOG.warning("FailureDetector: missing field %s in %s", e, key)
                continue
            out[session.instance_id] = session
        return out

    def _invoke_lost(self, pid: str) -> None:
        try:
            self._on_peer_lost(pid)
        except Exception:  # pragma: no cover — defensive logging
            _LOG.exception("on_peer_lost(%s) raised", pid)

    def _invoke_seen(self, pid: str, session: InstanceSession) -> None:
        try:
            self._on_peer_seen(pid, session)
        except Exception:  # pragma: no cover — defensive logging
            _LOG.exception("on_peer_seen(%s) raised", pid)

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------
    def known_peers(self) -> Set[str]:
        with self._iteration_lock:
            return set(self._known_peers)


# ---------------------------------------------------------------------------
# redis-py client factory — single source of truth for the flexkv_redis_db
# resolution rule.  All dist-reuse code paths that need a raw ``redis.Redis``
# (e.g. for ``RedisSessionClient`` / ``FailureDetector``) should go through
# this helper so the ``flexkv_redis_db`` override from ``CacheConfig`` is
# honoured in exactly one place.
# ---------------------------------------------------------------------------
def make_redis_client_from_cache_config(
    cache_config: Any,
    *,
    decode_responses: bool = True,
    socket_connect_timeout: Optional[float] = None,
) -> Any:
    """Construct a ``redis.Redis`` client from the given ``CacheConfig``.

    Pulls ``host`` / ``port`` / ``password`` / ``flexkv_redis_db`` (with
    legacy fallback to ``0``) off the config and returns a ready-to-use
    client.  Importing ``redis`` is deferred so callers that run in a
    CPU-only unit-test environment without the redis-py dependency still
    get a useful ``ImportError``.

    Args:
        cache_config: A :class:`~flexkv.common.config.CacheConfig` or any
            duck-typed object exposing the same attrs.
        decode_responses: Forwarded to ``redis.Redis`` — most FlexKV code
            expects ``str`` round-trips so this defaults to ``True``.
        socket_connect_timeout: Optional connect timeout in seconds.  Set
            this to a small value (e.g. 1.0) in tests to fail fast when
            Redis is absent.

    Returns:
        A ``redis.Redis`` instance bound to ``cache_config.flexkv_redis_db``.

    Raises:
        ImportError: If ``redis-py`` is not installed.
    """
    try:
        import redis as _redis  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            "redis-py is required for dist-reuse operations: pip install redis"
        ) from e

    kwargs: Dict[str, Any] = {
        "host": getattr(cache_config, "redis_host", "127.0.0.1"),
        "port": int(getattr(cache_config, "redis_port", 6379)),
        "db": int(getattr(cache_config, "flexkv_redis_db", 0)),
        "decode_responses": decode_responses,
    }
    password = getattr(cache_config, "redis_password", None)
    if password:
        kwargs["password"] = password
    if socket_connect_timeout is not None:
        kwargs["socket_connect_timeout"] = float(socket_connect_timeout)
    return _redis.Redis(**kwargs)
