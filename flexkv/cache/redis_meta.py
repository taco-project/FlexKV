"""Redis metadata layer with explicit ``SharingDomainNamespace`` scoping.

Phase 0 task 0-C: every Redis key produced here flows through a
:class:`SharingDomainNamespace`, so the legacy flat keys (``node:*`` /
``meta:*`` / ``buffer:*`` / ``CPUB:block:*`` / ``SSDB:block:*`` /
``PCFSB:block:*``) become ``sd:<sd_key>:node:*`` / ``sd:<sd_key>:meta:*`` /
... .  Per design doc §4.7 we go all-in on the new layout — there is no
backward compatibility for the bare keys.

Callers that don't care about sharing domains (legacy single-instance
dist_reuse) can pass ``SharingDomainKey.default()`` and continue to work as
before; the only observable difference is the Redis key prefix.

The ``RedisMetaChannel`` Python wrapper keeps the same surface as before
but its underlying C++ ``blocks_key`` argument is now expected to carry the
**full namespace** (``sd:<sd_key>:<device_prefix>``) so that
``make_block_key`` produces ``sd:<sd_key>:<device_prefix>:block:<nid>:<hash>``
without any further changes on the C++ side.
"""

from __future__ import annotations
from typing import Iterable, List, Tuple, Optional, Union, Dict, Set, cast
from dataclasses import dataclass
from enum import IntEnum
from uuid import uuid1
import threading
import time
import atexit
import signal
import sys
try:  # redis-py
    import redis as _redis
except Exception:  # pragma: no cover
    _redis = None  # type: ignore

# Import C++ dist extensions (RedisMetaChannel, BlockMeta).  Optional when
# built without FLEXKV_ENABLE_P2P=1.
_CRedisMetaChannel = None  # type: ignore
_CBlockMeta = None  # type: ignore
try:
    import flexkv.c_ext
    from flexkv.c_ext import RedisMetaChannel as _CRedisMetaChannel, BlockMeta as _CBlockMeta  # type: ignore
except (ImportError, AttributeError):
    pass

from flexkv.common.dist_reuse import (
    SharingDomainKey,
    SharingDomainNamespace,
)


__all__ = [
    "NodeState",
    "BlockMeta",
    "RedisMetaChannel",
    "RedisNodeInfo",
    "RedisMeta",
    "dist_available",
]


class NodeState(IntEnum):
    NODE_STATE_NORMAL = 0
    NODE_STATE_ABOUT_TO_EVICT = 1
    NODE_STATE_EVICTED = 2


@dataclass
class BlockMeta:
    ph: int = 0
    pb: int = 0
    nid: int = 0
    hash: int = 0
    lt: int = 0
    state: NodeState = NodeState.NODE_STATE_NORMAL

    def to_c(self) -> "_CBlockMeta":
        if _CBlockMeta is None:
            raise RuntimeError(
                "Distributed KV cache (P2P/Redis) is not built. "
                "Rebuild FlexKV with FLEXKV_ENABLE_P2P=1 and install Redis dependencies (e.g. libhiredis-dev)."
            )
        cm = _CBlockMeta()
        cm.ph = int(self.ph)
        cm.pb = int(self.pb)
        cm.nid = int(self.nid)
        cm.hash = int(self.hash)
        cm.lt = int(self.lt)
        cm.state = int(self.state)
        return cm

    @staticmethod
    def from_c(cm: "_CBlockMeta") -> "BlockMeta":
        if _CBlockMeta is None:
            raise RuntimeError(
                "Distributed KV cache (P2P/Redis) is not built. "
                "Rebuild FlexKV with FLEXKV_ENABLE_P2P=1 and install Redis dependencies (e.g. libhiredis-dev)."
            )
        return BlockMeta(
            ph=int(cm.ph),
            pb=int(cm.pb),
            nid=int(cm.nid),
            hash=int(cm.hash),
            lt=int(cm.lt),
            state=NodeState(int(cm.state))
        )


def dist_available() -> bool:
    """Return True if distributed (P2P/Redis) KV cache C++ extension is built (FLEXKV_ENABLE_P2P=1)."""
    return _CRedisMetaChannel is not None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _resolve_namespace(
    namespace: Optional[SharingDomainNamespace],
) -> SharingDomainNamespace:
    """Coerce ``namespace`` to a non-None ``SharingDomainNamespace``.

    ``None`` collapses to the degenerate single-SD namespace, which keeps
    legacy dist_reuse callers compatible (the only observable difference is
    the new ``sd:__default__:...`` key prefix instead of the old bare keys).
    """
    if namespace is None:
        return SharingDomainNamespace(SharingDomainKey.default())
    if isinstance(namespace, SharingDomainKey):
        return SharingDomainNamespace(namespace)
    if isinstance(namespace, SharingDomainNamespace):
        return namespace
    raise TypeError(
        f"namespace must be SharingDomainNamespace or SharingDomainKey, got {type(namespace).__name__}"
    )


def _channel_blocks_key(namespace: SharingDomainNamespace, device_prefix: str) -> str:
    """Build the C++ ``blocks_key`` argument such that
    ``make_block_key(nid, hash)`` produces
    ``sd:<sd_key>:<device_prefix>:block:<nid>:<hash_hex>``.

    The C++ side concatenates ``<blocks_key>:block:<nid>:<hash>``, so we
    pre-pend the SD prefix here.
    """
    if device_prefix:
        return f"{namespace.prefix}:{device_prefix}"
    return namespace.prefix


# ---------------------------------------------------------------------------
# RedisMetaChannel — thin Python wrapper over the C++ extension
# ---------------------------------------------------------------------------
class RedisMetaChannel:
    """Wraps the C++ ``RedisMetaChannel`` so callers don't need to format
    SD-aware keys themselves.

    The constructor takes a fully-qualified ``blocks_key`` (already
    SD-prefixed by :func:`_channel_blocks_key`).  Existing call sites that
    used to pass ``"CPUB"`` / ``"SSDB"`` / ``"PCFSB"`` should now go through
    :meth:`RedisMeta.get_redis_meta_channel` which performs the SD prefix
    composition centrally.
    """

    def __init__(
        self,
        host: str,
        port: int,
        node_id: int,
        local_ip: str,
        blocks_key: str = "blocks",
        password: str = "",
        db: int = 0,
    ) -> None:
        if _CRedisMetaChannel is None:
            raise ImportError(
                "Distributed KV cache (P2P/Redis) is not built. "
                "Rebuild FlexKV with FLEXKV_ENABLE_P2P=1 and install Redis dependencies (e.g. libhiredis-dev, redis-tools)."
            )
        # ``db`` is a recent addition (matches ``CacheConfig.flexkv_redis_db``).
        # Older C++ builds may not accept the kwarg yet, so fall back to the
        # 6-arg constructor for backward compatibility during the rollout.
        try:
            self._c = _CRedisMetaChannel(
                host, int(port), int(node_id), str(local_ip),
                str(blocks_key), str(password), int(db),
            )
        except TypeError:
            # Legacy C++ build without ``db`` support.  Callers asking for
            # ``db != 0`` on a legacy build get a loud error — silent fallback
            # would corrupt key isolation.
            if int(db) != 0:
                raise ImportError(
                    "C++ RedisMetaChannel does not accept a ``db`` argument — "
                    "rebuild FlexKV with the updated csrc/dist/redis_meta_channel.* "
                    "to use CacheConfig.flexkv_redis_db != 0."
                )
            self._c = _CRedisMetaChannel(
                host, int(port), int(node_id), str(local_ip),
                str(blocks_key), str(password),
            )
        self._blocks_key = str(blocks_key)
        self._db = int(db)

    @property
    def blocks_key(self) -> str:
        return self._blocks_key

    def connect(self) -> bool:
        return bool(self._c.connect())

    @property
    def node_id(self) -> int:
        return int(self._c.get_node_id())

    @property
    def local_ip(self) -> str:
        return str(self._c.get_local_ip())

    def make_block_key(self, node_id: int, hash_value: int) -> str:
        return str(self._c.make_block_key(int(node_id), int(hash_value)))

    def publish_one(self, meta: BlockMeta) -> bool:
        """publish single BlockMeta to Redis"""
        return self._c.publish_one(meta.to_c())

    def publish_batch(self, metas: Iterable[BlockMeta], batch_size: int = 100) -> bool:
        """batch publish BlockMeta to Redis"""
        cms = [m.to_c() for m in metas]
        return self._c.publish_batch(cms, int(batch_size))

    def list_keys(self, pattern: str) -> List[str]:
        return list(self._c.list_keys(pattern))

    def list_node_keys(self) -> List[str]:
        """List ``sd:<sd_key>:node:*`` keys in this channel's SD.

        The C++ side scans for ``<blocks_key_root>:node:*`` where
        ``blocks_key_root`` is everything before the first ``:`` device-prefix
        component.  As of Phase 0 we re-implement the scan in Python (using
        the matching ``sd:<sd_key>:node:*`` pattern) so that this method
        works regardless of whether the C++ side has been rebuilt with the
        SD-aware ``list_node_keys`` of task 0-D.  Once the rebuilt C++ is
        rolled out everywhere the Python fallback can be removed.
        """
        # Best-effort: prefer C++ implementation if it's been updated; fall
        # back to pattern-based scan that mirrors the SD layout.
        try:
            keys = list(self._c.list_node_keys())
            # The C++ pre-task-0-D version returns "node:*" — those don't
            # belong to the SD layout, drop them.
            keys = [k for k in keys if k.startswith("sd:")]
        except Exception:
            keys = []
        if keys:
            return keys
        # Fallback: derive node-pattern from the channel's blocks_key
        # (everything before the optional device-prefix tail).
        sd_prefix = self._derive_sd_prefix()
        pattern = f"{sd_prefix}:node:*" if sd_prefix else "node:*"
        return list(self._c.list_keys(pattern))

    def list_block_keys(self, node_id: int) -> List[str]:
        return list(self._c.list_block_keys(int(node_id)))

    def list_all_block_keys(self) -> List[str]:
        """Global SCAN over every block key in this SD (design doc §4.7.1.2).

        Phase 0 stub: prefer the C++ method when present, otherwise fall back
        to ``list_keys(<blocks_key>:block:*)``.  Task 0-D will replace the
        fallback with a native C++ call once the bindings are rebuilt.
        """
        try:
            return list(self._c.list_all_block_keys())
        except Exception:
            return list(self._c.list_keys(f"{self._blocks_key}:block:*"))

    def hmget_field_for_keys(self, keys: Iterable[str], field: str) -> List[str]:
        return list(self._c.hmget_field_for_keys(list(keys), field))

    def hmget_two_fields_for_keys(self, keys: Iterable[str], f1: str, f2: str) -> List[Tuple[str, str]]:
        return [(a, b) for a, b in self._c.hmget_two_fields_for_keys(list(keys), f1, f2)]

    def renew_node_leases(self, node_id: int, new_lt: int, batch_size: int = 200) -> bool:
        """batch update lease time for specified node"""
        return self._c.renew_node_leases(int(node_id), int(new_lt), int(batch_size))

    def update_block_state_batch(self, node_id: int, hashes: Iterable[int], state: int, batch_size: int = 200) -> bool:
        return self._c.update_block_state_batch(int(node_id), list(int(h) for h in hashes), int(state), int(batch_size))

    def delete_blockmeta_batch(self, node_id: int, hashes: Iterable[int], batch_size: int = 200) -> bool:
        return self._c.delete_blockmeta_batch(int(node_id), list(int(h) for h in hashes), int(batch_size))

    # --- helpers ----------------------------------------------------------
    def _derive_sd_prefix(self) -> str:
        """Return the ``sd:<sd_key>`` portion of ``self._blocks_key`` if any.

        The convention (see :func:`_channel_blocks_key`) is::

            blocks_key = sd:<sd_key>:<device_prefix>     # full SD path
            blocks_key = sd:<sd_key>                      # SD-only (no device prefix)
            blocks_key = blocks                           # legacy / tests

        Under the simplified dist_reuse design (CP not in the SD key), the
        serialized SD has 4 segments: ``<model_id>:pp<r>/<s>:tpn<i>/<c>:nsa<0|1>``.
        With the leading ``sd:`` literal that makes ``sd:<sd_key>`` exactly
        5 colon-separated parts.
        """
        bk = self._blocks_key
        if not bk.startswith("sd:"):
            return ""
        # Format: sd:<model_id>:pp/<>:tpn/<>:nsa<0|1>[:<device>]
        parts = bk.split(":")
        # ``sd``, ``<model_id>``, ``pp...``, ``tpn...``, ``nsa...`` = 5 parts.
        if len(parts) < 5:
            return bk  # malformed but caller will get an empty result anyway
        return ":".join(parts[:5])


# ---------------------------------------------------------------------------
# RedisNodeInfo — heartbeat + active-node discovery, scoped to one SD
# ---------------------------------------------------------------------------
class RedisNodeInfo:
    """Manages the ``sd:<sd_key>:node:<id>`` key family.

    Each instance owns exactly one SD's namespace.  The Master typically
    creates one ``RedisNodeInfo`` per SD it participates in (task 0-G);
    Remote nodes create a single ``RedisNodeInfo`` for their own SD
    (task 0-F).
    """

    DEFAULT_NODE_TTL_SECONDS: int = 30

    def __init__(
        self,
        host: str,
        port: int,
        local_ip: str,
        password: str = "",
        node_ttl_seconds: int = 0,
        *,
        namespace: Optional[SharingDomainNamespace] = None,
        db: int = 0,
    ) -> None:
        if _redis is None:
            raise ImportError("redis-py is required: pip install redis")
        self.host = host
        self.port = int(port)
        self.local_ip = str(local_ip)
        self.password = str(password)
        # Honour ``CacheConfig.flexkv_redis_db`` so all FlexKV clients land
        # on the same logical db; defaulting to 0 keeps legacy behaviour.
        self.db: int = int(db)
        self.uuid = str(uuid1())
        self.node_ttl_seconds: int = node_ttl_seconds if node_ttl_seconds > 0 else self.DEFAULT_NODE_TTL_SECONDS
        self.heartbeat_interval_seconds: float = max(1.0, self.node_ttl_seconds / 3.0)

        self._namespace: SharingDomainNamespace = _resolve_namespace(namespace)

        self._node_id: Optional[int] = None
        self._running = False
        self._listener_thread: Optional[threading.Thread] = None
        self._heartbeat_thread: Optional[threading.Thread] = None
        self.current_node_id_set: Set[int] = set()
        self._client: Optional["_redis.Redis"] = None
        self._sub_client: Optional["_redis.Redis"] = None
        self._cleanup_done = False

        atexit.register(self._cleanup_on_exit)
        # Some hosting environments (e.g. unit tests, threadpool workers)
        # don't allow signal handlers in non-main threads.  Be tolerant.
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except (ValueError, OSError):  # pragma: no cover
            pass

    def __del__(self) -> None:
        try:
            self._cleanup_on_exit()
        except Exception:
            pass

    # -- properties --------------------------------------------------------
    @property
    def namespace(self) -> SharingDomainNamespace:
        return self._namespace

    @property
    def sd_key_str(self) -> str:
        return self._namespace.serialized_sd

    # -- connection lifecycle ---------------------------------------------
    def _get_client(self) -> "_redis.Redis":
        return _redis.Redis(
            host=self.host,
            port=self.port,
            db=self.db,
            password=self.password if self.password else None,
            decode_responses=True,
            health_check_interval=30,
            socket_keepalive=True,
        )

    def connect(self) -> bool:
        try:
            self._client = self._get_client()
            self._client.ping()

            self._running = True
            self._listener_thread = threading.Thread(
                target=self._listener_worker,
                name=f"redis-node-info-listener[{self.sd_key_str}]",
                daemon=True,
            )
            self._listener_thread.start()

            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_worker,
                name=f"redis-node-heartbeat[{self.sd_key_str}]",
                daemon=True,
            )
            self._heartbeat_thread.start()

            return True
        except Exception:
            return False

    def disconnect(self) -> None:
        self._running = False
        if self._listener_thread and self._listener_thread.is_alive():
            self._listener_thread.join(timeout=2.0)
        self._listener_thread = None

        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2.0)
        self._heartbeat_thread = None

        if self._client:
            self._client.close()
            self._client = None
        if self._sub_client:
            self._sub_client.close()
            self._sub_client = None

    def _signal_handler(self, signum: int, frame) -> None:  # pragma: no cover
        print(f"received signal {signum}, starting cleanup of RedisNodeInfo[{self.sd_key_str}]...")
        self._cleanup()
        sys.exit(0)

    def _cleanup_on_exit(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        if self._cleanup_done:
            return
        self._cleanup_done = True
        try:
            if self._node_id is not None:
                self.unregister_node()
            self.disconnect()
        except Exception:
            pass

    # -- node registration -------------------------------------------------
    def _pubsub_channel(self) -> str:
        # Per-SD pub/sub channel so cross-SD events don't pollute each other.
        return f"flexkv_node_id_updated:{self.sd_key_str}"

    def register_node(self) -> Optional[int]:
        """Allocate a new global node_id, write ``sd:<sd_key>:node:<id>`` with TTL."""
        if not self._client:
            return None
        try:
            # SD-scoped stale cleanup: drop any same-IP, different-UUID keys
            # in this SD before claiming a new id.
            self._cleanup_stale_nodes_by_ip()

            # Atomic global counter — node_ids are globally unique even
            # though the keys are SD-scoped.  This makes BlockMeta.nid
            # unambiguous when two SDs of the same instance look at each
            # other's metadata (rare but possible during failover).
            # redis-py 5.x types ``incr`` as ``Awaitable[int] | int``; we're
            # always sync here — cast to silence mypy.
            node_id = cast(int, self._client.incr("global:node_id"))
            self._node_id = node_id

            node_key = self._namespace.node_key(node_id)
            self._client.hset(node_key, mapping={
                "node_id": str(node_id),
                "ip": self.local_ip,
                "local_ip": self.local_ip,
                "uuid": self.uuid,
                "status": "active",
                "timestamp": str(int(time.time())),
                "sd_key": self.sd_key_str,
            })
            self._client.expire(node_key, self.node_ttl_seconds)
            self._client.publish(self._pubsub_channel(), str(node_id))
            return node_id
        except Exception:
            return None

    def unregister_node(self) -> bool:
        if not self._client or self._node_id is None:
            return False
        try:
            node_id = self._node_id
            node_key = self._namespace.node_key(node_id)
            self._client.delete(node_key)
            self._cleanup_node_data(node_id)
            self._client.publish(self._pubsub_channel(), str(node_id))
            self._node_id = None
            return True
        except Exception:
            return False

    @property
    def node_id(self) -> Optional[int]:
        return self._node_id

    def get_uuid(self) -> str:
        return self.uuid

    def get_active_node_ids(self) -> List[int]:
        return list(self.current_node_id_set)

    def is_node_active(self, node_id: int) -> bool:
        return node_id in self.current_node_id_set

    # -- heartbeat / listener ---------------------------------------------
    def _heartbeat_worker(self) -> None:
        heartbeat_client: Optional["_redis.Redis"] = None
        while self._running:
            try:
                if heartbeat_client is None:
                    heartbeat_client = self._get_client()
                if self._node_id is not None:
                    node_key = self._namespace.node_key(self._node_id)
                    heartbeat_client.expire(node_key, self.node_ttl_seconds)
                    heartbeat_client.hset(node_key, "timestamp", str(int(time.time())))
            except Exception:
                if heartbeat_client:
                    try:
                        heartbeat_client.close()
                    except Exception:
                        pass
                    heartbeat_client = None
            for _ in range(int(self.heartbeat_interval_seconds * 10)):
                if not self._running:
                    break
                time.sleep(0.1)
        if heartbeat_client:
            try:
                heartbeat_client.close()
            except Exception:
                pass

    def _listener_worker(self) -> None:
        backoff = 0.5
        ch = self._pubsub_channel()
        while self._running:
            try:
                self._sub_client = self._get_client()
                pubsub = self._sub_client.pubsub()
                pubsub.subscribe(ch)
                for message in pubsub.listen():
                    if not self._running:
                        break
                    if message["type"] == "message" and message["channel"] == ch:
                        self.scan_active_nodes()
                break
            except Exception:
                time.sleep(backoff)
                backoff = min(backoff * 2, 5.0)
            finally:
                if self._sub_client:
                    try:
                        self._sub_client.close()
                    except Exception:
                        pass
                self._sub_client = None

    # -- discovery ---------------------------------------------------------
    def scan_active_nodes(self) -> None:
        if not self._client:
            return
        try:
            new_active_nodes: Set[int] = set()
            cursor = 0
            pattern = self._namespace.node_key_pattern()
            prefix = f"{self._namespace.prefix}:node:"
            while True:
                cursor, keys = cast(
                    Tuple[int, List[str]],
                    self._client.scan(cursor=cursor, match=pattern, count=100),
                )
                for key in keys:
                    if not key.startswith(prefix):
                        continue
                    try:
                        node_id = int(key[len(prefix):])
                        new_active_nodes.add(node_id)
                    except (ValueError, IndexError):
                        continue
                if cursor == 0:
                    break

            disappeared = self.current_node_id_set - new_active_nodes
            if disappeared:
                for stale_nid in disappeared:
                    if stale_nid == self._node_id:
                        continue
                    self._cleanup_node_data(stale_nid)
            self.current_node_id_set = new_active_nodes
        except Exception:
            pass

    def _cleanup_stale_nodes_by_ip(self) -> None:
        if not self._client:
            return
        try:
            cursor = 0
            stale_node_ids: List[int] = []
            pattern = self._namespace.node_key_pattern()
            prefix = f"{self._namespace.prefix}:node:"
            while True:
                cursor, keys = cast(
                    Tuple[int, List[str]],
                    self._client.scan(cursor=cursor, match=pattern, count=100),
                )
                for key in keys:
                    if not key.startswith(prefix):
                        continue
                    try:
                        nid = int(key[len(prefix):])
                    except (ValueError, IndexError):
                        continue
                    data = cast(Dict[str, str], self._client.hgetall(key) or {})
                    node_ip = data.get("ip", "") or data.get("local_ip", "")
                    node_uuid = data.get("uuid", "")
                    if node_ip == self.local_ip and node_uuid != self.uuid:
                        stale_node_ids.append(nid)
                if cursor == 0:
                    break

            for stale_nid in stale_node_ids:
                print(
                    f"[RedisNodeInfo:{self.sd_key_str}] Cleaning up stale "
                    f"node:{stale_nid} (same IP={self.local_ip}, different UUID)"
                )
                self._client.delete(self._namespace.node_key(stale_nid))
                self._cleanup_node_data(stale_nid)

            if stale_node_ids:
                self._client.publish(self._pubsub_channel(), "cleanup")
        except Exception:
            pass

    def _cleanup_node_data(self, node_id: int) -> None:
        """Drop every key associated with a dead node in this SD.

        Removes ``meta:<node_id>``, ``buffer:<node_id>:*`` and the per-device
        ``...:<DEVICE>:block:<node_id>:*`` families.  All under
        ``sd:<sd_key>:`` so other SDs are unaffected.
        """
        if not self._client:
            return
        try:
            # meta key (single key, no SCAN)
            meta_key = self._namespace.meta_key(node_id)
            self._client.delete(meta_key)

            # buffer keys: sd:<sd>:buffer:<node_id>:*
            cursor = 0
            buffer_pattern = f"{self._namespace.prefix}:buffer:{int(node_id)}:*"
            buffer_keys: List[str] = []
            while True:
                cursor, keys = cast(
                    Tuple[int, List[str]],
                    self._client.scan(cursor=cursor, match=buffer_pattern, count=200),
                )
                buffer_keys.extend(keys)
                if cursor == 0:
                    break
            if buffer_keys:
                for i in range(0, len(buffer_keys), 500):
                    self._client.delete(*buffer_keys[i:i + 500])

            # block keys, both legacy device-prefix flavours and the new
            # device-less SD-only flavour:
            #   sd:<sd>:CPUB:block:<nid>:*
            #   sd:<sd>:SSDB:block:<nid>:*
            #   sd:<sd>:PCFSB:block:<nid>:*
            #   sd:<sd>:block:<nid>:*  (when no device prefix is used)
            patterns = [
                f"{self._namespace.prefix}:CPUB:block:{int(node_id)}:*",
                f"{self._namespace.prefix}:SSDB:block:{int(node_id)}:*",
                f"{self._namespace.prefix}:PCFSB:block:{int(node_id)}:*",
                self._namespace.block_key_pattern_for_node(node_id),
            ]
            for pat in patterns:
                cursor = 0
                block_keys: List[str] = []
                while True:
                    cursor, keys = cast(
                        Tuple[int, List[str]],
                        self._client.scan(cursor=cursor, match=pat, count=500),
                    )
                    block_keys.extend(keys)
                    if cursor == 0:
                        break
                if block_keys:
                    for i in range(0, len(block_keys), 500):
                        self._client.delete(*block_keys[i:i + 500])
        except Exception as e:
            print(f"[RedisNodeInfo:{self.sd_key_str}] Warning: failed to clean up data for node {node_id}: {e}")


# ---------------------------------------------------------------------------
# RedisMeta — top-level facade
# ---------------------------------------------------------------------------
class RedisMeta:
    """Top-level wrapper that owns one SD's :class:`RedisNodeInfo` plus a
    cached redis-py client for non-block metadata writes (``meta:`` /
    ``buffer:`` / ``flexkv:instance:*``)."""

    def __init__(
        self,
        host: str,
        port: int,
        password: Optional[str] = None,
        local_ip: str = "127.0.0.1",
        decode_responses: bool = True,
        node_ttl_seconds: int = 0,
        *,
        namespace: Optional[SharingDomainNamespace] = None,
        db: int = 0,
    ) -> None:
        if _redis is None:  # pragma: no cover
            raise ImportError("redis-py is required: pip install redis")
        self.host = host
        self.port = int(port)
        self.local_ip = str(local_ip)
        # Logical Redis db — comes from ``CacheConfig.flexkv_redis_db``.
        # Kept as an instance attr so every ``self._client()`` call picks
        # up the same value; must match ``RedisNodeInfo.db`` below or the
        # node-set and block-set end up on different dbs!
        self.db: int = int(db)
        self.password = password
        self.decode_responses = bool(decode_responses)
        self._node_id: Optional[int] = None

        self._namespace: SharingDomainNamespace = _resolve_namespace(namespace)

        self._init_lock = threading.Lock()
        self._initialized = False
        self._init_error: Optional[Exception] = None

        self.nodeinfo = RedisNodeInfo(
            host, port, local_ip, password or "",
            node_ttl_seconds=node_ttl_seconds,
            namespace=self._namespace,
            db=self.db,
        )
        self._uuid = self.nodeinfo.get_uuid()

    # -- properties --------------------------------------------------------
    @property
    def namespace(self) -> SharingDomainNamespace:
        return self._namespace

    @property
    def sd_key_str(self) -> str:
        return self._namespace.serialized_sd

    def _client(self) -> "_redis.Redis":
        return _redis.Redis(
            host=self.host, port=self.port, db=self.db,
            password=self.password, decode_responses=self.decode_responses,
        )

    # -- lifecycle ---------------------------------------------------------
    def init_meta(self) -> Optional[int]:
        with self._init_lock:
            if self._initialized:
                if self._init_error:
                    raise self._init_error
                return self._node_id
            try:
                if not self.nodeinfo.connect():
                    raise RuntimeError("Failed to connect to Redis via RedisNodeInfo")
                node_id = self.nodeinfo.register_node()
                if node_id is None:
                    raise RuntimeError("Failed to register node via RedisNodeInfo")
                self._node_id = node_id
                self.nodeinfo.scan_active_nodes()
                self._initialized = True
                return node_id
            except Exception as e:
                self._init_error = e
                return None

    def get_node_id(self) -> int:
        if self._node_id is None:
            raise RuntimeError("node_id is not registered yet. Call init_meta() first.")
        return int(self._node_id)

    def is_initialized(self) -> bool:
        with self._init_lock:
            return self._initialized

    def get_init_error(self) -> Optional[Exception]:
        with self._init_lock:
            return self._init_error

    # -- channel factory ---------------------------------------------------
    def get_redis_meta_channel(self, device_prefix: str = "") -> "RedisMetaChannel":
        """Build a C++-backed ``RedisMetaChannel`` whose key-prefix is
        ``sd:<sd_key>[:<device_prefix>]``.

        The legacy parameter name ``blocks_key`` (which used to carry
        ``"CPUB"`` / ``"SSDB"`` / ``"PCFSB"``) is replaced by the more
        explicit ``device_prefix`` to make the SD prefixing obvious in
        callers.  Callers that pass ``device_prefix=""`` get a channel
        whose ``make_block_key`` produces ``sd:<sd_key>:block:<nid>:<hash>``
        (used by Master nodes that don't multi-tier blocks across CPU /
        SSD / PCFS).
        """
        nid = self.get_node_id()
        pwd = "" if (self.password is None or str(self.password).lower() == "none") else str(self.password)
        bk = _channel_blocks_key(self._namespace, str(device_prefix))
        channel = RedisMetaChannel(
            self.host, int(self.port), int(nid),
            self.local_ip, bk, pwd, db=int(self.db),
        )
        if not channel.connect():
            raise RuntimeError("Failed to connect to Redis")
        return channel

    def unregister_node(self, node_id: Optional[int] = None) -> None:
        if self.nodeinfo:
            self.nodeinfo.unregister_node()
        self._node_id = None

    def get_uuid(self) -> str:
        return self._uuid

    def get_active_node_ids(self) -> List[int]:
        if self.nodeinfo:
            return self.nodeinfo.get_active_node_ids()
        return []

    def is_node_active(self, node_id: int) -> bool:
        if self.nodeinfo:
            return self.nodeinfo.is_node_active(node_id)
        return False

    # -- pcfs file-nodeid mapping (legacy; SD-scoped now) ------------------
    def add_node_ids(self, node_ids: Iterable[Union[int, str]]) -> int:
        nid = self.get_node_id()
        values = [str(v) for v in node_ids]
        if not values:
            return 0
        r = self._client()
        return int(cast(int, r.rpush(f"{self._namespace.prefix}:pcfs:{nid}", *values)))

    def load_pcfs_file_nodeids(self) -> Dict[int, List[int]]:
        r = self._client()
        result: Dict[int, List[int]] = {}
        try:
            cursor = 0
            pattern = f"{self._namespace.prefix}:pcfs:*"
            prefix = f"{self._namespace.prefix}:pcfs:"
            while True:
                cursor, keys = cast(
                    Tuple[int, List[str]],
                    r.scan(cursor=cursor, match=pattern, count=100),
                )
                for key in keys:
                    if not isinstance(key, str):
                        key = str(key)
                    if not key.startswith(prefix):
                        continue
                    try:
                        node_id = int(key[len(prefix):])
                    except Exception:
                        continue
                    try:
                        values = cast(List[str], r.lrange(key, 0, -1) or [])
                        file_nodeids = [int(v) for v in values]
                    except Exception:
                        file_nodeids = []
                    result[node_id] = file_nodeids
                if cursor == 0:
                    break
        except Exception:
            return result
        return result

    # -- buffer registration ----------------------------------------------
    def regist_buffer(self, mrs: Iterable[object]) -> int:
        nid = self.get_node_id()
        r = self._client()
        pipe = r.pipeline()
        processed = 0
        for mr in mrs:
            if isinstance(mr, dict):
                ptr = mr.get("buffer_ptr")
                size = mr.get("buffer_size")
            elif isinstance(mr, (tuple, list)) and len(mr) >= 2:
                ptr, size = mr[0], mr[1]
            else:
                continue
            if ptr is None or size is None:
                continue
            key = self._namespace.buffer_key(nid, int(ptr))
            pipe.hset(key, mapping={"buffer_size": int(size)})
            processed += 1
        if processed:
            pipe.execute()
        return processed

    def unregist_buffer(self, buffer_ptr: Union[int, str]) -> bool:
        nid = self.get_node_id()
        key = self._namespace.buffer_key(nid, int(buffer_ptr))
        r = self._client()
        exists = bool(r.exists(key))
        if exists:
            r.delete(key)
            return True
        return False

    # -- node meta hash ----------------------------------------------------
    def regist_node_meta(self, node_id: int, addr: str, zmq_addr: str, cpu_buffer_ptr: int, ssd_buffer_ptr: int) -> None:
        r = self._client()
        key = self._namespace.meta_key(node_id)
        r.hset(key, mapping={
            "node_id": int(node_id),
            "addr": str(addr),
            "zmq_addr": str(zmq_addr),
            "cpu_buffer_ptr": int(cpu_buffer_ptr),
            "ssd_buffer_ptr": int(ssd_buffer_ptr),
        })

    def get_node_meta(self, node_id: int) -> dict:
        r = self._client()
        key = self._namespace.meta_key(node_id)
        data = cast(Dict[str, str], r.hgetall(key) or {})
        if not data:
            return {}
        out: Dict[str, Union[int, str]] = {}
        nid = data.get("node_id")
        out["node_id"] = int(nid) if nid is not None and nid != "" else int(node_id)
        out["addr"] = data.get("addr", "")
        out["zmq_addr"] = data.get("zmq_addr", "")
        cb = data.get("cpu_buffer_ptr")
        sb = data.get("ssd_buffer_ptr")
        out["cpu_buffer_ptr"] = int(cb) if cb is not None and cb != "" else 0
        out["ssd_buffer_ptr"] = int(sb) if sb is not None and sb != "" else 0
        return out

    def unregist_node_meta(self, node_id: int) -> bool:
        r = self._client()
        key = self._namespace.meta_key(node_id)
        return bool(r.delete(key))

    def set_node_id(self, node_id: int) -> None:
        self._node_id = int(node_id)

    # -- instance-level helpers (cross-SD; design doc §4.7.1.6) -----------
    def register_instance_sd_nodes(self, instance_id: str, sd_to_nid: Dict[str, int]) -> None:
        """Write the ``sd_key → node_id`` mapping for one full FlexKV instance.

        Called once on Master startup after collecting all Remote ack'ed
        node_ids.  Peers consume this via :meth:`load_instance_sd_nodes`
        when first discovering a new instance via the failure detector.
        """
        if not sd_to_nid:
            return
        r = self._client()
        key = SharingDomainNamespace.instance_sd_nodes_key(instance_id)
        # Stringify values so HSET round-trips cleanly through redis-py.
        mapping = {str(sd): str(int(nid)) for sd, nid in sd_to_nid.items()}
        r.hset(key, mapping=mapping)

    def load_instance_sd_nodes(self, instance_id: str) -> Dict[str, int]:
        r = self._client()
        key = SharingDomainNamespace.instance_sd_nodes_key(instance_id)
        # redis-py 5.x types ``hgetall`` as ``Awaitable[dict] | dict`` so that
        # the same stub serves both sync and async clients.  We're always in
        # sync mode here — cast defensively.
        data = cast(Dict[str, str], r.hgetall(key) or {})
        out: Dict[str, int] = {}
        for sd_str, nid_str in data.items():
            try:
                out[str(sd_str)] = int(nid_str)
            except (TypeError, ValueError):
                continue
        return out

    def unregister_instance_sd_nodes(self, instance_id: str) -> bool:
        r = self._client()
        key = SharingDomainNamespace.instance_sd_nodes_key(instance_id)
        return bool(r.delete(key))
