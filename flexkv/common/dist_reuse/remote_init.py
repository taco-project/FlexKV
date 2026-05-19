"""Remote-side bootstrap helper for sharing-domain aware dist_reuse.

Phase 0 task 0-F:  when ``CacheConfig.enable_sharing_domain=True``, a
``TransferManagerOnRemote`` needs to:

1. Create a per-SD ``RedisMeta`` and register itself as a node under the
   ``sd:<sd_key>:*`` namespace.
2. Initialize Mooncake TransferEngine and register its local CPU block
   pool as a P2P source.
3. Publish its ``(sd_key, distributed_node_id, mooncake_addr, zmq_addr)``
   tuple back to the Master via a :class:`RemoteReadyMsg`.

This module isolates that logic so ``transfer_manager.py`` stays minimally
intrusive — the existing ``_initialize_with_config`` path only needs to
call :meth:`RemoteDistReuseInitializer.bootstrap` when the config says
sharing-domain is enabled.

The class has **no hard dependency on Mooncake or the C++ extension** —
both are looked up lazily at runtime, so the module can be imported (and
the type checked) on CPU-only test machines.  The only required runtime
dep is redis-py, which is already a peer dep of ``flexkv.cache.redis_meta``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Optional

from flexkv.common.dist_reuse import (
    RemoteReadyMsg,
    SharingDomainKey,
    SharingDomainNamespace,
    encode_coord_message,
)


__all__ = [
    "BootstrapResult",
    "RemoteDistReuseInitializer",
]


_LOG = logging.getLogger("flexkv.dist_reuse.remote_init")


# ---------------------------------------------------------------------------
# Result payload
# ---------------------------------------------------------------------------
@dataclass
class BootstrapResult:
    """The outcome of a Remote bootstrap.

    ``ready_msg`` is ready to be sent to the Master via its ZMQ result
    socket (``encode_coord_message`` applied beforehand if the transport
    is JSON-oriented; pickle-based transports can send the dataclass
    directly).  ``redis_meta`` and ``mooncake_engine`` are held by the
    Remote so their lifetime matches the worker process.
    """

    sd_key: SharingDomainKey
    namespace: SharingDomainNamespace
    redis_meta: Any              # RedisMeta — typed as Any to avoid hard import
    mooncake_engine: Any         # MooncakeTransferEngine — same
    distributed_node_id: int
    ready_msg: RemoteReadyMsg


# ---------------------------------------------------------------------------
# Initializer
# ---------------------------------------------------------------------------
class RemoteDistReuseInitializer:
    """Drives the bootstrap sequence for a single Remote node.

    Usage (pseudocode, actual wiring lives in
    ``TransferManagerOnRemote._initialize_with_config``)::

        init = RemoteDistReuseInitializer(
            cache_config=self.cache_config,
            sd_key_str=config_msg["sd_key"],
            instance_id=config_msg["instance_id"],
            session_epoch=config_msg["session_epoch"],
            cpu_block_pool=self._cpu_block_pool,
            local_zmq_addr=self._local_zmq_addr,
            redis_meta_factory=_make_redis_meta,      # Optional
            mooncake_engine_factory=_make_mooncake,   # Optional
        )
        result = init.bootstrap()
        self._redis_meta = result.redis_meta
        self._mooncake_engine = result.mooncake_engine
        self._report_ready_to_master(result.ready_msg)

    Factory callables let the production wiring inject the real
    ``RedisMeta`` / ``MooncakeTransferEngine``, while unit tests can pass
    in stubs (used by :mod:`tests.test_remote_dist_reuse_initializer`).
    """

    # ---- default factories --------------------------------------------
    @staticmethod
    def _default_redis_meta_factory(
        cache_config: Any, namespace: SharingDomainNamespace
    ) -> Any:
        # Lazy import to keep ``flexkv.cache.redis_meta`` out of the
        # critical module graph for CPU-only envs.
        from flexkv.cache.redis_meta import RedisMeta

        return RedisMeta(
            host=cache_config.redis_host,
            port=cache_config.redis_port,
            password=cache_config.redis_password,
            local_ip=cache_config.local_ip,
            node_ttl_seconds=cache_config.node_ttl_seconds,
            namespace=namespace,
            db=int(getattr(cache_config, "flexkv_redis_db", 0)),
        )

    @staticmethod
    def _default_mooncake_factory(cache_config: Any) -> Any:
        # Lazy import so the Mooncake wheel isn't required at import time.
        # The public alias is ``MoonCakeTransferEngineWrapper`` (camelcase).
        from flexkv.mooncakeEngineWrapper import MoonCakeTransferEngineWrapper  # type: ignore[attr-defined]

        engine = MoonCakeTransferEngineWrapper(cache_config.mooncake_config_path)
        return engine

    # ---- lifecycle ----------------------------------------------------
    def __init__(
        self,
        *,
        cache_config: Any,
        sd_key_str: str,
        instance_id: str,
        session_epoch: str,
        cpu_buffer_ptr: int,
        cpu_buffer_size: int,
        local_zmq_addr: str,
        redis_meta_factory: Optional[Callable[[Any, SharingDomainNamespace], Any]] = None,
        mooncake_engine_factory: Optional[Callable[[Any], Any]] = None,
    ) -> None:
        self._cache_config = cache_config
        self._sd_key: SharingDomainKey = SharingDomainKey.deserialize(sd_key_str)
        self._namespace: SharingDomainNamespace = SharingDomainNamespace(self._sd_key)
        self._instance_id = str(instance_id)
        self._session_epoch = str(session_epoch)
        self._cpu_buffer_ptr = int(cpu_buffer_ptr)
        self._cpu_buffer_size = int(cpu_buffer_size)
        self._local_zmq_addr = str(local_zmq_addr)
        self._redis_meta_factory = redis_meta_factory or self._default_redis_meta_factory
        self._mooncake_engine_factory = mooncake_engine_factory or self._default_mooncake_factory

    # ---- main entrypoint ---------------------------------------------
    def bootstrap(self) -> BootstrapResult:
        """Run the three-step sequence; return a :class:`BootstrapResult`.

        Any failure propagates the original exception — callers are
        expected to treat that as fatal (the instance is "co-destined" with
        its Master; design doc §4.3.1).
        """
        # 1. Redis side: register node, get global distributed_node_id.
        redis_meta = self._redis_meta_factory(self._cache_config, self._namespace)
        node_id = redis_meta.init_meta()
        if node_id is None:
            err = getattr(redis_meta, "get_init_error", lambda: None)()
            raise RuntimeError(
                f"[RemoteDistReuseInit:{self._sd_key.serialize()}] "
                f"redis init_meta() failed: {err!r}"
            )
        _LOG.info(
            "[RemoteDistReuseInit:%s] registered as node_id=%d",
            self._sd_key.serialize(),
            node_id,
        )

        # 2. Mooncake side: init engine, register the CPU block pool buffer,
        # publish node meta back to Redis so the Master's peer discovery can
        # find us.
        mooncake_engine = self._mooncake_engine_factory(self._cache_config)
        _init_mooncake_if_needed(mooncake_engine, self._cache_config)

        regist = getattr(mooncake_engine, "regist_buffer", None)
        if regist is None:
            raise AttributeError("mooncake engine lacks regist_buffer()")
        regist(self._cpu_buffer_ptr, self._cpu_buffer_size)

        # Record buffer + node meta in Redis so peers can resolve the
        # (nid -> mooncake_addr, zmq_addr, cpu_buffer_ptr) triple.
        redis_meta.regist_buffer([{
            "buffer_ptr": self._cpu_buffer_ptr,
            "buffer_size": self._cpu_buffer_size,
        }])
        mooncake_addr = _safe_call(mooncake_engine, "get_engine_addr", default="")
        redis_meta.regist_node_meta(
            node_id=node_id,
            addr=str(mooncake_addr),
            zmq_addr=self._local_zmq_addr,
            cpu_buffer_ptr=self._cpu_buffer_ptr,
            ssd_buffer_ptr=0,
        )

        # 3. Build the ready message for the Master.
        ready_msg = RemoteReadyMsg(
            sender_instance_id=self._instance_id,
            sender_epoch=self._session_epoch,
            request_id=-1,
            sd_key=self._sd_key.serialize(),
            distributed_node_id=int(node_id),
            mooncake_addr=str(mooncake_addr),
            zmq_addr=self._local_zmq_addr,
        )

        return BootstrapResult(
            sd_key=self._sd_key,
            namespace=self._namespace,
            redis_meta=redis_meta,
            mooncake_engine=mooncake_engine,
            distributed_node_id=int(node_id),
            ready_msg=ready_msg,
        )

    # ---- encoding --------------------------------------------------
    @staticmethod
    def encode_ready(msg: RemoteReadyMsg) -> dict:
        """Convenience: turn a :class:`RemoteReadyMsg` into its wire
        ``dict`` form (handy when the ZMQ transport prefers JSON)."""
        return encode_coord_message(msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _init_mooncake_if_needed(engine: Any, cache_config: Any) -> None:
    """Call ``engine.init(mooncake_config_path)`` iff the engine exposes
    an ``init`` hook.  The Mooncake Python wrapper calls its own init in
    the constructor, so this is only needed for user-supplied stubs."""
    init_fn = getattr(engine, "init", None)
    if callable(init_fn):
        path = getattr(cache_config, "mooncake_config_path", None)
        if path is not None:
            init_fn(path)


def _safe_call(obj: Any, method: str, default: Any = None) -> Any:
    fn = getattr(obj, method, None)
    if fn is None:
        return default
    try:
        return fn()
    except Exception as e:
        _LOG.warning("%s.%s() raised: %s", type(obj).__name__, method, e)
        return default
