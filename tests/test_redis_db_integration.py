"""Integration test against a real Redis: verify ``flexkv_redis_db``
actually switches the logical database at the protocol level.

Auto-skips when Redis is not reachable.  Uses two different db numbers
(primary db ``0`` vs override db ``15`` — both always exist on a stock
Redis) and asserts that:

  * a key written from a ``RedisMeta`` bound to db=0 is **not** visible
    to a client bound to db=15, and vice-versa;
  * ``RedisSessionClient`` built through the factory respects db=15.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Iterator, List

import pytest

REDIS_HOST = os.environ.get("FLEXKV_TEST_REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("FLEXKV_TEST_REDIS_PORT", "6379"))

try:
    import redis as _redis  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    pytest.skip("redis-py not installed", allow_module_level=True)


def _probe_db(db: int) -> bool:
    try:
        c = _redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=db,
                         socket_connect_timeout=1.0, decode_responses=True)
        return bool(c.ping())
    except Exception:
        return False


if not (_probe_db(0) and _probe_db(15)):
    pytest.skip(
        f"Redis not reachable at {REDIS_HOST}:{REDIS_PORT} on both db=0 and db=15",
        allow_module_level=True,
    )


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_rm = _load("_rm_db_it", REPO_ROOT / "flexkv" / "cache" / "redis_meta.py")
RedisMeta = _rm.RedisMeta

from flexkv.common.dist_reuse import (  # noqa: E402
    RedisSessionClient,
    SharingDomainKey,
    SharingDomainNamespace,
    make_redis_client_from_cache_config,
    make_session_epoch,
)


@pytest.fixture
def tracked_db0() -> Iterator["_redis.Redis"]:
    c = _redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0,
                     decode_responses=True)
    keys: List[str] = []
    c._test_tracked_keys = keys  # type: ignore[attr-defined]
    yield c
    for k in keys:
        try:
            c.delete(k)
        except Exception:
            pass


@pytest.fixture
def tracked_db15() -> Iterator["_redis.Redis"]:
    c = _redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=15,
                     decode_responses=True)
    keys: List[str] = []
    c._test_tracked_keys = keys  # type: ignore[attr-defined]
    yield c
    for k in keys:
        try:
            c.delete(k)
        except Exception:
            pass


def _track(client, *keys: str) -> None:
    client._test_tracked_keys.extend(keys)  # type: ignore[attr-defined]


class _FakeCacheConfig:
    """Minimal stand-in for ``CacheConfig`` — keeps this test file free of
    the torch/zmq import chain that the real ``flexkv.common.config`` drags
    in (same trick as ``test_redis_meta_namespace.py``)."""

    def __init__(self, *, db: int):
        self.redis_host = REDIS_HOST
        self.redis_port = REDIS_PORT
        self.redis_password = None
        self.flexkv_redis_db = db


# ===========================================================================
# Real-Redis: writes to db=15 are invisible on db=0 (and vice-versa)
# ===========================================================================
def test_redismeta_db15_key_not_visible_on_db0(tracked_db0, tracked_db15):
    sd = SharingDomainKey(
        model_id="db-test-" + uuid.uuid4().hex[:10],
        pp_rank=0, pp_size=1, tp_node_idx=0, tp_node_count=1,
        is_nsa=False,
    )
    ns = SharingDomainNamespace(sd)

    meta_db15 = RedisMeta(
        host=REDIS_HOST, port=REDIS_PORT, local_ip="10.0.99.31",
        namespace=ns, db=15,
    )
    try:
        nid = meta_db15.init_meta()
        assert nid is not None, "init_meta failed on db=15"
        node_key = ns.node_key(nid)
        _track(tracked_db15, node_key)

        # Key is present on db=15 but NOT on db=0.
        assert tracked_db15.exists(node_key) == 1, (
            "Expected node key on db=15 after RedisMeta(db=15).init_meta()"
        )
        assert tracked_db0.exists(node_key) == 0, (
            f"Key {node_key} leaked to db=0 — db selection is broken!"
        )

        # Conversely: drop a sentinel on db=0 and assert db=15 doesn't see it.
        sentinel_key = f"sentinel:{uuid.uuid4().hex[:8]}"
        tracked_db0.set(sentinel_key, "db0-value")
        _track(tracked_db0, sentinel_key)
        assert tracked_db0.get(sentinel_key) == "db0-value"
        assert tracked_db15.exists(sentinel_key) == 0, (
            "db=15 client saw a db=0 sentinel — dbs are not isolated!"
        )
    finally:
        meta_db15.unregister_node()


def test_make_redis_client_from_cache_config_uses_db(tracked_db0, tracked_db15):
    """RedisSessionClient built with factory client must land on the
    right db."""
    cfg = _FakeCacheConfig(db=15)
    client = make_redis_client_from_cache_config(cfg)

    instance_id = f"dbsess-{uuid.uuid4().hex[:8]}"
    sess = RedisSessionClient(
        redis_client=client,
        instance_id=instance_id,
        epoch=make_session_epoch(),
        ttl_seconds=5,
    )
    sess.register()
    _track(tracked_db15, sess.key)

    # Session key present on db=15, absent on db=0.
    assert tracked_db15.exists(sess.key) == 1
    assert tracked_db0.exists(sess.key) == 0


def test_redismetachannel_python_wrapper_records_db():
    """Pure-python smoke test — verify the C++ ctor is invoked with db=N.

    We don't actually exercise the C++ binary here (would need a GPU build);
    we just assert the Python wrapper forwards the arg correctly.  The real
    C++ SELECT behaviour is exercised by
    ``test_redismeta_db15_key_not_visible_on_db0`` above, which uses only
    the Python ``RedisMeta`` path (redis-py supports SELECT natively).
    """
    # The wrapper ctor may raise ImportError when _CRedisMetaChannel is None
    # (no FLEXKV_ENABLE_P2P build).  Skip gracefully in that case.
    if _rm._CRedisMetaChannel is None:
        pytest.skip("flexkv.c_ext.RedisMetaChannel not built (FLEXKV_ENABLE_P2P=0)")
    # Otherwise try the real construction against Redis on db=15.
    ch = _rm.RedisMetaChannel(
        host=REDIS_HOST, port=REDIS_PORT, node_id=99999,
        local_ip="127.0.0.1", blocks_key="sd:dbsmoke:CPUB",
        password="", db=15,
    )
    assert ch._db == 15
    # connect() should succeed even on db=15 (assuming server allows it).
    assert ch.connect() is True
