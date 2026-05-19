"""Integration tests against a **real** Redis server.

Complement to the FakeRedis-based unit tests — FakeRedis cannot reproduce
real TTL expiry timing, real HMGET/SCAN pipelining, or HGETALL ordering,
so the Phase 0 / Phase 1 code paths that depend on those behaviours need a
live Redis to be exercised end-to-end.

Usage::

    FLEXKV_TEST_REDIS_HOST=127.0.0.1 \
    FLEXKV_TEST_REDIS_PORT=6379 \
    pytest tests/test_redis_integration.py -v

All tests auto-skip when Redis is not reachable.  Each test uses a
randomly-generated ``model_id`` / ``instance_id`` so the shared Redis
(which may already contain ~1000 keys from other tenants) is not disturbed,
and teardown deletes only keys under the test's own SD / instance prefix.

``RedisMeta`` is loaded via ``importlib.util`` to bypass
``flexkv.cache.__init__``'s forced ``import flexkv.c_ext`` — consistent
with the other dist_reuse unit tests.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Iterator, List, Tuple

import pytest

# ---------------------------------------------------------------------------
# Redis availability probe — gates the whole module.
# ---------------------------------------------------------------------------
REDIS_HOST = os.environ.get("FLEXKV_TEST_REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.environ.get("FLEXKV_TEST_REDIS_PORT", "6379"))
# RedisMeta hard-codes db=0; keep consistent for the RedisMeta tests.
REDIS_DB = int(os.environ.get("FLEXKV_TEST_REDIS_DB", "0"))

try:
    import redis as _redis  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover
    pytest.skip("redis-py not installed", allow_module_level=True)


def _redis_available() -> bool:
    try:
        r = _redis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
            socket_connect_timeout=1.0, decode_responses=True,
        )
        return bool(r.ping())
    except Exception:
        return False


if not _redis_available():
    pytest.skip(
        f"Redis not reachable at {REDIS_HOST}:{REDIS_PORT} db={REDIS_DB}",
        allow_module_level=True,
    )


# ---------------------------------------------------------------------------
# Dynamically load RedisMeta via importlib (bypasses c_ext dependency).
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_redis_meta_module():
    src = REPO_ROOT / "flexkv" / "cache" / "redis_meta.py"
    spec = importlib.util.spec_from_file_location("_redis_meta_it", str(src))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


_rm = _load_redis_meta_module()
RedisMeta = _rm.RedisMeta
RedisNodeInfo = _rm.RedisNodeInfo

from flexkv.common.dist_reuse import (  # noqa: E402
    FailureDetector,
    MasterCoordinator,
    RedisSessionClient,
    SharingDomainKey,
    SharingDomainNamespace,
    make_session_epoch,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def raw_client() -> Iterator["_redis.Redis"]:
    """Yield a decode_responses=True client + track+cleanup test keys.

    We never ``flushdb`` — the Redis is shared.  Tests should ``_track`` any
    keys they write so the teardown can remove exactly those.
    """
    client = _redis.Redis(
        host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB,
        decode_responses=True,
    )
    created_keys: List[str] = []
    client._test_tracked_keys = created_keys  # type: ignore[attr-defined]
    yield client
    for k in created_keys:
        try:
            client.delete(k)
        except Exception:
            pass


@pytest.fixture
def sd_key() -> SharingDomainKey:
    """Produce a unique SD key per test via random ``model_id``."""
    return SharingDomainKey(
        model_id="itm" + uuid.uuid4().hex[:12],
        pp_rank=0, pp_size=1,
        tp_node_idx=0, tp_node_count=1,
        is_nsa=False,
    )


@pytest.fixture
def namespace(sd_key: SharingDomainKey) -> SharingDomainNamespace:
    return SharingDomainNamespace(sd_key)


@pytest.fixture
def instance_id() -> str:
    return f"itinst-{uuid.uuid4().hex[:12]}"


def _track(client, *keys: str) -> None:
    """Mark keys for teardown deletion."""
    client._test_tracked_keys.extend(keys)  # type: ignore[attr-defined]


def _delete_test_sd(client, namespace: SharingDomainNamespace) -> None:
    """Delete every key under the test's SD prefix (scoped SCAN)."""
    pattern = f"{namespace.prefix}:*"
    cursor = 0
    to_delete: List[str] = []
    while True:
        cursor, keys = client.scan(cursor=cursor, match=pattern, count=200)
        to_delete.extend(keys)
        if cursor == 0:
            break
    if to_delete:
        client.delete(*to_delete)


# ===========================================================================
# Group 1 — RedisMeta / RedisNodeInfo on a live Redis
# ===========================================================================
class TestRedisMetaLive:
    """Verify the SD-aware Redis layout end-to-end on a real server."""

    def test_init_meta_writes_sd_scoped_node_key(
        self, raw_client, namespace,
    ):
        meta = RedisMeta(
            host=REDIS_HOST, port=REDIS_PORT, local_ip="10.0.99.1",
            namespace=namespace,
        )
        try:
            nid = meta.init_meta()
            assert nid is not None and nid >= 0
            node_key = namespace.node_key(nid)
            _track(raw_client, node_key)

            # Key must exist under sd:<sd>:node:<id>, with IP field.
            assert raw_client.exists(node_key) == 1
            data = raw_client.hgetall(node_key)
            assert data.get("ip") == "10.0.99.1"
        finally:
            meta.unregister_node()
            _delete_test_sd(raw_client, namespace)

    def test_two_disjoint_sds_are_isolated(self, raw_client):
        """Two different SDs must not see each other's active nodes."""
        sd_a = SharingDomainKey(
            model_id="itm-a-" + uuid.uuid4().hex[:8],
            pp_rank=0, pp_size=1, tp_node_idx=0, tp_node_count=1,
            is_nsa=False,
        )
        sd_b = SharingDomainKey(
            model_id="itm-b-" + uuid.uuid4().hex[:8],
            pp_rank=0, pp_size=1, tp_node_idx=0, tp_node_count=1,
            is_nsa=False,
        )
        ns_a = SharingDomainNamespace(sd_a)
        ns_b = SharingDomainNamespace(sd_b)
        assert ns_a.prefix != ns_b.prefix

        meta_a = RedisMeta(
            host=REDIS_HOST, port=REDIS_PORT, local_ip="10.0.99.1",
            namespace=ns_a,
        )
        meta_b = RedisMeta(
            host=REDIS_HOST, port=REDIS_PORT, local_ip="10.0.99.2",
            namespace=ns_b,
        )
        try:
            nid_a = meta_a.init_meta()
            nid_b = meta_b.init_meta()
            assert nid_a is not None and nid_b is not None
            assert nid_a != nid_b  # globally unique from INCR

            # A sees its own node.
            meta_a.nodeinfo.current_node_id_set.clear()
            meta_a.nodeinfo.scan_active_nodes()
            assert nid_a in meta_a.nodeinfo.current_node_id_set
            assert nid_b not in meta_a.nodeinfo.current_node_id_set

            # B sees its own node, not A's.
            meta_b.nodeinfo.current_node_id_set.clear()
            meta_b.nodeinfo.scan_active_nodes()
            assert nid_b in meta_b.nodeinfo.current_node_id_set
            assert nid_a not in meta_b.nodeinfo.current_node_id_set
        finally:
            meta_a.unregister_node()
            meta_b.unregister_node()
            _delete_test_sd(raw_client, ns_a)
            _delete_test_sd(raw_client, ns_b)

    def test_regist_node_meta_and_buffer(self, raw_client, namespace):
        meta = RedisMeta(
            host=REDIS_HOST, port=REDIS_PORT, local_ip="10.0.99.5",
            namespace=namespace,
        )
        try:
            nid = meta.init_meta()
            assert nid is not None

            meta.regist_node_meta(
                node_id=nid, addr="10.0.99.5",
                zmq_addr="tcp://10.0.99.5:18000",
                cpu_buffer_ptr=0xAABBCC, ssd_buffer_ptr=0,
            )
            meta_key = namespace.meta_key(nid)
            _track(raw_client, meta_key)
            assert raw_client.exists(meta_key) == 1

            data = raw_client.hgetall(meta_key)
            assert data.get("addr") == "10.0.99.5"
            assert data.get("zmq_addr") == "tcp://10.0.99.5:18000"
            assert int(data.get("cpu_buffer_ptr", "0")) == 0xAABBCC

            loaded = meta.get_node_meta(nid)
            assert loaded.get("zmq_addr") == "tcp://10.0.99.5:18000"

            # regist_buffer writes under sd:<sd>:buffer:<nid>:<ptr>
            count = meta.regist_buffer([
                (0xDEADBEEF, 1024),
                {"buffer_ptr": 0xC0FFEE, "buffer_size": 2048},
            ])
            assert count == 2
            buf1 = f"{namespace.prefix}:buffer:{nid}:{0xDEADBEEF}"
            buf2 = f"{namespace.prefix}:buffer:{nid}:{0xC0FFEE}"
            _track(raw_client, buf1, buf2)
            assert raw_client.exists(buf1) == 1
            assert raw_client.exists(buf2) == 1
            d1 = raw_client.hgetall(buf1)
            d2 = raw_client.hgetall(buf2)
            assert int(d1.get("buffer_size", "0")) == 1024
            assert int(d2.get("buffer_size", "0")) == 2048
        finally:
            meta.unregister_node()
            _delete_test_sd(raw_client, namespace)

    def test_instance_sd_nodes_roundtrip(
        self, raw_client, namespace, instance_id,
    ):
        """register_instance_sd_nodes → load_instance_sd_nodes is symmetric."""
        meta = RedisMeta(
            host=REDIS_HOST, port=REDIS_PORT, local_ip="10.0.99.10",
            namespace=namespace,
        )
        try:
            sd_map = {
                "mmm:pp0/2:tpn0/1:nsa0": 700,
                "mmm:pp1/2:tpn0/1:nsa0": 701,
            }
            meta.register_instance_sd_nodes(instance_id, sd_map)

            key = SharingDomainNamespace.instance_sd_nodes_key(instance_id)
            _track(raw_client, key)
            assert raw_client.exists(key) == 1

            loaded = meta.load_instance_sd_nodes(instance_id)
            assert loaded == sd_map

            # Missing instance returns empty dict.
            missing = meta.load_instance_sd_nodes(f"nope-{uuid.uuid4().hex[:8]}")
            assert missing == {}
        finally:
            _delete_test_sd(raw_client, namespace)

    def test_prefix_is_unique_no_crosstalk_with_other_tenants(
        self, raw_client, namespace,
    ):
        """Sanity: our random ``model_id`` ensures no pre-existing collision
        with the ~1000 keys from other users of this shared Redis."""
        pattern = f"{namespace.prefix}:*"
        cursor = 0
        existing = 0
        while True:
            cursor, keys = raw_client.scan(cursor=cursor, match=pattern, count=200)
            existing += len(keys)
            if cursor == 0:
                break
        assert existing == 0, (
            f"Unexpected pre-existing keys under {pattern}: {existing}"
        )


# ===========================================================================
# Group 2 — RedisSessionClient + FailureDetector with real TTL
# ===========================================================================
class TestSessionTTLLive:
    """FakeRedis cannot truly expire keys — this is where we verify the
    co-destined-failure model's actual wall-clock behaviour.
    """

    def test_session_register_renew_unregister(self, raw_client, instance_id):
        sess = RedisSessionClient(
            redis_client=raw_client,
            instance_id=instance_id,
            epoch=make_session_epoch(),
            ttl_seconds=5,
        )
        sess.register()
        _track(raw_client, sess.key)

        # Key must exist with TTL in (0, 5].
        assert raw_client.exists(sess.key) == 1
        ttl = raw_client.ttl(sess.key)
        assert 0 < ttl <= 5

        # Payload JSON round-trips.
        payload = json.loads(raw_client.get(sess.key))
        assert payload["instance_id"] == instance_id
        assert payload["epoch"] == sess.epoch

        # renew() bumps TTL back up.
        time.sleep(1.5)
        old_ttl = raw_client.ttl(sess.key)
        sess.renew()
        new_ttl = raw_client.ttl(sess.key)
        assert new_ttl > old_ttl

        sess.unregister()
        assert raw_client.exists(sess.key) == 0

    def test_session_expires_after_ttl_without_renew(
        self, raw_client, instance_id,
    ):
        """Core property: if renew() stops, the key really vanishes."""
        sess = RedisSessionClient(
            redis_client=raw_client,
            instance_id=instance_id,
            epoch=make_session_epoch(),
            ttl_seconds=1,
        )
        sess.register()
        _track(raw_client, sess.key)
        assert raw_client.exists(sess.key) == 1

        time.sleep(1.6)
        assert raw_client.exists(sess.key) == 0

    def test_failure_detector_fires_peer_lost_on_ttl_expiry(self, raw_client):
        """Deterministic via poll_once() — no thread race."""
        self_id = f"detector-{uuid.uuid4().hex[:6]}"
        peer_id = f"peer-{uuid.uuid4().hex[:6]}"

        lost_events: List[str] = []
        seen_events: List[Tuple[str, str]] = []

        peer_sess = RedisSessionClient(
            redis_client=raw_client,
            instance_id=peer_id,
            epoch=make_session_epoch(),
            ttl_seconds=1,
        )
        peer_sess.register()
        _track(raw_client, peer_sess.key)

        detector = FailureDetector(
            redis_client=raw_client,
            self_instance_id=self_id,
            poll_interval_seconds=0.1,
            on_peer_lost=lambda pid: lost_events.append(pid),
            on_peer_seen=lambda pid, s: seen_events.append((pid, s.epoch)),
        )
        # 1st poll: peer is alive → on_peer_seen fires.
        detector.poll_once()
        assert any(pid == peer_id for pid, _ in seen_events), (
            f"Expected on_peer_seen for {peer_id}; got {seen_events!r}"
        )

        # Let peer expire.
        time.sleep(1.6)
        assert raw_client.exists(peer_sess.key) == 0

        # 2nd poll: peer vanished → on_peer_lost fires.
        detector.poll_once()
        assert peer_id in lost_events, (
            f"Expected {peer_id} in lost_events; got {lost_events!r}"
        )

    def test_detector_ignores_self_and_handles_epoch_change(self, raw_client):
        self_id = f"self-{uuid.uuid4().hex[:6]}"
        peer_id = f"peer-{uuid.uuid4().hex[:6]}"

        # Both self and peer are alive initially.
        self_sess = RedisSessionClient(
            redis_client=raw_client, instance_id=self_id,
            epoch=make_session_epoch(), ttl_seconds=5,
        )
        self_sess.register()
        _track(raw_client, self_sess.key)

        peer_epoch_1 = make_session_epoch()
        peer_sess = RedisSessionClient(
            redis_client=raw_client, instance_id=peer_id,
            epoch=peer_epoch_1, ttl_seconds=5,
        )
        peer_sess.register()
        _track(raw_client, peer_sess.key)

        seen: List[Tuple[str, str]] = []
        lost: List[str] = []
        detector = FailureDetector(
            redis_client=raw_client,
            self_instance_id=self_id,
            poll_interval_seconds=0.1,
            on_peer_lost=lambda pid: lost.append(pid),
            on_peer_seen=lambda pid, s: seen.append((pid, s.epoch)),
        )
        detector.poll_once()
        # Self is excluded by contract.
        assert not any(pid == self_id for pid, _ in seen)
        assert any(pid == peer_id for pid, _ in seen)

        # Restart peer with new epoch (simulating crash + restart).
        baseline = len(seen)
        peer_sess_2 = RedisSessionClient(
            redis_client=raw_client, instance_id=peer_id,
            epoch=make_session_epoch(), ttl_seconds=5,
        )
        assert peer_sess_2.epoch != peer_epoch_1
        peer_sess_2.register()

        detector.poll_once()
        new_events = seen[baseline:]
        assert any(
            pid == peer_id and epoch == peer_sess_2.epoch
            for pid, epoch in new_events
        ), f"Expected epoch-change on_peer_seen; got {new_events!r}"


# ===========================================================================
# Group 3 — MasterCoordinator end-to-end with live session
# ===========================================================================
class TestMasterCoordinatorLive:
    """Exercise the Master-side composition over real Redis sessions."""

    def test_coordinator_lifecycle_with_live_session(
        self, raw_client, instance_id,
    ):
        self_sd = SharingDomainKey(
            model_id="mc-" + uuid.uuid4().hex[:6],
            pp_rank=0, pp_size=1, tp_node_idx=0, tp_node_count=1,
            is_nsa=False,
        )
        epoch = make_session_epoch()

        sess = RedisSessionClient(
            redis_client=raw_client,
            instance_id=instance_id, epoch=epoch, ttl_seconds=5,
        )
        sess.register()
        _track(raw_client, sess.key)

        coord = MasterCoordinator(
            self_sd=self_sd,
            instance_id=instance_id,
            session_epoch=epoch,
        )
        try:
            coord.expect_remotes(0)
            # The session key is alive on the server.
            assert raw_client.ttl(sess.key) > 0
            # MasterCoordinator composed without exception with a live Redis.
        finally:
            coord.shutdown()

    def test_register_and_load_instance_mapping_e2e(
        self, raw_client, namespace, instance_id,
    ):
        """Master registers sd_key → node_id mapping in Redis; read it back
        via two distinct paths (Python helper + raw HGETALL)."""
        meta = RedisMeta(
            host=REDIS_HOST, port=REDIS_PORT, local_ip="10.0.99.20",
            namespace=namespace,
        )
        try:
            mapping = {
                f"sd-{i}-{uuid.uuid4().hex[:4]}": 800 + i for i in range(4)
            }
            meta.register_instance_sd_nodes(instance_id, mapping)
            key = SharingDomainNamespace.instance_sd_nodes_key(instance_id)
            _track(raw_client, key)

            loaded = meta.load_instance_sd_nodes(instance_id)
            assert loaded == mapping

            raw_hash = raw_client.hgetall(key)
            assert {k: int(v) for k, v in raw_hash.items()} == mapping
        finally:
            _delete_test_sd(raw_client, namespace)
