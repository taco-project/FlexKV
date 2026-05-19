"""Unit tests for ``flexkv.cache.redis_meta`` after Phase 0 task 0-C migration.

These tests validate that every Redis key emitted by ``RedisMeta`` /
``RedisNodeInfo`` is **SD-scoped** via the new
``SharingDomainNamespace``.  They use :mod:`tests._dist_reuse_fakes` to
stand in for a real Redis server so no network (and no Redis install) is
required.

The test module does **not** import ``flexkv.cache.redis_meta`` via the
normal package path, because ``flexkv.cache.__init__`` unconditionally
loads the CUDA-linked C++ extension (``flexkv.c_ext``).  On CPU-only CI
workers that fails before we can get to our code.  Instead we load
``redis_meta.py`` directly via :mod:`importlib.util` after patching the
``redis`` module to hand out our :class:`FakeRedis` instances.
"""
from __future__ import annotations

import importlib.util
import os
import sys
import types
import unittest.mock as mock
from pathlib import Path
from typing import Any

import pytest

from flexkv.common.dist_reuse import SharingDomainKey, SharingDomainNamespace

# Shared fake Redis server (see tests/_dist_reuse_fakes.py).
sys.path.insert(0, str(Path(__file__).parent))
from _dist_reuse_fakes import FakeRedis, ManualClock  # noqa: E402


# ---------------------------------------------------------------------------
# Module loader: import ``flexkv.cache.redis_meta`` without importing the
# ``flexkv.cache`` package init (which pulls in CUDA).  Returns a fresh copy
# on each call so module-level state can't leak between test functions.
# ---------------------------------------------------------------------------
def _load_redis_meta(fake_client_factory) -> Any:
    """Load ``flexkv.cache.redis_meta`` directly from source.

    ``fake_client_factory`` is a callable ``() -> FakeRedis`` used to
    replace ``redis.Redis`` for the duration of this module's lifecycle.
    Returns the freshly loaded module object.
    """
    pkg_root = Path(__file__).resolve().parent.parent
    src = pkg_root / "flexkv" / "cache" / "redis_meta.py"
    assert src.exists(), f"missing source file {src}"

    # Stub out redis-py so ``import redis as _redis`` inside redis_meta.py
    # hands us a module whose ``Redis(...)`` constructor returns our fake.
    fake_redis_mod = types.ModuleType("redis")

    def _ctor(*args, **kwargs):  # noqa: ARG001 — mimic redis.Redis signature
        return fake_client_factory()

    fake_redis_mod.Redis = _ctor  # type: ignore[attr-defined]
    # redis_meta.py only uses ``redis.Redis``; no other attrs referenced.

    # Patch sys.modules for the import that will happen inside spec.loader.exec_module.
    original_redis = sys.modules.get("redis")
    original_cache_pkg = sys.modules.get("flexkv.cache.redis_meta")
    sys.modules["redis"] = fake_redis_mod

    try:
        spec = importlib.util.spec_from_file_location(
            "_rm_under_test", str(src),
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        # ``@dataclass`` looks the module up via ``sys.modules[cls.__module__]``.
        # Register the module BEFORE executing it so decorators don't see None.
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        if original_redis is None:
            sys.modules.pop("redis", None)
        else:
            sys.modules["redis"] = original_redis
        if original_cache_pkg is not None:
            sys.modules["flexkv.cache.redis_meta"] = original_cache_pkg


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def sd():
    return SharingDomainKey(
        model_id="abc",
        pp_rank=0, pp_size=2,
        tp_node_idx=0, tp_node_count=1,
        is_nsa=False,
    )


@pytest.fixture
def ns(sd):
    return SharingDomainNamespace(sd)


@pytest.fixture
def shared_fake():
    """One FakeRedis shared across all ``redis.Redis(...)`` calls in a test,
    so the heartbeat/listener threads and the main thread see consistent
    state."""
    return FakeRedis()


@pytest.fixture
def rm_module(shared_fake):
    return _load_redis_meta(lambda: shared_fake)


# ---------------------------------------------------------------------------
# _channel_blocks_key
# ---------------------------------------------------------------------------
def test_channel_blocks_key_composition(rm_module, ns):
    composed = rm_module._channel_blocks_key(ns, "CPUB")
    assert composed == f"{ns.prefix}:CPUB"

    composed_no_device = rm_module._channel_blocks_key(ns, "")
    assert composed_no_device == ns.prefix


def test_channel_blocks_key_with_sd_only_ns(rm_module):
    default_ns = SharingDomainNamespace(SharingDomainKey.default())
    assert rm_module._channel_blocks_key(default_ns, "SSDB").startswith("sd:__default__:")


# ---------------------------------------------------------------------------
# _resolve_namespace
# ---------------------------------------------------------------------------
def test_resolve_namespace_default(rm_module):
    ns = rm_module._resolve_namespace(None)
    assert isinstance(ns, SharingDomainNamespace)
    assert ns.sd_key.model_id == "__default__"


def test_resolve_namespace_accepts_sd_key(rm_module):
    sd = SharingDomainKey.default()
    ns = rm_module._resolve_namespace(sd)
    assert ns.sd_key == sd


def test_resolve_namespace_rejects_garbage(rm_module):
    with pytest.raises(TypeError):
        rm_module._resolve_namespace("not-a-namespace")


# ---------------------------------------------------------------------------
# RedisNodeInfo registration
# ---------------------------------------------------------------------------
class TestNodeRegistration:
    def _make(self, rm_module, ns, fake):
        # Don't let signal handlers interfere with pytest.
        with mock.patch("signal.signal"):
            info = rm_module.RedisNodeInfo(
                host="fake", port=0,
                local_ip="10.0.0.1",
                password="",
                node_ttl_seconds=60,
                namespace=ns,
            )
        return info

    def test_register_writes_sd_scoped_key(self, rm_module, shared_fake, ns):
        info = self._make(rm_module, ns, shared_fake)

        # Bypass connect() / heartbeat thread: just wire up the client.
        info._client = shared_fake
        node_id = info.register_node()
        assert node_id is not None
        assert node_id >= 1  # global:node_id incremented
        # Verify the key is SD-scoped, NOT bare "node:<id>"
        expected_key = ns.node_key(node_id)
        assert shared_fake.hget(expected_key, "node_id") == str(node_id)
        assert shared_fake.hget(expected_key, "sd_key") == ns.serialized_sd
        # And make sure the legacy bare key is *not* present
        assert shared_fake.exists(f"node:{node_id}") == 0

    def test_namespace_property(self, rm_module, shared_fake, ns):
        info = self._make(rm_module, ns, shared_fake)
        assert info.namespace == ns
        assert info.sd_key_str == ns.serialized_sd

    def test_two_infos_isolate_per_sd(self, rm_module, shared_fake):
        ns_a = SharingDomainNamespace(SharingDomainKey(
            model_id="m", pp_rank=0, pp_size=1, tp_node_idx=0,
            tp_node_count=1, is_nsa=False,
        ))
        ns_b = SharingDomainNamespace(SharingDomainKey(
            model_id="m", pp_rank=1, pp_size=2, tp_node_idx=0,
            tp_node_count=1, is_nsa=False,
        ))
        info_a = self._make(rm_module, ns_a, shared_fake)
        info_b = self._make(rm_module, ns_b, shared_fake)
        info_a._client = shared_fake
        info_b._client = shared_fake
        nid_a = info_a.register_node()
        nid_b = info_b.register_node()
        assert nid_a != nid_b

        # Scanning via SD-A pattern finds only SD-A's key.
        info_a.scan_active_nodes()
        info_b.scan_active_nodes()
        assert info_a.get_active_node_ids() == [nid_a]
        assert info_b.get_active_node_ids() == [nid_b]


# ---------------------------------------------------------------------------
# RedisMeta.get_redis_meta_channel — blocks_key composition
# ---------------------------------------------------------------------------
class TestRedisMetaChannelFactory:
    def test_device_prefix_composes_into_blocks_key(
        self, rm_module, shared_fake, ns, monkeypatch
    ):
        # Intercept RedisMetaChannel so we don't need the C++ ext to be built.
        captured = {}

        class _StubChannel:
            def __init__(self, host, port, node_id, local_ip, blocks_key, password, db=0):
                captured["blocks_key"] = blocks_key
                captured["host"] = host
                captured["node_id"] = node_id

            def connect(self):
                return True

        monkeypatch.setattr(rm_module, "RedisMetaChannel", _StubChannel)

        with mock.patch("signal.signal"):
            meta = rm_module.RedisMeta(
                host="h", port=6379, password="pw",
                local_ip="10.0.0.1", decode_responses=True,
                node_ttl_seconds=60, namespace=ns,
            )
        meta._node_id = 42

        _ = meta.get_redis_meta_channel(device_prefix="CPUB")
        assert captured["blocks_key"] == f"{ns.prefix}:CPUB"

        _ = meta.get_redis_meta_channel(device_prefix="")
        assert captured["blocks_key"] == ns.prefix

    def test_positional_device_prefix(self, rm_module, shared_fake, ns, monkeypatch):
        captured = {}

        class _StubChannel:
            def __init__(self, host, port, node_id, local_ip, blocks_key, password, db=0):
                captured["blocks_key"] = blocks_key

            def connect(self):
                return True

        monkeypatch.setattr(rm_module, "RedisMetaChannel", _StubChannel)

        with mock.patch("signal.signal"):
            meta = rm_module.RedisMeta(
                host="h", port=6379, password=None,
                local_ip="10.0.0.1", namespace=ns,
            )
        meta._node_id = 7
        # Legacy call style from hie_cache_engine.py uses positional arg.
        _ = meta.get_redis_meta_channel("SSDB")
        assert captured["blocks_key"] == f"{ns.prefix}:SSDB"


# ---------------------------------------------------------------------------
# RedisMeta.regist_buffer / regist_node_meta use SD-scoped keys
# ---------------------------------------------------------------------------
class TestRedisMetaBufferAndNodeMeta:
    def _make(self, rm_module, ns):
        with mock.patch("signal.signal"):
            meta = rm_module.RedisMeta(
                host="h", port=0, password=None,
                local_ip="10.0.0.1", namespace=ns,
            )
        meta._node_id = 5
        return meta

    def test_regist_buffer_sd_scoped(self, rm_module, shared_fake, ns):
        meta = self._make(rm_module, ns)
        meta.regist_buffer([{"buffer_ptr": 0xABC, "buffer_size": 1024}])
        # Expected key: sd:<sd>:buffer:5:2748
        expected_key = ns.buffer_key(5, 0xABC)
        assert shared_fake.hget(expected_key, "buffer_size") == "1024"
        # No legacy key
        assert shared_fake.exists(f"buffer:5:{0xABC}") == 0

    def test_unregist_buffer(self, rm_module, shared_fake, ns):
        meta = self._make(rm_module, ns)
        meta.regist_buffer([(0xDEF, 256)])
        ok = meta.unregist_buffer(0xDEF)
        assert ok is True
        assert shared_fake.exists(ns.buffer_key(5, 0xDEF)) == 0
        # Double unregister returns False
        assert meta.unregist_buffer(0xDEF) is False

    def test_regist_node_meta_sd_scoped(self, rm_module, shared_fake, ns):
        meta = self._make(rm_module, ns)
        meta.regist_node_meta(
            node_id=5,
            addr="10.0.0.1:5555",
            zmq_addr="tcp://10.0.0.1:6666",
            cpu_buffer_ptr=0x1000,
            ssd_buffer_ptr=0x2000,
        )
        expected = ns.meta_key(5)
        assert shared_fake.hget(expected, "addr") == "10.0.0.1:5555"
        assert shared_fake.hget(expected, "zmq_addr") == "tcp://10.0.0.1:6666"
        assert shared_fake.hget(expected, "cpu_buffer_ptr") == "4096"
        assert shared_fake.exists(f"meta:5") == 0  # no legacy key

    def test_get_node_meta_round_trip(self, rm_module, shared_fake, ns):
        meta = self._make(rm_module, ns)
        meta.regist_node_meta(5, "addr", "zmq", 100, 200)
        info = meta.get_node_meta(5)
        assert info == {
            "node_id": 5, "addr": "addr", "zmq_addr": "zmq",
            "cpu_buffer_ptr": 100, "ssd_buffer_ptr": 200,
        }

    def test_get_node_meta_missing(self, rm_module, shared_fake, ns):
        meta = self._make(rm_module, ns)
        assert meta.get_node_meta(9999) == {}

    def test_unregist_node_meta(self, rm_module, shared_fake, ns):
        meta = self._make(rm_module, ns)
        meta.regist_node_meta(5, "addr", "zmq", 100, 200)
        assert meta.unregist_node_meta(5) is True
        assert meta.unregist_node_meta(5) is False


# ---------------------------------------------------------------------------
# Instance-level cross-SD keys (design doc §4.7.1.6)
# ---------------------------------------------------------------------------
class TestInstanceSdNodes:
    def _make(self, rm_module, ns):
        with mock.patch("signal.signal"):
            return rm_module.RedisMeta(
                host="h", port=0, password=None,
                local_ip="10.0.0.1", namespace=ns,
            )

    def test_register_load_round_trip(self, rm_module, shared_fake, ns):
        meta = self._make(rm_module, ns)
        meta.register_instance_sd_nodes(
            instance_id="inst-001",
            sd_to_nid={
                "abc:pp0/2:tpn0/2:nsa0": 1,
                "abc:pp1/2:tpn0/2:nsa0": 2,
                "abc:pp0/2:tpn1/2:nsa0": 3,
            },
        )
        got = meta.load_instance_sd_nodes("inst-001")
        assert got == {
            "abc:pp0/2:tpn0/2:nsa0": 1,
            "abc:pp1/2:tpn0/2:nsa0": 2,
            "abc:pp0/2:tpn1/2:nsa0": 3,
        }

    def test_missing_instance_returns_empty(self, rm_module, shared_fake, ns):
        meta = self._make(rm_module, ns)
        assert meta.load_instance_sd_nodes("never-seen") == {}

    def test_unregister(self, rm_module, shared_fake, ns):
        meta = self._make(rm_module, ns)
        meta.register_instance_sd_nodes("inst-002", {"sd0": 1})
        assert meta.unregister_instance_sd_nodes("inst-002") is True
        assert meta.load_instance_sd_nodes("inst-002") == {}

    def test_register_empty_mapping_is_noop(self, rm_module, shared_fake, ns):
        meta = self._make(rm_module, ns)
        meta.register_instance_sd_nodes("inst-003", {})
        # Nothing should have been written.
        key = SharingDomainNamespace.instance_sd_nodes_key("inst-003")
        assert shared_fake.exists(key) == 0


# ---------------------------------------------------------------------------
# _cleanup_node_data
# ---------------------------------------------------------------------------
class TestCleanupNodeData:
    def _make_info(self, rm_module, ns, fake):
        with mock.patch("signal.signal"):
            info = rm_module.RedisNodeInfo(
                host="h", port=0, local_ip="ip",
                namespace=ns,
            )
        info._client = fake
        return info

    def test_cleanup_removes_all_sd_scoped_keys(self, rm_module, shared_fake, ns):
        info = self._make_info(rm_module, ns, shared_fake)
        # Seed a variety of keys belonging to node 5.
        shared_fake.hset(ns.meta_key(5), mapping={"foo": "bar"})
        shared_fake.hset(ns.buffer_key(5, 0x100), mapping={"sz": "1"})
        shared_fake.hset(f"{ns.prefix}:CPUB:block:5:abc", mapping={"x": "1"})
        shared_fake.hset(f"{ns.prefix}:SSDB:block:5:def", mapping={"x": "1"})
        # Keys for a different node — must survive.
        shared_fake.hset(ns.meta_key(6), mapping={"foo": "keep"})
        shared_fake.hset(f"{ns.prefix}:CPUB:block:6:abc", mapping={"x": "keep"})

        info._cleanup_node_data(5)

        # Node 5 is gone
        assert shared_fake.exists(ns.meta_key(5)) == 0
        assert shared_fake.exists(ns.buffer_key(5, 0x100)) == 0
        assert shared_fake.exists(f"{ns.prefix}:CPUB:block:5:abc") == 0
        assert shared_fake.exists(f"{ns.prefix}:SSDB:block:5:def") == 0
        # Node 6 is untouched
        assert shared_fake.exists(ns.meta_key(6)) == 1
        assert shared_fake.exists(f"{ns.prefix}:CPUB:block:6:abc") == 1


# ---------------------------------------------------------------------------
# PCFS file-nodeid mapping
# ---------------------------------------------------------------------------
class TestPcfsMapping:
    def _make(self, rm_module, ns):
        with mock.patch("signal.signal"):
            meta = rm_module.RedisMeta(
                host="h", port=0, local_ip="ip", namespace=ns,
            )
        meta._node_id = 1
        return meta

    def test_add_and_load(self, rm_module, shared_fake, ns):
        meta = self._make(rm_module, ns)
        meta.add_node_ids([10, 20, 30])
        # Simulate another node on the same SD pushing its own list.
        shared_fake.rpush(f"{ns.prefix}:pcfs:2", "100", "200")

        loaded = meta.load_pcfs_file_nodeids()
        assert loaded == {1: [10, 20, 30], 2: [100, 200]}
