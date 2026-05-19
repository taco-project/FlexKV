"""Tests for ``CacheConfig.flexkv_redis_db`` wiring.

Phase D follow-up: verify that the single ``flexkv_redis_db`` config option
flows through every FlexKV Redis client:

  * ``CacheConfig`` defaults + ``FlexKVUserConfig`` override merge.
  * ``RedisMeta`` / ``RedisNodeInfo`` constructor accepts ``db=`` and the
    raw ``redis-py`` client is bound to that db.
  * ``RedisMetaChannel`` (Python wrapper) forwards ``db`` to the C++ ctor.
  * ``make_redis_client_from_cache_config`` reads the right attr.

These tests do **not** require a live Redis — they only check the
Python-visible plumbing.  For real-server behaviour (``SELECT <db>`` on
the wire + key isolation across dbs) see ``test_redis_db_integration.py``.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_module_from_source(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# Load redis_meta.py directly (avoids flexkv.cache.__init__'s c_ext import).
_rm = _load_module_from_source(
    REPO_ROOT / "flexkv" / "cache" / "redis_meta.py",
    "_rm_db_wiring_test",
)
RedisMeta = _rm.RedisMeta
RedisNodeInfo = _rm.RedisNodeInfo


# Load config directly so we don't depend on whether config.so shadows .py.
_cfg = _load_module_from_source(
    REPO_ROOT / "flexkv" / "common" / "config.py",
    "_cfg_db_wiring_test",
)
CacheConfig = _cfg.CacheConfig


# ---------------------------------------------------------------------------
# CacheConfig default + override
# ---------------------------------------------------------------------------
class TestCacheConfigFlexkvRedisDb:
    def test_default_is_zero(self):
        cc = CacheConfig()
        assert cc.flexkv_redis_db == 0

    def test_can_set_db_explicitly(self):
        cc = CacheConfig(flexkv_redis_db=7)
        assert cc.flexkv_redis_db == 7

    def test_enable_sharing_domain_does_not_affect_db(self):
        cc = CacheConfig(enable_p2p_cpu=True, flexkv_redis_db=3)
        # Auto-enable kicks in, but the db override stays.
        assert cc.enable_sharing_domain is True
        assert cc.flexkv_redis_db == 3


# ---------------------------------------------------------------------------
# RedisNodeInfo — db arg reaches the redis-py Redis() constructor
# ---------------------------------------------------------------------------
class TestRedisNodeInfoDbForwarding:
    def test_default_db_is_zero(self):
        ni = RedisNodeInfo(
            host="127.0.0.1", port=6379, local_ip="127.0.0.1",
        )
        assert ni.db == 0

    def test_db_arg_is_stored(self):
        ni = RedisNodeInfo(
            host="127.0.0.1", port=6379, local_ip="127.0.0.1",
            db=15,
        )
        assert ni.db == 15

    def test_get_client_passes_db_kwarg(self):
        """_get_client must call redis.Redis(..., db=<configured>)."""
        ni = RedisNodeInfo(
            host="127.0.0.1", port=6379, local_ip="127.0.0.1",
            db=7,
        )
        with patch.object(_rm, "_redis") as redis_mod_mock:
            redis_mod_mock.Redis = MagicMock(return_value="fake_client")
            client = ni._get_client()
            redis_mod_mock.Redis.assert_called_once()
            _, kwargs = redis_mod_mock.Redis.call_args
            assert kwargs.get("db") == 7
            assert kwargs.get("host") == "127.0.0.1"
            assert kwargs.get("port") == 6379
            assert client == "fake_client"


# ---------------------------------------------------------------------------
# RedisMeta — db propagates to nodeinfo AND to its own `_client()`
# ---------------------------------------------------------------------------
class TestRedisMetaDbForwarding:
    def test_default_db_is_zero(self):
        meta = RedisMeta(
            host="127.0.0.1", port=6379, local_ip="127.0.0.1",
        )
        assert meta.db == 0
        assert meta.nodeinfo.db == 0

    def test_db_arg_propagates_to_nodeinfo(self):
        meta = RedisMeta(
            host="127.0.0.1", port=6379, local_ip="127.0.0.1",
            db=11,
        )
        assert meta.db == 11
        # Must propagate to the inner RedisNodeInfo so node-heartbeat
        # and block-metadata land on the same logical db.
        assert meta.nodeinfo.db == 11

    def test_client_uses_configured_db(self):
        """_client() must invoke redis.Redis(..., db=<configured>)."""
        meta = RedisMeta(
            host="127.0.0.1", port=6379, local_ip="127.0.0.1",
            db=9,
        )
        with patch.object(_rm, "_redis") as redis_mod_mock:
            redis_mod_mock.Redis = MagicMock(return_value="fake_client")
            meta._client()
            _, kwargs = redis_mod_mock.Redis.call_args
            assert kwargs.get("db") == 9


# ---------------------------------------------------------------------------
# RedisMetaChannel wrapper — db flows to the C++ constructor
# ---------------------------------------------------------------------------
class TestRedisMetaChannelDbForwarding:
    def _make_wrapper(self, **kwargs):
        """Build the Python wrapper with the C++ class stubbed out."""
        fake_c = MagicMock()
        # By default the stub accepts any arg shape; individual tests
        # override this to simulate legacy builds.
        with patch.object(_rm, "_CRedisMetaChannel", MagicMock(return_value=fake_c)):
            return _rm.RedisMetaChannel(**kwargs), fake_c

    def test_db_is_forwarded_to_cpp_ctor(self):
        captured_args = {}

        def fake_ctor(*args):
            captured_args["args"] = args
            return MagicMock()

        with patch.object(_rm, "_CRedisMetaChannel", side_effect=fake_ctor):
            _rm.RedisMetaChannel(
                host="h", port=6379, node_id=1, local_ip="127.0.0.1",
                blocks_key="sd:xx:CPUB", password="", db=5,
            )
        # Last positional arg must be db=5.
        assert captured_args["args"][-1] == 5
        assert captured_args["args"][0] == "h"
        assert captured_args["args"][1] == 6379
        assert captured_args["args"][4] == "sd:xx:CPUB"

    def test_legacy_cpp_build_accepts_default_db_zero(self):
        """Legacy C++ build (6-arg ctor) must still work for db=0."""
        call_counter = {"n": 0}

        def legacy_ctor(*args):
            call_counter["n"] += 1
            # First call raises TypeError (simulating 6-arg legacy signature);
            # second call (with 6 args) succeeds.
            if call_counter["n"] == 1:
                raise TypeError("too many arguments")
            return MagicMock()

        with patch.object(_rm, "_CRedisMetaChannel", side_effect=legacy_ctor):
            w = _rm.RedisMetaChannel(
                host="h", port=6379, node_id=1, local_ip="127.0.0.1",
                blocks_key="sd:xx:CPUB", password="", db=0,
            )
        assert call_counter["n"] == 2  # first raised, second succeeded
        assert w._db == 0

    def test_legacy_cpp_build_rejects_nonzero_db(self):
        """db != 0 on a legacy build must raise ImportError loudly."""
        def legacy_ctor(*args):
            raise TypeError("too many arguments")

        with patch.object(_rm, "_CRedisMetaChannel", side_effect=legacy_ctor):
            with pytest.raises(ImportError, match="rebuild FlexKV"):
                _rm.RedisMetaChannel(
                    host="h", port=6379, node_id=1, local_ip="127.0.0.1",
                    blocks_key="sd:xx:CPUB", password="", db=5,
                )


# ---------------------------------------------------------------------------
# make_redis_client_from_cache_config — single source of truth
# ---------------------------------------------------------------------------
class TestMakeRedisClientHelper:
    def test_helper_reads_flexkv_redis_db(self):
        from flexkv.common.dist_reuse import make_redis_client_from_cache_config

        class _FakeCfg:
            redis_host = "1.2.3.4"
            redis_port = 6400
            redis_password = "secret"
            flexkv_redis_db = 13

        import redis as _r
        with patch.object(_r, "Redis") as mock_redis:
            mock_redis.return_value = "ok"
            client = make_redis_client_from_cache_config(_FakeCfg())
            assert client == "ok"
            _, kwargs = mock_redis.call_args
            assert kwargs["host"] == "1.2.3.4"
            assert kwargs["port"] == 6400
            assert kwargs["db"] == 13
            assert kwargs["password"] == "secret"
            assert kwargs["decode_responses"] is True

    def test_helper_falls_back_when_attr_missing(self):
        """Duck-typed config without flexkv_redis_db → db=0."""
        from flexkv.common.dist_reuse import make_redis_client_from_cache_config

        class _MinimalCfg:
            redis_host = "h"
            redis_port = 6379
            redis_password = None
            # no flexkv_redis_db attr

        import redis as _r
        with patch.object(_r, "Redis") as mock_redis:
            mock_redis.return_value = "ok"
            make_redis_client_from_cache_config(_MinimalCfg())
            _, kwargs = mock_redis.call_args
            assert kwargs["db"] == 0
            assert "password" not in kwargs  # None → omitted
