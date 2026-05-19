"""Unit tests for :class:`RedisMetaChannel` SD-prefix derivation logic.

The C++ ``RedisMetaChannel::list_node_keys`` strips the device suffix from
``blocks_key`` to produce the per-SD node scan pattern.  The Python wrapper
``_derive_sd_prefix`` mirrors that logic for the fallback path that runs
when the C++ extension is older than Batch B (or built without
FLEXKV_ENABLE_P2P).  The two implementations MUST stay in sync — this test
enforces that by exercising every SD + device permutation the design-doc
allows.

We load ``redis_meta.py`` directly via importlib (see
``tests/test_redis_meta_namespace.py`` for the rationale) and build a
``RedisMetaChannel`` **without** connecting to Redis — the method under
test is purely string manipulation on ``self._blocks_key``.
"""
from __future__ import annotations

import importlib.util
import sys
import types
import unittest.mock as mock
from pathlib import Path
from typing import Any

import pytest

from flexkv.common.dist_reuse import SharingDomainKey, SharingDomainNamespace

sys.path.insert(0, str(Path(__file__).parent))
from _dist_reuse_fakes import FakeRedis  # noqa: E402


def _load_redis_meta():
    pkg_root = Path(__file__).resolve().parent.parent
    src = pkg_root / "flexkv" / "cache" / "redis_meta.py"
    fake_redis_mod = types.ModuleType("redis")
    fake_redis_mod.Redis = lambda *a, **kw: FakeRedis()  # type: ignore[attr-defined]
    original_redis = sys.modules.get("redis")
    sys.modules["redis"] = fake_redis_mod
    try:
        spec = importlib.util.spec_from_file_location("_rm_cc_ut", str(src))
        assert spec is not None and spec.loader is not None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod  # so @dataclass can resolve the module
        spec.loader.exec_module(mod)
        return mod
    finally:
        if original_redis is None:
            sys.modules.pop("redis", None)
        else:
            sys.modules["redis"] = original_redis


@pytest.fixture(scope="module")
def rm():
    return _load_redis_meta()


def _sd_key(**overrides) -> str:
    """Serialize an SD key for test input construction.

    Uses the simplified schema: no cp_rank / cp_size fields.
    """
    defaults = dict(
        model_id="abc123",
        pp_rank=0, pp_size=1,
        tp_node_idx=0, tp_node_count=1,
        is_nsa=False,
    )
    defaults.update(overrides)
    return SharingDomainKey(**defaults).serialize()


class TestDeriveSdPrefix:
    """Matches the logic in ``csrc/dist/redis_meta_channel.cpp:list_node_keys``.

    Layout (simplified — no cp segment):
      blocks_key = sd:<sd_key>          → SD-only;   4 colons inside sd_key
      blocks_key = sd:<sd_key>:<device> → SD+device; one extra colon
      blocks_key = "blocks" / "CPUB"    → legacy;    0–1 colons
    """

    def _make_channel(self, rm, blocks_key: str) -> Any:
        # Construct without connecting: instantiate the wrapper directly
        # around a stub that captures ``blocks_key``.
        wrapper = rm.RedisMetaChannel.__new__(rm.RedisMetaChannel)
        wrapper._blocks_key = blocks_key
        return wrapper

    @pytest.mark.parametrize("device", ["", "CPUB", "SSDB", "PCFSB"])
    @pytest.mark.parametrize("pp,tpn,nsa", [
        (1, 1, False),
        (2, 1, False),
        (1, 2, False),
        (1, 1, True),
        (2, 2, True),  # upper bound under the simplified design (PP × tpn)
    ])
    def test_sd_plus_optional_device(self, rm, device, pp, tpn, nsa):
        sd_str = _sd_key(pp_size=pp, tp_node_count=tpn, is_nsa=nsa)
        blocks_key = f"sd:{sd_str}" + (f":{device}" if device else "")
        ch = self._make_channel(rm, blocks_key)
        derived = ch._derive_sd_prefix()
        expected = f"sd:{sd_str}"
        assert derived == expected, f"blocks_key={blocks_key!r} produced {derived!r}"

    def test_legacy_bare_key_collapses_to_empty(self, rm):
        for bk in ("blocks", "CPUB", "SSDB", "", "something-else"):
            ch = self._make_channel(rm, bk)
            assert ch._derive_sd_prefix() == ""


class TestListNodeKeysPattern:
    """Exercise the pattern that ``RedisMetaChannel.list_node_keys``
    fallback would scan when the C++ side is out-of-date."""

    def test_sd_aware_pattern(self, rm, monkeypatch):
        captured = {}

        class _FakeCExt:
            """Stands in for ``flexkv.c_ext.RedisMetaChannel``."""
            def list_node_keys(self):
                # Simulate a pre-Batch-B C++ that still returns bare ``node:*``
                # keys.  The wrapper must discard them and fall back to the
                # Python scan.
                return ["node:1", "node:2"]

            def list_keys(self, pattern):
                captured["pattern"] = pattern
                return ["sd:abc:pp0/1:tpn0/1:nsa0:node:99"]

        wrapper = rm.RedisMetaChannel.__new__(rm.RedisMetaChannel)
        wrapper._c = _FakeCExt()
        wrapper._blocks_key = "sd:abc:pp0/1:tpn0/1:nsa0:CPUB"
        # The fallback should scan the *SD-scoped* pattern, NOT bare node:*
        keys = wrapper.list_node_keys()
        assert captured["pattern"] == "sd:abc:pp0/1:tpn0/1:nsa0:node:*"
        assert keys == ["sd:abc:pp0/1:tpn0/1:nsa0:node:99"]
