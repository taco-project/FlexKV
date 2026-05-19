"""Unit tests for ``flexkv.cache.sharing_domain_namespace`` (Phase 0 task 0-B)."""
from __future__ import annotations

import pytest

from flexkv.common.dist_reuse.sharing_domain import SharingDomainKey
from flexkv.common.dist_reuse.sharing_domain_namespace import (
    INSTANCE_KEY_PREFIX,
    SD_KEY_PREFIX,
    SharingDomainNamespace,
)


@pytest.fixture
def ns():
    sd = SharingDomainKey(
        model_id="abc123",
        pp_rank=1, pp_size=2,
        tp_node_idx=0, tp_node_count=2,
        is_nsa=True,
    )
    return SharingDomainNamespace(sd)


def test_constructor_rejects_non_sd_key():
    with pytest.raises(TypeError):
        SharingDomainNamespace("not-an-sd-key")  # type: ignore[arg-type]


def test_prefix_format(ns):
    assert ns.prefix == f"{SD_KEY_PREFIX}:abc123:pp1/2:tpn0/2:nsa1"
    assert ns.serialized_sd == "abc123:pp1/2:tpn0/2:nsa1"


@pytest.mark.parametrize("builder,fmt", [
    ("node_key", "{prefix}:node:{nid}"),
    ("meta_key", "{prefix}:meta:{nid}"),
])
def test_simple_keys(ns, builder, fmt):
    actual = getattr(ns, builder)(7)
    assert actual == fmt.format(prefix=ns.prefix, nid=7)


def test_buffer_key(ns):
    assert ns.buffer_key(7, 0xDEADBEEF) == f"{ns.prefix}:buffer:7:{0xDEADBEEF}"


def test_block_key_lower_hex(ns):
    # 0x10ab... should render as 10abcdef, not 0X10ABCDEF
    assert ns.block_key(7, 0x10ABCDEF) == f"{ns.prefix}:block:7:10abcdef"


def test_block_key_handles_negative_hash(ns):
    """C++ side may produce signed int64 hashes; we mask to 64 bits so the
    hex never carries a leading sign."""
    h = -1  # 64-bit two's complement => 0xFFFFFFFFFFFFFFFF
    assert ns.block_key(0, h) == f"{ns.prefix}:block:0:ffffffffffffffff"


def test_aggregate_key(ns):
    assert ns.aggregate_key(0xCAFEBABE) == f"{ns.prefix}:aggregate:cafebabe"


def test_scan_patterns(ns):
    assert ns.node_key_pattern() == f"{ns.prefix}:node:*"
    assert ns.meta_key_pattern() == f"{ns.prefix}:meta:*"
    assert ns.buffer_key_pattern() == f"{ns.prefix}:buffer:*"
    assert ns.block_key_pattern() == f"{ns.prefix}:block:*"
    assert ns.block_key_pattern_for_node(7) == f"{ns.prefix}:block:7:*"


# ---------------------------------------------------------------------------
# Cross-SD instance keys
# ---------------------------------------------------------------------------
class TestInstanceKeys:
    def test_session_key(self):
        assert (
            SharingDomainNamespace.instance_session_key("inst-001")
            == f"{INSTANCE_KEY_PREFIX}:inst-001:session"
        )

    def test_sd_nodes_key(self):
        assert (
            SharingDomainNamespace.instance_sd_nodes_key("inst-001")
            == f"{INSTANCE_KEY_PREFIX}:inst-001:sd_nodes"
        )

    def test_session_pattern(self):
        assert (
            SharingDomainNamespace.instance_session_key_pattern()
            == f"{INSTANCE_KEY_PREFIX}:*:session"
        )

    @pytest.mark.parametrize("bad_id", ["", "has space", "has:colon", "$dollar"])
    def test_rejects_bad_id(self, bad_id):
        with pytest.raises(ValueError):
            SharingDomainNamespace.instance_session_key(bad_id)
        with pytest.raises(ValueError):
            SharingDomainNamespace.instance_sd_nodes_key(bad_id)

    def test_parse_round_trip(self):
        for pid in ["inst-001", "abc.v2", "x_y_z", "ABC123"]:
            key = SharingDomainNamespace.instance_session_key(pid)
            parsed = SharingDomainNamespace.parse_instance_session_key(key)
            assert parsed == pid

    @pytest.mark.parametrize("bad", [
        "not-a-flexkv-key",
        "flexkv:instance:foo:notsession",
        "flexkv:instance::session",  # empty instance_id
    ])
    def test_parse_rejects_bad(self, bad):
        with pytest.raises(ValueError):
            SharingDomainNamespace.parse_instance_session_key(bad)


# ---------------------------------------------------------------------------
# Equality / hashability
# ---------------------------------------------------------------------------
class TestEquality:
    def test_equal_namespaces_share_hash(self):
        sd = SharingDomainKey.default()
        a = SharingDomainNamespace(sd)
        b = SharingDomainNamespace(sd)
        assert a == b
        assert hash(a) == hash(b)

    def test_different_sd_keys_unequal(self):
        a = SharingDomainNamespace(SharingDomainKey.default())
        b_sd = SharingDomainKey(
            model_id="m", pp_rank=0, pp_size=2,
            tp_node_idx=0, tp_node_count=1,
            is_nsa=False,
        )
        b = SharingDomainNamespace(b_sd)
        assert a != b
