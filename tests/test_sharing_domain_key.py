"""Unit tests for ``flexkv.common.dist_reuse.sharing_domain``.

Phase 0 — simplified design (CP not in sd_key, ``is_nsa_cp`` renamed to
``is_nsa``).  See
``docs/dist_reuse/dist_reuse_with_cp_pp_multinode_tp_simplified.md`` for the
authoritative definition of the current schema.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from flexkv.common.dist_reuse.sharing_domain import (
    DEFAULT_MODEL_ID,
    SharingDomainKey,
    derive_model_id,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@dataclass
class _FakeModelConfig:
    """Minimal stub of ``ModelConfig`` for ``from_model_config`` tests.

    Matches the post-rename field set: uses ``is_nsa`` (not ``is_nsa_cp``).
    Also carries ``nnodes`` so the node-granularity PP collapse can be
    exercised (``nnodes=pp_size`` for the typical pipelined-across-nodes
    case; ``nnodes=1, pp_size>1`` is rejected by the factory).
    """

    num_layers: int = 32
    num_kv_heads: int = 8
    head_size: int = 128
    use_mla: bool = False
    dtype: Any = "bfloat16"
    pp_rank: int = 0
    pp_size: int = 1
    nnodes: int = 1
    tp_node_idx: int = 0
    tp_node_count: int = 1
    # CP info is *not* in the SD key (simplified design §4.5), but
    # ModelConfig still carries attn_cp_* fields for other purposes — the
    # from_model_config factory just doesn't read them.
    attn_cp_rank: int = 0
    attn_cp_size: int = 1
    is_nsa: bool = False
    model_id: Any = None


# ---------------------------------------------------------------------------
# derive_model_id
# ---------------------------------------------------------------------------
class TestDeriveModelId:
    def test_stable_across_calls(self):
        a = derive_model_id(num_layers=32, num_kv_heads=8, head_size=128,
                            dtype="bfloat16", use_mla=False)
        b = derive_model_id(num_layers=32, num_kv_heads=8, head_size=128,
                            dtype="bfloat16", use_mla=False)
        assert a == b
        assert len(a) == 16
        # All hex chars
        int(a, 16)

    def test_changes_with_field(self):
        ref = derive_model_id(num_layers=32, num_kv_heads=8, head_size=128,
                              dtype="bfloat16", use_mla=False)
        for tweak in (
            dict(num_layers=64),
            dict(num_kv_heads=16),
            dict(head_size=256),
            dict(dtype="float16"),
            dict(use_mla=True),
        ):
            kwargs = dict(num_layers=32, num_kv_heads=8, head_size=128,
                          dtype="bfloat16", use_mla=False)
            kwargs.update(tweak)
            assert derive_model_id(**kwargs) != ref, f"tweak {tweak} should change model_id"


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------
class TestSharingDomainKeyValidation:
    @pytest.fixture
    def base_kwargs(self):
        return dict(
            model_id="abc123",
            pp_node_idx=0, pp_node_count=1,
            tp_node_idx=0, tp_node_count=1,
            is_nsa=False,
        )

    def test_construct_ok(self, base_kwargs):
        sd = SharingDomainKey(**base_kwargs)
        assert sd.model_id == "abc123"
        assert sd.is_master()

    def test_cp_fields_rejected(self, base_kwargs):
        """CP is not in the SD key any more — legacy cp_* kwargs must fail."""
        base_kwargs["cp_rank"] = 0
        base_kwargs["cp_size"] = 4
        with pytest.raises(TypeError):
            SharingDomainKey(**base_kwargs)

    def test_is_nsa_cp_kwarg_rejected(self, base_kwargs):
        """Old ``is_nsa_cp`` kwarg must not silently slip through — the new
        canonical name is ``is_nsa`` and constructors are strict."""
        base_kwargs.pop("is_nsa")
        base_kwargs["is_nsa_cp"] = False
        with pytest.raises(TypeError):
            SharingDomainKey(**base_kwargs)

    @pytest.mark.parametrize("bad_id", ["", "has:colon", "has space", None, 123])
    def test_bad_model_id(self, base_kwargs, bad_id):
        base_kwargs["model_id"] = bad_id
        with pytest.raises((ValueError, TypeError)):
            SharingDomainKey(**base_kwargs)

    @pytest.mark.parametrize("field,value", [
        ("pp_node_count", 0),
        ("tp_node_count", -1),
    ])
    def test_bad_count(self, base_kwargs, field, value):
        base_kwargs[field] = value
        with pytest.raises((ValueError, TypeError)):
            SharingDomainKey(**base_kwargs)

    def test_idx_out_of_range(self, base_kwargs):
        base_kwargs.update(pp_node_idx=2, pp_node_count=2)  # max idx = count - 1 = 1
        with pytest.raises(ValueError):
            SharingDomainKey(**base_kwargs)

    def test_negative_idx(self, base_kwargs):
        base_kwargs["tp_node_idx"] = -1
        with pytest.raises(ValueError):
            SharingDomainKey(**base_kwargs)

    def test_is_nsa_must_be_bool(self, base_kwargs):
        base_kwargs["is_nsa"] = 1  # type: ignore[arg-type]
        with pytest.raises(ValueError):
            SharingDomainKey(**base_kwargs)


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------
class TestSharingDomainKeySerialization:
    @pytest.mark.parametrize("sd", [
        SharingDomainKey.default(),
        SharingDomainKey(
            model_id="abc123",
            pp_node_idx=0, pp_node_count=2,
            tp_node_idx=1, tp_node_count=2,
            is_nsa=True,
        ),
        SharingDomainKey(
            model_id="model-v1.2_x",
            pp_node_idx=0, pp_node_count=1,
            tp_node_idx=0, tp_node_count=1,
            is_nsa=False,
        ),
    ])
    def test_round_trip(self, sd):
        s = sd.serialize()
        # Sanity: contains all four segments separated by ':'
        assert s.count(":") == 3
        out = SharingDomainKey.deserialize(s)
        assert out == sd
        assert out.serialize() == s

    def test_serialize_format(self):
        sd = SharingDomainKey(
            model_id="abc123",
            pp_node_idx=1, pp_node_count=2,
            tp_node_idx=0, tp_node_count=2,
            is_nsa=True,
        )
        assert sd.serialize() == "abc123:ppn1/2:tpn0/2:nsa1"

    def test_serialize_no_cp_segment(self):
        """Regression: the key must never contain ':cp<...>' — CP is gone."""
        sd = SharingDomainKey(
            model_id="m", pp_node_idx=0, pp_node_count=1,
            tp_node_idx=0, tp_node_count=1, is_nsa=False,
        )
        assert ":cp" not in sd.serialize()

    def test_serialize_uses_ppn_prefix_not_pp(self):
        """Regression: the PP segment uses the node-granularity prefix
        ``ppn``.  The legacy rank-granularity prefix ``pp<rank>/<size>``
        is gone."""
        sd = SharingDomainKey(
            model_id="m", pp_node_idx=0, pp_node_count=2,
            tp_node_idx=0, tp_node_count=1, is_nsa=False,
        )
        s = sd.serialize()
        assert ":ppn0/2:" in s
        assert ":pp0/2:" not in s

    @pytest.mark.parametrize("bad", [
        "abc",                               # missing fields
        "abc:ppn0/1:tpn0/1",                 # missing nsa
        "abc:ppn0/1:tpn0/1:cp0/1:nsa0",      # legacy 5-segment form with cp
        "abc:foo0/1:tpn0/1:nsa0",            # wrong ppn prefix
        "abc:pp0/1:tpn0/1:nsa0",             # legacy rank-granularity prefix
        "abc:ppn0:tpn0/1:nsa0",              # missing '/'
        "abc:ppn0/1:tpn0/1:nsa2",            # bad nsa value
        "abc:ppnX/1:tpn0/1:nsa0",            # non-int
    ])
    def test_deserialize_rejects_bad(self, bad):
        with pytest.raises(ValueError):
            SharingDomainKey.deserialize(bad)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
class TestFromModelConfig:
    def test_basic(self):
        mc = _FakeModelConfig()
        sd = SharingDomainKey.from_model_config(mc)
        assert sd.is_master()
        assert sd.pp_node_count == 1 and sd.tp_node_count == 1
        # model_id should be a 16-char hex digest, not the placeholder.
        assert sd.model_id != DEFAULT_MODEL_ID
        assert len(sd.model_id) == 16

    def test_explicit_model_id_takes_precedence(self):
        mc = _FakeModelConfig(model_id="my-model")
        sd = SharingDomainKey.from_model_config(mc)
        assert sd.model_id == "my-model"

    def test_overrides(self):
        # PP=2 across nnodes=2 — a legitimate cross-node PP layout.
        mc = _FakeModelConfig(pp_size=2, nnodes=2)
        sd_pp1 = SharingDomainKey.from_model_config(
            mc, overrides={"pp_node_idx": 1},
        )
        assert sd_pp1.pp_node_idx == 1
        assert not sd_pp1.is_master()

    def test_single_node_pp_gt_1_rejected(self):
        """Single-node PP>1 must not be allowed to derive an SD key — the
        per-rank CPU pool only stores a layer slice, so a single sd_key
        would alias incompatible bytes across PP ranks.  Until the CPU
        pool is reworked to full-layer storage, this combo is hard-blocked
        at the SD-key factory."""
        for pp in (2, 4, 8):
            mc = _FakeModelConfig(pp_size=pp, nnodes=1)
            with pytest.raises(ValueError, match="single-node PP>1"):
                SharingDomainKey.from_model_config(mc)

    def test_pp_collapses_to_node_granularity(self):
        """Cross-node PP=4 over nnodes=2 should collapse rank pairs to the
        same ``pp_node_idx`` (rank 0/1 → node 0, rank 2/3 → node 1)."""
        # Rank 1 lives on node 0 (pp_per_node = 4//2 = 2 → 1 // 2 == 0).
        mc = _FakeModelConfig(pp_size=4, nnodes=2, pp_rank=1)
        sd_node0 = SharingDomainKey.from_model_config(mc)
        assert sd_node0.pp_node_idx == 0
        assert sd_node0.pp_node_count == 2

        # Rank 2 lives on node 1 (2 // 2 == 1).
        mc = _FakeModelConfig(pp_size=4, nnodes=2, pp_rank=2)
        sd_node1 = SharingDomainKey.from_model_config(mc)
        assert sd_node1.pp_node_idx == 1
        assert sd_node1.pp_node_count == 2

    def test_pp1_multinode_keeps_pp_node_count_1(self):
        """PP=1 across multiple nodes (cross-node TP only) must keep
        pp_node_count=1 — PP doesn't span nodes here."""
        mc = _FakeModelConfig(pp_size=1, nnodes=2, tp_node_count=2)
        sd = SharingDomainKey.from_model_config(mc)
        assert sd.pp_node_count == 1 and sd.pp_node_idx == 0
        assert sd.tp_node_count == 2

    def test_unknown_override_rejected(self):
        """Unknown overrides must raise — in particular the removed cp_rank."""
        with pytest.raises(ValueError):
            SharingDomainKey.from_model_config(
                _FakeModelConfig(), overrides={"bogus": 1},
            )

    def test_cp_rank_override_rejected(self):
        """Regression: cp_rank used to be a valid override under the old
        schema.  After simplification it must be rejected."""
        with pytest.raises(ValueError):
            SharingDomainKey.from_model_config(
                _FakeModelConfig(), overrides={"cp_rank": 1},
            )

    def test_legacy_pp_rank_override_rejected(self):
        """Regression: ``pp_rank`` was a valid override under the
        rank-granularity schema; the SD key now only knows about
        ``pp_node_idx`` so the legacy override must be rejected to
        catch stale callers."""
        mc = _FakeModelConfig(pp_size=2, nnodes=2)
        with pytest.raises(ValueError):
            SharingDomainKey.from_model_config(
                mc, overrides={"pp_rank": 1},
            )

    def test_cp_fields_in_model_config_ignored(self):
        """Having attn_cp_* on the model config must NOT leak into the SD key."""
        mc = _FakeModelConfig(attn_cp_rank=2, attn_cp_size=4)
        sd = SharingDomainKey.from_model_config(mc)
        # Cannot read cp_rank from sd — the field does not exist any more.
        assert not hasattr(sd, "cp_rank")
        assert not hasattr(sd, "cp_size")
        # The legacy pp_rank/pp_size rank-granularity fields are gone too.
        assert not hasattr(sd, "pp_rank")
        assert not hasattr(sd, "pp_size")
        # Serialization must also not contain a cp segment.
        assert ":cp" not in sd.serialize()
        # (ppn=0, tpn=0) is still the master SD.
        assert sd.is_master()

    def test_is_nsa_picked_up(self):
        mc = _FakeModelConfig(is_nsa=True)
        sd = SharingDomainKey.from_model_config(mc)
        assert sd.is_nsa is True


# ---------------------------------------------------------------------------
# enumerate_peers
# ---------------------------------------------------------------------------
class TestEnumeratePeers:
    @pytest.mark.parametrize("ppn,tpn,expected", [
        (1, 1, 1),
        (2, 1, 2),
        (1, 2, 2),
        (2, 2, 4),  # current-scope upper bound (pp_node_count=2 × tp_node_count=2)
    ])
    def test_count(self, ppn, tpn, expected):
        sd = SharingDomainKey(
            model_id="m",
            pp_node_idx=0, pp_node_count=ppn,
            tp_node_idx=0, tp_node_count=tpn,
            is_nsa=False,
        )
        peers = sd.enumerate_peers()
        assert len(peers) == expected
        assert sd.total_sd_count() == expected

    def test_upper_bound_is_four(self):
        """Regression: the SD count upper bound is now 4 (pp_node_count × tp_node_count),
        not 32 like in the legacy design that included CP in the SD key."""
        sd = SharingDomainKey(
            model_id="m", pp_node_idx=0, pp_node_count=2,
            tp_node_idx=0, tp_node_count=2, is_nsa=False,
        )
        assert sd.total_sd_count() == 4

    def test_unique_serialization(self):
        sd = SharingDomainKey(
            model_id="m", pp_node_idx=0, pp_node_count=2,
            tp_node_idx=0, tp_node_count=2, is_nsa=True,
        )
        peers = sd.enumerate_peers()
        serialized = {p.serialize() for p in peers}
        assert len(serialized) == 4, "4 SDs must produce 4 distinct keys"
        # Every SD has the same model_id and is_nsa as the original.
        assert all(p.model_id == "m" for p in peers)
        assert all(p.is_nsa is True for p in peers)
        # Exactly one master.
        masters = [p for p in peers if p.is_master()]
        assert len(masters) == 1

    def test_iter_dunder(self):
        sd = SharingDomainKey.default()
        assert list(sd) == sd.enumerate_peers()


# ---------------------------------------------------------------------------
# default()
# ---------------------------------------------------------------------------
class TestDefault:
    def test_default_is_master(self):
        sd = SharingDomainKey.default()
        assert sd.is_master()
        assert sd.pp_node_count == sd.tp_node_count == 1
        assert not sd.is_nsa
        assert sd.model_id == DEFAULT_MODEL_ID

    def test_default_serialize_round_trip(self):
        sd = SharingDomainKey.default()
        out = SharingDomainKey.deserialize(sd.serialize())
        assert out == sd

    def test_default_total_sd_count(self):
        assert SharingDomainKey.default().total_sd_count() == 1


# ---------------------------------------------------------------------------
# Hashability
# ---------------------------------------------------------------------------
class TestHashable:
    def test_can_be_dict_key(self):
        a = SharingDomainKey.default()
        b = SharingDomainKey.default()
        d = {a: 1}
        assert d[b] == 1  # equal keys collapse

    def test_distinct_keys_distinct_hash(self):
        sd = SharingDomainKey(
            model_id="m", pp_node_idx=0, pp_node_count=2,
            tp_node_idx=0, tp_node_count=1,
            is_nsa=False,
        )
        peers = sd.enumerate_peers()
        # Set semantics: 2 peers => 2 hashes (modulo collisions, which are
        # vanishingly unlikely for tuples of small ints).
        assert len({hash(p) for p in peers}) == 2
