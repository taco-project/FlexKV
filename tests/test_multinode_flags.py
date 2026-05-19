"""Phase 1 task 1-A — decoupled multinode flags on ``ModelConfig``.

These flags (``is_multinode_tp`` / ``is_multinode_cp``) intentionally cover
*orthogonal* physical situations that historically were conflated under a
single ``is_multinode`` switch in the sglang connector.  This regression
suite locks down the documented semantics:

  * ``is_multinode_tp`` is True when one TP group physically spans
    > 1 node.  This is the *only* dimension that affects SD-Remote
    construction (since ``tp_node_count > 1`` enters the SD key).

  * ``is_multinode_cp`` is True when CP > 1 AND the CP group physically
    crosses node boundaries.  Under sglang's standard megatron-style
    topology this is *always False* (see ``ModelConfig.is_multinode_cp``
    docstring); we still test the flag so future deployments that break
    that assumption are caught early.

Critically, *neither* flag must influence the ``SharingDomainKey`` —
``is_multinode_cp`` is a transport-layer hint, not an SD identity.
"""

from __future__ import annotations

import pytest

from flexkv.common.config import ModelConfig


def _make(
    *,
    tp_size: int,
    nnodes: int,
    pp_size: int = 1,
    dp_size: int = 1,
    attn_cp_size: int = 1,
    enable_dp_attention: bool = False,
) -> ModelConfig:
    """Build a ModelConfig with just the topology fields needed for these
    properties.  The other fields keep their defaults — they don't influence
    ``is_multinode_*``.

    NOTE: ``tp_rank`` and ``node_rank`` were moved out of ``ModelConfig``
    into ``RankInfo`` by the RankInfo refactor (PR #165), so we no
    longer pass them here.  ``is_multinode_tp`` / ``is_multinode_cp``
    are pure cluster-topology properties — they read ``tp_size``,
    ``pp_size``, ``nnodes`` and ``attn_cp_size`` only.
    """
    return ModelConfig(
        num_layers=1, num_kv_heads=1, head_size=1,
        tp_size=tp_size,
        pp_size=pp_size,
        dp_size=dp_size,
        nnodes=nnodes,
        attn_cp_size=attn_cp_size,
        enable_dp_attention=enable_dp_attention,
    )


# ===========================================================================
# is_multinode_tp
# ===========================================================================
class TestIsMultinodeTp:
    def test_single_node_tp(self):
        # 8-way TP on one node — TP fits inside a single host.
        mc = _make(tp_size=8, nnodes=1)
        assert mc.is_multinode_tp is False
        assert mc.tp_node_count == 1

    def test_cross_node_tp(self):
        # 16-way TP across 2 nodes (8 GPUs per node).
        mc = _make(tp_size=16, nnodes=2)
        assert mc.is_multinode_tp is True
        assert mc.tp_node_count == 2

    def test_pp_alone_does_not_imply_multinode_tp(self):
        # PP=2 across 2 nodes (8-way TP each), TP itself is single-node.
        mc = _make(tp_size=8, nnodes=2, pp_size=2)
        assert mc.is_multinode_tp is False
        assert mc.nnodes_per_tp_group == 1


# ===========================================================================
# is_multinode_cp
# ===========================================================================
class TestIsMultinodeCp:
    """In FlexKV's topology model, CP is a sub-partition *inside* TP, not a
    top-level GPU dimension.  ``total_gpus = tp × pp × dp`` (CP excluded).
    A CP group occupies ``attn_cp_size`` GPUs carved out of the
    ``tp_size`` assigned to one TP group, so CP crosses nodes iff
    ``attn_cp_size > tp_size_per_node``.
    """

    def test_cp_disabled_is_false(self):
        mc = _make(tp_size=8, nnodes=1, attn_cp_size=1)
        assert mc.is_multinode_cp is False

    def test_cp_fits_inside_one_tp_node_share(self):
        # Single-node TP=8, CP=2 — tp_size_per_node=8 ≥ cp=2 ⇒ CP fits.
        mc = _make(tp_size=8, nnodes=1, attn_cp_size=2)
        assert mc.is_multinode_cp is False
        assert mc.tp_size_per_node == 8

    def test_cp_equals_tp_per_node_share(self):
        # tp_size_per_node == attn_cp_size: CP exactly fills one node's TP slice.
        # Strictly intra-node — not crossing.
        mc = _make(tp_size=4, nnodes=2, attn_cp_size=2, pp_size=1)
        # nnodes_per_tp_group = 2 → tp_size_per_node = 4/2 = 2 == cp ⇒ False
        assert mc.tp_size_per_node == 2
        assert mc.is_multinode_cp is False

    def test_cp_exceeds_tp_per_node_share_crosses(self):
        # tp_size_per_node < attn_cp_size: CP must cross nodes.
        # tp=4, nnodes=2, pp=1 → tp_size_per_node = 2; cp=4 > 2 ⇒ True.
        mc = _make(tp_size=4, nnodes=2, attn_cp_size=4, pp_size=1)
        assert mc.tp_size_per_node == 2
        assert mc.is_multinode_cp is True

    def test_megatron_typical_deployments_are_intra_node(self):
        """Production-style configurations (CP ≤ tp_size_per_node) — all False."""
        for tp, nnodes, pp, cp in [
            (8, 1, 1, 2),    # single-node TP=8, CP=2
            (8, 2, 1, 2),    # tp_size_per_node=4, CP=2 → fits
            (8, 2, 2, 2),    # PP=2, tp_size_per_node=8, CP=2 → fits (per-PP-stage TP single-node)
            (16, 2, 1, 2),   # tp_size_per_node=8, CP=2 → fits
        ]:
            mc = _make(tp_size=tp, nnodes=nnodes, pp_size=pp, attn_cp_size=cp)
            assert mc.is_multinode_cp is False, (
                f"is_multinode_cp should be False (tp={tp}, nnodes={nnodes}, "
                f"pp={pp}, cp={cp}); got True "
                f"(tp_size_per_node={mc.tp_size_per_node})"
            )


# ===========================================================================
# Independence
# ===========================================================================
class TestIndependence:
    """is_multinode_tp and is_multinode_cp must be physically independent."""

    def test_only_tp_multinode(self):
        mc = _make(tp_size=16, nnodes=2)  # cross-node TP, no CP
        assert mc.is_multinode_tp is True
        assert mc.is_multinode_cp is False

    def test_both_false_baseline(self):
        mc = _make(tp_size=4, nnodes=1)
        assert mc.is_multinode_tp is False
        assert mc.is_multinode_cp is False


# ===========================================================================
# Property: flags do NOT influence SharingDomainKey
# ===========================================================================
class TestNoSdKeyLeak:
    """Regression: changing CP must never alter ``SharingDomainKey``.

    The simplified design (§4.5) explicitly excludes CP from the SD key.
    We assert that here so any future change which accidentally feeds
    CP-dimension info into ``from_model_config`` will fail this test.
    """

    def test_cp_size_does_not_affect_sd_key(self):
        from flexkv.common.dist_reuse.sharing_domain import SharingDomainKey

        mc_no_cp = _make(tp_size=4, nnodes=1, attn_cp_size=1)
        mc_with_cp = _make(tp_size=4, nnodes=1, attn_cp_size=2)
        # The two configs differ in CP only — but SD must not know.
        assert mc_no_cp.attn_cp_size != mc_with_cp.attn_cp_size
        sd1 = SharingDomainKey.from_model_config(mc_no_cp)
        sd2 = SharingDomainKey.from_model_config(mc_with_cp)
        assert sd1 == sd2
        assert sd1.serialize() == sd2.serialize()

    def test_serialized_sd_never_contains_cp(self):
        """Belt-and-braces check on the on-the-wire format itself."""
        from flexkv.common.dist_reuse.sharing_domain import SharingDomainKey

        mc = _make(tp_size=4, nnodes=1, attn_cp_size=2)
        sd = SharingDomainKey.from_model_config(mc)
        s = sd.serialize()
        assert ":cp" not in s, f"Serialized SD must never contain ':cp...' segment, got: {s}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
