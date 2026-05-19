"""Scalability sanity tests for :class:`SharingDomainKey` under the simplified
dist_reuse schema.

Design doc §4.1 / §4.5 (simplified): CP is **not** part of the SD key.  The
SD-count upper bound is ``pp_size × tp_node_count`` (≤ 2 in the current
deployment, ≤ 4 in the future-extension scope ``PP=2 × tp_node_count=2``).

We don't enforce a *hard* cap on those dims at the data-structure layer —
this test confirms that:

* :meth:`SharingDomainKey.enumerate_peers` returns exactly
  ``pp_size × tp_node_count`` unique SDs for every shape;
* every enumerated SD serializes to a distinct, round-trippable string;
* the serialization is purely textual so it can be used verbatim as a
  Redis key-namespace prefix;
* :func:`graph_needs_gpu_clear` returns True iff the peer's pp_rank
  differs from self's (TP-node dim alone keeps the same slot_mapping,
  so no GPU clear is required);
* :meth:`total_sd_count` equals ``len(enumerate_peers())``.

Covers all shapes listed in the simplified design doc §4.1 table, plus
two stress-sized configurations.
"""
from __future__ import annotations

import hashlib
from typing import Dict, Set, Tuple

import pytest

from flexkv.common.dist_reuse import (
    SharingDomainKey,
    SharingDomainNamespace,
    graph_needs_gpu_clear,
)


MODEL_ID = "scaletest"


def _sd(pp_rank=0, pp_size=1, tpn_idx=0, tpn_count=1,
        is_nsa=False) -> SharingDomainKey:
    """Build an SD key with the simplified schema (no cp_* fields)."""
    return SharingDomainKey(
        model_id=MODEL_ID,
        pp_rank=pp_rank, pp_size=pp_size,
        tp_node_idx=tpn_idx, tp_node_count=tpn_count,
        is_nsa=is_nsa,
    )


# ---------------------------------------------------------------------------
# Shapes drawn from the simplified design doc §4.1 + a few stress cases.
# (pp, tpn, label)
# ---------------------------------------------------------------------------
SHAPES = [
    (1, 1, "1-SD degenerate"),
    (2, 1, "PP=2 (current scope)"),
    (1, 2, "cross-node TP=2 (current scope)"),
    (2, 2, "PP=2 × cross-node TP=2 (future-extension upper bound, 4 SDs)"),
    # Stress: confirm no hidden caps.  Real deployments never go here.
    (4, 1, "PP=4 (stress)"),
    (1, 8, "tp_node_count=8 (stress)"),
    (4, 4, "PP=4 × tp_node_count=4 (stress, 16 SDs)"),
]


# ===========================================================================
# Enumerate completeness + uniqueness
# ===========================================================================
@pytest.mark.parametrize("pp,tpn,label", SHAPES)
def test_enumerate_peers_is_complete_cartesian_product(pp, tpn, label):
    """``enumerate_peers`` must return every SD in the instance exactly once
    — product count = pp × tpn, no duplicates.  CP no longer contributes."""
    master = _sd(pp_size=pp, tpn_count=tpn)
    peers = master.enumerate_peers()

    # Count matches the Cartesian product PP × tpn.
    assert len(peers) == pp * tpn == master.total_sd_count()

    # Every (pp_rank, tp_node_idx) pair appears exactly once.
    pairs: Dict[Tuple[int, int], int] = {}
    for p in peers:
        pairs[(p.pp_rank, p.tp_node_idx)] = pairs.get((p.pp_rank, p.tp_node_idx), 0) + 1
    assert set(pairs.values()) == {1}, (
        f"[{label}] duplicate SDs detected in enumerate_peers: {pairs}"
    )
    # And every coordinate in range.
    for (pr, ti), _ in pairs.items():
        assert 0 <= pr < pp
        assert 0 <= ti < tpn


@pytest.mark.parametrize("pp,tpn,label", SHAPES)
def test_enumerated_sds_serialize_to_distinct_strings(pp, tpn, label):
    """Every SD in the same instance must produce a unique serialize().

    This is the core guarantee that lets us use ``sd.serialize()`` as a
    Redis key prefix without worrying about cross-SD key collisions.
    """
    master = _sd(pp_size=pp, tpn_count=tpn)
    serials: Set[str] = {p.serialize() for p in master.enumerate_peers()}
    assert len(serials) == pp * tpn, (
        f"[{label}] serialize() collided for some SDs — Redis key namespaces "
        f"would overlap.  Unique serials: {len(serials)} / expected "
        f"{pp * tpn}"
    )


@pytest.mark.parametrize("pp,tpn,label", SHAPES)
def test_serialize_deserialize_round_trip(pp, tpn, label):
    master = _sd(pp_size=pp, tpn_count=tpn)
    for p in master.enumerate_peers():
        s = p.serialize()
        back = SharingDomainKey.deserialize(s)
        assert back == p, (
            f"[{label}] round-trip failed: {p!r} → {s!r} → {back!r}"
        )


# ===========================================================================
# Simplified-design contract: CP must not show up in the SD key
# ===========================================================================
def test_no_cp_segment_in_any_serialized_sd():
    """Regression for the simplification: ``serialize()`` must not contain
    a ``:cp<...>:`` segment in any enumerated SD, regardless of shape."""
    for pp, tpn, _label in SHAPES:
        for p in _sd(pp_size=pp, tpn_count=tpn).enumerate_peers():
            assert ":cp" not in p.serialize(), (
                f"unexpected CP segment in serialized SD: {p.serialize()!r}"
            )


def test_total_sd_count_is_only_pp_times_tpn():
    """Regression: enabling CP at the model-config level must not multiply
    the SD count.  In the simplified schema only pp × tpn matters."""
    # The simplified SharingDomainKey doesn't even accept CP kwargs, so we
    # just confirm the formula on the most commonly-exercised shapes.
    assert _sd(pp_size=1, tpn_count=1).total_sd_count() == 1
    assert _sd(pp_size=2, tpn_count=1).total_sd_count() == 2
    assert _sd(pp_size=1, tpn_count=2).total_sd_count() == 2
    assert _sd(pp_size=2, tpn_count=2).total_sd_count() == 4


# ===========================================================================
# No hard-coded upper bounds — each dim independently scalable
# ===========================================================================
def test_no_hardcoded_limit_on_any_single_dim():
    """Each dim independently scaled to 16 must still enumerate cleanly.

    If any of the two dims were capped internally (e.g. by a fixed-size
    array), this test would blow up.
    """
    assert len(_sd(pp_size=16).enumerate_peers()) == 16
    assert len(_sd(tpn_count=16).enumerate_peers()) == 16


def test_stress_16_sds_enumerate_within_budget():
    """``4 × 4 = 16`` SDs: enumerate is O(N) and cheap."""
    import time
    master = _sd(pp_size=4, tpn_count=4)
    t0 = time.perf_counter()
    peers = master.enumerate_peers()
    elapsed = time.perf_counter() - t0
    assert len(peers) == 16
    # Cheap generator should be near-instant on CPU.
    assert elapsed < 0.1, f"enumerate_peers took {elapsed:.3f}s, expected < 0.1s"


# ===========================================================================
# graph_needs_gpu_clear semantic stays correct under the simplified schema
# ===========================================================================
def test_graph_needs_gpu_clear_only_pp_dim_forces_clear():
    """Full 2-D grid sweep — under the simplified schema the rule is:

        clear?  ⇔  peer.pp_rank != self.pp_rank

    Crossing the TP-node boundary alone does NOT require clearing
    (TP shares the slot_mapping across all of its ranks).  CP is not in
    the SD key any more so there's no CP component to this check.
    """
    master = _sd(pp_size=4, tpn_count=4)
    # Self → no clear.
    assert graph_needs_gpu_clear(master, master) is False

    for peer in master.enumerate_peers():
        expect_clear = peer.pp_rank != master.pp_rank
        got = graph_needs_gpu_clear(master, peer)
        assert got == expect_clear, (
            f"graph_needs_gpu_clear({master!r}, {peer!r}) = {got}; "
            f"expected {expect_clear} (rule: pp_rank differs)"
        )


# ===========================================================================
# Namespace-level: each SD's Redis prefix is a proper subset partition
# ===========================================================================
@pytest.mark.parametrize("pp,tpn,label", [
    (2, 2, "current upper bound: 4 SDs"),
    (4, 4, "stress: 16 SDs"),
])
def test_each_sd_has_disjoint_redis_prefix(pp, tpn, label):
    master = _sd(pp_size=pp, tpn_count=tpn)
    prefixes: Set[str] = set()
    for p in master.enumerate_peers():
        ns = SharingDomainNamespace(p)
        prefixes.add(ns.prefix)
    # prefixes[i] is the full leading string for all keys in SD i.
    # They must be unique, and none may be a prefix of another (otherwise a
    # SCAN sd:<i>:* would leak into SD j).
    sorted_prefixes = sorted(prefixes)
    for i in range(len(sorted_prefixes) - 1):
        a, b = sorted_prefixes[i], sorted_prefixes[i + 1]
        assert not b.startswith(a + ":"), (
            f"[{label}] prefix '{a}' is a proper prefix of '{b}' — SCANs would leak"
        )


# ===========================================================================
# model_id invariance: same topology + same model_arch ⇒ same sd_key strings
# ===========================================================================
def test_model_id_is_stable_across_calls():
    """``model_id`` must be deterministic across runs — serialize() is a
    pure function of the 6 dataclass fields (no hidden randomness)."""
    sd1 = _sd(pp_size=4, tpn_count=4, pp_rank=1, tpn_idx=2)
    sd2 = _sd(pp_size=4, tpn_count=4, pp_rank=1, tpn_idx=2)
    assert sd1.serialize() == sd2.serialize()
    h1 = hashlib.sha1(sd1.serialize().encode()).hexdigest()
    h2 = hashlib.sha1(sd2.serialize().encode()).hexdigest()
    assert h1 == h2


# ===========================================================================
# Master role count: exactly ONE SD in the instance is the Master
# ===========================================================================
@pytest.mark.parametrize("pp,tpn,label", SHAPES)
def test_exactly_one_master_in_any_instance(pp, tpn, label):
    master = _sd(pp_size=pp, tpn_count=tpn)
    peers = master.enumerate_peers()
    masters = [p for p in peers if p.is_master()]
    assert len(masters) == 1, (
        f"[{label}] expected exactly 1 Master SD in the instance; got {len(masters)}"
    )
    # And it has all-zero ranks.
    m = masters[0]
    assert m.pp_rank == 0 and m.tp_node_idx == 0
