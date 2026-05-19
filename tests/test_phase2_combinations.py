"""Phase 2 / Phase 3 combinations — simplified schema (CP not in SD key).

Design doc §4.10.1 (simplified):

* Current deployment (prefill crosses ≤ 2 nodes):
  - ``PP=2, tp_node_count=1`` — 2 SDs (PP-Remote lives on another node);
  - ``PP=1, tp_node_count=2`` — 2 SDs (TP-Remote lives on another node).
* Future-extension scope (not currently deployed):
  - ``PP=2, tp_node_count=2`` — 4 SDs.

CP (``cp_size``) does **not** multiply the SD count — each cp_rank inside
the same CP group shares the SD with the sync_leader, and data is dispatched
in-process via sglang's scatter (see simplified design doc §4.5).  This is
the main regression the tests below guard against.
"""
from __future__ import annotations

from typing import Set, Tuple

import pytest

from flexkv.common.dist_reuse import (
    SharingDomainKey,
    SharingDomainNamespace,
    build_sharing_domain_handles,
    graph_needs_gpu_clear,
)


MODEL_ID = "phase2-simplified"


def _sd(pp_rank=0, pp_size=1, tpn_idx=0, tpn_count=1,
        is_nsa=False) -> SharingDomainKey:
    """Build an SD key with the simplified schema (no cp_* fields)."""
    return SharingDomainKey(
        model_id=MODEL_ID,
        pp_rank=pp_rank, pp_size=pp_size,
        tp_node_idx=tpn_idx, tp_node_count=tpn_count,
        is_nsa=is_nsa,
    )


# ===========================================================================
# §4.10.1 row: PP=2, tp_node_count=1 → 2 SDs (current deployment case A)
# ===========================================================================
class TestPhase2PPOnly:
    SELF_SD = _sd(pp_rank=0, pp_size=2)
    EXPECTED_SD_COUNT = 2

    def test_enumerate_yields_2_distinct_sds(self):
        peers = self.SELF_SD.enumerate_peers()
        assert len(peers) == self.EXPECTED_SD_COUNT
        # Both (pp=0) and (pp=1) present, each with tp_node_idx=0.
        pp_ranks: Set[int] = set()
        for p in peers:
            pp_ranks.add(p.pp_rank)
            assert p.tp_node_idx == 0 and p.tp_node_count == 1
            assert p.pp_size == 2
        assert pp_ranks == {0, 1}

    def test_exactly_one_master(self):
        peers = self.SELF_SD.enumerate_peers()
        masters = [p for p in peers if p.is_master()]
        assert len(masters) == 1
        assert masters[0].pp_rank == 0

    def test_non_master_peer_needs_gpu_clear(self):
        peers = [p for p in self.SELF_SD.enumerate_peers() if p != self.SELF_SD]
        assert len(peers) == 1
        # Peer differs in pp_rank → clear required.
        assert graph_needs_gpu_clear(self.SELF_SD, peers[0]) is True


# ===========================================================================
# §4.10.1 row: PP=1, tp_node_count=2 → 2 SDs (current deployment case B)
# ===========================================================================
class TestPhase2CrossNodeTPOnly:
    SELF_SD = _sd(tpn_idx=0, tpn_count=2)
    EXPECTED_SD_COUNT = 2

    def test_enumerate_yields_2_distinct_sds(self):
        peers = self.SELF_SD.enumerate_peers()
        assert len(peers) == self.EXPECTED_SD_COUNT
        tp_nodes: Set[int] = set()
        for p in peers:
            tp_nodes.add(p.tp_node_idx)
            assert p.pp_rank == 0 and p.pp_size == 1
            assert p.tp_node_count == 2
        assert tp_nodes == {0, 1}

    def test_non_master_peer_does_not_need_gpu_clear(self):
        """Cross-TP-node only differs in the head shard, not in slot_mapping,
        so the graph can be forwarded as-is without a GPU clear."""
        peers = [p for p in self.SELF_SD.enumerate_peers() if p != self.SELF_SD]
        assert len(peers) == 1
        assert graph_needs_gpu_clear(self.SELF_SD, peers[0]) is False


# ===========================================================================
# §4.10.1 future-extension row: PP=2 × tp_node_count=2 → 4 SDs (full Phase 3)
# ===========================================================================
class TestPhase2MaxConfig:
    SELF_SD = _sd(pp_size=2, tpn_count=2)
    EXPECTED_SD_COUNT = 2 * 2  # 4

    def test_enumerate_yields_4_distinct_sds(self):
        peers = self.SELF_SD.enumerate_peers()
        assert len(peers) == self.EXPECTED_SD_COUNT
        serialized = {p.serialize() for p in peers}
        assert len(serialized) == self.EXPECTED_SD_COUNT
        # Every (pp, tpn) pair present exactly once.
        pairs: Set[Tuple[int, int]] = set()
        for p in peers:
            pairs.add((p.pp_rank, p.tp_node_idx))
        assert pairs == {(pp, tpn) for pp in range(2) for tpn in range(2)}

    def test_clear_decision_counts(self):
        """In a 4-SD instance with self at (pp=0, tpn=0):

            peer (pp=0, tpn=1) → TP-only differ → no clear
            peer (pp=1, tpn=0) → pp differs     → clear
            peer (pp=1, tpn=1) → pp differs     → clear

        So 2 of 3 non-self peers require gpu-clear.
        """
        clear_count = 0
        no_clear_count = 0
        for peer in self.SELF_SD.enumerate_peers():
            if peer == self.SELF_SD:
                continue
            if graph_needs_gpu_clear(self.SELF_SD, peer):
                clear_count += 1
            else:
                no_clear_count += 1
        total_peers = self.EXPECTED_SD_COUNT - 1  # 3 non-self
        assert clear_count + no_clear_count == total_peers
        assert no_clear_count == 1, "only the TP-only-differ peer avoids gpu clear"
        assert clear_count == 2

    def test_namespace_prefixes_are_pairwise_disjoint(self):
        """No SD's Redis key prefix is a prefix of another's (would leak SCANs)."""
        peers = self.SELF_SD.enumerate_peers()
        prefixes = [SharingDomainNamespace(p).prefix for p in peers]
        # All unique.
        assert len(set(prefixes)) == len(prefixes)
        # None is a proper prefix of another (followed by ':').
        for i, a in enumerate(prefixes):
            for j, b in enumerate(prefixes):
                if i == j:
                    continue
                assert not b.startswith(a + ":"), (
                    f"prefix '{a}' is a proper prefix of '{b}' — Redis SCANs would leak"
                )


# ===========================================================================
# build_sharing_domain_handles under the current-deployment cap (2 SDs)
# ===========================================================================
class TestBuildHandlesCurrentDeployment:
    def test_pp2_one_remote_handle(self):
        self_sd = _sd(pp_size=2)
        peer = _sd(pp_rank=1, pp_size=2)
        specs = build_sharing_domain_handles(
            self_sd=self_sd,
            remote_endpoints_by_sd={
                peer.serialize(): _FakeEndpoint(
                    ip="10.0.0.1",
                    gpu_register_port="5000",
                    command_port="5001",
                    result_port="5002",
                ),
            },
        )
        assert len(specs) == 2
        assert specs[0].mode == "process" and specs[0].sd_key.is_master()
        assert specs[1].mode == "remote" and specs[1].sd_key == peer


# ===========================================================================
# Supporting test doubles
# ===========================================================================
class _FakeEndpoint:
    """Duck-type of flexkv.common.config.RemoteEndpoint, sufficient for
    build_sharing_domain_handles to treat us as a valid endpoint."""

    def __init__(self, ip, gpu_register_port, command_port, result_port):
        self.ip = ip
        self.gpu_register_port = gpu_register_port
        self.command_port = command_port
        self.result_port = result_port
