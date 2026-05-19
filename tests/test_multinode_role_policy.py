"""§2.4 — Multi-node role decision policy tests.

The connector-side split of ``is_multinode_tp`` vs. ``is_multinode_cp``
is currently *not* plumbed into ``flexkv_connector.py`` (sglang/) —
doing that requires a two-machine GPU setup to verify.  Until then we
pin the decision table at the policy-function level so that when the
actual connector swap lands, it already has a stable, tested contract
to call into.

These tests are torch-free by design: pure logic over ``RankTopology``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flexkv.integration.multinode_policy import (  # noqa: E402
    RankTopology,
    RemoteProcessRole,
    decide_remote_role,
    is_sync_leader,
)


# ---------------------------------------------------------------------------
# Trivial validation
# ---------------------------------------------------------------------------
class TestValidation:
    def test_nnodes_must_be_positive(self):
        with pytest.raises(ValueError):
            decide_remote_role(RankTopology(
                nnodes=0, node_rank=0, local_rank=0,
                is_multinode_tp=False, is_multinode_cp=False,
            ))

    def test_node_rank_in_range(self):
        with pytest.raises(ValueError):
            decide_remote_role(RankTopology(
                nnodes=2, node_rank=5, local_rank=0,
                is_multinode_tp=True, is_multinode_cp=False,
            ))

    def test_local_rank_non_negative(self):
        with pytest.raises(ValueError):
            decide_remote_role(RankTopology(
                nnodes=2, node_rank=1, local_rank=-1,
                is_multinode_tp=True, is_multinode_cp=False,
            ))


# ---------------------------------------------------------------------------
# Single-node instance → NO_REMOTE regardless of any flag.
# ---------------------------------------------------------------------------
class TestSingleNode:
    @pytest.mark.parametrize("tp,cp", [(False, False), (True, False),
                                        (False, True), (True, True)])
    def test_single_node_never_spawns_remote(self, tp, cp):
        topo = RankTopology(
            nnodes=1, node_rank=0, local_rank=0,
            is_multinode_tp=tp, is_multinode_cp=cp,
        )
        assert decide_remote_role(topo) is RemoteProcessRole.NO_REMOTE


# ---------------------------------------------------------------------------
# Master node (node_rank == 0) never spawns a remote itself.
# ---------------------------------------------------------------------------
class TestMasterNode:
    def test_master_with_multinode_tp_is_master(self):
        topo = RankTopology(
            nnodes=2, node_rank=0, local_rank=0,
            is_multinode_tp=True, is_multinode_cp=False,
        )
        assert decide_remote_role(topo) is RemoteProcessRole.MASTER

    def test_master_with_multinode_cp_is_master(self):
        topo = RankTopology(
            nnodes=2, node_rank=0, local_rank=0,
            is_multinode_tp=False, is_multinode_cp=True,
        )
        assert decide_remote_role(topo) is RemoteProcessRole.MASTER

    def test_master_with_both_flags_is_master(self):
        topo = RankTopology(
            nnodes=2, node_rank=0, local_rank=0,
            is_multinode_tp=True, is_multinode_cp=True,
        )
        assert decide_remote_role(topo) is RemoteProcessRole.MASTER

    def test_master_with_nothing_crossing_nodes_is_no_remote(self):
        """Multi-node deployment but THIS instance is single-node
        (only DP crosses; each DP instance stays on one node)."""
        topo = RankTopology(
            nnodes=2, node_rank=0, local_rank=0,
            is_multinode_tp=False, is_multinode_cp=False,
        )
        assert decide_remote_role(topo) is RemoteProcessRole.NO_REMOTE


# ---------------------------------------------------------------------------
# Off-master nodes — the interesting routing table.
# ---------------------------------------------------------------------------
class TestOffMasterRouting:
    def test_multinode_tp_only_is_full_sd_remote(self):
        topo = RankTopology(
            nnodes=2, node_rank=1, local_rank=0,
            is_multinode_tp=True, is_multinode_cp=False,
        )
        assert decide_remote_role(topo) is RemoteProcessRole.SD_REMOTE_FULL

    def test_multinode_cp_only_is_cp_registration_stub(self):
        topo = RankTopology(
            nnodes=2, node_rank=1, local_rank=0,
            is_multinode_tp=False, is_multinode_cp=True,
        )
        assert decide_remote_role(topo) is RemoteProcessRole.CP_PEER_REGISTRATION_ONLY

    def test_multinode_tp_wins_over_cp(self):
        """When BOTH flags are True on an off-master rank, TP takes
        priority — TP-split SDs cannot be served by a CP-only stub."""
        topo = RankTopology(
            nnodes=2, node_rank=1, local_rank=0,
            is_multinode_tp=True, is_multinode_cp=True,
        )
        assert decide_remote_role(topo) is RemoteProcessRole.SD_REMOTE_FULL

    def test_neither_tp_nor_cp_multinode_still_spawns_full_remote_today(self):
        """Legacy bug-compat: today's connector treats any
        ``nnodes>1 and node_rank>0 and local_rank==0`` case as
        ``SD_REMOTE_FULL`` (PP crossing nodes uses this path).  We
        preserve that during the migration; the TODO in
        multinode_policy.py tracks the eventual cleanup.
        """
        topo = RankTopology(
            nnodes=2, node_rank=1, local_rank=0,
            is_multinode_tp=False, is_multinode_cp=False,
        )
        assert decide_remote_role(topo) is RemoteProcessRole.SD_REMOTE_FULL


# ---------------------------------------------------------------------------
# Sync-leader helper.
# ---------------------------------------------------------------------------
class TestSyncLeader:
    def test_default_rule_is_local_and_node_rank_zero(self):
        topo = RankTopology(
            nnodes=2, node_rank=0, local_rank=0,
            is_multinode_tp=True, is_multinode_cp=False,
        )
        assert is_sync_leader(topo) is True

        topo2 = RankTopology(
            nnodes=2, node_rank=1, local_rank=0,
            is_multinode_tp=True, is_multinode_cp=False,
        )
        assert is_sync_leader(topo2) is False

        topo3 = RankTopology(
            nnodes=2, node_rank=0, local_rank=1,
            is_multinode_tp=True, is_multinode_cp=False,
        )
        assert is_sync_leader(topo3) is False

    def test_explicit_hint_overrides_default(self):
        """If the caller hands us ``is_sync_leader=True`` (e.g. coming
        from sglang's own group metadata), respect it even if the
        default heuristic would say otherwise."""
        topo = RankTopology(
            nnodes=2, node_rank=1, local_rank=7,
            is_multinode_tp=True, is_multinode_cp=False,
            is_sync_leader=True,
        )
        assert is_sync_leader(topo) is True

        topo2 = RankTopology(
            nnodes=2, node_rank=0, local_rank=0,
            is_multinode_tp=True, is_multinode_cp=False,
            is_sync_leader=False,
        )
        assert is_sync_leader(topo2) is False
