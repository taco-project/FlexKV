"""Phase D-3 unit tests
(proposal_unify_with_graph_dispatch_2026-05-15.md §6.4).

Covers two pieces of functionality introduced in D-3:

1. ``_filter_graph_inplace_by_target_node_ids`` — module-level
   utility extracted out of ``TransferManagerOnRemote`` so the
   Master's in-proc / inter-proc handles can use it too.  In
   particular tests:

   * legacy graphs (no ``target_node_ids`` set) are untouched
   * ops not addressed to ``self_nid`` are dropped
   * dropped ops are removed from kept ops' ``predecessors`` /
     ``successors`` so the graph engine does not deadlock on a
     filtered-away dependency

2. ``GlobalCacheEngine._maybe_attach_multi_sd_peerh2h_ops`` — fans the
   GET-path PEERH2H op out to one clone per peer SD, with
   ``target_node_ids`` and ``src_block_node_ids`` derived from the
   ``AggregateRadixTree``'s ``ready_sds`` map.  Tests:

   * single-SD instance / dist_reuse-off  → no clones, master op
     untouched (legacy bit-identical behaviour)
   * multi-SD instance with all peer SDs ready  → master op stamped
     with ``target_node_ids=[self_node_id]`` + one clone per peer SD
   * peer SD missing from ``ready_sds``  → that SD is silently
     skipped (gate will reject the GET downstream)

The tests are pure Python: they construct a real
``TransferOpGraph`` and a stub ``GlobalCacheEngine`` instance with
just the attributes the helper touches.  No GPU / mooncake /
TransferEngine startup is needed.
"""
from __future__ import absolute_import

from typing import Dict, Iterable, List, Optional

import numpy as np
import pytest

from flexkv.common.transfer import (
    TransferOp,
    TransferOpGraph,
    TransferType,
)
from flexkv.transfer_manager import _filter_graph_inplace_by_target_node_ids


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _mk_op(
    transfer_type: TransferType,
    *,
    src=(0, 1),
    dst=(2, 3),
    target_node_ids: Optional[List[int]] = None,
) -> TransferOp:
    # NOTE: ``layer_id`` / ``layer_granularity`` were removed from the
    # base ``TransferOp`` dataclass (commit abbf299 + RankInfo refactor)
    # — those fields now live only on ``LayerwiseTransferOp``.  D-3
    # filter-graph behaviour is independent of layer info, so we simply
    # don't supply them here.
    return TransferOp(
        graph_id=-1,  # add_transfer_op stamps the real graph_id
        transfer_type=transfer_type,
        src_block_ids=np.array(src, dtype=np.int64),
        dst_block_ids=np.array(dst, dtype=np.int64),
        target_node_ids=list(target_node_ids) if target_node_ids else None,
    )


# ===========================================================================
# Section 1: filter utility
# ===========================================================================
class TestFilterUtility:
    def test_legacy_graph_no_target_node_ids_unchanged(self):
        """Op with target_node_ids=None must always be kept, regardless
        of self_nid.  This is the legacy single-SD / cross-machine TP
        contract — Phase D-3 must not regress it."""
        g = TransferOpGraph()
        a = _mk_op(TransferType.D2H)              # no target → legacy
        b = _mk_op(TransferType.H2D)              # no target → legacy
        g.add_transfer_op(a)
        g.add_transfer_op(b)
        g.add_dependency(b.op_id, a.op_id)        # b depends on a

        dropped = _filter_graph_inplace_by_target_node_ids(g, self_nid=42)

        assert dropped == 0
        assert g.num_ops == 2
        assert b.op_id in [op.op_id for op in g._op_map.values()]
        assert a.op_id in [op.op_id for op in g._op_map.values()]
        assert a.op_id in b.predecessors

    def test_self_nid_negative_keeps_everything(self):
        """Before dist_reuse bootstrap finishes, self_nid is -1 and the
        helper must behave like a no-op so legacy paths run as before.
        """
        g = TransferOpGraph()
        a = _mk_op(TransferType.PEERH2H, target_node_ids=[7])
        g.add_transfer_op(a)
        dropped = _filter_graph_inplace_by_target_node_ids(g, self_nid=-1)
        assert dropped == 0
        assert g.num_ops == 1

    def test_drops_ops_not_addressed_to_self(self):
        """Master's own op (target=self) is kept; peer-SD clone (target
        != self) is dropped."""
        g = TransferOpGraph()
        master_op = _mk_op(TransferType.PEERH2H, target_node_ids=[10])
        peer_op = _mk_op(TransferType.PEERH2H, target_node_ids=[11])
        g.add_transfer_op(master_op)
        g.add_transfer_op(peer_op)

        dropped = _filter_graph_inplace_by_target_node_ids(g, self_nid=10)

        assert dropped == 1
        assert g.num_ops == 1
        assert master_op.op_id in g._op_map
        assert peer_op.op_id not in g._op_map

    def test_dependency_repair_dropped_predecessor_does_not_deadlock(self):
        """If the dropped op was a predecessor of a kept op, the kept op
        must NOT carry the dropped op_id in its predecessors set —
        otherwise ``take_ready_ops`` would never schedule it.
        """
        g = TransferOpGraph()
        # master op_peerh2h (kept on self=10)
        master_peerh2h = _mk_op(
            TransferType.PEERH2H, target_node_ids=[10],
        )
        # peer-SD clone (dropped on self=10)
        peer_clone = _mk_op(
            TransferType.PEERH2H, target_node_ids=[11],
        )
        # op_h2d depends on BOTH (legacy + Phase D-3 dependency)
        op_h2d = _mk_op(TransferType.H2D)         # target=None → kept
        g.add_transfer_op(master_peerh2h)
        g.add_transfer_op(peer_clone)
        g.add_transfer_op(op_h2d)
        g.add_dependency(op_h2d.op_id, master_peerh2h.op_id)
        g.add_dependency(op_h2d.op_id, peer_clone.op_id)
        assert op_h2d.predecessors == {
            master_peerh2h.op_id, peer_clone.op_id,
        }

        dropped = _filter_graph_inplace_by_target_node_ids(g, self_nid=10)

        assert dropped == 1
        # Surviving op_h2d must still wait on master_peerh2h, but the
        # filtered-away peer_clone must be gone from its predecessors.
        assert op_h2d.predecessors == {master_peerh2h.op_id}
        # Conversely master_peerh2h.successors must lose op_h2d? No —
        # successors only loses references to *dropped* ops; op_h2d
        # is kept.  Sanity check.
        assert op_h2d.op_id in master_peerh2h.successors

    def test_dependency_repair_clears_orphan_successors(self):
        """If we drop op X that listed kept op Y in its successors, the
        helper should not leave a dangling ref to a dropped op_id in
        any kept op's successors set either (defensive symmetry — even
        if no current code path relies on it)."""
        g = TransferOpGraph()
        kept = _mk_op(TransferType.D2H)            # target=None → kept
        dropped_op = _mk_op(
            TransferType.PEERH2H, target_node_ids=[99],
        )
        g.add_transfer_op(kept)
        g.add_transfer_op(dropped_op)
        # kept depends on dropped_op (so kept.predecessors has it,
        # dropped_op.successors has kept)
        g.add_dependency(kept.op_id, dropped_op.op_id)

        _filter_graph_inplace_by_target_node_ids(g, self_nid=10)

        # No reference to the dropped op_id anywhere in kept op's sets.
        assert dropped_op.op_id not in kept.predecessors
        assert dropped_op.op_id not in kept.successors

    def test_kept_op_with_only_dropped_predecessors_becomes_ready(self):
        """If filtering empties an op's predecessors, the op should be
        moved back to ``_ready_ops`` so the graph engine schedules it.
        """
        g = TransferOpGraph()
        peer_a = _mk_op(TransferType.PEERH2H, target_node_ids=[11])
        peer_b = _mk_op(TransferType.PEERH2H, target_node_ids=[12])
        local_h2d = _mk_op(TransferType.H2D)       # target=None → kept

        g.add_transfer_op(peer_a)
        g.add_transfer_op(peer_b)
        g.add_transfer_op(local_h2d)
        g.add_dependency(local_h2d.op_id, peer_a.op_id)
        g.add_dependency(local_h2d.op_id, peer_b.op_id)
        # add_dependency moves successor out of _ready_ops
        assert local_h2d.op_id not in g._ready_ops

        _filter_graph_inplace_by_target_node_ids(g, self_nid=10)

        # All predecessors gone → local_h2d back in _ready_ops so the
        # engine actually schedules it.
        assert local_h2d.predecessors == set()
        assert local_h2d.op_id in g._ready_ops


# ===========================================================================
# Section 2: _maybe_attach_multi_sd_peerh2h_ops on GlobalCacheEngine
# ===========================================================================
#
# We don't construct a real GlobalCacheEngine (heavy: needs a torch
# device / RedisMeta / etc.).  We instead test the unbound method
# against a duck-typed stub that exposes only the attributes the
# helper reaches for: ``_master_coord`` (a ``MasterCoordinator``-like
# object).
#
# This keeps the test pure-Python while pinning the behavioural
# contract.

class _StubSequenceMeta:
    """Just enough of ``SequenceMeta`` for the helper.

    Helper paths used:

    * ``sequence_meta.gen_hashes()`` — must be safe to call
    * ``sequence_meta.block_hashes`` — np.ndarray[int64]
    """
    def __init__(self, hashes: List[int]):
        self.block_hashes = np.array(hashes, dtype=np.int64)

    def gen_hashes(self) -> None:
        # Hashes are pre-computed in the constructor.
        return


class _StubReadyEntry:
    def __init__(self, ready_sds: Dict[str, int]):
        self.ready_sds = dict(ready_sds)


class _StubMasterCoord:
    """Mimics enough of ``MasterCoordinator`` for the helper."""

    def __init__(
        self,
        *,
        self_sd_str: str,
        sd_to_nid: Dict[str, int],
        ready_sds: Optional[Dict[int, Dict[str, int]]] = None,
    ):
        # ``self_sd`` exposes ``serialize()`` like the real key.
        class _SD:
            def __init__(self, s):
                self._s = s
            def serialize(self) -> str:
                return self._s
        self.self_sd = _SD(self_sd_str)
        self._sd_to_nid = dict(sd_to_nid)
        # prefix_hash → ready_sds map (per prefix)
        self._ready_sds_per_prefix = ready_sds or {}

    def get_sd_to_nid_map(self) -> Dict[str, int]:
        return dict(self._sd_to_nid)

    def match_fully_ready(self, prefix_hash: int):
        rs = self._ready_sds_per_prefix.get(int(prefix_hash))
        if rs is None:
            return None
        return _StubReadyEntry(rs)


class _StubGlobalCacheEngine:
    """The helper is bound to ``self._master_coord`` only."""
    def __init__(self, master_coord):
        self._master_coord = master_coord


def _bind_helper():
    """Pull the helper off the real class so we can call it bound to
    a stub instance — keeps the production code under test honest."""
    from flexkv.cache.cache_engine import GlobalCacheEngine
    return GlobalCacheEngine._maybe_attach_multi_sd_peerh2h_ops


class TestMaybeAttachMultiSDPeerH2H:
    def _make_graph_with_master_op(self):
        g = TransferOpGraph()
        master_peerh2h = TransferOp(
            graph_id=g.graph_id,
            transfer_type=TransferType.PEERH2H,
            src_block_ids=np.array([10, 11, 12], dtype=np.int64),
            dst_block_ids=np.array([20, 21, 22], dtype=np.int64),
            remote_node_ids=np.array([7, 7, 7], dtype=np.int64),
            src_block_node_ids=np.array([7, 7, 7], dtype=np.int64),
        )
        g.add_transfer_op(master_peerh2h)
        return g, master_peerh2h

    def test_no_master_coord_legacy_passthrough(self):
        helper = _bind_helper()
        g, master_op = self._make_graph_with_master_op()
        seq = _StubSequenceMeta([0xAA])
        stub = _StubGlobalCacheEngine(master_coord=None)

        clones = helper(
            stub,
            transfer_graph=g,
            op_peerh2h=master_op,
            sequence_meta=seq,
            prefix_terminal_block_idx=0,
        )

        assert clones == []
        assert master_op.target_node_ids is None  # untouched
        assert g.num_ops == 1                     # no clones added

    def test_single_sd_instance_no_clones(self):
        helper = _bind_helper()
        g, master_op = self._make_graph_with_master_op()
        seq = _StubSequenceMeta([0xAA])
        coord = _StubMasterCoord(
            self_sd_str="sd-master",
            sd_to_nid={"sd-master": 100},
        )
        stub = _StubGlobalCacheEngine(master_coord=coord)

        clones = helper(
            stub,
            transfer_graph=g,
            op_peerh2h=master_op,
            sequence_meta=seq,
            prefix_terminal_block_idx=0,
        )

        assert clones == []
        # Critical: the master op stays untouched in the single-SD case
        # (otherwise the in-proc handle filter would drop it spuriously).
        assert master_op.target_node_ids is None
        assert g.num_ops == 1

    def test_bootstrap_not_finished_no_clones(self):
        """``get_sd_to_nid_map`` returns {} until the master's own
        node_id has been registered.  The helper must short-circuit and
        leave the master op untouched."""
        helper = _bind_helper()
        g, master_op = self._make_graph_with_master_op()
        seq = _StubSequenceMeta([0xAA])
        coord = _StubMasterCoord(
            self_sd_str="sd-master",
            sd_to_nid={},  # bootstrap not done yet
        )
        stub = _StubGlobalCacheEngine(master_coord=coord)

        clones = helper(
            stub,
            transfer_graph=g,
            op_peerh2h=master_op,
            sequence_meta=seq,
            prefix_terminal_block_idx=0,
        )

        assert clones == []
        assert master_op.target_node_ids is None
        assert g.num_ops == 1

    def test_multi_sd_all_ready_clones_per_peer(self):
        helper = _bind_helper()
        g, master_op = self._make_graph_with_master_op()
        prefix_hash = 0xCAFE
        seq = _StubSequenceMeta([prefix_hash])

        # 3 SDs total: master + 2 peers.  ``ready_sds`` records each
        # SD's contributing peer's node_id (note: master SD is owned
        # locally so node_id == master's own node_id).
        coord = _StubMasterCoord(
            self_sd_str="sd-master",
            sd_to_nid={
                "sd-master": 100,
                "sd-peer-A": 200,
                "sd-peer-B": 300,
            },
            ready_sds={
                prefix_hash: {
                    "sd-master": 100,
                    "sd-peer-A": 555,    # peer instance's nid on SD-A
                    "sd-peer-B": 666,    # peer instance's nid on SD-B
                },
            },
        )
        stub = _StubGlobalCacheEngine(master_coord=coord)

        clones = helper(
            stub,
            transfer_graph=g,
            op_peerh2h=master_op,
            sequence_meta=seq,
            prefix_terminal_block_idx=0,
        )

        # Master op stamped to itself.
        assert master_op.target_node_ids == [100]
        # Two clones added.
        assert len(clones) == 2
        # Total ops in graph: master + 2 clones = 3.
        assert g.num_ops == 3

        clones_by_target = {tuple(c.target_node_ids): c for c in clones}
        assert (200,) in clones_by_target
        assert (300,) in clones_by_target

        clone_a = clones_by_target[(200,)]
        clone_b = clones_by_target[(300,)]

        # src/dst block_ids mirror the master op (mirror assumption).
        np.testing.assert_array_equal(
            clone_a.src_block_ids, master_op.src_block_ids,
        )
        np.testing.assert_array_equal(
            clone_a.dst_block_ids, master_op.dst_block_ids,
        )
        # src_block_node_ids point to the peer instance's nid on each
        # peer SD (pulled from ready_sds).
        np.testing.assert_array_equal(
            clone_a.src_block_node_ids,
            np.array([555, 555, 555], dtype=np.int64),
        )
        np.testing.assert_array_equal(
            clone_b.src_block_node_ids,
            np.array([666, 666, 666], dtype=np.int64),
        )
        # transfer_type preserved.
        assert clone_a.transfer_type == TransferType.PEERH2H
        assert clone_b.transfer_type == TransferType.PEERH2H

    def test_multi_sd_partial_ready_skips_unacked_peers(self):
        """If a peer SD has not yet acked (missing from ``ready_sds``
        or value < 0), the helper skips it.  The downstream gate
        (``_sharing_domain_gate_get``) then rejects the GET when
        ``match_fully_ready`` returns None on the same prefix.

        This test only validates the per-SD skip behaviour; the gate
        is exercised by ``test_cache_engine_dist_reuse_gate.py``.
        """
        helper = _bind_helper()
        g, master_op = self._make_graph_with_master_op()
        prefix_hash = 0xC0DE
        seq = _StubSequenceMeta([prefix_hash])

        coord = _StubMasterCoord(
            self_sd_str="sd-master",
            sd_to_nid={
                "sd-master": 100,
                "sd-peer-A": 200,
                "sd-peer-B": 300,
            },
            ready_sds={
                prefix_hash: {
                    "sd-master": 100,
                    "sd-peer-A": 555,
                    # sd-peer-B intentionally missing → not yet acked
                },
            },
        )
        stub = _StubGlobalCacheEngine(master_coord=coord)

        clones = helper(
            stub,
            transfer_graph=g,
            op_peerh2h=master_op,
            sequence_meta=seq,
            prefix_terminal_block_idx=0,
        )

        # Only peer A has a clone; peer B is silently skipped.
        assert len(clones) == 1
        assert clones[0].target_node_ids == [200]
        # Master op still tagged (2-of-3 SDs visible is enough — gate
        # decides the GET fate elsewhere).
        assert master_op.target_node_ids == [100]
        assert g.num_ops == 2

    def test_multi_sd_no_aggregate_entry_no_clones(self):
        """``match_fully_ready`` returns None (gate will reject) →
        helper bails out without mutating the graph.  Avoids
        polluting the graph with clones that would never get a
        contributor."""
        helper = _bind_helper()
        g, master_op = self._make_graph_with_master_op()
        seq = _StubSequenceMeta([0xDEAD])
        coord = _StubMasterCoord(
            self_sd_str="sd-master",
            sd_to_nid={
                "sd-master": 100,
                "sd-peer-A": 200,
            },
            ready_sds={},  # no entry for any prefix
        )
        stub = _StubGlobalCacheEngine(master_coord=coord)

        clones = helper(
            stub,
            transfer_graph=g,
            op_peerh2h=master_op,
            sequence_meta=seq,
            prefix_terminal_block_idx=0,
        )

        assert clones == []
        # Master op left alone — gate will reject the GET so no
        # contradictory tagging is needed.
        assert master_op.target_node_ids is None
        assert g.num_ops == 1


# ===========================================================================
# Section 3: end-to-end small fixture — filter pipeline preserves
# the per-SD invariant for a realistic GET graph
# ===========================================================================
class TestEndToEndFilterPipeline:
    """Build a graph that mimics what ``_get_impl_local`` produces in
    the multi-SD reuse case, run it through the filter on each SD's
    handle, and verify each SD ends up with exactly the ops it should
    execute and a ready ``op_h2d`` whose predecessors collapse to the
    SD's own peerh2h op."""

    def _build_get_graph(self):
        g = TransferOpGraph()
        # master op_peerh2h, target=master_nid
        master_peerh2h = TransferOp(
            graph_id=g.graph_id,
            transfer_type=TransferType.PEERH2H,
            src_block_ids=np.array([0, 1], dtype=np.int64),
            dst_block_ids=np.array([10, 11], dtype=np.int64),
            target_node_ids=[100],
        )
        # peer-A clone, target=peer_a_nid
        peer_a_clone = TransferOp(
            graph_id=g.graph_id,
            transfer_type=TransferType.PEERH2H,
            src_block_ids=np.array([0, 1], dtype=np.int64),
            dst_block_ids=np.array([10, 11], dtype=np.int64),
            target_node_ids=[200],
        )
        # peer-B clone
        peer_b_clone = TransferOp(
            graph_id=g.graph_id,
            transfer_type=TransferType.PEERH2H,
            src_block_ids=np.array([0, 1], dtype=np.int64),
            dst_block_ids=np.array([10, 11], dtype=np.int64),
            target_node_ids=[300],
        )
        # op_h2d — no SD tag, every handle keeps it.
        op_h2d = TransferOp(
            graph_id=g.graph_id,
            transfer_type=TransferType.H2D,
            src_block_ids=np.array([10, 11], dtype=np.int64),
            dst_block_ids=np.array([1000, 1001], dtype=np.int64),
        )
        g.add_transfer_op(master_peerh2h)
        g.add_transfer_op(peer_a_clone)
        g.add_transfer_op(peer_b_clone)
        g.add_transfer_op(op_h2d)
        # H2D depends on every PEERH2H (master + each peer clone)
        g.add_dependency(op_h2d.op_id, master_peerh2h.op_id)
        g.add_dependency(op_h2d.op_id, peer_a_clone.op_id)
        g.add_dependency(op_h2d.op_id, peer_b_clone.op_id)
        return g, master_peerh2h, peer_a_clone, peer_b_clone, op_h2d

    def _deep_copy_graph(self, g):
        import pickle
        return pickle.loads(pickle.dumps(g))

    def test_master_handle_keeps_only_self_peerh2h_and_h2d(self):
        g, master_op, peer_a, peer_b, h2d = self._build_get_graph()
        master_g = self._deep_copy_graph(g)

        dropped = _filter_graph_inplace_by_target_node_ids(
            master_g, self_nid=100,
        )
        assert dropped == 2

        kept_ids = set(master_g._op_map.keys())
        assert master_op.op_id in kept_ids
        assert h2d.op_id in kept_ids
        assert peer_a.op_id not in kept_ids
        assert peer_b.op_id not in kept_ids

        # h2d must depend ONLY on master's own peerh2h after filtering.
        master_h2d = master_g._op_map[h2d.op_id]
        assert master_h2d.predecessors == {master_op.op_id}

    def test_peer_a_handle_keeps_only_clone_and_h2d(self):
        g, master_op, peer_a, peer_b, h2d = self._build_get_graph()
        peer_a_g = self._deep_copy_graph(g)

        dropped = _filter_graph_inplace_by_target_node_ids(
            peer_a_g, self_nid=200,
        )
        assert dropped == 2

        kept_ids = set(peer_a_g._op_map.keys())
        assert peer_a.op_id in kept_ids
        assert h2d.op_id in kept_ids
        assert master_op.op_id not in kept_ids
        assert peer_b.op_id not in kept_ids

        peer_h2d = peer_a_g._op_map[h2d.op_id]
        assert peer_h2d.predecessors == {peer_a.op_id}

    def test_peer_b_handle_keeps_only_clone_and_h2d(self):
        g, master_op, peer_a, peer_b, h2d = self._build_get_graph()
        peer_b_g = self._deep_copy_graph(g)

        dropped = _filter_graph_inplace_by_target_node_ids(
            peer_b_g, self_nid=300,
        )
        assert dropped == 2
        peer_h2d = peer_b_g._op_map[h2d.op_id]
        assert peer_h2d.predecessors == {peer_b.op_id}


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
