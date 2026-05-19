"""Python reference implementation of the single-node matching constraint
(design doc §4.7.1.4, implemented in C++ at
``csrc/dist/distributed_radix_tree.cpp::match_prefix`` lines 579-686).

This test file does **not** exercise the C++ code (that needs a GPU/CUDA
build to link).  Instead it re-implements the *core invariant* of the
constraint in pure Python and exhaustively checks the algorithm's
externally-visible behaviour on synthetic radix-node sequences:

    * locks onto the FIRST block's ``node_id`` as the canonical
      ``matched_node_id`` for the whole match;
    * as soon as a block with a different ``node_id`` is encountered,
      the match truncates — that block and every subsequent block (in
      the same or later nodes) is NOT included;
    * the truncation happens at **block granularity**, not node
      granularity — a node whose first 3 blocks belong to the locked
      node_id but whose 4th block belongs to a different one contributes
      only 3 blocks to the result;
    * ``matched_node_id`` stays ``-1`` iff the query didn't match any
      block at all (empty result).

These are **exactly** the guarantees the downstream Master needs in
order to issue a coordinated GET to a single Remote peer — see design
doc §4.4.1 / §4.12.2 (1).  Any future C++ refactor that changes the
algorithm must keep these guarantees; this reference implementation
serves as the executable spec.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pytest


# ---------------------------------------------------------------------------
# Minimal radix-node shape (just enough to drive the algorithm).
# ---------------------------------------------------------------------------
@dataclass
class _Node:
    """One radix node — a run of (hash, physical_block, block_node_id) triples."""
    hashes: List[int]
    pbs: List[int]
    bnis: List[int]  # must be len == len(hashes)

    def __post_init__(self) -> None:
        assert len(self.hashes) == len(self.pbs) == len(self.bnis), (
            f"inconsistent node: {len(self.hashes)=} {len(self.pbs)=} {len(self.bnis)=}"
        )

    def size(self) -> int:
        return len(self.hashes)


@dataclass
class _MatchResult:
    prefix_blocks_num: int
    physical_blocks: List[int]
    matched_node_id: int  # -1 if no match


# ---------------------------------------------------------------------------
# Python reference implementation of match_prefix.
# Mirrors csrc/dist/distributed_radix_tree.cpp:579-686 line-for-line
# (minus the lease-time / renew-queue machinery which is irrelevant here).
# ---------------------------------------------------------------------------
def match_prefix_ref(
    nodes: List[_Node],
    query_hashes: List[int],
) -> _MatchResult:
    """Reference Python implementation of the single-node matching rule.

    ``nodes`` is the flattened path from the root down to the deepest
    reachable node for the query — i.e. at this point in the algorithm
    we've already resolved which *sequence* of radix nodes the query
    would descend.  (In the real C++ code this happens dynamically via
    ``lookup_child``; for testing we feed the resolved path directly so
    we can focus on the single-node constraint itself.)
    """
    prefix_blocks_num = 0
    pb_out: List[int] = []
    matched_node_id = -1  # lock-on-first-block sentinel

    num_query = len(query_hashes)

    for node in nodes:
        if prefix_blocks_num >= num_query:
            break
        node_size = node.size()
        remaining = num_query - prefix_blocks_num
        blocks_to_check = min(node_size, remaining)

        # Step 1: how many blocks at the head of this node match the query?
        matched = 0
        for i in range(blocks_to_check):
            if node.hashes[i] == query_hashes[prefix_blocks_num + i]:
                matched += 1
            else:
                break

        # Step 2: among those ``matched`` blocks, how many can actually be
        # taken before the single-node constraint trips?
        if matched > 0:
            actually_copied = 0
            for i in range(matched):
                block_nid = node.bnis[i]
                if matched_node_id == -1:
                    matched_node_id = block_nid  # lock the first one
                elif block_nid != matched_node_id:
                    # Constraint trip — truncate AT this block.
                    matched = actually_copied
                    break
                pb_out.append(node.pbs[i])
                actually_copied += 1

            prefix_blocks_num += matched

        # Step 3: if we couldn't consume the entire node (either hash
        # mismatch or single-node trip), stop descending.
        if matched < node_size:
            break

    return _MatchResult(
        prefix_blocks_num=prefix_blocks_num,
        physical_blocks=pb_out,
        matched_node_id=matched_node_id,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestSingleNodeMatchingConstraint:
    def test_empty_query_yields_empty_result(self):
        nodes = [_Node([10, 11], [100, 101], [1, 1])]
        r = match_prefix_ref(nodes, [])
        assert r.prefix_blocks_num == 0
        assert r.physical_blocks == []
        assert r.matched_node_id == -1

    def test_no_nodes_yields_empty_result(self):
        r = match_prefix_ref([], [10, 11])
        assert r.prefix_blocks_num == 0
        assert r.physical_blocks == []
        assert r.matched_node_id == -1

    def test_full_match_single_node_all_same_nid(self):
        """Classic full-hit: 3 blocks all belong to node 7."""
        nodes = [_Node(
            hashes=[10, 11, 12],
            pbs=[100, 101, 102],
            bnis=[7, 7, 7],
        )]
        r = match_prefix_ref(nodes, [10, 11, 12])
        assert r.prefix_blocks_num == 3
        assert r.physical_blocks == [100, 101, 102]
        assert r.matched_node_id == 7

    def test_partial_match_hash_mismatch_mid_node(self):
        """Hash mismatch at index 2: stop at index 2 regardless of nid."""
        nodes = [_Node(
            hashes=[10, 11, 99, 13],
            pbs=[100, 101, 102, 103],
            bnis=[5, 5, 5, 5],
        )]
        r = match_prefix_ref(nodes, [10, 11, 12, 13])
        assert r.prefix_blocks_num == 2
        assert r.physical_blocks == [100, 101]
        assert r.matched_node_id == 5

    def test_single_node_trip_mid_node(self):
        """Same node contains blocks from 2 different nids — truncate
        at the first different nid."""
        nodes = [_Node(
            hashes=[10, 11, 12, 13],
            pbs=[100, 101, 102, 103],
            bnis=[7, 7, 8, 8],  # first 2 are nid=7, next 2 are nid=8
        )]
        r = match_prefix_ref(nodes, [10, 11, 12, 13])
        # Locked on nid=7 for the first block; at block index 2 (nid=8)
        # the trip fires → only first 2 blocks taken.
        assert r.prefix_blocks_num == 2
        assert r.physical_blocks == [100, 101]
        assert r.matched_node_id == 7

    def test_single_node_trip_at_first_block_of_second_node(self):
        """Descend to child only if the full first node was consumed —
        here node 1 is fully consumed (all nid=5), node 2 starts with
        nid=9 which trips the constraint."""
        nodes = [
            _Node([10, 11], [100, 101], [5, 5]),
            _Node([12, 13], [102, 103], [9, 9]),
        ]
        r = match_prefix_ref(nodes, [10, 11, 12, 13])
        # First node consumed (locked nid=5 for both blocks).  Second
        # node's first block has nid=9 != 5 → trip at block 0 of node 2.
        assert r.prefix_blocks_num == 2
        assert r.physical_blocks == [100, 101]
        assert r.matched_node_id == 5

    def test_single_node_trip_happens_before_hash_mismatch_check(self):
        """When both constraints would fire, single-node wins first
        (determines truncation point).  Design doc requires that we
        NEVER copy a block with a foreign nid, even if its hash is
        otherwise valid."""
        nodes = [_Node(
            hashes=[10, 11, 12, 13],
            pbs=[100, 101, 102, 103],
            bnis=[3, 3, 4, 99],  # trip at index 2 due to nid
        )]
        # Query matches all 4 hashes — but nid constraint stops at index 2.
        r = match_prefix_ref(nodes, [10, 11, 12, 13])
        assert r.prefix_blocks_num == 2
        assert r.physical_blocks == [100, 101]
        assert r.matched_node_id == 3

    def test_trip_at_very_first_block_does_not_happen(self):
        """The first block always succeeds (it defines matched_node_id)."""
        nodes = [_Node(
            hashes=[10, 11],
            pbs=[100, 101],
            bnis=[42, 42],
        )]
        r = match_prefix_ref(nodes, [10, 11])
        assert r.prefix_blocks_num == 2
        assert r.matched_node_id == 42  # locked on first block's nid

    def test_query_longer_than_available_nodes(self):
        """If the query extends past the last node, match stops at end."""
        nodes = [_Node([10, 11], [100, 101], [1, 1])]
        r = match_prefix_ref(nodes, [10, 11, 12, 13])
        assert r.prefix_blocks_num == 2
        assert r.physical_blocks == [100, 101]
        assert r.matched_node_id == 1

    def test_query_shorter_than_first_node(self):
        """Query consumes only part of the first node — matched_node_id
        still reflects the locked nid even if the node has other nids
        after the query end."""
        nodes = [_Node(
            hashes=[10, 11, 12, 13],
            pbs=[100, 101, 102, 103],
            bnis=[5, 5, 9, 9],  # would trip at index 2 if we got there
        )]
        r = match_prefix_ref(nodes, [10, 11])  # query of length 2
        assert r.prefix_blocks_num == 2
        assert r.physical_blocks == [100, 101]
        assert r.matched_node_id == 5  # locked on first block's nid

    def test_matched_node_id_persists_across_nodes(self):
        """Multi-node match: matched_node_id locked on node 1's first
        block must still apply to node 2's blocks (they must all be the
        same nid or trip)."""
        nodes = [
            _Node([10, 11], [100, 101], [3, 3]),
            _Node([12, 13], [102, 103], [3, 3]),
        ]
        r = match_prefix_ref(nodes, [10, 11, 12, 13])
        assert r.prefix_blocks_num == 4
        assert r.physical_blocks == [100, 101, 102, 103]
        assert r.matched_node_id == 3

    def test_matched_node_id_minus_one_iff_no_match(self):
        """The only way to get matched_node_id == -1 is zero matched blocks."""
        # Case A: first hash mismatch.
        nodes_a = [_Node([999], [100], [1])]
        r_a = match_prefix_ref(nodes_a, [10, 11])
        assert r_a.prefix_blocks_num == 0
        assert r_a.matched_node_id == -1

        # Case B: empty nodes list.
        r_b = match_prefix_ref([], [10, 11])
        assert r_b.prefix_blocks_num == 0
        assert r_b.matched_node_id == -1

    def test_pb_write_index_matches_prefix_blocks_num(self):
        """The C++ code writes ``pb_write`` physical blocks and returns
        a narrow(0, pb_write) tensor.  ``len(physical_blocks)`` must
        equal ``prefix_blocks_num`` in every branch."""
        # Trip mid-match.
        nodes = [_Node(
            hashes=[10, 11, 12, 13],
            pbs=[100, 101, 102, 103],
            bnis=[7, 7, 8, 8],
        )]
        r = match_prefix_ref(nodes, [10, 11, 12, 13])
        assert len(r.physical_blocks) == r.prefix_blocks_num

        # Full match.
        nodes2 = [_Node([20, 21], [200, 201], [9, 9])]
        r2 = match_prefix_ref(nodes2, [20, 21])
        assert len(r2.physical_blocks) == r2.prefix_blocks_num

        # No match.
        nodes3 = [_Node([999], [100], [1])]
        r3 = match_prefix_ref(nodes3, [20])
        assert len(r3.physical_blocks) == r3.prefix_blocks_num == 0


class TestRegressionScenarios:
    """Specific scenarios from design doc §4.7.1.4 / §4.4.1."""

    def test_design_doc_scenario_cross_remote_leaked_into_merge(self):
        """Per design doc: even if a single radix node contains blocks
        from multiple peer_instances (legal after merge_root_child), the
        match must truncate at the boundary so the Master only issues a
        coordinated GET to ONE peer."""
        # Node had 5 blocks, first 3 from peer_instance_A (nid 100),
        # last 2 merged in from peer_instance_B (nid 200).
        nodes = [_Node(
            hashes=[1, 2, 3, 4, 5],
            pbs=[10, 20, 30, 40, 50],
            bnis=[100, 100, 100, 200, 200],
        )]
        r = match_prefix_ref(nodes, [1, 2, 3, 4, 5])
        # Must stop at 3 blocks, all from peer A.
        assert r.prefix_blocks_num == 3
        assert r.physical_blocks == [10, 20, 30]
        assert r.matched_node_id == 100

    @pytest.mark.parametrize("trip_at", [1, 2, 3, 4, 5])
    def test_trip_at_arbitrary_position_is_exact(self, trip_at: int):
        """Parametrize the trip position and confirm byte-exact cut."""
        # 6-block node, trip at index ``trip_at``.
        n = 6
        bnis = [7] * trip_at + [8] * (n - trip_at)
        nodes = [_Node(
            hashes=list(range(10, 10 + n)),
            pbs=list(range(100, 100 + n)),
            bnis=bnis,
        )]
        r = match_prefix_ref(nodes, list(range(10, 10 + n)))
        assert r.prefix_blocks_num == trip_at
        assert r.physical_blocks == list(range(100, 100 + trip_at))
        assert r.matched_node_id == 7
