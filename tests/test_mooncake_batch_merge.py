# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Batch-merge unit tests for the mooncake-store lanes.

Covers the pure-Python glue in ``flexkv.common.transfer`` that does NOT need
the mooncake SDK:

* ``_merge_ops`` / ``_merge_swa_ops`` — hash preservation, mix rejection.
* ``merge_to_batch_graph`` — 12-bucket layout (main+SWA × 6 transfer types),
  mooncake dependency edges, ``batch_end_op_id`` VIRTUAL-sink semantics.
* Layerwise + mooncake coexistence: REMOTE2H stays standalone and is a
  predecessor of the fused LAYERWISE op (prefetch before H2D).
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pytest

from flexkv.common.transfer import (
    TransferOp,
    TransferOpGraph,
    TransferType,
    _merge_ops,
    _merge_swa_ops,
    merge_to_batch_graph,
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _mk_op(
    transfer_type: TransferType,
    src: List[int],
    dst: List[int],
    *,
    is_swa: bool = False,
    kv_hashes: Optional[List[str]] = None,
    swa_hashes: Optional[List[str]] = None,
    graph_id: int = 0,
    dp_client_id: int = 0,
) -> TransferOp:
    return TransferOp(
        graph_id=graph_id,
        transfer_type=transfer_type,
        src_block_ids=np.array(src, dtype=np.int64),
        dst_block_ids=np.array(dst, dtype=np.int64),
        dp_client_id=dp_client_id,
        is_swa=is_swa,
        mooncake_store_block_hashes=np.array(kv_hashes) if kv_hashes else None,
        mooncake_store_swa_block_hashes=swa_hashes,
    )


def _one_op_graph(op: TransferOp, graph_id: int = 0) -> TransferOpGraph:
    g = TransferOpGraph()
    g.set_graph_id(graph_id)
    op.graph_id = graph_id
    g.add_transfer_op(op)
    return g


def _graph_of(ops: List[TransferOp], graph_id: int = 0) -> TransferOpGraph:
    """Build a graph containing multiple TransferOps under the same graph_id."""
    g = TransferOpGraph()
    g.set_graph_id(graph_id)
    for op in ops:
        op.graph_id = graph_id
        g.add_transfer_op(op)
    return g


def _find_ops(graph: TransferOpGraph, *, transfer_type: TransferType,
              is_swa: bool = False) -> List[TransferOp]:
    return [op for op in graph._op_map.values()
            if op.transfer_type == transfer_type
            and getattr(op, "is_swa", False) == is_swa]


# --------------------------------------------------------------------------
# _merge_ops (main-KV)
# --------------------------------------------------------------------------

class TestMergeMainKvOps:
    def test_hash_concat_in_block_order(self):
        g = TransferOpGraph(); g.set_graph_id(0)
        ops = [
            _mk_op(TransferType.H2REMOTE, [0, 1], [0, 0], kv_hashes=["a0", "a1"]),
            _mk_op(TransferType.H2REMOTE, [2], [0], kv_hashes=["b0"]),
        ]
        merged = _merge_ops(ops, TransferType.H2REMOTE, g, [], {})
        assert merged is not None
        np.testing.assert_array_equal(
            merged.src_block_ids, np.array([0, 1, 2], dtype=np.int64))
        assert list(merged.mooncake_store_block_hashes) == ["a0", "a1", "b0"]

    def test_no_hash_when_no_op_carries_one(self):
        g = TransferOpGraph(); g.set_graph_id(0)
        merged = _merge_ops(
            [_mk_op(TransferType.H2D, [0], [0])], TransferType.H2D, g, [], {})
        assert merged is not None
        assert merged.mooncake_store_block_hashes is None

    def test_mix_hash_and_no_hash_raises(self):
        g = TransferOpGraph(); g.set_graph_id(0)
        ops = [
            _mk_op(TransferType.H2REMOTE, [0], [0], kv_hashes=["a0"]),
            _mk_op(TransferType.H2REMOTE, [1], [0]),  # no hash
        ]
        with pytest.raises(ValueError, match="cannot merge mooncake and non-mooncake"):
            _merge_ops(ops, TransferType.H2REMOTE, g, [], {})

    def test_swa_op_rejected(self):
        g = TransferOpGraph(); g.set_graph_id(0)
        with pytest.raises(ValueError, match="must go through _merge_swa_ops"):
            _merge_ops(
                [_mk_op(TransferType.H2D, [0], [0], is_swa=True)],
                TransferType.H2D, g, [], {})

    def test_unexpected_swa_hashes_rejected(self):
        g = TransferOpGraph(); g.set_graph_id(0)
        with pytest.raises(ValueError, match="mooncake_store_swa_block_hashes"):
            _merge_ops(
                [_mk_op(TransferType.H2REMOTE, [0], [0], swa_hashes=["x"])],
                TransferType.H2REMOTE, g, [], {})


# --------------------------------------------------------------------------
# _merge_swa_ops
# --------------------------------------------------------------------------

class TestMergeSwaOps:
    def test_local_lane_no_hashes(self):
        g = TransferOpGraph(); g.set_graph_id(0)
        ops = [
            _mk_op(TransferType.H2D, [4], [0], is_swa=True),
            _mk_op(TransferType.H2D, [9], [0], is_swa=True),
        ]
        merged = _merge_swa_ops(ops, TransferType.H2D, g, [], {})
        assert merged is not None
        assert merged.is_swa is True
        np.testing.assert_array_equal(
            merged.src_block_ids, np.array([4, 9], dtype=np.int64))
        assert merged.mooncake_store_swa_block_hashes is None

    def test_remote_lane_hash_concat(self):
        g = TransferOpGraph(); g.set_graph_id(0)
        ops = [
            _mk_op(TransferType.H2REMOTE, [4], [0], is_swa=True, swa_hashes=["tx"]),
            _mk_op(TransferType.H2REMOTE, [9], [0], is_swa=True, swa_hashes=["ty"]),
        ]
        merged = _merge_swa_ops(ops, TransferType.H2REMOTE, g, [], {})
        assert merged is not None
        assert merged.is_swa is True
        assert merged.mooncake_store_swa_block_hashes == ["tx", "ty"]

    def test_remote_lane_missing_hash_raises(self):
        g = TransferOpGraph(); g.set_graph_id(0)
        ops = [_mk_op(TransferType.H2REMOTE, [0], [0], is_swa=True)]  # no hash
        with pytest.raises(ValueError, match="missing mooncake_store_swa_block_hashes"):
            _merge_swa_ops(ops, TransferType.H2REMOTE, g, [], {})

    def test_local_lane_carrying_hash_rejected(self):
        g = TransferOpGraph(); g.set_graph_id(0)
        ops = [_mk_op(TransferType.H2D, [0], [0], is_swa=True, swa_hashes=["x"])]
        with pytest.raises(ValueError, match="only allowed on H2REMOTE / REMOTE2H"):
            _merge_swa_ops(ops, TransferType.H2D, g, [], {})

    def test_non_swa_rejected(self):
        g = TransferOpGraph(); g.set_graph_id(0)
        ops = [_mk_op(TransferType.H2D, [0], [0])]  # is_swa=False
        with pytest.raises(ValueError, match="must have is_swa=True"):
            _merge_swa_ops(ops, TransferType.H2D, g, [], {})

    def test_carrying_main_kv_hash_rejected(self):
        g = TransferOpGraph(); g.set_graph_id(0)
        ops = [_mk_op(TransferType.H2REMOTE, [0], [0], is_swa=True,
                      kv_hashes=["should_not_be_here"])]
        with pytest.raises(ValueError, match="unexpected mooncake_store_block_hashes"):
            _merge_swa_ops(ops, TransferType.H2REMOTE, g, [], {})


# --------------------------------------------------------------------------
# merge_to_batch_graph — mooncake dependencies + batch_end_op VIRTUAL sink
# --------------------------------------------------------------------------

def _find_virtual(graph: TransferOpGraph) -> List[TransferOp]:
    return _find_ops(graph, transfer_type=TransferType.VIRTUAL, is_swa=False)


class TestBatchMergeMooncakeDependencies:
    def test_get_remote2h_gates_h2d(self):
        r2h = _mk_op(TransferType.REMOTE2H, [0], [0], kv_hashes=["h"])
        h2d = _mk_op(TransferType.H2D, [0], [0])
        merged, _, _ = merge_to_batch_graph(1, [_graph_of([r2h, h2d])], [-1], {})
        (m_h2d,) = _find_ops(merged, transfer_type=TransferType.H2D)
        (m_r2h,) = _find_ops(merged, transfer_type=TransferType.REMOTE2H)
        assert m_r2h.op_id in m_h2d.predecessors

    def test_put_d2h_gates_h2remote(self):
        d2h = _mk_op(TransferType.D2H, [0], [0])
        h2r = _mk_op(TransferType.H2REMOTE, [0], [0], kv_hashes=["h"])
        merged, _, _ = merge_to_batch_graph(1, [_graph_of([d2h, h2r])], [-1], {})
        (m_d2h,) = _find_ops(merged, transfer_type=TransferType.D2H)
        (m_h2r,) = _find_ops(merged, transfer_type=TransferType.H2REMOTE)
        assert m_d2h.op_id in m_h2r.predecessors

    def test_swa_get_remote2h_gates_swa_h2d(self):
        r2h = _mk_op(TransferType.REMOTE2H, [0], [0], is_swa=True, swa_hashes=["t"])
        h2d = _mk_op(TransferType.H2D, [0], [0], is_swa=True)
        merged, _, _ = merge_to_batch_graph(1, [_graph_of([r2h, h2d])], [-1], {})
        (m_h2d,) = _find_ops(merged, transfer_type=TransferType.H2D, is_swa=True)
        (m_r2h,) = _find_ops(merged, transfer_type=TransferType.REMOTE2H, is_swa=True)
        assert m_r2h.op_id in m_h2d.predecessors

    def test_swa_put_d2h_gates_swa_h2remote(self):
        d2h = _mk_op(TransferType.D2H, [0], [0], is_swa=True)
        h2r = _mk_op(TransferType.H2REMOTE, [0], [0], is_swa=True, swa_hashes=["t"])
        merged, _, _ = merge_to_batch_graph(1, [_graph_of([d2h, h2r])], [-1], {})
        (m_d2h,) = _find_ops(merged, transfer_type=TransferType.D2H, is_swa=True)
        (m_h2r,) = _find_ops(merged, transfer_type=TransferType.H2REMOTE, is_swa=True)
        assert m_d2h.op_id in m_h2r.predecessors


class TestBatchEndOpIdVirtualSink:
    def test_get_main_only_no_virtual(self):
        h2d = _mk_op(TransferType.H2D, [0], [0])
        merged, end, _ = merge_to_batch_graph(1, [_one_op_graph(h2d)], [-1], {})
        assert not _find_virtual(merged)
        (m_h2d,) = _find_ops(merged, transfer_type=TransferType.H2D)
        assert end == m_h2d.op_id

    def test_get_swa_only_no_virtual(self):
        swa_h2d = _mk_op(TransferType.H2D, [0], [0], is_swa=True)
        merged, end, _ = merge_to_batch_graph(1, [_one_op_graph(swa_h2d)], [-1], {})
        assert not _find_virtual(merged)
        (m_swa_h2d,) = _find_ops(merged, transfer_type=TransferType.H2D, is_swa=True)
        assert end == m_swa_h2d.op_id

    def test_get_main_and_swa_virtual_sink(self):
        h2d = _mk_op(TransferType.H2D, [0], [0])
        swa_h2d = _mk_op(TransferType.H2D, [0], [0], is_swa=True)
        merged, end, _ = merge_to_batch_graph(1, [_graph_of([h2d, swa_h2d])], [-1], {})
        (sink,) = _find_virtual(merged)
        assert end == sink.op_id
        (m_h2d,) = _find_ops(merged, transfer_type=TransferType.H2D, is_swa=False)
        (m_swa_h2d,) = _find_ops(merged, transfer_type=TransferType.H2D, is_swa=True)
        assert m_h2d.op_id in sink.predecessors
        assert m_swa_h2d.op_id in sink.predecessors

    def test_put_main_only_no_virtual(self):
        d2h = _mk_op(TransferType.D2H, [0], [0])
        merged, end, _ = merge_to_batch_graph(1, [_one_op_graph(d2h)], [-1], {})
        assert not _find_virtual(merged)
        (m_d2h,) = _find_ops(merged, transfer_type=TransferType.D2H)
        assert end == m_d2h.op_id

    def test_put_main_and_swa_virtual_sink(self):
        d2h = _mk_op(TransferType.D2H, [0], [0])
        swa_d2h = _mk_op(TransferType.D2H, [0], [0], is_swa=True)
        merged, end, _ = merge_to_batch_graph(1, [_graph_of([d2h, swa_d2h])], [-1], {})
        (sink,) = _find_virtual(merged)
        assert end == sink.op_id
        (m_d2h,) = _find_ops(merged, transfer_type=TransferType.D2H, is_swa=False)
        (m_swa_d2h,) = _find_ops(merged, transfer_type=TransferType.D2H, is_swa=True)
        assert m_d2h.op_id in sink.predecessors
        assert m_swa_d2h.op_id in sink.predecessors

    def test_prefetch_only_remote2h_single_terminal(self):
        # Only mooncake staging present (no H2D / D2H). end -> the remote op.
        r2h = _mk_op(TransferType.REMOTE2H, [0], [0], kv_hashes=["h"])
        merged, end, _ = merge_to_batch_graph(1, [_one_op_graph(r2h)], [-1], {})
        assert not _find_virtual(merged)
        (m_r2h,) = _find_ops(merged, transfer_type=TransferType.REMOTE2H)
        assert end == m_r2h.op_id

    def test_prefetch_main_and_swa_remote2h_virtual_sink(self):
        # Prefetch with both full-KV and SWA REMOTE2H (no H2D): batch end must
        # wait for BOTH leaves, not just the first one found.
        r2h = _mk_op(TransferType.REMOTE2H, [0], [0], kv_hashes=["h"])
        swa_r2h = _mk_op(TransferType.REMOTE2H, [0], [0], is_swa=True, swa_hashes=["t"])
        merged, end, _ = merge_to_batch_graph(
            1, [_graph_of([r2h, swa_r2h])], [-1], {})
        (sink,) = _find_virtual(merged)
        assert end == sink.op_id
        (m_r2h,) = _find_ops(merged, transfer_type=TransferType.REMOTE2H, is_swa=False)
        (m_swa_r2h,) = _find_ops(merged, transfer_type=TransferType.REMOTE2H, is_swa=True)
        assert m_r2h.op_id in sink.predecessors
        assert m_swa_r2h.op_id in sink.predecessors

    def test_put_no_d2h_h2disk_and_h2remote_virtual_sink(self):
        # Degenerate PUT without D2H: both leaf lanes must be terminals.
        h2disk = _mk_op(TransferType.H2DISK, [0], [0])
        h2remote = _mk_op(TransferType.H2REMOTE, [0], [0], kv_hashes=["h"])
        merged, end, _ = merge_to_batch_graph(
            1, [_graph_of([h2disk, h2remote])], [-1], {})
        (sink,) = _find_virtual(merged)
        assert end == sink.op_id
        (m_disk,) = _find_ops(merged, transfer_type=TransferType.H2DISK)
        (m_remote,) = _find_ops(merged, transfer_type=TransferType.H2REMOTE)
        assert m_disk.op_id in sink.predecessors
        assert m_remote.op_id in sink.predecessors


# --------------------------------------------------------------------------
# Layerwise + mooncake coexistence
# --------------------------------------------------------------------------

class TestLayerwiseMooncakeCoexistence:
    def test_mooncake_r2h_stays_standalone_with_layerwise_h2d(self):
        # Batch has both: mooncake REMOTE2H (prefetch stage) AND layerwise H2D
        # (layerwise stage). Expect: LAYERWISE op fuses local DISK2H+H2D+SWA,
        # while REMOTE2H remains as an independent op in the merged graph.
        r2h = _mk_op(TransferType.REMOTE2H, [0], [0], kv_hashes=["h"])
        h2d = _mk_op(TransferType.H2D, [0], [0])
        merged, end, _ = merge_to_batch_graph(
            1, [_graph_of([r2h, h2d])], [-1], {}, layerwise_transfer=True)
        # LAYERWISE op exists.
        lw_ops = _find_ops(merged, transfer_type=TransferType.LAYERWISE)
        assert len(lw_ops) == 1
        # REMOTE2H still standalone (not folded).
        r2h_ops = _find_ops(merged, transfer_type=TransferType.REMOTE2H)
        assert len(r2h_ops) == 1
        # LAYERWISE depends on REMOTE2H (prefetch-before-H2D); end is LAYERWISE.
        assert r2h_ops[0].op_id in lw_ops[0].predecessors
        assert end == lw_ops[0].op_id
        assert not _find_virtual(merged)

    def test_layerwise_only_no_mooncake_end_id_layerwise(self):
        # No mooncake ops: end should be the LAYERWISE op directly (no VIRTUAL).
        h2d = _mk_op(TransferType.H2D, [0], [0])
        merged, end, _ = merge_to_batch_graph(
            1, [_one_op_graph(h2d)], [-1], {}, layerwise_transfer=True)
        lw_ops = _find_ops(merged, transfer_type=TransferType.LAYERWISE)
        assert len(lw_ops) == 1
        assert not _find_virtual(merged)
        assert end == lw_ops[0].op_id

    def test_swa_mooncake_r2h_stays_standalone_with_layerwise(self):
        # Layerwise + SWA REMOTE2H mooncake. LAYERWISE handles local SWA H2D;
        # SWA REMOTE2H stands alone.
        swa_r2h = _mk_op(TransferType.REMOTE2H, [0], [0], is_swa=True, swa_hashes=["t"])
        swa_h2d = _mk_op(TransferType.H2D, [0], [0], is_swa=True)
        merged, end, _ = merge_to_batch_graph(
            1, [_graph_of([swa_r2h, swa_h2d])], [-1], {}, layerwise_transfer=True)
        lw_ops = _find_ops(merged, transfer_type=TransferType.LAYERWISE)
        assert len(lw_ops) == 1
        swa_r2h_ops = _find_ops(merged, transfer_type=TransferType.REMOTE2H, is_swa=True)
        assert len(swa_r2h_ops) == 1
        assert swa_r2h_ops[0].op_id in lw_ops[0].predecessors
        assert end == lw_ops[0].op_id
        assert not _find_virtual(merged)
