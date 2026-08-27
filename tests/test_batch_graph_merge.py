"""Two GET-side graph rewrites, pinned at the shape `merge_to_batch_graph` emits.

Both used to be hidden inside one fused op, and both fail *silently* -- wrong
bytes on the GPU, or a task reported done before its data landed -- rather than
raising, which is why they are pinned here rather than left to the e2e runs:

* **The resident/staged H2D split.** A match spanning the CPU and SSD tiers
  used to emit ONE H2D over both fragments, depending on the DISK2H that fills
  only the SSD fragment; that serialized two disjoint block sets. The split
  emits a resident lane (no predecessors) and a staged lane, so a CPU hit moves
  while the SSD read runs -- and each lane must bind its own GPU window.
* **The hoisted DISK2H of a layerwise GET.** It used to be "Step 0" inside the
  cpp ``layerwise_transfer``: one synchronous host read of every layer before
  the first CPU->GPU layer launched, invisible to the scheduler. It is now an
  ordinary op the LAYERWISE op depends on, and the LAYERWISE op must have no
  SSD fields left for cpp to re-read.
"""
import numpy as np

from flexkv.common.transfer import (
    TransferOp,
    TransferOpGraph,
    TransferType,
    merge_to_batch_graph,
)


def _op(graph, ttype, src, dst, **kwargs):
    op = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=ttype,
        src_block_ids=np.array(src, dtype=np.int64),
        dst_block_ids=np.array(dst, dtype=np.int64),
        **kwargs,
    )
    graph.add_transfer_op(op)
    return op


def _merge(graphs, ends, **kwargs):
    return merge_to_batch_graph(
        batch_id=0,
        transfer_graphs=graphs,
        task_end_op_ids=ends,
        op_callback_dict={},
        **kwargs,
    )


# --------------------------------------------------------------------------
# The resident / staged split
# --------------------------------------------------------------------------

def _split_graph():
    """fragment1 = 2 CPU-resident blocks, fragment2 = 3 blocks read from SSD."""
    graph = TransferOpGraph()
    disk2h = _op(graph, TransferType.DISK2H, [90, 91, 92], [12, 13, 14])
    resident = _op(graph, TransferType.H2D, [10, 11], [0, 1],
                   gpu_bind_offset=0, src_is_staged=False)
    staged = _op(graph, TransferType.H2D, [12, 13, 14], [2, 3, 4],
                 gpu_bind_offset=2, src_is_staged=True)
    graph.add_dependency(staged.op_id, disk2h.op_id)
    return graph, staged.op_id


def test_set_gpu_blocks_binds_each_lane_to_its_own_window():
    """The staged lane starts partway into the slot_mapping.

    Without gpu_bind_offset both lanes bind from index 0 and the staged blocks
    land on the resident blocks' GPU slots -- silent corruption, not a crash.
    """
    graph, _end = _split_graph()
    slot_mapping = np.array([100, 101, 102, 103, 104], dtype=np.int64)
    graph.set_gpu_blocks(slot_mapping)

    resident, staged = [op for op in graph._op_map.values()
                        if op.transfer_type == TransferType.H2D]
    np.testing.assert_array_equal(resident.dst_block_ids, [100, 101])
    np.testing.assert_array_equal(staged.dst_block_ids, [102, 103, 104])


def test_batch_merge_keeps_the_two_lanes_apart():
    """Merging the lanes back together would re-serialize the whole batch."""
    graphs, ends = [], []
    for _ in range(2):
        graph, end = _split_graph()
        graphs.append(graph)
        ends.append(end)

    merged, batch_end_op_id, _cbs = _merge(graphs, ends)

    h2d_ops = [op for op in merged._op_map.values()
               if op.transfer_type == TransferType.H2D]
    assert len(h2d_ops) == 2, "expected one resident and one staged merged H2D"
    (resident,) = [op for op in h2d_ops if not op.src_is_staged]
    (staged,) = [op for op in h2d_ops if op.src_is_staged]

    assert not resident.predecessors, \
        "merged resident lane must stay free of the batch's DISK2H"
    assert staged.predecessors, "merged staged lane must keep its DISK2H"
    # Both lanes carry 2 tasks' worth of blocks.
    assert resident.src_block_ids.size == 4
    assert staged.src_block_ids.size == 6

    # The batch is complete only when BOTH lanes have landed.
    sink = merged._op_map[batch_end_op_id]
    assert sink.predecessors == {resident.op_id, staged.op_id}, \
        "reporting on the staged lane alone would mark the GET done early"


# --------------------------------------------------------------------------
# The hoisted DISK2H of a layerwise GET
# --------------------------------------------------------------------------

def test_ssd_ids_land_on_a_standalone_disk2h():
    graph = TransferOpGraph()
    disk2h = _op(graph, TransferType.DISK2H, [90, 91], [10, 11])
    h2d = _op(graph, TransferType.H2D, [10, 11], [0, 1])
    graph.add_dependency(h2d.op_id, disk2h.op_id)

    merged, end_id, _ = _merge([graph], [h2d.op_id], layerwise_transfer=True)

    hoisted = [op for op in merged._op_map.values()
               if op.transfer_type == TransferType.DISK2H]
    assert len(hoisted) == 1, "expected exactly one hoisted DISK2H"
    np.testing.assert_array_equal(hoisted[0].src_block_ids, [90, 91])
    np.testing.assert_array_equal(hoisted[0].dst_block_ids, [10, 11])

    # The LAYERWISE op carries H2D ids only. It has no SSD fields left at all,
    # so cpp cannot re-read the blocks the DISK2H just wrote.
    lw = merged._op_map[end_id]
    assert not hasattr(lw, "src_block_ids_disk2h")
    assert lw.predecessors == {hoisted[0].op_id}
    np.testing.assert_array_equal(lw.src_block_ids_h2d, [10, 11])
