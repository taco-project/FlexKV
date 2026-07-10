"""merge_to_batch_graph: SWA + main per-op callbacks survive fusion."""
import numpy as np

from flexkv.common.transfer import (
    LayerwiseTransferOp,
    TransferOp,
    TransferOpGraph,
    TransferType,
    add_virtual_op_for_multiple_finished_ops,
    merge_to_batch_graph,
)


def _graph_with_get_ops(*, with_swa: bool) -> tuple[TransferOpGraph, dict[int, object]]:
    graph = TransferOpGraph()
    fired: list[str] = []

    disk2h = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=TransferType.DISK2H,
        src_block_ids=np.array([1], dtype=np.int64),
        dst_block_ids=np.array([10], dtype=np.int64),
    )
    h2d = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=TransferType.H2D,
        src_block_ids=np.array([10], dtype=np.int64),
        dst_block_ids=np.array([0], dtype=np.int64),
    )
    graph.add_transfer_op(disk2h)
    graph.add_transfer_op(h2d)
    graph.add_dependency(h2d.op_id, disk2h.op_id)

    callbacks = {
        disk2h.op_id: lambda: fired.append("main_disk2h"),
        h2d.op_id: lambda: fired.append("main_h2d"),
    }

    if with_swa:
        swa_disk2h = TransferOp(
            graph_id=graph.graph_id,
            transfer_type=TransferType.DISK2H,
            src_block_ids=np.array([2], dtype=np.int64),
            dst_block_ids=np.array([20], dtype=np.int64),
            is_swa=True,
        )
        swa_h2d = TransferOp(
            graph_id=graph.graph_id,
            transfer_type=TransferType.H2D,
            src_block_ids=np.array([20], dtype=np.int64),
            dst_block_ids=np.array([5], dtype=np.int64),
            is_swa=True,
        )
        graph.add_transfer_op(swa_disk2h)
        graph.add_transfer_op(swa_h2d)
        graph.add_dependency(swa_h2d.op_id, swa_disk2h.op_id)
        callbacks[swa_disk2h.op_id] = lambda: fired.append("swa_disk2h")
        callbacks[swa_h2d.op_id] = lambda: fired.append("swa_h2d")

    return graph, {"fired": fired, "callbacks": callbacks}


def test_layerwise_merge_combines_main_and_swa_callbacks():
    graph, ctx = _graph_with_get_ops(with_swa=True)
    merged, batch_end_op_id, op_callbacks = merge_to_batch_graph(
        batch_id=99,
        transfer_graphs=[graph],
        task_end_op_ids=[-1],
        op_callback_dict=ctx["callbacks"],
        layerwise_transfer=True,
    )

    lw_op = next(iter(merged._op_map.values()))
    assert batch_end_op_id == lw_op.op_id
    assert merged.num_ops == 1
    assert isinstance(lw_op, LayerwiseTransferOp)
    assert lw_op.op_id in op_callbacks

    op_callbacks[lw_op.op_id]()
    assert ctx["fired"] == ["main_disk2h", "main_h2d", "swa_disk2h", "swa_h2d"]


def test_non_layerwise_put_swa_callbacks_on_merged_ops():
    graph = TransferOpGraph()
    fired: list[str] = []

    d2h = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=TransferType.D2H,
        src_block_ids=np.array([0], dtype=np.int64),
        dst_block_ids=np.array([10], dtype=np.int64),
        is_swa=True,
    )
    h2disk = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=TransferType.H2DISK,
        src_block_ids=np.array([10], dtype=np.int64),
        dst_block_ids=np.array([1], dtype=np.int64),
        is_swa=True,
    )
    graph.add_transfer_op(d2h)
    graph.add_transfer_op(h2disk)
    graph.add_dependency(h2disk.op_id, d2h.op_id)

    callbacks = {
        d2h.op_id: lambda: fired.append("swa_d2h"),
        h2disk.op_id: lambda: fired.append("swa_h2disk"),
    }

    merged, batch_end_op_id, op_callbacks = merge_to_batch_graph(
        batch_id=100,
        transfer_graphs=[graph],
        task_end_op_ids=[h2disk.op_id],
        op_callback_dict=callbacks,
        layerwise_transfer=False,
    )

    swa_ops = {op.transfer_type: op for op in merged._op_map.values() if op.is_swa}
    assert len(swa_ops) == 2
    assert batch_end_op_id == swa_ops[TransferType.D2H].op_id

    for op in swa_ops.values():
        assert op.op_id in op_callbacks
        op_callbacks[op.op_id]()

    assert fired == ["swa_d2h", "swa_h2disk"]


def test_non_layerwise_get_swa_ops_and_callbacks():
    graph, ctx = _graph_with_get_ops(with_swa=True)
    merged, _, op_callbacks = merge_to_batch_graph(
        batch_id=101,
        transfer_graphs=[graph],
        task_end_op_ids=[-1],
        op_callback_dict=ctx["callbacks"],
        layerwise_transfer=False,
    )

    swa_ops = {op.transfer_type: op for op in merged._op_map.values() if op.is_swa}
    assert TransferType.DISK2H in swa_ops
    assert TransferType.H2D in swa_ops
    assert swa_ops[TransferType.DISK2H].op_id in op_callbacks
    assert swa_ops[TransferType.H2D].op_id in op_callbacks

    op_callbacks[swa_ops[TransferType.DISK2H].op_id]()
    op_callbacks[swa_ops[TransferType.H2D].op_id]()
    assert "swa_disk2h" in ctx["fired"]
    assert "swa_h2d" in ctx["fired"]


def _graph_with_put_ops(*, with_swa: bool) -> tuple[TransferOpGraph, dict[int, object]]:
    graph = TransferOpGraph()
    fired: list[str] = []

    d2h = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=TransferType.D2H,
        src_block_ids=np.array([0], dtype=np.int64),
        dst_block_ids=np.array([10], dtype=np.int64),
    )
    graph.add_transfer_op(d2h)
    finished = [d2h.op_id]
    callbacks = {d2h.op_id: lambda: fired.append("main_d2h")}

    if with_swa:
        swa_d2h = TransferOp(
            graph_id=graph.graph_id,
            transfer_type=TransferType.D2H,
            src_block_ids=np.array([1], dtype=np.int64),
            dst_block_ids=np.array([20], dtype=np.int64),
            is_swa=True,
        )
        graph.add_transfer_op(swa_d2h)
        finished.append(swa_d2h.op_id)
        callbacks[swa_d2h.op_id] = lambda: fired.append("swa_d2h")

    graph, task_end_op_id = add_virtual_op_for_multiple_finished_ops(
        graph, finished, 0)
    return graph, {
        "fired": fired,
        "callbacks": callbacks,
        "task_end_op_id": task_end_op_id,
    }


def test_non_layerwise_put_full_and_swa_use_virtual_barrier():
    graph, ctx = _graph_with_put_ops(with_swa=True)
    merged, batch_end_op_id, _ = merge_to_batch_graph(
        batch_id=102,
        transfer_graphs=[graph],
        task_end_op_ids=[ctx["task_end_op_id"]],
        op_callback_dict=ctx["callbacks"],
        layerwise_transfer=False,
    )

    barrier = merged._op_map[batch_end_op_id]
    assert barrier.transfer_type == TransferType.VIRTUAL
    merged_ops = {
        op.transfer_type: op
        for op in merged._op_map.values()
        if not op.is_swa and op.transfer_type != TransferType.VIRTUAL
    }
    swa_ops = {op.transfer_type: op for op in merged._op_map.values() if op.is_swa}
    assert barrier.predecessors == {
        merged_ops[TransferType.D2H].op_id,
        swa_ops[TransferType.D2H].op_id,
    }


def test_non_layerwise_get_full_and_swa_use_virtual_barrier():
    graph, ctx = _graph_with_get_ops(with_swa=True)
    finished_ops_ids = [
        op.op_id for op in graph._op_map.values()
        if op.transfer_type == TransferType.H2D
    ]
    graph, task_end_op_id = add_virtual_op_for_multiple_finished_ops(
        graph, finished_ops_ids, 0)

    merged, batch_end_op_id, _callbacks = merge_to_batch_graph(
        batch_id=103,
        transfer_graphs=[graph],
        task_end_op_ids=[task_end_op_id],
        op_callback_dict=ctx["callbacks"],
        layerwise_transfer=False,
    )

    barrier = merged._op_map[batch_end_op_id]
    assert barrier.transfer_type == TransferType.VIRTUAL
    merged_h2d = next(
        op for op in merged._op_map.values()
        if not op.is_swa and op.transfer_type == TransferType.H2D
    )
    swa_h2d = next(
        op for op in merged._op_map.values()
        if op.is_swa and op.transfer_type == TransferType.H2D
    )
    assert barrier.predecessors == {merged_h2d.op_id, swa_h2d.op_id}


def test_non_layerwise_batch_rejects_mixed_get_and_put_terminals():
    graph = TransferOpGraph()
    h2d = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=TransferType.H2D,
        src_block_ids=np.array([10], dtype=np.int64),
        dst_block_ids=np.array([0], dtype=np.int64),
    )
    d2h = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=TransferType.D2H,
        src_block_ids=np.array([1], dtype=np.int64),
        dst_block_ids=np.array([20], dtype=np.int64),
    )
    graph.add_transfer_op(h2d)
    graph.add_transfer_op(d2h)

    try:
        merge_to_batch_graph(
            batch_id=104,
            transfer_graphs=[graph],
            task_end_op_ids=[h2d.op_id],
            op_callback_dict={},
            layerwise_transfer=False,
        )
    except ValueError as exc:
        assert "cannot mix GET and PUT" in str(exc)
    else:
        raise AssertionError("mixed GET/PUT batch must be rejected")
