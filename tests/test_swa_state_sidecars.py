from __future__ import annotations

import numpy as np
import torch

from flexkv.common.config import LayerGroupSpec
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import (
    LayerwiseTransferOp,
    TransferOp,
    TransferOpGraph,
    TransferType,
    merge_to_batch_graph,
)


def test_swa_state_sidecars_share_one_byte_flat_host_block() -> None:
    """C4 state rows are packed beside SWA KV under the same SWA page id."""
    page_size = 256
    ring_size = 8
    groups = [
        LayerGroupSpec(
            num_layers=3,
            num_kv_heads=1,
            head_size=585,
            layer_indices=[0, 1, 2],
            dtype=torch.uint8,
        ),
        LayerGroupSpec(
            num_layers=2,
            num_kv_heads=1,
            head_size=2304,
            layer_indices=[0, 2],
            dtype=torch.float32,
            compress_ratio=page_size // ring_size,
        ),
        LayerGroupSpec(
            num_layers=2,
            num_kv_heads=1,
            head_size=512,
            layer_indices=[0, 2],
            dtype=torch.float32,
            compress_ratio=page_size // ring_size,
        ),
    ]
    layout = KVCacheLayout(
        type=KVCacheLayoutType.BLOCKFIRST,
        num_layer=3,
        num_block=7,
        tokens_per_block=page_size,
        num_head=1,
        head_size=585,
        is_mla=True,
        layer_groups=groups,
    )

    expected_block_bytes = (
        3 * page_size * 585
        + 2 * ring_size * 2304 * torch.float32.itemsize
        + 2 * ring_size * 512 * torch.float32.itemsize
    )
    assert layout.get_block_stride() == expected_block_bytes
    assert layout.kv_shape == torch.Size([7, expected_block_bytes])


def test_layerwise_fuses_heterogeneous_swa_state_h2d() -> None:
    """Layerwise GET packs main-KV + SWA/state into one LAYERWISE op."""
    graph = TransferOpGraph()
    main_h2d = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=TransferType.H2D,
        src_block_ids=np.array([11], dtype=np.int64),
        dst_block_ids=np.array([21], dtype=np.int64),
    )
    swa_h2d = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=TransferType.H2D,
        src_block_ids=np.array([31], dtype=np.int64),
        dst_block_ids=np.array([41], dtype=np.int64),
        is_swa=True,
    )
    graph.add_transfer_op(main_h2d)
    graph.add_transfer_op(swa_h2d)

    merged, task_end, _ = merge_to_batch_graph(
        batch_id=99,
        transfer_graphs=[graph],
        task_end_op_ids=[main_h2d.op_id],
        op_callback_dict={},
        layerwise_transfer=True,
        counter_id=2,
    )

    layerwise_ops = [
        op for op in merged._op_map.values() if isinstance(op, LayerwiseTransferOp)
    ]
    standalone_swa = [
        op
        for op in merged._op_map.values()
        if op.transfer_type == TransferType.H2D and op.is_swa
    ]
    assert len(layerwise_ops) == 1
    assert len(standalone_swa) == 0
    assert np.array_equal(
        layerwise_ops[0].swa_src_block_ids_h2d, np.array([31], dtype=np.int64)
    )
    assert np.array_equal(
        layerwise_ops[0].swa_dst_block_ids_h2d, np.array([41], dtype=np.int64)
    )
    assert task_end == layerwise_ops[0].op_id


def test_swa_multi_layer_false_keeps_sidecar_h2d_as_predecessor() -> None:
    """The compatibility switch keeps SWA/state outside LAYERWISE."""
    graph = TransferOpGraph()
    main_h2d = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=TransferType.H2D,
        src_block_ids=np.array([11], dtype=np.int64),
        dst_block_ids=np.array([21], dtype=np.int64),
    )
    swa_h2d = TransferOp(
        graph_id=graph.graph_id,
        transfer_type=TransferType.H2D,
        src_block_ids=np.array([31], dtype=np.int64),
        dst_block_ids=np.array([41], dtype=np.int64),
        is_swa=True,
    )
    graph.add_transfer_op(main_h2d)
    graph.add_transfer_op(swa_h2d)

    merged, task_end, _ = merge_to_batch_graph(
        batch_id=99,
        transfer_graphs=[graph],
        task_end_op_ids=[main_h2d.op_id],
        op_callback_dict={},
        layerwise_transfer=True,
        counter_id=2,
    )

    layerwise_ops = [
        op for op in merged._op_map.values() if isinstance(op, LayerwiseTransferOp)
    ]
    standalone_swa = [
        op
        for op in merged._op_map.values()
        if op.transfer_type == TransferType.H2D and op.is_swa
    ]
    assert len(layerwise_ops) == 1
    assert len(standalone_swa) == 1
    assert standalone_swa[0].op_id in layerwise_ops[0].predecessors
    assert layerwise_ops[0].swa_src_block_ids_h2d.size == 0
    assert layerwise_ops[0].swa_dst_block_ids_h2d.size == 0
    assert task_end == layerwise_ops[0].op_id
