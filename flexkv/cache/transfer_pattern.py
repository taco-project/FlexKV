from typing import List, Optional, Tuple

import numpy as np
import torch

from flexkv.common.transfer import TransferType
from flexkv.common.transfer import TransferOp, TransferOpGraph


def add_virtal_op_for_mutiple_finished_ops(
    graph: TransferOpGraph,
    finished_ops_ids: List[int]
)->Tuple[TransferOpGraph, int]:
    if len(finished_ops_ids) == 0:
        return graph, -1
    elif len(finished_ops_ids) == 1:
        return graph, finished_ops_ids[0]
    else:
        op = TransferOp(
            graph_id = graph.graph_id,
            transfer_type = TransferType.VIRTUAL,
            src_block_ids = np.array([], dtype=np.int64),
            dst_block_ids = np.array([], dtype=np.int64),
            layer_id = -1,
            layer_granularity = -1,
        )
        graph.add_transfer_op(op)
        for op_id in finished_ops_ids:
            graph.add_dependency(op.op_id, op_id)
        return graph, op.op_id

def convert_read_graph_to_layer_wise_graph(
    transfer_graph: TransferOpGraph,
    finished_ops_ids: List[int],
    layer_num: int,
    layer_granularity: int,
) -> Tuple[TransferOpGraph, List[int]]:
    """
    Convert the input read transfer graph into a layer-wise transfer graph
    according to the given granularity. Each op will be split into
    (layer_num // layer_granularity) ops, and the original dependency relationships are preserved.
    """
    assert layer_num % layer_granularity == 0
    num_splits = layer_num // layer_granularity
    #reuse the graph id to assure that graph <-> request is one-to-one
    new_graph = TransferOpGraph()  # TODO: no need to create a new graph
    new_graph.set_graph_id(transfer_graph.graph_id)
    # Map from original op id to a list of new op ids (length = num_splits)
    opid2splitopids = {}
    layer_wise_virtual_op_ids = []
    # Split all ops
    for op_id, op in transfer_graph._op_map.items():
        split_op_ids = []
        for i in range(num_splits):
            # Copy op, modify layer_id and layer_granularity
            new_op = TransferOp(
                graph_id=new_graph.graph_id,
                transfer_type=op.transfer_type,
                src_block_ids=op.src_block_ids,
                dst_block_ids=op.dst_block_ids,
                layer_id=i * layer_granularity,
                layer_granularity=layer_granularity,
                # Inherit these fields directly
                dp_id=op.dp_id,
            )
            new_graph.add_transfer_op(new_op)
            split_op_ids.append(new_op.op_id)
        opid2splitopids[op_id] = split_op_ids

    # add splited ops to the finished_ops_ids
    for i in range(num_splits):
        layer_wise_virtual_op_ids.append(opid2splitopids[finished_ops_ids[0]][i])

    # Copy dependency relationships (preserve original dependencies between ops)
    for op_id, op in transfer_graph._op_map.items():
        for succ_id in op.successors:
            for i in range(num_splits):
                new_graph.add_dependency(
                    opid2splitopids[succ_id][i],
                    opid2splitopids[op_id][i]
                )

    # Add dependencies between split ops of the same original op (i.e., i-th depends on (i-1)-th)
    for op_id, split_op_ids in opid2splitopids.items():
        if transfer_graph._op_map[op_id].transfer_type == TransferType.VIRTUAL:
            continue
        for i in range(1, num_splits):
            new_graph.add_dependency(split_op_ids[i], split_op_ids[i - 1])

    return new_graph, layer_wise_virtual_op_ids

def create_read_graph_gds(
    gpu_blocks: torch.Tensor,
    gds_blocks: torch.Tensor,
    gpu_device_id: int = 0,
    layer_num: int = 1,
    graph: Optional[TransferOpGraph] = None,
)->Tuple[TransferOpGraph, List[int]]:
    """
    Create a read transfer graph with GDS->GPU operations
    
    Args:
        gpu_blocks: GPU block IDs to read into
        gds_blocks: GDS block IDs to read from
        gpu_device_id: GPU device ID
        layer_num: Number of layers to transfer
        graph: Optional existing graph to add operations to
        
    Returns:
        graph: TransferOpGraph
        ops_to_be_tracked: List[int]: a list of transfer ops that can indicate
        the completion of some key operations
    """
    assert len(gpu_blocks) == len(gds_blocks)
    if graph is None:
        graph = TransferOpGraph()
    assert len(gpu_blocks) > 0
    
    op = TransferOp(
        graph_id = graph.graph_id,
        transfer_type = TransferType.GDS2D,
        src_descriptor = TransferDescriptor(
            device_type = DeviceType.GDS,
            physical_block_ids=gds_blocks,
        ),
        dst_descriptor = TransferDescriptor(
            device_type = DeviceType.GPU,
            physical_block_ids=gpu_blocks,
            device_id = gpu_device_id
        ),
        layer_id = 0,
        layer_granularity = layer_num,
    )
    graph.add_transfer_op(op)
    return graph, [op.op_id]

def create_write_graph_gds(
    gpu_blocks: torch.Tensor,
    gds_blocks: torch.Tensor,
    gpu_device_id: int = 0,
    layer_num: int = 1,
    graph: Optional[TransferOpGraph] = None,
)->Tuple[TransferOpGraph, List[int]]:
    """
    Create a write transfer graph with GPU->GDS operations
    
    Args:
        gpu_blocks: GPU block IDs to write from
        gds_blocks: GDS block IDs to write to
        gpu_device_id: GPU device ID
        layer_num: Number of layers to transfer
        graph: Optional existing graph to add operations to
        
    Returns:
        graph: TransferOpGraph
        layer_wise_ops: List[int]: a list of transfer ops that can indicate
        the completion of each layer or each layer for each tp rank
    """
    assert len(gpu_blocks) == len(gds_blocks)
    if graph is None:
        graph = TransferOpGraph()
    assert len(gpu_blocks) > 0
    
    op = TransferOp(
        graph_id = graph.graph_id,
        transfer_type = TransferType.D2GDS,
        src_descriptor = TransferDescriptor(
            device_type = DeviceType.GPU,
            physical_block_ids=gpu_blocks,
            device_id = gpu_device_id
        ),
        dst_descriptor = TransferDescriptor(
            device_type = DeviceType.GDS,
            physical_block_ids=gds_blocks,
        ),
        layer_id = 0,
        layer_granularity = layer_num
    )
    graph.add_transfer_op(op)
    return graph, [op.op_id]
