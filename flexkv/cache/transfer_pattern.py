from typing import List, Tuple

import torch

from flexkv.common.transfer import DeviceType, TransferType
from flexkv.common.transfer import TransferOp, TransferOpGraph, TransferDescriptor, TransferIDAllocator

def create_read_graph_cpu_storage(
    graph: TransferOpGraph = None,
    gpu_blocks: torch.Tensor,
    cpu_blocks: torch.Tensor,
    storage_blocks: torch.Tensor,
    gpu_device_id: int = 0,
    layer_num: int = 1,
)->Tuple[TransferOpGraph, List[int]]:
    """
    Create a read transfer graph with (REMOTE_STORAGE / SSD)->CPU->GPU operations
    storage_blocks: the blocks of storage that are used as a lower-level storage backend,
    including ssd or remote storage. This can be empty, which means cpu-only kvcache. 
    Returns:
        graph: TransferOpGraph
        ops_to_be_tracked: List[int]: a list of transfer ops that can indicate
        the completion of some key operations
    """
    if graph is None:
        graph = TransferOpGraph(TransferIDAllocator.allocate_graph_id())
    if len(storage_blocks) == 0:
        op = TransferOp(
            transfer_op_id = TransferIDAllocator.allocate_op_id(),
            transfer_graph_id = graph.transfer_graph_id,
            transfer_type = TransferType.H2D,
            src_descriptor = TransferDescriptor(
                device_type = DeviceType.CPU,
                physical_block_ids=cpu_blocks,
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
        return graph, [op.transfer_op_id]
    elif len(ssd_blocks) < len(cpu_blocks):
        op1 = TransferOp(
            transfer_op_id = TransferIDAllocator.allocate_op_id(),
            transfer_graph_id = graph.transfer_graph_id,
            transfer_type = TransferType.DISK2H,
            src_descriptor = TransferDescriptor(
                device_type = DeviceType.SSD,
                physical_block_ids=ssd_blocks,
            ),
            dst_descriptor = TransferDescriptor(
                device_type = DeviceType.CPU,
                physical_block_ids=cpu_blocks[-len(ssd_blocks):]
            ),
            layer_id = 0,
            layer_granularity = layer_num
        )
        graph.add_transfer_op(op1)
        op2 = TransferOp(
            transfer_op_id = TransferIDAllocator.allocate_op_id(),
            transfer_graph_id = graph.transfer_graph_id,
            transfer_type = TransferType.H2D,
            src_descriptor = TransferDescriptor(
                device_type = DeviceType.CPU,
                physical_block_ids=cpu_blocks[-len(ssd_blocks):]
            ),
            dst_descriptor = TransferDescriptor(
                device_type = DeviceType.GPU,
                physical_block_ids=gpu_blocks[-len(ssd_blocks):],
                device_id = gpu_device_id
            ),
            layer_id = 0,
            layer_granularity = layer_num
        )
        graph.add_transfer_op(op2)
        op3 = TransferOp(
            transfer_op_id = TransferIDAllocator.allocate_op_id(),
            transfer_graph_id = graph.transfer_graph_id,
            transfer_type = TransferType.H2D,
            src_descriptor = TransferDescriptor(
                device_type = DeviceType.CPU,
                physical_block_ids=cpu_blocks[:len(cpu_blocks) - len(ssd_blocks)]
            ),
            dst_descriptor = TransferDescriptor(
                device_type = DeviceType.GPU,
                physical_block_ids=gpu_blocks[:len(cpu_blocks) - len(ssd_blocks)],
                device_id = gpu_device_id
            ),
            layer_id = 0,
            layer_granularity = layer_num
        )
        graph.add_transfer_op(op3)

        graph.add_dependency(op2.transfer_op_id, op1.transfer_op_id)
        return graph, [op2.transfer_op_id, op3.transfer_op_id]
    else:
        op1 = TransferOp(
            transfer_op_id = TransferIDAllocator.allocate_op_id(),
            transfer_graph_id = graph.transfer_graph_id,
            transfer_type=TransferType.DISK2H,
            src_descriptor=TransferDescriptor(
                device_type=DeviceType.SSD,
                physical_block_ids=ssd_blocks,
            ),
            dst_descriptor=TransferDescriptor(
                device_type=DeviceType.CPU,
                physical_block_ids=cpu_blocks,
            ),
            layer_id = 0,
            layer_granularity = layer_num
        )
        graph.add_transfer_op(op1)
        op2 = TransferOp(
            transfer_op_id = TransferIDAllocator.allocate_op_id(),
            transfer_graph_id = graph.transfer_graph_id,
            transfer_type=TransferType.H2D,
            src_descriptor=TransferDescriptor(
                device_type=DeviceType.CPU,
                physical_block_ids=cpu_blocks,
            ),
            dst_descriptor=TransferDescriptor(
                device_type=DeviceType.GPU,
                physical_block_ids=gpu_blocks,
            ),
            layer_id = 0,
            layer_granularity = layer_num
        )
        graph.add_transfer_op(op2)
        graph.add_dependency(op2.transfer_op_id, op1.transfer_op_id)
        return graph, [op2.transfer_op_id]

def create_read_graph_cpu_ssd_remote(
    gpu_blocks: torch.Tensor,
    cpu_blocks: torch.Tensor,
    ssd_blocks: torch.Tensor,
    remote_blocks: torch.Tensor,
    gpu_device_id: int = 0,
    layer_num: int = 1,
    layer_granularity: int = 1,
    write_back_to_ssd: bool = True,
)->Tuple[TransferOpGraph, List[int]]:
    """
    Create a read transfer graph with (REMOTE_STORAGE + SSD)->CPU->GPU operations
    Returns:
        graph: TransferOpGraph
        finished_ops_ids: List[int]: a list of transfer ops that can indicate
        the completion of each layer or each layer for each tp rank
    """
    graph = TransferOpGraph(TransferIDAllocator.allocate_graph_id())
    graph, finished_ops_ids = create_read_graph_cpu_storage(graph=graph,
                                  gpu_blocks=gpu_blocks[:-len(remote_blocks)],
                                  cpu_blocks=cpu_blocks[:-len(remote_blocks)],
                                  storage_blocks=ssd_blocks[:-len(remote_blocks)],
                                  gpu_device_id=gpu_device_id,
                                  layer_num=layer_num)
    op_r2h = TransferOp(
        transfer_op_id = TransferIDAllocator.allocate_op_id(),
        transfer_graph_id = graph.transfer_graph_id,
        transfer_type = TransferType.REMOTE2H,
        src_descriptor = TransferDescriptor(
            device_type = DeviceType.REMOTE,
            physical_block_ids=remote_blocks,
        ),
        dst_descriptor = TransferDescriptor(
            device_type = DeviceType.CPU,
            physical_block_ids=cpu_blocks[-len(remote_blocks):],
        ),
        layer_id = 0,
        layer_granularity = layer_wise_transfer_num
    )
    graph.add_transfer_op(op_r2h)
    op_h2d = TransferOp(
        transfer_op_id = TransferIDAllocator.allocate_op_id(),
        transfer_graph_id = graph.transfer_graph_id,
        transfer_type = TransferType.H2D,
        src_descriptor = TransferDescriptor(
            device_type = DeviceType.CPU,
            physical_block_ids=cpu_blocks[-len(remote_blocks):],
        ),
        dst_descriptor = TransferDescriptor(
            device_type = DeviceType.GPU,
            physical_block_ids=gpu_blocks[-len(remote_blocks):],
            device_id = gpu_device_id
        ),
        layer_id = 0,
        layer_granularity = layer_wise_transfer_num
    )
    graph.add_transfer_op(op_h2d)
    graph.add_dependency(op_h2d.transfer_op_id, op_r2h.transfer_op_id)
    if write_back_to_ssd:
        op_h2disk = TransferOp(
            transfer_op_id = TransferIDAllocator.allocate_op_id(),
            transfer_graph_id = graph.transfer_graph_id,
            transfer_type = TransferType.H2DISK,
            src_descriptor = TransferDescriptor(
                device_type = DeviceType.CPU,
                physical_block_ids=cpu_blocks[-len(remote_blocks):],
            ),
            dst_descriptor = TransferDescriptor(
                device_type = DeviceType.SSD,
                physical_block_ids=ssd_blocks[-len(remote_blocks):],
            ),
            layer_id = 0,
            layer_granularity = layer_wise_transfer_num
        )
        graph.add_transfer_op(op_h2disk)
        graph.add_dependency(op_h2disk.transfer_op_id, op_r2h.transfer_op_id)
    finished_ops_ids.append(op_h2d.transfer_op_id)
    return graph, finished_ops_ids

def create_write_graph_cpu_storage(
    graph: TransferOpGraph = None,
    gpu_blocks: torch.Tensor,
    cpu_blocks: torch.Tensor,
    storage_blocks: torch.Tensor,
    gpu_device_id: int = 0,
    layer_num: int = 1,
)->Tuple[TransferOpGraph, List[int]]:
    """
    Create a write transfer graph with CPU->REMOTE_STORAGE / SSD operations
    storage_blocks: the blocks of storage that are used as a lower-level storage backend,
    including ssd or remote storage. This can be empty, which means cpu-only kvcache. 
    Write op granularity is larger: gpu->cpu is put into the same op.
    Returns:
        graph: TransferOpGraph
        layer_wise_ops: List[int]: a list of transfer ops that can indicate
        the completion of each layer or each layer for each tp rank
    """
    if graph is None:
        graph = TransferOpGraph(TransferIDAllocator.allocate_graph_id())
    op_d2h = TransferOp(
        transfer_op_id = TransferIDAllocator.allocate_op_id(),
        transfer_graph_id = graph.transfer_graph_id,
        transfer_type = TransferType.D2H,
        src_descriptor = TransferDescriptor(
            device_type = DeviceType.GPU,
            physical_block_ids=gpu_blocks,
            device_id = gpu_device_id
        ),
        dst_descriptor = TransferDescriptor(
            device_type = DeviceType.CPU,
            physical_block_ids=cpu_blocks[-len(gpu_blocks):],
        ),
        layer_id = 0,
        layer_granularity = layer_num
    )
    graph.add_transfer_op(op_d2h)
    if len(storage_blocks) == 0:
        return graph, [op_d2h.transfer_op_id]
    else:
        op_h2disk = TransferOp(
            transfer_op_id = TransferIDAllocator.allocate_op_id(),
            transfer_graph_id = graph.transfer_graph_id,
            transfer_type = TransferType.H2DISK,
            src_descriptor = TransferDescriptor(
                device_type = DeviceType.CPU,
                physical_block_ids=cpu_blocks[-len(storage_blocks):],
            ),
            dst_descriptor = TransferDescriptor(
                device_type = DeviceType.SSD,
                physical_block_ids=storage_blocks,
            ),
            layer_id = 0,
            layer_granularity = layer_num
        )
        graph.add_transfer_op(op_h2disk)
        graph.add_dependency(op_h2disk.transfer_op_id, op_d2h.transfer_op_id)
        return graph, [op_h2disk.transfer_op_id]

def create_write_graph_cpu_ssd_remote(
    gpu_blocks: torch.Tensor,
    cpu_blocks: torch.Tensor,
    ssd_blocks: torch.Tensor,
    remote_blocks: torch.Tensor,
    gpu_device_id: int = 0,
    layer_num: int = 1,
)->Tuple[TransferOpGraph, List[int]]:
    """
    Create a write transfer graph with CPU->REMOTE_STORAGE + SSD operations
    Returns:
        graph: TransferOpGraph
        layer_wise_ops: List[int]: a list of transfer ops that can indicate
        the completion of each layer or each layer for each tp rank
    """
    graph = TransferOpGraph(TransferIDAllocator.allocate_graph_id())
    op_d2h = TransferOp(
        transfer_op_id = TransferIDAllocator.allocate_op_id(),
        transfer_graph_id = graph.transfer_graph_id,
        transfer_type = TransferType.D2H,
        src_descriptor = TransferDescriptor(
            device_type = DeviceType.GPU,
            physical_block_ids=gpu_blocks,
            device_id = gpu_device_id
        ),
        dst_descriptor = TransferDescriptor(
            device_type = DeviceType.CPU,
            physical_block_ids=cpu_blocks[-len(gpu_blocks):],
        ),
        layer_id = 0,
        layer_granularity = layer_num
    )
    graph.add_transfer_op(op_d2h)
    if len(ssd_blocks) != 0:
        op_h2disk = TransferOp(
            transfer_op_id = TransferIDAllocator.allocate_op_id(),
            transfer_graph_id = graph.transfer_graph_id,
            transfer_type = TransferType.H2DISK,
            src_descriptor = TransferDescriptor(
                device_type = DeviceType.CPU,
                physical_block_ids=cpu_blocks[-len(gpu_blocks):],
            ),
            dst_descriptor = TransferDescriptor(
                device_type = DeviceType.SSD,
                physical_block_ids=ssd_blocks,
            ),
            layer_id = 0,
            layer_granularity = layer_num
        )
        graph.add_transfer_op(op_h2disk)
        graph.add_dependency(op_h2disk.transfer_op_id, op_d2h.transfer_op_id)
    if len(remote_blocks) != 0:
        op_h2remote = TransferOp(
            transfer_op_id = TransferIDAllocator.allocate_op_id(),
            transfer_graph_id = graph.transfer_graph_id,
            transfer_type = TransferType.H2REMOTE,
            src_descriptor = TransferDescriptor(
                device_type = DeviceType.CPU,
                physical_block_ids=cpu_blocks[-len(remote_blocks):],
            ),
            dst_descriptor = TransferDescriptor(
                device_type = DeviceType.REMOTE,
                physical_block_ids=remote_blocks,
            ),
            layer_id = 0,
            layer_granularity = layer_num
        )
        graph.add_transfer_op(op_h2remote)
        graph.add_dependency(op_h2remote.transfer_op_id, op_d2h.transfer_op_id)
    return graph, [op_d2h.transfer_op_id]

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
    new_graph = TransferOpGraph(TransferIDAllocator.allocate_graph_id())
    # Map from original op id to a list of new op ids (length = num_splits)
    opid2splitopids = {}
    layer_wise_virtual_op_ids = []
    # Split all ops
    for op_id, op in transfer_graph._op_map.items():
        split_op_ids = []
        for i in range(num_splits):
            # Copy op, modify layer_id and layer_granularity
            new_op = TransferOp(
                transfer_op_id=TransferIDAllocator.allocate_op_id(),
                transfer_graph_id=new_graph.transfer_graph_id,
                transfer_type=op.transfer_type,
                src_descriptor=op.src_descriptor,
                dst_descriptor=op.dst_descriptor,
                layer_id=i * layer_granularity,
                layer_granularity=layer_granularity,
                # Inherit these fields directly
                tp_rank=op.tp_rank,
                tp_world_size=op.tp_world_size,
            )
            new_graph.add_transfer_op(new_op)
            split_op_ids.append(new_op.transfer_op_id)
        opid2splitopids[op_id] = split_op_ids

    #add virtual ops that mark the finish of each layer
    for i in range(num_splits):
        op = TransferOp(
            transfer_op_id = TransferIDAllocator.allocate_op_id(),
            transfer_graph_id = new_graph.transfer_graph_id,
            transfer_type = TransferType.VIRTUAL
        )
        new_graph.add_transfer_op(op)
        layer_wise_virtual_op_ids.append(op.transfer_op_id)
        # add dependencies between virtual ops and finished ops
        for h2d_op in finished_ops_ids:
            new_graph.add_dependency(op.transfer_op_id, opid2splitopids[h2d_op][i])

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
        for i in range(1, num_splits):
            new_graph.add_dependency(split_op_ids[i], split_op_ids[i - 1])

    return new_graph, layer_wise_virtual_op_ids

