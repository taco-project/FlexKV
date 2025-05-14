from typing import List, Tuple

import torch

from flexkv.common.transfer import DeviceType, TransferType
from flexkv.common.transfer import TransferOp, TransferOpGraph, TransferDescriptor, TransferIDAllocator


def create_read_transfer_graph(
    ssd_blocks: torch.Tensor,
    cpu_blocks: torch.Tensor,
    gpu_blocks: torch.Tensor,
    gpu_device_id: int = 0,
    layer_num: int = 1,
    layer_granularity: int = 1,
) -> Tuple[TransferOpGraph, List[int]]:
    """
    Create a transfer graph with SSD->CPU->GPU operations
    Returns:
        graph: TransferOpGraph
        layer_wise_ops: List[int]: a list of transfer ops that can indicate
        the completion of each layer or each layer for each tp rank
    """
    graph = TransferOpGraph(TransferIDAllocator.allocate_graph_id())
    assert layer_num % layer_granularity == 0
    layer_wise_transfer_num = layer_num // layer_granularity
    if len(ssd_blocks) == 0:
        layer_wise_ops = []
        for i in range(layer_wise_transfer_num):
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
                layer_id = i * layer_granularity,
                layer_granularity = layer_granularity
            )
            graph.add_transfer_op(op)
            layer_wise_ops.append(op.transfer_op_id)
            if (i > 0):
                graph.add_dependency(op.transfer_op_id, layer_wise_ops[i - 1])
        return graph, layer_wise_ops
    elif len(ssd_blocks) < len(cpu_blocks):
        cpu_layer_wise_ops = []
        cpu_ssd_depend_ops = []
        ssd_layer_wise_ops = []
        virtual_ops = []
        for i in range(layer_wise_transfer_num):
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
                layer_id = i * layer_granularity,
                layer_granularity = layer_granularity
            )
            graph.add_transfer_op(op1)
            ssd_layer_wise_ops.append(op1.transfer_op_id)
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
                layer_id = i * layer_granularity,
                layer_granularity = layer_granularity
            )
            graph.add_transfer_op(op2)
            cpu_ssd_depend_ops.append(op2.transfer_op_id)
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
                layer_id = i * layer_granularity,
                layer_granularity = layer_granularity
            )
            graph.add_transfer_op(op3)
            cpu_layer_wise_ops.append(op3.transfer_op_id)
            op4 = TransferOp(
                transfer_op_id = TransferIDAllocator.allocate_op_id(),
                transfer_graph_id = graph.transfer_graph_id,
                transfer_type = TransferType.VIRTUAL
            )
            graph.add_transfer_op(op4)
            virtual_ops.append(op4.transfer_op_id)

            graph.add_dependency(op2.transfer_op_id, op1.transfer_op_id)
            graph.add_dependency(op4.transfer_op_id, op2.transfer_op_id)
            graph.add_dependency(op4.transfer_op_id, op3.transfer_op_id)
            if (i > 0):
                graph.add_dependency(op1.transfer_op_id, ssd_layer_wise_ops[i - 1])
                graph.add_dependency(op2.transfer_op_id, cpu_ssd_depend_ops[i - 1])
                graph.add_dependency(op3.transfer_op_id, cpu_layer_wise_ops[i - 1])
        return graph, virtual_ops
    else:
        cpu_layer_wise_op_ids = []
        ssd_layer_wise_op_ids = []
        for i in range(layer_wise_transfer_num):
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
                layer_id = i * layer_granularity,
                layer_granularity = layer_granularity
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
                layer_id = i * layer_granularity,
                layer_granularity = layer_granularity
            )
            graph.add_transfer_op(op2)
            cpu_layer_wise_op_ids.append(op2.transfer_op_id)
            ssd_layer_wise_op_ids.append(op1.transfer_op_id)
            graph.add_dependency(op2.transfer_op_id, op1.transfer_op_id)
            if (i > 0):
                graph.add_dependency(op2.transfer_op_id, cpu_layer_wise_op_ids[i - 1])
                graph.add_dependency(op1.transfer_op_id, ssd_layer_wise_op_ids[i - 1])
        return graph, cpu_layer_wise_op_ids

# NOTE write through now
# now we don't support layer-wise write
def create_write_transfer_graph(
    ssd_blocks: torch.Tensor,
    cpu_blocks: torch.Tensor,
    gpu_blocks: torch.Tensor,
    gpu_device_id: int = 0,
    layer_num: int = 1
) -> TransferOpGraph:
    assert len(gpu_blocks) == len(cpu_blocks)
    graph = TransferOpGraph(TransferIDAllocator.allocate_graph_id())
    if len(ssd_blocks) == 0:
        op1 = TransferOp(
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
                physical_block_ids=cpu_blocks,
            ),
            layer_id = 0,  # all layers
            layer_granularity = layer_num
        )
        graph.add_transfer_op(op1)
        return graph
    elif len(ssd_blocks) < len(cpu_blocks):
        op1 = TransferOp(
            transfer_op_id = TransferIDAllocator.allocate_op_id(),
            transfer_graph_id = graph.transfer_graph_id,
            transfer_type = TransferType.D2H,
            src_descriptor = TransferDescriptor(
                device_type = DeviceType.GPU,
                physical_block_ids=gpu_blocks[-len(ssd_blocks):],
                device_id = gpu_device_id
            ),
            dst_descriptor = TransferDescriptor(
                device_type = DeviceType.CPU,
                physical_block_ids=cpu_blocks[-len(ssd_blocks):],
            ),
            layer_id = 0,  # all layers
            layer_granularity = layer_num
        )
        graph.add_transfer_op(op1)
        op2 = TransferOp(
            transfer_op_id = TransferIDAllocator.allocate_op_id(),
            transfer_graph_id = graph.transfer_graph_id,
            transfer_type = TransferType.H2DISK,
            src_descriptor = TransferDescriptor(
                device_type = DeviceType.CPU,
                physical_block_ids=cpu_blocks[-len(ssd_blocks):],
            ),
            dst_descriptor = TransferDescriptor(
                device_type = DeviceType.SSD,
                physical_block_ids=ssd_blocks,
            ),
            layer_id = 0,  # all layers
            layer_granularity = layer_num
        )
        graph.add_transfer_op(op2)
        graph.add_dependency(op2.transfer_op_id, op1.transfer_op_id)
        op3 = TransferOp(
            transfer_op_id = TransferIDAllocator.allocate_op_id(),
            transfer_graph_id = graph.transfer_graph_id,
            transfer_type = TransferType.D2H,
            src_descriptor = TransferDescriptor(
                device_type = DeviceType.GPU    ,
                physical_block_ids=gpu_blocks[:len(cpu_blocks) - len(ssd_blocks)],
                device_id = gpu_device_id
            ),
            dst_descriptor = TransferDescriptor(
                device_type = DeviceType.CPU,
                physical_block_ids=cpu_blocks[:len(cpu_blocks) - len(ssd_blocks)],
            ),
            layer_id = 0,  # all layers
            layer_granularity = layer_num
        )
        graph.add_transfer_op(op3)
        #op4 = TransferOp(
        #    transfer_op_id = TransferIDAllocator.allocate_op_id(),
        #    transfer_graph_id = graph.transfer_graph_id,
        #    transfer_type = TransferType.VIRTUAL
        #)
        #graph.add_transfer_op(op4)
        #graph.add_dependency(op4.transfer_op_id, op3.transfer_op_id)
        #graph.add_dependency(op4.transfer_op_id, op1.transfer_op_id)
        return graph
    else:
        op1 = TransferOp(
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
                physical_block_ids=cpu_blocks,
            ),
            layer_id = 0,  # all layers
            layer_granularity = layer_num
        )
        graph.add_transfer_op(op1)
        op2 = TransferOp(
            transfer_op_id = TransferIDAllocator.allocate_op_id(),
            transfer_graph_id = graph.transfer_graph_id,
            transfer_type = TransferType.H2DISK,
            src_descriptor = TransferDescriptor(
                device_type = DeviceType.CPU,
                physical_block_ids=cpu_blocks,
            ),
            dst_descriptor = TransferDescriptor(
                device_type = DeviceType.SSD,
                physical_block_ids=ssd_blocks,
            ),
            layer_id = 0,  # all layers
            layer_granularity = layer_num
        )
        graph.add_transfer_op(op2)
        graph.add_dependency(op2.transfer_op_id, op1.transfer_op_id)
        return graph
