from typing import List

import torch

from flexkv.common.transfer import DeviceType, TransferType
from flexkv.common.transfer import TransferOp, TransferOpGraph, TransferDescriptor, TransferIDAllocator


def create_read_transfer_graph(
    ssd_blocks: torch.Tensor,
    cpu_blocks: torch.Tensor,
    gpu_blocks: torch.Tensor,
    gpu_device_id: int = 0,
) -> TransferOpGraph:
    """Create a transfer graph with SSD->CPU->GPU operations"""
    graph = TransferOpGraph(TransferIDAllocator.allocate_graph_id())
    graph.block_meta_to_free = {}
    if len(cpu_blocks) != 0:
        graph.block_meta_to_free[DeviceType.CPU] = cpu_blocks
    if len(ssd_blocks) != 0:
        graph.block_meta_to_free[DeviceType.SSD] = ssd_blocks
    if len(ssd_blocks) == 0:
        op1 = TransferOp(
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
            )
        )
        graph.add_transfer_op(op1)
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
            )
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
            )
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
            )
        )
        graph.add_transfer_op(op3)
        graph.add_dependency(op2.transfer_op_id, op1.transfer_op_id)
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
            )
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
            )
        )
        graph.add_transfer_op(op2)
        graph.add_dependency(op2.transfer_op_id, op1.transfer_op_id)
    return graph

#NOTE write through now
def create_write_transfer_graph(
    ssd_blocks: torch.Tensor,
    cpu_blocks: torch.Tensor,
    gpu_blocks: torch.Tensor,
    gpu_device_id: int = 0,
) -> TransferOpGraph:
    assert len(gpu_blocks) == len(cpu_blocks)
    graph = TransferOpGraph(TransferIDAllocator.allocate_graph_id())
    graph.block_meta_to_free = {}
    if len(ssd_blocks) != 0:
        graph.block_meta_to_free[DeviceType.SSD] = ssd_blocks
    if len(cpu_blocks) != 0:
        graph.block_meta_to_free[DeviceType.CPU] = cpu_blocks
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
            )
        )
        graph.add_transfer_op(op1)
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
                )
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
            )
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
            )
        )
        graph.add_transfer_op(op3)
        graph.add_dependency(op3.transfer_op_id, op2.transfer_op_id)
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
            )
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
            )
        )
        graph.add_transfer_op(op2)
        graph.add_dependency(op2.transfer_op_id, op1.transfer_op_id)
    return graph
