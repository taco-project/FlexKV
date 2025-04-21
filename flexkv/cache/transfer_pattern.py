from typing import List

import torch

from flexkv.common.transfer import TransferOp, TransferOpGraph, DeviceType, TransferType, TransferDescriptor


global_transfer_op_id = 0
global_transfer_graph_id = 0

def create_read_transfer_graph(
    ssd_blocks: List[int],
    cpu_blocks: List[int],
    gpu_blocks: List[int],
    gpu_device_id: int = 0,
) -> TransferOpGraph:
    """Create a transfer graph with SSD->CPU->GPU operations"""
    global global_transfer_op_id
    global global_transfer_graph_id
    graph = TransferOpGraph(global_transfer_graph_id)
    if len(ssd_blocks) == 0:
        op1 = TransferOp(
            transfer_op_id = global_transfer_op_id,
            transfer_graph_id = global_transfer_graph_id,
            transfer_type = TransferType.H2D,
            src_descriptor = TransferDescriptor(
                device_type = DeviceType.CPU,
                physical_block_ids=torch.tensor(cpu_blocks),
            ),
            dst_descriptor = TransferDescriptor(
                device_type = DeviceType.GPU,
                physical_block_ids=torch.tensor(gpu_blocks),
                device_id = gpu_device_id
            )
        )
        graph.add_transfer_op(op1)
        global_transfer_op_id += 1
    elif len(ssd_blocks) < len(cpu_blocks):
        op1 = TransferOp(
            transfer_op_id = global_transfer_op_id,
            transfer_graph_id = global_transfer_graph_id,
            transfer_type = TransferType.DISK2H,
            src_descriptor = TransferDescriptor(
                device_type = DeviceType.SSD,
                physical_block_ids=torch.tensor(ssd_blocks)
            ),
            dst_descriptor = TransferDescriptor(
                device_type = DeviceType.CPU,
                physical_block_ids=torch.tensor(cpu_blocks[-len(ssd_blocks):])
            )
        )
        graph.add_transfer_op(op1)
        global_transfer_op_id += 1
        op2 = TransferOp(
            transfer_op_id = global_transfer_op_id,
            transfer_graph_id = global_transfer_graph_id,
            transfer_type = TransferType.H2D,
            src_descriptor = TransferDescriptor(
                device_type = DeviceType.CPU,
                physical_block_ids=torch.tensor(cpu_blocks[-len(ssd_blocks):])
            ),
            dst_descriptor = TransferDescriptor(
                device_type = DeviceType.GPU,
                physical_block_ids=torch.tensor(gpu_blocks[-len(ssd_blocks):]),
                device_id = gpu_device_id
            )
        )
        graph.add_transfer_op(op2)
        global_transfer_op_id += 1
        op3 = TransferOp(
            transfer_op_id = global_transfer_op_id,
            transfer_graph_id = global_transfer_graph_id,
            transfer_type = TransferType.H2D,
            src_descriptor = TransferDescriptor(
                device_type = DeviceType.CPU,
                physical_block_ids=torch.tensor(cpu_blocks[:len(cpu_blocks) - len(ssd_blocks)])
            ),
            dst_descriptor = TransferDescriptor(
                device_type = DeviceType.GPU,
                physical_block_ids=torch.tensor(gpu_blocks[:len(cpu_blocks) - len(ssd_blocks)]),
                device_id = gpu_device_id
            )
        )
        graph.add_transfer_op(op3)
        global_transfer_op_id += 1
        graph.add_dependency(op2.transfer_op_id, op1.transfer_op_id)
    else:
        op1 = TransferOp(
            transfer_op_id = global_transfer_op_id,
            transfer_graph_id = global_transfer_graph_id,
            transfer_type=TransferType.DISK2H,
            src_descriptor=TransferDescriptor(
                device_type=DeviceType.SSD,
                physical_block_ids=torch.tensor(ssd_blocks)
            ),
            dst_descriptor=TransferDescriptor(
                device_type=DeviceType.CPU,
                physical_block_ids=torch.tensor(cpu_blocks)
            )
        )
        graph.add_transfer_op(op1)
        global_transfer_op_id += 1
        op2 = TransferOp(
            transfer_op_id = global_transfer_op_id,
            transfer_graph_id = global_transfer_graph_id,
            transfer_type=TransferType.H2D,
            src_descriptor=TransferDescriptor(
                device_type=DeviceType.CPU,
                physical_block_ids=torch.tensor(cpu_blocks)
            ),
            dst_descriptor=TransferDescriptor(
                device_type=DeviceType.GPU,
                physical_block_ids=torch.tensor(gpu_blocks)
            )
        )
        graph.add_transfer_op(op2)
        global_transfer_op_id += 1
        graph.add_dependency(op2.transfer_op_id, op1.transfer_op_id)
    global_transfer_graph_id += 1
    return graph

#NOTE write through now
def create_write_transfer_graph(
    ssd_blocks: List[int],
    cpu_blocks: List[int],
    gpu_blocks: List[int],
    gpu_device_id: int = 0,
) -> TransferOpGraph:
    assert len(gpu_blocks) == len(cpu_blocks)
    global global_transfer_op_id
    global global_transfer_graph_id
    graph = TransferOpGraph(global_transfer_graph_id)
    if len(ssd_blocks) == 0:
        op1 = TransferOp(
            transfer_op_id = global_transfer_op_id,
            transfer_graph_id = global_transfer_graph_id,
            transfer_type = TransferType.D2H,
            src_descriptor = TransferDescriptor(
                device_type = DeviceType.GPU,
                physical_block_ids=torch.tensor(gpu_blocks),
                device_id = gpu_device_id
            ),
            dst_descriptor = TransferDescriptor(
                device_type = DeviceType.CPU,
                physical_block_ids=torch.tensor(cpu_blocks),
            )
        )
        graph.add_transfer_op(op1)
        global_transfer_op_id += 1
    elif len(ssd_blocks) < len(cpu_blocks):
        op1 = TransferOp(
            transfer_op_id = global_transfer_op_id,
            transfer_graph_id = global_transfer_graph_id,
            transfer_type = TransferType.D2H,
            src_descriptor = TransferDescriptor(
                device_type = DeviceType.GPU,
                physical_block_ids=torch.tensor(gpu_blocks[-len(ssd_blocks):]),
                device_id = gpu_device_id
            ),
            dst_descriptor = TransferDescriptor(
                device_type = DeviceType.CPU,
                physical_block_ids=torch.tensor(cpu_blocks[-len(ssd_blocks):]),
                )
        )
        graph.add_transfer_op(op1)
        global_transfer_op_id += 1
        op2 = TransferOp(
            transfer_op_id = global_transfer_op_id,
            transfer_graph_id = global_transfer_graph_id,
            transfer_type = TransferType.H2DISK,
            src_descriptor = TransferDescriptor(
                device_type = DeviceType.CPU,
                physical_block_ids=torch.tensor(cpu_blocks[-len(ssd_blocks):]),
            ),
            dst_descriptor = TransferDescriptor(
                device_type = DeviceType.SSD,
                physical_block_ids=torch.tensor(ssd_blocks)
            )
        )
        graph.add_transfer_op(op2)
        global_transfer_op_id += 1
        graph.add_dependency(op2.transfer_op_id, op1.transfer_op_id)
        op3 = TransferOp(
            transfer_op_id = global_transfer_op_id,
            transfer_graph_id = global_transfer_graph_id,
            transfer_type = TransferType.D2H,
            src_descriptor = TransferDescriptor(
                device_type = DeviceType.GPU    ,
                physical_block_ids=torch.tensor(gpu_blocks[:len(cpu_blocks) - len(ssd_blocks)]),
                device_id = gpu_device_id
            ),
            dst_descriptor = TransferDescriptor(
                device_type = DeviceType.CPU,
                physical_block_ids=torch.tensor(cpu_blocks[:len(cpu_blocks) - len(ssd_blocks)]),
            )
        )
        graph.add_transfer_op(op3)
        global_transfer_op_id += 1
    else:
        op1 = TransferOp(
            transfer_op_id = global_transfer_op_id,
            transfer_graph_id = global_transfer_graph_id,
            transfer_type = TransferType.D2H,
            src_descriptor = TransferDescriptor(
                device_type = DeviceType.GPU,
                physical_block_ids=torch.tensor(gpu_blocks),
                device_id = gpu_device_id
            ),
            dst_descriptor = TransferDescriptor(
                device_type = DeviceType.CPU,
                physical_block_ids=torch.tensor(cpu_blocks)
            )
        )
        graph.add_transfer_op(op1)
        global_transfer_op_id += 1
        op2 = TransferOp(
            transfer_op_id = global_transfer_op_id,
            transfer_graph_id = global_transfer_graph_id,
            transfer_type = TransferType.H2DISK,
            src_descriptor = TransferDescriptor(
                device_type = DeviceType.CPU,
                physical_block_ids=torch.tensor(cpu_blocks),
            ),
            dst_descriptor = TransferDescriptor(
                device_type = DeviceType.SSD,
                physical_block_ids=torch.tensor(ssd_blocks)
            )
        )
        graph.add_transfer_op(op2)
        global_transfer_op_id += 1
        graph.add_dependency(op2.transfer_op_id, op1.transfer_op_id)
    global_transfer_graph_id += 1
    return graph
