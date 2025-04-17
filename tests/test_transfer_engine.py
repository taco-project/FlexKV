import torch
import os
from typing import List, Tuple
from flexkv.storage.storage_engine import StorageEngine
from flexkv.transfer.transfer_engine import TransferEngine
from flexkv.common.transfer import (
    TransferOp, TransferOpGraph, DeviceType, 
    TransferType, TransferDescriptor
)
from flexkv.common.storage import AccessibleHandle, AccessHandleType
from flexkv.cache.transfer_pattern import (
    create_read_transfer_graph,
    create_write_transfer_graph
)

def create_test_data(layer_num: int, gpu_block_num: int, block_size: int):
    """Create test GPU blocks with known values"""
    gpu_blocks = [
        torch.randint(0, 100, 
                    size=(2, gpu_block_num, block_size),
                    dtype=torch.float16, 
                    device="cuda")
        for _ in range(layer_num)
    ]
    return gpu_blocks

def setup_storage_engine(
    gpu_blocks: List[torch.Tensor],
    token_per_block: int,
    head_num: int,
    head_dim: int,
    layer_num: int,
    gpu_block_num: int,
    cpu_block_num: int,
    ssd_block_num: int,
    ssd_path: str
) -> StorageEngine:
    """Setup storage engine with GPU, CPU and SSD allocators"""
    storage_engine = StorageEngine()
    
    # Create GPU allocator from existing blocks
    gpu_success = storage_engine.create_allocator(
        device_type=DeviceType.GPU,
        tensor_shape=(layer_num, 2, gpu_block_num, token_per_block * head_num * head_dim),
        dtype=torch.float16,
        device_id=0,
        raw_data=gpu_blocks
    )
    print(f"GPU allocator creation: {'success' if gpu_success else 'failed'}")
    
    # Create CPU allocator
    cpu_success = storage_engine.create_allocator(
        device_type=DeviceType.CPU,
        tensor_shape=(layer_num, 2, cpu_block_num, token_per_block * head_num * head_dim),
        dtype=torch.float16,
        pin_memory=True  # Use pinned memory for better transfer performance
    )
    print(f"CPU allocator creation: {'success' if cpu_success else 'failed'}")
    
    # Create SSD allocator
    ssd_success = storage_engine.create_allocator(
        device_type=DeviceType.SSD,
        tensor_shape=(layer_num, 2, ssd_block_num, token_per_block * head_num * head_dim),
        dtype=torch.float16,
        file_path=ssd_path
    )
    print(f"SSD allocator creation: {'success' if ssd_success else 'failed'}")
    
    return storage_engine

def main():
    # Test parameters
    ssd_path = "storage.cache"

    layer_num = 4
    gpu_block_num = 100
    cpu_block_num = 200
    ssd_block_num = 300

    token_per_block = 4
    head_num = 8
    head_dim = 128
    
    # Create test data
    print("Creating test data...")
    gpu_blocks = create_test_data(layer_num, 
                                  gpu_block_num, 
                                  token_per_block * head_num * head_dim)
    # Setup storage engine
    print("\nSetting up storage engine...")
    storage_engine = setup_storage_engine(
        gpu_blocks=gpu_blocks,
        token_per_block=token_per_block,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        gpu_block_num=gpu_block_num,
        cpu_block_num=cpu_block_num,
        ssd_block_num=ssd_block_num,
        ssd_path=ssd_path
    )
    
    # Get handles for transfer engine
    print("\nGetting handles...")
    gpu_handle = storage_engine.get_allocator_handle(
        device_type=DeviceType.GPU,
        device_id=0,
    )
    cpu_handle = storage_engine.get_allocator_handle(
        device_type=DeviceType.CPU,
    )
    
    ssd_handle = storage_engine.get_allocator_handle(
        device_type=DeviceType.SSD,
    )
    
    # Setup transfer engine
    print("\nSetting up transfer engine...")
    transfer_engine = TransferEngine(
        gpu_handles=[gpu_handle],
        cpu_handle=cpu_handle,
        ssd_handle=ssd_handle
    )
    
    print("\nCreating transfer graph...")
    tranfer_graph_num = 4
    gpu_transfer_block_num = 16
    assert gpu_transfer_block_num * tranfer_graph_num <= gpu_block_num
    gpu_block_ids = [
        [i for i in range(2 + gpu_transfer_block_num * j, 2 + gpu_transfer_block_num * (j+1))]
        for j in range(tranfer_graph_num)
    ]
    cpu_block_ids = [
        [i for i in range(3 + gpu_transfer_block_num * j, 3 + gpu_transfer_block_num * (j+1))]
        for j in range(tranfer_graph_num)
    ]
    ssd_block_ids = [
        [i for i in range(4 + gpu_transfer_block_num * j, 4 + gpu_transfer_block_num * (j+1))]
        for j in range(tranfer_graph_num)
    ]
    launched_graph_ids = []
    gpu_block_ground_truth = [
        gpu_blocks[i].clone()
        for i in range(layer_num)
    ]
    # Create and submit write transfer graph, write through ssd
    for i in range(tranfer_graph_num):
        transfer_graph_write = create_write_transfer_graph(
            ssd_blocks=ssd_block_ids[i],
            cpu_blocks=cpu_block_ids[i],
            gpu_blocks=gpu_block_ids[i]
        )
        transfer_engine.submit_transfer_graph(transfer_graph_write)
        launched_graph_ids.append(transfer_graph_write.transfer_graph_id)
    # Wait for completion
    print("Waiting for transfers to complete...")
    while True:
        completed_graphs = transfer_engine.get_completed_graphs(timeout=0.1)
        if len(completed_graphs) > 0:
            for completed_graph in completed_graphs:
                if completed_graph.transfer_graph_id in launched_graph_ids:
                    launched_graph_ids.remove(completed_graph.transfer_graph_id)
                    print(f"Transfer graph {completed_graph.transfer_graph_id} completed")
        if len(launched_graph_ids) == 0:
            break
    # Create and submit read transfer graph, ssd partial read
    for i in range(tranfer_graph_num):
        transfer_graph_read = create_read_transfer_graph(
            ssd_blocks=ssd_block_ids[i][:gpu_transfer_block_num//2] if i % 2 == 0 else [],
            cpu_blocks=cpu_block_ids[i],
            gpu_blocks=gpu_block_ids[i]
        )
        transfer_engine.submit_transfer_graph(transfer_graph_read)
        launched_graph_ids.append(transfer_graph_read.transfer_graph_id)
    # Wait for completion
    print("Waiting for transfers to complete...")
    while True:
        completed_graphs = transfer_engine.get_completed_graphs(timeout=0.1)
        if len(completed_graphs) > 0:
            for completed_graph in completed_graphs:
                if completed_graph.transfer_graph_id in launched_graph_ids:
                    launched_graph_ids.remove(completed_graph.transfer_graph_id)
                    print(f"Transfer graph {completed_graph.transfer_graph_id} completed")
        if len(launched_graph_ids) == 0:
            break    
    # Verify results
    print("\nVerifying results...")
    for i in range(layer_num):
        assert torch.allclose(gpu_blocks[i], gpu_block_ground_truth[i])
    print("\nTest completed")
    # Cleanup
    print("\nCleaning up...")
    transfer_engine.shutdown()
    print("Test completed")

if __name__ == "__main__":
    main()