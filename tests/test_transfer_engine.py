"""
Transfer Engine Unit Tests

This module contains comprehensive unit tests for the TransferEngine component,
which handles data transfers between different storage tiers (GPU, CPU, SSD).

Test Functions Overview:
1. test_gpu_cpu_round_trip: Tests round-trip data transfers between GPU and CPU
   - Parameterized by: tp_size, dp_size, num_gpu_blocks, transfer_block_num
   - Validates data consistency after GPU->CPU->GPU transfers

2. test_ssd_round_trip: Tests round-trip data transfers involving SSD storage
   - Parameterized by: num_gpu_blocks, transfer_block_num, enable_ssd_cache
   - Validates data consistency after GPU->CPU->SSD->CPU->GPU transfers

3. test_concurrent_mixed_transfers: Tests multiple concurrent read/write transfers
   - Parameterized by: num_concurrent_transfers, blocks_per_transfer, include_ssd
   - Validates correctness of mixed read/write transfer graphs running concurrently

usage example:
    python -m pytest tests/test_transfer_engine.py::test_gpu_cpu_round_trip -v --tb=short
Each test validates both transfer completion and data correctness to ensure
the TransferEngine maintains data integrity across all transfer operations.
"""

import os
import time
import tempfile
from typing import List, Dict, Tuple
import multiprocessing as mp

import pytest
import torch

from flexkv.cache.transfer_pattern import (
    create_read_graph_cpu_storage,
    create_write_graph_cpu_storage,
    create_read_graph_cpu_ssd_remote,
    create_write_graph_cpu_ssd_remote
)
from flexkv.common.config import ModelConfig, CacheConfig
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import DeviceType, TransferIDAllocator
from flexkv.storage.storage_engine import StorageEngine
from flexkv.transfer.transfer_engine import TransferEngine


@pytest.fixture
def temp_cache_files():
    """Create temporary cache files for testing"""
    temp_files = []
    for i in range(2):
        fd, path = tempfile.mkstemp(suffix=f'_cache_{i}.tmp')
        os.close(fd)
        temp_files.append(path)
    
    yield temp_files
    
    # Cleanup temporary files
    for path in temp_files:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


@pytest.fixture
def model_config():
    """Create default model configuration"""
    return ModelConfig(
        num_layers=16,
        num_kv_heads=32,
        head_size=128,
        dtype=torch.float16,
        use_mla=False,
        tp_size=1,
        dp_size=1
    )


@pytest.fixture
def cache_config(temp_cache_files):
    """Create default cache configuration"""
    return CacheConfig(
        tokens_per_block=16,
        enable_cpu=True,
        enable_ssd=True,
        enable_remote=False,
        num_cpu_blocks=128,
        num_ssd_blocks=256,
        num_remote_blocks=256,
        use_gds=False,
        use_pinned_memory=True,
        ssd_cache_path=temp_cache_files,
        remote_cache_path=["remote_cache1", "remote_cache2"],
        remote_config_custom={
            "pcfs_fsid": "f_l91fz6",
            "pcfs_port": 31,
            "pcfs_ip": "172.21.16.177",
            "pcfs_parent_nodeid": 144115188075855883
        }
    )


def create_test_gpu_blocks(model_config: ModelConfig, cache_config: CacheConfig, num_gpu_blocks: int) -> Dict[int, List[torch.Tensor]]:
    """Create test GPU block data with random values"""
    gpu_blocks = {}
    for gpu_id in range(model_config.tp_size * model_config.dp_size):
        gpu_blocks[gpu_id] = []
        device_name = f"cuda:{gpu_id}" if gpu_id < torch.cuda.device_count() else "cuda:0"
        
        for layer_id in range(model_config.num_layers):
            # Create KV cache tensor: [2, num_blocks, tokens_per_block, num_heads, head_size]
            kv_tensor = torch.randint(
                0, 100,
                size=(2, num_gpu_blocks, cache_config.tokens_per_block, 
                      model_config.num_kv_heads // model_config.tp_size, model_config.head_size),
                dtype=model_config.dtype,
                device=device_name
            )
            gpu_blocks[gpu_id].append(kv_tensor)
    
    return gpu_blocks


def setup_kv_layouts(model_config: ModelConfig, cache_config: CacheConfig, num_gpu_blocks: int):
    """Setup KV cache layouts for different storage tiers"""
    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERWISE,
        num_layer=model_config.num_layers,
        num_block=num_gpu_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads // model_config.tp_size,
        head_size=model_config.head_size,
        is_mla=model_config.use_mla
    )
    
    cpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERWISE,
        num_layer=model_config.num_layers,
        num_block=cache_config.num_cpu_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        is_mla=model_config.use_mla
    )
    
    ssd_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERWISE,
        num_layer=model_config.num_layers,
        num_block=cache_config.num_ssd_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        is_mla=model_config.use_mla
    )
    
    cache_config.gpu_kv_layout = gpu_layout
    cache_config.cpu_kv_layout = cpu_layout
    cache_config.ssd_kv_layout = ssd_layout


class DataVerifier:
    """Utility class for verifying data transfer correctness"""
    
    @staticmethod
    def verify_gpu_cpu_transfer(
        gpu_ground_truth: Dict[int, List[torch.Tensor]], 
        cpu_blocks: List[torch.Tensor], 
        gpu_block_ids: torch.Tensor, 
        cpu_block_ids: torch.Tensor, 
        model_config: ModelConfig, 
        cache_config: CacheConfig,
        start_gpu_id: int,
        gpu_num: int
    ):
        """Verify correctness of GPU to CPU data transfer"""
        cpu_block_num = cache_config.num_cpu_blocks
        tokens_per_block = cache_config.tokens_per_block
        num_heads = model_config.num_kv_heads
        head_size = model_config.head_size
        
        for gpu_id in range(gpu_num):
            start_head = gpu_id * model_config.num_kv_heads // gpu_num
            end_head = (gpu_id + 1) * model_config.num_kv_heads // gpu_num
            gpu_layers = gpu_ground_truth[start_gpu_id + gpu_id]
            for layer_id in range(model_config.num_layers):
                gpu_gt_data = gpu_layers[layer_id][:, gpu_block_ids, :, :, :].cpu()
                cpu_data = cpu_blocks[layer_id].reshape(2, cpu_block_num, tokens_per_block, num_heads, head_size)
                gpu_data_cpu = cpu_data[:, cpu_block_ids, :,start_head:end_head, :]
                assert torch.allclose(gpu_data_cpu, gpu_gt_data, rtol=1e-5, atol=1e-6), \
                    f"GPU->CPU data mismatch at gpu_id {gpu_id}, layer {layer_id}"

    @staticmethod
    def verify_round_trip_consistency(
        original_data: Dict[int, List[torch.Tensor]], 
        final_data: Dict[int, List[torch.Tensor]], 
        block_ids: torch.Tensor,
        model_config: ModelConfig,
        gpu_id: int = 0
    ):
        """Verify that round-trip transfer maintains data consistency"""
        for layer_id in range(model_config.num_layers):
            for kv_idx in range(2):
                for block_id in block_ids:
                    original_block = original_data[gpu_id][layer_id][kv_idx, block_id]
                    final_block = final_data[gpu_id][layer_id][kv_idx, block_id]
                    
                    assert torch.allclose(original_block, final_block, rtol=1e-5, atol=1e-6), \
                        f"Round-trip consistency failed at layer {layer_id}, kv {kv_idx}, block {block_id}"


def wait_for_transfer_completion(transfer_engine: TransferEngine, expected_graph_ids: List[int], max_wait_time: float = 15.0) -> bool:
    """Wait for transfer graphs to complete and return success status"""
    completed_graph_ids = set()
    start_time = time.time()
    
    while len(completed_graph_ids) < len(expected_graph_ids) and (time.time() - start_time) < max_wait_time:
        results = transfer_engine.get_completed_graphs_and_ops(timeout=0.1)
        for graph_id, op_id in results:
            if op_id == -1:  # Graph completion
                completed_graph_ids.add(graph_id)
        time.sleep(0.001)
    
    return len(completed_graph_ids) == len(expected_graph_ids)


@pytest.mark.parametrize("tp_size,dp_size", [(1, 1), (2, 1), (2, 2)])
@pytest.mark.parametrize("num_gpu_blocks", [64, 128])
@pytest.mark.parametrize("transfer_block_num", [16])
def test_gpu_cpu_round_trip(model_config, cache_config, tp_size, dp_size, num_gpu_blocks, transfer_block_num):
    """
    Test round-trip data transfers between GPU and CPU
    
    This test validates:
    1. GPU -> CPU transfer correctness
    2. CPU -> GPU transfer correctness  
    3. Round-trip data consistency (GPU -> CPU -> GPU)
    
    Parameterized by:
    - tp_size, dp_size: Tensor and data parallelism configurations
    - num_gpu_blocks: Number of GPU blocks to test with
    - transfer_block_num: Number of blocks to transfer in each operation
    """
    total_gpus = tp_size * dp_size
    if torch.cuda.device_count() < total_gpus:
        pytest.skip(f"Need at least {total_gpus} CUDA devices")
    
    if transfer_block_num > num_gpu_blocks:
        pytest.skip(f"transfer_block_num ({transfer_block_num}) > num_gpu_blocks ({num_gpu_blocks})")
    
    # Reset ID allocator
    TransferIDAllocator.reset()
    
    # Update model config
    model_config.tp_size = tp_size
    model_config.dp_size = dp_size
    
    # Setup configurations
    cache_config.enable_ssd = False
    setup_kv_layouts(model_config, cache_config, num_gpu_blocks)
    
    # Create GPU blocks and save ground truth
    gpu_blocks = create_test_gpu_blocks(model_config, cache_config, num_gpu_blocks)
    
    # Setup storage engine and transfer engine
    storage_engine = StorageEngine(model_config, cache_config, gpu_blocks)
    gpu_handles = [storage_engine.get_allocator_handle(DeviceType.GPU, i) for i in range(total_gpus)]
    cpu_handle = storage_engine.get_allocator_handle(DeviceType.CPU)
    
    transfer_engine = TransferEngine(
        gpu_handles=gpu_handles,
        model_config=model_config,
        cache_config=cache_config,
        cpu_handle=cpu_handle
    )
    
    # Test each DP group separately
    for dp_id in range(dp_size):
        gpu_block_ids = torch.arange(0, transfer_block_num, dtype=torch.int64)
        cpu_block_ids = torch.arange(dp_id * transfer_block_num, (dp_id + 1) * transfer_block_num, dtype=torch.int64)
        
        # Step 1: GPU -> CPU transfer
        write_graph, _ = create_write_graph_cpu_storage(
            graph_id=TransferIDAllocator.allocate_graph_id(),
            gpu_blocks=gpu_block_ids,
            cpu_blocks=cpu_block_ids,
            ssd_blocks=torch.tensor([], dtype=torch.int64),
            gpu_device_id=dp_id * tp_size,
            layer_num=model_config.num_layers
        )
        write_graph.bind_to_dp_group(dp_id)
        transfer_engine.submit_transfer_graph(write_graph)
        
        # Wait for write completion
        assert wait_for_transfer_completion(transfer_engine, [write_graph.transfer_graph_id]), \
            f"GPU->CPU transfer failed for DP group {dp_id}"
        
        # Verify GPU -> CPU transfer
        start_gpu_id= dp_id * tp_size
        DataVerifier.verify_gpu_cpu_transfer(
            gpu_blocks, cpu_handle.data, gpu_block_ids, cpu_block_ids, 
            model_config, cache_config, start_gpu_id, tp_size,
        )
        
        # Clear GPU blocks for read test
        for tp_id in range(tp_size):
            global_gpu_id = dp_id * tp_size + tp_id
            for layer_id in range(model_config.num_layers):
                gpu_blocks[global_gpu_id][layer_id].zero_()
        
        # Step 2: CPU -> GPU transfer
        read_graph, _ = create_read_graph_cpu_storage(
            graph_id=TransferIDAllocator.allocate_graph_id(),
            gpu_blocks=gpu_block_ids,
            cpu_blocks=cpu_block_ids,
            ssd_blocks=torch.tensor([], dtype=torch.int64),
            gpu_device_id=dp_id * tp_size,
            layer_num=model_config.num_layers
        )
        read_graph.bind_to_dp_group(dp_id)
        transfer_engine.submit_transfer_graph(read_graph)
        
        # Wait for read completion
        assert wait_for_transfer_completion(transfer_engine, [read_graph.transfer_graph_id]), \
            f"CPU->GPU transfer failed for DP group {dp_id}"
        
        # Verify round-trip consistency
        DataVerifier.verify_gpu_cpu_transfer(
            gpu_blocks, cpu_handle.data, gpu_block_ids, cpu_block_ids,
            model_config, cache_config, start_gpu_id, tp_size,
        )
    
    # Cleanup
    transfer_engine.shutdown()


@pytest.mark.parametrize("num_gpu_blocks", [64, 128])
@pytest.mark.parametrize("transfer_block_num", [8, 16])
def test_ssd_round_trip(model_config, cache_config, num_gpu_blocks, transfer_block_num):
    """
    Test round-trip data transfers involving SSD storage
    
    This test validates:
    1. GPU -> CPU -> SSD transfer chain
    2. SSD -> CPU -> GPU transfer chain
    3. Full round-trip data consistency
    
    Parameterized by:
    - num_gpu_blocks: Number of GPU blocks to test with
    - transfer_block_num: Number of blocks to transfer
    """
    if torch.cuda.device_count() == 0:
        pytest.skip("No CUDA devices available")
    
    if transfer_block_num > num_gpu_blocks:
        pytest.skip(f"transfer_block_num ({transfer_block_num}) > num_gpu_blocks ({num_gpu_blocks})")
    
    # Reset ID allocator
    TransferIDAllocator.reset()
    
    # Setup configurations
    cache_config.enable_ssd = True
    setup_kv_layouts(model_config, cache_config, num_gpu_blocks)
    if (model_config.tp_size * model_config.dp_size) > 1:
        pytest.skip("SSD transfer test is not supported for multi-GPU")
    
    # Create GPU blocks and save ground truth
    gpu_blocks = create_test_gpu_blocks(model_config, cache_config, num_gpu_blocks)
    original_gpu_data = {gpu_id: [layer.clone() for layer in layers] for gpu_id, layers in gpu_blocks.items()}
    
    # Setup storage engine and transfer engine
    storage_engine = StorageEngine(model_config, cache_config, gpu_blocks)
    gpu_handles = [storage_engine.get_allocator_handle(DeviceType.GPU, i) 
                   for i in range(model_config.tp_size * model_config.dp_size)]
    cpu_handle = storage_engine.get_allocator_handle(DeviceType.CPU)
    ssd_handle = storage_engine.get_allocator_handle(DeviceType.SSD)
    
    transfer_engine = TransferEngine(
        gpu_handles=gpu_handles,
        model_config=model_config,
        cache_config=cache_config,
        cpu_handle=cpu_handle,
        ssd_handle=ssd_handle
    )
    
    # Prepare transfer block IDs
    gpu_block_ids = torch.arange(0, transfer_block_num, dtype=torch.int64)
    cpu_block_ids = torch.arange(0, transfer_block_num, dtype=torch.int64)
    ssd_block_ids = torch.arange(0, transfer_block_num, dtype=torch.int64)
    
    # Step 1: GPU -> CPU -> SSD write
    write_graph, _ = create_write_graph_cpu_storage(
        graph_id=TransferIDAllocator.allocate_graph_id(),
        gpu_blocks=gpu_block_ids,
        cpu_blocks=cpu_block_ids,
        ssd_blocks=ssd_block_ids,
        gpu_device_id=0,
        layer_num=model_config.num_layers
    )
    write_graph.bind_to_dp_group(0)
    transfer_engine.submit_transfer_graph(write_graph)
    
    # Wait for write completion
    assert wait_for_transfer_completion(transfer_engine, [write_graph.transfer_graph_id]), \
        "GPU->CPU->SSD write transfer failed"
    
    # Clear GPU blocks for read test
    for gpu_id in range(model_config.tp_size * model_config.dp_size):
        for layer_id in range(model_config.num_layers):
            gpu_blocks[gpu_id][layer_id].zero_()
    
    # Step 2: SSD -> CPU -> GPU read
    read_graph, _ = create_read_graph_cpu_storage(
        graph_id=TransferIDAllocator.allocate_graph_id(),
        gpu_blocks=gpu_block_ids,
        cpu_blocks=cpu_block_ids,
        ssd_blocks=ssd_block_ids,
        gpu_device_id=0,
        layer_num=model_config.num_layers
    )
    read_graph.bind_to_dp_group(0)
    transfer_engine.submit_transfer_graph(read_graph)
    
    # Wait for read completion
    assert wait_for_transfer_completion(transfer_engine, [read_graph.transfer_graph_id]), \
        "SSD->CPU->GPU read transfer failed"
    
    # Verify full round-trip consistency
    DataVerifier.verify_round_trip_consistency(
        original_gpu_data, gpu_blocks, gpu_block_ids, model_config, 0
    )
    
    # Cleanup
    transfer_engine.shutdown()


@pytest.mark.parametrize("num_concurrent_transfers", [4])
@pytest.mark.parametrize("blocks_per_transfer", [16])
@pytest.mark.parametrize("include_ssd", [True, False])
def test_concurrent_mixed_transfers(model_config, cache_config, num_concurrent_transfers, blocks_per_transfer, include_ssd):
    """
    Test multiple concurrent read/write transfers
    
    This test validates:
    1. Multiple write transfers running concurrently
    2. Multiple read transfers running concurrently
    3. Mixed read/write transfers running concurrently
    4. Data correctness across all concurrent operations
    
    Parameterized by:
    - num_concurrent_transfers: Number of concurrent transfer graphs
    - blocks_per_transfer: Number of blocks per transfer operation
    - include_ssd: Whether to include SSD in transfer operations
    """
    if torch.cuda.device_count() == 0:
        pytest.skip("No CUDA devices available")
    if (model_config.tp_size * model_config.dp_size) > 1:
        pytest.skip("Concurrent transfer test is not supported for multi-GPU")
    
    total_blocks_needed = num_concurrent_transfers * blocks_per_transfer * 2  # For both read and write
    num_gpu_blocks = max(128, total_blocks_needed)

    cache_config.num_cpu_blocks = num_gpu_blocks
    cache_config.num_ssd_blocks = num_gpu_blocks
    
    # Reset ID allocator
    TransferIDAllocator.reset()
    
    # Setup configurations
    cache_config.enable_ssd = include_ssd
    setup_kv_layouts(model_config, cache_config, num_gpu_blocks)
    
    # Create GPU blocks and save ground truth
    gpu_blocks = create_test_gpu_blocks(model_config, cache_config, num_gpu_blocks)
    original_gpu_data = {gpu_id: [layer.clone() for layer in layers] for gpu_id, layers in gpu_blocks.items()}
    
    # Setup storage engine and transfer engine
    storage_engine = StorageEngine(model_config, cache_config, gpu_blocks)
    gpu_handles = [storage_engine.get_allocator_handle(DeviceType.GPU, i) 
                   for i in range(model_config.tp_size * model_config.dp_size)]
    cpu_handle = storage_engine.get_allocator_handle(DeviceType.CPU)
    ssd_handle = storage_engine.get_allocator_handle(DeviceType.SSD) if include_ssd else None
    
    transfer_engine = TransferEngine(
        gpu_handles=gpu_handles,
        model_config=model_config,
        cache_config=cache_config,
        cpu_handle=cpu_handle,
        ssd_handle=ssd_handle
    )
    
    # Create concurrent write transfers
    write_graphs = []
    
    for i in range(num_concurrent_transfers):
        start_block = i * blocks_per_transfer
        end_block = start_block + blocks_per_transfer
        
        gpu_block_ids = torch.arange(start_block, end_block, dtype=torch.int64)
        cpu_block_ids = torch.arange(start_block, end_block, dtype=torch.int64)
        ssd_block_ids = torch.arange(start_block, end_block, dtype=torch.int64) if include_ssd else torch.tensor([], dtype=torch.int64)
        
        
        write_graph, _ = create_write_graph_cpu_storage(
            graph_id=TransferIDAllocator.allocate_graph_id(),
            gpu_blocks=gpu_block_ids,
            cpu_blocks=cpu_block_ids,
            ssd_blocks=ssd_block_ids,
            gpu_device_id=0,
            layer_num=model_config.num_layers
        )
        write_graph.bind_to_dp_group(0)
        write_graphs.append(write_graph)
    
    # Submit all write transfers
    for graph in write_graphs:
        transfer_engine.submit_transfer_graph(graph)
    
    # Wait for all writes to complete
    write_graph_ids = [graph.transfer_graph_id for graph in write_graphs]
    assert wait_for_transfer_completion(transfer_engine, write_graph_ids, max_wait_time=20.0), \
        "Concurrent write transfers failed to complete"
    
    # Clear GPU blocks for read test
    for gpu_id in range(model_config.tp_size * model_config.dp_size):
        for layer_id in range(model_config.num_layers):
            gpu_blocks[gpu_id][layer_id].zero_()
    
    # Create concurrent read transfers (using different GPU blocks)
    read_graphs = []
    
    for i in range(num_concurrent_transfers):
        gpu_block_ids = torch.arange(i * blocks_per_transfer, (i + 1) * blocks_per_transfer, dtype=torch.int64)
        cpu_block_ids = torch.arange(i * blocks_per_transfer, (i + 1) * blocks_per_transfer, dtype=torch.int64)
        ssd_block_ids = torch.arange(i * blocks_per_transfer, (i + 1) * blocks_per_transfer, dtype=torch.int64) if include_ssd else torch.tensor([], dtype=torch.int64)
        
        read_graph, _ = create_read_graph_cpu_storage(
            graph_id=TransferIDAllocator.allocate_graph_id(),
            gpu_blocks=gpu_block_ids,
            cpu_blocks=cpu_block_ids,
            ssd_blocks=ssd_block_ids,
            gpu_device_id=0,
            layer_num=model_config.num_layers
        )
        read_graph.bind_to_dp_group(0)
        read_graphs.append(read_graph)
    
    # Submit all read transfers
    for graph in read_graphs:
        transfer_engine.submit_transfer_graph(graph)
    
    # Wait for all reads to complete
    read_graph_ids = [graph.transfer_graph_id for graph in read_graphs]
    assert wait_for_transfer_completion(transfer_engine, read_graph_ids, max_wait_time=20.0), \
        "Concurrent read transfers failed to complete"
    
    # Verify round-trip consistency
    DataVerifier.verify_round_trip_consistency(
        original_gpu_data, gpu_blocks, gpu_block_ids, model_config, 0
    )
    
    # Cleanup
    transfer_engine.shutdown()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
