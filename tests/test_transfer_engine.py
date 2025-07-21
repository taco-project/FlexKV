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
from contextlib import suppress

import pytest
import torch

from flexkv.cache.transfer_pattern import (
    create_read_graph_cpu_storage,
    create_write_graph_cpu_storage,
)
from flexkv.common.config import ModelConfig, CacheConfig
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import DeviceType
from flexkv.storage.storage_engine import StorageEngine
from flexkv.transfer.transfer_engine import TransferEngine

# Import utilities from test_utils
from tests.test_utils import (
    wait_for_transfer_completion,
    skip_if_no_cuda,
    skip_if_insufficient_gpus,
    generate_gpu_blocks_with_ground_truth,
    verify_data
)

@pytest.mark.parametrize("tp_size,dp_size", [(1, 1), (2, 1), (2, 2)])
@pytest.mark.parametrize("num_gpu_blocks", [128])
@pytest.mark.parametrize("transfer_block_num", [16])
@pytest.mark.parametrize("use_mla", [False, True])
@pytest.mark.parametrize("underlying_layout_type", [KVCacheLayoutType.LAYERWISE, KVCacheLayoutType.BLOCKWISE])
def test_gpu_cpu_round_trip(model_config,
                            cache_config,
                            test_config,
                            tp_size,
                            dp_size,
                            num_gpu_blocks,
                            transfer_block_num,
                            use_mla,
                            underlying_layout_type):
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
    skip_if_insufficient_gpus(total_gpus)

    if transfer_block_num > num_gpu_blocks:
        pytest.skip(f"transfer_block_num ({transfer_block_num}) > num_gpu_blocks ({num_gpu_blocks})")

    # Update model config
    model_config.use_mla = use_mla
    model_config.tp_size = tp_size
    model_config.dp_size = dp_size

    # Create a copy of test_config to avoid modifying the fixture
    test_config_copy = test_config.copy()
    test_config_copy['num_gpu_blocks'] = num_gpu_blocks

    cache_config.cpu_kv_layout_type = underlying_layout_type
    # Setup configurations
    cache_config.enable_ssd = False

    gpu_blocks, dp_wise_gpu_blocks_gt, gpu_kv_layout = \
        generate_gpu_blocks_with_ground_truth(model_config, cache_config, test_config_copy)
    # Setup storage engine and transfer engine
    storage_engine = StorageEngine(model_config, cache_config)
    for gpu_id, gpu_block in gpu_blocks.items():
        storage_engine.register_gpu_blocks(gpu_block, gpu_kv_layout, device_id=gpu_id, dtype=model_config.dtype)
    gpu_handles = [storage_engine.get_storage_handle(DeviceType.GPU, i) for i in range(total_gpus)]
    cpu_handle = storage_engine.get_storage_handle(DeviceType.CPU)

    transfer_engine = TransferEngine(
        gpu_handles=gpu_handles,
        model_config=model_config,
        cache_config=cache_config,
        cpu_handle=cpu_handle
    )
    transfer_engine.start()

    # Test each DP group separately
    for dp_id in range(dp_size):
        gpu_block_ids = torch.arange(0, transfer_block_num, dtype=torch.int64)
        cpu_block_ids = torch.arange(dp_id * transfer_block_num, (dp_id + 1) * transfer_block_num, dtype=torch.int64)

        # Step 1: GPU -> CPU transfer
        write_graph, _ = create_write_graph_cpu_storage(
            gpu_blocks=gpu_block_ids,
            cpu_blocks=cpu_block_ids,
            ssd_blocks=torch.tensor([], dtype=torch.int64),
            gpu_device_id=dp_id * tp_size,
            layer_num=model_config.num_layers
        )
        write_graph.bind_to_dp_group(dp_id)
        transfer_engine.submit_transfer_graph(write_graph)

        # Wait for write completion
        assert wait_for_transfer_completion(transfer_engine, [write_graph.graph_id]), \
            f"GPU->CPU transfer failed for DP group {dp_id}"

        # Clear GPU blocks for read test
        for tp_id in range(tp_size):
            global_gpu_id = dp_id * tp_size + tp_id
            for layer_id in range(model_config.num_layers):
                gpu_blocks[global_gpu_id][layer_id][:,gpu_block_ids].zero_()

        # Step 2: CPU -> GPU transfer
        read_graph, _ = create_read_graph_cpu_storage(
            gpu_blocks=gpu_block_ids,
            cpu_blocks=cpu_block_ids,
            ssd_blocks=torch.tensor([], dtype=torch.int64),
            gpu_device_id=dp_id * tp_size,
            layer_num=model_config.num_layers
        )
        read_graph.bind_to_dp_group(dp_id)
        transfer_engine.submit_transfer_graph(read_graph)
        # Wait for read completion
        assert wait_for_transfer_completion(transfer_engine, [read_graph.graph_id]), \
            f"CPU->GPU transfer failed for DP group {dp_id}"

    verify_data(gpu_blocks, dp_wise_gpu_blocks_gt, model_config.num_kv_heads,
                model_config.tp_size, model_config.dp_size, model_config.num_layers, model_config.use_mla)
    # Cleanup
    transfer_engine.shutdown()


@pytest.mark.parametrize("num_gpu_blocks", [64, 128])
@pytest.mark.parametrize("transfer_block_num", [8, 16])
@pytest.mark.parametrize("use_mla", [True, False])
@pytest.mark.parametrize("iouring_entries", [0, 512])
def test_ssd_round_trip(model_config,
                        cache_config,
                        test_config,
                        num_gpu_blocks,
                        transfer_block_num,
                        use_mla,
                        iouring_entries):
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
    skip_if_no_cuda()

    if transfer_block_num > num_gpu_blocks:
        pytest.skip(f"transfer_block_num ({transfer_block_num}) > num_gpu_blocks ({num_gpu_blocks})")

    # Setup configurations
    cache_config.enable_ssd = True
    cache_config.ssd_cache_iouring_entries = iouring_entries
    model_config.use_mla = use_mla

    # Create a copy of test_config to avoid modifying the fixture
    test_config_copy = test_config.copy()
    test_config_copy['num_gpu_blocks'] = num_gpu_blocks

    gpu_blocks, dp_wise_gpu_blocks_gt, gpu_kv_layout = \
        generate_gpu_blocks_with_ground_truth(model_config, cache_config, test_config_copy)
    if (model_config.tp_size * model_config.dp_size) > 1:
        pytest.skip("SSD transfer test is not supported for multi-GPU")

    # Setup storage engine and transfer engine
    storage_engine = StorageEngine(model_config, cache_config)
    for gpu_id, gpu_block in gpu_blocks.items():
        storage_engine.register_gpu_blocks(gpu_block, gpu_kv_layout, device_id=gpu_id, dtype=model_config.dtype)
    gpu_handles = [storage_engine.get_storage_handle(DeviceType.GPU, i)
                   for i in range(model_config.tp_size * model_config.dp_size)]
    cpu_handle = storage_engine.get_storage_handle(DeviceType.CPU)
    ssd_handle = storage_engine.get_storage_handle(DeviceType.SSD)

    transfer_engine = TransferEngine(
        gpu_handles=gpu_handles,
        model_config=model_config,
        cache_config=cache_config,
        cpu_handle=cpu_handle,
        ssd_handle=ssd_handle
    )
    transfer_engine.start()
    # Prepare transfer block IDs
    gpu_block_ids = torch.arange(0, transfer_block_num, dtype=torch.int64)
    cpu_block_ids = torch.arange(0, transfer_block_num, dtype=torch.int64)
    ssd_block_ids = torch.arange(0, transfer_block_num, dtype=torch.int64)

    # Step 1: GPU -> CPU -> SSD write
    write_graph, _ = create_write_graph_cpu_storage(
        gpu_blocks=gpu_block_ids,
        cpu_blocks=cpu_block_ids,
        ssd_blocks=ssd_block_ids,
        gpu_device_id=0,
        layer_num=model_config.num_layers
    )
    write_graph.bind_to_dp_group(0)
    transfer_engine.submit_transfer_graph(write_graph)

    # Wait for write completion
    assert wait_for_transfer_completion(transfer_engine, [write_graph.graph_id]), \
        "GPU->CPU->SSD write transfer failed"

    # Clear GPU blocks for read test
    for gpu_id in range(model_config.tp_size * model_config.dp_size):
        for layer_id in range(model_config.num_layers):
            gpu_blocks[gpu_id][layer_id][:,gpu_block_ids].zero_()

    # Step 2: SSD -> CPU -> GPU read
    read_graph, _ = create_read_graph_cpu_storage(
        gpu_blocks=gpu_block_ids,
        cpu_blocks=cpu_block_ids,
        ssd_blocks=ssd_block_ids,
        gpu_device_id=0,
        layer_num=model_config.num_layers
    )
    read_graph.bind_to_dp_group(0)
    transfer_engine.submit_transfer_graph(read_graph)

    # Wait for read completion
    assert wait_for_transfer_completion(transfer_engine, [read_graph.graph_id]), \
        "SSD->CPU->GPU read transfer failed"

    verify_data(gpu_blocks, dp_wise_gpu_blocks_gt, model_config.num_kv_heads,
                model_config.tp_size, model_config.dp_size, model_config.num_layers, model_config.use_mla)

    # Cleanup
    transfer_engine.shutdown()


@pytest.mark.parametrize("num_concurrent_transfers", [4])
@pytest.mark.parametrize("blocks_per_transfer", [16])
@pytest.mark.parametrize("include_ssd", [True, False])
@pytest.mark.parametrize("use_mla", [True, False])
@pytest.mark.parametrize("iouring_entries", [0, 512])
def test_concurrent_mixed_transfers(model_config,
                                    cache_config,
                                    test_config,
                                    num_concurrent_transfers,
                                    blocks_per_transfer,
                                    include_ssd,
                                    use_mla,
                                    iouring_entries):
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
    model_config.use_mla = use_mla
    skip_if_no_cuda()

    if (model_config.tp_size * model_config.dp_size) > 1:
        pytest.skip("Concurrent transfer test is not supported for multi-GPU")

    total_blocks_needed = num_concurrent_transfers * blocks_per_transfer * 2  # For both read and write
    num_gpu_blocks = max(128, total_blocks_needed)

    cache_config.num_cpu_blocks = num_gpu_blocks
    cache_config.num_ssd_blocks = num_gpu_blocks
    cache_config.ssd_cache_iouring_entries = iouring_entries

    # Setup configurations
    cache_config.enable_ssd = include_ssd

    # Create a copy of test_config to avoid modifying the fixture
    test_config_copy = test_config.copy()
    test_config_copy['num_gpu_blocks'] = num_gpu_blocks

    gpu_blocks, dp_wise_gpu_blocks_gt, gpu_kv_layout = \
        generate_gpu_blocks_with_ground_truth(model_config, cache_config, test_config_copy)

    # Setup storage engine and transfer engine
    storage_engine = StorageEngine(model_config, cache_config)
    for gpu_id, gpu_block in gpu_blocks.items():
        storage_engine.register_gpu_blocks(gpu_block, gpu_kv_layout, device_id=gpu_id, dtype=model_config.dtype)
    gpu_handles = [storage_engine.get_storage_handle(DeviceType.GPU, i)
                   for i in range(model_config.tp_size * model_config.dp_size)]
    cpu_handle = storage_engine.get_storage_handle(DeviceType.CPU)
    ssd_handle = storage_engine.get_storage_handle(DeviceType.SSD) if include_ssd else None

    transfer_engine = TransferEngine(
        gpu_handles=gpu_handles,
        model_config=model_config,
        cache_config=cache_config,
        cpu_handle=cpu_handle,
        ssd_handle=ssd_handle
    )

    transfer_engine.start()
    # Create concurrent write transfers
    write_graphs = []

    for i in range(num_concurrent_transfers):
        start_block = i * blocks_per_transfer
        end_block = start_block + blocks_per_transfer

        gpu_block_ids = torch.arange(start_block, end_block, dtype=torch.int64)
        cpu_block_ids = torch.arange(start_block, end_block, dtype=torch.int64)
        ssd_block_ids = torch.arange(start_block, end_block, dtype=torch.int64) \
            if include_ssd else torch.tensor([], dtype=torch.int64)


        write_graph, _ = create_write_graph_cpu_storage(
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
    write_graph_ids = [graph.graph_id for graph in write_graphs]
    assert wait_for_transfer_completion(transfer_engine, write_graph_ids, max_wait_time=20.0), \
        "Concurrent write transfers failed to complete"

    # Clear GPU blocks for read test
    for gpu_id in range(model_config.tp_size * model_config.dp_size):
        gpu_block_ids = torch.arange(0, (num_concurrent_transfers + 1) * blocks_per_transfer, dtype=torch.int64)
        for layer_id in range(model_config.num_layers):
            gpu_blocks[gpu_id][layer_id][:,gpu_block_ids].zero_()

    # Create concurrent read transfers (using different GPU blocks)
    read_graphs = []

    for i in range(num_concurrent_transfers):
        gpu_block_ids = torch.arange(i * blocks_per_transfer, (i + 1) * blocks_per_transfer, dtype=torch.int64)
        cpu_block_ids = torch.arange(i * blocks_per_transfer, (i + 1) * blocks_per_transfer, dtype=torch.int64)
        ssd_block_ids = torch.arange(i * blocks_per_transfer, (i + 1) * blocks_per_transfer, dtype=torch.int64) \
            if include_ssd else torch.tensor([], dtype=torch.int64)

        read_graph, _ = create_read_graph_cpu_storage(
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
    read_graph_ids = [graph.graph_id for graph in read_graphs]
    assert wait_for_transfer_completion(transfer_engine, read_graph_ids, max_wait_time=20.0), \
        "Concurrent read transfers failed to complete"

    verify_data(gpu_blocks, dp_wise_gpu_blocks_gt, model_config.num_kv_heads,
                model_config.tp_size, model_config.dp_size, model_config.num_layers, model_config.use_mla)

    # Cleanup
    transfer_engine.shutdown()

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
