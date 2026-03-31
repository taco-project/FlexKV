"""Tests for KVManagerGroup and KVTPClientGroup in flexkv.kv_group."""

import copy
import time
import os
import multiprocessing as mp

import pytest
import torch
import numpy as np

from flexkv.common.config import ModelConfig, CacheConfig, GLOBAL_CONFIG_FROM_ENV
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.request import KVResponseStatus
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.kvmanager import KVManager
from flexkv.server.client import KVTPClient
from flexkv.kv_group import KVManagerGroup, KVTPClientGroup
from flexkv.common.debug import flexkv_logger

from common_utils import (
    DEFAULT_MODEL_CONFIG, DEFAULT_CACHE_CONFIG,
    generate_request_pair, block_ids_2_slot_mapping,
    skip_if_insufficient_gpus, create_gpu_kv_layout, GPUKVCacheVerifier,
)


# ---------------------------------------------------------------------------
# Helper: run tp_client process (no indexer)
# ---------------------------------------------------------------------------

def _run_tp_client_no_indexer(dp_client_id, tp_rank, server_recv_port,
                              model_config, cache_config, num_gpu_blocks,
                              child_conn, gpu_layout_type):
    """Run a plain KVTPClient process (no indexer)."""
    try:
        device_id = tp_rank + dp_client_id * model_config.tp_size
        tp_client = KVTPClient(server_recv_port, dp_client_id, device_id)

        gpu_kv_layout = create_gpu_kv_layout(model_config, cache_config,
                                             num_gpu_blocks, gpu_layout_type)
        gpu_blocks = []
        if gpu_layout_type == 0:
            for _ in range(model_config.num_layers):
                gpu_blocks.append(
                    torch.empty(size=tuple(gpu_kv_layout.kv_shape[1:]),
                                dtype=model_config.dtype).cuda(device_id))
        elif gpu_layout_type == 2:
            kv_dim = 1 if model_config.use_mla else 2
            for _ in range(model_config.num_layers * kv_dim):
                gpu_blocks.append(
                    torch.empty(size=tuple(gpu_kv_layout.kv_shape[2:]),
                                dtype=model_config.dtype).cuda(device_id))
        else:
            raise ValueError(f"Invalid GPU layout type: {gpu_layout_type}")

        tp_client.register_to_server(gpu_blocks, gpu_kv_layout)

        if child_conn is not None:
            shared = [TensorSharedHandle(t) for t in gpu_blocks]
            child_conn.send(shared)
            child_conn.close()

        while True:
            time.sleep(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        if child_conn is not None:
            child_conn.send(None)
            child_conn.close()


# ---------------------------------------------------------------------------
# Helper: run tp_client process (with indexer via KVTPClientGroup)
# ---------------------------------------------------------------------------

def _run_tp_client_with_indexer(dp_client_id, tp_rank, server_recv_port,
                                model_config, cache_config, num_gpu_blocks,
                                child_conn, gpu_layout_type,
                                indexer_gpu_register_port,
                                indexer_model_config, indexer_cache_config):
    """Run a KVTPClientGroup process that registers both main and indexer caches."""
    try:
        device_id = tp_rank + dp_client_id * model_config.tp_size

        gpu_kv_layout = create_gpu_kv_layout(model_config, cache_config,
                                             num_gpu_blocks, gpu_layout_type)
        # Main GPU blocks
        gpu_blocks = []
        if gpu_layout_type == 0:
            for _ in range(model_config.num_layers):
                gpu_blocks.append(
                    torch.empty(size=tuple(gpu_kv_layout.kv_shape[1:]),
                                dtype=model_config.dtype).cuda(device_id))
        elif gpu_layout_type == 2:
            kv_dim = 1 if model_config.use_mla else 2
            for _ in range(model_config.num_layers * kv_dim):
                gpu_blocks.append(
                    torch.empty(size=tuple(gpu_kv_layout.kv_shape[2:]),
                                dtype=model_config.dtype).cuda(device_id))
        else:
            raise ValueError(f"Invalid GPU layout type for indexer test: {gpu_layout_type}")

        # Indexer GPU blocks (MLA-style: 3D tensors)
        indexer_blocks = []
        indexer_tpb = indexer_cache_config.tokens_per_block
        for _ in range(indexer_model_config.num_layers):
            indexer_blocks.append(
                torch.empty(num_gpu_blocks, indexer_tpb,
                            indexer_model_config.head_size,
                            dtype=indexer_model_config.dtype).cuda(device_id))

        indexer_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=indexer_model_config.num_layers,
            num_block=num_gpu_blocks,
            tokens_per_block=indexer_tpb,
            num_head=1,
            head_size=indexer_model_config.head_size,
            is_mla=True,
        )

        tp_group = KVTPClientGroup(
            gpu_register_port=server_recv_port + "_gpu_register",
            client_id=dp_client_id,
            device_id=device_id,
            indexer_gpu_register_port=indexer_gpu_register_port,
        )
        tp_group.register_to_server(
            kv_caches=gpu_blocks,
            gpu_layout=gpu_kv_layout,
            indexer_buffers=indexer_blocks,
            indexer_layout=indexer_layout,
        )

        if child_conn is not None:
            shared = [TensorSharedHandle(t) for t in gpu_blocks]
            child_conn.send(shared)
            child_conn.close()

        while True:
            time.sleep(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        if child_conn is not None:
            child_conn.send(None)
            child_conn.close()


def _run_tp_client_group_no_indexer(gpu_register_port, model_config, cache_config,
                                    num_gpu_blocks, gpu_layout_type, child_conn):
    """Module-level worker for test_kv_tp_client_group_without_indexer (picklable by spawn)."""
    try:
        device_id = 0
        gpu_kv_layout = create_gpu_kv_layout(model_config, cache_config,
                                             num_gpu_blocks, gpu_layout_type)
        gpu_blocks = []
        for _ in range(model_config.num_layers):
            gpu_blocks.append(
                torch.empty(size=tuple(gpu_kv_layout.kv_shape[1:]),
                            dtype=model_config.dtype).cuda(device_id))

        tp_group = KVTPClientGroup(
            gpu_register_port=gpu_register_port,
            client_id=0,
            device_id=device_id,
        )
        assert tp_group.tp_client is not None
        assert tp_group.indexer_tp_client is None

        tp_group.register_to_server(kv_caches=gpu_blocks, gpu_layout=gpu_kv_layout)
        child_conn.send("OK")
        child_conn.close()
        while True:
            time.sleep(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        child_conn.send(f"FAIL: {e}")
        child_conn.close()


def _run_tp_client_group_with_indexer(main_gpu_register_port, model_config, cache_config,
                                      num_gpu_blocks, gpu_layout_type,
                                      indexer_gpu_register_port,
                                      indexer_model_config, indexer_cache_config,
                                      child_conn):
    """Module-level worker for test_kv_tp_client_group_with_indexer (picklable by spawn)."""
    try:
        device_id = 0
        gpu_kv_layout = create_gpu_kv_layout(model_config, cache_config,
                                             num_gpu_blocks, gpu_layout_type)
        gpu_blocks = []
        for _ in range(model_config.num_layers):
            gpu_blocks.append(
                torch.empty(size=tuple(gpu_kv_layout.kv_shape[1:]),
                            dtype=model_config.dtype).cuda(device_id))

        # Indexer blocks
        indexer_blocks = []
        indexer_tpb = indexer_cache_config.tokens_per_block
        for _ in range(indexer_model_config.num_layers):
            indexer_blocks.append(
                torch.empty(num_gpu_blocks, indexer_tpb,
                            indexer_model_config.head_size,
                            dtype=indexer_model_config.dtype).cuda(device_id))

        indexer_layout = KVCacheLayout(
            type=KVCacheLayoutType.LAYERFIRST,
            num_layer=indexer_model_config.num_layers,
            num_block=num_gpu_blocks,
            tokens_per_block=indexer_tpb,
            num_head=1,
            head_size=indexer_model_config.head_size,
            is_mla=True,
        )

        tp_group = KVTPClientGroup(
            gpu_register_port=main_gpu_register_port,
            client_id=0,
            device_id=device_id,
            indexer_gpu_register_port=indexer_gpu_register_port,
        )
        assert tp_group.tp_client is not None
        assert tp_group.indexer_tp_client is not None

        tp_group.register_to_server(
            kv_caches=gpu_blocks,
            gpu_layout=gpu_kv_layout,
            indexer_buffers=indexer_blocks,
            indexer_layout=indexer_layout,
        )
        child_conn.send("OK")
        child_conn.close()
        while True:
            time.sleep(1)
    except Exception as e:
        import traceback
        traceback.print_exc()
        child_conn.send(f"FAIL: {e}")
        child_conn.close()


def _shutdown_processes(procs):
    for p in procs:
        if p.is_alive():
            p.terminate()
            p.join(timeout=5)
            if p.is_alive():
                p.kill()
                p.join(timeout=2)


# ===================================================================
# Test: KVManagerGroup without indexer
# ===================================================================

@pytest.mark.parametrize("model_config", [{"tp_size": 1, "dp_size": 1}], indirect=True)
@pytest.mark.parametrize("cache_config", [
    {"enable_cpu": True, "enable_ssd": False, "num_cpu_blocks": 1024},
], indirect=True)
@pytest.mark.parametrize("test_config", [
    {"num_gpu_blocks": 256, "requests_per_block": 16, "initial_write_ratio": 0.4},
], indirect=True)
@pytest.mark.parametrize("gpu_layout_type", [0])
def test_kv_manager_group_without_indexer(model_config, cache_config, test_config,
                                          gpu_layout_type):
    """KVManagerGroup without indexer should behave identically to plain KVManager."""
    tp_size = model_config.tp_size
    tokens_per_block = cache_config.tokens_per_block
    num_gpu_blocks = test_config["num_gpu_blocks"]
    block_per_request = test_config["requests_per_block"]
    initial_write_ratio = test_config["initial_write_ratio"]
    num_requests = num_gpu_blocks // block_per_request

    skip_if_insufficient_gpus(tp_size)

    # Create KVManagerGroup *without* indexer
    group = KVManagerGroup(model_config, cache_config)
    assert group.indexer_kv_manager is None
    group.start()

    # Spawn tp_client
    mp_ctx = mp.get_context("spawn")
    parent_conn, child_conn = mp_ctx.Pipe()
    tp_proc = mp_ctx.Process(
        target=_run_tp_client_no_indexer,
        args=(0, 0, group.kv_manager.gpu_register_port,
              model_config, cache_config, num_gpu_blocks, child_conn,
              gpu_layout_type),
        daemon=True,
    )
    tp_proc.start()

    shared = parent_conn.recv()
    parent_conn.close()
    assert shared is not None

    while not group.is_ready():
        time.sleep(0.5)

    request_pairs = [
        generate_request_pair(i, block_per_request, num_gpu_blocks,
                              tokens_per_block, 1)
        for i in range(num_requests)
    ]
    initial_write_num = int(num_requests * initial_write_ratio)

    # Put flow
    for token_ids, block_ids, dp_id in request_pairs[:initial_write_num]:
        tid = group.put_async(
            token_ids=token_ids,
            slot_mapping=block_ids_2_slot_mapping(block_ids, tokens_per_block),
        )
        group.wait([tid], completely=True)

    # Get flow
    total_hit = 0
    total_miss = 0
    for i in range(initial_write_num):
        token_ids, block_ids, dp_id = request_pairs[i]
        slot_mapping = block_ids_2_slot_mapping(block_ids, tokens_per_block)
        req_id, _ = group.get_match(token_ids=token_ids)
        group.launch(req_id, slot_mapping)
        results = group.wait(req_id, completely=True)
        for resp in results.values():
            assert resp.status == KVResponseStatus.SUCCESS
            total_hit += resp.return_mask.sum().item()
            total_miss += len(resp.return_mask) - resp.return_mask.sum().item()

    assert total_miss == 0, f"Expected 0 cache miss, got {total_miss}"
    print(f"[test_kv_manager_group_without_indexer] PASSED  hit={total_hit}")

    _shutdown_processes([tp_proc])
    group.shutdown()


# ===================================================================
# Test: KVManagerGroup with indexer
# ===================================================================

@pytest.mark.parametrize("model_config", [{"tp_size": 1, "dp_size": 1}], indirect=True)
@pytest.mark.parametrize("cache_config", [
    {"enable_cpu": True, "enable_ssd": False, "num_cpu_blocks": 1024},
], indirect=True)
@pytest.mark.parametrize("test_config", [
    {"num_gpu_blocks": 256, "requests_per_block": 16, "initial_write_ratio": 0.4},
], indirect=True)
@pytest.mark.parametrize("gpu_layout_type", [0])
def test_kv_manager_group_with_indexer(model_config, cache_config, test_config,
                                       gpu_layout_type):
    """KVManagerGroup with indexer should auto-create and wait indexer tasks."""
    tp_size = model_config.tp_size
    tokens_per_block = cache_config.tokens_per_block
    num_gpu_blocks = test_config["num_gpu_blocks"]
    block_per_request = test_config["requests_per_block"]
    initial_write_ratio = test_config["initial_write_ratio"]
    num_requests = num_gpu_blocks // block_per_request

    skip_if_insufficient_gpus(tp_size)

    indexer_model_config = ModelConfig(
        num_layers=model_config.num_layers,
        num_kv_heads=1, head_size=64, use_mla=True,
        dtype=torch.uint8,
        tp_size=model_config.tp_size, dp_size=model_config.dp_size,
    )
    indexer_cache_config = CacheConfig(
        tokens_per_block=cache_config.tokens_per_block,
        enable_cpu=cache_config.enable_cpu,
        enable_ssd=cache_config.enable_ssd,
        num_cpu_blocks=cache_config.num_cpu_blocks,
    )
    indexer_server_recv_port = GLOBAL_CONFIG_FROM_ENV.server_recv_port + "_indexer"
    indexer_gpu_register_port = indexer_server_recv_port + "_gpu_register"

    group = KVManagerGroup(
        model_config, cache_config,
        indexer_model_config=indexer_model_config,
        indexer_cache_config=indexer_cache_config,
        indexer_server_recv_port=indexer_server_recv_port,
        indexer_gpu_register_port=indexer_gpu_register_port,
    )
    assert group.indexer_kv_manager is not None
    group.start()

    # Spawn tp_client with indexer
    mp_ctx = mp.get_context("spawn")
    parent_conn, child_conn = mp_ctx.Pipe()
    tp_proc = mp_ctx.Process(
        target=_run_tp_client_with_indexer,
        args=(0, 0, GLOBAL_CONFIG_FROM_ENV.server_recv_port,
              model_config, cache_config, num_gpu_blocks, child_conn,
              gpu_layout_type, indexer_gpu_register_port,
              indexer_model_config, indexer_cache_config),
        daemon=True,
    )
    tp_proc.start()

    shared = parent_conn.recv()
    parent_conn.close()
    assert shared is not None

    while not group.is_ready():
        time.sleep(0.5)

    request_pairs = [
        generate_request_pair(i, block_per_request, num_gpu_blocks,
                              tokens_per_block, 1)
        for i in range(num_requests)
    ]
    initial_write_num = int(num_requests * initial_write_ratio)

    # Put flow
    for token_ids, block_ids, dp_id in request_pairs[:initial_write_num]:
        tid = group.put_async(
            token_ids=token_ids,
            slot_mapping=block_ids_2_slot_mapping(block_ids, tokens_per_block),
        )
        group.wait([tid], completely=True)

    assert len(group._indexer_task_map) == 0

    # Get flow
    total_hit = 0
    total_miss = 0
    for i in range(initial_write_num):
        token_ids, block_ids, dp_id = request_pairs[i]
        slot_mapping = block_ids_2_slot_mapping(block_ids, tokens_per_block)
        req_id, _ = group.get_match(token_ids=token_ids)
        group.launch(req_id, slot_mapping)
        results = group.wait(req_id, completely=True)
        for resp in results.values():
            assert resp.status == KVResponseStatus.SUCCESS
            total_hit += resp.return_mask.sum().item()
            total_miss += len(resp.return_mask) - resp.return_mask.sum().item()

    assert len(group._indexer_task_map) == 0
    assert total_miss == 0, f"Expected 0 cache miss, got {total_miss}"
    print(f"[test_kv_manager_group_with_indexer] PASSED  hit={total_hit}")

    _shutdown_processes([tp_proc])
    group.shutdown()


# ===================================================================
# Test: KVTPClientGroup without indexer
# ===================================================================

@pytest.mark.parametrize("model_config", [{"tp_size": 1, "dp_size": 1}], indirect=True)
@pytest.mark.parametrize("cache_config", [
    {"enable_cpu": True, "enable_ssd": False, "num_cpu_blocks": 1024},
], indirect=True)
@pytest.mark.parametrize("test_config", [
    {"num_gpu_blocks": 256, "requests_per_block": 16, "initial_write_ratio": 0.4},
], indirect=True)
@pytest.mark.parametrize("gpu_layout_type", [0])
def test_kv_tp_client_group_without_indexer(model_config, cache_config, test_config,
                                            gpu_layout_type):
    """KVTPClientGroup without indexer should register main caches only."""
    tp_size = model_config.tp_size
    num_gpu_blocks = test_config["num_gpu_blocks"]

    skip_if_insufficient_gpus(tp_size)

    # We need a KVManager to act as the server side
    kvmanager = KVManager(model_config, cache_config)
    kvmanager.start()

    mp_ctx = mp.get_context("spawn")
    parent_conn, child_conn = mp_ctx.Pipe()

    proc = mp_ctx.Process(
        target=_run_tp_client_group_no_indexer,
        args=(kvmanager.gpu_register_port, model_config, cache_config,
              num_gpu_blocks, gpu_layout_type, child_conn),
        daemon=True,
    )
    proc.start()

    result = parent_conn.recv()
    parent_conn.close()
    assert result == "OK", f"Worker failed: {result}"

    while not kvmanager.is_ready():
        time.sleep(0.5)

    print("[test_kv_tp_client_group_without_indexer] PASSED")
    _shutdown_processes([proc])
    kvmanager.shutdown()


# ===================================================================
# Test: KVTPClientGroup with indexer
# ===================================================================

@pytest.mark.parametrize("model_config", [{"tp_size": 1, "dp_size": 1}], indirect=True)
@pytest.mark.parametrize("cache_config", [
    {"enable_cpu": True, "enable_ssd": False, "num_cpu_blocks": 1024},
], indirect=True)
@pytest.mark.parametrize("test_config", [
    {"num_gpu_blocks": 256, "requests_per_block": 16, "initial_write_ratio": 0.4},
], indirect=True)
@pytest.mark.parametrize("gpu_layout_type", [0])
def test_kv_tp_client_group_with_indexer(model_config, cache_config, test_config,
                                         gpu_layout_type):
    """KVTPClientGroup with indexer should register both main and indexer caches."""
    tp_size = model_config.tp_size
    num_gpu_blocks = test_config["num_gpu_blocks"]

    skip_if_insufficient_gpus(tp_size)

    indexer_model_config = ModelConfig(
        num_layers=model_config.num_layers,
        num_kv_heads=1, head_size=64, use_mla=True,
        dtype=torch.uint8,
        tp_size=model_config.tp_size, dp_size=model_config.dp_size,
    )
    indexer_cache_config = CacheConfig(
        tokens_per_block=cache_config.tokens_per_block,
        enable_cpu=cache_config.enable_cpu,
        enable_ssd=cache_config.enable_ssd,
        num_cpu_blocks=cache_config.num_cpu_blocks,
    )
    indexer_server_recv_port = GLOBAL_CONFIG_FROM_ENV.server_recv_port + "_indexer"
    indexer_gpu_register_port = indexer_server_recv_port + "_gpu_register"

    # Create main + indexer KVManagers as server side
    main_kvmanager = KVManager(model_config, cache_config)
    indexer_kvmanager = KVManager(
        indexer_model_config, indexer_cache_config,
        server_recv_port=indexer_server_recv_port,
        gpu_register_port=indexer_gpu_register_port,
    )
    main_kvmanager.start()
    indexer_kvmanager.start()

    mp_ctx = mp.get_context("spawn")
    parent_conn, child_conn = mp_ctx.Pipe()

    proc = mp_ctx.Process(
        target=_run_tp_client_group_with_indexer,
        args=(main_kvmanager.gpu_register_port, model_config, cache_config,
              num_gpu_blocks, gpu_layout_type,
              indexer_gpu_register_port,
              indexer_model_config, indexer_cache_config,
              child_conn),
        daemon=True,
    )
    proc.start()

    result = parent_conn.recv()
    parent_conn.close()
    assert result == "OK", f"Worker failed: {result}"

    while not main_kvmanager.is_ready():
        time.sleep(0.5)
    while not indexer_kvmanager.is_ready():
        time.sleep(0.5)

    print("[test_kv_tp_client_group_with_indexer] PASSED")
    _shutdown_processes([proc])
    main_kvmanager.shutdown()
    indexer_kvmanager.shutdown()
