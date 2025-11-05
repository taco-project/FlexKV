import time
import os
import shutil

import pytest
import torch
import multiprocessing as mp
from multiprocessing import Process, Pipe

from flexkv.common.config import ModelConfig, CacheConfig, GLOBAL_CONFIG_FROM_ENV
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.request import KVResponseStatus
from flexkv.kvtask import KVTaskEngine
from flexkv.kvmanager import KVManager
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.server.client import KVTPClient
from flexkv.common.debug import flexkv_logger

# Import utilities from test_utils
from test_utils import (
    DEFAULT_MODEL_CONFIG, DEFAULT_CACHE_CONFIG, DEFAULT_TEST_CONFIG,
    generate_request_pair, verify_data, block_ids_2_slot_mapping,
    generate_gpu_blocks_with_ground_truth, skip_if_insufficient_gpus,
    create_gpu_kv_layout, GPUKVCacheVerifier
)

def run_tp_client(dp_client_id,
                  tp_rank,
                  server_recv_port,
                  model_config,
                  cache_config,
                  num_gpu_blocks,
                  child_conn,
                  gpu_layout_type):
    """Run tp_client process"""
    try:
        device_id = tp_rank + dp_client_id * model_config.tp_size
        tp_client = KVTPClient(server_recv_port, dp_client_id, device_id)

        gpu_kv_layout = create_gpu_kv_layout(model_config, cache_config, num_gpu_blocks, gpu_layout_type)

        # Create GPU blocks for this tp_rank in the tp_client process
        gpu_blocks_for_tp = []
        if gpu_layout_type == 0:
            for _ in range(model_config.num_layers):
                gpu_blocks_for_tp.append(
                    torch.empty(size=tuple(gpu_kv_layout.kv_shape[1:]), dtype=model_config.dtype).cuda(device_id)
                )
        elif gpu_layout_type == 1:
            gpu_blocks_for_tp.append(
                torch.empty(size=tuple(gpu_kv_layout.kv_shape[:]), dtype=model_config.dtype).cuda(device_id)
            )
        elif gpu_layout_type == 2:
            for _ in range(model_config.num_layers * 2):
                gpu_blocks_for_tp.append(
                    torch.empty(size=tuple(gpu_kv_layout.kv_shape[2:]), dtype=model_config.dtype).cuda(device_id)
                )
        else:
            raise ValueError(f"Invalid GPU layout type: {gpu_layout_type}")
        tp_client.register_to_server(gpu_blocks_for_tp, gpu_kv_layout)

        # Send GPU blocks back to main process via pipe if connection provided
        if child_conn is not None:
            print(f"[TP Client {tp_rank}] Converting {len(gpu_blocks_for_tp)} GPU blocks to TensorSharedHandle")
            shared_gpu_blocks = [TensorSharedHandle(tensor) for tensor in gpu_blocks_for_tp]
            child_conn.send(shared_gpu_blocks)
            print(f"[TP Client {tp_rank}] Sent GPU blocks to main process via pipe")
            child_conn.close()

        # Keep the process running
        while True:
            time.sleep(1)
    except Exception as e:
        print(f"[TP Client {tp_rank}] Exception occurred: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        if child_conn is not None:
            child_conn.send(None)
            child_conn.close()

def shutdown_tp_client(tp_client_processes):
    for tp_process in tp_client_processes:
        if tp_process.is_alive():
            tp_process.terminate()
            tp_process.join(timeout=5)
            if tp_process.is_alive():
                print(f"Force killing tp_client process {tp_process.pid}")
                tp_process.kill()
                tp_process.join(timeout=2)

@pytest.mark.parametrize("model_config", [
    {'tp_size': 1, 'dp_size': 1},
    {'tp_size': 2, 'dp_size': 2},
    {'dtype': torch.float32},
    {'use_mla': True},
    {'tp_size': 4, 'dp_size': 1, 'use_mla': True},
], indirect=True)
@pytest.mark.parametrize("cache_config", [
    {'enable_cpu': True, 'enable_ssd': False, 'num_cpu_blocks': 1024},
    {'enable_cpu': True, 'enable_ssd': True},
    {'enable_cpu': True, 'enable_ssd': True, 'num_ssd_blocks': 256},
    # GDS test configs
    {'enable_cpu': True, 'enable_gds': True, 'enable_ssd': False, \
        'num_gds_blocks': 512, 'gds_cache_dir': ["./gdstest"]},
], indirect=True)
@pytest.mark.parametrize("test_config", [
    {'num_gpu_blocks': 512, 'requests_per_block': 16, 'initial_write_ratio': 0.4},
], indirect=True)
@pytest.mark.parametrize("gpu_layout_type", [
    0,
    1,
    2,
])
def test_kvmanager(model_config, cache_config, test_config, gpu_layout_type):
    tp_size = model_config.tp_size
    dp_size = model_config.dp_size

    tokens_per_block = cache_config.tokens_per_block
    num_cpu_blocks = cache_config.num_cpu_blocks
    num_ssd_blocks = cache_config.num_ssd_blocks
    num_gds_blocks = cache_config.num_gds_blocks

    enable_cpu = cache_config.enable_cpu
    enable_ssd = cache_config.enable_ssd
    enable_remote = cache_config.enable_remote
    enable_gds = cache_config.enable_gds

    num_gpu_blocks = test_config["num_gpu_blocks"]
    block_per_request = test_config['requests_per_block']
    initial_write_ratio = test_config['initial_write_ratio']

    num_requests = num_gpu_blocks // block_per_request

    # Skip tests based on GPU availability and configuration
    skip_if_insufficient_gpus(tp_size * dp_size)

    if enable_gds and os.environ.get("FLEXKV_GDS_TEST", "0") == "0":
        pytest.skip("skip because GDS test is not enabled")

    if enable_remote:
        pytest.skip("skip because enable_remote is not supported")

    if dp_size > 1:
         #note that for now only dp_size=1 is supported
        pytest.skip("skip because server-client mode is not ready for dp_size > 1")

    import uuid
    gpu_register_port = f"ipc:///tmp/flexkv_gpu_{uuid.uuid4().hex[:8]}"
    server_recv_port = f"ipc:///tmp/flexkv_srv_{uuid.uuid4().hex[:8]}"
    kvmanager = KVManager(model_config, cache_config, gpu_register_port, server_recv_port)
    kvmanager.start()

    # Create pipes for each tp_client to send GPU blocks back
    mp_ctx = mp.get_context('spawn')
    pipe_connections = []
    tp_client_processes = []

    for tp_rank in range(tp_size):
        parent_conn, child_conn = mp_ctx.Pipe()
        pipe_connections.append(parent_conn)

        tp_client_process = mp_ctx.Process(
            target=run_tp_client,
            args=(0, tp_rank, gpu_register_port, model_config, cache_config, \
                num_gpu_blocks + tp_rank, child_conn, gpu_layout_type),
            daemon=True
        )
        tp_client_processes.append(tp_client_process)
        tp_client_process.start()

    # Collect GPU blocks from all tp_client processes
    print(f"[Main Process] Waiting to receive GPU blocks from {tp_size} TP client processes...")
    all_gpu_blocks = []

    for tp_rank, parent_conn in enumerate(pipe_connections):
        try:
            shared_gpu_blocks = parent_conn.recv()
            if shared_gpu_blocks is not None:
                all_gpu_blocks.append(shared_gpu_blocks)
                print(f"[Main Process] Received GPU blocks from TP client {tp_rank}")
            else:
                print(f"[Main Process] TP client {tp_rank} failed to create GPU blocks")
            parent_conn.close()
        except Exception as e:
            print(f"[Main Process] Error receiving from TP client {tp_rank}: {e}")

    # Create GPUKVCacheVerifier with collected GPU blocks
    if all_gpu_blocks and len(all_gpu_blocks) == tp_size:
        print(f"[Main Process] Creating GPUKVCacheVerifier with GPU blocks from {len(all_gpu_blocks)} TP clients")

        # Get gpu_kv_layout from cache_config for GPUKVCacheVerifier
        gpu_kv_layout = create_gpu_kv_layout(model_config, cache_config, num_gpu_blocks, gpu_layout_type)

        gpu_kv_verifier = GPUKVCacheVerifier(
            shared_gpu_blocks=all_gpu_blocks,
            gpu_kv_layout=gpu_kv_layout,
            tp_size=model_config.tp_size,
            tokens_per_block=cache_config.tokens_per_block,
            dtype=model_config.dtype,
            gpu_layout_type=gpu_layout_type
        )
        print("[Main Process] GPUKVCacheVerifier created successfully")
    else:
        print(f"[Main Process] Failed to collect GPU blocks from all TP clients. "
              f"Got {len(all_gpu_blocks)} out of {tp_size}")
        gpu_kv_verifier = None

    while not kvmanager.is_ready():
        time.sleep(1)
        flexkv_logger.info("waiting for flexkv to be ready")

    num_remote_blocks = cache_config.num_remote_blocks
    request_pairs = [generate_request_pair(i, block_per_request, num_gpu_blocks, tokens_per_block, dp_size)
                     for i in range(num_requests)]
    initial_write_num = int(num_requests * initial_write_ratio)
    print("writing initial data...")
    put_ids = []
    for token_ids, block_ids, dp_id in request_pairs[:initial_write_num]:
        if gpu_kv_verifier is not None:
            gpu_kv_verifier.fill_gpu_blocks(token_ids, block_ids)
        write_request = kvmanager.put_async(
            token_ids=token_ids,
            slot_mapping=block_ids_2_slot_mapping(block_ids, tokens_per_block),
            token_mask=None,
            dp_id=dp_id,
        )
        kvmanager.wait([write_request], completely=True)

    #corner case: input token length for put is less than tokens_per_block
    write_request = kvmanager.put_async(
        token_ids=torch.randint(0, 100, size=(8,), dtype=torch.int64),
        slot_mapping=block_ids_2_slot_mapping(torch.arange(0,1, dtype=torch.int64), tokens_per_block, actual_length=8),
        token_mask=None,
        dp_id=0,
    )
    kvmanager.wait([write_request], completely=True)
    #corner case: input token length is long enough, but the mask is less than tokens_per_block
    #my_mask = torch.zeros(16, dtype=torch.bool)
    #my_mask[0:8] = True
    #write_request = kvmanager.put_async(
    #    token_ids=torch.randint(0, 100, size=(16,), dtype=torch.int64),
    #    slot_mapping=block_ids_2_slot_mapping(torch.arange(0,1, dtype=torch.int64), tokens_per_block, actual_length=8),
    #    token_mask=my_mask,
    #    dp_id=0,
    #)
    #kvmanager.wait_for_graph_finished(write_request)

    print(f"initial data {initial_write_num} written")
    total_cache_hit = 0
    total_cache_miss = 0
    running_get_requests = []
    running_put_requests = []
    req_id2block_ids = {}
    req_id2token_ids = {}
    flexkv_id2req_id = {}
    start_time = time.time()
    print(f"the initial {initial_write_num} write done,performing mixed read/write...")
    for i in range(initial_write_num, num_requests):
        print(f"performing mixed read/write {i} / {num_requests} ...")
        read_idx = i - initial_write_num
        token_ids, block_ids, dp_id = request_pairs[read_idx]
        slot_mapping = block_ids_2_slot_mapping(block_ids, tokens_per_block)
        request_id, _ = kvmanager.get_match(
            token_ids=token_ids,
            layer_granularity=-1,
            token_mask=None,
            dp_id=dp_id,
        )
        kvmanager.launch(request_id, slot_mapping)
        flexkv_id2req_id[request_id] = read_idx
        running_get_requests.append(request_id)
        req_id2block_ids[request_id] = block_ids
        req_id2token_ids[request_id] = token_ids
        token_ids, block_ids, dp_id = request_pairs[i]
        if gpu_kv_verifier is not None:
            gpu_kv_verifier.fill_gpu_blocks(token_ids, block_ids)
        request_id = kvmanager.put_async(
            token_ids=token_ids,
            slot_mapping=block_ids_2_slot_mapping(block_ids, tokens_per_block),
            token_mask=None,
            dp_id=dp_id,
        )
        flexkv_id2req_id[request_id] = i
        print(f"write flexkv request_id {request_id} to req_id {i}")
        running_put_requests.append(request_id)
        min_block_num = min(num_cpu_blocks, num_gpu_blocks)
        if (len(running_get_requests) + len(running_put_requests) >= min_block_num // block_per_request - 2 or
            i % initial_write_num == initial_write_num - 1 or
            i == num_requests - 1):
            if len(running_put_requests) > 0:
                kvmanager.wait(running_put_requests, completely=True)
            if len(running_get_requests) > 0:
                return_results = kvmanager.wait(running_get_requests, completely=True)
                if gpu_kv_verifier is not None:
                    for req_id, kvresponse in return_results.items():
                        assert kvresponse.status == KVResponseStatus.SUCCESS
                        valid_fetched_tokens = kvresponse.return_mask.sum().item() // \
                            tokens_per_block * tokens_per_block
                        token_ids = req_id2token_ids[req_id]
                        block_ids = req_id2block_ids[req_id]
                        assert gpu_kv_verifier.verify_kv_blocks(
                            token_ids[:valid_fetched_tokens],
                            block_ids[:valid_fetched_tokens//tokens_per_block])
                for kvresponse in return_results.values():
                    assert kvresponse.status == KVResponseStatus.SUCCESS
                    total_cache_hit += kvresponse.return_mask.sum().item()
                    total_cache_miss += len(kvresponse.return_mask) - kvresponse.return_mask.sum().item()
            running_get_requests = []
            running_put_requests = []
    if len(running_get_requests) > 0:
        return_results = kvmanager.wait(running_get_requests, completely=True)
        if gpu_kv_verifier is not None:
            for req_id, kvresponse in return_results.items():
                assert kvresponse.status == KVResponseStatus.SUCCESS
                valid_fetched_tokens = kvresponse.return_mask.sum().item() // tokens_per_block * tokens_per_block
                token_ids = req_id2token_ids[req_id]
                block_ids = req_id2block_ids[req_id]
                assert gpu_kv_verifier.verify_kv_blocks(
                    token_ids[:valid_fetched_tokens],
                    block_ids[:valid_fetched_tokens//tokens_per_block])
        running_get_requests = []
    if len(running_put_requests) > 0:
        kvmanager.wait(running_put_requests, completely=True)
        running_put_requests = []
    print("mixed read/write done")
    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time: {total_time} s")
    print(f"Total cache hit rate: {total_cache_hit / (total_cache_hit + total_cache_miss)}")
    if enable_cpu and num_cpu_blocks >= num_gpu_blocks or \
        enable_ssd and num_ssd_blocks >= num_gpu_blocks or \
        enable_remote and num_remote_blocks >= num_gpu_blocks or \
        enable_gds and num_gds_blocks >= num_gpu_blocks:
        assert total_cache_miss == 0
    shutdown_tp_client(tp_client_processes)
    kvmanager.shutdown()

    # Only verify data in direct mode
    # verify_data(gpu_blocks, dp_wise_gpu_blocks_gt, num_kv_heads, tp_size, dp_size, num_layers, use_mla)
    if total_cache_miss == 0:
        return
    elif total_cache_miss > 0:
        print(f"verify skipped, because of total_cache_miss={total_cache_miss} > 0")
