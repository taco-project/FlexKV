"""
Distributed P2P Interleaved Test - Node A

Test Scenario:
- 100 shared requests generated with the same seed on both nodes
- Node A writes: even_ids = [0, 2, 4, ..., 98]  (50 requests)
- Node B writes: odd_ids  = [1, 3, 5, ..., 99]  (50 requests)
- After sync, Node A reads: [0-49] (25 local + 25 remote from B)
- After sync, Node B reads: [50-99] (25 local + 25 remote from A)
"""

import time
import json
import numpy as np
import torch
import os
from multiprocessing import Process, Pipe

from flexkv.common.config import ModelConfig, CacheConfig
from flexkv.kvmanager import KVManager
from flexkv.common.debug import flexkv_logger
from flexkv.common.request import KVResponseStatus
from test_utils import GPUKVCacheVerifier, DEFAULT_CACHE_CONFIG, DEFAULT_MODEL_CONFIG
from test_utils import block_ids_2_slot_mapping


def generate_shared_requests(num_requests=100, tokens_per_block=4, blocks_per_request=16):
    """
    Generate shared request set with deterministic seed
    Both Node A and B will generate the same requests
    """
    # Use fixed seed for reproducibility across nodes
    np.random.seed(42)
    torch.manual_seed(42)
    
    requests = []
    for req_id in range(num_requests):
        token_length = tokens_per_block * blocks_per_request
        # Generate unique token_ids for each request
        token_ids = torch.arange(
            req_id * token_length,
            (req_id + 1) * token_length,
            dtype=torch.int64
        )
        
        # Block IDs are unique per request
        block_ids = torch.arange(
            req_id * blocks_per_request,
            (req_id + 1) * blocks_per_request,
            dtype=torch.int64
        )
        
        requests.append((req_id, token_ids, block_ids))
    
    return requests


def create_gpu_kv_layout(model_config, cache_config, num_gpu_blocks):
    """Create GPU KV cache layout"""
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
    
    gpu_kv_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERWISE,
        num_layer=model_config.num_layers,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        num_block=num_gpu_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        is_mla=model_config.use_mla,
    )
    return gpu_kv_layout


def run_tp_client(dp_client_id, tp_rank, server_recv_port, model_config, cache_config, num_gpu_blocks, child_conn):
    """Run TP client process"""
    try:
        from flexkv.server.client import KVTPClient
        from flexkv.common.memory_handle import TensorSharedHandle
        
        device_id = tp_rank + dp_client_id * model_config.tp_size
        tp_client = KVTPClient(server_recv_port, dp_client_id, device_id)
        
        gpu_kv_layout = create_gpu_kv_layout(model_config, cache_config, num_gpu_blocks)
        
        # Create GPU blocks
        gpu_blocks_for_tp = []
        for _ in range(model_config.num_layers):
            gpu_blocks_for_tp.append(
                torch.empty(size=tuple(gpu_kv_layout.kv_shape[1:]), dtype=model_config.dtype).cuda(device_id)
            )
        tp_client.register_to_server(gpu_blocks_for_tp, gpu_kv_layout)
        
        # Send GPU blocks back via pipe
        if child_conn is not None:
            print(f"[TP Client {tp_rank}] Converting {len(gpu_blocks_for_tp)} GPU blocks to TensorSharedHandle")
            shared_gpu_blocks = [TensorSharedHandle(tensor) for tensor in gpu_blocks_for_tp]
            child_conn.send(shared_gpu_blocks)
            print(f"[TP Client {tp_rank}] Sent GPU blocks to main process via pipe")
            child_conn.close()
        
        # Keep running
        while True:
            time.sleep(1)
    except Exception as e:
        print(f"[TP Client {tp_rank}] Error: {e}")
        if child_conn is not None:
            child_conn.send(None)
            child_conn.close()


def test_node_a():
    """Node A: Write even IDs, Read 0-49"""
    print("=" * 80)
    print("Node A: Distributed P2P Interleaved Test")
    print("=" * 80)
    
    # Print environment info
    print(f"Node A IP: {os.getenv('NODE_A_IP', '10.6.131.9')}")
    print(f"Node B IP: {os.getenv('NODE_B_IP', '10.6.131.10')}")
    print(f"Redis Host: {os.getenv('REDIS_HOST', '10.6.131.10')}")
    print("=" * 80)
    
    # Configuration
    model_config = ModelConfig(**DEFAULT_MODEL_CONFIG)
    model_config.tp_size = 1
    cache_config = CacheConfig(**DEFAULT_CACHE_CONFIG)
    num_total_request = 100
    cache_config.tokens_per_block = 4
    cache_config.enable_ssd = True
    cache_config.enable_kv_sharing = True
    cache_config.enable_p2p_cpu = True
    cache_config.enable_p2p_ssd = True
    cache_config.redis_host = "10.6.131.10"
    cache_config.redis_port = 6379
    cache_config.redis_password = "redis-serving-passwd"
    cache_config.local_zmq_ip = "10.6.131.9"
    cache_config.local_zmq_port = 5555
    cache_config.local_ip = "10.6.131.9"
    cache_config.num_cpu_blocks = 300
    cache_config.num_ssd_blocks = 1000
    cache_config.ssd_cache_dir = "/data/flexkv_ssd/"
    cache_config.refresh_batch_size = 256
    cache_config.lease_ttl_ms = 10000          # 10秒租约
    cache_config.renew_lease_ms = 4000         # 5秒续约（降低Redis负载）
    cache_config.remote_rebuild_interval_ms = 10000  # 10秒重建远程索引
    cache_config.safety_ttl_ms = 100
    cache_config.swap_block_threshold = 100
    
    # 主动式驱逐策略：提前预留buffer空间
    cache_config.evict_start_threshold = 0.7  # CPU使用率达80%就开始驱逐
    cache_config.evict_ratio = 0.1            # 每次至少驱逐15%的blocks
    
    num_gpu_blocks = 512
    tokens_per_block = cache_config.tokens_per_block
    
    import uuid
    gpu_register_port = f"ipc:///tmp/flexkv_gpu_{uuid.uuid4().hex[:8]}"
    server_recv_port = f"ipc:///tmp/flexkv_srv_{uuid.uuid4().hex[:8]}"
    
    # Initialize KVManager
    kvmanager = KVManager(model_config, cache_config, gpu_register_port, server_recv_port)
    kvmanager.start()
    
    # Start TP client
    print("[Main Process] Waiting to receive GPU blocks from 1 TP client processes...")
    pipe_connections = []
    tp_client_processes = []
    
    for tp_rank in range(model_config.tp_size):
        parent_conn, child_conn = Pipe()
        pipe_connections.append(parent_conn)
        
        tp_client_process = Process(
            target=run_tp_client,
            args=(0, tp_rank, gpu_register_port, model_config, cache_config, num_gpu_blocks, child_conn),
            daemon=True
        )
        tp_client_processes.append(tp_client_process)
        tp_client_process.start()
    
    # Collect GPU blocks
    all_gpu_blocks = []
    for tp_rank, parent_conn in enumerate(pipe_connections):
        try:
            shared_gpu_blocks = parent_conn.recv()
            if shared_gpu_blocks is not None:
                all_gpu_blocks.append(shared_gpu_blocks)
                print(f"[Main Process] Received GPU blocks from TP client {tp_rank}")
            parent_conn.close()
        except Exception as e:
            print(f"[Main Process] Error receiving from TP client {tp_rank}: {e}")
    
    # Create GPUKVCacheVerifier
    gpu_kv_verifier = None
    if all_gpu_blocks and len(all_gpu_blocks) == model_config.tp_size:
        print(f"[Main Process] Creating GPUKVCacheVerifier with GPU blocks from {len(all_gpu_blocks)} TP clients")
        gpu_kv_layout = create_gpu_kv_layout(model_config, cache_config, num_gpu_blocks)
        
        gpu_kv_verifier = GPUKVCacheVerifier(
            shared_gpu_blocks=all_gpu_blocks,
            gpu_kv_layout=gpu_kv_layout,
            tp_size=model_config.tp_size,
            tokens_per_block=cache_config.tokens_per_block,
            dtype=model_config.dtype
        )
        print("[Main Process] GPUKVCacheVerifier created successfully")
    
    # Wait for KVManager to be ready
    while not kvmanager.is_ready():
        time.sleep(1)
        flexkv_logger.info("waiting for flexkv to be ready")
    
    print("\n" + "=" * 80)
    print("Phase 1: Generate Shared Requests")
    print("=" * 80)
    
    # Generate 100 shared requests
    all_requests = generate_shared_requests(num_requests=num_total_request, tokens_per_block=tokens_per_block, blocks_per_request=4)
    print(f"Generated {len(all_requests)} shared requests")
    #for req_id, token_ids, block_ids in all_requests:
    #    print(f"Request {req_id}: Token IDs: {token_ids}, Block IDs: {block_ids}")
    print("\n" + "=" * 80)
    print("Phase 2: Node A Writes EVEN IDs [0, 2, 4, ..., 98]")
    print("=" * 80)
    
    # Node A writes even IDs
    my_write_ids = list(range(0, num_total_request, 2))  # [0, 2, 4, ..., 98]
    print(f"Node A will write {len(my_write_ids)} requests: {my_write_ids[:10]}...")
    
    write_start_time = time.time()
    for req_id in my_write_ids:
        _, token_ids, block_ids = all_requests[req_id]
        
        # Fill GPU blocks with ground truth
        if gpu_kv_verifier is not None:
            gpu_kv_verifier.fill_gpu_blocks(token_ids, block_ids)
        
        # Put into FlexKV
        slot_mapping = block_ids_2_slot_mapping(block_ids, tokens_per_block)
        write_request = kvmanager.put_async(
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            token_mask=None,
            dp_id=0,
        )
        kvmanager.wait([write_request], completely=True)
        time.sleep(1)
        if (req_id + 1) % 10 == 0:
            print(f"  Written {len([x for x in my_write_ids if x <= req_id])}/{len(my_write_ids)} requests")
    
    write_elapsed = time.time() - write_start_time
    print(f"Node A write completed in {write_elapsed:.2f} seconds")
    
    print("\n" + "=" * 80)
    print("Phase 3: Wait for Remote Index Synchronization")
    print("=" * 80)
    
    # Wait for remote index to sync (rebuild_interval_ms = 1000ms)
    sync_wait_time = 20.0
    print(f"Waiting {sync_wait_time} seconds for Node B's index to sync...")
    time.sleep(sync_wait_time)
    print("Sync wait completed")
    
    print("\n" + "=" * 80)
    print("Phase 4: Node A Reads [0-49] (25 local + 25 remote)")
    print("=" * 80)
    
    # Node A reads 0-49
    my_read_ids = list(range(0, num_total_request // 2))  # [0, 1, 2, ..., 49]
    print(f"Node A will read {len(my_read_ids)} requests")
    print(f"  Expected local hits:  {len([x for x in my_read_ids if x % 2 == 0])} (even IDs)")
    print(f"  Expected remote hits: {len([x for x in my_read_ids if x % 2 == 1])} (odd IDs from Node B)")
    print(f"  Expected total hits: {len(my_read_ids)}")
    
    read_start_time = time.time()
    local_hits = 0
    remote_hits = 0
    verification_passed = 0
    verification_failed = 0
    
    for req_id in my_read_ids:
        _, token_ids, block_ids = all_requests[req_id]
        
        # Get from FlexKV
        slot_mapping = block_ids_2_slot_mapping(block_ids, tokens_per_block)
        request_id, _ = kvmanager.get_match(
            token_ids=token_ids,
            layer_granularity=-1,
            token_mask=None,
            dp_id=0,
        )
        kvmanager.launch(request_id, slot_mapping)
        return_results = kvmanager.wait([request_id], completely=True)
        
        # Verify
        kvresponse = return_results[request_id]
        if kvresponse.status != KVResponseStatus.SUCCESS:
            verification_failed += 1
            print(f"  ❌ Request {req_id}: {kvresponse.status}")
            continue
        
        valid_fetched_tokens = kvresponse.return_mask.sum().item() // tokens_per_block * tokens_per_block
        
        # All tokens should be fetched
        if valid_fetched_tokens == len(token_ids):
            # Count local vs remote
            if req_id % 2 == 0:
                local_hits += 1
            else:
                remote_hits += 1
            # Verify data correctness
            if gpu_kv_verifier is not None:
                if gpu_kv_verifier.verify_kv_blocks(
                    token_ids[:valid_fetched_tokens],
                    block_ids[:valid_fetched_tokens // tokens_per_block]
                ):
                    verification_passed += 1
                else:
                    verification_failed += 1
                    print(f"  ❌ Request {req_id}: Data verification FAILED")
        else:
            verification_failed += 1
            print(f"  ❌ Request {req_id}: Only {valid_fetched_tokens}/{len(token_ids)} tokens fetched")
        
        time.sleep(0.5)
        if (req_id + 1) % 10 == 0:
            print(f"  Read {req_id + 1}/{len(my_read_ids)} requests")
    
    read_elapsed = time.time() - read_start_time
    
    print("\n" + "=" * 80)
    print("Phase 5: Results Summary")
    print("=" * 80)
    
    print(f"\n📊 Node A Statistics:")
    print(f"  Write Phase:")
    print(f"    - Requests written: {len(my_write_ids)}")
    print(f"    - Time elapsed: {write_elapsed:.2f} seconds")
    print(f"    - Throughput: {len(my_write_ids) / write_elapsed:.2f} req/s")
    
    print(f"\n  Read Phase:")
    print(f"    - Total reads: {len(my_read_ids)}")
    print(f"    - Local hits:  {local_hits} ({local_hits / len(my_read_ids) * 100:.1f}%)")
    print(f"    - Remote hits: {remote_hits} ({remote_hits / len(my_read_ids) * 100:.1f}%)")
    print(f"    - Time elapsed: {read_elapsed:.2f} seconds")
    print(f"    - Throughput: {len(my_read_ids) / read_elapsed:.2f} req/s")
    
    print(f"\n  Verification:")
    print(f"    - ✅ Passed: {verification_passed}")
    print(f"    - ❌ Failed: {verification_failed}")
    
    # Assertions
    assert verification_failed == 0, f"Verification failed for {verification_failed} requests"
    
    print(f"\n✅ All tests passed!")
    print("=" * 80)
    final_wait = 200
    while final_wait > 0:
        time.sleep(1)
        final_wait -= 1
        print(f"Waiting {final_wait} seconds for final cleanup...")
    
    # Cleanup
    for tp_process in tp_client_processes:
        if tp_process.is_alive():
            tp_process.terminate()
            tp_process.join(timeout=5)
    
    kvmanager.shutdown()


if __name__ == "__main__":
    test_node_a()


