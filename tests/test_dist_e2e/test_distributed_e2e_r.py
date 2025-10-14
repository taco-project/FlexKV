import time
import json

import torch
from multiprocessing import Process, Pipe
from flexkv.common.config import ModelConfig, CacheConfig
from flexkv.kvmanager import KVManager
from flexkv.common.debug import flexkv_logger
from test_utils import GPUKVCacheVerifier, DEFAULT_CACHE_CONFIG, DEFAULT_MODEL_CONFIG
from test_utils import block_ids_2_slot_mapping
from flexkv.cache.redis_meta import RedisMeta
from test_distributed_e2e_l import create_gpu_kv_layout, run_tp_client
def put_into_flexKV():
    #necessary to initialize the model_config and cache_config
    #may need more config for distributed kv reuse
    model_config = ModelConfig(**DEFAULT_MODEL_CONFIG)
    cache_config = CacheConfig(**DEFAULT_CACHE_CONFIG)

    cache_config.tokens_per_block = 4
    cache_config.enable_ssd = True
    cache_config.enable_kv_sharing = True
    cache_config.enable_p2p_cpu = True
    cache_config.enable_p2p_ssd = True
    # cache_config.index_accel = True
    cache_config.redis_host = "10.6.131.10"
    cache_config.redis_port = 6379
    cache_config.redis_password = "redis-serving-passwd"
    cache_config.local_zmq_ip = "10.6.131.9"
    cache_config.local_zmq_port = 5555
    cache_config.local_ip = "10.6.131.9"
    cache_config.num_cpu_blocks = 100
    cache_config.num_ssd_blocks = 1000
    cache_config.num_remote_blocks = 2000
    cache_config.ssd_cache_dir =  "/data/flexkv_ssd/"

    # redis_meta = RedisMeta(
    #     cache_config.redis_host,
    #     cache_config.redis_port,
    #     cache_config.redis_password,
    #     "127.0.0.1"
    # )
    # redis_meta.init_meta()
    # node_id = redis_meta.get_node_id()
    # print(f"[Node B] node id: {node_id}")
    # cache_config.distributed_node_id = node_id

    num_gpu_blocks = 128
    import uuid
    gpu_register_port = f"ipc:///tmp/flexkv_gpu_{uuid.uuid4().hex[:8]}"
    server_recv_port = f"ipc:///tmp/flexkv_srv_{uuid.uuid4().hex[:8]}"
    kvmanager = KVManager(model_config, cache_config, gpu_register_port, server_recv_port)
    kvmanager.start()

    # the following 3 steps are directly copied from test_kvmanager.py
    # 1. Create pipes for each tp_client to send GPU blocks back
    pipe_connections = []
    tp_client_processes = []

    for tp_rank in range(model_config.tp_size):
        parent_conn, child_conn = Pipe()
        pipe_connections.append(parent_conn)

        tp_client_process = Process(
            target=run_tp_client,
            args=(0, tp_rank, gpu_register_port, model_config, cache_config, num_gpu_blocks + tp_rank, child_conn),
            daemon=True
        )
        tp_client_processes.append(tp_client_process)
        tp_client_process.start()

    # 2. Collect GPU blocks from all tp_client processes
    print(f"[Main Process] Waiting to receive GPU blocks from {model_config.tp_size} TP client processes...")
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

    # 3. Create GPUKVCacheVerifier with collected GPU blocks
    if all_gpu_blocks and len(all_gpu_blocks) == model_config.tp_size:
        print(f"[Main Process] Creating GPUKVCacheVerifier with GPU blocks from {len(all_gpu_blocks)} TP clients")

        # Get gpu_kv_layout from cache_config for GPUKVCacheVerifier
        gpu_kv_layout = create_gpu_kv_layout(model_config, cache_config, num_gpu_blocks)

        gpu_kv_verifier = GPUKVCacheVerifier(
            shared_gpu_blocks=all_gpu_blocks,
            gpu_kv_layout=gpu_kv_layout,
            tp_size=model_config.tp_size,
            tokens_per_block=cache_config.tokens_per_block,
            dtype=model_config.dtype
        )
        print("[Main Process] GPUKVCacheVerifier created successfully")
    else:
        print(f"[Main Process] Failed to collect GPU blocks from all TP clients. "
              f"Got {len(all_gpu_blocks)} out of {model_config.tp_size}")
        gpu_kv_verifier = None

    while not kvmanager.is_ready():
        time.sleep(1)
        flexkv_logger.info("waiting for flexkv to be ready")

    #put a test request into flexKV and wait
    block_ids = torch.arange(0, 4, dtype=torch.int64)
    token_ids = torch.arange(0, 16, dtype=torch.int64)
    if gpu_kv_verifier is not None:
        gpu_kv_verifier.fill_gpu_blocks(token_ids, block_ids)
    write_request = kvmanager.put_async(
        token_ids=token_ids,
        slot_mapping=block_ids_2_slot_mapping(block_ids, cache_config.tokens_per_block),
        token_mask=None,
        dp_id=0,
    )
    kvmanager.wait([write_request], completely=True)
    print("put into flexkv finished. start waiting...")
    while True:
        time.sleep(1)

if __name__ == "__main__":
    put_into_flexKV()









