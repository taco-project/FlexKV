import time
import json

import torch
from multiprocessing import Process, Pipe
from flexkv.common.config import ModelConfig, CacheConfig
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.request import KVResponseStatus
from flexkv.kvmanager import KVManager
from flexkv.common.debug import flexkv_logger
from test_utils import GPUKVCacheVerifier, DEFAULT_MODEL_CONFIG, DEFAULT_CACHE_CONFIG
from test_utils import block_ids_2_slot_mapping
# from tests.test_kvmanager import run_tp_client, create_gpu_kv_layout
from flexkv.cache.redis_meta import RedisMeta
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.server.client import KVTPClient

def create_gpu_kv_layout(model_config, cache_config, num_gpu_blocks):
    """Create GPU KV layout"""
    num_layers = model_config.num_layers
    num_kv_heads = model_config.num_kv_heads
    head_size = model_config.head_size
    use_mla = model_config.use_mla
    tp_size = model_config.tp_size
    tokens_per_block = cache_config.tokens_per_block

    tpgroup_gpu_kv_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERWISE,
        num_layer=num_layers,
        num_block=num_gpu_blocks,
        tokens_per_block=tokens_per_block,
        num_head=num_kv_heads,
        head_size=head_size,
        is_mla=model_config.use_mla
    )
    gpu_kv_layout = tpgroup_gpu_kv_layout.div_head(tp_size) if not use_mla else tpgroup_gpu_kv_layout
    return gpu_kv_layout

def run_tp_client(dp_client_id, tp_rank, server_recv_port, model_config, cache_config, num_gpu_blocks, child_conn):
    """Run tp_client process"""
    try:
        device_id = tp_rank + dp_client_id * model_config.tp_size
        tp_client = KVTPClient(server_recv_port, dp_client_id, device_id)

        gpu_kv_layout = create_gpu_kv_layout(model_config, cache_config, num_gpu_blocks)

        # Create GPU blocks for this tp_rank in the tp_client process
        gpu_blocks_for_tp = []
        for _ in range(model_config.num_layers):
            gpu_blocks_for_tp.append(
                torch.empty(size=tuple(gpu_kv_layout.kv_shape[1:]), dtype=model_config.dtype).cuda(device_id)
            )
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
        if child_conn is not None:
            child_conn.send(None)
            child_conn.close()


def initial_cache_config(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return CacheConfig(tokens_per_block = data["tokens_per_block"],
                       enable_cpu = data["enable_cpu"],
                       enable_ssd=data["enable_ssd"],
                       enable_p2p_cpu = data["enable_p2p_cpu"],
                       enable_p2p_ssd= data["enable_p2p_ssd"],
                       enable_kv_sharing = data["enable_kv_sharing"],
                       index_accel=True,
                       cpu_kv_layout_type=KVCacheLayoutType.BLOCKWISE,
                       ssd_kv_layout_type=KVCacheLayoutType.BLOCKWISE,
                       num_cpu_blocks=data["num_cpu_blocks"],
                       num_ssd_blocks=data["num_ssd_blocks"],
                       num_remote_blocks = data["num_remote_blocks"],
                       ssd_cache_dir=data["ssd_cache_dir"],
                       ssd_cache_iouring_entries = data["ssd_cache_iouring_entries"],
                       local_zmq_ip = data["local_zmq_ip"],
                       redis_host = data["redis_host"],
                       redis_password = data["redis_password"],
                       evict_ratio = data["evict_ratio"]
                       )

def initial_model_config():
    return ModelConfig(num_layers=61, num_kv_heads=128, head_size=128, use_mla=True)

def try_get_from_flexKV():
    #necessary to initialize the model_config and cache_config
    #may need more config for distributed kv reuse
    model_config = ModelConfig(**DEFAULT_MODEL_CONFIG)
    cache_config = CacheConfig(**DEFAULT_CACHE_CONFIG)

    cache_config.tokens_per_block = 4
    cache_config.enable_ssd = True
    cache_config.enable_kv_sharing = True
    # cache_config.index_accel = True
    cache_config.enable_p2p_cpu = True
    cache_config.enable_p2p_ssd = True
    cache_config.redis_host = "10.6.131.10"
    cache_config.redis_port = 6379
    cache_config.redis_password = "redis-serving-passwd"
    cache_config.local_zmq_ip = "10.6.131.10"
    cache_config.local_zmq_port = 5555
    cache_config.local_ip = "10.6.131.10"
    cache_config.num_cpu_blocks = 100
    cache_config.num_ssd_blocks = 1000
    cache_config.num_remote_blocks = 2000
    cache_config.ssd_cache_dir =  "/data1/flexkv_ssd/"
    
    

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
    
    time.sleep(20)
    
    block_ids = torch.arange(0, 2, dtype=torch.int64)
    token_ids = torch.arange(0, 8, dtype=torch.int64)
    slot_mapping = block_ids_2_slot_mapping(block_ids, cache_config.tokens_per_block)
    request_id, _ = kvmanager.get_match(
        token_ids=token_ids,
        layer_granularity=-1,
        token_mask=None,
        dp_id=0,
    )
    print("request_id: ",request_id)
    kvmanager.launch(request_id, slot_mapping)
    return_results = kvmanager.wait(request_id, completely=True)
    for kvresponse in return_results.values():
        assert kvresponse.status == KVResponseStatus.SUCCESS
        valid_fetched_tokens = kvresponse.return_mask.sum().item() // \
            cache_config.tokens_per_block * cache_config.tokens_per_block
        print(f"valid_fetched_tokens: {valid_fetched_tokens}")
        assert gpu_kv_verifier.verify_kv_blocks(
            token_ids[:valid_fetched_tokens],
            block_ids[:valid_fetched_tokens//cache_config.tokens_per_block])
    print("get from flexKV done")
    
    
if __name__ == "__main__":
    # test_kvmanager("./cache_config_l.json")
    try_get_from_flexKV()