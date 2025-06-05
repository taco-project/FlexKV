import torch
from typing import Tuple
import time
import tempfile

from multiprocessing import Process
from flexkv.server.server import KVServer
from flexkv.server.client import KVDPClient, KVTPClient
from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.debug import init_logger

logger = init_logger(__name__)


num_layers = 32
num_kv_heads = 8
head_size = 128
num_cpu_blocks = 300
num_gpu_blocks = 30
tp_size = 2
dp_size = 1
tokens_per_block = 4
cpu_shape = (num_layers, 2, num_cpu_blocks, tokens_per_block, num_kv_heads, head_size)
gpu_shape = (num_layers, 2, num_gpu_blocks, tokens_per_block, num_kv_heads, head_size)

model_config = ModelConfig(num_layers=num_layers,
                            num_kv_heads=num_kv_heads,
                            head_size=head_size,
                            element_size=2,
                            use_mla=False,
                            tp_size=tp_size)

cache_config = CacheConfig(enable_cpu=True,
                            enable_ssd=False,
                            enable_remote=False,
                            use_gds=False,
                            use_pinned_memory=True,
                            tokens_per_block=tokens_per_block,
                            num_cpu_blocks=num_cpu_blocks,)



def run_dp_client(server_recv_port):
    """Client process function"""
    logger.info(f"start dp client process")
    # Initialize client
    dp_client = KVDPClient(server_recv_port, model_config)
            
    logger.info(f"start tp client process")
    tp_client_processes: list[Process] = []
    for tp_rank in range(model_config.tp_size):
        tp_client_process = Process(
            target=run_tp_client,
            args=(dp_client.dp_client_id, tp_rank, tp_rank, server_recv_port,),
            daemon=True,
        )
        tp_client_process.start()
        tp_client_processes.append(tp_client_process)
        
    # wait tp client registered
    time.sleep(5)
    
    # get/put/wait
    tokens_per_block = gpu_shape[3]
    token_ids = torch.randint(0, 1000, (64, ))
    token_mask = torch.ones(64)
    gpu_physical_block_ids = torch.tensor([i for i in range(64//tokens_per_block)]).pin_memory()
    slot_mapping = gpu_physical_block_ids.repeat_interleave(tokens_per_block) * tokens_per_block
    
    # Example workload
    # Send some requests
    request_id1 = dp_client.put_async(token_ids, slot_mapping, token_mask)
    
    token_ids = torch.randint(0, 1000, (64, ))
    token_mask = torch.ones(64)
    gpu_physical_block_ids = torch.tensor([20-i for i in range(64//tokens_per_block)]).pin_memory()
    slot_mapping = gpu_physical_block_ids.repeat_interleave(tokens_per_block) * tokens_per_block
    request_id2 = dp_client.put_async(token_ids, slot_mapping, token_mask)
    # Process responses
    results = dp_client.wait([request_id1, request_id2])
    logger.info(f"Client {dp_client.dp_client_id} got result for request {request_id1}")
    logger.info(f"Client {dp_client.dp_client_id} got result for request {request_id2}")
    logger.info(results)
    
    token_ids[40:] += 1
    request_id3 = dp_client.get_async(token_ids, slot_mapping, token_mask)
    results = dp_client.wait([request_id3])
    logger.info(f"Client {dp_client.dp_client_id} got result for request {request_id3}")
    logger.info(results)
    
    logger.info(f"dp client tasks finish")

    
def run_tp_client(dp_client_id, tp_rank, device_id, server_recv_port):
    
    tp_client = KVTPClient(server_recv_port, dp_client_id, device_id, tp_rank)
    
    gpu_blocks = [torch.rand(size=gpu_shape[1:], dtype=torch.float16).cuda(device_id)
                    for layer_id in range(gpu_shape[0])]
    gpu_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERWISE,
        num_layer=num_layers,
        num_block=num_gpu_blocks,
        tokens_per_block=tokens_per_block,
        num_head=num_kv_heads,
        head_size=head_size,
        is_mla=False,
        kv_shape=gpu_shape,
    )
    
    tp_client.register_to_server(gpu_blocks, gpu_layout)
    
    while True:
        time.sleep(1)


def run_server(server_recv_port):
    kvserver = KVServer(model_config, cache_config, server_recv_port)
    kvserver.run()

def main():
    server_recv_port = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
    server_process = Process(
        target=run_server,
        args=(server_recv_port, ),
    )
    server_process.start()
    
    # wait for server start
    time.sleep(5)

    client_processes: list[Process] = []

    for dp_rank in range(dp_size):
        # Create client processes
        dp_client_precess = Process(
            target=run_dp_client,
            args=(server_recv_port, ),
        )
        dp_client_precess.start()
        
        client_processes.append(dp_client_precess)
        
    try:
        # Keep main process running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server_process.terminate()
        for dp_process in client_processes:
            dp_process.terminate()
            

if __name__ == "__main__":
    main()
