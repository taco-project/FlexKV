import tempfile
from multiprocessing import Process
import argparse
import json
import time
from dataclasses import dataclass

import torch

from flexkv.server.client import KVDPClient, KVTPClient
from flexkv.server.server import KVServer
from flexkv.common.config import ModelConfig, CacheConfig
from flexkv.common.storage import KVCacheLayoutType, KVCacheLayout
from flexkv.common.debug import flexkv_logger
from utils import load_config

flexkv_logger.set_level("INFO")


@dataclass
class BenchmarkConfig:
    num_layers_to_transfer: int
    batch_size: int
    sequence_length: int
    cache_ratio: float

def run_server(model_config, cache_config, server_recv_port):
    """Run server process"""
    kvserver = KVServer(model_config, cache_config, server_recv_port)
    kvserver.run()
    time.sleep(10)

def run_tp_client(dp_client_id, tp_rank, server_recv_port, model_config, cache_config):
    """Run tp_client process"""
    device_id = tp_rank + dp_client_id * model_config.tp_size
    tp_client = KVTPClient(server_recv_port, dp_client_id, device_id, tp_rank)

    num_gpu_blocks = cache_config.num_gpu_blocks

    gpu_kv_layout = KVCacheLayout(
        type=cache_config.gpu_kv_layout_type,
        num_layer=model_config.num_layers,
        num_block=num_gpu_blocks,
        tokens_per_block=cache_config.tokens_per_block,
        num_head=model_config.num_kv_heads,
        head_size=model_config.head_size,
        is_mla=model_config.use_mla,
    )

    # Create GPU blocks for this tp_rank in the tp_client process
    gpu_blocks_for_tp = []
    for _ in range(model_config.num_layers):
        gpu_blocks_for_tp.append(
            torch.empty(size=tuple(gpu_kv_layout.kv_shape[1:]), dtype=model_config.dtype).cuda(device_id)
        )
    tp_client.register_to_server(gpu_blocks_for_tp, gpu_kv_layout)
    # Keep the process running
    while True:
        time.sleep(1)

def shutdown(server_process, dp_client, tp_client_processes, server_recv_port):
    """Shutdown all processes"""
    try:
        # Send a shutdown request to the server
        from flexkv.server.request import ShutdownRequest
        shutdown_request = ShutdownRequest(dp_client_id=dp_client.dp_client_id)
        dp_client.send_to_server.send_pyobj(shutdown_request)

        # Wait a bit for graceful shutdown
        time.sleep(3)
    except Exception as e:
        print(f"Error sending shutdown request: {e}")

    # Terminate tp_client processes
    for tp_process in tp_client_processes:
        if tp_process.is_alive():
            tp_process.terminate()
            tp_process.join(timeout=5)
            if tp_process.is_alive():
                print(f"Force killing tp_client process {tp_process.pid}")
                tp_process.kill()
                tp_process.join(timeout=2)

    # Terminate server process
    if server_process.is_alive():
        server_process.terminate()
        server_process.join(timeout=10)
        if server_process.is_alive():
            print(f"Force killing server process {server_process.pid}")
            server_process.kill()
            server_process.join(timeout=5)

    # Clean up temporary file
    import os
    if server_recv_port.startswith('ipc://'):
        temp_file = server_recv_port[6:]  # Remove 'ipc://' prefix
        try:
            if os.path.exists(temp_file):
                os.unlink(temp_file)
        except Exception as e:
            print(f"Error cleaning up temporary file: {e}")

def benchmark_kvmanager(model_config, cache_config, benchmark_config, server_recv_port):
    if model_config.tp_size * model_config.dp_size > torch.cuda.device_count():
        raise ValueError(f"tp_size {model_config.tp_size} * dp_size {model_config.dp_size} is greater than "
                         f"the number of available GPUs {torch.cuda.device_count()}")
    print(f"{model_config = }")
    print(f"{cache_config = }")
    print(f"{benchmark_config = }")
    server_process = Process(
        target=run_server,
        args=(model_config, cache_config, server_recv_port),
        daemon=False
    )
    server_process.start()
    time.sleep(5)
    dp_client = KVDPClient(server_recv_port, model_config)
    tp_client_processes = []

    sequence_length = benchmark_config.sequence_length
    batch_size = benchmark_config.batch_size
    num_required_gpu_blocks = sequence_length * batch_size // cache_config.tokens_per_block
    cache_config.num_gpu_blocks = num_required_gpu_blocks
    print(f"allocate {num_required_gpu_blocks} gpu blocks for benchmark")
    for tp_rank in range(model_config.tp_size):
        tp_client_process = Process(
            target=run_tp_client,
            args=(dp_client.dp_client_id, tp_rank, server_recv_port,
                    model_config, cache_config),
            daemon=True
        )
        tp_client_process.start()
        tp_client_processes.append(tp_client_process)
    time.sleep(5)

    batch_sequence_tensor = []
    batch_slot_mapping = []
    cache_length = int(sequence_length * benchmark_config.cache_ratio)

    # generate requests
    for i in range(batch_size):
        batch_sequence_tensor.append(torch.randint(0, 100000, (sequence_length, ), dtype=torch.int64))
        batch_slot_mapping.append(torch.arange(i * sequence_length, (i+1) * sequence_length, dtype=torch.int64))

    # benchmark put
    start_time = time.time()
    put_ids = []
    if benchmark_config.cache_ratio > 0:
        for i in range(batch_size):
            put_ids.append(dp_client.put_async(batch_sequence_tensor[i][:cache_length],
                                            batch_slot_mapping[i][:cache_length],
                                            token_mask=None))
    put_result = dp_client.wait(put_ids)
    end_time = time.time()
    time.sleep(1)
    elapsed_time_put = end_time - start_time
    put_tokens = 0
    for _, return_mask in put_result.items():
        put_tokens += return_mask.sum().item()
    transfer_data_size_GB = put_tokens * model_config.token_size_in_bytes / 1024 / 1024 / 1024
    transfer_bandwidth_put = transfer_data_size_GB / elapsed_time_put
    print(f"put {put_tokens} tokens, data_size: {transfer_data_size_GB:.3f} GB, "
          f"time: {elapsed_time_put*1000:.2f}ms, bandwidth: {transfer_bandwidth_put:.2f} GB/s")

    #benchmark get
    start_time = time.time()
    get_ids = []
    for i in range(batch_size):
        get_ids.append(dp_client.get_async(batch_sequence_tensor[i],
                                           batch_slot_mapping[i],
                                           token_mask=None))
    get_result = dp_client.wait(get_ids)
    end_time = time.time()
    elapsed_time_get = end_time - start_time
    cached_tokens = 0
    all_tokens = 0
    for _, return_mask in get_result.items():
        cached_tokens += return_mask.sum().item()
        all_tokens += len(return_mask)
    transfer_data_size_GB = cached_tokens * model_config.token_size_in_bytes / 1024 / 1024 / 1024
    transfer_bandwidth_get = transfer_data_size_GB / elapsed_time_get
    print(f"get {cached_tokens} tokens, data_size: {transfer_data_size_GB:.3f} GB, "
          f"cache_ratio: {cached_tokens * 100 / all_tokens:.2f}%, "
          f"time: {elapsed_time_get*1000:.2f}ms, bandwidth: {transfer_bandwidth_get:.2f} GB/s")

    shutdown(server_process, dp_client, tp_client_processes, server_recv_port)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="benchmarks/example_config.json")
    # benchmark config
    parser.add_argument("--num_layers", type=int, default=-1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--sequence_length", type=int, default=512)
    parser.add_argument("--cache_ratio", type=float, default=1)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    benchmark_config = BenchmarkConfig(
        num_layers_to_transfer=args.num_layers,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        cache_ratio=args.cache_ratio
    )
    model_config, cache_config = load_config(args.config)
    # pad sequence length to divisible by tokens_per_block
    benchmark_config.sequence_length = \
        ((benchmark_config.sequence_length - 1) // cache_config.tokens_per_block + 1) * cache_config.tokens_per_block
    server_recv_port = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
    benchmark_kvmanager(model_config, cache_config, benchmark_config, server_recv_port)
