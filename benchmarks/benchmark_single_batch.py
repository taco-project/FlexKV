import tempfile
from multiprocessing import Process
import argparse
import time
from dataclasses import dataclass

import torch

from flexkv.server.client import KVTPClient
from flexkv.common.storage import KVCacheLayout
from flexkv.common.debug import flexkv_logger
from flexkv.common.config import ModelConfig, CacheConfig
from utils import load_config
from flexkv.kvmanager import KVManager
from flexkv.kvtask import KVResponseStatus

flexkv_logger.set_level("INFO")


@dataclass
class BenchmarkConfig:
    num_layers_to_transfer: int
    batch_size: int
    sequence_length: int
    cache_ratio: float
    clear_cpu_cache: bool

def run_tp_client(dp_client_id, tp_rank, server_recv_port, model_config, cache_config):
    """Run tp_client process"""
    device_id = tp_rank + dp_client_id * model_config.tp_size
    tp_client = KVTPClient(server_recv_port, dp_client_id, device_id)

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

def shutdown_tp_client(tp_client_processes):
    for tp_process in tp_client_processes:
        if tp_process.is_alive():
            tp_process.terminate()
            tp_process.join(timeout=5)
            if tp_process.is_alive():
                print(f"Force killing tp_client process {tp_process.pid}")
                tp_process.kill()
                tp_process.join(timeout=2)

def benchmark_flexkv(model_config: ModelConfig,
                     cache_config: CacheConfig,
                     benchmark_config: BenchmarkConfig,
                     gpu_register_port: str,
                     server_recv_port: str):
    if model_config.tp_size * model_config.dp_size > torch.cuda.device_count():
        raise ValueError(f"tp_size {model_config.tp_size} * dp_size {model_config.dp_size} is greater than "
                         f"the number of available GPUs {torch.cuda.device_count()}")
    print(f"{benchmark_config = }")
    kvmanager = KVManager(model_config, cache_config, gpu_register_port, server_recv_port)
    kvmanager.start()

    tp_client_processes = []

    sequence_length = benchmark_config.sequence_length
    batch_size = benchmark_config.batch_size
    num_required_gpu_blocks = sequence_length * batch_size // cache_config.tokens_per_block
    cache_config.num_gpu_blocks = num_required_gpu_blocks
    print(f"allocate {num_required_gpu_blocks} gpu blocks for benchmark")
    for tp_rank in range(model_config.tp_size):
        tp_client_process = Process(
            target=run_tp_client,
            args=(0, tp_rank, gpu_register_port,
                    model_config, cache_config),
            daemon=True
        )
        tp_client_process.start()
        tp_client_processes.append(tp_client_process)

    while not kvmanager.is_ready():
        time.sleep(3)
        flexkv_logger.info("waiting for flexkv to be ready")
    flexkv_logger.info("flexkv is ready")

    batch_sequence_tensor = []
    batch_slot_mapping = []
    cache_length = int(sequence_length * benchmark_config.cache_ratio)

    # generate requests
    for i in range(batch_size):
        batch_sequence_tensor.append(torch.randint(0, 100000, (sequence_length, ), dtype=torch.int64))
        batch_slot_mapping.append(torch.arange(i * sequence_length, (i+1) * sequence_length, dtype=torch.int64))

    # benchmark put
    start_time = time.time()
    batch_put_ids = []
    if benchmark_config.cache_ratio > 0:
        for i in range(batch_size):
            task_id = kvmanager.put_async(batch_sequence_tensor[i][:cache_length],
                                          batch_slot_mapping[i][:cache_length],
                                          token_mask=None)
            batch_put_ids.append(task_id)
    put_result = kvmanager.wait(batch_put_ids, completely=True)
    end_time = time.time()

    if benchmark_config.clear_cpu_cache:
        kvmanager._clear_cpu_cache()

    elapsed_time_put = end_time - start_time
    put_tokens = 0
    for _, response in put_result.items():
        if response.status == KVResponseStatus.SUCCESS:
            put_tokens += response.return_mask.sum().item()
    transfer_data_size_GB = put_tokens * model_config.token_size_in_bytes / 1024 / 1024 / 1024
    transfer_bandwidth_put = transfer_data_size_GB / elapsed_time_put
    print(f"put {put_tokens} tokens, data_size: {transfer_data_size_GB:.3f} GB, "
          f"time: {elapsed_time_put*1000:.2f}ms, bandwidth: {transfer_bandwidth_put:.2f} GB/s")

    all_tokens = 0
    start_time = time.time()
    batch_get_ids = []
    for i in range(batch_size):
        all_tokens += len(batch_sequence_tensor[i])
        task_id, _ = kvmanager.get_match(batch_sequence_tensor[i],
                                      token_mask=None)
        batch_get_ids.append(task_id)
    get_match_time = time.time() - start_time
    kvmanager.launch(batch_get_ids, batch_slot_mapping)
    get_result = kvmanager.wait(batch_get_ids)
    elapsed_time_get = time.time() - start_time
    cached_tokens = 0
    for _, response in get_result.items():
        if response.status == KVResponseStatus.SUCCESS:
            cached_tokens += response.return_mask.sum().item()
    transfer_data_size_GB = cached_tokens * model_config.token_size_in_bytes / 1024 / 1024 / 1024
    transfer_bandwidth_get = transfer_data_size_GB / elapsed_time_get
    print(f"get {cached_tokens} tokens, data_size: {transfer_data_size_GB:.3f} GB, "
          f"cache_ratio: {cached_tokens * 100 / all_tokens:.2f}%, "
          f"match time: {get_match_time*1000:.2f}ms, "
          f"e2e time: {elapsed_time_get*1000:.2f}ms, "
          f"bandwidth: {transfer_bandwidth_get:.2f} GB/s")

    shutdown_tp_client(tp_client_processes)
    kvmanager.shutdown()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="benchmarks/example_config.json")
    # benchmark config
    parser.add_argument("--num-layers", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--cache-ratio", type=float, default=1)
    parser.add_argument("--clear-cpu-cache", action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    benchmark_config = BenchmarkConfig(
        num_layers_to_transfer=args.num_layers,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        cache_ratio=args.cache_ratio,
        clear_cpu_cache=args.clear_cpu_cache
    )
    model_config, cache_config = load_config(args.config)
    #cache_config.num_cpu_blocks = 8192 - 2048
    # pad sequence length to divisible by tokens_per_block
    benchmark_config.sequence_length = \
        ((benchmark_config.sequence_length - 1) // cache_config.tokens_per_block + 1) * cache_config.tokens_per_block
    import uuid
    gpu_register_port = f"ipc:///tmp/flexkv_gpu_{uuid.uuid4().hex[:8]}"
    server_recv_port = f"ipc:///tmp/flexkv_srv_{uuid.uuid4().hex[:8]}"

    benchmark_flexkv(model_config, cache_config, benchmark_config, gpu_register_port, server_recv_port)
