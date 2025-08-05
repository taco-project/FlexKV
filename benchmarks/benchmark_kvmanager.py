import os
import tempfile
from multiprocessing import Process
import argparse
import json
import time
from dataclasses import dataclass

import torch

from flexkv.server.client import KVDPClient, KVTPClient
from flexkv.server.server import KVServer, SchedulerServer
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

def shutdown_tp_client(tp_client_processes):
    for tp_process in tp_client_processes:
        if tp_process.is_alive():
            tp_process.terminate()
            tp_process.join(timeout=5)
            if tp_process.is_alive():
                print(f"Force killing tp_client process {tp_process.pid}")
                tp_process.kill()
                tp_process.join(timeout=2)

class FlexkvWrapper:
    def __init__(self, model_config, cache_config, server_recv_port):
        self.model_config = model_config
        self.cache_config = cache_config
        self.server_recv_port = server_recv_port

        self.use_scheduler_server = model_config.dp_size == 1
        if self.use_scheduler_server:
            self.launch_scheduler_server()
        else:
            self.launch_server()

    def launch_server(self):
        def server_process():
            kvserver = KVServer(self.model_config, self.cache_config, self.server_recv_port)
            kvserver.run()
            time.sleep(10)
        self.server_process = Process(
            target=server_process,
            daemon=False
        )
        self.server_process.start()
        time.sleep(5)
        self.dp_client = KVDPClient(self.server_recv_port, self.model_config)

    def launch_scheduler_server(self):
        self.scheduler_server = SchedulerServer(self.model_config, self.cache_config, self.server_recv_port)
        self.scheduler_server.start_server_thread()
        time.sleep(10)

    @property
    def dp_client_id(self):
        if self.use_scheduler_server:
            return 0
        else:
            return self.dp_client.dp_client_id

    def put_async(self, token_ids, slot_mapping, token_mask=None):
        if self.use_scheduler_server:
            return self.scheduler_server.put_async(token_ids, slot_mapping, token_mask)
        else:
            return self.dp_client.put_async(token_ids, slot_mapping, token_mask)

    def get_async(self, token_ids, slot_mapping, token_mask=None):
        if self.use_scheduler_server:
            return self.scheduler_server.get_async(token_ids, slot_mapping, token_mask)
        else:
            return self.dp_client.get_async(token_ids, slot_mapping, token_mask)

    def wait(self, request_ids):
        if self.use_scheduler_server:
            return self.scheduler_server.wait(request_ids)
        else:
            return self.dp_client.wait(request_ids)

    def try_wait(self, request_ids):
        if self.use_scheduler_server:
            return self.scheduler_server.try_wait(request_ids)
        else:
            return self.dp_client.try_wait(request_ids)

    def check_running(self):
        if self.use_scheduler_server:
            return self.scheduler_server.check_running()
        else:
            return self.dp_client.check_running()

    def shutdown(self):
        if not self.use_scheduler_server:
            try:
                # Send a shutdown request to the server
                self.dp_client.shutdown()
                # Wait a bit for graceful shutdown
                time.sleep(3)
            except Exception as e:
                print(f"Error sending shutdown request: {e}")
            if self.server_process.is_alive():
                self.server_process.terminate()
                self.server_process.join(timeout=10)
                if self.server_process.is_alive():
                    print(f"Force killing server process {self.server_process.pid}")
                    self.server_process.kill()
                    self.server_process.join(timeout=5)
            if self.server_recv_port.startswith('ipc://'):
                temp_file = self.server_recv_port[6:]  # Remove 'ipc://' prefix
                try:
                    if os.path.exists(temp_file):
                        os.unlink(temp_file)
                except Exception as e:
                    print(f"Error cleaning up temporary file: {e}")
        else:
            self.scheduler_server.shutdown()

def benchmark_kvmanager(model_config, cache_config, benchmark_config, server_recv_port):
    if model_config.tp_size * model_config.dp_size > torch.cuda.device_count():
        raise ValueError(f"tp_size {model_config.tp_size} * dp_size {model_config.dp_size} is greater than "
                         f"the number of available GPUs {torch.cuda.device_count()}")
    print(f"{model_config = }")
    print(f"{cache_config = }")
    print(f"{benchmark_config = }")
    flexkv_wrapper = FlexkvWrapper(model_config, cache_config, server_recv_port)

    tp_client_processes = []

    sequence_length = benchmark_config.sequence_length
    batch_size = benchmark_config.batch_size
    num_required_gpu_blocks = sequence_length * batch_size // cache_config.tokens_per_block
    cache_config.num_gpu_blocks = num_required_gpu_blocks
    print(f"allocate {num_required_gpu_blocks} gpu blocks for benchmark")
    for tp_rank in range(model_config.tp_size):
        tp_client_process = Process(
            target=run_tp_client,
            args=(flexkv_wrapper.dp_client_id, tp_rank, server_recv_port,
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

    while not flexkv_wrapper.check_running():
        time.sleep(0.1)
        print("waiting for flexkv wrapper to be ready")
    # benchmark put
    start_time = time.time()
    put_ids = []
    if benchmark_config.cache_ratio > 0:
        for i in range(batch_size):
            put_ids.append(flexkv_wrapper.put_async(batch_sequence_tensor[i][:cache_length],
                                            batch_slot_mapping[i][:cache_length],
                                            token_mask=None))
    put_result = flexkv_wrapper.wait(put_ids)
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
        get_ids.append(flexkv_wrapper.get_async(batch_sequence_tensor[i],
                                           batch_slot_mapping[i],
                                           token_mask=None))
    get_result = flexkv_wrapper.wait(get_ids)
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

    shutdown_tp_client(tp_client_processes)
    flexkv_wrapper.shutdown()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="benchmarks/example_config.json")
    # benchmark config
    parser.add_argument("--num-layers", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--cache-ratio", type=float, default=1)
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
    #cache_config.num_cpu_blocks = 8192 - 2048
    # pad sequence length to divisible by tokens_per_block
    benchmark_config.sequence_length = \
        ((benchmark_config.sequence_length - 1) // cache_config.tokens_per_block + 1) * cache_config.tokens_per_block
    server_recv_port = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
    benchmark_kvmanager(model_config, cache_config, benchmark_config, server_recv_port)
