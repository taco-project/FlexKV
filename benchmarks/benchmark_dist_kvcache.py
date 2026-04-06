"""
Benchmark for FlexKV distributed KVCache in server_client_mode.

This script tests the put/get performance of FlexKV when running in
server_client_mode with distributed KVCache sharing enabled (enable_p2p_cpu).

Prerequisites:
  - A running Redis server (default: 127.0.0.1:6379)
  - At least 1 GPU available
  - FlexKV built with distributed support (FLEXKV_ENABLE_P2P=1)

Usage:
  # Basic usage with default config
  python benchmarks/benchmark_dist_kvcache.py --config benchmarks/example_dist_config.yml

  # Custom parameters
  python benchmarks/benchmark_dist_kvcache.py \
      --config benchmarks/example_dist_config.yml \
      --batch-size 4 \
      --sequence-length 2048 \
      --cache-ratio 0.5 \
      --num-users 10 \
      --num-turns 3

  # Multi-turn conversation benchmark only
  python benchmarks/benchmark_dist_kvcache.py \
      --config benchmarks/example_dist_config.yml \
      --mode multiturn \
      --num-users 20 \
      --num-turns 5
"""

import os
import argparse
import json
import tempfile
import time
from multiprocessing import Process
from dataclasses import dataclass

import torch
import numpy as np

from flexkv.server.client import KVTPClient
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.config import (
    ModelConfig, CacheConfig, UserConfig,
    update_default_config_from_user_config, parse_path_list,
    GLOBAL_CONFIG_FROM_ENV,
)
from flexkv.common.debug import flexkv_logger
from flexkv.kvmanager import KVManager
from flexkv.kvtask import KVResponseStatus

from utils import generate_random_multiturn

flexkv_logger.set_level("INFO")


def load_dist_config(config_path: str):
    """Load config with distributed KVCache support.

    Extends the standard load_config to handle distributed-specific fields:
      enable_p2p_cpu, enable_p2p_ssd, enable_3rd_remote,
      redis_host, redis_port, local_ip, redis_password,
      server_client_mode, etc.
    """
    import yaml

    with open(config_path) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)
    print(f"Loaded config: {config}")

    model_config = ModelConfig()
    cache_config = CacheConfig()
    user_config = UserConfig()

    # Model config
    model_config.num_layers = config["num_layers"]
    model_config.num_kv_heads = config["num_kv_heads"]
    model_config.head_size = config["head_size"]
    model_config.dtype = eval(f"torch.{config['dtype']}")
    model_config.use_mla = config["use_mla"]
    model_config.tp_size = config["tp_size"]
    model_config.dp_size = config["dp_size"]
    cache_config.tokens_per_block = config["tokens_per_block"]

    # Cache size config
    if "cpu_cache_gb" in config:
        user_config.cpu_cache_gb = config["cpu_cache_gb"]
    if "ssd_cache_gb" in config:
        user_config.ssd_cache_gb = config["ssd_cache_gb"]
    if "ssd_cache_dir" in config:
        user_config.ssd_cache_dir = parse_path_list(config["ssd_cache_dir"])
    if "enable_gds" in config:
        user_config.enable_gds = config["enable_gds"]

    # Distributed KVCache config
    if "enable_p2p_cpu" in config:
        user_config.enable_p2p_cpu = config["enable_p2p_cpu"]
    if "enable_p2p_ssd" in config:
        user_config.enable_p2p_ssd = config["enable_p2p_ssd"]
    if "enable_3rd_remote" in config:
        user_config.enable_3rd_remote = config["enable_3rd_remote"]

    # Redis config
    if "redis_host" in config:
        user_config.redis_host = config["redis_host"]
    if "redis_port" in config:
        user_config.redis_port = config["redis_port"]
    if "local_ip" in config:
        user_config.local_ip = config["local_ip"]
    if "redis_password" in config:
        user_config.redis_password = config["redis_password"]

    # Auto-generate mooncake config JSON and set MOONCAKE_CONFIG_PATH if P2P is enabled
    if config.get("enable_p2p_cpu", False) or config.get("enable_p2p_ssd", False):
        if "MOONCAKE_CONFIG_PATH" not in os.environ:
            mooncake_config = {
                "engine_ip": config.get("mooncake_engine_ip", config.get("local_ip", "127.0.0.1")),
                "engine_port": config.get("mooncake_engine_port", 5555),
                "metadata_backend": config.get("mooncake_metadata_backend", "redis"),
                "metadata_server": config.get("mooncake_metadata_server",
                    f"redis://{config.get('redis_host', '127.0.0.1')}:{config.get('redis_port', 6379)}"),
                "metadata_server_auth": config.get("mooncake_metadata_server_auth",
                    config.get("redis_password", "")),
                "protocol": config.get("mooncake_protocol", "tcp"),
                "device_name": config.get("mooncake_device_name", ""),
            }
            # Write to a temp file that persists until process exits
            mooncake_config_fd, mooncake_config_path = tempfile.mkstemp(
                suffix=".json", prefix="mooncake_config_"
            )
            with os.fdopen(mooncake_config_fd, "w") as f:
                json.dump(mooncake_config, f, indent=2)
            os.environ["MOONCAKE_CONFIG_PATH"] = mooncake_config_path
            print(f"[INFO] Auto-generated mooncake config at: {mooncake_config_path}")
            print(f"[INFO] Mooncake config: {json.dumps(mooncake_config, indent=2)}")
        else:
            mooncake_config_path = os.environ['MOONCAKE_CONFIG_PATH']
            print(f"[INFO] Using existing MOONCAKE_CONFIG_PATH: {mooncake_config_path}")

        # Store mooncake_config_path in cache_config so it survives spawn subprocesses via pickle
        cache_config.mooncake_config_path = mooncake_config_path

    update_default_config_from_user_config(model_config, cache_config, user_config)

    # Handle server_client_mode from config
    if config.get("server_client_mode", False):
        os.environ["FLEXKV_SERVER_CLIENT_MODE"] = "1"
        GLOBAL_CONFIG_FROM_ENV.server_client_mode = True

    return model_config, cache_config


@dataclass
class BenchmarkConfig:
    # Single batch benchmark params
    batch_size: int = 1
    sequence_length: int = 1024
    cache_ratio: float = 1.0
    clear_cpu_cache: bool = False

    # Multi-turn benchmark params
    num_users: int = 10
    num_turns: int = 3
    system_prompt_length: int = 100
    input_length: int = 512
    output_length: int = 64

    # General
    mode: str = "all"  # "single", "multiturn", or "all"


def run_tp_client(dp_client_id, tp_rank, gpu_register_port, model_config, cache_config, num_gpu_blocks):
    """Run tp_client process to register GPU blocks"""
    device_id = tp_rank + dp_client_id * model_config.tp_size
    tp_client = KVTPClient(gpu_register_port, dp_client_id, device_id)

    gpu_kv_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERFIRST,
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


def shutdown_tp_clients(tp_client_processes):
    """Terminate all tp_client processes"""
    for tp_process in tp_client_processes:
        if tp_process.is_alive():
            tp_process.terminate()
            tp_process.join(timeout=5)
            if tp_process.is_alive():
                print(f"Force killing tp_client process {tp_process.pid}")
                tp_process.kill()
                tp_process.join(timeout=2)


def benchmark_single_batch(kvmanager, model_config, cache_config, bench_config):
    """Benchmark single batch put/get with distributed KVCache"""
    print("\n" + "=" * 60)
    print("  Single Batch Benchmark (Distributed KVCache)")
    print("=" * 60)

    sequence_length = bench_config.sequence_length
    batch_size = bench_config.batch_size
    cache_length = int(sequence_length * bench_config.cache_ratio)

    print(f"  batch_size={batch_size}, sequence_length={sequence_length}, "
          f"cache_ratio={bench_config.cache_ratio}, cache_length={cache_length}")

    # Generate random sequences
    batch_sequence_tensor = []
    batch_slot_mapping = []
    for i in range(batch_size):
        batch_sequence_tensor.append(torch.randint(0, 100000, (sequence_length,), dtype=torch.int64))
        batch_slot_mapping.append(torch.arange(i * sequence_length, (i + 1) * sequence_length, dtype=torch.int64))

    # ---- Benchmark PUT ----
    print("\n--- PUT Phase ---")
    start_time = time.time()
    batch_put_ids = []
    if bench_config.cache_ratio > 0:
        for i in range(batch_size):
            task_id = kvmanager.put_async(
                batch_sequence_tensor[i][:cache_length],
                batch_slot_mapping[i][:cache_length],
                token_mask=None,
            )
            batch_put_ids.append(task_id)
    put_result = kvmanager.wait(batch_put_ids, completely=True)
    end_time = time.time()

    elapsed_time_put = end_time - start_time
    put_tokens = 0
    for _, response in put_result.items():
        if response.status == KVResponseStatus.SUCCESS:
            put_tokens += response.return_mask.sum().item()
    transfer_data_size_GB = put_tokens * model_config.token_size_in_bytes / (1024 ** 3)
    transfer_bandwidth_put = transfer_data_size_GB / elapsed_time_put if elapsed_time_put > 0 else 0
    print(f"  PUT: {put_tokens} tokens, data_size: {transfer_data_size_GB:.3f} GB, "
          f"time: {elapsed_time_put * 1000:.2f}ms, bandwidth: {transfer_bandwidth_put:.2f} GB/s")

    if bench_config.clear_cpu_cache:
        kvmanager._clear_cpu_cache()

    # ---- Benchmark GET ----
    print("\n--- GET Phase ---")
    all_tokens = 0
    start_time = time.time()
    batch_get_ids = []
    for i in range(batch_size):
        all_tokens += len(batch_sequence_tensor[i])
        task_id, _ = kvmanager.get_match(batch_sequence_tensor[i], token_mask=None)
        batch_get_ids.append(task_id)
    get_match_time = time.time() - start_time

    kvmanager.launch(batch_get_ids, batch_slot_mapping, as_batch=True, layerwise_transfer=False)
    get_result = kvmanager.wait(batch_get_ids)
    elapsed_time_get = time.time() - start_time

    cached_tokens = 0
    for _, response in get_result.items():
        if response.status == KVResponseStatus.SUCCESS:
            cached_tokens += response.return_mask.sum().item()
    transfer_data_size_GB = cached_tokens * model_config.token_size_in_bytes / (1024 ** 3)
    transfer_bandwidth_get = transfer_data_size_GB / elapsed_time_get if elapsed_time_get > 0 else 0
    print(f"  GET: {cached_tokens}/{all_tokens} tokens, data_size: {transfer_data_size_GB:.3f} GB, "
          f"cache_ratio: {cached_tokens * 100 / all_tokens:.2f}%, "
          f"match time: {get_match_time * 1000:.2f}ms, "
          f"e2e time: {elapsed_time_get * 1000:.2f}ms, "
          f"bandwidth: {transfer_bandwidth_get:.2f} GB/s")

    return {
        "put_tokens": put_tokens,
        "put_time_ms": elapsed_time_put * 1000,
        "put_bandwidth_GBs": transfer_bandwidth_put,
        "get_cached_tokens": cached_tokens,
        "get_total_tokens": all_tokens,
        "get_cache_ratio": cached_tokens / all_tokens if all_tokens > 0 else 0,
        "get_match_time_ms": get_match_time * 1000,
        "get_e2e_time_ms": elapsed_time_get * 1000,
        "get_bandwidth_GBs": transfer_bandwidth_get,
    }


def benchmark_multiturn(kvmanager, model_config, cache_config, bench_config):
    """Benchmark multi-turn conversation with distributed KVCache"""
    print("\n" + "=" * 60)
    print("  Multi-Turn Conversation Benchmark (Distributed KVCache)")
    print("=" * 60)
    print(f"  num_users={bench_config.num_users}, num_turns={bench_config.num_turns}, "
          f"system_prompt_length={bench_config.system_prompt_length}, "
          f"input_length={bench_config.input_length}, output_length={bench_config.output_length}")

    # Generate multi-turn requests
    reqs = generate_random_multiturn(
        num_user_requests=bench_config.num_users,
        num_turns=bench_config.num_turns,
        system_prompt_length=bench_config.system_prompt_length,
        input_length=bench_config.input_length,
        output_length=bench_config.output_length,
    )

    total_get_requests = 0
    total_put_requests = 0
    cache_hit_ratios = []
    total_put_time = 0
    total_get_time = 0
    total_put_tokens = 0
    total_get_cached_tokens = 0
    total_get_all_tokens = 0

    request_id = 0
    for req in reqs:
        fake_slot_mapping = torch.arange(req.token_mask.sum(), dtype=torch.int64)

        if req.request_type == "get":
            total_get_requests += 1
            total_get_all_tokens += req.token_mask.sum().item()

            start_time = time.time()
            task_id, _ = kvmanager.get_match(
                req.token_ids,
                token_mask=torch.ones_like(torch.from_numpy(req.token_ids) if isinstance(req.token_ids, np.ndarray) else req.token_ids),
            )
            kvmanager.launch([task_id], [fake_slot_mapping.numpy()])
            result = kvmanager.wait([task_id])
            elapsed = time.time() - start_time
            total_get_time += elapsed

            for _, response in result.items():
                if response.status == KVResponseStatus.SUCCESS and response.return_mask is not None:
                    cached = response.return_mask.sum().item()
                    total_get_cached_tokens += cached
                    ratio = cached / req.token_mask.sum().item()
                    cache_hit_ratios.append(ratio)
                else:
                    cache_hit_ratios.append(0.0)

        elif req.request_type == "put":
            total_put_requests += 1

            start_time = time.time()
            task_id = kvmanager.put_async(
                req.token_ids,
                fake_slot_mapping.numpy(),
                token_mask=None,
            )
            result = kvmanager.wait([task_id], completely=True)
            elapsed = time.time() - start_time
            total_put_time += elapsed

            for _, response in result.items():
                if response.status == KVResponseStatus.SUCCESS and response.return_mask is not None:
                    total_put_tokens += response.return_mask.sum().item()

        request_id += 1

    # Print results
    print(f"\n--- Results ---")
    print(f"  Total requests: {len(reqs)} (GET: {total_get_requests}, PUT: {total_put_requests})")
    print(f"  PUT: {total_put_tokens} tokens, total time: {total_put_time * 1000:.2f}ms, "
          f"avg time: {total_put_time * 1000 / max(total_put_requests, 1):.2f}ms/req")
    print(f"  GET: {total_get_cached_tokens}/{total_get_all_tokens} tokens cached, "
          f"total time: {total_get_time * 1000:.2f}ms, "
          f"avg time: {total_get_time * 1000 / max(total_get_requests, 1):.2f}ms/req")

    if cache_hit_ratios:
        sorted_ratios = sorted(cache_hit_ratios)
        avg_ratio = sum(sorted_ratios) / len(sorted_ratios)
        print(f"  Cache hit ratio: avg={avg_ratio * 100:.2f}%, "
              f"min={sorted_ratios[0] * 100:.2f}%, "
              f"median={sorted_ratios[len(sorted_ratios) // 2] * 100:.2f}%, "
              f"max={sorted_ratios[-1] * 100:.2f}%")

    return {
        "total_requests": len(reqs),
        "get_requests": total_get_requests,
        "put_requests": total_put_requests,
        "put_tokens": total_put_tokens,
        "put_total_time_ms": total_put_time * 1000,
        "get_cached_tokens": total_get_cached_tokens,
        "get_total_tokens": total_get_all_tokens,
        "get_total_time_ms": total_get_time * 1000,
        "avg_cache_hit_ratio": sum(cache_hit_ratios) / len(cache_hit_ratios) if cache_hit_ratios else 0,
    }


def main(args):
    # Load config
    model_config, cache_config = load_dist_config(args.config)

    bench_config = BenchmarkConfig(
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        cache_ratio=args.cache_ratio,
        clear_cpu_cache=args.clear_cpu_cache,
        num_users=args.num_users,
        num_turns=args.num_turns,
        system_prompt_length=args.system_prompt_length,
        input_length=args.input_length,
        output_length=args.output_length,
        mode=args.mode,
    )

    # Pad sequence length to be divisible by tokens_per_block
    bench_config.sequence_length = (
        ((bench_config.sequence_length - 1) // cache_config.tokens_per_block + 1)
        * cache_config.tokens_per_block
    )

    num_gpu_blocks = bench_config.sequence_length * bench_config.batch_size // cache_config.tokens_per_block
    # Ensure enough GPU blocks for multi-turn mode too
    if bench_config.mode in ("multiturn", "all"):
        max_tokens_per_user = (
            bench_config.system_prompt_length
            + bench_config.num_turns * (bench_config.input_length + bench_config.output_length)
        )
        multiturn_blocks = max_tokens_per_user * bench_config.num_users // cache_config.tokens_per_block
        num_gpu_blocks = max(num_gpu_blocks, multiturn_blocks)
    # Add some extra blocks for safety
    num_gpu_blocks = int(num_gpu_blocks * 1.5) + 64

    if model_config.tp_size * model_config.dp_size > torch.cuda.device_count():
        raise ValueError(
            f"tp_size {model_config.tp_size} * dp_size {model_config.dp_size} > "
            f"available GPUs {torch.cuda.device_count()}"
        )

    print("=" * 60)
    print("  FlexKV Distributed KVCache Benchmark (server_client_mode)")
    print("=" * 60)
    print(f"  model_config: {model_config}")
    print(f"  cache_config: {cache_config}")
    print(f"  enable_kv_sharing: {cache_config.enable_kv_sharing}")
    print(f"  enable_p2p_cpu: {cache_config.enable_p2p_cpu}")
    print(f"  redis: {cache_config.redis_host}:{cache_config.redis_port}")
    print(f"  num_gpu_blocks: {num_gpu_blocks}")
    print(f"  bench_config: {bench_config}")

    # Create KVManager (this will start KVServer in server_client_mode)
    kvmanager = KVManager(model_config, cache_config)
    kvmanager.start()

    # Start tp_client processes to register GPU blocks
    tp_client_processes = []
    for tp_rank in range(model_config.tp_size):
        tp_process = Process(
            target=run_tp_client,
            args=(0, tp_rank, kvmanager.gpu_register_port,
                  model_config, cache_config, num_gpu_blocks),
            daemon=True,
        )
        tp_process.start()
        tp_client_processes.append(tp_process)

    # Wait for system to be ready
    print("\nWaiting for FlexKV to be ready...")
    wait_start = time.time()
    while not kvmanager.is_ready():
        time.sleep(1)
        elapsed = time.time() - wait_start
        if elapsed > 120:
            print("ERROR: Timeout waiting for FlexKV to be ready (120s)")
            shutdown_tp_clients(tp_client_processes)
            kvmanager.shutdown()
            return
        if int(elapsed) % 10 == 0 and int(elapsed) > 0:
            print(f"  Still waiting... ({int(elapsed)}s)")
    print(f"FlexKV is ready! (took {time.time() - wait_start:.1f}s)")

    try:
        results = {}

        if bench_config.mode in ("single", "all"):
            results["single_batch"] = benchmark_single_batch(
                kvmanager, model_config, cache_config, bench_config
            )

        if bench_config.mode in ("multiturn", "all"):
            results["multiturn"] = benchmark_multiturn(
                kvmanager, model_config, cache_config, bench_config
            )

        # Print summary
        print("\n" + "=" * 60)
        print("  Benchmark Summary")
        print("=" * 60)
        for name, result in results.items():
            print(f"\n  [{name}]")
            for k, v in result.items():
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
                else:
                    print(f"    {k}: {v}")

    finally:
        print("\nShutting down...")
        shutdown_tp_clients(tp_client_processes)
        kvmanager.shutdown()
        print("Done.")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark FlexKV distributed KVCache in server_client_mode"
    )
    parser.add_argument("--config", type=str, default="benchmarks/example_dist_config.yml",
                        help="Path to config YAML file")
    parser.add_argument("--mode", type=str, default="all", choices=["single", "multiturn", "all"],
                        help="Benchmark mode: single batch, multi-turn, or both")

    # Single batch params
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for single batch benchmark")
    parser.add_argument("--sequence-length", type=int, default=1024, help="Sequence length per request")
    parser.add_argument("--cache-ratio", type=float, default=1.0, help="Ratio of tokens to cache in PUT phase")
    parser.add_argument("--clear-cpu-cache", action="store_true", help="Clear CPU cache between PUT and GET")

    # Multi-turn params
    parser.add_argument("--num-users", type=int, default=10, help="Number of simulated users")
    parser.add_argument("--num-turns", type=int, default=3, help="Number of conversation turns per user")
    parser.add_argument("--system-prompt-length", type=int, default=100, help="System prompt length in tokens")
    parser.add_argument("--input-length", type=int, default=512, help="Input length per turn in tokens")
    parser.add_argument("--output-length", type=int, default=64, help="Output length per turn in tokens")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
