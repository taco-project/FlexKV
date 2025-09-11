#!/usr/bin/env python3
"""
SchedulerServer Usage Example

Demonstrates how to use the new SchedulerServer to replace the original KVServer + KVDPClient mode
"""

import torch
import time
from multiprocessing import Process
from flexkv.common.config import ModelConfig, CacheConfig
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.server.scheduler_server import SchedulerServer


def run_tp_client_process(dp_client_id, tp_rank, device_id, server_recv_port, model_config, gpu_kv_layout):
    """Run TP client process"""
    from flexkv.server.client import KVTPClient

    print(f"Starting TP client: dp_client_id={dp_client_id}, tp_rank={tp_rank}, device_id={device_id}")

    try:
        # Set CUDA device for this process
        if torch.cuda.is_available():
            torch.cuda.set_device(device_id)
            # Initialize CUDA context
            torch.cuda.init()
            # Clear cache
            torch.cuda.empty_cache()

        tp_client = KVTPClient(server_recv_port, dp_client_id, device_id)

        # Create GPU blocks for this TP client
        gpu_blocks = []
        for layer_id in range(model_config.num_layers):
            kv_dim = 2 if not model_config.use_mla else 1
            kv_tensor = torch.zeros(
                size=(kv_dim, gpu_kv_layout.num_block, gpu_kv_layout.tokens_per_block,
                      model_config.num_kv_heads // model_config.tp_size,
                      model_config.head_size),
                dtype=model_config.dtype,
                device=f"cuda:{device_id}"
            )
            gpu_blocks.append(kv_tensor)

        print(f"TP client {tp_rank} registering to server...")
        # Register to server
        tp_client.register_to_server(gpu_blocks, gpu_kv_layout)
        print(f"TP client {tp_rank} registered to server")

        # Keep TP client running
        while True:
            time.sleep(1)

    except Exception as e:
        print(f"TP client {tp_rank} error: {e}")
        import traceback
        traceback.print_exc()
        raise


def main():
    # Configuration parameters
    num_layers = 32
    num_kv_heads = 8
    head_size = 128
    num_cpu_blocks = 300
    num_gpu_blocks = 30
    tp_size = 1
    tokens_per_block = 4

    # Create model and cache configuration
    model_config = ModelConfig(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        use_mla=False,
        tp_size=tp_size,
        dtype=torch.float16
    )

    cache_config = CacheConfig(
        enable_cpu=True,
        enable_ssd=False,
        enable_remote=False,
        use_gds=False,
        tokens_per_block=tokens_per_block,
        num_cpu_blocks=num_cpu_blocks,
    )

    # Create GPU KV layout
    gpu_kv_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERWISE,
        num_layer=num_layers,
        num_block=num_gpu_blocks,
        tokens_per_block=tokens_per_block,
        num_head=num_kv_heads // tp_size,
        head_size=head_size,
        is_mla=False
    )

    # Create SchedulerServer (integrates server and dpclient functionality)
    scheduler_server = SchedulerServer(
        model_config=model_config,
        cache_config=cache_config,
        server_recv_port="ipc:///tmp/scheduler_server_example"  # TPClient connects to this port
    )

    # Start background server thread to handle TPClient registration
    scheduler_server.start_server_thread()

    print("SchedulerServer started!")
    print(f"TPClient can connect to: {scheduler_server.get_server_port()}")
    print("Starting TP client processes...")

    # Start TP client processes
    tp_client_processes = []
    for tp_rank in range(tp_size):
        device_id = tp_rank  # Use TP rank as device ID
        # Check available GPUs
        available_gpus = torch.cuda.device_count()
        if device_id >= available_gpus:
            device_id = device_id % available_gpus
            print(f"Warning: Using GPU {device_id} for TP rank {tp_rank} (not enough GPUs)")

        tp_client_process = Process(
            target=run_tp_client_process,
            args=(0, tp_rank, device_id, scheduler_server.get_server_port(), model_config, gpu_kv_layout),
            daemon=True
        )
        tp_client_process.start()
        tp_client_processes.append(tp_client_process)
        print(f"Started TP client process for rank {tp_rank} on device {device_id}")

    print("Waiting for all TP clients to register...")

    time.sleep(5)

    # Now we can directly use scheduler_server without network communication
    # Example: Create some test data (following benchmark_kvmanager.py pattern)
    batch_size = 4
    seq_len = 128

    print("\n=== Generating test data ===")
    # Generate separate sequences for each request (correct approach)
    batch_token_ids = []
    batch_slot_mappings = []
    batch_token_masks = []

    for i in range(batch_size):
        # Each sequence is independent (seq_len,) shape
        token_ids = torch.randint(0, 1000, (seq_len,))
        slot_mapping = torch.arange(i * seq_len, (i + 1) * seq_len)
        token_mask = torch.ones(seq_len, dtype=torch.bool)

        batch_token_ids.append(token_ids)
        batch_slot_mappings.append(slot_mapping)
        batch_token_masks.append(token_mask)

    print(f"Generated {batch_size} sequences, each with {seq_len} tokens")

    print("\n=== Executing PUT Operations ===")
    # PUT operations - each sequence processed separately
    start_time = time.time()
    put_task_ids = []
    for i in range(batch_size):
        task_id = scheduler_server.put_async(
            token_ids=batch_token_ids[i],
            slot_mapping=batch_slot_mappings[i],
            token_mask=batch_token_masks[i]
        )
        if task_id:
            put_task_ids.append(task_id)
            print(f"PUT task {task_id} created for sequence {i}")

    put_time = (time.time() - start_time) * 1000
    print(f"Created {len(put_task_ids)} PUT tasks, time: {put_time:.2f}ms")
    time.sleep(2)
    print("\n=== Executing GET Operations ===")
    # GET operations - each sequence processed separately
    start_time = time.time()
    get_task_ids = []
    for i in range(batch_size):
        task_id = scheduler_server.get_async(
            token_ids=batch_token_ids[i],
            slot_mapping=batch_slot_mappings[i],
            token_mask=batch_token_masks[i]
        )
        if task_id:
            get_task_ids.append(task_id)
            print(f"GET task {task_id} created for sequence {i}")

    get_time = (time.time() - start_time) * 1000
    print(f"Created {len(get_task_ids)} GET tasks, time: {get_time:.2f}ms")

    print("\n=== Waiting for All Tasks to Complete ===")
    # Wait for all tasks to complete - can wait for multiple tasks at once
    all_task_ids = put_task_ids + get_task_ids
    if all_task_ids:
        start_time = time.time()
        masks = scheduler_server.wait(all_task_ids)
        wait_time = (time.time() - start_time) * 1000
        print(f"All {len(all_task_ids)} tasks completed, time: {wait_time:.2f}ms")

        # Analyze results
        if masks:
            total_tokens = 0
            for task_id, mask in masks.items():
                if mask is not None:
                    tokens = mask.sum().item() if hasattr(mask, 'sum') else len(mask)
                    total_tokens += tokens
                    print(f"Task {task_id}: {tokens} tokens processed")

    print("\n=== Trying Non-blocking Wait ===")
    # Create a few more tasks and try non-blocking wait
    extra_task_ids = []
    for i in range(2):
        task_id = scheduler_server.put_async(
            token_ids=batch_token_ids[i][:5],  # Use first 5 tokens
            slot_mapping=batch_slot_mappings[i][:5],
            token_mask=batch_token_masks[i][:5]
        )
        if task_id:
            extra_task_ids.append(task_id)

    if extra_task_ids:
        # Immediately try to wait (might not be completed yet)
        masks = scheduler_server.try_wait(extra_task_ids)
        if masks:
            print(f"Tasks {extra_task_ids} completed immediately")
        else:
            print(f"Tasks {extra_task_ids} not ready yet, will wait...")
            masks = scheduler_server.wait(extra_task_ids)
            print(f"Tasks {extra_task_ids} completed after wait")

    print("\n✅ All operations completed successfully!")


    # Clean up resources
    print("\n=== Shutting down SchedulerServer ===")
    scheduler_server.shutdown()
    print("SchedulerServer has been shut down")

    # Terminate TP client processes
    print("Terminating TP client processes...")
    for i, process in enumerate(tp_client_processes):
        process.terminate()
        process.join(timeout=2)
        if process.is_alive():
            process.kill()
        print(f"TP client process {i} terminated")


if __name__ == "__main__":
    main()
