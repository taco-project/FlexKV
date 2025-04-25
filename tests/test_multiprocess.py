import torch
from multiprocessing import Process
from flexkv.server.server import KVServer
from flexkv.server.client import KVClient
from typing import Tuple
import time

def run_client(client_id: int, client_conn, gpu_device_id: int,
               gpu_shape: Tuple[int, int, int, int, int, int]):
    """Client process function"""
    # Initialize client
    client = KVClient(client_conn, client_id)

    # Register device memory
    token_ids = torch.randn(64)
    token_mask = torch.ones(64)
    gpu_blocks = [torch.tensor(gpu_shape[1:]).cuda(gpu_device_id)
                    for i in range(gpu_shape[0])]
    gpu_physical_block_ids = torch.tensor([i for i in range(64//gpu_shape[3])]).pin_memory()

    # Register with server
    client.register_device_memory(gpu_blocks)

    # Example workload
    #while True:
        # Send some requests
    request_id1 = client.put_async(token_ids, token_mask, gpu_physical_block_ids)
    request_id2 = client.put_async(token_ids, token_mask, gpu_physical_block_ids)
    # Process responses
    results = client.wait([request_id1, request_id2])
    print(f"Client {client_id} got result for request {request_id1}")
    print(f"Client {client_id} got result for request {request_id2}")
    print(results[0])
    print(results[1])

    #    time.sleep(0.01)  # Avoid busy waiting

def main():
    # Create and start server
    num_layers = 32
    num_heads = 8
    head_dim = 128
    num_cpu_blocks = 300
    num_gpu_blocks = 30
    num_clients = 1
    token_per_block = 4
    cpu_shape = (num_layers, 2, num_cpu_blocks, token_per_block, num_heads, head_dim)
    gpu_shape = (num_layers, 2, num_gpu_blocks, token_per_block, num_heads, head_dim)

    dtype = torch.float16
    server = KVServer(cpu_shape = cpu_shape, dtype = dtype)
    server.start()

    # Create client processes
    client_processes = []
    for i in range(num_clients):  # Create 2 clients
        # Get connection for new client
        client_id, client_conn = server.register_client()

        # Create and start client process
        process = Process(
            target=run_client,
            args=(client_id, client_conn, i, gpu_shape)  # Assuming each client uses a different GPU
        )
        process.start()
        client_processes.append(process)

    try:
        # Keep main process running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Clean shutdown
        server.terminate()
        for process in client_processes:
            process.terminate()

if __name__ == "__main__":
    main()
