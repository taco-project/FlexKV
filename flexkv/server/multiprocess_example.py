import torch
from multiprocessing import Process
from flexkv.server.server import KVServer
from flexkv.server.client import KVClient
import time

def run_client(client_id: int, client_conn, gpu_device_id: int):
    """Client process function"""
    # Initialize client
    client = KVClient(client_conn, client_id)
    
    # Register device memory
    token_ids = torch.randn(100, 512).cuda(gpu_device_id)
    token_mask = torch.ones(100).cuda(gpu_device_id)
    gpu_physical_block_ids = torch.arange(100).cuda(gpu_device_id)
    
    # Register with server
    client.register_device_memory(token_ids, token_mask, gpu_physical_block_ids)
    
    # Example workload
    while True:
        # Send some requests
        request_id = client.get_async(token_ids[:10], token_mask[:10], gpu_physical_block_ids[:10])
        
        # Process responses
        results = client.process_responses()
        for req_id, mask in results:
            print(f"Client {client_id} got result for request {req_id}")
        
        time.sleep(0.01)  # Avoid busy waiting

def main():
    # Create and start server
    server = KVServer()
    server.start()
    
    # Create client processes
    client_processes = []
    for i in range(2):  # Create 2 clients
        # Get connection for new client
        client_id, client_conn = server.register_client()
        
        # Create and start client process
        process = Process(
            target=run_client,
            args=(client_id, client_conn, i)  # Assuming each client uses a different GPU
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