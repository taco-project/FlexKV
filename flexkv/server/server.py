from multiprocessing import Process, Queue, Pipe, Connection
from threading import Thread
from typing import Dict, List, Optional, Tuple
import torch
from dataclasses import dataclass
from ..kvmanager import KVManager
from ..common.memory_handle import CUDAIPCHandle
@dataclass
class ServerRequest:
    client_id: int
    request_id: int
    request_type: str  # 'put' or 'get' 'wait' or 'register'
    token_ids: Optional[torch.Tensor] = None
    token_mask: Optional[torch.Tensor] = None
    gpu_blocks: Optional[List[CUDAIPCHandle]] = None
    gpu_physical_block_ids: Optional[torch.Tensor] = None
    wait_request_ids: Optional[List[int]] = None

@dataclass
class ServerResponse:
    client_id: int
    request_id: int
    mask: Optional[List[torch.Tensor]] = None
    success: bool = True
    error_msg: str = ""

class KVServer(Process):
    def __init__(self):
        super().__init__()
        self.request_queue = Queue()
        self.client_pipes: Dict[int, Connection] = {}
        self.client_counter = 0
        self.kvmanager = KVManager()
        self.global_to_client_map: Dict[int, Tuple[int, int]] = {}
        self._running = True
        
    def register_client(self) -> Tuple[int, Connection]:
        """Register a new client and return client_id and connection"""
        server_conn, client_conn = Pipe()
        client_id = self.client_counter
        self.client_pipes[client_id] = server_conn
        self.client_counter += 1
        return client_id, client_conn

    def _poll_client_requests(self):
        """Thread function to poll client connections for requests"""
        while self._running:
            # Check each client connection for new requests
            for client_id, conn in self.client_pipes.items():
                if conn.poll():  # Check if there's data available
                    try:
                        request = conn.recv()
                        self.request_queue.put(request)
                    except EOFError:
                        # Handle client disconnect
                        print(f"Client {client_id} disconnected")
                        continue

    def _handle_register_request(self, request: Request):
        """Handle memory registration request"""
        client_id = request.client_id
        
        shared_tensors = []
        for handle in request.handles:
            tensor = import_cuda_tensor(handle)
            shared_tensors.append(tensor)
        return shared_tensors

    def run(self):
        """Main server loop"""
        # Start polling thread
        polling_thread = Thread(target=self._poll_client_requests)
        polling_thread.daemon = True  # Thread will exit when main process exits
        polling_thread.start()

        while self._running:
            # Process pending requests
            while not self.request_queue.empty():
                request: ServerRequest = self.request_queue.get()
                
                if request.request_type == 'register':
                    shared_tensors = self._handle_register_request(request)
                    self.kvmanager.register_device_memory(
                        request.client_id, #also device id
                        shared_tensors,
                    ) #this is allowed to be blocking now
                    response = ServerResponse(request.client_id, request.request_id)
                    self.client_pipes[request.client_id].send(response)
                    
                elif request.request_type == 'put':
                    self.kvmanager.put_async(
                        request.client_id, #also device id
                        request.request_id,
                        request.token_ids,
                        request.token_mask,
                        request.gpu_physical_block_ids
                    )
                    #response = ServerResponse(request.client_id, request.request_id)
                    
                elif request.request_type == 'get':
                    self.kvmanager.get_async(
                        request.client_id, #also device id
                        request.request_id,
                        request.token_ids,
                        request.token_mask,
                        request.gpu_physical_block_ids
                    )
                    #response = ServerResponse(request.client_id, request.request_id)
                elif request.request_type == 'wait':
                    #Note: to avoid blocking other clients, we use try_wait api here
                    masks = self.kvmanager.try_wait(
                        request.client_id, #also device id
                        request.request_id,
                        request.wait_request_ids
                    )
                    if masks is not None:
                        response = ServerResponse(request.client_id, request.request_id, masks)
                        self.client_pipes[request.client_id].send(response)
                    else:
                        self.request_queue.put(request)
                # Send response back to client

    def shutdown(self):
        """Shutdown the server"""
        self._running = False
        # Close all client connections
        for conn in self.client_pipes.values():
            conn.close()