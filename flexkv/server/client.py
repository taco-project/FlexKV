from multiprocessing import Queue, Connection
from queue import Queue as ThreadQueue
from typing import Dict, List, Optional, Tuple
import torch
from .server import ServerRequest, ServerResponse

class KVClient:
    def __init__(self, server_connection: Connection, client_id: int):
        self.conn = server_connection
        self.client_id = client_id
        self.request_counter = 0
        self.pending_requests = ThreadQueue()  # Store pending requests
        self.completed_results = ThreadQueue()  # Store completed results
        
    def register_device_memory(self, 
                             gpu_blocks: List[torch.Tensor]):
        """Register device memory with server"""
        if not gpu_blocks or not gpu_blocks[0].is_cuda:
            raise ValueError("GPU blocks must be CUDA tensors")
        
        device_id = gpu_blocks[0].device.index
        handles = []
        
        for tensor in gpu_blocks:
            if tensor.device.index != device_id:
                raise ValueError("All tensors must be on the same GPU")
            
            # 导出CUDA IPC handle
            handle = export_cuda_tensor(tensor)
            handles.append(handle)
        
        # 发送handles给server
        request = Request(
            client_id=self.client_id,
            request_id=self.request_counter,
            request_type='register',
            handles=handles
        )
        self.request_counter += 1
        self.conn.send(request)
        return request.request_id


    def put_async(self,
                 token_ids: torch.Tensor,
                 token_mask: Optional[torch.Tensor],
                 gpu_physical_block_ids: torch.Tensor) -> int:
        """Send put request to server"""
        request = ServerRequest(
            client_id=self.client_id,
            request_id=self.request_counter,
            request_type='put',
            token_ids=token_ids,
            token_mask=token_mask,
            gpu_physical_block_ids=gpu_physical_block_ids
        )
        self.request_counter += 1
        self.pending_requests.put(request.request_id)
        self.conn.send(request)
        return request.request_id

    def get_async(self,
                 token_ids: torch.Tensor,
                 token_mask: Optional[torch.Tensor],
                 gpu_physical_block_ids: torch.Tensor) -> int:
        """Send get request to server"""
        request = ServerRequest(
            client_id=self.client_id,
            request_id=self.request_counter,
            request_type='get',
            token_ids=token_ids,
            token_mask=token_mask,
            gpu_physical_block_ids=gpu_physical_block_ids
        )
        self.request_counter += 1
        self.pending_requests.put(request.request_id)
        self.conn.send(request)
        return request.request_id

    def _process_responses(self):
        """Process any available responses from server"""
        while self.conn.poll():  # Check if there are any messages
            response: ServerResponse = self.conn.recv()
            #results.append((response.request_id, response.mask))
            self.pending_requests.remove(response.request_id)
            self.completed_results.put(response)
        
    #this is blocking
    def wait(self, request_ids: List[int]) -> List[torch.Tensor]:
        request = ServerRequest(
            client_id=self.client_id,
            request_id=self.request_counter,
            request_type='wait',
            wait_request_ids=request_ids
        )
        self.request_counter += 1
        self.conn.send(request)
        found_response = []
        founded_request_num = 0
        while founded_request_num < len(request_ids):
            self._process_responses()
            while not self.completed_results.empty():
                response = self.completed_results.get()
                if response.request_id in request_ids:
                    found_response.append(response.mask)
                    founded_request_num += 1
                else:
                    self.completed_results.put(response)
            sleep(0.001)
            
        return found_response