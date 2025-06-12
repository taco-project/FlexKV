import time
from multiprocessing import Queue
from multiprocessing.connection import Connection
from queue import Queue as ThreadQueue
from typing import Dict, List, Optional, Tuple

import tempfile
import torch
import zmq

from flexkv.common.config import ModelConfig
from flexkv.common.debug import init_logger
from flexkv.common.memory_handle import export_layer_tensor_handle
from flexkv.common.storage import KVCacheLayout
from flexkv.server.util import get_zmq_socket
from flexkv.server.request import (
    RegisterDPClientRequest,
    RegisterTPClientRequest,
    PutRequest,
    GetRequest,
    WaitRequest,
    TryWaitRequest,
    Response
)


logger = init_logger(__name__)


class KVDPClient:
    def __init__(
        self,
        server_recv_port: str,
        model_config: ModelConfig,
    ):
        # Init inter-process communication
        context = zmq.Context(2)
        self.send_to_server = get_zmq_socket(
            context, zmq.PUSH, server_recv_port, False
        )
        client_recv_port = f"ipc://{tempfile.NamedTemporaryFile(delete=True).name}"
        self.recv_from_server = get_zmq_socket(
            context, zmq.PULL, client_recv_port, True
        )
        self.dp_client_id = self.register_to_server(model_config, client_recv_port)
        
        logger.info(f"KVDPClient Initialized! [DP Client ID]: {self.dp_client_id}")
    
    def register_to_server(
        self,
        model_config: ModelConfig,
        client_recv_port: str,
    ) -> int:
        register_req = RegisterDPClientRequest(model_config, client_recv_port)
        
        self.send_to_server.send_pyobj(register_req)
        # blocking
        response: Response = self.recv_from_server.recv_pyobj()
        if response.success:
            logger.info(f"DP client registered successfully! DP client id: {response.dp_client_id}")
            return response.dp_client_id
        else:
            logger.error(f"DP client registeration fialed: {response.error_msg}")
            raise
    
    def put_async(
        self,
        token_ids: torch.Tensor,
        slot_mapping: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> int:
        req = PutRequest(self.dp_client_id, token_ids, slot_mapping, token_mask)
        
        self.send_to_server.send_pyobj(req)
        response: Response = self.recv_from_server.recv_pyobj()
        
        if response.success:
            logger.info(f"put_async task: {response.task_id} created.")
            return response.task_id
        else:
            logger.error(f"put_async task in DP {self.dp_client_id} create failed.")
            return None
    
    def get_async(
        self,
        token_ids: torch.Tensor,
        slot_mapping: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> int:
        req = GetRequest(self.dp_client_id, token_ids, slot_mapping, token_mask)
        
        self.send_to_server.send_pyobj(req)
        response: Response = self.recv_from_server.recv_pyobj()
        
        if response.success:
            logger.info(f"get_async task: {response.task_id} created.")
            return response.task_id
        else:
            logger.error(f"get_async task in DP {self.dp_client_id} create failed.")
            return None
        
    def wait(
        self,
        wait_task_ids: List[int],
    ) -> Dict[int, torch.Tensor]:
        req = WaitRequest(self.dp_client_id, None, wait_task_ids)
        
        self.send_to_server.send_pyobj(req)
        response: Response = self.recv_from_server.recv_pyobj()
        
        if response.success:
            logger.info(f"wait tasks: {wait_task_ids} finished.")
            return response.masks
        else:
            logger.error(f"wait tasks: {wait_task_ids} in DP {self.dp_client_id} failed.")
            return None

    def try_wait(
        self,
        try_wait_task_ids: List[int],
    ) -> Dict[int, torch.Tensor]:
        req = TryWaitRequest(self.dp_client_id, None, try_wait_task_ids)
        
        self.send_to_server.send_pyobj(req)
        response: Response = self.recv_from_server.recv_pyobj()
        
        if response.success:
            logger.info(f"try_wait tasks: {try_wait_task_ids} finished.")
            return response.masks
        else:
            logger.error(f"try_wait tasks: {try_wait_task_ids} in DP {self.dp_client_id} failed.")
            return None
    
class KVTPClient:
    def __init__(
        self,
        server_recv_port: str,
        dp_client_id: int,
        device_id: int,
        tp_rank: int,
    ):
        # Init inter-process communication
        context = zmq.Context(2)
        self.send_to_server = get_zmq_socket(
            context, zmq.PUSH, server_recv_port, False
        )
        self.client_recv_port = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
        self.recv_from_server = get_zmq_socket(
            context, zmq.PULL, self.client_recv_port, True
        )
        
        self.dp_client_id = dp_client_id
        self.device_id = device_id
        self.tp_rank = tp_rank
        
        logger.info(f"KVTPClient {tp_rank} of KVDPClient {self.dp_client_id} Initialized!")
    
    def register_to_server(
        self,
        kv_caches: List[torch.Tensor],
        kv_layout: KVCacheLayout,
    ):
        if not kv_caches or not kv_caches[0].is_cuda:
            raise ValueError("GPU blocks must be CUDA tensors")

        handles = []
        for layer_id, tensor in enumerate(kv_caches):
            if tensor.device.index != self.device_id:
                raise ValueError(f"All tensors must be on specified device: {self.device_id}")

            handle = export_layer_tensor_handle(tensor, layer_id, kv_layout)
            handles.append(handle)
        
        register_req = RegisterTPClientRequest(
            self.dp_client_id, 
            self.tp_rank, 
            self.device_id, 
            self.client_recv_port,
            handles
        )
        
        self.send_to_server.send_pyobj(register_req)
        # blocking
        response: Response = self.recv_from_server.recv_pyobj()
        if response.success:
            logger.info(f"TP client of DP client {self.dp_client_id} registered successfully!")
        else:
            logger.error(f"TP client of DP client {self.dp_client_id} registeration fialed: {response.error_msg}")
            raise



if __name__ == "__main__":
    num_layers = 32
    num_kv_heads = 8
    head_size = 128
    num_cpu_blocks = 300
    tp_size = 2
    tokens_per_block = 4

    model_config = ModelConfig(num_layers=num_layers,
                                num_kv_heads=num_kv_heads,
                                head_size=head_size,
                                use_mla=False,
                                tp_size=tp_size,
                                dtype=torch.float16)
    
    dp_client = KVDPClient("ipc:///tmp/tmp6isie_et", model_config)