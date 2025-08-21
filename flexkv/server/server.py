from collections import deque
from typing import Optional, Dict, List

import tempfile
import zmq
import torch
import time
import threading
from threading import Lock
import multiprocessing as mp
import socket
import os

from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.common.debug import flexkv_logger
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.kvtask import KVTaskEngine
from flexkv.server.utils import get_zmq_socket
from flexkv.server.request import (
    RegisterDPClientRequest,
    RegisterTPClientRequest,
    IsReadyRequest,
    PutRequest,
    GetRequest,
    PutMatchRequest,
    GetMatchRequest,
    LaunchTaskRequest,
    CancelTaskRequest,
    WaitRequest,
    TryWaitRequest,
    Response,
    ShutdownRequest,
    CheckRunningRequest,
)
import contextlib


def _is_port_in_use(port_or_endpoint: str) -> bool:
    """
    check if the port or IPC endpoint is in use by another process
    
    Args:
        port_or_endpoint: port number or IPC endpoint string (e.g. "ipc:///tmp/xxx" or "5555")
    
    Returns:
        bool: True if the port/endpoint is in use, False if it is free
    """
    try:
        if port_or_endpoint.startswith("ipc://"):
            # IPC endpoint: check if the file exists
            ipc_path = port_or_endpoint[6:]  # remove "ipc://" prefix
            return os.path.exists(ipc_path)
        elif port_or_endpoint.startswith("tcp://"):
            # TCP endpoint: parse host and port
            tcp_part = port_or_endpoint[6:]  # remove "tcp://" prefix
            if ':' in tcp_part:
                host, port_str = tcp_part.rsplit(':', 1)
                port = int(port_str)
            else:
                host = "localhost"
                port = int(tcp_part)
            
            # try to connect to the port to check if it is in use
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        else:
            # assume it is a pure port number
            port = int(port_or_endpoint)
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1)
            result = sock.connect_ex(("localhost", port))
            sock.close()
            return result == 0
    except (ValueError, OSError):
        # if parsing fails or connection fails, assume the port is free
        return False
"""
class TPClient:
    def __init__(
        self,
        send_to_client: zmq.Socket,
        tp_rank: int = 0,
        device_id: int = 0,
    ):
        self.tp_rank = tp_rank
        self.device_id = device_id
        self.send_to_client = send_to_client
"""

class DPClient:
    def __init__(
        self,
        client_id: int,
        send_to_client: zmq.Socket,
        tp_size: int = 1,
    ):
        self.client_id = client_id
        self.tp_size = tp_size

        self.send_to_client = send_to_client

        self.is_ready: bool = False
"""
    def register_tp_client(
        self,
        context: zmq.Context,
        client_recv_port: str,
        tp_rank: int = 0,
        device_id: int = 0,
    ) -> None:
        if tp_rank in self.tp_client_dict:
            flexkv_logger.error(f"TP rank: {tp_rank} in DP client: {self.client_id} has already registered.")
            raise
        if tp_rank >= self.tp_size:
            flexkv_logger.error(f"TP rank: {tp_rank} is larger than TP size of DP client: {self.client_id}.")
            raise

        send_to_client = get_zmq_socket(
            context, zmq.SocketType.PUSH, client_recv_port, False
        )

        self.tp_client_dict[tp_rank] = TPClient(send_to_client, tp_rank, device_id)

        flexkv_logger.info(f"TP rank: {tp_rank} in DP client: {self.client_id} registered successfully.")

        if len(self.tp_client_dict) == self.tp_size:
            self.is_ready = True
            flexkv_logger.info(f"All the TP clients in DP client: {self.client_id} has registered. "
                           f"DP client: {self.client_id} is ready!")
"""

class ClientManager:
    def __init__(
        self,
        max_num_dp_client: int = 1,
    ):
        #assert max_num_dp_client == 1, f"currently only support dp=1"
        self.free_client_ids = deque(range(max_num_dp_client))
        self.client_dict: Dict[int, DPClient] = {}

    def register_dp_client(
        self,
        context: zmq.Context,
        client_recv_port: str,
        tp_size: int = 1,
    ) -> int:
        if len(self.free_client_ids) == 0:
            flexkv_logger.error("Client full. DP client registration failed.")
            raise
        client_id = self.free_client_ids.popleft()
        send_to_client = get_zmq_socket(
            context, zmq.SocketType.PUSH, client_recv_port, False
        )

        self.client_dict[client_id] = DPClient(
            client_id=client_id,
            tp_size=tp_size,
            send_to_client=send_to_client,
        )
        flexkv_logger.info(f"DP client {client_id} registered successfully")

        return client_id
    """
    def register_tp_client(
        self,
        context: zmq.Context,
        dp_client_id: int,
        client_recv_port: str,
        tp_rank: int,
        device_id: int
    ) -> None:
        if dp_client_id not in self.client_dict:
            flexkv_logger.error(f"DP client: {dp_client_id} has not registered.")
            raise
        self.client_dict[dp_client_id].register_tp_client(
            context, client_recv_port, tp_rank, device_id)
    """
    def delete_dp_client(self, client_id: int) -> None:
        if client_id not in self.client_dict:
            flexkv_logger.error(f"DP client: {client_id} dosen't exist. Delete failed.")
            raise
        self.client_dict.pop(client_id)
        self.free_client_ids.appendleft(client_id)
        flexkv_logger.info(f"Delete DP client: {client_id} succeeded.")

    def get_zmq(self, dp_client_id: int, tp_rank: int = -1) -> zmq.Socket:
        dp_client = self.client_dict[dp_client_id]
        if tp_rank == -1:
            return dp_client.send_to_client
        else:
            return dp_client.tp_client_dict[tp_rank].send_to_client

    def is_dp_client_ready(self, dp_client_id: int) -> bool:
        if dp_client_id in self.client_dict:
            return self.client_dict[dp_client_id].is_ready
        return False

class KVServerHandle:
    def __init__(self, process: mp.Process, ready_event: mp.Event):
        self.process = process
        self.ready_event = ready_event
    
    def shutdown(self) -> None:
        self.process.join(timeout=5)
        if self.process.is_alive():
            flexkv_logger.info("force terminate the server process")
            self.process.terminate()
            self.process.join()

    def __del__(self) -> None:
        if self.process.is_alive():
            self.shutdown()

class KVServer:
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        gpu_register_port: str,
        server_recv_port: str
    ):

        # Init inter-process communication
        self.context = zmq.Context(2)
        self.recv_from_client = get_zmq_socket(
            self.context, zmq.SocketType.PULL, server_recv_port, True)

        self.client_manager = ClientManager(max_num_dp_client=model_config.dp_size)
        self.kv_task_engine = KVTaskEngine(model_config, cache_config, gpu_register_port, False)
        self.kv_task_engine.start()
        self._is_ready = True

        self.req_counter = 0

        flexkv_logger.info(f"Server Initialized! [Recv Port]: {server_recv_port}")
        self._running = False

    def is_ready(self) -> bool:
        return self._is_ready

    @staticmethod
    def _server_process(model_config: ModelConfig, 
                       cache_config: CacheConfig,
                       gpu_register_port: str,
                       server_recv_port: str,
                       ready_event: mp.Event) -> None:
        
        server = KVServer(model_config, cache_config, gpu_register_port, server_recv_port)
        ready_event.set()
        server.run()
        
    @classmethod
    def create_server(cls,
                      model_config: ModelConfig,
                      cache_config: CacheConfig,
                      gpu_register_port: str,
                      server_recv_port: Optional[str] = None) -> 'KVServerHandle':
        if server_recv_port is None:
            server_recv_port = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
        #if _is_port_in_use(server_recv_port):
        #    flexkv_logger.info(f"port {server_recv_port} is in use, skip starting new kvserver")
        #    return None
        #else:
        #    flexkv_logger.info(f"port {server_recv_port} is free, starting new kvserver")
        mp.set_start_method("spawn")
        ready_event = mp.Event()
        process = mp.Process(target=cls._server_process,
                             args=(model_config, cache_config, gpu_register_port, server_recv_port, ready_event))
        process.start()
        flexkv_logger.info(f"KVServer process started, PID: {process.pid}")
        
        return KVServerHandle(process, ready_event)

    def run(self) -> None:
        """Main server loop"""

        # TODO: handle error and return error response
        # TODO: support check finish
        self._running = True
        while self._running:
            try:
                flexkv_logger.info("start waiting for req")
                req = self.recv_from_client.recv_pyobj()
                flexkv_logger.info(f"recv req: {type(req)}")

                # register dp client
                if isinstance(req, RegisterDPClientRequest):
                    self._verify_model_config(req.model_config)
                    client_id = self.client_manager.register_dp_client(
                        self.context,
                        req.client_recv_port,
                        req.model_config.tp_size
                    )
                    response = Response(client_id)
                    result_zmq = self.client_manager.get_zmq(client_id)
                    result_zmq.send_pyobj(response)

                elif isinstance(req, IsReadyRequest):
                    is_ready = self.kv_task_engine.is_ready()
                    response = Response(req.dp_client_id, is_ready=is_ready)
                    result_zmq = self.client_manager.get_zmq(
                        req.dp_client_id)
                    result_zmq.send_pyobj(response)

                elif isinstance(req, GetRequest):
                    #assert self.client_manager.is_dp_client_ready(req.dp_client_id)
                    req_id = self.kv_task_engine.get_async(
                        token_ids=torch.from_numpy(req.token_ids),
                        slot_mapping=torch.from_numpy(req.slot_mapping),
                        token_mask=torch.from_numpy(req.token_mask) if req.token_mask is not None else None,
                        layer_granularity=-1,
                        dp_id=req.dp_client_id,
                        task_id=req.task_id,
                    )

                elif isinstance(req, PutRequest):
                    #assert self.client_manager.is_dp_client_ready(req.dp_client_id)
                    req_id = self.kv_task_engine.put_async(
                        token_ids=torch.from_numpy(req.token_ids),
                        slot_mapping=torch.from_numpy(req.slot_mapping),
                        token_mask=torch.from_numpy(req.token_mask) if req.token_mask is not None else None,
                        dp_id=req.dp_client_id,
                        task_id=req.task_id,
                    )
        
                elif isinstance(req, GetMatchRequest):
                    #assert self.client_manager.is_dp_client_ready(req.dp_client_id)
                    req_id, mask = self.kv_task_engine.get_match(
                        token_ids=torch.from_numpy(req.token_ids),
                        slot_mapping=torch.from_numpy(req.slot_mapping),
                        token_mask=torch.from_numpy(req.token_mask) if req.token_mask is not None else None,
                    )
                    response = Response(req.dp_client_id, task_id=req_id, mask=mask)
                    result_zmq = self.client_manager.get_zmq(
                        req.dp_client_id)
                    result_zmq.send_pyobj(response)

                elif isinstance(req, PutMatchRequest):
                    #assert self.client_manager.is_dp_client_ready(req.dp_client_id)
                    req_id, mask = self.kv_task_engine.put_match(
                        token_ids=torch.from_numpy(req.token_ids),
                        slot_mapping=torch.from_numpy(req.slot_mapping),
                        token_mask=torch.from_numpy(req.token_mask) if req.token_mask is not None else None,
                    )
                    response = Response(req.dp_client_id, task_id=req_id, mask=mask)
                    result_zmq = self.client_manager.get_zmq(
                        req.dp_client_id)
                    result_zmq.send_pyobj(response)

                elif isinstance(req, WaitRequest):
                    kv_responses = self.kv_task_engine.wait(
                        req.wait_task_ids,
                        timeout=req.wait_timeout,
                    )
                    response = Response(req.dp_client_id, status=kv_responses)
                    result_zmq = self.client_manager.get_zmq(
                        req.dp_client_id)
                    result_zmq.send_pyobj(response)

                elif isinstance(req, TryWaitRequest):
                    kv_responses = self.kv_task_engine.try_wait(
                        req.try_wait_task_ids,
                    )
                    response = Response(req.dp_client_id, status=kv_responses)
                    result_zmq = self.client_manager.get_zmq(
                        req.dp_client_id)
                    result_zmq.send_pyobj(response)

                elif isinstance(req, ShutdownRequest):
                    flexkv_logger.info(f"Received shutdown request from DP client {req.dp_client_id}")
                    # Gracefully shutdown the server
                    self._running = False
                    # Send response back to client
                    response = Response(req.dp_client_id, success=True)
                    result_zmq = self.client_manager.get_zmq(req.dp_client_id)
                    result_zmq.send_pyobj(response)
                    break
        
                else:
                    raise TypeError(f"Unregonized RequestType: {type(req)}")

            except zmq.ZMQError as e:
                flexkv_logger.error(f"ZMQ Error: {e}", exc_info=True)
            except Exception as e:
                flexkv_logger.error(f"Error: {e}", exc_info=True)

        # Cleanup after shutdown
        flexkv_logger.info("Server shutting down, cleaning up...")
        if hasattr(self, 'kvmanager'):
            self.kvmanager.shutdown()
        flexkv_logger.info("Server shutdown complete")


    def _verify_model_config(
        self,
        model_config: ModelConfig) -> bool:
        # TODO
        return True

    def __del__(self) -> None:
        self.kvmanager.shutdown()

if __name__ == "__main__":
    import torch
    num_layers = 32
    num_kv_heads = 8
    head_size = 128
    num_cpu_blocks = 300
    num_gpu_blocks = 30
    tp_size = 2
    tokens_per_block = 4

    gpu_kv_layout = KVCacheLayout(
        type=KVCacheLayoutType.LAYERWISE,
        num_layer=num_layers,
        num_block=num_gpu_blocks,
        tokens_per_block=tokens_per_block,
        num_head=num_kv_heads//tp_size,
        head_size=head_size,
        is_mla=False
    )

    model_config = ModelConfig(num_layers=num_layers,
                                num_kv_heads=num_kv_heads,
                                head_size=head_size,
                                use_mla=False,
                                tp_size=tp_size,
                                dtype=torch.float16)

    cache_config = CacheConfig(enable_cpu=True,
                                enable_ssd=False,
                                enable_remote=False,
                                use_gds=False,
                                use_pinned_memory=True,
                                tokens_per_block=tokens_per_block,
                                num_cpu_blocks=num_cpu_blocks,)

    kv_server = KVServer(model_config, cache_config, "ipc:///tmp/tmp6isie_et")
    kv_server.run()
