from collections import deque
from typing import Optional, Dict, List, Union

import tempfile
import zmq
import torch
import time
import threading
from threading import Lock
import multiprocessing as mp
import socket
import os
import subprocess
import textwrap

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
    StartRequest,
    ShutdownRequest,
    CheckRunningRequest,
    PrefetchRequest,
)
import contextlib

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
        client_id: Optional[int] = None,
    ) -> int:
        if client_id is None:
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
    def __init__(self, process: Union[mp.Process, 'subprocess.Popen']):
        self.process = process

class KVServer:
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        gpu_register_port: str,
        server_recv_port: str,
        total_clients: int = 0,
    ):

        # Init inter-process communication
        self.context = zmq.Context(2)
        self.recv_from_client = get_zmq_socket(
            self.context, zmq.SocketType.PULL, server_recv_port, True)

        # Use total_clients if provided (multi-instance mode), otherwise use dp_size
        max_clients = total_clients if total_clients > 0 else model_config.dp_size
        self.client_manager = ClientManager(max_num_dp_client=max_clients)
        self.kv_task_engine = KVTaskEngine(model_config, cache_config, gpu_register_port)

        self.req_counter = 0
        self._is_ready = False
        self._running = False

        # Request handler dispatch table
        self.request_handlers = {
            StartRequest: self._handle_start_request,
            RegisterDPClientRequest: self._handle_register_dp_client_request,
            IsReadyRequest: self._handle_is_ready_request,
            GetRequest: self._handle_get_request,
            PutRequest: self._handle_put_request,
            GetMatchRequest: self._handle_get_match_request,
            PutMatchRequest: self._handle_put_match_request,
            PrefetchRequest: self._handle_prefetch_request,
            WaitRequest: self._handle_wait_request,
            LaunchTaskRequest: self._handle_launch_task_request,
            CancelTaskRequest: self._handle_cancel_task_request,
            TryWaitRequest: self._handle_try_wait_request,
            ShutdownRequest: self._handle_shutdown_request,
        }

    def is_ready(self) -> bool:
        return self._is_ready

    def start_server(self) -> None:
        self.kv_task_engine.start()
        self._is_ready = True

    @staticmethod
    def _server_process(model_config: ModelConfig,
                       cache_config: CacheConfig,
                       gpu_register_port: str,
                       server_recv_port: str,
                       total_clients: int = 0) -> None:

        server = KVServer(model_config, cache_config, gpu_register_port, server_recv_port, total_clients)
        server.run()

    @classmethod
    def create_server(cls,
                      model_config: ModelConfig,
                      cache_config: CacheConfig,
                      gpu_register_port: str,
                      server_recv_port: Optional[str] = None,
                      total_clients: int = 0,
                      child_env: Optional[dict] = None,
                      inherit_env: bool = True) -> 'KVServerHandle':

        # Set spawn method for CUDA compatibility
        with contextlib.suppress(RuntimeError):
            mp.set_start_method("spawn")

        # Prepare environment variables for child process
        if child_env is not None or not inherit_env:
            # Use subprocess for better environment control
            import subprocess
            import pickle
            import sys

            # Prepare environment
            if inherit_env:
                env = os.environ.copy()
                if child_env:
                    env.update(child_env)
            else:
                env = child_env or {}
            
            # Remove CUDA_VISIBLE_DEVICES so server can see all GPUs
            env.pop('CUDA_VISIBLE_DEVICES', None)
            env.update({"FLEXKV_INSTANCE_NUM": str(total_clients // model_config.dp_size)})
            # Serialize arguments
            args_data = pickle.dumps((model_config, cache_config, gpu_register_port, server_recv_port, total_clients))

            # Start subprocess
            flexkv_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            server_script = textwrap.dedent(f'''
                import pickle
                import sys
                sys.path.insert(0, "{flexkv_root}")
                from flexkv.server.server import KVServer

                args_data = {args_data!r}
                model_config, cache_config, gpu_register_port, server_recv_port, total_clients = pickle.loads(args_data)
                server = KVServer(model_config, cache_config, gpu_register_port, server_recv_port, total_clients)
                server.run()
            ''').strip()
            process = subprocess.Popen([
                sys.executable, '-c', server_script
            ], env=env)

            flexkv_logger.info(f"KVServer subprocess started, PID: {process.pid}, total_clients: {total_clients}")
            return KVServerHandle(process)
        else:
            # Use multiprocessing as before
            process = mp.Process(target=cls._server_process,
                                 args=(model_config, cache_config, gpu_register_port, server_recv_port, total_clients))
            process.start()
            flexkv_logger.info(f"KVServer process started, PID: {process.pid}, total_clients: {total_clients}")
            return KVServerHandle(process)

    def run(self) -> None:
        """Main server loop"""

        # TODO: handle error and return error response
        # TODO: support check finish
        flexkv_logger.info("Servering waiting to be started")
        req = self.recv_from_client.recv_pyobj()
        if isinstance(req, StartRequest):
            flexkv_logger.info(f"Received start request from DP client {req.dp_client_id}, "
                               f"Starting server...")
            self.start_server()
        else:
            raise TypeError(f"Received RequestType: {type(req)} from DP client "
                            f"{req.dp_client_id} before the start request")
        self._running = True
        while self._running:
            try:
                flexkv_logger.info("start waiting for req")
                req = self.recv_from_client.recv_pyobj()
                flexkv_logger.info(f"recv req: {type(req)} from DP client {req.dp_client_id}")

                # Use dispatch table for request handling
                req_type = type(req)
                handler = self.request_handlers.get(req_type)

                if handler is None:
                    raise TypeError(f"Unrecognized RequestType: {req_type}")

                # Call the corresponding handler method
                handler(req)

                # If the request is a shutdown request, exit the loop
                if req_type == ShutdownRequest:
                    break

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

    # Request Handler Methods

    def _handle_start_request(self, req: StartRequest) -> None:
        """Handle start request"""
        flexkv_logger.info(f"Received start request from DP client {req.dp_client_id}")

    def _handle_register_dp_client_request(self, req: RegisterDPClientRequest) -> None:
        """Handle DP client registration request"""
        self._verify_model_config(req.model_config)
        client_id = self.client_manager.register_dp_client(
            self.context,
            req.client_recv_port,
            req.model_config.tp_size,
            req.dp_client_id,
        )

    def _handle_is_ready_request(self, req: IsReadyRequest) -> None:
        """Handle ready state check request"""
        is_ready = self.kv_task_engine.is_ready()
        response = Response(req.dp_client_id, is_ready=is_ready)
        result_zmq = self.client_manager.get_zmq(req.dp_client_id)
        result_zmq.send_pyobj(response)

    def _handle_get_request(self, req: GetRequest) -> None:
        """Handle Get request"""
        req_id = self.kv_task_engine.get_async(
            task_id=req.task_id,
            token_ids=req.token_ids,
            slot_mapping=req.slot_mapping,
            token_mask=req.token_mask,
            layer_granularity=req.layer_granularity,
            dp_id=req.dp_client_id,
        )

    def _handle_put_request(self, req: PutRequest) -> None:
        """Handle Put request"""
        req_id = self.kv_task_engine.put_async(
            token_ids=req.token_ids,
            slot_mapping=req.slot_mapping,
            token_mask=req.token_mask,
            dp_id=req.dp_client_id,
            task_id=req.task_id,
        )

    def _handle_get_match_request(self, req: GetMatchRequest) -> None:
        """Handle GetMatch request"""
        req_id, mask = self.kv_task_engine.get_match(
            token_ids=req.token_ids,
            token_mask=req.token_mask,
            layer_granularity=req.layer_granularity,
            dp_id=req.dp_client_id,
            task_id=req.task_id,
        )
        response = Response(req.dp_client_id, task_id=req_id, mask=mask)
        result_zmq = self.client_manager.get_zmq(req.dp_client_id)
        result_zmq.send_pyobj(response)

    def _handle_put_match_request(self, req: PutMatchRequest) -> None:
        """Handle PutMatch request"""
        req_id, mask = self.kv_task_engine.put_match(
            token_ids=req.token_ids,
            token_mask=req.token_mask,
            dp_id=req.dp_client_id,
            task_id=req.task_id,
        )
        response = Response(req.dp_client_id, task_id=req_id, mask=mask)
        result_zmq = self.client_manager.get_zmq(req.dp_client_id)
        result_zmq.send_pyobj(response)

    def _handle_prefetch_request(self, req: PrefetchRequest) -> None:
        """Handle Prefetch request"""
        task_id = self.kv_task_engine.prefetch_async(
            token_ids=req.token_ids,
            dp_id=req.dp_client_id,
            task_id=req.task_id
        )

    def _handle_launch_task_request(self, req: LaunchTaskRequest) -> None:
        """Handle LaunchTask request"""
        self.kv_task_engine.launch_tasks(req.task_ids, req.slot_mappings, req.as_batch, req.batch_id)

    def _handle_cancel_task_request(self, req: CancelTaskRequest) -> None:
        """Handle CancelTask request"""
        self.kv_task_engine.cancel_tasks(req.task_ids)

    def _handle_wait_request(self, req: WaitRequest) -> None:
        """Handle Wait request"""
        kv_responses = self.kv_task_engine.wait(
            req.wait_task_ids,
            timeout=req.wait_timeout,
            completely=req.completely,
        )
        response = Response(req.dp_client_id, status=kv_responses)
        result_zmq = self.client_manager.get_zmq(req.dp_client_id)
        result_zmq.send_pyobj(response)

    def _handle_try_wait_request(self, req: TryWaitRequest) -> None:
        """Handle TryWait request"""
        kv_responses = self.kv_task_engine.try_wait(
            req.try_wait_task_ids,
        )
        response = Response(req.dp_client_id, status=kv_responses)
        result_zmq = self.client_manager.get_zmq(req.dp_client_id)
        result_zmq.send_pyobj(response)

    def _handle_shutdown_request(self, req: ShutdownRequest) -> None:
        """Handle shutdown request"""
        flexkv_logger.info(f"Received shutdown request from DP client {req.dp_client_id}")
        self._running = False

    def __del__(self) -> None:
        self.kv_task_engine.shutdown()

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
        type=KVCacheLayoutType.LAYERFIRST,
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
                                enable_gds=False,
                                tokens_per_block=tokens_per_block,
                                num_cpu_blocks=num_cpu_blocks,)

    kv_server = KVServer(model_config, cache_config, "ipc:///tmp/tmp6isie_et")
    kv_server.run()
