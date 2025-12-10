import time
from multiprocessing import Lock, Queue
from multiprocessing.connection import Connection
from queue import Queue as ThreadQueue
from typing import Dict, List, Optional, Tuple, Callable

import tempfile
import torch
import zmq
import numpy as np

from flexkv.common.config import ModelConfig, CacheConfig
from flexkv.common.debug import flexkv_logger
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.storage import KVCacheLayout
from flexkv.common.request import KVResponseStatus, KVResponse
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
    CheckRunningRequest,
    StartRequest,
    ShutdownRequest,
    Response
)

class KVDPClient:
    def __init__(
        self,
        server_recv_port: str,
        model_config: ModelConfig,
        dp_client_id: int,
    ):
        # Init inter-process communication
        context = zmq.Context(2)
        self.send_to_server = get_zmq_socket(
            context, zmq.SocketType.PUSH, server_recv_port, False
        )
        self.client_recv_port = f"ipc://{tempfile.NamedTemporaryFile(delete=True).name}"
        self.recv_from_server = get_zmq_socket(
            context, zmq.SocketType.PULL, self.client_recv_port, True
        )
        self.dp_client_id = dp_client_id
        self.model_config = model_config

        self._task_id_range = (self.dp_client_id * 10000000, (self.dp_client_id + 1) * 10000000)
        self._task_id_counter = self._task_id_range[0]
        self._task_id_lock = Lock()
        flexkv_logger.info(f"KVDPClient Initialized! [DP Client ID]: {self.dp_client_id}")

    def _get_task_id(self) -> int:
        with self._task_id_lock:
            old_value = self._task_id_counter
            self._task_id_counter += 1
            if self._task_id_counter >= self._task_id_range[1]:
                self._task_id_counter = self._task_id_range[0]
            return old_value

    def start_server_and_register(self) -> None:
        #start server and register
        req = StartRequest(self.dp_client_id)
        self.send_to_server.send_pyobj(req)
        self.register_to_server(self.model_config, self.client_recv_port)

    def register_to_server(
        self,
        model_config: ModelConfig,
        client_recv_port: str,
    ) -> None:
        register_req = RegisterDPClientRequest(self.dp_client_id, model_config, client_recv_port)
        self.send_to_server.send_pyobj(register_req)
        flexkv_logger.info(f"DP client {self.dp_client_id} registered to server request sent!")

    def is_ready(
        self,
    ) -> bool:
        req = IsReadyRequest(self.dp_client_id)
        self.send_to_server.send_pyobj(req)
        response: Response = self.recv_from_server.recv_pyobj()
        return response.is_ready

    def put_async(
        self,
        token_ids: np.ndarray,
        slot_mapping: np.ndarray,
        token_mask: Optional[np.ndarray],
    ) -> int:
        req = PutRequest(self.dp_client_id,
                         token_ids,
                         slot_mapping,
                         token_mask if token_mask is not None else None,
                         self._get_task_id())
        self.send_to_server.send_pyobj(req)
        return req.task_id

    def put_match(
        self,
        token_ids: np.ndarray,
        token_mask: Optional[np.ndarray],
    ) -> Optional[Tuple[int, np.ndarray]]:
        req = PutMatchRequest(self.dp_client_id,
                              token_ids,
                              token_mask if token_mask is not None else None,
                              self._get_task_id())
        self.send_to_server.send_pyobj(req)
        response: Response = self.recv_from_server.recv_pyobj()
        if response.error_msg is None:
            return response.task_id, response.mask
        else:
            flexkv_logger.error(f"put_match failed, error_msg: {response.error_msg}")
            return None

    def get_async(
        self,
        token_ids: np.ndarray,
        slot_mapping: np.ndarray,
        token_mask: Optional[np.ndarray],
        layer_granularity: int,
    ) -> int:
        req = GetRequest(self.dp_client_id,
                         token_ids,
                         slot_mapping,
                         token_mask if token_mask is not None else None,
                         self._get_task_id(),
                         layer_granularity)
        self.send_to_server.send_pyobj(req)
        return req.task_id

    def get_match(
        self,
        token_ids: np.ndarray,
        token_mask: Optional[np.ndarray],
        layer_granularity: int,
    ) -> Optional[Tuple[int, np.ndarray]]:
        req = GetMatchRequest(self.dp_client_id,
                              token_ids,
                              token_mask if token_mask is not None else None,
                              layer_granularity,
                              self._get_task_id())
        self.send_to_server.send_pyobj(req)
        response: Response = self.recv_from_server.recv_pyobj()
        if response.error_msg is None:
            return req.task_id, response.mask
        else:
            flexkv_logger.error(f"get_match failed, error_msg: {response.error_msg}")
            return None

    def launch_tasks(
        self,
        task_ids: List[int],
        slot_mappings: List[np.ndarray],
        as_batch: bool = False,
    ) -> List[int]:
        batch_id = -1
        if as_batch:
            batch_id = self._get_task_id()
        req = LaunchTaskRequest(self.dp_client_id, task_ids, slot_mappings, as_batch, batch_id)
        self.send_to_server.send_pyobj(req)
        return [batch_id] if as_batch else task_ids

    def cancel_task(
        self,
        task_ids: List[int],
    ) -> None:
        req = CancelTaskRequest(self.dp_client_id, task_ids)
        self.send_to_server.send_pyobj(req)

    def wait(
        self,
        wait_task_ids: List[int],
        wait_timeout: float = 20.0,
        completely: bool = False,
    ) -> Optional[Dict[int, KVResponse]]:
        req = WaitRequest(self.dp_client_id, None, wait_task_ids, wait_timeout, completely)
        self.send_to_server.send_pyobj(req)
        response: Response = self.recv_from_server.recv_pyobj()
        if response.status is not None:
            for k, v in response.status.items():
                if v.status != KVResponseStatus.SUCCESS:
                    flexkv_logger.error(f"wait task {k} failed: {v.status}")
            return response.status
        else:
            flexkv_logger.error(f"wait tasks: {wait_task_ids} in DP {self.dp_client_id} failed.")
            return None

    def try_wait(
        self,
        try_wait_task_ids: List[int],
    ) -> Optional[Dict[int, KVResponse]]:
        req = TryWaitRequest(self.dp_client_id, None, try_wait_task_ids)

        self.send_to_server.send_pyobj(req)
        response: Response = self.recv_from_server.recv_pyobj()
        if response.status is not None:
            for k, v in response.status.items():
                if v.status != KVResponseStatus.SUCCESS:
                    flexkv_logger.error(f"try_wait task {k} failed: {v.status}")
            return response.status
        else:
            flexkv_logger.error(f"try_wait tasks: {try_wait_task_ids} in DP {self.dp_client_id} failed.")
            return None

    def shutdown(self) -> None:
        req = ShutdownRequest(self.dp_client_id)
        self.send_to_server.send_pyobj(req)

class KVTPClient:
    def __init__(
        self,
        gpu_register_port: str,
        dp_client_id: int,
        device_id: int,
    ):
        # Init inter-process communication
        context = zmq.Context(2)
        self.send_to_server = get_zmq_socket(
            context, zmq.SocketType.PUSH, gpu_register_port, False
        )

        self.dp_client_id = dp_client_id
        self.device_id = device_id

        flexkv_logger.info(f"KVTPClient {device_id} of KVDPClient {self.dp_client_id} Initialized!")

    def register_to_server(
        self,
        kv_caches: List[torch.Tensor],
        kv_layout: KVCacheLayout,
        override_device_id: Optional[int] = None,
    ) -> None:
        if not kv_caches or not kv_caches[0].is_cuda:
            raise ValueError("GPU blocks must be CUDA tensors")

        # Use override_device_id if provided, otherwise use self.device_id
        device_id = override_device_id if override_device_id is not None else self.device_id

        handles = []
        for _, tensor in enumerate(kv_caches):
            handle = TensorSharedHandle(tensor, device_id)
            handles.append(handle)

        register_req = RegisterTPClientRequest(
            self.dp_client_id,
            device_id,
            handles,
            kv_layout
        )

        self.send_to_server.send_pyobj(register_req, flags=zmq.NOBLOCK)


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
