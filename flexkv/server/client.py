import time
from multiprocessing import Lock, Queue
from multiprocessing.connection import Connection
from queue import Queue as ThreadQueue
from typing import Dict, List, Optional, Tuple

import tempfile
import torch
import zmq

from flexkv.common.config import ModelConfig
from flexkv.common.debug import flexkv_logger
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.storage import KVCacheLayout
from flexkv.server.utils import get_zmq_socket
from flexkv.server.request import (
    RegisterDPClientRequest,
    RegisterTPClientRequest,
    PutRequest,
    GetRequest,
    WaitRequest,
    TryWaitRequest,
    CheckRunningRequest,
    ShutdownRequest,
    Response
)


class KVDPClient:
    def __init__(
        self,
        server_recv_port: str,
        model_config: ModelConfig,
    ):
        # Init inter-process communication
        context = zmq.Context(2)
        self.send_to_server = get_zmq_socket(
            context, zmq.SocketType.PUSH, server_recv_port, False
        )
        client_recv_port = f"ipc://{tempfile.NamedTemporaryFile(delete=True).name}"
        self.recv_from_server = get_zmq_socket(
            context, zmq.SocketType.PULL, client_recv_port, True
        )
        self.dp_client_id = self.register_to_server(model_config, client_recv_port)

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
            flexkv_logger.info(f"DP client registered successfully! DP client id: {response.dp_client_id}")
            return response.dp_client_id
        else:
            flexkv_logger.error(f"DP client registeration fialed: {response.error_msg}")
            raise

    def put_async(
        self,
        token_ids: torch.Tensor,
        slot_mapping: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> Optional[int]:
        start_time = time.time()
        req = PutRequest(self.dp_client_id,
                         token_ids.numpy(),
                         slot_mapping.numpy(),
                         token_mask.numpy() if token_mask is not None else None,
                         self._get_task_id())
        self.send_to_server.send_pyobj(req)
        end_time = time.time()
        flexkv_logger.info(f"[dpclient] put_async task: {req.task_id} created. "
                           f"time: {(end_time - start_time)*1000:.2f}ms")
        return req.task_id

    def get_async(
        self,
        token_ids: torch.Tensor,
        slot_mapping: torch.Tensor,
        token_mask: Optional[torch.Tensor],
    ) -> Optional[int]:
        start_time = time.time()
        req = GetRequest(self.dp_client_id,
                         token_ids.numpy(),
                         slot_mapping.numpy(),
                         token_mask.numpy() if token_mask is not None else None,
                         self._get_task_id())

        self.send_to_server.send_pyobj(req)
        end_time = time.time()
        flexkv_logger.info(f"[dpclient] get_async task: {req.task_id} created. "
                           f"time: {(end_time - start_time)*1000:.2f}ms")
        return req.task_id

    def wait(
        self,
        wait_task_ids: List[int],
        wait_timeout: float = 20.0,
    ) -> Optional[Dict[int, torch.Tensor]]:
        req = WaitRequest(self.dp_client_id, None, wait_task_ids, wait_timeout)

        self.send_to_server.send_pyobj(req)
        response: Response = self.recv_from_server.recv_pyobj()
        if response.masks is not None:
            response.masks = {k: torch.from_numpy(v) for k, v in response.masks.items()}
        if response.success:
            flexkv_logger.info(f"wait tasks: {wait_task_ids} finished.")
            return response.masks
        else:
            flexkv_logger.error(f"wait tasks: {wait_task_ids} in DP {self.dp_client_id} failed.")
            return None

    def try_wait(
        self,
        try_wait_task_ids: List[int],
    ) -> Optional[Dict[int, torch.Tensor]]:
        req = TryWaitRequest(self.dp_client_id, None, try_wait_task_ids)

        self.send_to_server.send_pyobj(req)
        response: Response = self.recv_from_server.recv_pyobj()
        if response.masks is not None:
            response.masks = {k: torch.from_numpy(v) for k, v in response.masks.items()}
        if response.success:
            flexkv_logger.info(f"try_wait tasks: {try_wait_task_ids} finished.")
            return response.masks
        else:
            flexkv_logger.error(f"try_wait tasks: {try_wait_task_ids} in DP {self.dp_client_id} failed.")
            return None

    def check_running(self) -> bool:
        req = CheckRunningRequest(self.dp_client_id)
        self.send_to_server.send_pyobj(req)
        response: Response = self.recv_from_server.recv_pyobj()
        return response.running

    def shutdown(self) -> None:
        req = ShutdownRequest(self.dp_client_id)
        self.send_to_server.send_pyobj(req)
        response: Response = self.recv_from_server.recv_pyobj()
        if response.success:
            flexkv_logger.info(f"DP client {self.dp_client_id} shutdown successfully.")
        else:
            flexkv_logger.error(f"DP client {self.dp_client_id} shutdown failed.")
            raise

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
            context, zmq.SocketType.PUSH, server_recv_port, False
        )
        self.client_recv_port = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
        self.recv_from_server = get_zmq_socket(
            context, zmq.SocketType.PULL, self.client_recv_port, True
        )

        self.dp_client_id = dp_client_id
        self.device_id = device_id
        self.tp_rank = tp_rank

        flexkv_logger.info(f"KVTPClient {tp_rank} of KVDPClient {self.dp_client_id} Initialized!")

    def register_to_server(
        self,
        kv_caches: List[torch.Tensor],
        kv_layout: KVCacheLayout,
    ) -> None:
        if not kv_caches or not kv_caches[0].is_cuda:
            raise ValueError("GPU blocks must be CUDA tensors")

        handles = []
        for _, tensor in enumerate(kv_caches):
            if tensor.device.index != self.device_id:
                raise ValueError(f"All tensors must be on specified device: {self.device_id}")

            handle = TensorSharedHandle(tensor)
            handles.append(handle)

        register_req = RegisterTPClientRequest(
            self.dp_client_id,
            self.tp_rank,
            self.device_id,
            self.client_recv_port,
            handles,
            kv_layout
        )

        self.send_to_server.send_pyobj(register_req)
        # blocking
        response: Response = self.recv_from_server.recv_pyobj()
        if response.success:
            flexkv_logger.info(f"TP client of DP client {self.dp_client_id} registered successfully!")
        else:
            flexkv_logger.error(
                f"TP client of DP client {self.dp_client_id} registeration fialed: {response.error_msg}"
            )
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
