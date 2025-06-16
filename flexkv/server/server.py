from collections import deque
from typing import Optional, Dict

import tempfile
import zmq

from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.common.debug import flexkv_logger
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.kvmanager import KVManager
from flexkv.server.util import get_zmq_socket
from flexkv.server.request import (
    RegisterDPClientRequest,
    RegisterTPClientRequest,
    PutRequest,
    GetRequest,
    WaitRequest,
    TryWaitRequest,
    Response,
)


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


class DPClient:
    def __init__(
        self,
        client_id: int,
        send_to_client: zmq.Socket,
        tp_size: int = 1,
    ):
        self.client_id = client_id
        self.tp_size = tp_size
        self.tp_client_dict: Dict[int, TPClient] = {}

        self.send_to_client = send_to_client

        self.is_ready: bool = False

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


class KVServer:
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        server_recv_port: Optional[str] = None
    ):

        # Init inter-process communication
        self.context = zmq.Context(2)
        if server_recv_port is None:
            server_recv_port = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
        self.recv_from_client = get_zmq_socket(
            self.context, zmq.SocketType.PULL, server_recv_port, True)

        self.client_manager = ClientManager(max_num_dp_client=model_config.dp_size)
        self.kvmanager = KVManager(model_config, cache_config)

        if self.kvmanager.is_ready():
            self.kvmanager.start()

        self.req_counter = 0

        flexkv_logger.info(f"Server Initialized! [Recv Port]: {server_recv_port}")
        # self._running = True


    def run(self) -> None:
        """Main server loop"""

        # TODO: handle error and return error response
        # TODO: support check finish
        while True:
            try:
                flexkv_logger.info("start wait for req")
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


                elif isinstance(req, RegisterTPClientRequest):
                    self.client_manager.register_tp_client(
                        self.context,
                        req.dp_client_id,
                        req.client_recv_port,
                        req.tp_rank,
                        req.device_id,
                    )

                    # register GPU Memory
                    self.kvmanager.register_single_gpu_blocks(req.handles, req.dp_client_id, req.tp_rank)

                    response = Response(req.dp_client_id)
                    result_zmq = self.client_manager.get_zmq(
                        req.dp_client_id, req.tp_rank)
                    result_zmq.send_pyobj(response)

                    if self.kvmanager.is_ready():
                        self.kvmanager.start()

                elif isinstance(req, GetRequest):
                    assert self.client_manager.is_dp_client_ready(req.dp_client_id)
                    req_id = self.kvmanager.get_async(
                        token_ids=req.token_ids,
                        slot_mapping=req.slot_mapping,
                        token_mask=req.token_mask,
                        layer_granularity=-1,
                        dp_id=req.dp_client_id,
                    )
                    response = Response(req.dp_client_id, req_id)
                    result_zmq = self.client_manager.get_zmq(
                        req.dp_client_id)
                    result_zmq.send_pyobj(response)

                elif isinstance(req, PutRequest):
                    assert self.client_manager.is_dp_client_ready(req.dp_client_id)
                    #print(f"put request: {req.token_ids} from dp {req.dp_client_id}")
                    req_id = self.kvmanager.put_async(
                        token_ids=req.token_ids,
                        slot_mapping=req.slot_mapping,
                        token_mask=req.token_mask,
                        dp_id=req.dp_client_id,
                    )
                    response = Response(req.dp_client_id, req_id)
                    result_zmq = self.client_manager.get_zmq(
                        req.dp_client_id)
                    result_zmq.send_pyobj(response)

                elif isinstance(req, WaitRequest):
                    # TODO: support TP client wait
                    # TODO: try_wait api? asyncio?
                    masks = self.kvmanager.wait(
                        req.wait_task_ids,
                    )
                    response = Response(req.dp_client_id, masks=masks)
                    result_zmq = self.client_manager.get_zmq(
                        req.dp_client_id)
                    result_zmq.send_pyobj(response)

                elif isinstance(req, TryWaitRequest):
                    # TODO: support TP client try_wait
                    masks = self.kvmanager.try_wait(
                        req.try_wait_task_ids,
                    )
                    response = Response(req.dp_client_id, masks=masks)
                    result_zmq = self.client_manager.get_zmq(
                        req.dp_client_id)
                    result_zmq.send_pyobj(response)

                else:
                    raise TypeError(f"Unregonized RequestType: {type(req)}")

            except zmq.ZMQError as e:
                flexkv_logger.error(f"ZMQ Error: {e}", exc_info=True)
            except Exception as e:
                flexkv_logger.error(f"Error: {e}", exc_info=True)


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
    tp_size = 2
    tokens_per_block = 4

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
