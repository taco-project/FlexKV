from collections import deque
from typing import Optional, Dict, List

import tempfile
import zmq
import torch
import time
import threading
from threading import Lock

from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.common.debug import flexkv_logger
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.kvmanager import KVManager
from flexkv.server.utils import get_zmq_socket
from flexkv.server.request import (
    RegisterDPClientRequest,
    RegisterTPClientRequest,
    IsReadyRequest,
    PutRequest,
    GetRequest,
    WaitRequest,
    TryWaitRequest,
    Response,
    ShutdownRequest,
    CheckRunningRequest,
)
import contextlib


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
        server_recv_port: Optional[str] = None,
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
            flexkv_logger.info("KVManager is ready, starting with worker initialization...")
            self.kvmanager.start()

        self.req_counter = 0

        flexkv_logger.info(f"Server Initialized! [Recv Port]: {server_recv_port}")
        # self._running = True


    def run(self) -> None:
        """Main server loop"""

        # TODO: handle error and return error response
        # TODO: support check finish
        self._running = True
        while self._running:
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
                    self.kvmanager.register_single_gpu_blocks(req.handles,
                                                            req.gpu_layout,
                                                            req.dp_client_id,
                                                            req.tp_rank)

                    response = Response(req.dp_client_id)
                    result_zmq = self.client_manager.get_zmq(
                        req.dp_client_id, req.tp_rank)
                    result_zmq.send_pyobj(response)

                    if self.kvmanager.is_ready():
                        flexkv_logger.info("All TP clients registered, starting KVManager...")
                        self.kvmanager.start()

                elif isinstance(req, IsReadyRequest):
                    is_ready = self.kvmanager.is_ready()
                    response = Response(req.dp_client_id, is_ready=is_ready)
                    result_zmq = self.client_manager.get_zmq(
                        req.dp_client_id)
                    result_zmq.send_pyobj(response)

                elif isinstance(req, GetRequest):
                    assert self.client_manager.is_dp_client_ready(req.dp_client_id)
                    req_id = self.kvmanager.get_async(
                        token_ids=torch.from_numpy(req.token_ids),
                        slot_mapping=torch.from_numpy(req.slot_mapping),
                        token_mask=torch.from_numpy(req.token_mask) if req.token_mask is not None else None,
                        layer_granularity=-1,
                        dp_id=req.dp_client_id,
                        task_id=req.task_id,
                    )
                    if req.task_id == -1:
                        response = Response(req.dp_client_id, req_id)
                        result_zmq = self.client_manager.get_zmq(
                            req.dp_client_id)
                        result_zmq.send_pyobj(response)

                elif isinstance(req, PutRequest):
                    assert self.client_manager.is_dp_client_ready(req.dp_client_id)
                    req_id = self.kvmanager.put_async(
                        token_ids=torch.from_numpy(req.token_ids),
                        slot_mapping=torch.from_numpy(req.slot_mapping),
                        token_mask=torch.from_numpy(req.token_mask) if req.token_mask is not None else None,
                        dp_id=req.dp_client_id,
                        task_id=req.task_id,
                    )
                    if req.task_id == -1:
                        response = Response(req.dp_client_id, req_id)
                        result_zmq = self.client_manager.get_zmq(
                            req.dp_client_id)
                        result_zmq.send_pyobj(response)

                elif isinstance(req, WaitRequest):
                    # TODO: support TP client wait
                    masks = self.kvmanager.wait(
                        req.wait_task_ids,
                        timeout=req.wait_timeout,
                    )
                    if masks is not None:
                        # Convert to numpy arrays for serialization
                        masks = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in masks.items()}
                    response = Response(req.dp_client_id, masks=masks)
                    result_zmq = self.client_manager.get_zmq(
                        req.dp_client_id)
                    result_zmq.send_pyobj(response)

                elif isinstance(req, TryWaitRequest):
                    # TODO: support TP client try_wait
                    masks = self.kvmanager.try_wait(
                        req.try_wait_task_ids,
                    )
                    if masks is not None:
                        # Convert to numpy arrays for serialization
                        masks = {k: v.numpy() if isinstance(v, torch.Tensor) else v for k, v in masks.items()}
                    response = Response(req.dp_client_id, masks=masks)
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

                elif isinstance(req, CheckRunningRequest):
                    response = Response(req.dp_client_id, success=True, running=self.kvmanager.is_running())
                    result_zmq = self.client_manager.get_zmq(req.dp_client_id)
                    result_zmq.send_pyobj(response)

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


class SchedulerServer:
    """
    Scheduler server that merges the functionality of KVServer and KVDPClient.
    Note that this class is ONLY FOR CASES WHEN DP_SIZE = 1.

    This class can:
    1. Directly call KVManager methods to avoid inter-process communication latency
    2. Accept registration requests from TPClient
    3. Provide the same interface as KVDPClient (put_async, get_async, wait, try_wait)
    """

    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        server_recv_port: Optional[str] = None,
    ):
        self.model_config = model_config
        self.cache_config = cache_config

        # Initialize KVManager (similar to KVServer)
        self.kvmanager = KVManager(model_config, cache_config)

        # Start KVManager if it's ready (e.g., when no TP clients are needed)
        if self.kvmanager.is_ready():
            try:
                self.kvmanager.start()
                flexkv_logger.info("KVManager started during initialization")
            except Exception as e:
                flexkv_logger.warning(f"KVManager start failed during initialization: {e}")

        # For TPClient compatibility, we need a server to receive TPClient registration requests
        self.context = zmq.Context(2)
        if server_recv_port is None:
            server_recv_port = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"

        self.server_recv_port = server_recv_port
        self.recv_from_client = get_zmq_socket(
            self.context, zmq.SocketType.PULL, server_recv_port, True)

        # Manage TP clients
        self.tp_size = model_config.tp_size
        self.tp_client_dict: Dict[int, TPClient] = {}
        self.is_ready: bool = False

        # DP client related
        self.dp_client_id = 0  # Fixed to 0 because we merged scheduler and server
        self._task_id_range = (self.dp_client_id * 10000000, (self.dp_client_id + 1) * 10000000)
        self._task_id_counter = self._task_id_range[0]
        self._task_id_lock = Lock()

        # Server thread control
        self._running = False
        self._server_thread = None

        flexkv_logger.info(f"SchedulerServer Initialized! [Recv Port]: {server_recv_port}")

    def _get_task_id(self) -> int:
        """Generate unique task ID"""
        with self._task_id_lock:
            old_value = self._task_id_counter
            self._task_id_counter += 1
            if self._task_id_counter >= self._task_id_range[1]:
                self._task_id_counter = self._task_id_range[0]
            return old_value

    def start_server_thread(self) -> None:
        """Start background server thread to handle TPClient requests"""
        if self._server_thread is not None and self._server_thread.is_alive():
            flexkv_logger.warning("Server thread is already running")
            return

        self._running = True
        self._server_thread = threading.Thread(target=self._server_loop, daemon=True)
        self._server_thread.start()
        flexkv_logger.info("SchedulerServer background thread started")

    def _server_loop(self) -> None:
        """Background server loop to handle requests from TPClient"""
        while self._running:
            try:
                # Set non-blocking receive to allow checking _running status
                try:
                    req = self.recv_from_client.recv_pyobj(zmq.NOBLOCK)
                except zmq.Again:
                    time.sleep(0.001)  # Brief sleep to avoid busy waiting
                    continue

                flexkv_logger.info(f"SchedulerServer received request: {type(req)}")

                if isinstance(req, RegisterTPClientRequest):
                    self._handle_tp_registration(req)
                elif isinstance(req, ShutdownRequest):
                    flexkv_logger.info("Received shutdown request from TP client")
                    response = Response(req.dp_client_id, success=True)
                    # Since we don't know which TP client sent the shutdown request,
                    # we send response to all registered TP clients
                    self._running = False
                    for tp_client in self.tp_client_dict.values():
                        tp_client.send_to_client.send_pyobj(response)
                    break
                else:
                    flexkv_logger.error(f"Unrecognized RequestType in SchedulerServer: {type(req)}")

            except zmq.ZMQError as e:
                if e.errno == zmq.ETERM:
                    break  # Context terminated
                flexkv_logger.error(f"ZMQ Error in SchedulerServer: {e}", exc_info=True)
            except Exception as e:
                flexkv_logger.error(f"Error in SchedulerServer: {e}", exc_info=True)
            time.sleep(0.0001)

        flexkv_logger.info("SchedulerServer background thread stopped")

    def _handle_tp_registration(self, req: RegisterTPClientRequest) -> None:
        """Handle TP Client registration request"""
        tp_rank = req.tp_rank

        if tp_rank in self.tp_client_dict:
            flexkv_logger.error(f"TP rank: {tp_rank} has already registered.")
            response = Response(req.dp_client_id, success=False,
                              error_msg=f"TP rank {tp_rank} already registered")
        elif tp_rank >= self.tp_size:
            flexkv_logger.error(f"TP rank: {tp_rank} is larger than TP size: {self.tp_size}.")
            response = Response(req.dp_client_id, success=False,
                              error_msg=f"TP rank {tp_rank} exceeds TP size {self.tp_size}")
        else:
            try:
                # Create connection to TP client
                send_to_client = get_zmq_socket(
                    self.context, zmq.SocketType.PUSH, req.client_recv_port, False
                )

                self.tp_client_dict[tp_rank] = TPClient(send_to_client, tp_rank, req.device_id)

                # Register GPU Memory to KVManager
                self.kvmanager.register_single_gpu_blocks(
                    req.handles,
                    req.gpu_layout,
                    self.dp_client_id,  # Use fixed dp_client_id = 0
                    req.tp_rank
                )

                flexkv_logger.info(f"TP rank: {tp_rank} registered successfully.")

                # Check if all TP clients have registered
                if len(self.tp_client_dict) == self.tp_size:
                    self.is_ready = True
                    # Always start kvmanager when all TP clients are registered
                    try:
                        flexkv_logger.info("All TP clients registered, starting KVManager...")
                        self.kvmanager.start()
                        flexkv_logger.info("KVManager started successfully")
                    except Exception as e:
                        flexkv_logger.warning(f"KVManager start failed or already started: {e}")
                    flexkv_logger.info("All TP clients registered. SchedulerServer is ready!")

                response = Response(req.dp_client_id, success=True)

            except Exception as e:
                flexkv_logger.error(f"Failed to register TP client {tp_rank}: {e}")
                response = Response(req.dp_client_id, success=False, error_msg=str(e))

        # Send response to TP client
        if tp_rank in self.tp_client_dict:
            self.tp_client_dict[tp_rank].send_to_client.send_pyobj(response)

    def put_async(
        self,
        token_ids: torch.Tensor,
        slot_mapping: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Optional[int]:
        """
        Asynchronous PUT operation, directly calling KVManager (no network communication required)

        Args:
            token_ids: Token IDs tensor
            slot_mapping: Slot mapping tensor
            token_mask: Optional token mask tensor

        Returns:
            Task ID if successful, None otherwise
        """
        start_time = time.time()

        if not self.is_ready:
            flexkv_logger.error("SchedulerServer is not ready (not all TP clients registered)")
            return None

        try:
            task_id = self._get_task_id()
            req_id = self.kvmanager.put_async(
                token_ids=token_ids,
                slot_mapping=slot_mapping,
                token_mask=token_mask,
                dp_id=self.dp_client_id,
                task_id=task_id,
            )

            end_time = time.time()
            flexkv_logger.info(f"[SchedulerServer] put_async task: {task_id} created. "
                              f"time: {(end_time - start_time)*1000:.2f}ms")
            return task_id

        except Exception as e:
            flexkv_logger.error(f"put_async failed: {e}")
            return None

    def get_async(
        self,
        token_ids: torch.Tensor,
        slot_mapping: torch.Tensor,
        token_mask: Optional[torch.Tensor] = None,
    ) -> Optional[int]:
        """
        Asynchronous GET operation, directly calling KVManager (no network communication required)

        Args:
            token_ids: Token IDs tensor
            slot_mapping: Slot mapping tensor
            token_mask: Optional token mask tensor

        Returns:
            Task ID if successful, None otherwise
        """
        start_time = time.time()

        if not self.is_ready:
            flexkv_logger.error("SchedulerServer is not ready (not all TP clients registered)")
            return None

        try:
            task_id = self._get_task_id()
            req_id = self.kvmanager.get_async(
                token_ids=token_ids,
                slot_mapping=slot_mapping,
                token_mask=token_mask,
                layer_granularity=-1,
                dp_id=self.dp_client_id,
                task_id=task_id,
            )

            end_time = time.time()
            flexkv_logger.info(f"[SchedulerServer] get_async task: {task_id} created. "
                              f"time: {(end_time - start_time)*1000:.2f}ms")
            return task_id

        except Exception as e:
            flexkv_logger.error(f"get_async failed: {e}")
            return None

    def wait(
        self,
        wait_task_ids: List[int],
        wait_timeout: float = 20.0,
    ) -> Optional[Dict[int, torch.Tensor]]:
        """
        Wait for specified tasks to complete, directly calling KVManager (no network communication required)

        Args:
            wait_task_ids: List of task IDs to wait for

        Returns:
            Dictionary mapping task IDs to result masks, None if failed
        """
        try:
            masks = self.kvmanager.wait(wait_task_ids, timeout=wait_timeout)
            flexkv_logger.info(f"[SchedulerServer] wait tasks: {wait_task_ids} finished.")
            return masks

        except Exception as e:
            flexkv_logger.error(f"wait failed: {e}")
            return None

    def try_wait(
        self,
        try_wait_task_ids: List[int],
    ) -> Optional[Dict[int, torch.Tensor]]:
        """
        Non-blocking wait for specified tasks, directly calling KVManager (no network communication required)

        Args:
            try_wait_task_ids: List of task IDs to try waiting for

        Returns:
            Dictionary mapping task IDs to result masks, None if not ready or failed
        """
        try:
            masks = self.kvmanager.try_wait(try_wait_task_ids)
            if masks is not None:
                flexkv_logger.info(f"[SchedulerServer] try_wait tasks: {try_wait_task_ids} finished.")
            return masks

        except Exception as e:
            flexkv_logger.error(f"try_wait failed: {e}")
            return None

    def check_running(self) -> bool:
        return self.kvmanager.is_running()

    def shutdown(self) -> None:
        """Shutdown SchedulerServer"""
        flexkv_logger.info("Shutting down SchedulerServer...")

        # Stop server thread
        self._running = False
        if self._server_thread is not None and self._server_thread.is_alive():
            self._server_thread.join(timeout=5.0)

        # Shutdown KVManager
        if hasattr(self, 'kvmanager'):
            self.kvmanager.shutdown()

        # Close ZMQ context
        #if hasattr(self, 'context'):
        #    self.context.term()

        flexkv_logger.info("SchedulerServer shutdown complete")

    def get_server_port(self) -> str:
        """Get server receive port for TPClient to use"""
        return self.server_recv_port

    def __del__(self) -> None:
        """Destructor"""
        with contextlib.suppress(Exception):
            self.shutdown()


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
