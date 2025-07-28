import threading
import time
from collections import deque
from multiprocessing import Lock
from typing import Optional, Dict, List
import tempfile

import torch
import zmq

from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.common.debug import flexkv_logger
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.storage import KVCacheLayout
from flexkv.kvmanager import KVManager
from flexkv.server.utils import get_zmq_socket
from flexkv.server.request import (
    RegisterTPClientRequest,
    Response,
    ShutdownRequest,
)


class TPClient:
    """TP Client representation in scheduler server"""
    def __init__(
        self,
        send_to_client: zmq.Socket,
        tp_rank: int = 0,
        device_id: int = 0,
    ):
        self.tp_rank = tp_rank
        self.device_id = device_id
        self.send_to_client = send_to_client


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
        server_recv_port: Optional[str] = None
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
        self._task_id_counter = (self.dp_client_id + 1) * 10000000
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
                    flexkv_logger.info(f"Received shutdown request from TP client")
                    response = Response(req.dp_client_id, success=True)
                    # Since we don't know which TP client sent the shutdown request, we send response to all registered TP clients
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
                        self.kvmanager.start()
                        flexkv_logger.info("KVManager started successfully")
                    except Exception as e:
                        flexkv_logger.warning(f"KVManager start failed or already started: {e}")
                    flexkv_logger.info(f"All TP clients registered. SchedulerServer is ready!")
                
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
    ) -> Optional[Dict[int, torch.Tensor]]:
        """
        Wait for specified tasks to complete, directly calling KVManager (no network communication required)
        
        Args:
            wait_task_ids: List of task IDs to wait for
            
        Returns:
            Dictionary mapping task IDs to result masks, None if failed
        """
        try:
            masks = self.kvmanager.wait(wait_task_ids)
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
        try:
            self.shutdown()
        except:
            pass


if __name__ == "__main__":
    import torch
    from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
    
    # Example configuration
    num_layers = 32
    num_kv_heads = 8
    head_size = 128
    num_cpu_blocks = 300
    num_gpu_blocks = 30
    tp_size = 2
    tokens_per_block = 4

    model_config = ModelConfig(
        num_layers=num_layers,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        use_mla=False,
        tp_size=tp_size,
        dtype=torch.float16
    )

    cache_config = CacheConfig(
        enable_cpu=True,
        enable_ssd=False,
        enable_remote=False,
        use_gds=False,
        use_pinned_memory=True,
        tokens_per_block=tokens_per_block,
        num_cpu_blocks=num_cpu_blocks,
    )

    # Create SchedulerServer
    scheduler_server = SchedulerServer(model_config, cache_config, "ipc:///tmp/scheduler_server_test")
    
    # Start background server thread
    scheduler_server.start_server_thread()
    
    print(f"SchedulerServer started, server port: {scheduler_server.get_server_port()}")
    print("Ready to accept TP client registrations and handle direct calls...")
    
    # You can directly call put_async, get_async, wait, try_wait methods here
    # No network communication required
    
    try:
        # Keep the program running
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
        scheduler_server.shutdown() 