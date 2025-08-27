import multiprocessing as mp
import time
import queue
from queue import Queue
from typing import Dict, Optional, List, Tuple
from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe, Event
import zmq
import tempfile
import threading
import numpy as np

from flexkv.common.transfer import TransferOpGraph
from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.common.debug import flexkv_logger
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.transfer import DeviceType
from flexkv.common.storage import KVCacheLayout
from flexkv.storage.storage_engine import StorageEngine
from flexkv.transfer.transfer_engine import TransferEngine
from flexkv.server.utils import get_zmq_socket
from flexkv.server.request import RegisterTPClientRequest, Response


class TransferManager:
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_register_port: str):
        self.model_config = model_config
        self.cache_config = cache_config
        self.gpu_register_port = gpu_register_port

        self.gpu_layout: Optional[KVCacheLayout] = None
        self.all_gpu_blocks: Dict[int, List[TensorSharedHandle]] = {}  # device_id -> gpu_blocks

        self.context = zmq.Context(2)
        self.recv_from_client = get_zmq_socket(
            self.context, zmq.SocketType.PULL, gpu_register_port, True)
        self.client_dict: Dict[int, zmq.Socket] = {}

        self.transfer_engine: Optional[TransferEngine] = None
        self.storage_engine = StorageEngine(self.model_config, self.cache_config)

    def _handle_gpu_blocks_registration(self, req: RegisterTPClientRequest) -> None:
        device_id = req.device_id

        if device_id in self.all_gpu_blocks:
            flexkv_logger.error(f"GPU {device_id} has already registered.")
            response = Response(req.dp_client_id, success=False,
                              error_msg=f"GPU {device_id} already registered")
        elif device_id >= self.model_config.tp_size * self.model_config.dp_size:
            flexkv_logger.error(f"GPU {device_id} is larger than TP size: "
                                f"{self.model_config.tp_size * self.model_config.dp_size}.")
            response = Response(req.dp_client_id, success=False,
                              error_msg=f"GPU {device_id} exceeds TP size "
                                        f"{self.model_config.tp_size * self.model_config.dp_size}")
        else:
            try:
                response = Response(req.dp_client_id)
                send_to_client = get_zmq_socket(
                    self.context, zmq.SocketType.PUSH, req.client_recv_port, False)
                send_to_client.send_pyobj(response)
                self.client_dict[device_id] = send_to_client

                self.all_gpu_blocks[device_id] = req.handles
                if self.gpu_layout is None:
                    self.gpu_layout = req.gpu_layout
                elif self.gpu_layout != req.gpu_layout:
                    flexkv_logger.error(f"GPU {device_id} has different GPU layout: "
                                        f"{self.gpu_layout} != {req.gpu_layout}")
                    raise ValueError(f"GPU {device_id} has different GPU layout: "
                                     f"{self.gpu_layout} != {req.gpu_layout}")
                flexkv_logger.info(f"GPU {device_id} registered successfully")
            except Exception as e:
                flexkv_logger.error(f"Failed to register GPU {device_id}: {e}")
                response = Response(req.dp_client_id, success=False,
                                  error_msg=f"Failed to register GPU {device_id}: {e}")

        if device_id in self.client_dict:
            self.client_dict[device_id].send_pyobj(response)

    def _register_gpu_blocks_via_socket(self) -> None:
        try:
            flexkv_logger.info(f"GPU tensor registration server started on port {self.gpu_register_port}")

            expected_gpus = self.model_config.tp_size * self.model_config.dp_size

            while len(self.all_gpu_blocks) < expected_gpus:
                try:
                    req = self.recv_from_client.recv_pyobj(zmq.NOBLOCK)
                except zmq.Again:
                    time.sleep(0.001)
                    continue

                if isinstance(req, RegisterTPClientRequest):
                    flexkv_logger.info(f"Received GPU blocks registration request: {type(req)}")
                    self._handle_gpu_blocks_registration(req)
                else:
                    flexkv_logger.error(f"Unrecognized RequestType in SchedulerServer: {type(req)}")

            flexkv_logger.info(f"All {expected_gpus} GPUs registered successfully")

        except Exception as e:
            flexkv_logger.error(f"Error in GPU registration server: {e}")
            raise
        finally:
            pass
            # TODO: fix the socket close issue
            # self.recv_from_client.close()
            # self.context.term()

    def initialize_transfer_engine(self) -> None:
        self._register_gpu_blocks_via_socket()

        assert self.gpu_layout is not None
        assert len(self.all_gpu_blocks) == self.model_config.tp_size * self.model_config.dp_size
        for device_id, gpu_blocks_wrapper in self.all_gpu_blocks.items():
            self.storage_engine.register_gpu_blocks(gpu_blocks_wrapper,
                                                    self.gpu_layout,
                                                    device_id,
                                                    dtype=self.model_config.dtype)
        self.gpu_handles = [
            self.storage_engine.get_storage_handle(DeviceType.GPU, i)
            for i in range(self.model_config.tp_size * self.model_config.dp_size)
        ]
        cpu_handle = self.storage_engine.get_storage_handle(DeviceType.CPU) \
            if self.cache_config.enable_cpu else None
        ssd_handle = self.storage_engine.get_storage_handle(DeviceType.SSD) \
            if self.cache_config.enable_ssd else None
        remote_handle = (
            self.storage_engine.get_storage_handle(DeviceType.REMOTE) \
            if self.cache_config.enable_remote \
            else None
        )
        self.transfer_engine = TransferEngine(gpu_handles=self.gpu_handles,
                                              model_config=self.model_config,
                                              cache_config=self.cache_config,
                                              cpu_handle=cpu_handle,
                                              ssd_handle=ssd_handle,
                                              remote_handle=remote_handle)

    def submit(self, transfer_graph: TransferOpGraph) -> None:
        self.transfer_engine.submit_transfer_graph(transfer_graph)

    def wait(self, timeout: Optional[float] = None) -> List[Tuple[int, int]]:
        return self.transfer_engine.get_completed_graphs_and_ops(timeout)

    def start(self) -> None:
        self.transfer_engine.start()

    def shutdown(self) -> None:
        self.transfer_engine.shutdown()


class TransferManagerHandleBase(ABC):
    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        pass

    @abstractmethod
    def submit(self, transfer_graph: TransferOpGraph) -> None:
        pass

    @abstractmethod
    def wait(self, timeout: Optional[float] = None) -> List[Tuple[int, int]]:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass


class TransferManagerIntraProcessHandle(TransferManagerHandleBase):
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_register_port: str):
        self.transfer_manager = TransferManager(model_config, cache_config, gpu_register_port)
        self._is_ready = False

    def start(self) -> None:
        self.transfer_manager.initialize_transfer_engine()
        self.transfer_manager.start()
        self._is_ready = True

    def is_ready(self) -> bool:
        return self._is_ready

    def submit(self, transfer_graph: TransferOpGraph) -> None:
        self.transfer_manager.submit(transfer_graph)

    def wait(self, timeout: Optional[float] = None) -> List[Tuple[int, int]]:
        return self.transfer_manager.wait(timeout)

    def shutdown(self) -> None:
        self.transfer_manager.shutdown()


class TransferManagerInterProcessHandle(TransferManagerHandleBase):
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_register_port: str):
        mp.set_start_method('spawn', force=True)

        self.model_config = model_config
        self.cache_config = cache_config
        self.gpu_register_port = gpu_register_port

        self.command_parent_conn, self.command_child_conn = Pipe()
        self.result_parent_conn, self.result_child_conn = Pipe()

        self.process: Optional[Process] = None
        self.ready_event = Event()

        self._completed_results: List[Tuple[int, int]] = []

    def _start_process(self) -> None:
        if self.process is not None and self.process.is_alive():
            return

        self.process = Process(
            target=self._process_worker,
            args=(self.model_config,
                  self.cache_config,
                  self.command_child_conn,
                  self.result_child_conn,
                  self.gpu_register_port,
                  self.ready_event),
            daemon=False
        )
        self.process.start()

    def _process_worker(self,
                        model_config: ModelConfig,
                        cache_config: CacheConfig,
                        command_conn,
                        result_conn,
                        gpu_register_port: str,
                        ready_event) -> None:
        try:
            transfer_manager = TransferManager(model_config, cache_config, gpu_register_port)
            transfer_manager.initialize_transfer_engine()
            transfer_manager.start()
            ready_event.set()
            while True:
                try:
                    if command_conn.poll(timeout=0.0001):
                        request = command_conn.recv()
                        request_type = request.get('type')
                        if request_type == 'submit':
                            transfer_manager.submit(request['transfer_graph'])
                        else:
                            flexkv_logger.error(f"Unrecognized request type: {request_type}")
                    try:
                        finished_ops = transfer_manager.wait(0.0001)
                        if finished_ops:
                            result_conn.send(finished_ops)
                    except queue.Empty:
                        pass
                except Exception as e:
                    flexkv_logger.error(f"Error in transfer manager process: {e}")

        except Exception as e:
            flexkv_logger.error(f"Failed to initialize transfer manager process: {e}")
        finally:
            command_conn.close()
            result_conn.close()

    def start(self) -> None:
        self._start_process()

    def is_ready(self) -> bool:
        return self.ready_event.is_set()

    def submit(self, transfer_graph: TransferOpGraph) -> None:
        self.command_parent_conn.send({
            'type': 'submit',
            'transfer_graph': transfer_graph
        })

    def wait(self, timeout: Optional[float] = None) -> List[Tuple[int, int]]:
        finished_ops: List[Tuple[int, int]] = []
        try:
            if self.result_parent_conn.poll(timeout=timeout):
                finished_ops += self.result_parent_conn.recv()
                while self.result_parent_conn.poll():
                    finished_ops += self.result_parent_conn.recv()
        except EOFError:
            pass

        return finished_ops

    def shutdown(self) -> None:
        if self.process is not None:
            self.process.terminate()
            self.process.join(timeout=5.0)
            if self.process.is_alive():
                self.process.kill()
                self.process.join()

        self.command_parent_conn.close()
        self.result_parent_conn.close()

    def __del__(self):
        self.shutdown()


class TransferManagerHandle:
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 use_separate_process: bool = True,
                 gpu_register_port: Optional[str] = None):
        if gpu_register_port is None:
            gpu_register_port = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
        if use_separate_process:
            self._handle: TransferManagerHandleBase = TransferManagerInterProcessHandle(
                model_config, cache_config, gpu_register_port
            )
        else:
            self._handle: TransferManagerHandleBase = TransferManagerIntraProcessHandle(
                model_config, cache_config, gpu_register_port
            )

    def start(self) -> None:
        self._handle.start()

    def is_ready(self) -> bool:
        return self._handle.is_ready()

    def submit(self, transfer_graph: TransferOpGraph) -> None:
        self._handle.submit(transfer_graph)

    def wait(self, timeout: Optional[float] = None) -> List[Tuple[int, int]]:
        return self._handle.wait(timeout)

    def shutdown(self) -> None:
        self._handle.shutdown()
