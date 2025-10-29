import os
import multiprocessing as mp
import time
import queue
from queue import Queue
from typing import Dict, Optional, List, Tuple, Any
from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe, Event
from sympy.assumptions.assume import true
import zmq
import tempfile
import threading
import numpy as np
import textwrap
import subprocess
import pickle
import sys

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
from flexkv.common.debug import flexkv_logger


class TransferManager:
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_register_port: str):
        self.model_config = model_config
        self.cache_config = cache_config
        self.gpu_register_port = gpu_register_port

        self.all_gpu_layouts: Dict[int, KVCacheLayout] = {}
        self.all_gpu_blocks: Dict[int, List[TensorSharedHandle]] = {}  # device_id -> gpu_blocks

        self.context = zmq.Context(2)
        self.recv_from_client = get_zmq_socket(
            self.context, zmq.SocketType.PULL, gpu_register_port, True)

        self.transfer_engine: Optional[TransferEngine] = None
        self.storage_engine = StorageEngine(self.model_config, self.cache_config)

    def _handle_gpu_blocks_registration(self, req: RegisterTPClientRequest) -> None:
        device_id = req.device_id

        if device_id in self.all_gpu_blocks:
            flexkv_logger.error(f"GPU {device_id} has already registered.")
        elif device_id >= self.model_config.tp_size * self.model_config.dp_size:
            flexkv_logger.error(f"GPU {device_id} is larger than TP size: "
                                f"{self.model_config.tp_size * self.model_config.dp_size}.")
        else:
            try:
                self.all_gpu_blocks[device_id] = req.handles
                self.all_gpu_layouts[device_id] = req.gpu_layout
                flexkv_logger.info(f"GPU {device_id} registered successfully")
            except Exception as e:
                flexkv_logger.error(f"Failed to register GPU {device_id}: {e}")

    def _register_gpu_blocks_via_socket(self) -> None:
        try:
            flexkv_logger.info(f"GPU tensor registration server started on port {self.gpu_register_port}")

            expected_gpus = self.model_config.tp_size * self.model_config.dp_size
            flexkv_logger.info(f"{self.model_config.tp_size=}, {self.model_config.dp_size=}, {expected_gpus=}")
            while len(self.all_gpu_blocks) < expected_gpus:
                try:
                    # Recv from: flexkv.server.client.KVTPClient.register_to_server
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

        assert len(self.all_gpu_layouts) == self.model_config.tp_size * self.model_config.dp_size
        assert len(self.all_gpu_blocks) == self.model_config.tp_size * self.model_config.dp_size
        for device_id, gpu_blocks_wrapper in self.all_gpu_blocks.items():
            self.storage_engine.register_gpu_blocks(gpu_blocks_wrapper,
                                                    self.all_gpu_layouts[device_id],
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

def get_master_host_and_ports_from_env() -> Tuple[str, Tuple[str, str, str]]:
    master_host = os.getenv("MASTER_HOST", "localhost")
    master_ports = os.getenv("MASTER_PORTS", "5556,5557,5558")
    master_ports = tuple(master_ports.split(","))
    return "tcp://" + master_host, master_ports

class TransferManagerOnRemote(TransferManager):
    """
    TransferManager for remote mode, used for multi-node tensor parallelism.
    """
    def __init__(self):
        self.master_host, self.master_ports = get_master_host_and_ports_from_env()

        self.context = zmq.Context()
        self.command_socket = self.context.socket(zmq.PULL)
        self.command_socket.setsockopt(zmq.LINGER, 0)
        self.result_socket = self.context.socket(zmq.PUSH)
        self.result_socket.setsockopt(zmq.LINGER, 0)
        self.query_socket = self.context.socket(zmq.REP)
        self.query_socket.setsockopt(zmq.LINGER, 0)

        self._shutdown_flag = False
        self._is_ready = False

        # key: graph_id, value: task_end_op_id
        self._active_graphs: Dict[int, int] = {}
        self._active_graphs_lock = threading.Lock()

        self._worker_thread: threading.Thread | None = None

        self._connect_to_master_transfer_manager()

        self._initialize_with_config()

    def _connect_to_master_transfer_manager(self) -> None:
        try:
            command_addr = f"{self.master_host}:{self.master_ports[0]}"
            self.command_socket.connect(command_addr)
            flexkv_logger.debug(f"Connected to master command port at {command_addr}")

            result_addr = f"{self.master_host}:{self.master_ports[1]}"
            self.result_socket.connect(result_addr)
            flexkv_logger.debug(f"Connected to master result port at {result_addr}")

            query_addr = f"{self.master_host}:{self.master_ports[2]}"
            self.query_socket.connect(query_addr)
            flexkv_logger.debug(f"Connected to master query port at {query_addr}")

            flexkv_logger.debug("Successfully connected to master transfer manager")

        except Exception as e:
            flexkv_logger.error(f"Failed to connect to master transfer manager: {e}")
            raise

    def _initialize_with_config(self) -> None:
        flexkv_logger.info(f"Waiting for config from master at {self.master_host}:{self.master_ports[0]}")
        config_msg = self.command_socket.recv_pyobj()
        if isinstance(config_msg, dict) and config_msg.get('type') == 'config':
            self.model_config = config_msg.get('model_config')
            self.cache_config = config_msg.get('cache_config')
            self.gpu_register_port = config_msg.get('gpu_register_port')
            flexkv_logger.info(f"Received config from master, {self.model_config = }, \
                {self.cache_config = }, {self.gpu_register_port = }.")
        else:
            raise RuntimeError(f"Expected config message, got: {config_msg}")

        super().__init__(self.model_config, self.cache_config, self.gpu_register_port)

    def _polling_worker(self) -> None:
        flexkv_logger.info("Polling worker thread started")

        poller = zmq.Poller()
        poller.register(self.command_socket, zmq.POLLIN)
        poller.register(self.query_socket, zmq.POLLIN)

        while not self._shutdown_flag:
            try:
                socks = dict(poller.poll(timeout=0.001))

                if self.command_socket in socks:
                    try:
                        message = self.command_socket.recv_pyobj(zmq.NOBLOCK)

                        if isinstance(message, dict) and message.get('type') == 'submit':
                            graph = message.get('graph')
                            task_end_op_id = message.get('task_end_op_id', -1)

                            if graph is not None:
                                graph_id = graph.graph_id

                                with self._active_graphs_lock:
                                    self._active_graphs[graph_id] = task_end_op_id

                                self.submit(graph)
                            else:
                                flexkv_logger.warning("Received submit message without graph")
                        else:
                            flexkv_logger.warning(f"Unexpected command message: {message}")
                    except zmq.Again:
                        pass

                if self.query_socket in socks:
                    try:
                        query_msg = self.query_socket.recv_pyobj(zmq.NOBLOCK)

                        if isinstance(query_msg, dict) and query_msg.get('type') == 'query_ready':
                            response = {'ready': self._is_ready}
                            self.query_socket.send_pyobj(response)
                        else:
                            response = {'error': 'unknown query type'}
                            self.query_socket.send_pyobj(response)
                            flexkv_logger.warning(f"Unknown query message: {query_msg}")
                    except zmq.Again:
                        pass

                try:
                    completed = self.wait(timeout=0.001)

                    if completed:
                        with self._active_graphs_lock:
                            for graph_id, op_id in completed:
                                if graph_id in self._active_graphs:
                                    task_end_op_id = self._active_graphs[graph_id]

                                    if task_end_op_id != -1 and op_id == task_end_op_id:
                                        self.result_socket.send_pyobj(
                                            [task_end_op_id, graph_id]
                                        )
                                    if op_id == -1:
                                        self.result_socket.send_pyobj([-1, graph_id])
                                        del self._active_graphs[graph_id]

                except queue.Empty:
                    pass

            except Exception as e:
                if not self._shutdown_flag:
                    flexkv_logger.error(f"Error in polling worker: {e}")
                    time.sleep(0.01)

        poller.unregister(self.command_socket)
        poller.unregister(self.query_socket)

    def start(self) -> None:
        self.initialize_transfer_engine()
        super().start()

        self._is_ready = true

        self._worker_thread = threading.Thread(
            target=self._polling_worker, daemon=True
        )
        self._worker_thread.start()

        flexkv_logger.info("TransferManagerOnRemote started successfully")

    def shutdown(self) -> None:
        flexkv_logger.info("Shutting down TransferManagerOnRemote")

        self._shutdown_flag = True
        self._is_ready = False

        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

        super().shutdown()

        try:
            self.command_socket.close()
            self.result_socket.close()
            self.query_socket.close()
            self.context.term()
        except Exception as e:
            flexkv_logger.error(f"Error closing sockets: {e}")

        flexkv_logger.info("TransferManagerOnRemote shutdown complete")

    def __del__(self) -> None:
        if not self._shutdown_flag:
            self.shutdown()

    # @classmethod
    # def create_process(cls, **kwargs: Any) -> Process:
    #     def _run():
    #         instance = cls(**kwargs)
    #         instance.start()
    #         if hasattr(instance, '_worker_thread') and instance._worker_thread is not None:
    #             instance._worker_thread.join()  # block until worker thread exits
    #     process = Process(target=_run, daemon=False)
    #     process.start()
    #     return process
    
    @classmethod
    def create_process(cls, **kwargs: Any) -> Process:
        import tempfile
        import os
        
        # Serialize the class and kwargs
        cls_data = pickle.dumps(cls)
        kwargs_data = pickle.dumps(kwargs)
        
        # Create temporary files for serialized data
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.cls') as f:
            f.write(cls_data)
            cls_file = f.name
            
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.kwargs') as f:
            f.write(kwargs_data)
            kwargs_file = f.name
        
        # Prepare environment - remove MPI-related variables to avoid conflicts
        env = os.environ.copy()
        mpi_vars = [k for k in env.keys() if any(prefix in k for prefix in ['MPI', 'OMPI', 'PMI', 'UCX'])]
        for var in mpi_vars:
            env.pop(var, None)
        env['MPI4PY_RC_INITIALIZE'] = 'false'
        env['PYTHONUNBUFFERED'] = '1'  # Ensure output is unbuffered
        
        # CRITICAL: Remove CUDA_VISIBLE_DEVICES to allow access to all GPUs
        # TransferManager needs to access all physical GPUs for IPC
        if 'CUDA_VISIBLE_DEVICES' in env:
            flexkv_logger.info(f"Removing CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} for TransferManager subprocess")
            env.pop('CUDA_VISIBLE_DEVICES', None)
        
        # Create the subprocess script
        transfer_manager_script = textwrap.dedent(f'''
            import os
            import sys
            import pickle
            import tempfile
            
            # Immediately disable MPI to avoid conflicts
            os.environ['MPI4PY_RC_INITIALIZE'] = 'false'
    
            # Add FlexKV to Python path
            sys.path.insert(0, "/cfs_zhongwei/rongwei/FlexKV")

            try:
                # Load the class and kwargs
                with open("{cls_file}", "rb") as f:
                    cls = pickle.load(f)
                
                with open("{kwargs_file}", "rb") as f:
                    kwargs = pickle.load(f)
                
                # Create and start TransferManager instance
                instance = cls(**kwargs)
                instance.start()
                
                # Keep running until worker thread exits
                if hasattr(instance, '_worker_thread') and instance._worker_thread is not None:
                    instance._worker_thread.join()
                    
            except Exception as e:
                print(f"Error in TransferManager subprocess: {{e}}", file=sys.stderr)
                sys.exit(1)
            finally:
                # Clean up temporary files
                try:
                    os.unlink("{cls_file}")
                    os.unlink("{kwargs_file}")
                except Exception:
                    pass
        ''').strip()
        
        # Start the subprocess
        process = subprocess.Popen([
            sys.executable, '-c', transfer_manager_script
        ], env=env, stdout=None, stderr=None, text=True)  # None = inherit parent's stdout/stderr
        flexkv_logger.info(f"TransferManager subprocess started, PID: {process.pid}")
        
        # Clean up temporary files after subprocess completes
        def cleanup_files():
            # Wait for subprocess to complete before cleaning up files
            process.wait()
            try:
                os.unlink(cls_file)
                os.unlink(kwargs_file)
            except Exception:
                pass
        
        import threading
        cleanup_thread = threading.Thread(target=cleanup_files, daemon=True)
        cleanup_thread.start()
        
        # Return a wrapper that mimics multiprocessing.Process interface
        class SubprocessWrapper:
            def __init__(self, popen_process):
                self._popen = popen_process
                self.pid = popen_process.pid
                
            def join(self, timeout=None):
                return self._popen.wait(timeout)
                
            def close(self):
                # Close the subprocess pipes
                if self._popen.stdout:
                    self._popen.stdout.close()
                if self._popen.stderr:
                    self._popen.stderr.close()
                if self._popen.stdin:
                    self._popen.stdin.close()
        
        return SubprocessWrapper(process)

class TransferManagerHandleBase(ABC):
    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        pass

    @abstractmethod
    def submit(self, transfer_graph: TransferOpGraph, task_end_op_id: int = -1) -> None:
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

    def submit(self, transfer_graph: TransferOpGraph, task_end_op_id: int = -1) -> None:
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
        self.mp_ctx = mp.get_context('spawn')

        self.model_config = model_config
        self.cache_config = cache_config
        self.gpu_register_port = gpu_register_port

        self.command_parent_conn, self.command_child_conn = self.mp_ctx.Pipe()
        self.result_parent_conn, self.result_child_conn = self.mp_ctx.Pipe()

        self.process: Optional[Process] = None
        self.start_event = self.mp_ctx.Event()
        self.ready_event = self.mp_ctx.Event()

        self._completed_results: List[Tuple[int, int]] = []

    def _start_process(self) -> None:
        if self.process is not None and self.process.is_alive():
            return
        
        self.process = self.mp_ctx.Process(
            target=self._process_worker,
            args=(self.model_config,
                  self.cache_config,
                  self.command_child_conn,
                  self.result_child_conn,
                  self.gpu_register_port,
                  self.ready_event,
                  self.start_event),
            daemon=False
        )
        self.process.start()

    def _process_worker(self,
                        model_config: ModelConfig,
                        cache_config: CacheConfig,
                        command_conn,
                        result_conn,
                        gpu_register_port: str,
                        ready_event,
                        start_event) -> None:
        try:
            start_event.set()
            os.environ['MPI4PY_RC_INITIALIZE'] = 'false'
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
        os.environ['MPI4PY_RC_INITIALIZE'] = 'false'
        self._start_process()
        self.start_event.wait()
        os.environ['MPI4PY_RC_INITIALIZE'] = 'true'

    def is_ready(self) -> bool:
        return self.ready_event.is_set()

    def submit(self, transfer_graph: TransferOpGraph, task_end_op_id: int = -1) -> None:
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


class TranserManagerMultiNodeHandle(TransferManagerHandleBase):
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_register_port: str,
                 master_host: str,
                 master_ports: Tuple[str, str, str]):  # command, result, query
        self.model_config = model_config
        self.cache_config = cache_config
        self.gpu_register_port = gpu_register_port

        self.master_host = master_host
        self.master_ports = master_ports

        self.context = zmq.Context()
        self.command_socket = self.context.socket(zmq.PUSH)
        self.command_socket.setsockopt(zmq.LINGER, 0)
        self.result_socket = self.context.socket(zmq.PULL)
        self.result_socket.setsockopt(zmq.LINGER, 0)
        self.query_socket = self.context.socket(zmq.REQ)
        self.query_socket.setsockopt(zmq.LINGER, 0)
        self.query_socket.setsockopt(zmq.REQ_RELAXED, 1)
        self.query_socket.setsockopt(zmq.REQ_CORRELATE, 1)
        self.query_socket.setsockopt(zmq.RCVTIMEO, 1000)

        self._shutdown_flag = False
        self._connected = False

        self._result_buffer: List[Tuple[int, int]] = []
        self._result_buffer_lock = threading.Lock()

        self._bind_master_ports()

        self._polling_thread: threading.Thread | None = None

    def _bind_master_ports(self) -> None:
        try:
            command_addr = f"{self.master_host}:{self.master_ports[0]}"
            self.command_socket.bind(command_addr)
            flexkv_logger.debug(f"Master bound command port at {command_addr}")

            result_addr = f"{self.master_host}:{self.master_ports[1]}"
            self.result_socket.bind(result_addr)
            flexkv_logger.debug(f"Master bound result port at {result_addr}")

            query_addr = f"{self.master_host}:{self.master_ports[2]}"
            self.query_socket.bind(query_addr)
            flexkv_logger.debug(f"Master bound query port at {query_addr}")

            self.result_socket.setsockopt(zmq.RCVTIMEO, 0)

            self._connected = True
            flexkv_logger.debug("Master transfer manager ready for remote connections")

        except Exception as e:
            flexkv_logger.error(f"Master failed to bind ports: {e}")
            raise

    def send_config_to_remotes(self) -> None:
        flexkv_logger.info(f"Sending config to remote at {self.master_host}:{self.master_ports[0]}")
        try:
            config_msg = {
                'type': 'config',
                'model_config': self.model_config,
                'cache_config': self.cache_config,
                'gpu_register_port': self.gpu_register_port
            }
            self.command_socket.send_pyobj(config_msg)
            flexkv_logger.info(f"Config sent to remote at {self.master_host}:{self.master_ports[0]}")
        except Exception as e:
            flexkv_logger.error(f"Failed to send config to remote: {e}")

    def _polling_worker(self) -> None:
        while not self._shutdown_flag:
            try:
                result = self.result_socket.recv_pyobj(zmq.NOBLOCK)
                if isinstance(result, list) and len(result) == 2:
                    op_id, graph_id = result

                    with self._result_buffer_lock:
                        self._result_buffer.append((graph_id, op_id))
                else:
                    flexkv_logger.warning(f"Unexpected result format from remote: {result}")

            except zmq.Again:
                time.sleep(0.001)
            except Exception as e:
                if not self._shutdown_flag:
                    flexkv_logger.error(f"Error in polling thread: {e}")
                    time.sleep(0.01)

    def start(self) -> None:
        self._polling_thread = threading.Thread(target=self._polling_worker, daemon=True)
        self._polling_thread.start()

    def is_ready(self) -> bool:
        if not self._connected:
            flexkv_logger.warning("Master not ready: ports not bound yet")
            return False

        try:
            query_msg = {'type': 'query_ready'}
            self.query_socket.send_pyobj(query_msg)

            response = self.query_socket.recv_pyobj()
            if response.get('ready'):
                return True
            else:
                flexkv_logger.warning(f"Remote not ready, response: {response}")
                return False

        except zmq.Again:
            flexkv_logger.warning("Timeout waiting for ready response from remote")
            return False
        except Exception as e:
            flexkv_logger.error(f"Error checking remote ready status: {e}")

            return False

    def submit(self, transfer_graph: TransferOpGraph, task_end_op_id: int = -1) -> None:
        if not self._connected:
            flexkv_logger.warning("Not connected to remote transfer manager")
            return

        try:
            message = {
                'type': 'submit',
                'graph': transfer_graph,
                'task_end_op_id': task_end_op_id
            }
            self.command_socket.send_pyobj(message)

        except Exception as e:
            flexkv_logger.error(f"Failed to submit graph to remote: {e}")

    def wait(self, timeout: float | None = None) -> List[Tuple[int, int]]:
        start_time = time.time()
        results = []

        while True:
            with self._result_buffer_lock:
                if self._result_buffer:
                    results.extend(self._result_buffer)
                    self._result_buffer.clear()
                    break
                elif timeout is not None and (time.time() - start_time) >= timeout:
                    break

            time.sleep(0.001)

        return results

    def shutdown(self) -> None:
        flexkv_logger.info("Shutting down TransferManagerMultiNodeHandle")

        self._shutdown_flag = True

        if self._polling_thread is not None and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=5.0)

        try:
            self.command_socket.close()
            self.result_socket.close()
            self.query_socket.close()
            self.context.term()
        except Exception as e:
            flexkv_logger.error(f"Error closing sockets: {e}")

        flexkv_logger.info("TransferManagerMultiNodeHandle shutdown complete")


class TransferManagerHandle:
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_register_port: Optional[str] = None,
                 mode: str = "process",
                 **kwargs): # process or thread or remote
        if gpu_register_port is None:
            gpu_register_port = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
        if mode == "process":
            self._handle: TransferManagerHandleBase = TransferManagerInterProcessHandle(
                model_config, cache_config, gpu_register_port
            )
        elif mode == "thread":
            self._handle: TransferManagerHandleBase = TransferManagerIntraProcessHandle(
                model_config, cache_config, gpu_register_port
            )
        elif mode == "remote":
            master_host = kwargs["master_host"]
            master_ports = kwargs["master_ports"]
            self._handle: TransferManagerHandleBase = TranserManagerMultiNodeHandle(
                model_config, cache_config, gpu_register_port, master_host, master_ports
            )
        else:
            raise ValueError(f"Invalid mode: {mode}, must be process, thread or remote")

    def start(self) -> None:
        self._handle.start()

    def is_ready(self) -> bool:
        return self._handle.is_ready()

    def submit(self, transfer_graph: TransferOpGraph, task_end_op_id: int = -1) -> None:
        self._handle.submit(transfer_graph, task_end_op_id)

    def wait(self, timeout: Optional[float] = None) -> List[Tuple[int, int]]:
        return self._handle.wait(timeout)

    def shutdown(self) -> None:
        self._handle.shutdown()
