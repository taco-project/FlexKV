import queue
import threading
import time
from multiprocessing import Queue as MPQueue
from queue import Queue
from typing import Dict, List, Optional, Tuple, Union

import contextlib
import nvtx
import torch

from flexkv.common.debug import flexkv_logger
from flexkv.common.storage import StorageHandle
from flexkv.common.transfer import TransferOp, TransferOpGraph, TransferType
from flexkv.common.transfer import get_nvtx_range_color
from flexkv.transfer.scheduler import TransferScheduler
from flexkv.transfer.worker import (
    WorkerHandle,
    CPUSSDDiskTransferWorker,
    CPURemoteTransferWorker,
    GPUCPUTransferWorker,
    tpGPUCPUTransferWorker,
)
from flexkv.common.config import CacheConfig, ModelConfig


class TransferEngine:
    def __init__(self,
        gpu_handles: List[StorageHandle],
        model_config: ModelConfig,
        cache_config: CacheConfig,
        cpu_handle: Optional[StorageHandle] = None,
        ssd_handle: Optional[StorageHandle] = None,
        remote_handle: Optional[StorageHandle] = None):
        """
        Initialize transfer engine

        Args:
            gpu_handles: List of GPU handles
            cpu_handle: CPU handle
            ssd_handle: Optional SSD handle
            remote_handle: Optional remote handle
        """
        # Initialize scheduler
        self.scheduler = TransferScheduler()
        self.task_queue: Queue[TransferOpGraph] = Queue()
        self.completed_queue: Queue[Tuple[int, int]] = Queue()
        self.finished_ops_queue: MPQueue[int] = MPQueue()
        self.op_id_to_op: Dict[int, TransferOp] = {}
        self.gpu_handles = gpu_handles
        self._cpu_handle = cpu_handle
        self._ssd_handle = ssd_handle
        self._remote_handle = remote_handle

        self.op_id_to_nvtx_range: Dict[int, str] = {}

        self.dp_size = model_config.dp_size
        self.tp_size = model_config.tp_size

        assert len(gpu_handles) == self.dp_size * self.tp_size
        self._running = False

    def _init_workers(self) -> None:
        if self._running:
            return
        self._worker_map: Dict[TransferType, Union[WorkerHandle, List[WorkerHandle]]] = {}

        assert self._cpu_handle is not None
        if self.tp_size == 1:
            self.gpucpu_workers: List[WorkerHandle] = [
                GPUCPUTransferWorker.create_worker(
                    worker_id=i,
                    finished_ops_queue=self.finished_ops_queue,
                    gpu_blocks=self.gpu_handles[i].get_tensor_handle_list(),
                    cpu_blocks=self._cpu_handle.get_tensor_list(),
                    gpu_kv_layout=self.gpu_handles[i].kv_layout,
                    cpu_kv_layout=self._cpu_handle.kv_layout,
                    dtype=self.gpu_handles[i].dtype,
                    gpu_device_id=i,
                )
                for i in range(self.dp_size)
            ]
        else:
            self.gpucpu_workers = [
                tpGPUCPUTransferWorker.create_worker(
                    worker_id=i,
                    finished_ops_queue=self.finished_ops_queue,
                    gpu_blocks=[self.gpu_handles[j].get_tensor_handle_list() \
                                for j in range(i * self.tp_size, (i + 1) * self.tp_size)],
                    cpu_blocks=self._cpu_handle.get_tensor_list(),
                    gpu_kv_layout=self.gpu_handles[i].kv_layout,
                    cpu_kv_layout=self._cpu_handle.kv_layout,
                    dtype=self.gpu_handles[i].dtype,
                    tp_group_size=self.tp_size,
                    dp_group_id=i,
                )
                for i in range(self.dp_size)
            ]
        self._worker_map[TransferType.H2D] = self.gpucpu_workers
        self._worker_map[TransferType.D2H] = self.gpucpu_workers

        if self._ssd_handle is not None and self._cpu_handle is not None:
            self.cpussd_read_worker: WorkerHandle = CPUSSDDiskTransferWorker.create_worker(
                worker_id=10,
                finished_ops_queue=self.finished_ops_queue,
                cpu_blocks=self._cpu_handle.get_tensor_list(),
                ssd_file=self._ssd_handle.get_file_list(),
                cpu_kv_layout=self._cpu_handle.kv_layout,
                ssd_kv_layout=self._ssd_handle.kv_layout,
                dtype=self._cpu_handle.dtype,
            )
            self.cpussd_write_worker: WorkerHandle = CPUSSDDiskTransferWorker.create_worker(
                worker_id=11,
                finished_ops_queue=self.finished_ops_queue,
                cpu_blocks=self._cpu_handle.get_tensor_list(),
                ssd_file=self._ssd_handle.get_file_list(),
                cpu_kv_layout=self._cpu_handle.kv_layout,
                ssd_kv_layout=self._ssd_handle.kv_layout,
                dtype=self._cpu_handle.dtype,
            )
            self._worker_map[TransferType.H2DISK] = self.cpussd_write_worker
            self._worker_map[TransferType.DISK2H] = self.cpussd_read_worker
        if self._remote_handle is not None and self._cpu_handle is not None:
            self.remotecpu_read_worker: WorkerHandle = CPURemoteTransferWorker.create_worker(
                worker_id=20,
                finished_ops_queue=self.finished_ops_queue,
                cpu_blocks=self._cpu_handle.get_tensor_list(),
                remote_file=self._remote_handle.get_file_list(),
                cpu_kv_layout=self._cpu_handle.kv_layout,
                remote_kv_layout=self._remote_handle.kv_layout,
                dtype=self._cpu_handle.dtype,
                remote_config_custom=self._remote_handle.remote_config_custom,
            )
            self.remotecpu_write_worker: WorkerHandle = CPURemoteTransferWorker.create_worker(
                worker_id=21,
                finished_ops_queue=self.finished_ops_queue,
                cpu_blocks=self._cpu_handle.get_tensor_list(),
                remote_file=self._remote_handle.get_file_list(),
                cpu_kv_layout=self._cpu_handle.kv_layout,
                remote_kv_layout=self._remote_handle.kv_layout,
                dtype=self._cpu_handle.dtype,
                remote_config_custom=self._remote_handle.remote_config_custom,
            )
            self._worker_map[TransferType.H2REMOTE] = self.remotecpu_write_worker
            self._worker_map[TransferType.REMOTE2H] = self.remotecpu_read_worker
        if len(self._worker_map) == 0:
            raise ValueError("No workers initialized, please check the config")
        # Wait for all workers to ready
        for worker in self._worker_map.values():
            if isinstance(worker, List):
                for w in worker:
                    w.ready_event.wait(timeout=60)
            else:
                worker.ready_event.wait(timeout=60)
        # Start scheduler thread
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self._scheduler_thread.start()
    
    def start(self) -> None:
        self._init_workers()

    def _scheduler_loop(self) -> None:
        """Main scheduler loop"""
        while self._running:
            # Process new transfer graphs
            new_graphs_num = 0
            while True:
                try:
                    transfer_graph = self.task_queue.get_nowait()
                    self.scheduler.add_transfer_graph(transfer_graph)
                    new_graphs_num += 1
                except queue.Empty:
                    break
            # Collect finished ops
            finished_ops: List[TransferOp] = []
            while True:
                try:
                    op_id = self.finished_ops_queue.get_nowait()
                    op = self.op_id_to_op[op_id]
                    self.completed_queue.put((op.graph_id, op.op_id))
                    finished_ops.append(op)
                    del self.op_id_to_op[op_id]
                except queue.Empty:
                    break
            for op in finished_ops:
                nvtx.end_range(self.op_id_to_nvtx_range[op.op_id])
                self.op_id_to_nvtx_range.pop(op.op_id)
            if finished_ops or new_graphs_num > 0:
                # Schedule next operations
                completed_graph_ids, next_ops = self.scheduler.schedule(finished_ops)
                # Distribute new ops to workers
                for op in next_ops:
                    if op.transfer_type == TransferType.VIRTUAL:
                        self.completed_queue.put((op.graph_id, op.op_id))
                    else:
                        self.op_id_to_op[op.op_id] = op
                        self._assign_op_to_worker(op)
                # Handle completed graphs
                for graph_id in completed_graph_ids:
                    self.completed_queue.put((graph_id, -1))
            time.sleep(0.0001)  # Prevent busy waiting

    def _assign_op_to_worker(self, op: TransferOp) -> None:
        self.op_id_to_nvtx_range[op.op_id] = nvtx.start_range(f"schedule {op.transfer_type.name} "
                                                                       f"op_id: {op.op_id}, "
                                                                       f"graph_id: {op.graph_id}, "
                                                                       f"successors: {op.successors}",
                                                                       color=get_nvtx_range_color(op.graph_id))
        """Assign operation to appropriate worker"""
        if op.transfer_type == TransferType.VIRTUAL:
            return
        if op.transfer_type not in self._worker_map:
            raise ValueError(f"Unsupported transfer type: {op.transfer_type}")

        worker = self._worker_map[op.transfer_type]
        if isinstance(worker, List):
            worker[op.dp_id].submit_transfer(op)
        else:
            worker.submit_transfer(op)

    def submit_transfer_graph(self, transfer_graph: TransferOpGraph) -> None:
        """Submit a transfer graph for execution"""
        self.task_queue.put(transfer_graph)

    def get_completed_graphs_and_ops(self, timeout: Optional[float] = None) -> List[Tuple[int, int]]:
        """Get IDs of all completed transfer graphs at current moment

        Args:
            timeout: Optional timeout for the first graph retrieval

        Returns:
            List of completed graph IDs. Empty list if no graphs are completed.
        """
        completed_graph_ids: List[Tuple[int, int]] = []

        if self.completed_queue.empty():
            return completed_graph_ids

        try:
            first_graph = self.completed_queue.get(timeout=timeout)
            completed_graph_ids.append(first_graph)

            while not self.completed_queue.empty():
                completed_graph_ids.append(self.completed_queue.get_nowait())

        except queue.Empty:
            pass

        return completed_graph_ids

    def shutdown(self) -> None:
        """Shutdown the transfer engine"""
        try:
            self._running = False
            self._scheduler_thread.join(timeout=5)

            # shutdown all workers
            for worker in self._worker_map.values():
                if isinstance(worker, List):
                    for w in worker:
                        w.shutdown()
                else:
                    worker.shutdown()
        except Exception as e:
            flexkv_logger.error(f"Error during shutdown: {e}")
        finally:
            with contextlib.suppress(Exception):
                while not self.finished_ops_queue.empty():
                    self.finished_ops_queue.get_nowait()

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
