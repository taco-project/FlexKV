from typing import Dict, List, Optional, Tuple
import threading
import contextlib
import time
from queue import Queue
import queue

import torch
import nvtx
from .scheduler import TransferScheduler
from .worker import TransferWorker, GPUCPUTransferWorker, CPUSSDDiskTransferWorker
from ..common.transfer import TransferOp, TransferOpGraph, DeviceType, TransferType
from ..common.storage import AccessibleHandle
import multiprocessing as mp
from multiprocessing import Queue as MPQueue
import copy
from ..common.debug import debuginfo
from ..common.transfer import get_nvtx_range_color, get_nvtx_default_color

class TransferEngine:
    def __init__(self,
        gpu_handles: List[AccessibleHandle],
        cpu_handle: AccessibleHandle,
        ssd_handle: AccessibleHandle = None,
        remote_handle: AccessibleHandle = None):
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
        self.task_queue = Queue()
        self.completed_queue = Queue()
        self.finished_ops_queue = MPQueue()
        self.op_id_to_op = {}

        self.op_id_to_nvtx_range = {}

        # Initialize workers
        self.gpucpu_workers = [
            GPUCPUTransferWorker.create_worker(
                worker_id=i,
                gpu_blocks=gpu_handles[i].data,
                cpu_blocks=cpu_handle.data,
                finished_ops_queue=self.finished_ops_queue,
                gpu_kv_layout=gpu_handles[i].kv_layout,
                cpu_kv_layout=cpu_handle.kv_layout,
                dtype=gpu_handles[i].dtype,
                gpu_device_id=gpu_handles[i].gpu_device_id,
            )
            for i in range(len(gpu_handles))
        ]

        # Wait for GPU-CPU workers to initialize
        self.cpussd_read_worker = None
        self.cpussd_write_worker = None
        self.remotecpu_worker = None
        if ssd_handle is not None:
            self.cpussd_read_worker = CPUSSDDiskTransferWorker.create_worker(
                worker_id=0,
                cpu_blocks=cpu_handle.data,
                ssd_file=ssd_handle.data,
                finished_ops_queue=self.finished_ops_queue,
                cpu_kv_layout=cpu_handle.kv_layout,
                ssd_kv_layout=ssd_handle.kv_layout,
                dtype=cpu_handle.dtype,
            )
            self.cpussd_write_worker = CPUSSDDiskTransferWorker.create_worker(
                worker_id=1,
                cpu_blocks=cpu_handle.data,
                ssd_file=ssd_handle.data,
                finished_ops_queue=self.finished_ops_queue,
                cpu_kv_layout=cpu_handle.kv_layout,
                ssd_kv_layout=ssd_handle.kv_layout,
                dtype=cpu_handle.dtype,
            )
        # TODO replace with the cpu-remoteFileSystem transfer worker
        if remote_handle is not None:
            self.remotecpu_worker = CPUSSDDiskTransferWorker.create_worker(
                worker_id=0,
                cpu_blocks=cpu_handle.data,
                ssd_file=remote_handle.data,
                finished_ops_queue=self.finished_ops_queue,
                cpu_kv_layout=cpu_handle.kv_layout,
                ssd_kv_layout=remote_handle.kv_layout,
                dtype=cpu_handle.dtype,
            )
        # Wait for all workers to ready
        for worker in self.gpucpu_workers:
            worker.ready_event.wait(timeout=60)
        if self.cpussd_read_worker is not None:
            self.cpussd_read_worker.ready_event.wait(timeout=60)
        if self.cpussd_write_worker is not None:
            self.cpussd_write_worker.ready_event.wait(timeout=60)
        if self.remotecpu_worker is not None:
            self.remotecpu_worker.ready_event.wait(timeout=60)

        # Start scheduler thread
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self._scheduler_thread.start()

    def _scheduler_loop(self):
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
            finished_ops = []
            while True:
                try:
                    op_id = self.finished_ops_queue.get_nowait()
                    op = self.op_id_to_op[op_id]
                    self.completed_queue.put((op.transfer_graph_id, op.transfer_op_id))
                    finished_ops.append(op)
                    del self.op_id_to_op[op_id]
                except queue.Empty:
                    break
            for op in finished_ops:
                nvtx.end_range(self.op_id_to_nvtx_range[op.transfer_op_id])
                self.op_id_to_nvtx_range.pop(op.transfer_op_id)
            if finished_ops or new_graphs_num > 0:
                # Schedule next operations
                completed_graph_ids, next_ops = self.scheduler.schedule(finished_ops)
                # Handle completed graphs
                for graph_id in completed_graph_ids:
                    self.completed_queue.put((graph_id, -1))
                # Distribute new ops to workers
                for op in next_ops:
                    if op.transfer_type == TransferType.VIRTUAL:
                        self.completed_queue.put((op.transfer_graph_id, op.transfer_op_id))
                    else:
                        self.op_id_to_op[op.transfer_op_id] = op
                        self._assign_op_to_worker(op)
            time.sleep(0.001)  # Prevent busy waiting

    def _assign_op_to_worker(self, op: TransferOp):
        self.op_id_to_nvtx_range[op.transfer_op_id] = nvtx.start_range(f"schedule {op.transfer_type.name} "
                                                                       f"op_id: {op.transfer_op_id}, "
                                                                       f"graph_id: {op.transfer_graph_id}, "
                                                                       f"successors: {op.successors}",
                                                                       color=get_nvtx_range_color(op.transfer_graph_id))
        """Assign operation to appropriate worker"""
        # Determine worker type based on transfer type
        if op.transfer_type in [
            TransferType.H2D,
            TransferType.D2H
        ]:
            if op.transfer_type == TransferType.H2D:
                gpu_device_id = op.dst_descriptor.device_id
            else:
                gpu_device_id = op.src_descriptor.device_id
            self.gpucpu_workers[gpu_device_id].submit_transfer(op)
        elif op.transfer_type in [
            TransferType.H2DISK,
            TransferType.DISK2H
        ]:
            if op.transfer_type == TransferType.H2DISK:
                self.cpussd_write_worker.submit_transfer(op)
            else:
                self.cpussd_read_worker.submit_transfer(op)
        elif op.transfer_type in [
            TransferType.H2REMOTE,
            TransferType.REMOTE2H
        ]:
            self.remotecpu_worker.submit_transfer(copy.deepcopy(op))
        elif op.transfer_type == TransferType.VIRTUAL:
            pass
        else:
            raise ValueError(f"Unsupported transfer type: {op.transfer_type}")

    def submit_transfer_graph(self, transfer_graph: TransferOpGraph):
        """Submit a transfer graph for execution"""
        self.task_queue.put(transfer_graph)

    def get_completed_graphs_and_ops(self, timeout: Optional[float] = None) -> List[Tuple[int, int]]:
        """Get IDs of all completed transfer graphs at current moment

        Args:
            timeout: Optional timeout for the first graph retrieval

        Returns:
            List of completed graph IDs. Empty list if no graphs are completed.
        """
        completed_graph_ids = []

        if self.completed_queue.empty():
            return completed_graph_ids

        try:
            first_graph = self.completed_queue.get(timeout=timeout)
            completed_graph_ids.append(first_graph)

            while not self.completed_queue.empty():
                completed_graph_ids.append(self.completed_queue.get_nowait())

        except Queue.Empty:
            pass

        return completed_graph_ids

    def shutdown(self):
        """Shutdown the transfer engine"""
        try:
            self._running = False
            self._scheduler_thread.join(timeout=5)

            # shutdown all workers
            for worker in self.gpucpu_workers:
                worker.shutdown()
                print("clear gpu cpu worker")
            if self.cpussd_read_worker is not None:
                self.cpussd_read_worker.shutdown()
                print("clear cpu ssd read worker")
            if self.cpussd_write_worker is not None:
                self.cpussd_write_worker.shutdown()
                print("clear cpu ssd write worker")
            if self.remotecpu_worker is not None:
                self.remotecpu_worker.shutdown()
                print("clear cpu remote worker")

            # clear finished_ops_queue
            with contextlib.suppress(Exception):
                while not self.finished_ops_queue.empty():
                    self.finished_ops_queue.get_nowait()

            # clear CUDA cache
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        except Exception as e:
            debuginfo.error(f"Error during shutdown: {e}")
            # terminate all workers
            for worker in self.gpucpu_workers:
                if worker.process.is_alive():
                    worker.process.terminate()
                    worker.process.join()
            if self.cpussd_read_worker is not None and self.cpussd_read_worker.process.is_alive():
                self.cpussd_read_worker.process.terminate()
                self.cpussd_read_worker.process.join()
            if self.cpussd_write_worker is not None and self.cpussd_write_worker.process.is_alive():
                self.cpussd_write_worker.process.terminate()
                self.cpussd_write_worker.process.join()
            if self.remotecpu_worker is not None and self.remotecpu_worker.process.is_alive():
                self.remotecpu_worker.process.terminate()
                self.remotecpu_worker.process.join()
