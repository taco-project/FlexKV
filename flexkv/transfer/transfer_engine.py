from typing import Dict, List, Optional
from queue import Queue
import threading
import time
import torch
from .scheduler import TransferScheduler
from .worker import TransferWorker, GPUCPUTransferWorker, CPUSSDDiskTransferWorker
from ..common.transfer import TransferOp, TransferOpGraph, DeviceType, TransferType
from ..common.storage import AccessibleHandle

class TransferEngine:
    def __init__(self,
        gpu_handles: List[AccessibleHandle],
        cpu_handle: AccessibleHandle,
        ssd_handle: AccessibleHandle = None):
        """
        Initialize transfer engine
        
        Args:
            gpu_handles: List of GPU handles
            cpu_handle: CPU handle
            ssd_handle: Optional SSD handle
        """
        # Initialize scheduler
        self.scheduler = TransferScheduler()
        
        self.max_batch_size = 32

        # Initialize queues
        self.task_queue = Queue()  # Queue for new transfer graphs
        self.completed_queue = Queue()  # Queue for completed transfer graphs
        self.finished_ops_queue = Queue()  # Queue for finished operations
        
        # Initialize workers
        self.gpucpu_workers = [
            GPUCPUTransferWorker(
                worker_id=i,
                gpu_blocks_ptrs=gpu_handles[i].data,
                cpu_blocks_ptrs=cpu_handle.data,
                finished_queue=self.finished_ops_queue,
                gpu_kv_shape=gpu_handles[i].kv_shape,
                cpu_kv_shape=cpu_handle.kv_shape,
                dtype=gpu_handles[i].dtype,
                max_batch_size=self.max_batch_size,
                gpu_device_id=i,    
            )
            for i in range(len(gpu_handles))
        ]
        if ssd_handle is not None:
            self.cpussd_worker = CPUSSDDiskTransferWorker(
                worker_id=0,
                cpu_block_ptrs=cpu_handle.data,
                ssd_file=ssd_handle.data,
                cpu_kv_shape=cpu_handle.kv_shape,
                ssd_kv_shape=ssd_handle.kv_shape,
                dtype=cpu_handle.dtype,
                max_batch_size=self.max_batch_size,
                finished_queue=self.finished_ops_queue,
            )
        else:
            self.cpussd_worker = None
        
        # Start scheduler thread
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self._scheduler_thread.start()

    def _scheduler_loop(self):
        """Main scheduler loop"""
        while self._running:
            # Process new transfer graphs
            new_graphs_num = 0
            while not self.task_queue.empty():
                transfer_graph = self.task_queue.get()
                self.scheduler.add_transfer_graph(transfer_graph)
                new_graphs_num += 1
            # Collect finished ops
            finished_ops = []
            while not self.finished_ops_queue.empty():
                op = self.finished_ops_queue.get()
                finished_ops.append(op)
            
            if finished_ops or new_graphs_num > 0:
                # Schedule next operations
                completed_graphs, next_ops = self.scheduler.schedule(finished_ops)
                
                # Handle completed graphs
                for graph in completed_graphs:
                    self.completed_queue.put(graph)
                
                # Distribute new ops to workers
                for op in next_ops:
                    self._assign_op_to_worker(op)
            
            time.sleep(0.001)  # Prevent busy waiting

    def _assign_op_to_worker(self, op: TransferOp):
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
            self.cpussd_worker.submit_transfer(op)
        else:
            raise ValueError(f"Unsupported transfer type: {op.transfer_type}")

    def submit_transfer_graph(self, transfer_graph: TransferOpGraph):
        """Submit a transfer graph for execution"""
        self.task_queue.put(transfer_graph)

    def get_completed_graph(self, timeout: Optional[float] = None) -> Optional[int]:
        """Get ID of a completed transfer graph"""
        if self.completed_queue.empty():
            return None
        return self.completed_queue.get(timeout=timeout)

    def report_finished_op(self, op: TransferOp, graph_id: int):
        """Report a finished operation"""
        self.finished_ops_queue.put((op, graph_id))

    def shutdown(self):
        """Shutdown the transfer engine"""
        self._running = False
        self._scheduler_thread.join()
        
        # Shutdown all workers
        for worker_type in self.workers:
            for worker in self.workers[worker_type]:
                worker.shutdown()