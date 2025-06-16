import copy
import multiprocessing as mp
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from multiprocessing import Queue as MPQueue
from queue import Empty
from threading import Thread
from typing import List, Any, Dict, Union

import ctypes
import numpy as np
import nvtx
import torch

from flexkv import c_ext

from flexkv.c_ext import transfer_kv_layers, transfer_kv_blocks_ssd, TPTransferThreadGroup
from flexkv.common.debug import flexkv_logger
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.storage import KVCacheLayout
from flexkv.common.transfer import TransferOp, TransferType
from flexkv.common.transfer import get_nvtx_range_color

try:
    from flexkv.c_ext import transfer_kv_blocks_remote
except ImportError:
    transfer_kv_blocks_remote = None


cudart = ctypes.CDLL('libcudart.so')

def cudaHostRegister(tensor: torch.Tensor) -> None:
    """Register a CPU tensor with CUDA for pinned memory access"""
    ptr = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()
    ret = cudart.cudaHostRegister(ctypes.c_void_p(ptr), ctypes.c_size_t(size), 1) # 1 means cudaHostRegisterPortable
    if ret != 0:
        raise RuntimeError(f"cudaHostRegister failed with error code {ret}")

def cudaHostUnregister(tensor: torch.Tensor) -> None:
    """Unregister a CPU tensor from CUDA for pinned memory access"""
    ptr = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()
    ret = cudart.cudaHostUnregister(ctypes.c_void_p(ptr))


@dataclass
class WorkerTransferOp:
    transfer_op_id: int
    transfer_graph_id: int
    transfer_type: TransferType
    src_block_ids: np.ndarray
    dst_block_ids: np.ndarray
    layer_id: int
    layer_granularity: int
    # successors: List[int]

    def __init__(self, transfer_op: TransferOp):
        self.transfer_op_id = transfer_op.transfer_op_id
        self.transfer_graph_id = transfer_op.transfer_graph_id
        self.transfer_type = transfer_op.transfer_type
        self.src_block_ids = transfer_op.src_descriptor.physical_block_ids.numpy()
        self.dst_block_ids = transfer_op.dst_descriptor.physical_block_ids.numpy()
        self.layer_id = transfer_op.layer_id
        self.layer_granularity = transfer_op.layer_granularity
        # self.successors = list(transfer_op.successors)  # for nvtx

class TransferWorkerBase(ABC):
    def __init__(self,
                 worker_id: int,
                 transfer_queue: MPQueue,
                 finished_ops_queue: MPQueue):
        self.worker_id = worker_id
        self.transfer_queue: MPQueue[WorkerTransferOp] = transfer_queue
        self.finished_ops_queue: MPQueue[int] = finished_ops_queue

    def _get_layer_ptrs(self, layer_blocks: List[torch.Tensor]) -> torch.Tensor:
        layer_ptrs = torch.zeros(
            len(layer_blocks),
            dtype=torch.int64,
            device="cpu",
            pin_memory=True,
        )
        for lay_id in range(len(layer_blocks)):
            layer_ptrs[lay_id] = layer_blocks[lay_id][0].data_ptr()
        return layer_ptrs

    @classmethod
    def create_worker(cls, worker_id: int, finished_ops_queue: MPQueue, *args: Any, **kwargs: Any) -> 'WorkerHandle':
        """Generic worker creation template method"""
        transfer_queue: mp.Queue[WorkerTransferOp] = mp.Queue()
        ready_event = mp.Event()

        process = mp.Process(
            target=cls._worker_process,
            args=(worker_id, transfer_queue, finished_ops_queue, ready_event, *args),
            kwargs=kwargs,
            daemon=True
        )
        process.start()

        return WorkerHandle(worker_id, transfer_queue, process, ready_event)

    @classmethod
    def _worker_process(cls, worker_id: int, transfer_queue: MPQueue, finished_ops_queue: MPQueue,
                       ready_event: Any, *args: Any, **kwargs: Any) -> None:
        worker = cls(worker_id, transfer_queue, finished_ops_queue, *args, **kwargs)
        ready_event.set()
        worker.run()

    @abstractmethod
    def _transfer_impl(
        self,
        src_block_ids: np.ndarray,
        dst_block_ids: np.ndarray,
        transfer_type: TransferType,
        layer_id: int,
        layer_granularity: int,
        **kwargs: Any
    ) -> None:
        pass

    def _log_transfer_performance(self, transfer_op: WorkerTransferOp,
                                  start_time: float, end_time: float,
                                  block_size: int, num_layers: int = -1) -> None:
        """Common method to log transfer performance"""
        if num_layers == -1:
            num_layers = transfer_op.layer_granularity

        transfer_size = (
            len(transfer_op.src_block_ids) * block_size * 2 * num_layers
        )

        flexkv_logger.info(
            f"{transfer_op.transfer_type.name} transfer request: {transfer_op.transfer_op_id} finished "
            f"transfer data size: {transfer_size} bytes "
            f"transfer time: {end_time - start_time:.4f} s "
            f"transfer bandwidth: {transfer_size / (end_time - start_time) / 1e9:.2f} GB/s"
        )

    @abstractmethod
    def launch_transfer(self, transfer_op: WorkerTransferOp) -> None:
        pass

    def run(self) -> None:
        """main loop for worker process"""
        while True:
            try:
                op = self.transfer_queue.get(timeout=0.001)
                if op is None:
                    break
                try:
                    nvtx.push_range(f"launch {op.transfer_type.name} op_id: {op.transfer_op_id}, "
                                        f"graph_id: {op.transfer_graph_id}, "
                                        f"num_blocks: {len(op.src_block_ids)}",
                                        color=get_nvtx_range_color(op.transfer_graph_id))
                    self.launch_transfer(op)
                    nvtx.pop_range()
                except Exception as e:
                    flexkv_logger.error(f"Error launching transfer: {e}\n"
                                  f"Failed transfer op: {op}")
                self.finished_ops_queue.put(op.transfer_op_id)
            except Empty:
                continue

class WorkerHandle:
    """handle for worker process"""
    def __init__(self, worker_id: int, transfer_queue: mp.Queue, process: mp.Process, ready_event: Any):
        self.worker_id = worker_id
        self.transfer_queue = transfer_queue
        self.process = process
        self.ready_event = ready_event

    def submit_transfer(self, op: TransferOp) -> None:
        self.transfer_queue.put(WorkerTransferOp(op))

    def shutdown(self) -> None:
        self.transfer_queue.put(None)
        # set timeout to 5 seconds
        self.process.join(timeout=5)
        if self.process.is_alive():
            print("force terminate the worker process")
            self.process.terminate()
            self.process.join()

class GPUCPUTransferWorker(TransferWorkerBase):
    def __init__(self,
                 worker_id: int,
                 transfer_queue: MPQueue,
                 finished_ops_queue: MPQueue,
                 gpu_blocks: List[TensorSharedHandle],
                 cpu_blocks: List[torch.Tensor],
                 gpu_kv_layout: KVCacheLayout,
                 cpu_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 gpu_device_id: int) -> None:
        # initialize worker in a new process
        super().__init__(worker_id, transfer_queue, finished_ops_queue)

        # Register CPU tensors with CUDA
        for cpu_block in cpu_blocks:
            cudaHostRegister(cpu_block)

        self.gpu_blocks = [wrapper.get_tensor() for wrapper in gpu_blocks]
        self.cpu_blocks = cpu_blocks

        # Get pointers first
        self.gpu_blocks_ptrs = self._get_layer_ptrs(self.gpu_blocks)
        self.cpu_blocks_ptrs = self._get_layer_ptrs(self.cpu_blocks)

        self.num_layers = len(self.gpu_blocks_ptrs)
        self.num_gpu_blocks = gpu_kv_layout.num_block
        self.num_cpu_blocks = cpu_kv_layout.num_block

        self.block_size = cpu_kv_layout.kv_shape[3:].numel()
        self.dtype = dtype

        self.gpu_layer_ptrs = self.gpu_blocks_ptrs
        self.cpu_layer_ptrs = self.cpu_blocks_ptrs

        self.gpu_kv_stride_in_bytes = (
            self.num_gpu_blocks * self.block_size * self.dtype.itemsize
        )
        self.cpu_kv_stride_in_bytes = (
            self.num_cpu_blocks * self.block_size * self.dtype.itemsize
        )
        self.gpu_block_stride_in_bytes = self.block_size * self.dtype.itemsize
        self.cpu_block_stride_in_bytes = self.block_size * self.dtype.itemsize

        # set GPU device
        if gpu_device_id != -1:
            torch.cuda.set_device(gpu_device_id)
        self.transfer_stream = torch.cuda.Stream()
        self.transfer_sms = 4

    def _transfer_impl(
        self,
        src_block_ids: np.ndarray,
        dst_block_ids: np.ndarray,
        transfer_type: TransferType,
        layer_id: int,
        layer_granularity: int,
        **kwargs: Any,
    ) -> None:
        assert src_block_ids.dtype == np.int64
        assert dst_block_ids.dtype == np.int64
        assert len(src_block_ids) == len(dst_block_ids)

        use_ce_transfer = kwargs.get("use_ce_transfer", False)

        if transfer_type == TransferType.H2D:
            gpu_block_ids = dst_block_ids
            cpu_block_ids = src_block_ids
        elif transfer_type == TransferType.D2H:
            gpu_block_ids = src_block_ids
            cpu_block_ids = dst_block_ids
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type} for GPUCPUTransferWorker")

        gpu_block_id_list = torch.from_numpy(gpu_block_ids).to(dtype=torch.int64).pin_memory()
        cpu_block_id_list = torch.from_numpy(cpu_block_ids).to(dtype=torch.int64).pin_memory()

        assert len(gpu_block_id_list) == len(cpu_block_id_list)

        if len(gpu_block_id_list) == 0:
            return

        layer_id_list = torch.arange(layer_id, layer_id + layer_granularity, dtype=torch.int32)

        chunk_size_in_bytes = self.block_size * self.dtype.itemsize
        gpu_layer_ptrs = self.gpu_layer_ptrs[layer_id_list].contiguous().pin_memory()
        cpu_layer_ptrs = self.cpu_layer_ptrs[layer_id_list].contiguous().pin_memory()
        if transfer_type == TransferType.H2D:
            transfer_kv_layers(
                gpu_block_id_list,
                gpu_layer_ptrs,
                self.gpu_kv_stride_in_bytes,
                self.gpu_block_stride_in_bytes,
                cpu_block_id_list,
                cpu_layer_ptrs,
                self.cpu_kv_stride_in_bytes,
                self.cpu_block_stride_in_bytes,
                chunk_size_in_bytes,
                self.transfer_sms,
                True,
                use_ce_transfer,
            )
        elif transfer_type == TransferType.D2H:
            transfer_kv_layers(
                cpu_block_id_list,
                cpu_layer_ptrs,
                self.cpu_kv_stride_in_bytes,
                self.cpu_block_stride_in_bytes,
                gpu_block_id_list,
                gpu_layer_ptrs,
                self.gpu_kv_stride_in_bytes,
                self.gpu_block_stride_in_bytes,
                chunk_size_in_bytes,
                self.transfer_sms,
                False,
                use_ce_transfer,
            )

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> None:
        layer_id = transfer_op.layer_id
        layer_granularity = transfer_op.layer_granularity
        if layer_id == -1:
            layer_id = 0
        if layer_granularity == -1:
            layer_granularity = self.num_layers

        with torch.cuda.stream(self.transfer_stream):
            start_time = time.time()
            self._transfer_impl(
                transfer_op.src_block_ids,
                transfer_op.dst_block_ids,
                transfer_op.transfer_type,
                layer_id,
                layer_granularity,
                use_ce_transfer=False,
            )
            end_time = time.time()

            self._log_transfer_performance(
                transfer_op, start_time, end_time,
                self.block_size * self.dtype.itemsize
            )

class tpGPUCPUTransferWorker(TransferWorkerBase):
    def __init__(self,
                 worker_id: int,
                 transfer_queue: MPQueue,
                 finished_ops_queue: MPQueue,
                 gpu_blocks: List[List[TensorSharedHandle]],
                 cpu_blocks: List[torch.Tensor],
                 gpu_kv_layout: KVCacheLayout,
                 cpu_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 tp_group_size: int,
                 dp_group_id: int):

        super().__init__(worker_id, transfer_queue, finished_ops_queue)
        assert len(gpu_blocks) == tp_group_size
        # Handle tensor import for multi-process case
        imported_gpu_blocks = []
        for handles_in_one_gpu in gpu_blocks:
            blocks_in_one_gpu = []
            for handle in handles_in_one_gpu:
                blocks_in_one_gpu.append(handle.get_tensor())
            imported_gpu_blocks.append(blocks_in_one_gpu)
        self.gpu_blocks = imported_gpu_blocks
        self.dtype = dtype

        self.num_gpus = len(self.gpu_blocks)
        self.tp_group_size = tp_group_size
        self.dp_group_id = dp_group_id

        for cpu_block in cpu_blocks:
            cudaHostRegister(cpu_block)

        self.num_layers = gpu_kv_layout.num_layer
        self.num_gpu_blocks = gpu_kv_layout.num_block
        self.num_cpu_blocks = cpu_kv_layout.num_block

        self.cpu_block_size = cpu_kv_layout.kv_shape[3:].numel()
        self.gpu_block_size = gpu_kv_layout.kv_shape[3:].numel()

        #self.cpu_layer_ptrs = self._get_layer_ptrs(cpu_blocks)
        #self.gpu_layer_ptrs = self._get_layer_ptrs(gpu_blocks)

        self.cpu_kv_stride_in_bytes = (
            self.num_cpu_blocks * self.cpu_block_size * self.dtype.itemsize
        )
        self.gpu_kv_stride_in_bytes = (
            self.num_gpu_blocks * self.gpu_block_size * self.dtype.itemsize
        )
        self.gpu_block_stride_in_bytes = self.gpu_block_size * self.dtype.itemsize
        self.cpu_block_stride_in_bytes = self.cpu_block_size * self.dtype.itemsize

        self.cpu_chunk_size_in_bytes = self.cpu_block_size * self.dtype.itemsize
        self.gpu_chunk_size_in_bytes = self.gpu_block_size * self.dtype.itemsize

        self.transfer_sms = 4

        self.tp_transfer_thread_group = TPTransferThreadGroup(self.num_gpus, self.gpu_blocks, cpu_blocks,dp_group_id)

    def _transfer_impl(self,
                       src_block_ids: np.ndarray,
                       dst_block_ids: np.ndarray,
                       transfer_type: TransferType,
                       layer_id: int,
                       layer_granularity: int,
                       **kwargs: Any,
                       )->None:
        assert src_block_ids.dtype == np.int64
        assert dst_block_ids.dtype == np.int64
        assert len(src_block_ids) == len(dst_block_ids)

        use_ce_transfer = kwargs.get("use_ce_transfer", False)

        if transfer_type == TransferType.H2D:
            gpu_block_ids = dst_block_ids
            cpu_block_ids = src_block_ids
        elif transfer_type == TransferType.D2H:
            gpu_block_ids = src_block_ids
            cpu_block_ids = dst_block_ids
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type} for tpGPUCPUTransferWorker")

        gpu_block_id_list = torch.from_numpy(gpu_block_ids).to(dtype=torch.int64).pin_memory()
        cpu_block_id_list = torch.from_numpy(cpu_block_ids).to(dtype=torch.int64).pin_memory()

        assert len(gpu_block_id_list) == len(cpu_block_id_list)

        if len(gpu_block_id_list) == 0:
            return

        if transfer_type == TransferType.H2D:
            self.tp_transfer_thread_group.tp_group_transfer(
                gpu_block_id_list,
                self.gpu_kv_stride_in_bytes,
                self.gpu_block_stride_in_bytes,
                self.gpu_chunk_size_in_bytes,
                cpu_block_id_list,
                self.cpu_kv_stride_in_bytes,
                self.cpu_block_stride_in_bytes,
                self.cpu_chunk_size_in_bytes,
                self.transfer_sms,
                True,
                use_ce_transfer,
                layer_id,
                layer_granularity,
            )
        elif transfer_type == TransferType.D2H:
            self.tp_transfer_thread_group.tp_group_transfer(
                cpu_block_id_list,
                self.cpu_kv_stride_in_bytes,
                self.cpu_block_stride_in_bytes,
                self.cpu_chunk_size_in_bytes,
                gpu_block_id_list,
                self.gpu_kv_stride_in_bytes,
                self.gpu_block_stride_in_bytes,
                self.gpu_chunk_size_in_bytes,
                self.transfer_sms,
                False,
                use_ce_transfer,
                layer_id,
                layer_granularity,
            )

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> None:
        layer_id = transfer_op.layer_id
        layer_granularity = transfer_op.layer_granularity
        if layer_id == -1:
            layer_id = 0
        if layer_granularity == -1:
            layer_granularity = self.num_layers

        start_time = time.time()
        self._transfer_impl(
            transfer_op.src_block_ids,
            transfer_op.dst_block_ids,
            transfer_op.transfer_type,
            layer_id,
            layer_granularity,
            use_ce_transfer=False,
        )
        end_time = time.time()

        self._log_transfer_performance(
            transfer_op, start_time, end_time,
            self.cpu_block_size * self.dtype.itemsize
        )


class CPUSSDDiskTransferWorker(TransferWorkerBase):
    def __init__(self,
                 worker_id: int,
                 transfer_queue: MPQueue,
                 finished_ops_queue: MPQueue,
                 cpu_blocks: List[torch.Tensor],
                 ssd_file: List[str],
                 cpu_kv_layout: KVCacheLayout,
                 ssd_kv_layout: KVCacheLayout,
                 dtype: torch.dtype):
        super().__init__(worker_id, transfer_queue, finished_ops_queue)
        self.ssd_files = ssd_file
        self.num_ssd_files = len(ssd_file)

        self.num_layers = cpu_kv_layout.num_layer
        self.num_cpu_blocks = cpu_kv_layout.num_block
        self.num_ssd_blocks = ssd_kv_layout.num_block
        self.round_robin = 1

        if self.num_ssd_blocks % self.num_ssd_files != 0:
            raise ValueError(f"num_ssd_blocks {self.num_ssd_blocks} "
                             f"is not divisible by num_ssd_files {self.num_ssd_files}")
        self.num_ssd_blocks_per_file = self.num_ssd_blocks // self.num_ssd_files
        if self.num_ssd_blocks_per_file % self.round_robin != 0:
            raise ValueError(f"num_ssd_blocks_per_file {self.num_ssd_blocks_per_file} "
                             f"is not divisible by round_robin {self.round_robin}")

        self.block_size = cpu_kv_layout.kv_shape[3:].numel()
        self.dtype = dtype

        self.cpu_blocks = cpu_blocks

        self.cpu_layer_ptrs = self._get_layer_ptrs(cpu_blocks)

        self.cpu_layer_stride_in_bytes = (
            self.num_cpu_blocks * self.block_size * self.dtype.itemsize * 2
        )
        self.ssd_layer_stride_in_bytes = (
            self.num_ssd_blocks * self.block_size * self.dtype.itemsize * 2
        )
        self.ssd_layer_stride_in_bytes_per_file = self.ssd_layer_stride_in_bytes // self.num_ssd_files
        self.cpu_kv_stride_in_bytes = (
            self.num_cpu_blocks * self.block_size * self.dtype.itemsize
        )
        self.ssd_kv_stride_in_bytes = (
            self.num_ssd_blocks * self.block_size * self.dtype.itemsize
        )
        self.ssd_kv_stride_in_bytes_per_file = self.ssd_kv_stride_in_bytes // self.num_ssd_files
        self.ssd_block_stride_in_bytes = self.block_size * self.dtype.itemsize
        self.cpu_block_stride_in_bytes = self.block_size * self.dtype.itemsize

        self.chunk_size_in_bytes = self.block_size * self.dtype.itemsize

    def _transfer_impl(
        self,
        src_block_ids: np.ndarray,
        dst_block_ids: np.ndarray,
        transfer_type: TransferType,
        layer_id: int,
        layer_granularity: int,
        **kwargs: Any,
    ) -> None:
        assert src_block_ids.dtype == np.int64
        assert dst_block_ids.dtype == np.int64
        assert len(src_block_ids) == len(dst_block_ids)

        if transfer_type == TransferType.H2DISK:
            ssd_block_ids = dst_block_ids
            cpu_block_ids = src_block_ids
        elif transfer_type == TransferType.DISK2H:
            ssd_block_ids = src_block_ids
            cpu_block_ids = dst_block_ids
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type} for CPUSSDDiskTransferWorker")

        # this means partial read hit cpu and other hit ssd
        # or partial write hit ssd and none hit cpu
        ssd_block_id_list = torch.from_numpy(ssd_block_ids).to(dtype=torch.int64)
        cpu_block_id_list = torch.from_numpy(cpu_block_ids).to(dtype=torch.int64)

        layer_id_list = torch.arange(layer_id, layer_id + layer_granularity, dtype=torch.int32)

        transfer_kv_blocks_ssd(
            filename_list=self.ssd_files,
            cpu_layer_id_list=layer_id_list,
            cpu_layer_ptrs_tensor=self.cpu_layer_ptrs,
            ssd_block_ids=ssd_block_id_list,
            cpu_block_ids=cpu_block_id_list,
            cpu_kv_stride_in_bytes=self.cpu_kv_stride_in_bytes,
            ssd_layer_stride_in_bytes=self.ssd_layer_stride_in_bytes_per_file,
            ssd_block_stride_in_bytes=self.ssd_block_stride_in_bytes,
            ssd_kv_stride_in_bytes=self.ssd_kv_stride_in_bytes_per_file,
            block_size_in_bytes=self.chunk_size_in_bytes,
            total_layers=self.num_layers,
            is_read=(transfer_type == TransferType.DISK2H),
            round_robin=self.round_robin,
            use_mmap=False,  # TODO: fix bug when use mmap
            num_threads_per_file=16,
        )

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> None:
        layer_id = transfer_op.layer_id
        layer_granularity = transfer_op.layer_granularity
        if layer_id == -1:
            layer_id = 0
        if layer_granularity == -1:
            layer_granularity = self.num_layers
        start_time = time.time()
        self._transfer_impl(
            transfer_op.src_block_ids,
            transfer_op.dst_block_ids,
            transfer_op.transfer_type,
            transfer_op.layer_id,
            transfer_op.layer_granularity,
        )
        end_time = time.time()

        self._log_transfer_performance(
            transfer_op, start_time, end_time,
            self.block_size * self.dtype.itemsize, self.num_layers
        )


class CPURemoteTransferWorker(TransferWorkerBase):
    def __init__(self,
                 worker_id: int,
                 transfer_queue: MPQueue,
                 finished_ops_queue: MPQueue,
                 cpu_blocks: List[torch.Tensor],
                 remote_file: List[str],
                 cpu_kv_layout: KVCacheLayout,
                 remote_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 remote_config_custom: Dict[str, Any]):
        if transfer_kv_blocks_remote is None:
            raise RuntimeError("transfer_kv_blocks_remote not available, please build with FLEXKV_ENABLE_CFS=1")
        super().__init__(worker_id, transfer_queue, finished_ops_queue)
        self.remote_files = remote_file
        self.num_remote_files = len(remote_file)

        self.num_layers = cpu_kv_layout.num_layer
        self.num_cpu_blocks = cpu_kv_layout.num_block
        self.num_remote_blocks = remote_kv_layout.num_block
        self.round_robin = 1

        if self.num_remote_blocks % self.num_remote_files != 0:
            raise ValueError(f"num_remote_blocks {self.num_remote_blocks} "
                             f"is not divisible by num_remote_files {self.num_remote_blocks}")
        self.num_remote_blocks_per_file = self.num_remote_blocks // self.num_remote_files
        if self.num_remote_blocks_per_file % self.round_robin != 0:
            raise ValueError(f"num_remote_blocks_per_file {self.num_remote_blocks_per_file} "
                             f"is not divisible by round_robin {self.round_robin}")

        self.block_size = cpu_kv_layout.kv_shape[3:].numel()
        self.dtype = dtype

        self.cpu_blocks = cpu_blocks

        self.cpu_layer_ptrs = self._get_layer_ptrs(cpu_blocks)

        self.cpu_layer_stride_in_bytes = (
            self.num_cpu_blocks * self.block_size * self.dtype.itemsize * 2
        )
        self.remote_layer_stride_in_bytes = (
            self.num_remote_blocks * self.block_size * self.dtype.itemsize * 2
        )
        self.remote_layer_stride_in_bytes_per_file = self.remote_layer_stride_in_bytes // self.num_remote_files
        self.cpu_kv_stride_in_bytes = (
            self.num_cpu_blocks * self.block_size * self.dtype.itemsize
        )
        self.remote_kv_stride_in_bytes = (
            self.num_remote_blocks * self.block_size * self.dtype.itemsize
        )
        self.remote_kv_stride_in_bytes_per_file = self.remote_kv_stride_in_bytes // self.num_remote_files
        self.remote_block_stride_in_bytes = self.block_size * self.dtype.itemsize
        self.cpu_block_stride_in_bytes = self.block_size * self.dtype.itemsize

        self.chunk_size_in_bytes = self.block_size * self.dtype.itemsize
        # 144115188075855883 only use int not c_types.u_int64
        if not remote_config_custom:
            raise RuntimeError("remote_config_custom is not provided")
        pcfs_fsid = remote_config_custom.get("pcfs_fsid")
        pcfs_port = remote_config_custom.get("pcfs_port")
        pcfs_ip = remote_config_custom.get("pcfs_ip")
        pcfs_parent_nodeid = remote_config_custom.get("pcfs_parent_nodeid")
        if None in (pcfs_fsid, pcfs_port, pcfs_ip, pcfs_parent_nodeid):
            raise RuntimeError("Some required PCFS config fields are missing")
        self.pcfs = c_ext.Pcfs(pcfs_fsid, pcfs_port, pcfs_ip, False, pcfs_parent_nodeid)
        if not self.pcfs.init():
            raise RuntimeError(f"PCFS init failed: fsid={pcfs_fsid}, ip={pcfs_ip}")
        self.file_nodeid_list = []
        for remote_file_single in remote_file:
            nodeid = self.pcfs.lookup_or_create_file(
            remote_file_single,
            (self.remote_layer_stride_in_bytes_per_file * self.num_layers))
            if nodeid == 0:
                raise RuntimeError(f"lookup or create file failed for file: {remote_file_single}")
            self.file_nodeid_list.append(nodeid)

        c_ext.set_pcfs_instance(self.pcfs)

    def _transfer_impl(
        self,
        src_block_ids: np.ndarray,
        dst_block_ids: np.ndarray,
        transfer_type: TransferType,
        layer_id: int,
        layer_granularity: int,
        **kwargs: Any
    ) -> None:
        assert dst_block_ids.dtype == np.int64
        assert src_block_ids.dtype == np.int64
        assert len(src_block_ids) == len(dst_block_ids)

        if layer_id == -1:
            layer_id = 0
        if layer_granularity == -1:
            layer_granularity = self.num_layers

        # this means partial read hit cpu and other hit remote
        # or partial write hit remote and none hit cpu
        remote_block_ids = dst_block_ids
        cpu_block_ids = src_block_ids
        remote_block_id_list = torch.from_numpy(remote_block_ids).pin_memory().to(dtype=torch.int64)
        cpu_block_id_list = torch.from_numpy(cpu_block_ids).pin_memory().to(dtype=torch.int64)

        layer_id_list = torch.arange(layer_id, layer_id + layer_granularity, dtype=torch.int32)
        transfer_kv_blocks_remote(
            file_nodeid_list=self.file_nodeid_list,
            cpu_layer_id_list=layer_id_list,
            cpu_layer_ptrs_tensor=self.cpu_layer_ptrs,
            remote_block_ids=remote_block_id_list,
            cpu_block_ids=cpu_block_id_list,
            cpu_kv_stride_in_bytes=self.cpu_kv_stride_in_bytes,
            remote_layer_stride_in_bytes=self.remote_layer_stride_in_bytes_per_file,
            remote_block_stride_in_bytes=self.remote_block_stride_in_bytes,
            remote_kv_stride_in_bytes=self.remote_kv_stride_in_bytes_per_file,
            block_size_in_bytes=self.chunk_size_in_bytes,
            total_layers=self.num_layers,
            is_read=(transfer_type == TransferType.REMOTE2H),
            round_robin=self.round_robin,
            use_mmap=False,  # TODO: fix bug when use mmap
            num_threads_per_file=32,
        )

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> None:
        layer_id = transfer_op.layer_id
        layer_granularity = transfer_op.layer_granularity
        if layer_id == -1:
            layer_id = 0
        if layer_granularity == -1:
            layer_granularity = self.num_layers
        start_time = time.time()
        self._transfer_impl(
            transfer_op.src_block_ids,
            transfer_op.dst_block_ids,
            transfer_op.transfer_type,
            transfer_op.layer_id,
            transfer_op.layer_granularity,
        )
        end_time = time.time()

        self._log_transfer_performance(
            transfer_op, start_time, end_time,
            self.block_size * self.dtype.itemsize, self.num_layers
        )
