from abc import ABC, abstractmethod
from typing import List, Any
import torch

import threading
from flexkv.common.transfer import TransferOp, TransferType, DeviceType, TransferDescriptor
from flexkv.c_ext import transfer_kv_layers, transfer_kv_blocks_ssd
import time
from threading import Thread
import copy
import numpy as np
from dataclasses import dataclass

import nvtx

from flexkv.common.transfer import get_nvtx_default_color, get_nvtx_range_color
from flexkv.common.debug import debuginfo
from flexkv.common.storage import KVCacheLayout
import multiprocessing as mp
from multiprocessing import Queue as MPQueue
from queue import Empty
import ctypes

cudart = ctypes.CDLL('libcudart.so')

mp.set_start_method('spawn', force=True)

def cudaHostRegister(tensor):
    """Register a CPU tensor with CUDA for pinned memory access"""
    ptr = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()
    ret = cudart.cudaHostRegister(ctypes.c_void_p(ptr), ctypes.c_size_t(size), 0)
    if ret != 0:
        raise RuntimeError(f"cudaHostRegister failed with error code {ret}")

def cudaHostUnregister(tensor):
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

class TransferWorker(ABC):
    def __init__(self, worker_id: int, finished_ops_queue: MPQueue):
        self.worker_id = worker_id
        self.transfer_queue = MPQueue()
        self.finished_ops_queue = finished_ops_queue

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

    @abstractmethod
    def _transfer_impl(self, **kwargs)->None:
        pass

    @abstractmethod
    def launch_transfer(self, transfer_op: WorkerTransferOp)->None:
        pass

    def run(self):
        # Signal initialization complete
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
                                        # f"successors: {op.successors}",
                                        color=get_nvtx_range_color(op.transfer_graph_id))
                    self.launch_transfer(op)
                    nvtx.pop_range()
                except Exception as e:
                    debuginfo.error(f"Error launching transfer: {e}\n"
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

    def submit_transfer(self, op: TransferOp):
        self.transfer_queue.put(WorkerTransferOp(op))

    def shutdown(self):
        self.transfer_queue.put(None)
        # set timeout to 5 seconds
        self.process.join(timeout=5)
        if self.process.is_alive():
            print("force terminate the worker process")
            self.process.terminate()
            self.process.join()

class GPUCPUTransferWorker(TransferWorker):
    @classmethod
    def create_worker(cls,
                     worker_id: int,
                     gpu_blocks: List[torch.Tensor],
                     cpu_blocks: List[torch.Tensor],
                     finished_ops_queue: MPQueue,
                     gpu_kv_layout: KVCacheLayout,
                     cpu_kv_layout: KVCacheLayout,
                     dtype: torch.dtype,
                     gpu_device_id: int = -1):
        transfer_queue = mp.Queue()
        ready_event = mp.Event()

        process = mp.Process(
            target=cls._worker_process,
            args=(worker_id, gpu_blocks, cpu_blocks,
                  transfer_queue, finished_ops_queue,
                  gpu_kv_layout, cpu_kv_layout, dtype, gpu_device_id, ready_event),
            daemon=True
        )
        process.start()

        return WorkerHandle(worker_id, transfer_queue, process, ready_event)

    @classmethod
    def _worker_process(cls,
                       worker_id: int,
                       gpu_blocks: List[torch.Tensor],
                       cpu_blocks: List[torch.Tensor],
                       transfer_queue: MPQueue,
                       finished_ops_queue: MPQueue,
                       gpu_kv_layout: KVCacheLayout,
                       cpu_kv_layout: KVCacheLayout,
                       dtype: torch.dtype,
                       gpu_device_id: int,
                       ready_event: Any):
        # create worker in a new process
        worker = cls(worker_id, gpu_blocks, cpu_blocks,
                    transfer_queue, finished_ops_queue,
                    gpu_kv_layout, cpu_kv_layout, dtype,
                    gpu_device_id)
        ready_event.set()
        worker.run()

    def __init__(self,
                 worker_id: int,
                 gpu_blocks: List[torch.Tensor],
                 cpu_blocks: List[torch.Tensor],
                 transfer_queue: MPQueue,
                 finished_ops_queue: MPQueue,
                 gpu_kv_layout: KVCacheLayout,
                 cpu_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 gpu_device_id: int = -1):
        # initialize worker in a new process
        super().__init__(worker_id, finished_ops_queue)

        # Register CPU tensors with CUDA
        for cpu_block in cpu_blocks:
            cudaHostRegister(cpu_block)

        # Get pointers first
        self.gpu_blocks_ptrs = self._get_layer_ptrs(gpu_blocks)
        self.cpu_blocks_ptrs = self._get_layer_ptrs(cpu_blocks)

        self.gpu_blocks = gpu_blocks
        self.cpu_blocks = cpu_blocks

        self.transfer_queue = transfer_queue

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
        gpu_block_ids: np.ndarray,
        cpu_block_ids: np.ndarray,
        transfer_type: TransferType,
        layer_id: int = -1,
        layer_granularity: int = -1,
        non_blocking: bool = False,
        use_ce_transfer: bool = False,
    ) -> None:
        if layer_id == -1:
            layer_id = 0
        if layer_granularity == -1:
            layer_granularity = self.num_layers

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
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type}")
        # if non_blocking:
        #torch.cuda.synchronize()  # TODO: remove this ?

    def launch_transfer(self, transfer_op: WorkerTransferOp):
        with torch.cuda.stream(self.transfer_stream):
            if transfer_op.transfer_type == TransferType.H2D:
                cpu_block_ids = transfer_op.src_block_ids
                gpu_block_ids = transfer_op.dst_block_ids
            elif transfer_op.transfer_type == TransferType.D2H:
                cpu_block_ids = transfer_op.dst_block_ids
                gpu_block_ids = transfer_op.src_block_ids
            else:
                raise ValueError(f"Invalid transfer type: {transfer_op.transfer_type}")
            start_time = time.time()
            self._transfer_impl(
                gpu_block_ids,
                cpu_block_ids,
                transfer_op.transfer_type,
                layer_id=transfer_op.layer_id,
                layer_granularity=transfer_op.layer_granularity,
                non_blocking=True,
                use_ce_transfer=False,
            )
            # debuginfo.info("TRANSFER stream synchronized")
            end_time = time.time()
            transfer_size = (
                len(gpu_block_ids)
                * self.block_size
                * self.dtype.itemsize
                * 2
                * transfer_op.layer_granularity
            )
            debuginfo.info(
                f"gpu cpu tranfer request: {transfer_op.transfer_op_id} finished "
                f"request type is {transfer_op.transfer_type} "
                f"transfer data size is {transfer_size} bytes "
                f"transfer time is {end_time - start_time:.4f} s "
                f"transfer bandwidth is "
                f"{transfer_size / (end_time - start_time) / 1e9:.2f} GB/s"
            )

class CPUSSDDiskTransferWorker(TransferWorker):
    @classmethod
    def create_worker(cls,
                     worker_id: int,
                     cpu_blocks: List[torch.Tensor],
                     ssd_file: List[str],
                     finished_ops_queue: MPQueue,
                     cpu_kv_layout: KVCacheLayout,
                     ssd_kv_layout: KVCacheLayout,
                     dtype: torch.dtype):
        transfer_queue = mp.Queue()
        ready_event = mp.Event()

        process = mp.Process(
            target=cls._worker_process,
            args=(worker_id, cpu_blocks, ssd_file,
                  transfer_queue, finished_ops_queue,
                  cpu_kv_layout, ssd_kv_layout, dtype, ready_event),
            daemon=True
        )
        process.start()

        return WorkerHandle(worker_id, transfer_queue, process, ready_event)

    @classmethod
    def _worker_process(cls,
                       worker_id: int,
                       cpu_blocks: List[torch.Tensor],
                       ssd_file: List[str],
                       transfer_queue: MPQueue,
                       finished_ops_queue: MPQueue,
                       cpu_kv_layout: KVCacheLayout,
                       ssd_kv_layout: KVCacheLayout,
                       dtype: torch.dtype,
                       ready_event: Any):
        worker = cls(worker_id, cpu_blocks, ssd_file,
                    transfer_queue, finished_ops_queue,
                    cpu_kv_layout, ssd_kv_layout, dtype)
        ready_event.set()
        worker.run()

    def __init__(self,
                 worker_id: int,
                 cpu_blocks: List[torch.Tensor],
                 ssd_file: List[str],
                 transfer_queue: MPQueue,
                 finished_ops_queue: MPQueue,
                 cpu_kv_layout: KVCacheLayout,
                 ssd_kv_layout: KVCacheLayout,
                 dtype: torch.dtype):
        super().__init__(worker_id, finished_ops_queue)
        self.ssd_files = ssd_file
        self.num_ssd_files = len(ssd_file)
        self.transfer_queue = transfer_queue

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
        cpu_block_ids: np.ndarray,
        ssd_block_ids: np.ndarray,
        transfer_type: TransferType,
        layer_id: int = -1,
        layer_granularity: int = -1,
        non_blocking: bool = False,
    ) -> None:
        assert ssd_block_ids.dtype == np.int64
        assert cpu_block_ids.dtype == np.int64

        if layer_id == -1:
            layer_id = 0
        if layer_granularity == -1:
            layer_granularity = self.num_layers

        # this means partial read hit cpu and other hit ssd
        # or partial write hit ssd and none hit cpu
        ssd_block_id_list = torch.from_numpy(ssd_block_ids).pin_memory().to(dtype=torch.int64)
        cpu_block_id_list = torch.from_numpy(cpu_block_ids).pin_memory().to(dtype=torch.int64)
        if len(ssd_block_id_list) != len(cpu_block_id_list):
            assert len(ssd_block_id_list) < len(cpu_block_id_list)
            cpu_block_id_list = cpu_block_ids[
                -len(ssd_block_id_list) :
            ]
            debuginfo.debug(
                f"cpu block id num {len(cpu_block_id_list)} "
                f"ssd block id num {len(ssd_block_id_list)}"
            )

        if len(ssd_block_id_list) == 0:
            return

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

    def launch_transfer(self, transfer_op: WorkerTransferOp):
        #TODO remove this when remote worker is ready
        if transfer_op.transfer_type == TransferType.DISK2H or transfer_op.transfer_type == TransferType.REMOTE2H:
            cpu_block_ids = transfer_op.dst_block_ids
            ssd_block_ids = transfer_op.src_block_ids
            transfer_op.transfer_type = TransferType.DISK2H
        else:
            cpu_block_ids = transfer_op.src_block_ids
            ssd_block_ids = transfer_op.dst_block_ids
            transfer_op.transfer_type = TransferType.H2DISK
        start_time = time.time()
        self._transfer_impl(
            cpu_block_ids,
            ssd_block_ids,
            transfer_op.transfer_type,
            layer_id=transfer_op.layer_id,
            layer_granularity=transfer_op.layer_granularity,
            non_blocking=True,
        )
        end_time = time.time()
        transfer_size = (
            len(ssd_block_ids)
            * self.block_size
            * self.dtype.itemsize
            * 2
            * self.num_layers
        )
        debuginfo.info(
            f"ssd tranfer request: {transfer_op.transfer_op_id} finished "
            f"request type is {transfer_op.transfer_type} "
            f"transfer data size is {transfer_size} bytes "
            f"transfer time is {end_time - start_time:.4f} s "
            f"transfer bandwidth is "
            f"{transfer_size / (end_time - start_time) / 1e9:.2f} GB/s"
        )
