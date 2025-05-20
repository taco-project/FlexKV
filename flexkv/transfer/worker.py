from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
import torch

import threading
from flexkv.common.transfer import TransferOp, TransferType, DeviceType, TransferDescriptor
from flexkv.common.storage import AccessibleHandle, AccessHandleType
from flexkv.c_ext import transfer_kv_layers, transfer_kv_blocks_ssd
import time
from threading import Thread

from flexkv.common.debug import debuginfo
import torch.cuda.nvtx as nvtx
import multiprocessing as mp
from multiprocessing import Process, Queue
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

class TransferWorker(ABC):
    def __init__(self, worker_id: int, finished_ops_queue: Queue):
        self.worker_id = worker_id
        self.transfer_queue = Queue()
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
    def _transfer_impl(self, transfer: TransferOp)->None:
        pass

    @abstractmethod
    def launch_transfer(self, transfers: List[TransferOp])->None:
        pass

class WorkerHandle:
    """handle for worker process"""
    def __init__(self, transfer_queue: mp.Queue, process: mp.Process):
        self.transfer_queue = transfer_queue
        self.process = process
        
    def submit_transfer(self, transfer: TransferOp):
        self.transfer_queue.put(transfer)
        
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
                     finished_ops_queue: mp.Queue,
                     gpu_kv_shape: Tuple[int, ...],
                     cpu_kv_shape: Tuple[int, ...],
                     dtype: torch.dtype,
                     gpu_device_id: int = -1):
        transfer_queue = mp.Queue()
        
        process = mp.Process(
            target=cls._worker_process,
            args=(worker_id, gpu_blocks, cpu_blocks, 
                  transfer_queue, finished_ops_queue,
                  gpu_kv_shape, cpu_kv_shape, dtype, gpu_device_id),
            daemon=True
        )
        process.start()
        
        return WorkerHandle(transfer_queue, process)

    @classmethod
    def _worker_process(cls,
                       worker_id: int,
                       gpu_blocks: List[torch.Tensor],
                       cpu_blocks: List[torch.Tensor],
                       transfer_queue: mp.Queue,
                       finished_ops_queue: mp.Queue,
                       gpu_kv_shape: Tuple[int, ...],
                       cpu_kv_shape: Tuple[int, ...],
                       dtype: torch.dtype,
                       gpu_device_id: int):
        # create worker in a new process
        worker = cls(worker_id, gpu_blocks, cpu_blocks,
                    transfer_queue, finished_ops_queue,
                    gpu_kv_shape, cpu_kv_shape, dtype, gpu_device_id)
        worker.run()

    def __init__(self,
                 worker_id: int,
                 gpu_blocks: List[torch.Tensor],
                 cpu_blocks: List[torch.Tensor], 
                 transfer_queue: mp.Queue,
                 finished_ops_queue: mp.Queue,
                 gpu_kv_shape: Tuple[int, ...],
                 cpu_kv_shape: Tuple[int, ...],
                 dtype: torch.dtype,
                 gpu_device_id: int = -1):
        # initialize worker in a new process
        self.worker_id = worker_id

        # Register CPU tensors with CUDA
        for cpu_block in cpu_blocks:
            cudaHostRegister(cpu_block)

        # Get pointers first
        self.gpu_blocks_ptrs = self._get_layer_ptrs(gpu_blocks)
        self.cpu_blocks_ptrs = self._get_layer_ptrs(cpu_blocks)

        self.gpu_blocks = gpu_blocks
        self.cpu_blocks = cpu_blocks

        self.transfer_queue = transfer_queue
        self.finished_ops_queue = finished_ops_queue
        
        self.num_layers = len(self.gpu_blocks_ptrs)
        self.num_gpu_blocks = gpu_kv_shape[2]
        self.num_cpu_blocks = cpu_kv_shape[2]
        self.block_size = cpu_kv_shape[3]
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

    def run(self):
        """main loop for worker process"""
        
        while True:
            try:
                op = self.transfer_queue.get(timeout=0.001)
                if op is None:
                    for cpu_block in self.cpu_blocks:
                        cudaHostUnregister(cpu_block)
                    del self.cpu_blocks
                    del self.gpu_blocks
                    torch.cuda.synchronize()
                    break
                try:
                    self.launch_transfer(op)
                except Exception as e:
                    debuginfo.error(f"Error launching transfer: {e}\n"
                                    f"Failed transfer op: {op}")
                self.finished_ops_queue.put(op)
                
            except Empty:
                continue

    def _transfer_impl(
        self,
        gpu_descriptor: TransferDescriptor,
        cpu_descriptor: TransferDescriptor,
        transfer_type: TransferType,
        layer_id: int = -1,
        layer_granularity: int = -1,
        non_blocking: bool = False,
        use_ce_transfer: bool = False,
    ) -> None:
        assert gpu_descriptor.device_type == DeviceType.GPU
        assert cpu_descriptor.device_type == DeviceType.CPU

        if layer_id == -1:
            layer_id = 0
        if layer_granularity == -1:
            layer_granularity = self.num_layers

        gpu_block_id_list = gpu_descriptor.physical_block_ids.pin_memory().to(dtype=torch.int64)
        cpu_block_id_list = cpu_descriptor.physical_block_ids.pin_memory().to(dtype=torch.int64)

        assert len(gpu_block_id_list) == len(cpu_block_id_list)

        if len(gpu_block_id_list) == 0:
            return

        layer_id_list = torch.arange(layer_id, layer_id + layer_granularity, dtype=torch.int32)

        chunk_size_in_bytes = self.block_size * self.dtype.itemsize
        nvtx.range_push("Transfer KV Layers")
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
        nvtx.range_pop()
        # if non_blocking:
        #torch.cuda.synchronize()  # TODO: remove this ?

    def launch_transfer(self, transfer_op: TransferOp):
        with torch.cuda.stream(self.transfer_stream):
            if transfer_op.transfer_type == TransferType.H2D:
                cpu_descriptor = transfer_op.src_descriptor
                gpu_descriptor = transfer_op.dst_descriptor
            elif transfer_op.transfer_type == TransferType.D2H:
                cpu_descriptor = transfer_op.dst_descriptor
                gpu_descriptor = transfer_op.src_descriptor
            else:
                raise ValueError(f"Invalid transfer type: {transfer_op.transfer_type}")
            start_time = time.time()
            nvtx.range_push("GPU CPUTransfer")
            self._transfer_impl(
                gpu_descriptor,
                cpu_descriptor,
                transfer_op.transfer_type,
                layer_id=transfer_op.layer_id,
                layer_granularity=transfer_op.layer_granularity,
                non_blocking=True,
                use_ce_transfer=False,
            )
            nvtx.range_pop()
            # debuginfo.info("TRANSFER stream synchronized")
            end_time = time.time()
            transfer_size = (
                len(gpu_descriptor.physical_block_ids)
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
                     ssd_file: str,
                     finished_ops_queue: mp.Queue,
                     cpu_kv_shape: Tuple[int, ...],
                     ssd_kv_shape: Tuple[int, ...],
                     dtype: torch.dtype):
        transfer_queue = mp.Queue()
        
        #NOTE cpublocks in cpussd worker has not to be pinned
        process = mp.Process(
            target=cls._worker_process,
            args=(worker_id, cpu_blocks, ssd_file,
                  transfer_queue, finished_ops_queue,
                  cpu_kv_shape, ssd_kv_shape, dtype),
            daemon=True
        )
        process.start()
        
        return WorkerHandle(transfer_queue, process)

    @classmethod
    def _worker_process(cls,
                       worker_id: int,
                       cpu_blocks: List[torch.Tensor],
                       ssd_file: str,
                       transfer_queue: mp.Queue,
                       finished_ops_queue: mp.Queue,
                       cpu_kv_shape: Tuple[int, ...],
                       ssd_kv_shape: Tuple[int, ...],
                       dtype: torch.dtype):
        worker = cls(worker_id, cpu_blocks, ssd_file,
                    transfer_queue, finished_ops_queue,
                    cpu_kv_shape, ssd_kv_shape, dtype)
        worker.run()

    def __init__(self,
                 worker_id: int,
                 cpu_blocks: List[torch.Tensor],
                 ssd_file: str,
                 transfer_queue: mp.Queue,
                 finished_ops_queue: mp.Queue,
                 cpu_kv_shape: Tuple[int, ...],
                 ssd_kv_shape: Tuple[int, ...],
                 dtype: torch.dtype):
        self.worker_id = worker_id
        print(f"IN SSD WORKER: the ssd file is {ssd_file}")
        self.ssd_file = ssd_file
        self.transfer_queue = transfer_queue
        self.finished_ops_queue = finished_ops_queue
        
        self.num_layers = cpu_kv_shape[0]
        self.num_cpu_blocks = cpu_kv_shape[2]
        self.num_ssd_blocks = ssd_kv_shape[2]

        self.block_size = cpu_kv_shape[3]
        self.dtype = dtype

        self.cpu_blocks = cpu_blocks
        
        self.cpu_layer_ptrs = self._get_layer_ptrs(cpu_blocks)
        
        self.cpu_layer_stride_in_bytes = (
            self.num_cpu_blocks * self.block_size * self.dtype.itemsize * 2
        )
        self.ssd_layer_stride_in_bytes = (
            self.num_ssd_blocks * self.block_size * self.dtype.itemsize * 2
        )
        self.cpu_kv_stride_in_bytes = (
            self.num_cpu_blocks * self.block_size * self.dtype.itemsize
        )
        self.ssd_kv_stride_in_bytes = (
            self.num_ssd_blocks * self.block_size * self.dtype.itemsize
        )
        self.ssd_block_stride_in_bytes = self.block_size * self.dtype.itemsize
        self.cpu_block_stride_in_bytes = self.block_size * self.dtype.itemsize

        self.chunk_size_in_bytes = self.block_size * self.dtype.itemsize

        #self.transfer_thread = Thread(target=self._transfer_worker, daemon=True)
        #self.transfer_thread.start()

    def run(self):
        """main loop for worker process"""
        while True:
            try:
                op = self.transfer_queue.get(timeout=0.001)
                if op is None:
                    del self.cpu_blocks
                    break
                try:
                    self.launch_transfer(op)
                except Exception as e:
                    debuginfo.error(f"Error launching transfer: {e}\n"
                                  f"Failed transfer op: {op}")
                self.finished_ops_queue.put(op)
                
            except Empty:
                continue

    def _transfer_impl(
        self,
        cpu_descriptor: TransferDescriptor,
        ssd_descriptor: TransferDescriptor,
        transfer_type: TransferType,
        layer_id: int = -1,
        layer_granularity: int = -1,
        non_blocking: bool = False,
    ) -> None:
        debuginfo.info(f"ssd transfer {transfer_type} happens")
        assert ssd_descriptor.device_type == DeviceType.SSD
        assert cpu_descriptor.device_type == DeviceType.CPU

        if layer_id == -1:
            layer_id = 0
        if layer_granularity == -1:
            layer_granularity = self.num_layers

        # this means partial read hit cpu and other hit ssd
        # or partial write hit ssd and none hit cpu
        debuginfo.info(f"ssd transfer {transfer_type} happens")
        ssd_block_id_list = ssd_descriptor.physical_block_ids.pin_memory().to(dtype=torch.int64)
        cpu_block_id_list = cpu_descriptor.physical_block_ids.pin_memory().to(dtype=torch.int64)
        if len(ssd_block_id_list) != len(cpu_block_id_list):
            assert len(ssd_block_id_list) < len(cpu_block_id_list)
            cpu_block_id_list = cpu_descriptor.physical_block_ids[
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
            filename=self.ssd_file,
            cpu_layer_id_list=layer_id_list,
            cpu_layer_ptrs_tensor=self.cpu_layer_ptrs,
            ssd_block_ids=ssd_block_id_list,
            cpu_block_ids=cpu_block_id_list,
            cpu_kv_stride_in_bytes=self.cpu_kv_stride_in_bytes,
            ssd_layer_stride_in_bytes=self.ssd_layer_stride_in_bytes,
            ssd_block_stride_in_bytes=self.ssd_block_stride_in_bytes,
            ssd_kv_stride_in_bytes=self.ssd_kv_stride_in_bytes,
            block_size_in_bytes=self.chunk_size_in_bytes,
            total_layers=self.num_layers,
            is_read=(transfer_type == TransferType.DISK2H),
            verbose=False
        )

    def launch_transfer(self, transfer_op: TransferOp):
        if transfer_op.transfer_type == TransferType.DISK2H:
            cpu_descriptor = transfer_op.dst_descriptor
            ssd_descriptor = transfer_op.src_descriptor
        else:
            cpu_descriptor = transfer_op.src_descriptor
            ssd_descriptor = transfer_op.dst_descriptor
        start_time = time.time()
        nvtx.range_push("CPU SSD Transfer")
        self._transfer_impl(
            cpu_descriptor,
            ssd_descriptor,
            transfer_op.transfer_type,
            layer_id=transfer_op.layer_id,
            layer_granularity=transfer_op.layer_granularity,
            non_blocking=True,
        )
        nvtx.range_pop()
        end_time = time.time()
        transfer_size = (
            len(ssd_descriptor.physical_block_ids)
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
