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
from flexkv.common.debug import debuginfo
from flexkv.common.debug import init_logger
from flexkv.common.memory_handle import KVCacheTensorHandle, import_layer_tensor_handle
from flexkv.common.storage import KVCacheLayout
from flexkv.common.transfer import TransferOp, TransferType, DeviceType, TransferDescriptor
from flexkv.common.transfer import get_nvtx_default_color, get_nvtx_range_color

try:
    from flexkv.c_ext import transfer_kv_blocks_remote
except ImportError:
    transfer_kv_blocks_remote = None


logger = init_logger(__name__)

cudart = ctypes.CDLL('libcudart.so')

def cudaHostRegister(tensor: torch.Tensor):
    """Register a CPU tensor with CUDA for pinned memory access"""
    ptr = tensor.data_ptr()
    size = tensor.numel() * tensor.element_size()
    ret = cudart.cudaHostRegister(ctypes.c_void_p(ptr), ctypes.c_size_t(size), 1) # 1 means cudaHostRegisterPortable
    if ret != 0:
        raise RuntimeError(f"cudaHostRegister failed with error code {ret}")

def cudaHostUnregister(tensor: torch.Tensor):
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
                     gpu_device_id: int = -1,
                     zmq_gpu_handle: list[KVCacheTensorHandle] = None):
        transfer_queue = mp.Queue()
        ready_event = mp.Event()

        process = mp.Process(
            target=cls._worker_process,
            args=(worker_id, gpu_blocks, cpu_blocks,
                  transfer_queue, finished_ops_queue,
                  gpu_kv_layout, cpu_kv_layout, dtype, 
                  gpu_device_id, ready_event, zmq_gpu_handle),
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
                       ready_event: Any,
                       zmq_gpu_handle: list[KVCacheTensorHandle]):
        # create worker in a new process
        if zmq_gpu_handle is not None:
            gpu_blocks = []
            for layer_handle in zmq_gpu_handle:
                tensor, layer_id = import_layer_tensor_handle(layer_handle)
                gpu_blocks.append(tensor)
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

class tpGPUCPUTransferWorker(GPUCPUTransferWorker):
    @classmethod
    def create_worker(cls,
                     worker_id: int,
                     gpu_blocks: Union[List[List[torch.Tensor]], List[List[KVCacheTensorHandle]]],
                     cpu_blocks: List[torch.Tensor],
                     finished_ops_queue: MPQueue,
                     gpu_kv_layout: KVCacheLayout,
                     cpu_kv_layout: KVCacheLayout,
                     dtype: torch.dtype,
                     tp_group_size: int,
                     dp_group_id: int):
        transfer_queue = mp.Queue()
        ready_event = mp.Event()

        process = mp.Process(
            target=cls._worker_process,
            args=(worker_id, gpu_blocks, cpu_blocks,
                  transfer_queue, finished_ops_queue,
                  gpu_kv_layout, cpu_kv_layout, dtype, 
                  tp_group_size, dp_group_id, ready_event),
            daemon=True
        )
        process.start()

        return WorkerHandle(worker_id, transfer_queue, process, ready_event)
    
    @classmethod
    def _worker_process(cls,
                        worker_id: int,
                        gpu_blocks: Union[List[List[torch.Tensor]], List[List[KVCacheTensorHandle]]],
                        cpu_blocks: List[torch.Tensor],
                        transfer_queue: MPQueue,
                        finished_ops_queue: MPQueue,
                        gpu_kv_layout: KVCacheLayout,
                        cpu_kv_layout: KVCacheLayout,
                        dtype: torch.dtype,
                        tp_group_size: int,
                        dp_group_id: int,
                        ready_event: Any):
        assert len(gpu_blocks) == tp_group_size
        # handles exported from other processes
        if isinstance(gpu_blocks[0][0], KVCacheTensorHandle):
            imported_gpu_blocks = []
            for handles_in_one_gpu in gpu_blocks:
                blocks_in_one_gpu = []
                for handle in handles_in_one_gpu:
                    tensor, layer_id = import_layer_tensor_handle(handle)
                    blocks_in_one_gpu.append(tensor)
                imported_gpu_blocks.append(blocks_in_one_gpu)
            gpu_blocks = imported_gpu_blocks
        else: # raw gpu blocks
            imported_gpu_blocks = gpu_blocks

        worker = cls(worker_id, imported_gpu_blocks, cpu_blocks,
                    transfer_queue, finished_ops_queue,
                    gpu_kv_layout, cpu_kv_layout, dtype,
                    tp_group_size, dp_group_id)
        ready_event.set()
        worker.run()

    def __init__(self,
                 worker_id: int,
                 gpu_blocks: List[List[torch.Tensor]],
                 cpu_blocks: List[torch.Tensor],
                 transfer_queue: MPQueue,
                 finished_ops_queue: MPQueue,
                 gpu_kv_layout: KVCacheLayout,
                 cpu_kv_layout: KVCacheLayout,
                 dtype: torch.dtype, 
                 tp_group_size: int, 
                 dp_group_id: int):

        self.worker_id = worker_id
        self.transfer_queue = MPQueue()
        self.finished_ops_queue = finished_ops_queue
        self.dtype = dtype

        self.num_gpus = len(gpu_blocks)
        self.tp_group_size = tp_group_size
        self.dp_group_id = dp_group_id

        for cpu_block in cpu_blocks:
            cudaHostRegister(cpu_block)

        self.transfer_queue = transfer_queue

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

        self.tp_transfer_thread_group = TPTransferThreadGroup(self.num_gpus, gpu_blocks, cpu_blocks,dp_group_id)

    def _transfer_impl(self,
                       gpu_block_ids: np.ndarray,
                       cpu_block_ids: np.ndarray,
                       transfer_type: TransferType,
                       layer_id: int = -1,
                       layer_granularity: int = -1,
                       non_blocking: bool = False,
                       use_ce_transfer: bool = False,
                       )->None:
        if layer_id == -1:
            layer_id = 0
        if layer_granularity == -1:
            layer_granularity = self.num_layers
            
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
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type}")
        
    def launch_transfer(self, transfer_op: WorkerTransferOp):
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
            len(cpu_block_ids)
            * self.cpu_block_size
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
        if transfer_op.transfer_type == TransferType.DISK2H:
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


class CPURemoteTransferWorker(TransferWorker):
    @classmethod
    def create_worker(cls,
                     worker_id: int,
                     cpu_blocks: List[torch.Tensor],
                     remote_file: List[str],
                     finished_ops_queue: MPQueue,
                     cpu_kv_layout: KVCacheLayout,
                     remote_kv_layout: KVCacheLayout,
                     dtype: torch.dtype,
                     remote_config_custom: Dict[str, Any]):
        transfer_queue = mp.Queue()
        ready_event = mp.Event()

        process = mp.Process(
            target=cls._worker_process,
            args=(worker_id, cpu_blocks, remote_file,
                  transfer_queue, finished_ops_queue,
                  cpu_kv_layout, remote_kv_layout, dtype, ready_event, remote_config_custom),
            daemon=True
        )
        process.start()

        return WorkerHandle(worker_id, transfer_queue, process, ready_event)

    @classmethod
    def _worker_process(cls,
                       worker_id: int,
                       cpu_blocks: List[torch.Tensor],
                       remote_file: List[str],
                       transfer_queue: MPQueue,
                       finished_ops_queue: MPQueue,
                       cpu_kv_layout: KVCacheLayout,
                       remote_kv_layout: KVCacheLayout,
                       dtype: torch.dtype,
                       ready_event: Any, 
                       remote_config_custom:  Dict[str, Any]):
        worker = cls(worker_id, cpu_blocks, remote_file,
                    transfer_queue, finished_ops_queue,
                    cpu_kv_layout, remote_kv_layout, dtype, remote_config_custom)
        ready_event.set()
        worker.run()

    def __init__(self,
                 worker_id: int,
                 cpu_blocks: List[torch.Tensor],
                 remote_file: List[str],
                 transfer_queue: MPQueue,
                 finished_ops_queue: MPQueue,
                 cpu_kv_layout: KVCacheLayout,
                 remote_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 remote_config_custom: Dict[str, Any]):
        if transfer_kv_blocks_remote is None:
            raise RuntimeError("transfer_kv_blocks_remote not available, please build with FLEXKV_ENABLE_CFS=1")
        super().__init__(worker_id, finished_ops_queue)
        self.remote_files = remote_file
        self.num_remote_files = len(remote_file)
        self.transfer_queue = transfer_queue

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
        cpu_block_ids: np.ndarray,
        remote_block_ids: np.ndarray,
        transfer_type: TransferType,
        layer_id: int = -1,
        layer_granularity: int = -1,
        non_blocking: bool = False,
    ) -> None:
        assert remote_block_ids.dtype == np.int64
        assert cpu_block_ids.dtype == np.int64

        if layer_id == -1:
            layer_id = 0
        if layer_granularity == -1:
            layer_granularity = self.num_layers

        # this means partial read hit cpu and other hit remote
        # or partial write hit remote and none hit cpu
        remote_block_id_list = torch.from_numpy(remote_block_ids).pin_memory().to(dtype=torch.int64)
        cpu_block_id_list = torch.from_numpy(cpu_block_ids).pin_memory().to(dtype=torch.int64)
        if len(remote_block_id_list) != len(cpu_block_id_list):
            assert len(remote_block_id_list) < len(cpu_block_id_list)
            cpu_block_id_list = cpu_block_ids[
                -len(remote_block_id_list) :
            ]
            debuginfo.debug(
                f"cpu block id num {len(cpu_block_id_list)} "
                f"remote block id num {len(remote_block_id_list)}"
            )

        if len(remote_block_id_list) == 0:
            return

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

    def launch_transfer(self, transfer_op: WorkerTransferOp):
        #TODO remove this when remote worker is ready
        if transfer_op.transfer_type == TransferType.REMOTE2H:
            cpu_block_ids = transfer_op.dst_block_ids
            remote_block_ids = transfer_op.src_block_ids
            transfer_op.transfer_type = TransferType.REMOTE2H
        else:
            cpu_block_ids = transfer_op.src_block_ids
            remote_block_ids = transfer_op.dst_block_ids
            transfer_op.transfer_type = TransferType.H2REMOTE
        start_time = time.time()
        self._transfer_impl(
            cpu_block_ids,
            remote_block_ids,
            transfer_op.transfer_type,
            layer_id=transfer_op.layer_id,
            layer_granularity=transfer_op.layer_granularity,
            non_blocking=True,
        )
        end_time = time.time()
        transfer_size = (
            len(remote_block_ids)
            * self.block_size
            * self.dtype.itemsize
            * 2
            * self.num_layers
        )
        debuginfo.info(
            f"remote tranfer request: {transfer_op.transfer_op_id} finished "
            f"request type is {transfer_op.transfer_type} "
            f"transfer data size is {transfer_size} bytes "
            f"transfer time is {end_time - start_time:.4f} s "
            f"transfer bandwidth is "
            f"{transfer_size / (end_time - start_time) / 1e9:.2f} GB/s"
        )