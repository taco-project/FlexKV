import copy
import torch.multiprocessing as mp
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch.multiprocessing import Queue as MPQueue, Pipe as MPPipe
from multiprocessing.connection import Connection
from threading import Thread
from typing import List, Any, Dict, Union, Optional

import ctypes
import numpy as np
import nvtx
import torch

from flexkv import c_ext

from flexkv.c_ext import transfer_kv_blocks, transfer_kv_blocks_ssd, TPTransferThreadGroup
from flexkv.common.debug import flexkv_logger
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import TransferOp, TransferType, PartitionBlockType
from flexkv.common.transfer import get_nvtx_range_color
from flexkv.common.config import CacheConfig

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
    layer_id: int
    layer_granularity: int
    slot_id: int
    valid_block_num: int
    src_block_ids: np.ndarray
    dst_block_ids: np.ndarray
    # successors: List[int]

    def __init__(self, transfer_op: TransferOp):
        self.transfer_op_id = transfer_op.op_id
        self.transfer_graph_id = transfer_op.graph_id
        self.transfer_type = transfer_op.transfer_type
        self.layer_id = transfer_op.layer_id
        self.layer_granularity = transfer_op.layer_granularity
        self.slot_id = transfer_op.slot_id
        self.valid_block_num = transfer_op.valid_block_num
        if self.slot_id == -1:
            self.src_block_ids = transfer_op.src_block_ids
            self.dst_block_ids = transfer_op.dst_block_ids
        else:
            self.src_block_ids = np.empty(0)
            self.dst_block_ids = np.empty(0)
        # self.successors = list(transfer_op.successors)  # for nvtx

class TransferWorkerBase(ABC):
    _worker_id_counter = 0
    _worker_id_lock = threading.Lock()

    def __init__(self,
                 worker_id: int,
                 transfer_conn: Connection,  # receive end of pipe
                 finished_ops_queue: MPQueue,
                 src_buffer_tensor: torch.Tensor,
                 dst_buffer_tensor: torch.Tensor):
        self.worker_id = worker_id
        self.transfer_conn = transfer_conn  # receive end of pipe
        self.finished_ops_queue: MPQueue[int] = finished_ops_queue

        flexkv_logger.info(f"[TransferWorkerBase] src data ptr: {src_buffer_tensor.storage().data_ptr()}")
        flexkv_logger.info(f"[TransferWorkerBase] dst data ptr: {dst_buffer_tensor.storage().data_ptr()}")
        self.src_shared_tensor = src_buffer_tensor
        self.dst_shared_tensor = dst_buffer_tensor
        cudaHostRegister(self.src_shared_tensor)
        cudaHostRegister(self.dst_shared_tensor)

    @classmethod
    def _get_worker_id(cls) -> int:
        with cls._worker_id_lock:
            worker_id = cls._worker_id_counter
            cls._worker_id_counter += 1
            return worker_id

    def _get_layer_ptrs(self, layer_blocks: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        if isinstance(layer_blocks, torch.Tensor):
            layer_blocks = [layer_blocks]
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
    def create_worker(cls, finished_ops_queue: MPQueue, src_buffer_tensor: torch.Tensor,
                      dst_buffer_tensor: torch.Tensor, *args: Any, **kwargs: Any) -> 'WorkerHandle':
        """Generic worker creation template method"""
        parent_conn, child_conn = MPPipe()  # create pipe
        ready_event = mp.Event()
        worker_id = cls._get_worker_id()

        process = mp.Process(
            target=cls._worker_process,
            args=(worker_id, child_conn, finished_ops_queue, src_buffer_tensor, 
                  dst_buffer_tensor, ready_event, *args),            
            kwargs=kwargs,
            daemon=True
        )
        process.start()

        return WorkerHandle(worker_id, parent_conn, process, ready_event)

    @classmethod
    def _worker_process(cls, worker_id: int, transfer_conn: Connection, finished_ops_queue: MPQueue,
                        src_buffer_tensor: torch.Tensor, dst_buffer_tensor: torch.Tensor,
                       ready_event: Any, *args: Any, **kwargs: Any) -> None:
        worker = cls(worker_id, transfer_conn, finished_ops_queue, src_buffer_tensor, 
                     dst_buffer_tensor, *args, **kwargs)
        ready_event.set()
        worker.run()

    @abstractmethod
    def _transfer_impl(
        self,
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
        transfer_type: TransferType,
        layer_id: int,
        layer_granularity: int,
        **kwargs: Any
    ) -> None:
        pass

    
    def get_transfer_block_ids(self, transfer_op: WorkerTransferOp) ->tuple[torch.Tensor, torch.Tensor]:
        slot_id = transfer_op.slot_id
        valid_block_num = transfer_op.valid_block_num
        if slot_id == -1:
            assert len(transfer_op.src_block_ids) == valid_block_num and len(transfer_op.dst_block_ids) == valid_block_num
            src_block_ids = torch.from_numpy(transfer_op.src_block_ids).to(dtype=torch.int64).pin_memory()
            dst_block_ids = torch.from_numpy(transfer_op.src_block_ids).to(dtype=torch.int64).pin_memory()
        else:
            src_block_ids = self.src_shared_tensor[slot_id, :valid_block_num]
            dst_block_ids = self.dst_shared_tensor[slot_id, :valid_block_num]

        return src_block_ids, dst_block_ids

    def _log_transfer_performance(self,
                                  transfer_op: WorkerTransferOp,
                                  transfer_size: int,
                                  start_time: float,
                                  end_time: float) -> None:
        """Common method to log transfer performance"""
        flexkv_logger.info(
            f"{transfer_op.transfer_type.name} transfer request: {transfer_op.transfer_op_id} finished "
            f"transfer data size: {transfer_size / (1024 * 1024 * 1024)} GB "
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
                if self.transfer_conn.poll(timeout=0.0001):  # check if data available
                    op = self.transfer_conn.recv()
                    if op is None:
                        break
                    batch_ops = [op]
                    while self.transfer_conn.poll(timeout=0):
                        op = self.transfer_conn.recv()
                        if op is None:
                            break
                        batch_ops.append(op)
                    for op in batch_ops:
                        try:
                            nvtx.push_range(f"launch {op.transfer_type.name} op_id: {op.transfer_op_id}, "
                                                f"graph_id: {op.transfer_graph_id}, "
                                                f"num_blocks: {op.valid_block_num}",
                                                color=get_nvtx_range_color(op.transfer_graph_id))
                            self.launch_transfer(op)
                            nvtx.pop_range()
                        except Exception as e:
                            flexkv_logger.error(f"Error launching transfer: {e}\n"
                                        f"Failed transfer op: {op}")
                        self.finished_ops_queue.put(op.transfer_op_id)
                else:
                    continue
            except EOFError:
                # Connection closed
                break
            except Exception as e:
                flexkv_logger.error(f"Error in worker run loop: {e}")
                continue

class WorkerHandle:
    """handle for worker process"""
    def __init__(self, worker_id: int, transfer_conn: Connection, process: mp.Process, ready_event: Any):
        self.worker_id = worker_id
        self.transfer_conn = transfer_conn
        self.process = process
        self.ready_event = ready_event

    def submit_transfer(self, op: TransferOp) -> None:
        self.transfer_conn.send(WorkerTransferOp(op))

    def shutdown(self) -> None:
        try:
            self.transfer_conn.send(None)
            self.transfer_conn.close()
        except (BrokenPipeError, OSError):
            pass  # Pipe already closed
        # set timeout to 5 seconds
        self.process.join(timeout=5)
        if self.process.is_alive():
            print("force terminate the worker process")
            self.process.terminate()
            self.process.join()

    def __del__(self) -> None:
        if self.process.is_alive():
            self.shutdown()

class GPUCPUTransferWorker(TransferWorkerBase):  # this worker only supports non-tp and non-dp case
    def __init__(self,
                 worker_id: int,
                 transfer_conn: Connection,
                 finished_ops_queue: MPQueue,
                 src_buffer_tensor: torch.Tensor,
                 dst_buffer_tensor: torch.Tensor,
                 gpu_blocks: List[TensorSharedHandle],
                 cpu_blocks: torch.Tensor,
                 gpu_kv_layout: KVCacheLayout,
                 cpu_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 gpu_device_id: int,
                 use_ce_transfer_h2d: bool = False,
                 use_ce_transfer_d2h: bool = False,
                 transfer_sms_h2d: int = 8,
                 transfer_sms_d2h: int = 8) -> None:
        # initialize worker in a new process
        super().__init__(worker_id, transfer_conn, finished_ops_queue, src_buffer_tensor, dst_buffer_tensor)
        # Register CPU tensors with CUDA
        cudaHostRegister(cpu_blocks)
        self.gpu_blocks = [wrapper.get_tensor() for wrapper in gpu_blocks]
        # Get pointers first
        self.gpu_blocks_ptrs = self._get_layer_ptrs(self.gpu_blocks)
        self.gpu_layer_ptrs = self.gpu_blocks_ptrs

        self.cpu_tensor = cpu_blocks

        self.dtype = dtype
        self.is_mla = gpu_kv_layout.is_mla

        self.num_layers = gpu_kv_layout.num_layer
        gpu_kv_layout_per_layer = gpu_kv_layout.div_layer(self.num_layers)

        # a chunk can be located by layer_id * layer_stride + kv_id * kv_stride + block_id * block_stride
        self.chunk_size_in_bytes = gpu_kv_layout_per_layer.get_chunk_size() * self.dtype.itemsize
        self.gpu_kv_stride_in_bytes = gpu_kv_layout_per_layer.get_kv_stride() * self.dtype.itemsize
        self.gpu_block_stride_in_bytes = gpu_kv_layout_per_layer.get_block_stride() * self.dtype.itemsize

        self.cpu_layer_stride_in_bytes = cpu_kv_layout.get_layer_stride() * self.dtype.itemsize
        self.cpu_kv_stride_in_bytes = cpu_kv_layout.get_kv_stride() * self.dtype.itemsize
        self.cpu_block_stride_in_bytes = cpu_kv_layout.get_block_stride() * self.dtype.itemsize

        if not gpu_kv_layout.type == KVCacheLayoutType.LAYERWISE:
            raise ValueError("Only layerwise layout is supported for GPU")
        # set GPU device
        if gpu_device_id != -1:
            torch.cuda.set_device(gpu_device_id)
        self.transfer_stream = torch.cuda.Stream()
        self.transfer_sms_h2d = transfer_sms_h2d
        self.transfer_sms_d2h = transfer_sms_d2h
        self.use_ce_transfer_h2d = use_ce_transfer_h2d
        self.use_ce_transfer_d2h = use_ce_transfer_d2h

    def _transfer_impl(
        self,
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
        transfer_type: TransferType,
        layer_id: int,
        layer_granularity: int,
        **kwargs: Any,
    ) -> None:
        assert src_block_ids.dtype == torch.int64
        assert dst_block_ids.dtype == torch.int64
        assert len(src_block_ids) == len(dst_block_ids)

        if transfer_type == TransferType.H2D:
            gpu_block_id_list = dst_block_ids
            cpu_block_id_list = src_block_ids
            use_ce_transfer = self.use_ce_transfer_h2d
            transfer_sms = self.transfer_sms_h2d
        elif transfer_type == TransferType.D2H:
            gpu_block_id_list = src_block_ids
            cpu_block_id_list = dst_block_ids
            use_ce_transfer = self.use_ce_transfer_d2h
            transfer_sms = self.transfer_sms_d2h
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type} for GPUCPUTransferWorker")

        assert len(gpu_block_id_list) == len(cpu_block_id_list)

        if len(gpu_block_id_list) == 0:
            return

        layer_id_list = torch.arange(layer_id, layer_id + layer_granularity, dtype=torch.int32)

        gpu_layer_ptrs = self.gpu_layer_ptrs[layer_id_list].contiguous().pin_memory()

        transfer_kv_blocks(
            gpu_block_id_list,
            gpu_layer_ptrs,
            self.gpu_kv_stride_in_bytes,
            self.gpu_block_stride_in_bytes,
            cpu_block_id_list,
            self.cpu_tensor,
            self.cpu_kv_stride_in_bytes,
            self.cpu_layer_stride_in_bytes,
            self.cpu_block_stride_in_bytes,
            self.chunk_size_in_bytes,
            layer_id,
            transfer_sms,
            transfer_type == TransferType.H2D,
            use_ce_transfer,
            self.is_mla,
        )

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> None:
        layer_id = transfer_op.layer_id
        layer_granularity = transfer_op.layer_granularity
        if layer_id == -1:
            layer_id = 0
        if layer_granularity == -1:
            layer_granularity = self.num_layers

        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op)

        with torch.cuda.stream(self.transfer_stream):
            start_time = time.time()
            self._transfer_impl(
                src_block_ids,
                dst_block_ids,
                transfer_op.transfer_type,
                layer_id,
                layer_granularity,
            )
            end_time = time.time()

            kv_dim = 2 if not self.is_mla else 1
            transfer_size = self.chunk_size_in_bytes * layer_granularity * transfer_op.valid_block_num * kv_dim

            self._log_transfer_performance(
                transfer_op,
                transfer_size,
                start_time,
                end_time,
            )

class tpGPUCPUTransferWorker(TransferWorkerBase):
    def __init__(self,
                 worker_id: int,
                 transfer_conn: Connection,
                 finished_ops_queue: MPQueue,
                 src_buffer_tensor: torch.Tensor,
                 dst_buffer_tensor: torch.Tensor,
                 gpu_blocks: List[List[TensorSharedHandle]],
                 cpu_blocks: torch.Tensor,
                 gpu_kv_layout: KVCacheLayout,
                 cpu_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 tp_group_size: int,
                 dp_group_id: int,
                 use_ce_transfer_h2d: bool = False,
                 use_ce_transfer_d2h: bool = False,
                 transfer_sms_h2d: int = 8,
                 transfer_sms_d2h: int = 8):

        super().__init__(worker_id, transfer_conn, finished_ops_queue, src_buffer_tensor, dst_buffer_tensor)
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
        self.is_mla = gpu_kv_layout.is_mla

        self.num_gpus = len(self.gpu_blocks)
        self.tp_group_size = tp_group_size
        self.dp_group_id = dp_group_id

        cudaHostRegister(cpu_blocks)

        self.num_layers = gpu_kv_layout.num_layer
        gpu_kv_layout_per_layer = gpu_kv_layout.div_layer(self.num_layers)

        self.gpu_chunk_size_in_bytes = gpu_kv_layout_per_layer.get_chunk_size() * self.dtype.itemsize
        self.gpu_kv_stride_in_bytes = gpu_kv_layout_per_layer.get_kv_stride() * self.dtype.itemsize
        self.gpu_block_stride_in_bytes = gpu_kv_layout_per_layer.get_block_stride() * self.dtype.itemsize

        self.cpu_chunk_size_in_bytes = cpu_kv_layout.get_chunk_size() * self.dtype.itemsize
        self.cpu_layer_stride_in_bytes = cpu_kv_layout.get_layer_stride() * self.dtype.itemsize
        self.cpu_kv_stride_in_bytes = cpu_kv_layout.get_kv_stride() * self.dtype.itemsize
        self.cpu_block_stride_in_bytes = cpu_kv_layout.get_block_stride() * self.dtype.itemsize

        if not gpu_kv_layout.type == KVCacheLayoutType.LAYERWISE:
            raise ValueError("Only layerwise layout is supported for GPU")

        self.transfer_sms_h2d = transfer_sms_h2d
        self.transfer_sms_d2h = transfer_sms_d2h
        self.use_ce_transfer_h2d = use_ce_transfer_h2d
        self.use_ce_transfer_d2h = use_ce_transfer_d2h

        self.tp_transfer_thread_group = TPTransferThreadGroup(self.num_gpus, self.gpu_blocks, cpu_blocks, dp_group_id)


    def _transfer_impl(self,
                       src_block_ids: torch.Tensor,
                       dst_block_ids: torch.Tensor,
                       transfer_type: TransferType,
                       layer_id: int,
                       layer_granularity: int,
                       **kwargs: Any,
                       )->None:
        assert src_block_ids.dtype == torch.int64
        assert dst_block_ids.dtype == torch.int64
        assert len(src_block_ids) == len(dst_block_ids)

        if transfer_type == TransferType.H2D:
            gpu_block_id_list = dst_block_ids
            cpu_block_id_list = src_block_ids
            use_ce_transfer = self.use_ce_transfer_h2d
            transfer_sms = self.transfer_sms_h2d
        elif transfer_type == TransferType.D2H:
            gpu_block_id_list = src_block_ids
            cpu_block_id_list = dst_block_ids
            use_ce_transfer = self.use_ce_transfer_d2h
            transfer_sms = self.transfer_sms_d2h
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type} for tpGPUCPUTransferWorker")


        assert len(gpu_block_id_list) == len(cpu_block_id_list)

        if len(gpu_block_id_list) == 0:
            return

        self.tp_transfer_thread_group.tp_group_transfer(
            gpu_block_id_list,
            self.gpu_kv_stride_in_bytes,
            self.gpu_block_stride_in_bytes,
            self.gpu_chunk_size_in_bytes,
            cpu_block_id_list,
            self.cpu_kv_stride_in_bytes,
            self.cpu_layer_stride_in_bytes,
            self.cpu_block_stride_in_bytes,
            self.cpu_chunk_size_in_bytes,
            transfer_sms,
            transfer_type == TransferType.H2D,
            use_ce_transfer,
            layer_id,
            layer_granularity,
            self.is_mla,
        )


    def launch_transfer(self, transfer_op: WorkerTransferOp) -> None:
        layer_id = transfer_op.layer_id
        layer_granularity = transfer_op.layer_granularity
        if layer_id == -1:
            layer_id = 0
        if layer_granularity == -1:
            layer_granularity = self.num_layers

        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op)

        start_time = time.time()
        self._transfer_impl(
            src_block_ids,
            dst_block_ids,
            transfer_op.transfer_type,
            layer_id,
            layer_granularity,
        )
        end_time = time.time()

        kv_dim = 2 if not self.is_mla else 1
        transfer_size = self.cpu_chunk_size_in_bytes * layer_granularity * transfer_op.valid_block_num * kv_dim

        self._log_transfer_performance(
            transfer_op,
            transfer_size,
            start_time,
            end_time,
        )

class CPUSSDDiskTransferWorker(TransferWorkerBase):
    def __init__(self,
                 worker_id: int,
                 transfer_conn: Connection,
                 finished_ops_queue: MPQueue,
                 src_buffer_tensor: torch.Tensor,
                 dst_buffer_tensor: torch.Tensor,
                 cpu_blocks: torch.Tensor,
                 ssd_files: Dict[int, List[str]],  # ssd_device_id -> file_paths
                 cpu_kv_layout: KVCacheLayout,
                 ssd_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 num_blocks_per_file: int,
                 cache_config: CacheConfig):
        super().__init__(worker_id, transfer_conn, finished_ops_queue, src_buffer_tensor, dst_buffer_tensor)
        self.ssd_files = ssd_files
        self.num_blocks_per_file = num_blocks_per_file
        self.num_files = sum(len(file_list) for file_list in ssd_files.values())

        self.num_layers = cpu_kv_layout.num_layer
        self.num_cpu_blocks = cpu_kv_layout.num_block
        self.round_robin = 1

        self.dtype = dtype

        self.cpu_blocks = cpu_blocks
        self.cpu_layer_ptrs = self._get_layer_ptrs(cpu_blocks)

        self.is_mla = cpu_kv_layout.is_mla

        if cpu_kv_layout.type != ssd_kv_layout.type:
            raise ValueError("no support for different CPU and SSD KV cache layout type")

        ssd_kv_layout_per_file = ssd_kv_layout.div_block(self.num_files, padding=True)

        self.chunk_size_in_bytes = cpu_kv_layout.get_chunk_size() * self.dtype.itemsize
        self.block_stride_in_bytes = cpu_kv_layout.get_block_stride() * self.dtype.itemsize
        self.cpu_kv_stride_in_bytes = cpu_kv_layout.get_kv_stride() * self.dtype.itemsize
        self.cpu_layer_stride_in_bytes = cpu_kv_layout.get_layer_stride() * self.dtype.itemsize
        self.ssd_kv_stride_in_bytes = ssd_kv_layout_per_file.get_kv_stride() * self.dtype.itemsize
        self.ssd_layer_stride_in_bytes = ssd_kv_layout_per_file.get_layer_stride() * self.dtype.itemsize

        try:
            self.ioctx = c_ext.SSDIOCTX(ssd_files, len(ssd_files), cache_config.ssd_cache_iouring_entries,
                cache_config.ssd_cache_iouring_flags)
        except Exception as e:
            flexkv_logger.error(f"Error setting ssd ioctx: {e}\n")
            raise RuntimeError("SSD Worker init failed") from e

    def _transfer_impl(
        self,
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
        transfer_type: TransferType,
        layer_id: int,
        layer_granularity: int,
        **kwargs: Any,
    ) -> None:
        assert src_block_ids.dtype == torch.int64
        assert dst_block_ids.dtype == torch.int64
        assert len(src_block_ids) == len(dst_block_ids)

        if transfer_type == TransferType.H2DISK:
            ssd_block_id_list = dst_block_ids
            cpu_block_id_list = src_block_ids
        elif transfer_type == TransferType.DISK2H:
            ssd_block_id_list = src_block_ids
            cpu_block_id_list = dst_block_ids
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type} for CPUSSDDiskTransferWorker")


        layer_id_list = torch.arange(layer_id, layer_id + layer_granularity, dtype=torch.int32)

        transfer_kv_blocks_ssd(
            ioctx=self.ioctx,
            cpu_layer_id_list=layer_id_list,
            cpu_tensor_ptr=self.cpu_layer_ptrs[0].item(),
            ssd_block_ids=ssd_block_id_list,
            cpu_block_ids=cpu_block_id_list,
            cpu_layer_stride_in_bytes=self.cpu_layer_stride_in_bytes,
            cpu_kv_stride_in_bytes=self.cpu_kv_stride_in_bytes,
            ssd_layer_stride_in_bytes=self.ssd_layer_stride_in_bytes,
            ssd_kv_stride_in_bytes=self.ssd_kv_stride_in_bytes,
            chunk_size_in_bytes=self.chunk_size_in_bytes,
            block_stride_in_bytes=self.block_stride_in_bytes,
            is_read=(transfer_type == TransferType.DISK2H),
            num_blocks_per_file=self.num_blocks_per_file,
            round_robin=self.round_robin,
            num_threads_per_device=32,
            is_mla=self.is_mla,
        )

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> None:
        layer_id = transfer_op.layer_id
        layer_granularity = transfer_op.layer_granularity
        if layer_id == -1:
            layer_id = 0
        if layer_granularity == -1:
            layer_granularity = self.num_layers

        src_block_ids , dst_block_ids = self.get_transfer_block_ids(transfer_op)

        start_time = time.time()
        self._transfer_impl(
            src_block_ids,
            dst_block_ids,
            transfer_op.transfer_type,
            transfer_op.layer_id,
            transfer_op.layer_granularity,
        )
        end_time = time.time()

        kv_dim = 2 if not self.is_mla else 1
        transfer_size = self.chunk_size_in_bytes * layer_granularity * transfer_op.valid_block_num * kv_dim

        self._log_transfer_performance(
            transfer_op,
            transfer_size,
            start_time,
            end_time,
        )

class CPURemoteTransferWorker(TransferWorkerBase):
    def __init__(self,
                 worker_id: int,
                 transfer_conn: Connection,
                 finished_ops_queue: MPQueue,
                 src_buffer_tensor: torch.Tensor,
                 dst_buffer_tensor: torch.Tensor,
                 cpu_blocks: List[torch.Tensor],
                 remote_file: List[str],
                 cpu_kv_layout: KVCacheLayout,
                 remote_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 remote_config_custom: Dict[str, Any]):
        if transfer_kv_blocks_remote is None:
            raise RuntimeError("transfer_kv_blocks_remote not available, please build with FLEXKV_ENABLE_CFS=1")
        super().__init__(worker_id, transfer_conn, finished_ops_queue, src_buffer_tensor, dst_buffer_tensor)

        self.cpu_layer_ptrs = self._get_layer_ptrs(cpu_blocks)
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

        self.block_size = cpu_kv_layout.get_chunk_size()
        self.dtype = dtype

        self.is_mla = cpu_kv_layout.is_mla
        kv_dim = 2 if not self.is_mla else 1

        self.cpu_blocks = cpu_blocks

        self.cpu_layer_ptrs = self._get_layer_ptrs(cpu_blocks)

        self.cpu_layer_stride_in_bytes = (
            self.num_cpu_blocks * self.block_size * self.dtype.itemsize * kv_dim
        )
        self.remote_layer_stride_in_bytes = (
            self.num_remote_blocks * self.block_size * self.dtype.itemsize * kv_dim
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
        need_create = False
        for remote_file_single in remote_file:
            nodeid = self.pcfs.lookup_or_create_file(
            remote_file_single,
            (self.remote_layer_stride_in_bytes_per_file * self.num_layers), need_create)
            if nodeid == 0:
                raise RuntimeError(f"lookup or create file failed for file: {remote_file_single}")
            self.file_nodeid_list.append(nodeid)

        c_ext.set_pcfs_instance(self.pcfs)

    def _transfer_impl(
        self,
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
        transfer_type: TransferType,
        layer_id: int,
        layer_granularity: int,
        **kwargs: Any
    ) -> None:
        assert src_block_ids.dtype == torch.int64
        assert dst_block_ids.dtype == torch.int64
        assert len(src_block_ids) == len(dst_block_ids)

        if layer_id == -1:
            layer_id = 0
        if layer_granularity == -1:
            layer_granularity = self.num_layers

        # this means partial read hit cpu and other hit remote
        # or partial write hit remote and none hit cpu

        if transfer_type == TransferType.H2REMOTE:
            remote_block_id_list = dst_block_ids
            cpu_block_id_list = src_block_ids
        elif transfer_type == TransferType.REMOTE2H:
            remote_block_id_list = src_block_ids
            cpu_block_id_list = dst_block_ids
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type} for CPUSSDDiskTransferWorker")

        layer_id_list = torch.arange(layer_id, layer_id + layer_granularity, dtype=torch.int32)
        transfer_kv_blocks_remote(
            file_nodeid_list=self.file_nodeid_list,
            cpu_layer_id_list=layer_id_list,
            cpu_tensor_ptr=self.cpu_layer_ptrs[0].item(),
            remote_block_ids=remote_block_id_list,
            cpu_block_ids=cpu_block_id_list,
            cpu_layer_stride_in_bytes=self.cpu_layer_stride_in_bytes,
            cpu_kv_stride_in_bytes=self.cpu_kv_stride_in_bytes,
            remote_layer_stride_in_bytes=self.remote_layer_stride_in_bytes_per_file,
            remote_block_stride_in_bytes=self.remote_block_stride_in_bytes,
            remote_kv_stride_in_bytes=self.remote_kv_stride_in_bytes_per_file,
            block_size_in_bytes=self.chunk_size_in_bytes,
            total_layers=self.num_layers,
            is_read=(transfer_type == TransferType.REMOTE2H),
            partition_block_type=PartitionBlockType.SEQUENTIAL.value, # use sequential
            round_robin=self.round_robin,
            num_remote_blocks_per_file=self.num_remote_blocks_per_file,
            use_mmap=False,  # TODO: fix bug when use mmap
            num_threads_per_file=32,
            is_mla=self.is_mla,
        )

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> None:
        layer_id = transfer_op.layer_id
        layer_granularity = transfer_op.layer_granularity
        if layer_id == -1:
            layer_id = 0
        if layer_granularity == -1:
            layer_granularity = self.num_layers

        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op)

        start_time = time.time()
        self._transfer_impl(
            src_block_ids,
            dst_block_ids,
            transfer_op.transfer_type,
            transfer_op.layer_id,
            transfer_op.layer_granularity,
        )
        end_time = time.time()

        kv_dim = 2 if not self.is_mla else 1
        transfer_size = self.chunk_size_in_bytes * layer_granularity * transfer_op.valid_block_num * kv_dim

        self._log_transfer_performance(
            transfer_op,
            transfer_size,
            start_time,
            end_time,
        )
