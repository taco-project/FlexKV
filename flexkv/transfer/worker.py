import os
import copy
import torch.multiprocessing as mp
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch.multiprocessing import Queue as MPQueue, Pipe as MPPipe
from multiprocessing.connection import Connection
from threading import Thread
from typing import List, Any, Dict, Union, Optional, Tuple

import ctypes
import numpy as np
import nvtx
import torch
import zmq
import json

from flexkv import c_ext

from flexkv.c_ext import transfer_kv_blocks, transfer_kv_blocks_ssd, \
    transfer_kv_blocks_gds, transfer_kv_blocks_gds_nvmf, TPTransferThreadGroup, TPGDSTransferThreadGroup, FileExtent
from flexkv.common.debug import flexkv_logger
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import TransferOp, TransferType, PartitionBlockType
from flexkv.common.transfer import get_nvtx_range_color
from flexkv.common.config import CacheConfig, GLOBAL_CONFIG_FROM_ENV, MooncakeTransferEngineConfig
from flexkv.mooncakeEngineWrapper import MoonCakeTransferEngineWrapper
from flexkv.transfer.zmqHelper import NotifyMsg, NotifyStatus, SSDZMQServer, SSDZMQClient
from flexkv.cache.redis_meta import RedisMeta, RedisMetaChannel
from flexkv.transfer.utils import group_blocks_by_node_and_segment, group_blocks_by_node, split_contiguous_blocks, RemoteSSD2HMetaInfo, NodeMetaInfo, RDMATaskInfo
try:
    from flexkv.c_ext import (
        transfer_kv_blocks_remote,
        shared_transfer_kv_blocks_remote_read,
    )
except ImportError:
    transfer_kv_blocks_remote = None
    shared_transfer_kv_blocks_remote_read = None


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
    src_slot_id: int
    dst_slot_id: int
    valid_block_num: int
    src_block_ids: np.ndarray
    dst_block_ids: np.ndarray
    src_block_node_ids: Optional[np.ndarray]
    # successors: List[int]

    def __init__(self, transfer_op: TransferOp):
        self.transfer_op_id = transfer_op.op_id
        self.transfer_graph_id = transfer_op.graph_id
        self.transfer_type = transfer_op.transfer_type
        self.layer_id = transfer_op.layer_id
        self.layer_granularity = transfer_op.layer_granularity
        self.src_slot_id = transfer_op.src_slot_id
        self.dst_slot_id = transfer_op.dst_slot_id
        self.valid_block_num = transfer_op.valid_block_num
        # Always preserve optional src_block_node_ids from TransferOp
        self.src_block_node_ids = transfer_op.src_block_node_ids

        if self.src_slot_id == -1:
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
                 op_buffer_tensor: torch.Tensor):
        self.worker_id = worker_id
        self.transfer_conn = transfer_conn  # receive end of pipe
        self.finished_ops_queue: MPQueue[int] = finished_ops_queue

        self.op_buffer_tensor = op_buffer_tensor
        cudaHostRegister(self.op_buffer_tensor) 

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
    def create_worker(cls,
                      mp_ctx: Any,
                      finished_ops_queue: MPQueue,
                      op_buffer_tensor: torch.Tensor,
                      *args: Any, **kwargs: Any) -> 'WorkerHandle':
        """Generic worker creation template method."""
        
        parent_conn, child_conn = mp_ctx.Pipe()  # create pipe
        ready_event = mp_ctx.Event()
        worker_id = cls._get_worker_id()

        process = mp_ctx.Process(
            target=cls._worker_process,
            args=(worker_id, child_conn, finished_ops_queue, op_buffer_tensor, ready_event, *args),
            kwargs=kwargs,
            daemon=True
        )
        process.start()

        return WorkerHandle(worker_id, parent_conn, process, ready_event)

    @classmethod
    def _worker_process(cls, worker_id: int, transfer_conn: Connection, finished_ops_queue: MPQueue,
                        op_buffer_tensor: torch.Tensor, ready_event: Any, *args: Any, **kwargs: Any) -> None:
        # Note: MPI initialization prevention is handled by create_safe_process
        # Environment variables are set before this function is called
        worker = cls(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor, *args, **kwargs)
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

    def get_transfer_block_ids(self,
                               transfer_op: WorkerTransferOp,
                               pinned: bool = True) ->tuple[torch.Tensor, torch.Tensor]:
        """
        Get transfer block ids from op buffer tensor or directly from op
        Args:
            transfer_op: WorkerTransferOp
            pinned: whether to pin the block ids tensor
        Returns:
            tuple[torch.Tensor, torch.Tensor]: src_block_ids and dst_block_ids
        """
        src_slot_id = transfer_op.src_slot_id
        dst_slot_id = transfer_op.dst_slot_id
        valid_block_num = transfer_op.valid_block_num
        
        if src_slot_id == -1:
            src_block_ids = torch.from_numpy(transfer_op.src_block_ids).to(dtype=torch.int64)
            if pinned:
                src_block_ids = src_block_ids.pin_memory()
        else:
            src_block_ids = self.op_buffer_tensor[src_slot_id, :valid_block_num]
            
        if dst_slot_id == -1:
            dst_block_ids = torch.from_numpy(transfer_op.dst_block_ids).to(dtype=torch.int64)
            if pinned:
                dst_block_ids = dst_block_ids.pin_memory()
        else:
            dst_block_ids = self.op_buffer_tensor[dst_slot_id, :valid_block_num]

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
    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
        pass

    def run(self) -> None:
        """main loop for worker process"""
        while True:
            try:
                if self.transfer_conn.poll(timeout=0.0001):  # check if data available
                    op = self.transfer_conn.recv()
                    if op is None:
                        # shut down zmq listening server of peer2cpuTransferWorker
                        if hasattr(self, "shutdown") and callable(getattr(self, "shutdown")):
                            try:
                                self.shutdown()
                            except Exception as e:
                                flexkv_logger.error(f"Error when shut down worker: {e}")
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
                            transfer_status = self.launch_transfer(op)
                            nvtx.pop_range()
                        except Exception as e:
                            flexkv_logger.error(f"Error launching transfer: {e}\n"
                                        f"Failed transfer op: {op}")
                        if transfer_status:
                            ## only put the op when transfer success
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
                 op_buffer_tensor: torch.Tensor,
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
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)
        # Register CPU tensors with CUDA
        cudaHostRegister(cpu_blocks)
        self.gpu_blocks = [wrapper.get_tensor() for wrapper in gpu_blocks]
        # Get pointers first
        self.gpu_blocks_ptrs = self._get_layer_ptrs(self.gpu_blocks)
        self.gpu_tensor_ptrs = self.gpu_blocks_ptrs

        self.cpu_tensor = cpu_blocks

        self.dtype = dtype
        self.is_mla = gpu_kv_layout.is_mla

        self.num_layers = gpu_kv_layout.num_layer
        gpu_kv_layout_per_layer = gpu_kv_layout.div_layer(self.num_layers)

        # a chunk can be located by layer_id * layer_stride + kv_id * kv_stride + block_id * block_stride
        self.chunk_size_in_bytes = gpu_kv_layout_per_layer.get_chunk_size() * self.dtype.itemsize
        self.gpu_kv_stride_in_bytes = gpu_kv_layout_per_layer.get_kv_stride() * self.dtype.itemsize
        self.gpu_block_stride_in_bytes = gpu_kv_layout_per_layer.get_block_stride() * self.dtype.itemsize
        self.gpu_layer_stride_in_bytes = gpu_kv_layout_per_layer.get_layer_stride() * self.dtype.itemsize

        self.cpu_layer_stride_in_bytes = cpu_kv_layout.get_layer_stride() * self.dtype.itemsize
        self.cpu_kv_stride_in_bytes = cpu_kv_layout.get_kv_stride() * self.dtype.itemsize
        self.cpu_block_stride_in_bytes = cpu_kv_layout.get_block_stride() * self.dtype.itemsize

        if len(self.gpu_blocks) == 1:
            self.gpu_block_type_ = 1
        elif len(self.gpu_blocks) == self.num_layers:
            self.gpu_block_type_ = 0
        elif len(self.gpu_blocks) == self.num_layers * 2:
            self.gpu_block_type_ = 2
        else:
            raise ValueError(f"Invalid GPU block type: {len(self.gpu_blocks)}")
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

        gpu_tensor_ptrs = self.gpu_blocks_ptrs.contiguous().pin_memory()

        transfer_kv_blocks(
            gpu_block_id_list,
            gpu_tensor_ptrs,
            self.gpu_kv_stride_in_bytes,
            self.gpu_block_stride_in_bytes,
            self.gpu_layer_stride_in_bytes,
            cpu_block_id_list,
            self.cpu_tensor,
            self.cpu_kv_stride_in_bytes,
            self.cpu_layer_stride_in_bytes,
            self.cpu_block_stride_in_bytes,
            self.chunk_size_in_bytes,
            layer_id,
            layer_granularity,
            transfer_sms,
            transfer_type == TransferType.H2D,
            use_ce_transfer,
            self.is_mla,
            self.gpu_block_type_,
        )

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
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
        return True
        
class tpGPUCPUTransferWorker(TransferWorkerBase):
    def __init__(self,
                 worker_id: int,
                 transfer_conn: Connection,
                 finished_ops_queue: MPQueue,
                 op_buffer_tensor: torch.Tensor,
                 gpu_blocks: List[List[TensorSharedHandle]],
                 cpu_blocks: torch.Tensor,
                 gpu_kv_layouts: List[KVCacheLayout],
                 cpu_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 tp_group_size: int,
                 dp_group_id: int,
                 use_ce_transfer_h2d: bool = False,
                 use_ce_transfer_d2h: bool = False,
                 transfer_sms_h2d: int = 8,
                 transfer_sms_d2h: int = 8):

        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)
        assert len(gpu_blocks) == tp_group_size
        # Handle tensor import for multi-process case
        imported_gpu_blocks = []
        for handles_in_one_gpu in gpu_blocks:
            blocks_in_one_gpu = []
            for handle in handles_in_one_gpu:
                blocks_in_one_gpu.append(handle.get_tensor())
            imported_gpu_blocks.append(blocks_in_one_gpu)
        self.gpu_blocks = imported_gpu_blocks
        self.dtype = dtype # note this should be quantized data type
        self.is_mla = gpu_kv_layouts[0].is_mla

        self.num_gpus = len(self.gpu_blocks)
        self.tp_group_size = tp_group_size
        self.dp_group_id = dp_group_id

        cudaHostRegister(cpu_blocks)

        self.num_layers = gpu_kv_layouts[0].num_layer
        # here the chunk size doesn't include the layer info
        self.gpu_chunk_sizes_in_bytes = [gpu_kv_layout.get_chunk_size() * self.dtype.itemsize \
                                for gpu_kv_layout in gpu_kv_layouts]
        self.gpu_kv_strides_in_bytes = [gpu_kv_layout.get_kv_stride() * self.dtype.itemsize \
                                for gpu_kv_layout in gpu_kv_layouts]
        self.gpu_block_strides_in_bytes = [gpu_kv_layout.get_block_stride() * self.dtype.itemsize \
                                for gpu_kv_layout in gpu_kv_layouts]
        self.gpu_layer_strides_in_bytes = [gpu_kv_layout.get_layer_stride() * self.dtype.itemsize \
                                for gpu_kv_layout in gpu_kv_layouts]

        self.cpu_chunk_size_in_bytes = cpu_kv_layout.get_chunk_size() * self.dtype.itemsize
        self.cpu_layer_stride_in_bytes = cpu_kv_layout.get_layer_stride() * self.dtype.itemsize
        self.cpu_kv_stride_in_bytes = cpu_kv_layout.get_kv_stride() * self.dtype.itemsize
        self.cpu_block_stride_in_bytes = cpu_kv_layout.get_block_stride() * self.dtype.itemsize

        self.transfer_sms_h2d = transfer_sms_h2d
        self.transfer_sms_d2h = transfer_sms_d2h
        self.use_ce_transfer_h2d = use_ce_transfer_h2d
        self.use_ce_transfer_d2h = use_ce_transfer_d2h

        gpu_kv_strides_tensor = torch.tensor(self.gpu_kv_strides_in_bytes, dtype=torch.int64)
        gpu_block_strides_tensor = torch.tensor(self.gpu_block_strides_in_bytes, dtype=torch.int64)
        gpu_chunk_sizes_tensor = torch.tensor(self.gpu_chunk_sizes_in_bytes, dtype=torch.int64)
        gpu_layer_strides_tensor = torch.tensor(self.gpu_layer_strides_in_bytes, dtype=torch.int64)
        self.tp_transfer_thread_group = TPTransferThreadGroup(self.num_gpus, self.gpu_blocks, cpu_blocks, dp_group_id,
                                                              self.num_layers, gpu_kv_strides_tensor,
                                                              gpu_block_strides_tensor, gpu_layer_strides_tensor,
                                                              gpu_chunk_sizes_tensor)


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


    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
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
        return True

class CPUSSDDiskTransferWorker(TransferWorkerBase):
    def __init__(self,
                 worker_id: int,
                 transfer_conn: Connection,
                 finished_ops_queue: MPQueue,
                 op_buffer_tensor: torch.Tensor,
                 cpu_blocks: torch.Tensor,
                 ssd_files: Dict[int, List[str]],  # ssd_device_id -> file_paths
                 cpu_kv_layout: KVCacheLayout,
                 ssd_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 num_blocks_per_file: int,
                 cache_config: CacheConfig):
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)
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
            self.ioctx = c_ext.SSDIOCTX(ssd_files, len(ssd_files), GLOBAL_CONFIG_FROM_ENV.iouring_entries,
                GLOBAL_CONFIG_FROM_ENV.iouring_flags)
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

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
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
            layer_id,           # Use corrected value, not transfer_op.layer_id
            layer_granularity,  # Use corrected value, not transfer_op.layer_granularity
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

        return True
    
class CPURemoteTransferWorker(TransferWorkerBase):
    def __init__(self,
                 worker_id: int,
                 transfer_conn: Connection,
                 finished_ops_queue: MPQueue,
                 op_buffer_tensor: torch.Tensor,
                 cpu_blocks: List[torch.Tensor],
                 remote_file: List[str],
                 cpu_kv_layout: KVCacheLayout,
                 remote_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 remote_config_custom: Dict[str, Any],
                 enable_pcfs_sharing: bool = False):
        if transfer_kv_blocks_remote is None:
            raise RuntimeError("transfer_kv_blocks_remote not available, please build with FLEXKV_ENABLE_CFS=1")
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)

        self.cpu_layer_ptrs = self._get_layer_ptrs(cpu_blocks)
        self.remote_files = remote_file
        self.num_remote_files = len(remote_file)

        self.num_layers = cpu_kv_layout.num_layer
        self.num_cpu_blocks = cpu_kv_layout.num_block
        self.num_remote_blocks = remote_kv_layout.num_block
        self.round_robin = 1
        self.enable_pcfs_sharing = enable_pcfs_sharing

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
                # Use PCFS shared transfer for read operations when PCFS sharing is enabled
        if self.enable_pcfs_sharing and transfer_type == TransferType.REMOTE2H:
            # For PCFS sharing, we need to construct cfs_blocks_partition and cpu_blocks_partition
            # based on the file_nodeids from the transfer operation
            # Optional: per-source-block node ids for remote routing (numpy.ndarray)
            src_block_node_ids = kwargs.get("src_block_node_ids")
            if src_block_node_ids is not None and not isinstance(src_block_node_ids, np.ndarray):
                raise TypeError("src_block_node_ids must be a numpy.ndarray if provided")
            
            assert len(src_block_node_ids) == len(remote_block_id_list)
            
            # Construct cfs_blocks_partition and cpu_blocks_partition
            # This is a simplified implementation - in practice, you might need more sophisticated logic
            
            # Group blocks by file_nodeid (simplified grouping logic)
            files_set = set(src_block_node_ids)
            file_nodeids_list = list(files_set)
            
            # Initialize partitions with proper size
            cfs_blocks_partition = [[] for _ in range(len(file_nodeids_list))]
            cpu_blocks_partition = [[] for _ in range(len(file_nodeids_list))]
            
            # Create mapping from file_nodeid to partition index
            file2fid_dict = {file_nodeid: fid for fid, file_nodeid in enumerate(file_nodeids_list)}
            #因为每个flexkv的文件数量是相同的，所以total_file_num是相同的，后面用全局block_id计算block_id_in_file时，需要除以total_file_num
            total_file_num = len(self.file_nodeid_list)
            for i in range(len(remote_block_id_list)):
                file_nodeid = src_block_node_ids[i]
                fid = file2fid_dict[file_nodeid]
                
                # Calculate block_id_in_file using the same logic as C++
                # This should match the C++ implementation in pcfs.cpp
                block_id_in_file = int(((remote_block_id_list[i] / self.round_robin) / total_file_num) * self.round_robin + (remote_block_id_list[i] % self.round_robin))
                
                cfs_blocks_partition[fid].append(block_id_in_file)
                cpu_blocks_partition[fid].append(cpu_block_id_list[i].item())
            
            # Use the new shared transfer function
            shared_transfer_kv_blocks_remote_read(
                file_nodeid_list=file_nodeids_list,
                cfs_blocks_partition_list=cfs_blocks_partition,
                cpu_blocks_partition_list=cpu_blocks_partition,
                cpu_layer_id_list=layer_id_list,
                cpu_tensor_ptr=self.cpu_layer_ptrs[0].item(),
                cpu_layer_stride_in_bytes=self.cpu_layer_stride_in_bytes,
                cpu_kv_stride_in_bytes=self.cpu_kv_stride_in_bytes,
                cfs_layer_stride_in_bytes=self.remote_layer_stride_in_bytes_per_file,
                cfs_block_stride_in_bytes=self.remote_block_stride_in_bytes,
                cfs_kv_stride_in_bytes=self.remote_kv_stride_in_bytes_per_file,
                block_size_in_bytes=self.chunk_size_in_bytes,
                total_layers=self.num_layers,
                is_mla=self.is_mla,
                num_threads_per_file=32,
            )
        else:
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

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
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
            layer_id,           # Use corrected value, not transfer_op.layer_id
            layer_granularity,  # Use corrected value, not transfer_op.layer_granularity
            src_block_node_ids=transfer_op.src_block_node_ids,
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

class GDSTransferWorker(TransferWorkerBase):
    def __init__(
        self,
        worker_id: int,
        transfer_conn: Connection,
        finished_ops_queue: MPQueue,
        op_buffer_tensor: torch.Tensor,
        gpu_blocks: List[TensorSharedHandle],
        ssd_files: Dict[int, List[str]],
        num_blocks_per_file: int,
        gpu_kv_layout: KVCacheLayout,
        ssd_kv_layout: KVCacheLayout,
        dtype: torch.dtype,
        gpu_device_id: int = 0,
    ) -> None:
        """
        Initialize GDS Transfer Worker

        Args:
            worker_id: Worker ID
            transfer_queue: Queue for incoming transfer operations
            finished_ops_queue: Queue for completed operations
            gpu_blocks: GPU memory block handles
            ssd_files: Dict of SSD file paths (ssd_device_id -> file_paths)
            num_blocks_per_file: Number of blocks per file
            gpu_kv_layout: Layout of GPU KV cache
            ssd_kv_layout: Layout of SSD KV cache
            dtype: Data type
            gpu_device_id: GPU device ID
        """
        # Initialize base class first
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)

        self.gpu_blocks = [wrapper.get_tensor() for wrapper in gpu_blocks]
        self.gpu_blocks_ptrs = self._get_layer_ptrs(self.gpu_blocks)
        self.gpu_layer_ptrs = self.gpu_blocks_ptrs
        self.num_blocks_per_file = num_blocks_per_file
        self.num_files = sum(len(file_list) for file_list in ssd_files.values())

        # Use same round_robin as SSD transfer to ensure consistent block mapping
        self.round_robin = 1
        # Create GDSManager from file paths in this worker process
        self.gds_manager = c_ext.GDSManager(
            ssd_files,
            len(ssd_files),
            self.round_robin
        )
        
        if not self.gds_manager.is_ready():
            raise RuntimeError(f"Failed to initialize GDS Manager in worker {worker_id}: "
                               f"{self.gds_manager.get_last_error()}")

        self.dtype = dtype
        self.is_mla = gpu_kv_layout.is_mla

        # Layout information
        self.num_layers = gpu_kv_layout.num_layer
        gpu_kv_layout_per_layer = gpu_kv_layout.div_layer(self.num_layers)
        ssd_kv_layout_per_file = ssd_kv_layout.div_block(self.num_files, padding=True)

        # GPU layout calculations
        self.chunk_size_in_bytes = gpu_kv_layout_per_layer.get_chunk_size() * self.dtype.itemsize
        self.gpu_kv_stride_in_bytes = gpu_kv_layout_per_layer.get_kv_stride() * self.dtype.itemsize
        self.gpu_block_stride_in_bytes = gpu_kv_layout_per_layer.get_block_stride() * self.dtype.itemsize
        self.gpu_layer_stride_in_bytes = gpu_kv_layout_per_layer.get_layer_stride() * self.dtype.itemsize

        # SSD layout calculations
        self.ssd_layer_stride_in_bytes = ssd_kv_layout_per_file.get_layer_stride() * self.dtype.itemsize
        self.ssd_kv_stride_in_bytes = ssd_kv_layout_per_file.get_kv_stride() * self.dtype.itemsize
        self.ssd_block_stride_in_bytes = ssd_kv_layout_per_file.get_block_stride() * self.dtype.itemsize

        if len(self.gpu_blocks) == 1:
            self.gpu_block_type_ = 1  # TRTLLM
        elif len(self.gpu_blocks) == self.num_layers:
            self.gpu_block_type_ = 0  # VLLM
        elif len(self.gpu_blocks) == self.num_layers * 2:
            self.gpu_block_type_ = 2  # SGLANG
        else:
            raise ValueError(f"Invalid GPU block type: {len(self.gpu_blocks)}")

        # Set GPU device and create stream
        self.gpu_device_id = gpu_device_id
        if gpu_device_id != -1:
            torch.cuda.set_device(gpu_device_id)
        self.transfer_stream = torch.cuda.Stream()

    def _transfer_impl(
        self,
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
        transfer_type: TransferType,
        layer_id: int,
        layer_granularity: int,
        **kwargs: Any,
    ) -> None:
        """Implement actual transfer between GPU and SSD"""
        assert src_block_ids.dtype == torch.int64
        assert dst_block_ids.dtype == torch.int64
        assert len(src_block_ids) == len(dst_block_ids)

        if layer_id == -1:
            layer_id = 0
        if layer_granularity == -1:
            layer_granularity = self.num_layers

        # Convert to tensors
        # SSD uses DISK2D/D2DISK transfer types (same as traditional SSD I/O)
        if transfer_type == TransferType.DISK2D:
            # SSD to GPU via GDS path: src=SSD, dst=GPU
            ssd_block_id_list = src_block_ids
            gpu_block_id_list = dst_block_ids
        elif transfer_type == TransferType.D2DISK:
            # GPU to SSD via GDS path: src=GPU, dst=SSD
            gpu_block_id_list = src_block_ids
            ssd_block_id_list = dst_block_ids
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type} for GDSTransferWorker. Expected DISK2D or D2DISK.")

        if len(ssd_block_id_list) == 0:
            return

        # Process transfer for each layer
        layer_id_list = torch.arange(layer_id, layer_id + layer_granularity, dtype=torch.int32)

        # Determine if this is a read operation
        is_read = (transfer_type == TransferType.DISK2D)

        # Use the optimized C++ function for KV block transfers
        # Note: topology information (files, devices, round_robin) is now encapsulated in gds_manager
        try:
            transfer_kv_blocks_gds(
                self.gds_manager,               # GDS manager (contains topology info)
                layer_id_list,                  # GPU layer IDs to process
                self.gpu_layer_ptrs,            # GPU layer pointers tensor
                ssd_block_id_list,              # SSD block IDs
                gpu_block_id_list,              # GPU block IDs
                self.gpu_kv_stride_in_bytes,    # GPU K-V stride
                self.gpu_block_stride_in_bytes, # GPU block stride
                self.gpu_layer_stride_in_bytes, # GPU layer stride
                self.ssd_layer_stride_in_bytes, # SSD layer stride
                self.ssd_block_stride_in_bytes, # SSD block stride
                self.ssd_kv_stride_in_bytes,    # SSD K-V stride
                self.chunk_size_in_bytes,       # Chunk size
                0,                              # SSD copy offset
                self.num_blocks_per_file,       # Blocks per file
                self.num_layers,                # Total layers
                is_read,                        # Read or write
                False,                          # Verbose logging
                self.is_mla,                    # MLA
                self.gpu_block_type_            # GPU block type
            )

        except Exception as e:
            flexkv_logger.error(f"GDS transfer failed: {e}")
            raise RuntimeError(f"Failed to transfer KV blocks: {e}") from e

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> None:
        """Launch a GDS transfer operation"""
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


class tpGDSTransferWorker(TransferWorkerBase):
    def __init__(
        self,
        worker_id: int,
        transfer_conn: Connection,
        finished_ops_queue: MPQueue,
        op_buffer_tensor: torch.Tensor,
        gpu_blocks: List[List[TensorSharedHandle]],
        ssd_files: Dict[int, List[str]],
        num_blocks_per_file: int,
        gpu_kv_layouts: List[KVCacheLayout],
        ssd_kv_layout: KVCacheLayout,
        dtype: torch.dtype,
        tp_group_size: int,
        dp_group_id: int,
    ) -> None:
        """
        Initialize TP GDS Transfer Worker

        Args:
            worker_id: Worker ID
            transfer_queue: Queue for incoming transfer operations
            finished_ops_queue: Queue for completed operations
            gpu_blocks: List of GPU memory block handles for each GPU in TP group
            ssd_files: Dict of SSD file paths
            num_blocks_per_file: Number of blocks per file
            gpu_kv_layouts: Layout of GPU KV cache
            ssd_kv_layout: Layout of SSD KV cache
            dtype: Data type
            tp_group_size: Size of tensor parallel group
            dp_group_id: Data parallel group ID
        """
        # Initialize base class first
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)

        assert len(gpu_blocks) == tp_group_size
        # Handle tensor import for multi-process case
        imported_gpu_blocks = []
        for handles_in_one_gpu in gpu_blocks:
            blocks_in_one_gpu = []
            for handle in handles_in_one_gpu:
                blocks_in_one_gpu.append(handle.get_tensor())
            imported_gpu_blocks.append(blocks_in_one_gpu)
        self.gpu_blocks = imported_gpu_blocks
        self.num_blocks_per_file = num_blocks_per_file
        self.num_files = sum(len(file_list) for file_list in ssd_files.values())

        self.dtype = dtype
        self.is_mla = gpu_kv_layouts[0].is_mla
        self.num_gpus = len(self.gpu_blocks)
        self.tp_group_size = tp_group_size
        self.dp_group_id = dp_group_id

        # Layout information
        self.num_layers = gpu_kv_layouts[0].num_layer
        ssd_kv_layout_per_file = ssd_kv_layout.div_block(self.num_files, padding=True)

        # GPU layout calculations
        self.gpu_chunk_sizes_in_bytes = [gpu_kv_layout.get_chunk_size() * self.dtype.itemsize \
                                         for gpu_kv_layout in gpu_kv_layouts]
        self.gpu_kv_strides_in_bytes = [gpu_kv_layout.get_kv_stride() * self.dtype.itemsize \
                                        for gpu_kv_layout in gpu_kv_layouts]
        self.gpu_block_strides_in_bytes = [gpu_kv_layout.get_block_stride() * self.dtype.itemsize \
                                           for gpu_kv_layout in gpu_kv_layouts]
        self.gpu_layer_strides_in_bytes = [gpu_kv_layout.get_layer_stride() * self.dtype.itemsize \
                                           for gpu_kv_layout in gpu_kv_layouts]

        # SSD layout calculations
        self.ssd_chunk_size_in_bytes = ssd_kv_layout_per_file.get_chunk_size() * self.dtype.itemsize
        self.ssd_layer_stride_in_bytes = ssd_kv_layout_per_file.get_layer_stride() * self.dtype.itemsize
        self.ssd_kv_stride_in_bytes = ssd_kv_layout_per_file.get_kv_stride() * self.dtype.itemsize
        self.ssd_block_stride_in_bytes = ssd_kv_layout_per_file.get_block_stride() * self.dtype.itemsize

        # Create TP GDS Transfer Thread Group
        gpu_kv_strides_tensor = torch.tensor(self.gpu_kv_strides_in_bytes, dtype=torch.int64)
        gpu_block_strides_tensor = torch.tensor(self.gpu_block_strides_in_bytes, dtype=torch.int64)
        gpu_layer_strides_tensor = torch.tensor(self.gpu_layer_strides_in_bytes, dtype=torch.int64)
        gpu_chunk_sizes_tensor = torch.tensor(self.gpu_chunk_sizes_in_bytes, dtype=torch.int64)
        self.tp_gds_transfer_thread_group = TPGDSTransferThreadGroup(
            self.num_gpus, self.gpu_blocks, ssd_files, dp_group_id, self.num_layers,
            gpu_kv_strides_tensor, gpu_block_strides_tensor, gpu_layer_strides_tensor, gpu_chunk_sizes_tensor)

    def _transfer_impl(self,
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

        # GDS uses DISK2D/D2DISK transfer types (same as traditional SSD I/O)
        if transfer_type == TransferType.D2DISK:
            gpu_block_ids = src_block_ids
            ssd_block_ids = dst_block_ids
            is_read = False  # GPU -> SSD via GDS (write)
        elif transfer_type == TransferType.DISK2D:
            gpu_block_ids = dst_block_ids
            ssd_block_ids = src_block_ids
            is_read = True   # SSD -> GPU via GDS (read)
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type} for tpGDSTransferWorker. Expected DISK2D or D2DISK.")

        gpu_block_id_list = gpu_block_ids
        ssd_block_id_list = ssd_block_ids

        assert len(gpu_block_id_list) == len(ssd_block_id_list)

        if len(gpu_block_id_list) == 0:
            return

        self.tp_gds_transfer_thread_group.tp_group_transfer(
            gpu_block_id_list,
            ssd_block_id_list,
            self.ssd_layer_stride_in_bytes,
            self.ssd_kv_stride_in_bytes,
            self.ssd_block_stride_in_bytes,
            self.ssd_chunk_size_in_bytes,
            self.num_blocks_per_file,
            is_read,
            layer_id,
            layer_granularity,
            self.is_mla,
        )

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> None:
        """Launch a TP GDS transfer operation"""
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
        transfer_size = self.ssd_chunk_size_in_bytes * layer_granularity * transfer_op.valid_block_num * kv_dim

        self._log_transfer_performance(
            transfer_op,
            transfer_size,
            start_time,
            end_time,
        )

class PEER2CPUTransferWorker(TransferWorkerBase):
    def __init__(self,
        worker_id: int,
        transfer_conn: Connection,
        finished_ops_queue: MPQueue,
        op_buffer_tensor: torch.Tensor,
        cpu_blocks: torch.Tensor,
        cpu_kv_layout: KVCacheLayout,
        remote_kv_layout: KVCacheLayout,
        dtype: torch.dtype,
        cache_config: CacheConfig,
        ssd_kv_layout: KVCacheLayout = None,
        ssd_files: Dict[int, List[str]] = None,  # ssd_device_id -> file_paths
        num_blocks_per_file: int = 0,
    ):
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)
        self.cpu_layer_ptrs = self._get_layer_ptrs(cpu_blocks)
        self.num_layers = cpu_kv_layout.num_layer
        self.num_cpu_blocks = cpu_kv_layout.num_block
        self.block_size = cpu_kv_layout.get_chunk_size()
        self.dtype = dtype
        self.cpu_kv_layout = cpu_kv_layout
        self.remote_kv_layout = remote_kv_layout

        self.is_mla = cpu_kv_layout.is_mla
        self.kv_dim = 2 if not self.is_mla else 1

        self.cpu_blocks = cpu_blocks  ## shared memory
        self.cache_config = cache_config
        self.dst_buffer_ptr = self.cpu_blocks.data_ptr()

        self.mooncake_transfer_engine = None
        # self.zmq_listen_addr = ""

        self.zmq_listen_addr = (
            f"tcp://{cache_config.local_zmq_ip}:{cache_config.local_zmq_port}"
        )

        ## initialize distributed environment
        if self.cache_config.enable_kv_sharing:
            # step1: initialize the redis meta client for node info
            self.redis_meta_client = RedisMeta(
                self.cache_config.redis_host,
                self.cache_config.redis_port,
                self.cache_config.redis_password,
                self.cache_config.local_ip,
            )
            self.redis_meta_client.set_node_id(self.cache_config.distributed_node_id)

            # Persistent NodeMetaInfo Pool for skip redis operation when getting 
            # NodeMetaInfo according to node_id
            # assuming that every flexkv progress has unique node id
            self.node_metas: Dict[int, NodeMetaInfo] = {}
            assert self.redis_meta_client != None


            # step2: initialize mooncake transfer engine for the whole flexkv
            # NOTE:now we read the config file by env paras
            mooncake_config_path = os.environ["MOONCAKE_CONFIG_PATH"]
            self.mooncake_config = MooncakeTransferEngineConfig.from_file(
                mooncake_config_path
            )
            self.mooncake_transfer_engine = MoonCakeTransferEngineWrapper(
                self.mooncake_config
            )
            assert (
                self.mooncake_transfer_engine != None
            ), "PEER2CPUTransferWorker: initilaize mooncake transfer engine failed"

            # step3: register local cpu buffer to mooncake transfer engine
            total_cpu_blocks_size = (
                self.cpu_blocks.numel() * self.cpu_blocks.element_size()
            )
            regist_buffer_status = self.mooncake_transfer_engine.regist_buffer(
                self.cpu_blocks.data_ptr(), total_cpu_blocks_size
            )
            assert (
                regist_buffer_status == 0
            ), "PEER2CPUTransferWorker: regist cpu buffer to mooncake transfer engine"

            # step4: regist node info into redis server
            self.regist_node_meta(
                self.cpu_blocks.data_ptr(),
                self.cpu_blocks.data_ptr(),
                self.zmq_listen_addr,
            )

        ## when enable p2p ssd, we need start a zmq server to recive the meta info from remote node, 
        # and allocate a cpu buffer for ssd to cpu copy
        if self.cache_config.enable_p2p_ssd:
            assert ssd_kv_layout is not None, "Invalid ssd kv layout!"
            ## init the cpu buffer for ssd to cpu copy
            # NOTE: now we allocate 500 blocks for test
            self.tmp_cpu_buffer_layout = KVCacheLayout(
                self.cpu_kv_layout.type,
                self.cpu_kv_layout.num_layer,
                self.cache_config.num_tmp_cpu_blocks,
                self.cpu_kv_layout.tokens_per_block,
                self.cpu_kv_layout.num_head,
                self.cpu_kv_layout.head_size,
                self.cpu_kv_layout.is_mla,
                self.cpu_kv_layout._kv_shape,
            )
            self.tmp_cpu_buffer = torch.empty(
                self.tmp_cpu_buffer_layout.get_total_elements(),
                dtype=self.dtype,
                device="cpu",
                pin_memory=True,
            )
            self.mooncake_transfer_engine.regist_buffer(
                self.tmp_cpu_buffer.data_ptr(),
                self.tmp_cpu_buffer.numel() * self.tmp_cpu_buffer.element_size(),
            )

            ## start the zmq server and client
            self.zmq_server = SSDZMQServer(cache_config.local_zmq_ip, cache_config.local_zmq_port, self.ssd_handle_loop)
            self.zmq_client = SSDZMQClient(cache_config.local_zmq_ip, cache_config.local_zmq_port+1)
            
            ## ssd copy to temp cpu buffer related
            self.ssd_files = ssd_files
            self.num_blocks_per_file = num_blocks_per_file
            self.num_files = sum(len(file_list) for file_list in ssd_files.values())

            ssd_kv_layout_per_file = ssd_kv_layout.div_block(self.num_files, padding=True)

            self.chunk_size_in_bytes = (
                self.tmp_cpu_buffer_layout.get_chunk_size() * self.dtype.itemsize
            )
            self.block_stride_in_bytes = (
                self.tmp_cpu_buffer_layout.get_block_stride() * self.dtype.itemsize
            )
            self.cpu_kv_stride_in_bytes = (
                self.tmp_cpu_buffer_layout.get_kv_stride() * self.dtype.itemsize
            )
            self.cpu_layer_stride_in_bytes = (
                self.tmp_cpu_buffer_layout.get_layer_stride() * self.dtype.itemsize
            )
            self.ssd_kv_stride_in_bytes = (
                ssd_kv_layout_per_file.get_kv_stride() * self.dtype.itemsize
            )
            self.ssd_layer_stride_in_bytes = (
                ssd_kv_layout_per_file.get_layer_stride() * self.dtype.itemsize
            )
            
       
            self.round_robin = 1
            # initialize ssd ioctx
            try:
                self.ioctx = c_ext.SSDIOCTX(
                    ssd_files,
                    len(ssd_files),
                    GLOBAL_CONFIG_FROM_ENV.iouring_entries,
                    GLOBAL_CONFIG_FROM_ENV.iouring_flags,
                )
            except Exception as e:
                flexkv_logger.error(f"Error setting ssd ioctx: {e}\n")
                raise RuntimeError("SSD Worker init failed") from e

        ## unique task id counter for remote ssd to cpu transfer task
        self.remote_ssd_task_id_counter = 0
        self.task_id_lock = threading.Lock()

    #============================ common part ========================
    def gen_task_id(self) -> int:
        """
        generate a unique task id for remote ssd to cpu transfer task
        Returns:
            int: task id
        """
        with self.task_id_lock:
            old_value = self.remote_ssd_task_id_counter
            self.remote_ssd_task_id_counter += 1
            return old_value

    def shutdown(self):
        self.zmq_server.shutdown()
        self.zmq_client.shutdown()
        # unregist buffer in mooncake engine
        self.mooncake_transfer_engine.unregist_buffer(self.cpu_blocks.data_ptr())
        if self.cache_config.enable_p2p_ssd:
            self.mooncake_transfer_engine.unregist_buffer(self.tmp_cpu_buffer.data_ptr())
        # unregist node info from redis server
        self.unregist_node_meta()

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
        layer_id = transfer_op.layer_id
        layer_granularity = transfer_op.layer_granularity
        if layer_id == -1:
            layer_id = 0
        if layer_granularity == -1:
            layer_granularity = self.num_layers
        task_info_list = self.op_parser(transfer_op, layer_id, layer_granularity)

        start_time = time.time()
        transfered_size = 0
        transfer_finished = True

        for task_info in task_info_list:
            # NOTE: here one task_info represent data transfer from one node 
            ret = self._batch_transfer_impl(
                task_info,
                transfer_op.transfer_type,
                layer_id,
                layer_granularity,
            )
            if not ret:
                transfer_finished = False
                break
            transfered_size += task_info.data_size

        end_time = time.time()

        # kv_dim = 2 if not self.is_mla else 1
        # transfer_size = (
        #     self.block_size * layer_granularity * transfer_op.valid_block_num * kv_dim
        # )

        self._log_transfer_performance(
            transfer_op,
            transfered_size,
            start_time,
            end_time,
        )
        return transfer_finished
    
    def _batch_transfer_impl(self, 
        task_info: RDMATaskInfo,    
        transfer_type: TransferType,
        layer_id: int,
        layer_granularity: int,
        **kwargs,):
        if transfer_type == TransferType.PEERH2H:
            ret = self.mooncake_transfer_engine.batch_transfer_sync_read(
                task_info.peer_engine_addr, task_info.src_ptrs, task_info.dst_ptrs, task_info.data_lens
            )
            if ret != 0:
                flexkv_logger.error(f"RDMA transfer failed with error code: {ret}")
                return False
        elif transfer_type == TransferType.PEERSSD2H:
          # remote ssd to local cpu transfer by two side zmq and one side rdma write
            # step1: construct the meta info
            remote_ssd_to_cpu_meta = RemoteSSD2HMetaInfo(
                task_id=task_info.task_id,
                cpu_block_ids=task_info.dst_block_ids,
                ssd_block_ids=task_info.src_block_ids,
                peer_engine_addr=task_info.local_engine_addr,
                peer_cpu_base_ptr=self.dst_buffer_ptr,
                peer_zmq_status_addr=self.zmq_client.get_addr(),
                data_size=task_info.data_size,
                layer_id=layer_id,
                layer_granularity=layer_granularity,
            )
            #flexkv_logger.info(
            #    f"[PEERSSD2H] Sending meta: task_id={task_info.task_id}, "
            #    f"ssd_block_ids={task_info.src_block_ids}, cpu_block_ids={task_info.dst_block_ids}, "
            #    f"peer_engine_addr={task_info.local_engine_addr}, peer_zmq_addr={task_info.peer_zmq_addr}"
            #)
            ## step2: send the meta info to remote node
            if not self.zmq_client.send_meta_info(remote_ssd_to_cpu_meta, task_info.peer_zmq_addr):
                flexkv_logger.error(
                    f"Send remote ssd to cpu meta info to {task_info.peer_zmq_addr} failed"
                )
                return False

            ## step3: wait for remote node to send data transfer complete notify
            ret = self.zmq_client.wait_transfer_notify(
                task_info.peer_engine_addr, task_info.task_id
            )
            if not ret:
                flexkv_logger.error(
                    f"Wait remote ssd to cpu transfer task {task_info.task_id} notify from {task_info.peer_engine_addr} failed with error code: {ret}"
                )
                return False
        else:
            raise ValueError(
                f"Invalid transfer type: {transfer_type} for PEER2CPUTransferWorker"
            )
        return True

    def _transfer_impl(
        self,
        task_info: RDMATaskInfo,
        transfer_type: TransferType,
        layer_id: int,
        layer_granularity: int,
        **kwargs,
    ):
        if transfer_type == TransferType.PEERH2H:
            # remote cpu to local cpu transfer by one-side rdma read
            for i in range(len(task_info.src_ptrs)):
                ret = self.mooncake_transfer_engine.transfer_sync_read(task_info.peer_zmq_addr, task_info.src_ptrs[i], 
                                                                       task_info.dst_ptrs[i], task_info.data_lens[i])
                if ret != 0:
                    flexkv_logger.error(f"transfer_sync_write failed with error code: {ret}")
                    return False
        elif transfer_type == TransferType.PEERSSD2H:
            # remote ssd to local cpu transfer by two side zmq and one side rdma write
            # step1: construct the meta info
            remote_ssd_to_cpu_meta = RemoteSSD2HMetaInfo(
                task_id=task_info.task_id,
                cpu_block_ids=task_info.dst_block_ids,
                ssd_block_ids=task_info.src_block_ids,
                peer_engine_addr=task_info.local_engine_addr,
                peer_cpu_base_ptr=self.dst_buffer_ptr,
                peer_zmq_status_addr=self.zmq_client.get_addr(),
                data_size=task_info.data_size,
                layer_id=layer_id,
                layer_granularity=layer_granularity,
            )
            flexkv_logger.info(
                f"[_transfer_impl] Sending task_id={task_info.task_id}, "
                f"ssd_block_ids={task_info.src_block_ids}, "
                f"cpu_block_ids={task_info.dst_block_ids} to {task_info.peer_zmq_addr}"
            )
            ## step2: send the meta info to remote node
            if not self.zmq_client.send_meta_info(remote_ssd_to_cpu_meta, task_info.peer_zmq_addr):
                flexkv_logger.error(
                    f"Send remote ssd to cpu meta info to {task_info.peer_zmq_addr} failed"
                )
                return False

            ## step3: wait for remote node to send data transfer complete notify
            ret = self.zmq_client.wait_transfer_notify(
                task_info.peer_engine_addr, task_info.task_id
            )
            if not ret:
                flexkv_logger.error(
                    f"Wait remote ssd to cpu transfer task {task_info.task_id} notify from {task_info.peer_engine_addr} failed with error code: {ret}"
                )
                return False

        else:
            raise ValueError(
                f"Invalid transfer type: {transfer_type} for PEER2CPUTransferWorker"
            )

        return True
    
    def op_parser(
        self, transfer_op: WorkerTransferOp, layer_id: int, layer_granularity: int
    ) -> List[RDMATaskInfo]:
        """
        parse the transfer op to a list of RDMATaskInfo
        1. group the blocks by remote node id, each segment is a list of continuous blocks (segment is the smallest transmission unit)
        2. using corresponding distributed op parser to parse the op and create RDMATaskInfo for each segment
        5. return the list of RDMATaskInfo
        Parameters:
            transfer_op (WorkerTransferOp): the transfer op to be parsed
        Returns:
            List[RDMATaskInfo]: the list of RDMATaskInfo
        """
        assert (
            transfer_op.transfer_type == TransferType.PEERH2H
            or transfer_op.transfer_type == TransferType.PEERSSD2H
        ), f"PEER2CPUTransferWorker only support PEERH2H or PEERSSD2H, but get {transfer_op.transfer_type}"

        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op, False)

        assert len(src_block_ids) == len(dst_block_ids)

        src_block_node_ids = transfer_op.src_block_node_ids

        # step1: group the blocks by remote node id and remote block source type, each segment is a list of continuous blocks
        #flexkv_logger.info(
        #    f"[PEER2CPUTransferWorker] src_block_ids: {src_block_ids} \n \
        #                        dst_block_ids: {dst_block_ids} \n \
        #                        src_block_node_ids: {src_block_node_ids} \n"
        #)
        task_info_list = []

        if transfer_op.transfer_type == TransferType.PEERH2H:
            groups = group_blocks_by_node_and_segment(
                src_block_ids, dst_block_ids, src_block_node_ids
            )
            task_info_list = self._dist_cpu_op_parser(
                groups, layer_id, layer_granularity
            )
        elif transfer_op.transfer_type == TransferType.PEERSSD2H:
            groups = group_blocks_by_node(
                src_block_ids, dst_block_ids, src_block_node_ids
            )
            task_info_list = self._dist_ssd_op_parser(groups)
        else:
            raise RuntimeError(
                f"Unsurpported transfer_type {transfer_op.transfer_type} in PEER2CPUTransferWorker"
            )

        return task_info_list

    #========================== distrbuted ssd related ==========================
    #========================== local behaviors
    def _dist_ssd_op_parser(self, groups: Dict[int, Dict[str, List[int]]]):
        """
        Distributed ssd op parser
        1. for each segment, get the remote ssd blocks and local cpu blocks
        2. create RDMATaskInfo for each segment
        Args:
            groups (Dict[int, Dict[str, List[int]]]): the grouped blocks

        Returns:
            task_info_list: the list of RDMATaskInfo, each task refers to one data transfer operation
        """
        ## parse ssd
        # TODO: now we only support blockwise layout, need support layerwise layout

        task_info_list = []

        for node_id, segment in groups.items():
            ##NOTE: for ssd scenario, each node will only have one set of src and dst block ids
            peer_node_info = self.get_node_meta(node_id)
            if peer_node_info is None:
                return []
            peer_zmq_addr = peer_node_info.zmq_addr
            peer_engine_addr = peer_node_info.engine_addr
            assert (
                peer_zmq_addr != ""
            ), f"Node {node_id} zmq addr not found in redis server"
            
            src_blocks = segment["src"]
            dst_blocks = segment["dst"]
            assert len(src_blocks) == len(dst_blocks)
            
            data_size = self.cpu_kv_layout.get_block_stride() * self.dtype.itemsize * len(src_blocks)
            ssd_task_id = self.gen_task_id()
            task_info_list.append(
                RDMATaskInfo(
                    ssd_task_id,
                    self.mooncake_transfer_engine.get_engine_addr(),  ## for ssd transfer, peer engine addr refers to local mooncake engine
                    peer_engine_addr,
                    peer_zmq_addr,
                    None,
                    None,
                    src_blocks,
                    dst_blocks,
                    [], # not used in ssd transfer
                    data_size=data_size
                )
            )
        return task_info_list


    #=============================remote behaviors
    
    def meta_info_parser(self, recv_msg: str):
        recv_dict = json.loads(recv_msg)
        return RemoteSSD2HMetaInfo.from_dict(recv_dict)

    def ssd_handle_loop(self):
        flexkv_logger.info(
            f"Node {self.cache_config.distributed_node_id} Listening on {self.zmq_listen_addr}"
        )
        while not self.zmq_server.shutdown_event.is_set():
            try:
                ## step1: recv and parse the message into meta info
                try:
                    message = self.zmq_server.listen_socket.recv().decode("utf-8")
                except zmq.Again:
                    time.sleep(0.001)
                    continue
                if not message:
                    self.zmq_server.listen_socket.send(b"ERROR")  
                    continue

                recv_meta = self.meta_info_parser(message)
                if not recv_meta:
                    self.zmq_server.listen_socket.send(b"ERROR")  
                    flexkv_logger.warning("Can not parse RemoteSSD2HMetaInfo using recieved message")
                    continue
                
                flexkv_logger.info(
                    f"[ssd_handle_loop] Received task_id={recv_meta.task_id}, "
                    f"ssd_block_ids={recv_meta.ssd_block_ids}, "
                    f"cpu_block_ids={recv_meta.cpu_block_ids}"
                )
                
                self.zmq_server.listen_socket.send(b"OK")  
                
                failure_msg = NotifyMsg(mooncake_engine_addr=self.mooncake_transfer_engine.get_engine_addr(),
                                                task_id=recv_meta.task_id, status = NotifyStatus.FAIL) ## used when transfer fails
                success_msg = NotifyMsg(mooncake_engine_addr=self.mooncake_transfer_engine.get_engine_addr(),
                                                task_id=recv_meta.task_id, status = NotifyStatus.SUCCESS) ## used when transfer fails
                
                # step2: ckeck the recieved info, early return if check error
                if len(recv_meta.ssd_block_ids) == 0 or len(recv_meta.cpu_block_ids) == 0 \
                    or len(recv_meta.cpu_block_ids)!=len(recv_meta.ssd_block_ids):
                        flexkv_logger.warning(
                            "Invalid cpu_block_ids or ssd_block_ids, skipping this transfer..."
                        )
                        self.zmq_server.send_transfer_status(recv_meta.peer_zmq_status_addr, failure_msg)
                        continue
                    
                # TODO: we need to support dynamic temp buffer or split the ssd transfer request if number of ssd blocks is larger than num_tmp_cpu_blocks
                # now we just refuse this transfer by returning a failure status
                if len(recv_meta.ssd_block_ids)>self.cache_config.num_tmp_cpu_blocks:
                    flexkv_logger.warning(
                            f"The number of ssd_block_ids is larger than {self.cache_config.num_tmp_cpu_blocks},  can not do transfer now"
                        )
                    self.zmq_server.send_transfer_status(recv_meta.peer_zmq_status_addr, failure_msg)
                    continue
                
                ## step3: do copy data from ssd to cpu
                # NOTE: this block ids is a corresponding relationship with self.tmp_cpu_buffer, for every transfer req we reuse the local cpu buffer
                local_cpu_buffer_block_ids = torch.arange(0, len(recv_meta.ssd_block_ids), dtype = torch.int64)
                local_cpu_start_idx = 0
                
                # seperate the blocks to get the longest continuous blocks
                groups = split_contiguous_blocks(recv_meta.ssd_block_ids, recv_meta.cpu_block_ids)

                all_copy_complete = True
                src_ptr_list = []
                dst_ptr_list = []
                data_size_list = []
                
                for item in groups:
                    # in this loop we do two things:
                    # 1. copy ssd data to cpu for each segment
                    # 2. calculate the start ptr of local cpu blocks and dst cpu blocks for each segment and record them
                    ssd_block_ids_per_seg = torch.tensor(item["src"], dtype=torch.int64)
                    dst_cpu_block_ids_per_seg = torch.tensor(item["dst"], dtype=torch.int64)
                    
                    if len(ssd_block_ids_per_seg) == 0:
                        all_copy_complete = False                     
                        break
                    # get corresponding temp cpu block ids
                    local_cpu_buffer_block_ids_per_seg = local_cpu_buffer_block_ids[local_cpu_start_idx: local_cpu_start_idx+len(ssd_block_ids_per_seg)]
                    local_cpu_start_idx += len(ssd_block_ids_per_seg)
                    
                    layer_id = recv_meta.layer_id
                    layer_granularity = recv_meta.layer_granularity
                    layer_id_list = torch.arange(
                        layer_id, layer_id + layer_granularity, dtype=torch.int32
                    )
                    if not self.copy_ssd_data_to_dram(
                        layer_id_list, ssd_block_ids_per_seg, local_cpu_buffer_block_ids_per_seg
                    ):
                        flexkv_logger.error("Copy ssd data to dram failed!")
                        all_copy_complete = False
                        break

                    src_ptrs, src_block_size = self.get_cpu_buffer_block_start_ptr(
                        local_cpu_buffer_block_ids_per_seg,
                        self.tmp_cpu_buffer.data_ptr(),
                        layer_id,
                        layer_granularity,
                    )

                    dst_ptrs, dst_block_size = self.get_cpu_buffer_block_start_ptr(
                        dst_cpu_block_ids_per_seg,
                        recv_meta.peer_cpu_base_ptr,
                        layer_id,
                        layer_granularity,
                    )
                    assert src_block_size == dst_block_size, "Block size mismatch between src and dst"

                    for _ in range(len(src_ptrs)):
                        data_size_list.append(src_block_size * len(local_cpu_buffer_block_ids_per_seg))
                    src_ptr_list.extend(src_ptrs)
                    dst_ptr_list.extend(dst_ptrs)
                    assert len(src_ptr_list) == len(data_size_list) and len(dst_ptr_list) == len(data_size_list)

                ## step4: do rdma transfer and send notify
                if not all_copy_complete:
                    self.zmq_server.send_transfer_status(recv_meta.peer_zmq_status_addr, failure_msg)
                    continue
                
                if not self.write_data_back_to_peer(recv_meta.peer_engine_addr, src_ptr_list, dst_ptr_list, data_size_list):
                    self.zmq_server.send_transfer_status(recv_meta.peer_zmq_status_addr, failure_msg)
                    flexkv_logger.error(f"Failed to write data back to peer")
                    continue
                    
                self.zmq_server.send_transfer_status(recv_meta.peer_zmq_status_addr, success_msg)

            except Exception as e:
                flexkv_logger.error(f"Unexpected error in ssd_handle_loop: {e}")
                time.sleep(0.001)

    def copy_ssd_data_to_dram(
        self, layer_id_list: torch.Tensor, ssd_block_id_list: torch.Tensor, cpu_block_id_list: torch.Tensor
    ):  
        assert len(ssd_block_id_list) == len(cpu_block_id_list)
        flexkv_logger.info(f"copy ssd blocks:{ssd_block_id_list} to cpu blocks: {cpu_block_id_list}" )
        try:
            transfer_kv_blocks_ssd(
                ioctx=self.ioctx,
                cpu_layer_id_list=layer_id_list,
                cpu_tensor_ptr=self.tmp_cpu_buffer.data_ptr(),  ## copy ssd data to tmp cpu buffer
                ssd_block_ids=ssd_block_id_list,
                cpu_block_ids=cpu_block_id_list,
                cpu_layer_stride_in_bytes=self.cpu_layer_stride_in_bytes,
                cpu_kv_stride_in_bytes=self.cpu_kv_stride_in_bytes,
                ssd_layer_stride_in_bytes=self.ssd_layer_stride_in_bytes,
                ssd_kv_stride_in_bytes=self.ssd_kv_stride_in_bytes,
                chunk_size_in_bytes=self.chunk_size_in_bytes,
                block_stride_in_bytes=self.block_stride_in_bytes,
                is_read=True,
                num_blocks_per_file=self.num_blocks_per_file,
                round_robin=self.round_robin,
                num_threads_per_device=32,
                is_mla=self.is_mla,
            )
        except Exception as e:
            flexkv_logger.error(f"Copy data from ssd to cpu failed: {e}")
            return False
        return True
    
    def write_data_back_to_peer(
        self, 
        peer_address: str, 
        src_ptr_list: List[int], 
        dst_ptr_list: List[int], 
        data_size_list: List[int]
    ):
        flexkv_logger.info(f"Write data back to peer from src: {src_ptr_list} to {dst_ptr_list}")
        ret = self.mooncake_transfer_engine.batch_transfer_sync_write(peer_address, src_ptr_list, dst_ptr_list, data_size_list)
        return True if ret == 0 else False

        
    #============================== distrbuted cpu related ==========================

    def _dist_cpu_op_parser(
        self,
        groups: Dict[int, List[Dict[str, List[int]]]],
        layer_id: int,
        layer_granularity: int,
    ):
        """
        Distributed cpu op parser
        1. for each segment, get the remote cpu ptrs and local cpu ptrs
        2. create RDMATaskInfo for each segment

        Inputs:
            groups (Dict[int, List[Dict[str, List[int]]]]): the grouped blocks
            layer_id (int): start layer id
            layer_granularity (int): number of layers to be transferred

        Returns:
            task_info_list: the list of RDMATaskInfo, each task refers to the data transfer of one node
        """

        task_info_list = []

        for node_id, segments in groups.items():
            # step1: get the remote meta info
            src_meta = self.get_node_meta(node_id)
            if src_meta is None:
                return [] # NOTE: if we miss part of meta data, then we should abort this request
            peer_engine_addr = src_meta.engine_addr
            src_ptr_list = []
            dst_ptr_list = []
            data_size_list = []
            for seg in segments:
                src_blocks = seg["src"]
                dst_blocks = seg["dst"]

                # step2: calculate the src and dst block start ptrs
                src_block_start_ptrs, src_data_size_per_block = (
                    self.get_cpu_buffer_block_start_ptr(
                        src_blocks,
                        src_meta.cpu_bufer_base_ptr,  # the cpu buffer ptr on remote machine
                        layer_id,
                        layer_granularity,
                    )
                )


                dst_block_start_ptrs, dst_data_size_per_block = (
                    self.get_cpu_buffer_block_start_ptr(
                        dst_blocks,
                        self.dst_buffer_ptr,  # the cpu buffer ptr on local machine
                        layer_id,
                        layer_granularity,
                    )
                )
 
                assert (
                    src_data_size_per_block == dst_data_size_per_block
                ), "src and dst blocks have different layout"
                
                
                for _ in range(len(src_block_start_ptrs)):
                    data_size = src_data_size_per_block * len(src_blocks)
                    data_size_list.append(data_size)
                src_ptr_list.extend(src_block_start_ptrs)
                dst_ptr_list.extend(dst_block_start_ptrs)
                assert len(data_size_list) == len(src_ptr_list) and len(data_size_list) == len(dst_ptr_list)
                
            flexkv_logger.info(
                f"[PEER2CPUTransferWorker]: remote cpu op parser src_ptr_list: {src_ptr_list}, dst_ptr_list: {dst_ptr_list} "
            )
            # step3: create RDMATaskInfo for each segment
            # NOTE: block wise layout: only one start ptr for each segment
            #       layer wise layout: multiple start ptrs for each segment, the number of start ptrs equals to layer_granularity * kv_dim

            task_info_list.append(
                  RDMATaskInfo(
                    0,
                    "",
                    peer_engine_addr,
                    "",
                    src_ptr_list,
                    dst_ptr_list,
                    None,
                    None,
                    data_size_list,
                    data_size = sum(data_size_list)
                )
            )
            
        return task_info_list

    #================================== utils =================================
    def get_cpu_buffer_block_start_ptr(
        self,
        cpu_blocks: List[int],
        cpu_base_ptr: int,
        layer_start_id: int = 0,
        layer_granularity: int = -1,
    ) -> Tuple[List[int], int]:
        """
        Get the cpu buffer block start ptrs for the given cpu blocks.
        We have two layout types in flexkv, layerwise and blockwise.
        1) For layerwise layout,although the cpu blocks are continous, we need to calculate the start ptrs for each layer and each kv dim. So
        the number of start ptrs equals to layer_granularity * kv_dim.
        2) For blockwise layout, the cpu blocks are continuous, so we only need to calculate the start ptr for the first block. So
        the number of start ptrs is 1.
        3) For other layout types, raise error.

        Parameters:
            cpu_blocks (List[int]): the list of cpu block ids, continuous
            cpu_base_ptr (int): the base ptr of the cpu buffer
            layer_start_id (int): the start layer id, only used for layerwise layout
            layer_granularity (int): the number of layers to be transferred, only used for layerwise layout
        Returns:
            Tuple(List[int], int): the list of cpu buffer block start ptrs and data size per block (used for calculate total data size)
        """

        # assuming that remote cpu buffer layout is the same as local cpu buffer layout
        assert self.cpu_kv_layout.type == self.remote_kv_layout.type
        src_block_ptrs = []
        
        # Get the first block ID and handle different input types
        if isinstance(cpu_blocks, torch.Tensor):
            block_id_int = int(cpu_blocks[0].item())
        elif isinstance(cpu_blocks, list):
            first_elem = cpu_blocks[0]
            if isinstance(first_elem, torch.Tensor):
                block_id_int = int(first_elem.item())
            else:
                block_id_int = int(first_elem)
        else:
            raise ValueError(f"Invalid cpu_blocks type: {type(cpu_blocks)}")
        
        if self.cpu_kv_layout.type == KVCacheLayoutType.LAYERFIRST:
            for layer_id in range(layer_start_id, layer_start_id + layer_granularity):
                for kv_id in range(self.kv_dim):
                    element_offset = (
                        (
                            ((layer_id * self.kv_dim) + kv_id)
                            * self.cpu_kv_layout.num_block
                            + block_id_int
                        )
                        * self.cpu_kv_layout.get_block_stride()
                        * self.dtype.itemsize
                    )
                    src_block_ptrs.append(cpu_base_ptr + element_offset)

        elif self.cpu_kv_layout.type == KVCacheLayoutType.BLOCKFIRST:
            block_volume = self.cpu_kv_layout.get_block_stride()
            element_offset = block_id_int * block_volume * self.dtype.itemsize
            src_block_ptrs.append(cpu_base_ptr + element_offset)
        else:
            raise ValueError(f"Invalid KVCacheLayoutType: {self.cpu_kv_layout.type}")
        data_size_per_block = self.cpu_kv_layout.get_block_stride() * self.dtype.itemsize
        
        return  src_block_ptrs, data_size_per_block

    ### redis client helper functions
    def regist_node_meta(self, cpu_buffer_base_ptr: int, ssd_buffer_base_ptr: int, zmq_addr: str):
        self.redis_meta_client.regist_node_meta(self.redis_meta_client.get_node_id(), self.mooncake_transfer_engine.get_engine_addr(), 
                                                zmq_addr, cpu_buffer_base_ptr, ssd_buffer_base_ptr)
        #NOTE: maybe useless
        node_meta_info = NodeMetaInfo(
            self.redis_meta_client.get_node_id(),
            self.mooncake_transfer_engine.get_engine_addr(),
            zmq_addr,
            cpu_buffer_base_ptr,
            ssd_buffer_base_ptr
        )
        self.node_metas[self.redis_meta_client.get_node_id()] = node_meta_info
        flexkv_logger.info(f"Registered node {self.redis_meta_client.get_node_id()} to Redis.")
    
    def unregist_node_meta(self, node_id: int = None) -> None:
        self.redis_meta_client.unregist_node_meta(self.redis_meta_client.get_node_id())
        flexkv_logger.info(f"Unregistered node {self.redis_meta_client.get_node_id()} from Redis.")

    def get_node_meta(self, node_id: int) -> Optional[NodeMetaInfo]:
        # TODO: how to remove the invalid node meta info in node_metas
        """Get the node meta info by node id."""
        if not node_id in self.node_metas:
            ## fetch from redis
            node_redis_data = self.redis_meta_client.get_node_meta(node_id)
            if not node_redis_data:
                flexkv_logger.error(f"Node {node_id} meta not found in Redis.")
                return None

            node_meta = NodeMetaInfo.from_dict(node_redis_data)

            self.node_metas[node_id] = node_meta
            flexkv_logger.info(f"Fetched node {node_id} meta from Redis.")

        return self.node_metas[node_id]

class GDSNVMfTransferWorker(TransferWorkerBase):
    def __init__(
        self,
        worker_id: int,
        transfer_conn: Connection,
        finished_ops_queue: MPQueue,
        op_buffer_tensor: torch.Tensor,

        gpu_blocks: List[TensorSharedHandle],
        gpu_kv_layout: KVCacheLayout,
        nvmf_targets: Dict[int, Dict[str, str]],
        ssd_kv_layout: KVCacheLayout, # Assume symmetric nodes with identical KV layout
        num_files: int,

        dtype: torch.dtype,
        cache_config: CacheConfig,

        gpu_device_id: int = 0
    ) -> None:
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)

        self.gpu_blocks = [wrapper.get_tensor() for wrapper in gpu_blocks]
        self.gpu_blocks_ptrs = self._get_layer_ptrs(self.gpu_blocks)
        self.gpu_layer_ptrs = self.gpu_blocks_ptrs

        self.num_files = num_files

        # Initialize Redis client
        # NOTE: We only need hiredis client for file extents.
        self.redis_client = RedisMeta(
            cache_config.redis_host,
            cache_config.redis_port,
            cache_config.redis_password,
            cache_config.local_ip
        )
        self.redis_client.set_node_id(cache_config.distributed_node_id)
        redis_channel: RedisMetaChannel = self.redis_client.get_redis_meta_channel()
        try:
            channel = getattr(redis_channel, '_c', redis_channel)
        except Exception:
            raise RuntimeError('Failed to get hiredis channel')
        # Use same round_robin as SSD transfer to ensure consistent block mapping
        self.round_robin = 1

        self.dtype = dtype
        self.is_mla = gpu_kv_layout.is_mla

        # Layout information
        self.num_layers = gpu_kv_layout.num_layer
        gpu_kv_layout_per_layer = gpu_kv_layout.div_layer(self.num_layers)
        ssd_kv_layout_per_file = ssd_kv_layout.div_block(self.num_files, padding=True)

        # GPU layout calculations
        self.chunk_size_in_bytes = gpu_kv_layout_per_layer.get_chunk_size() * self.dtype.itemsize
        self.gpu_kv_stride_in_bytes = gpu_kv_layout_per_layer.get_kv_stride() * self.dtype.itemsize
        self.gpu_block_stride_in_bytes = gpu_kv_layout_per_layer.get_block_stride() * self.dtype.itemsize
        self.gpu_layer_stride_in_bytes = gpu_kv_layout_per_layer.get_layer_stride() * self.dtype.itemsize

        # SSD layout calculations
        self.ssd_layer_stride_in_bytes = ssd_kv_layout_per_file.get_layer_stride() * self.dtype.itemsize
        self.ssd_kv_stride_in_bytes = ssd_kv_layout_per_file.get_kv_stride() * self.dtype.itemsize
        self.ssd_block_stride_in_bytes = ssd_kv_layout_per_file.get_block_stride() * self.dtype.itemsize

        # Create GDSNVMfManager from NVMe device names in this worker process
        self.gds_nvmf_manager = c_ext.GDSNVMfManager(
            nvmf_targets,
            self.chunk_size_in_bytes
            channel,
            self.round_robin
        )
        if not self.gds_nvmf_manager.is_ready():
            raise RuntimeError(f'Failed to initialize GDSNVMf Manager in worker {worker_id}: '
                               f'{self.gds_nvmf_manager.get_last_error()}')

        if len(self.gpu_blocks) == 1:
            self.gpu_block_type_ = 1  # TRTLLM
        elif len(self.gpu_blocks) == self.num_layers:
            self.gpu_block_type_ = 0  # VLLM
        elif len(self.gpu_blocks) == self.num_layers * 2:
            self.gpu_block_type_ = 2  # SGLANG
        else:
            raise ValueError(f'Invalid GPU block type: {len(self.gpu_blocks)}')

        self.gpu_device_id = gpu_device_id
        if gpu_device_id != -1:
            torch.cuda.set_device(gpu_device_id)
        self.transfer_stream = torch.cuda.Stream()

    def _transfer_impl(
        self,
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
        transfer_type: TransferType,
        layer_id: int,
        layer_granularity: int,
        **kwargs: Any
    ) -> None:
        assert transfer_type == TransferType.PEERSSD2D, 'GDSNVMfTransferWorker is read-only.'

        src_block_node_ids: np.ndarray = kwargs['src_block_node_ids']
        assert len(src_block_ids) == len(src_block_node_ids) == len(dst_block_ids)
        if len(src_block_ids) == 0:
            return

        assert src_block_ids.dtype == torch.int64
        assert dst_block_ids.dtype == torch.int64

        layer_id_list = torch.arange(layer_id, layer_id + layer_granularity, dtype=torch.int32)
        try:
            transfer_kv_blocks_gds_nvmf(
                self.gds_nvmf_manager,          # GDS manager for NVMf (contains topology info)
                layer_id_list,                  # GPU layer IDs to process
                self.gpu_layer_ptrs,            # GPU layer pointers tensor
                src_block_ids,                  # SSD block IDs
                src_block_node_ids,             #
                dst_block_ids,                  # GPU block IDs

                self.gpu_kv_stride_in_bytes,    # GPU K-V stride
                self.gpu_block_stride_in_bytes, # GPU block stride
                self.gpu_layer_stride_in_bytes, # GPU layer stride
                self.ssd_layer_stride_in_bytes, # SSD layer stride
                self.ssd_block_stride_in_bytes, # SSD block stride
                self.ssd_kv_stride_in_bytes,    # SSD K-V stride

                self.num_layers,                # Total layers

                #False,                          # Verbose logging
                self.is_mla,                    # MLA
                self.gpu_block_type_            # GPU block type
            )
        except Exception as e:
            flexkv_logger.error(f'GDSNVMf transfer failed: {e}')
            raise RuntimeError(f'Failed to transfer KV blocks: {e}') from e

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
        layer_id: int = transfer_op.layer_id
        if layer_id == -1:
            layer_id = 0

        layer_granularity: int = transfer_op.layer_granularity
        if layer_granularity == -1:
            layer_granularity = self.num_layers

        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op, pinned=False) # Pin mem?

        with torch.cuda.stream(self.transfer_stream):
            start_time = time.time()
            self._transfer_impl(
                src_block_ids,
                dst_block_ids,
                transfer_op.transfer_type,
                layer_id,
                layer_granularity,
                src_block_node_ids=transfer_op.src_block_node_ids
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
        # TODO: Check read status
        return True
