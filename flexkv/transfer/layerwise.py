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

from flexkv.c_ext import transfer_kv_blocks, transfer_kv_blocks_ssd, \
    transfer_kv_blocks_gds, TPTransferThreadGroup, TPGDSTransferThreadGroup
from flexkv.common.debug import flexkv_logger
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import TransferOp, TransferType, PartitionBlockType
from flexkv.common.transfer import get_nvtx_range_color
from flexkv.common.config import CacheConfig, GLOBAL_CONFIG_FROM_ENV

try:
    from flexkv.c_ext import transfer_kv_blocks_remote
except ImportError:
    transfer_kv_blocks_remote = None

from flexkv.transfer.worker import WorkerTransferOp
from flexkv.transfer.worker import TransferWorkerBase, cudaHostRegister

class LayerwiseTransferWorker(TransferWorkerBase):
    def __init__(self,
                 worker_id: int,
                 transfer_conn: Connection,
                 finished_ops_queue: MPQueue,
                 op_buffer_tensor: torch.Tensor,
                 gpu_blocks: List[List[TensorSharedHandle]],
                 cpu_blocks: torch.Tensor,
                 ssd_files: Dict[int, List[str]],
                 gpu_kv_layouts: List[KVCacheLayout],
                 cpu_kv_layout: KVCacheLayout,
                 ssd_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 tp_group_size: int,
                 dp_group_id: int,
                 gpu_device_id: int,
                 num_blocks_per_file: int,
                 use_ce_transfer_h2d: bool = False,
                 use_ce_transfer_d2h: bool = False,
                 transfer_sms_h2d: int = 8,
                 transfer_sms_d2h: int = 8) -> None:
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)
        assert len(gpu_blocks) == tp_group_size
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

        # initialize GPU storage
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

        self.gpu_blocks = [wrapper.get_tensor() for wrapper in gpu_blocks]

        # 确定 GPU block 类型
        if len(self.gpu_blocks) == 1:
            self.gpu_block_type_ = 1  # TRTLLM
        elif len(self.gpu_blocks) == self.num_layers:
            self.gpu_block_type_ = 0  # VLLM
        elif len(self.gpu_blocks) == self.num_layers * 2:
            self.gpu_block_type_ = 2  # SGLANG
        else:
            raise ValueError(f"Invalid GPU block type: {len(self.gpu_blocks)}")

        # initialize CPU storage
        flexkv_logger.info(f"[LayerwiseWorker-{worker_id}] Pinning CPU Memory: "
                           f"{cpu_blocks.numel() * cpu_blocks.element_size() / (1024 ** 3):.2f} GB")
        cudaHostRegister(cpu_blocks)
        self.cpu_blocks = cpu_blocks

        self.cpu_chunk_size_in_bytes = cpu_kv_layout.get_chunk_size() * self.dtype.itemsize
        self.cpu_kv_stride_in_bytes = cpu_kv_layout.get_kv_stride() * self.dtype.itemsize
        self.cpu_block_stride_in_bytes = cpu_kv_layout.get_block_stride() * self.dtype.itemsize
        self.cpu_layer_stride_in_bytes = cpu_kv_layout.get_layer_stride() * self.dtype.itemsize

        self.use_ce_transfer_h2d = use_ce_transfer_h2d
        self.use_ce_transfer_d2h = use_ce_transfer_d2h
        self.transfer_sms_h2d = transfer_sms_h2d
        self.transfer_sms_d2h = transfer_sms_d2h

        # initialize SSD storage
        self.ssd_files = ssd_files
        self.num_blocks_per_file = num_blocks_per_file
        self.num_files = sum(len(file_list) for file_list in ssd_files.values())
        self.round_robin = 1

        ssd_kv_layout_per_file = ssd_kv_layout.div_block(self.num_files, padding=True)
        self.ssd_kv_stride_in_bytes = ssd_kv_layout_per_file.get_kv_stride() * self.dtype.itemsize
        self.ssd_layer_stride_in_bytes = ssd_kv_layout_per_file.get_layer_stride() * self.dtype.itemsize
        self.ssd_block_stride_in_bytes = ssd_kv_layout_per_file.get_block_stride() * self.dtype.itemsize

        # assert self.ssd_block_stride_in_bytes == self.cpu_block_stride_in_bytes

        try:
            self.ioctx = c_ext.SSDIOCTX(
                ssd_files,
                len(ssd_files),
                GLOBAL_CONFIG_FROM_ENV.iouring_entries,
                GLOBAL_CONFIG_FROM_ENV.iouring_flags
            )
        except Exception as e:
            flexkv_logger.error(f"Error setting ssd ioctx: {e}\n")
            raise RuntimeError("SSD Worker init failed") from e

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
                      **kwargs: Any) -> None:
        pass

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> None:
        layer_id = transfer_op.layer_id
        layer_granularity = transfer_op.layer_granularity
        if layer_id == -1:
            layer_id = 0
        if layer_granularity == -1:
            layer_granularity = self.num_layers

        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op)

        self._transfer_impl(
            src_block_ids,
            dst_block_ids,
            transfer_op.transfer_type,
            layer_id,
            layer_granularity,
        )
