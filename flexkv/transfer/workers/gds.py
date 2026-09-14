"""GPU <-> SSD transfers over GPUDirect Storage.

``GDSTransferWorker`` is the non-TP path and ``tpGDSTransferWorker`` the TP
path, mirroring the GPU<->CPU pair.
"""
import contextlib
import logging
import math
import os
import copy
import signal

import torch.multiprocessing as mp
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from torch.multiprocessing import Queue as MPQueue, Pipe as MPPipe
from multiprocessing.connection import Connection
from threading import Thread
from typing import List, Any, Dict, Union, Optional, Tuple

import numpy as np
import nvtx
import torch
import zmq
import json

from flexkv import c_ext

from flexkv.c_ext import transfer_kv_blocks, transfer_kv_blocks_ssd, TPTransferThreadGroup

# GDS imports are optional (only available when compiled with FLEXKV_ENABLE_GDS=1)
try:
    from flexkv.c_ext import transfer_kv_blocks_gds, TPGDSTransferThreadGroup
except ImportError:
    transfer_kv_blocks_gds = None
    TPGDSTransferThreadGroup = None

from flexkv.common.debug import flexkv_logger
from flexkv.common.memory_handle import TensorSharedHandle, release_vmm_tensor
from flexkv.common.storage import KVCacheLayout, KVCacheLayoutType
from flexkv.common.transfer import TransferOp, TransferType, PartitionBlockType
from flexkv.common.transfer import get_nvtx_range_color, LayerwiseTransferOp
from flexkv.common.config import (
    CacheConfig, GLOBAL_CONFIG_FROM_ENV, MooncakeTransferEngineConfig, LayerGroupSpec,
)
from flexkv.storage.allocator import HugePageTensorHandle, materialize_worker_tensor
from flexkv.transfer.host_buffer import (
    allocate_host_buffer,
    cudaHostRegister,
    safe_cuda_host_unregister,
)


from flexkv.transfer.compression.common.strategy import (
    CompressionStrategy,
    NullCompressionStrategy,
)
from flexkv.transfer.worker_op import (
    WorkerLayerwiseTransferOp,
    WorkerTransferOp,
    WorkerTransferResult,
)
from flexkv.transfer import trace
from flexkv.mooncakeEngineWrapper import MoonCakeTransferEngineWrapper
from flexkv.external.mooncake_store_keys import PoolKind, build_key
from flexkv.external.mooncake_fault_inject import inject_mooncake_fault, is_mooncake_fault_inject_enabled
from flexkv.transfer.zmqHelper import NotifyMsg, NotifyStatus, SSDZMQServer, SSDZMQClient
from flexkv.cache.redis_meta import RedisMeta
from flexkv.transfer.utils import (
    group_blocks_by_node_and_segment,
    group_blocks_by_node,
    split_contiguous_blocks,
    RemoteSSD2HMetaInfo,
    NodeMetaInfo,
    RDMATaskInfo,
)
from flexkv.transfer.nixlutil import (
    NIXL_CPU_FILE_BACKENDS,
    NIXL_GPU_FILE_BACKENDS,
    NixlAgentSession,
    normalize_nixl_file_plugin_name,
    file_path_for_ssd_block,
    gpu_chunk_u8_view,
    kv_chunk_byte_offset_in_block,
    ssd_chunk_byte_offset_in_file,
)
try:
    from flexkv.c_ext import (
        transfer_kv_blocks_remote,
        shared_transfer_kv_blocks_remote_read,
    )
except ImportError:
    transfer_kv_blocks_remote = None
    shared_transfer_kv_blocks_remote_read = None


from flexkv.transfer.workers.runtime import TransferWorkerBase, ensure_cuda_device, import_tensor_handles


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
        layer_groups: Optional[List[LayerGroupSpec]] = None,
        gpu_blocks_per_group: Optional[List[List[TensorSharedHandle]]] = None,
        gpu_layouts_per_group: Optional[List[KVCacheLayout]] = None,
    ) -> None:
        """
        Initialize GDS Transfer Worker
        """
        # Initialize base class first
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)

        ensure_cuda_device(gpu_device_id)
        self._pin_op_buffer()
        self.gpu_blocks = import_tensor_handles(gpu_blocks)
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
        self.kv_dim = gpu_kv_layout.kv_dim
        self.num_kv_heads = gpu_kv_layout.num_kv_heads
        self.has_multi_group = layer_groups is not None

        # Layout information
        self.num_layers = gpu_kv_layout.num_layer

        if self.has_multi_group:
            self._init_multi_group_gds(
                gpu_kv_layout, ssd_kv_layout, layer_groups,
                gpu_blocks_per_group, gpu_layouts_per_group)
        else:
            gpu_kv_layout_per_layer = gpu_kv_layout.div_layer(self.num_layers)
            ssd_kv_layout_per_file = ssd_kv_layout.div_block(self.num_files, padding=True)

            # GPU layout calculations — compute strides from actual tensor to handle
            # different attention backend layouts (flash_attn vs triton/flashinfer).
            self.chunk_size_in_bytes = gpu_kv_layout_per_layer.get_chunk_size() * self.dtype.itemsize
            # Bytes per KV block (all layers); used by transfer tracing for bw.
            self._bytes_per_block = self.chunk_size_in_bytes * self.num_layers * self.kv_dim
            gpu_strides = self._get_gpu_strides_from_tensor(
                self.gpu_blocks[0], gpu_kv_layout.tokens_per_block,
                self.dtype.itemsize, self.kv_dim,
            ) if len(self.gpu_blocks) > 1 else None
            if gpu_strides is not None:
                self.gpu_kv_stride_in_bytes = gpu_strides[0]
                self.gpu_block_stride_in_bytes = gpu_strides[1]
                self.gpu_layer_stride_in_bytes = gpu_strides[2]
            else:
                self.gpu_kv_stride_in_bytes = gpu_kv_layout.get_kv_stride() * self.dtype.itemsize
                self.gpu_block_stride_in_bytes = gpu_kv_layout.get_block_stride() * self.dtype.itemsize
                self.gpu_layer_stride_in_bytes = gpu_kv_layout.get_layer_stride() * self.dtype.itemsize

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
        self.transfer_stream = torch.cuda.Stream()

    def _init_multi_group_gds(
        self,
        gpu_kv_layout: KVCacheLayout,
        ssd_kv_layout: KVCacheLayout,
        layer_groups: List[LayerGroupSpec],
        gpu_blocks_per_group: Optional[List[List[TensorSharedHandle]]],
        gpu_layouts_per_group: Optional[List[KVCacheLayout]],
    ) -> None:
        """Initialize per-group GDS transfer parameters.

        SSD buffer is byte-flat (uint8) in multi-group mode; per-group strides
        use g.dtype.itemsize so groups with different element sizes (e.g.
        bf16 main + uint8 indexer) interleave correctly within a block.
        """
        kv_dim = self.kv_dim
        tpb = ssd_kv_layout.tokens_per_block

        # Multi-group BLOCKFIRST: get_block_stride() returns bytes_per_block
        # directly (already accounts for tp_size and per-group dtype sizes).
        self.ssd_block_stride_in_bytes = ssd_kv_layout.get_block_stride()

        self.group_gds_params: list = []
        # Keep imported CUDA-IPC tensors alive: _get_layer_ptrs() records only
        # raw data_ptr()s below, so dropping the tensors would free the IPC
        # mapping and dangle the stored pointers.
        self._multi_group_gpu_blocks_keepalive: list = []
        ssd_offset_bytes = 0

        for gi, g in enumerate(layer_groups):
            # Per-group dtype: indexer uses uint8 even when main KV is bf16/fp16.
            dtype_size_g = g.dtype.itemsize
            # Compressed groups: tpb_g = tpb // compress_ratio.
            tpb_g = tpb // g.compress_ratio
            chunk_elements = tpb_g * g.num_kv_heads * g.head_size
            ssd_layer_stride = kv_dim * chunk_elements * dtype_size_g
            ssd_kv_stride = chunk_elements * dtype_size_g

            # GPU strides from per-group layout — compute from actual tensor to
            # handle different attention backend layouts (flash_attn vs triton).
            if gpu_layouts_per_group is not None:
                gpu_layout = gpu_layouts_per_group[gi]
                group_gpu_blocks = import_tensor_handles(gpu_blocks_per_group[gi])
                gpu_strides = self._get_gpu_strides_from_tensor(
                    group_gpu_blocks[0], tpb_g, dtype_size_g, self.kv_dim,
                ) if len(group_gpu_blocks) > 1 else None
                if gpu_strides is not None:
                    gpu_kv_stride, gpu_block_stride, gpu_layer_stride = gpu_strides
                else:
                    gpu_kv_stride = gpu_layout.get_kv_stride() * dtype_size_g
                    gpu_block_stride = gpu_layout.get_block_stride() * dtype_size_g
                    gpu_layer_stride = gpu_layout.get_layer_stride() * dtype_size_g
                gpu_chunk_size = chunk_elements * dtype_size_g
            else:
                gpu_kv_stride = self.gpu_kv_stride_in_bytes
                gpu_block_stride = self.gpu_block_stride_in_bytes
                gpu_layer_stride = self.gpu_layer_stride_in_bytes
                gpu_chunk_size = chunk_elements * dtype_size_g

            # GPU pointers for this group
            if gpu_blocks_per_group is not None:
                group_gpu_blocks = import_tensor_handles(gpu_blocks_per_group[gi])
                self._multi_group_gpu_blocks_keepalive.append(group_gpu_blocks)
                group_gpu_ptrs = self._get_layer_ptrs(group_gpu_blocks)
            else:
                group_gpu_ptrs = self.gpu_layer_ptrs

            self.group_gds_params.append({
                'num_layers': g.num_layers,
                'gpu_ptrs': group_gpu_ptrs,
                'gpu_kv_stride': gpu_kv_stride,
                'gpu_block_stride': gpu_block_stride,
                'gpu_layer_stride': gpu_layer_stride,
                'chunk_size': gpu_chunk_size,
                'ssd_layer_stride': ssd_layer_stride,
                'ssd_kv_stride': ssd_kv_stride,
                'ssd_copy_offset': ssd_offset_bytes,
            })

            ssd_offset_bytes += g.num_layers * kv_dim * chunk_elements * dtype_size_g

        flexkv_logger.info(
            f"GDSTransferWorker multi-group initialized: {len(layer_groups)} groups, "
            f"ssd_block_stride={self.ssd_block_stride_in_bytes} bytes"
        )

    def _transfer_impl(
        self,
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
        transfer_type: TransferType,
        **kwargs: Any,
    ) -> None:
        """Implement actual transfer between GPU and SSD"""
        assert src_block_ids.dtype == torch.int64
        assert dst_block_ids.dtype == torch.int64
        assert len(src_block_ids) == len(dst_block_ids)

        # SSD uses DISK2D/D2DISK transfer types (same as traditional SSD I/O)
        if transfer_type == TransferType.DISK2D:
            ssd_block_id_list = src_block_ids
            gpu_block_id_list = dst_block_ids
        elif transfer_type == TransferType.D2DISK:
            gpu_block_id_list = src_block_ids
            ssd_block_id_list = dst_block_ids
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type} for GDSTransferWorker. "
                             f"Expected DISK2D or D2DISK.")

        if len(ssd_block_id_list) == 0:
            return

        is_read = (transfer_type == TransferType.DISK2D)

        try:
            if self.has_multi_group:
                for gp in self.group_gds_params:
                    g_gpu = gpu_block_id_list
                    g_ssd = ssd_block_id_list

                    layer_id_list = torch.arange(0, gp['num_layers'], dtype=torch.int32)
                    transfer_kv_blocks_gds(
                        self.gds_manager,
                        layer_id_list,
                        gp['gpu_ptrs'],
                        g_ssd,
                        g_gpu,
                        gp['gpu_kv_stride'],
                        gp['gpu_block_stride'],
                        gp['gpu_layer_stride'],
                        gp['ssd_layer_stride'],
                        self.ssd_block_stride_in_bytes,
                        gp['ssd_kv_stride'],
                        gp['chunk_size'],
                        gp['ssd_copy_offset'],
                        self.num_blocks_per_file,
                        gp['num_layers'],
                        is_read,
                        False,
                        self.kv_dim,
                        self.gpu_block_type_,
                        self.gpu_device_id,
                    )
            else:
                # Uniform: whole-model transfer
                layer_id_list = torch.arange(0, self.num_layers, dtype=torch.int32)
                transfer_kv_blocks_gds(
                    self.gds_manager,
                    layer_id_list,
                    self.gpu_layer_ptrs,
                    ssd_block_id_list,
                    gpu_block_id_list,
                    self.gpu_kv_stride_in_bytes,
                    self.gpu_block_stride_in_bytes,
                    self.gpu_layer_stride_in_bytes,
                    self.ssd_layer_stride_in_bytes,
                    self.ssd_block_stride_in_bytes,
                    self.ssd_kv_stride_in_bytes,
                    self.chunk_size_in_bytes,
                    0,
                    self.num_blocks_per_file,
                    self.num_layers,
                    is_read,
                    False,
                    self.kv_dim,
                    self.gpu_block_type_,
                    self.gpu_device_id,
                )

        except Exception as e:
            flexkv_logger.error(f"GDS transfer failed: {e}")
            raise RuntimeError(f"Failed to transfer KV blocks: {e}") from e

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
        """Launch a GDS transfer operation"""
        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op)

        with torch.cuda.stream(self.transfer_stream):
            start_time = time.time()
            self._transfer_impl(
                src_block_ids,
                dst_block_ids,
                transfer_op.transfer_type,
            )
            end_time = time.time()

            if self.has_multi_group:
                transfer_size = 0
                for gp in self.group_gds_params:
                    transfer_size += gp['chunk_size'] * gp['num_layers'] * transfer_op.valid_block_num * self.kv_dim
            else:
                transfer_size = self.chunk_size_in_bytes * self.num_layers * transfer_op.valid_block_num * self.kv_dim

            self._log_transfer_performance(
                transfer_op,
                transfer_size,
                start_time,
                end_time,
            )
        return True


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
        layer_groups: Optional[List[LayerGroupSpec]] = None,
        gpu_blocks_per_group: Optional[List[List[List[TensorSharedHandle]]]] = None,
        gpu_layouts_per_group: Optional[List[List[KVCacheLayout]]] = None,
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
            tp_group_size: Effective tp-group size on this node
                (``effective_tp_size_per_node`` =
                ``tp_size_per_node × cp_size_per_node``).
            layer_groups: Optional per-group KV layouts for heterogeneous models
                (including DSA/NSA indexer-as-group).
        """
        # Initialize base class first
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)

        assert len(gpu_blocks) == tp_group_size
        if gpu_blocks and gpu_blocks[0]:
            ensure_cuda_device(gpu_blocks[0][0].device)
        self._pin_op_buffer()
        # Handle tensor import for multi-process case — set_device per GPU first.
        imported_gpu_blocks = []
        for handles_in_one_gpu in gpu_blocks:
            imported_gpu_blocks.append(import_tensor_handles(handles_in_one_gpu))
        self.gpu_blocks = imported_gpu_blocks
        self.num_blocks_per_file = num_blocks_per_file
        self.num_files = sum(len(file_list) for file_list in ssd_files.values())

        self.dtype = dtype
        self.kv_dim = gpu_kv_layouts[0].kv_dim
        self.num_kv_heads = gpu_kv_layouts[0].num_kv_heads
        self.num_gpus = len(self.gpu_blocks)
        self.tp_group_size = tp_group_size
        self.has_multi_group = layer_groups is not None

        # Layout information
        self.num_layers = gpu_kv_layouts[0].num_layer

        if self.has_multi_group:
            self._init_tp_multi_group_gds(
                gpu_kv_layouts, ssd_kv_layout, layer_groups,
                gpu_blocks_per_group, gpu_layouts_per_group,
                ssd_files)
        else:
            ssd_kv_layout_per_file = ssd_kv_layout.div_block(self.num_files, padding=True)
            self.ssd_chunk_size_in_bytes = ssd_kv_layout_per_file.get_chunk_size() * self.dtype.itemsize
            self.chunk_size_in_bytes = self.ssd_chunk_size_in_bytes
            self.ssd_block_stride_in_bytes = ssd_kv_layout_per_file.get_block_stride() * self.dtype.itemsize
            # Bytes per KV block (all layers); used by transfer tracing for bw.
            self._bytes_per_block = self.chunk_size_in_bytes * self.num_layers * self.kv_dim
            if self.num_kv_heads > 1:
                ssd_kv_layout_per_file = ssd_kv_layout_per_file.div_head(self.tp_group_size)

            # GPU layout calculations — compute strides from actual tensor to handle
            # different attention backend layouts (flash_attn vs triton/flashinfer).
            dtype_sz = self.dtype.itemsize
            tpb = gpu_kv_layouts[0].tokens_per_block
            self.gpu_chunk_sizes_in_bytes = []
            self.gpu_kv_strides_in_bytes = []
            self.gpu_block_strides_in_bytes = []
            self.gpu_layer_strides_in_bytes = []
            for i, gpu_kv_layout in enumerate(gpu_kv_layouts):
                gpu_strides = self._get_gpu_strides_from_tensor(
                    self.gpu_blocks[i][0], tpb, dtype_sz, self.kv_dim,
                ) if len(self.gpu_blocks[i]) > 1 else None
                if gpu_strides is not None:
                    kv_s, blk_s, layer_s = gpu_strides
                else:
                    kv_s = gpu_kv_layout.get_kv_stride() * dtype_sz
                    blk_s = gpu_kv_layout.get_block_stride() * dtype_sz
                    layer_s = gpu_kv_layout.get_layer_stride() * dtype_sz
                self.gpu_chunk_sizes_in_bytes.append(gpu_kv_layout.get_chunk_size() * dtype_sz)
                self.gpu_kv_strides_in_bytes.append(kv_s)
                self.gpu_block_strides_in_bytes.append(blk_s)
                self.gpu_layer_strides_in_bytes.append(layer_s)

            # SSD layout calculations
            self.ssd_layer_stride_in_bytes = ssd_kv_layout_per_file.get_layer_stride() * self.dtype.itemsize
            self.ssd_kv_stride_in_bytes = ssd_kv_layout_per_file.get_kv_stride() * self.dtype.itemsize
            self.ssd_tp_stride_in_bytes = (self.ssd_block_stride_in_bytes // self.tp_group_size
                                           if self.num_kv_heads > 1 else self.ssd_block_stride_in_bytes)

            # Resolve pointers in Python
            gpu_block_ptrs_flat = [
                self.gpu_blocks[i][j].data_ptr()
                for i in range(self.num_gpus)
                for j in range(len(self.gpu_blocks[i]))
            ]
            gpu_device_ids = [self.gpu_blocks[i][0].device.index for i in range(self.num_gpus)]
            num_tensors_per_gpu = len(self.gpu_blocks[0])

            # Create TP GDS Transfer Thread Group
            self.tp_gds_transfer_thread_group = TPGDSTransferThreadGroup(
                self.num_gpus,
                gpu_block_ptrs_flat,
                num_tensors_per_gpu,
                ssd_files,
                self.num_layers,
                self.gpu_kv_strides_in_bytes,
                self.gpu_block_strides_in_bytes,
                self.gpu_layer_strides_in_bytes,
                self.gpu_chunk_sizes_in_bytes,
                gpu_device_ids,
            )

    def _init_tp_multi_group_gds(
        self,
        gpu_kv_layouts: List[KVCacheLayout],
        ssd_kv_layout: KVCacheLayout,
        layer_groups: List[LayerGroupSpec],
        gpu_blocks_per_group: Optional[List[List[List[TensorSharedHandle]]]],
        gpu_layouts_per_group: Optional[List[List[KVCacheLayout]]],
        ssd_files: Dict[int, List[str]],
    ) -> None:
        """Initialize per-group TPGDSTransferThreadGroup instances.

        SSD buffer is byte-flat (uint8) in multi-group mode; per-group strides
        use g.dtype.itemsize so groups with different element sizes (e.g.
        bf16 main + uint8 indexer) interleave correctly within a block.
        """
        kv_dim = self.kv_dim
        tpb = ssd_kv_layout.tokens_per_block

        # Multi-group BLOCKFIRST: get_block_stride() returns bytes_per_block
        # directly (already accounts for tp_size and per-group dtype sizes).
        self.ssd_block_stride_in_bytes = ssd_kv_layout.get_block_stride()

        self.group_tp_gds_params: list = []
        # Keep imported CUDA-IPC tensors alive: only data_ptr()s are recorded
        # below, so dropping the tensors would free the IPC mapping and dangle
        # the stored pointers.
        self._multi_group_gpu_blocks_keepalive: list = []
        ssd_offset_bytes = 0

        gpu_device_ids = [self.gpu_blocks[i][0].device.index for i in range(self.num_gpus)]

        for gi, g in enumerate(layer_groups):
            # Per-group dtype: indexer uses uint8 even when main KV is bf16/fp16.
            dtype_size_g = g.dtype.itemsize
            # Compressed groups: tpb_g = tpb // compress_ratio.
            tpb_g = tpb // g.compress_ratio
            chunk_elements = tpb_g * g.num_kv_heads * g.head_size
            ssd_layer_stride = kv_dim * chunk_elements * dtype_size_g
            ssd_kv_stride = chunk_elements * dtype_size_g
            # TP stride for SSD: partition the block across TP ranks
            ssd_tp_stride = self.ssd_block_stride_in_bytes // self.tp_group_size if self.num_kv_heads > 1 \
                else self.ssd_block_stride_in_bytes

            # Per-group GPU strides and pointers
            if gpu_blocks_per_group is not None and gpu_layouts_per_group is not None:
                gpu_kv_strides = []
                gpu_block_strides = []
                gpu_layer_strides = []
                gpu_chunk_sizes = []
                gpu_ptrs_flat = []
                num_tensors = None

                for gpu_idx in range(self.num_gpus):
                    grp_layout = gpu_layouts_per_group[gi][gpu_idx]
                    grp_handles = gpu_blocks_per_group[gi][gpu_idx]
                    grp_tensors = [h.get_tensor() for h in grp_handles]
                    self._multi_group_gpu_blocks_keepalive.append(grp_tensors)

                    gpu_strides = self._get_gpu_strides_from_tensor(
                        grp_tensors[0], tpb_g, dtype_size_g, self.kv_dim,
                    ) if len(grp_tensors) > 1 else None
                    if gpu_strides is not None:
                        kv_s, blk_s, layer_s = gpu_strides
                    else:
                        kv_s = grp_layout.get_kv_stride() * dtype_size_g
                        blk_s = grp_layout.get_block_stride() * dtype_size_g
                        layer_s = grp_layout.get_layer_stride() * dtype_size_g
                    gpu_kv_strides.append(kv_s)
                    gpu_block_strides.append(blk_s)
                    gpu_layer_strides.append(layer_s)
                    gpu_chunk_sizes.append(chunk_elements * dtype_size_g)

                    for t in grp_tensors:
                        gpu_ptrs_flat.append(t.data_ptr())
                    if num_tensors is None:
                        num_tensors = len(grp_tensors)
            else:
                gpu_kv_strides = []
                gpu_block_strides = []
                gpu_layer_strides = []
                for i, layout in enumerate(gpu_kv_layouts):
                    gpu_strides = self._get_gpu_strides_from_tensor(
                        self.gpu_blocks[i][0], tpb_g, dtype_size_g, self.kv_dim,
                    ) if len(self.gpu_blocks[i]) > 1 else None
                    if gpu_strides is not None:
                        kv_s, blk_s, layer_s = gpu_strides
                    else:
                        kv_s = layout.get_kv_stride() * dtype_size_g
                        blk_s = layout.get_block_stride() * dtype_size_g
                        layer_s = layout.get_layer_stride() * dtype_size_g
                    gpu_kv_strides.append(kv_s)
                    gpu_block_strides.append(blk_s)
                    gpu_layer_strides.append(layer_s)
                gpu_chunk_sizes = [chunk_elements * dtype_size_g] * self.num_gpus
                gpu_ptrs_flat = [
                    self.gpu_blocks[i][j].data_ptr()
                    for i in range(self.num_gpus)
                    for j in range(len(self.gpu_blocks[i]))
                ]
                num_tensors = len(self.gpu_blocks[0])

            tp_gds_group = TPGDSTransferThreadGroup(
                self.num_gpus,
                gpu_ptrs_flat,
                num_tensors,
                ssd_files,
                g.num_layers,
                gpu_kv_strides,
                gpu_block_strides,
                gpu_layer_strides,
                gpu_chunk_sizes,
                gpu_device_ids,
            )

            self.group_tp_gds_params.append({
                'num_layers': g.num_layers,
                'tp_gds_group': tp_gds_group,
                'ssd_layer_stride': ssd_layer_stride,
                'ssd_kv_stride': ssd_kv_stride,
                'ssd_tp_stride': ssd_tp_stride,
                'ssd_copy_offset': ssd_offset_bytes,
            })

            ssd_offset_bytes += g.num_layers * kv_dim * chunk_elements * dtype_size_g

        flexkv_logger.info(
            f"tpGDSTransferWorker multi-group initialized: {len(layer_groups)} groups, "
            f"ssd_block_stride={self.ssd_block_stride_in_bytes} bytes"
        )

    def _transfer_impl(self,
                       src_block_ids: torch.Tensor,
                       dst_block_ids: torch.Tensor,
                       transfer_type: TransferType,
                       **kwargs: Any,
                       ) -> None:
        assert src_block_ids.dtype == torch.int64
        assert dst_block_ids.dtype == torch.int64
        assert len(src_block_ids) == len(dst_block_ids)

        # GDS uses DISK2D/D2DISK transfer types
        if transfer_type == TransferType.D2DISK:
            gpu_block_ids = src_block_ids
            ssd_block_ids = dst_block_ids
            is_read = False
        elif transfer_type == TransferType.DISK2D:
            gpu_block_ids = dst_block_ids
            ssd_block_ids = src_block_ids
            is_read = True
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type} for tpGDSTransferWorker. "
                             f"Expected DISK2D or D2DISK.")

        gpu_block_id_list = gpu_block_ids
        ssd_block_id_list = ssd_block_ids

        assert len(gpu_block_id_list) == len(ssd_block_id_list)

        if len(gpu_block_id_list) == 0:
            return

        if self.has_multi_group:
            for gp in self.group_tp_gds_params:
                gp['tp_gds_group'].tp_group_transfer(
                    gpu_block_id_list,
                    ssd_block_id_list,
                    gp['ssd_layer_stride'],
                    gp['ssd_kv_stride'],
                    self.ssd_block_stride_in_bytes,
                    gp['ssd_tp_stride'],
                    self.num_blocks_per_file,
                    is_read,
                    0,  # layer_id always 0 for per-group
                    gp['num_layers'],
                    self.kv_dim,
                    self.num_kv_heads,
                )
        else:
            self.tp_gds_transfer_thread_group.tp_group_transfer(
                gpu_block_id_list,
                ssd_block_id_list,
                self.ssd_layer_stride_in_bytes,
                self.ssd_kv_stride_in_bytes,
                self.ssd_block_stride_in_bytes,
                self.ssd_tp_stride_in_bytes,
                self.num_blocks_per_file,
                is_read,
                0,
                self.num_layers,
                self.kv_dim,
                self.num_kv_heads,
            )

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
        """Launch a TP GDS transfer operation"""
        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op)

        start_time = time.time()
        self._transfer_impl(
            src_block_ids,
            dst_block_ids,
            transfer_op.transfer_type,
        )
        end_time = time.time()

        if self.has_multi_group:
            transfer_size = 0
            for gp in self.group_tp_gds_params:
                transfer_size += gp['ssd_kv_stride'] * gp['num_layers'] * transfer_op.valid_block_num * self.kv_dim
        else:
            transfer_size = self.ssd_chunk_size_in_bytes * self.num_layers * transfer_op.valid_block_num * self.kv_dim

        self._log_transfer_performance(
            transfer_op,
            transfer_size,
            start_time,
            end_time,
        )

        return True


