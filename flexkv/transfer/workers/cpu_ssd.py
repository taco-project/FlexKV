"""CPU <-> local SSD transfers (io_uring)."""
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


from flexkv.transfer.workers.runtime import TransferWorkerBase


class CPUSSDDiskTransferWorker(TransferWorkerBase):
    def __init__(self,
                 worker_id: int,
                 transfer_conn: Connection,
                 finished_ops_queue: MPQueue,
                 op_buffer_tensor: torch.Tensor,
                 cpu_blocks: Union[torch.Tensor, HugePageTensorHandle],
                 ssd_files: Dict[int, List[str]],  # ssd_device_id -> file_paths
                 cpu_kv_layout: KVCacheLayout,
                 ssd_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 num_blocks_per_file: int,
                 cache_config: CacheConfig,
                 compressor: Optional[CompressionStrategy] = None,
                 layer_groups: Optional[List[LayerGroupSpec]] = None):
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)
        self._pin_op_buffer()
        cpu_blocks = materialize_worker_tensor(cpu_blocks)
        self.ssd_files = ssd_files
        self.num_blocks_per_file = num_blocks_per_file
        self.num_files = sum(len(file_list) for file_list in ssd_files.values())

        self.num_layers = cpu_kv_layout.num_layer
        self.num_cpu_blocks = cpu_kv_layout.num_block
        self.round_robin = 1

        self.dtype = dtype

        self.cpu_blocks = cpu_blocks
        self.cpu_layer_ptrs = self._get_layer_ptrs(cpu_blocks)

        self.kv_dim = cpu_kv_layout.kv_dim
        self.num_kv_heads = cpu_kv_layout.num_kv_heads
        self.cpu_layout_type = cpu_kv_layout.type
        self.has_multi_group = layer_groups is not None

        if cpu_kv_layout.type != ssd_kv_layout.type:
            raise ValueError("no support for different CPU and SSD KV cache layout type")

        if self.has_multi_group:
            self._init_multi_group_ssd(cpu_kv_layout, ssd_kv_layout, layer_groups)
        else:
            ssd_kv_layout_per_file = ssd_kv_layout.div_block(self.num_files, padding=True)

            self.chunk_size_in_bytes = cpu_kv_layout.get_chunk_size() * self.dtype.itemsize
            self.block_stride_in_bytes = cpu_kv_layout.get_block_stride() * self.dtype.itemsize
            self.cpu_kv_stride_in_bytes = cpu_kv_layout.get_kv_stride() * self.dtype.itemsize
            self.cpu_layer_stride_in_bytes = cpu_kv_layout.get_layer_stride() * self.dtype.itemsize
            self.ssd_kv_stride_in_bytes = ssd_kv_layout_per_file.get_kv_stride() * self.dtype.itemsize
            self.ssd_layer_stride_in_bytes = ssd_kv_layout_per_file.get_layer_stride() * self.dtype.itemsize
            # Bytes per KV block (all layers); used by transfer tracing for bw.
            self._bytes_per_block = self.chunk_size_in_bytes * self.num_layers * self.kv_dim

        try:
            self.ioctx = c_ext.SSDIOCTX(ssd_files, len(ssd_files), GLOBAL_CONFIG_FROM_ENV.iouring_entries,
                GLOBAL_CONFIG_FROM_ENV.iouring_flags)
        except Exception as e:
            flexkv_logger.error(f"Error setting ssd ioctx: {e}\n")
            raise RuntimeError("SSD Worker init failed") from e

        self._compressor = compressor or NullCompressionStrategy()
        self._compressor.attach(self)

    def _init_multi_group_ssd(
        self,
        cpu_kv_layout: KVCacheLayout,
        ssd_kv_layout: KVCacheLayout,
        layer_groups: List[LayerGroupSpec],
    ) -> None:
        """Initialize CPU<->SSD multi-group parameters.

        CPU and SSD share an identical per-block byte layout (BLOCKFIRST),
        so multi-group SSD transfers move whole blocks as opaque blobs —
        no per-group / per-tp_rank slicing needed at the IO layer.
        """
        # Multi-group BLOCKFIRST: get_block_stride() returns bytes_per_block
        # directly (already accounts for tp_size and per-group dtype sizes).
        self.block_stride_in_bytes = cpu_kv_layout.get_block_stride()

        flexkv_logger.info(
            f"CPUSSDDiskTransferWorker multi-group initialized: {len(layer_groups)} groups, "
            f"block_stride={self.block_stride_in_bytes} bytes"
        )

    def _transfer_impl(
        self,
        src_block_ids: torch.Tensor,
        dst_block_ids: torch.Tensor,
        transfer_type: TransferType,
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

        is_read = (transfer_type == TransferType.DISK2H)
        cpu_base_ptr = self.cpu_layer_ptrs[0].item()

        if self.has_multi_group:
            # CPU and SSD share an identical per-block byte layout in multi-group
            # mode, so each block can be transferred as one opaque blob — no
            # per-group / per-tp_rank loop needed. num_layers=1,
            # layer_stride=chunk_size=block_stride, one KV region makes
            # the kernel issue exactly one pread/pwrite of block_stride bytes
            # per block, sidestepping the sub-4KiB chunk hazard for highly
            # compressed groups (e.g. DSv4 indexer at compress_ratio=128).
            #
            # kv_dim=1 for the same reason: block_stride already covers K and V,
            # and kv_stride is 0, so looping the kv axis would repeat the very
            # same I/O.
            one_layer_id = torch.tensor([0], dtype=torch.int32)
            transfer_kv_blocks_ssd(
                self.ioctx,
                one_layer_id,
                cpu_base_ptr,
                ssd_block_id_list,
                cpu_block_id_list,
                self.block_stride_in_bytes,
                0,
                self.block_stride_in_bytes,
                0,
                self.block_stride_in_bytes,
                self.block_stride_in_bytes,
                is_read,
                self.num_blocks_per_file,
                self.round_robin,
                32,
                1,  # kv_dim
                ssd_io_opt=GLOBAL_CONFIG_FROM_ENV.ssd_io_opt,
            )
        else:
            layer_id_list = torch.arange(0, self.num_layers, dtype=torch.int32)

            transfer_kv_blocks_ssd(
                self.ioctx,
                layer_id_list,
                cpu_base_ptr,
                ssd_block_id_list,
                cpu_block_id_list,
                self.cpu_layer_stride_in_bytes,
                self.cpu_kv_stride_in_bytes,
                self.ssd_layer_stride_in_bytes,
                self.ssd_kv_stride_in_bytes,
                self.chunk_size_in_bytes,
                self.block_stride_in_bytes,
                is_read,
                self.num_blocks_per_file,
                self.round_robin,
                32,
                self.kv_dim,
                ssd_io_opt=GLOBAL_CONFIG_FROM_ENV.ssd_io_opt,
            )

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op)
        if self.has_multi_group:
            # Multi-group (heterogeneous KV) path — compression not supported here.
            start_time = time.time()
            self._transfer_impl(
                src_block_ids,
                dst_block_ids,
                transfer_op.transfer_type,
            )
            end_time = time.time()
            # Total transfer size across all groups
            transfer_size = self.block_stride_in_bytes * transfer_op.valid_block_num
            self._log_transfer_performance(
                transfer_op,
                transfer_size,
                start_time,
                end_time,
            )
        else:
            # Uniform path — supports (optional) nvcomp compression.
            self._compressor.run(
                self, src_block_ids=src_block_ids,
                dst_block_ids=dst_block_ids, op=transfer_op)
        return True

