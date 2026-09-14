"""GPU <-> CPU transfer workers.

``GPUCPUTransferWorker`` is the non-TP path (one device, a Python-side stream)
and ``tpGPUCPUTransferWorker`` the TP path (``TPTransferThreadGroup``, one
thread and one stream per rank).
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


def _validate_multi_group_chunk_layout(
    group_chunk_size: int,
    layout_chunk_size: int,
    group_index: int,
    group_tpb: int,
    layout_tpb: int,
    head_size: int,
    compress_ratio: int,
) -> None:
    """Reject a transfer descriptor that disagrees with GPU storage."""
    if group_chunk_size != layout_chunk_size:
        raise ValueError(
            "Multi-group chunk/layout mismatch for group "
            f"{group_index}: group_chunk={group_chunk_size} B, "
            f"layout_chunk={layout_chunk_size} B, "
            f"group_tpb={group_tpb}, layout_tpb={layout_tpb}, "
            f"head_size={head_size}, compress_ratio={compress_ratio}"
        )



class GPUCPUTransferWorker(TransferWorkerBase):  # this worker only supports non-tp and non-dp case
    def __init__(self,
                 worker_id: int,
                 transfer_conn: Connection,
                 finished_ops_queue: MPQueue,
                 op_buffer_tensor: torch.Tensor,
                 gpu_blocks: List[TensorSharedHandle],
                 cpu_blocks: Union[torch.Tensor, HugePageTensorHandle],
                 gpu_kv_layout: KVCacheLayout,
                 cpu_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 gpu_device_id: int,
                 use_ce_transfer_h2d: bool = False,
                 use_ce_transfer_d2h: bool = False,
                 transfer_num_cta_h2d: int = 4,
                 transfer_num_cta_d2h: int = 4,
                 compressor: Optional[CompressionStrategy] = None,
                 layer_groups: Optional[List[LayerGroupSpec]] = None,
                 gpu_blocks_per_group: Optional[List[List[TensorSharedHandle]]] = None,
                 gpu_layouts_per_group: Optional[List[KVCacheLayout]] = None) -> None:
        # initialize worker in a new process
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)

        # Bind CUDA device BEFORE host-register / IPC import / Stream creation.
        ensure_cuda_device(gpu_device_id)

        self._pin_op_buffer()
        # Register CPU tensors with CUDA
        cpu_blocks = materialize_worker_tensor(cpu_blocks)
        flexkv_logger.info(f"Pinning CPU Memory: {cpu_blocks.numel() * cpu_blocks.element_size() / (1024 ** 3):.2f} GB")
        self._register_host_tensor(cpu_blocks, "cpu_kv_pool")

        self.gpu_device_id = gpu_device_id
        self._gpu_block_count = len(gpu_blocks)
        self.gpu_blocks = import_tensor_handles(gpu_blocks)
        # Get pointers first
        self.gpu_blocks_ptrs = self._get_layer_ptrs(self.gpu_blocks)
        self.gpu_tensor_ptrs = self.gpu_blocks_ptrs

        self.cpu_tensor = cpu_blocks

        self.dtype = dtype
        self.kv_dim = gpu_kv_layout.kv_dim
        self.num_kv_heads = gpu_kv_layout.num_kv_heads
        self.cpu_is_blockfirst = (
            cpu_kv_layout.type == KVCacheLayoutType.BLOCKFIRST
        )

        self.num_layers = gpu_kv_layout.num_layer
        self.layer_groups = layer_groups

        if layer_groups is not None and gpu_blocks_per_group is not None and gpu_layouts_per_group is not None:
            # Multi-group mode: compute per-group strides
            self._init_multi_group(
                gpu_blocks_per_group, gpu_layouts_per_group,
                cpu_kv_layout, layer_groups,
            )
        else:
            # Uniform mode: existing code path
            self.group_transfer_params = None

            # a chunk can be located by layer_id * layer_stride + kv_id * kv_stride + block_id * block_stride
            self.chunk_size_in_bytes = gpu_kv_layout.get_chunk_size() * self.dtype.itemsize
            # Bytes per KV block (all layers); used by transfer tracing for bw.
            self._bytes_per_block = self.chunk_size_in_bytes * self.num_layers * self.kv_dim

            # Compute GPU strides from actual tensor to handle different attention
            # backend layouts (flash_attn: [2,N,B,H,D], triton: [N,2,B,H,D]).
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

            self.cpu_layer_stride_in_bytes = cpu_kv_layout.get_layer_stride() * self.dtype.itemsize
            self.cpu_kv_stride_in_bytes = cpu_kv_layout.get_kv_stride() * self.dtype.itemsize
            self.cpu_block_stride_in_bytes = cpu_kv_layout.get_block_stride() * self.dtype.itemsize

        # gpu_block_type_ is a framework-level tag (0=VLLM, 1=TRTLLM, 2=SGLANG).
        # In multi-group mode all groups share the same per-layer GPU layout, so
        # judge from any one group; the flat self.gpu_blocks would over-count.
        if self.group_transfer_params is None:
            ref_blocks_len = len(self.gpu_blocks)
            ref_num_layers = self.num_layers
        else:
            ref_blocks_len = len(gpu_blocks_per_group[0])
            ref_num_layers = layer_groups[0].num_layers

        if ref_blocks_len == 1:
            self.gpu_block_type_ = 1
        elif ref_blocks_len == ref_num_layers:
            self.gpu_block_type_ = 0
        elif ref_blocks_len == ref_num_layers * 2:
            self.gpu_block_type_ = 2
        else:
            raise ValueError(
                f"Invalid GPU block type: ref_blocks_len={ref_blocks_len}, "
                f"ref_num_layers={ref_num_layers}"
            )

        self.transfer_stream = torch.cuda.Stream()
        self.transfer_num_cta_h2d = transfer_num_cta_h2d
        self.transfer_num_cta_d2h = transfer_num_cta_d2h
        self.use_ce_transfer_h2d = use_ce_transfer_h2d
        self.use_ce_transfer_d2h = use_ce_transfer_d2h

        self.ce_path_opt = GLOBAL_CONFIG_FROM_ENV.ce_path_opt
        self.ce_segment_threshold = GLOBAL_CONFIG_FROM_ENV.ce_segment_threshold
        self.ce_enable_memcpy2d = GLOBAL_CONFIG_FROM_ENV.enable_ce_memcpy2d

        self._compressor = compressor or NullCompressionStrategy()
        self._compressor.attach(self)

    def _init_multi_group(
        self,
        gpu_blocks_per_group: List[List[TensorSharedHandle]],
        gpu_layouts_per_group: List[KVCacheLayout],
        cpu_kv_layout: KVCacheLayout,
        layer_groups: List[LayerGroupSpec],
    ) -> None:
        """Initialize per-group transfer parameters for models with mixed KV shapes."""
        kv_dim = self.kv_dim
        tpb = cpu_kv_layout.tokens_per_block
        cpu_layout_type = cpu_kv_layout.type
        num_cpu_blocks = cpu_kv_layout.num_block

        # CPU buffer is sized in BYTES per block (see KVCacheLayout._compute_kv_shape).
        # cpu_kv_layout.get_block_stride() returns bytes_per_block directly for
        # multi-group BLOCKFIRST.  For LAYERFIRST, num_cpu_blocks is the contiguous
        # dim count and we keep elements-based per-group accumulation below.
        total_block_bytes = (
            cpu_kv_layout.get_block_stride()
            if cpu_layout_type == KVCacheLayoutType.BLOCKFIRST else None
        )

        self.group_transfer_params: list = []
        # Keep the imported CUDA-IPC tensors alive for the worker's lifetime.
        # _get_layer_ptrs() below records only their raw data_ptr()s; if the
        # tensors themselves were allowed to go out of scope, PyTorch would
        # release the underlying CUDA IPC mapping and the stored pointers would
        # dangle, so the per-group transfer would read/write freed device
        # memory (observed as the indexer group silently restoring zeros).
        self._multi_group_gpu_blocks_keepalive: list = []
        cpu_offset_bytes = 0  # byte offset of this group within a CPU block

        for gi, (g, gpu_layout) in enumerate(zip(layer_groups, gpu_layouts_per_group)):
            # Per-group dtype: indexer uses uint8 even when main KV is bf16/fp16.
            dtype_size_g = g.dtype.itemsize

            # Resolve GPU tensors for this group
            group_gpu_blocks = import_tensor_handles(gpu_blocks_per_group[gi])
            self._multi_group_gpu_blocks_keepalive.append(group_gpu_blocks)
            group_gpu_ptrs = self._get_layer_ptrs(group_gpu_blocks)

            # Compressed groups: GPU tensor's tokens dim equals tpb_g, not tpb.
            tpb_g = tpb // g.compress_ratio
            chunk_elements = tpb_g * g.num_kv_heads * g.head_size

            # GPU strides: compute from actual tensor to handle different
            # attention backend layouts (flash_attn vs triton/flashinfer).
            gpu_chunk_size = chunk_elements * dtype_size_g
            # Fail closed before submitting a native transfer if the
            # declarative LayerGroupSpec disagrees with the actual tensor
            # layout. This catches page-packed GLM DSA indexer buffers
            # (tpb=1, one 8448-byte row) being described as tpb=64.
            layout_chunk_size = gpu_layout.get_chunk_size() * dtype_size_g
            _validate_multi_group_chunk_layout(
                gpu_chunk_size,
                layout_chunk_size,
                gi,
                tpb_g,
                gpu_layout.tokens_per_block,
                g.head_size,
                g.compress_ratio,
            )
            t0 = group_gpu_blocks[0]
            gpu_strides = self._get_gpu_strides_from_tensor(t0, tpb_g, dtype_size_g, self.kv_dim)
            if gpu_strides is not None:
                gpu_kv_stride, gpu_block_stride, gpu_layer_stride = gpu_strides
            else:
                gpu_kv_stride = gpu_layout.get_kv_stride() * dtype_size_g
                gpu_block_stride = gpu_layout.get_block_stride() * dtype_size_g
                gpu_layer_stride = gpu_layout.get_layer_stride() * dtype_size_g

            # CPU strides: depend on layout type.  All values are in bytes; the
            # CPU buffer underlying self.cpu_tensor is uint8 for multi-group.
            if cpu_layout_type == KVCacheLayoutType.BLOCKFIRST:
                # BLOCKFIRST: [num_block, bytes_per_block]; within a block,
                # data is laid out group-by-group (each group's region holds
                # its own layer0_k, layer0_v, layer1_k, ... bytes).
                cpu_layer_stride = kv_dim * chunk_elements * dtype_size_g
                cpu_block_stride = total_block_bytes
                cpu_kv_stride = chunk_elements * dtype_size_g
            else:
                # LAYERFIRST: [all_layers, kv_dim, num_block, tpb, heads, head_dim]
                cpu_layer_stride = kv_dim * num_cpu_blocks * chunk_elements * dtype_size_g
                cpu_block_stride = chunk_elements * dtype_size_g
                cpu_kv_stride = num_cpu_blocks * chunk_elements * dtype_size_g

            self.group_transfer_params.append({
                'gpu_ptrs': group_gpu_ptrs,
                'chunk_size': gpu_chunk_size,
                'gpu_kv_stride': gpu_kv_stride,
                'gpu_block_stride': gpu_block_stride,
                'gpu_layer_stride': gpu_layer_stride,
                'cpu_layer_stride': cpu_layer_stride,
                'cpu_block_stride': cpu_block_stride,
                'cpu_kv_stride': cpu_kv_stride,
                'cpu_offset_bytes': cpu_offset_bytes,
                'num_layers': g.num_layers,
                'kv_dim': gpu_layout.kv_dim,
            })

            # Advance CPU byte offset for next group.
            if cpu_layout_type == KVCacheLayoutType.BLOCKFIRST:
                cpu_offset_bytes += g.num_layers * kv_dim * chunk_elements * dtype_size_g
            else:
                cpu_offset_bytes += (
                    g.num_layers * kv_dim * num_cpu_blocks * chunk_elements * dtype_size_g
                )

        flexkv_logger.info(
            f"Multi-group transfer initialized: {len(layer_groups)} groups, "
            f"total_block_bytes={total_block_bytes}"
        )

    def _control_suspend_gpu(self, payload: Any) -> int:
        if self.group_transfer_params is not None:
            raise NotImplementedError(
                "GPU hot remap does not support multi-group KV layouts"
            )
        if not self.gpu_blocks:
            return 0
        with torch.cuda.device(self.gpu_device_id):
            torch.cuda.synchronize()
        old_blocks = self.gpu_blocks
        self.gpu_blocks = []
        self.gpu_blocks_ptrs.zero_()
        self.gpu_tensor_ptrs = self.gpu_blocks_ptrs
        released = sum(release_vmm_tensor(tensor) for tensor in old_blocks)
        if released != len(old_blocks):
            raise RuntimeError(
                f"Expected {len(old_blocks)} VMM mappings, released {released}"
            )
        return released

    def _control_resume_gpu(
        self, gpu_blocks: List[TensorSharedHandle]
    ) -> int:
        if self.gpu_blocks:
            raise RuntimeError("GPU blocks are already registered")
        if len(gpu_blocks) != self._gpu_block_count:
            raise ValueError(
                f"Expected {self._gpu_block_count} GPU blocks, "
                f"got {len(gpu_blocks)}"
            )
        self.gpu_blocks = import_tensor_handles(gpu_blocks)
        self.gpu_blocks_ptrs = self._get_layer_ptrs(self.gpu_blocks)
        self.gpu_tensor_ptrs = self.gpu_blocks_ptrs
        return len(self.gpu_blocks)

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

        if transfer_type == TransferType.H2D:
            gpu_block_id_list = dst_block_ids
            cpu_block_id_list = src_block_ids
            use_ce_transfer = self.use_ce_transfer_h2d
            transfer_num_cta = self.transfer_num_cta_h2d
        elif transfer_type == TransferType.D2H:
            gpu_block_id_list = src_block_ids
            cpu_block_id_list = dst_block_ids
            use_ce_transfer = self.use_ce_transfer_d2h
            transfer_num_cta = self.transfer_num_cta_d2h
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type} for GPUCPUTransferWorker")

        assert len(gpu_block_id_list) == len(cpu_block_id_list)

        if len(gpu_block_id_list) == 0:
            return

        if self.group_transfer_params is not None:
            # Multi-group transfer: one call per group
            for gp in self.group_transfer_params:
                gpu_ptrs = gp['gpu_ptrs'].contiguous().pin_memory()
                # Offset the CPU tensor to this group's region.  cpu_tensor is
                # uint8 in multi-group mode, so this slice is byte-addressed.
                cpu_tensor_for_group = self.cpu_tensor.view(-1)[
                    gp['cpu_offset_bytes']:
                ]

                transfer_kv_blocks(
                    gpu_block_id_list,
                    gpu_ptrs,
                    gp['gpu_kv_stride'],
                    gp['gpu_block_stride'],
                    gp['gpu_layer_stride'],
                    cpu_block_id_list,
                    cpu_tensor_for_group,
                    gp['cpu_kv_stride'],
                    gp['cpu_layer_stride'],
                    gp['cpu_block_stride'],
                    gp['chunk_size'],
                    0,                   # start_layer_id (always 0 within group)
                    gp['num_layers'],    # all layers in this group
                    transfer_num_cta,
                    transfer_type == TransferType.H2D,
                    use_ce_transfer,
                    self.kv_dim,
                    self.num_kv_heads,
                    self.gpu_block_type_,
                    True,  # sync
                    self.ce_path_opt,
                    self.ce_segment_threshold,
                    -1,  # ce_force_path
                    self.ce_enable_memcpy2d,
                    self.cpu_is_blockfirst,
                    enable_transfer_trace=GLOBAL_CONFIG_FROM_ENV.enable_transfer_trace,
                )
        else:
            # Uniform transfer: single call (whole-model)
            transfer_kv_blocks(
                gpu_block_id_list,
                self.gpu_blocks_ptrs,
                self.gpu_kv_stride_in_bytes,
                self.gpu_block_stride_in_bytes,
                self.gpu_layer_stride_in_bytes,
                cpu_block_id_list,
                self.cpu_tensor,
                self.cpu_kv_stride_in_bytes,
                self.cpu_layer_stride_in_bytes,
                self.cpu_block_stride_in_bytes,
                self.chunk_size_in_bytes,
                0,                  # start_layer_id (whole-model)
                self.num_layers,    # layer_granularity = all layers
                transfer_num_cta,
                transfer_type == TransferType.H2D,
                use_ce_transfer,
                self.kv_dim,
                self.num_kv_heads,
                self.gpu_block_type_,
                True,  # sync
                self.ce_path_opt,
                self.ce_segment_threshold,
                -1,  # ce_force_path
                self.ce_enable_memcpy2d,
                self.cpu_is_blockfirst,
                enable_transfer_trace=GLOBAL_CONFIG_FROM_ENV.enable_transfer_trace,
            )

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
        nvtx_range = nvtx.start_range(
            message=f"GPUCPUWorker.launch_transfer[{transfer_op.transfer_op_id}]",
            color="purple")

        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op)

        try:
            with torch.cuda.stream(self.transfer_stream):
                if self.group_transfer_params is not None:
                    # Multi-group (heterogeneous KV) path — compression is not
                    # supported here; issue the per-group transfers inline.
                    start_time = time.time()
                    self._transfer_impl(
                        src_block_ids,
                        dst_block_ids,
                        transfer_op.transfer_type,
                    )
                    end_time = time.time()
                    transfer_size = 0
                    for gp in self.group_transfer_params:
                        transfer_size += gp['chunk_size'] * gp['num_layers'] * transfer_op.valid_block_num * self.kv_dim
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
        finally:
            nvtx.end_range(nvtx_range)

        return True

class tpGPUCPUTransferWorker(TransferWorkerBase):
    def __init__(self,
                 worker_id: int,
                 transfer_conn: Connection,
                 finished_ops_queue: MPQueue,
                 op_buffer_tensor: torch.Tensor,
                 gpu_blocks: List[List[TensorSharedHandle]],
                 cpu_blocks: Union[torch.Tensor, HugePageTensorHandle],
                 gpu_kv_layouts: List[KVCacheLayout],
                 cpu_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 tp_group_size: int,
                 use_ce_transfer_h2d: bool = False,
                 use_ce_transfer_d2h: bool = False,
                 transfer_num_cta_h2d: int = 4,
                 transfer_num_cta_d2h: int = 4,
                 compressor: Optional[CompressionStrategy] = None,
                 layer_groups: Optional[List[LayerGroupSpec]] = None,
                 gpu_blocks_per_group: Optional[List[List[List[TensorSharedHandle]]]] = None,
                 gpu_layouts_per_group: Optional[List[List[KVCacheLayout]]] = None):

        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)
        assert len(gpu_blocks) == tp_group_size
        cpu_blocks = materialize_worker_tensor(cpu_blocks)
        # Bind primary GPU + pin op buffer before any CUDA IPC import.
        if gpu_blocks and gpu_blocks[0]:
            ensure_cuda_device(gpu_blocks[0][0].device)
        self._pin_op_buffer()
        # Handle tensor import for multi-process case — set_device per GPU first.
        imported_gpu_blocks = []
        for handles_in_one_gpu in gpu_blocks:
            imported_gpu_blocks.append(import_tensor_handles(handles_in_one_gpu))
        self._gpu_block_counts = [len(handles) for handles in gpu_blocks]
        self.gpu_blocks = imported_gpu_blocks
        self.dtype = dtype # note this should be quantized data type
        self.kv_dim = gpu_kv_layouts[0].kv_dim
        self.num_kv_heads = gpu_kv_layouts[0].num_kv_heads

        self.num_gpus = len(self.gpu_blocks)
        self.tp_group_size = tp_group_size
        self.layer_groups = layer_groups
        self.cpu_tensor = cpu_blocks

        flexkv_logger.info(f"Pinning CPU Memory: {cpu_blocks.numel() * cpu_blocks.element_size() / (1024 ** 3):.2f} GB")
        self._register_host_tensor(cpu_blocks, "tp_cpu_kv_pool")

        self.num_layers = gpu_kv_layouts[0].num_layer

        self.transfer_num_cta_h2d = transfer_num_cta_h2d
        self.transfer_num_cta_d2h = transfer_num_cta_d2h
        self.use_ce_transfer_h2d = use_ce_transfer_h2d
        self.use_ce_transfer_d2h = use_ce_transfer_d2h

        # Read KV shared across ranks D2H mode from global config
        self.kv_shared_across_ranks_mode = GLOBAL_CONFIG_FROM_ENV.kv_shared_across_ranks_mode
        flexkv_logger.debug(f"[tpGPUCPUTransferWorker] kv_shared_across_ranks_mode={self.kv_shared_across_ranks_mode}")

        if layer_groups is not None and gpu_blocks_per_group is not None and gpu_layouts_per_group is not None:
            self._init_tp_multi_group(
                gpu_blocks_per_group, gpu_layouts_per_group,
                cpu_kv_layout, layer_groups,
            )
        else:
            self.tp_group_transfer_groups = None

            # Compute GPU strides from actual tensor to handle different attention
            # backend layouts (flash_attn: [2,N,B,H,D], triton: [N,2,B,H,D]).
            # Each GPU may have different strides, so compute per-GPU.
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

            self.cpu_is_blockfirst = (
                cpu_kv_layout.type == KVCacheLayoutType.BLOCKFIRST
            )
            self.cpu_block_stride_in_bytes = cpu_kv_layout.get_block_stride() * self.dtype.itemsize
            self.cpu_chunk_size_in_bytes = cpu_kv_layout.get_chunk_size() * self.dtype.itemsize
            self.chunk_size_in_bytes = self.cpu_chunk_size_in_bytes
            # Bytes per KV block (all layers); used by transfer tracing for bw.
            self._bytes_per_block = self.chunk_size_in_bytes * self.num_layers * self.kv_dim
            # tp has effect on the layout of the cpu tensor
            # the tp dim should always be right after the block dim
            # on both blockfirst layout and layerfirst layout
            if cpu_kv_layout.type == KVCacheLayoutType.BLOCKFIRST and self.num_kv_heads > 1:
                cpu_kv_layout = cpu_kv_layout.div_head(self.tp_group_size)

            self.cpu_layer_stride_in_bytes = cpu_kv_layout.get_layer_stride() * self.dtype.itemsize
            self.cpu_kv_stride_in_bytes = cpu_kv_layout.get_kv_stride() * self.dtype.itemsize
            self.cpu_tp_stride_in_bytes = self.cpu_block_stride_in_bytes // self.tp_group_size

            # Resolve pointers in Python (where storage is valid); pass them to C++ so we avoid
            # "Tensor that doesn't have storage" when C++ calls .data_ptr() on tensors passed
            # across the pybind11 boundary from a spawn'd subprocess (shared memory / CUDA IPC).
            gpu_block_ptrs_flat = [
                self.gpu_blocks[i][j].data_ptr()
                for i in range(self.num_gpus)
                for j in range(len(self.gpu_blocks[i]))
            ]
            cpu_blocks_ptr = cpu_blocks.data_ptr()
            gpu_device_ids = [self.gpu_blocks[i][0].device.index for i in range(self.num_gpus)]
            num_tensors_per_gpu = len(self.gpu_blocks[0])

            self.tp_transfer_thread_group = TPTransferThreadGroup(
                self.num_gpus,
                gpu_block_ptrs_flat,
                num_tensors_per_gpu,
                cpu_blocks_ptr,
                self.num_layers,
                self.gpu_kv_strides_in_bytes,
                self.gpu_block_strides_in_bytes,
                self.gpu_layer_strides_in_bytes,
                self.gpu_chunk_sizes_in_bytes,
                gpu_device_ids,
                GLOBAL_CONFIG_FROM_ENV.ce_segment_threshold,
                GLOBAL_CONFIG_FROM_ENV.ce_path_opt,
                GLOBAL_CONFIG_FROM_ENV.enable_ce_memcpy2d,
                self.cpu_is_blockfirst,
                self.num_kv_heads,
                ce_gather_threads=GLOBAL_CONFIG_FROM_ENV.ce_gather_threads,
                ce_gather_nt=GLOBAL_CONFIG_FROM_ENV.ce_gather_nt,
            )

        self._compressor = compressor or NullCompressionStrategy()
        self._compressor.attach(self)

    def _init_tp_multi_group(
        self,
        gpu_blocks_per_group: List[List[List[TensorSharedHandle]]],
        gpu_layouts_per_group: List[List[KVCacheLayout]],
        cpu_kv_layout: KVCacheLayout,
        layer_groups: List[LayerGroupSpec],
    ) -> None:
        """Initialize per-group TPTransferThreadGroup instances.

        CPU buffer is byte-flat (uint8) in multi-group mode: each block has
        size kv_shape[1] = bytes_per_block (see KVCacheLayout._compute_kv_shape).
        Per-group strides use g.dtype.itemsize so groups with different element
        sizes (e.g. bf16 main + uint8 indexer) interleave correctly within a
        block.
        """
        kv_dim = self.kv_dim
        tpb = cpu_kv_layout.tokens_per_block
        cpu_layout_type = cpu_kv_layout.type
        num_cpu_blocks = cpu_kv_layout.num_block

        # For BLOCKFIRST multi-group, get_block_stride() returns bytes_per_block
        # directly (already accounts for tp_size and per-group dtype sizes).
        total_block_bytes = (
            cpu_kv_layout.get_block_stride()
            if cpu_layout_type == KVCacheLayoutType.BLOCKFIRST else None
        )

        self.tp_group_transfer_groups: list = []
        # Keep imported CUDA-IPC tensors alive for the worker's lifetime:
        # TPTransferThreadGroup below stores only their raw data_ptr()s, so if
        # the tensors were dropped PyTorch would release the IPC mapping and the
        # pointers would dangle (mirrors GPUCPUTransferWorker._init_multi_group
        # and LayerwiseWorker._init_multi_group).
        self._multi_group_gpu_blocks_keepalive: list = []
        cpu_offset_bytes = 0

        for gi, g in enumerate(layer_groups):
            # Per-group dtype: indexer uses uint8 even when main KV is bf16/fp16.
            dtype_size_g = g.dtype.itemsize

            # gpu_blocks_per_group[gi] = list of per-GPU handle lists for this group
            # gpu_blocks_per_group[gi][gpu_idx] = handles for this group on GPU gpu_idx
            group_gpu_blocks_per_gpu = gpu_blocks_per_group[gi]

            # Import tensors from handles (bind CUDA device per GPU first)
            imported_group_blocks = []
            for handles_in_one_gpu in group_gpu_blocks_per_gpu:
                imported_group_blocks.append(import_tensor_handles(handles_in_one_gpu))
            self._multi_group_gpu_blocks_keepalive.append(imported_group_blocks)

            # Build flat pointer list for this group
            gpu_block_ptrs_flat = [
                imported_group_blocks[i][j].data_ptr()
                for i in range(self.num_gpus)
                for j in range(len(imported_group_blocks[i]))
            ]
            gpu_device_ids = [imported_group_blocks[i][0].device.index for i in range(self.num_gpus)]
            num_tensors_per_gpu = len(imported_group_blocks[0])

            # Compressed groups: GPU tensor's tokens dim equals tpb_g.
            tpb_g = tpb // g.compress_ratio

            # Per-group GPU strides: compute from actual tensor to handle different
            # attention backend layouts (flash_attn vs triton/flashinfer).
            group_gpu_layouts = gpu_layouts_per_group[gi]  # one layout per GPU
            gpu_kv_strides = []
            gpu_block_strides = []
            gpu_layer_strides = []
            gpu_chunk_sizes = []
            for i, layout in enumerate(group_gpu_layouts):
                gpu_strides = self._get_gpu_strides_from_tensor(
                    imported_group_blocks[i][0], tpb_g, dtype_size_g, self.kv_dim,
                ) if len(imported_group_blocks[i]) > 1 else None
                if gpu_strides is not None:
                    kv_s, blk_s, layer_s = gpu_strides
                else:
                    kv_s = layout.get_kv_stride() * dtype_size_g
                    blk_s = layout.get_block_stride() * dtype_size_g
                    layer_s = layout.get_layer_stride() * dtype_size_g
                gpu_kv_strides.append(kv_s)
                gpu_block_strides.append(blk_s)
                gpu_layer_strides.append(layer_s)
                gpu_chunk_sizes.append(layout.get_chunk_size() * dtype_size_g)

            chunk_elements = tpb_g * g.num_kv_heads * g.head_size

            # CPU strides for this group (all in bytes)
            if cpu_layout_type == KVCacheLayoutType.BLOCKFIRST:
                cpu_block_stride = total_block_bytes
                cpu_layer_stride = kv_dim * chunk_elements * dtype_size_g
                cpu_kv_stride = chunk_elements * dtype_size_g
                cpu_tp_stride = cpu_block_stride // self.tp_group_size
            else:
                cpu_block_stride = chunk_elements * dtype_size_g
                cpu_layer_stride = kv_dim * num_cpu_blocks * chunk_elements * dtype_size_g
                cpu_kv_stride = num_cpu_blocks * chunk_elements * dtype_size_g
                cpu_tp_stride = cpu_block_stride // self.tp_group_size

            # CPU tensor offset for this group (cpu_tensor is uint8 in multi-group)
            cpu_blocks_ptr = self.cpu_tensor.view(-1)[cpu_offset_bytes:].data_ptr()

            tp_thread_group = TPTransferThreadGroup(
                self.num_gpus,
                gpu_block_ptrs_flat,
                num_tensors_per_gpu,
                cpu_blocks_ptr,
                g.num_layers,
                gpu_kv_strides,
                gpu_block_strides,
                gpu_layer_strides,
                gpu_chunk_sizes,
                gpu_device_ids,
                GLOBAL_CONFIG_FROM_ENV.ce_segment_threshold,
                GLOBAL_CONFIG_FROM_ENV.ce_path_opt,
                GLOBAL_CONFIG_FROM_ENV.enable_ce_memcpy2d,
                (cpu_layout_type == KVCacheLayoutType.BLOCKFIRST),
                self.num_kv_heads,
            )

            self.tp_group_transfer_groups.append({
                'tp_thread_group': tp_thread_group,
                'cpu_kv_stride': cpu_kv_stride,
                'cpu_layer_stride': cpu_layer_stride,
                'cpu_block_stride': cpu_block_stride,
                'cpu_tp_stride': cpu_tp_stride,
                'cpu_offset_bytes': cpu_offset_bytes,
                'num_layers': g.num_layers,
                'chunk_size': chunk_elements * dtype_size_g,
            })

            # Advance CPU byte offset for next group
            if cpu_layout_type == KVCacheLayoutType.BLOCKFIRST:
                cpu_offset_bytes += g.num_layers * kv_dim * chunk_elements * dtype_size_g
            else:
                cpu_offset_bytes += (
                    g.num_layers * kv_dim * num_cpu_blocks * chunk_elements * dtype_size_g
                )

        flexkv_logger.info(
            f"TP multi-group transfer initialized: {len(layer_groups)} groups, "
            f"total_block_bytes={total_block_bytes}"
        )


    def _control_suspend_gpu(self, payload: Any) -> int:
        if self.tp_group_transfer_groups is not None:
            raise NotImplementedError(
                "GPU hot remap does not support multi-group KV layouts"
            )
        if not self.gpu_blocks:
            return 0
        zero_ptrs = [0] * sum(self._gpu_block_counts)
        self.tp_transfer_thread_group.update_gpu_block_ptrs(zero_ptrs)
        old_blocks = self.gpu_blocks
        self.gpu_blocks = []
        released = sum(
            release_vmm_tensor(tensor)
            for blocks_in_one_gpu in old_blocks
            for tensor in blocks_in_one_gpu
        )
        expected = sum(self._gpu_block_counts)
        if released != expected:
            raise RuntimeError(
                f"Expected {expected} VMM mappings, released {released}"
            )
        return released

    def _control_resume_gpu(
        self, gpu_blocks: List[List[TensorSharedHandle]]
    ) -> int:
        if self.gpu_blocks:
            raise RuntimeError("GPU blocks are already registered")
        counts = [len(handles) for handles in gpu_blocks]
        if counts != self._gpu_block_counts:
            raise ValueError(
                f"Expected GPU block counts {self._gpu_block_counts}, got {counts}"
            )
        imported_gpu_blocks = [
            import_tensor_handles(handles) for handles in gpu_blocks
        ]
        gpu_block_ptrs_flat = [
            tensor.data_ptr()
            for blocks_in_one_gpu in imported_gpu_blocks
            for tensor in blocks_in_one_gpu
        ]
        self.tp_transfer_thread_group.update_gpu_block_ptrs(
            gpu_block_ptrs_flat
        )
        self.gpu_blocks = imported_gpu_blocks
        return len(gpu_block_ptrs_flat)

    def _transfer_impl(self,
                       src_block_ids: torch.Tensor,
                       dst_block_ids: torch.Tensor,
                       transfer_type: TransferType,
                       **kwargs: Any,
                       )->None:
        assert src_block_ids.dtype == torch.int64
        assert dst_block_ids.dtype == torch.int64
        assert len(src_block_ids) == len(dst_block_ids)

        if transfer_type == TransferType.H2D:
            gpu_block_id_list = dst_block_ids
            cpu_block_id_list = src_block_ids
            use_ce_transfer = self.use_ce_transfer_h2d
            transfer_num_cta = self.transfer_num_cta_h2d
        elif transfer_type == TransferType.D2H:
            gpu_block_id_list = src_block_ids
            cpu_block_id_list = dst_block_ids
            use_ce_transfer = self.use_ce_transfer_d2h
            transfer_num_cta = self.transfer_num_cta_d2h
        else:
            raise ValueError(f"Invalid transfer type: {transfer_type} for tpGPUCPUTransferWorker")


        assert len(gpu_block_id_list) == len(cpu_block_id_list)

        if len(gpu_block_id_list) == 0:
            return

        if self.tp_group_transfer_groups is not None:
            # Multi-group transfer: one call per group
            for gp in self.tp_group_transfer_groups:
                g_gpu = gpu_block_id_list
                g_cpu = cpu_block_id_list

                gp['tp_thread_group'].tp_group_transfer(
                    g_gpu,
                    g_cpu,
                    gp['cpu_kv_stride'],
                    gp['cpu_layer_stride'],
                    gp['cpu_block_stride'],
                    gp['cpu_tp_stride'],
                    transfer_num_cta,
                    transfer_type == TransferType.H2D,
                    use_ce_transfer,
                    0,                 # start_layer_id (always 0 within group)
                    gp['num_layers'],  # all layers in this group
                    self.kv_dim,
                    self.num_kv_heads,
                    self.kv_shared_across_ranks_mode,
                )
        else:
            self.tp_transfer_thread_group.tp_group_transfer(
                gpu_block_id_list,
                cpu_block_id_list,
                self.cpu_kv_stride_in_bytes,
                self.cpu_layer_stride_in_bytes,
                self.cpu_block_stride_in_bytes,
                self.cpu_tp_stride_in_bytes,
                transfer_num_cta,
                transfer_type == TransferType.H2D,
                use_ce_transfer,
                0,                  # start_layer_id (whole-model)
                self.num_layers,    # layer_granularity = all layers
                self.kv_dim,
                self.num_kv_heads,
                self.kv_shared_across_ranks_mode,
            )


    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op)
        if self.tp_group_transfer_groups is not None:
            # Multi-group (heterogeneous KV) path — compression not supported here.
            start_time = time.time()
            self._transfer_impl(
                src_block_ids,
                dst_block_ids,
                transfer_op.transfer_type,
            )
            end_time = time.time()

            transfer_size = 0
            for gp in self.tp_group_transfer_groups:
                transfer_size += gp['chunk_size'] * gp['num_layers'] * transfer_op.valid_block_num * self.kv_dim

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

