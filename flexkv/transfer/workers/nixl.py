"""CPU/GPU <-> file transfers driven by a NIXL agent."""
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


class NixlTransferWorker(TransferWorkerBase):
    """KV cache transfer via NIXL FILE backends: GDS_MT (GPU↔file) or POSIX / 3FS (CPU↔file).

    Both ``gpu_kv_layout`` and ``cpu_kv_layout`` are required so GPU, CPU, and SSD (per-file)
    byte strides are always defined; only the tensors needed for the chosen backend must be
    provided (``gpu_blocks`` for GDS_MT, ``cpu_blocks`` for POSIX/3FS).
    """

    def __init__(
        self,
        worker_id: int,
        transfer_conn: Connection,
        finished_ops_queue: MPQueue,
        op_buffer_tensor: torch.Tensor,
        nixl_backend: str,
        ssd_files: Dict[int, List[str]],
        num_blocks_per_file: int,
        dtype: torch.dtype,
        ssd_kv_layout: KVCacheLayout,
        gpu_kv_layout: KVCacheLayout,
        cpu_kv_layout: KVCacheLayout,
        nixl_extra_config: Optional[Dict[str, Any]] = None,
        gpu_blocks: Optional[List[TensorSharedHandle]] = None,
        cpu_blocks: Optional[torch.Tensor] = None,
        gpu_device_id: int = 0,
    ) -> None:
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)

        be = normalize_nixl_file_plugin_name(str(nixl_backend).upper())
        if be not in NIXL_GPU_FILE_BACKENDS and be not in NIXL_CPU_FILE_BACKENDS:
            raise ValueError(
                f"nixl_backend must be one of "
                f"{sorted(NIXL_GPU_FILE_BACKENDS | NIXL_CPU_FILE_BACKENDS)}, got {nixl_backend}"
            )
        if be in NIXL_GPU_FILE_BACKENDS and gpu_blocks is None:
            raise ValueError("GDS_MT requires gpu_blocks")
        if be in NIXL_CPU_FILE_BACKENDS and cpu_blocks is None:
            raise ValueError("POSIX/3FS require cpu_blocks")

        # Pin after optional GPU bind so GPU backends do not create a GPU0 context.
        if be in NIXL_GPU_FILE_BACKENDS:
            ensure_cuda_device(gpu_device_id)
        self._pin_op_buffer()
        if (
            gpu_kv_layout.num_layer != cpu_kv_layout.num_layer
            or gpu_kv_layout.kv_dim != cpu_kv_layout.kv_dim
            or gpu_kv_layout.num_kv_heads != cpu_kv_layout.num_kv_heads
        ):
            raise ValueError(
                "gpu_kv_layout and cpu_kv_layout must match on num_layer, "
                "kv_dim and num_kv_heads"
            )

        self.nixl_backend = be
        self.ssd_files = ssd_files
        self.num_blocks_per_file = num_blocks_per_file
        self.num_files = sum(len(fl) for fl in ssd_files.values())
        self.num_devices = len(ssd_files)
        self.num_files_per_device = len(ssd_files[0])
        self.round_robin = 1
        self.dtype = dtype

        self.num_layers = gpu_kv_layout.num_layer
        self.kv_dim = gpu_kv_layout.kv_dim
        self.num_kv_heads = gpu_kv_layout.num_kv_heads

        # SSD / file-side layout (same for every NIXL FILE backend).
        ssd_pf = ssd_kv_layout.div_block(self.num_files, padding=True)
        self.ssd_layer_stride_in_bytes = (
            ssd_pf.get_layer_stride() * self.dtype.itemsize
        )
        self.ssd_kv_stride_in_bytes = ssd_pf.get_kv_stride() * self.dtype.itemsize
        self.ssd_block_stride_in_bytes = (
            ssd_pf.get_block_stride() * self.dtype.itemsize
        )

        # GPU pool strides (per tensor layout).
        gpu_pl = gpu_kv_layout.div_layer(self.num_layers)
        self.gpu_chunk_size_in_bytes = gpu_pl.get_chunk_size() * self.dtype.itemsize
        self.gpu_kv_stride_in_bytes = (
            gpu_kv_layout.get_kv_stride() * self.dtype.itemsize
        )
        self.gpu_block_stride_in_bytes = (
            gpu_kv_layout.get_block_stride() * self.dtype.itemsize
        )
        self.gpu_layer_stride_in_bytes = (
            gpu_kv_layout.get_layer_stride() * self.dtype.itemsize
        )

        # CPU pool strides (DRAM side for POSIX / 3FS).
        self.cpu_chunk_size_in_bytes = (
            cpu_kv_layout.get_chunk_size() * self.dtype.itemsize
        )
        self.mem_block_stride_in_bytes = (
            cpu_kv_layout.get_block_stride() * self.dtype.itemsize
        )
        self.mem_kv_stride_in_bytes = (
            cpu_kv_layout.get_kv_stride() * self.dtype.itemsize
        )
        self.mem_layer_stride_in_bytes = (
            cpu_kv_layout.get_layer_stride() * self.dtype.itemsize
        )

        self._session = NixlAgentSession(be, nixl_extra_config or {})

        if be in NIXL_GPU_FILE_BACKENDS:
            self.gpu_blocks = import_tensor_handles(gpu_blocks)  # type: ignore[arg-type]
            if len(self.gpu_blocks) == 1:
                self.gpu_block_type_ = 1
            elif len(self.gpu_blocks) == self.num_layers:
                self.gpu_block_type_ = 0
            elif len(self.gpu_blocks) == self.num_layers * 2:
                self.gpu_block_type_ = 2
            else:
                raise ValueError(
                    f"Invalid GPU block count for NIXL: {len(self.gpu_blocks)}"
                )
            self.chunk_size_in_bytes = self.gpu_chunk_size_in_bytes
            self.gpu_device_id = gpu_device_id
            self.transfer_stream = torch.cuda.Stream()
            if not self._session.prepare_all_ssd_files(self.ssd_files):
                raise RuntimeError("NIXL: prepare_all_ssd_files failed")
            if not self._session.prepare_vram_gpu(self.gpu_blocks):
                raise RuntimeError("NIXL: prepare_vram_gpu failed")
        else:
            self.cpu_blocks = cpu_blocks  # type: ignore[assignment]
            flexkv_logger.info(
                f"NixlTransferWorker ({be}): pinning CPU pool "
                f"{cpu_blocks.numel() * cpu_blocks.element_size() / (1024 ** 3):.2f} GiB"
            )
            self._register_host_tensor(cpu_blocks, "nixl_cpu_pool")  # type: ignore[arg-type]
            if cpu_kv_layout.type != ssd_kv_layout.type:
                raise ValueError(
                    "CPU and SSD KV layout types must match for NIXL FILE transfer"
                )
            self.chunk_size_in_bytes = self.cpu_chunk_size_in_bytes
            if not self._session.prepare_all_ssd_files(self.ssd_files):
                raise RuntimeError("NIXL: prepare_all_ssd_files failed")
            if not self._session.prepare_dram_cpu(self.cpu_blocks):
                raise RuntimeError("NIXL: prepare_dram_cpu failed")

        # Bytes per KV block (all layers); used by transfer tracing for bw.
        self._bytes_per_block = self.chunk_size_in_bytes * self.num_layers * self.kv_dim

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

        if layer_id == -1:
            layer_id = 0
        if layer_granularity == -1:
            layer_granularity = self.num_layers

        if self.nixl_backend in NIXL_GPU_FILE_BACKENDS:
            if transfer_type == TransferType.DISK2D:
                ssd_block_ids, mem_block_ids = src_block_ids, dst_block_ids
                direction = "READ"
            elif transfer_type == TransferType.D2DISK:
                mem_block_ids, ssd_block_ids = src_block_ids, dst_block_ids
                direction = "WRITE"
            else:
                raise ValueError(
                    f"GDS_MT NixlTransferWorker expects DISK2D or D2DISK, got {transfer_type}"
                )
        else:
            if transfer_type == TransferType.DISK2H:
                ssd_block_ids, mem_block_ids = src_block_ids, dst_block_ids
                direction = "READ"
            elif transfer_type == TransferType.H2DISK:
                mem_block_ids, ssd_block_ids = src_block_ids, dst_block_ids
                direction = "WRITE"
            else:
                raise ValueError(
                    f"POSIX/3FS NixlTransferWorker expects DISK2H or H2DISK, got {transfer_type}"
                )

        n = ssd_block_ids.numel()
        if n == 0:
            return

        kv_dim = self.kv_dim
        layer_end = layer_id + layer_granularity

        file_paths: List[str] = []
        region_offsets: List[int] = []
        region_lens: List[int] = []

        if self.nixl_backend in NIXL_GPU_FILE_BACKENDS:
            gpu_tensors: List[torch.Tensor] = []
            for i in range(n):
                ssd_b = int(ssd_block_ids[i].item())
                mem_b = int(mem_block_ids[i].item())
                path, block_in_file = file_path_for_ssd_block(
                    self.ssd_files,
                    ssd_b,
                    self.num_devices,
                    self.num_files_per_device,
                    self.round_robin,
                )
                for lid in range(layer_id, layer_end):
                    for kv in range(kv_dim):
                        sob = ssd_chunk_byte_offset_in_file(
                            lid,
                            kv,
                            block_in_file,
                            self.ssd_layer_stride_in_bytes,
                            self.ssd_kv_stride_in_bytes,
                            self.ssd_block_stride_in_bytes,
                            self.kv_dim,
                        )
                        gview = gpu_chunk_u8_view(
                            self.gpu_blocks,
                            self.gpu_block_type_,
                            self.num_layers,
                            mem_b,
                            lid,
                            kv,
                            self.gpu_kv_stride_in_bytes,
                            self.gpu_block_stride_in_bytes,
                            self.gpu_layer_stride_in_bytes,
                            self.chunk_size_in_bytes,
                            self.kv_dim,
                        )
                        gpu_tensors.append(gview)
                        file_paths.append(path)
                        region_offsets.append(sob)
                        region_lens.append(self.chunk_size_in_bytes)

            ok = self._session.xfer_vram_file(
                direction, gpu_tensors, file_paths, region_lens, region_offsets
            )
            if not ok:
                raise RuntimeError("NIXL GDS_MT transfer failed")
            torch.cuda.synchronize()
        else:
            base = self.cpu_blocks.data_ptr()
            dram_ptr_len: List[Tuple[int, int]] = []
            for i in range(n):
                ssd_b = int(ssd_block_ids[i].item())
                mem_b = int(mem_block_ids[i].item())
                path, block_in_file = file_path_for_ssd_block(
                    self.ssd_files,
                    ssd_b,
                    self.num_devices,
                    self.num_files_per_device,
                    self.round_robin,
                )
                for lid in range(layer_id, layer_end):
                    for kv in range(kv_dim):
                        cob = kv_chunk_byte_offset_in_block(
                            lid,
                            kv,
                            mem_b,
                            self.mem_layer_stride_in_bytes,
                            self.mem_kv_stride_in_bytes,
                            self.mem_block_stride_in_bytes,
                            self.kv_dim,
                        )
                        sob = ssd_chunk_byte_offset_in_file(
                            lid,
                            kv,
                            block_in_file,
                            self.ssd_layer_stride_in_bytes,
                            self.ssd_kv_stride_in_bytes,
                            self.ssd_block_stride_in_bytes,
                            self.kv_dim,
                        )
                        dram_ptr_len.append((base + cob, self.chunk_size_in_bytes))
                        file_paths.append(path)
                        region_offsets.append(sob)
                        region_lens.append(self.chunk_size_in_bytes)

            ok = self._session.xfer_dram_file(
                direction, dram_ptr_len, file_paths, region_lens, region_offsets
            )
            if not ok:
                raise RuntimeError(f"NIXL {self.nixl_backend} CPU↔file transfer failed")

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
        lid = transfer_op.layer_id
        lg = transfer_op.layer_granularity
        if lid == -1:
            lid = 0
        if lg == -1:
            lg = self.num_layers

        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op)

        # GDS_MT runs NIXL VRAM xfers on a dedicated stream; POSIX/3FS are CPU-only.
        stream_ctx = (
            torch.cuda.stream(self.transfer_stream)
            if self.nixl_backend in NIXL_GPU_FILE_BACKENDS
            else contextlib.nullcontext()
        )
        with stream_ctx:
            start_time = time.time()
            self._transfer_impl(
                src_block_ids,
                dst_block_ids,
                transfer_op.transfer_type,
                lid,
                lg,
            )
            end_time = time.time()
            kv_dim = self.kv_dim
            transfer_size = (
                self.chunk_size_in_bytes * lg * transfer_op.valid_block_num * kv_dim
            )
            self._log_transfer_performance(
                transfer_op, transfer_size, start_time, end_time
            )
        return True


