"""CPU <-> local SSD transfers.

Native engine is io_uring (``c_ext.SSDIOCTX`` + ``transfer_kv_blocks_ssd``); a
``StorageBackend`` replaces it wholesale when one is supplied.
"""

import time
from multiprocessing.connection import Connection
from typing import Any, Dict, List, Optional, Union

import torch
from torch.multiprocessing import Queue as MPQueue

from flexkv import c_ext
from flexkv.c_ext import transfer_kv_blocks_ssd
from flexkv.common.config import (
    CacheConfig,
    GLOBAL_CONFIG_FROM_ENV,
    LayerGroupSpec,
)
from flexkv.common.debug import flexkv_logger
from flexkv.common.storage import KVCacheLayout
from flexkv.common.transfer import TransferType
from flexkv.storage.allocator import HugePageTensorHandle, materialize_worker_tensor
from flexkv.transfer.backends import StorageBackend
from flexkv.transfer.compression.common.strategy import (
    CompressionStrategy,
    NullCompressionStrategy,
)
from flexkv.transfer.geometry import (
    ChunkStrides,
    DiskSide,
    EdgeGeometry,
    HostSide,
)
from flexkv.transfer.worker_op import WorkerTransferOp
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
                 layer_groups: Optional[List[LayerGroupSpec]] = None,
                 backend: Optional[StorageBackend] = None):
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

        self.cpu_kv_layout = cpu_kv_layout
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
            # Per-file block stride: unused by the io_uring kernel (which takes
            # num_blocks_per_file and derives it), but a backend addressing the
            # file directly needs it, and deriving it here keeps the two views
            # of the same file from drifting.
            self.ssd_block_stride_in_bytes = (
                ssd_kv_layout_per_file.get_block_stride() * self.dtype.itemsize)

        if backend is None:
            # io_uring is this edge's native engine; a backend replaces it
            # wholesale, so do not open descriptors it will never use.
            try:
                self.ioctx = c_ext.SSDIOCTX(ssd_files, len(ssd_files), GLOBAL_CONFIG_FROM_ENV.iouring_entries,
                    GLOBAL_CONFIG_FROM_ENV.iouring_flags)
            except Exception as e:
                flexkv_logger.error(f"Error setting ssd ioctx: {e}\n")
                raise RuntimeError("SSD Worker init failed") from e

        self._compressor = compressor or NullCompressionStrategy()
        self._compressor.attach(self)
        self._attach_backend(backend, self._build_geometry())

    def _build_geometry(self) -> EdgeGeometry:
        """This edge, in the terms every CPU<->SSD engine reads it in.

        Under a heterogeneous layout there is no uniform (layer, kv) chunk on
        either side -- both blocks are opaque blobs of identical byte layout,
        which is exactly what makes the native io_uring path able to move them
        as one I/O -- so both ``strides`` are None and a chunk-addressing
        engine is refused by name.
        """
        cpu_block_stride = self.block_stride_in_bytes
        if self.has_multi_group:
            cpu_strides = ssd_strides = None
            # CPU and SSD share an identical per-block byte layout here.
            ssd_block_stride = cpu_block_stride
            bytes_per_block = cpu_block_stride
        else:
            cpu_strides = ChunkStrides(
                chunk_bytes=self.chunk_size_in_bytes,
                kv_stride=self.cpu_kv_stride_in_bytes,
                layer_stride=self.cpu_layer_stride_in_bytes,
                block_stride=cpu_block_stride,
            )
            ssd_block_stride = self.ssd_block_stride_in_bytes
            ssd_strides = ChunkStrides(
                # The two sides move the same bytes, so one chunk size covers
                # both; the SSD layout differs only in how blocks are spread
                # across files.
                chunk_bytes=self.chunk_size_in_bytes,
                kv_stride=self.ssd_kv_stride_in_bytes,
                layer_stride=self.ssd_layer_stride_in_bytes,
                block_stride=ssd_block_stride,
            )
            bytes_per_block = (
                self.chunk_size_in_bytes * self.num_layers * self.kv_dim)
        return EdgeGeometry(
            num_layers=self.num_layers,
            kv_dim=self.kv_dim,
            num_kv_heads=self.num_kv_heads,
            dtype=self.dtype,
            has_multi_group=self.has_multi_group,
            bytes_per_block=bytes_per_block,
            cpu=HostSide(
                layout=self.cpu_kv_layout,
                blocks=self.cpu_blocks,
                layer_ptrs=self.cpu_layer_ptrs,
                block_stride=cpu_block_stride,
                strides=cpu_strides,
            ),
            ssd=DiskSide(
                block_stride=ssd_block_stride,
                strides=ssd_strides,
            ),
        )

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
        if self._backend is not None:
            return self._run_backend(transfer_op)
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
