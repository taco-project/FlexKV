"""CPU <-> remote store transfers (PCFS, mooncake-store).

Also holds the mooncake registration-region helpers, which only these workers
call.
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


from flexkv.transfer.workers.runtime import TransferWorkerBase


def _split_mooncake_registration_regions(
    base_ptr: int,
    logical_size: int,
    mapped_size: int,
    block_size: int,
    max_mr_size: int,
    size_alignment: int,
    pointer_alignment: int,
    mr_split_policy: str = "strict",
) -> List[Tuple[int, int]]:
    """Split a mapped KV pool according to an explicit MR policy.

    ``strict`` keeps every region aligned to the KV block, external mapping,
    and HugePage. ``block_boundary`` keeps every KV block inside one MR while
    allowing derived MR pointers/sizes to be externally unaligned.
    """
    values = {
        "base_ptr": base_ptr,
        "logical_size": logical_size,
        "mapped_size": mapped_size,
        "block_size": block_size,
        "max_mr_size": max_mr_size,
        "size_alignment": size_alignment,
        "pointer_alignment": pointer_alignment,
    }
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if mapped_size < logical_size:
        raise ValueError(
            "HugePage mapped length is smaller than the logical CPU pool: "
            f"mapped={mapped_size}, logical={logical_size}"
        )
    if mr_split_policy not in {"strict", "block_boundary"}:
        raise ValueError(
            "mr_split_policy must be one of {'strict', 'block_boundary'}, got "
            f"{mr_split_policy!r}"
        )
    if logical_size % block_size != 0:
        raise ValueError(
            "Logical CPU pool must contain whole KV blocks: "
            f"logical_size={logical_size}, block_size={block_size}"
        )
    if base_ptr % pointer_alignment != 0:
        raise ValueError(
            "Mooncake MR base pointer is not HugePage aligned: "
            f"ptr=0x{base_ptr:x}, alignment={pointer_alignment}"
        )
    if mapped_size % size_alignment != 0:
        raise ValueError(
            "Mooncake mapped size is not externally aligned: "
            f"mapped_size={mapped_size}, alignment={size_alignment}"
        )
    if mapped_size <= max_mr_size:
        return [(base_ptr, mapped_size)]

    regions: List[Tuple[int, int]] = []
    if mr_split_policy == "block_boundary":
        mapped_padding = mapped_size - logical_size
        regular_region_size = (max_mr_size // block_size) * block_size
        final_logical_capacity = (
            (max_mr_size - mapped_padding) // block_size
        ) * block_size
        if regular_region_size <= 0 or final_logical_capacity <= 0:
            raise ValueError(
                "Mooncake max MR size cannot hold one KV block plus mapping tail: "
                f"max_mr_size={max_mr_size}, block_size={block_size}, "
                f"mapped_padding={mapped_padding}"
            )
        offset = 0
        while logical_size - offset > final_logical_capacity:
            required = logical_size - offset - final_logical_capacity
            size = min(
                regular_region_size,
                ((required + block_size - 1) // block_size) * block_size,
            )
            regions.append((base_ptr + offset, size))
            offset += size
        regions.append((base_ptr + offset, mapped_size - offset))
    else:
        region_unit = math.lcm(block_size, size_alignment, pointer_alignment)
        aligned_region_size = (max_mr_size // region_unit) * region_unit
        if aligned_region_size <= 0:
            raise ValueError(
                "Mooncake max MR size cannot hold one aligned KV region: "
                f"max_mr_size={max_mr_size}, region_unit={region_unit}"
            )
        offset = 0
        while offset < mapped_size:
            remaining = mapped_size - offset
            size = (
                remaining
                if remaining <= max_mr_size
                else aligned_region_size
            )
            regions.append((base_ptr + offset, size))
            offset += size

    for index, (ptr, size) in enumerate(regions):
        is_last = index == len(regions) - 1
        if (
            mr_split_policy == "strict"
            and not is_last
            and size % block_size != 0
        ):
            raise ValueError(
                "Non-final Mooncake MR is not KV-block aligned: "
                f"index={index}, size={size}, block_size={block_size}"
            )
        if mr_split_policy == "strict" and ptr % pointer_alignment != 0:
            raise ValueError(
                "Mooncake MR pointer is not HugePage aligned: "
                f"ptr=0x{ptr:x}, alignment={pointer_alignment}"
            )
        if mr_split_policy == "strict" and size % size_alignment != 0:
            raise ValueError(
                "Mooncake MR size is not externally aligned: "
                f"size={size}, alignment={size_alignment}"
            )
        if size > max_mr_size:
            raise ValueError(
                "Mooncake MR exceeds configured maximum: "
                f"size={size}, max_mr_size={max_mr_size}"
            )

    # Explicit last index, not ``regions[-1]``: release builds cythonize this
    # module with wraparound=False, under which a negative index on a real
    # list is not folded to len + i -- it reads off the front of the object.
    last_ptr, last_size = regions[len(regions) - 1]
    if last_ptr + last_size != base_ptr + mapped_size:
        raise ValueError("Mooncake MR split does not cover mapped extent")
    if base_ptr + logical_size > last_ptr + last_size:
        raise ValueError("Mooncake MR split does not cover logical KV pool")
    return regions


def _register_mooncake_regions(
    client: Any, regions: List[Tuple[int, int]]
) -> List[Tuple[int, int]]:
    """Register regions transactionally and roll back a partial failure."""
    registered: List[Tuple[int, int]] = []
    try:
        for ptr, size in regions:
            client.register_buffer(ptr, size)
            registered.append((ptr, size))
    except Exception:
        for ptr, _ in reversed(registered):
            try:
                client.unregister_buffer(ptr)
            except Exception as rollback_error:
                flexkv_logger.error(
                    "Mooncake MR rollback failed for "
                    f"ptr=0x{ptr:x}: {rollback_error}"
                )
        raise
    return registered


def _unregister_mooncake_regions(
    client: Any, regions: List[Tuple[int, int]]
) -> None:
    """Best-effort reverse-order cleanup for registered Mooncake MRs."""
    for ptr, size in reversed(regions):
        try:
            client.unregister_buffer(ptr)
        except Exception as error:
            flexkv_logger.error(
                "Mooncake MR unregister failed for "
                f"ptr=0x{ptr:x} size={size}: {error}"
            )
class CPURemoteTransferWorker(TransferWorkerBase):
    def __init__(self,
                 worker_id: int,
                 transfer_conn: Connection,
                 finished_ops_queue: MPQueue,
                 op_buffer_tensor: torch.Tensor,
                 cpu_blocks: Union[List[torch.Tensor], torch.Tensor, HugePageTensorHandle],
                 remote_file: List[str],
                 cpu_kv_layout: KVCacheLayout,
                 remote_kv_layout: KVCacheLayout,
                 dtype: torch.dtype,
                 remote_config_custom: Dict[str, Any],
                 enable_pcfs_sharing: bool = False):
        if transfer_kv_blocks_remote is None:
            raise RuntimeError("transfer_kv_blocks_remote not available, please build with FLEXKV_ENABLE_CFS=1")
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)
        self._pin_op_buffer()

        cpu_blocks = materialize_worker_tensor(cpu_blocks)

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

        self.has_multi_group = (
            getattr(cpu_kv_layout, "layer_groups", None) is not None
        )
        if self.has_multi_group:
            if (
                cpu_kv_layout.type != KVCacheLayoutType.BLOCKFIRST
                or remote_kv_layout.type != KVCacheLayoutType.BLOCKFIRST
            ):
                raise ValueError(
                    "Multi-group CPU/remote transfer requires BLOCKFIRST layouts"
                )
            # A heterogeneous BLOCKFIRST block is already one byte-flat blob.
            # Present it to the existing remote kernel as one MLA layer.
            self.block_size = cpu_kv_layout.get_block_stride()
            self.num_layers = 1
        else:
            self.block_size = cpu_kv_layout.get_chunk_size()
        self.dtype = dtype

        self.kv_dim = 1 if self.has_multi_group else cpu_kv_layout.kv_dim
        self.num_kv_heads = cpu_kv_layout.num_kv_heads

        self.cpu_blocks = cpu_blocks

        self.cpu_layer_ptrs = self._get_layer_ptrs(cpu_blocks)

        self.cpu_layer_stride_in_bytes = (
            self.num_cpu_blocks * self.block_size * self.dtype.itemsize * self.kv_dim
        )
        self.remote_layer_stride_in_bytes = (
            self.num_remote_blocks * self.block_size * self.dtype.itemsize * self.kv_dim
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
        # Bytes per KV block (all layers); used by transfer tracing for bw.
        self._bytes_per_block = self.chunk_size_in_bytes * self.num_layers * self.kv_dim
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
        **kwargs: Any
    ) -> None:
        assert src_block_ids.dtype == torch.int64
        assert dst_block_ids.dtype == torch.int64
        assert len(src_block_ids) == len(dst_block_ids)

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

        layer_id_list = torch.arange(0, self.num_layers, dtype=torch.int32)
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
                block_id_in_file = int(
                    ((remote_block_id_list[i] / self.round_robin) / total_file_num)
                    * self.round_robin
                    + (remote_block_id_list[i] % self.round_robin)
                )

                cfs_blocks_partition[fid].append(block_id_in_file)
                cpu_blocks_partition[fid].append(cpu_block_id_list[i].item())

            # Use the new shared transfer function
            shared_transfer_kv_blocks_remote_read(
                file_nodeids_list,
                cfs_blocks_partition,
                cpu_blocks_partition,
                layer_id_list,
                self.cpu_layer_ptrs[0].item(),
                self.cpu_layer_stride_in_bytes,
                self.cpu_kv_stride_in_bytes,
                self.remote_layer_stride_in_bytes_per_file,
                self.remote_block_stride_in_bytes,
                self.remote_kv_stride_in_bytes_per_file,
                self.chunk_size_in_bytes,
                self.num_layers,
                self.kv_dim,
                num_threads_per_file=32,
            )
        else:
            transfer_kv_blocks_remote(
                self.file_nodeid_list,
                layer_id_list,
                self.cpu_layer_ptrs[0].item(),
                remote_block_id_list,
                cpu_block_id_list,
                self.cpu_layer_stride_in_bytes,
                self.cpu_kv_stride_in_bytes,
                self.remote_layer_stride_in_bytes_per_file,
                self.remote_block_stride_in_bytes,
                self.remote_kv_stride_in_bytes_per_file,
                self.chunk_size_in_bytes,
                self.num_layers,
                (transfer_type == TransferType.REMOTE2H),
                PartitionBlockType.SEQUENTIAL.value,
                self.round_robin,
                self.num_remote_blocks_per_file,
                False,
                32,
                self.kv_dim,
            )

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> bool:
        src_block_ids, dst_block_ids = self.get_transfer_block_ids(transfer_op)

        start_time = time.time()
        self._transfer_impl(
            src_block_ids,
            dst_block_ids,
            transfer_op.transfer_type,
            src_block_node_ids=transfer_op.src_block_node_ids,
        )
        end_time = time.time()
        transfer_size = self.chunk_size_in_bytes * self.num_layers * transfer_op.valid_block_num * self.kv_dim

        self._log_transfer_performance(
            transfer_op,
            transfer_size,
            start_time,
            end_time,
        )

class MooncakeStoreTransferWorker(TransferWorkerBase):
    """Mooncake-store remote KV I/O worker (main KV and SWA pools)."""

    def __init__(
        self,
        worker_id: int,
        transfer_conn: Connection,
        finished_ops_queue: MPQueue,
        op_buffer_tensor: torch.Tensor,
        cpu_blocks: Union[List[torch.Tensor], torch.Tensor, HugePageTensorHandle],
        cpu_kv_layout: "KVCacheLayout",
        dtype: torch.dtype,
        cache_config: "CacheConfig",
        pool_kind: PoolKind = PoolKind.KV,
        override_global_segment_size: Optional[int] = None,
    ) -> None:
        super().__init__(worker_id, transfer_conn, finished_ops_queue, op_buffer_tensor)
        self.pp_rank = int(getattr(cache_config, 'mooncake_store_pp_rank', 0) or 0)
        self.pp_size = int(getattr(cache_config, 'mooncake_store_pp_size', 1) or 1)
        self.node_layer_start = int(getattr(cache_config, 'mooncake_store_node_layer_start', 0) or 0)
        self.node_layer_end = int(getattr(cache_config, 'mooncake_store_node_layer_end', 0) or 0)
        self.total_layers = int(getattr(cache_config, 'mooncake_store_total_layers', 0) or 0)
        self.pool_kind = pool_kind

        mapped_size = (
            int(cpu_blocks.aligned)
            if isinstance(cpu_blocks, HugePageTensorHandle)
            else None
        )
        cpu_blocks = materialize_worker_tensor(cpu_blocks)
        # Mooncake owns the RDMA registration and this worker only uses host
        # pointers. A second CUDA/HIP registration of the same shared pool is
        # redundant and can exhaust the host mapping budget for large caches.
        flexkv_logger.info(
            "[MooncakeStoreTransferWorker] skip CUDA host registration for "
            "the CPU KV pool; Mooncake owns the external MR"
        )
        self.cpu_layer_ptrs = self._get_layer_ptrs(cpu_blocks)
        self.num_layers: int = cpu_kv_layout.num_layer
        self.num_cpu_blocks: int = cpu_kv_layout.num_block
        self.dtype = dtype
        self.cpu_kv_layout = cpu_kv_layout
        assert self.cpu_kv_layout.type == KVCacheLayoutType.BLOCKFIRST
        self.kv_dim = cpu_kv_layout.kv_dim
        self.num_kv_heads = cpu_kv_layout.num_kv_heads
        self.cpu_blocks = cpu_blocks
        self.cache_config = cache_config
        self._cpu_buffer = cpu_blocks[0] if isinstance(cpu_blocks, (list, tuple)) else cpu_blocks
        # Opaque whole-block I/O: multi-group CPU layout is byte-flat
        # ([num_block, bytes_per_block]); get_chunk_size() is invalid there.
        self.block_size_bytes = self._block_size_bytes(cpu_kv_layout, dtype)
        # Bytes per KV block (all layers); used by transfer tracing for bw.
        self._bytes_per_block = self.block_size_bytes

        from flexkv.external.mooncake_store_utils import MooncakeStoreClient, MooncakeStoreConfig
        store_config = MooncakeStoreConfig.from_file(
            self.cache_config,
            override_global_segment_size=override_global_segment_size,
        )
        self.mooncake_client = MooncakeStoreClient(store_config)
        self._mooncake_registered_regions: List[Tuple[int, int]] = []
        base_ptr = self._cpu_buffer.data_ptr()
        logical_size = self._cpu_buffer.numel() * self._cpu_buffer.element_size()
        if mapped_size is None:
            regions = [(base_ptr, logical_size)]
        else:
            hugepage_size = int(self.cache_config.hugepage_size_bytes)
            size_alignment = int(
                os.getenv(
                    "FLEXKV_HUGEPAGE_MAPPING_ALIGNMENT_BYTES",
                    str(hugepage_size),
                )
            )
            max_mr_size = int(self.cache_config.mooncake_max_mr_size_bytes)
            regions = _split_mooncake_registration_regions(
                base_ptr=base_ptr,
                logical_size=logical_size,
                mapped_size=mapped_size,
                block_size=self.block_size_bytes,
                max_mr_size=max_mr_size,
                size_alignment=size_alignment,
                pointer_alignment=hugepage_size,
                mr_split_policy=self.cache_config.mooncake_mr_split_policy,
            )
        flexkv_logger.info(
            "[MooncakeStoreTransferWorker] registering external MRs: "
            f"logical_size={logical_size} mapped_size={mapped_size or logical_size} "
            f"mr_split_policy={self.cache_config.mooncake_mr_split_policy} "
            f"regions={regions}"
        )
        self._mooncake_registered_regions = _register_mooncake_regions(
            self.mooncake_client, regions
        )

    def shutdown(self) -> None:
        """Best-effort cleanup; tolerant of partially-failed ``__init__``."""
        try:
            client = getattr(self, "mooncake_client", None)
            regions = getattr(self, "_mooncake_registered_regions", [])
            if client is not None:
                _unregister_mooncake_regions(client, regions)
            self._mooncake_registered_regions = []
        finally:
            super().shutdown()

    @staticmethod
    def _block_size_bytes(cpu_kv_layout: "KVCacheLayout", dtype: torch.dtype) -> int:
        """Bytes per CPU block for mooncake put/get addressing.

        Multi-group BLOCKFIRST stores ``bytes_per_block`` directly in
        ``kv_shape[1]`` (via ``get_block_stride()``) — do not multiply by
        ``dtype.itemsize``. Single-group layouts still use element count ×
        itemsize.
        """
        if cpu_kv_layout.layer_groups is not None:
            return int(cpu_kv_layout.get_block_stride())
        return int(cpu_kv_layout.get_elements_per_block() * dtype.itemsize)

    def _transfer_impl(self, cpu_ptrs, block_sizes, keys,
                       transfer_type: TransferType) -> List[bool]:
        if transfer_type == TransferType.H2REMOTE:
            put_results = self.mooncake_client.batch_put(keys, cpu_ptrs, block_sizes)
            if not all(put_results):
                flexkv_logger.warning(f"Mooncake-store batch put partially failed: {put_results}")
            return put_results
        elif transfer_type == TransferType.REMOTE2H:
            get_results = self.mooncake_client.batch_get(keys, cpu_ptrs, block_sizes)

            if is_mooncake_fault_inject_enabled():
                get_results = inject_mooncake_fault(get_results, transfer_type)
                flexkv_logger.info(f"Mooncake-store batch get results after fault injection: {get_results}")

            if not all(get_results):
                flexkv_logger.warning(
                    f"Mooncake-store batch get partially failed: {get_results}")
            return get_results
        else:
            raise ValueError(
                f"MooncakeStoreTransferWorker only supports H2REMOTE/REMOTE2H, got {transfer_type}")

    def launch_transfer(self, transfer_op: WorkerTransferOp) -> WorkerTransferResult:
        expected_blocks = len(transfer_op.src_block_ids)
        block_sizes: List[int] = []
        start_time = time.time()
        try:
            if self.pool_kind == PoolKind.SWA:
                cpu_ptrs, block_sizes, keys = self._preprocess_swa(transfer_op)
            else:
                cpu_ptrs, block_sizes, keys = self._preprocess_kv(transfer_op)
            if len(keys) != expected_blocks:
                raise ValueError(
                    "Mooncake key count does not match transfer block count: "
                    f"keys={len(keys)}, blocks={expected_blocks}")
            block_results = self._transfer_impl(
                cpu_ptrs, block_sizes, keys, transfer_op.transfer_type)
            if len(block_results) != expected_blocks:
                raise ValueError(
                    "Mooncake result count does not match transfer block count: "
                    f"results={len(block_results)}, blocks={expected_blocks}")
            normalized_results = tuple(bool(result) for result in block_results)
        except Exception:
            # A completed failure must still reach the scheduler. Otherwise the
            # graph never completes and its reserved cache blocks remain leaked.
            flexkv_logger.error(
                "Mooncake transfer failed; reporting all blocks unsuccessful "
                f"for op_id={transfer_op.transfer_op_id}",
                exc_info=True,
            )
            normalized_results = (False,) * expected_blocks
        end_time = time.time()
        try:
            transfer_size = sum(block_sizes)
            self._log_transfer_performance(
                transfer_op, transfer_size, start_time, end_time)
        except Exception:
            flexkv_logger.error(
                "Mooncake transfer performance logging failed; reporting the "
                f"operation unsuccessful for op_id={transfer_op.transfer_op_id}",
                exc_info=True,
            )
            normalized_results = (False,) * expected_blocks
        if not all(normalized_results):
            flexkv_logger.warning(
                "Mooncake transfer partially failed: "
                f"op_id={transfer_op.transfer_op_id}, "
                f"successful={sum(normalized_results)}/{len(normalized_results)}")
        return WorkerTransferResult(
            transfer_op_id=transfer_op.transfer_op_id,
            block_results=normalized_results,
        )

    def _preprocess_kv(self, transfer_op: WorkerTransferOp):
        cpu_block_ids = (
            transfer_op.dst_block_ids
            if transfer_op.transfer_type == TransferType.REMOTE2H
            else transfer_op.src_block_ids
        )
        assert transfer_op.mooncake_store_block_hashes is not None
        block_size_bytes = self.block_size_bytes
        base_ptr = self._cpu_buffer.data_ptr()
        cpu_ptrs, block_sizes, keys = [], [], []
        for i, blk_id in enumerate(cpu_block_ids):
            key = build_key(
                transfer_op.mooncake_store_block_hashes[i],
                PoolKind.KV,
                pp_rank=self.pp_rank,
                pp_size=self.pp_size,
                node_layer_start=self.node_layer_start,
                node_layer_end=self.node_layer_end,
                total_layers=self.total_layers,
            )
            cpu_ptrs.append(base_ptr + int(blk_id) * block_size_bytes)
            block_sizes.append(block_size_bytes)
            keys.append(key)
        return cpu_ptrs, block_sizes, keys

    def _preprocess_swa(self, transfer_op: WorkerTransferOp):
        """Build (cpu_ptrs, sizes, keys) for the SWA mooncake lane.

        Each request contributes one (CPU slot id, tail_hash) pair; after batch
        merging, ``cpu_block_ids`` and ``mooncake_store_swa_block_hashes`` both
        hold N entries in the same order (one key per slot).
        """
        tail_hashes = transfer_op.mooncake_store_swa_block_hashes
        if tail_hashes is None:
            raise ValueError(
                "SWA mooncake transfer requires mooncake_store_swa_block_hashes")
        cpu_block_ids = (
            transfer_op.dst_block_ids
            if transfer_op.transfer_type == TransferType.REMOTE2H
            else transfer_op.src_block_ids
        )
        if len(tail_hashes) != len(cpu_block_ids):
            raise ValueError(
                "SWA mooncake transfer requires len(swa_block_hashes) == "
                f"len(cpu_block_ids): got {len(tail_hashes)} vs {len(cpu_block_ids)}")
        block_size_bytes = self.block_size_bytes
        base_ptr = self._cpu_buffer.data_ptr()
        cpu_ptrs: List[int] = []
        block_sizes: List[int] = []
        keys: List[str] = []
        for i, blk_id in enumerate(cpu_block_ids):
            cpu_ptrs.append(base_ptr + int(blk_id) * block_size_bytes)
            block_sizes.append(block_size_bytes)
            keys.append(build_key(
                str(tail_hashes[i]),
                PoolKind.SWA,
                pp_rank=self.pp_rank,
                pp_size=self.pp_size,
                node_layer_start=self.node_layer_start,
                node_layer_end=self.node_layer_end,
                total_layers=self.total_layers,
            ))
        return cpu_ptrs, block_sizes, keys

    def _postprocess(self, transfer_op: WorkerTransferOp) -> None:
        pass
