# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import queue
import threading
import time
import multiprocessing as mp
import selectors
import os
from queue import Queue
from typing import Dict, List, Optional, Tuple, Union

import contextlib
import nvtx
import numpy as np

from flexkv.common.debug import flexkv_logger
from flexkv.common.storage import StorageHandle
from flexkv.common.transfer import TransferOp, TransferOpGraph, TransferType, CompletedOp, LayerwiseTransferOp
from flexkv.common.transfer import get_nvtx_range_color
from flexkv.common.storage import KVCacheLayoutType
from flexkv.transfer.scheduler import TransferScheduler
from flexkv.transfer.worker import (
    WorkerHandle,
    CPUSSDDiskTransferWorker,
    CPURemoteTransferWorker,
    GPUCPUTransferWorker,
    tpGPUCPUTransferWorker,
    GDSTransferWorker,
    tpGDSTransferWorker,
    PEER2CPUTransferWorker,
)
from flexkv.transfer.layerwise import LayerwiseTransferWorker
from flexkv.common.config import CacheConfig, ModelConfig, GLOBAL_CONFIG_FROM_ENV
from flexkv.common.ring_buffer import SharedOpPool


def register_op_to_buffer(op: TransferOp, pin_buffer: SharedOpPool) -> None:
    """
    Register transfer operation to buffer with device type prefixes.

    Device type prefixes prevent hash collisions when different device types
    use the same block ID values (e.g., CPU block 0 vs SSD block 0).
    """
    if op.transfer_type == TransferType.LAYERWISE:
        return
    # Map TransferType to (src_device_type, dst_device_type) for hash prefix
    # This prevents hash collisions when different devices use the same block IDs
    transfer_type_to_devices = {
        TransferType.D2H: (1, 2),      # GPU -> CPU
        TransferType.H2D: (2, 1),      # CPU -> GPU
        TransferType.H2DISK: (2, 3),   # CPU -> SSD
        TransferType.DISK2H: (3, 2),   # SSD -> CPU
        TransferType.DISK2D: (3, 1),   # SSD -> GPU
        TransferType.D2DISK: (1, 3),   # GPU -> SSD
        TransferType.H2REMOTE: (2, 4), # CPU -> REMOTE
        TransferType.REMOTE2H: (4, 2), # REMOTE -> CPU
        TransferType.PEERH2H: (5, 2),  # PEER_CPU -> CPU
        TransferType.H2PEERH: (2, 5),  # CPU -> PEER_CPU
        TransferType.PEERSSD2H: (6, 2),# PEER_SSD -> CPU
        TransferType.H2PEERSSD: (2, 6),# CPU -> PEER_SSD
    }

    src_device, dst_device = transfer_type_to_devices.get(op.transfer_type, (0, 0))

    op.src_slot_id = pin_buffer.allocate_slot(op.src_block_ids, device_type_prefix=src_device)
    op.dst_slot_id = pin_buffer.allocate_slot(op.dst_block_ids, device_type_prefix=dst_device)

def free_op_from_buffer(op: TransferOp, pin_buffer: SharedOpPool) -> None:
    if op.src_slot_id != -1:
        pin_buffer.free_slot(op.src_slot_id)
    if op.dst_slot_id != -1:
        pin_buffer.free_slot(op.dst_slot_id)

class TransferEngine:
    def __init__(self,
        gpu_handles: Dict[int, List[StorageHandle]],
        model_config: ModelConfig,
        cache_config: CacheConfig,
        cpu_handle: Optional[StorageHandle] = None,
        ssd_handle: Optional[StorageHandle] = None,
        remote_handle: Optional[StorageHandle] = None,
        indexer_gpu_handles: Optional[Dict[int, List[StorageHandle]]] = None,
        indexer_cpu_handle: Optional[StorageHandle] = None,
        indexer_ssd_handle: Optional[StorageHandle] = None,
        indexer_remote_handle: Optional[StorageHandle] = None):
        """
        Initialize transfer engine

        Args:
            gpu_handles: Dict mapping dp_client_id -> list of GPU handles for that TP group
            cpu_handle: CPU handle
            ssd_handle: Optional SSD handle
            remote_handle: Optional remote handle
        """
        self.model_config: ModelConfig = model_config
        self.cache_config: CacheConfig = cache_config

        # Use spawn context for CUDA compatibility
        self.mp_ctx = mp.get_context('spawn')

        # Initialize scheduler
        self.scheduler = TransferScheduler()
        # Use mp.Queue instead of queue.Queue to enable selector monitoring
        self.task_queue = self.mp_ctx.Queue()
        # Use mp.Queue for completed_queue to enable daemon process to monitor it via selector
        self.completed_queue = self.mp_ctx.Queue()
        self.finished_ops_queue = self.mp_ctx.Queue()
        self.op_id_to_op: Dict[int, TransferOp] = {}

        # Create shutdown pipe for zero-latency selector
        self.shutdown_read_fd, self.shutdown_write_fd = os.pipe()
        self.gpu_handle_groups = gpu_handles  # dp_client_id -> list of GPU handles for that TP group
        self._cpu_handle = cpu_handle
        self._ssd_handle = ssd_handle
        self._remote_handle = remote_handle
        self._cache_config = cache_config
        # TODO: is this correct?
        self._enable_pcfs_sharing = (
            GLOBAL_CONFIG_FROM_ENV.index_accel and cache_config.enable_kv_sharing
        )

        self._indexer_gpu_handles = indexer_gpu_handles
        self._indexer_cpu_handle = indexer_cpu_handle
        self._indexer_ssd_handle = indexer_ssd_handle
        self._indexer_remote_handle = indexer_remote_handle

        self.pin_buffer = SharedOpPool(2048, self.cache_config.num_cpu_blocks)

        self.op_id_to_nvtx_range: Dict[int, str] = {}

        # self.dp_size = model_config.dp_size
        self.tp_size = model_config.tp_size
        self.num_gpu_groups = len(self.gpu_handle_groups)
        self._running = False
        self._has_indexer = False

        self._indexer_op_to_parent_op: Dict[int, int] = {}
        self._indexer_op_map: Dict[int, TransferOp] = {}

    def _init_workers(self) -> None:
        if self._running:
            return
        self._worker_map: Dict[TransferType, Union[WorkerHandle, List[WorkerHandle]]] = {}

        assert self._cpu_handle is not None
        # Use num_gpu_groups to support multi-instance mode
        # Use gpu_device_id from StorageHandle for correct CUDA device selection
        if self.tp_size == 1:
            self.h2d_workers: List[WorkerHandle] = [
                GPUCPUTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    gpu_blocks=gpu_handles[0].get_tensor_handle_list(),
                    cpu_blocks=self._cpu_handle.get_tensor(),
                    gpu_kv_layout=gpu_handles[0].kv_layout,
                    cpu_kv_layout=self._cpu_handle.kv_layout,
                    dtype=gpu_handles[0].dtype,
                    gpu_device_id=gpu_handles[0].gpu_device_id,
                    use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                    use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                    transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                    transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                )
                for _, gpu_handles in self.gpu_handle_groups.items()
            ]
            self.d2h_workers: List[WorkerHandle] = [
                GPUCPUTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    gpu_blocks=gpu_handles[0].get_tensor_handle_list(),
                    cpu_blocks=self._cpu_handle.get_tensor(),
                    gpu_kv_layout=gpu_handles[0].kv_layout,
                    cpu_kv_layout=self._cpu_handle.kv_layout,
                    dtype=gpu_handles[0].dtype,
                    gpu_device_id=gpu_handles[0].gpu_device_id,
                    use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                    use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                    transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                    transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                )
                for _, gpu_handles in self.gpu_handle_groups.items()
            ]
        else:
            self.h2d_workers = [
                tpGPUCPUTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    gpu_blocks=[gpu_handle.get_tensor_handle_list() for gpu_handle in gpu_handles],
                    cpu_blocks=self._cpu_handle.get_tensor(),
                    gpu_kv_layouts=[gpu_handle.kv_layout for gpu_handle in gpu_handles],
                    cpu_kv_layout=self._cpu_handle.kv_layout,
                    dtype=gpu_handles[0].dtype,
                    tp_group_size=self.tp_size,
                    dp_group_id=dp_client_id,
                    is_nsa_cp=getattr(self.model_config, "is_nsa_cp", False),
                    cp_size=getattr(self.model_config, "cp_size", 1),
                    use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                    use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                    transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                    transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                )
                for dp_client_id, gpu_handles in self.gpu_handle_groups.items()
            ]
            self.d2h_workers = [
                tpGPUCPUTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    gpu_blocks=[gpu_handle.get_tensor_handle_list() for gpu_handle in gpu_handles],
                    cpu_blocks=self._cpu_handle.get_tensor(),
                    gpu_kv_layouts=[gpu_handle.kv_layout for gpu_handle in gpu_handles],
                    cpu_kv_layout=self._cpu_handle.kv_layout,
                    dtype=gpu_handles[0].dtype,
                    tp_group_size=self.tp_size,
                    dp_group_id=dp_client_id,
                    is_nsa_cp=getattr(self.model_config, "is_nsa_cp", False),
                    cp_size=getattr(self.model_config, "cp_size", 1),
                    use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                    use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                    transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                    transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                )
                for dp_client_id, gpu_handles in self.gpu_handle_groups.items()
            ]
        self._worker_map[TransferType.H2D] = self.h2d_workers
        self._worker_map[TransferType.D2H] = self.d2h_workers

        if self._ssd_handle is not None and self._cpu_handle is not None:
            self.cpussd_read_worker: WorkerHandle = CPUSSDDiskTransferWorker.create_worker(
                mp_ctx=self.mp_ctx,
                finished_ops_queue=self.finished_ops_queue,
                op_buffer_tensor = self.pin_buffer.get_buffer(),
                cpu_blocks=self._cpu_handle.get_tensor(),
                ssd_files=self._ssd_handle.get_file_list(),
                cpu_kv_layout=self._cpu_handle.kv_layout,
                ssd_kv_layout=self._ssd_handle.kv_layout,
                dtype=self._cpu_handle.dtype,
                num_blocks_per_file=self._ssd_handle.num_blocks_per_file,
                cache_config=self._cache_config,
            )
            self.cpussd_write_worker: WorkerHandle = CPUSSDDiskTransferWorker.create_worker(
                mp_ctx=self.mp_ctx,
                finished_ops_queue=self.finished_ops_queue,
                op_buffer_tensor = self.pin_buffer.get_buffer(),
                cpu_blocks=self._cpu_handle.get_tensor(),
                ssd_files=self._ssd_handle.get_file_list(),
                cpu_kv_layout=self._cpu_handle.kv_layout,
                ssd_kv_layout=self._ssd_handle.kv_layout,
                dtype=self._cpu_handle.dtype,
                num_blocks_per_file=self._ssd_handle.num_blocks_per_file,
                cache_config=self._cache_config,
            )
            self._worker_map[TransferType.H2DISK] = self.cpussd_write_worker
            self._worker_map[TransferType.DISK2H] = self.cpussd_read_worker
        if self._remote_handle is not None and self._cpu_handle is not None:
            self.remotecpu_read_worker: WorkerHandle = CPURemoteTransferWorker.create_worker(
                mp_ctx=self.mp_ctx,
                finished_ops_queue=self.finished_ops_queue,
                op_buffer_tensor = self.pin_buffer.get_buffer(),
                cpu_blocks=self._cpu_handle.get_tensor(),
                remote_file=self._remote_handle.get_file_list(),
                cpu_kv_layout=self._cpu_handle.kv_layout,
                remote_kv_layout=self._remote_handle.kv_layout,
                dtype=self._cpu_handle.dtype,
                remote_config_custom=self._remote_handle.remote_config_custom,
                enable_pcfs_sharing=self._enable_pcfs_sharing,
            )
            self.remotecpu_write_worker: WorkerHandle = CPURemoteTransferWorker.create_worker(
                mp_ctx=self.mp_ctx,
                finished_ops_queue=self.finished_ops_queue,
                op_buffer_tensor = self.pin_buffer.get_buffer(),
                cpu_blocks=self._cpu_handle.get_tensor(),
                remote_file=self._remote_handle.get_file_list(),
                cpu_kv_layout=self._cpu_handle.kv_layout,
                remote_kv_layout=self._remote_handle.kv_layout,
                dtype=self._cpu_handle.dtype,
                remote_config_custom=self._remote_handle.remote_config_custom,
            )
            self._worker_map[TransferType.H2REMOTE] = self.remotecpu_write_worker
            self._worker_map[TransferType.REMOTE2H] = self.remotecpu_read_worker
        if self.cache_config.enable_gds:
            if self.tp_size == 1:
                self.gds_workers = [
                    GDSTransferWorker.create_worker(
                        mp_ctx=self.mp_ctx,
                        finished_ops_queue=self.finished_ops_queue,
                        op_buffer_tensor=self.pin_buffer.get_buffer(),
                        gpu_blocks=gpu_handles[0].get_tensor_handle_list(),
                        ssd_files=self._ssd_handle.get_file_list(),
                        num_blocks_per_file=self._ssd_handle.num_blocks_per_file,
                        gpu_kv_layout=gpu_handles[0].kv_layout,
                        ssd_kv_layout=self._ssd_handle.kv_layout,
                        dtype=self._ssd_handle.dtype,
                        gpu_device_id=gpu_handles[0].gpu_device_id,
                    )
                    for _, gpu_handles in self.gpu_handle_groups.items()
                ]
            else:
                self.gds_workers = [
                    tpGDSTransferWorker.create_worker(
                        mp_ctx=self.mp_ctx,
                        finished_ops_queue=self.finished_ops_queue,
                        op_buffer_tensor=self.pin_buffer.get_buffer(),
                        gpu_blocks=[gpu_handle.get_tensor_handle_list() for gpu_handle in gpu_handles],
                        ssd_files=self._ssd_handle.get_file_list(),
                        num_blocks_per_file=self._ssd_handle.num_blocks_per_file,
                        gpu_kv_layouts=[gpu_handle.kv_layout for gpu_handle in gpu_handles],
                        ssd_kv_layout=self._ssd_handle.kv_layout,
                        dtype=self._ssd_handle.dtype,
                        tp_group_size=self.tp_size,
                        dp_group_id=dp_client_id,
                    )
                    for dp_client_id, gpu_handles in self.gpu_handle_groups.items()
                ]
            self._worker_map[TransferType.DISK2D] = self.gds_workers
            self._worker_map[TransferType.D2DISK] = self.gds_workers
        if GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer:
            ssd_files = {} if self._ssd_handle is None else self._ssd_handle.get_file_list()
            ssd_kv_layout = None if self._ssd_handle is None else self._ssd_handle.kv_layout
            num_blocks_per_file = 0 if self._ssd_handle is None else self._ssd_handle.num_blocks_per_file
            _is_nsa_cp = getattr(self.model_config, 'is_nsa_cp', False)
            _cp_size = getattr(self.model_config, 'cp_size', 1)
            # For CP, each CP rank connects via eventfd; for TP, each TP rank connects.
            _eventfd_group_size = _cp_size if _is_nsa_cp and _cp_size > 1 else self.tp_size
            self.layerwise_workers = [
                LayerwiseTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    gpu_blocks=[handle.get_tensor_handle_list() for handle in gpu_handles],
                    cpu_blocks=self._cpu_handle.get_tensor(),
                    ssd_files=ssd_files,
                    gpu_kv_layouts=[handle.kv_layout for handle in gpu_handles],
                    cpu_kv_layout=self._cpu_handle.kv_layout,
                    ssd_kv_layout=ssd_kv_layout,
                    dtype=gpu_handles[0].dtype,
                    tp_group_size=_eventfd_group_size,
                    dp_group_id=dp_client_id,
                    pp_rank=self.model_config.pp_rank,
                    pp_size=self.model_config.pp_size,
                    dp_size=self.model_config.dp_size,
                    dp_rank=dp_client_id,
                    num_blocks_per_file=num_blocks_per_file,
                    use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                    use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                    h2d_cta_num=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                    d2h_cta_num=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                    is_nsa_cp=_is_nsa_cp,
                )
                for dp_client_id, gpu_handles in self.gpu_handle_groups.items()
            ]
            self._worker_map[TransferType.LAYERWISE] = self.layerwise_workers

        if self.cache_config.enable_kv_sharing and self._cpu_handle is not None and (self.cache_config.enable_p2p_cpu \
            or (self._ssd_handle and self.cache_config.enable_p2p_ssd)):
            ## NOTE:if we have the cpu handle and enable p2p cpu transfer we need this worker
            ## (currently we inplement cpu and ssd distributed transfer in one worker)

            flexkv_logger.info("[transfer_engine] initializing the PEER2CPUTransferWorker!")
            self.cpu_remote_cpu_worker: WorkerHandle = PEER2CPUTransferWorker.create_worker(
                mp_ctx=self.mp_ctx,
                finished_ops_queue=self.finished_ops_queue,
                op_buffer_tensor = self.pin_buffer.get_buffer(),
                cpu_blocks=self._cpu_handle.get_tensor(),
                cpu_kv_layout=self._cpu_handle.kv_layout,
                # TODO: get remote kv_layout, now we can assume that remote kv layout is same as current node
                remote_kv_layout=self._cpu_handle.kv_layout,
                dtype=self._cpu_handle.dtype,
                cache_config = self.cache_config,
                ssd_kv_layout = self._ssd_handle.kv_layout if self._ssd_handle else None,
                ssd_files = self._ssd_handle.get_file_list() if self._ssd_handle else None,
                num_blocks_per_file = self._ssd_handle.num_blocks_per_file if self._ssd_handle else None,
                mooncake_config_path = getattr(self.cache_config, 'mooncake_config_path', None) or os.environ.get("MOONCAKE_CONFIG_PATH"),
            )
            # NOTE: now peerH2H and peerSSD2H op use the same worker
            if self.cache_config.enable_p2p_cpu:
                self._worker_map[TransferType.PEERH2H] = self.cpu_remote_cpu_worker
            if self.cache_config.enable_p2p_ssd:
                self._worker_map[TransferType.PEERSSD2H] = self.cpu_remote_cpu_worker

        # Initialize indexer workers
        if (self._indexer_gpu_handles is not None
                and self._indexer_cpu_handle is not None):
            self._indexer_finished_ops_queue = self.mp_ctx.Queue()
            self._indexer_worker_map: Dict[TransferType, Union[WorkerHandle, List[WorkerHandle]]] = {}
            if self.tp_size == 1:
                self._indexer_h2d_workers = [
                    GPUCPUTransferWorker.create_worker(
                        mp_ctx=self.mp_ctx,
                        finished_ops_queue=self._indexer_finished_ops_queue,
                        op_buffer_tensor=self.pin_buffer.get_buffer(),
                        gpu_blocks=indexer_gpu_handles_list[0].get_tensor_handle_list(),
                        cpu_blocks=self._indexer_cpu_handle.get_tensor(),
                        gpu_kv_layout=indexer_gpu_handles_list[0].kv_layout,
                        cpu_kv_layout=self._indexer_cpu_handle.kv_layout,
                        dtype=indexer_gpu_handles_list[0].dtype,
                        gpu_device_id=indexer_gpu_handles_list[0].gpu_device_id,
                        use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                        use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                        transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                        transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                    )
                    for _, indexer_gpu_handles_list in self._indexer_gpu_handles.items()
                ]
                self._indexer_d2h_workers = [
                    GPUCPUTransferWorker.create_worker(
                        mp_ctx=self.mp_ctx,
                        finished_ops_queue=self._indexer_finished_ops_queue,
                        op_buffer_tensor=self.pin_buffer.get_buffer(),
                        gpu_blocks=indexer_gpu_handles_list[0].get_tensor_handle_list(),
                        cpu_blocks=self._indexer_cpu_handle.get_tensor(),
                        gpu_kv_layout=indexer_gpu_handles_list[0].kv_layout,
                        cpu_kv_layout=self._indexer_cpu_handle.kv_layout,
                        dtype=indexer_gpu_handles_list[0].dtype,
                        gpu_device_id=indexer_gpu_handles_list[0].gpu_device_id,
                        use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                        use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                        transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                        transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                    )
                    for _, indexer_gpu_handles_list in self._indexer_gpu_handles.items()
                ]
            else:
                self._indexer_h2d_workers = [
                    tpGPUCPUTransferWorker.create_worker(
                        mp_ctx=self.mp_ctx,
                        finished_ops_queue=self._indexer_finished_ops_queue,
                        op_buffer_tensor=self.pin_buffer.get_buffer(),
                        gpu_blocks=[h.get_tensor_handle_list() for h in indexer_gpu_handles_list],
                        cpu_blocks=self._indexer_cpu_handle.get_tensor(),
                        gpu_kv_layouts=[h.kv_layout for h in indexer_gpu_handles_list],
                        cpu_kv_layout=self._indexer_cpu_handle.kv_layout,
                        dtype=indexer_gpu_handles_list[0].dtype,
                        tp_group_size=self.tp_size,
                        dp_group_id=dp_client_id,
                        is_nsa_cp=getattr(self.model_config, "is_nsa_cp", False),
                        cp_size=getattr(self.model_config, "cp_size", 1),
                        use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                        use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                        transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                        transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                    )
                    for dp_client_id, indexer_gpu_handles_list in self._indexer_gpu_handles.items()
                ]
                self._indexer_d2h_workers = [
                    tpGPUCPUTransferWorker.create_worker(
                        mp_ctx=self.mp_ctx,
                        finished_ops_queue=self._indexer_finished_ops_queue,
                        op_buffer_tensor=self.pin_buffer.get_buffer(),
                        gpu_blocks=[h.get_tensor_handle_list() for h in indexer_gpu_handles_list],
                        cpu_blocks=self._indexer_cpu_handle.get_tensor(),
                        gpu_kv_layouts=[h.kv_layout for h in indexer_gpu_handles_list],
                        cpu_kv_layout=self._indexer_cpu_handle.kv_layout,
                        dtype=indexer_gpu_handles_list[0].dtype,
                        tp_group_size=self.tp_size,
                        dp_group_id=dp_client_id,
                        is_nsa_cp=getattr(self.model_config, "is_nsa_cp", False),
                        cp_size=getattr(self.model_config, "cp_size", 1),
                        use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                        use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                        transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                        transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                    )
                    for dp_client_id, indexer_gpu_handles_list in self._indexer_gpu_handles.items()
                ]
            self._indexer_worker_map[TransferType.H2D] = self._indexer_h2d_workers
            self._indexer_worker_map[TransferType.D2H] = self._indexer_d2h_workers
            if self._indexer_ssd_handle is not None and self._indexer_cpu_handle is not None:
                self._indexer_h2disk_worker = CPUSSDDiskTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self._indexer_finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    cpu_blocks=self._indexer_cpu_handle.get_tensor(),
                    ssd_files=self._indexer_ssd_handle.get_file_list(),
                    cpu_kv_layout=self._indexer_cpu_handle.kv_layout,
                    ssd_kv_layout=self._indexer_ssd_handle.kv_layout,
                    dtype=self._indexer_cpu_handle.dtype,
                    num_blocks_per_file=self._indexer_ssd_handle.num_blocks_per_file,
                    cache_config=self._cache_config,
                )
                self._indexer_disk2h_worker = CPUSSDDiskTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self._indexer_finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    cpu_blocks=self._indexer_cpu_handle.get_tensor(),
                    ssd_files=self._indexer_ssd_handle.get_file_list(),
                    cpu_kv_layout=self._indexer_cpu_handle.kv_layout,
                    ssd_kv_layout=self._indexer_ssd_handle.kv_layout,
                    dtype=self._indexer_cpu_handle.dtype,
                    num_blocks_per_file=self._indexer_ssd_handle.num_blocks_per_file,
                    cache_config=self._cache_config,
                )
                self._indexer_worker_map[TransferType.H2DISK] = self._indexer_h2disk_worker
                self._indexer_worker_map[TransferType.DISK2H] = self._indexer_disk2h_worker
                flexkv_logger.info("TransferEngine: indexer SSD workers initialized")
            if self._indexer_remote_handle is not None and self._indexer_cpu_handle is not None:
                self._indexer_h2remote_worker = CPURemoteTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self._indexer_finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    cpu_blocks=self._indexer_cpu_handle.get_tensor(),
                    remote_file=self._indexer_remote_handle.get_file_list(),
                    cpu_kv_layout=self._indexer_cpu_handle.kv_layout,
                    remote_kv_layout=self._indexer_remote_handle.kv_layout,
                    dtype=self._indexer_cpu_handle.dtype,
                    remote_config_custom=self._indexer_remote_handle.remote_config_custom,
                    enable_pcfs_sharing=self._enable_pcfs_sharing,
                )
                self._indexer_remote2h_worker = CPURemoteTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self._indexer_finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    cpu_blocks=self._indexer_cpu_handle.get_tensor(),
                    remote_file=self._indexer_remote_handle.get_file_list(),
                    cpu_kv_layout=self._indexer_cpu_handle.kv_layout,
                    remote_kv_layout=self._indexer_remote_handle.kv_layout,
                    dtype=self._indexer_cpu_handle.dtype,
                    remote_config_custom=self._indexer_remote_handle.remote_config_custom,
                )
                self._indexer_worker_map[TransferType.H2REMOTE] = self._indexer_h2remote_worker
                self._indexer_worker_map[TransferType.REMOTE2H] = self._indexer_remote2h_worker
                flexkv_logger.info("TransferEngine: indexer Remote workers initialized")
            if self.cache_config.enable_gds and self._indexer_ssd_handle is not None:
                if self.tp_size == 1:
                    self._indexer_gds_workers = [
                        GDSTransferWorker.create_worker(
                            mp_ctx=self.mp_ctx,
                            finished_ops_queue=self._indexer_finished_ops_queue,
                            op_buffer_tensor=self.pin_buffer.get_buffer(),
                            gpu_blocks=indexer_gpu_handles_list[0].get_tensor_handle_list(),
                            ssd_files=self._indexer_ssd_handle.get_file_list(),
                            num_blocks_per_file=self._indexer_ssd_handle.num_blocks_per_file,
                            gpu_kv_layout=indexer_gpu_handles_list[0].kv_layout,
                            ssd_kv_layout=self._indexer_ssd_handle.kv_layout,
                            dtype=self._indexer_ssd_handle.dtype,
                            gpu_device_id=indexer_gpu_handles_list[0].gpu_device_id,
                        )
                        for _, indexer_gpu_handles_list in self._indexer_gpu_handles.items()
                    ]
                else:
                    self._indexer_gds_workers = [
                        tpGDSTransferWorker.create_worker(
                            mp_ctx=self.mp_ctx,
                            finished_ops_queue=self._indexer_finished_ops_queue,
                            op_buffer_tensor=self.pin_buffer.get_buffer(),
                            gpu_blocks=[h.get_tensor_handle_list() for h in indexer_gpu_handles_list],
                            ssd_files=self._indexer_ssd_handle.get_file_list(),
                            num_blocks_per_file=self._indexer_ssd_handle.num_blocks_per_file,
                            gpu_kv_layouts=[h.kv_layout for h in indexer_gpu_handles_list],
                            ssd_kv_layout=self._indexer_ssd_handle.kv_layout,
                            dtype=self._indexer_ssd_handle.dtype,
                            tp_group_size=self.tp_size,
                            dp_group_id=dp_client_id,
                        )
                        for dp_client_id, indexer_gpu_handles_list in self._indexer_gpu_handles.items()
                    ]
                self._indexer_worker_map[TransferType.DISK2D] = self._indexer_gds_workers
                self._indexer_worker_map[TransferType.D2DISK] = self._indexer_gds_workers
                flexkv_logger.info("TransferEngine: indexer GDS workers initialized")
            if self.cache_config.enable_kv_sharing and self._indexer_cpu_handle is not None and (
                    self.cache_config.enable_p2p_cpu
                    or (self._indexer_ssd_handle and self.cache_config.enable_p2p_ssd)):
                flexkv_logger.info("[transfer_engine] initializing the indexer PEER2CPUTransferWorker!")
                self._indexer_cpu_remote_cpu_worker: WorkerHandle = PEER2CPUTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self._indexer_finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    cpu_blocks=self._indexer_cpu_handle.get_tensor(),
                    cpu_kv_layout=self._indexer_cpu_handle.kv_layout,
                    remote_kv_layout=self._indexer_cpu_handle.kv_layout,
                    dtype=self._indexer_cpu_handle.dtype,
                    cache_config=self._cache_config,
                    ssd_kv_layout=self._indexer_ssd_handle.kv_layout if self._indexer_ssd_handle else None,
                    ssd_files=self._indexer_ssd_handle.get_file_list() if self._indexer_ssd_handle else None,
                    num_blocks_per_file=self._indexer_ssd_handle.num_blocks_per_file if self._indexer_ssd_handle else None,
                )
                if self.cache_config.enable_p2p_cpu:
                    self._indexer_worker_map[TransferType.PEERH2H] = self._indexer_cpu_remote_cpu_worker
                if self.cache_config.enable_p2p_ssd:
                    self._indexer_worker_map[TransferType.PEERSSD2H] = self._indexer_cpu_remote_cpu_worker
                flexkv_logger.info("TransferEngine: indexer P2P workers initialized")
            if GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer:
                indexer_ssd_files = {} if self._indexer_ssd_handle is None else self._indexer_ssd_handle.get_file_list()
                indexer_ssd_kv_layout = None if self._indexer_ssd_handle is None else self._indexer_ssd_handle.kv_layout
                indexer_num_blocks_per_file = 0 if self._indexer_ssd_handle is None else self._indexer_ssd_handle.num_blocks_per_file
                self._indexer_layerwise_workers = [
                    LayerwiseTransferWorker.create_worker(
                        mp_ctx=self.mp_ctx,
                        finished_ops_queue=self._indexer_finished_ops_queue,
                        op_buffer_tensor=self.pin_buffer.get_buffer(),
                        gpu_blocks=[h.get_tensor_handle_list() for h in indexer_gpu_handles_list],
                        cpu_blocks=self._indexer_cpu_handle.get_tensor(),
                        ssd_files=indexer_ssd_files,
                        gpu_kv_layouts=[h.kv_layout for h in indexer_gpu_handles_list],
                        cpu_kv_layout=self._indexer_cpu_handle.kv_layout,
                        ssd_kv_layout=indexer_ssd_kv_layout,
                        dtype=indexer_gpu_handles_list[0].dtype,
                        tp_group_size=self.tp_size,
                        dp_group_id=dp_client_id,
                        pp_rank=self.model_config.pp_rank,
                        pp_size=self.model_config.pp_size,
                        dp_size=self.model_config.dp_size,
                        dp_rank=dp_client_id,
                        num_blocks_per_file=indexer_num_blocks_per_file,
                        use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                        use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                        h2d_cta_num=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                        d2h_cta_num=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                        enable_eventfd=False,
                    )
                    for dp_client_id, indexer_gpu_handles_list in self._indexer_gpu_handles.items()
                ]
                self._indexer_worker_map[TransferType.LAYERWISE] = self._indexer_layerwise_workers
                flexkv_logger.info("TransferEngine: indexer Layerwise workers initialized")
            self._has_indexer = True
            flexkv_logger.info(
                f"TransferEngine: indexer inline workers initialized "
                f"({len(self._indexer_h2d_workers)} H2D + {len(self._indexer_d2h_workers)} D2H)")

        if len(self._worker_map) == 0:
            raise ValueError("No workers initialized, please check the config")
        # Wait for all main KV workers to ready
        for transfer_type, worker in self._worker_map.items():
            if isinstance(worker, List):
                for w in worker:
                    flexkv_logger.info(f"waiting for {transfer_type.name} worker {w.worker_id} to ready")
                    w.ready_event.wait()
                    flexkv_logger.info(f"{transfer_type.name} worker {w.worker_id} is ready")
            else:
                flexkv_logger.info(f"waiting for {transfer_type.name} worker {worker.worker_id} to ready")
                worker.ready_event.wait()
                flexkv_logger.info(f"{transfer_type.name} worker {worker.worker_id} is ready")
        # Wait for all indexer workers to ready
        if self._has_indexer:
            for transfer_type, worker in self._indexer_worker_map.items():
                if isinstance(worker, List):
                    for w in worker:
                        flexkv_logger.info(f"waiting for indexer {transfer_type.name} worker {w.worker_id} to ready")
                        w.ready_event.wait()
                        flexkv_logger.info(f"indexer {transfer_type.name} worker {w.worker_id} is ready")
                else:
                    flexkv_logger.info(f"waiting for indexer {transfer_type.name} worker {worker.worker_id} to ready")
                    worker.ready_event.wait()
                    flexkv_logger.info(f"indexer {transfer_type.name} worker {worker.worker_id} is ready")
        # Start scheduler thread
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self._scheduler_thread.start()

    def start(self) -> None:
        self._init_workers()

    def _scheduler_loop(self) -> None:
        """Event-driven scheduler loop using selectors (ZERO LATENCY with shutdown pipe)"""
        from flexkv.common.debug import flexkv_logger

        # Setup selector to monitor both queues simultaneously
        sel = selectors.DefaultSelector()

        # Register both queues for monitoring
        sel.register(self.task_queue._reader, selectors.EVENT_READ, data="new_graph")
        sel.register(self.finished_ops_queue._reader, selectors.EVENT_READ, data="finished_op")

        # Register indexer finished_ops_queue when indexer is enabled
        if self._has_indexer:
            sel.register(self._indexer_finished_ops_queue._reader, selectors.EVENT_READ, data="indexer_finished_op")

        # Register shutdown pipe for zero-latency shutdown
        sel.register(self.shutdown_read_fd, selectors.EVENT_READ, data="shutdown")

        flexkv_logger.info("TransferEngine scheduler loop started with ZERO-LATENCY selector (timeout=None)")

        while self._running:
            try:
                # Complete blocking with NO TIMEOUT for zero latency!
                # Shutdown via pipe signal instead of timeout
                events = sel.select(timeout=None)

                new_graphs_num = 0
                finished_ops: List[TransferOp] = []
                should_shutdown = False

                # Process events from selector
                for key, mask in events:
                    if key.data == "shutdown":
                        # Shutdown signal received via pipe
                        flexkv_logger.info("Scheduler loop received shutdown signal via pipe")
                        should_shutdown = True
                        break

                    elif key.data == "new_graph":
                        # Process new transfer graphs (batch get all available)
                        nvtx_r1 = nvtx.start_range(message="transfer scheduler. get new graphs", color="orange")
                        # Get all available graphs in one go to reduce system calls
                        while True:
                            try:
                                transfer_graph = self.task_queue.get_nowait()
                                # Handle batch submission (list of graphs)
                                graphs = transfer_graph if isinstance(transfer_graph, list) else [transfer_graph]
                                for graph in graphs:
                                    self.scheduler.add_transfer_graph(graph)
                                new_graphs_num += len(graphs)
                            except queue.Empty:
                                break
                        nvtx.end_range(nvtx_r1)

                    elif key.data == "finished_op":
                        # Collect finished ops from main KV worker (batch get all available)
                        nvtx_r2 = nvtx.start_range(message="transfer scheduler. collect finished ops", color="orange")
                        # Get all available ops in one go to reduce system calls
                        while True:
                            try:
                                op_id = self.finished_ops_queue.get_nowait()
                                op = self.op_id_to_op[op_id]
                                op.pending_count -= 1
                                if op.pending_count == 0:
                                    self._finalize_op(op, finished_ops)
                            except queue.Empty:
                                break
                        nvtx.end_range(nvtx_r2)

                    elif key.data == "indexer_finished_op":
                        # Collect finished ops from indexer worker (batch get all available)
                        nvtx_r2i = nvtx.start_range(message="transfer scheduler. collect indexer finished ops", color="blue")
                        while True:
                            try:
                                op_id = self._indexer_finished_ops_queue.get_nowait()
                                assert op_id in self._indexer_op_to_parent_op, (
                                    f"[TransferEngine] Indexer op {op_id} not found in "
                                    f"_indexer_op_to_parent_op. All indexer ops must be "
                                    f"registered with a parent op."
                                )
                                indexer_op = self._indexer_op_map.pop(op_id)
                                free_op_from_buffer(indexer_op, self.pin_buffer)
                                parent_op_id = self._indexer_op_to_parent_op.pop(op_id)
                                parent_op = self.op_id_to_op[parent_op_id]
                                parent_op.pending_count -= 1
                                if parent_op.pending_count == 0:
                                    self._finalize_op(parent_op, finished_ops)
                            except queue.Empty:
                                break
                        nvtx.end_range(nvtx_r2i)

                # Exit loop if shutdown requested
                if should_shutdown:
                    break

                # End NVTX ranges for finished ops
                for op in finished_ops:
                    nvtx.end_range(self.op_id_to_nvtx_range[op.op_id])
                    self.op_id_to_nvtx_range.pop(op.op_id)

                # Schedule next operations
                nvtx_r3 = nvtx.start_range(message="transfer scheduler. schedule next ops", color="orange")
                if finished_ops or new_graphs_num > 0:
                    completed_graph_ids, next_ops = self.scheduler.schedule(finished_ops)
                    # Distribute new ops to workers
                    for op in next_ops:
                        if op.transfer_type == TransferType.VIRTUAL:
                            self.completed_queue.put(CompletedOp(graph_id=op.graph_id, op_id=op.op_id))
                        else:
                            self.op_id_to_op[op.op_id] = op
                            # copy block ids into buffer and update slot id info
                            register_op_to_buffer(op, self.pin_buffer)
                            self._assign_op_to_worker(op)
                    # Handle completed graphs
                    for graph_id in completed_graph_ids:
                        self.completed_queue.put(CompletedOp.completed_graph(graph_id))
                nvtx.end_range(nvtx_r3)

            except Exception as e:
                flexkv_logger.error(f"Error in scheduler loop: {e}")
                time.sleep(0.001)  # Fallback on error

        # Cleanup
        sel.close()
        flexkv_logger.info("TransferEngine scheduler loop stopped")

    def _finalize_op(self, op: TransferOp, finished_ops: List[TransferOp]) -> None:
        """Finalize a completed op: release pin buffer, notify upper layer, and clean up.

        Called only when op.pending_count reaches 0, i.e., all workers (main KV + indexer)
        have completed this op. This ensures atomic eviction semantics.
        """
        free_op_from_buffer(op, self.pin_buffer)
        # Compute transfer metrics for this completed op
        num_blocks = len(op.src_block_ids) if op.src_block_ids is not None else 0
        num_bytes = num_blocks * self.cache_config.tokens_per_block * self.model_config.token_size_in_bytes
        transfer_type_str = op.transfer_type.value if op.transfer_type != TransferType.VIRTUAL else None
        self.completed_queue.put(CompletedOp(
            graph_id=op.graph_id,
            op_id=op.op_id,
            transfer_type=transfer_type_str,
            num_blocks=num_blocks,
            num_bytes=num_bytes,
        ))
        finished_ops.append(op)
        del self.op_id_to_op[op.op_id]

    def _assign_op_to_worker(self, op: TransferOp) -> None:
        self.op_id_to_nvtx_range[op.op_id] = nvtx.start_range(f"schedule {op.transfer_type.name} "
                                                                       f"op_id: {op.op_id}, "
                                                                       f"graph_id: {op.graph_id}, "
                                                                       f"successors: {op.successors}",
                                                                       color=get_nvtx_range_color(op.graph_id))
        """Assign operation to appropriate worker"""
        if op.transfer_type == TransferType.VIRTUAL:
            return
        if op.transfer_type not in self._worker_map:
            raise ValueError(f"Unsupported transfer type: {op.transfer_type}")

        if self._has_indexer and op.transfer_type in self._indexer_worker_map:
            # Indexer maps 1:1 with main KV blocks, use block_ids directly.
            src_page_ids = op.src_block_ids
            dst_page_ids = op.dst_block_ids
            num_pages = src_page_ids.size

            if num_pages > 0:
                # Always create a separate indexer_op to avoid sharing the same op
                # object between indexer worker and main KV worker.
                indexer_op = TransferOp(
                    graph_id=op.graph_id,
                    transfer_type=op.transfer_type,
                    src_block_ids=src_page_ids,
                    dst_block_ids=dst_page_ids,
                    layer_id=op.layer_id,
                    layer_granularity=op.layer_granularity,
                    dp_id=op.dp_id,
                )
                register_op_to_buffer(indexer_op, self.pin_buffer)
                self._indexer_op_to_parent_op[indexer_op.op_id] = op.op_id
                self._indexer_op_map[indexer_op.op_id] = indexer_op
                op.pending_count += 1

                flexkv_logger.debug(
                    f"[TransferEngine] Created indexer op {indexer_op.op_id} "
                    f"for parent op {op.op_id}: {num_pages} pages, "
                    f"type={op.transfer_type.name}"
                )

                indexer_worker = self._indexer_worker_map[op.transfer_type]
                if isinstance(indexer_worker, List):
                    indexer_worker[op.dp_id].submit_transfer(indexer_op)
                else:
                    indexer_worker.submit_transfer(indexer_op)

        worker = self._worker_map[op.transfer_type]
        if isinstance(worker, List):
            worker[op.dp_id].submit_transfer(op)
        else:
            worker.submit_transfer(op)

    def submit_transfer_graph(self, transfer_graph: Union[TransferOpGraph, List[TransferOpGraph]]) -> None:
        """Submit a transfer graph for execution"""
        nvtx_range = nvtx.start_range(message="TransferEngine.submit_transfer_graph", color="green")
        if not isinstance(transfer_graph, List):
            transfer_graph = [transfer_graph]
        self.task_queue.put(transfer_graph)
        nvtx.end_range(nvtx_range)

    def get_completed_graphs_and_ops(self, timeout: Optional[float] = None) -> List[CompletedOp]:
        """Get IDs of all completed transfer graphs at current moment

        Args:
            timeout: Optional timeout for the first graph retrieval

        Returns:
            List of CompletedOp objects. Empty list if no graphs are completed.
        """
        completed_ops: List[CompletedOp] = []

        if self.completed_queue.empty():
            return completed_ops

        try:
            first_op = self.completed_queue.get(timeout=timeout)
            completed_ops.append(first_op)

            while not self.completed_queue.empty():
                completed_op = self.completed_queue.get_nowait()
                completed_ops.append(completed_op)

        except queue.Empty:
            pass

        return completed_ops

    def shutdown(self) -> None:
        """Shutdown the transfer engine"""
        try:
            if not self._running:
                return
            self._running = False

            # Send shutdown signal via pipe to wake up selector immediately
            try:
                os.write(self.shutdown_write_fd, b'1')
            except (OSError, BrokenPipeError) as e:
                # Pipe already closed, that's ok
                flexkv_logger.debug(f"Shutdown pipe already closed during write: {e}")

            self._scheduler_thread.join(timeout=5)

            # Close shutdown pipe
            try:
                os.close(self.shutdown_read_fd)
                os.close(self.shutdown_write_fd)
            except OSError as e:
                # Only ignore EBADF (bad file descriptor, already closed)
                if e.errno != 9:  # errno.EBADF = 9
                    flexkv_logger.warning(f"Unexpected error closing shutdown pipes: {e}")
                else:
                    flexkv_logger.debug(f"Shutdown pipes already closed: {e}")

            # shutdown indexer workers first
            if self._has_indexer:
                for worker in self._indexer_worker_map.values():
                    if isinstance(worker, List):
                        for w in worker:
                            w.shutdown()
                    else:
                        worker.shutdown()
            # shutdown main KV workers
            for worker in self._worker_map.values():
                if isinstance(worker, List):
                    for w in worker:
                        w.shutdown()
                else:
                    worker.shutdown()
        except Exception as e:
            flexkv_logger.error(f"Error during shutdown: {e}")
        finally:
            with contextlib.suppress(Exception):
                while not self.finished_ops_queue.empty():
                    self.finished_ops_queue.get_nowait()

            torch.cuda.empty_cache()
            torch.cuda.synchronize()
