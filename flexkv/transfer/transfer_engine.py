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
from typing import Dict, List, Optional, Tuple, Union

import contextlib
import nvtx
import numpy as np
import torch

from flexkv.common.debug import flexkv_logger
from flexkv.common.storage import StorageHandle
from flexkv.common.transfer import TransferOp, TransferOpGraph, TransferType, CompletedOp, WorkerKey
from flexkv.common.transfer import get_nvtx_range_color
from flexkv.transfer.scheduler import TransferScheduler
from flexkv.transfer.worker import (
    WorkerHandle,
    CPUSSDDiskTransferWorker,
    CPURemoteTransferWorker,
    GPUCPUTransferWorker,
    tpGPUCPUTransferWorker,
    GDSTransferWorker,
    tpGDSTransferWorker,
    NixlTransferWorker,
    PEER2CPUTransferWorker,
    MooncakeStoreTransferWorker,
)
from flexkv.external.mooncake_store_keys import PoolKind
from flexkv.transfer.compression import build_compressors
from flexkv.transfer.layerwise import (
    LayerwiseTransferWorker,
    build_layerwise_eventfd_socket_path,
)
from flexkv.common.config import (
    CacheConfig, LayerGroupSpec, ModelConfig, GLOBAL_CONFIG_FROM_ENV,
)
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
        gpu_handles: Dict[WorkerKey, List[StorageHandle]],
        model_config: ModelConfig,
        cache_config: CacheConfig,
        cpu_handle: Optional[StorageHandle] = None,
        ssd_handle: Optional[StorageHandle] = None,
        remote_handle: Optional[StorageHandle] = None,
        gpu_blocks_per_group: Optional[Dict[WorkerKey, List]] = None,
        gpu_layouts_per_group: Optional[Dict[WorkerKey, List]] = None,
        swa_gpu_handles: Optional[Dict[WorkerKey, List[StorageHandle]]] = None,
        swa_cpu_handle: Optional[StorageHandle] = None,
        swa_ssd_handle: Optional[StorageHandle] = None,
        swa_remote_handle: Optional[StorageHandle] = None,
        swa_layer_groups: Optional[List[LayerGroupSpec]] = None,
        swa_gpu_blocks_per_group: Optional[Dict[WorkerKey, List]] = None,
        swa_gpu_layouts_per_group: Optional[Dict[WorkerKey, List]] = None,
        ):
        """
        Initialize transfer engine

        Args:
            gpu_handles: Dict mapping WorkerKey(dp_rank, pp_rank) -> list of GPU handles for that TP group
            model_config: global ModelConfig (parallelism sizes; no per-rank index)
            cache_config: global CacheConfig
            cpu_handle: CPU handle
            ssd_handle: Optional SSD handle
            remote_handle: Optional remote handle
            gpu_blocks_per_group: Per-group GPU handles, keyed by WorkerKey
            gpu_layouts_per_group: Per-group GPU layouts, keyed by WorkerKey
        """
        self.model_config: ModelConfig = model_config
        self.cache_config: CacheConfig = cache_config

        first_handles = next(iter(gpu_handles.values()))
        self._num_layers_for_local_pp_stage = first_handles[0].kv_layout.num_layer

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
        self.gpu_handle_groups = gpu_handles  # WorkerKey -> list of GPU handles for that TP group
        self._cpu_handle = cpu_handle
        self._ssd_handle = ssd_handle
        self._remote_handle = remote_handle
        self._gpu_blocks_per_group = gpu_blocks_per_group
        self._gpu_layouts_per_group = gpu_layouts_per_group

        # SWA handles and workers
        self._swa_gpu_handles = swa_gpu_handles
        self._swa_cpu_handle = swa_cpu_handle
        self._swa_ssd_handle = swa_ssd_handle
        self._swa_remote_handle = swa_remote_handle
        self._swa_layer_groups = (
            swa_cpu_handle.kv_layout.layer_groups
            if swa_cpu_handle is not None
            and swa_cpu_handle.kv_layout.layer_groups is not None
            else swa_layer_groups
        )
        self._swa_gpu_blocks_per_group = swa_gpu_blocks_per_group
        self._swa_gpu_layouts_per_group = swa_gpu_layouts_per_group
        if self._swa_layer_groups is not None and (
            self._swa_gpu_blocks_per_group is None
            or self._swa_gpu_layouts_per_group is None
        ):
            raise ValueError(
                "SWA multi-group layout is missing per-group GPU handles/layouts"
            )
        self._has_swa = (swa_gpu_handles is not None and len(swa_gpu_handles) > 0
                         and swa_cpu_handle is not None)
        self._cache_config = cache_config
        # TODO: is this correct?
        self._enable_pcfs_sharing = (
            GLOBAL_CONFIG_FROM_ENV.index_accel and cache_config.enable_kv_sharing
        )

        self.pin_buffer = SharedOpPool(2048, self.cache_config.num_cpu_blocks)

        self.op_id_to_nvtx_range: Dict[int, str] = {}

        self.num_gpu_groups = len(self.gpu_handle_groups)
        self._running = False

        self._child_id_to_child: Dict[int, TransferOp] = {}
        self._child_to_parent_op_id: Dict[int, int] = {}

        self._compressors = build_compressors(
            cpu_handle=self._cpu_handle,
            ssd_handle=self._ssd_handle,
            cache_config=self.cache_config,
            model_config=self.model_config,
            gpu_handle_groups=self.gpu_handle_groups,
            layerwise_enabled=GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer,
        )

        # Used for LAYERWISE PP fan-out: a parent op spawns one replica per PP
        # sibling worker; each replica's completion decrements the parent's
        # pending_count and the parent finalizes when count hits 0.
        self._child_id_to_child: Dict[int, TransferOp] = {}
        self._child_to_parent_op_id: Dict[int, int] = {}

    def _get_multi_group_kwargs_tp1(self, worker_key: WorkerKey) -> dict:
        """Get multi-group kwargs for TP=1 workers (GPUCPU / GDS)."""
        if (self.model_config.layer_groups is None or
                self._gpu_blocks_per_group is None or
                worker_key not in self._gpu_blocks_per_group):
            return {}
        # For TP=1, there's one device per WorkerKey
        # _gpu_blocks_per_group[worker_key][0] = per-group handle lists for that device
        per_device_group_blocks = self._gpu_blocks_per_group[worker_key][0]
        per_device_group_layouts = self._gpu_layouts_per_group[worker_key][0]
        if per_device_group_blocks is None or per_device_group_layouts is None:
            return {}
        return dict(
            layer_groups=self.model_config.layer_groups,
            gpu_blocks_per_group=per_device_group_blocks,
            gpu_layouts_per_group=per_device_group_layouts,
        )

    def _get_multi_group_kwargs_tp(self, worker_key: WorkerKey) -> dict:
        """Get multi-group kwargs for TP>1 workers (tpGPUCPU / tpGDS)."""
        if (self.model_config.layer_groups is None or
                self._gpu_blocks_per_group is None or
                worker_key not in self._gpu_blocks_per_group):
            return {}
        # For TP>1, _gpu_blocks_per_group[worker_key] has tp_size entries (one per device)
        # Each entry is List[List[TensorSharedHandle]] (per-group handle lists for that device)
        per_device_data = self._gpu_blocks_per_group[worker_key]
        per_device_layouts = self._gpu_layouts_per_group[worker_key]
        if per_device_data[0] is None or per_device_layouts[0] is None:
            return {}

        num_groups = len(self.model_config.layer_groups)
        num_devices = len(per_device_data)

        # Restructure: from [device][group] -> [group][device]
        # gpu_blocks_per_group[group_idx][device_idx] = handles for that group on that device
        blocks_by_group = []
        layouts_by_group = []
        for gi in range(num_groups):
            group_blocks_per_device = [per_device_data[di][gi] for di in range(num_devices)]
            group_layouts_per_device = [per_device_layouts[di][gi] for di in range(num_devices)]
            blocks_by_group.append(group_blocks_per_device)
            layouts_by_group.append(group_layouts_per_device)

        return dict(
            layer_groups=self.model_config.layer_groups,
            gpu_blocks_per_group=blocks_by_group,
            gpu_layouts_per_group=layouts_by_group,
        )

    def _get_swa_multi_group_kwargs_tp1(self, worker_key: WorkerKey) -> dict:
        """Return DSv4 SWA/state sidecar groups for a one-device worker."""
        if (
            self._swa_layer_groups is None
            or self._swa_gpu_blocks_per_group is None
            or self._swa_gpu_layouts_per_group is None
            or worker_key not in self._swa_gpu_blocks_per_group
        ):
            return {}
        per_device_blocks = self._swa_gpu_blocks_per_group[worker_key][0]
        per_device_layouts = self._swa_gpu_layouts_per_group[worker_key][0]
        if per_device_blocks is None or per_device_layouts is None:
            return {}
        return dict(
            layer_groups=self._swa_layer_groups,
            gpu_blocks_per_group=per_device_blocks,
            gpu_layouts_per_group=per_device_layouts,
        )

    def _get_swa_multi_group_kwargs_tp(self, worker_key: WorkerKey) -> dict:
        """Return SWA/state sidecar groups reshaped as [group][device]."""
        if (
            self._swa_layer_groups is None
            or self._swa_gpu_blocks_per_group is None
            or self._swa_gpu_layouts_per_group is None
            or worker_key not in self._swa_gpu_blocks_per_group
        ):
            return {}
        per_device_blocks = self._swa_gpu_blocks_per_group[worker_key]
        per_device_layouts = self._swa_gpu_layouts_per_group[worker_key]
        if per_device_blocks[0] is None or per_device_layouts[0] is None:
            return {}
        num_groups = len(self._swa_layer_groups)
        num_devices = len(per_device_blocks)
        return dict(
            layer_groups=self._swa_layer_groups,
            gpu_blocks_per_group=[
                [per_device_blocks[di][gi] for di in range(num_devices)]
                for gi in range(num_groups)
            ],
            gpu_layouts_per_group=[
                [per_device_layouts[di][gi] for di in range(num_devices)]
                for gi in range(num_groups)
            ],
        )

    def _get_swa_multi_group_kwargs_tp1(self, worker_key: WorkerKey) -> dict:
        """Return DSv4 SWA/state sidecar groups for a one-device worker."""
        if (
            self._swa_layer_groups is None
            or self._swa_gpu_blocks_per_group is None
            or self._swa_gpu_layouts_per_group is None
            or worker_key not in self._swa_gpu_blocks_per_group
        ):
            return {}
        per_device_blocks = self._swa_gpu_blocks_per_group[worker_key][0]
        per_device_layouts = self._swa_gpu_layouts_per_group[worker_key][0]
        if per_device_blocks is None or per_device_layouts is None:
            return {}
        return dict(
            layer_groups=self._swa_layer_groups,
            gpu_blocks_per_group=per_device_blocks,
            gpu_layouts_per_group=per_device_layouts,
        )

    def _get_swa_multi_group_kwargs_tp(self, worker_key: WorkerKey) -> dict:
        """Return SWA/state sidecar groups reshaped as [group][device]."""
        if (
            self._swa_layer_groups is None
            or self._swa_gpu_blocks_per_group is None
            or self._swa_gpu_layouts_per_group is None
            or worker_key not in self._swa_gpu_blocks_per_group
        ):
            return {}
        per_device_blocks = self._swa_gpu_blocks_per_group[worker_key]
        per_device_layouts = self._swa_gpu_layouts_per_group[worker_key]
        if per_device_blocks[0] is None or per_device_layouts[0] is None:
            return {}
        num_groups = len(self._swa_layer_groups)
        num_devices = len(per_device_blocks)
        return dict(
            layer_groups=self._swa_layer_groups,
            gpu_blocks_per_group=[
                [per_device_blocks[di][gi] for di in range(num_devices)]
                for gi in range(num_groups)
            ],
            gpu_layouts_per_group=[
                [per_device_layouts[di][gi] for di in range(num_devices)]
                for gi in range(num_groups)
            ],
        )

    def _get_layerwise_swa_kwargs(self, worker_key: WorkerKey) -> dict:
        """SWA args for LayerwiseTransferWorker (uniform or multi-group).

        When SWA is enabled, layerwise GET always binds SWA (and any C4 state
        sidecars) into the LAYERWISE worker rather than a standalone H2D worker.
        """
        if not self._has_swa:
            return {}
        swa_ssd_files = (
            self._swa_ssd_handle.get_file_list()
            if self._swa_ssd_handle is not None else None)
        swa_ssd_kv_layout = (
            self._swa_ssd_handle.kv_layout
            if self._swa_ssd_handle is not None else None)
        swa_num_blocks_per_file = (
            self._swa_ssd_handle.num_blocks_per_file
            if self._swa_ssd_handle is not None else 0)

        if self._swa_layer_groups is not None:
            mg = self._get_swa_multi_group_kwargs_tp(worker_key)
            if not mg:
                return {}
            return dict(
                swa_cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
                swa_cpu_kv_layout=self._swa_cpu_handle.kv_layout,
                swa_ssd_files=swa_ssd_files,
                swa_ssd_kv_layout=swa_ssd_kv_layout,
                swa_num_blocks_per_file=swa_num_blocks_per_file,
                swa_layer_groups=mg["layer_groups"],
                swa_gpu_blocks_per_group=mg["gpu_blocks_per_group"],
                swa_gpu_layouts_per_group=mg["gpu_layouts_per_group"],
            )

        return dict(
            swa_gpu_blocks=[
                h.get_tensor_handle_list()
                for h in self._swa_gpu_handles[worker_key]
            ],
            swa_cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
            swa_gpu_kv_layouts=[
                h.kv_layout for h in self._swa_gpu_handles[worker_key]
            ],
            swa_cpu_kv_layout=self._swa_cpu_handle.kv_layout,
            swa_dtype=self._swa_gpu_handles[worker_key][0].dtype,
            swa_ssd_files=swa_ssd_files,
            swa_ssd_kv_layout=swa_ssd_kv_layout,
            swa_num_blocks_per_file=swa_num_blocks_per_file,
        )

    def _init_workers(self) -> None:
        if self._running:
            return
        self._worker_map: Dict[TransferType, Union[WorkerHandle, Dict[WorkerKey, WorkerHandle]]] = {}

        assert self._cpu_handle is not None
        _enable_layerwise = GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer
        _fuse_swa_into_layerwise = (
            _enable_layerwise and self.cache_config.swa_multi_layer
        )
        # Use num_gpu_groups to support multi-instance mode
        # Use gpu_device_id from StorageHandle for correct CUDA device selection
        
        # H2D worker
        if not _enable_layerwise:
            if self.model_config.effective_tp_size_per_node == 1:
                self.h2d_workers: Dict[WorkerKey, WorkerHandle] = {
                    worker_key: GPUCPUTransferWorker.create_worker(
                        mp_ctx=self.mp_ctx,
                        finished_ops_queue=self.finished_ops_queue,
                        op_buffer_tensor=self.pin_buffer.get_buffer(),
                        gpu_blocks=gpu_handles[0].get_tensor_handle_list(),
                        cpu_blocks=self._cpu_handle.get_worker_tensor(),
                        gpu_kv_layout=gpu_handles[0].kv_layout,
                        cpu_kv_layout=self._cpu_handle.kv_layout,
                        dtype=gpu_handles[0].dtype,
                        gpu_device_id=gpu_handles[0].gpu_device_id,
                        use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                        use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                        transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                        transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                        compressor=self._compressors["gpu_cpu"],
                        **self._get_multi_group_kwargs_tp1(worker_key),
                    )
                    for worker_key, gpu_handles in self.gpu_handle_groups.items()
                }
            else:
                self.h2d_workers = {
                    worker_key: tpGPUCPUTransferWorker.create_worker(
                        mp_ctx=self.mp_ctx,
                        finished_ops_queue=self.finished_ops_queue,
                        op_buffer_tensor=self.pin_buffer.get_buffer(),
                        gpu_blocks=[gpu_handle.get_tensor_handle_list() for gpu_handle in gpu_handles],
                        cpu_blocks=self._cpu_handle.get_worker_tensor(),
                        gpu_kv_layouts=[gpu_handle.kv_layout for gpu_handle in gpu_handles],
                        cpu_kv_layout=self._cpu_handle.kv_layout,
                        dtype=gpu_handles[0].dtype,
                        tp_group_size=self.model_config.effective_tp_size_per_node,
                        use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                        use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                        transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                        transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                        compressor=self._compressors["gpu_cpu_tp"],
                        **self._get_multi_group_kwargs_tp(worker_key),
                    )
                    for worker_key, gpu_handles in self.gpu_handle_groups.items()
                }
            self._worker_map[TransferType.H2D] = self.h2d_workers

        # D2H worker
        if self.model_config.effective_tp_size_per_node == 1:
            self.d2h_workers: Dict[WorkerKey, WorkerHandle] = {
                worker_key: GPUCPUTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    gpu_blocks=gpu_handles[0].get_tensor_handle_list(),
                    cpu_blocks=self._cpu_handle.get_worker_tensor(),
                    gpu_kv_layout=gpu_handles[0].kv_layout,
                    cpu_kv_layout=self._cpu_handle.kv_layout,
                    dtype=gpu_handles[0].dtype,
                    gpu_device_id=gpu_handles[0].gpu_device_id,
                    use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                    use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                    transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                    transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                    compressor=self._compressors["gpu_cpu"],
                    **self._get_multi_group_kwargs_tp1(worker_key),
                )
                for worker_key, gpu_handles in self.gpu_handle_groups.items()
            }
        else:
            self.d2h_workers = {
                worker_key: tpGPUCPUTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    gpu_blocks=[gpu_handle.get_tensor_handle_list() for gpu_handle in gpu_handles],
                    cpu_blocks=self._cpu_handle.get_worker_tensor(),
                    gpu_kv_layouts=[gpu_handle.kv_layout for gpu_handle in gpu_handles],
                    cpu_kv_layout=self._cpu_handle.kv_layout,
                    dtype=gpu_handles[0].dtype,
                    tp_group_size=self.model_config.effective_tp_size_per_node,
                    use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                    use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                    transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                    transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                    compressor=self._compressors["gpu_cpu_tp"],
                    **self._get_multi_group_kwargs_tp(worker_key),
                )
                for worker_key, gpu_handles in self.gpu_handle_groups.items()
            }
        self._worker_map[TransferType.D2H] = self.d2h_workers

        if self._ssd_handle is not None and self._cpu_handle is not None:
            ssd_layer_groups = self.model_config.layer_groups
            # DISK2H worker
            if not _enable_layerwise:
                self.cpussd_read_worker: WorkerHandle = CPUSSDDiskTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor = self.pin_buffer.get_buffer(),
                    cpu_blocks=self._cpu_handle.get_worker_tensor(),
                    ssd_files=self._ssd_handle.get_file_list(),
                    cpu_kv_layout=self._cpu_handle.kv_layout,
                    ssd_kv_layout=self._ssd_handle.kv_layout,
                    dtype=self._cpu_handle.dtype,
                    num_blocks_per_file=self._ssd_handle.num_blocks_per_file,
                    cache_config=self._cache_config,
                    compressor=self._compressors["cpu_ssd"],
                    layer_groups=ssd_layer_groups,
                )
                self._worker_map[TransferType.DISK2H] = self.cpussd_read_worker

            # H2DISK worker
            self.cpussd_write_worker: WorkerHandle = CPUSSDDiskTransferWorker.create_worker(
                mp_ctx=self.mp_ctx,
                finished_ops_queue=self.finished_ops_queue,
                op_buffer_tensor = self.pin_buffer.get_buffer(),
                cpu_blocks=self._cpu_handle.get_worker_tensor(),
                ssd_files=self._ssd_handle.get_file_list(),
                cpu_kv_layout=self._cpu_handle.kv_layout,
                ssd_kv_layout=self._ssd_handle.kv_layout,
                dtype=self._cpu_handle.dtype,
                num_blocks_per_file=self._ssd_handle.num_blocks_per_file,
                cache_config=self._cache_config,
                compressor=self._compressors["cpu_ssd"],
                layer_groups=ssd_layer_groups,
            )
            self._worker_map[TransferType.H2DISK] = self.cpussd_write_worker
        if self._remote_handle is not None and self._cpu_handle is not None:
            self.remotecpu_read_worker: WorkerHandle = CPURemoteTransferWorker.create_worker(
                mp_ctx=self.mp_ctx,
                finished_ops_queue=self.finished_ops_queue,
                op_buffer_tensor = self.pin_buffer.get_buffer(),
                cpu_blocks=self._cpu_handle.get_worker_tensor(),
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
                cpu_blocks=self._cpu_handle.get_worker_tensor(),
                remote_file=self._remote_handle.get_file_list(),
                cpu_kv_layout=self._cpu_handle.kv_layout,
                remote_kv_layout=self._remote_handle.kv_layout,
                dtype=self._cpu_handle.dtype,
                remote_config_custom=self._remote_handle.remote_config_custom,
            )
            self._worker_map[TransferType.H2REMOTE] = self.remotecpu_write_worker
            self._worker_map[TransferType.REMOTE2H] = self.remotecpu_read_worker
        elif (getattr(self.cache_config, 'use_mooncake_store_backend', False)
              and self._cpu_handle is not None):
            self.mooncake_store_worker: WorkerHandle = MooncakeStoreTransferWorker.create_worker(
                mp_ctx=self.mp_ctx,
                finished_ops_queue=self.finished_ops_queue,
                op_buffer_tensor=self.pin_buffer.get_buffer(),
                cpu_blocks=self._cpu_handle.get_worker_tensor(),
                cpu_kv_layout=self._cpu_handle.kv_layout,
                dtype=self._cpu_handle.dtype,
                cache_config=self.cache_config,
                pool_kind=PoolKind.KV,
            )
            self._worker_map[TransferType.H2REMOTE] = self.mooncake_store_worker
            self._worker_map[TransferType.REMOTE2H] = self.mooncake_store_worker
            flexkv_logger.info(
                "[TransferEngine] mooncake-store workers created for H2REMOTE/REMOTE2H")
        if self.cache_config.enable_gds:
            assert self._ssd_handle is not None
            if self.cache_config.enable_nixl:
                flexkv_logger.info(
                    "[transfer_engine] GDS path using NixlTransferWorker (NIXL GDS_MT)"
                )
                if self.model_config.effective_tp_size_per_node != 1:
                    raise RuntimeError(
                        "enable_nixl requires effective_tp_size_per_node==1 (validated in KVTaskManager)"
                    )
                self.gds_workers: Dict[WorkerKey, WorkerHandle] = {
                    worker_key: NixlTransferWorker.create_worker(
                        mp_ctx=self.mp_ctx,
                        finished_ops_queue=self.finished_ops_queue,
                        op_buffer_tensor=self.pin_buffer.get_buffer(),
                        nixl_backend="GDS_MT",
                        ssd_files=self._ssd_handle.get_file_list(),
                        num_blocks_per_file=self._ssd_handle.num_blocks_per_file,
                        dtype=self._ssd_handle.dtype,
                        ssd_kv_layout=self._ssd_handle.kv_layout,
                        gpu_kv_layout=gpu_handles[0].kv_layout,
                        cpu_kv_layout=self._cpu_handle.kv_layout,
                        nixl_extra_config=self.cache_config.nixl_extra_config,
                        gpu_blocks=gpu_handles[0].get_tensor_handle_list(),
                        cpu_blocks=None,
                        gpu_device_id=gpu_handles[0].gpu_device_id,
                    )
                    for worker_key, gpu_handles in self.gpu_handle_groups.items()
                }
            elif self.model_config.effective_tp_size_per_node == 1:
                self.gds_workers: Dict[WorkerKey, WorkerHandle] = {
                    worker_key: GDSTransferWorker.create_worker(
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
                        **self._get_multi_group_kwargs_tp1(worker_key),
                    )
                    for worker_key, gpu_handles in self.gpu_handle_groups.items()
                }
            else:
                self.gds_workers = {
                    worker_key: tpGDSTransferWorker.create_worker(
                        mp_ctx=self.mp_ctx,
                        finished_ops_queue=self.finished_ops_queue,
                        op_buffer_tensor=self.pin_buffer.get_buffer(),
                        gpu_blocks=[gpu_handle.get_tensor_handle_list() for gpu_handle in gpu_handles],
                        ssd_files=self._ssd_handle.get_file_list(),
                        num_blocks_per_file=self._ssd_handle.num_blocks_per_file,
                        gpu_kv_layouts=[gpu_handle.kv_layout for gpu_handle in gpu_handles],
                        ssd_kv_layout=self._ssd_handle.kv_layout,
                        dtype=self._ssd_handle.dtype,
                        tp_group_size=self.model_config.effective_tp_size_per_node,
                        **self._get_multi_group_kwargs_tp(worker_key),
                    )
                    for worker_key, gpu_handles in self.gpu_handle_groups.items()
                }
            self._worker_map[TransferType.DISK2D] = self.gds_workers
            self._worker_map[TransferType.D2DISK] = self.gds_workers
        if GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer:
            ssd_files = {} if self._ssd_handle is None else self._ssd_handle.get_file_list()
            ssd_kv_layout = None if self._ssd_handle is None else self._ssd_handle.kv_layout
            num_blocks_per_file = 0 if self._ssd_handle is None else self._ssd_handle.num_blocks_per_file

            self.layerwise_workers: Dict[WorkerKey, WorkerHandle] = {}
            for worker_key, gpu_handles in self.gpu_handle_groups.items():
                _layerwise_eventfd_socket = build_layerwise_eventfd_socket_path(
                    dp_client_id=worker_key.dp_client_id,
                    pp_rank=worker_key.pp_rank,
                    model_config=self.model_config,
                )

                worker = LayerwiseTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    gpu_blocks=[handle.get_tensor_handle_list() for handle in gpu_handles],
                    cpu_blocks=self._cpu_handle.get_worker_tensor(),
                    ssd_files=ssd_files,
                    gpu_kv_layouts=[handle.kv_layout for handle in gpu_handles],
                    cpu_kv_layout=self._cpu_handle.kv_layout,
                    ssd_kv_layout=ssd_kv_layout,
                    dtype=gpu_handles[0].dtype,
                    tp_group_size=self.model_config.effective_tp_size_per_node,
                    layerwise_eventfd_socket=_layerwise_eventfd_socket,
                    num_blocks_per_file=num_blocks_per_file,
                    use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                    use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                    h2d_cta_num=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                    d2h_cta_num=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                    # Fuse main-KV + uniform/multi-group SWA into one LAYERWISE op.
                    **self._get_layerwise_swa_kwargs(worker_key),
                    **self._get_multi_group_kwargs_tp(worker_key),
                )
                self.layerwise_workers[worker_key] = worker

                flexkv_logger.debug(
                    f"[TransferEngine] Created layerwise worker for {worker_key}: "
                    f"effective_tp_size_per_node={self.model_config.effective_tp_size_per_node}, "
                    f"layer_groups={'yes' if self.model_config.layer_groups else 'no'}, "
                    f"has_ssd={len(ssd_files) > 0}")

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
                cpu_blocks=self._cpu_handle.get_worker_tensor(),
                cpu_kv_layout=self._cpu_handle.kv_layout,
                # TODO: get remote kv_layout, now we can assume that remote kv layout is same as current node
                remote_kv_layout=self._cpu_handle.kv_layout,
                dtype=self._cpu_handle.dtype,
                cache_config = self.cache_config,
                ssd_kv_layout = self._ssd_handle.kv_layout if self._ssd_handle else None,
                ssd_files = self._ssd_handle.get_file_list() if self._ssd_handle else None,
                num_blocks_per_file = self._ssd_handle.num_blocks_per_file if self._ssd_handle else 0,
                mooncake_config_path = getattr(self.cache_config, 'mooncake_config_path', None) or os.environ.get("MOONCAKE_CONFIG_PATH"),
            )
            # NOTE: now peerH2H and peerSSD2H op use the same worker
            if self.cache_config.enable_p2p_cpu:
                self._worker_map[TransferType.PEERH2H] = self.cpu_remote_cpu_worker
            if self.cache_config.enable_p2p_ssd:
                self._worker_map[TransferType.PEERSSD2H] = self.cpu_remote_cpu_worker

        # ---- SWA dedicated worker map ----
        # Reuses GPUCPUTransferWorker / tpGPUCPUTransferWorker exactly like the
        # main-KV H2D/D2H workers, but bound to the dedicated SWA GPU/CPU pools
        # and submitting completion onto the shared finished_ops_queue.
        # Uniform SWA uses the legacy single-group worker. DSv4 state sidecars
        # reuse this channel with heterogeneous multi-group worker arguments.
        if self._has_swa:
            self._swa_worker_map: Dict[TransferType, Dict[WorkerKey, WorkerHandle]] = {}
            # When layerwise is on, SWA H2D always runs inside LAYERWISE
            # (uniform via launch_swa_h2d_layer_, multi-group via
            # launch_swa_mg_h2d_layer_). Standalone SWA H2D workers are only
            # created when layerwise transfer is disabled.
            if not _enable_layerwise:
                if self.model_config.effective_tp_size_per_node == 1:
                    self._swa_h2d_workers: Dict[WorkerKey, WorkerHandle] = {
                        worker_key: GPUCPUTransferWorker.create_worker(
                            mp_ctx=self.mp_ctx,
                            finished_ops_queue=self.finished_ops_queue,
                            op_buffer_tensor=self.pin_buffer.get_buffer(),
                            gpu_blocks=swa_handles[0].get_tensor_handle_list(),
                            cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
                            gpu_kv_layout=swa_handles[0].kv_layout,
                            cpu_kv_layout=self._swa_cpu_handle.kv_layout,
                            dtype=swa_handles[0].dtype,
                            gpu_device_id=swa_handles[0].gpu_device_id,
                            use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                            use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                            transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                            transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                            **self._get_swa_multi_group_kwargs_tp1(worker_key),
                        )
                        for worker_key, swa_handles in self._swa_gpu_handles.items()
                    }
                else:
                    self._swa_h2d_workers = {
                        worker_key: tpGPUCPUTransferWorker.create_worker(
                            mp_ctx=self.mp_ctx,
                            finished_ops_queue=self.finished_ops_queue,
                            op_buffer_tensor=self.pin_buffer.get_buffer(),
                            gpu_blocks=[h.get_tensor_handle_list() for h in swa_handles],
                            cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
                            gpu_kv_layouts=[h.kv_layout for h in swa_handles],
                            cpu_kv_layout=self._swa_cpu_handle.kv_layout,
                            dtype=swa_handles[0].dtype,
                            tp_group_size=self.model_config.effective_tp_size_per_node,
                            use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                            use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                            transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                            transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                            **self._get_swa_multi_group_kwargs_tp(worker_key),
                        )
                        for worker_key, swa_handles in self._swa_gpu_handles.items()
                    }
                self._swa_worker_map[TransferType.H2D] = self._swa_h2d_workers
                flexkv_logger.info("TransferEngine: swa H2D workers initialized")
            # D2H swa worker
            if self.model_config.effective_tp_size_per_node == 1:
                self._swa_d2h_workers: Dict[WorkerKey, WorkerHandle] = {
                    worker_key: GPUCPUTransferWorker.create_worker(
                        mp_ctx=self.mp_ctx,
                        finished_ops_queue=self.finished_ops_queue,
                        op_buffer_tensor=self.pin_buffer.get_buffer(),
                        gpu_blocks=swa_handles[0].get_tensor_handle_list(),
                        cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
                        gpu_kv_layout=swa_handles[0].kv_layout,
                        cpu_kv_layout=self._swa_cpu_handle.kv_layout,
                        dtype=swa_handles[0].dtype,
                        gpu_device_id=swa_handles[0].gpu_device_id,
                        use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                        use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                        transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                        transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                        **self._get_swa_multi_group_kwargs_tp1(worker_key),
                    )
                    for worker_key, swa_handles in self._swa_gpu_handles.items()
                }
            else:
                self._swa_d2h_workers = {
                    worker_key: tpGPUCPUTransferWorker.create_worker(
                        mp_ctx=self.mp_ctx,
                        finished_ops_queue=self.finished_ops_queue,
                        op_buffer_tensor=self.pin_buffer.get_buffer(),
                        gpu_blocks=[h.get_tensor_handle_list() for h in swa_handles],
                        cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
                        gpu_kv_layouts=[h.kv_layout for h in swa_handles],
                        cpu_kv_layout=self._swa_cpu_handle.kv_layout,
                        dtype=swa_handles[0].dtype,
                        tp_group_size=self.model_config.effective_tp_size_per_node,
                        use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                        use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                        transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
                        transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
                        **self._get_swa_multi_group_kwargs_tp(worker_key),
                        )
                    for worker_key, swa_handles in self._swa_gpu_handles.items()
                }

            self._swa_worker_map[TransferType.D2H] = self._swa_d2h_workers
            flexkv_logger.info("TransferEngine: swa D2H workers initialized")

            if self._swa_ssd_handle is not None and self._swa_cpu_handle is not None:
                self.swa_h2disk_worker: WorkerHandle = CPUSSDDiskTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
                    ssd_files=self._swa_ssd_handle.get_file_list(),
                    cpu_kv_layout=self._swa_cpu_handle.kv_layout,
                    ssd_kv_layout=self._swa_ssd_handle.kv_layout,
                    dtype=self._swa_cpu_handle.dtype,
                    num_blocks_per_file=self._swa_ssd_handle.num_blocks_per_file,
                    cache_config=self._cache_config,
                    layer_groups=self._swa_layer_groups,
                )
                self._swa_worker_map[TransferType.H2DISK] = self.swa_h2disk_worker

                self.swa_disk2h_worker: WorkerHandle = CPUSSDDiskTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
                    ssd_files=self._swa_ssd_handle.get_file_list(),
                    cpu_kv_layout=self._swa_cpu_handle.kv_layout,
                    ssd_kv_layout=self._swa_ssd_handle.kv_layout,
                    dtype=self._swa_cpu_handle.dtype,
                    num_blocks_per_file=self._swa_ssd_handle.num_blocks_per_file,
                    cache_config=self._cache_config,
                    layer_groups=self._swa_layer_groups,
                )
                self._swa_worker_map[TransferType.DISK2H] = self.swa_disk2h_worker
                flexkv_logger.info("TransferEngine: swa CPU<->SSD workers initialized")


            # ---- SWA CPU<->Remote workers -----------------------------------
            if (getattr(self.cache_config, 'use_mooncake_store_backend', False)
                    and self._swa_cpu_handle is not None):
                self.swa_mooncake_store_worker: WorkerHandle = (
                    MooncakeStoreTransferWorker.create_worker(
                        mp_ctx=self.mp_ctx,
                        finished_ops_queue=self.finished_ops_queue,
                        op_buffer_tensor=self.pin_buffer.get_buffer(),
                        cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
                        cpu_kv_layout=self._swa_cpu_handle.kv_layout,
                        dtype=self._swa_cpu_handle.dtype,
                        cache_config=self.cache_config,
                        pool_kind=PoolKind.SWA,
                        override_global_segment_size=0,
                    ))
                self._swa_worker_map[TransferType.REMOTE2H] = self.swa_mooncake_store_worker
                self._swa_worker_map[TransferType.H2REMOTE] = self.swa_mooncake_store_worker
                flexkv_logger.info(
                    "TransferEngine: swa mooncake-store workers initialized")
            elif self._swa_remote_handle is not None and self._swa_cpu_handle is not None:
                self.swa_remotecpu_read_worker: WorkerHandle = CPURemoteTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
                    remote_file=self._swa_remote_handle.get_file_list(),
                    cpu_kv_layout=self._swa_cpu_handle.kv_layout,
                    remote_kv_layout=self._swa_remote_handle.kv_layout,
                    dtype=self._swa_cpu_handle.dtype,
                    remote_config_custom=self._swa_remote_handle.remote_config_custom,
                    enable_pcfs_sharing=self._enable_pcfs_sharing,
                )
                self.swa_remotecpu_write_worker: WorkerHandle = CPURemoteTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
                    remote_file=self._swa_remote_handle.get_file_list(),
                    cpu_kv_layout=self._swa_cpu_handle.kv_layout,
                    remote_kv_layout=self._swa_remote_handle.kv_layout,
                    dtype=self._swa_cpu_handle.dtype,
                    remote_config_custom=self._swa_remote_handle.remote_config_custom,
                )
                self._swa_worker_map[TransferType.REMOTE2H] = self.swa_remotecpu_read_worker
                self._swa_worker_map[TransferType.H2REMOTE] = self.swa_remotecpu_write_worker
                flexkv_logger.info("TransferEngine: swa CPU<->Remote workers initialized")


            if self.cache_config.enable_gds and self._swa_ssd_handle is not None:
                if self.model_config.effective_tp_size_per_node == 1:
                    self._swa_gds_workers: Dict[WorkerKey, WorkerHandle] = {
                        worker_key: GDSTransferWorker.create_worker(
                            mp_ctx=self.mp_ctx,
                            finished_ops_queue=self.finished_ops_queue,
                            op_buffer_tensor=self.pin_buffer.get_buffer(),
                            gpu_blocks=swa_handles[0].get_tensor_handle_list(),
                            ssd_files=self._swa_ssd_handle.get_file_list(),
                            num_blocks_per_file=self._swa_ssd_handle.num_blocks_per_file,
                            gpu_kv_layout=swa_handles[0].kv_layout,
                            ssd_kv_layout=self._swa_ssd_handle.kv_layout,
                            dtype=swa_handles[0].dtype,
                            gpu_device_id=swa_handles[0].gpu_device_id,
                            **self._get_swa_multi_group_kwargs_tp1(worker_key),
                        )
                        for worker_key, swa_handles in self._swa_gpu_handles.items()
                    }
                else:
                    self._swa_gds_workers = {
                        worker_key: tpGDSTransferWorker.create_worker(
                            mp_ctx=self.mp_ctx,
                            finished_ops_queue=self.finished_ops_queue,
                            op_buffer_tensor=self.pin_buffer.get_buffer(),
                            gpu_blocks=[h.get_tensor_handle_list() for h in swa_handles],
                            ssd_files=self._swa_ssd_handle.get_file_list(),
                            num_blocks_per_file=self._swa_ssd_handle.num_blocks_per_file,
                            gpu_kv_layouts=[h.kv_layout for h in swa_handles],
                            ssd_kv_layout=self._swa_ssd_handle.kv_layout,
                            dtype=swa_handles[0].dtype,
                            tp_group_size=self.model_config.effective_tp_size_per_node,
                            **self._get_swa_multi_group_kwargs_tp(worker_key),
                        )
                        for worker_key, swa_handles in self._swa_gpu_handles.items()
                    }
                self._swa_worker_map[TransferType.DISK2D] = self._swa_gds_workers
                self._swa_worker_map[TransferType.D2DISK] = self._swa_gds_workers
                flexkv_logger.info("TransferEngine: swa GDS workers initialized")
            self._has_swa = True
            # Must mirror the create condition above.
            if not _enable_layerwise:
                flexkv_logger.info(
                    f"TransferEngine: swa workers initialized "
                    f"({len(self._swa_h2d_workers)} H2D + {len(self._swa_d2h_workers)} D2H)")
            else:
                flexkv_logger.info(
                    f"TransferEngine: swa inline workers initialized "
                    f"(H2D fused into layerwise, {len(self._swa_d2h_workers)} D2H)")

        if len(self._worker_map) == 0:
            raise ValueError("No workers initialized, please check the config")

        def _wait_worker_ready(
            worker: WorkerHandle,
            transfer_type: TransferType,
            worker_key: Optional[WorkerKey] = None,
        ) -> None:
            """Wait for ready_event, but fail fast if the process already died."""
            label = (
                f"{transfer_type.name} worker {worker.worker_id}"
                + (f" key={worker_key}" if worker_key is not None else "")
            )
            while not worker.ready_event.wait(timeout=5.0):
                if not worker.process.is_alive():
                    raise RuntimeError(
                        f"{label} died during init "
                        f"(exitcode={worker.process.exitcode}); "
                        f"see worker traceback above (often CUDA OOM from "
                        f"wrong-device context on GPU0)"
                    )
                flexkv_logger.debug(f"still waiting for {label} to ready")
            flexkv_logger.debug(f"{label} is ready")

        # Wait for all main KV workers to ready
        for transfer_type, worker in self._worker_map.items():
            if isinstance(worker, dict):
                for wk, w in worker.items():
                    _wait_worker_ready(w, transfer_type, wk)
            else:
                _wait_worker_ready(worker, transfer_type)

        # Wait for all SWA dedicated workers to be ready
        if self._has_swa:

            for transfer_type, worker in self._swa_worker_map.items():
                if isinstance(worker, dict):
                    for wk, w in worker.items():
                        _wait_worker_ready(w, transfer_type, wk)
                else:
                    _wait_worker_ready(worker, transfer_type)

        # Startup assertions: verify layerwise mode worker map consistency
        if _enable_layerwise:
            assert TransferType.H2D not in self._worker_map, \
                "H2D worker should not exist in layerwise mode (fused into layerwise worker)"
            assert TransferType.DISK2H not in self._worker_map, \
                "DISK2H worker should not exist in layerwise mode (fused into layerwise worker)"
            assert TransferType.LAYERWISE in self._worker_map, \
                "LAYERWISE worker must exist when layerwise transfer is enabled"

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
                                if op_id in self._child_to_parent_op_id:
                                    # Replica op (LAYERWISE PP fan-out): decrement parent's
                                    # pending_count and finalize parent when all replicas done.
                                    parent_op_id = self._child_to_parent_op_id.pop(op_id)
                                    child_op = self._child_id_to_child.pop(op_id)
                                    free_op_from_buffer(child_op, self.pin_buffer)
                                    if op_id in self.op_id_to_nvtx_range:
                                        nvtx.end_range(self.op_id_to_nvtx_range.pop(op_id))
                                    parent_op = self.op_id_to_op[parent_op_id]
                                    parent_op.pending_count -= 1
                                    if parent_op.pending_count == 0:
                                        self._finalize_op(parent_op, finished_ops)
                                    flexkv_logger.debug(
                                        f"[TransferEngine] child op {op_id} completed, "
                                        f"parent op {parent_op_id} pending_count={parent_op.pending_count}")
                                else:
                                    op = self.op_id_to_op[op_id]
                                    op.pending_count -= 1
                                    if op.pending_count == 0:
                                        self._finalize_op(op, finished_ops)
                            except queue.Empty:
                                break
                        nvtx.end_range(nvtx_r2)

                # Exit loop if shutdown requested
                if should_shutdown:
                    break

                # End NVTX ranges for finished ops
                for op in finished_ops:
                    nvtx_range = self.op_id_to_nvtx_range.pop(op.op_id, None)
                    if nvtx_range is not None:
                        nvtx.end_range(nvtx_range)

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
                            # Unified rule for both main-KV and SWA paths:
                            # only register here when the resolved worker_map
                            # entry is a single worker (no PP fan-out). For
                            # dict-keyed entries (H2D/D2H), each replica is
                            # registered inside _assign_op_to_worker /
                            # _assign_swa_op_to_worker per PP sibling.
                            if getattr(op, "is_swa", False):
                                resolved_worker = self._swa_worker_map.get(op.transfer_type)
                            else:
                                resolved_worker = self._worker_map.get(op.transfer_type)
                            if resolved_worker is not None and not isinstance(resolved_worker, dict):
                                register_op_to_buffer(op, self.pin_buffer)
                            self._assign_op_to_worker(op)
                    # Handle completed graphs
                    for graph_id in completed_graph_ids:
                        self.completed_queue.put(CompletedOp.completed_graph(graph_id))
                nvtx.end_range(nvtx_r3)

            except Exception as e:
                flexkv_logger.error(
                    f"Error in scheduler loop: {type(e).__name__}: {e!r} "
                    f"| op_id_to_op keys={list(self.op_id_to_op.keys())[:16]} "
                    f"(total={len(self.op_id_to_op)}) "
                    f"| child->parent keys={list(self._child_to_parent_op_id.keys())[:16]} "
                    f"(total={len(self._child_to_parent_op_id)}) "
                    f"| nvtx_range keys={list(self.op_id_to_nvtx_range.keys())[:16]} "
                    f"(total={len(self.op_id_to_nvtx_range)})",
                    exc_info=True,
                )
                time.sleep(0.001)  # Fallback on error

        # Cleanup
        sel.close()
        flexkv_logger.info("TransferEngine scheduler loop stopped")

    def _finalize_op(self, op: TransferOp, finished_ops: List[TransferOp]) -> None:
        """Finalize a completed op: release pin buffer, notify upper layer, and clean up.

        Called only when op.pending_count reaches 0, i.e., all PP-sibling replica
        workers have completed this op. This ensures atomic eviction semantics.
        """
        # Unified rule: free the parent op buffer here only if the parent itself
        # was registered upstream (single-worker path). For dict-keyed (PP fan-out)
        # entries the parent was never registered; each replica was registered and
        # freed individually in the scheduler's child completion path.
        if getattr(op, "is_swa", False):
            resolved_worker = self._swa_worker_map.get(op.transfer_type)
        else:
            resolved_worker = self._worker_map.get(op.transfer_type)
        if resolved_worker is not None and not isinstance(resolved_worker, dict):
            free_op_from_buffer(op, self.pin_buffer)
        # Compute transfer metrics for this completed op.
        # Use layer_groups-aware token size so overlapping main/indexer groups
        # report their combined byte count.
        num_blocks = len(op.src_block_ids) if op.src_block_ids is not None else 0
        total_token_bytes = self.model_config.token_size_in_bytes
        total_layers = self.model_config.num_layers
        avg_bytes_per_layer = total_token_bytes // max(1, total_layers)
        token_size_in_bytes_per_pp_stage = self._num_layers_for_local_pp_stage * avg_bytes_per_layer
        num_bytes = num_blocks * self.cache_config.tokens_per_block * token_size_in_bytes_per_pp_stage
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

    @staticmethod
    def _match_pp_siblings(
        worker_map: Dict[WorkerKey, WorkerHandle],
        dp_client_id: int,
    ) -> List[WorkerKey]:
        """Return every WorkerKey whose flat DP slice equals ``dp_client_id``.

        After flattening, a single int fully identifies the DP slice —
        PP siblings are the worker_keys that share it across pp_rank.
        """
        return [wk for wk in worker_map.keys() if wk.dp_client_id == dp_client_id]

    def _assign_layerwise_op_to_workers(self, op: TransferOp) -> None:
        """Fan-out a LAYERWISE op symmetrically to every local PP-stage
        sibling worker matching ``op.dp_client_id``."""
        from flexkv.common.transfer import LayerwiseTransferOp
        assert isinstance(op, LayerwiseTransferOp)

        worker_map = self._worker_map[TransferType.LAYERWISE]
        assert isinstance(worker_map, dict), \
            "LAYERWISE worker map must be a Dict[WorkerKey, WorkerHandle]"

        sibling_keys = self._match_pp_siblings(worker_map, op.dp_client_id)
        if not sibling_keys:
            raise ValueError(
                f"No LAYERWISE worker found matching "
                f"dp_client_id={op.dp_client_id}; "
                f"available worker keys={list(worker_map.keys())}"
            )

        for wk in sibling_keys:
            replica = LayerwiseTransferOp(
                graph_id=op.graph_id,
                src_block_ids_h2d=op.src_block_ids_h2d.copy(),
                dst_block_ids_h2d=op.dst_block_ids_h2d.copy(),
                src_block_ids_disk2h=op.src_block_ids_disk2h.copy(),
                dst_block_ids_disk2h=op.dst_block_ids_disk2h.copy(),
                # SWA ids must be carried through PP fan-out replicas, otherwise
                # each PP sibling's worker would only see main-KV ids and the SWA
                # layer-fused branch in cpp would be silently skipped.
                swa_src_block_ids_h2d=op.swa_src_block_ids_h2d.copy(),
                swa_dst_block_ids_h2d=op.swa_dst_block_ids_h2d.copy(),
                swa_src_block_ids_disk2h=op.swa_src_block_ids_disk2h.copy(),
                swa_dst_block_ids_disk2h=op.swa_dst_block_ids_disk2h.copy(),
                dp_client_id=op.dp_client_id,
                counter_id=op.counter_id,
            )
            register_op_to_buffer(replica, self.pin_buffer)
            self._child_id_to_child[replica.op_id] = replica
            self._child_to_parent_op_id[replica.op_id] = op.op_id
            self.op_id_to_nvtx_range[replica.op_id] = nvtx.start_range(
                f"schedule {replica.transfer_type.name}_REPLICA op_id: {replica.op_id}, "
                f"graph_id: {replica.graph_id}, worker_key={wk}",
                color=get_nvtx_range_color(replica.graph_id))
            op.pending_count += 1
            worker_map[wk].submit_transfer(replica)
            flexkv_logger.debug(
                f"[TransferEngine] LAYERWISE fan-out: "
                f"parent_op_id={op.op_id}, replica_op_id={replica.op_id}, "
                f"worker_key={wk}, pending_count={op.pending_count}")

    def _assign_swa_op_to_worker(self, op: TransferOp) -> None:
        """Route a graph-built ``is_swa=True`` op to the SWA worker map.

        Structurally identical to the main-KV dispatch path:
          * dict worker_entry (H2D/D2H, keyed by WorkerKey for PP siblings)
            -> PP fan-out: derive one replica per sibling, register each,
               track in _child_to_parent_op_id, pending_count++ per replica.
            This is needed because each PP stage holds its own slice of SWA
            layers, exactly like the main-KV path: a single submit to one
            sibling would silently drop the other stages\' SWA data.
          * single-instance worker_entry (CPU<->SSD / CPU<->Remote)
            -> no fan-out; pending_count++ and submit op directly.
            register_op_to_buffer + op_id_to_op are done by the scheduler
            upstream for this branch, exactly like main-KV single-worker.
        """
        if op.transfer_type not in self._swa_worker_map:
            raise ValueError(f"Unsupported SWA transfer type: {op.transfer_type}")
        worker_entry = self._swa_worker_map[op.transfer_type]

        if isinstance(worker_entry, dict):
            sibling_keys = self._match_pp_siblings(worker_entry, op.dp_client_id)
            if not sibling_keys:
                raise ValueError(
                    f"No SWA_{op.transfer_type.name} worker found matching "
                    f"dp_client_id={op.dp_client_id}; "
                    f"available worker keys={list(worker_entry.keys())}"
                )
            for wk in sibling_keys:
                replica = TransferOp(
                    graph_id=op.graph_id,
                    transfer_type=op.transfer_type,
                    src_block_ids=op.src_block_ids.copy(),
                    dst_block_ids=op.dst_block_ids.copy(),
                    dp_client_id=op.dp_client_id,
                    is_swa=True,
                    mooncake_store_swa_block_hashes=(
                        list(op.mooncake_store_swa_block_hashes)
                        if op.mooncake_store_swa_block_hashes is not None else None),
                )
                register_op_to_buffer(replica, self.pin_buffer)
                self._child_id_to_child[replica.op_id] = replica
                self._child_to_parent_op_id[replica.op_id] = op.op_id
                self.op_id_to_nvtx_range[replica.op_id] = nvtx.start_range(
                    f"schedule SWA_{op.transfer_type.name}_REPLICA op_id: {replica.op_id}, "
                    f"graph_id: {replica.graph_id}, worker_key={wk}",
                    color=get_nvtx_range_color(replica.graph_id),
                )
                op.pending_count += 1
                worker_entry[wk].submit_transfer(replica)
                flexkv_logger.debug(
                    f"[TransferEngine] SWA_{op.transfer_type.name} fan-out: "
                    f"parent_op_id={op.op_id}, replica_op_id={replica.op_id}, "
                    f"worker_key={wk}, pending_count={op.pending_count}"
                )
        else:
            self.op_id_to_nvtx_range[op.op_id] = nvtx.start_range(
                f"schedule SWA_{op.transfer_type.name} op_id: {op.op_id}, "
                f"graph_id: {op.graph_id}, successors: {op.successors}",
                color=get_nvtx_range_color(op.graph_id),
            )
            op.pending_count += 1
            worker_entry.submit_transfer(op)
            flexkv_logger.debug(
                f"[TransferEngine] Submitted SWA op {op.op_id}: "
                f"type={op.transfer_type.name}, single-worker, "
                f"blocks={op.src_block_ids.size}, pending_count={op.pending_count}"
            )

    def _assign_op_to_worker(self, op: TransferOp) -> None:
        """Assign operation to appropriate worker."""
        if op.transfer_type == TransferType.VIRTUAL:
            return

        if op.is_swa:
            # SWA ops are built directly in the transfer graph (is_swa=True)
            # and routed to _swa_worker_map; they are NOT derived from main-KV
            # ops at dispatch time.
            self._assign_swa_op_to_worker(op)
            return

        if op.transfer_type not in self._worker_map:
            raise ValueError(f"Unsupported transfer type: {op.transfer_type}")

        if op.transfer_type == TransferType.LAYERWISE:
            self._assign_layerwise_op_to_workers(op)
            return

        worker = self._worker_map[op.transfer_type]
        if isinstance(worker, dict):
            sibling_keys = self._match_pp_siblings(worker, op.dp_client_id)
            if not sibling_keys:
                raise ValueError(
                    f"No MAIN_KV_{op.transfer_type.name} worker found matching "
                    f"dp_client_id={op.dp_client_id}; "
                    f"available worker keys={list(worker.keys())}"
                )
            for wk in sibling_keys:
                replica = TransferOp(
                    graph_id=op.graph_id,
                    transfer_type=op.transfer_type,
                    src_block_ids=op.src_block_ids.copy(),
                    dst_block_ids=op.dst_block_ids.copy(),
                    dp_client_id=op.dp_client_id,
                    mooncake_store_block_hashes=(
                        op.mooncake_store_block_hashes.copy()
                        if op.mooncake_store_block_hashes is not None else None),
                )
                register_op_to_buffer(replica, self.pin_buffer)
                self._child_id_to_child[replica.op_id] = replica
                self._child_to_parent_op_id[replica.op_id] = op.op_id
                self.op_id_to_nvtx_range[replica.op_id] = nvtx.start_range(
                    f"schedule {replica.transfer_type.name}_REPLICA op_id: {replica.op_id}, "
                    f"graph_id: {replica.graph_id}, worker_key={wk}",
                    color=get_nvtx_range_color(replica.graph_id))
                op.pending_count += 1
                worker[wk].submit_transfer(replica)
                flexkv_logger.debug(
                    f"[TransferEngine] MAIN_KV_{op.transfer_type.name} fan-out: "
                    f"parent_op_id={op.op_id}, replica_op_id={replica.op_id}, "
                    f"worker_key={wk}, pending_count={op.pending_count}")
        else:
            self.op_id_to_nvtx_range[op.op_id] = nvtx.start_range(
                f"schedule {op.transfer_type.name} "
                f"op_id: {op.op_id}, graph_id: {op.graph_id}, "
                f"successors: {op.successors}",
                color=get_nvtx_range_color(op.graph_id),
            )
            op.pending_count += 1
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

            # shutdown main KV workers
            for worker in self._worker_map.values():
                if isinstance(worker, dict):
                    for w in worker.values():
                        w.shutdown()
                else:
                    worker.shutdown()
            # shutdown SWA dedicated workers
            for worker in getattr(self, "_swa_worker_map", {}).values():
                if isinstance(worker, dict):
                    for w in worker.values():
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
