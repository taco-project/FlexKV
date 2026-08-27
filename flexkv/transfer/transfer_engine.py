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
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import contextlib
import nvtx
import numpy as np
import torch

from flexkv.common.debug import flexkv_logger
from flexkv.common.pool import PoolId
from flexkv.common.storage import StorageHandle
from flexkv.common.transfer import TransferOp, TransferOpGraph, TransferType, CompletedOp, WorkerKey
from flexkv.common.transfer import get_nvtx_range_color
from flexkv.transfer.scheduler import TransferScheduler
from flexkv.transfer import trace
from flexkv.transfer.workers import (
    WorkerHandle,
    CPUSSDDiskTransferWorker,
    CPURemoteTransferWorker,
    GPUCPUTransferWorker,
    GDSTransferWorker,
    PEER2CPUTransferWorker,
)
# NIXL / PCFS / mooncake-store are I/O *engines*, not edges: each plugs into
# the worker that already owns its edge. See flexkv/transfer/backends.py.
from flexkv.transfer.backends import (
    MooncakeStoreBackend,
    NixlFileBackend,
    PcfsRemoteBackend,
)
from flexkv.external.mooncake_store_keys import PoolKind
from flexkv.transfer.compression import build_compressors, NullCompressionStrategy
from flexkv.transfer.completion import CompletionContract
from flexkv.transfer.layer_eventfd import build_layerwise_eventfd_socket_path
from flexkv.transfer.worker_op import WorkerTransferResult
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


def _te_bounded_cuda_sync(timeout_s: float) -> None:
    """torch.cuda.synchronize() with a wall-clock cap.

    Runs the sync in a daemon thread so a wedged GPU cannot prevent
    TransferEngine.shutdown from returning. Failure / timeout is logged
    but not raised — this is called from the shutdown finally.
    """
    if not (torch.cuda.is_available() and torch.cuda.is_initialized()):
        return
    done = threading.Event()
    err: List[BaseException] = []

    def _run() -> None:
        try:
            torch.cuda.synchronize()
        except BaseException as e:  # noqa: BLE001
            err.append(e)
        finally:
            done.set()

    t = threading.Thread(
        target=_run, name="flexkv-te-cuda-drain", daemon=True,
    )
    t.start()
    if not done.wait(timeout=timeout_s):
        flexkv_logger.warning(
            f"TransferEngine.shutdown: cuda synchronize did not finish in "
            f"{timeout_s:.0f}s (GPU likely wedged); continuing"
        )
        return
    if err:
        flexkv_logger.warning(
            f"TransferEngine.shutdown: cuda synchronize failed: {err[0]!r}"
        )

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
        self._gpu_mappings_suspended = False

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

        # Failure propagation state; see _handle_failed_op for the flow.
        # Graphs with a failed op, awaiting drain of their in-flight ops
        # before the graph-level failure message is emitted.
        self._failed_graph_ids: Set[int] = set()
        # Ops with at least one failed replica: must be discarded, not
        # finalized, when their pending_count drains to zero.
        self._failed_parent_op_ids: Set[int] = set()

    # ---- multi-group worker kwargs -------------------------------------------
    # There used to be four of these (main/SWA x TP=1/TP>1).  They differed only
    # in which three attributes they read and whether they transposed, so the
    # pool is a parameter and the transpose is a branch.  The design doc calls
    # the duplication out by name.
    def _multi_group_pool(self, pool_id: PoolId) -> tuple:
        """(layer_groups, blocks_by_worker_key, layouts_by_worker_key)."""
        if pool_id is PoolId.SWA:
            return (self._swa_layer_groups,
                    self._swa_gpu_blocks_per_group,
                    self._swa_gpu_layouts_per_group)
        return (self.model_config.layer_groups,
                self._gpu_blocks_per_group,
                self._gpu_layouts_per_group)

    def _get_multi_group_kwargs(self, worker_key: WorkerKey, *,
                                pool_id: PoolId = PoolId.FULL_KV) -> dict:
        """Multi-group handles/layouts for one worker, or {} if not applicable.

        Transposes the registry's [device][group] into the [group][device] a
        worker consumes.  There used to be a ``tp=False`` mode returning a
        single device's own [group] lists; its only callers were the singular
        GPU<->CPU and GDS worker classes, and both have been merged into their
        TP-shaped counterparts (one device is just ``num_devices == 1``).
        """
        layer_groups, blocks_by_key, layouts_by_key = self._multi_group_pool(pool_id)
        if (layer_groups is None or blocks_by_key is None
                or layouts_by_key is None or worker_key not in blocks_by_key):
            return {}

        per_device_blocks = blocks_by_key[worker_key]
        per_device_layouts = layouts_by_key[worker_key]
        # Device 0 standing in for "this worker registered nothing" is the
        # historical emptiness check; keep it rather than inventing a new one.
        if per_device_blocks[0] is None or per_device_layouts[0] is None:
            return {}

        num_groups = len(layer_groups)
        num_devices = len(per_device_blocks)
        return dict(
            layer_groups=layer_groups,
            gpu_blocks_per_group=[
                [per_device_blocks[di][gi] for di in range(num_devices)]
                for gi in range(num_groups)
            ],
            gpu_layouts_per_group=[
                [per_device_layouts[di][gi] for di in range(num_devices)]
                for gi in range(num_groups)
            ],
        )

    def _get_swa_kwargs(self, worker_key: WorkerKey) -> dict:
        """SWA pool args, for any worker that carries SWA as a second pool.

        Both ``GPUCPUTransferWorker`` and ``LayerwiseTransferWorker`` take SWA
        this way now: it is one more pool on the worker that already owns this
        TP group's GPUs, not a worker of its own. Which is why this is no
        longer ``_get_layerwise_swa_kwargs`` -- nothing in it was layerwise.

        No SSD args: neither worker reads SSD -- the SSD->CPU read is a
        standalone DISK2H op the merge emits.
        """
        if not self._has_swa:
            return {}
        # Uniform and multi-group are mutually exclusive, not layered: a worker
        # given both cannot tell which description of the same pool is
        # authoritative, and LayerwiseWorker rejects the pair outright.
        if self._swa_layer_groups is not None:
            mg = self._get_multi_group_kwargs(worker_key, pool_id=PoolId.SWA)
            if not mg:
                return {}
            return dict(
                swa_cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
                swa_cpu_kv_layout=self._swa_cpu_handle.kv_layout,
                swa_dtype=self._swa_gpu_handles[worker_key][0].dtype,
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
        )

    def _create_gpu_cpu_worker(
        self, worker_key: WorkerKey, gpu_handles: list,
        completion: CompletionContract = CompletionContract.WHOLE,
        layerwise_eventfd_socket: Optional[str] = None,
    ) -> WorkerHandle:
        """Spawn one CPU<->GPU worker for a TP group of any size.

        H2D and D2H differ only in which transfer types they are registered
        for -- the worker itself handles both directions -- so they share this.
        SWA rides along as a second pool for the same reason: it is the same
        GPUs, the same direction and the same layout family, differing only in
        which pool the block ids index.

        LAYERWISE is the same worker once more, differing only in
        ``completion``: PER_LAYER makes it open the consumer's eventfd UDS and
        post a fd per model layer. That is why there is no layerwise worker
        class any more -- per-layer completion is a contract, not a pool
        layout, and everything else about the transfer is identical.
        """
        assert self._cpu_handle is not None
        return GPUCPUTransferWorker.create_worker(
            mp_ctx=self.mp_ctx,
            finished_ops_queue=self.finished_ops_queue,
            op_buffer_tensor=self.pin_buffer.get_buffer(),
            gpu_blocks=[h.get_tensor_handle_list() for h in gpu_handles],
            cpu_blocks=self._cpu_handle.get_worker_tensor(),
            gpu_kv_layouts=[h.kv_layout for h in gpu_handles],
            cpu_kv_layout=self._cpu_handle.kv_layout,
            dtype=gpu_handles[0].dtype,
            tp_group_size=self.model_config.effective_tp_size_per_node,
            use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
            use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
            transfer_num_cta_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_h2d,
            transfer_num_cta_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_num_cta_d2h,
            compressor=self._compressors["gpu_cpu"],
            completion=completion,
            layerwise_eventfd_socket=layerwise_eventfd_socket,
            **self._get_multi_group_kwargs(worker_key),
            **self._get_swa_kwargs(worker_key),
        )

    def _create_gds_worker(
        self, worker_key: WorkerKey, gpu_handles: list, ssd_handle: Any,
        *, pool_id: PoolId = PoolId.FULL_KV, backend: Any = None,
    ) -> WorkerHandle:
        """Spawn one GPU<->SSD GDS worker for a TP group of any size.

        Same merge as ``_create_gpu_cpu_worker``: ``TPGDSTransferThreadGroup``
        already spawns one thread, one stream and one ``GDSManager`` per GPU,
        and ``num_gpus == 1`` *is* the non-TP case, so there is nothing for a
        separate singular worker class to do. This used to be four call sites
        -- main/SWA crossed with tp==1/tp>1 -- differing only in whether the
        gpu args were scalars or one-element lists.

        ``ssd_handle`` is this pool's SSD handle. Different pools are different
        files here, which is why a non-default pool's GDS edge is still its own
        worker rather than a second pool the way GPU<->CPU is.

        ``backend`` replaces cuFile with another engine on the same edge --
        ``NixlFileBackend("GDS_MT", ...)`` is the one in tree. That used to be
        a whole separate worker class re-deriving the geometry computed here.
        """
        return GDSTransferWorker.create_worker(
            mp_ctx=self.mp_ctx,
            finished_ops_queue=self.finished_ops_queue,
            op_buffer_tensor=self.pin_buffer.get_buffer(),
            gpu_blocks=[h.get_tensor_handle_list() for h in gpu_handles],
            ssd_files=ssd_handle.get_file_list(),
            num_blocks_per_file=ssd_handle.num_blocks_per_file,
            gpu_kv_layouts=[h.kv_layout for h in gpu_handles],
            ssd_kv_layout=ssd_handle.kv_layout,
            # Which side names the dtype differs by pool, and the two call
            # sites disagreed before the merge: the main path took the SSD
            # handle's, SWA the GPU handle's. They should agree; preserved
            # rather than unified so the merge stays behaviour-preserving.
            dtype=(ssd_handle.dtype if pool_id is PoolId.FULL_KV
                   else gpu_handles[0].dtype),
            tp_group_size=self.model_config.effective_tp_size_per_node,
            backend=backend,
            **self._get_multi_group_kwargs(worker_key, pool_id=pool_id),
        )

    def _init_workers(self) -> None:
        if self._running:
            return
        # Registry is per-pool and created on demand; clear rather than
        # rebind, so a retry after a rolled-back init starts from empty
        # without leaving a stale dict reachable through the compat views.
        self._workers.clear()

        assert self._cpu_handle is not None
        # When layerwise is on, SWA/state H2D is always fused into the LAYERWISE
        # worker (no standalone swa_multi_layer switch).
        _enable_layerwise = GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer
        # Use num_gpu_groups to support multi-instance mode
        # Use gpu_device_id from StorageHandle for correct CUDA device selection
        
        # H2D / D2H workers. One worker class for every TP width: tp==1 is
        # num_gpus==1 inside TPTransferThreadGroup, which spawns one thread and
        # one stream per GPU either way.
        if not _enable_layerwise:
            self.h2d_workers: Dict[WorkerKey, WorkerHandle] = {
                worker_key: self._create_gpu_cpu_worker(worker_key, gpu_handles)
                for worker_key, gpu_handles in self.gpu_handle_groups.items()
            }
            self._register_worker(PoolId.FULL_KV, TransferType.H2D, self.h2d_workers)

        self.d2h_workers: Dict[WorkerKey, WorkerHandle] = {
            worker_key: self._create_gpu_cpu_worker(worker_key, gpu_handles)
            for worker_key, gpu_handles in self.gpu_handle_groups.items()
        }
        self._register_worker(PoolId.FULL_KV, TransferType.D2H, self.d2h_workers)

        if self._ssd_handle is not None and self._cpu_handle is not None:
            ssd_layer_groups = self.model_config.layer_groups
            # DISK2H worker. Under layerwise the merge emits a standalone DISK2H
            # op the LAYERWISE op depends on, so the worker is unconditional.
            if _enable_layerwise:
                # The layerwise per-layer H2D reads raw CPU blocks; a
                # compressing DISK2H worker would write a different byte layout
                # than it expects. check_engine_nvcomp_enable() already disables
                # nvcomp under layerwise, so this only guards against that
                # coupling being broken later.
                assert isinstance(self._compressors["cpu_ssd"],
                                  NullCompressionStrategy), \
                    ("layerwise requires an uncompressed CPU<->SSD path "
                     "(layerwise H2D reads raw CPU blocks)")
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
            self._register_worker(PoolId.FULL_KV, TransferType.DISK2H, self.cpussd_read_worker)

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
            self._register_worker(PoolId.FULL_KV, TransferType.H2DISK, self.cpussd_write_worker)
        if self._remote_handle is not None and self._cpu_handle is not None:
            self.remotecpu_read_worker: WorkerHandle = CPURemoteTransferWorker.create_worker(
                mp_ctx=self.mp_ctx,
                finished_ops_queue=self.finished_ops_queue,
                op_buffer_tensor = self.pin_buffer.get_buffer(),
                cpu_blocks=self._cpu_handle.get_worker_tensor(),
                cpu_kv_layout=self._cpu_handle.kv_layout,
                dtype=self._cpu_handle.dtype,
                backend=PcfsRemoteBackend(
                    remote_files=self._remote_handle.get_file_list(),
                    remote_kv_layout=self._remote_handle.kv_layout,
                    remote_config_custom=self._remote_handle.remote_config_custom,
                    enable_pcfs_sharing=self._enable_pcfs_sharing,
                ),
            )
            self.remotecpu_write_worker: WorkerHandle = CPURemoteTransferWorker.create_worker(
                mp_ctx=self.mp_ctx,
                finished_ops_queue=self.finished_ops_queue,
                op_buffer_tensor = self.pin_buffer.get_buffer(),
                cpu_blocks=self._cpu_handle.get_worker_tensor(),
                cpu_kv_layout=self._cpu_handle.kv_layout,
                dtype=self._cpu_handle.dtype,
                backend=PcfsRemoteBackend(
                    remote_files=self._remote_handle.get_file_list(),
                    remote_kv_layout=self._remote_handle.kv_layout,
                    remote_config_custom=self._remote_handle.remote_config_custom,
                ),
            )
            self._register_worker(PoolId.FULL_KV, TransferType.H2REMOTE, self.remotecpu_write_worker)
            self._register_worker(PoolId.FULL_KV, TransferType.REMOTE2H, self.remotecpu_read_worker)
        elif (getattr(self.cache_config, 'use_mooncake_store_backend', False)
              and self._cpu_handle is not None):
            self.mooncake_store_worker: WorkerHandle = CPURemoteTransferWorker.create_worker(
                mp_ctx=self.mp_ctx,
                finished_ops_queue=self.finished_ops_queue,
                op_buffer_tensor=self.pin_buffer.get_buffer(),
                cpu_blocks=self._cpu_handle.get_worker_tensor(),
                cpu_kv_layout=self._cpu_handle.kv_layout,
                dtype=self._cpu_handle.dtype,
                backend=MooncakeStoreBackend(
                    cache_config=self.cache_config,
                    pool_kind=PoolKind.KV,
                ),
            )
            self._register_worker(PoolId.FULL_KV, TransferType.H2REMOTE, self.mooncake_store_worker)
            self._register_worker(PoolId.FULL_KV, TransferType.REMOTE2H, self.mooncake_store_worker)
            flexkv_logger.info(
                "[TransferEngine] mooncake-store workers created for H2REMOTE/REMOTE2H")
        if self.cache_config.enable_gds:
            assert self._ssd_handle is not None
            if self.cache_config.enable_nixl:
                flexkv_logger.info(
                    "[transfer_engine] GDS edge using the NIXL GDS_MT backend"
                )
                if self.model_config.effective_tp_size_per_node != 1:
                    raise RuntimeError(
                        "enable_nixl requires effective_tp_size_per_node==1 (validated in KVTaskManager)"
                    )
                # Same worker, same edge geometry, different engine: only the
                # library that issues the read changes.
                self.gds_workers: Dict[WorkerKey, WorkerHandle] = {
                    worker_key: self._create_gds_worker(
                        worker_key, gpu_handles, self._ssd_handle,
                        backend=NixlFileBackend(
                            "GDS_MT",
                            self._ssd_handle.get_file_list(),
                            self.cache_config.nixl_extra_config,
                        ),
                    )
                    for worker_key, gpu_handles in self.gpu_handle_groups.items()
                }
            else:
                self.gds_workers: Dict[WorkerKey, WorkerHandle] = {
                    worker_key: self._create_gds_worker(
                        worker_key, gpu_handles, self._ssd_handle)
                    for worker_key, gpu_handles in self.gpu_handle_groups.items()
                }
            self._register_worker(PoolId.FULL_KV, TransferType.DISK2D, self.gds_workers)
            self._register_worker(PoolId.FULL_KV, TransferType.D2DISK, self.gds_workers)
        if GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer:
            # Same worker class as H2D/D2H, same pools, same regions. The only
            # difference is the completion contract: PER_LAYER makes it receive
            # the consumer's eventfds and post one per model layer. SWA and the
            # multi-group state pools ride along exactly as they do for H2D --
            # they are pools of this worker, not a separate worker.
            self.layerwise_workers: Dict[WorkerKey, WorkerHandle] = {}
            for worker_key, gpu_handles in self.gpu_handle_groups.items():
                self.layerwise_workers[worker_key] = self._create_gpu_cpu_worker(
                    worker_key,
                    gpu_handles,
                    completion=CompletionContract.PER_LAYER,
                    layerwise_eventfd_socket=build_layerwise_eventfd_socket_path(
                        dp_client_id=worker_key.dp_client_id,
                        pp_rank=worker_key.pp_rank,
                        model_config=self.model_config,
                    ),
                )

                flexkv_logger.debug(
                    f"[TransferEngine] Created layerwise worker for {worker_key}: "
                    f"effective_tp_size_per_node={self.model_config.effective_tp_size_per_node}, "
                    f"layer_groups={'yes' if self.model_config.layer_groups else 'no'}")

            self._register_worker(PoolId.FULL_KV, TransferType.LAYERWISE, self.layerwise_workers)

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
                mooncake_config_path = (getattr(self.cache_config, 'mooncake_config_path', None)
                                        or os.environ.get("MOONCAKE_CONFIG_PATH")),
            )
            # NOTE: now peerH2H and peerSSD2H op use the same worker
            if self.cache_config.enable_p2p_cpu:
                self._register_worker(PoolId.FULL_KV, TransferType.PEERH2H, self.cpu_remote_cpu_worker)
            if self.cache_config.enable_p2p_ssd:
                self._register_worker(PoolId.FULL_KV, TransferType.PEERSSD2H, self.cpu_remote_cpu_worker)

        # ---- SWA workers for the tiers that own separate storage -----------
        # GPU<->CPU is NOT here any more: SWA is a second pool inside the same
        # GPUCPUTransferWorker (see _get_swa_kwargs and worker._Pool), because
        # it is the same GPUs and the same direction with only the block ids
        # pointing at a different pool. What remains are the tiers where SWA
        # really does address different storage -- its own SSD files, its own
        # remote namespace -- and where there is therefore nothing to merge.
        if self._has_swa:
            if self._swa_ssd_handle is not None and self._swa_cpu_handle is not None:
                swa_h2disk_worker = CPUSSDDiskTransferWorker.create_worker(
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
                self._register_worker(PoolId.SWA, TransferType.H2DISK, swa_h2disk_worker)

                swa_disk2h_worker = CPUSSDDiskTransferWorker.create_worker(
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
                self._register_worker(PoolId.SWA, TransferType.DISK2H, swa_disk2h_worker)
                flexkv_logger.info("TransferEngine: swa CPU<->SSD workers initialized")


            # ---- SWA CPU<->Remote workers -----------------------------------
            if (getattr(self.cache_config, 'use_mooncake_store_backend', False)
                    and self._swa_cpu_handle is not None):
                swa_mooncake_store_worker = (
                    CPURemoteTransferWorker.create_worker(
                        mp_ctx=self.mp_ctx,
                        finished_ops_queue=self.finished_ops_queue,
                        op_buffer_tensor=self.pin_buffer.get_buffer(),
                        cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
                        cpu_kv_layout=self._swa_cpu_handle.kv_layout,
                        dtype=self._swa_cpu_handle.dtype,
                        backend=MooncakeStoreBackend(
                            cache_config=self.cache_config,
                            pool_kind=PoolKind.SWA,
                            override_global_segment_size=0,
                        ),
                    ))
                self._register_worker(PoolId.SWA, TransferType.REMOTE2H, swa_mooncake_store_worker)
                self._register_worker(PoolId.SWA, TransferType.H2REMOTE, swa_mooncake_store_worker)
                flexkv_logger.info(
                    "TransferEngine: swa mooncake-store workers initialized")
            elif self._swa_remote_handle is not None and self._swa_cpu_handle is not None:
                swa_remotecpu_read_worker = CPURemoteTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
                    cpu_kv_layout=self._swa_cpu_handle.kv_layout,
                    dtype=self._swa_cpu_handle.dtype,
                    backend=PcfsRemoteBackend(
                        remote_files=self._swa_remote_handle.get_file_list(),
                        remote_kv_layout=self._swa_remote_handle.kv_layout,
                        remote_config_custom=self._swa_remote_handle.remote_config_custom,
                        enable_pcfs_sharing=self._enable_pcfs_sharing,
                    ),
                )
                swa_remotecpu_write_worker = CPURemoteTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor=self.pin_buffer.get_buffer(),
                    cpu_blocks=self._swa_cpu_handle.get_worker_tensor(),
                    cpu_kv_layout=self._swa_cpu_handle.kv_layout,
                    dtype=self._swa_cpu_handle.dtype,
                    backend=PcfsRemoteBackend(
                        remote_files=self._swa_remote_handle.get_file_list(),
                        remote_kv_layout=self._swa_remote_handle.kv_layout,
                        remote_config_custom=self._swa_remote_handle.remote_config_custom,
                    ),
                )
                self._register_worker(PoolId.SWA, TransferType.REMOTE2H, swa_remotecpu_read_worker)
                self._register_worker(PoolId.SWA, TransferType.H2REMOTE, swa_remotecpu_write_worker)
                flexkv_logger.info("TransferEngine: swa CPU<->Remote workers initialized")


            if self.cache_config.enable_gds and self._swa_ssd_handle is not None:
                # SWA GDS stays a worker of its own -- unlike GPU<->CPU, this
                # tier really does address different storage (its own SSD
                # files), so there is no pool to merge it into.
                swa_gds_workers: Dict[WorkerKey, WorkerHandle] = {
                    worker_key: self._create_gds_worker(
                        worker_key, swa_handles, self._swa_ssd_handle,
                        pool_id=PoolId.SWA)
                    for worker_key, swa_handles in self._swa_gpu_handles.items()
                }
                self._register_worker(PoolId.SWA, TransferType.DISK2D, swa_gds_workers)
                self._register_worker(PoolId.SWA, TransferType.D2DISK, swa_gds_workers)
                flexkv_logger.info("TransferEngine: swa GDS workers initialized")
            flexkv_logger.info(
                "TransferEngine: swa GPU<->CPU is a pool on the main workers; "
                f"separate-storage swa workers: "
                f"{sorted(t.name for t in self._workers.get(PoolId.SWA, {}))}")

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

        # Wait for every worker of every pool. One loop: a SWA-only worker is
        # a worker, and skipping it used to depend on a separate ``_has_swa``
        # test that could disagree with what was actually registered.
        # ``_collect_worker_handles`` is not reused here because the label
        # wants the transfer type and worker key, which it drops.
        for worker_map in self._workers.values():
            for transfer_type, worker in worker_map.items():
                if isinstance(worker, dict):
                    for wk, w in worker.items():
                        _wait_worker_ready(w, transfer_type, wk)
                else:
                    _wait_worker_ready(worker, transfer_type)

        # Startup assertions: verify layerwise mode worker map consistency
        if _enable_layerwise:
            assert TransferType.H2D not in self._worker_map, \
                "H2D worker should not exist in layerwise mode (fused into layerwise worker)"
            # The SSD read is a standalone DISK2H op the merge emits; it has
            # nowhere to run without this worker.
            assert (self._ssd_handle is None or self._cpu_handle is None
                    or TransferType.DISK2H in self._worker_map), \
                ("DISK2H worker must exist under layerwise "
                 "(the LAYERWISE op does not perform the SSD read)")
            assert TransferType.LAYERWISE in self._worker_map, \
                "LAYERWISE worker must exist when layerwise transfer is enabled"

        # Start scheduler thread
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self._scheduler_thread.start()

    # ---- the worker registry -------------------------------------------------
    # One registry keyed by (pool, transfer type). There used to be two maps,
    # ``_worker_map`` and ``_swa_worker_map``, walked by two loops, cleared in
    # two places, dispatched through two branches -- all because the pool was a
    # bool that had to be turned back into a container by every reader. A pool
    # is a key, so it keys the map.
    #
    # A pool that has no worker of its own for a transfer type falls back to
    # FULL_KV's: for GPU<->CPU, SWA *is* a pool on the main worker, and the
    # miss is the answer, not an error. Only the tiers where SWA addresses
    # genuinely separate storage (its own SSD files, its own remote namespace)
    # register under PoolId.SWA.

    @property
    def _workers(self) -> Dict[PoolId, Dict[TransferType, Any]]:
        """``{pool_id: {transfer_type: handle-or-WorkerKey-dict}}``.

        Created on demand rather than in ``__init__`` so the rollback and
        shutdown paths -- which run when init failed partway -- and the tests
        that build a bare engine with ``object.__new__`` all see an empty
        registry instead of an ``AttributeError``.
        """
        workers = self.__dict__.get("_pool_workers")
        if workers is None:
            workers = self.__dict__["_pool_workers"] = {}
        return workers

    def _register_worker(
        self,
        pool_id: PoolId,
        transfer_type: TransferType,
        worker: Union[WorkerHandle, Dict[WorkerKey, WorkerHandle]],
    ) -> None:
        self._workers.setdefault(pool_id, {})[transfer_type] = worker

    @property
    def _worker_map(self) -> Dict[TransferType, Union[WorkerHandle, Dict[WorkerKey, WorkerHandle]]]:
        """The full-KV workers, as the flat map this class has always exposed.

        A live view, not a copy: ``self._worker_map[X] = w`` still registers,
        and the tests that build a bare engine by assigning this attribute
        still work (see the setter).
        """
        return self._workers.setdefault(PoolId.FULL_KV, {})

    @_worker_map.setter
    def _worker_map(
        self, value: Dict[TransferType, Union[WorkerHandle, Dict[WorkerKey, WorkerHandle]]]
    ) -> None:
        self._workers[PoolId.FULL_KV] = value

    def _collect_worker_handles(self) -> List[WorkerHandle]:
        """Every distinct worker handle this engine owns, each listed once.

        The registry is keyed by ``(pool, TransferType)``, and one worker
        answers several of those keys -- mooncake-store and the peer worker
        each serve a read and a write type, GDS serves DISK2D and D2DISK, and
        a GPU<->CPU worker serves both pools -- so a plain walk yields the same
        handle repeatedly. That matters because the caller starts one shutdown
        *thread per element*: duplicates meant two threads racing on one
        process's join/terminate/close.

        Deduped by identity, not by ``worker_id``: two handles are the same
        worker only if they are the same object, and identity needs no
        assumption about how ids are assigned. Insertion order is preserved so
        shutdown logs stay stable.
        """
        handles: List[WorkerHandle] = []
        seen: Set[int] = set()

        def _add(handle: WorkerHandle) -> None:
            if id(handle) not in seen:
                seen.add(id(handle))
                handles.append(handle)

        for worker_map in self._workers.values():
            for worker in worker_map.values():
                if isinstance(worker, dict):
                    for sub in worker.values():
                        _add(sub)
                else:
                    _add(worker)
        return handles

    def _shutdown_worker_handles(self, handles: List[WorkerHandle]) -> None:
        """Stop worker processes in parallel (send sentinel + join/unregister)."""
        if not handles:
            return
        flexkv_logger.info(
            f"TransferEngine: stopping {len(handles)} worker(s) in parallel"
        )

        def _shutdown_one(handle: WorkerHandle) -> None:
            try:
                handle.shutdown()
            except Exception as e:
                flexkv_logger.error(
                    f"Error shutting down worker {handle.worker_id}: {e}"
                )

        threads = [
            threading.Thread(
                target=_shutdown_one,
                args=(h,),
                name=f"flexkv-worker-shutdown-{h.worker_id}",
            )
            for h in handles
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def _rollback_init_workers(self, err: BaseException) -> None:
        """Best-effort cleanup when spawn/ready fails before engine is running."""
        handles = self._collect_worker_handles()
        if not handles:
            return
        flexkv_logger.error(
            f"TransferEngine init failed ({err}); "
            f"rolling back {len(handles)} already-created worker(s)"
        )
        self._shutdown_worker_handles(handles)
        self._workers.clear()

    def start(self) -> None:
        try:
            self._init_workers()
        except Exception as e:
            # Covers: mid-spawn exception, ready timeout/death, startup asserts.
            # Child-side __init__ failure also unpins inside the worker process;
            # this rolls back sibling workers that already became ready.
            if not self._running:
                self._rollback_init_workers(e)
            raise

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
                                payload = self.finished_ops_queue.get_nowait()
                                # Payload forms:
                                #   WorkerTransferResult (partial block outcomes)
                                #   int (legacy success)
                                #   (op_id|WorkerTransferResult, ok)
                                #   (op_id|WorkerTransferResult, ok, metrics)
                                op_succeeded = True
                                metrics = None
                                block_results = None
                                if isinstance(payload, WorkerTransferResult):
                                    op_id = payload.transfer_op_id
                                    block_results = payload.block_results
                                elif isinstance(payload, tuple):
                                    if len(payload) >= 3:
                                        first, op_succeeded, metrics = (
                                            payload[0], payload[1], payload[2])
                                    else:
                                        first, op_succeeded = payload[0], payload[1]
                                    if isinstance(first, WorkerTransferResult):
                                        op_id = first.transfer_op_id
                                        block_results = first.block_results
                                    else:
                                        op_id = first
                                else:
                                    op_id = payload
                                if not op_succeeded:
                                    # Keep trace state consistent even on failure.
                                    trace.dec_inflight()
                                    trace.consume_submit_ns(op_id)
                                    self._handle_failed_op(op_id)
                                    continue
                                if op_id in self._child_to_parent_op_id:
                                    # Replica op (LAYERWISE PP fan-out): decrement parent's
                                    # pending_count and finalize parent when all replicas done.
                                    parent_op_id = self._child_to_parent_op_id.pop(op_id)
                                    child_op = self._child_id_to_child.pop(op_id)
                                    self._merge_block_results(child_op, block_results)
                                    free_op_from_buffer(child_op, self.pin_buffer)
                                    if op_id in self.op_id_to_nvtx_range:
                                        nvtx.end_range(self.op_id_to_nvtx_range.pop(op_id))
                                    self._emit_xfer_trace(op_id, metrics)
                                    parent_op = self.op_id_to_op[parent_op_id]
                                    self._merge_block_results(
                                        parent_op, child_op.block_results)
                                    parent_op.pending_count -= 1
                                    if parent_op.pending_count == 0:
                                        self._finalize_or_discard(parent_op, finished_ops)
                                    flexkv_logger.debug(
                                        f"[TransferEngine] child op {op_id} completed, "
                                        f"parent op {parent_op_id} pending_count={parent_op.pending_count}")
                                else:
                                    op = self.op_id_to_op[op_id]
                                    self._merge_block_results(op, block_results)
                                    op.pending_count -= 1
                                    self._emit_xfer_trace(op_id, metrics)
                                    if op.pending_count == 0:
                                        self._finalize_or_discard(op, finished_ops)
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
                            if self._op_buffer_registered_here(op):
                                register_op_to_buffer(op, self.pin_buffer)
                            self._assign_op_to_worker(op)
                    # Handle completed graphs
                    for graph_id in completed_graph_ids:
                        self.completed_queue.put(CompletedOp.completed_graph(graph_id))
                nvtx.end_range(nvtx_r3)

                # Outside the dispatch block: a tick may consist solely of a
                # failure report, with no finished op and no new graph.
                if self._failed_graph_ids:
                    self._emit_drained_graph_failures()

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

    def _op_buffer_registered_here(self, op: TransferOp) -> bool:
        """The 'unified rule' shared by dispatch, _finalize_op and
        _discard_failed_op: a parent op's pin buffer is registered (and thus
        freed) at this level only when its worker_map entry resolves to a
        single worker. Dict-keyed entries (PP fan-out) register and free each
        replica individually."""
        try:
            resolved_worker = self._worker_entry_for(op)
        except ValueError:
            return False
        return not isinstance(resolved_worker, dict)

    def _finalize_or_discard(self, op: TransferOp, finished_ops: List[TransferOp]) -> None:
        """Route a fully-drained op: discard if any replica of it failed,
        finalize (completion message + successor scheduling) otherwise."""
        if op.op_id in self._failed_parent_op_ids:
            self._discard_failed_op(op)
        else:
            self._finalize_op(op, finished_ops)

    def _emit_xfer_trace(self, op_id: int, metrics) -> None:
        """Print one ``[XFER]`` line for a completed op (transfer tracing).

        Combines the worker-computed timing metrics with the scheduler-side
        e2e (submit -> detect) and current backlog. All trace.* calls no-op
        when ``FLEXKV_TRANSFER_TRACE`` is unset, so this is safe to call
        unconditionally on every finished op.
        """
        e2e_ms = trace.consume_submit_ns(op_id)
        trace.dec_inflight()
        trace.record_xfer(op_id, metrics, e2e_ms)

    def _handle_failed_op(self, op_id: int) -> None:
        """A worker reported a failed transfer for ``op_id``.

        Bookkeeping mirrors the completion path (replica maps, pending
        counts, pin buffer, nvtx) so nothing leaks, but the op never reaches
        finished_ops (its successors must not run) and its graph is marked
        failed: the scheduler stops dispatching the graph's remaining ops,
        and once every already-dispatched op of the graph has drained the
        loop emits a graph-level failure to the task layer.
        """
        graph_id = None
        if op_id in self._child_to_parent_op_id:
            parent_op_id = self._child_to_parent_op_id.pop(op_id)
            child_op = self._child_id_to_child.pop(op_id)
            free_op_from_buffer(child_op, self.pin_buffer)
            if op_id in self.op_id_to_nvtx_range:
                nvtx.end_range(self.op_id_to_nvtx_range.pop(op_id))
            parent_op = self.op_id_to_op.get(parent_op_id)
            if parent_op is not None:
                graph_id = parent_op.graph_id
                parent_op.pending_count -= 1
                self._failed_parent_op_ids.add(parent_op_id)
                if parent_op.pending_count == 0:
                    self._discard_failed_op(parent_op)
        else:
            op = self.op_id_to_op.get(op_id)
            if op is not None:
                graph_id = op.graph_id
                op.pending_count -= 1
                self._failed_parent_op_ids.add(op_id)
                if op.pending_count == 0:
                    self._discard_failed_op(op)
        if graph_id is not None:
            flexkv_logger.error(
                f"[TransferEngine] transfer op {op_id} of graph {graph_id} "
                f"failed; failing the graph and draining its in-flight ops")
            self._failed_graph_ids.add(graph_id)
            self.scheduler.fail_graph(graph_id)

    def _discard_failed_op(self, op: TransferOp) -> None:
        """Release a fully-drained op that must not complete (it failed, or a
        replica of it did): same pin-buffer rule and cleanup as _finalize_op,
        minus the completion message and successor scheduling."""
        if self._op_buffer_registered_here(op):
            free_op_from_buffer(op, self.pin_buffer)
        if op.op_id in self.op_id_to_nvtx_range:
            nvtx.end_range(self.op_id_to_nvtx_range.pop(op.op_id))
        self.op_id_to_op.pop(op.op_id, None)
        self._failed_parent_op_ids.discard(op.op_id)

    def _emit_drained_graph_failures(self) -> None:
        """Report each failed graph to the task layer once its dispatched ops
        have all drained, so the task's rollback never races an in-flight op's
        completion callback."""
        for graph_id in list(self._failed_graph_ids):
            if any(op.graph_id == graph_id for op in self.op_id_to_op.values()):
                continue
            self.completed_queue.put(CompletedOp.failed_graph(graph_id))
            self._failed_graph_ids.discard(graph_id)

    def _finalize_op(self, op: TransferOp, finished_ops: List[TransferOp]) -> None:
        """Finalize a completed op: release pin buffer, notify upper layer, and clean up.

        Called only when op.pending_count reaches 0, i.e., all PP-sibling replica
        workers have completed this op. This ensures atomic eviction semantics.
        """
        # Unified rule: free the parent op buffer here only if the parent itself
        # was registered upstream (single-worker path). For dict-keyed (PP fan-out)
        # entries the parent was never registered; each replica was registered and
        # freed individually in the scheduler's child completion path.
        if self._op_buffer_registered_here(op):
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
            block_results=op.block_results,
        ))
        finished_ops.append(op)
        del self.op_id_to_op[op.op_id]

    @staticmethod
    def _merge_block_results(
        op: TransferOp,
        block_results: Optional[Tuple[bool, ...]],
    ) -> None:
        """Accumulate per-worker outcomes; every participating worker must win."""
        if block_results is None:
            return
        normalized = tuple(bool(result) for result in block_results)
        if len(normalized) != len(op.src_block_ids):
            flexkv_logger.error(
                f"Completion result length mismatch for op {op.op_id}: "
                f"results={len(normalized)}, blocks={len(op.src_block_ids)}")
            normalized = (False,) * len(op.src_block_ids)
        if op.block_results is None:
            op.block_results = normalized
        else:
            op.block_results = tuple(
                old and new for old, new in zip(op.block_results, normalized))

    @staticmethod
    def _match_pp_siblings(
        worker_map: Dict[WorkerKey, WorkerHandle],
        dp_client_id: int,
    ) -> List[WorkerKey]:
        """Return every WorkerKey whose flat DP slice equals ``dp_client_id``.

        After flattening, a single int fully identifies the DP slice —
        PP siblings are the worker_keys that share it across pp_rank.
        """
        return [wk for wk in worker_map if wk.dp_client_id == dp_client_id]

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
                # SWA ids must be carried through PP fan-out replicas, otherwise
                # each PP sibling's worker would only see main-KV ids and the SWA
                # layer-fused branch in cpp would be silently skipped.
                swa_src_block_ids_h2d=op.swa_src_block_ids_h2d.copy(),
                swa_dst_block_ids_h2d=op.swa_dst_block_ids_h2d.copy(),
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

    def _worker_entry_for(
        self, op: TransferOp
    ) -> Union[WorkerHandle, Dict[WorkerKey, WorkerHandle]]:
        """The worker (or PP-sibling dict) that serves this op.

        Looked up by (pool, transfer type), falling back to the FULL_KV pool's
        worker. The fallback is the normal case, not an error path: for
        GPU<->CPU a non-default pool is a *pool on the same worker* -- same
        GPUs, same direction, same regions -- and the worker picks it from
        ``op.pool_id``. A pool only registers a worker of its own for the
        tiers where it addresses genuinely different storage (its own SSD
        files, its own remote namespace).
        """
        if op.pool_id is not PoolId.FULL_KV:
            entry = self._workers.get(op.pool_id, {}).get(op.transfer_type)
            if entry is not None:
                return entry
        main = self._workers.get(PoolId.FULL_KV, {})
        if op.transfer_type not in main:
            kind = "" if op.pool_id is PoolId.FULL_KV else f"{op.pool_id.name} "
            raise ValueError(f"Unsupported {kind}transfer type: {op.transfer_type}")
        return main[op.transfer_type]

    def _assign_op_to_worker(self, op: TransferOp) -> None:
        """Assign operation to appropriate worker.

        One path for main KV and SWA. They used to be two near-identical
        functions (``_assign_swa_op_to_worker`` said so in its own docstring:
        "structurally identical to the main-KV dispatch path"); the only real
        difference was which map to look in, which ``_worker_entry_for``
        answers, and which hash list a replica carries.
        """
        if op.transfer_type == TransferType.VIRTUAL:
            return

        if op.transfer_type == TransferType.LAYERWISE:
            if op.transfer_type not in self._worker_map:
                raise ValueError(f"Unsupported transfer type: {op.transfer_type}")
            self._assign_layerwise_op_to_workers(op)
            return

        worker = self._worker_entry_for(op)
        label = f"{op.pool_id.name}_{op.transfer_type.name}"
        if isinstance(worker, dict):
            # PP fan-out: each stage holds its own slice of the layers, so a
            # single submit to one sibling would silently drop the others'.
            sibling_keys = self._match_pp_siblings(worker, op.dp_client_id)
            if not sibling_keys:
                raise ValueError(
                    f"No {label} worker found matching "
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
                    # pool_id, not is_swa: the replica must address the same
                    # slot-id space as its parent, and a bool could only carry
                    # two of them.
                    pool_id=op.pool_id,
                    mooncake_store_block_hashes=(
                        op.mooncake_store_block_hashes.copy()
                        if op.mooncake_store_block_hashes is not None else None),
                    mooncake_store_swa_block_hashes=(
                        list(op.mooncake_store_swa_block_hashes)
                        if op.mooncake_store_swa_block_hashes is not None else None),
                )
                register_op_to_buffer(replica, self.pin_buffer)
                self._child_id_to_child[replica.op_id] = replica
                self._child_to_parent_op_id[replica.op_id] = op.op_id
                self.op_id_to_nvtx_range[replica.op_id] = nvtx.start_range(
                    f"schedule {label}_REPLICA op_id: {replica.op_id}, "
                    f"graph_id: {replica.graph_id}, worker_key={wk}",
                    color=get_nvtx_range_color(replica.graph_id))
                op.pending_count += 1
                worker[wk].submit_transfer(replica)
                flexkv_logger.debug(
                    f"[TransferEngine] {label} fan-out: "
                    f"parent_op_id={op.op_id}, replica_op_id={replica.op_id}, "
                    f"worker_key={wk}, pending_count={op.pending_count}")
        else:
            # No fan-out; register_op_to_buffer + op_id_to_op were done by the
            # scheduler upstream for this branch.
            self.op_id_to_nvtx_range[op.op_id] = nvtx.start_range(
                f"schedule {label} "
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
        """Drain completed ops, blocking up to ``timeout`` for the first one.

        The old early-return-on-empty ignored ``timeout`` and busy-spun the
        result thread at 100% CPU, starving the scheduler loop under high QPS.
        """
        completed_ops: List[CompletedOp] = []

        try:
            if timeout is None or timeout <= 0:
                # Non-blocking drain.
                if self.completed_queue.empty():
                    return completed_ops
                first_op = self.completed_queue.get_nowait()
            else:
                first_op = self.completed_queue.get(timeout=timeout)
            completed_ops.append(first_op)
        except queue.Empty:
            return completed_ops

        # Drain whatever else is immediately available.
        while not self.completed_queue.empty():
            try:
                completed_ops.append(self.completed_queue.get_nowait())
            except queue.Empty:
                break

        return completed_ops

    def suspend_gpu_mappings(self) -> int:
        """Drain worker pipes and release imported vLLM VMM mappings."""
        if self._gpu_mappings_suspended:
            return 0
        if GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer:
            raise NotImplementedError(
                "GPU hot remap does not support layerwise transfer"
            )
        if self.model_config.layer_groups is not None:
            raise NotImplementedError(
                "GPU hot remap does not support multi-group KV layouts"
            )
        if self._has_swa:
            raise NotImplementedError(
                "GPU hot remap does not support SWA KV pools"
            )
        if self.op_id_to_op or not self.task_queue.empty():
            raise RuntimeError(
                "Cannot suspend GPU mappings with transfers in flight"
            )
        released = 0
        for workers in (self.h2d_workers, self.d2h_workers):
            for worker in workers.values():
                released += int(worker.control("suspend_gpu"))
        self._gpu_mappings_suspended = True
        return released

    def resume_gpu_mappings(
        self, gpu_handle_groups: Dict[WorkerKey, List[StorageHandle]]
    ) -> int:
        """Import fresh post-wake VMM handles into existing workers."""
        if not self._gpu_mappings_suspended:
            raise RuntimeError("GPU mappings are not suspended")
        if set(gpu_handle_groups) != set(self.gpu_handle_groups):
            raise ValueError(
                "GPU worker groups changed across sleep: "
                f"old={set(self.gpu_handle_groups)}, "
                f"new={set(gpu_handle_groups)}"
            )
        imported = 0
        for worker_key, handles in gpu_handle_groups.items():
            # Always the plural shape now: one worker class, one payload shape,
            # for every TP width.
            payload = [handle.get_tensor_handle_list() for handle in handles]
            imported += int(
                self.h2d_workers[worker_key].control("resume_gpu", payload)
            )
            imported += int(
                self.d2h_workers[worker_key].control("resume_gpu", payload)
            )
        self.gpu_handle_groups = gpu_handle_groups
        self._gpu_mappings_suspended = False
        return imported

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

            # Shutdown all workers in parallel so large cudaHostUnregister
            # work overlaps across processes instead of stacking timeouts.
            self._shutdown_worker_handles(self._collect_worker_handles())
        except Exception as e:
            flexkv_logger.error(f"Error during shutdown: {e}")
        finally:
            with contextlib.suppress(Exception):
                while not self.finished_ops_queue.empty():
                    self.finished_ops_queue.get_nowait()

            torch.cuda.empty_cache()
            # Bounded sync: a wedged GPU here would keep the TM process alive
            # past shutdown, forcing the parent-side SIGTERM/SIGKILL. Workers
            # have already unpinned; TM does not itself hold CPU pin refs.
            _te_bounded_cuda_sync(timeout_s=15.0) # hardcode for now
