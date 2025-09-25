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
from queue import Queue
from typing import Dict, List, Optional, Tuple, Union

import contextlib
import nvtx
import torch

from flexkv.common.debug import flexkv_logger
from flexkv.common.storage import StorageHandle
from flexkv.common.transfer import TransferOp, TransferOpGraph, TransferType
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
from flexkv.common.config import CacheConfig, ModelConfig, GLOBAL_CONFIG_FROM_ENV
from flexkv.common.ring_buffer import SharedOpPool


def register_op_to_buffer(op: TransferOp, pin_buffer: SharedOpPool) -> None:
    op.src_slot_id = pin_buffer.allocate_slot(op.src_block_ids)
    op.dst_slot_id = pin_buffer.allocate_slot(op.dst_block_ids)

def free_op_from_buffer(op: TransferOp, pin_buffer: SharedOpPool) -> None:
    if op.src_slot_id != -1:
        pin_buffer.free_slot(op.src_slot_id)
    if op.dst_slot_id != -1:
        pin_buffer.free_slot(op.dst_slot_id)

class TransferEngine:
    def __init__(self,
        gpu_handles: List[StorageHandle],
        model_config: ModelConfig,
        cache_config: CacheConfig,
        cpu_handle: Optional[StorageHandle] = None,
        ssd_handle: Optional[StorageHandle] = None,
        remote_handle: Optional[StorageHandle] = None):
        """
        Initialize transfer engine

        Args:
            gpu_handles: List of GPU handles
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
        self.task_queue: Queue[TransferOpGraph] = Queue()
        self.completed_queue: Queue[Tuple[int, int]] = Queue()
        self.finished_ops_queue = self.mp_ctx.Queue()
        self.op_id_to_op: Dict[int, TransferOp] = {}
        self.gpu_handles = gpu_handles
        self._cpu_handle = cpu_handle
        self._ssd_handle = ssd_handle
        self._remote_handle = remote_handle
        self._cache_config = cache_config
        self._enable_pcfs_sharing = cache_config.index_accel and cache_config.enable_kv_sharing

        self.pin_buffer = SharedOpPool(2048, self.cache_config.num_cpu_blocks)

        self.op_id_to_nvtx_range: Dict[int, str] = {}

        self.dp_size = model_config.dp_size
        self.tp_size = model_config.tp_size

        assert len(gpu_handles) == self.dp_size * self.tp_size
        self._running = False

    def _init_workers(self) -> None:
        if self._running:
            return
        self._worker_map: Dict[TransferType, Union[WorkerHandle, List[WorkerHandle]]] = {}

        assert self._cpu_handle is not None
        if self.tp_size == 1:
            self.gpucpu_workers: List[WorkerHandle] = [
                GPUCPUTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor = self.pin_buffer.get_buffer(),
                    gpu_blocks=self.gpu_handles[i].get_tensor_handle_list(),
                    cpu_blocks=self._cpu_handle.get_tensor(),
                    gpu_kv_layout=self.gpu_handles[i].kv_layout,
                    cpu_kv_layout=self._cpu_handle.kv_layout,
                    dtype=self.gpu_handles[i].dtype,
                    gpu_device_id=i,
                    use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                    use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                    transfer_sms_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_sms_h2d,
                    transfer_sms_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_sms_d2h,
                )
                for i in range(self.dp_size)
            ]
        else:
            self.gpucpu_workers = [
                tpGPUCPUTransferWorker.create_worker(
                    mp_ctx=self.mp_ctx,
                    finished_ops_queue=self.finished_ops_queue,
                    op_buffer_tensor = self.pin_buffer.get_buffer(),
                    gpu_blocks=[self.gpu_handles[j].get_tensor_handle_list() \
                                for j in range(i * self.tp_size, (i + 1) * self.tp_size)],
                    cpu_blocks=self._cpu_handle.get_tensor(),
                    gpu_kv_layouts=[self.gpu_handles[i].kv_layout \
                        for i in range(i * self.tp_size, (i + 1) * self.tp_size)],
                    cpu_kv_layout=self._cpu_handle.kv_layout,
                    dtype=self.gpu_handles[i].dtype,
                    tp_group_size=self.tp_size,
                    dp_group_id=i,
                    use_ce_transfer_h2d=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_h2d,
                    use_ce_transfer_d2h=GLOBAL_CONFIG_FROM_ENV.use_ce_transfer_d2h,
                    transfer_sms_h2d=GLOBAL_CONFIG_FROM_ENV.transfer_sms_h2d,
                    transfer_sms_d2h=GLOBAL_CONFIG_FROM_ENV.transfer_sms_d2h,
                )
                for i in range(self.dp_size)
            ]
        self._worker_map[TransferType.H2D] = self.gpucpu_workers
        self._worker_map[TransferType.D2H] = self.gpucpu_workers

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
                        gpu_blocks=self.gpu_handles[i].get_tensor_handle_list(),
                        ssd_files=self._ssd_handle.get_file_list(),
                        num_blocks_per_file=self._ssd_handle.num_blocks_per_file,
                        gpu_kv_layout=self.gpu_handles[i].kv_layout,
                        ssd_kv_layout=self._ssd_handle.kv_layout,
                        dtype=self._ssd_handle.dtype,
                        gpu_device_id=i,
                    )
                    for i in range(self.dp_size)
                ]
            else:
                self.gds_workers = [
                    tpGDSTransferWorker.create_worker(
                        mp_ctx=self.mp_ctx,
                        finished_ops_queue=self.finished_ops_queue,
                        op_buffer_tensor=self.pin_buffer.get_buffer(),
                        gpu_blocks=[self.gpu_handles[j].get_tensor_handle_list() \
                                    for j in range(i * self.tp_size, (i + 1) * self.tp_size)],
                        ssd_files=self._ssd_handle.get_file_list(),
                        num_blocks_per_file=self._ssd_handle.num_blocks_per_file,
                        gpu_kv_layouts=[self.gpu_handles[j].kv_layout \
                                       for j in range(i * self.tp_size, (i + 1) * self.tp_size)],
                        ssd_kv_layout=self._ssd_handle.kv_layout,
                        dtype=self._ssd_handle.dtype,
                        tp_group_size=self.tp_size,
                        dp_group_id=i,
                    )
                    for i in range(self.dp_size)
                ]
            # GDS workers handle DISK2D/D2DISK operations using the GDS transfer path
            self._worker_map[TransferType.DISK2D] = self.gds_workers
            self._worker_map[TransferType.D2DISK] = self.gds_workers
            
        if self._cpu_handle is not None and self.cache_config.enable_kv_sharing and self.cache_config.enable_p2p_cpu:
            ## NOTE:if we have the cpu handle and enable p2p cpu transfer we need this worker 
            ## (currently we inplement cpu and ssd distributed transfer in one worker)
            if self.cache_config.enable_p2p_ssd and not self.cache_config.enable_p2p_cpu:
                raise ValueError("enable_p2p_ssd requires enable_p2p_cpu to be True")
            flexkv_logger.info(f"[transfer_engine] initializing the PEER2CPUTransferWorker!")
            self.cpu_remote_cpu_worker: WorkerHandle = PEER2CPUTransferWorker.create_worker(
                finished_ops_queue=self.finished_ops_queue,
                op_buffer_tensor = self.pin_buffer.get_buffer(),
                cpu_blocks=self._cpu_handle.get_tensor(),
                cpu_kv_layout=self._cpu_handle.kv_layout,
                ssd_kv_layout = self._ssd_handle.kv_layout,
                # TODO: get remote kv_layout, now we can assume that remote kv layout is same as current node
                remote_kv_layout=self._cpu_handle.kv_layout, 
                dtype=self._cpu_handle.dtype,
                cache_config = self.cache_config,
                ssd_files = self._ssd_handle.get_file_list(),
                num_blocks_per_file = self._ssd_handle.num_blocks_per_file
            )
            # NOTE: now peerH2H and peerSSD2H op use the same worker
            if self.cache_config.enable_p2p_cpu:
                self._worker_map[TransferType.PEERH2H] = self.cpu_remote_cpu_worker
            if self.cache_config.enable_p2p_ssd:
                self._worker_map[TransferType.PEERSSD2H] = self.cpu_remote_cpu_worker

            
        if len(self._worker_map) == 0:
            raise ValueError("No workers initialized, please check the config")
        # Wait for all workers to ready
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
        # Start scheduler thread
        self._running = True
        self._scheduler_thread = threading.Thread(target=self._scheduler_loop)
        self._scheduler_thread.start()

    def start(self) -> None:
        self._init_workers()

    def _scheduler_loop(self) -> None:
        """Main scheduler loop"""
        while self._running:
            # Process new transfer graphs
            new_graphs_num = 0
            while True:
                try:
                    transfer_graph = self.task_queue.get_nowait()
                    self.scheduler.add_transfer_graph(transfer_graph)
                    new_graphs_num += 1
                except queue.Empty:
                    break
            # Collect finished ops
            finished_ops: List[TransferOp] = []
            while True:
                try:
                    op_id = self.finished_ops_queue.get_nowait()
                    op = self.op_id_to_op[op_id]
                    free_op_from_buffer(op, self.pin_buffer)
                    self.completed_queue.put((op.graph_id, op.op_id))
                    finished_ops.append(op)
                    del self.op_id_to_op[op_id]
                except queue.Empty:
                    break
            for op in finished_ops:
                nvtx.end_range(self.op_id_to_nvtx_range[op.op_id])
                self.op_id_to_nvtx_range.pop(op.op_id)
            if finished_ops or new_graphs_num > 0:
                # Schedule next operations
                completed_graph_ids, next_ops = self.scheduler.schedule(finished_ops)
                # Distribute new ops to workers
                for op in next_ops:
                    if op.transfer_type == TransferType.VIRTUAL:
                        self.completed_queue.put((op.graph_id, op.op_id))
                    else:
                        self.op_id_to_op[op.op_id] = op
                        # copy block ids into buffer and update slot id info
                        register_op_to_buffer(op, self.pin_buffer)
                        self._assign_op_to_worker(op)
                # Handle completed graphs
                for graph_id in completed_graph_ids:
                    self.completed_queue.put((graph_id, -1))
            time.sleep(0.001)  # Prevent busy waiting

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

        worker = self._worker_map[op.transfer_type]
        if isinstance(worker, List):
            worker[op.dp_id].submit_transfer(op)
        else:
            worker.submit_transfer(op)

    def submit_transfer_graph(self, transfer_graph: TransferOpGraph) -> None:
        """Submit a transfer graph for execution"""
        self.task_queue.put(transfer_graph)

    def get_completed_graphs_and_ops(self, timeout: Optional[float] = None) -> List[Tuple[int, int]]:
        """Get IDs of all completed transfer graphs at current moment

        Args:
            timeout: Optional timeout for the first graph retrieval

        Returns:
            List of completed graph IDs. Empty list if no graphs are completed.
        """
        completed_graph_ids: List[Tuple[int, int]] = []

        if self.completed_queue.empty():
            return completed_graph_ids

        try:
            first_graph = self.completed_queue.get(timeout=timeout)
            completed_graph_ids.append(first_graph)

            while not self.completed_queue.empty():
                completed_graph_ids.append(self.completed_queue.get_nowait())

        except queue.Empty:
            pass

        return completed_graph_ids

    def shutdown(self) -> None:
        """Shutdown the transfer engine"""
        try:
            if not self._running:
                return
            self._running = False
            self._scheduler_thread.join(timeout=5)

            # shutdown all workers
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
