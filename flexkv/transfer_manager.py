import os
import multiprocessing as mp
import signal
import time
import queue
import selectors
from queue import Queue
from typing import Dict, Optional, List, Tuple, Any
from abc import ABC, abstractmethod
from multiprocessing import Process, Pipe, Event
from sympy.assumptions.assume import true
import torch
import zmq
import nvtx
import tempfile
import threading
import numpy as np
import textwrap
import subprocess
import pickle
import sys

from flexkv.common.transfer import TransferOpGraph, CompletedOp, WorkerKey
from flexkv.common.config import (
    CacheConfig, LayerGroupSpec, ModelConfig,
    recompute_cache_block_counts, GLOBAL_CONFIG_FROM_ENV,
)
from flexkv.common.debug import flexkv_logger
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.transfer import DeviceType
from flexkv.common.storage import KVCacheLayout
from flexkv.storage.storage_engine import StorageEngine
from flexkv.transfer.transfer_engine import TransferEngine
from flexkv.server.utils import get_zmq_socket
from flexkv.server.request import RegistrationKey, RegisterTPClientRequest, Response


class TransferManager:
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_register_port: str):
        self.model_config = model_config
        self.cache_config = cache_config
        self.gpu_register_port = gpu_register_port
        self.instance_num = self.model_config.instance_num
        # Calculate total expected GPUs on this node across all instances
        self.expected_gpus = self.instance_num * self.model_config.gpus_per_node

        self.all_gpu_layouts: Dict[RegistrationKey, KVCacheLayout] = {}
        self.all_gpu_blocks: Dict[RegistrationKey, List[TensorSharedHandle]] = {}
        self.gpu_worker_key_mapping: Dict[RegistrationKey, WorkerKey] = {}
        self.gpu_device_id_mapping: Dict[RegistrationKey, int] = {}

        # Multi-group storage for heterogeneous KV shapes, including DSA/NSA
        # indexer-as-group. None for uniform single-shape registrations.
        self.all_gpu_layouts_per_group: Dict[
            RegistrationKey, Optional[List[KVCacheLayout]]
        ] = {}
        self.all_gpu_blocks_per_group: Dict[
            RegistrationKey, Optional[List[List[TensorSharedHandle]]]
        ] = {}

        # SWA dedicated GPU pool (channel B): independent of the main-KV pool.
        # Logical registration key -> SWA handles / layout. Populated only when
        # the client provides swa_handles (DSv4 sliding-window-attention pool).
        self.all_swa_gpu_blocks: Dict[RegistrationKey, List[TensorSharedHandle]] = {}
        self.all_swa_gpu_layouts: Dict[RegistrationKey, KVCacheLayout] = {}
        self.all_swa_gpu_layouts_per_group: Dict[
            RegistrationKey, Optional[List[KVCacheLayout]]
        ] = {}
        self.all_swa_gpu_blocks_per_group: Dict[
            RegistrationKey, Optional[List[List[TensorSharedHandle]]]
        ] = {}
        self.swa_layer_groups: Optional[List[LayerGroupSpec]] = None

        self.context = zmq.Context(2)
        self.recv_from_client = get_zmq_socket(
            self.context, zmq.SocketType.PULL, gpu_register_port, True)
        self.gpu_control_port = f"{gpu_register_port}_control"
        self.gpu_control_socket = get_zmq_socket(
            self.context, zmq.SocketType.REP, self.gpu_control_port, True
        )
        self._gpu_suspended = False
        self._pending_resume_registrations: Dict[
            RegistrationKey, RegisterTPClientRequest
        ] = {}
        # The REP socket above is bound unconditionally, so *every* deployment
        # mode has to service it -- an unserved REP endpoint turns a
        # suspend/resume call into a silent 120s RCVTIMEO stall on the client.
        # The subprocess mode drives it from its selector loop; the other two
        # modes use the listener thread below.
        self._gpu_control_shutdown = threading.Event()
        self._gpu_control_thread: Optional[threading.Thread] = None

        self.transfer_engine: Optional[TransferEngine] = None
        self.storage_engine: Optional[StorageEngine] = None
        flexkv_logger.info(f"Initialized TransferManager with config successfully, "
                           f"instance_num={self.instance_num}, expected_gpus={self.expected_gpus}")

    def _handle_gpu_blocks_registration(self, req: RegisterTPClientRequest) -> None:
        registration_key = req.registration_key

        if registration_key in self.all_gpu_blocks:
            # A duplicate (dp_client_id, intra_client_id) means the framework
            # adapter handed us the same logical identity for two different
            # workers -- typically because it never passes intra_client_id and
            # every TP rank collapses onto ...0. Registration can then never
            # reach expected_gpus, so _register_gpu_blocks_via_socket spins
            # forever printing "Still waiting for GPU registrations: k/N".
            # Say so explicitly instead of leaving an unexplained hang.
            flexkv_logger.error(
                f"GPU worker {registration_key} has already registered. "
                f"A duplicate registration key from a different worker means "
                f"registration can never complete and init will hang. Check "
                f"that the framework adapter passes a per-worker-unique "
                f"intra_client_id (registered so far: "
                f"{sorted(self.all_gpu_blocks)}).")
        else:
            try:
                self.all_gpu_blocks[registration_key] = req.handles
                self.all_gpu_layouts[registration_key] = req.gpu_layout
                self.gpu_device_id_mapping[registration_key] = req.device_id
                self.gpu_worker_key_mapping[registration_key] = WorkerKey(
                    dp_client_id=req.dp_client_id,
                    pp_rank=req.pp_rank,
                )
                # Store multi-group info (None when uniform single-shape registration).
                # This covers heterogeneous shapes and DSA/NSA indexer-as-group.
                self.all_gpu_layouts_per_group[registration_key] = req.gpu_layouts
                self.all_gpu_blocks_per_group[registration_key] = req.handles_per_group
                # Store SWA GPU data if present.
                if getattr(req, "swa_handles", None) is not None and req.swa_layout is not None:
                    self.all_swa_gpu_blocks[registration_key] = req.swa_handles
                    self.all_swa_gpu_layouts[registration_key] = req.swa_layout
                    self.all_swa_gpu_layouts_per_group[registration_key] = (
                        req.swa_gpu_layouts
                    )
                    self.all_swa_gpu_blocks_per_group[registration_key] = (
                        req.swa_handles_per_group
                    )
                    if req.swa_layer_groups is not None:
                        if self.swa_layer_groups is None:
                            self.swa_layer_groups = req.swa_layer_groups
                        elif self.swa_layer_groups != req.swa_layer_groups:
                            raise ValueError(
                                "SWA layer groups differ across GPU registrations"
                            )
                    flexkv_logger.info(
                        f"GPU worker {registration_key}: registered SWA handles "
                        f"({len(req.swa_handles)} tensors, "
                        f"groups={len(req.swa_layer_groups or [])})"
                    )
                # Propagate layer_groups to model_config (first registration wins).
                # token_size_in_bytes / num_cpu_blocks recompute downstream depends on this.
                if req.layer_groups is not None and self.model_config.layer_groups is None:
                    self.model_config.layer_groups = req.layer_groups
                    flexkv_logger.info(
                        f"Set model_config.layer_groups from GPU worker "
                        f"{registration_key}: "
                        f"{[(g.num_layers, g.num_kv_heads, g.head_size) for g in req.layer_groups]}"
                    )
            except Exception as e:
                flexkv_logger.error(
                    f"Failed to register GPU worker {registration_key}: {e}")

    def handle_gpu_control(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle synchronous sleep/wake mapping lifecycle requests."""
        if self.transfer_engine is None or self.storage_engine is None:
            raise RuntimeError("Transfer engine is not initialized")

        request_type = request.get("type")
        if request_type == "suspend_gpu":
            if not self._gpu_suspended:
                released = self.transfer_engine.suspend_gpu_mappings()
                self._gpu_suspended = True
                self._pending_resume_registrations.clear()
                flexkv_logger.info(
                    f"Suspended FlexKV GPU mappings: released={released}"
                )
            else:
                released = 0
            return {"ok": True, "released_mappings": released}

        if request_type != "resume_gpu":
            raise ValueError(f"Unknown GPU control request: {request_type}")
        if not self._gpu_suspended:
            raise RuntimeError("GPU mappings are not suspended")

        registration = request.get("registration")
        if not isinstance(registration, RegisterTPClientRequest):
            raise TypeError("resume_gpu requires RegisterTPClientRequest")
        if registration.registration_key not in self.all_gpu_blocks:
            raise KeyError(
                f"Unknown registration key {registration.registration_key}"
            )
        if (
            registration.layer_groups is not None
            or registration.handles_per_group is not None
            or registration.swa_handles is not None
        ):
            raise NotImplementedError(
                "GPU hot remap currently supports uniform main KV only"
            )
        self._pending_resume_registrations[
            registration.registration_key
        ] = registration

        ready = (
            len(self._pending_resume_registrations) == self.expected_gpus
        )
        imported = 0
        if ready:
            grouped_gpu_handles = {}
            for registration_key in sorted(
                self._pending_resume_registrations
            ):
                fresh = self._pending_resume_registrations[registration_key]
                self.all_gpu_blocks[registration_key] = fresh.handles
                self.all_gpu_layouts[registration_key] = fresh.gpu_layout
                handle = self.storage_engine.get_storage_handle(
                    DeviceType.GPU, fresh.device_id
                )
                handle.data = fresh.handles
                handle.kv_layout = fresh.gpu_layout
                worker_key = self.gpu_worker_key_mapping[registration_key]
                grouped_gpu_handles.setdefault(worker_key, []).append(handle)
            imported = self.transfer_engine.resume_gpu_mappings(
                grouped_gpu_handles
            )
            self._pending_resume_registrations.clear()
            self._gpu_suspended = False
            flexkv_logger.info(
                f"Resumed FlexKV GPU mappings: imported={imported}"
            )

        return {
            "ok": True,
            "ready": ready,
            "registered": (
                self.expected_gpus if ready
                else len(self._pending_resume_registrations)
            ),
            "imported_mappings": imported,
        }

    def drain_gpu_control_requests(self) -> int:
        """Drain all requests after a ZeroMQ FD edge notification."""
        processed = 0
        while True:
            try:
                request = self.gpu_control_socket.recv_pyobj(zmq.NOBLOCK)
            except zmq.Again:
                break
            try:
                response = self.handle_gpu_control(request)
            except Exception as e:
                flexkv_logger.exception("GPU mapping lifecycle request failed")
                response = {"ok": False, "error": str(e)}
            self.gpu_control_socket.send_pyobj(response)
            processed += 1
        return processed

    def start_gpu_control_listener(self) -> None:
        """Serve the GPU control REP socket from a dedicated thread.

        For the subprocess mode the selector loop in
        ``TransferManagerInterProcessHandle._process_worker`` already drains
        this socket, so it must not call this.  Thread mode and remote mode
        have no such loop: without this thread the bound REP endpoint accepts
        connections and then never answers, so a vLLM sleep/wake call blocks
        for the client's full 120s RCVTIMEO and then fails.
        """
        if self._gpu_control_thread is not None:
            return
        self._gpu_control_shutdown.clear()
        self._gpu_control_thread = threading.Thread(
            target=self._gpu_control_listener,
            name="flexkv-gpu-control",
            daemon=True,
        )
        self._gpu_control_thread.start()

    def _gpu_control_listener(self) -> None:
        poller = zmq.Poller()
        poller.register(self.gpu_control_socket, zmq.POLLIN)
        try:
            while not self._gpu_control_shutdown.is_set():
                try:
                    # Milliseconds -- zmq.Poller, unlike socket RCVTIMEO, does
                    # not take seconds.
                    if not poller.poll(timeout=100):
                        continue
                    self.drain_gpu_control_requests()
                except zmq.ZMQError:
                    # Context terminated during shutdown.
                    break
                except Exception:
                    if not self._gpu_control_shutdown.is_set():
                        flexkv_logger.exception(
                            "GPU control listener failed; retrying"
                        )
                        time.sleep(0.01)
        finally:
            try:
                poller.unregister(self.gpu_control_socket)
            except Exception:
                pass

    def stop_gpu_control_listener(self) -> None:
        self._gpu_control_shutdown.set()
        thread = self._gpu_control_thread
        self._gpu_control_thread = None
        if thread is not None and thread.is_alive():
            thread.join(timeout=1.0)

    def _register_gpu_blocks_via_socket(self) -> None:
        try:
            flexkv_logger.info(f"GPU tensor registration server started on port {self.gpu_register_port}, "
                               f"expected {self.expected_gpus} GPUs to register "
                               f"(instance_num={self.instance_num}, gpus_per_node={self.model_config.gpus_per_node}, "
                               f"total_gpus={self.model_config.total_gpus}, nnodes={self.model_config.nnodes})")
            last_log_time = time.time()
            while len(self.all_gpu_blocks) < self.expected_gpus:
                try:
                    # Recv from: flexkv.server.client.KVTPClient.register_to_server
                    req = self.recv_from_client.recv_pyobj(zmq.NOBLOCK)
                except zmq.Again:
                    # Periodically log waiting status for debugging
                    now = time.time()
                    if now - last_log_time >= 5.0:
                        registered_keys = sorted(self.all_gpu_blocks.keys())
                        flexkv_logger.info(
                            f"Still waiting for GPU registrations: "
                            f"{len(self.all_gpu_blocks)}/{self.expected_gpus} registered "
                            f"(registered_keys={registered_keys}, "
                            f"port={self.gpu_register_port})")
                        last_log_time = now
                    time.sleep(0.001)
                    continue

                if isinstance(req, RegisterTPClientRequest):
                    flexkv_logger.info(f"Received GPU blocks registration request: {type(req)}, "
                                       f"registration_key={req.registration_key}, "
                                       f"device_id={req.device_id}, "
                                       f"dp_client_id={req.dp_client_id}, pp_rank={req.pp_rank}")
                    self._handle_gpu_blocks_registration(req)
                    flexkv_logger.info(f"GPU worker {req.registration_key} registered successfully, "
                                       f"waiting for {self.expected_gpus - len(self.all_gpu_blocks)} GPUs to register")
                else:
                    flexkv_logger.error(f"Unrecognized RequestType in SchedulerServer: {type(req)}")

            flexkv_logger.info(f"All {self.expected_gpus} GPUs registered successfully")

        except Exception as e:
            flexkv_logger.error(f"Error in GPU registration server: {e}")
            raise
        finally:
            pass
            # TODO: fix the socket close issue
            # self.recv_from_client.close()
            # self.context.term()

    def initialize_transfer_engine(self) -> None:
        flexkv_logger.info("Initializing TransferEngine...")
        self._register_gpu_blocks_via_socket()

        assert len(self.all_gpu_layouts) == self.expected_gpus, \
            f"Expected {self.expected_gpus} GPU layouts, got {len(self.all_gpu_layouts)}"
        assert len(self.all_gpu_blocks) == self.expected_gpus, \
            f"Expected {self.expected_gpus} GPU blocks, got {len(self.all_gpu_blocks)}"
        num_layers_per_pp_stage = next(iter(self.all_gpu_layouts.values())).num_layer

        # Recompute block counts once layer_groups are known (heterogeneous /
        # multi-pool models).  Must match CacheEngine mempool sizing in the
        # main process — sglang DSv4 applies the same recompute before
        # KVManager; this path covers late discovery at GPU registration.
        recompute_cache_block_counts(self.model_config, self.cache_config)

        self.storage_engine = StorageEngine(
            self.model_config,
            self.cache_config,
            num_layers_per_pp_stage,
            swa_layer_groups=self.swa_layer_groups,
        )

        # Logical registration identity is separate from the CUDA device ID.
        for registration_key, gpu_blocks_wrapper in self.all_gpu_blocks.items():
            device_id = self.gpu_device_id_mapping[registration_key]
            self.storage_engine.register_gpu_blocks(
                gpu_blocks_wrapper,
                self.all_gpu_layouts[registration_key],
                device_id,
                dtype=self.model_config.dtype,
            )

        # Register SWA dedicated GPU pool.
        for registration_key, swa_blocks in self.all_swa_gpu_blocks.items():
            device_id = self.gpu_device_id_mapping[registration_key]
            self.storage_engine.register_swa_gpu_blocks(
                swa_blocks,
                self.all_swa_gpu_layouts[registration_key],
                device_id,
                dtype=torch.uint8,
            )
            flexkv_logger.info(
                f"StorageEngine registered SWA GPU pool for device {device_id}"
            )

        # Group GPU handles by WorkerKey
        grouped_gpu_handles: Dict[WorkerKey, List] = {}
        # Per-group data, also keyed by WorkerKey, for multi-group support
        # (heterogeneous KV / indexer-as-group)
        grouped_gpu_blocks_per_group: Optional[Dict[WorkerKey, List]] = None
        grouped_gpu_layouts_per_group: Optional[Dict[WorkerKey, List]] = None
        has_multi_group = self.model_config.layer_groups is not None
        if has_multi_group:
            grouped_gpu_blocks_per_group = {}
            grouped_gpu_layouts_per_group = {}

        for registration_key in sorted(self.all_gpu_blocks.keys()):
            worker_key = self.gpu_worker_key_mapping[registration_key]
            device_id = self.gpu_device_id_mapping[registration_key]
            if worker_key not in grouped_gpu_handles:
                grouped_gpu_handles[worker_key] = []
            grouped_gpu_handles[worker_key].append(
                self.storage_engine.get_storage_handle(DeviceType.GPU, device_id))

            if has_multi_group:
                if worker_key not in grouped_gpu_blocks_per_group:
                    grouped_gpu_blocks_per_group[worker_key] = []
                    grouped_gpu_layouts_per_group[worker_key] = []
                grouped_gpu_blocks_per_group[worker_key].append(
                    self.all_gpu_blocks_per_group[registration_key])
                grouped_gpu_layouts_per_group[worker_key].append(
                    self.all_gpu_layouts_per_group[registration_key])

        cpu_handle = self.storage_engine.get_storage_handle(DeviceType.CPU) \
            if self.cache_config.enable_cpu else None
        ssd_handle = self.storage_engine.get_storage_handle(DeviceType.SSD) \
            if self.cache_config.enable_ssd else None
        use_mooncake_store = self.cache_config.use_mooncake_store_backend
        remote_handle = (
            self.storage_engine.get_storage_handle(DeviceType.REMOTE) \
            if self.cache_config.enable_remote and not use_mooncake_store \
            else None
        )
        # Group SWA GPU handles by WorkerKey, mirroring the main-KV grouping,
        # so the dedicated SWA worker map can be built per TP group.
        swa_gpu_handles: Optional[Dict[WorkerKey, List]] = None
        swa_grouped_gpu_blocks_per_group: Optional[Dict[WorkerKey, List]] = None
        swa_grouped_gpu_layouts_per_group: Optional[Dict[WorkerKey, List]] = None
        if self.swa_layer_groups is not None:
            swa_grouped_gpu_blocks_per_group = {}
            swa_grouped_gpu_layouts_per_group = {}
        if self.storage_engine.has_storage_handle(DeviceType.CPU, is_swa=True):
            swa_gpu_handles = {}
            for registration_key in sorted(self.all_swa_gpu_blocks.keys()):
                device_id = self.gpu_device_id_mapping[registration_key]
                if self.storage_engine.get_storage_handle(DeviceType.GPU, device_id, is_swa=True):
                    worker_key = self.gpu_worker_key_mapping[registration_key]
                    if worker_key not in swa_gpu_handles:
                        swa_gpu_handles[worker_key] = []
                    swa_gpu_handles[worker_key].append(
                        self.storage_engine.get_storage_handle(DeviceType.GPU, device_id, is_swa=True))
                    if self.swa_layer_groups is not None:
                        swa_grouped_gpu_blocks_per_group.setdefault(
                            worker_key, []
                        ).append(self.all_swa_gpu_blocks_per_group[registration_key])
                        swa_grouped_gpu_layouts_per_group.setdefault(
                            worker_key, []
                        ).append(self.all_swa_gpu_layouts_per_group[registration_key])

        swa_cpu_handle =(
         self.storage_engine.get_storage_handle(DeviceType.CPU, is_swa=True)
         if self.storage_engine.has_storage_handle(DeviceType.CPU, is_swa=True)
         else None
         )
        swa_ssd_handle = (
         self.storage_engine.get_storage_handle(DeviceType.SSD, is_swa=True)
         if self.storage_engine.has_storage_handle(DeviceType.SSD, is_swa=True)
         else None
         )
        swa_remote_handle = (
         self.storage_engine.get_storage_handle(DeviceType.REMOTE, is_swa=True)
         if self.storage_engine.has_storage_handle(DeviceType.REMOTE, is_swa=True)
         else None
         )

        self.transfer_engine = TransferEngine(
            gpu_handles=grouped_gpu_handles,
            model_config=self.model_config,
            cache_config=self.cache_config,
            cpu_handle=cpu_handle,
            ssd_handle=ssd_handle,
            remote_handle=remote_handle,
            gpu_blocks_per_group=grouped_gpu_blocks_per_group,
            gpu_layouts_per_group=grouped_gpu_layouts_per_group,
            swa_gpu_handles=swa_gpu_handles,
            swa_cpu_handle=swa_cpu_handle,
            swa_ssd_handle=swa_ssd_handle,
            swa_remote_handle=swa_remote_handle,
            swa_layer_groups=self.swa_layer_groups,
            swa_gpu_blocks_per_group=swa_grouped_gpu_blocks_per_group,
            swa_gpu_layouts_per_group=swa_grouped_gpu_layouts_per_group,
        )
        flexkv_logger.info(
            f"Initialized TransferEngine successfully, "
            f"grouped_gpu_handles keys={list(grouped_gpu_handles.keys())}, "
            f"num_gpu_groups={len(grouped_gpu_handles)}"
        )

    def submit(self, transfer_graph: TransferOpGraph) -> None:
        self.transfer_engine.submit_transfer_graph(transfer_graph)

    def submit_batch(self, transfer_graphs: List[TransferOpGraph]) -> None:
        self.transfer_engine.submit_transfer_graph(transfer_graphs)

    def wait(self, timeout: Optional[float] = None) -> List[CompletedOp]:
        return self.transfer_engine.get_completed_graphs_and_ops(timeout)

    def start(self) -> None:
        self.transfer_engine.start()

    def shutdown(self) -> None:
        if hasattr(self, 'transfer_engine'):
            self.transfer_engine.shutdown()

class TransferManagerOnRemote(TransferManager):
    """
    TransferManager for remote mode, used for multi-node tensor parallelism.
    """
    def __init__(
        self,
        master_host: str,
        master_ports: Tuple[str, str, str],
    ):
        self.master_host = master_host
        self.master_ports = master_ports
        flexkv_logger.info(
            f"[TransferManagerOnRemote] master endpoint: "
            f"host={master_host!r}, ports={master_ports}"
        )

        self.context = zmq.Context()
        self.command_socket = self.context.socket(zmq.PULL)
        self.command_socket.setsockopt(zmq.LINGER, 0)
        self.result_socket = self.context.socket(zmq.PUSH)
        self.result_socket.setsockopt(zmq.LINGER, 0)
        self.query_socket = self.context.socket(zmq.REP)
        self.query_socket.setsockopt(zmq.LINGER, 0)

        self._shutdown_flag = False
        self._is_ready = False

        # key: graph_id, value: task_end_op_id
        self._active_graphs: Dict[int, int] = {}
        self._active_graphs_lock = threading.Lock()

        self._pending_graphs: Dict[int, Tuple[TransferOpGraph, int]] = {}
        self._pending_slot_mappings: Dict[int, np.ndarray] = {}
        self._pending_lock = threading.Lock()

        self._worker_thread: threading.Thread | None = None

        self._connect_to_master_transfer_manager()

        self._initialize_with_config()
        flexkv_logger.info("Initialized TransferManagerOnRemote with config successfully")

    def _connect_to_master_transfer_manager(self) -> None:
        try:
            command_addr = f"tcp://{self.master_host}:{self.master_ports[0]}"
            self.command_socket.connect(command_addr)
            flexkv_logger.debug(f"Connected to master command port at {command_addr}")

            result_addr = f"tcp://{self.master_host}:{self.master_ports[1]}"
            self.result_socket.connect(result_addr)
            flexkv_logger.debug(f"Connected to master result port at {result_addr}")

            query_addr = f"tcp://{self.master_host}:{self.master_ports[2]}"
            self.query_socket.connect(query_addr)
            flexkv_logger.debug(f"Connected to master query port at {query_addr}")

            flexkv_logger.debug("Successfully connected to master transfer manager")

        except Exception as e:
            flexkv_logger.error(f"Failed to connect to master transfer manager: {e}")
            raise

    def _initialize_with_config(self) -> None:
        flexkv_logger.info(f"Waiting for config from master at {self.master_host}:{self.master_ports[0]}")
        config_msg = self.command_socket.recv_pyobj()
        if isinstance(config_msg, dict) and config_msg.get('type') == 'config':
            self.model_config = config_msg.get('model_config')
            self.cache_config = config_msg.get('cache_config')
            self.gpu_register_port = config_msg.get('gpu_register_port')
            flexkv_logger.info(f"Received config from master, {self.model_config = }, \
                {self.cache_config = }, {self.gpu_register_port = }.")
        else:
            raise RuntimeError(f"Expected config message, got: {config_msg}")
        flexkv_logger.info("Received config from master successfully")
        super().__init__(self.model_config, self.cache_config, self.gpu_register_port)

    def _polling_worker(self) -> None:
        flexkv_logger.info("Polling worker thread started")

        poller = zmq.Poller()
        poller.register(self.command_socket, zmq.POLLIN)
        poller.register(self.query_socket, zmq.POLLIN)
        # Inherited from TransferManager.__init__, which binds it
        # unconditionally. Nothing else in this process serves it.
        poller.register(self.gpu_control_socket, zmq.POLLIN)

        while not self._shutdown_flag:
            try:
                socks = dict(poller.poll(timeout=0.001))

                if self.command_socket in socks:
                    try:
                        message = self.command_socket.recv_pyobj(zmq.NOBLOCK)

                        if isinstance(message, dict):
                            msg_type = message.get('type')
                            if msg_type == 'submit':
                                graph = message.get('graph')
                                task_end_op_id = message.get('task_end_op_id', -1)

                                if graph is not None:
                                    self._handle_submit(graph, task_end_op_id)
                                else:
                                    flexkv_logger.warning("Received submit message without graph")
                            elif msg_type == 'submit_batch':
                                graphs = message.get('graphs', [])
                                for graph in graphs:
                                    graph_id = graph.graph_id
                                    with self._active_graphs_lock:
                                        self._active_graphs[graph_id] = -1
                                    self.submit(graph)
                            elif msg_type == 'set_slot_mapping':
                                task_id = message.get('task_id')
                                slot_mapping = message.get('slot_mapping')
                                self._handle_set_slot_mapping(task_id, slot_mapping)
                            else:
                                flexkv_logger.warning(f"Unexpected command message: {message}")
                        else:
                            flexkv_logger.warning(f"Unexpected command message type: {type(message)}")
                    except zmq.Again:
                        pass

                if self.gpu_control_socket in socks:
                    self.drain_gpu_control_requests()

                if self.query_socket in socks:
                    try:
                        query_msg = self.query_socket.recv_pyobj(zmq.NOBLOCK)

                        if isinstance(query_msg, dict) and query_msg.get('type') == 'query_ready':
                            response = {'ready': self._is_ready}
                            self.query_socket.send_pyobj(response)
                        else:
                            response = {'error': 'unknown query type'}
                            self.query_socket.send_pyobj(response)
                            flexkv_logger.warning(f"Unknown query message: {query_msg}")
                    except zmq.Again:
                        pass

                try:
                    completed = self.wait(timeout=0.001)

                    if completed:
                        with self._active_graphs_lock:
                            for completed_op in completed:
                                if completed_op.graph_id in self._active_graphs:
                                    task_end_op_id = self._active_graphs[completed_op.graph_id]

                                    if task_end_op_id != -1 and completed_op.op_id == task_end_op_id:
                                        end_op = CompletedOp(graph_id=completed_op.graph_id, op_id=task_end_op_id)
                                        self.result_socket.send_pyobj(end_op)
                                    if (completed_op.is_graph_completed()
                                            or completed_op.is_graph_failed()):
                                        self.result_socket.send_pyobj(completed_op)
                                        del self._active_graphs[completed_op.graph_id]

                except queue.Empty:
                    pass

            except Exception as e:
                if not self._shutdown_flag:
                    flexkv_logger.error(f"Error in polling worker: {e}")
                    time.sleep(0.01)

        poller.unregister(self.command_socket)
        poller.unregister(self.query_socket)
        poller.unregister(self.gpu_control_socket)

    def _handle_set_slot_mapping(self, task_id: int, slot_mapping: np.ndarray) -> None:
        """Handle set_slot_mapping message from FlexKVConnector.

        When the graph (with cleared GPU blocks) arrived earlier, we can immediately
        set_gpu_blocks and submit.  Otherwise, store the slot_mapping and wait
        for the graph to arrive later.
        """
        graph = None
        task_end_op_id = -1
        with self._pending_lock:
            if task_id in self._pending_graphs:
                # Graph already arrived, set GPU blocks and prepare for submit
                graph, task_end_op_id = self._pending_graphs.pop(task_id)
                graph.set_gpu_blocks(slot_mapping)
                flexkv_logger.debug(
                    f"[TransferManagerOnRemote] set_slot_mapping: "
                    f"graph for task_id={task_id} submitted (graph arrived first)"
                )
            else:
                # Graph not yet arrived, store slot_mapping for later matching
                self._pending_slot_mappings[task_id] = slot_mapping
                flexkv_logger.debug(
                    f"[TransferManagerOnRemote] set_slot_mapping: "
                    f"slot_mapping stored for task_id={task_id}, waiting for graph"
                )
                return

        # Submit graph to transfer engine
        with self._active_graphs_lock:
            self._active_graphs[graph.graph_id] = task_end_op_id
        self.submit(graph)

    def _handle_submit(self, graph: TransferOpGraph, task_end_op_id: int = -1) -> None:
        """Handle submit message with pending matching support.

        If slot_mapping already arrived, set_gpu_blocks and submit immediately.
        Otherwise, store graph in pending_graphs for later matching.
        """
        task_id = graph.graph_id  # Use graph_id as task_id for matching
        with self._pending_lock:
            if task_id in self._pending_slot_mappings:
                # slot_mapping already arrived, set GPU blocks and submit
                slot_mapping = self._pending_slot_mappings.pop(task_id)
                graph.set_gpu_blocks(slot_mapping)
                flexkv_logger.debug(
                    f"[TransferManagerOnRemote] submit: "
                    f"graph for task_id={task_id} submitted (slot_mapping arrived first)"
                )
            else:
                # slot_mapping not yet arrived, store graph and task_end_op_id for later matching
                self._pending_graphs[task_id] = (graph, task_end_op_id)
                flexkv_logger.debug(
                    f"[TransferManagerOnRemote] submit: "
                    f"graph stored for task_id={task_id}, waiting for slot_mapping"
                )
                return  # Don't submit yet, wait for slot_mapping

        # Submit graph to transfer engine
        with self._active_graphs_lock:
            self._active_graphs[graph.graph_id] = task_end_op_id
        self.submit(graph)

    def start(self) -> None:
        self.initialize_transfer_engine()
        super().start()

        self._is_ready = True

        self._worker_thread = threading.Thread(
            target=self._polling_worker, daemon=True
        )
        self._worker_thread.start()

        flexkv_logger.info("TransferManagerOnRemote started successfully")

    def shutdown(self) -> None:
        flexkv_logger.info("Shutting down TransferManagerOnRemote")

        self._shutdown_flag = True
        self._is_ready = False

        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5.0)

        super().shutdown()

        try:
            self.command_socket.close()
            self.result_socket.close()
            self.query_socket.close()
            self.context.term()
        except Exception as e:
            flexkv_logger.error(f"Error closing sockets: {e}")

        flexkv_logger.info("TransferManagerOnRemote shutdown complete")

    def __del__(self) -> None:
        if not self._shutdown_flag:
            self.shutdown()

    @classmethod
    def create_process(cls, **kwargs: Any) -> Process:
        # Serialize the class and kwargs
        cls_data = pickle.dumps(cls)
        kwargs_data = pickle.dumps(kwargs)

        # Create temporary files for serialized data
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.cls') as f:
            f.write(cls_data)
            cls_file = f.name

        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.kwargs') as f:
            f.write(kwargs_data)
            kwargs_file = f.name

        # Prepare environment - remove MPI-related variables to avoid conflicts
        env = os.environ.copy()
        # CRITICAL: Remove CUDA_VISIBLE_DEVICES to allow access to all GPUs
        # TransferManager needs to access all physical GPUs for IPC
        if 'CUDA_VISIBLE_DEVICES' in env:
            flexkv_logger.info(f"Removing CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']} "
                               "for TransferManager subprocess")
            env.pop('CUDA_VISIBLE_DEVICES', None)

        # Create the subprocess script
        transfer_manager_script = textwrap.dedent(f'''
            import os
            import sys
            import pickle
            import tempfile
            from flexkv.common.debug import flexkv_logger

            # Immediately disable MPI to avoid conflicts
            os.environ['MPI4PY_RC_INITIALIZE'] = 'false'

            try:
                # Load the class and kwargs
                with open("{cls_file}", "rb") as f:
                    cls = pickle.load(f)

                with open("{kwargs_file}", "rb") as f:
                    kwargs = pickle.load(f)

                # Create and start TransferManagerOnRemote instance
                flexkv_logger.info(f"Creating TransferManagerOnRemote instance...")
                instance = cls(**kwargs)
                flexkv_logger.info(f"Starting TransferManagerOnRemote instance...")
                instance.start()
                flexkv_logger.info(f"TransferManager instance started successfully")

                # Keep running until worker thread exits
                if hasattr(instance, '_worker_thread') and instance._worker_thread is not None:
                    instance._worker_thread.join()

            except Exception as e:
                print(f"Error in TransferManager subprocess: {{e}}", file=sys.stderr)
                sys.exit(1)
            finally:
                # Clean up temporary files
                try:
                    os.unlink("{cls_file}")
                    os.unlink("{kwargs_file}")
                except Exception:
                    pass
        ''').strip()

        # Start the subprocess
        process = subprocess.Popen([
            sys.executable, '-c', transfer_manager_script
        ], env=env, stdout=None, stderr=None, text=True)  # None = inherit parent's stdout/stderr
        flexkv_logger.info(f"TransferManager subprocess started, PID: {process.pid}")

        # Clean up temporary files after subprocess completes
        def cleanup_files():
            # Wait for subprocess to complete before cleaning up files
            process.wait()
            try:
                os.unlink(cls_file)
                os.unlink(kwargs_file)
            except Exception:
                pass

        cleanup_thread = threading.Thread(target=cleanup_files, daemon=True)
        cleanup_thread.start()

        # Return a wrapper that mimics multiprocessing.Process interface
        class SubprocessWrapper:
            def __init__(self, popen_process):
                self._popen = popen_process
                self.pid = popen_process.pid

            def is_alive(self):
                return self._popen.poll() is None

            def terminate(self):
                self._popen.terminate()

            def join(self, timeout=None):
                return self._popen.wait(timeout)

            def close(self):
                # Close the subprocess pipes
                if self._popen.stdout:
                    self._popen.stdout.close()
                if self._popen.stderr:
                    self._popen.stderr.close()
                if self._popen.stdin:
                    self._popen.stdin.close()

        return SubprocessWrapper(process)

class TransferManagerHandleBase(ABC):
    @abstractmethod
    def start(self) -> None:
        pass

    @abstractmethod
    def is_ready(self) -> bool:
        pass

    @abstractmethod
    def submit(self, transfer_graph: TransferOpGraph, task_end_op_id: int = -1) -> None:
        pass

    @abstractmethod
    def submit_batch(self, transfer_graphs: List[TransferOpGraph]) -> None:
        pass

    @abstractmethod
    def wait(self, timeout: Optional[float] = None) -> List[CompletedOp]:
        pass

    @abstractmethod
    def shutdown(self) -> None:
        pass


class TransferManagerIntraProcessHandle(TransferManagerHandleBase):
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_register_port: str):
        self.transfer_manager = TransferManager(model_config, cache_config, gpu_register_port)
        self._is_ready = False

    def start(self) -> None:
        self.transfer_manager.initialize_transfer_engine()
        self.transfer_manager.start()
        # No selector loop here (that is the subprocess mode), so the bound GPU
        # control REP socket needs its own listener.
        self.transfer_manager.start_gpu_control_listener()
        self._is_ready = True

    def is_ready(self) -> bool:
        return self._is_ready

    def submit(self, transfer_graph: TransferOpGraph, task_end_op_id: int = -1) -> None:
        self.transfer_manager.submit(transfer_graph)

    def submit_batch(self, transfer_graphs: List[TransferOpGraph]) -> None:
        self.transfer_manager.submit_batch(transfer_graphs)

    def wait(self, timeout: Optional[float] = None) -> List[CompletedOp]:
        return self.transfer_manager.wait(timeout)

    def shutdown(self) -> None:
        self.transfer_manager.stop_gpu_control_listener()
        self.transfer_manager.shutdown()


class TransferManagerInterProcessHandle(TransferManagerHandleBase):
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_register_port: str):
        self.mp_ctx = mp.get_context('spawn')

        self.model_config = model_config
        self.cache_config = cache_config
        self.gpu_register_port = gpu_register_port

        self.command_parent_conn, self.command_child_conn = self.mp_ctx.Pipe()
        self.result_parent_conn, self.result_child_conn = self.mp_ctx.Pipe()

        self.process: Optional[Process] = None
        self.start_event = self.mp_ctx.Event()
        self.ready_event = self.mp_ctx.Event()

        self._completed_results: List[CompletedOp] = []

    def _start_process(self) -> None:
        if self.process is not None and self.process.is_alive():
            return

        flexkv_logger.debug(
            f"Spawning TransferManager subprocess: "
            f"tp_size={self.model_config.tp_size}, dp_size={self.model_config.dp_size}, "
            f"gpu_register_port={self.gpu_register_port}")
        self.process = self.mp_ctx.Process(
            target=self._process_worker,
            args=(self.model_config,
                  self.cache_config,
                  self.command_child_conn,
                  self.result_child_conn,
                  self.gpu_register_port,
                  self.ready_event,
                  self.start_event),
            daemon=False
        )
        self.process.start()
        flexkv_logger.debug(f"TransferManager subprocess spawned, pid={self.process.pid}")

    def _process_worker(self,
                        model_config: ModelConfig,
                        cache_config: CacheConfig,
                        command_conn,
                        result_conn,
                        gpu_register_port: str,
                        ready_event,
                        start_event) -> None:
        # Automatically reap child processes (daemon transfer workers) to
        # prevent zombie accumulation.  Use a handler that calls waitpid()
        # with WNOHANG so that multiprocessing.Process.join() still works
        # correctly (SIG_IGN would cause join() to raise ChildProcessError).
        def _reap_children(signum, frame):
            while True:
                try:
                    pid, _ = os.waitpid(-1, os.WNOHANG)
                    if pid == 0:
                        break
                except ChildProcessError:
                    break
        signal.signal(signal.SIGCHLD, _reap_children)

        # Ignore Ctrl+C (SIGINT): process-group SIGINT would race with parent
        # kill_process_tree(SIGKILL). Only SIGTERM / {'type':'shutdown'} should
        # trigger paired worker unregister. SIGKILL cannot be handled.
        def _on_sigterm(signum, frame):
            flexkv_logger.warning(
                f"TransferManager process received signal {signum}; exiting for graceful cleanup"
            )
            raise SystemExit(0)

        try:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
            signal.signal(signal.SIGTERM, _on_sigterm)
        except Exception as e:
            flexkv_logger.warning(
                f"Failed to install TransferManager shutdown signal handlers: {e}"
            )

        try:
            flexkv_logger.debug(f"_process_worker started, pid={os.getpid()}, "
                               f"gpu_register_port={gpu_register_port}")
            start_event.set()
            os.environ['MPI4PY_RC_INITIALIZE'] = 'false'
            transfer_manager = TransferManager(model_config, cache_config, gpu_register_port)
            transfer_manager.initialize_transfer_engine()
            transfer_manager.start()
            flexkv_logger.debug("TransferEngine started successfully, setting ready_event")
            ready_event.set()

            # Setup selector for event-driven processing (complete zero polling!)
            sel = selectors.DefaultSelector()
            sel.register(command_conn.fileno(), selectors.EVENT_READ, data="command")
            # Also monitor completed_queue for finished ops (now it's mp.Queue with _reader)
            sel.register(transfer_manager.transfer_engine.completed_queue._reader,
                        selectors.EVENT_READ, data="finished_ops")
            sel.register(transfer_manager.gpu_control_socket,
                         selectors.EVENT_READ, data="gpu_control")

            flexkv_logger.info(
                "TransferManager daemon process started with selector-based "
                "event monitoring (command + finished_ops)"
            )

            should_exit = False
            while not should_exit:
                try:
                    # Event-driven: wait for command OR finished_ops.
                    # Graceful exit via {'type': 'shutdown'} command (or SIGTERM→SystemExit).
                    events = sel.select(timeout=None)

                    # Process all events
                    has_finished_ops = False

                    for key, mask in events:
                        if key.data == "command":
                            # New command available
                            inner_range = nvtx.start_range(message="TransferManagerInter.process_worker.req", color="red")
                            try:
                                request = command_conn.recv()
                            except (EOFError, BrokenPipeError, ConnectionResetError) as e:
                                # Parent (scheduler) died without sending shutdown.
                                # Break out and let finally run transfer_manager.shutdown()
                                # so workers get paired cudaHostUnregister.
                                flexkv_logger.warning(
                                    f"TransferManager command pipe closed ({e!r}); "
                                    "parent likely crashed. Exiting for graceful cleanup."
                                )
                                should_exit = True
                                nvtx.end_range(inner_range)
                                break
                            request_type = request.get('type')
                            if request_type == 'submit':
                                transfer_manager.submit(request['transfer_graph'])
                            elif request_type == 'submit_batch':
                                transfer_manager.submit_batch(request['transfer_graphs'])
                            elif request_type == 'shutdown':
                                flexkv_logger.info(
                                    "TransferManager received shutdown command; "
                                    "leaving event loop for graceful cleanup"
                                )
                                should_exit = True
                            else:
                                flexkv_logger.error(f"Unrecognized request type: {request_type}")
                            nvtx.end_range(inner_range)

                        elif key.data == "finished_ops":
                            # Selector reports finished_ops queue has data
                            has_finished_ops = True

                        elif key.data == "gpu_control":
                            # ZeroMQ exposes an edge-triggered FD, so one event
                            # must consume every request that is already queued.
                            transfer_manager.drain_gpu_control_requests()

                    # Only collect finished_ops if selector reported data available
                    if has_finished_ops and not should_exit:
                        inner_range = nvtx.start_range(message="TransferManagerInter.process_worker.results", color="red")
                        try:
                            # Directly get from completed_queue without timeout to avoid poll
                            finished_ops = []
                            completed_queue = transfer_manager.transfer_engine.completed_queue
                            while not completed_queue.empty():
                                try:
                                    finished_ops.append(completed_queue.get_nowait())
                                except queue.Empty:
                                    break

                            if finished_ops:
                                result_conn.send(finished_ops)
                        except Exception as e:
                            flexkv_logger.error(f"Error collecting finished ops: {e}")
                        nvtx.end_range(inner_range)

                except (EOFError, BrokenPipeError, ConnectionResetError) as e:
                    # Fallback: any IPC-broken exception bubbling up here also
                    # means the parent is gone — exit for graceful cleanup.
                    flexkv_logger.warning(
                        f"TransferManager IPC error ({e!r}); "
                        "parent likely crashed. Exiting for graceful cleanup."
                    )
                    should_exit = True
                except Exception as e:
                    flexkv_logger.error(f"Error in transfer manager process: {e}")

        except Exception as e:
            flexkv_logger.error(f"Failed to initialize transfer manager process: {e}")
        finally:
            # Cleanup selector (only if it was created)
            if 'sel' in locals():
                try:
                    sel.close()
                except Exception as e:
                    flexkv_logger.error(f"Error closing selector: {e}")

            # Gracefully shut down transfer engine and its worker subprocesses
            if 'transfer_manager' in locals():
                try:
                    flexkv_logger.info("TransferManager process: shutting down transfer engine")
                    transfer_manager.shutdown()
                except Exception as e:
                    flexkv_logger.error(f"Error shutting down transfer manager: {e}")

            try:
                command_conn.close()
            except Exception:
                pass
            try:
                result_conn.close()
            except Exception:
                pass
            flexkv_logger.info("TransferManager process cleanup complete")

    def start(self) -> None:
        os.environ['MPI4PY_RC_INITIALIZE'] = 'false'
        self._start_process()
        self.start_event.wait()
        os.environ['MPI4PY_RC_INITIALIZE'] = 'true'

    def is_ready(self) -> bool:
        return self.ready_event.is_set()

    def submit(self, transfer_graph: TransferOpGraph, task_end_op_id: int = -1) -> None:
        nvtx_range = nvtx.start_range(message="TransferManagerInterProcessHandle.submit", color="green")
        self.command_parent_conn.send({
            'type': 'submit',
            'transfer_graph': transfer_graph
        })
        nvtx.end_range(nvtx_range)

    def submit_batch(self, transfer_graphs: List[TransferOpGraph]) -> None:
        # Batch submit to reduce IPC overhead
        nvtx_range = nvtx.start_range(
            message=f"TransferManagerInterProcessHandle.submit_batch count={len(transfer_graphs)}",
            color="green"
        )
        self.command_parent_conn.send({
            'type': 'submit_batch',
            'transfer_graphs': transfer_graphs
        })
        nvtx.end_range(nvtx_range)

    def wait(self, timeout: Optional[float] = None) -> List[CompletedOp]:
        finished_ops: List[CompletedOp] = []
        try:
            if self.result_parent_conn.poll(timeout=timeout):
                received_ops = self.result_parent_conn.recv()
                finished_ops += received_ops
                while self.result_parent_conn.poll():
                    received_ops = self.result_parent_conn.recv()
                    finished_ops += received_ops
        except EOFError:
            pass

        return finished_ops

    def shutdown(self) -> None:
        if self.process is None:
            return

        if self.process.is_alive():
            try:
                flexkv_logger.info(
                    "Sending graceful shutdown command to TransferManager subprocess "
                    f"(pid={self.process.pid})"
                )
                self.command_parent_conn.send({'type': 'shutdown'})
            except (BrokenPipeError, OSError, EOFError) as e:
                flexkv_logger.warning(
                    f"Failed to send TransferManager shutdown command: {e}; "
                    "falling back to terminate"
                )
                self.process.terminate()

            timeout = float(GLOBAL_CONFIG_FROM_ENV.transfer_manager_shutdown_timeout_s)
            self.process.join(timeout=timeout)
            if self.process.is_alive():
                flexkv_logger.warning(
                    f"TransferManager still alive after {timeout:.0f}s graceful wait; "
                    "terminating"
                )
                self.process.terminate()
                self.process.join(timeout=30)
                if self.process.is_alive():
                    flexkv_logger.warning(
                        "TransferManager still alive after terminate; killing"
                    )
                    self.process.kill()
                    self.process.join()

        try:
            self.command_parent_conn.close()
        except Exception:
            pass
        try:
            self.result_parent_conn.close()
        except Exception:
            pass
        self.process = None

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass


class TransferManagerMultiNodeHandle(TransferManagerHandleBase):
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_register_port: str,
                 master_host: str,
                 master_ports: Tuple[str, str, str]):  # command, result, query
        self.model_config = model_config
        self.cache_config = cache_config
        self.gpu_register_port = gpu_register_port

        self.master_host = master_host
        self.master_ports = master_ports

        self.context = zmq.Context()
        self.command_socket = self.context.socket(zmq.PUSH)
        self.command_socket.setsockopt(zmq.LINGER, 0)
        self.result_socket = self.context.socket(zmq.PULL)
        self.result_socket.setsockopt(zmq.LINGER, 0)
        self.query_socket = self.context.socket(zmq.REQ)
        self.query_socket.setsockopt(zmq.LINGER, 0)
        self.query_socket.setsockopt(zmq.REQ_RELAXED, 1)
        self.query_socket.setsockopt(zmq.REQ_CORRELATE, 1)
        self.query_socket.setsockopt(zmq.RCVTIMEO, 1000)

        self._shutdown_flag = False
        self._connected = False

        self._result_buffer: List[CompletedOp] = []
        self._result_buffer_lock = threading.Lock()

        self._bind_master_ports()

        self._polling_thread: threading.Thread | None = None

    def _bind_master_ports(self) -> None:
        try:
            command_addr = f"tcp://{self.master_host}:{self.master_ports[0]}"
            self.command_socket.bind(command_addr)
            flexkv_logger.info(f"Master bound command port at {command_addr}")

            result_addr = f"tcp://{self.master_host}:{self.master_ports[1]}"
            self.result_socket.bind(result_addr)
            flexkv_logger.info(f"Master bound result port at {result_addr}")

            query_addr = f"tcp://{self.master_host}:{self.master_ports[2]}"
            self.query_socket.bind(query_addr)
            flexkv_logger.info(f"Master bound query port at {query_addr}")

            self.result_socket.setsockopt(zmq.RCVTIMEO, 0)

            self._connected = True
            flexkv_logger.info("Master transfer manager ready for remote connections")

        except Exception as e:
            flexkv_logger.error(f"Master failed to bind ports: {e}")
            try:
                self.command_socket.close()
                self.result_socket.close()
                self.query_socket.close()
                self.context.term()
            except Exception:
                pass
            raise

    def send_config_to_remotes(self) -> None:
        flexkv_logger.info(f"Sending config to remote at {self.master_host}:{self.master_ports[0]}")
        try:
            config_msg = {
                'type': 'config',
                'model_config': self.model_config,
                'cache_config': self.cache_config,
                'gpu_register_port': self.gpu_register_port
            }
            self.command_socket.send_pyobj(config_msg)
            flexkv_logger.info(f"Config sent to remote at {self.master_host}:{self.master_ports[0]}")
        except Exception as e:
            flexkv_logger.error(f"Failed to send config to remote: {e}")

    def _polling_worker(self) -> None:
        while not self._shutdown_flag:
            try:
                result = self.result_socket.recv_pyobj(zmq.NOBLOCK)
                if isinstance(result, CompletedOp):
                    with self._result_buffer_lock:
                        self._result_buffer.append(result)
                else:
                    flexkv_logger.warning(f"Unexpected result format from remote: {result}")

            except zmq.Again:
                time.sleep(0.001)
            except Exception as e:
                if not self._shutdown_flag:
                    flexkv_logger.error(f"Error in polling thread: {e}")
                    time.sleep(0.01)

    def start(self) -> None:
        self._polling_thread = threading.Thread(target=self._polling_worker, daemon=True)
        self._polling_thread.start()

    def is_ready(self) -> bool:
        if not self._connected:
            flexkv_logger.warning("Master not ready: ports not bound yet")
            return False

        try:
            query_msg = {'type': 'query_ready'}
            self.query_socket.send_pyobj(query_msg)

            response = self.query_socket.recv_pyobj()
            if response.get('ready'):
                return True
            else:
                flexkv_logger.warning(f"Remote not ready, response: {response}")
                return False

        except zmq.Again:
            flexkv_logger.warning("Timeout waiting for ready response from remote")
            return False
        except Exception as e:
            flexkv_logger.error(f"Error checking remote ready status: {e}")

            return False

    def submit(self, transfer_graph: TransferOpGraph, task_end_op_id: int = -1) -> None:
        if not self._connected:
            flexkv_logger.warning("Not connected to remote transfer manager")
            return

        try:
            message = {
                'type': 'submit',
                'graph': transfer_graph,
                'task_end_op_id': task_end_op_id
            }
            self.command_socket.send_pyobj(message)

        except Exception as e:
            flexkv_logger.error(f"Failed to submit graph to remote: {e}")

    def submit_batch(self, transfer_graphs: List[TransferOpGraph]) -> None:
        if not self._connected:
            flexkv_logger.warning("Not connected to remote transfer manager")
            return

        try:
            message = {
                'type': 'submit_batch',
                'graphs': transfer_graphs
            }
            self.command_socket.send_pyobj(message)

        except Exception as e:
            flexkv_logger.error(f"Failed to submit batch graphs to remote: {e}")

    def wait(self, timeout: float | None = None) -> List[CompletedOp]:
        start_time = time.time()
        results = []

        while True:
            with self._result_buffer_lock:
                if self._result_buffer:
                    results.extend(self._result_buffer)
                    self._result_buffer.clear()
                    break
                elif timeout is not None and (time.time() - start_time) >= timeout:
                    break

            time.sleep(0.001)

        return results

    def shutdown(self) -> None:
        flexkv_logger.info("Shutting down TransferManagerMultiNodeHandle")

        self._shutdown_flag = True

        if self._polling_thread is not None and self._polling_thread.is_alive():
            self._polling_thread.join(timeout=5.0)

        try:
            self.command_socket.close()
            self.result_socket.close()
            self.query_socket.close()
            self.context.term()
        except Exception as e:
            flexkv_logger.error(f"Error closing sockets: {e}")

        flexkv_logger.info("TransferManagerMultiNodeHandle shutdown complete")


class TransferManagerHandle:
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_register_port: Optional[str] = None,
                 mode: str = "process",
                 **kwargs): # process or thread or remote
        flexkv_logger.debug(
            f"Creating TransferManagerHandle: mode={mode}, "
            f"tp_size={model_config.tp_size}, dp_size={model_config.dp_size}, "
            f"pp_size={model_config.pp_size}, nnodes={model_config.nnodes}, "
            f"gpu_register_port={gpu_register_port}")
        if gpu_register_port is None:
            gpu_register_port = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
        if mode == "process":
            self._handle: TransferManagerHandleBase = TransferManagerInterProcessHandle(
                model_config, cache_config, gpu_register_port
            )
        elif mode == "thread":
            self._handle: TransferManagerHandleBase = TransferManagerIntraProcessHandle(
                model_config, cache_config, gpu_register_port
            )
        elif mode == "remote":
            master_host = kwargs["master_host"]
            master_ports = kwargs["master_ports"]
            self._handle: TransferManagerHandleBase = TransferManagerMultiNodeHandle(
                model_config, cache_config, gpu_register_port, master_host, master_ports
            )
        else:
            raise ValueError(f"Invalid mode: {mode}, must be process, thread or remote")

    def start(self) -> None:
        self._handle.start()

    def is_ready(self) -> bool:
        return self._handle.is_ready()

    def submit(self, transfer_graph: TransferOpGraph, task_end_op_id: int = -1) -> None:
        self._handle.submit(transfer_graph, task_end_op_id)

    def submit_batch(self, transfer_graphs: List[TransferOpGraph]) -> None:
        self._handle.submit_batch(transfer_graphs)

    def wait(self, timeout: Optional[float] = None) -> List[CompletedOp]:
        return self._handle.wait(timeout)

    def shutdown(self) -> None:
        self._handle.shutdown()
