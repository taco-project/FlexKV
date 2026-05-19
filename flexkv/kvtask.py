import time
from typing import Dict, Optional, List, Union, Tuple
import threading
from enum import Enum
from dataclasses import dataclass
from typing import Callable
import multiprocessing as mp
import copy
from expiring_dict import ExpiringDict
import nvtx
import numpy as np

from flexkv.common.config import CacheConfig, ModelConfig, GLOBAL_CONFIG_FROM_ENV
from flexkv.common.debug import flexkv_logger
from flexkv.common.block import hash_token
from flexkv.common.transfer import TransferOpGraph, merge_to_batch_graph, get_nvtx_default_color, CompletedOp
from flexkv.common.tracer import FlexKVTracer
from flexkv.cache.cache_engine import GlobalCacheEngine, DEFAULT_CACHE_STRATEGY, CPUONLY_CACHE_STRATEGY
from flexkv.transfer_manager import TransferManagerHandle, TransferManagerOnRemote
from flexkv.common.request import KVResponseStatus, KVResponse
from flexkv.cache.redis_meta import RedisMeta
from flexkv.integration.dynamo.collector import KVEventCollector
from flexkv.transfer_manager import TransferManagerMultiNodeHandle

class TaskStatus(Enum):
    # slot mapping is not ready
    UNREADY = "unready"
    # waiting for the task to be launched
    READY = "ready"
    # in transfer
    RUNNING = "running"
    # transfer completed
    COMPLETED = "completed"
    # transfer cancelled
    CANCELLED = "cancelled"
    # transfer failed
    FAILED = "failed"

class TaskType(Enum):
    GET = "get"
    PUT = "put"
    PREFETCH = "prefetch"
    BATCH_GET = "batch_get"
    BATCH_PUT = "batch_put"

@dataclass
class KVTask:
    # task descriptor
    task_id: int
    task_type: TaskType
    task_end_op_id: int
    task_end_op_finished: bool
    status: TaskStatus

    # params
    token_ids: np.ndarray
    slot_mapping: np.ndarray
    token_mask: Optional[np.ndarray]

    # cache engine return
    graph: TransferOpGraph
    return_mask: Union[np.ndarray, list[np.ndarray]]
    callback: Optional[Union[Callable, List[Callable]]]
    op_callback_dict: Dict[int, Callable]

    # batch: points to the batch task id if this task was merged into a batch
    batch_task_id: Optional[int] = None

    def is_completed(self) -> bool:
        return self.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED]

TASK_STATUS_TO_RESPONSE_STATUS = {
    TaskStatus.COMPLETED: KVResponseStatus.SUCCESS,
    TaskStatus.CANCELLED: KVResponseStatus.CANCELLED,
    TaskStatus.FAILED: KVResponseStatus.FAILED,
    TaskStatus.RUNNING: KVResponseStatus.SUCCESS, # for early return: still running, but success
}

def convert_to_response_status(task_status: TaskStatus) -> KVResponseStatus:
    return TASK_STATUS_TO_RESPONSE_STATUS[task_status]

class KVTaskManager:
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_register_port: Optional[str] = None,
                 redis_meta: RedisMeta = None,
                 event_collector: Optional[KVEventCollector] = None
                 ):
        self.model_config = model_config
        self.cache_config = cache_config

        flexkv_logger.info(
            f"[KVTaskEngine] topology: {self.model_config}"
        )

        self.cache_engine = GlobalCacheEngine(cache_config, model_config, redis_meta, event_collector)

        if not self.model_config.use_trtllm_subprocess:
            self.transfer_handles = [TransferManagerHandle(
                model_config,
                cache_config,
                mode="process",
                gpu_register_port=gpu_register_port
            )]
        else:
            # When using FlexKV with TensorRT-LLM, we use remote mode to transfer data
            #  to avoid the way we launch subprocess in FlexKV
            #  conflict with TensorRT-LLM's MPI initialization.
            sub_host = self.model_config.trtllm_subprocess_host
            sub_ports = self.model_config.trtllm_subprocess_ports
            self.remote_process = TransferManagerOnRemote.create_process(
                master_host=sub_host,
                master_ports=sub_ports,
            )
            self.transfer_handles = [
                TransferManagerHandle(
                    model_config,
                    cache_config,
                    mode="remote",
                    gpu_register_port=gpu_register_port,
                    master_host=sub_host,
                    master_ports=sub_ports,
                )
            ]
            self.transfer_handles[0]._handle.send_config_to_remotes()

        if self.model_config.nnodes > 1:
            self.transfer_handles.append(TransferManagerHandle(
                model_config,
                cache_config,
                mode="remote",
                gpu_register_port=gpu_register_port,
                master_host=self.model_config.master_host,
                master_ports=self.model_config.master_ports,
            ))
            self.transfer_handles[-1]._handle.send_config_to_remotes()

        # Phase 0 task 0-G: when sharing-domain is on, replace / augment the
        # handle list with one handle per SD (Master SD + every peer SD's
        # Remote).  This runs after the legacy construction above so that
        # legacy single-Remote and sharing-domain paths stay independent —
        # the legacy branch leaves ``self.transfer_handles`` untouched when
        # ``enable_sharing_domain`` is False.
        self._master_coordinator = None
        if getattr(self.cache_config, "enable_sharing_domain", False):
            self._setup_sharing_domain_handles(gpu_register_port=gpu_register_port)

        self.tasks: ExpiringDict[int, KVTask] = ExpiringDict(max_age_seconds=1800, max_len=100000) # 30 minutes

        # hash(token_ids) -> task_id
        self.prefetch_tasks: ExpiringDict[int, int] = ExpiringDict(max_age_seconds=1800, max_len=100000) # 30 minutes
        self._gen_prefetch_key = lambda token_ids, namespace: hash_token(token_ids, namespace)

        self.graph_to_task: Dict[int, int] = {}

        self.uncompleted_ops: Dict[int, int] = {}  # op_id -> completed_count
        self.uncompleted_graphs: Dict[int, int] = {}  # graph_id -> completed_count
        self.required_completed_count: int = len(self.transfer_handles)

        self.task_id_counter = 0
        self.task_id_lock = threading.Lock()

        self.running_tasks: int = 0

    def start(self) -> None:
        for transfer_handle in self.transfer_handles:
            transfer_handle.start()

    def is_ready(self) -> bool:
        return all(transfer_handle.is_ready() for transfer_handle in self.transfer_handles)

    def __del__(self) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        if hasattr(self, "transfer_handles") and self.transfer_handles is not None:
            for transfer_handle in self.transfer_handles:
                transfer_handle.shutdown()
        if hasattr(self, "remote_process") and self.remote_process is not None:
            assert self.remote_process.is_alive()
            self.remote_process.terminate()
            self.remote_process.join()
            self.remote_process.close()
            self.remote_process = None
        # Phase 0 task 0-G: tear down sharing-domain background threads.
        if getattr(self, "_master_coordinator", None) is not None:
            try:
                # Batch-F: drop cache_engine's references before the
                # coordinator goes away so any in-flight completion
                # callback (_on_peer_sd_completed_op /
                # handle_failure_report) sees a clean None instead of
                # a half-torn-down object.
                if hasattr(self, "cache_engine") and self.cache_engine is not None:
                    try:
                        self.cache_engine.detach_dist_reuse()
                    except Exception:
                        pass
                self._master_coordinator.shutdown()
            except Exception as e:
                flexkv_logger.warning(f"MasterCoordinator.shutdown() raised: {e}")
            self._master_coordinator = None

    def _setup_sharing_domain_handles(self, *, gpu_register_port: Optional[str]) -> None:
        """Populate ``self.transfer_handles`` with one handle per SD and
        create a :class:`MasterCoordinator` on the Master node.

        No-op when ``cache_config.enable_sharing_domain`` is False — the
        caller already gates this.  This method is **best-effort**: if
        the user hasn't populated ``remote_endpoints_by_sd`` yet (e.g.
        single-SD degenerate mode), we keep the legacy handle list and
        just construct the coordinator for the local SD.
        """
        from flexkv.common.dist_reuse import (
            MasterCoordinator,
            SharingDomainKey,
            build_sharing_domain_handles,
            make_session_epoch,
        )

        self_sd = SharingDomainKey.from_model_config(self.model_config)

        # Single-SD degenerate case (no sharing) — no extra handles needed.
        if self_sd.total_sd_count() <= 1:
            flexkv_logger.info(
                "[DistReuse] Master SD is the only SD in the instance; "
                "skipping multi-SD handle construction."
            )
            # Still spin up a MasterCoordinator so aggregate-radix hooks
            # work uniformly (it will just track 1 SD).
            instance_id = self.cache_config.instance_id or f"inst-{self_sd.serialize()}"
            self._master_coordinator = MasterCoordinator(
                self_sd=self_sd,
                instance_id=instance_id,
                session_epoch=self.cache_config.session_epoch or make_session_epoch(),
            )
            self._master_coordinator.expect_remotes(0)
            # Single-SD path still attaches the coord state to the cache
            # engine so refcount / aggregate / failure-detection hooks work.
            self._wire_dist_reuse_coord_dispatcher()
            return

        try:
            specs = build_sharing_domain_handles(
                self_sd=self_sd,
                remote_endpoints_by_sd=self.cache_config.remote_endpoints_by_sd,
            )
        except KeyError as e:
            flexkv_logger.warning(
                f"[DistReuse] could not build multi-SD handles: {e}. "
                f"Falling back to legacy handle list."
            )
            return

        # Drop the legacy handles and rebuild from scratch — we're on the
        # multi-SD path now.
        for h in self.transfer_handles:
            try:
                h.shutdown()
            except Exception:  # pragma: no cover
                pass
        self.transfer_handles = []

        # Build new handle list per spec.
        for spec in specs:
            if spec.mode == "process":
                handle = TransferManagerHandle(
                    self.model_config, self.cache_config,
                    mode="process", gpu_register_port=gpu_register_port,
                )
            else:
                ep = spec.endpoint
                master_host = ep.ip
                master_ports = (ep.gpu_register_port, ep.command_port, ep.result_port)
                handle = TransferManagerHandle(
                    self.model_config, self.cache_config,
                    mode="remote",
                    gpu_register_port=gpu_register_port,
                    master_host=master_host,
                    master_ports=master_ports,
                )
                handle._handle.set_target_sd_key(spec.sd_key.serialize())
                handle._handle.send_config_to_remotes()
            self.transfer_handles.append(handle)

        self.required_completed_count = len(self.transfer_handles)

        # Create the Master coordinator; it will accept RemoteReadyMsg
        # acks as Remote nodes finish their dist_reuse bootstrap.
        instance_id = self.cache_config.instance_id or f"inst-{self_sd.serialize()}"
        self._master_coordinator = MasterCoordinator(
            self_sd=self_sd,
            instance_id=instance_id,
            session_epoch=self.cache_config.session_epoch or make_session_epoch(),
            refcount_leak_timeout_seconds=self.cache_config.refcount_leak_timeout_seconds,
        )
        self._master_coordinator.expect_remotes(len(specs) - 1)

        # Batch-F: wire up the cross-SD coordinator now that we have both
        # the MasterCoordinator (aggregate radix + refcount owner) and
        # the fully-populated transfer_handles list (one per SD).
        self._wire_dist_reuse_coord_dispatcher()

    def _wire_dist_reuse_coord_dispatcher(self) -> None:
        """Phase D-2 (proposal_unify_with_graph_dispatch_2026-05-15.md
        §6.3 / §3.5): wire the master-side completion sink so peer-SD
        ``CompletedOp(sd_key, contributing_node_id)`` flowing back through
        each Remote handle's ``_polling_worker`` is routed to the
        cache_engine's ``_on_peer_sd_completed_op`` method.

        Replaces the pre-Phase-D ``CoordinationCoordinator +
        set_coord_ack_sink`` wiring (Phase D-4 deleted the latter
        entirely).  ``set_coord_ack_sink`` is still used here, but only
        for the surviving ``FailureReportMsg`` channel — see
        ``MasterCoordinator.handle_failure_report``.

        Idempotent: safe to call multiple times.
        Degenerates to a single-SD no-op when the instance has exactly
        one SD (``total_sd_count == 1``).
        """
        if self._master_coordinator is None:
            return

        self_sd = self._master_coordinator.self_sd
        total = self_sd.total_sd_count()
        if total <= 1:
            # Single-SD instance: still attach the master_coord (for
            # refcount / aggregate / failure detection), but no cross-SD
            # dispatch is needed.
            self.cache_engine.attach_dist_reuse(self._master_coordinator)
            return

        # Phase D-2: register the master's completion sink on every
        # peer-SD handle so CompletedOp(sd_key=..., contributing_node_id=...)
        # gets routed to GlobalCacheEngine._on_peer_sd_completed_op.
        sink = self.cache_engine._on_peer_sd_completed_op
        # Phase D-4: also register a FailureReportMsg sink so peer
        # data-plane failures invalidate aggregate-radix prefixes.
        failure_sink = self._master_coordinator.handle_failure_report
        # Phase D-3 (proposal_unify_with_graph_dispatch_2026-05-15.md
        # §6.4): the Master's own in-proc / inter-proc handle must also
        # honour ``target_node_ids`` filtering so peer-SD clone ops
        # (D-2 PUT D2H clones, D-3 GET PEERH2H clones) are NOT executed
        # by the Master's local TransferEngine.  Without this the
        # Master would either waste GPU bandwidth (D-2 D2H mirror is
        # idempotent) or pull data from peer-SD mooncake endpoints it
        # never connected to (D-3 PEERH2H — silent corruption).
        master_self_nid = int(getattr(self.cache_config, "distributed_node_id", -1))
        for h in self.transfer_handles:
            inner = h._handle
            if hasattr(inner, "set_completion_sink"):
                try:
                    inner.set_completion_sink(sink)
                except Exception as e:  # pragma: no cover
                    flexkv_logger.warning(
                        f"[DistReuse] set_completion_sink failed: {e}"
                    )
            if hasattr(inner, "set_coord_ack_sink"):
                try:
                    inner.set_coord_ack_sink(failure_sink)
                except Exception as e:  # pragma: no cover
                    flexkv_logger.warning(
                        f"[DistReuse] set_coord_ack_sink failed: {e}"
                    )
            # Phase D-3: only the Master's own in-proc / inter-proc
            # handle exposes ``set_dist_reuse_node_id``; the multi-node
            # remote handles do their filtering on the Remote side via
            # ``TransferManagerOnRemote._filter_graph_by_target_node_ids``
            # and ``_dist_reuse_node_id`` set during bootstrap.
            if hasattr(inner, "set_dist_reuse_node_id") and master_self_nid >= 0:
                try:
                    inner.set_dist_reuse_node_id(master_self_nid)
                except Exception as e:  # pragma: no cover
                    flexkv_logger.warning(
                        f"[DistReuse] set_dist_reuse_node_id failed: {e}"
                    )

        # Wire into the cache engine.  Cross-SD coordination flows
        # through the graph-dispatch path with per-op
        # ``target_node_ids`` filtering (Phase D-4); no separate coord
        # dispatcher is needed.
        self.cache_engine.attach_dist_reuse(self._master_coordinator)

    def create_get_task(self,
                        task_id: int,
                        token_ids: np.ndarray,
                        slot_mapping: np.ndarray,
                        dp_client_id: int,
                        token_mask: Optional[np.ndarray] = None,
                        is_fake_slot_mapping: bool = False,
                        temp_cache_strategy=DEFAULT_CACHE_STRATEGY,
                        namespace: Optional[List[str]] = None,
                        ) -> None:
        if task_id in self.tasks:
            raise ValueError(f"Task ID {task_id} already exists")
        graph, return_mask, callback, op_callback_dict, task_end_op_id = self.cache_engine.get(
            request_id=task_id,
            token_ids=token_ids,
            token_mask=token_mask,
            slot_mapping=slot_mapping,
            dp_client_id=dp_client_id,
            temp_cache_strategy=temp_cache_strategy,
            namespace=namespace)
        self.tasks[task_id] = KVTask(
            task_id=task_id,
            task_type=TaskType.GET,
            task_end_op_id=task_end_op_id,
            task_end_op_finished=False,
            status=TaskStatus.UNREADY if is_fake_slot_mapping else TaskStatus.READY,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            token_mask=token_mask,
            graph=graph,
            return_mask=return_mask,
            callback=callback,
            op_callback_dict=op_callback_dict)

        self.graph_to_task[graph.graph_id] = task_id

    def create_put_task(self,
                        task_id: int,
                        token_ids: np.ndarray,
                        slot_mapping: np.ndarray,
                        dp_client_id: int,
                        token_mask: Optional[np.ndarray] = None,
                        is_fake_slot_mapping: bool = False,
                        namespace: Optional[List[str]] = None,
                        ) -> None:
        if task_id in self.tasks:
            raise ValueError(f"Task ID {task_id} already exists")
        graph, return_mask, callback, op_callback_dict, task_end_op_id = self.cache_engine.put(
            request_id=task_id,
            token_ids=token_ids,
            token_mask=token_mask,
            slot_mapping=slot_mapping,
            dp_client_id=dp_client_id,
            namespace=namespace)
        self.tasks[task_id] = KVTask(
            task_id=task_id,
            task_type=TaskType.PUT,
            task_end_op_id=task_end_op_id,
            task_end_op_finished=False,
            status=TaskStatus.UNREADY if is_fake_slot_mapping else TaskStatus.READY,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            token_mask=token_mask,
            graph=graph,
            return_mask=return_mask,
            callback=callback,
            op_callback_dict=op_callback_dict)
        self.graph_to_task[graph.graph_id] = task_id

    def create_prefetch_task(self,
                            task_id: int,
                            token_ids: np.ndarray,
                            dp_client_id: int,
                            namespace: Optional[List[str]] = None,
                            ) -> None:
        if task_id in self.tasks:
            raise ValueError(f"Task ID {task_id} already exists")
        fake_slot_mapping = np.zeros_like(token_ids)
        fake_token_mask = np.ones_like(token_ids)
        temp_cache_strategy = copy.deepcopy(DEFAULT_CACHE_STRATEGY)
        temp_cache_strategy.ignore_gpu = True  # upload to CPU only
        temp_cache_strategy.ignore_gds = True
        graph, return_mask, callback, op_callback_dict, task_end_op_id = self.cache_engine.get(
            request_id=task_id,
            token_ids=token_ids,
            token_mask=fake_token_mask,
            slot_mapping=fake_slot_mapping,
            dp_client_id=dp_client_id,
            temp_cache_strategy=temp_cache_strategy,
            namespace=namespace)
        self.tasks[task_id] = KVTask(
            task_id=task_id,
            task_type=TaskType.PREFETCH,
            task_end_op_id=task_end_op_id,
            task_end_op_finished=False,
            status=TaskStatus.READY,  # gpu slots are not needed for prefetch
            token_ids=token_ids,
            slot_mapping=fake_slot_mapping,  # ignore slot_mapping for prefetch
            token_mask=fake_token_mask,  # ignore token_mask for prefetch
            graph=graph,
            return_mask=return_mask,
            callback=callback,
            op_callback_dict=op_callback_dict)

        self.prefetch_tasks[self._gen_prefetch_key(token_ids, namespace)] = task_id

        self.graph_to_task[graph.graph_id] = task_id

    def _launch_task(self, task_id: int) -> None:
        transfer_graph = self.check_task_ready(task_id)
        if transfer_graph is None:
            return
        nvtx.mark(f"launch task: task_id={task_id}, graph_id={transfer_graph.graph_id}")
        if transfer_graph.num_ops > 0:
            # Phase 0 task 0-K: compute per-handle GPU-clear decision *once*
            # so the sharing-domain-aware logic can be unit-tested separately.
            clear_flags = self._compute_gpu_clear_flags()
            for idx, transfer_handle in enumerate(self.transfer_handles):
                if isinstance(transfer_handle._handle, TransferManagerMultiNodeHandle):
                    if clear_flags[idx]:
                        graph_copy = copy.deepcopy(transfer_graph)
                        graph_copy.clear_gpu_blocks()
                        transfer_handle.submit(graph_copy, task_end_op_id=self.tasks[task_id].task_end_op_id)
                    else:
                        transfer_handle.submit(transfer_graph, task_end_op_id=self.tasks[task_id].task_end_op_id)
                else:
                    transfer_handle.submit(transfer_graph, task_end_op_id=self.tasks[task_id].task_end_op_id)

    def _compute_gpu_clear_flags(self) -> List[bool]:
        """Decide for each transfer handle whether its graph needs
        GPU-block clearing before send.

        Legacy (``enable_sharing_domain=False``) behaviour: match the
        pre-Batch-C rule — cross-machine PP needs clearing, cross-machine
        TP does not.

        Sharing-domain behaviour: consult the per-handle SD key and use
        :func:`graph_needs_gpu_clear` from :mod:`flexkv.common.dist_reuse`.
        """
        if not self.transfer_handles:
            return []

        # Legacy branch first.
        if not getattr(self.cache_config, "enable_sharing_domain", False):
            legacy_clear = (
                self.model_config.nnodes > 1 and self.model_config.pp_size > 1
            )
            out: List[bool] = []
            for h in self.transfer_handles:
                if isinstance(h._handle, TransferManagerMultiNodeHandle):
                    out.append(legacy_clear)
                else:
                    out.append(False)
            return out

        # Sharing-domain branch.
        from flexkv.common.dist_reuse import (
            SharingDomainKey,
            graph_needs_gpu_clear,
        )
        self_sd = SharingDomainKey.from_model_config(self.model_config)
        flags: List[bool] = []
        for h in self.transfer_handles:
            if not isinstance(h._handle, TransferManagerMultiNodeHandle):
                flags.append(False)
                continue
            peer_sd_str = getattr(h._handle, "_target_sd_key", None)
            if peer_sd_str is None:
                # Legacy remote with no SD tag — fall back to old rule.
                flags.append(
                    self.model_config.nnodes > 1 and self.model_config.pp_size > 1
                )
                continue
            try:
                peer_sd = SharingDomainKey.deserialize(peer_sd_str)
            except ValueError:
                flags.append(True)  # be conservative
                continue
            flags.append(graph_needs_gpu_clear(self_sd, peer_sd))
        return flags

    def _update_tasks(self, timeout: float = 0.001) -> None:
        completed_ops = self._get_completed_ops(timeout)
        for completed_op in completed_ops:
            if completed_op.graph_id not in self.graph_to_task:
                continue
            task_id = self.graph_to_task[completed_op.graph_id]
            task = self.tasks[task_id]
            if completed_op.is_graph_completed():
                self._mark_completed(task_id)
            elif completed_op.op_id == task.task_end_op_id:
                self.tasks[task_id].task_end_op_finished = True
            if completed_op.op_id in task.op_callback_dict:
                task.op_callback_dict[completed_op.op_id]()

    def _cancel_task(self, task_id: int) -> None:
        task = self.tasks[task_id]
        if task.is_completed():
            flexkv_logger.warning(f"Task {task_id} is already completed, cannot cancel")
            return
        if task.status == TaskStatus.RUNNING:
            flexkv_logger.warning(f"Task {task_id} is running, cannot cancel")
            return
        if task.status == TaskStatus.CANCELLED:
            flexkv_logger.warning(f"Task {task_id} is already cancelled, cannot cancel")
            return
        task.status = TaskStatus.CANCELLED
        self.graph_to_task.pop(task.graph.graph_id, None)

    def check_completed(self, task_id: int, completely: bool = False) -> bool:
        task = self.tasks[task_id]
        if task.batch_task_id is not None:
            return self.check_completed(task.batch_task_id, completely)
        self._process_empty_graph(task_id)
        if completely:
            return task.is_completed()
        # For tasks with callback (e.g., PUT tasks that need to call insert_and_publish),
        # we must wait until _mark_completed is called (i.e., is_completed() returns True)
        # to ensure the callback is executed before returning success.
        #if task.callback is not None:
        #    return task.is_completed()
        return task.is_completed() or task.task_end_op_finished

    def set_slot_mappings(self,
                          task_ids: List[int],
                          slot_mappings: List[np.ndarray]) -> None:
        for task_id, slot_mapping in zip(task_ids, slot_mappings):
            self._set_slot_mapping_impl(task_id, slot_mapping)

    def _set_slot_mapping_impl(self, task_id: int, slot_mapping: np.ndarray) -> None:
        task = self.tasks[task_id]
        if task.status != TaskStatus.UNREADY:
            return
        graph_ids = self.cache_engine.slot_mapping_to_block_ids(slot_mapping,
                                                                self.cache_config.tokens_per_block)
        task.graph.set_gpu_blocks(graph_ids)
        task.status = TaskStatus.READY

    def _gen_task_id(self) -> int:
        with self.task_id_lock:
            old_value = self.task_id_counter
            self.task_id_counter += 1
            return old_value

    def check_task_ready(self, task_id: int) -> TransferOpGraph:
        task = self.tasks[task_id]
        if task.is_completed():
            return None
        if task.status != TaskStatus.READY:
            raise ValueError(f"Task {task_id} status is {task.status}, cannot launch")
        task.status = TaskStatus.RUNNING
        return task.graph

    def _mark_completed(self, task_id: int) -> None:
        task = self.tasks[task_id]
        if task.is_completed():
            return
        if task.callback:
            if isinstance(task.callback, list):
                for callback in task.callback:
                    callback()
            else:
                task.callback()
        task.status = TaskStatus.COMPLETED
        task.task_end_op_finished = True
        self.graph_to_task.pop(task.graph.graph_id)

    def _process_empty_graph(self, task_id: int) -> None:
        task = self.tasks[task_id]
        if task.graph.num_ops == 0:
            self._mark_completed(task_id)

    def _get_completed_ops(self, timeout: Optional[float] = None) -> List[CompletedOp]:
        results = []
        for transfer_handle in self.transfer_handles:
            completed_ops = transfer_handle.wait(timeout)
            for completed_op in completed_ops:
                if completed_op.is_graph_completed():
                    completed_count = self.uncompleted_graphs.get(completed_op.graph_id, 0) + 1
                    if completed_count == self.required_completed_count:
                        results.append(completed_op)
                        self.uncompleted_graphs.pop(completed_op.graph_id, None)
                    else:
                        self.uncompleted_graphs[completed_op.graph_id] = completed_count
                else:
                    completed_count = self.uncompleted_ops.get(completed_op.op_id, 0) + 1
                    if completed_count == self.required_completed_count:
                        results.append(completed_op)
                        self.uncompleted_ops.pop(completed_op.op_id, None)
                    else:
                        self.uncompleted_ops[completed_op.op_id] = completed_count
        return results

class KVTaskEngine(KVTaskManager):
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_register_port: Optional[str] = None,
                 redis_meta: Optional[RedisMeta] = None,
                 event_collector: Optional[KVEventCollector] = None
                 ):
        super().__init__(model_config, cache_config, gpu_register_port, redis_meta, event_collector)
        self.tracer = FlexKVTracer()
        self.tracer.trace_config(model_config, cache_config, gpu_layout=None)

    def get_async(self,
                  token_ids: np.ndarray,
                  slot_mapping: np.ndarray,
                  dp_client_id: int = 0,
                  token_mask: Optional[np.ndarray] = None,
                  task_id: int = -1,
                  namespace: Optional[List[str]] = None) -> Tuple[int, np.ndarray]:
        # self._sync_prefetch(token_ids, namespace)
        task_id, return_mask = self._get_match_impl(token_ids,
                                                    slot_mapping,
                                                    is_fake_slot_mapping=False,
                                                    token_mask=token_mask,
                                                    dp_client_id=dp_client_id,
                                                    task_id=task_id,
                                                    namespace=namespace)
        # trace get request
        self.tracer.trace_request(
            request_type="GET",
            request_id=task_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            token_mask=token_mask,
            dp_client_id=dp_client_id
        )
        self._launch_task(task_id)
        return task_id, return_mask

    def put_async(self,
                  token_ids: np.ndarray,
                  slot_mapping: np.ndarray,
                  dp_client_id: int = 0,
                  token_mask: Optional[np.ndarray] = None,
                  task_id: int = -1,
                  namespace: Optional[List[str]] = None) -> Tuple[int, np.ndarray]:
        task_id, return_mask = self._put_match_impl(token_ids,
                                                    slot_mapping,
                                                    is_fake_slot_mapping=False,
                                                    token_mask=token_mask,
                                                    dp_client_id=dp_client_id,
                                                    task_id=task_id,
                                                    namespace=namespace)
        # trace put request
        self.tracer.trace_request(
            request_type="PUT",
            request_id=task_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            token_mask=token_mask,
            dp_client_id=dp_client_id
        )
        self._launch_task(task_id)
        return task_id, return_mask

    def _wait_impl(self,
                   task_ids: List[int],
                   timeout: float = 20.0,
                   completely: bool = False,
                   only_return_finished: bool = False,
                   ) -> Dict[int, KVResponse]:
        return_responses = {}
        start_time = time.time()
        is_timeout = timeout == 0.0

        self._update_tasks(timeout=0)

        for task_id in task_ids:
            nvtx_range = nvtx.start_range(message=f"KVTask.wait[{task_id}]", color="red")
            while True:
                if task_id not in self.tasks:
                    flexkv_logger.error(f"task_id {task_id} not submitted into flexKV")
                    return_responses[task_id] = KVResponse(
                        status=KVResponseStatus.NOTFOUND,
                        task_id=task_id,
                        return_mask=None
                    )
                    break
                elif self.tasks[task_id].status == TaskStatus.UNREADY:
                    flexkv_logger.warning(f"task_id {task_id} is unready")
                    return_responses[task_id] = KVResponse(
                        status=KVResponseStatus.UNREADY,
                        task_id=task_id,
                        return_mask=None
                    )
                    break
                elif self.check_completed(task_id, completely=completely):
                    effective_id = self.tasks[task_id].batch_task_id or task_id
                    return_responses[task_id] = KVResponse(
                        status=convert_to_response_status(self.tasks[effective_id].status),
                        task_id=task_id,
                        return_mask=self.tasks[task_id].return_mask
                    )
                    break
                elif only_return_finished:
                    break
                elif time.time() - start_time > timeout:
                    is_timeout = True
                if is_timeout:
                    return_responses[task_id] = KVResponse(
                        status=KVResponseStatus.TIMEOUT,
                        task_id=task_id,
                        return_mask=None
                    )
                    break
                self._update_tasks(timeout=0.001)
            nvtx.end_range(nvtx_range)
        return return_responses

    def try_wait(self, task_ids: Union[int, List[int]]) -> Dict[int, KVResponse]:
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        nvtx.mark(f"try_wait task_ids: {task_ids}")
        # trace try_wait request
        self.tracer.trace_wait_request(
            wait_type="try_wait",
            task_ids=task_ids,
            timeout=None,  # try_wait doesn't have explicit timeout
            completely=False
        )
        return_responses = self._wait_impl(task_ids,
                                           completely=False,
                                           only_return_finished=True)
        return return_responses

    def wait(self,
             task_ids: Union[int, List[int]],
             timeout: float = 20.0,
             completely: bool = False) -> Dict[int, KVResponse]:
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        nvtx.push_range(f"wait task_ids: {task_ids}", color=get_nvtx_default_color())
        # trace wait request
        self.tracer.trace_wait_request(
            wait_type="wait",
            task_ids=task_ids,
            timeout=timeout,
            completely=completely
        )
        return_responses = self._wait_impl(task_ids, timeout, completely=completely)
        nvtx.pop_range()
        return return_responses

    def _sync_prefetch(self, token_ids: np.ndarray, namespace: Optional[List[str]] = None) -> None:
        prefetch_task_id = self.prefetch_tasks.get(self._gen_prefetch_key(token_ids, namespace), None)
        if prefetch_task_id is not None:
            start_time = time.time()
            self.wait([prefetch_task_id], completely=True)
            end_time = time.time()
            flexkv_logger.debug(f"sync prefetch task {prefetch_task_id} cost {(end_time - start_time) * 1000} ms")

    def get_match(self,
                  token_ids: np.ndarray,
                  dp_client_id: int = 0,
                  token_mask: Optional[np.ndarray] = None,
                  cpu_only: bool = False,
                  task_id: int = -1,
                  namespace: Optional[List[str]] = None) -> Tuple[int, np.ndarray]:
        nvtx.push_range(f"get match: task_id={task_id}", color=get_nvtx_default_color())
        # self._sync_prefetch(token_ids, namespace)
        if token_mask is None:
            token_mask = np.ones_like(token_ids, dtype=bool)
        fake_slot_mapping = np.zeros_like(token_ids[token_mask])
        result_task_id, return_mask = self._get_match_impl(token_ids,
                                                           fake_slot_mapping,
                                                           is_fake_slot_mapping=True,
                                                           token_mask=token_mask,
                                                           dp_client_id=dp_client_id,
                                                           cpu_only=cpu_only,
                                                           task_id=task_id,
                                                           namespace=namespace)
        # trace get match request
        self.tracer.trace_request(
            request_type="GET_MATCH",
            request_id=result_task_id,
            token_ids=token_ids,
            slot_mapping=fake_slot_mapping,
            token_mask=token_mask,
            dp_client_id=dp_client_id
        )
        nvtx.pop_range()
        return result_task_id, return_mask

    def _get_match_impl(self,
                  token_ids: np.ndarray,
                  slot_mapping: np.ndarray,
                  dp_client_id: int,
                  is_fake_slot_mapping: bool = False,
                  token_mask: Optional[np.ndarray] = None,
                  cpu_only: bool = False,
                  task_id: int = -1,
                  namespace: Optional[List[str]] = None) -> Tuple[int, np.ndarray]:
        if token_mask is None:
            token_mask = np.ones_like(token_ids)
        if task_id == -1:
            task_id = self._gen_task_id()
        temp_cache_strategy = DEFAULT_CACHE_STRATEGY
        if cpu_only:
            temp_cache_strategy = CPUONLY_CACHE_STRATEGY
        nvtx.push_range(f"get match: task_id={task_id}", color=get_nvtx_default_color())
        self.create_get_task(task_id=task_id,
                             token_ids=token_ids,
                             slot_mapping=slot_mapping,
                             dp_client_id=dp_client_id,
                             token_mask=token_mask,
                             is_fake_slot_mapping=is_fake_slot_mapping,
                             temp_cache_strategy=temp_cache_strategy,
                             namespace=namespace)
        self._process_empty_graph(task_id)
        nvtx.pop_range()
        return task_id, self.tasks[task_id].return_mask

    def put_match(self,
                  token_ids: np.ndarray,
                  dp_client_id: int = 0,
                  token_mask: Optional[np.ndarray] = None,
                  task_id: int = -1,
                  namespace: Optional[List[str]] = None) -> Tuple[int, np.ndarray]:
        fake_slot_mapping = np.zeros_like(token_ids)
        result_task_id, return_mask = self._put_match_impl(token_ids,
                                                           fake_slot_mapping,
                                                           is_fake_slot_mapping=True,
                                                           token_mask=token_mask,
                                                           dp_client_id=dp_client_id,
                                                           task_id=task_id,
                                                           namespace=namespace)
        # trace put match request
        self.tracer.trace_request(
            request_type="PUT_MATCH",
            request_id=result_task_id,
            token_ids=token_ids,
            slot_mapping=fake_slot_mapping,
            token_mask=token_mask,
            dp_client_id=dp_client_id
        )
        return result_task_id, return_mask

    def _put_match_impl(self,
                        token_ids: np.ndarray,
                        slot_mapping: np.ndarray,
                        dp_client_id: int,
                        is_fake_slot_mapping: bool = False,
                        token_mask: Optional[np.ndarray] = None,
                        task_id: int = -1,
                        namespace: Optional[List[str]] = None) -> Tuple[int, np.ndarray]:
        if token_mask is None:
            token_mask = np.ones_like(token_ids)
        if task_id == -1:
            task_id = self._gen_task_id()
        nvtx.push_range(f"put match: task_id={task_id}", color=get_nvtx_default_color())
        self.create_put_task(task_id=task_id,
                             token_ids=token_ids,
                             slot_mapping=slot_mapping,
                             dp_client_id=dp_client_id,
                             token_mask=token_mask,
                             is_fake_slot_mapping=is_fake_slot_mapping,
                             namespace=namespace)
        self._process_empty_graph(task_id)
        nvtx.pop_range()
        return task_id, self.tasks[task_id].return_mask

    def prefetch_async(self,
                       token_ids: np.ndarray,
                       dp_client_id: int = 0,
                       task_id: int = -1,
                       namespace: Optional[List[str]] = None) -> int:
        if task_id == -1:
            task_id = self._gen_task_id()
        nvtx.push_range(f"prefetch match: task_id={task_id}", color=get_nvtx_default_color())
        self.create_prefetch_task(task_id, token_ids, dp_client_id=dp_client_id, namespace=namespace)
        self._process_empty_graph(task_id)
        nvtx.pop_range()
        # trace prefetch async request
        self.tracer.trace_request(
            request_type="PREFETCH_ASYNC",
            request_id=task_id,
            token_ids=token_ids,
            slot_mapping=np.zeros_like(token_ids),
            token_mask=np.ones_like(token_ids),
            dp_client_id=dp_client_id
        )
        self._launch_task(task_id)
        return task_id

    def merge_to_batch_kvtask(self,
                              batch_id: int,
                              task_ids: List[int],
                              batch_task_type: TaskType,
                              layerwise_transfer: bool = False,
                              counter_id: int = 0) -> TransferOpGraph:
        op_callback_dict = {}
        task_end_op_ids = []
        callbacks = []
        transfer_graphs = []
        return_masks = []
        expected_type = TaskType.GET if batch_task_type == TaskType.BATCH_GET else TaskType.PUT
        for task_id in task_ids:
            assert self.tasks[task_id].task_type == expected_type, \
                f"only {expected_type.value} task can be launched as {batch_task_type.value}"
            transfer_graph = self.check_task_ready(task_id)
            if transfer_graph is not None and transfer_graph.num_ops > 0:
                transfer_graphs.append(transfer_graph)
                op_callback_dict.update(self.tasks[task_id].op_callback_dict)
                task_end_op_ids.append(self.tasks[task_id].task_end_op_id)
                callbacks.append(self.tasks[task_id].callback)
                return_masks.append(self.tasks[task_id].return_mask)
        batch_task_graph, task_end_op_id, op_callback_dict = merge_to_batch_graph(batch_id,
                                                                                  transfer_graphs,
                                                                                  task_end_op_ids,
                                                                                  op_callback_dict,
                                                                                  layerwise_transfer,
                                                                                  counter_id)
        self.tasks[batch_id] = KVTask(
            task_id=batch_id,
            token_ids=np.concatenate([self.tasks[task_id].token_ids for task_id in task_ids]),
            slot_mapping=np.concatenate([self.tasks[task_id].slot_mapping for task_id in task_ids]),
            token_mask=np.concatenate([self.tasks[task_id].token_mask for task_id in task_ids]),
            task_type=batch_task_type,
            task_end_op_id=task_end_op_id,
            task_end_op_finished=False,
            status=TaskStatus.READY,
            graph=batch_task_graph,
            return_mask=return_masks,
            callback=callbacks,
            op_callback_dict=op_callback_dict,
        )
        self.graph_to_task[batch_task_graph.graph_id] = batch_id
        for task_id in task_ids:
            self.graph_to_task.pop(self.tasks[task_id].graph.graph_id, None)
            self.tasks[task_id].batch_task_id = batch_id
        return batch_task_graph

    def launch_tasks(self,
                    task_ids: List[int],
                    slot_mappings: List[np.ndarray],
                    as_batch: bool = False,
                    batch_id: int = -1,
                    layerwise_transfer: bool = False,
                    counter_id: int = 0) -> List[int]:
        assert isinstance(slot_mappings[0], np.ndarray)
        # trace launch tasks
        self.tracer.trace_launch_tasks(task_ids, slot_mappings, as_batch)
        self.set_slot_mappings(task_ids, slot_mappings)

        # Batch optimization: collect all transfer graphs first
        nvtx_range = nvtx.start_range(message=f"KVTaskEngine.launch_tasks batch={len(task_ids)}", color="blue")

        all_get = all(self.tasks[tid].task_type == TaskType.GET for tid in task_ids)
        all_put = all(self.tasks[tid].task_type == TaskType.PUT for tid in task_ids)
        if (len(task_ids) > 1 or layerwise_transfer) and as_batch and (all_get or all_put):
            if batch_id == -1:
                batch_id = self._gen_task_id()
            if layerwise_transfer:
                if not GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer:
                    flexkv_logger.warning("layerwise transfer is not enabled")
                    layerwise_transfer = False
                elif not all_get:
                    flexkv_logger.warning("only support layerwise get")
                    layerwise_transfer = False
            batch_task_type = TaskType.BATCH_GET if all_get else TaskType.BATCH_PUT
            batch_task_graph = self.merge_to_batch_kvtask(
                batch_id, task_ids, batch_task_type, layerwise_transfer, counter_id
            )
            transfer_graphs = [batch_task_graph]
            self.tasks[batch_id].status = TaskStatus.RUNNING
            task_ids = [batch_id]
        else:
            transfer_graphs = []
            for task_id in task_ids:
                transfer_graph = self.check_task_ready(task_id)
                if transfer_graph is not None and transfer_graph.num_ops > 0:
                    transfer_graphs.append(transfer_graph)

        # Submit all graphs in batch to reduce IPC overhead
        if transfer_graphs:
            for transfer_handle in self.transfer_handles:
                transfer_handle.submit_batch(transfer_graphs)

        nvtx.end_range(nvtx_range)
        return task_ids

    def cancel_tasks(self, task_ids: Union[int, List[int]]) -> None:
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        for task_id in task_ids:
            self._cancel_task(task_id)

    def _clear_cpu_cache(self) -> None:
        self.cache_engine.cpu_cache_engine.reset()
