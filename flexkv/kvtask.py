import logging
import time
from typing import Dict, Optional, List, Union, Tuple
import threading
from enum import Enum
from dataclasses import dataclass, field, replace
from typing import Callable
import multiprocessing as mp
import copy
from expiring_dict import ExpiringDict
import nvtx
import numpy as np

from flexkv.common.config import CacheConfig, ModelConfig, GLOBAL_CONFIG_FROM_ENV
from flexkv.common.debug import flexkv_logger
from flexkv.common.block import hash_token
from flexkv.common.transfer import (
    CompletedOp,
    DeviceType,
    TransferOpGraph,
    TransferType,
    get_nvtx_default_color,
    invoke_op_callback,
    merge_to_batch_graph,
)
from flexkv.common.tracer import FlexKVTracer
from flexkv.cache.cache_engine import (
    GlobalCacheEngine,
    CacheStrategy,
    DEFAULT_CACHE_STRATEGY,
    CPUONLY_CACHE_STRATEGY,
)
from flexkv.transfer_manager import TransferManagerHandle, TransferManagerOnRemote
from flexkv.common.request import KVResponseStatus, KVResponse
from flexkv.cache.redis_meta import RedisMeta
from flexkv.integration.dynamo.collector import KVEventCollector
from flexkv.metrics.collector import get_global_collector
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
    transfer_failed: bool = False

    # SWA GPU slot_mapping (SWA-pool token index space), bound LATE at launch —
    # the SWA counterpart to slot_mapping. None when the request has no SWA ops.
    swa_slot_mapping: Optional[np.ndarray] = None
    created_ns: int = field(default_factory=time.perf_counter_ns)

    # True after wait()/try_wait() has produced a response for this task.
    request_returned: bool = False

    # Prefetch mooncake outcomes captured from CompletedOp (not from
    # task.graph ops — transfer engines may mutate a deepcopy).
    # Finalize return_mask (A) is the Full REMOTE2H success prefix clamped by
    # deferred publish; for joint SWA prefetch it is further gated so only
    # Full+SWA both succeeding reports a non-zero mask (else 0). SWA bitmaps
    # + ``prefetch_has_swa_remote`` also feed the joint outcome METRIC —
    # SWA lives in a separate slot space and is not summed into return_mask.
    prefetch_full_block_results: Optional[Tuple[bool, ...]] = None
    prefetch_swa_block_results: Optional[Tuple[bool, ...]] = None
    prefetch_has_swa_remote: bool = False
    prefetch_namespace: Optional[List[str]] = None
    prefetch_swa_aware: bool = False

    def is_completed(self) -> bool:
        return self.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED]

    def shed_heavy_resources(self) -> None:
        # Keep status and return_mask so a task whose response has not been
        # returned can still be observed by wait().
        self.graph = None
        self.token_ids = None
        self.slot_mapping = None
        self.token_mask = None
        self.callback = None

TASK_STATUS_TO_RESPONSE_STATUS = {
    TaskStatus.COMPLETED: KVResponseStatus.SUCCESS,
    TaskStatus.CANCELLED: KVResponseStatus.CANCELLED,
    TaskStatus.FAILED: KVResponseStatus.FAILED,
    TaskStatus.RUNNING: KVResponseStatus.SUCCESS, # for early return: still running, but success
}

def convert_to_response_status(task_status: TaskStatus) -> KVResponseStatus:
    return TASK_STATUS_TO_RESPONSE_STATUS[task_status]


def _longest_success_prefix(block_results: Tuple[bool, ...]) -> int:
    """Longest contiguous True prefix of per-block transfer results."""
    prefix = 0
    for succeeded in block_results:
        if not succeeded:
            break
        prefix += 1
    return prefix


class KVTaskManager:
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_register_port: Optional[str] = None,
                 redis_meta: RedisMeta = None,
                 event_collector: Optional[KVEventCollector] = None
                 ):
        if not cache_config.enable_cpu:
            raise ValueError("enable_cpu must be True")
        # Mooncake store is a remote backend that does not require local SSD.
        # Keep this aligned with CacheConfig validation in common/config.py.
        if (cache_config.enable_remote and not cache_config.enable_cpu):
            raise ValueError("enable_cpu must be True if enable_remote is True")
        if not cache_config.enable_cpu and not cache_config.enable_gds:
            raise ValueError("enable_gds must be True if enable_cpu is False")
        if cache_config.enable_gds and not cache_config.enable_ssd:
            raise ValueError("enable_ssd must be True if enable_gds is True")
        if cache_config.enable_kv_sharing and cache_config.enable_gds:
            raise ValueError("enable_kv_sharing and enable_gds cannot be used at the same time")
        if cache_config.enable_nixl and not cache_config.enable_gds:
            raise ValueError("enable_nixl requires enable_gds to be True")
        if cache_config.enable_nixl and model_config.effective_tp_size_per_node > 1:
            raise ValueError(
                "enable_nixl GPU-SSD path currently requires effective_tp_size_per_node==1 "
                "(no tpNixlTransferWorker)"
            )
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

        self.tasks: ExpiringDict[int, KVTask] = ExpiringDict(ttl=1800) # 30 minutes

        # hash(token_ids) -> task_id
        self.prefetch_tasks: ExpiringDict[int, int] = ExpiringDict(ttl=1800) # 30 minutes
        self._gen_prefetch_key = lambda token_ids, namespace: hash_token(token_ids, namespace)

        self.graph_to_task: Dict[int, int] = {}

        # (graph_id, op_id) -> completed_count; graph-keyed so a failed
        # graph's stale per-op counters can be purged.
        self.uncompleted_ops: Dict[Tuple[int, int], int] = {}
        self.uncompleted_op_results: Dict[Tuple[int, int], CompletedOp] = {}
        # graph_id -> (terminal_count, any_failed) across the N handles.
        self.uncompleted_graphs: Dict[int, Tuple[int, bool]] = {}
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

    @staticmethod
    def _operation_name(task_type: TaskType) -> str:
        if task_type in (TaskType.GET, TaskType.BATCH_GET):
            return "get"
        if task_type in (TaskType.PUT, TaskType.BATCH_PUT):
            return "put"
        return "prefetch"

    def _log_task_created(self, task: KVTask) -> None:
        if not flexkv_logger.is_enabled_for(logging.DEBUG):
            return
        graph_id = task.graph.graph_id if task.graph is not None else -1
        tokens = len(task.token_ids) if task.token_ids is not None else 0
        flexkv_logger.debug(
            "[FlexKV-IO] operation=%s act=create status=%s blocks=%d "
            "flexkv_task_id=%d graph_id=%d tokens=%d graph_ops=%d",
            self._operation_name(task.task_type),
            task.status.value,
            tokens // self.cache_config.tokens_per_block,
            task.task_id,
            graph_id,
            tokens,
            task.graph.num_ops if task.graph is not None else 0,
        )

    def _log_task_terminal(self, task: KVTask, status: TaskStatus) -> None:
        graph_ops = task.graph.num_ops if task.graph is not None else 0
        if status == TaskStatus.COMPLETED:
            level = logging.INFO if graph_ops else logging.DEBUG
        else:
            level = logging.WARNING
        if not flexkv_logger.is_enabled_for(level):
            return
        graph_id = task.graph.graph_id if task.graph is not None else -1
        duration_s = (time.perf_counter_ns() - task.created_ns) / 1e9
        if level == logging.INFO:
            log = flexkv_logger.info
        elif level == logging.DEBUG:
            log = flexkv_logger.debug
        else:
            log = flexkv_logger.warning
        log(
            "[FlexKV-IO] operation=%s act=complete status=%s "
            "flexkv_task_id=%d graph_id=%d task_time=%.4fs",
            self._operation_name(task.task_type),
            convert_to_response_status(status).value,
            task.task_id,
            graph_id,
            duration_s,
        )

    def create_get_task(self,
                        task_id: int,
                        token_ids: np.ndarray,
                        slot_mapping: np.ndarray,
                        dp_client_id: int,
                        token_mask: Optional[np.ndarray] = None,
                        is_fake_slot_mapping: bool = False,
                        temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
                        namespace: Optional[List[str]] = None,
                        swa_aware: bool = False,
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
            namespace=namespace,
            swa_aware=swa_aware)
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
        self._log_task_created(self.tasks[task_id])

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
        self._log_task_created(self.tasks[task_id])

    def create_prefetch_task(self,
                            task_id: int,
                            token_ids: np.ndarray,
                            dp_client_id: int,
                            namespace: Optional[List[str]] = None,
                            swa_aware: bool = False,
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
            namespace=namespace,
            swa_aware=swa_aware)
        prefetch_has_swa_remote = any(
            op.transfer_type == TransferType.REMOTE2H and getattr(op, "is_swa", False)
            for op in graph._op_map.values()
        )
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
            op_callback_dict=op_callback_dict,
            prefetch_has_swa_remote=prefetch_has_swa_remote,
            prefetch_namespace=namespace,
            prefetch_swa_aware=swa_aware)

        self.prefetch_tasks[self._gen_prefetch_key(token_ids, namespace)] = task_id

        self.graph_to_task[graph.graph_id] = task_id
        self._log_task_created(self.tasks[task_id])

    def _launch_task(self, task_id: int) -> None:
        transfer_graph = self.check_task_ready(task_id)
        if transfer_graph is None:
            return
        nvtx.mark(f"launch task: task_id={task_id}, graph_id={transfer_graph.graph_id}")
        if transfer_graph.num_ops > 0:
            for transfer_handle in self.transfer_handles:
                # For remote handles: deepcopy graph and clear GPU blocks when
                # it's a cross-machine PP handle (different PP stages have
                # different GPU block_ids).  Cross-machine TP handles share
                # the same slot_mapping, so no clear is needed.
                if isinstance(transfer_handle._handle, TransferManagerMultiNodeHandle):
                    if self.model_config.nnodes > 1 and self.model_config.pp_size > 1:
                        # Cross-machine PP: each PP rank has different GPU blocks
                        graph_copy = copy.deepcopy(transfer_graph)
                        graph_copy.clear_gpu_blocks()
                        transfer_handle.submit(graph_copy, task_end_op_id=self.tasks[task_id].task_end_op_id)
                    else:
                        # Cross-machine TP: same slot_mapping across TP ranks
                        transfer_handle.submit(transfer_graph, task_end_op_id=self.tasks[task_id].task_end_op_id)
                else:
                    transfer_handle.submit(transfer_graph, task_end_op_id=self.tasks[task_id].task_end_op_id)

    def _update_tasks(self, timeout: float = 0.001) -> None:
        completed_ops = self._get_completed_ops(timeout)
        metrics_collector = get_global_collector()
        for completed_op in completed_ops:
            if completed_op.graph_id not in self.graph_to_task:
                continue
            task_id = self.graph_to_task[completed_op.graph_id]
            task = self.tasks[task_id]
            if completed_op.is_graph_failed():
                self._fail_task(task_id)
                continue
            # A failed pull invalidates the current request and must trigger
            # fallback. Mooncake uploads retain the existing asynchronous PUT
            # completion contract (the task-end D2H may precede H2REMOTE).
            graph_op = getattr(task.graph, "_op_map", {}).get(
                completed_op.op_id)
            is_swa_op = graph_op is not None and getattr(graph_op, "is_swa", False)
            is_remote_load = (
                completed_op.transfer_type == TransferType.REMOTE2H.value
                or (graph_op is not None
                    and graph_op.transfer_type == TransferType.REMOTE2H)
            )
            expects_block_results = (
                graph_op is not None
                and (graph_op.mooncake_store_block_hashes is not None
                     or graph_op.mooncake_store_swa_block_hashes is not None)
            ) ## only mooncake store related ops expect block results now
            missing_results = (
                completed_op.block_results is None and expects_block_results)
            failed_blocks = (
                completed_op.block_results is not None
                and not all(completed_op.block_results)
            )
            # REMOTE2H policy + mooncake bitmap snapshot (CompletionOp itself;
            # do not rely on TransferOp.block_results on task.graph — engines
            # may mutate a submitted deepcopy).
            if is_remote_load:
                if is_swa_op:
                    # Joint prefetch: SWA partial/missing MUST NOT fail the task.
                    # Commit-time joint guard uses the bitmap with Full's mask.
                    if task.task_type == TaskType.PREFETCH:
                        task.prefetch_has_swa_remote = True
                        if completed_op.block_results is not None:
                            task.prefetch_swa_block_results = tuple(
                                bool(x) for x in completed_op.block_results)
                else:
                    if missing_results:
                        task.transfer_failed = True
                    elif failed_blocks:
                        # Prefetch: L==0 fails eagerly; L>0 waits for graph
                        # completion so joint SWA can still shape commit.
                        if task.task_type == TaskType.PREFETCH:
                            assert completed_op.block_results is not None
                            if _longest_success_prefix(
                                    tuple(completed_op.block_results)) == 0:
                                task.transfer_failed = True
                        else:
                            task.transfer_failed = True
                    if (task.task_type == TaskType.PREFETCH
                            and completed_op.block_results is not None):
                        task.prefetch_full_block_results = tuple(
                            bool(x) for x in completed_op.block_results)
            # Record transfer metrics for completed ops (post-completion statistics)
            # All three counters (ops_total, blocks_total, bytes_total) are updated
            # here after transfer completion, providing accurate post-transfer metrics.
            if metrics_collector is not None and completed_op.transfer_type is not None:
                if task.task_type in (TaskType.GET, TaskType.PREFETCH, TaskType.BATCH_GET):
                    operation = "get"
                elif task.task_type == TaskType.PUT:
                    operation = "put"
                else:
                    operation = "unknown"
                metrics_collector.record_transfer_completed(
                    completed_op.transfer_type,
                    completed_op.num_blocks,
                    completed_op.num_bytes,
                    operation,
                )
            if task.status == TaskStatus.CANCELLED and task.callback is None:
                # Cache was reset while this task was in flight: reset_cache()
                # cleared its callbacks and freed the radix nodes / mempool blocks
                flexkv_logger.warning(
                    f"task {task_id}: transfer op {completed_op.op_id} completed "
                    "after reset_cache(), callback no longer exists and will be skipped."
                )
                continue
            has_callback = completed_op.op_id in task.op_callback_dict
            if has_callback:
                try:
                    invoke_op_callback(
                        task.op_callback_dict[completed_op.op_id], completed_op)
                except Exception:
                    task.transfer_failed = True
                    flexkv_logger.error(
                        "Transfer op callback failed: "
                        f"graph_id={completed_op.graph_id}, "
                        f"op_id={completed_op.op_id}",
                        exc_info=True,
                    )
            if completed_op.is_graph_completed():
                # _mark_completed runs deferred commit first, then (for
                # prefetch) finalizes return_mask from the Full REMOTE2H
                # success bitmap — report "how much remote this task pulled".
                self._mark_completed(task_id)
            elif completed_op.op_id == task.task_end_op_id:
                self.tasks[task_id].task_end_op_finished = True

    def _narrow_return_mask_to_prefix_blocks(
            self, task: "KVTask", num_success_blocks: int) -> None:
        """Rewrite prefetch return_mask to the first ``num_success_blocks``.

        Prefetch builds a contiguous True span for planned REMOTE2H blocks.
        After partial mooncake success, only ``[:L]`` is published/usable.
        """
        mask = task.return_mask
        if mask is None or isinstance(mask, list):
            return
        if num_success_blocks <= 0:
            task.return_mask = np.zeros_like(mask, dtype=np.bool_)
            return
        true_idx = np.flatnonzero(mask)
        if true_idx.size == 0:
            return
        start = int(true_idx[0])
        orig_end = int(true_idx[-1]) + 1
        tpb = self.cache_config.tokens_per_block
        end = min(start + num_success_blocks * tpb, orig_end, mask.shape[0])
        new_mask = np.zeros_like(mask, dtype=np.bool_)
        new_mask[start:end] = True
        task.return_mask = new_mask

    @staticmethod
    def _prefetch_published_remote_blocks(task: "KVTask") -> Optional[int]:
        """CPU deferred-publish remote block count after graph callback.

        Returns ``None`` when this task has no deferred-publish tracker
        (non-mooncake / legacy path). Returns ``0`` when commit discarded
        or failed to mount anything matchable.
        """
        callbacks = task.callback
        if callbacks is None:
            return None
        if not isinstance(callbacks, list):
            callbacks = [callbacks]
        saw_tracker = False
        published: Optional[int] = None
        for callback in callbacks:
            keywords = getattr(callback, "keywords", None) or {}
            for pending in keywords.get("deferred_inserts") or []:
                if getattr(pending, "device_type", None) != DeviceType.CPU:
                    continue
                publish_result = getattr(pending, "publish_result", None)
                if publish_result is None:
                    continue
                saw_tracker = True
                blocks = publish_result.published_remote_blocks
                if blocks is None or publish_result.failed:
                    blocks = 0
                published = (
                    int(blocks) if published is None
                    else min(published, int(blocks)))
        if not saw_tracker:
            return None
        return 0 if published is None else published

    def _finalize_prefetch_return_mask(self, task: "KVTask") -> None:
        """Report reusable Full REMOTE2H tokens for this prefetch (A).

        ``sum(return_mask)`` is the storage/L3 accounting number consumed by
        sglang as ``storage_hit_length``. It must NOT include a pre-existing
        CPU prefix (f1) or DISK2H.

        Base length is the longest success prefix of this task's Full mooncake
        REMOTE2H, clamped by CPU deferred-publish length so transfer-success /
        commit-discard cannot over-report.

        Joint Full+SWA prefetch (``prefetch_has_swa_remote``): subsequent
        ``swa_aware`` GET uses ``min(full, swa)`` and commit only mounts SWA
        when Full covers the whole planned span. Reporting therefore requires
        **both** Full transfer/publish complete (L == planned) **and** SWA
        transfer success; otherwise the mask is cleared to 0 even if Full
        alone was mounted. SWA tokens are never summed into the mask.

        Opaque backends without ``block_results`` leave the plan-time mask
        unchanged when there is no SWA remote op.
        """
        full_results = task.prefetch_full_block_results
        if full_results is not None:
            mounted_full_len = _longest_success_prefix(full_results)
        else:
            mounted_full_len = None

        published_remote = self._prefetch_published_remote_blocks(task)
        if published_remote is not None:
            if mounted_full_len is None:
                mounted_full_len = published_remote
            else:
                mounted_full_len = min(mounted_full_len, published_remote)

        # Length reported to callers (may be zeroed by the joint SWA gate).
        report_len = mounted_full_len
        if task.prefetch_has_swa_remote:
            swa_results = task.prefetch_swa_block_results
            swa_ok = (
                swa_results is not None
                and len(swa_results) > 0
                and all(swa_results)
            )
            if full_results is None:
                # Joint path without Full bitmaps cannot prove both succeeded.
                report_len = 0
            else:
                planned_full_len = len(full_results)
                if (report_len is None
                        or report_len != planned_full_len
                        or not swa_ok):
                    report_len = 0

        if report_len is not None:
            self._narrow_return_mask_to_prefix_blocks(task, report_len)

        # Joint / full-only outcome metric (mooncake bitmaps).
        # Classified from mounted Full length + SWA bitmap — not from the
        # caller-facing report_len gate — so full_only_swa_lost remains visible.
        if full_results is None:
            return

        assert mounted_full_len is not None
        if not task.prefetch_has_swa_remote:
            outcome = "full_only"
        else:
            planned_full_len = len(full_results)
            swa_results = task.prefetch_swa_block_results
            swa_ok = (
                swa_results is not None
                and len(swa_results) > 0
                and all(swa_results)
            )
            if mounted_full_len == 0:
                outcome = "all_failed"
            elif mounted_full_len == planned_full_len and swa_ok:
                outcome = "full_and_swa"
            elif mounted_full_len == planned_full_len and not swa_ok:
                outcome = "full_only_swa_lost"
            else:
                outcome = "partial_full"
        metrics_collector = get_global_collector()
        if metrics_collector is not None:
            metrics_collector.record_joint_prefetch_outcome(outcome)

    @staticmethod
    def _abort_task_plans(task: "KVTask") -> None:
        """Run the abort path of every plan handle the task carries (a batch
        task carries one per merged sub-task). Handles predating the abort API
        are skipped."""
        callbacks = task.callback if isinstance(task.callback, list) \
            else [task.callback]
        for callback in callbacks:
            abort = getattr(callback, "abort", None)
            if abort is not None:
                abort()

    def _fail_task(self, task_id: int) -> None:
        """A transfer op of this task's graph failed and the graph has fully
        drained. Roll the plan back instead of completing it: ops that did
        finish already ran their callbacks (their nodes are ready and their
        data is valid, so abort keeps them), while nodes whose transfer never
        ran are still unready and get removed with their blocks recycled.
        The task terminates as FAILED so wait() reports the failure instead
        of a misleading TIMEOUT."""
        if task_id not in self.tasks:
            return
        task = self.tasks[task_id]
        if task.is_completed():
            return
        flexkv_logger.error(f"[KVTaskEngine] task {task_id} FAILED: a transfer "
                            f"op of graph {task.graph.graph_id} failed")
        self._abort_task_plans(task)
        task.status = TaskStatus.FAILED
        task.task_end_op_finished = True
        self.graph_to_task.pop(task.graph.graph_id, None)
        task.shed_heavy_resources()
        if task.request_returned:
            self._release_task(task_id)

    def _cancel_task(self, task_id: int) -> None:
        if task_id not in self.tasks:
            return
        task = self.tasks[task_id]
        if not task.is_completed():
            # A task whose graph never launched still holds everything its
            # plan acquired at create time: locked radix nodes, CPU staging
            # blocks, and is_ready=False index nodes that only a completion
            # callback could publish. Dropping the task without aborting leaks
            # all of it -- the staging blocks become unreachable (mempool
            # exhaustion) and the unready nodes are permanently unevictable
            # holes that also shadow future puts of the same prefix. Abort
            # rolls those back; RUNNING tasks keep the old behavior (their
            # graph is in flight and completion callbacks will still fire).
            if task.status in (TaskStatus.UNREADY, TaskStatus.READY):
                self._abort_task_plans(task)
            task.status = TaskStatus.CANCELLED
            self._log_task_terminal(task, TaskStatus.CANCELLED)
        self._release_task(task_id)

    def check_completed(self, task_id: int, completely: bool = False) -> bool:
        task = self.tasks[task_id]
        self._process_empty_graph(task_id)
        # Prefetch must wait for the graph terminal only. Joint Full+SWA graphs
        # may mark an early task_end (e.g. SWA REMOTE2H) while Full is still
        # in flight; SUCCESS before _finalize_prefetch_return_mask /
        # deferred commit would advertise a planned mask that is not yet ready.
        if task.task_type == TaskType.PREFETCH:
            completely = True
        if completely:
            return task.is_completed()
        # A partial-capable backend may finish the data-path sink after already
        # reporting failed blocks. Wait for graph completion so cleanup runs and
        # the caller observes FAILED instead of an early RUNNING-as-success result.
        if task.transfer_failed:
            return task.is_completed()
        # For tasks with callback (e.g., PUT tasks that need to call insert_and_publish),
        # we must wait until _mark_completed is called (i.e., is_completed() returns True)
        # to ensure the callback is executed before returning success.
        #if task.callback is not None:
        #    return task.is_completed()
        return task.is_completed() or task.task_end_op_finished

    def set_slot_mappings(self,
                          task_ids: List[int],
                          slot_mappings: List[np.ndarray],
                          swa_slot_mappings: Optional[List[Optional[np.ndarray]]] = None) -> None:
        if swa_slot_mappings is None:
            swa_slot_mappings = [None] * len(task_ids)
        for task_id, slot_mapping, swa_slot_mapping in zip(task_ids, slot_mappings, swa_slot_mappings):
            self._set_slot_mapping_impl(task_id, slot_mapping, swa_slot_mapping)

    def _set_slot_mapping_impl(self,
                               task_id: int,
                               slot_mapping: np.ndarray,
                               swa_slot_mapping: Optional[np.ndarray] = None) -> None:
        task = self.tasks[task_id]
        if task.status != TaskStatus.UNREADY:
            return
        graph_ids = self.cache_engine.slot_mapping_to_block_ids(slot_mapping,
                                                                self.cache_config.tokens_per_block)
        # Late-bind the GPU-side SWA slots via the unified set_gpu_blocks(gpu,
        # swa_gpu) path (PR#191). SWA is page-granular, so the mapping folds by
        # the same stride as full-KV (slot_mapping_to_block_ids).
        # A None swa_slot_mapping leaves the graph's SWA ops at their built ids.
        swa_sm = swa_slot_mapping if swa_slot_mapping is not None else task.swa_slot_mapping
        swa_graph_ids = None
        if swa_sm is not None:
            swa_graph_ids = self.cache_engine.slot_mapping_to_block_ids(
                swa_sm, self.cache_config.tokens_per_block)
        task.graph.set_gpu_blocks(graph_ids, swa_graph_ids)
        task.slot_mapping = slot_mapping
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
        if flexkv_logger.is_enabled_for(logging.DEBUG):
            flexkv_logger.debug(
                "[FlexKV-IO] operation=%s act=launch status=running "
                "flexkv_task_id=%d graph_id=%d",
                self._operation_name(task.task_type),
                task.task_id,
                task.graph.graph_id,
            )
        return task.graph

    def _release_task(self, task_id: int) -> None:
        if task_id not in self.tasks:
            return
        task = self.tasks[task_id]
        if task.graph is not None:
            self.graph_to_task.pop(task.graph.graph_id, None)
        self.tasks.pop(task_id, None)

    def _mark_completed(self, task_id: int) -> None:
        task = self.tasks[task_id]
        if task.is_completed():
            return
        if task.callback:
            callbacks = (
                task.callback if isinstance(task.callback, list)
                else [task.callback]
            )
            for callback in callbacks:
                try:
                    callback()
                except Exception:
                    task.transfer_failed = True
                    flexkv_logger.error(
                        f"Transfer graph callback failed for task_id={task_id}",
                        exc_info=True,
                    )
        # Deferred commit (tree mount) has just run. Finalize return_mask from
        # the Full REMOTE2H success bitmap (how much remote this task pulled).
        # On fail we still finalize so outcome metrics (e.g. all_failed) land.
        if task.task_type == TaskType.PREFETCH:
            self._finalize_prefetch_return_mask(task)
        task.status = (
            TaskStatus.FAILED if task.transfer_failed else TaskStatus.COMPLETED)
        task.task_end_op_finished = True
        self._log_task_terminal(task, TaskStatus.COMPLETED)
        self.graph_to_task.pop(task.graph.graph_id, None)
        task.shed_heavy_resources()
        if task.request_returned:
            self._release_task(task_id)

    def _process_empty_graph(self, task_id: int) -> None:
        task = self.tasks[task_id]
        if task.graph is None:
            return
        if task.graph.num_ops == 0:
            self._mark_completed(task_id)

    def _get_completed_ops(self, timeout: Optional[float] = None) -> List[CompletedOp]:
        results = []
        # Keep lightweight test/fallback managers created with ``__new__``
        # compatible with the pre-bitmap state shape.
        if not hasattr(self, "uncompleted_op_results"):
            self.uncompleted_op_results = {}
        for transfer_handle in self.transfer_handles:
            completed_ops = transfer_handle.wait(timeout)
            for completed_op in completed_ops:
                if completed_op.op_id == -1:
                    # Graph-level terminal message, completed OR failed. Every
                    # handle received the same graph, so the task terminates
                    # only after all of them have reported a terminal state --
                    # aborting on the first failure would recycle plan blocks
                    # a sibling engine is still writing into. Any failure
                    # among the N outcomes fails the graph.
                    graph_id = completed_op.graph_id
                    count, failed = self.uncompleted_graphs.get(graph_id, (0, False))
                    count += 1
                    failed = failed or completed_op.is_graph_failed()
                    if count == self.required_completed_count:
                        self.uncompleted_graphs.pop(graph_id, None)
                        if failed:
                            # A failed handle never finalizes some of the
                            # graph's ops, so their N-way per-op counters can
                            # never complete: purge them rather than leak.
                            # Safe because each handle's terminal message
                            # follows all its per-op messages (per-handle
                            # FIFO), so nothing can arrive for this graph
                            # after the Nth terminal and resurrect a counter.
                            stale = [key for key in self.uncompleted_ops
                                     if key[0] == graph_id]
                            for key in stale:
                                self.uncompleted_ops.pop(key, None)
                                self.uncompleted_op_results.pop(key, None)
                            results.append(CompletedOp.failed_graph(graph_id))
                        else:
                            results.append(completed_op)
                    else:
                        self.uncompleted_graphs[graph_id] = (count, failed)
                else:
                    op_key = (completed_op.graph_id, completed_op.op_id)
                    completed_count = self.uncompleted_ops.get(op_key, 0) + 1
                    aggregate = self._merge_completed_op(
                        self.uncompleted_op_results.get(op_key),
                        completed_op,
                    )
                    if completed_count == self.required_completed_count:
                        results.append(aggregate)
                        self.uncompleted_ops.pop(op_key, None)
                        self.uncompleted_op_results.pop(op_key, None)
                    else:
                        self.uncompleted_ops[op_key] = completed_count
                        self.uncompleted_op_results[op_key] = aggregate
        return results

    @staticmethod
    def _merge_completed_op(
        current: Optional[CompletedOp],
        incoming: CompletedOp,
    ) -> CompletedOp:
        """Combine multi-handle outcomes; a block succeeds only everywhere."""
        if current is None:
            if incoming.block_results is None:
                return incoming
            expected_blocks = incoming.num_blocks or len(incoming.block_results)
            block_results = (
                tuple(bool(result) for result in incoming.block_results)
                if len(incoming.block_results) == expected_blocks
                else (False,) * expected_blocks
            )
            return replace(
                incoming,
                num_blocks=expected_blocks,
                block_results=block_results,
            )

        block_results = None
        if (current.block_results is not None
                or incoming.block_results is not None):
            # Once one handle reports a bitmap, every handle must report the
            # same width. Missing or malformed data is a fail-closed result.
            expected_blocks = max(
                current.num_blocks,
                incoming.num_blocks,
                len(current.block_results or ()),
                len(incoming.block_results or ()),
            )

            def normalize(results: Optional[Tuple[bool, ...]]) -> Tuple[bool, ...]:
                if results is None or len(results) != expected_blocks:
                    return (False,) * expected_blocks
                return tuple(bool(result) for result in results)

            left = normalize(current.block_results)
            right = normalize(incoming.block_results)
            block_results = tuple(a and b for a, b in zip(left, right))
        return replace(
            current,
            transfer_type=current.transfer_type or incoming.transfer_type,
            num_blocks=max(
                current.num_blocks,
                incoming.num_blocks,
                len(block_results or ()),
            ),
            num_bytes=max(current.num_bytes, incoming.num_bytes),
            block_results=block_results,
        )

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
                    task = self.tasks[task_id]
                    return_responses[task_id] = KVResponse(
                        status=convert_to_response_status(task.status),
                        task_id=task_id,
                        return_mask=task.return_mask
                    )
                    task.request_returned = True
                    if task.is_completed():
                        self._release_task(task_id)
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
                  namespace: Optional[List[str]] = None,
                  swa_aware: bool = False) -> Tuple[int, np.ndarray]:
        """Match a prefix and build the load graph; return (task_id, return_mask).

        With ``swa_aware=True`` the Full-KV transfer is clamped to the reusable
        SWA window (``usable = min(full_hit, swa_hit)``) from the same single radix
        match: past that window the Full-KV bytes would feed stale KV to the
        SWA-layer attention. The SWA window is the trailing block of the returned
        mask (page-granular), which the caller reads directly — there is no
        separate SWA mask. ``swa_aware=False`` (default) is the plain path,
        untouched.
        """
        nvtx.push_range(f"get match: task_id={task_id}", color=get_nvtx_default_color())
        # self._sync_prefetch(token_ids, namespace)
        # Flush pending D2H completions so set_ready callbacks run before
        # we check the radix tree.  Without this, blocks offloaded between
        # scheduler steps remain "not ready" until the next try_wait call,
        # which comes too late (after get_match).
        self._update_tasks(timeout=0)
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
                                                           namespace=namespace,
                                                           swa_aware=swa_aware)
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
                  namespace: Optional[List[str]] = None,
                  swa_aware: bool = False) -> Tuple[int, np.ndarray]:
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
                             namespace=namespace,
                             swa_aware=swa_aware)
        self._process_empty_graph(task_id)
        nvtx.pop_range()
        return task_id, self.tasks[task_id].return_mask

    def put_match(self,
                  token_ids: np.ndarray,
                  dp_client_id: int = 0,
                  token_mask: Optional[np.ndarray] = None,
                  task_id: int = -1,
                  namespace: Optional[List[str]] = None) -> Tuple[int, np.ndarray]:
        self._update_tasks(timeout=0)
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
                       namespace: Optional[List[str]] = None,
                       swa_aware: bool = False) -> int:
        """Launch a prefetch task; return its task_id.

        The launch call is fire-and-forget: it publishes the plan and hands the
        graph to the transfer engine. Progress and usable-token accounting are
        polled later via ``try_wait``/``wait`` — ``KVResponse.return_mask`` is
        rewritten in ``_finalize_prefetch_return_mask`` to the reusable Full
        REMOTE2H length (clamped by deferred publish; for joint SWA only when
        Full+SWA both succeed, else 0). Callers that want that count should
        read ``sum(return_mask)`` on the response, not any launch-time value.
        Compute H2D length still comes from a subsequent local ``get_match``
        against the CPU tree.
        """
        if task_id == -1:
            task_id = self._gen_task_id()
        nvtx.push_range(f"prefetch match: task_id={task_id}", color=get_nvtx_default_color())
        self.create_prefetch_task(task_id, token_ids, dp_client_id=dp_client_id, namespace=namespace, swa_aware=swa_aware)
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
        # When layerwise is on, SWA (+ optional C4 state sidecars) always folds
        # into the fused LAYERWISE op via launch_swa_h2d_layer_ /
        # launch_swa_mg_h2d_layer_.
        batch_task_graph, task_end_op_id, op_callback_dict = merge_to_batch_graph(
            batch_id,
            transfer_graphs,
            task_end_op_ids,
            op_callback_dict,
            layerwise_transfer,
            counter_id,
        )
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
        if flexkv_logger.is_enabled_for(logging.INFO):
            operation = self._operation_name(batch_task_type)
            flexkv_logger.info(
                "[FlexKV-IO] operation=%s act=merge status=ready direction=%s "
                "child_task_ids=%s flexkv_batch_task_id=%d graph_id=%d "
                "mode=%s graph_ops=%d",
                operation,
                "H2D" if operation == "get" else "D2H",
                ",".join(str(task_id) for task_id in task_ids),
                batch_id,
                batch_task_graph.graph_id,
                "layerwise" if layerwise_transfer else "no-layerwise",
                batch_task_graph.num_ops,
            )
        for task_id in task_ids:
            child_task = self.tasks[task_id]
            if child_task.graph is not None:
                self.graph_to_task.pop(child_task.graph.graph_id, None)
            self.tasks.pop(task_id, None)
        return batch_task_graph

    def launch_tasks(self,
                    task_ids: List[int],
                    slot_mappings: List[np.ndarray],
                    swa_slot_mappings: Optional[List[Optional[np.ndarray]]] = None,
                    as_batch: bool = False,
                    batch_id: int = -1,
                    layerwise_transfer: bool = False,
                    counter_id: int = 0) -> List[int]:
        assert isinstance(slot_mappings[0], np.ndarray)
        # trace launch tasks
        self.tracer.trace_launch_tasks(task_ids, slot_mappings, as_batch)
        self.set_slot_mappings(task_ids, slot_mappings, swa_slot_mappings)

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

    def reset_cache(self) -> None:
        """Invalidate the cache across ALL tiers (CPU + SSD + remote): drop the
        whole prefix tree and return every block to the mempool.

        Used after a weight update (e.g. verl RL rollout) so that KV computed
        against stale weights is never reused.

        We do NOT drain or cancel in-flight transfers here. verl issues the
        reset at a rollout/weight-update boundary where no new generation
        requests are being served, so in practice no task is ongoing. If any
        task IS still in flight we only warn: resetting the radix tree +
        mempool is cheap and the stale-weight invalidation must not be blocked
        on transfer completion. This mirrors vLLM's own reset_encoder_cache /
        reset_mm_cache, which likewise only warn on has_unfinished_requests().
        """
        ongoing = sum(1 for t in list(self.tasks.values()) if not t.is_completed())
        if ongoing:
            flexkv_logger.warning(
                f"reset_cache called while {ongoing} task(s) are still in flight; "
                f"resetting anyway. In-flight transfers may target blocks that are "
                f"being freed — ensure reset is issued at a quiesced boundary."
            )

        # Invalidate in-flight tasks' callbacks BEFORE dropping the cache. The
        # callback / op_callback_dict partials close over the exact radix nodes
        # and mempool blocks we are about to free; if a late transfer fired them
        # after reset they would unlock/set_ready a deleted node, or recycle a
        # block into the freshly-emptied mempool (which raises "already free").
        # Clear them here so the dispatch in _update_tasks has nothing to fire.
        # Note: reset_cache() runs on the same thread as the callback dispatch
        # (_update_tasks), so no lock is needed. We keep the graph_to_task
        # mapping so a late-completing op still resolves to its task and warns.
        for task_id, task in list(self.tasks.items()):
            if task.is_completed():
                continue  # already-fired callbacks are harmless
            task.callback = None
            task.op_callback_dict = {}
            task.status = TaskStatus.CANCELLED

        # Drop index (radix tree) + mempool on every tier. CRadixTreeIndex (C++)
        # and the pure-Python RadixTreeIndex both expose reset(); GlobalCacheEngine
        # fans out to whichever tier engines are enabled.
        self.cache_engine.reset()  # GlobalCacheEngine.reset()
