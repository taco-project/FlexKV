import time
import heapq
from typing import Dict, Optional, List, Union, Tuple
import threading
from enum import Enum
from dataclasses import dataclass, field
from typing import Callable
from queue import Queue, Empty
import multiprocessing as mp
import copy
import os
from expiring_dict import ExpiringDict
import nvtx
import numpy as np

from flexkv.common.config import CacheConfig, ModelConfig, GLOBAL_CONFIG_FROM_ENV
from flexkv.common.debug import flexkv_logger
from flexkv.common.block import hash_token
from flexkv.common.transfer import TransferOpGraph, TransferOp, TransferType, merge_to_batch_graph, get_nvtx_default_color, CompletedOp, DeviceType
from flexkv.common.tracer import FlexKVTracer
from flexkv.cache.cache_engine import GlobalCacheEngine, DEFAULT_CACHE_STRATEGY, CPUONLY_CACHE_STRATEGY
from flexkv.transfer_manager import TransferManagerHandle, TransferManagerOnRemote
from flexkv.common.request import KVResponseStatus, KVResponse
from flexkv.transfer_manager import (
    resolve_master_host_and_ports,
    get_trtllm_subprocess_host_and_ports_from_env
)
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

    dp_rank: int = 0
    pp_rank: int = 0

    # batch: points to the batch task id if this task was merged into a batch
    batch_task_id: Optional[int] = None

    # ---- Prefetch-specific fields ----
    # Token count reserved in _prefetch_tokens_occupied (for PREFETCH tasks only).
    # Used by _mark_completed to decrement the exact amount that was added,
    # independent of any later modifications to return_mask.
    prefetch_tokens_reserved: int = 0

    # ---- Resource references (for structured cleanup) ----
    # Extracted from callback's partial.keywords at task creation time.
    # Used by complete_task() / abort_task() instead of hacking partial internals.
    node_to_unlock: Optional[Dict] = None     # {DeviceType: (node, ready_len)}
    buffer_to_free: Optional[Dict] = None     # {DeviceType: np.ndarray}
    is_put: bool = False

    # ---- Deferred insert info (prefetch only) ----
    # When prefetch mode skips immediate insert in _get_impl_local, this dict
    # holds the info needed to do insert(is_ready=True) after transfer completes.
    # Keys: 'sequence_meta', 'num_insert_blocks', 'cpu_block_ids'
    deferred_insert_info: Optional[Dict] = None

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


@dataclass
class PrefetchBatchExecutor:
    """Manages batched execution of a large prefetch DISK2H transfer.

    Instead of submitting the entire DISK2H transfer as one monolithic graph
    (which blocks until the C++ io_uring call completes), this executor splits
    it into multiple mini-graphs of ``batch_size_blocks`` each.  Between
    batches, a termination flag is checked so that the prefetch can be
    interrupted mid-flight, releasing CPU blocks for the untransferred portion.

    Lifecycle:
        1. ``create_prefetch_task()`` runs normally → full graph/callback/node_to_unlock.
        2. ``_create_batch_executor()`` extracts the DISK2H op's src/dst blocks.
        3. The original graph is NOT submitted.  Instead, the executor thread
           creates and submits one mini-graph per batch.
        4. After each batch completes, the executor checks ``_terminated``.
        5. On completion or termination, ``_finalize_batch_executor()`` cleans up.

    Thread-safety:
        ``mark_terminate()`` / ``is_terminated()`` are safe to call from any
        thread (protected by ``_lock``).  ``wait_current_batch()`` /
        ``notify_batch_done()`` use a ``threading.Event``.
    """
    task_id: int

    # ---- Blocks extracted from the original DISK2H op ----
    ssd_block_ids: np.ndarray       # all SSD source blocks
    cpu_block_ids: np.ndarray       # all CPU destination blocks (already allocated)

    # ---- Batch configuration ----
    batch_size_blocks: int = 32                     # blocks per mini-graph

    # ---- Execution state ----
    total_blocks: int = 0
    completed_blocks: int = 0
    completed_batches: int = 0

    # ---- Termination control (thread-safe) ----
    _terminated: bool = field(default=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ---- Batch synchronization ----
    # The executor thread waits on this event; the main thread's _update_tasks
    # sets it when the current mini-graph's CompletedOp arrives.
    _batch_done_event: threading.Event = field(default_factory=threading.Event, repr=False)

    # ---- Current mini-graph tracking ----
    _current_mini_graph_id: Optional[int] = field(default=None, repr=False)
    # Flag: set by notify_batch_done to indicate IO completion (vs terminate wakeup)
    _io_done: bool = field(default=False, repr=False)

    # ------------------------------------------------------------------
    # Termination API
    # ------------------------------------------------------------------

    def mark_terminate(self) -> None:
        """Request termination.  Thread-safe; may be called from scheduler thread."""
        with self._lock:
            self._terminated = True
        # Wake up executor thread if it's waiting for a batch
        self._batch_done_event.set()

    def is_terminated(self) -> bool:
        """Check termination flag.  Lock-free read (bool is atomic in CPython)."""
        return self._terminated

    # ------------------------------------------------------------------
    # Batch synchronization API
    # ------------------------------------------------------------------

    def wait_current_batch(self, timeout: float) -> bool:
        """Block until the current mini-graph completes or *timeout* elapses.

        Called by the **executor thread**.
        Returns ``True`` if the event was set (batch done), ``False`` on timeout.
        """
        return self._batch_done_event.wait(timeout=timeout)

    def notify_batch_done(self) -> None:
        """Signal that the current mini-graph has completed.

        Called by the **main thread** from ``_update_tasks``.
        """
        self._io_done = True
        self._batch_done_event.set()

    def reset_batch_event(self) -> None:
        """Prepare for the next batch.  Called by executor thread before submit."""
        self._io_done = False
        self._batch_done_event.clear()

    # ------------------------------------------------------------------
    # Computed properties
    # ------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        return self.completed_blocks >= self.total_blocks

    @property
    def next_batch_start(self) -> int:
        return self.completed_blocks

    @property
    def next_batch_end(self) -> int:
        return min(self.completed_blocks + self.batch_size_blocks, self.total_blocks)

    @property
    def remaining_blocks(self) -> int:
        return max(0, self.total_blocks - self.completed_blocks)

    @property
    def progress_ratio(self) -> float:
        if self.total_blocks == 0:
            return 1.0
        return self.completed_blocks / self.total_blocks


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
        if cache_config.enable_remote and not cache_config.enable_ssd:
            raise ValueError("enable_ssd must be True if enable_remote is True")
        if not cache_config.enable_cpu and not cache_config.enable_gds:
            raise ValueError("enable_gds must be True if enable_cpu is False")
        if cache_config.enable_gds and not cache_config.enable_ssd:
            raise ValueError("enable_ssd must be True if enable_gds is True")
        if cache_config.enable_kv_sharing and cache_config.enable_gds:
            raise ValueError("enable_kv_sharing and enable_gds cannot be used at the same time")
        self.cache_config = cache_config
        self.model_config = model_config
        self._check_config(model_config, cache_config)

        # ---- Multi-node topology ----
        nnodes = self.model_config.nnodes
        pp_size = self.model_config.pp_size
        tp_size = self.model_config.tp_size

        total_gpus = tp_size * pp_size
        if total_gpus % nnodes != 0:
            raise ValueError(
                f"[KVTaskEngine] cannot derive gpus_per_node: "
                f"tp*pp={total_gpus} not divisible by nnodes={nnodes}"
            )
        gpus_per_node = total_gpus // nnodes

        self.nnodes_per_tp_group = max(
            (tp_size + gpus_per_node - 1) // gpus_per_node, 1
        )
        if self.nnodes_per_tp_group > 2:
            raise ValueError(
                f"Only support 2-nodes TP for now, but got "
                f"nnodes_per_tp_group={self.nnodes_per_tp_group} "
                f"(tp_size={tp_size}, gpus_per_node={gpus_per_node})"
            )

        if tp_size % self.nnodes_per_tp_group != 0:
            raise ValueError(
                f"[KVTaskEngine] tp_size={tp_size} not divisible by "
                f"nnodes_per_tp_group={self.nnodes_per_tp_group}"
            )
        tp_size_per_node = tp_size // self.nnodes_per_tp_group

        flexkv_logger.info(
            f"[KVTaskEngine] topology: "
            f"nnodes={nnodes}, "
            f"node_rank={self.model_config.node_rank}, "
            f"gpus_per_node={gpus_per_node}, "
            f"tp_size={tp_size}, "
            f"pp_size={pp_size}, "
            f"dp_size={self.model_config.dp_size}, "
            f"nnodes_per_tp_group={self.nnodes_per_tp_group}, "
            f"tp_size_per_node={tp_size_per_node}, "
            f"master_host={self.model_config.master_host!r}"
        )

        self.cache_engine = GlobalCacheEngine(cache_config, model_config, redis_meta, event_collector)

        combine_with_trtllm = os.getenv("FLEXKV_WITH_TRTLLM", "0") == "1"
        if not combine_with_trtllm:
            self.transfer_handles = [TransferManagerHandle(
                self.model_config,
                self.cache_config,
                mode="process",
                gpu_register_port=gpu_register_port
            )]
        else:
            # When using FlexKV with TensorRT-LLM, we use remote mode to transfer data
            #  to avoid the way we launch subprocess in FlexKV
            #  conflict with TensorRT-LLM's MPI initialization
            master_host, master_ports = get_trtllm_subprocess_host_and_ports_from_env()
            self.remote_process = TransferManagerOnRemote.create_process(mode="TrtllmSubprocess")
            self.transfer_handles = [
                TransferManagerHandle(
                    self.model_config,
                    self.cache_config,
                    mode="remote",
                    gpu_register_port=gpu_register_port,
                    master_host=master_host,
                    master_ports=master_ports
                )
            ]
            self.transfer_handles[0]._handle.send_config_to_remotes()

        if self.model_config.nnodes > 1:
            master_host, master_ports = resolve_master_host_and_ports(
                master_host=self.model_config.master_host
            )
            self.transfer_handles.append(TransferManagerHandle(
                self.model_config,
                self.cache_config,
                mode="remote",
                gpu_register_port=gpu_register_port,
                master_host=master_host,
                master_ports=master_ports
            ))
            self.transfer_handles[-1]._handle.send_config_to_remotes()

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

        self._prefetch_token_threshold: int = GLOBAL_CONFIG_FROM_ENV.prefetch_token_threshold
        # Token-level flow control for prefetch.
        # Capacity = cpu_total_tokens × ratio.  Prefetch tasks whose cumulative
        # actual SSD→CPU token count would exceed this are discarded before enqueue.
        if self.cache_engine.cpu_cache_engine is not None:
            cpu_total_tokens = (
                self.cache_engine.cpu_cache_engine.num_total_blocks
                * self.cache_config.tokens_per_block
            )
            ratio = GLOBAL_CONFIG_FROM_ENV.prefetch_capacity_ratio
            self._prefetch_token_capacity = max(0, int(cpu_total_tokens * ratio))
        else:
            self._prefetch_token_capacity = 0  # no CPU cache → no limit
        self._prefetch_tokens_occupied: int = 0

        # ---- Batched prefetch executor infrastructure ----
        self._prefetch_enabled = GLOBAL_CONFIG_FROM_ENV.prefetch_enabled
        self._prefetch_batch_size = GLOBAL_CONFIG_FROM_ENV.prefetch_batch_size
        self._prefetch_batch_timeout = GLOBAL_CONFIG_FROM_ENV.prefetch_batch_timeout
        # mini-graph_id → PrefetchBatchExecutor (for routing CompletedOps in _update_tasks)
        self._pending_mini_graphs: Dict[int, PrefetchBatchExecutor] = {}
        # task_id → PrefetchBatchExecutor (for terminate_prefetch lookup)
        self._task_to_executor: Dict[int, PrefetchBatchExecutor] = {}
        # Deferred insert queue: (task_id, completed_cpu_block_ids) pairs
        # collected by executor thread, executed by main thread in _update_tasks
        self._deferred_inserts_queue: List[tuple] = []
        # Queue for the executor thread to pick up new executors
        self._prefetch_executor_queue: Queue = Queue()
        # Executor thread: started once at init, blocks on queue.get() when idle.
        if self._prefetch_enabled:
            self._prefetch_executor_thread = threading.Thread(
                target=self._prefetch_executor_loop,
                daemon=True,
                name="flexkv-prefetch-batch-executor",
            )
            self._prefetch_executor_thread.start()
        else:
            self._prefetch_executor_thread = None

        flexkv_logger.info(
            f"prefetch_enabled: {self._prefetch_enabled}, "
            f"prefetch_capacity_ratio: {GLOBAL_CONFIG_FROM_ENV.prefetch_capacity_ratio}, " 
            f"prefetch_token_capacity: {self._prefetch_token_capacity}, "
            f"prefetch_batch_size: {self._prefetch_batch_size}, "
            f"prefetch_batch_timeout: {self._prefetch_batch_timeout}, "
        )


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

    def create_get_task(self,
                        task_id: int,
                        token_ids: np.ndarray,
                        slot_mapping: np.ndarray,
                        token_mask: Optional[np.ndarray] = None,
                        layer_granularity: int = -1,
                        dp_rank: int = 0,
                        pp_rank: int = 0,
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
            layer_num=self.model_config.num_layers_per_pp_stage,
            layer_granularity=layer_granularity,
            dp_rank=dp_rank,
            pp_rank=pp_rank,
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
            dp_rank=dp_rank,
            pp_rank=pp_rank,
            graph=graph,
            return_mask=return_mask,
            callback=callback,
            op_callback_dict=op_callback_dict)

        self.graph_to_task[graph.graph_id] = task_id

    def create_put_task(self,
                        task_id: int,
                        token_ids: np.ndarray,
                        slot_mapping: np.ndarray,
                        token_mask: Optional[np.ndarray] = None,
                        dp_rank: int = 0,
                        pp_rank: int = 0,
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
            layer_num=self.model_config.num_layers_per_pp_stage,
            dp_rank=dp_rank,
            pp_rank=pp_rank,
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
            dp_rank=dp_rank,
            pp_rank=pp_rank,
            graph=graph,
            return_mask=return_mask,
            callback=callback,
            op_callback_dict=op_callback_dict)
        self.graph_to_task[graph.graph_id] = task_id

    def create_prefetch_task(self,
                            task_id: int,
                            token_ids: np.ndarray,
                            pp_rank: int = 0,
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
            layer_num=self.model_config.num_layers_per_pp_stage,
            dp_rank=0,  # dp_rank irrelevant: prefetch only uploads to CPU (ignore_gpu=True)
            pp_rank=pp_rank,
            temp_cache_strategy=temp_cache_strategy,
            namespace=namespace)
        node_to_unlock, buffer_to_free, is_put, deferred_insert_info = self._extract_resource_refs(callback)
        self.tasks[task_id] = KVTask(
            task_id=task_id,
            task_type=TaskType.PREFETCH,
            task_end_op_id=task_end_op_id,
            task_end_op_finished=False,
            status=TaskStatus.READY,  # gpu slots are not needed for prefetch
            token_ids=token_ids,
            slot_mapping=fake_slot_mapping,  # ignore slot_mapping for prefetch
            token_mask=fake_token_mask,  # ignore token_mask for prefetch
            dp_rank=0,  # ignore dp_rank for prefetch
            pp_rank=pp_rank,
            graph=graph,
            return_mask=return_mask,
            callback=callback,
            op_callback_dict=op_callback_dict,
            node_to_unlock=node_to_unlock,
            buffer_to_free=buffer_to_free,
            is_put=is_put,
            deferred_insert_info=deferred_insert_info)

        self.prefetch_tasks[self._gen_prefetch_key(token_ids, namespace)] = task_id

        self.graph_to_task[graph.graph_id] = task_id

    @staticmethod
    def _extract_resource_refs(callback) -> tuple:
        """Extract node_to_unlock, buffer_to_free, is_put, deferred_insert_info from callback's partial.keywords."""
        keywords = getattr(callback, 'keywords', {}) if callback is not None else {}
        node_to_unlock = keywords.get('node_to_unlock', {})
        buffer_to_free = keywords.get('buffer_to_free', {})
        is_put = keywords.get('is_put', False)
        deferred_insert_info = keywords.get('deferred_insert_info', None)
        return node_to_unlock, buffer_to_free, is_put, deferred_insert_info

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
        for completed_op in completed_ops:
            # Route mini-graph completions to the batch executor
            if completed_op.graph_id in self._pending_mini_graphs:
                executor = self._pending_mini_graphs[completed_op.graph_id]
                if completed_op.is_graph_completed():
                    executor.notify_batch_done()
                    # Don't pop here — the executor thread does it after waking up
                continue
            # Route non-mini-graph completions to the task
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

        # Execute deferred inserts collected by batch executor thread.
        # This runs on main thread where radix tree operations are safe.
        if self._prefetch_enabled:
            self._drain_deferred_inserts()
            
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
        # Release resources: unlock (no set_ready) + recycle buffer_to_free
        if task.node_to_unlock:
            try:
                self.cache_engine.abort_task(task)
            except Exception as e:
                flexkv_logger.error(f"_cancel_task abort error for task {task_id}: {e}")
            task.node_to_unlock = None
            task.buffer_to_free = None
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
        # Prefetch discard path uses structured resource refs (abort_task handles
        # unlock without set_ready). Normal GET/PUT always use original callback.
        if task.task_type == TaskType.PREFETCH and task.node_to_unlock:
            self.cache_engine.complete_task(task)
            task.node_to_unlock = None  # prevent double-processing
        elif task.callback:
            if isinstance(task.callback, list):
                for callback in task.callback:
                    callback()
            else:
                task.callback()
        if task.task_type == TaskType.PREFETCH:
            # Release token capacity: use the exact value recorded at prefetch time,
            # not return_mask (which may be modified by batch executor on termination).
            if task.prefetch_tokens_reserved > 0:
                self._prefetch_tokens_occupied = max(
                    0, self._prefetch_tokens_occupied - task.prefetch_tokens_reserved
                )
        task.status = TaskStatus.COMPLETED
        task.task_end_op_finished = True
        self.graph_to_task.pop(task.graph.graph_id, None)

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

    def _check_config(self, model_config: ModelConfig, cache_config: CacheConfig) -> None:
        if cache_config.enable_remote:
            if cache_config.remote_cache_path is None:

                if cache_config.remote_file_prefix is None:
                    raise ValueError("remote_file_prefix must be provided when remote_cache_path is None")

                if cache_config.remote_file_num is None or cache_config.remote_file_num <= 0:
                    raise ValueError("remote_file_num must be a positive integer")

                cache_config.remote_cache_path = [
                    f"{cache_config.remote_file_prefix}_{i}"
                    for i in range(cache_config.remote_file_num)
                ]

            if cache_config.remote_cache_size_mode == "block_num":
                if cache_config.num_remote_blocks is None:
                    raise ValueError("num_remote_blocks must not None if use block_num model")
            elif cache_config.remote_cache_size_mode == "file_size":
                if cache_config.remote_file_size is None:
                    raise ValueError("remote_file_size must not None if use file_size model")
                if model_config.use_mla:
                    kv_size = (
                        model_config.num_layers_per_pp_stage
                        * cache_config.tokens_per_block
                        * model_config.num_kv_heads
                        * model_config.head_size
                        * model_config.dtype.itemsize
                    )
                else:
                    kv_size = (
                        model_config.num_layers_per_pp_stage
                        * 2
                        * cache_config.tokens_per_block
                        * model_config.num_kv_heads
                        * model_config.head_size
                        * model_config.dtype.itemsize
                    )
                cache_config.num_remote_blocks = cache_config.remote_file_size // kv_size * cache_config.remote_file_num

            else:
                raise ValueError("remote_cache_size_mode must block_num or file_size model")

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
                  token_mask: Optional[np.ndarray] = None,
                  layer_granularity: int = -1,
                  dp_rank: int = 0,
                  pp_rank: int = 0,
                  task_id: int = -1,
                  namespace: Optional[List[str]] = None) -> Tuple[int, np.ndarray]:
        self._drain_deferred_inserts()
        # self._sync_prefetch(token_ids, namespace)
        task_id, return_mask = self._get_match_impl(token_ids,
                                                    slot_mapping,
                                                    is_fake_slot_mapping=False,
                                                    token_mask=token_mask,
                                                    layer_granularity=layer_granularity,
                                                    dp_rank=dp_rank,
                                                    pp_rank=pp_rank,
                                                    task_id=task_id,
                                                    namespace=namespace)
        # trace get request
        self.tracer.trace_request(
            request_type="GET",
            request_id=task_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            token_mask=token_mask,
            layer_granularity=layer_granularity,
            dp_rank=dp_rank,
            pp_rank=pp_rank
        )
        self._launch_task(task_id)
        return task_id, return_mask

    def put_async(self,
                  token_ids: np.ndarray,
                  slot_mapping: np.ndarray,
                  token_mask: Optional[np.ndarray] = None,
                  dp_rank: int = 0,
                  pp_rank: int = 0,
                  task_id: int = -1,
                  namespace: Optional[List[str]] = None) -> Tuple[int, np.ndarray]:
        self._drain_deferred_inserts()
        task_id, return_mask = self._put_match_impl(token_ids,
                                                    slot_mapping,
                                                    is_fake_slot_mapping=False,
                                                    token_mask=token_mask,
                                                    dp_rank=dp_rank,
                                                    pp_rank=pp_rank,
                                                    task_id=task_id,
                                                    namespace=namespace)
        # trace put request
        self.tracer.trace_request(
            request_type="PUT",
            request_id=task_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            token_mask=token_mask,
            layer_granularity=-1,  # put has no layer_granularity parameter
            dp_rank=dp_rank,
            pp_rank=pp_rank
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
                  token_mask: Optional[np.ndarray] = None,
                  layer_granularity: int = -1,
                  dp_rank: int = 0,
                  pp_rank: int = 0,
                  cpu_only: bool = False,
                  task_id: int = -1,
                  namespace: Optional[List[str]] = None) -> Tuple[int, np.ndarray]:
        self._drain_deferred_inserts()
        nvtx.push_range(f"get match: task_id={task_id}", color=get_nvtx_default_color())
        # self._sync_prefetch(token_ids, namespace)
        if token_mask is None:
            token_mask = np.ones_like(token_ids, dtype=bool)
        fake_slot_mapping = np.zeros_like(token_ids[token_mask])
        result_task_id, return_mask = self._get_match_impl(token_ids,
                                                           fake_slot_mapping,
                                                           is_fake_slot_mapping=True,
                                                           token_mask=token_mask,
                                                           layer_granularity=layer_granularity,
                                                           dp_rank=dp_rank,
                                                           pp_rank=pp_rank,
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
            layer_granularity=layer_granularity,
            dp_rank=dp_rank,
            pp_rank=pp_rank
        )
        nvtx.pop_range()
        return result_task_id, return_mask

    def _get_match_impl(self,
                  token_ids: np.ndarray,
                  slot_mapping: np.ndarray,
                  is_fake_slot_mapping: bool = False,
                  token_mask: Optional[np.ndarray] = None,
                  layer_granularity: int = -1,
                  dp_rank: int = 0,
                  pp_rank: int = 0,
                  cpu_only: bool = False,
                  task_id: int = -1,
                  namespace: Optional[List[str]] = None) -> Tuple[int, np.ndarray]:
        if token_mask is None:
            token_mask = np.ones_like(token_ids)
        if layer_granularity == -1:
            layer_granularity = self.model_config.num_layers_per_pp_stage
        if task_id == -1:
            task_id = self._gen_task_id()
        temp_cache_strategy = DEFAULT_CACHE_STRATEGY
        if cpu_only:
            temp_cache_strategy = CPUONLY_CACHE_STRATEGY
        nvtx.push_range(f"get match: task_id={task_id}", color=get_nvtx_default_color())
        self.create_get_task(task_id,
                             token_ids,
                             slot_mapping,
                             token_mask,
                             layer_granularity,
                             dp_rank,
                             pp_rank=pp_rank,
                             is_fake_slot_mapping=is_fake_slot_mapping,
                             temp_cache_strategy=temp_cache_strategy,
                             namespace=namespace)
        self._process_empty_graph(task_id)
        nvtx.pop_range()
        return task_id, self.tasks[task_id].return_mask

    def put_match(self,
                  token_ids: np.ndarray,
                  token_mask: Optional[np.ndarray] = None,
                  dp_rank: int = 0,
                  pp_rank: int = 0,
                  task_id: int = -1,
                  namespace: Optional[List[str]] = None) -> Tuple[int, np.ndarray]:
        self._drain_deferred_inserts()
        fake_slot_mapping = np.zeros_like(token_ids)
        result_task_id, return_mask = self._put_match_impl(token_ids,
                                                           fake_slot_mapping,
                                                           is_fake_slot_mapping=True,
                                                           token_mask=token_mask,
                                                           dp_rank=dp_rank,
                                                           pp_rank=pp_rank,
                                                           task_id=task_id,
                                                           namespace=namespace)
        # trace put match request
        self.tracer.trace_request(
            request_type="PUT_MATCH",
            request_id=result_task_id,
            token_ids=token_ids,
            slot_mapping=fake_slot_mapping,
            token_mask=token_mask,
            layer_granularity=-1,  # put has no layer_granularity parameter
            dp_rank=dp_rank,
            pp_rank=pp_rank
        )
        return result_task_id, return_mask

    def _put_match_impl(self,
                        token_ids: np.ndarray,
                        slot_mapping: np.ndarray,
                        is_fake_slot_mapping: bool = False,
                        token_mask: Optional[np.ndarray] = None,
                        dp_rank: int = 0,
                        pp_rank: int = 0,
                        task_id: int = -1,
                        namespace: Optional[List[str]] = None) -> Tuple[int, np.ndarray]:
        if token_mask is None:
            token_mask = np.ones_like(token_ids)
        if task_id == -1:
            task_id = self._gen_task_id()
        nvtx.push_range(f"put match: task_id={task_id}", color=get_nvtx_default_color())
        self.create_put_task(task_id,
                             token_ids,
                             slot_mapping,
                             token_mask,
                             dp_rank,
                             pp_rank=pp_rank,
                             is_fake_slot_mapping=is_fake_slot_mapping,
                             namespace=namespace)
        self._process_empty_graph(task_id)
        nvtx.pop_range()
        return task_id, self.tasks[task_id].return_mask

    def prefetch_async(self,
                       token_ids: np.ndarray,
                       dp_rank: int = 0,
                       pp_rank: int = 0,
                       task_id: int = -1,
                       namespace: Optional[List[str]] = None) -> Tuple[int, int]:
        self._drain_deferred_inserts()
        if task_id == -1:
            task_id = self._gen_task_id()
        nvtx.push_range(f"prefetch match: task_id={task_id}", color=get_nvtx_default_color())
        self.create_prefetch_task(task_id, token_ids, pp_rank=pp_rank, namespace=namespace)
        self._process_empty_graph(task_id)
        nvtx.pop_range()

        # ---- Phase1: early return if no substantive hit ----
        # If the transfer graph was empty (no SSD/Remote hit), the task was
        # already completed by _process_empty_graph → _mark_completed (callback
        # executed, resources released).  Clean up bookkeeping and return -1
        # so the caller (connector) does not track a no-op prefetch.
        task = self.tasks.get(task_id)
        if task is not None and task.is_completed():
            self.prefetch_tasks.pop(self._gen_prefetch_key(token_ids, namespace), None)
            del self.tasks[task_id]
            flexkv_logger.info(f"[FlexKV] prefetch task {task_id} completed by _process_empty_graph, early return")
            return -1, 0


        # ---- Phase2: token-level flow control ----
        # Precise SSD→CPU token count from return_mask (set by cache_engine.get
        # during create_prefetch_task). If current prefetch would exceed capacity, 
        # discard the task immediately and release all resources allocated by 
        # create_prefetch_task
        actual_prefetch_tokens = 0
        if task is not None and task.return_mask is not None:
            actual_prefetch_tokens = int(np.sum(task.return_mask))

        # Token-level flow control (before enqueue/launch) 
        discard = False
        # first level check: if actual transfer is below threshold, discard the task immediately
        if actual_prefetch_tokens < self._prefetch_token_threshold:
            flexkv_logger.info(f"[FlexKV] prefetch task {task_id} discarded because actual transfer is below threshold {self._prefetch_token_threshold}")
            discard = True

        # second level check: if actual transfer would exceed capacity, discard the task immediately
        if self._prefetch_token_capacity > 0 and actual_prefetch_tokens > 0 and self._prefetch_tokens_occupied + actual_prefetch_tokens > self._prefetch_token_capacity:
            flexkv_logger.info(f"[FlexKV] prefetch task {task_id} discarded because actual transfer would exceed capacity {self._prefetch_token_capacity}")
            discard = True

        if discard and task is not None and not task.is_completed():
            # Release all resources allocated by create_prefetch_task.
            # With deferred insert, node_to_unlock is empty (no insert → no lock),
            # but buffer_to_free[CPU] still holds allocated blocks that must be recycled.
            if task.node_to_unlock:
                # actually won't reach here
                try:
                    self.cache_engine.abort_task(task)
                except Exception as e:
                    flexkv_logger.error(
                        f"[FlexKV] prefetch discard abort error for task {task_id}: {e}")
                task.node_to_unlock = None
                task.buffer_to_free = None
            elif task.buffer_to_free:
                # Deferred insert path: no node to unlock, but CPU blocks need recycling
                for device_type, blocks in task.buffer_to_free.items():
                    if len(blocks) > 0:
                        self.cache_engine.cache_engines[device_type].recycle(blocks)
                task.buffer_to_free = None

            # Clean up task bookkeeping
            self.graph_to_task.pop(task.graph.graph_id, None)
            self.prefetch_tasks.pop(self._gen_prefetch_key(token_ids, namespace), None)
            del self.tasks[task_id]
            # trace prefetch discard
            self.tracer.trace_request(
                request_type="PREFETCH_ASYNC",
                request_id=task_id,
                token_ids=token_ids,
                slot_mapping=np.zeros_like(token_ids),
                token_mask=np.ones_like(token_ids),
                layer_granularity=-1,
                dp_rank=dp_rank,
                pp_rank=pp_rank
            )
            return -1, 0

        # trace prefetch async request
        self.tracer.trace_request(
            request_type="PREFETCH_ASYNC",
            request_id=task_id,
            token_ids=token_ids,
            slot_mapping=np.zeros_like(token_ids),
            token_mask=np.ones_like(token_ids),
            layer_granularity=-1,
            dp_rank=dp_rank,
            pp_rank=pp_rank
        )

        # ---- Phase3: enqueue/launch ----
        # Track tokens occupied by this prefetch and record the exact value 
        # so _mark_completed can decrement correctly,
        # even if return_mask is later modified (e.g. by batch executor termination).
        if actual_prefetch_tokens > 0:
            self._prefetch_tokens_occupied += actual_prefetch_tokens
            # Record the exact value so _mark_completed can decrement correctly,
            # even if return_mask is later modified (e.g. by batch executor termination).
            if task is not None:
                task.prefetch_tokens_reserved = actual_prefetch_tokens

        # Route to batch executor thread for all prefetch tasks with DISK2H ops.
        batch_executor = self._create_batch_executor(task_id)

        if batch_executor is not None:
            # Route to batch executor thread — do NOT submit the original graph.
            flexkv_logger.info(f"[FlexKV] prefetch task {task_id} routed to batch executor")
            self._prefetch_executor_queue.put(batch_executor)
        else:
            # No DISK2H op found — should not happen after _process_empty_graph.
            # Recycle allocated blocks and clean up to avoid resource leak.
            flexkv_logger.warning(f"[FlexKV] prefetch task {task_id} has no DISK2H op, cleaning up")
            if task is not None and task.buffer_to_free:
                for device_type, blocks in task.buffer_to_free.items():
                    if len(blocks) > 0:
                        self.cache_engine.cache_engines[device_type].recycle(blocks)
                task.buffer_to_free = None
            if actual_prefetch_tokens > 0:
                self._prefetch_tokens_occupied = max(
                    0, self._prefetch_tokens_occupied - actual_prefetch_tokens)
            self.graph_to_task.pop(task.graph.graph_id, None)
            self.prefetch_tasks.pop(self._gen_prefetch_key(token_ids, namespace), None)
            del self.tasks[task_id]
            return -1, actual_prefetch_tokens


        return task_id, actual_prefetch_tokens



    # ------------------------------------------------------------------
    # Deferred insert for prefetch (zombie-free design)
    # ------------------------------------------------------------------

    def _drain_deferred_inserts(self) -> None:
        """Consume pending deferred inserts queued by executor thread.

        MUST be called from main thread only. This ensures radix tree
        operations (match/insert) are safe from concurrent access.

        Called at the beginning of main-thread entry points that operate
        on the radix tree: get_match, get_async, put_match, put_async,
        prefetch_async.
        """
        if self._deferred_inserts_queue:
            num_items = len(self._deferred_inserts_queue)
            for task_id, cpu_block_ids in self._deferred_inserts_queue:
                self._execute_deferred_insert(task_id, cpu_block_ids)
            self._deferred_inserts_queue.clear()
            # Monitor unready blocks after drain
            try:
                unready = self.cache_engine.cpu_cache_engine.index.total_unready_blocks()
                free = self.cache_engine.cpu_cache_engine.mempool.num_free_blocks
                total_pool = self.cache_engine.cpu_cache_engine.mempool.num_total_blocks
                flexkv_logger.info(
                    f"[FlexKV] _drain_deferred_inserts: processed {num_items} items, "
                    f"unready_blocks={unready}, free_blocks={free}/{total_pool}")
            except Exception:
                pass

    def _execute_deferred_insert(self, task_id: int, completed_cpu_block_ids: np.ndarray) -> None:
        """Execute deferred insert after prefetch transfer completes.

        Re-matches the CPU radix tree and inserts the transferred blocks
        with is_ready=True. This avoids zombie nodes because insert only
        happens after data is fully transferred.

        Args:
            task_id: The task whose deferred insert should be executed.
            completed_cpu_block_ids: CPU block IDs that were successfully transferred.
        """
        task = self.tasks.get(task_id)
        if task is None or task.deferred_insert_info is None:
            return

        info = task.deferred_insert_info
        sequence_meta = info['sequence_meta']
        num_insert_blocks = info['num_insert_blocks']

        if len(completed_cpu_block_ids) == 0:
            task.deferred_insert_info = None
            return

        try:
            cpu_engine = self.cache_engine.cpu_cache_engine
            # Step 1: Re-match CPU radix tree (tree may have changed)
            match_result = cpu_engine.match(sequence_meta)
            num_matched = match_result.num_matched_blocks

            # Step 2: Calculate actual blocks to insert
            actual_insert = num_insert_blocks - num_matched

            if actual_insert <= 0:
                # All already in tree (another request inserted them)
                cpu_engine.recycle(completed_cpu_block_ids)
                flexkv_logger.debug(
                    f"[FlexKV] deferred insert for task {task_id}: "
                    f"skipped (already in tree), recycled {len(completed_cpu_block_ids)} blocks")
            elif actual_insert > len(completed_cpu_block_ids):
                # Prefix was evicted, only insert what we have
                actual_insert = len(completed_cpu_block_ids)
                adjusted_num_insert = num_matched + actual_insert
                node = cpu_engine.insert(sequence_meta,
                                         completed_cpu_block_ids,
                                         num_insert_blocks=adjusted_num_insert,
                                         is_ready=True,
                                         match_result=match_result)
                if node is None:
                    cpu_engine.recycle(completed_cpu_block_ids)
                flexkv_logger.debug(
                    f"[FlexKV] deferred insert for task {task_id}: "
                    f"partial insert {actual_insert} blocks (prefix evicted)")
            elif actual_insert < len(completed_cpu_block_ids):
                # Some overlap with existing tree — recycle duplicates
                blocks_to_insert = completed_cpu_block_ids[:actual_insert]
                blocks_to_recycle = completed_cpu_block_ids[actual_insert:]
                cpu_engine.recycle(blocks_to_recycle)
                node = cpu_engine.insert(sequence_meta,
                                         blocks_to_insert,
                                         num_insert_blocks=num_insert_blocks,
                                         is_ready=True,
                                         match_result=match_result)
                if node is None:
                    cpu_engine.recycle(blocks_to_insert)
                flexkv_logger.debug(
                    f"[FlexKV] deferred insert for task {task_id}: "
                    f"inserted {actual_insert} blocks, recycled {len(blocks_to_recycle)} duplicates")
            else:
                # Exact match: insert all completed blocks
                node = cpu_engine.insert(sequence_meta,
                                         completed_cpu_block_ids,
                                         num_insert_blocks=num_insert_blocks,
                                         is_ready=True,
                                         match_result=match_result)
                if node is None:
                    cpu_engine.recycle(completed_cpu_block_ids)
                    flexkv_logger.debug(
                        f"[FlexKV] deferred insert for task {task_id}: "
                        f"insert returned None, recycled {len(completed_cpu_block_ids)} blocks")
                else:
                    flexkv_logger.debug(
                        f"[FlexKV] deferred insert for task {task_id}: "
                        f"inserted {len(completed_cpu_block_ids)} blocks")

            # Clear buffer_to_free[CPU] to prevent double-free:
            # The blocks are now either in the tree or recycled above.
            if task.buffer_to_free and DeviceType.CPU in task.buffer_to_free:
                task.buffer_to_free[DeviceType.CPU] = np.array([], dtype=np.int64)

        except Exception as e:
            flexkv_logger.error(
                f"[FlexKV] deferred insert failed for task {task_id}: {e}")
            # Safety: recycle blocks to prevent leak
            try:
                self.cache_engine.cpu_cache_engine.recycle(completed_cpu_block_ids)
            except Exception:
                pass
            if task.buffer_to_free and DeviceType.CPU in task.buffer_to_free:
                task.buffer_to_free[DeviceType.CPU] = np.array([], dtype=np.int64)

        task.deferred_insert_info = None

    # ------------------------------------------------------------------
    # Batched prefetch executor
    # ------------------------------------------------------------------

    def _create_batch_executor(self, task_id: int) -> Optional[PrefetchBatchExecutor]:
        """Create a PrefetchBatchExecutor from an already-created prefetch task.

        Inspects the task's transfer graph for a DISK2H op.  If one exists and
        is large enough to benefit from batching, returns a new executor with
        the op's block arrays extracted.  Returns ``None`` when batching is not
        needed (no DISK2H, too few blocks, or feature disabled).

        The caller is responsible for NOT submitting the original graph when a
        batch executor is returned — the executor will create its own
        mini-graphs.
        """
        if not self._prefetch_enabled:
            return None

        task = self.tasks.get(task_id)
        if task is None or task.graph.num_ops == 0:
            return None

        # Find the DISK2H op in the graph
        disk2h_op: Optional[TransferOp] = None
        for op_id, op in task.graph._op_map.items():
            if op.transfer_type == TransferType.DISK2H:
                disk2h_op = op
                break

        if disk2h_op is None:
            flexkv_logger.error(f"No DISK2H op found for task {task_id}")
            # No DISK2H (pure CPU hit or empty graph) — no need to batch
            return None

        total_blocks = len(disk2h_op.src_block_ids)

        executor = PrefetchBatchExecutor(
            task_id=task_id,
            ssd_block_ids=disk2h_op.src_block_ids,
            cpu_block_ids=disk2h_op.dst_block_ids,
            batch_size_blocks=self._prefetch_batch_size,
            total_blocks=total_blocks,
        )

        # Register executor for terminate_prefetch lookup
        self._task_to_executor[task_id] = executor

        return executor

    def _prefetch_executor_loop(self) -> None:
        """Main loop of the batch executor thread.

        Processes PrefetchBatchExecutors one at a time from the queue.
        Each executor is driven to completion (or termination) before the
        next one is picked up.
        """
        while True:
            try:
                executor = self._prefetch_executor_queue.get()
                if executor is None:
                    break  # shutdown signal
                self._execute_prefetch_batched(executor)
            except Exception as e:
                flexkv_logger.error(
                    f"[FlexKV] prefetch batch executor loop error: {e}")

    def _execute_prefetch_batched(self, executor: PrefetchBatchExecutor) -> None:
        """Drive one PrefetchBatchExecutor to completion or termination.

        Called on the executor thread.  For each batch:
        1. Create a mini TransferOpGraph with a slice of the DISK2H blocks.
        2. Register it in ``_pending_mini_graphs`` so the main thread can
           route its CompletedOp.
        3. Submit to TransferManager.
        4. Wait on ``executor._batch_done_event`` (set by main thread).
        5. Check termination flag; if set, stop and finalize.
        """
        try:
            # Mark task as RUNNING (batch executor bypasses _launch_task which
            # normally sets this via check_task_ready).
            task = self.tasks.get(executor.task_id)
            if task is not None:
                task.status = TaskStatus.RUNNING

            while not executor.is_terminated() and not executor.is_complete:
                start = executor.next_batch_start
                end = executor.next_batch_end

                # Create mini-graph for this batch
                mini_graph = TransferOpGraph()
                mini_op = TransferOp(
                    graph_id=mini_graph.graph_id,
                    transfer_type=TransferType.DISK2H,
                    src_block_ids=executor.ssd_block_ids[start:end].copy(),
                    dst_block_ids=executor.cpu_block_ids[start:end].copy(),
                    layer_id=0,
                    layer_granularity=self.model_config.num_layers,
                )
                flexkv_logger.info(f"[FlexKV] prefetch batch {executor.completed_batches} for task {executor.task_id} creating mini-graph {mini_graph.graph_id} with {len(executor.ssd_block_ids[start:end])} blocks")
                mini_graph.add_transfer_op(mini_op)

                # Register so _update_tasks can route the CompletedOp back
                executor.reset_batch_event()
                executor._current_mini_graph_id = mini_graph.graph_id
                self._pending_mini_graphs[mini_graph.graph_id] = executor

                # Submit to all transfer handles
                for transfer_handle in self.transfer_handles:
                    transfer_handle.submit(mini_graph)

                # Wait for main thread to signal batch completion
                batch_done = executor.wait_current_batch(
                    timeout=self._prefetch_batch_timeout
                )

                if batch_done and not executor.is_terminated():
                    # Normal completion: IO done, not terminated
                    self._pending_mini_graphs.pop(mini_graph.graph_id, None)
                    executor._current_mini_graph_id = None
                    executor.completed_blocks = end
                    executor.completed_batches += 1

                elif executor.is_terminated():
                    # Terminated: mark_terminate() woke us up.
                    # The current batch IO may still be in flight — must wait for
                    # it to complete before recycling blocks (prevents dirty writes
                    # from io_uring writing to already-recycled CPU memory).
                    if not executor._io_done:
                        # IO hasn't completed yet — wait for actual CompletedOp.
                        executor.reset_batch_event()
                        batch_done = executor.wait_current_batch(
                            timeout=self._prefetch_batch_timeout
                        )

                    # Now the current batch IO is definitely done (or timed out).
                    self._pending_mini_graphs.pop(mini_graph.graph_id, None)
                    executor._current_mini_graph_id = None
                    if executor._io_done:
                        # Current batch IO completed — count it as done
                        executor.completed_blocks = end
                        executor.completed_batches += 1
                    # Break out of the loop — don't submit more batches
                    break

                else:
                    # Batch timed out (neither done nor terminated)
                    self._pending_mini_graphs.pop(mini_graph.graph_id, None)
                    executor._current_mini_graph_id = None
                    flexkv_logger.warning(
                        f"[FlexKV] prefetch batch {executor.completed_batches} "
                        f"for task {executor.task_id} timed out after "
                        f"{self._prefetch_batch_timeout}s"
                    )
                    executor.mark_terminate()
        except Exception as e:
            flexkv_logger.error(
                f"[FlexKV] _execute_prefetch_batched error for task "
                f"{executor.task_id}: {e}"
            )
            if not executor.is_terminated():
                executor.mark_terminate()
        finally:
            self._finalize_batch_executor(executor)

    def _finalize_batch_executor(self, executor: PrefetchBatchExecutor) -> None:
        """Clean up after a batch executor completes or is terminated.

        With deferred insert design: no node was inserted during task creation,
        so there is no zombie node to worry about. We simply:
        - Collect completed blocks for deferred insert (main thread will execute)
        - Recycle uncompleted blocks immediately
        - Mark task as completed
        """
        task = self.tasks.get(executor.task_id)
        completed = executor.completed_blocks
        total = executor.total_blocks

        if executor.is_complete and not executor.is_terminated():
            # ---- Full success ----
            # Queue deferred insert for all blocks (main thread will execute)
            if task is not None and task.deferred_insert_info is not None:
                self._deferred_inserts_queue.append(
                    (executor.task_id, executor.cpu_block_ids.copy())
                )
            # Clear buffer_to_free[CPU] to prevent _mark_completed callback from
            # recycling blocks that will be inserted into radix tree by deferred insert.
            if task is not None and task.buffer_to_free and DeviceType.CPU in task.buffer_to_free:
                task.buffer_to_free[DeviceType.CPU] = np.array([], dtype=np.int64)
            flexkv_logger.info(
                f"[FlexKV] prefetch batch for task {executor.task_id} completed: "
                f"{completed}/{total} blocks "
                f"in {executor.completed_batches} batches"
            )
        else:
            # ---- Partial completion or terminated ----
            # Queue deferred insert for completed portion only
            if task is not None and task.deferred_insert_info is not None and completed > 0:
                self._deferred_inserts_queue.append(
                    (executor.task_id, executor.cpu_block_ids[:completed].copy())
                )

            # Recycle uncompleted blocks immediately (mempool is thread-safe)
            if completed < total:
                remaining_blocks = executor.cpu_block_ids[completed:]
                try:
                    self.cache_engine.cpu_cache_engine.recycle(remaining_blocks)
                except Exception as e:
                    flexkv_logger.error(
                        f"[FlexKV] batch executor recycle remaining error for task "
                        f"{executor.task_id}: {e}")

            # Clear buffer_to_free[CPU] to prevent double-recycle in callback:
            # - completed blocks will be handled by deferred insert
            # - remaining blocks already recycled above
            if task is not None and task.buffer_to_free and DeviceType.CPU in task.buffer_to_free:
                task.buffer_to_free[DeviceType.CPU] = np.array([], dtype=np.int64)

            # Update return_mask to reflect actual completion
            if task is not None and task.return_mask is not None:
                completed_tokens = completed * self.cache_config.tokens_per_block
                task.return_mask[completed_tokens:] = False

            flexkv_logger.info(
                f"[FlexKV] prefetch batch for task {executor.task_id} terminated: "
                f"{completed}/{total} blocks "
                f"({executor.progress_ratio:.1%})"
            )

        # ---- Common cleanup ----
        self._task_to_executor.pop(executor.task_id, None)

        # Monitor unready blocks after prefetch finalize
        try:
            unready = self.cache_engine.cpu_cache_engine.index.total_unready_blocks()
            free = self.cache_engine.cpu_cache_engine.mempool.num_free_blocks
            total_pool = self.cache_engine.cpu_cache_engine.mempool.num_total_blocks
            flexkv_logger.info(
                f"[FlexKV] prefetch task {executor.task_id} finalized: "
                f"unready_blocks={unready}, free_blocks={free}/{total_pool}, "
                f"deferred_queue_len={len(self._deferred_inserts_queue)}")
        except Exception:
            pass

        # Mark task as completed (releases prefetch token capacity).
        # node_to_unlock is empty for prefetch (no insert was done), so
        # _mark_completed just updates status + releases token capacity.
        if task is not None and not task.is_completed():
            self._mark_completed(executor.task_id)

    # ------------------------------------------------------------------
    # Terminate API
    # ------------------------------------------------------------------

    def terminate_prefetch(self, task_id: int) -> int:
        """Request termination of an in-flight batched prefetch.

        If the task is being executed by a PrefetchBatchExecutor, sets its
        terminated flag.  The executor thread will stop after the current
        batch completes and release remaining resources.

        If the task was submitted via the original (non-batched) path, this
        is a no-op — the C++ transfer cannot be interrupted.
        """
        if task_id not in self.tasks:
            return 0
        task = self.tasks[task_id]
        if task.is_completed():
            return 0

        executor = self._task_to_executor.get(task_id)
        if executor is not None:
            executor.mark_terminate()
            flexkv_logger.info(
                f"[FlexKV] terminate_prefetch: task {task_id} "
                f"(completed {executor.completed_blocks}/{executor.total_blocks})"
            )
            return executor.completed_blocks * self.cache_config.tokens_per_block
        else:
            # Non-batch path (_launch_task): cannot interrupt C++ IO.
            # Check if the task already completed naturally.
            task = self.tasks.get(task_id)
            if task is not None and task.is_completed():
                loaded = int(np.sum(task.return_mask)) if task.return_mask is not None else 0
                return loaded
            flexkv_logger.debug(
                f"[FlexKV] terminate_prefetch: task {task_id} not in batch mode, "
                f"cannot interrupt"
            )
            return 0

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
            dp_rank=self.tasks[task_ids[0]].dp_rank,
            pp_rank=self.tasks[task_ids[0]].pp_rank,
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
