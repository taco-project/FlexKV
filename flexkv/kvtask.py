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


    def is_completed(self) -> bool:
        return self.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED, TaskStatus.FAILED]

    def shed_heavy_resources(self) -> None:
        # Drop heavy fields once the task terminates. Keeps status / return_mask / batch link 
        # so wait() can still report the result; only when the user observes it (wait/try_wait) or cancels it.
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

    def create_get_task(self,
                        task_id: int,
                        token_ids: np.ndarray,
                        slot_mapping: np.ndarray,
                        dp_client_id: int,
                        token_mask: Optional[np.ndarray] = None,
                        is_fake_slot_mapping: bool = False,
                        temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
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
            if completed_op.is_graph_completed():
                self._mark_completed(task_id)
            elif completed_op.op_id == task.task_end_op_id:
                self.tasks[task_id].task_end_op_finished = True
            if completed_op.op_id in task.op_callback_dict:
                task.op_callback_dict[completed_op.op_id]()

    def _cancel_task(self, task_id: int) -> None:
        if task_id not in self.tasks:
            return
        task = self.tasks[task_id]
        if not task.is_completed():
            task.status = TaskStatus.CANCELLED
        self._release_task(task_id)

    def check_completed(self, task_id: int, completely: bool = False) -> bool:
        task = self.tasks[task_id]
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
        return task.graph

    def _release_task(self, task_id: int) -> None:
        """Remove the task record entirely. Called when the user observes (wait/try_wait)
        or cancels the task. """
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
            if isinstance(task.callback, list):
                for callback in task.callback:
                    callback()
            else:
                task.callback()
        task.status = TaskStatus.COMPLETED
        task.task_end_op_finished = True
        self.graph_to_task.pop(task.graph.graph_id, None)
        task.shed_heavy_resources()

    def _process_empty_graph(self, task_id: int) -> None:
        task = self.tasks[task_id]
        if task.graph is None:
            return
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
                    return_responses[task_id] = KVResponse(
                        status=convert_to_response_status(self.tasks[task_id].status),
                        task_id=task_id,
                        return_mask=self.tasks[task_id].return_mask
                    )
                    if self.tasks[task_id].is_completed():
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
            child_task = self.tasks[task_id]
            if child_task.graph is not None:
                self.graph_to_task.pop(child_task.graph.graph_id, None)
            self.tasks.pop(task_id, None)
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
