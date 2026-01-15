import time
from typing import Dict, Optional, List, Union, Tuple
import threading
from enum import Enum
from dataclasses import dataclass
from typing import Callable
import multiprocessing as mp
import copy
import os
from expiring_dict import ExpiringDict
import nvtx
import torch
import numpy as np

from flexkv.common.config import CacheConfig, ModelConfig, GLOBAL_CONFIG_FROM_ENV
from flexkv.common.debug import flexkv_logger
from flexkv.common.transfer import TransferOpGraph, merge_to_batch_graph, get_nvtx_default_color, CompletedOp
from flexkv.common.tracer import FlexKVTracer
from flexkv.cache.cache_engine import GlobalCacheEngine, DEFAULT_CACHE_STRATEGY
from flexkv.transfer_manager import TransferManagerHandle, TransferManagerOnRemote
from flexkv.common.request import KVResponseStatus, KVResponse
from flexkv.transfer_manager import (
    get_master_host_and_ports_from_env,
    get_trtllm_subprocess_host_and_ports_from_env
)
from flexkv.common.hash_utils import hash_array
from flexkv.cache.redis_meta import RedisMeta

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
    dp_id: int

    # cache engine return
    graph: TransferOpGraph
    return_mask: Union[np.ndarray, list[np.ndarray]]
    callback: Optional[Union[Callable, List[Callable]]]
    op_callback_dict: Dict[int, Callable]

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
                 redis_meta: RedisMeta = None
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

        self.is_multinode_tp = False
        self.tp_node_count = 1
        if self.model_config.tp_size > torch.cuda.device_count():
            if self.model_config.tp_size != torch.cuda.device_count() * 2:
                raise ValueError("Only support 2 nodes TP for now")
            assert self.model_config.dp_size == 1
            self.tp_node_count = self.model_config.tp_size // torch.cuda.device_count()
            self.is_multinode_tp = True

        self.cache_engine = GlobalCacheEngine(cache_config, model_config, redis_meta)

        model_config_for_transfer = copy.deepcopy(self.model_config)
        if self.is_multinode_tp:
            model_config_for_transfer.tp_size //= self.tp_node_count
            if not self.model_config.use_mla:
                model_config_for_transfer.num_kv_heads //= self.tp_node_count

        combine_with_trtllm = os.getenv("FLEXKV_WITH_TRTLLM", "0") == "1"
        if not combine_with_trtllm:
            self.transfer_handles = [TransferManagerHandle(
                model_config_for_transfer,
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
                    model_config_for_transfer,
                    self.cache_config,
                    mode="remote",
                    gpu_register_port=gpu_register_port,
                    master_host=master_host,
                    master_ports=master_ports
                )
            ]
            self.transfer_handles[0]._handle.send_config_to_remotes()

        if self.is_multinode_tp:
            master_host, master_ports = get_master_host_and_ports_from_env()
            self.transfer_handles.append(TransferManagerHandle(
                model_config_for_transfer,
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
        self._hash_func = lambda x: hash_array(x)

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
                        token_mask: Optional[np.ndarray] = None,
                        layer_granularity: int = -1,
                        dp_id: int = 0,
                        is_fake_slot_mapping: bool = False,
                        ) -> None:
        if task_id in self.tasks:
            raise ValueError(f"Task ID {task_id} already exists")
        graph, return_mask, callback, op_callback_dict, task_end_op_id = self.cache_engine.get(task_id,
                                                                               token_ids,
                                                                               token_mask,
                                                                               slot_mapping,
                                                                               self.model_config.num_layers,
                                                                               layer_granularity,
                                                                               dp_id)
        self.tasks[task_id] = KVTask(
            task_id=task_id,
            task_type=TaskType.GET,
            task_end_op_id=task_end_op_id,
            task_end_op_finished=False,
            status=TaskStatus.UNREADY if is_fake_slot_mapping else TaskStatus.READY,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            token_mask=token_mask,
            dp_id=dp_id,
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
                        dp_id: int = 0,
                        is_fake_slot_mapping: bool = False,
                        ) -> None:
        if task_id in self.tasks:
            raise ValueError(f"Task ID {task_id} already exists")
        graph, return_mask, callback, op_callback_dict, task_end_op_id = self.cache_engine.put(task_id,
                                                                               token_ids,
                                                                               token_mask,
                                                                               slot_mapping,
                                                                               self.model_config.num_layers,
                                                                               dp_id)
        self.tasks[task_id] = KVTask(
            task_id=task_id,
            task_type=TaskType.PUT,
            task_end_op_id=task_end_op_id,
            task_end_op_finished=False,
            status=TaskStatus.UNREADY if is_fake_slot_mapping else TaskStatus.READY,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            token_mask=token_mask,
            dp_id=dp_id,
            graph=graph,
            return_mask=return_mask,
            callback=callback,
            op_callback_dict=op_callback_dict)
        self.graph_to_task[graph.graph_id] = task_id

    def create_prefetch_task(self,
                            task_id: int,
                            token_ids: np.ndarray,
                            ) -> None:
        if task_id in self.tasks:
            raise ValueError(f"Task ID {task_id} already exists")
        fake_slot_mapping = np.zeros_like(token_ids)
        fake_token_mask = np.ones_like(token_ids)
        temp_cache_strategy = copy.deepcopy(DEFAULT_CACHE_STRATEGY)
        temp_cache_strategy.ignore_gpu = True  # upload to CPU only
        temp_cache_strategy.ignore_gds = True
        graph, return_mask, callback, op_callback_dict, task_end_op_id = self.cache_engine.get(task_id,
                                                                               token_ids,
                                                                               fake_token_mask,
                                                                               fake_slot_mapping,
                                                                               self.model_config.num_layers,
                                                                               temp_cache_strategy=temp_cache_strategy)
        self.tasks[task_id] = KVTask(
            task_id=task_id,
            task_type=TaskType.PREFETCH,
            task_end_op_id=task_end_op_id,
            task_end_op_finished=False,
            status=TaskStatus.READY,  # gpu slots are not needed for prefetch
            token_ids=token_ids,
            slot_mapping=fake_slot_mapping,  # ignore slot_mapping for prefetch
            token_mask=fake_token_mask,  # ignore token_mask for prefetch
            dp_id=0,  # ignore dp_id for prefetch
            graph=graph,
            return_mask=return_mask,
            callback=callback,
            op_callback_dict=op_callback_dict)

        self.prefetch_tasks[self._hash_func(token_ids)] = task_id

        self.graph_to_task[graph.graph_id] = task_id

    def _launch_task(self, task_id: int) -> None:
        transfer_graph = self.check_task_ready(task_id)
        if transfer_graph is None:
            return
        nvtx.mark(f"launch task: task_id={task_id}, graph_id={transfer_graph.graph_id}")
        if transfer_graph.num_ops > 0:
            for transfer_handle in self.transfer_handles:
                transfer_handle.submit(transfer_graph)

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
        self._process_empty_graph(task_id)
        task = self.tasks[task_id]
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
                        model_config.num_layers
                        * cache_config.tokens_per_block
                        * model_config.num_kv_heads
                        * model_config.head_size
                        * model_config.dtype.itemsize
                    )
                else:
                    kv_size = (
                        model_config.num_layers
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
                 redis_meta: Optional[RedisMeta] = None
                 ):
        super().__init__(model_config, cache_config, gpu_register_port, redis_meta)
        self.tracer = FlexKVTracer()
        self.tracer.trace_config(model_config, cache_config, gpu_layout=None)

    def get_async(self,
                  token_ids: np.ndarray,
                  slot_mapping: np.ndarray,
                  token_mask: Optional[np.ndarray] = None,
                  layer_granularity: int = -1,
                  dp_id: int = 0,
                  task_id: int = -1) -> Tuple[int, np.ndarray]:
        self._sync_prefetch(token_ids)
        task_id, return_mask = self._get_match_impl(token_ids,
                                                    slot_mapping,
                                                    is_fake_slot_mapping=False,
                                                    token_mask=token_mask,
                                                    layer_granularity=layer_granularity,
                                                    dp_id=dp_id,
                                                    task_id=task_id)
        # trace get request
        self.tracer.trace_request(
            request_type="GET",
            request_id=task_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            token_mask=token_mask,
            layer_granularity=layer_granularity,
            dp_id=dp_id
        )
        self._launch_task(task_id)
        return task_id, return_mask

    def put_async(self,
                  token_ids: np.ndarray,
                  slot_mapping: np.ndarray,
                  token_mask: Optional[np.ndarray] = None,
                  dp_id: int = 0,
                  task_id: int = -1) -> Tuple[int, np.ndarray]:
        task_id, return_mask = self._put_match_impl(token_ids,
                                                    slot_mapping,
                                                    is_fake_slot_mapping=False,
                                                    token_mask=token_mask,
                                                    dp_id=dp_id,
                                                    task_id=task_id)
        # trace put request
        self.tracer.trace_request(
            request_type="PUT",
            request_id=task_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            token_mask=token_mask,
            layer_granularity=-1,  # put has no layer_granularity parameter
            dp_id=dp_id
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

    def _sync_prefetch(self, token_ids: np.ndarray) -> None:
        prefetch_task_id = self.prefetch_tasks.get(self._hash_func(token_ids), None)
        if prefetch_task_id is not None:
            start_time = time.time()
            self.wait([prefetch_task_id], completely=True)
            end_time = time.time()
            flexkv_logger.debug(f"sync prefetch task {prefetch_task_id} cost {(end_time - start_time) * 1000} ms")

    def get_match(self,
                  token_ids: np.ndarray,
                  token_mask: Optional[np.ndarray] = None,
                  layer_granularity: int = -1,
                  dp_id: int = 0,
                  task_id: int = -1) -> Tuple[int, np.ndarray]:
        nvtx.push_range(f"get match: task_id={task_id}", color=get_nvtx_default_color())
        self._sync_prefetch(token_ids)
        if token_mask is None:
            token_mask = np.ones_like(token_ids, dtype=bool)
        fake_slot_mapping = np.zeros_like(token_ids[token_mask])
        result_task_id, return_mask = self._get_match_impl(token_ids,
                                                           fake_slot_mapping,
                                                           is_fake_slot_mapping=True,
                                                           token_mask=token_mask,
                                                           layer_granularity=layer_granularity,
                                                           dp_id=dp_id,
                                                           task_id=task_id)
        # trace get match request
        self.tracer.trace_request(
            request_type="GET_MATCH",
            request_id=result_task_id,
            token_ids=token_ids,
            slot_mapping=fake_slot_mapping,
            token_mask=token_mask,
            layer_granularity=layer_granularity,
            dp_id=dp_id
        )
        nvtx.pop_range()
        return result_task_id, return_mask

    def _get_match_impl(self,
                  token_ids: np.ndarray,
                  slot_mapping: np.ndarray,
                  is_fake_slot_mapping: bool = False,
                  token_mask: Optional[np.ndarray] = None,
                  layer_granularity: int = -1,
                  dp_id: int = 0,
                  task_id: int = -1) -> Tuple[int, np.ndarray]:
        if token_mask is None:
            token_mask = np.ones_like(token_ids)
        if layer_granularity == -1:
            layer_granularity = self.model_config.num_layers
        if task_id == -1:
            task_id = self._gen_task_id()
        nvtx.push_range(f"get match: task_id={task_id}", color=get_nvtx_default_color())
        self.create_get_task(task_id,
                             token_ids,
                             slot_mapping,
                             token_mask,
                             layer_granularity,
                             dp_id,
                             is_fake_slot_mapping=is_fake_slot_mapping)
        self._process_empty_graph(task_id)
        nvtx.pop_range()
        return task_id, self.tasks[task_id].return_mask

    def put_match(self,
                  token_ids: np.ndarray,
                  token_mask: Optional[np.ndarray] = None,
                  dp_id: int = 0,
                  task_id: int = -1) -> Tuple[int, np.ndarray]:
        fake_slot_mapping = np.zeros_like(token_ids)
        result_task_id, return_mask = self._put_match_impl(token_ids,
                                                           fake_slot_mapping,
                                                           is_fake_slot_mapping=True,
                                                           token_mask=token_mask,
                                                           dp_id=dp_id,
                                                           task_id=task_id)
        # trace put match request
        self.tracer.trace_request(
            request_type="PUT_MATCH",
            request_id=result_task_id,
            token_ids=token_ids,
            slot_mapping=fake_slot_mapping,
            token_mask=token_mask,
            layer_granularity=-1,  # put has no layer_granularity parameter
            dp_id=dp_id
        )
        return result_task_id, return_mask

    def _put_match_impl(self,
                        token_ids: np.ndarray,
                        slot_mapping: np.ndarray,
                        is_fake_slot_mapping: bool = False,
                        token_mask: Optional[np.ndarray] = None,
                        dp_id: int = 0,
                        task_id: int = -1) -> Tuple[int, np.ndarray]:
        if token_mask is None:
            token_mask = np.ones_like(token_ids)
        if task_id == -1:
            task_id = self._gen_task_id()
        nvtx.push_range(f"put match: task_id={task_id}", color=get_nvtx_default_color())
        self.create_put_task(task_id,
                             token_ids,
                             slot_mapping,
                             token_mask,
                             dp_id,
                             is_fake_slot_mapping=is_fake_slot_mapping)
        self._process_empty_graph(task_id)
        nvtx.pop_range()
        return task_id, self.tasks[task_id].return_mask

    def prefetch_async(self,
                       token_ids: np.ndarray,
                       dp_id: int = 0,
                       task_id: int = -1) -> int:
        if task_id == -1:
            task_id = self._gen_task_id()
        nvtx.push_range(f"prefetch match: task_id={task_id}", color=get_nvtx_default_color())
        self.create_prefetch_task(task_id, token_ids)
        self._process_empty_graph(task_id)
        nvtx.pop_range()
        # trace prefetch async request
        self.tracer.trace_request(
            request_type="PREFETCH_ASYNC",
            request_id=task_id,
            token_ids=token_ids,
            slot_mapping=np.zeros_like(token_ids),
            token_mask=np.ones_like(token_ids),
            layer_granularity=-1,
            dp_id=dp_id
        )
        self._launch_task(task_id)
        return task_id

    def merge_to_batch_kvtask(self,
                              batch_id: int,
                              task_ids: List[int],
                              layerwise_transfer: bool = False,
                              counter_id: int = 0) -> TransferOpGraph:
        op_callback_dict = {}
        task_end_op_ids = []
        callbacks = []
        transfer_graphs = []
        return_masks = []
        for task_id in task_ids:
            assert self.tasks[task_id].task_type == TaskType.GET, "only get task can be launched as batch"
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
            task_type=TaskType.BATCH_GET,
            task_end_op_id=task_end_op_id,
            task_end_op_finished=False,
            status=TaskStatus.READY,
            dp_id=self.tasks[task_ids[0]].dp_id,
            graph=batch_task_graph,
            return_mask=return_masks,
            callback=callbacks, # this is a list now
            op_callback_dict=op_callback_dict,
        )
        self.graph_to_task[batch_task_graph.graph_id] = batch_id
        # pop those tasks which are merged into batch
        for task_id in task_ids:
            self.graph_to_task.pop(self.tasks[task_id].graph.graph_id, None)
            self.tasks.pop(task_id, None)
        batch_task_graph = self.check_task_ready(batch_id)
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

        if as_batch:
            if batch_id == -1:
                batch_id = self._gen_task_id()
            if layerwise_transfer:
                if not GLOBAL_CONFIG_FROM_ENV.enable_layerwise_transfer:
                    flexkv_logger.warning("layerwise transfer is not enabled")
                    layerwise_transfer = False
                else:
                    for task_id in task_ids:
                        if self.tasks[task_id].task_type != TaskType.GET:
                            flexkv_logger.warning("only support layerwise get")
                            layerwise_transfer = False
                            break
            batch_task_graph = self.merge_to_batch_kvtask(batch_id, task_ids, layerwise_transfer, counter_id)
            transfer_graphs = [batch_task_graph]
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
