import time
from typing import Dict, Optional, List, Union, Tuple
import threading
from enum import Enum
from dataclasses import dataclass
from typing import Callable


from expiring_dict import ExpiringDict
import nvtx
import torch
import numpy as np

from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.common.debug import flexkv_logger
from flexkv.common.transfer import TransferOpGraph, get_nvtx_default_color
from flexkv.common.tracer import FlexKVTracer
from flexkv.cache.cache_engine import GlobalCacheEngine
from flexkv.transfer_manager import TransferManagerHandle
from flexkv.common.request import KVResponseStatus, KVResponse

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
    return_mask: np.ndarray
    callback: Optional[Callable]
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
                 use_separate_process: bool = True,
                 ):
        if not cache_config.enable_cpu:
            raise ValueError("enable_cpu must be True")
        if cache_config.enable_remote and not cache_config.enable_ssd:
            raise ValueError("enable_ssd must be True if enable_remote is True")
        if not cache_config.enable_cpu and not cache_config.enable_gds:
            raise ValueError("enable_gds must be True if enable_cpu is False")
        if cache_config.enable_ssd and cache_config.enable_gds:
            raise ValueError("enable_ssd and enable_gds cannot be used at the same time")
        self.cache_config = cache_config
        self.model_config = model_config
        self._check_config(model_config, cache_config)

        self.cache_engine = GlobalCacheEngine(cache_config, model_config)

        self.transfer_handle = TransferManagerHandle(
            self.model_config,
            self.cache_config,
            use_separate_process=use_separate_process,
            gpu_register_port=gpu_register_port
        )

        self.tasks: ExpiringDict[int, KVTask] = ExpiringDict(max_age_seconds=1800, max_len=100000) # 30 minutes
        self.graph_to_task: Dict[int, int] = {}

        self.task_id_counter = 0
        self.task_id_lock = threading.Lock()

        self.running_tasks: int = 0

    def start(self) -> None:
        self.transfer_handle.start()

    def is_ready(self) -> bool:
        return self.transfer_handle.is_ready()

    def __del__(self) -> None:
        self.shutdown()

    def shutdown(self) -> None:
        self.transfer_handle.shutdown()

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

    def _launch_task(self, task_id: int) -> None:
        task = self.tasks[task_id]
        if task.is_completed():
            return
        if task.status != TaskStatus.READY:
            raise ValueError(f"Task {task_id} status is {task.status}, cannot launch")
        transfer_graph = task.graph
        task.status = TaskStatus.RUNNING
        nvtx.mark(f"launch task: task_id={task_id}, graph_id={transfer_graph.graph_id}")
        if transfer_graph.num_ops > 0:
            self.transfer_handle.submit(transfer_graph)

    def _update_tasks(self, timeout: float = 0.001) -> None:
        completed_ops = self._get_completed_ops(timeout)
        for completed_graph_id, completed_op_id in completed_ops:
            if completed_graph_id not in self.graph_to_task:
                continue
            task_id = self.graph_to_task[completed_graph_id]
            task = self.tasks[task_id]
            if completed_op_id == -1:  # the graph is totally finished
                self._mark_completed(task_id)
            elif completed_op_id == task.task_end_op_id:
                self.tasks[task_id].task_end_op_finished = True
            if completed_op_id in task.op_callback_dict:
                task.op_callback_dict[completed_op_id]()

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

    def _mark_completed(self, task_id: int) -> None:
        task = self.tasks[task_id]
        if task.is_completed():
            return
        if task.callback:
            task.callback()
        task.status = TaskStatus.COMPLETED
        task.task_end_op_finished = True
        self.graph_to_task.pop(task.graph.graph_id)

    def _process_empty_graph(self, task_id: int) -> None:
        task = self.tasks[task_id]
        if task.graph.num_ops == 0:
            self._mark_completed(task_id)

    def _get_completed_ops(self, timeout: Optional[float] = None) -> List[Tuple[int, int]]:
        return self.transfer_handle.wait(timeout)

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
                 use_separate_process: bool = True,
                 ):
        super().__init__(model_config, cache_config, gpu_register_port, use_separate_process)
        self.tracer = FlexKVTracer(cache_config)

    def get_async(self,
                  token_ids: np.ndarray,
                  slot_mapping: np.ndarray,
                  token_mask: Optional[np.ndarray] = None,
                  layer_granularity: int = -1,
                  dp_id: int = 0,
                  task_id: int = -1) -> Tuple[int, np.ndarray]:
        task_id, return_mask = self._get_match_impl(token_ids,
                                                    slot_mapping,
                                                    is_fake_slot_mapping=False,
                                                    token_mask=token_mask,
                                                    layer_granularity=layer_granularity,
                                                    dp_id=dp_id,
                                                    task_id=task_id)
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
        return return_responses

    def try_wait(self, task_ids: Union[int, List[int]]) -> Dict[int, KVResponse]:
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        nvtx.mark(f"try_wait task_ids: {task_ids}")
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
        return_responses = self._wait_impl(task_ids, timeout, completely=completely)
        nvtx.pop_range()
        return return_responses

    def get_match(self,
                  token_ids: np.ndarray,
                  token_mask: Optional[np.ndarray] = None,
                  layer_granularity: int = -1,
                  dp_id: int = 0,
                  task_id: int = -1) -> Tuple[int, np.ndarray]:
        if token_mask is None:
            token_mask = np.ones_like(token_ids, dtype=bool)
        fake_slot_mapping = np.zeros_like(token_ids[token_mask])
        return self._get_match_impl(token_ids,
                                    fake_slot_mapping,
                                    is_fake_slot_mapping=True,
                                    token_mask=token_mask,
                                    layer_granularity=layer_granularity,
                                    dp_id=dp_id,
                                    task_id=task_id)

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
        return self._put_match_impl(token_ids,
                                    fake_slot_mapping,
                                    is_fake_slot_mapping=True,
                                    token_mask=token_mask,
                                    dp_id=dp_id,
                                    task_id=task_id)

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

    def launch_tasks(self,
                        task_ids: List[int],
                        slot_mappings: List[np.ndarray]) -> None:
        assert isinstance(slot_mappings[0], np.ndarray)
        self.set_slot_mappings(task_ids, slot_mappings)
        for task_id in task_ids:
            self._launch_task(task_id)

    def cancel_tasks(self, task_ids: Union[int, List[int]]) -> None:
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        for task_id in task_ids:
            self._cancel_task(task_id)

    def _clear_cpu_cache(self) -> None:
        self.cache_engine.cpu_cache_engine.reset()
