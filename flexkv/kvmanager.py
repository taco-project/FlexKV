import multiprocessing as mp
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Dict, Any, Optional
from typing import List, Callable, Union

import nvtx
import torch
from expiring_dict import ExpiringDict

from flexkv.cache.cache_engine import GlobalCacheEngine, TransferOpGraph
from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.common.debug import flexkv_logger
from flexkv.common.memory_handle import TensorSharedHandle
from flexkv.common.request import KVRequestType, KVRequest
from flexkv.common.transfer import DeviceType, get_nvtx_range_color, get_nvtx_default_color
from flexkv.common.storage import KVCacheLayout
from flexkv.common.exceptions import LogicError
from flexkv.common.tracer import FlexKVTracer
from flexkv.storage.storage_engine import StorageEngine
from flexkv.transfer.transfer_engine import TransferEngine


@dataclass
class RequestTracker:
    task_id: int
    task_type: KVRequestType
    return_mask: torch.Tensor
    callback: Optional[Callable]
    task_end_ops_ids: List[int]
    task_end_ops_status: List[bool]
    task_finished: bool = False


class KVManager:
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_layout: Optional[KVCacheLayout] = None,
                 gpu_blocks: Optional[Dict[int, List[TensorSharedHandle]]] = None):

        flexkv_logger.info(f"Initializing kvmanager...\nmodel_config: {model_config}\ncache_config: {cache_config}")

        mp.set_start_method('spawn', force=True)
        self.init_nvtx_range = nvtx.push_range("Initialize kvmanager", color=get_nvtx_default_color())

        if not cache_config.enable_cpu:
            raise ValueError("enable_cpu must be True")
        if cache_config.enable_remote and not cache_config.enable_ssd:
            raise ValueError("enable_ssd must be True if enable_remote is True")
        if not cache_config.enable_cpu and not cache_config.use_gds:
            raise ValueError("use_gds must be True if enable_cpu is False")
        self.cache_config = cache_config
        self.model_config = model_config

        self._verify_Model_Cache_config(model_config, cache_config)
        self.cache_engine = GlobalCacheEngine(cache_config, model_config)
        self.storage_engine = StorageEngine(self.model_config, self.cache_config)

        # Initialize tracer
        self.tracer = FlexKVTracer(cache_config)

        # Record configuration in tracer
        if gpu_layout is not None:
            self.tracer.trace_config(model_config, cache_config, gpu_layout)


        self.transfer_engine: Optional[TransferEngine] = None
        self.gpu_layout: Optional[KVCacheLayout] = gpu_layout

        self.running = False
        self.requests_tracker: ExpiringDict[int, RequestTracker] = ExpiringDict(1800) # 30 minutes
        self.graph_to_request: Dict[int, int] = {}
        self.taskid_to_nvtx_range: Dict[int, Any] = {}
        self.graphid_to_nvtx_range: Dict[int, Any] = {}

        self._task_id_counter = 0
        self.task_queue: Queue[KVRequest] = Queue()

        if gpu_blocks is None:
            gpu_blocks = {}

        self.num_gpus = len(gpu_blocks)
        self.all_gpu_blocks: Dict[int, List[TensorSharedHandle]] = gpu_blocks

        self.lock = threading.Lock()

        if self.num_gpus == self.model_config.tp_size * self.model_config.dp_size:
            self._init_transfer_engine()

    # Note that for now only after all the gpu blocks are added, we can initialize the transfer engine
    def _init_transfer_engine(self) -> None:
        assert self.gpu_layout is not None
        assert len(self.all_gpu_blocks) == self.model_config.tp_size * self.model_config.dp_size
        for device_id, gpu_blocks_wrapper in self.all_gpu_blocks.items():
            self.storage_engine.register_gpu_blocks(gpu_blocks_wrapper,
                                                    self.gpu_layout,
                                                    device_id,
                                                    dtype=self.model_config.dtype)
        self.gpu_handles = [
            self.storage_engine.get_storage_handle(DeviceType.GPU, i)
            for i in range(self.model_config.tp_size * self.model_config.dp_size)
        ]
        cpu_handle = self.storage_engine.get_storage_handle(DeviceType.CPU) if self.cache_config.enable_cpu else None
        ssd_handle = self.storage_engine.get_storage_handle(DeviceType.SSD) if self.cache_config.enable_ssd else None
        remote_handle = (
            self.storage_engine.get_storage_handle(DeviceType.REMOTE)
            if self.cache_config.enable_remote
            else None
        )
        self.transfer_engine = TransferEngine(self.gpu_handles,
                                              self.model_config,
                                              self.cache_config,
                                              cpu_handle,
                                              ssd_handle,
                                              remote_handle)

        nvtx.pop_range(self.init_nvtx_range)


    def is_ready(self) -> bool:
        return self.transfer_engine is not None

    def is_running(self) -> bool:
        return self.running

    def start(self) -> None:
        if self.running:
            flexkv_logger.warning("kvmanager is already running")
            return
        if not self.is_ready():
            raise ValueError("transfer engine is not ready, please add all gpu blocks first")
        if self.transfer_engine is not None:
            self.transfer_engine.start()
            self.running = True
        else:
            raise ValueError("transfer engine is not initialized, please call start() after all gpu blocks are added")

        self._worker_thread = threading.Thread(target=self._worker_loop)
        self._worker_thread.start()
        flexkv_logger.info("KVManager fully started and running")

    # the gpu_blocks of multiple gpus can be added post initialization.
    # the transfer engine will be initialized after we have all the intended gpu handles.
    def register_single_gpu_blocks(
        self,
        gpu_handles: List[TensorSharedHandle],
        gpu_layout: KVCacheLayout,
        dp_client_id: int = 0,
        tp_rank: int = 0,
    ) -> None:
        if self.transfer_engine is not None:
            raise ValueError("we have already get all gpu blocks")
        if self.gpu_layout is None:
            self.gpu_layout = gpu_layout
            self.tracer.trace_config(self.model_config, self.cache_config, self.gpu_layout)
        else:
            assert self.gpu_layout == gpu_layout
        self.all_gpu_blocks[tp_rank + dp_client_id * self.model_config.tp_size] = gpu_handles
        self.num_gpus += 1
        if self.num_gpus == self.model_config.tp_size * self.model_config.dp_size:
            self._init_transfer_engine()

    def _worker_loop(self) -> None:
        assert self.transfer_engine is not None
        while self.running:
            # deal with completed requests from the cache engine
            if not self.task_queue.empty():
                request = self.task_queue.get()
                if request.request_type == KVRequestType.SHUTDOWN:
                    self.shutdown()
                    break
                elif request.request_type == KVRequestType.GET:
                    nvtx.push_range(f"cache_engine.get request_id: {request.request_id}",
                                    color=get_nvtx_default_color())
                    graph, return_mask, callback, task_end_ops_ids = self.cache_engine.get(request.request_id,
                                                                                           request.token_ids,
                                                                                           request.token_mask,
                                                                                           request.slot_mapping,
                                                                                           self.model_config.num_layers,
                                                                                           request.layer_granularity,
                                                                                           request.dp_id)
                elif request.request_type == KVRequestType.PUT:
                    nvtx.push_range(f"cache_engine.put request_id: {request.request_id}",
                                    color=get_nvtx_default_color())
                    graph, return_mask, callback, task_end_ops_ids = self.cache_engine.put(request.request_id,
                                                                                           request.token_ids,
                                                                                           request.token_mask,
                                                                                           request.slot_mapping,
                                                                                           self.model_config.num_layers,
                                                                                           request.dp_id)
                else:
                    raise ValueError(f"Unknown request type: {request.request_type}")
                nvtx.pop_range()
                if graph.num_ops == 0: #early return
                    flexkv_logger.info(f"no transfer: "
                                   f"request_id = {request.request_id}, request_type = {request.request_type}")
                    layer_op_num = self.model_config.num_layers // request.layer_granularity \
                        if request.request_type == KVRequestType.GET else 1
                    self.requests_tracker[request.request_id] = RequestTracker(task_id=request.request_id,
                                                                               task_type=request.request_type,
                                                                               return_mask=return_mask,
                                                                               callback=None,
                                                                               task_end_ops_ids=[-1]*layer_op_num,
                                                                               task_end_ops_status=[True]*layer_op_num,
                                                                               task_finished=True)
                else:
                    self.graph_to_request[graph.graph_id] = request.request_id
                    self.graphid_to_nvtx_range[graph.graph_id] = nvtx.start_range(
                                                                            f"request id: {request.request_id}, "
                                                                            f"graph id: {graph.graph_id}",
                                                                            color=get_nvtx_range_color(graph.graph_id))
                    self.requests_tracker[request.request_id] = RequestTracker(task_id=request.request_id,
                                                                               task_type=request.request_type,
                                                                               return_mask=return_mask,
                                                                               callback=callback,
                                                                               task_end_ops_ids=task_end_ops_ids,
                                                                               task_end_ops_status=len(task_end_ops_ids)*[False],
                                                                               task_finished=False)
                    self.transfer_engine.submit_transfer_graph(graph)
            results = self.transfer_engine.get_completed_graphs_and_ops(timeout=0.001)
            for completed_graph_id, completed_op_id in results:
                request_id = self.graph_to_request[completed_graph_id]
                request_tracker = self.requests_tracker[request_id]
                if completed_op_id == -1:
                    if request_tracker.callback:
                        request_tracker.callback()
                    nvtx.end_range(self.graphid_to_nvtx_range[completed_graph_id])
                    self.graphid_to_nvtx_range.pop(completed_graph_id)
                    self.graph_to_request.pop(completed_graph_id)
                    nvtx.end_range(self.taskid_to_nvtx_range[request_tracker.task_id])
                    self.taskid_to_nvtx_range.pop(request_tracker.task_id)
                    request_tracker.task_finished = True
                elif completed_op_id in request_tracker.task_end_ops_ids:
                    request_tracker.task_end_ops_status[request_tracker.task_end_ops_ids.index(completed_op_id)] = True
                self.requests_tracker[request_id] = request_tracker
            time.sleep(0.0001)

    def _get_task_id(self) -> int:
        with self.lock:
            old_value = self._task_id_counter
            self._task_id_counter += 1
            return old_value

    def __del__(self) -> None:
        if hasattr(self, 'tracer'):
            self.tracer.flush()
        if self.running:
            self.shutdown()

    def shutdown(self) -> None:
        self.running = False
        # Flush tracer before shutdown
        if hasattr(self, 'tracer'):
            self.tracer.flush()
        flexkv_logger.info("kvmanager shutdown")
        self.task_queue.put(KVRequest(
            request_type=KVRequestType.SHUTDOWN,
            request_id=-1,
            token_ids=torch.empty(0),
            token_mask=torch.empty(0),
            slot_mapping=torch.empty(0),
        ))
        self._worker_thread.join()
        if self.transfer_engine is not None:
            self.transfer_engine.shutdown()

    def get_async(self,
                  token_ids: torch.Tensor,
                  slot_mapping: torch.Tensor,
                  token_mask: Optional[torch.Tensor] = None,
                  layer_granularity: int = -1,
                  dp_id: int = 0,
                  task_id: int = -1) -> int:
        if not self.running:
            raise ValueError("kvmanager is not running, please call start() first")
        if token_mask is None:
            token_mask = torch.ones_like(token_ids)
        if layer_granularity == -1:
            layer_granularity = self.model_config.num_layers
        if task_id == -1:
            task_id = self._get_task_id()
        # Trace the request
        self.tracer.trace_request(
            request_type="GET",
            request_id=task_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            token_mask=token_mask,
            layer_granularity=layer_granularity,
            dp_id=dp_id
        )
        nvtx.mark(f"GET request_id: {task_id}")
        self.taskid_to_nvtx_range[task_id] = nvtx.start_range(f"GET request_id: {task_id}",
                                                              color=get_nvtx_default_color())
        self.task_queue.put(KVRequest(
            request_type=KVRequestType.GET,
            request_id=task_id,
            token_ids=token_ids,
            token_mask=token_mask,
            slot_mapping=slot_mapping,
            layer_granularity=layer_granularity,
            dp_id=dp_id,
        ))
        self.requests_tracker[task_id] = RequestTracker(task_id=task_id,
                                                        task_type=KVRequestType.GET,
                                                        return_mask=torch.empty(0),
                                                        callback=None,
                                                        task_end_ops_ids=[],
                                                        task_end_ops_status=[],
                                                        task_finished=False)
        return task_id

    def put_async(self,
                  token_ids: torch.Tensor,
                  slot_mapping: torch.Tensor,
                  token_mask: Optional[torch.Tensor] = None,
                  dp_id: int = 0,
                  task_id: int = -1) -> int:
        if not self.running:
            raise ValueError("kvmanager is not running, please call start() first")
        if token_mask is None:
            token_mask = torch.ones_like(token_ids)
        if task_id == -1:
            task_id = self._get_task_id()
        # Trace the request
        self.tracer.trace_request(
            request_type="PUT",
            request_id=task_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            token_mask=token_mask,
            dp_id=dp_id
        )
        nvtx.mark(f"PUT request_id: {task_id}")
        self.taskid_to_nvtx_range[task_id] = nvtx.start_range(f"PUT request_id: {task_id}",
                                                              color=get_nvtx_default_color())
        self.task_queue.put(KVRequest(
            request_type=KVRequestType.PUT,
            request_id=task_id,
            token_ids=token_ids,
            token_mask=token_mask,
            slot_mapping=slot_mapping,
            dp_id=dp_id,
        ))
        self.requests_tracker[task_id] = RequestTracker(task_id=task_id,
                                                        task_type=KVRequestType.PUT,
                                                        return_mask=torch.empty(0),
                                                        callback=None,
                                                        task_end_ops_ids=[],
                                                        task_end_ops_status=[],
                                                        task_finished=False)
        return task_id

    # wait for the key op to be finished
    def wait(self, task_ids: Union[int, List[int]], timeout: float = 20.0) -> Dict[int, torch.Tensor]:
        # Trace the wait request
        self.tracer.trace_wait_request(
            wait_type="wait",
            task_ids=task_ids,
        )
        nvtx.mark(f"wait task_ids: {task_ids}")
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        num_completed_tasks = 0
        num_tasks = len(task_ids)
        return_masks = {}
        start_time = time.time()
        while num_completed_tasks < num_tasks:
            finished_task_ids = []
            for task_id in task_ids:
                if task_id not in self.requests_tracker:
                    flexkv_logger.error(f"task_id {task_id} not submitted into flexKV")
                    return_masks[task_id] = torch.empty(0, dtype=torch.bool) #if not found in tracker, the return mask is an empty tensor
                    num_completed_tasks += 1
                    finished_task_ids.append(task_id)
                    continue
                task_tracker = self.requests_tracker[task_id]
                if len(task_tracker.return_mask) == 0: #NOT READY
                    continue
                if all(task_tracker.task_end_ops_status):
                    num_completed_tasks += 1
                    return_masks[task_id] = task_tracker.return_mask
                    finished_task_ids.append(task_id)
            task_ids = [task_id for task_id in task_ids if task_id not in finished_task_ids]
            if time.time() - start_time > timeout:
                flexkv_logger.warning(f"wait task_ids: {task_ids} timeout, has to return now")
                for task_id in task_ids:
                    return_masks[task_id] = torch.empty(0, dtype=torch.bool) # return mask of timeout task is also an empty tensor
                nvtx.mark(f"wait task_ids: {task_ids} timeout")
                return return_masks
            time.sleep(0.0001)
        nvtx.mark(f"wait task_ids: {task_ids} done")
        return return_masks

    # wait for the whole task to be finished, including the key op and all other ops
    # this function is mainly designed for testing to avoid the frequency of writing is too high to use up memory blocks
    def wait_for_graph_finished(self,
                                task_ids: Union[int, List[int]],
                                timeout: float = 20.0) -> Dict[int, torch.Tensor]:
         # Trace the wait request
        self.tracer.trace_wait_request(
            wait_type="wait_for_graph_finished",
            task_ids=task_ids,
        )
        nvtx.mark(f"wait task_ids: {task_ids}")
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        num_completed_tasks = 0
        return_masks = {}
        start_time = time.time()
        while num_completed_tasks < len(task_ids):
            finished_task_ids = []
            for task_id in task_ids:
                if task_id not in self.requests_tracker:
                    flexkv_logger.error(f"task_id {task_id} not submitted into flexKV")
                    return_masks[task_id] = torch.empty(0) #if not found in tracker, the return mask is an empty tensor
                    num_completed_tasks += 1
                    finished_task_ids.append(task_id)
                    continue
                task_tracker = self.requests_tracker[task_id]
                if task_tracker.task_finished:
                    num_completed_tasks += 1
                    return_masks[task_id] = task_tracker.return_mask
                    finished_task_ids.append(task_id)
            task_ids = [task_id for task_id in task_ids if task_id not in finished_task_ids]
            if time.time() - start_time > timeout:
                flexkv_logger.warning(f"wait task_ids: {task_ids} timeout, has to return now")
                for task_id in task_ids:
                    return_masks[task_id] = torch.empty(0) # return mask of timeout task is also an empty tensor
                nvtx.mark(f"wait task_ids: {task_ids} timeout")
                return return_masks
            time.sleep(0.0001)
        nvtx.mark(f"wait task_ids: {task_ids} done")
        return return_masks

    # the try_wait api is used for server-client mode:
    # server process running the kvmanager should NOT be blocked by any single client
    def try_wait(self, task_ids: Union[int, List[int]]) -> Dict[int, torch.Tensor]:
        # Trace the wait request
        self.tracer.trace_wait_request(
            wait_type="try_wait",
            task_ids=task_ids,
        )
        return_masks: Dict[int, torch.Tensor] = {}
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        for task_id in task_ids:
            if task_id not in self.requests_tracker:
                flexkv_logger.error(f"task_id {task_id} not submitted into flexKV")
                return_masks[task_id] = torch.empty(0) #if not found in tracker, the return mask is an empty tensor
                continue
            task_tracker = self.requests_tracker[task_id]
            if len(task_tracker.task_end_ops_ids) == 0:
                mask = None
            elif all(task_tracker.task_end_ops_status):
                mask = task_tracker.return_mask
                return_masks[task_id] = mask
            else:
                mask = None

        return return_masks

    def wait_at_layer_group(self, task_id: int, layer_group_id: int, timeout: float = 20.0) -> torch.Tensor:
        # Trace the wait request
        self.tracer.trace_wait_request(
            wait_type="wait_at_layer_group",
            task_ids=task_id,
            layer_group_id=layer_group_id
        )
        nvtx.mark(f"wait task_id: {task_id}, layer_group_id: {layer_group_id}")
        start_time = time.time()
        while True:
            if task_id not in self.requests_tracker:
                flexkv_logger.error(f"task_id {task_id} not submitted into flexKV")
                return torch.empty(0) #if not found in tracker, the return mask is an empty tensor
            task_tracker = self.requests_tracker[task_id]
            if len(task_tracker.task_end_ops_ids) == 0:
                continue
            if task_tracker.task_end_ops_status[layer_group_id]:
                return task_tracker.return_mask
            if time.time() - start_time > timeout:
                flexkv_logger.warning(f"wait task_id: {task_id}, layer_group_id: {layer_group_id} "
                                      f"timeout, has to return now")
                return torch.empty(0) # return mask of timeout task is an empty tensor
            time.sleep(0.0001)

        # nvtx.mark(f"wait_at_layer_group task_id: {task_id}, layer_group_id: {layer_group_id} done")
        # return return_mask

    def try_wait_at_layer_group(self,
                                task_ids: Union[int, List[int]],
                                layer_group_id: int) -> Dict[int, torch.Tensor]:
        # Trace the wait request
        self.tracer.trace_wait_request(
            wait_type="try_wait_at_layer_group",
            task_ids=task_ids,
            layer_group_id=layer_group_id,
        )
        return_masks: Dict[int, torch.Tensor] = {}
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        for task_id in task_ids:
            if task_id not in self.requests_tracker:
                flexkv_logger.error(f"task_id {task_id} not submitted into flexKV")
                return_masks[task_id] = torch.empty(0) #if not found in tracker, the return mask is an empty tensor
                continue
            task_tracker = self.requests_tracker[task_id]
            if len(task_tracker.task_end_ops_ids) == 0:
                mask = torch.empty(0)
            elif task_tracker.task_end_ops_status[layer_group_id]:
                mask = task_tracker.return_mask
            else:
                mask = torch.empty(0)
            return_masks[task_id] = mask
        return return_masks

    def _verify_Model_Cache_config(self,
                                   model_config: ModelConfig,
                                   cache_config: CacheConfig):
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
