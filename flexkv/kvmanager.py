import multiprocessing as mp
import threading
import time
from dataclasses import dataclass
from queue import Queue
from typing import Dict, Mapping
from typing import List, Optional, Callable, Union

import nvtx
import torch

from flexkv.cache.cache_engine import GlobalCacheEngine, TransferOpGraph
from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.common.debug import init_logger, debuginfo
from flexkv.common.expiring_dict import ExpiringDict
from flexkv.common.memory_handle import KVCacheTensorHandle
from flexkv.common.request import cacheEngineRequestType, cacheEngineRequest
from flexkv.common.transfer import DeviceType, get_nvtx_range_color, get_nvtx_default_color
from flexkv.storage.storage_engine import StorageEngine
from flexkv.transfer.transfer_engine import TransferEngine


logger = init_logger(__name__)

@dataclass
class RequestTracker:
    task_id: int
    task_type: cacheEngineRequestType
    return_mask: torch.Tensor
    callback: Callable
    task_end_ops_ids: List[int]
    task_end_ops_status: List[bool]
    task_finished: bool = False


class KVManager:
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_blocks: Dict[int, List[torch.Tensor]] = None):

        mp.set_start_method('spawn', force=True)
        nvtx.push_range("Initialize kvmanager", color=get_nvtx_default_color())

        if not cache_config.enable_cpu:
            raise ValueError("enable_cpu must be True")
        if cache_config.enable_remote and not cache_config.enable_ssd:
            raise ValueError("enable_ssd must be True if enable_remote is True")
        if not cache_config.enable_cpu and not cache_config.use_gds:
            raise ValueError("use_gds must be True if enable_cpu is False")

        self.cache_config = cache_config
        self.model_config = model_config
        self.gpu_num = 0
        self.cache_engine = GlobalCacheEngine(cache_config, model_config)
        self.running = False

        # Two initialization paths:
        # 1) gpu_blocks are provided
        # 2) gpu_handles are provided
        if gpu_blocks is None:
            self.gpu_blocks = {}
            self.transfer_engine = None
            self.gpu_num = 0
            return

        self.gpu_num = len(gpu_blocks)
        self._init_after_gpu_blocks_added(gpu_blocks)

    # Note that for now only after all the gpu blocks are added, we can initialize the transfer engine
    def _init_after_gpu_blocks_added(self, gpu_blocks: Dict[int, List[torch.Tensor]]):

        assert len(gpu_blocks) == self.model_config.tp_size * self.model_config.dp_size
        self.storage_engine = StorageEngine(self.model_config, self.cache_config, gpu_blocks)
        self.gpu_handles = [
            self.storage_engine.get_allocator_handle(DeviceType.GPU, i)
            for i in range(self.model_config.tp_size * self.model_config.dp_size)
        ]
        cpu_handle = self.storage_engine.get_allocator_handle(DeviceType.CPU) if self.cache_config.enable_cpu else None
        ssd_handle = self.storage_engine.get_allocator_handle(DeviceType.SSD) if self.cache_config.enable_ssd else None
        remote_handle = (
            self.storage_engine.get_allocator_handle(DeviceType.REMOTE)
            if self.cache_config.enable_remote
            else None
        )
        self.transfer_engine = TransferEngine(self.gpu_handles,
                                              self.model_config,
                                              self.cache_config,
                                              cpu_handle,
                                              ssd_handle,
                                              remote_handle)

        self.requests_tracker: Mapping[int, RequestTracker] = ExpiringDict(expire_seconds=600)

        self.taskid_to_nvtx_range = {}
        self.graphid_to_nvtx_range = {}

        self._task_id_counter = 0
        self.task_queue: Queue[cacheEngineRequest] = Queue()

        self.running = True
        self._worker_thread = threading.Thread(target=self._worker_loop)
        self._worker_thread.start()
        self.lock = threading.Lock()

        nvtx.pop_range()

    # the gpu_blocks of multiple gpus can be added post initialization.
    # the transfer engine will be initialized after we have all the intended gpu handles.
    def add_single_gpu_blocks(
        self,
        gpu_handle: Union[List[KVCacheTensorHandle], List[torch.Tensor]],
        dp_client_id: int = 0,
        tp_rank: int = 0,
    ):
        if self.transfer_engine is not None:
            raise ValueError("we have already get all gpu blocks")
        self.gpu_blocks[tp_rank + dp_client_id * self.model_config.tp_size] = gpu_handle
        self.gpu_num += 1
        if self.gpu_num == self.model_config.tp_size * self.model_config.dp_size:
            self._init_after_gpu_blocks_added(self.gpu_blocks)

    def _worker_loop(self):
        while self.running:
            # deal with completed requests from the cache engine
            if not self.task_queue.empty():
                request = self.task_queue.get()
                if request is None:
                    break
                elif request.request_type == cacheEngineRequestType.GET:
                    nvtx.push_range(f"cache_engine.get request_id: {request.request_id}",
                                    color=get_nvtx_default_color())
                    graph, return_mask, callback, task_end_ops_ids = self.cache_engine.get(request.request_id,
                                                                                            request.token_ids,
                                                                                            request.token_mask,
                                                                                            request.slot_mapping,
                                                                                            self.model_config.num_layers,
                                                                                            request.layer_granularity)
                elif request.request_type == cacheEngineRequestType.PUT:
                    nvtx.push_range(f"cache_engine.put request_id: {request.request_id}",
                                    color=get_nvtx_default_color())
                    graph, return_mask, callback, task_end_ops_ids = self.cache_engine.put(request.request_id,
                                                                                        request.token_ids,
                                                                                        request.token_mask,
                                                                                        request.slot_mapping,
                                                                                        self.model_config.num_layers)
                else:
                    raise ValueError(f"Unknown request type: {request.request_type}")
                graph.bind_to_dp_group(request.dp_id)  # TODO: should call this here or in get/put?
                nvtx.pop_range()
                if graph.num_ops == 0: #early return
                    debuginfo.info(f"no transfer: "
                                   f"request_id = {request.request_id}, request_type = {request.request_type}")
                    layer_op_num = self.model_config.num_layers // request.layer_granularity \
                        if request.request_type == cacheEngineRequestType.GET else 1
                    self.requests_tracker[request.request_id] = RequestTracker(task_id=request.request_id,
                                                                            task_type=request.request_type,
                                                                            return_mask=return_mask,
                                                                            callback=None,
                                                                            task_end_ops_ids=[-1]*layer_op_num,
                                                                            task_end_ops_status=[True]*layer_op_num,
                                                                            task_finished=True)
                else:
                    self.graphid_to_nvtx_range[graph.transfer_graph_id] = nvtx.start_range(
                                                                            f"request id: {request.request_id}, "
                                                                            f"graph id: {graph.transfer_graph_id}",
                                                                            color=get_nvtx_range_color(graph.transfer_graph_id))
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
                request_tracker = self.requests_tracker[completed_graph_id]
                if completed_op_id == -1:
                    request_tracker.callback()
                    nvtx.end_range(self.graphid_to_nvtx_range[completed_graph_id])
                    self.graphid_to_nvtx_range.pop(completed_graph_id)
                    nvtx.end_range(self.taskid_to_nvtx_range[request_tracker.task_id])
                    self.taskid_to_nvtx_range.pop(request_tracker.task_id)
                    request_tracker.task_finished = True
                elif completed_op_id in request_tracker.task_end_ops_ids:
                    request_tracker.task_end_ops_status[request_tracker.task_end_ops_ids.index(completed_op_id)] = True
                self.requests_tracker[completed_graph_id] = request_tracker
            time.sleep(0.0001)

    def _get_task_id(self) -> int:
        with self.lock:
            old_value = self._task_id_counter
            self._task_id_counter += 1
            return old_value

    def __del__(self):
        if self.running:
            self.shutdown()

    def shutdown(self):
        self.running = False
        self._worker_thread.join()
        self.task_queue.put(None)
        self.transfer_engine.shutdown()

    def get_async(self,
                  token_ids: torch.Tensor,
                  slot_mapping: torch.Tensor,
                  token_mask: Optional[torch.Tensor] = None,
                  layer_granularity: int = -1,
                  dp_id: int = 0) -> int:
        if token_mask is None:
            token_mask = torch.ones_like(token_ids)
        if layer_granularity == -1:
            layer_granularity = self.model_config.num_layers
        task_id = self._get_task_id()
        nvtx.mark(f"GET request_id: {task_id}")
        self.taskid_to_nvtx_range[task_id] = nvtx.start_range(f"GET request_id: {task_id}",
                                                              color=get_nvtx_default_color())
        self.task_queue.put(cacheEngineRequest(
            request_type=cacheEngineRequestType.GET,
            request_id=task_id,
            token_ids=token_ids,
            token_mask=token_mask,
            slot_mapping=slot_mapping,
            layer_granularity=layer_granularity,
            dp_id=dp_id
        ))
        self.requests_tracker[task_id] = RequestTracker(task_id=task_id,
                                                        task_type=cacheEngineRequestType.GET,
                                                        return_mask=None,
                                                        callback=None,
                                                        task_end_ops_ids=[],
                                                        task_end_ops_status=[],
                                                        task_finished=False)
        return task_id

    def put_async(self,
                  token_ids: torch.Tensor,
                  slot_mapping: torch.Tensor,
                  token_mask: Optional[torch.Tensor] = None,
                  dp_id: int = 0) -> int:
        if token_mask is None:
            token_mask = torch.ones_like(token_ids)
        task_id = self._get_task_id()
        nvtx.mark(f"PUT request_id: {task_id}")
        self.taskid_to_nvtx_range[task_id] = nvtx.start_range(f"PUT request_id: {task_id}",
                                                              color=get_nvtx_default_color())
        self.task_queue.put(cacheEngineRequest(
            request_type=cacheEngineRequestType.PUT,
            request_id=task_id,
            token_ids=token_ids,
            token_mask=token_mask,
            slot_mapping=slot_mapping,
            dp_id=dp_id
        ))
        self.requests_tracker[task_id] = RequestTracker(task_id=task_id,
                                                        task_type=cacheEngineRequestType.PUT,
                                                        return_mask=None,
                                                        callback=None,
                                                        task_end_ops_ids=[],
                                                        task_end_ops_status=[],
                                                        task_finished=False)
        return task_id

    # wait for the key op to be finished
    def wait(self, task_ids: Union[int, List[int]]) -> Dict[int, torch.Tensor]:
        nvtx.mark(f"wait task_ids: {task_ids}")
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        num_completed_tasks = 0
        num_tasks = len(task_ids)
        return_masks = {}
        while num_completed_tasks < num_tasks:
            finished_task_ids = []
            for task_id in task_ids:
                task_tracker = self.requests_tracker[task_id]
                if len(task_tracker.task_end_ops_ids) == 0: #NOT READY
                    continue
                if all(task_tracker.task_end_ops_status):
                    num_completed_tasks += 1
                    return_masks[task_id] = task_tracker.return_mask
                    finished_task_ids.append(task_id)
            task_ids = [task_id for task_id in task_ids if task_id not in finished_task_ids]
            time.sleep(0.0001)
        nvtx.mark(f"wait task_ids: {task_ids} done")
        return return_masks

    # wait for the whole task to be finished, including the key op and all other ops
    # this function is mainly designed for testing to avoid the frequency of writing is too high to use up memory blocks
    def wait_for_graph_finished(self, task_ids: Union[int, List[int]]) -> Dict[int, torch.Tensor]:
        nvtx.mark(f"wait task_ids: {task_ids}")
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        num_completed_tasks = 0
        return_masks = {}
        while num_completed_tasks < len(task_ids):
            finished_task_ids = []
            for task_id in task_ids:
                task_tracker = self.requests_tracker[task_id]
                if task_tracker.task_finished:
                    num_completed_tasks += 1
                    return_masks[task_id] = task_tracker.return_mask
                    finished_task_ids.append(task_id)
            task_ids = [task_id for task_id in task_ids if task_id not in finished_task_ids]
            time.sleep(0.0001)
        nvtx.mark(f"wait task_ids: {task_ids} done")
        return return_masks

    # the try_wait api is used for server-client mode:
    # server process running the kvmanager should NOT be blocked by any single client
    def try_wait(self, task_ids: Union[int, List[int]]) -> Dict[int, torch.Tensor]:
        return_masks = {}
        
        if isinstance(task_ids, int):
            task_ids = [task_ids]
            
        for task_id in task_ids:
            task_tracker = self.requests_tracker[task_id]
            if len(task_tracker.task_end_ops_ids) == 0:
                mask = None
            elif all(task_tracker.task_end_ops_status):
                mask = task_tracker.return_mask
            else:
                mask = None
            
            return_masks[task_id] = mask

        return return_masks

    def wait_at_layer_group(self, task_id: int, layer_group_id: int):
        nvtx.mark(f"wait task_id: {task_id}, layer_group_id: {layer_group_id}")
        while True:
            task_tracker = self.requests_tracker[task_id]
            if len(task_tracker.task_end_ops_ids) == 0:
                continue
            if task_tracker.task_end_ops_status[layer_group_id]:
                return task_tracker.return_mask
            time.sleep(0.0001)

        # nvtx.mark(f"wait_at_layer_group task_id: {task_id}, layer_group_id: {layer_group_id} done")
        # return return_mask

    def try_wait_at_layer_group(self, task_ids: Union[int, List[int]], layer_group_id: int)->Dict[int, torch.Tensor]:
        return_masks = {}
        
        if isinstance(task_ids, int):
            task_ids = [task_ids]
            
        for task_id in task_ids:
            task_tracker = self.requests_tracker[task_id]
            if len(task_tracker.task_end_ops_ids) == 0:
                mask = None
            elif task_tracker.task_end_ops_status[layer_group_id]:
                mask = task_tracker.return_mask
            else:
                mask = None
            
            return_masks[task_id] = mask

        return return_masks
