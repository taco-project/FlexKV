from queue import Queue
from typing import List, Optional, Callable
import threading
import time

import torch
import nvtx

from dataclasses import dataclass
from typing import Dict

from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.cache.cache_engine import GlobalCacheEngine
from flexkv.storage.storage_engine import StorageEngine
from flexkv.transfer.transfer_engine import TransferEngine
from flexkv.common.transfer import DeviceType, get_nvtx_range_color, get_nvtx_default_color
from flexkv.common.request import cacheEngineRequestType, cacheEngineRequest

@dataclass
class TaskDescriptor:
    task_id: int
    return_mask: torch.Tensor
    callback: Callable

class KVManager:
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_blocks: List[List[torch.Tensor]] = None):
        nvtx.push_range("KVManager.__init__", color=get_nvtx_default_color())
        self.cache_engine = GlobalCacheEngine(cache_config, model_config)
        self.storage_engine = StorageEngine(model_config, cache_config, gpu_blocks)
        self.tp_size = model_config.tp_size
        self.gpu_handles = []
        self._model_config = model_config
        if gpu_blocks is not None:
            assert len(gpu_blocks) == self.tp_size
            self.gpu_handles = [
                self.storage_engine.get_allocator_handle(DeviceType.GPU, i)
                for i in range(self.tp_size)
            ]
            cpu_handle = self.storage_engine.get_allocator_handle(DeviceType.CPU) if cache_config.enable_cpu else None
            ssd_handle = self.storage_engine.get_allocator_handle(DeviceType.SSD) if cache_config.enable_ssd else None
            remote_handle = self.storage_engine.get_allocator_handle(DeviceType.REMOTE) if cache_config.enable_remote else None
            self.transfer_engine = TransferEngine(self.gpu_handles, cpu_handle, ssd_handle, remote_handle)

        self.transfer_gid_to_task = {}
        self.taskid_to_layerwise_ops = {}

        self.taskid_to_nvtx_range = {}
        self.graphid_to_nvtx_range = {}

        self._task_id_counter = 0
        self.task_queue = Queue()
        self.finished_queue = Queue()
        self.finished_ops_queue = Queue()
        self.running = True
        self._worker_thread = threading.Thread(target=self._worker_loop)
        self._worker_thread.start()
        self.lock = threading.Lock()

        nvtx.pop_range()

    # the gpu_blocks of multiple gpus can be added post initialization.
    # the transfer engine will be initialized after we have all the intended gpu handles.
    # NOTE: this add function must be called in the order of device_id for now
    def add_single_gpu_blocks(self, gpu_blocks: List[torch.Tensor], device_id: int = 0):
        if self.transfer_engine is not None:
            raise ValueError("we have already get all gpu blocks")
        self.storage_engine.add_single_gpu_blocks(gpu_blocks, device_id)
        gpu_handle = self.storage_engine.get_allocator_handle(DeviceType.GPU, device_id)
        self.gpu_handles.append(gpu_handle)
        if len(self.gpu_handles) == self.tp_size:
            cpu_handle = (
                self.storage_engine.get_allocator_handle(DeviceType.CPU)
                if self.cache_config.enable_cpu
                else None
            )
            ssd_handle = (
                self.storage_engine.get_allocator_handle(DeviceType.SSD)
                if self.cache_config.enable_ssd
                else None
            )
            remote_handle = (
                self.storage_engine.get_allocator_handle(DeviceType.REMOTE)
                if self.cache_config.enable_remote
                else None   
            )
            self.transfer_engine = TransferEngine(self.gpu_handles,
                                                  cpu_handle,
                                                  ssd_handle,
                                                  remote_handle)

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
                    graph, return_mask, callback, finished_ops_ids = self.cache_engine.get(request.token_ids,
                                                               request.token_mask,
                                                               request.slot_mapping,
                                                               self._model_config.num_layers,
                                                               request.layer_granularity)
                    if request.layer_granularity != -1:
                        self.taskid_to_layerwise_ops[request.request_id] = finished_ops_ids
                elif request.request_type == cacheEngineRequestType.PUT:
                    nvtx.push_range(f"cache_engine.put request_id: {request.request_id}",
                                    color=get_nvtx_default_color())
                    graph, return_mask, callback, finished_ops_ids = self.cache_engine.put(request.token_ids,
                                                                            request.token_mask,
                                                                            request.slot_mapping,
                                                                            self._model_config.num_layers)

                else:
                    raise ValueError(f"Unknown request type: {request.request_type}")
                nvtx.pop_range()

                self.graphid_to_nvtx_range[graph.transfer_graph_id] = nvtx.start_range(
                                                                            f"request id: {request.request_id}, "
                                                                            f"graph id: {graph.transfer_graph_id}",
                                                                            color=get_nvtx_range_color(graph.transfer_graph_id))
                self.transfer_engine.submit_transfer_graph(graph)
                self.transfer_gid_to_task[graph.transfer_graph_id] = TaskDescriptor(
                        request.request_id, return_mask, callback)
            results = self.transfer_engine.get_completed_graphs_and_ops(timeout=0.001)
            for completed_graph_id, completed_op_id in results:
                task_descriptor = self.transfer_gid_to_task[completed_graph_id]
                task_id = task_descriptor.task_id
                if completed_op_id == -1:
                    task_descriptor.callback()
                    nvtx.end_range(self.graphid_to_nvtx_range[completed_graph_id])
                    self.graphid_to_nvtx_range.pop(completed_graph_id)
                    self.finished_queue.put( #TODO do not return "return_mask" everytime
                        (task_id, completed_op_id, task_descriptor.return_mask)
                    )
                    self.transfer_gid_to_task.pop(completed_graph_id)
                    nvtx.end_range(self.taskid_to_nvtx_range[task_id])
                    self.taskid_to_nvtx_range.pop(task_id)
                else:
                    self.finished_ops_queue.put(
                        (task_id, completed_op_id, task_descriptor.return_mask)
                    )
            time.sleep(0.001)

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
                  layer_granularity: int = -1) -> int:
        if token_mask is None:
            token_mask = torch.ones_like(token_ids)
        if layer_granularity == -1:
            layer_granularity = self._model_config.num_layers
        task_id = self._get_task_id()
        self.taskid_to_nvtx_range[task_id] = nvtx.start_range(f"GET request_id: {task_id}",
                                                              color=get_nvtx_default_color())
        self.task_queue.put(cacheEngineRequest(
            request_type=cacheEngineRequestType.GET,
            request_id=task_id,
            token_ids=token_ids,
            token_mask=token_mask,
            slot_mapping=slot_mapping,
            layer_granularity=layer_granularity
        ))
        self.taskid_to_layerwise_ops[task_id] = []
        return task_id

    def put_async(self,
                  token_ids: torch.Tensor,
                  slot_mapping: torch.Tensor,
                  token_mask: Optional[torch.Tensor] = None) -> int:
        if token_mask is None:
            token_mask = torch.ones_like(token_ids)
        task_id = self._get_task_id()
        self.taskid_to_nvtx_range[task_id] = nvtx.start_range(f"PUT request_id: {task_id}",
                                                              color=get_nvtx_default_color())
        self.task_queue.put(cacheEngineRequest(
            request_type=cacheEngineRequestType.PUT,
            request_id=task_id,
            token_ids=token_ids,
            token_mask=token_mask,
            slot_mapping=slot_mapping
        ))
        return task_id

    #NOTE the wait() function and wait_at_layer() function should not be called at the same time,
    # because they clean each other
    #NOTE every task should be synced in some way, otherwise the queue will be full
    def wait(self, task_ids: List[int]) -> Dict[int, torch.Tensor]:
        num_completed_tasks = 0
        return_masks = {}
        while num_completed_tasks < len(task_ids):
            completed_task_id, op_id, return_mask = self.finished_queue.get()
            if completed_task_id in task_ids:
                if op_id == -1:
                    num_completed_tasks += 1
                    return_masks[completed_task_id] = return_mask
                    if completed_task_id in self.taskid_to_layerwise_ops:
                        self.taskid_to_layerwise_ops.pop(completed_task_id)
            else:
                self.finished_queue.put((completed_task_id, op_id, return_mask))
            time.sleep(0.001)
        return return_masks

    def wait_at_layer_group(self, task_id: int, layer_group_id: int, last_layer: bool = False):
        while True:
            completed_task_id, op_id, return_mask = self.finished_ops_queue.get()
            if completed_task_id == task_id:
                if op_id == self.taskid_to_layerwise_ops[task_id][layer_group_id]:
                    if last_layer:
                        self.wait([task_id])
                    return return_mask
                elif op_id == -1:
                    raise ValueError("should not happen since we are waiting at some layer")
                elif op_id < layer_group_id:
                    continue
            else:
                self.finished_ops_queue.put((completed_task_id, op_id, return_mask))
            time.sleep(0.001)
