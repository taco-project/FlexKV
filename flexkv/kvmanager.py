from queue import Queue
from typing import List
import threading
import time

import torch

from dataclasses import dataclass
from typing import Dict

from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.cache.cache_engine import GlobalCacheEngine
from flexkv.storage.storage_engine import StorageEngine
from flexkv.transfer.transfer_engine import TransferEngine
from flexkv.common.transfer import DeviceType
from flexkv.common.request import cacheEngineRequestType, cacheEngineRequest

@dataclass
class transferCleanupParams:
    def __init__(self,
                 task_id : int,
                 return_mask : torch.Tensor,
                 block_meta_to_free : Dict[DeviceType, torch.Tensor]):
        self.task_id = task_id
        self.return_mask = return_mask
        self.block_meta_to_free = block_meta_to_free

class KVManager:
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_blocks: List[List[torch.Tensor]] = None):
        self.cache_engine = GlobalCacheEngine(cache_config)
        self.storage_engine = StorageEngine(model_config, cache_config, gpu_blocks)
        self.tp_size = model_config.tp_size
        self.gpu_handles = []
        if gpu_blocks is not None:
            assert len(gpu_blocks) == self.tp_size
            self.gpu_handles = [
                self.storage_engine.get_allocator_handle(DeviceType.GPU, i)
                for i in range(self.tp_size)
            ]
            cpu_handle = self.storage_engine.get_allocator_handle(DeviceType.CPU) if cache_config.enable_cpu else None
            ssd_handle = self.storage_engine.get_allocator_handle(DeviceType.SSD) if cache_config.enable_ssd else None
            self.transfer_engine = TransferEngine(self.gpu_handles, cpu_handle, ssd_handle)

        self.transfer_gid_to_cleanups = {}

        self._task_id_counter = 0
        self.task_queue = Queue()
        self.finished_queue = Queue()
        self.running = True
        self._worker_thread = threading.Thread(target=self._worker_loop)
        self._worker_thread.start()

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
            self.transfer_engine = TransferEngine(self.gpu_handles,
                                                  cpu_handle,
                                                  ssd_handle)

    def _worker_loop(self):
        while self.running:
            # deal with completed requests from the cache engine
            if not self.task_queue.empty():
                request = self.task_queue.get()
                if request is None:
                    break
                elif request.request_type == cacheEngineRequestType.GET:
                    graph, return_mask = self.cache_engine.get(request.token_ids,
                                                               request.token_mask,
                                                               request.slot_mapping)
                elif request.request_type == cacheEngineRequestType.PUT:
                    graph, return_mask = self.cache_engine.put(request.token_ids,
                                                               request.token_mask,
                                                               request.slot_mapping)
                else:
                    raise ValueError(f"Unknown request type: {request.request_type}")
                self.transfer_engine.submit_transfer_graph(graph)
                self.transfer_gid_to_cleanups[graph.transfer_graph_id] = transferCleanupParams(
                        request.request_id, return_mask, graph.get_block_meta_to_free())
            completed_graph_ids = self.transfer_engine.get_completed_graphs(timeout=0.001)
            for completed_graph_id in completed_graph_ids:
                cleanup_params = self.transfer_gid_to_cleanups[completed_graph_id]
                task_id = cleanup_params.task_id
                self.cache_engine.cleanup_engines(cleanup_params.block_meta_to_free)
                self.finished_queue.put(
                    (task_id, cleanup_params.return_mask)
                )
                self.transfer_gid_to_cleanups.pop(completed_graph_id)
            time.sleep(0.001)

    def _get_task_id(self) -> int:
        self._task_id_counter += 1
        return self._task_id_counter

    def shutdown(self):
        self.running = False
        self._worker_thread.join()
        self.task_queue.put(None)
        self.transfer_engine.shutdown()

    def get_async(self,
                  token_ids: torch.Tensor,
                  slot_mapping: torch.Tensor,
                  token_mask: torch.Tensor = None) -> int:
        if token_mask is None:
            token_mask = torch.ones_like(token_ids)
        task_id = self._get_task_id()
        self.task_queue.put(cacheEngineRequest(
            request_type=cacheEngineRequestType.GET,
            request_id=task_id,
            token_ids=token_ids,
            token_mask=token_mask,
            slot_mapping=slot_mapping
        ))
        return task_id

    def put_async(self,
                  token_ids: torch.Tensor,
                  slot_mapping: torch.Tensor,
                  token_mask: torch.Tensor = None) -> int:
        if token_mask is None:
            token_mask = torch.ones_like(token_ids)
        task_id = self._get_task_id()
        self.task_queue.put(cacheEngineRequest(
            request_type=cacheEngineRequestType.PUT,
            request_id=task_id,
            token_ids=token_ids,
            token_mask=token_mask,
            slot_mapping=slot_mapping
        ))
        return task_id

    def wait(self, task_ids: List[int]) -> Dict[int, torch.Tensor]:
        num_completed_tasks = 0
        return_masks = {}
        while num_completed_tasks < len(task_ids):
            completed_task_id, return_mask = self.finished_queue.get()
            if completed_task_id in task_ids:
                num_completed_tasks += 1
                return_masks[completed_task_id] = return_mask
            else:
                self.finished_queue.put((completed_task_id, return_mask))
            time.sleep(0.001)
        return return_masks
