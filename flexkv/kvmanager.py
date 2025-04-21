from queue import Queue
from typing import List
import threading
import time

import torch

from flexkv.common.config import CacheConfig, StorageConfig
from flexkv.cache.cache_engine import GlobalCacheEngine
from flexkv.storage.storage_engine import StorageEngine
from flexkv.transfer.transfer_engine import TransferEngine


class KVManager:
    def __init__(self,
                 cache_config: CacheConfig,
                 storage_config: StorageConfig):
        self.cache_engine = GlobalCacheEngine(cache_config)
        self.storage_engine = StorageEngine(storage_config)
        self.transfer_engine = TransferEngine(storage_config)

        self.transfer_graph_id_to_task_id = {}

        self._task_id_counter = 0
        self.task_queue = Queue()
        self.finished_queue = Queue()

        self._worker_thread = threading.Thread(target=self._worker_loop)
        self._worker_thread.start()

    def _worker_loop(self):
        while True:
            completed_graph_ids = self.transfer_engine.get_completed_graphs(timeout=0.001)
            for completed_graph_id in completed_graph_ids:
                task_id = self.transfer_graph_id_to_task_id[completed_graph_id]
                # TODO: unlock block metas in index
                self.finished_queue.put(task_id)
            time.sleep(0.0001)

    def _get_task_id(self) -> int:
        self._task_id_counter += 1
        return self._task_id_counter

    def shutdown(self):
        self._worker_thread.join()

    def get_async(self,
                  token_ids: torch.Tensor,
                  token_mask: torch.Tensor,
                  slot_mapping: torch.Tensor) -> int:
        task_id = self._get_task_id()
        transfer_op_graph, return_mask = self.cache_engine.get(token_ids, token_mask, slot_mapping)
        self.task_queue.put((task_id, transfer_op_graph.transfer_graph_id))
        self.transfer_engine.submit_transfer_graph(transfer_op_graph)
        self.transfer_graph_id_to_task_id[transfer_op_graph.transfer_graph_id] = task_id
        return task_id

    def put_async(self,
                  token_ids: torch.Tensor,
                  token_mask: torch.Tensor,
                  slot_mapping: torch.Tensor) -> int:
        task_id = self._get_task_id()
        transfer_op_graph, return_mask = self.cache_engine.put(token_ids, token_mask, slot_mapping)
        self.task_queue.put((task_id, transfer_op_graph.transfer_graph_id))
        self.transfer_engine.submit_transfer_graph(transfer_op_graph)
        self.transfer_graph_id_to_task_id[transfer_op_graph.transfer_graph_id] = task_id
        return task_id

    def wait(self, task_ids: List[int]) -> None:
        num_completed_tasks = 0
        while num_completed_tasks < len(task_ids):
            completed_task_id = self.finished_queue.get()
            if completed_task_id in task_ids:
                num_completed_tasks += 1
            else:
                self.finished_queue.put(completed_task_id)
