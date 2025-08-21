from typing import Optional, Tuple, List, Dict, Union, Iterable
import time

import numpy as np
import torch

from flexkv.server.client import KVDPClient
from flexkv.server.server import KVServer, DPClient
from flexkv.kvtask import KVTaskEngine, KVResponse
from flexkv.common.config import ModelConfig, CacheConfig
from flexkv.common.debug import flexkv_logger


class KVManager:
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 gpu_register_port: Optional[str] = None,
                 server_recv_port: Optional[str] = None):
        flexkv_logger.info(f"{model_config = }")
        flexkv_logger.info(f"{cache_config = }")
        self.model_config = model_config
        self.cache_config = cache_config
        self.gpu_register_port = gpu_register_port
        self.server_recv_port = server_recv_port
        self.server_client_mode = model_config.dp_size > 1 # True #just for test
        flexkv_logger.info(f"server_client_mode: {self.server_client_mode}")
        if self.server_client_mode:
            self.server_handle = KVServer.create_server(model_config, cache_config, gpu_register_port, server_recv_port)
            self.dp_client = KVDPClient(self.server_recv_port, self.model_config)
        else:
            self.server_handle = None
            self.kv_task_engine = KVTaskEngine(model_config, cache_config, gpu_register_port)

    #def _launch_server(self) -> None:
    #    self.server = KVServer(self.model_config, self.cache_config, self.server_recv_port)
    #    self.server.run()
    #    time.sleep(10)
    #    self.dp_client = DPClient(self.server_recv_port, self.model_config)

    def start(self) -> None:
        if not self.server_client_mode:
            self.kv_task_engine.start()
        # for server client mode, we need to do nothing, because the start is actually called
        # when the server is created

    def is_ready(self) -> bool:
        if self.server_client_mode:
            return self.server_handle is not None and self.server_handle.ready_event.is_set()
        else:
            return self.kv_task_engine.is_ready()

    def shutdown(self) -> None:
        if self.server_client_mode:
            if self.server_handle is not None:
                self.server_handle.shutdown()
            else:
                flexkv_logger.error("Shutdown server failed, server is not created")
        else:
            self.kv_task_engine.shutdown()

    def get_async(self,
                  token_ids: Union[torch.Tensor, np.ndarray],
                  slot_mapping: Union[torch.Tensor, np.ndarray],
                  token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  layer_granularity: int = -1,
                  dp_id: int = 0,
                  ) -> int:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if isinstance(slot_mapping, torch.Tensor):
            slot_mapping = slot_mapping.numpy()
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.numpy()
        if self.server_client_mode:
            task_id = self.dp_client.get_async(token_ids,
                                            slot_mapping,
                                            token_mask,
                                            layer_granularity,
                                            dp_id)
        else:
            task_id, _ = self.kv_task_engine.get_async(token_ids,
                                                       slot_mapping,
                                                       token_mask,
                                                       layer_granularity,
                                                       dp_id)
        return task_id

    def get_match(self,
                  token_ids: Union[torch.Tensor, np.ndarray],
                  token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  layer_granularity: int = -1,
                  dp_id: int = 0,
                  ) -> Tuple[int, np.ndarray]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.numpy()
        if self.server_client_mode:
            task_id, mask = self.dp_client.get_match(token_ids,
                                                     token_mask,
                                                     layer_granularity,
                                                     dp_id)
        else:
            task_id, mask = self.kv_task_engine.get_match(token_ids,
                                                          token_mask,
                                                          layer_granularity,
                                                          dp_id)
        mask = torch.from_numpy(mask) if mask is not None else None
        return task_id, mask

    def put_async(self,
                  token_ids: Union[torch.Tensor, np.ndarray],
                  slot_mapping: Union[torch.Tensor, np.ndarray],
                  token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  dp_id: int = 0,
                  ) -> int:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if isinstance(slot_mapping, torch.Tensor):
            slot_mapping = slot_mapping.numpy()
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.numpy()
        if self.server_client_mode:
            task_id = self.dp_client.put_async(token_ids, slot_mapping, token_mask, dp_id)
        else:
            task_id, _ = self.kv_task_engine.put_async(token_ids, slot_mapping, token_mask, dp_id)
        return task_id

    def put_match(self,
                  token_ids: Union[torch.Tensor, np.ndarray],
                  token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  dp_id: int = 0,
                  ) -> Tuple[int, np.ndarray]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.numpy()
        if self.server_client_mode:
            task_id, mask = self.dp_client.put_match(token_ids, token_mask, dp_id)
        else:
            task_id, mask = self.kv_task_engine.put_match(token_ids, token_mask, dp_id)
        mask = torch.from_numpy(mask) if mask is not None else None
        return task_id, mask

    def launch(self,
               task_ids: Union[int, List[int]],
               slot_mappings: Union[np.ndarray, List[np.ndarray]]) -> None:
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if not isinstance(slot_mappings, List):
            slot_mappings = [slot_mappings]
        if isinstance(slot_mappings[0], torch.Tensor):
            slot_mappings = [slot_mapping.numpy() for slot_mapping in slot_mappings]
        if self.server_client_mode:
            for task_id in task_ids:
                self.dp_client.launch_task(task_id, slot_mappings)
        else:
            self.kv_task_engine.launch_transfer(task_ids, slot_mappings)

    def cancel_task(self, task_ids: Union[int, List[int]]) -> None:
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if self.server_client_mode:
            for task_id in task_ids:
                self.dp_client.cancel_task(task_id)
        else:
            for task_id in task_ids:
                self.kv_task_engine.cancel_task(task_id)

    def wait(self,
             task_ids: Union[int, List[int]],
             timeout: float = 20.0,
             completely: bool = False) -> Dict[int, KVResponse]:
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if self.server_client_mode:
            return self.dp_client.wait(task_ids, timeout, completely)
        else:
            return self.kv_task_engine.wait(task_ids, timeout, completely)

    def try_wait(self, task_ids: Union[int, List[int]]) -> Dict[int, KVResponse]:
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if self.server_client_mode:
            return self.dp_client.try_wait(task_ids)
        else:
            return self.kv_task_engine.try_wait(task_ids)
