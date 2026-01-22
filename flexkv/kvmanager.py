# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple, List, Dict, Union, Iterable
import time

import numpy as np
import torch

from flexkv.server.client import KVDPClient
from flexkv.server.server import KVServer, DPClient
from flexkv.kvtask import KVTaskEngine, KVResponse
from flexkv.common.config import ModelConfig, CacheConfig, GLOBAL_CONFIG_FROM_ENV
from flexkv.common.debug import flexkv_logger


class KVManager:
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 dp_client_id: int = 0,
                 server_recv_port: str = "",
                 gpu_register_port: str = ""):
        flexkv_logger.info(f"{model_config = }")
        flexkv_logger.info(f"{cache_config = }")
        flexkv_logger.info(f"{GLOBAL_CONFIG_FROM_ENV = }")
        self.model_config = model_config
        self.cache_config = cache_config
        self.instance_id = GLOBAL_CONFIG_FROM_ENV.instance_id
        self.instance_num = GLOBAL_CONFIG_FROM_ENV.instance_num

        if server_recv_port != "":
            self.server_recv_port = server_recv_port
        else:
            self.server_recv_port = GLOBAL_CONFIG_FROM_ENV.server_recv_port
        if gpu_register_port != "":
            self.gpu_register_port = gpu_register_port
        else:
            self.gpu_register_port = self.server_recv_port + "_gpu_register"

        # Multi-instance mode also requires server_client_mode
        self.server_client_mode = (model_config.dp_size > 1 or 
                                   self.instance_num > 1 or 
                                   GLOBAL_CONFIG_FROM_ENV.server_client_mode)
        self.dp_client_id = dp_client_id
        
        # Calculate global_client_id for multi-instance mode
        self.global_client_id = self.instance_id * model_config.dp_size + dp_client_id
        
        flexkv_logger.info(f"server_client_mode: {self.server_client_mode}, "
                          f"instance_id: {self.instance_id}, dp_client_id: {dp_client_id}, "
                          f"global_client_id: {self.global_client_id}")
        
        if self.server_client_mode:
            # Server should only be created once across all instances and dp ranks
            # Only instance_id == 0 and dp_client_id == 0 creates the server
            if self.instance_id == 0 and dp_client_id == 0:
                # Calculate total clients for all instances
                total_clients = self.instance_num * model_config.dp_size
                # inherit_env=False ensures server can see all GPUs
                self.server_handle = KVServer.create_server(model_config=model_config,
                                                            cache_config=cache_config,
                                                            gpu_register_port=self.gpu_register_port,
                                                            server_recv_port=self.server_recv_port,
                                                            total_clients=total_clients,
                                                            inherit_env=False)

            else:
                self.server_handle = None
            # Use global_client_id for server communication
            self.dp_client = KVDPClient(self.server_recv_port, self.model_config, self.global_client_id)
        else:
            self.server_handle = None
            self.kv_task_engine = KVTaskEngine(model_config, cache_config, self.gpu_register_port)
    @property
    def dpclient_id(self) -> int:
        return self.dp_client_id

    def start(self) -> None:
        if not self.server_client_mode:
            self.kv_task_engine.start()
        else:
            # send the start request to the server
            self.dp_client.start_server_and_register()

    def is_ready(self) -> bool:
        if self.server_client_mode:
            return self.dp_client.is_ready()
        else:
            return self.kv_task_engine.is_ready()

    def shutdown(self) -> None:
        if self.server_client_mode:
            self.dp_client.shutdown()
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
                                               layer_granularity)
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
                                                     layer_granularity)
        else:
            task_id, mask = self.kv_task_engine.get_match(token_ids,
                                                          token_mask,
                                                          layer_granularity,
                                                          dp_id)
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
            task_id = self.dp_client.put_async(token_ids, slot_mapping, token_mask)
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
            task_id, mask = self.dp_client.put_match(token_ids, token_mask)
        else:
            task_id, mask = self.kv_task_engine.put_match(token_ids, token_mask, dp_id)
        return task_id, mask

    def prefetch_async(self,
                       token_ids: np.ndarray,
                       dp_id: int = 0) -> int:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if self.server_client_mode:
            task_id = self.dp_client.prefetch_async(token_ids)
        else:
            task_id = self.kv_task_engine.prefetch_async(token_ids, dp_id=dp_id)
        return task_id

    def launch(self,
               task_ids: Union[int, List[int]],
               slot_mappings: Union[np.ndarray, List[np.ndarray], torch.Tensor, List[torch.Tensor]],
               as_batch: bool = False) -> List[int]:
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if not isinstance(slot_mappings, List):
            slot_mappings = [slot_mappings]
        if isinstance(slot_mappings[0], torch.Tensor):
            slot_mappings = [slot_mapping.numpy() for slot_mapping in slot_mappings]
        if self.server_client_mode:
            return self.dp_client.launch_tasks(task_ids, slot_mappings, as_batch)
        else:
            return self.kv_task_engine.launch_tasks(task_ids, slot_mappings, as_batch)
            
    def cancel(self, task_ids: Union[int, List[int]]) -> None:
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if self.server_client_mode:
            self.dp_client.cancel_tasks(task_ids)
        else:
            self.kv_task_engine.cancel_tasks(task_ids)

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

    # Only for testing
    def _clear_cpu_cache(self) -> None:
        if self.server_client_mode:
            flexkv_logger.error("clear_cache is not supported in server client mode")
            return
        else:
            self.kv_task_engine._clear_cpu_cache()
