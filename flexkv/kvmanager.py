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

import os
import subprocess
from typing import Optional, Tuple, List, Dict, Union, Iterable
import time

import numpy as np
import torch

from flexkv.server.client import KVDPClient
from flexkv.server.server import KVServer, DPClient
from flexkv.kvtask import KVTaskEngine, KVResponse
from flexkv.common.config import ModelConfig, CacheConfig, GLOBAL_CONFIG_FROM_ENV, MooncakeTransferEngineConfig
from flexkv.integration.dynamo.collector import KVEventCollector
from flexkv.common.debug import flexkv_logger
from flexkv.cache.redis_meta import RedisMeta


class KVManager:
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 dp_client_id: int = 0,
                 server_recv_port: str = "",
                 gpu_register_port: str = "",
                 event_collector: Optional[KVEventCollector] = None):
        flexkv_logger.info(f"{model_config = }")
        flexkv_logger.info(f"{cache_config = }")
        self.model_config = model_config
        self.cache_config = cache_config

        if server_recv_port != "":
            self.server_recv_port = server_recv_port
        else:
            self.server_recv_port = GLOBAL_CONFIG_FROM_ENV.server_recv_port
        if gpu_register_port != "":
            self.gpu_register_port = gpu_register_port
        else:
            self.gpu_register_port = self.server_recv_port + "_gpu_register"

        flexkv_logger.info(
            f"[KVManager] IPC ports: server_recv_port={self.server_recv_port}, "
            f"gpu_register_port={self.gpu_register_port}"
        )

        # Multi-instance mode also requires server_client_mode
        self.server_client_mode = (model_config.dp_size > 1 or
                                   model_config.instance_num > 1 or
                                   GLOBAL_CONFIG_FROM_ENV.server_client_mode)

        flexkv_logger.info(
            f"[KVManager] instance_num={model_config.instance_num}, dp_size={model_config.dp_size}, "
            f"server_client_mode={self.server_client_mode}"
        )

        self.redis_meta_client = None
        self.enable_mps = GLOBAL_CONFIG_FROM_ENV.enable_mps

        if self.server_client_mode:
            if dp_client_id == 0:
                self.server_handle = KVServer.create_server(model_config=model_config,
                                                            cache_config=cache_config,
                                                            gpu_register_port=self.gpu_register_port,
                                                            server_recv_port=self.server_recv_port,
                                                            inherit_env=False)

            else:
                self.server_handle = None
            self.dp_client = KVDPClient(
                self.server_recv_port,
                model_config=model_config,
                dp_client_id=dp_client_id,
            )
        else:
            # In non-server_client_mode, create RedisMeta here and pass to KVTaskEngine
            if self.cache_config.enable_kv_sharing:
                flexkv_logger.info(f"[kv manager] initializing RedisMeta and connection to "
                                   f"{self.cache_config.redis_host}:{self.cache_config.redis_port}")
                self.redis_meta_client = RedisMeta(
                    self.cache_config.redis_host,
                    self.cache_config.redis_port,
                    self.cache_config.redis_password,
                    self.cache_config.local_ip,
                    node_ttl_seconds=self.cache_config.node_ttl_seconds,
                )
                self.redis_meta_client.init_meta()
                # update distributed_node_id
                self.cache_config.distributed_node_id = self.redis_meta_client.get_node_id()

            self.server_handle = None
            self.kv_task_engine = KVTaskEngine(
                model_config,
                self.cache_config,
                self.gpu_register_port,
                redis_meta=self.redis_meta_client,
                event_collector=event_collector,
            )

    def start(self) -> None:
        if self.enable_mps:
            # try to start MPS
            subprocess.run(['nvidia-cuda-mps-control', '-d'], check=False)
            flexkv_logger.debug("MPS started")

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
            # Wait for the server process to exit after sending shutdown request
            if self.server_handle is not None:
                self.server_handle.shutdown()
                self.server_handle = None
        else:
            self.kv_task_engine.shutdown()

        if self.enable_mps:
            flexkv_logger.info(
                "MPS is enabled. To stop MPS daemon manually, run: "
                "'echo quit | nvidia-cuda-mps-control'"
            )

    def get_async(self,
                  token_ids: Union[torch.Tensor, np.ndarray],
                  slot_mapping: Union[torch.Tensor, np.ndarray],
                  token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  namespace: Optional[List[str]] = None,
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
                                               namespace=namespace)
        else:
            task_id, _ = self.kv_task_engine.get_async(
                token_ids=token_ids,
                slot_mapping=slot_mapping,
                token_mask=token_mask,
                namespace=namespace,
            )
        return task_id

    def get_match(self,
                  token_ids: Union[torch.Tensor, np.ndarray],
                  token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  cpu_only: bool = False,
                  namespace: Optional[List[str]] = None,
                  ) -> Tuple[int, np.ndarray]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.numpy()
        if self.server_client_mode:
            task_id, mask = self.dp_client.get_match(token_ids,
                                                     token_mask,
                                                     cpu_only=cpu_only,
                                                     namespace=namespace)
        else:
            task_id, mask = self.kv_task_engine.get_match(
                token_ids=token_ids,
                token_mask=token_mask,
                cpu_only=cpu_only,
                namespace=namespace,
            )
        return task_id, mask

    def put_async(self,
                  token_ids: Union[torch.Tensor, np.ndarray],
                  slot_mapping: Union[torch.Tensor, np.ndarray],
                  token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  namespace: Optional[List[str]] = None,
                  ) -> int:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if isinstance(slot_mapping, torch.Tensor):
            slot_mapping = slot_mapping.numpy()
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.numpy()
        if self.server_client_mode:
            task_id = self.dp_client.put_async(token_ids, slot_mapping, token_mask,
                                               namespace=namespace)
        else:
            task_id, _ = self.kv_task_engine.put_async(
                token_ids=token_ids,
                slot_mapping=slot_mapping,
                token_mask=token_mask,
                namespace=namespace,
            )
        return task_id

    def put_match(self,
                  token_ids: Union[torch.Tensor, np.ndarray],
                  token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  namespace: Optional[List[str]] = None,
                  ) -> Tuple[int, np.ndarray]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.numpy()
        if self.server_client_mode:
            task_id, mask = self.dp_client.put_match(token_ids, token_mask,
                                                     namespace=namespace)
        else:
            task_id, mask = self.kv_task_engine.put_match(
                token_ids=token_ids,
                token_mask=token_mask,
                namespace=namespace,
            )
        return task_id, mask

    def prefetch_async(self,
                       token_ids: np.ndarray,
                       namespace: Optional[List[str]] = None) -> int:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if self.server_client_mode:
            task_id = self.dp_client.prefetch_async(token_ids, namespace=namespace)
        else:
            task_id = self.kv_task_engine.prefetch_async(
                token_ids,
                namespace=namespace,
            )
        return task_id

    def launch(self,
               task_ids: Union[int, List[int]],
               slot_mappings: Union[np.ndarray, List[np.ndarray], torch.Tensor, List[torch.Tensor]],
               as_batch: bool = False,
               layerwise_transfer: bool = False,
               counter_id: int = 0) -> List[int]:
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if not isinstance(slot_mappings, List):
            slot_mappings = [slot_mappings]
        if isinstance(slot_mappings[0], torch.Tensor):
            slot_mappings = [slot_mapping.numpy() for slot_mapping in slot_mappings]
        if self.server_client_mode:
            return self.dp_client.launch_tasks(task_ids, slot_mappings, as_batch, layerwise_transfer, counter_id)
        else:
            return self.kv_task_engine.launch_tasks(
                task_ids,
                slot_mappings,
                as_batch=as_batch,
                layerwise_transfer=layerwise_transfer,
                counter_id=counter_id
            )

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

    # ------------------------------------------------------------------
    # SWA (Sliding Window Attention) Pool API
    # ------------------------------------------------------------------

    def swa_put(self,
                token_ids: Union[torch.Tensor, np.ndarray],
                swa_data: Union[torch.Tensor, np.ndarray],
                ) -> bool:
        """Offload an SWA page to the CPU cache.

        Args:
            token_ids: Full token sequence (used to derive endpoint hash for lookup key).
            swa_data: Raw SWA ring-buffer bytes (uint8 tensor/array, shape=[page_size_bytes]).

        Returns:
            True if the page was successfully stored, False otherwise.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()

        if not hasattr(self, '_swa_engine'):
            self._init_swa_pool()

        if self._swa_engine is None:
            return False

        from flexkv.common.hash_utils import HashType
        endpoint_hash = self._compute_swa_endpoint_hash(token_ids)
        slot = self._swa_engine.allocate(endpoint_hash)
        if slot is None:
            return False

        self._swa_storage.write_slot(slot, swa_data)
        self._swa_engine.set_ready(endpoint_hash, True)
        return True

    def swa_get(self,
                token_ids: Union[torch.Tensor, np.ndarray],
                ) -> Optional[np.ndarray]:
        """Restore an SWA page from the CPU cache.

        Uses TRAILING_PAGES hit policy: matches by the endpoint hash
        (hash of the last block in the token sequence).

        Args:
            token_ids: Full token sequence (used to derive endpoint hash).

        Returns:
            SWA page data as numpy array (uint8), or None if not cached.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()

        if not hasattr(self, '_swa_engine'):
            self._init_swa_pool()

        if self._swa_engine is None:
            return None

        endpoint_hash = self._compute_swa_endpoint_hash(token_ids)
        result = self._swa_engine.match(endpoint_hash)
        if not result.hit:
            return None

        data = self._swa_storage.read_slot(result.physical_block)
        if isinstance(data, torch.Tensor):
            return data.numpy()
        return data

    def swa_remove(self,
                   token_ids: Union[torch.Tensor, np.ndarray],
                   ) -> None:
        """Explicitly remove an SWA page from the cache.

        Args:
            token_ids: Full token sequence (endpoint hash derived from this).
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()

        if not hasattr(self, '_swa_engine') or self._swa_engine is None:
            return

        endpoint_hash = self._compute_swa_endpoint_hash(token_ids)
        self._swa_engine.remove(endpoint_hash)

    def _init_swa_pool(self) -> None:
        """Lazily initialize the SWA pool from cache_config.swa."""
        swa_cfg = self.cache_config.swa
        if swa_cfg is None or not swa_cfg.enabled:
            self._swa_engine = None
            self._swa_storage = None
            return

        from flexkv.swa.swa_cache_engine import SWACacheEngine
        from flexkv.swa.swa_storage import SWAStorage, SWAStorageConfig

        self._swa_engine = SWACacheEngine(
            num_slots=swa_cfg.num_slots,
            evict_ratio=swa_cfg.evict_ratio,
        )
        self._swa_storage = SWAStorage(
            SWAStorageConfig.from_pool_config(swa_cfg),
            pin_memory=True,
        )
        flexkv_logger.info(
            f"[KVManager] SWA pool initialized: "
            f"num_slots={swa_cfg.num_slots}, "
            f"page_size={swa_cfg.page_size_bytes} bytes "
            f"({swa_cfg.window_size} tokens x {swa_cfg.num_swa_layers} layers)"
        )

    def _compute_swa_endpoint_hash(self, token_ids: np.ndarray):
        """Compute endpoint hash for SWA lookup.

        Uses the hash of the last tokens_per_block tokens as the key,
        consistent with the main KV pool's RadixTree block hashing.
        """
        from flexkv.common.hash_utils import HashType, Hasher

        tokens_per_block = self.cache_config.tokens_per_block
        if len(token_ids) < tokens_per_block:
            # Short sequence: hash whatever we have
            h = Hasher.hash_tokens(token_ids)
        else:
            # Hash the last full block
            last_block = token_ids[-(len(token_ids) % tokens_per_block or tokens_per_block):]
            if len(last_block) == 0:
                last_block = token_ids[-tokens_per_block:]
            h = Hasher.hash_tokens(last_block)
        return HashType(int(h))
