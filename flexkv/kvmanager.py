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

        # Multi-instance identifiers: read from env-driven globals.
        # ``instance_id`` is per-process (set via FLEXKV_INSTANCE_ID); ``instance_num``
        # comes from the model config (FLEXKV_INSTANCE_NUM is mirrored there at parse time).
        self.instance_id = GLOBAL_CONFIG_FROM_ENV.instance_id
        self.instance_num = model_config.instance_num

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
            # In server_client_mode, RedisMeta is created and initialized inside KVServer
            # Server should only be created once across all instances and dp ranks
            if self.instance_id == 0 and dp_client_id == 0:
                total_clients = self.instance_num * model_config.dp_size
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
            self.kv_task_engine = KVTaskEngine(self.model_config, self.cache_config, self.gpu_register_port, redis_meta=self.redis_meta_client, event_collector=event_collector)

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

    def get_match_swa(self,
                      token_ids: Union[torch.Tensor, np.ndarray],
                      full_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
                      swa_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
                      cpu_only: bool = False,
                      namespace: Optional[List[str]] = None,
                      update_state_for_load: bool = True,
                      ) -> Tuple[int, np.ndarray, np.ndarray]:
        """Dual-mask SWA-aware match — deployment-dispatch wrapper.

        Mirrors :meth:`get_match`'s wrapper exactly: convert tensors to numpy and
        dispatch on ``server_client_mode``. ``full_mask`` is the original
        token_mask (1 = token not on GPU full pool, needs transfer); ``swa_mask``
        marks tokens missing on the GPU SWA pool.  Returns
        ``(task_id, return_mask_full, return_mask_swa)``; the matching itself lives
        in :meth:`KVTaskEngine.get_match_swa`.

        Both deployment forms are supported (this is the SWA analogue of
        ``get_match``'s ``if server_client_mode`` branch): in-process calls
        ``kv_task_engine.get_match_swa``; server mode forwards over the SWA RPC
        (``dp_client.get_match_swa`` → ``GetMatchSwaRequest`` →
        ``_handle_get_match_swa_request``).  If the RPC fails, it falls back to a
        no-hit reply (matching ``get_match``'s None-on-error tolerance) rather than
        crashing the scheduler.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if isinstance(full_mask, torch.Tensor):
            full_mask = full_mask.numpy()
        if isinstance(swa_mask, torch.Tensor):
            swa_mask = swa_mask.numpy()
        if self.server_client_mode:
            result = self.dp_client.get_match_swa(
                token_ids,
                full_mask,
                swa_mask,
                cpu_only=cpu_only,
                namespace=namespace,
                update_state_for_load=update_state_for_load,
            )
            if result is None:
                # RPC failed on the server; degrade to a no-hit reply so the
                # caller never unpacks None (same tolerance as get_match).
                empty = np.zeros_like(token_ids, dtype=np.bool_)
                return -1, empty, empty
            return result
        return self.kv_task_engine.get_match_swa(
            token_ids=token_ids,
            full_mask=full_mask,
            swa_mask=swa_mask,
            cpu_only=cpu_only,
            namespace=namespace,
            update_state_for_load=update_state_for_load,
        )

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
               swa_slot_mappings: Optional[Union[np.ndarray, List[Optional[np.ndarray]], torch.Tensor, List[Optional[torch.Tensor]]]] = None,
               as_batch: bool = False,
               layerwise_transfer: bool = False,
               counter_id: int = 0) -> List[int]:
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if not isinstance(slot_mappings, List):
            slot_mappings = [slot_mappings]
        if isinstance(slot_mappings[0], torch.Tensor):
            slot_mappings = [slot_mapping.numpy() for slot_mapping in slot_mappings]
        # SWA GPU slot_mappings (optional): the connector supplies these only when
        # it registered an SWA GPU pool and the request has an SWA reuse window.
        if swa_slot_mappings is not None and not isinstance(swa_slot_mappings, List):
            swa_slot_mappings = [swa_slot_mappings]
        if isinstance(swa_slot_mappings, List):
            swa_slot_mappings = [
                sm.numpy() if isinstance(sm, torch.Tensor) else sm
                for sm in swa_slot_mappings
            ]
        if self.server_client_mode:
            # server mode: forward SWA only if the RPC client supports it, else
            # degrade to full-KV launch (SWA stays CPU-resident) rather than crash.
            try:
                return self.dp_client.launch_tasks(task_ids, slot_mappings, as_batch,
                                                   layerwise_transfer, counter_id,
                                                   swa_slot_mappings=swa_slot_mappings)
            except TypeError:
                return self.dp_client.launch_tasks(task_ids, slot_mappings, as_batch,
                                                   layerwise_transfer, counter_id)
        else:
            return self.kv_task_engine.launch_tasks(
                task_ids,
                slot_mappings,
                swa_slot_mappings=swa_slot_mappings,
                as_batch=as_batch,
                layerwise_transfer=layerwise_transfer,
                counter_id=counter_id,
            )

    def cancel(self, task_ids: Union[int, List[int]]) -> None:
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if self.server_client_mode:
            # KVDPClient exposes the singular `cancel_task(List[int])`, not
            # `cancel_tasks`. The plural form was a typo that never fired
            # under prior workloads because release_load_state was only
            # called on rare alloc-rollback paths.
            self.dp_client.cancel_task(task_ids)
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

    # ===== SWA (Sliding Window Attention) =====
    #
    # SWA is NODE-MOUNTED on the Full-KV radix tree and matched via
    # ``get_match_swa`` (see above); SWA bytes move through the FlexKV transfer
    # engine (data plane). The legacy standalone-index / blob API (SWAIndex,
    # swa_put / swa_get / swa_available + node_swa_ops + SWAProductionManager)
    # has been removed. See deployments/swa_design/08_节点挂载SWA架构.md.

    def _get_cpu_cache_engine(self):
        """Return the in-process CPU cache engine (owns the radix tree + SWA pool)."""
        if self.server_client_mode:
            return None
        try:
            return self.kv_task_engine.cache_engine.cpu_cache_engine
        except AttributeError:
            return None

    def swa_unavailable_reason(self) -> Optional[str]:
        """Diagnostic: why SWA matching is unavailable, or None if it should work."""
        if self.cache_config.swa is None or not self.cache_config.swa.enabled:
            return "SWAPoolConfig is None or disabled"
        engine = self._get_cpu_cache_engine()
        if engine is None:
            return "cpu_cache_engine missing (no in-process engine)"
        if not getattr(engine, "swa_enabled", False):
            return "SWA host pool not initialized (call init_swa)"
        return None
