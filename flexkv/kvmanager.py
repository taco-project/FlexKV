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
from flexkv.common.block import SequenceMeta


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

    # ===== SWA (Sliding Window Attention) Integration =====

    def swa_unavailable_reason(self) -> Optional[str]:
        """Return why SWA production manager is unavailable, or None if it should work."""
        if self.cache_config.swa is None or not self.cache_config.swa.enabled:
            return "SWAPoolConfig is None or disabled"
        if self.server_client_mode:
            # SWA radix-tree ops run on FlexKV Server via SWAPut/SWAAvailable/SWAGet RPC.
            return None
        from flexkv.swa.node_swa_ops import swa_unavailable_reason as _reason_on_engine
        try:
            engine = self.kv_task_engine.cache_engine.cpu_cache_engine
        except AttributeError:
            return "kv_task_engine.cache_engine.cpu_cache_engine missing (no in-process engine)"
        return _reason_on_engine(self.cache_config, engine)

    def _get_cpu_cache_engine(self):
        """Return the CPU cache engine that owns the local radix tree.

        Only available in non-server_client_mode (the engine runs in-process).
        DSv4 SWA runs in this mode. Works for CacheEngineAccel (default,
        index_accel=1), CacheEngine, and HierarchyLRCacheEngine. Returns None if
        unavailable.
        """
        if self.server_client_mode:
            return None
        try:
            return self.kv_task_engine.cache_engine.cpu_cache_engine
        except AttributeError:
            return None

    @staticmethod
    def _engine_tree(engine):
        """Return the radix tree index of a cache engine, field-name agnostic.

        CacheEngineAccel / CacheEngine expose it as `index`; HierarchyLRCacheEngine
        (p2p path) as `local_index`. Returns None if neither is present.
        """
        tree = getattr(engine, "index", None)
        if tree is None:
            tree = getattr(engine, "local_index", None)
        return tree

    def _get_swa_production_manager(self):
        """Return the node-attached SWAProductionManager owned by the cache engine.

        The manager is created once (lazily) on the cache engine via init_swa(),
        so that SWA put/get (here) and cascade eviction (in the engine's take())
        share a single pool and a single source of truth (the SWA state stored on
        each CRadixNode).
        """
        if hasattr(self, '_swa_prod_manager'):
            return self._swa_prod_manager

        self._swa_prod_manager = None
        if self.cache_config.swa is None or not self.cache_config.swa.enabled:
            return None

        engine = self._get_cpu_cache_engine()
        if engine is None:
            return None

        # Node-attached SWA requires the C++ tree (drain_freed_swa_slots) and an
        # init_swa hook. CacheEngineAccel (default) and HierarchyLRCacheEngine
        # qualify; the pure-Python CacheEngine fallback does not.
        tree = self._engine_tree(engine)
        if tree is None or not hasattr(tree, "drain_freed_swa_slots") \
                or not hasattr(engine, "init_swa"):
            flexkv_logger.warning(
                "[KVManager] SWA enabled but cache engine does not support "
                "node-attached SWA; SWA disabled for this engine."
            )
            return None

        # Ensure the engine has a SWA manager and reuse it (single shared pool).
        if getattr(engine, "swa_manager", None) is None:
            engine.init_swa(self.cache_config.swa)
        self._swa_prod_manager = engine.swa_manager
        flexkv_logger.info(
            f"[KVManager] SWA production manager bound: "
            f"num_slots={self.cache_config.swa.num_slots}, "
            f"window_size={self.cache_config.swa.window_size}"
        )
        return self._swa_prod_manager

    def _match_node(self, token_ids: np.ndarray):
        """Resolve token_ids to the deepest matching CRadixNode in the local tree.

        Returns the last matched node, or None if there is no match (empty tree
        or no shared prefix). This is the single query path: SWA never does its
        own prefix matching — it only reads SWA state off the resolved node.
        """
        engine = self._get_cpu_cache_engine()
        if engine is None:
            return None
        tree = self._engine_tree(engine)
        if tree is None:
            return None
        seq = SequenceMeta(
            token_ids=np.asarray(token_ids, dtype=np.int64),
            tokens_per_block=self.cache_config.tokens_per_block,
        )
        seq.gen_hashes()
        if seq.num_blocks == 0:
            return None
        block_hashes_t = torch.from_numpy(seq.block_hashes).to(torch.int64)
        mr = tree.match_prefix(block_hashes_t, int(seq.num_blocks), False)
        if mr is None or int(mr.num_matched_blocks) == 0:
            return None
        return mr.last_node

    def swa_put(self,
                token_ids: Union[torch.Tensor, np.ndarray],
                swa_data: Union[torch.Tensor, np.ndarray, bytes],
                physical_block_ids: Optional[np.ndarray] = None) -> bool:
        """Store SWA data when request finishes (write-through).

        Tries the production manager first (for C++ radix tree path).
        Falls back to the Python SWAConnector path if production manager
        is not available.

        Args:
            token_ids: Full token sequence for the completed request.
            swa_data: SWA snapshot data to store.
            physical_block_ids: Optional physical block IDs from the radix tree
                insert (used for cascade eviction tracking in production path).

        Returns:
            True if stored successfully, False otherwise.
        """
        # SWA pool is process-local. Query the radix tree to resolve the node,
        # then store the snapshot on it. swa_put does NOT depend on the full-KV
        # put task having completed: the node only needs to exist in the tree
        # (the ready blocks are inserted synchronously). If it isn't there yet,
        # we skip storing — SWA can always be recomputed.
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()

        if self.server_client_mode:
            from flexkv.swa.node_swa_ops import _normalize_swa_data
            return self.dp_client.swa_put(token_ids, _normalize_swa_data(swa_data))

        from flexkv.swa.node_swa_ops import swa_put_on_engine
        engine = self._get_cpu_cache_engine()
        if engine is None:
            return False
        return swa_put_on_engine(self.cache_config, engine, token_ids, swa_data)

    def swa_get(self,
                token_ids: Union[torch.Tensor, np.ndarray]) -> Optional[Union[torch.Tensor, np.ndarray]]:
        """Get SWA data for prefix match hit. Returns data or None.

        Tries the production manager first (for C++ radix tree path).
        Falls back to the Python SWAConnector path if production manager
        is not available.

        Args:
            token_ids: Token prefix that was matched.

        Returns:
            SWA data buffer or None if not available.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()

        if self.server_client_mode:
            return self.dp_client.swa_get(token_ids)

        from flexkv.swa.node_swa_ops import swa_get_on_engine
        engine = self._get_cpu_cache_engine()
        if engine is None:
            return None
        return swa_get_on_engine(self.cache_config, engine, token_ids)

    def swa_available(self,
                      token_ids: Union[torch.Tensor, np.ndarray]) -> bool:
        """Check if SWA is available for given token_ids.

        Tries the production manager first (for C++ radix tree path).
        Falls back to the Python SWAConnector path if production manager
        is not available.

        Args:
            token_ids: Token prefix to check.

        Returns:
            True if SWA data is available for the trailing window.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()

        if self.server_client_mode:
            return self.dp_client.swa_available(token_ids)

        from flexkv.swa.node_swa_ops import swa_available_on_engine
        engine = self._get_cpu_cache_engine()
        if engine is None:
            return False
        return swa_available_on_engine(self.cache_config, engine, token_ids)
