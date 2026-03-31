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

from typing import Optional, Tuple, List, Dict, Union

import numpy as np
import torch

from flexkv.kvmanager import KVManager
from flexkv.server.client import KVTPClient
from flexkv.kvtask import KVResponse
from flexkv.common.config import ModelConfig, CacheConfig
from flexkv.common.storage import KVCacheLayout
from flexkv.common.debug import flexkv_logger
from flexkv.integration.dynamo.collector import KVEventCollector



class KVManagerGroup:
    """Orchestrates a main ``KVManager`` and an optional indexer ``KVManager``.

    Args:
        model_config: Model configuration for the main KV cache
            (num_layers, num_kv_heads, head_size, dtype, …).
        cache_config: Cache configuration for the main KV cache
            (tokens_per_block, num_cpu_blocks, enable_cpu, …).
        dp_client_id: Data-parallel client identifier (default ``0``).
        server_recv_port: Port string used by the main KVManager to connect
            to the FlexKV server.  Defaults to the value from environment.
        gpu_register_port: Port string used by the main KVManager for GPU
            block registration.  Defaults to ``server_recv_port + "_gpu_register"``.
        event_collector: Optional ``KVEventCollector`` for Dynamo-style KV
            event tracing.  Only attached to the main manager.
        indexer_model_config: Model configuration for the indexer KV cache.
            When *None* (default) no indexer manager is created.
        indexer_cache_config: Cache configuration for the indexer KV cache.
            Required when *indexer_model_config* is provided.
        indexer_server_recv_port: Port string for the indexer KVManager.
        indexer_gpu_register_port: GPU-register port for the indexer KVManager.
    """
    def __init__(
        self,
        model_config: ModelConfig,
        cache_config: CacheConfig,
        dp_client_id: int = 0,
        server_recv_port: str = "",
        gpu_register_port: str = "",
        event_collector: Optional[KVEventCollector] = None,
        indexer_model_config: Optional[ModelConfig] = None,
        indexer_cache_config: Optional[CacheConfig] = None,
        indexer_server_recv_port: str = "",
        indexer_gpu_register_port: str = "",
    ):
        # Create the main KVManager
        self._kv_manager = KVManager(
            model_config=model_config,
            cache_config=cache_config,
            dp_client_id=dp_client_id,
            server_recv_port=server_recv_port,
            gpu_register_port=gpu_register_port,
            event_collector=event_collector,
        )

        # Create the optional indexer KVManager
        self._indexer_kv_manager: Optional[KVManager] = None
        # Maps main task_id -> indexer task_id for coordinated lifecycle
        self._indexer_task_map: Dict[int, int] = {}

        if indexer_model_config is not None:
            flexkv_logger.info(
                "KVManagerGroup: creating indexer KVManager for sparse attention indexer cache")
            self._indexer_kv_manager = KVManager(
                model_config=indexer_model_config,
                cache_config=indexer_cache_config,
                dp_client_id=dp_client_id,
                server_recv_port=indexer_server_recv_port,
                gpu_register_port=indexer_gpu_register_port,
                event_collector=None,
            )

    @property
    def kv_manager(self) -> KVManager:
        """Direct access to the main ``KVManager``."""
        return self._kv_manager

    @property
    def indexer_kv_manager(self) -> Optional[KVManager]:
        """Direct access to the indexer ``KVManager`` (may be *None*)."""
        return self._indexer_kv_manager

    @property
    def dpclient_id(self) -> int:
        return self._kv_manager.dpclient_id

    @property
    def dp_client_id(self) -> int:
        return self._kv_manager.dp_client_id

    @property
    def model_config(self) -> ModelConfig:
        return self._kv_manager.model_config

    @property
    def cache_config(self) -> CacheConfig:
        return self._kv_manager.cache_config

    @property
    def server_client_mode(self) -> bool:
        return self._kv_manager.server_client_mode

    @property
    def global_client_id(self) -> int:
        return self._kv_manager.global_client_id


    def start(self) -> None:
        self._kv_manager.start()
        if self._indexer_kv_manager is not None:
            self._indexer_kv_manager.start()

    def is_ready(self) -> bool:
        ready = self._kv_manager.is_ready()
        if ready and self._indexer_kv_manager is not None:
            ready = self._indexer_kv_manager.is_ready()
        return ready

    def shutdown(self) -> None:
        self._kv_manager.shutdown()
        if self._indexer_kv_manager is not None:
            self._indexer_kv_manager.shutdown()


    def get_async(
        self,
        token_ids: Union[torch.Tensor, np.ndarray],
        slot_mapping: Union[torch.Tensor, np.ndarray],
        token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        layer_granularity: int = -1,
        dp_id: int = 0,
        namespace: Optional[List[str]] = None,
    ) -> int:
        task_id = self._kv_manager.get_async(
            token_ids, slot_mapping, token_mask, layer_granularity, dp_id, namespace)
        if self._indexer_kv_manager is not None:
            idx_task_id = self._indexer_kv_manager.get_async(
                token_ids=token_ids, slot_mapping=slot_mapping,
                dp_id=dp_id, namespace=namespace)
            self._indexer_task_map[task_id] = idx_task_id
        return task_id

    def get_match(
        self,
        token_ids: Union[torch.Tensor, np.ndarray],
        token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        layer_granularity: int = -1,
        dp_id: int = 0,
        namespace: Optional[List[str]] = None,
    ) -> Tuple[int, np.ndarray]:
        task_id, mask = self._kv_manager.get_match(
            token_ids, token_mask, layer_granularity, dp_id, namespace)
        if self._indexer_kv_manager is not None:
            idx_task_id, _ = self._indexer_kv_manager.get_match(
                token_ids=token_ids, dp_id=dp_id, namespace=namespace)
            self._indexer_task_map[task_id] = idx_task_id
        return task_id, mask

    def put_async(
        self,
        token_ids: Union[torch.Tensor, np.ndarray],
        slot_mapping: Union[torch.Tensor, np.ndarray],
        token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        dp_id: int = 0,
        namespace: Optional[List[str]] = None,
    ) -> int:
        task_id = self._kv_manager.put_async(
            token_ids, slot_mapping, token_mask, dp_id, namespace)
        if self._indexer_kv_manager is not None:
            idx_task_id = self._indexer_kv_manager.put_async(
                token_ids=token_ids, slot_mapping=slot_mapping,
                dp_id=dp_id, namespace=namespace)
            self._indexer_task_map[task_id] = idx_task_id
        return task_id

    def put_match(
        self,
        token_ids: Union[torch.Tensor, np.ndarray],
        token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        dp_id: int = 0,
        namespace: Optional[List[str]] = None,
    ) -> Tuple[int, np.ndarray]:
        task_id, mask = self._kv_manager.put_match(
            token_ids, token_mask, dp_id, namespace)
        if self._indexer_kv_manager is not None:
            idx_task_id, _ = self._indexer_kv_manager.put_match(
                token_ids=token_ids, dp_id=dp_id, namespace=namespace)
            self._indexer_task_map[task_id] = idx_task_id
        return task_id, mask

    def prefetch_async(
        self,
        token_ids: np.ndarray,
        dp_id: int = 0,
        namespace: Optional[List[str]] = None,
    ) -> int:
        return self._kv_manager.prefetch_async(token_ids, dp_id, namespace)


    def launch(
        self,
        task_ids: Union[int, List[int]],
        slot_mappings: Union[np.ndarray, List[np.ndarray], torch.Tensor, List[torch.Tensor]],
        as_batch: bool = False,
        **kwargs,
    ) -> List[int]:
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        if not isinstance(slot_mappings, list):
            slot_mappings = [slot_mappings]
        if isinstance(slot_mappings[0], torch.Tensor):
            slot_mappings = [sm.numpy() for sm in slot_mappings]

        # Launch corresponding indexer tasks (created during get_match/put_match)
        if self._indexer_kv_manager is not None:
            indexer_task_ids: List[int] = []
            indexer_slot_mappings: List[np.ndarray] = []
            for tid, sm in zip(task_ids, slot_mappings):
                idx_tid = self._indexer_task_map.get(tid)
                if idx_tid is not None:
                    indexer_task_ids.append(idx_tid)
                    indexer_slot_mappings.append(sm)
            if indexer_task_ids:
                self._indexer_kv_manager.launch(
                    task_ids=indexer_task_ids,
                    slot_mappings=indexer_slot_mappings)

        return self._kv_manager.launch(
            task_ids, slot_mappings, as_batch, **kwargs)

    def wait(
        self,
        task_ids: Union[int, List[int]],
        timeout: float = 20.0,
        completely: bool = False,
    ) -> Dict[int, KVResponse]:
        if isinstance(task_ids, int):
            task_ids = [task_ids]

        # Wait for corresponding indexer tasks
        if self._indexer_kv_manager is not None:
            indexer_task_ids = [self._indexer_task_map.pop(tid)
                                for tid in task_ids
                                if tid in self._indexer_task_map]
            if indexer_task_ids:
                self._indexer_kv_manager.wait(
                    task_ids=indexer_task_ids, timeout=timeout)

        return self._kv_manager.wait(task_ids, timeout, completely)

    def try_wait(self, task_ids: Union[int, List[int]]) -> Dict[int, KVResponse]:
        if isinstance(task_ids, int):
            task_ids = [task_ids]

        # Check corresponding indexer tasks completion
        if self._indexer_kv_manager is not None:
            pending = {tid: self._indexer_task_map[tid]
                       for tid in task_ids
                       if tid in self._indexer_task_map}
            if pending:
                indexer_results = self._indexer_kv_manager.try_wait(list(pending.values()))
                completed_idx_tids = set(indexer_results.keys())
                for main_tid, idx_tid in pending.items():
                    if idx_tid in completed_idx_tids:
                        self._indexer_task_map.pop(main_tid, None)

        return self._kv_manager.try_wait(task_ids)

    def cancel(self, task_ids: Union[int, List[int]]) -> None:
        if isinstance(task_ids, int):
            task_ids = [task_ids]
        # Cancel corresponding indexer tasks
        if self._indexer_kv_manager is not None:
            indexer_task_ids = [self._indexer_task_map.pop(tid)
                                for tid in task_ids
                                if tid in self._indexer_task_map]
            if indexer_task_ids:
                self._indexer_kv_manager.cancel(indexer_task_ids)
        self._kv_manager.cancel(task_ids)



class KVTPClientGroup:
    """Manages a main ``KVTPClient`` and an optional indexer ``KVTPClient``.

    Args:
        gpu_register_port: Port for the main KVTPClient.
        client_id: Client identifier.
        device_id: CUDA device id.
        indexer_gpu_register_port: Port for the indexer KVTPClient (optional).
    """

    def __init__(
        self,
        gpu_register_port: str,
        client_id: int,
        device_id: int,
        indexer_gpu_register_port: str = "",
    ):
        self._tp_client = KVTPClient(gpu_register_port, client_id, device_id)
        self._indexer_tp_client: Optional[KVTPClient] = None
        if indexer_gpu_register_port:
            flexkv_logger.info(
                f"KVTPClientGroup: creating indexer KVTPClient to {indexer_gpu_register_port}")
            self._indexer_tp_client = KVTPClient(
                indexer_gpu_register_port, client_id, device_id)


    @property
    def tp_client(self) -> KVTPClient:
        """Direct access to the underlying main KVTPClient (backward compat)."""
        return self._tp_client

    @property
    def indexer_tp_client(self) -> Optional[KVTPClient]:
        """Direct access to the underlying indexer KVTPClient (if any)."""
        return self._indexer_tp_client


    def register_to_server(
        self,
        kv_caches: list,
        gpu_layout: KVCacheLayout,
        indexer_buffers: Optional[list] = None,
        indexer_layout: Optional[KVCacheLayout] = None,
    ) -> None:
        """Register GPU KV cache buffers (and optional indexer buffers) to FlexKV.

        Args:
            kv_caches: List of main KV cache tensors.
            gpu_layout: Layout descriptor for the main KV caches.
            indexer_buffers: Optional list of indexer cache tensors.
            indexer_layout: Layout descriptor for the indexer caches.
                Must be provided when *indexer_buffers* is not empty.
        """
        self._tp_client.register_to_server(kv_caches, gpu_layout)
        flexkv_logger.info("Registered main KV caches to server")

        if (indexer_buffers is not None and len(indexer_buffers) > 0
                and self._indexer_tp_client is not None):
            assert indexer_layout is not None, (
                "indexer_layout must be provided when indexer_buffers is not empty")
            self._indexer_tp_client.register_to_server(indexer_buffers, indexer_layout)
            flexkv_logger.info(
                f"Registered indexer cache: {len(indexer_buffers)} layers, "
                f"num_blocks={indexer_layout.num_block}, "
                f"tokens_per_block={indexer_layout.tokens_per_block}, "
                f"head_size={indexer_layout.head_size}")
        elif indexer_buffers is not None and len(indexer_buffers) > 0:
            flexkv_logger.warning(
                "Detected indexer buffers but no indexer KVTPClient configured. "
                "Indexer transfers will be skipped.")


# ---------------------------------------------------------------------------
# Smoke tests (run with: python -m flexkv.kv_group)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import copy

    print("=" * 60)
    print("KVManagerGroup / KVTPClientGroup smoke tests")
    print("=" * 60)

    # -- Minimal configs for smoke testing --
    main_model_config = ModelConfig(
        num_layers=4,
        num_kv_heads=1,
        head_size=128,
        use_mla=True,
        dtype=torch.float16,
        tp_size=1,
        dp_size=1,
    )
    main_cache_config = CacheConfig(
        tokens_per_block=16,
        enable_cpu=True,
        enable_ssd=False,
        num_cpu_blocks=128,
    )

    indexer_model_config = ModelConfig(
        num_layers=4,
        num_kv_heads=1,
        head_size=64,
        use_mla=True,
        dtype=torch.uint8,
        tp_size=1,
        dp_size=1,
    )
    indexer_cache_config = copy.deepcopy(main_cache_config)

    # ---- Test 1: KVManagerGroup without indexer ----
    print("\n[Test 1] KVManagerGroup without indexer...")
    group_no_idx = KVManagerGroup(
        model_config=main_model_config,
        cache_config=main_cache_config,
    )
    assert group_no_idx.kv_manager is not None, "kv_manager should not be None"
    assert group_no_idx.indexer_kv_manager is None, "indexer_kv_manager should be None"
    print("  PASSED: indexer_kv_manager is None")

    # ---- Test 2: KVManagerGroup with indexer ----
    print("\n[Test 2] KVManagerGroup with indexer...")
    group_with_idx = KVManagerGroup(
        model_config=main_model_config,
        cache_config=main_cache_config,
        indexer_model_config=indexer_model_config,
        indexer_cache_config=indexer_cache_config,
        indexer_server_recv_port="test_indexer_port",
        indexer_gpu_register_port="test_indexer_gpu_port",
    )
    assert group_with_idx.kv_manager is not None, "kv_manager should not be None"
    assert group_with_idx.indexer_kv_manager is not None, "indexer_kv_manager should not be None"
    print("  PASSED: indexer_kv_manager is not None")

    # ---- Test 3: KVTPClientGroup without indexer ----
    print("\n[Test 3] KVTPClientGroup without indexer...")
    tp_group_no_idx = KVTPClientGroup(
        gpu_register_port="test_gpu_port",
        client_id=0,
        device_id=0,
    )
    assert tp_group_no_idx.tp_client is not None, "tp_client should not be None"
    assert tp_group_no_idx.indexer_tp_client is None, "indexer_tp_client should be None"
    print("  PASSED: indexer_tp_client is None")

    # ---- Test 4: KVTPClientGroup with indexer ----
    print("\n[Test 4] KVTPClientGroup with indexer...")
    tp_group_with_idx = KVTPClientGroup(
        gpu_register_port="test_gpu_port",
        client_id=0,
        device_id=0,
        indexer_gpu_register_port="test_indexer_gpu_port",
    )
    assert tp_group_with_idx.tp_client is not None, "tp_client should not be None"
    assert tp_group_with_idx.indexer_tp_client is not None, "indexer_tp_client should not be None"
    print("  PASSED: indexer_tp_client is not None")

    print("\n" + "=" * 60)
    print("All smoke tests PASSED")
    print("=" * 60)
