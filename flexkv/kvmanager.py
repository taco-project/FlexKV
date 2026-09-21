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

from flexkv import c_ext
from flexkv.server.client import KVDPClient
from flexkv.server.server import KVServer, DPClient
from flexkv.kvtask import KVTaskEngine, KVResponse
from flexkv.common.config import ModelConfig, CacheConfig, GLOBAL_CONFIG_FROM_ENV, MooncakeTransferEngineConfig
from flexkv.common.transfer import TransferOpGraph
from flexkv.integration.dynamo.collector import KVEventCollector
from flexkv.common.debug import eviction_log_aggregator, flexkv_logger
from flexkv.cache.redis_meta import RedisMeta


class KVManager:
    def __init__(self,
                 model_config: ModelConfig,
                 cache_config: CacheConfig,
                 dp_client_id: int = 0,
                 server_recv_port: str = "",
                 gpu_register_port: str = "",
                 event_collector: Optional[KVEventCollector] = None,
                 local_dp_client_id: Optional[int] = None):
        # Use the curated ``__str__`` summaries. Dataclass repr includes
        # credential-bearing fields such as ``redis_password``.
        flexkv_logger.info(
            "[FlexKV-CONFIG] operation=config act=load status=success "
            "component=kv_manager commit=%s model_config=%s cache_config=%s",
            getattr(c_ext, "__git_commit__", "unknown"),
            model_config,
            cache_config,
        )
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

        self.enable_radixshmem = GLOBAL_CONFIG_FROM_ENV.enable_radixshmem
        if self.enable_radixshmem and cache_config.enable_remote:
            # CacheEngineRadixShmem indexes the CPU tier in shm and reaches
            # peers over RDMA; but the 3rd-party (PCFS) tier has its own
            # Redis-published index and GET planner.
            raise ValueError(
                "radix_shmem and enable_remote (3rd-party remote storage) "
                "cannot be enabled at the same time"
            )
        if self.enable_radixshmem and cache_config.enable_ssd:
            raise ValueError(
                "radix_shmem backs the CPU tier only (index + SlotStore + peer "
                "pull); set ssd_cache_gb=0 / enable_ssd=False"
            )
        if self.enable_radixshmem and (cache_config.enable_p2p_cpu
                                     or cache_config.enable_p2p_ssd):
            # Peer reuse is the radix-server's (etcd + RDMA), switched on by the
            # radixshmem YAML making the cluster distributed; the Redis-backed
            # P2P paths these flags select must stay off.
            raise ValueError(
                "radix_shmem does its own peer reuse; set enable_p2p_cpu=False "
                "and enable_p2p_ssd=False (cross-node reuse follows the "
                "radixshmem YAML: expected_min_nodes / num_rht_shards)"
            )
        # Prefix of this host's radix regions and TE channels: the YAML's
        # cluster_id (plus the node name when several nodes share the host).
        self._shm_radix_id = None
        if self.enable_radixshmem:
            from flexkv.common.radixshmem_config import get_radixshmem_config
            self._shm_radix_id = get_radixshmem_config().local_id

        flexkv_logger.info(
            f"[KVManager] IPC ports: server_recv_port={self.server_recv_port}, "
            f"gpu_register_port={self.gpu_register_port}"

        )

        if self.enable_radixshmem:
            flexkv_logger.info(f"[KVManager] radix_shmem is enabled"
                               f"[KVManager] shm_radix_id: {self._shm_radix_id}")

        # Multi-instance mode also requires server_client_mode
        if self.enable_radixshmem:
            # Force server_client_mode False — KVServer is bypassed entirely.
            self.server_client_mode = False
        else:
            self.server_client_mode = (model_config.dp_size > 1 or
                                       model_config.instance_num > 1 or
                                       GLOBAL_CONFIG_FROM_ENV.server_client_mode)
        self.server_launch_mode = GLOBAL_CONFIG_FROM_ENV.server_launch_mode
        if self.server_launch_mode not in ("embedded", "external"):
            raise ValueError(
                "FLEXKV_SERVER_LAUNCH_MODE must be embedded or external, "
                f"got {self.server_launch_mode!r}"
            )
        if self.server_launch_mode == "external" and not self.server_client_mode:
            raise ValueError(
                "FLEXKV_SERVER_LAUNCH_MODE=external requires server-client mode"
            )

        self.dp_client_id = dp_client_id
        self.local_dp_client_id = (
            dp_client_id if local_dp_client_id is None else local_dp_client_id
        )

        flexkv_logger.info(
            f"[KVManager] instance_num={model_config.instance_num}, dp_size={model_config.dp_size}, "
            f"dp_client_id={self.dp_client_id}, "
            f"local_dp_client_id={self.local_dp_client_id}, "
            f"server_client_mode={self.server_client_mode}, "
            f"server_launch_mode={self.server_launch_mode}, "
            f"enable_radixshmem={self.enable_radixshmem}"
        )

        self.redis_meta_client = None
        self.enable_mps = GLOBAL_CONFIG_FROM_ENV.enable_mps
        self.owns_mps = self.enable_mps and self.server_launch_mode != "external"
        # The embedded radix-server subprocess — only the bootstrap process
        # holds this; others have None.
        self._shm_radix_server = None
        # TE-process handle — only the bootstrap process holds this.
        self._shm_te_process = None
        # Local KVTaskEngine for the radix-shmem path (per-DP).
        self.kv_task_engine = None
        self.server_handle = None

        if self.enable_radixshmem:
            self._init_radix_shmem_path(event_collector)
        elif self.server_client_mode:
            # One KVServer per node: with node-local DP the first rank of each
            # node owns it, so nodes 1..n-1 get their own server instead of
            # waiting on node 0's. Without node-local DP local_dp_client_id
            # equals dp_client_id and this is the previous condition.
            if self.server_launch_mode == "embedded" and self.local_dp_client_id == 0:
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

    def _init_radix_shmem_path(self,
                               event_collector: Optional[KVEventCollector]) -> None:
        """Initialize the radix-shmem multi-DP path.

        Everything shared by this inference instance's DP processes on the
        node — the radix shm regions and the single TE subprocess — is set up
        by the node-local bootstrap proc (local DP client 0) only. Every other
        proc builds its own KVTaskEngine and
        attaches: `CacheEngineRadixShmem` polls for its region, and the TE
        channel handle blocks in `ShmControlBlock.wait_ready`.

        Each CE process gets a disjoint graph/op id range so submissions to the
        single shared TE never collide.
        """
        from flexkv.common.transfer import TransferOp

        TransferOpGraph.set_graph_id_range(self.dp_client_id << 32,
                                           (self.dp_client_id + 1) << 32)
        TransferOp.set_op_id_range(self.dp_client_id << 32,
                                   (self.dp_client_id + 1) << 32)

        try:
            if self.local_dp_client_id == 0:
                self._bootstrap_radix_shmem()

            # KVTaskEngine reads GLOBAL_CONFIG_FROM_ENV.enable_radixshmem and builds a
            # RadixShmemCacheEngine (CPU tier = RadixClient on the radix-server).
            self.kv_task_engine = KVTaskEngine(
                self.model_config, self.cache_config,
                self.gpu_register_port,
                redis_meta=self.redis_meta_client,
                event_collector=event_collector,
                shm_te_server_id=self._shm_radix_id,
                shm_te_channel_id=self.local_dp_client_id,
            )
        except BaseException:
            # A failure after the TE / radix-server subprocesses were spawned
            # must not leave them running (the TE would wait for GPU
            # registrations forever).
            self._shutdown_radix_shmem_children()
            raise

    def _shutdown_radix_shmem_children(self) -> None:
        if self._shm_te_process is not None:
            self._shm_te_process.shutdown()
            self._shm_te_process = None
        if self._shm_radix_server is not None:
            self._shm_radix_server.shutdown()
            self._shm_radix_server = None

    def _bootstrap_radix_shmem(self) -> None:
        """Bootstrap proc (dp 0) only: bring up this node's radix-server (index +
        SlotStore + peer transfer) and spawn the shared TE.

        The server is a subprocess (``FLEXKV_RADIX_SERVER_LAUNCH_MODE=embedded``)
        or one the operator started (``external``); either way every FlexKV
        process attaches by name. Peer reuse needs no Redis address book any
        more: the server resolves peers through etcd and pulls their blocks
        itself (``RadixClient.pull_async`` from the prefetch path)."""
        from flexkv.server.shm_radix_bootstrap import (RadixServerProcess,
                                                       build_radix_server_config,
                                                       radix_socket_path)
        from flexkv.transfer_manager import TransferManagerShmTEProcess

        launch_mode = GLOBAL_CONFIG_FROM_ENV.radix_server_launch_mode
        if launch_mode not in ("embedded", "external"):
            raise ValueError(
                "FLEXKV_RADIX_SERVER_LAUNCH_MODE must be embedded or external, "
                f"got {launch_mode!r}"
            )
        if launch_mode == "embedded":
            server_cfg = build_radix_server_config(self.model_config, self.cache_config)
            self._shm_radix_server = RadixServerProcess(server_cfg).start()
            self.cache_config.distributed_node_id = int(
                self._shm_radix_server.cluster_rank)
            flexkv_logger.info(
                f"[kv manager] radix-server for {self._shm_radix_id} is up: "
                f"cluster rank {self.cache_config.distributed_node_id}"
            )
        else:
            from flexkv.common.radixshmem_config import get_radixshmem_config
            flexkv_logger.info(
                f"[kv manager] attaching to an external radix-server at "
                f"{get_radixshmem_config().endpoint or radix_socket_path(self._shm_radix_id)}"
            )

        total_clients = self.model_config.total_clients
        if self.model_config.local_dp_size is not None:
            total_clients = self.model_config.local_dp_size
        self._shm_te_process = TransferManagerShmTEProcess(
            self.model_config, self.cache_config,
            gpu_register_port=self.gpu_register_port,
            server_id=self._shm_radix_id,
            num_channels=total_clients,
        )
        self._shm_te_process.start()

    def start(self) -> None:
        if self.owns_mps:
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
        flexkv_logger.info("[FLEXKV] KVManager.shutdown begin.")
        eviction_log_aggregator.flush()
        if self.server_client_mode:
            if self.server_launch_mode == "external":
                self.dp_client.unregister()
            else:
                self.dp_client.shutdown()
                # Wait for the server process to exit after sending shutdown request
                if self.server_handle is not None:
                    self.server_handle.shutdown()
                    self.server_handle = None
        else:
            if self.kv_task_engine is not None:
                self.kv_task_engine.shutdown()

        # Multi-DP radix-shmem teardown — only the bootstrap proc owns these.
        # TE first: its workers map the server's SlotStore.
        self._shutdown_radix_shmem_children()

        if self.owns_mps:
            flexkv_logger.info(
                "MPS is enabled. To stop MPS daemon manually, run: "
                "'echo quit | nvidia-cuda-mps-control'"
            )
        flexkv_logger.info("[FLEXKV] KVManager.shutdown done.")

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
                dp_client_id=self.dp_client_id,
                namespace=namespace,
            )
        return task_id

    def get_match(self,
                  token_ids: Union[torch.Tensor, np.ndarray],
                  token_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
                  cpu_only: bool = False,
                  namespace: Optional[List[str]] = None,
                  swa_aware: bool = False,
                  ) -> Tuple[int, np.ndarray]:
        """Match a prefix and build the load graph; return (task_id, mask).

        ``swa_aware=True`` clamps the Full-KV transfer to the reusable SWA window
        (from the same single match); the SWA window is the trailing block of the
        returned mask, which the caller reads directly. ``swa_aware=False``
        (default) is the plain path.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if isinstance(token_mask, torch.Tensor):
            token_mask = token_mask.numpy()
        if self.server_client_mode:
            task_id, mask = self.dp_client.get_match(token_ids,
                                                     token_mask,
                                                     cpu_only=cpu_only,
                                                     namespace=namespace,
                                                     swa_aware=swa_aware)
        else:
            task_id, mask = self.kv_task_engine.get_match(
                token_ids=token_ids,
                token_mask=token_mask,
                cpu_only=cpu_only,
                dp_client_id=self.dp_client_id,
                namespace=namespace,
                swa_aware=swa_aware,
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
                dp_client_id=self.dp_client_id,
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
                dp_client_id=self.dp_client_id,
                namespace=namespace,
            )
        return task_id, mask

    def prefetch_async(self,
                       token_ids: np.ndarray,
                       namespace: Optional[List[str]] = None,
                       swa_aware: bool = False) -> int:
        """Launch prefetch; return the task_id.

        The prefetch is fire-and-forget at launch time. Callers poll progress
        via ``try_wait``/``wait`` — the returned ``KVResponse.return_mask`` is
        rewritten to the CPU-tree state at graph completion (post-commit), so
        ``sum(return_mask)`` is the authoritative usable-token count.

        ``swa_aware=True`` plans a joint Full+SWA REMOTE2H so the SWA snapshot
        lands on the local CPU SWA pool alongside the Full-KV prefix. The tree
        keeps the invariant "SWA present ⇒ Full ready up to this node" —
        partial Full or SWA failure frees the SWA slot; only the Full prefix
        (if any) stays on the tree.
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.numpy()
        if self.server_client_mode:
            task_id = self.dp_client.prefetch_async(
                token_ids, namespace=namespace, swa_aware=swa_aware)
        else:
            task_id = self.kv_task_engine.prefetch_async(
                token_ids,
                dp_client_id=self.dp_client_id,
                namespace=namespace,
                swa_aware=swa_aware,
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
            return self.dp_client.launch_tasks(
                task_ids=task_ids,
                slot_mappings=slot_mappings,
                swa_slot_mappings=swa_slot_mappings,
                as_batch=as_batch,
                layerwise_transfer=layerwise_transfer,
                counter_id=counter_id,
            )
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

    def reset(self) -> None:
        """Invalidate the cache across all tiers (CPU + SSD + remote): drop the
        radix tree and free the mempool.

        Call after a weight update so KV computed against stale weights is not
        reused. Works in both in-process and server-client mode. Cheap and
        idempotent (resetting an already-empty tree/mempool is a no-op).
        """
        if self.server_client_mode:
            self.dp_client.reset()
        else:
            self.kv_task_engine.reset_cache()
