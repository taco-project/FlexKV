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

import threading
import time
from functools import partial
from queue import Queue
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass, field

import numpy as np
import nvtx
import torch
from flexkv.c_ext import CRadixNode, CRadixTreeIndex, CMatchResult
from flexkv.cache.hie_cache_engine import HierarchyLRCacheEngine
from flexkv.cache.redis_meta import RedisMeta, dist_available

from flexkv.cache.mempool import Mempool
from flexkv.cache.radixtree import RadixTreeIndex, RadixNode, MatchResult
from flexkv.cache.transfer_pattern import add_virtual_op_for_multiple_finished_ops
from flexkv.common.block import SequenceMeta
from flexkv.common.config import CacheConfig, ModelConfig, GLOBAL_CONFIG_FROM_ENV
from flexkv.common.transfer import (
    DeviceType, TransferOpGraph, TransferOp, TransferType
)
from flexkv.common.debug import flexkv_logger
from flexkv.common.type import MatchResultAccel
from flexkv.integration.dynamo.collector import KVEventCollector
from flexkv.metrics import FlexKVMetricsCollector, init_global_collector, get_global_collector

DEVICE_TYPE: List[str] = ['CPU', 'GPU', 'SSD', 'REMOTE']
_VALID_EVICTION_POLICIES = {'lru', 'lfu', 'fifo', 'mru', 'filo'}

class CacheEngineAccel:
    def __init__(self,
                 device_type: DeviceType,
                 num_total_blocks: int,
                 tokens_per_block: int,
                 evict_ratio: float,
                 hit_reward_seconds: int = 0,
                 evict_start_threshold: float = 1.0,
                 eviction_policy: str = "lru",
                 event_collector: Optional[KVEventCollector] = None,
                 metrics_collector = None):
        if not isinstance(device_type, DeviceType):
            raise ValueError(f"Unknown device type: {device_type}")
        if num_total_blocks <= 0:
            raise ValueError(f"Invalid num_total_blocks: {num_total_blocks}")
        if tokens_per_block <= 0 or (tokens_per_block & (tokens_per_block - 1)) != 0:
            raise ValueError(f"Invalid tokens_per_block: {tokens_per_block}, "
                              f"tokens_per_block must be a power of 2")
        if eviction_policy not in _VALID_EVICTION_POLICIES:
            raise ValueError(f"Invalid eviction_policy: '{eviction_policy}'. "
                              f"Supported policies: {sorted(_VALID_EVICTION_POLICIES)}")

        self.device_type = device_type

        self.index = CRadixTreeIndex(tokens_per_block, num_total_blocks, hit_reward_seconds, eviction_policy)

        self.mempool = Mempool(num_total_blocks=num_total_blocks)

        self.tokens_per_block = tokens_per_block
        self.num_total_blocks = num_total_blocks
        self.evict_ratio = evict_ratio
        self.evict_start_threshold = evict_start_threshold

        self.event_collector = event_collector
        self._metrics_collector = metrics_collector

        # Dist-reuse eviction refcount guard (§2.2).  When a guard is
        # installed (typically by
        # :meth:`GlobalCacheEngine.attach_dist_reuse` broadcasting
        # ``MasterCoordinator.is_evictable`` down to every subengine),
        # the ``take`` path below calls the 4-arg
        # ``CRadixTreeIndex::evict`` overload that accepts a
        # ``std::function<bool(int64_t)>`` predicate.  Block ids for
        # which the predicate returns False are *not* recycled — the
        # block stays physically pinned until the in-flight coord GET
        # drains the refcount.  See
        # :cpp:func:`flexkv::CRadixTreeIndex::evict` (radix_tree.cpp)
        # for the authoritative implementation.
        #
        # The guard is optional — when ``None`` (default), ``take``
        # calls the legacy 3-arg overload and behaviour is byte-
        # identical to pre-§2.2.
        self._evict_guard_fn: Optional[Callable[[int], bool]] = None

    def set_evict_guard(self, fn: Optional[Callable[[int], bool]]) -> None:
        """Install (or remove) the refcount guard used in ``take``'s
        eviction path.  Called by
        :meth:`GlobalCacheEngine.attach_dist_reuse` / ``detach_dist_reuse``.
        """
        self._evict_guard_fn = fn

    def reset(self) -> None:
        self.index.reset()
        self.mempool.reset()

    def match(self, sequence_meta: SequenceMeta) -> MatchResultAccel:
        sequence_meta.gen_hashes()
        match_result = self.index.match_prefix(torch.from_numpy(sequence_meta.block_hashes).to(torch.int64),
                                              sequence_meta.num_blocks, True)
        # physical blocks (torch.Tensor -> numpy, zero-copy on CPU)
        phys = match_result.physical_blocks.cpu().numpy()
        # Extract single matched_node_id (single-node constraint)
        raw_nid = getattr(match_result, "matched_node_id", -1)
        single_node_id = int(raw_nid) if raw_nid is not None and raw_nid >= 0 else None
        # Broadcast matched_node_id to per-block array for downstream compat
        bnids_np = None
        if single_node_id is not None and len(phys) > 0:
            bnids_np = np.full(len(phys), single_node_id, dtype=np.uint32)
        return MatchResultAccel(
            num_ready_matched_blocks=match_result.num_ready_matched_blocks,
            num_matched_blocks=match_result.num_matched_blocks,
            last_ready_node=match_result.last_ready_node,
            last_node=match_result.last_node,
            last_node_matched_length=match_result.last_node_matched_length,
            physical_blocks=phys,
            matched_node_id=single_node_id,
            block_node_ids=bnids_np,
            matched_pos="remote" if self.device_type == DeviceType.REMOTE else "local",
        )

    def insert(self,
               sequence_meta: SequenceMeta,
               physical_block_ids: torch.Tensor,
               num_insert_blocks: int = -1,
               is_ready: bool = True,
               match_result: Optional[MatchResultAccel] = None) -> Optional[CRadixNode]:
        sequence_meta.gen_hashes()
        if match_result is None:
            node = self.index.insert(torch.from_numpy(physical_block_ids).to(torch.int64),
                                     torch.from_numpy(sequence_meta.block_hashes).to(torch.int64),
                                     sequence_meta.num_blocks,
                                     num_insert_blocks,
                                     is_ready)
        else:
            node = self.index.insert(torch.from_numpy(physical_block_ids).to(torch.int64),
                                     torch.from_numpy(sequence_meta.block_hashes).to(torch.int64),
                                     sequence_meta.num_blocks,
                                     num_insert_blocks,
                                     is_ready,
                                     match_result.last_node,
                                     match_result.num_matched_blocks,
                                     match_result.last_node_matched_length)

        if self.event_collector is not None:
            self.event_collector.publish_stored(
                block_hashes=sequence_meta.block_hashes[:None if num_insert_blocks == -1 else num_insert_blocks],
                block_size=self.tokens_per_block,
                medium=DEVICE_TYPE[self.device_type]
            )
        return node

    def lock_node(self, node: CRadixNode) -> None:
        self.index.lock(node)

    def unlock(self, node: CRadixNode) -> None:
        self.index.unlock(node)

    def set_ready(self, node: CRadixNode, ready: bool, ready_length: int) -> None:
        self.index.set_ready(node, ready, ready_length)

    def take(self,
             num_required_blocks: int,
             protected_node: Optional[CRadixNode] = None,
             strict: bool = True) -> torch.Tensor:
        # Calculate current utilization
        utilization = (self.mempool.num_total_blocks - self.mempool.num_free_blocks) / self.mempool.num_total_blocks if self.mempool.num_total_blocks > 0 else 0

        # Proactive eviction: trigger when utilization exceeds threshold OR when blocks are needed
        should_evict = (utilization >= self.evict_start_threshold) or (num_required_blocks > self.mempool.num_free_blocks)

        if should_evict:
            if protected_node is not None:
                self.index.lock(protected_node)

            # Calculate how many blocks to evict
            # Goal: maintain free blocks above (1 - evict_start_threshold) ratio
            target_free_blocks = int(self.mempool.num_total_blocks * (1.0 - self.evict_start_threshold))
            evict_to_reach_target = max(0, target_free_blocks - self.mempool.num_free_blocks)

            evict_block_num = max(
                num_required_blocks - self.mempool.num_free_blocks,  # At least meet current demand
                evict_to_reach_target,                               # Or reach target free ratio
                int(self.mempool.num_total_blocks * self.evict_ratio) if self.evict_ratio > 0 else 0  # Or minimum evict_ratio
            )

            if evict_block_num > 0:
                target_blocks = torch.zeros(evict_block_num, dtype=torch.int64)
                evicted_block_hashes = torch.zeros(evict_block_num, dtype=torch.int64)
                # §2.2 dist-reuse refcount guard: when a guard is
                # installed (``GlobalCacheEngine.attach_dist_reuse``
                # pushes ``MasterCoordinator.is_evictable`` down here),
                # call the 4-arg C++ overload so the eviction path
                # never recycles a block id that still has a refcount
                # > 0 from an in-flight coord GET.  Pure legacy path
                # (no guard, default production deployments) keeps the
                # 3-arg call unchanged — byte-identical to pre-§2.2
                # behaviour.
                if self._evict_guard_fn is not None:
                    num_evicted = self.index.evict(
                        target_blocks,
                        evicted_block_hashes,
                        evict_block_num,
                        self._evict_guard_fn,
                    )
                else:
                    num_evicted = self.index.evict(
                        target_blocks, evicted_block_hashes, evict_block_num
                    )
                if num_evicted != evict_block_num:
                    target_blocks.resize_(num_evicted)
                    evicted_block_hashes.resize_(num_evicted)
                target_blocks = target_blocks.numpy()
                self.mempool.recycle_blocks(target_blocks)

                # Record eviction metrics
                if self._metrics_collector is not None and num_evicted > 0:
                    self._metrics_collector.record_eviction(DEVICE_TYPE[self.device_type].lower(), num_evicted)

                if self.event_collector is not None:
                    self.event_collector.publish_removed(
                        block_hashes=evicted_block_hashes.numpy(),
                        medium=DEVICE_TYPE[self.device_type]
                    )
            if protected_node is not None:
                self.index.unlock(protected_node)

        if strict and num_required_blocks > self.mempool.num_free_blocks:
            raise RuntimeError(f"Not enough free blocks to take, "
                               f"required: {num_required_blocks}, "
                               f"available: {self.mempool.num_free_blocks}")
        num_allocated_blocks = min(num_required_blocks, self.mempool.num_free_blocks)
        allocated_blocks = self.mempool.allocate_blocks(num_allocated_blocks)

        # Record allocation metrics
        if self._metrics_collector is not None and num_allocated_blocks > 0:
            self._metrics_collector.record_allocation(DEVICE_TYPE[self.device_type].lower(), num_allocated_blocks)

        return allocated_blocks

    def recycle(self, physical_blocks: np.ndarray) -> None:
        self.mempool.recycle_blocks(physical_blocks)

class CacheEngine:
    def __init__(self,
                 device_type: DeviceType,
                 num_total_blocks: int,
                 tokens_per_block: int,
                 evict_ratio: float,
                 hit_reward_seconds: int = 0,
                 evict_start_threshold: float = 1.0,
                 eviction_policy: str = "lru",
                 event_collector: Optional[KVEventCollector] = None,
                 metrics_collector = None):
        if not isinstance(device_type, DeviceType):
            raise ValueError(f"Unknown device type: {device_type}")
        if num_total_blocks <= 0:
            raise ValueError(f"Invalid num_total_blocks: {num_total_blocks}")
        if tokens_per_block <= 0 or (tokens_per_block & (tokens_per_block - 1)) != 0:
            raise ValueError(f"Invalid tokens_per_block: {tokens_per_block}, "
                              f"tokens_per_block must be a power of 2")
        if eviction_policy not in _VALID_EVICTION_POLICIES:
            raise ValueError(f"Invalid eviction_policy: '{eviction_policy}'. "
                              f"Supported policies: {sorted(_VALID_EVICTION_POLICIES)}")

        self.device_type = device_type

        self.index = RadixTreeIndex(tokens_per_block=tokens_per_block, hit_reward_seconds=hit_reward_seconds, eviction_policy=eviction_policy)

        self.mempool = Mempool(num_total_blocks=num_total_blocks)

        self.tokens_per_block = tokens_per_block
        self.num_total_blocks = num_total_blocks
        self.evict_ratio = evict_ratio
        self.evict_start_threshold = evict_start_threshold

        self.event_collector = event_collector
        self._metrics_collector = metrics_collector

        # Dist-reuse eviction refcount guard.  When set,
        # ``RadixTreeIndex.evict`` will skip physical block ids where
        # ``is_evictable_fn(block_id)`` is False (i.e. the block is
        # pinned by an in-flight coord GET).  See
        # :meth:`GlobalCacheEngine.attach_dist_reuse`.
        self._evict_guard_fn: Optional[Callable[[int], bool]] = None

    def set_evict_guard(self, fn: Optional[Callable[[int], bool]]) -> None:
        """Install (or remove) the refcount guard used in ``take``'s
        eviction path.  Called by ``GlobalCacheEngine.attach_dist_reuse``.
        """
        self._evict_guard_fn = fn

    def reset(self) -> None:
        self.index.reset()
        self.mempool.reset()

    def match(self, sequence_meta: SequenceMeta) -> MatchResult:
        match_result = self.index.match_prefix(sequence_meta,
                                              update_cache_info=True)
        return match_result

    def insert(self,
               sequence_meta: SequenceMeta,
               physical_block_ids: np.ndarray,
               num_insert_blocks: int = -1,
               is_ready: bool = True,
               match_result: Optional[MatchResult] = None) -> Optional[RadixNode]:
        node = self.index.insert(sequence_meta,
                                 physical_block_ids,
                                 num_insert_blocks=num_insert_blocks,
                                 is_ready=is_ready,
                                 match_result=match_result)
        if self.event_collector is not None:
            self.event_collector.publish_stored(block_hashes=sequence_meta.block_hashes[:None if num_insert_blocks == -1 else num_insert_blocks],
                                                block_size=self.tokens_per_block,
                                                medium=DEVICE_TYPE[self.device_type])
        return node

    def lock_node(self, node: RadixNode) -> None:
        self.index.lock(node)

    def unlock(self, node: RadixNode) -> None:
        self.index.unlock(node)

    def set_ready(self, node: RadixNode, ready: bool, ready_length: int) -> None:
        self.index.set_ready(node, ready, ready_length)

    def take(self,
             num_required_blocks: int,
             protected_node: Optional[RadixNode] = None,
             strict: bool = True) -> np.ndarray:
        # Calculate current utilization
        utilization = (self.mempool.num_total_blocks - self.mempool.num_free_blocks) / self.mempool.num_total_blocks if self.mempool.num_total_blocks > 0 else 0

        # Proactive eviction: trigger when utilization exceeds threshold OR when blocks are needed
        should_evict = (utilization >= self.evict_start_threshold) or (num_required_blocks > self.mempool.num_free_blocks)

        if should_evict:
            if protected_node is not None:
                self.index.lock(protected_node)

            # Calculate how many blocks to evict
            # Goal: maintain free blocks above (1 - evict_start_threshold) ratio
            target_free_blocks = int(self.mempool.num_total_blocks * (1.0 - self.evict_start_threshold))
            evict_to_reach_target = max(0, target_free_blocks - self.mempool.num_free_blocks)

            evict_block_num = max(
                num_required_blocks - self.mempool.num_free_blocks,  # At least meet current demand
                evict_to_reach_target,                               # Or reach target free ratio
                int(self.mempool.num_total_blocks * self.evict_ratio) if self.evict_ratio > 0 else 0  # Or minimum evict_ratio
            )
            if evict_block_num > 0:
                evicted_blocks, evicted_block_hashes = self.index.evict(
                    evict_block_num,
                    is_evictable_fn=self._evict_guard_fn,
                )
                self.mempool.recycle_blocks(evicted_blocks)

                # Record eviction metrics
                if self._metrics_collector is not None and len(evicted_blocks) > 0:
                    self._metrics_collector.record_eviction(DEVICE_TYPE[self.device_type].lower(), len(evicted_blocks))

                if self.event_collector is not None:
                    self.event_collector.publish_removed(block_hashes=evicted_block_hashes,
                                                         medium=DEVICE_TYPE[self.device_type])
            if protected_node is not None:
                self.index.unlock(protected_node)

        if strict and num_required_blocks > self.mempool.num_free_blocks:
            raise RuntimeError("Not enough free blocks to take, ",
                               f"required: {num_required_blocks}, "
                               f"available: {self.mempool.num_free_blocks}")
        num_allocated_blocks = min(num_required_blocks, self.mempool.num_free_blocks)
        allocated_blocks = self.mempool.allocate_blocks(num_allocated_blocks)

        # Record allocation metrics
        if self._metrics_collector is not None and num_allocated_blocks > 0:
            self._metrics_collector.record_allocation(DEVICE_TYPE[self.device_type].lower(), num_allocated_blocks)

        return allocated_blocks

    def recycle(self, physical_blocks: np.ndarray) -> None:
        self.mempool.recycle_blocks(physical_blocks)

@dataclass
class CacheStrategy:
    # if True, will not put or get blocks from GPU
    ignore_gpu: bool = False
    # if True, will not put or get blocks from SSD
    ignore_ssd: bool = False
    # if True, will not get blocks from REMOTE
    ignore_remote: bool = False
    # if True, will not use GDS
    ignore_gds: bool = False

DEFAULT_CACHE_STRATEGY = CacheStrategy()

CPUONLY_CACHE_STRATEGY = CacheStrategy(ignore_gpu=False, ignore_ssd=True, ignore_remote=True, ignore_gds=True)

class GlobalCacheEngine:
    def __init__(self, cache_config: CacheConfig, model_config: ModelConfig, redis_meta: RedisMeta = None,
                 event_collector: Optional[KVEventCollector] = None):
        self.cache_config = cache_config
        self.model_config = model_config
        self.tokens_per_block = cache_config.tokens_per_block

        self.cpu_cache_engine = None
        self.ssd_cache_engine = None
        self.remote_cache_engine = None

        # --------------------------------------------------------------
        # Dist-reuse integration hooks (Phase 1 integration glue).
        #
        # Populated by :meth:`attach_dist_reuse` **after** construction
        # (the KVTaskManager owns the coordinator lifetimes and wires them
        # in once all handles / Remote ready ACKs are collected).  When
        # unset, every dist-reuse code path no-ops and the engine behaves
        # exactly like pre-Batch-E — this keeps legacy deployments
        # (``enable_sharing_domain=False``) byte-identical.
        #
        # ``_master_coord``  -> ``MasterCoordinator`` (owns the
        #                      ``AggregateRadixTree``, refcounts, and the
        #                      Layer-1 ``FailureDetector``).
        # Peer-lost hook is registered on the ``FailureDetector`` via
        # ``MasterCoordinator.set_peer_lost_hook`` and is mapped to
        # ``aggregate_radix.invalidate_by_peer_instance``.
        #
        # Phase D-4 (proposal_unify_with_graph_dispatch_2026-05-15.md):
        # the previous ``_coord_dispatcher -> CoordinationCoordinator``
        # field was removed.  Cross-SD coordination now flows through
        # the unified ``TransferOpGraph`` dispatch path with per-op
        # ``target_node_ids`` filtering on each Remote, and peer-SD
        # acks come back as ``CompletedOp(sd_key, contributing_node_id)``
        # on the master polling worker.
        # --------------------------------------------------------------
        self._master_coord = None  # type: ignore[assignment]

        # Phase D-2 (proposal_unify_with_graph_dispatch_2026-05-15.md
        # §6.3): the PUT-path graph-dispatch ack book.  Populated by
        # ``_notify_master_sd_ready`` when the Master's own SD finishes,
        # consumed by ``_on_peer_sd_completed_op`` when peer-SD
        # ``CompletedOp(sd_key, contributing_node_id)`` arrives via the
        # master polling thread's ``_completion_sink``.
        import threading as _threading
        self._pending_put_lock = _threading.Lock()
        self._pending_put_batches: Dict[int, list] = {}

        self.index_accel = GLOBAL_CONFIG_FROM_ENV.index_accel
        if cache_config.enable_kv_sharing:
            assert redis_meta is not None
            self.redis_meta = redis_meta
            self.node_id = self.redis_meta.get_node_id()
            self.enable_kv_sharing = True
        else:
            self.enable_kv_sharing = False
        self.cache_engines = {}

        self.evict_ratio = GLOBAL_CONFIG_FROM_ENV.evict_ratio
        self.evict_start_threshold = GLOBAL_CONFIG_FROM_ENV.evict_start_threshold
        self.hit_reward_seconds = GLOBAL_CONFIG_FROM_ENV.hit_reward_seconds
        self.eviction_policy = GLOBAL_CONFIG_FROM_ENV.eviction_policy

        # Initialize metrics collector for cache engine monitoring (before creating CacheEngines)
        self._metrics_collector = get_global_collector()
        if self._metrics_collector is None:
            self._metrics_collector = init_global_collector()

        need_dist = (
            (cache_config.enable_cpu and cache_config.enable_p2p_cpu)
            or (cache_config.enable_ssd and cache_config.enable_p2p_ssd)
            or (cache_config.enable_remote and cache_config.enable_kv_sharing)
        )
        if need_dist and not dist_available():
            raise RuntimeError(
                "Config enables distributed KV cache (P2P/Redis), but FlexKV was built without it. "
                "Rebuild with FLEXKV_ENABLE_P2P=1 and install Redis dependencies "
                "(e.g. libhiredis-dev, redis-tools). See README for full list."
            )

        if cache_config.enable_cpu:
            if cache_config.enable_p2p_cpu:
                self.cpu_cache_engine = HierarchyLRCacheEngine.from_cache_config(cache_config, self.node_id, DeviceType.CPU, meta=self.redis_meta)
            elif self.index_accel:
                self.cpu_cache_engine = CacheEngineAccel(DeviceType.CPU,
                                                cache_config.num_cpu_blocks,
                                                cache_config.tokens_per_block,
                                                self.evict_ratio,
                                                self.hit_reward_seconds,
                                                self.evict_start_threshold,
                                                self.eviction_policy,
                                                event_collector,
                                                self._metrics_collector)
            else:
                self.cpu_cache_engine = CacheEngine(DeviceType.CPU,
                                                cache_config.num_cpu_blocks,
                                                cache_config.tokens_per_block,
                                                self.evict_ratio,
                                                self.hit_reward_seconds,
                                                self.evict_start_threshold,
                                                self.eviction_policy,
                                                event_collector,
                                                self._metrics_collector)
            self.cache_engines[DeviceType.CPU] = self.cpu_cache_engine
        if cache_config.enable_ssd:
            if cache_config.enable_p2p_ssd:
                self.ssd_cache_engine = HierarchyLRCacheEngine.from_cache_config(cache_config, self.node_id, DeviceType.SSD, meta=self.redis_meta)
            elif self.index_accel:
                self.ssd_cache_engine = CacheEngineAccel(DeviceType.SSD,
                                                cache_config.num_ssd_blocks,
                                                cache_config.tokens_per_block,
                                                self.evict_ratio,
                                                self.hit_reward_seconds,
                                                self.evict_start_threshold,
                                                self.eviction_policy,
                                                event_collector,
                                                self._metrics_collector)
            else:
                self.ssd_cache_engine = CacheEngine(DeviceType.SSD,
                                                cache_config.num_ssd_blocks,
                                                cache_config.tokens_per_block,
                                                self.evict_ratio,
                                                self.hit_reward_seconds,
                                                self.evict_start_threshold,
                                                self.eviction_policy,
                                                event_collector,
                                                self._metrics_collector)
            self.cache_engines[DeviceType.SSD] = self.ssd_cache_engine
        if cache_config.enable_remote:
            if cache_config.enable_kv_sharing:
                # Build PCFSCacheEngine from CacheConfig directly (replacing RemotePCFSCacheEngine) TODO
                self.remote_cache_engine = HierarchyLRCacheEngine.from_cache_config(cache_config, self.node_id, DeviceType.REMOTE, meta=self.redis_meta)
            elif self.index_accel:
                self.remote_cache_engine = CacheEngineAccel(DeviceType.REMOTE,
                                                   cache_config.num_remote_blocks,
                                                   cache_config.tokens_per_block,
                                                   self.evict_ratio,
                                                   self.hit_reward_seconds,
                                                   self.evict_start_threshold,
                                                   self.eviction_policy,
                                                   None,
                                                   self._metrics_collector)
            else:
                self.remote_cache_engine = CacheEngine(DeviceType.REMOTE,
                                                   cache_config.num_remote_blocks,
                                                   cache_config.tokens_per_block,
                                                   self.evict_ratio,
                                                   self.hit_reward_seconds,
                                                   self.evict_start_threshold,
                                                   self.eviction_policy,
                                                   None,
                                                   self._metrics_collector)
            self.cache_engines[DeviceType.REMOTE] = self.remote_cache_engine

        #TODO move this to kvmanager.start()
        self.start()

        self._empty_get_return: Callable[[int], Tuple[TransferOpGraph, List[int], Dict, Dict, Dict, int]] = \
            lambda request_id: (TransferOpGraph.create_empty_graph(), [], {}, {}, {}, 0)
        self._empty_put_return: Callable[[int], Tuple[TransferOpGraph, List[int], Dict, Dict, Dict, int, int]] = \
            lambda request_id: (TransferOpGraph.create_empty_graph(), [], {}, {}, {}, 0, 0)

        # Update initial mempool stats
        self._update_mempool_metrics()

    def start(self) -> None:
        if self.cpu_cache_engine and self.cache_config.enable_p2p_cpu:
            self.cpu_cache_engine.start()
        if self.ssd_cache_engine and self.cache_config.enable_p2p_ssd:
            self.ssd_cache_engine.start()
        if self.remote_cache_engine and self.cache_config.enable_3rd_remote:
            self.remote_cache_engine.start()

    def reset(self) -> None:
        if self.cpu_cache_engine:
            self.cpu_cache_engine.reset()
        if self.ssd_cache_engine:
            self.ssd_cache_engine.reset()
        if self.remote_cache_engine:
            self.remote_cache_engine.reset()

    def _update_mempool_metrics(self) -> None:
        """Update memory pool metrics for all cache engines."""
        if self._metrics_collector is None:
            return
        for device_type, engine in self.cache_engines.items():
            if hasattr(engine, 'mempool'):
                device_label = DEVICE_TYPE[device_type].lower()
                self._metrics_collector.update_mempool_stats(
                    device_label,
                    engine.mempool.num_total_blocks,
                    engine.mempool.num_free_blocks
                )

    def _record_transfer_ops(self, transfer_graph: TransferOpGraph, operation: str) -> None:
        """Record metrics for all transfer operations in the graph.

        Args:
            transfer_graph: The transfer operation graph
            operation: Operation type ("get" or "put")
        """
        if self._metrics_collector is None:
            return
        for op in transfer_graph._op_map.values():
            if op.transfer_type != TransferType.VIRTUAL:
                transfer_type_str = op.transfer_type.value
                num_blocks = len(op.src_block_ids) if op.src_block_ids is not None else 0
                self._metrics_collector.record_transfer(transfer_type_str, num_blocks, operation)

    def get(self,
            request_id: int,
            token_ids: np.ndarray,
            token_mask: np.ndarray,
            slot_mapping: np.ndarray,
            dp_client_id: int,
            temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
            namespace: Optional[List[str]] = None) \
                 -> Tuple[TransferOpGraph, np.ndarray, Callable, Dict, int]:
        self._check_input(token_ids, token_mask, slot_mapping)

        aligned_length = (token_ids.shape[0] // self.tokens_per_block) * self.tokens_per_block

        aligned_token_ids = token_ids[:aligned_length]
        token_mask[aligned_length:] = False

        if aligned_length == 0 or not token_mask.any():
            transfer_graph = TransferOpGraph.create_empty_graph()
            return_mask = np.zeros_like(token_mask, dtype=np.bool_)
            callback = partial(self._transfer_callback, node_to_unlock={}, buffer_to_free={})
            return transfer_graph, return_mask, callback, {}, -1

        block_start_idx, block_end_idx = self._get_block_range(token_mask)
        assert block_end_idx == aligned_length // self.tokens_per_block
        gpu_block_ids = self.slot_mapping_to_block_ids(slot_mapping,
                                                       self.tokens_per_block)[:block_end_idx-block_start_idx]

        sequence_meta = SequenceMeta(token_ids=aligned_token_ids,
                                     tokens_per_block=self.cache_config.tokens_per_block,
                                     namespace=namespace)

        if not self.cache_config.enable_remote or temp_cache_strategy.ignore_remote:
            # from this entrance, we will also handle the case of peer_cpu and peer_ssd
            (transfer_graph, finished_ops_ids, node_to_unlock,
             op_node_to_ready, buffer_to_free, num_gpu_blocks_to_transfer) = \
                self._get_impl_local(
                    request_id,
                    sequence_meta,
                    block_start_idx,
                    block_end_idx,
                    gpu_block_ids,
                    temp_cache_strategy,
                    dp_client_id,
                )
        else:
            #TODO pcfs will be supported later
            (transfer_graph, finished_ops_ids, node_to_unlock,
             op_node_to_ready, buffer_to_free, num_gpu_blocks_to_transfer) = \
                self._get_impl_global(
                    request_id,
                    sequence_meta,
                    block_start_idx,
                    block_end_idx,
                    gpu_block_ids,
                    temp_cache_strategy,
                    dp_client_id,
                )

        transfer_graph, task_end_op_id = add_virtual_op_for_multiple_finished_ops(
            transfer_graph,
            finished_ops_ids,
            dp_client_id,
            )

        return_mask = np.zeros_like(token_mask, dtype=np.bool_)
        return_mask[block_start_idx* self.tokens_per_block:
                    (block_start_idx + num_gpu_blocks_to_transfer) * self.tokens_per_block] = True

        # ------------------------------------------------------------------
        # §2.1 dist_reuse GET main-path hook.
        #
        # When ``enable_sharing_domain`` is on AND the instance owns more
        # than one SD (PP>1 or tp_node_count>1), every cross-instance
        # reuse hit must clear a coordination barrier: the Master needs
        # every peer SD to ACK that it also has the prefix, otherwise
        # the GPU would receive half-assembled KV.  ``_sharing_domain_gate_get``
        # returns False on barrier failure — in which case we reset the
        # transfer_graph + return_mask to an empty result, equivalent to
        # a cache miss.  Upstream will fall back to normal prefill.
        #
        # For the single-SD degenerate case (most common today: PP=1 +
        # TP does not cross nodes) the gate is a no-op: ``coord_get``
        # has nothing to coordinate so this short-circuits.  Zero
        # regression for existing deployments that only use
        # ``enable_p2p_cpu`` but not ``enable_sharing_domain``.
        # ------------------------------------------------------------------
        barrier_ok = self._sharing_domain_gate_get(
            sequence_meta=sequence_meta,
            return_mask=return_mask,
            block_start_idx=block_start_idx,
            num_gpu_blocks_to_transfer=num_gpu_blocks_to_transfer,
        )
        if not barrier_ok:
            # Release anything we had locked during matching, reset to
            # the empty-return shape — equivalent to a cache miss so
            # upstream falls back to normal prefill.
            for device_type, (node, _) in (node_to_unlock or {}).items():
                try:
                    self.cache_engines[device_type].unlock(node)
                except Exception:
                    pass
            for device_type, blks in (buffer_to_free or {}).items():
                try:
                    if blks is not None and len(blks) > 0:
                        self.cache_engines[device_type].recycle(blks)
                except Exception:
                    pass
            empty_graph = TransferOpGraph.create_empty_graph()
            empty_graph.bind_to_worker(dp_rank, pp_rank)
            empty_mask = np.zeros_like(token_mask, dtype=np.bool_)
            empty_cb = partial(self._transfer_callback,
                               node_to_unlock={}, buffer_to_free={})
            return empty_graph, empty_mask, empty_cb, {}, -1

        # if layer_num // layer_granularity != 1:
        #     transfer_graph, finished_ops_ids = convert_read_graph_to_layer_wise_graph(transfer_graph=transfer_graph,
        #                                                                         finished_ops_ids=finished_ops_ids,
        #                                                                         layer_num=layer_num,
        #                                                                         layer_granularity=layer_granularity)

        for device_type in node_to_unlock:
            self.cache_engines[device_type].lock_node(node_to_unlock[device_type][0])

        callback = partial(self._transfer_callback,
                           node_to_unlock=node_to_unlock,
                           buffer_to_free=buffer_to_free)

        op_callback_dict = {} # dict, op_id -> callback
        for op_id in op_node_to_ready:
            op_callback_dict[op_id] = partial(self._op_callback,
                                              device_type=op_node_to_ready[op_id][0],
                                              node_to_ready=op_node_to_ready[op_id][1],
                                              ready_length=op_node_to_ready[op_id][2])

        # Record metrics for GET operation
        if self._metrics_collector is not None:
            self._record_transfer_ops(transfer_graph, "get")
            self._update_mempool_metrics()

        return transfer_graph, return_mask, callback, op_callback_dict, task_end_op_id

    def _get_impl_global(self,
            request_id: int,
            sequence_meta: SequenceMeta,
            block_mask_start: int,
            block_mask_end: int,
            gpu_block_ids: np.ndarray,
            temp_cache_strategy: CacheStrategy,
            dp_client_id: int) \
                 -> Tuple[TransferOpGraph, List[int], Dict, Dict, Dict, int]:
        """
        transfer pattern:

        GPU: (gpu cached) | fragment1 | fragment2      | fragment3      | (need compute)
                               ↑          ↑               ↑
        CPU:     ...      | fragment1 | fragment2(new) | fragment3(new) ← (from REMOTE)
                                          ↑               ↓
        SSD:     ...      | fragment1 | fragment2      | fragment3(new)

        """
        enable_gpu = not temp_cache_strategy.ignore_gpu
        enable_cpu = self.cache_config.enable_cpu
        enable_ssd = self.cache_config.enable_ssd
        enable_remote = self.cache_config.enable_remote and not temp_cache_strategy.ignore_remote
        assert enable_cpu and enable_remote
        assert self.cpu_cache_engine is not None
        assert self.remote_cache_engine is not None
        if self.index_accel:
            cpu_matched_result, ssd_matched_result, remote_matched_result = self.match_all_accel(sequence_meta)
        else:
            cpu_matched_result, ssd_matched_result, remote_matched_result = self.match_all(sequence_meta)
        cpu_matched_blocks = cpu_matched_result.physical_blocks[
            :cpu_matched_result.num_ready_matched_blocks][block_mask_start:block_mask_end]
        ssd_matched_blocks = ssd_matched_result.physical_blocks[
            :ssd_matched_result.num_ready_matched_blocks][block_mask_start:block_mask_end]
        remote_matched_blocks = remote_matched_result.physical_blocks[
            :remote_matched_result.num_ready_matched_blocks][block_mask_start:block_mask_end]
        shared_pcfs_read = self.cache_config.enable_kv_sharing and self.index_accel
        remote_file_nodeids = None
        if shared_pcfs_read:
            remote_file_nodeids = remote_matched_result.block_node_ids
        fragment123_num_blocks = max(len(cpu_matched_blocks), len(ssd_matched_blocks), len(remote_matched_blocks))
        #early return if no blocks to transfer
        if fragment123_num_blocks == 0:
            # All cache levels missed - record miss for all requested blocks
            if self._metrics_collector is not None:
                total_query_blocks = block_mask_end - block_mask_start
                if total_query_blocks > 0:
                    self._metrics_collector.record_cache_miss(total_query_blocks)
            return self._empty_get_return(request_id)
        assert fragment123_num_blocks <= len(gpu_block_ids)

        transfer_graph = TransferOpGraph()
        finished_ops_ids = []

        fragment1_num_blocks = len(cpu_matched_blocks)
        fragment2_num_blocks = max(len(ssd_matched_blocks) - len(cpu_matched_blocks), 0)
        fragment12_num_blocks = max(len(cpu_matched_blocks), len(ssd_matched_blocks))
        fragment3_num_blocks = max(len(remote_matched_blocks) - fragment12_num_blocks, 0)
        fragment23_num_blocks = fragment2_num_blocks + fragment3_num_blocks

        fragment123_gpu_blocks = gpu_block_ids[:fragment123_num_blocks]
        fragment123_cpu_blocks = cpu_matched_blocks
        fragment2_ssd_blocks = ssd_matched_blocks[-fragment2_num_blocks:]
        fragment3_remote_blocks = remote_matched_blocks[-fragment3_num_blocks:]
        fragment3_remote_file_nodeids = None
        if shared_pcfs_read:
            fragment3_remote_file_nodeids = remote_file_nodeids[-fragment3_num_blocks:]
        cpu_node_to_unlock = cpu_matched_result.last_ready_node
        ssd_node_to_unlock = ssd_matched_result.last_ready_node
        remote_node_to_unlock = remote_matched_result.last_ready_node
        cpu_blocks_to_free = np.array([], dtype=np.int64)

        if fragment23_num_blocks > 0:
            num_extra_required_blocks = fragment23_num_blocks
            fragment23_cpu_blocks = self.cpu_cache_engine.take(
                num_required_blocks=num_extra_required_blocks,
                protected_node=cpu_matched_result.last_node,
                strict=True
            )
            if len(fragment23_cpu_blocks) < num_extra_required_blocks:
                self.cpu_cache_engine.recycle(fragment23_cpu_blocks)
                # Record allocation failure (resource unavailable, not cache miss)
                if self._metrics_collector is not None:
                    self._metrics_collector.record_allocation_failure("global")
                return self._empty_get_return(request_id)
            fragment123_cpu_blocks = np.concatenate([fragment123_cpu_blocks, fragment23_cpu_blocks])
            # we only insert the buffer blocks to cpu cache engine only:
            # 1. the cpu cache engine satisfies prefix cache after insertion
            # 2. the sequence is all ready blocks
            if (cpu_matched_result.num_ready_matched_blocks >= block_mask_start and
                cpu_matched_result.num_ready_matched_blocks == cpu_matched_result.num_matched_blocks):
                cpu_node_to_unlock = self.cpu_cache_engine.insert(sequence_meta,
                                                                  fragment23_cpu_blocks,
                                                                  num_insert_blocks=fragment123_num_blocks + \
                                                                    block_mask_start,
                                                                  is_ready=False,
                                                                  match_result=cpu_matched_result)
            else:
                cpu_blocks_to_free = fragment23_cpu_blocks

        # Record cache hit/miss metrics after confirming successful allocation
        if self._metrics_collector is not None:
            total_query_blocks = block_mask_end - block_mask_start
            # CPU hit blocks (directly from CPU cache)
            self._metrics_collector.record_cache_hit("cpu", fragment1_num_blocks)
            # SSD hit blocks (blocks loaded from SSD)
            self._metrics_collector.record_cache_hit("ssd", fragment2_num_blocks)
            # Remote hit blocks (blocks loaded from remote)
            self._metrics_collector.record_cache_hit("remote", fragment3_num_blocks)
            # Miss blocks (not in any cache)
            miss_blocks = total_query_blocks - fragment123_num_blocks
            if miss_blocks > 0:
                self._metrics_collector.record_cache_miss(miss_blocks)

        op_disk2h = None
        if fragment2_num_blocks > 0:
            op_disk2h = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.DISK2H,
                src_block_ids = fragment2_ssd_blocks,
                dst_block_ids = fragment123_cpu_blocks[fragment1_num_blocks:fragment12_num_blocks],    
                dp_client_id = dp_client_id,
            )
            transfer_graph.add_transfer_op(op_disk2h)

        op_remote2h = None
        if fragment3_num_blocks > 0:
            op_remote2h = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.REMOTE2H,
                src_block_ids = fragment3_remote_blocks,
                dst_block_ids = fragment123_cpu_blocks[-fragment3_num_blocks:],
                src_block_node_ids = fragment3_remote_file_nodeids,
                dp_client_id = dp_client_id,
            )
            transfer_graph.add_transfer_op(op_remote2h)

        # prepare ssd blocks to transfer
        write_ssd_blocks_from_remote = False
        if (enable_ssd and
            op_remote2h is not None and
            ssd_matched_result.num_ready_matched_blocks >= block_mask_start and
            ssd_matched_result.num_ready_matched_blocks == ssd_matched_result.num_matched_blocks):
            # only when the above all are satisfied, we load data back from cpu to ssd
            write_ssd_blocks_from_remote = True
            fragment3_ssd_blocks = self.ssd_cache_engine.take(
                num_required_blocks=fragment3_num_blocks,
                protected_node=ssd_matched_result.last_node,
                strict=False
            )
            if len(fragment3_ssd_blocks) < fragment3_num_blocks:
                self.ssd_cache_engine.recycle(fragment3_ssd_blocks)
                write_ssd_blocks_from_remote = False
            if write_ssd_blocks_from_remote:
                op_h2disk = TransferOp(
                    graph_id = transfer_graph.graph_id,
                    transfer_type = TransferType.H2DISK,
                    src_block_ids = fragment123_cpu_blocks[-fragment3_num_blocks:],
                    dst_block_ids = fragment3_ssd_blocks,
                    dp_client_id = dp_client_id,
                )
                transfer_graph.add_transfer_op(op_h2disk)
                transfer_graph.add_dependency(op_h2disk.op_id, op_remote2h.op_id)

                ssd_node_to_unlock = self.ssd_cache_engine.insert(sequence_meta,
                                                                fragment3_ssd_blocks,
                                                                num_insert_blocks=fragment123_num_blocks + \
                                                                    block_mask_start,
                                                                is_ready=False,
                                                                match_result=ssd_matched_result)
        if enable_gpu:
            op_h2d = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.H2D,
                src_block_ids = fragment123_cpu_blocks,
                dst_block_ids = fragment123_gpu_blocks,
                dp_client_id = dp_client_id,
            )
            transfer_graph.add_transfer_op(op_h2d)
            if op_disk2h is not None:
                transfer_graph.add_dependency(op_h2d.op_id, op_disk2h.op_id)
            if op_remote2h is not None:
                transfer_graph.add_dependency(op_h2d.op_id, op_remote2h.op_id)
            finished_ops_ids.append(op_h2d.op_id)

        node_to_unlock = {}
        if cpu_node_to_unlock is not None:
            node_to_unlock[DeviceType.CPU] = (cpu_node_to_unlock, cpu_node_to_unlock.size())
        if ssd_node_to_unlock is not None:
            node_to_unlock[DeviceType.SSD] = (ssd_node_to_unlock, ssd_node_to_unlock.size())
        if remote_node_to_unlock is not None:
            node_to_unlock[DeviceType.REMOTE] = (remote_node_to_unlock, remote_node_to_unlock.size())

        buffer_to_free = {DeviceType.CPU: cpu_blocks_to_free}

        # NOTE: for now in build transfer graph, we assume that cpu works as a cache for ssd
        return (
            transfer_graph, finished_ops_ids, node_to_unlock, {}, buffer_to_free,
            len(fragment123_gpu_blocks) if enable_gpu else 0  # op_node_to_ready: {}
        )

    def _get_impl_local(self,
                        request_id: int,
                        sequence_meta: SequenceMeta,
                        block_mask_start: int,
                        block_mask_end: int,
                        gpu_block_ids: np.ndarray,
                        temp_cache_strategy: CacheStrategy,
                        dp_client_id: int) \
                            -> Tuple[TransferOpGraph, List[int], Dict, Dict, Dict, int]:
        """
        transfer pattern:

        GPU          : (gpu cached) | fragment1 | fragment2      | (need compute)
                               ↑          ↑
        CPU(+peerCPU):     ...      | fragment1 | fragment2(new) | (uncached)
                                          ↑
        SSD(+peerSSD):     ...      | fragment1 | fragment2      | (uncached)

        """
        nvtx_range = nvtx.start_range(message=f"CacheEngine.get_impl_local[{request_id}]", color="cyan")
        enable_gpu = not temp_cache_strategy.ignore_gpu
        enable_cpu = self.cache_config.enable_cpu
        enable_ssd = self.cache_config.enable_ssd and not temp_cache_strategy.ignore_ssd
        enable_gds = self.cache_config.enable_gds and not temp_cache_strategy.ignore_gds
        assert enable_cpu
        assert self.cpu_cache_engine is not None

        if self.index_accel:
            cpu_matched_result, ssd_matched_result = self.match_local_accel(sequence_meta, temp_cache_strategy, is_put=False, gpu_matched_blocks=block_mask_start)
        else:
            cpu_matched_result, ssd_matched_result = self.match_local(sequence_meta, temp_cache_strategy)


        # DEBUG: Log GET operation with hash info
        #if len(sequence_meta.block_hashes) > 0:
        #    print(f"[GET {request_id}] hash[0]={sequence_meta.block_hashes[0]}, CPU={cpu_matched_result.num_matched_blocks}/{cpu_matched_result.num_ready_matched_blocks}, SSD={ssd_matched_result.num_matched_blocks}/{ssd_matched_result.num_ready_matched_blocks}, pos_CPU={cpu_matched_result.matched_pos}, pos_SSD={ssd_matched_result.matched_pos}")

        # tailor the blocks to assure:
        # the blocks are needed by the mask & the blocks are ready
        cpu_matched_blocks = cpu_matched_result.physical_blocks[:cpu_matched_result.num_ready_matched_blocks]
        cpu_matched_blocks = cpu_matched_blocks[block_mask_start:block_mask_end]
        # if ssd disabled, len(ssd_physical_blocks) is 0
        ssd_matched_blocks = ssd_matched_result.physical_blocks[:ssd_matched_result.num_ready_matched_blocks]
        ssd_matched_blocks = ssd_matched_blocks[block_mask_start:block_mask_end]

        # TODO: is this possible?
        if len(cpu_matched_blocks) > len(ssd_matched_blocks):
            ssd_matched_blocks = np.array([], dtype=np.int64)

        fragment12_num_blocks = max(len(cpu_matched_blocks), len(ssd_matched_blocks))
        fragment1_num_blocks = len(cpu_matched_blocks)
        fragment2_num_blocks = max(len(ssd_matched_blocks) - len(cpu_matched_blocks), 0)
        #early return if no blocks to transfer
        if fragment12_num_blocks == 0:
            # All cache levels missed - record miss for all requested blocks
            if self._metrics_collector is not None:
                total_query_blocks = block_mask_end - block_mask_start
                if total_query_blocks > 0:
                    self._metrics_collector.record_cache_miss(total_query_blocks)
            nvtx.end_range(nvtx_range)
            return self._empty_get_return(request_id)
        assert fragment12_num_blocks <= len(gpu_block_ids)

        transfer_graph = TransferOpGraph()
        finished_ops_ids = []
        op_node_to_ready = {}

        fragment12_gpu_blocks = gpu_block_ids[:fragment12_num_blocks]
        fragment2_ssd_blocks = ssd_matched_blocks[-fragment2_num_blocks:]
        fragment1_cpu_blocks = cpu_matched_blocks[:fragment1_num_blocks]

        cpu_node_to_unlock = cpu_matched_result.last_ready_node
        ssd_node_to_unlock = ssd_matched_result.last_ready_node

        # prepare cpu blocks to transfer
        cpu_blocks_to_free = np.array([], dtype=np.int64)
        op_disk2h = None
        op_gds_transfer = None
        fragment2_cpu_blocks = None

        #allocated new cpu blocks for this request
        allocated_cpu_block_num = fragment2_num_blocks
        # NOTE: When matched_pos is "remote", we ALWAYS need to allocate local CPU blocks
        # to receive the data, regardless of whether we insert to local index or not
        if cpu_matched_result.matched_pos == "remote" and fragment1_num_blocks > 0:
            allocated_cpu_block_num += fragment1_num_blocks
        nvtx.push_range(f"take {allocated_cpu_block_num} cpu blocks", color="green")
        allocated_cpu_blocks = self.cpu_cache_engine.take(
            num_required_blocks=allocated_cpu_block_num,
            protected_node=cpu_matched_result.last_node,
            strict=False
        )
        nvtx.pop_range()
        # NOTE: not enough space to allocate, skip the request
        # there might be a better way to handle this
        if len(allocated_cpu_blocks) < allocated_cpu_block_num:
            self.cpu_cache_engine.recycle(allocated_cpu_blocks)
            # Record allocation failure (resource unavailable, not cache miss)
            if self._metrics_collector is not None:
                self._metrics_collector.record_allocation_failure("local")
            nvtx.end_range(nvtx_range)
            return self._empty_get_return(request_id)

        # Record cache hit/miss metrics after confirming successful allocation
        if self._metrics_collector is not None:
            total_query_blocks = block_mask_end - block_mask_start
            # CPU hit blocks (directly from CPU cache)
            self._metrics_collector.record_cache_hit("cpu", fragment1_num_blocks)
            # SSD hit blocks (blocks loaded from SSD to CPU)
            self._metrics_collector.record_cache_hit("ssd", fragment2_num_blocks)
            # Miss blocks (not in any cache)
            miss_blocks = total_query_blocks - fragment12_num_blocks
            if miss_blocks > 0:
                self._metrics_collector.record_cache_miss(miss_blocks)

        if cpu_matched_result.matched_pos == "remote" and fragment1_num_blocks > 0:
            fragment1_cpu_blocks_local = allocated_cpu_blocks[-fragment1_num_blocks:]
            op_peerh2h = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.PEERH2H,
                src_block_ids = fragment1_cpu_blocks,
                dst_block_ids = fragment1_cpu_blocks_local,
                remote_node_ids = cpu_matched_result.matched_node_ids,
                src_block_node_ids = cpu_matched_result.matched_node_ids,  # Add this for worker
                dp_client_id = dp_client_id,
            )
            transfer_graph.add_transfer_op(op_peerh2h)
            # Phase D-3 (proposal_unify_with_graph_dispatch_2026-05-15.md
            # §6.4): in a multi-SD instance, fan the PEERH2H out to one
            # clone per peer SD so each SD pulls its own slice from the
            # contributing peer instance through that SD's mooncake.
            # Returns [] for single-SD / dist_reuse-off / not-fully-ready,
            # in which case the legacy single-op path stays bit-identical.
            # ``block_mask_start + fragment1_num_blocks - 1`` is the index
            # in the *full* sequence of the last block we're reusing
            # — the prefix terminal block whose hash keys the
            # AggregateRadixTree fully-ready entry.
            peerh2h_clones = self._maybe_attach_multi_sd_peerh2h_ops(
                transfer_graph=transfer_graph,
                op_peerh2h=op_peerh2h,
                sequence_meta=sequence_meta,
                prefix_terminal_block_idx=int(
                    block_mask_start + fragment1_num_blocks - 1
                ),
            )
            #TODO here we dont combine peer cpu or local cpu match results, so we can safely add remote results to local cpu
            #TODO here assume all matched blocks are ready blocks for peer cpu
            if (cpu_matched_result.insert_to_local_cpu_index and
                cpu_matched_result.num_ready_matched_blocks >= block_mask_start and
                cpu_matched_result.num_ready_matched_blocks == cpu_matched_result.num_matched_blocks):
                cpu_node_to_unlock = self.cpu_cache_engine.insert(sequence_meta,
                                                                  fragment1_cpu_blocks_local,
                                                                  is_ready=False)
                op_node_to_ready[op_peerh2h.op_id] = (DeviceType.CPU, cpu_node_to_unlock, cpu_node_to_unlock.size())
            else:
                cpu_blocks_to_free = np.concatenate([cpu_blocks_to_free, fragment1_cpu_blocks_local])
        else:
            peerh2h_clones = []

        if fragment2_num_blocks > 0:
            if enable_gds:
                # For GDS, transfer directly from SSD to GPU using GDS transfer path (DISK2D)
                op_gds_transfer = TransferOp(
                    graph_id = transfer_graph.graph_id,
                    transfer_type = TransferType.DISK2D,
                    src_block_ids = fragment2_ssd_blocks,
                    dst_block_ids = fragment12_gpu_blocks[-fragment2_num_blocks:],
                    dp_client_id = dp_client_id,
                )
                transfer_graph.add_transfer_op(op_gds_transfer)
                finished_ops_ids.append(op_gds_transfer.op_id)
                op_node_to_ready[op_gds_transfer.op_id] = (DeviceType.SSD,
                                                           ssd_node_to_unlock,
                                                           ssd_node_to_unlock.size())
            else:
                fragment2_cpu_blocks = allocated_cpu_blocks[:fragment2_num_blocks]

                op_disk2h = TransferOp(
                    graph_id = transfer_graph.graph_id,
                    transfer_type = TransferType.PEERSSD2H if ssd_matched_result.matched_pos == "remote" else TransferType.DISK2H,
                    src_block_ids = fragment2_ssd_blocks,
                    dst_block_ids = fragment2_cpu_blocks,
                    remote_node_ids = ssd_matched_result.matched_node_ids if ssd_matched_result.matched_pos == "remote" else None,
                    src_block_node_ids = ssd_matched_result.matched_node_ids if ssd_matched_result.matched_pos == "remote" else None,
                    dp_client_id = dp_client_id,
                )
                transfer_graph.add_transfer_op(op_disk2h)
                # we only insert the buffer blocks to cpu cache engine only:
                # 1. the cpu cache engine satisfies prefix cache after insertion
                # 2. the sequence is all ready blocks
                # TODO: for simplicity, if we use peer cpu results, we dont insert the buffer ssd blocks to local cpu any more
                if (cpu_matched_result.matched_pos == "local" and
                    cpu_matched_result.num_ready_matched_blocks >= block_mask_start and
                    cpu_matched_result.num_ready_matched_blocks == cpu_matched_result.num_matched_blocks):
                    cpu_node_to_unlock = self.cpu_cache_engine.insert(sequence_meta,
                                                                    fragment2_cpu_blocks,
                                                                    num_insert_blocks=fragment12_num_blocks + \
                                                                        block_mask_start,
                                                                    is_ready=False,
                                                                    match_result=cpu_matched_result)
                    op_node_to_ready[op_disk2h.op_id] = (DeviceType.CPU, cpu_node_to_unlock, cpu_node_to_unlock.size())
                else:
                    cpu_blocks_to_free = np.concatenate([cpu_blocks_to_free, fragment2_cpu_blocks])
        if self.cache_config.enable_p2p_cpu and cpu_matched_result.matched_pos == "remote" and fragment1_num_blocks > 0:
            fragment1_cpu_blocks = fragment1_cpu_blocks_local

        if fragment2_cpu_blocks is not None:
            fragment12_cpu_blocks = np.concatenate([fragment1_cpu_blocks, fragment2_cpu_blocks])
        else:
            fragment12_cpu_blocks = fragment1_cpu_blocks

        if enable_gpu:
            op_h2d = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.H2D,
                src_block_ids = fragment12_cpu_blocks if not enable_gds else fragment1_cpu_blocks,
                dst_block_ids = fragment12_gpu_blocks if not enable_gds \
                    else fragment12_gpu_blocks[:fragment1_num_blocks],
                dp_client_id = dp_client_id,
            )
            transfer_graph.add_transfer_op(op_h2d)
            if op_disk2h is not None:
                transfer_graph.add_dependency(op_h2d.op_id, op_disk2h.op_id)
            if cpu_matched_result.matched_pos == "remote" and fragment1_num_blocks > 0:
                transfer_graph.add_dependency(op_h2d.op_id, op_peerh2h.op_id)
                # Phase D-3: H2D must wait for *every* peer-SD PEERH2H
                # clone to land its slice into the master CPU pool
                # before the GPU copy fires.  The peer-SD clones run on
                # their respective Remote handles and their
                # CompletedOp(success=True) flows back to the master
                # polling thread through D-2's _completion_sink, which
                # is what the graph dependency engine waits on.
                for clone in peerh2h_clones:
                    transfer_graph.add_dependency(op_h2d.op_id, clone.op_id)
            finished_ops_ids.append(op_h2d.op_id)

        node_to_unlock = {}
        if cpu_node_to_unlock is not None:
            node_to_unlock[DeviceType.CPU] = (cpu_node_to_unlock, cpu_node_to_unlock.size())
        if ssd_node_to_unlock is not None:
            node_to_unlock[DeviceType.SSD] = (ssd_node_to_unlock, ssd_node_to_unlock.size())
        buffer_to_free = {DeviceType.CPU: cpu_blocks_to_free}
        nvtx.end_range(nvtx_range)
        return (
            transfer_graph, finished_ops_ids, node_to_unlock, op_node_to_ready,
            buffer_to_free, len(fragment12_gpu_blocks) if enable_gpu else 0
        )

    def put(self,
            request_id: int,
            token_ids: np.ndarray,
            token_mask: np.ndarray,
            slot_mapping: np.ndarray,
            dp_client_id: int,
            temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
            namespace: Optional[List[str]] = None) \
                -> Tuple[TransferOpGraph, np.ndarray, Callable, Dict, int]:
        self._check_input(token_ids, token_mask, slot_mapping)
        # ignore the last incomplete block
        aligned_length = (token_ids.shape[0] // self.tokens_per_block) * self.tokens_per_block
        aligned_token_ids = token_ids[:aligned_length]
        token_mask[aligned_length:] = False
        block_start_idx, block_end_idx = self._get_block_range(token_mask)

        # the mask should has a prefix of True
        assert block_start_idx == 0

        gpu_block_ids = self.slot_mapping_to_block_ids(slot_mapping,
                                                       self.tokens_per_block)[:block_end_idx-block_start_idx]

        sequence_meta = SequenceMeta(token_ids=aligned_token_ids,
                                     tokens_per_block=self.cache_config.tokens_per_block,
                                     namespace=namespace)

        assert not temp_cache_strategy.ignore_gpu
        if not self.cache_config.enable_remote or temp_cache_strategy.ignore_remote:
            (transfer_graph, finished_ops_ids, node_to_unlock, op_node_to_ready,
             buffer_to_free, num_gpu_blocks_to_transfer, skipped_gpu_blocks) = \
                self._put_impl_local(
                    request_id,
                    sequence_meta,
                    block_start_idx,
                    block_end_idx,
                    gpu_block_ids,
                    temp_cache_strategy,
                    dp_client_id,
                )
        else:
            (transfer_graph, finished_ops_ids, node_to_unlock, op_node_to_ready,
             buffer_to_free, num_gpu_blocks_to_transfer, skipped_gpu_blocks) = \
                self._put_impl_global(
                    request_id,
                    sequence_meta,
                    block_start_idx,
                    block_end_idx,
                    gpu_block_ids,
                    temp_cache_strategy,
                    dp_client_id,
                )

        transfer_graph, task_end_op_id = add_virtual_op_for_multiple_finished_ops(
            transfer_graph,
            finished_ops_ids,
            dp_client_id,
        )

        return_mask = np.zeros_like(token_mask, dtype=np.bool_)
        return_mask[(block_start_idx + skipped_gpu_blocks)* self.tokens_per_block:
                    (block_start_idx + skipped_gpu_blocks + num_gpu_blocks_to_transfer) * self.tokens_per_block] = True

        for device_type in node_to_unlock:
            self.cache_engines[device_type].lock_node(node_to_unlock[device_type][0])

        # §2.1 dist_reuse PUT-path glue: once the local PUT really
        # lands its block meta in Redis (which happens inside
        # ``_transfer_callback`` via ``insert_and_publish``), we want
        # to tell the AggregateRadixTree that *this* SD now contributes
        # to the prefix.  ``_notify_sd_ready_on_put`` is a no-op when
        # dist_reuse isn't attached, so passing the parameters
        # unconditionally keeps the legacy path zero-overhead.
        sd_notify_kwargs = {
            "sequence_meta": sequence_meta,
            "inserted_block_ids": gpu_block_ids[
                skipped_gpu_blocks: skipped_gpu_blocks + num_gpu_blocks_to_transfer
            ] if num_gpu_blocks_to_transfer > 0 else None,
            "block_start_idx": int(block_start_idx + skipped_gpu_blocks),
            "num_blocks_inserted": int(num_gpu_blocks_to_transfer),
        }
        callback = partial(self._transfer_callback,
                           node_to_unlock=node_to_unlock,
                           buffer_to_free=buffer_to_free,
                           is_put=True,
                           sd_notify_kwargs=sd_notify_kwargs)

        op_callback_dict = {}
        for op_id in op_node_to_ready:
            op_callback_dict[op_id] = partial(self._op_callback,
                                              device_type=op_node_to_ready[op_id][0],
                                              node_to_ready=op_node_to_ready[op_id][1],
                                              ready_length=op_node_to_ready[op_id][2])

        # Record metrics for PUT operation
        if self._metrics_collector is not None:
            self._record_transfer_ops(transfer_graph, "put")
            self._update_mempool_metrics()

        return transfer_graph, return_mask, callback, op_callback_dict, task_end_op_id

    def _put_impl_global(self,
            request_id: int,
            sequence_meta: SequenceMeta,
            block_mask_start: int,
            block_mask_end: int,
            gpu_block_ids: np.ndarray,
            temp_cache_strategy: CacheStrategy,
            dp_client_id: int) \
                -> Tuple[TransferOpGraph, List[int], Dict, Dict, Dict, int, int]:
        """
        transfer pattern:

        GPU:   (skipped)  | fragment1      | fragment2      | (uncompleted block)
                               ↓                ↓
        CPU: (cpu cached) | fragment1(new) | fragment2(new) |
                                                ↓
        SSD:          (ssd cached)         | fragment2(new) |

        CPU:            ...           |     fragment3      |
                                               ↓ (from cpu)
        REMOTE:     (remote cached)   |   fragment3(new)   |

        """
        enable_gpu = not temp_cache_strategy.ignore_gpu
        enable_cpu = self.cache_config.enable_cpu
        enable_ssd = self.cache_config.enable_ssd and not temp_cache_strategy.ignore_ssd
        enable_remote = self.cache_config.enable_remote and not temp_cache_strategy.ignore_remote
        assert enable_gpu
        assert enable_cpu
        assert enable_remote
        assert self.cpu_cache_engine is not None
        assert self.remote_cache_engine is not None

        if self.index_accel:
            cpu_matched_result, ssd_matched_result, remote_matched_result = self.match_all_accel(sequence_meta,
                                                                                               temp_cache_strategy=temp_cache_strategy,
                                                                                               is_get=False)
        else:
            cpu_matched_result, ssd_matched_result, remote_matched_result = self.match_all(sequence_meta,
                                                                                           temp_cache_strategy=temp_cache_strategy)
        cpu_matched_blocks = cpu_matched_result.physical_blocks[
            :cpu_matched_result.num_matched_blocks][block_mask_start:block_mask_end]
        ssd_matched_blocks = ssd_matched_result.physical_blocks[
            :ssd_matched_result.num_matched_blocks][block_mask_start:block_mask_end]
        remote_matched_blocks = remote_matched_result.physical_blocks[
            :remote_matched_result.num_matched_blocks][block_mask_start:block_mask_end]

        num_skipped_blocks = len(cpu_matched_blocks)
        fragment12_num_blocks = len(gpu_block_ids) - num_skipped_blocks
        if fragment12_num_blocks == 0:
            return self._empty_put_return(request_id)
        fragment2_num_blocks = len(gpu_block_ids) - len(ssd_matched_blocks)
        if not enable_ssd:
            fragment2_num_blocks = 0
        fragment3_num_blocks = len(gpu_block_ids) - len(remote_matched_blocks)

        fragment12_gpu_blocks = gpu_block_ids[num_skipped_blocks:]

        fragment12_cpu_blocks = self.cpu_cache_engine.take(
            num_required_blocks=fragment12_num_blocks,
            protected_node = cpu_matched_result.last_node,
            strict=False
        )
        if len(fragment12_cpu_blocks) < fragment12_num_blocks:
            self.cpu_cache_engine.recycle(fragment12_cpu_blocks)
            return self._empty_put_return(request_id)
        put_to_ssd = False
        if enable_ssd and fragment2_num_blocks > 0:
            fragment2_ssd_blocks = self.ssd_cache_engine.take(
                num_required_blocks=fragment2_num_blocks,
                protected_node = ssd_matched_result.last_node,
                strict=False
            )
            if len(fragment2_ssd_blocks) == fragment2_num_blocks:
                put_to_ssd = True
            else:
                self.ssd_cache_engine.recycle(fragment2_ssd_blocks)
        else:
            fragment2_ssd_blocks = np.array([], dtype=np.int64)
        put_to_remote = False
        if fragment3_num_blocks > 0:
            fragment3_remote_blocks = self.remote_cache_engine.take(
                num_required_blocks=fragment3_num_blocks,
                protected_node = remote_matched_result.last_node,
                strict=False
            )
            if len(fragment3_remote_blocks) == fragment3_num_blocks:
                put_to_remote = True
            else:
                self.remote_cache_engine.recycle(fragment3_remote_blocks)
        else:
            fragment3_remote_blocks = np.array([], dtype=np.int64)

        transfer_graph = TransferOpGraph()
        finished_ops_ids = []

        op_d2h = TransferOp(
            graph_id = transfer_graph.graph_id,
            transfer_type = TransferType.D2H,
            src_block_ids = fragment12_gpu_blocks,
            dst_block_ids = fragment12_cpu_blocks,
            dp_client_id = dp_client_id,
        )
        transfer_graph.add_transfer_op(op_d2h)
        finished_ops_ids.append(op_d2h.op_id)

        # Phase D-2: tag self-SD + clone for each peer SD.
        self._maybe_attach_multi_sd_d2h_ops(transfer_graph, op_d2h)

        if put_to_ssd:
            if len(fragment12_cpu_blocks) < fragment2_num_blocks:
                num_needed_from_cpu_matched = fragment2_num_blocks - len(fragment12_cpu_blocks)
                fragment2_cpu_blocks = np.concatenate([cpu_matched_blocks[-num_needed_from_cpu_matched:], \
                    fragment12_cpu_blocks])
            else:
                fragment2_cpu_blocks = fragment12_cpu_blocks[-fragment2_num_blocks:]
            op_h2disk = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.H2DISK,
                src_block_ids = fragment2_cpu_blocks,
                dst_block_ids = fragment2_ssd_blocks,
                dp_client_id = dp_client_id,
            )
            transfer_graph.add_transfer_op(op_h2disk)

            transfer_graph.add_dependency(op_h2disk.op_id, op_d2h.op_id)

        if put_to_remote:
            if fragment3_num_blocks > fragment12_num_blocks:
                extra_num_cpu_blocks = fragment3_num_blocks - fragment12_num_blocks
                fragment3_cpu_blocks = np.concatenate([fragment12_cpu_blocks,
                                                  cpu_matched_blocks[-extra_num_cpu_blocks:]])
            else:
                fragment3_cpu_blocks = fragment12_cpu_blocks[-fragment3_num_blocks:]
            op_h2remote = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.H2REMOTE,
                src_block_ids = fragment3_cpu_blocks,
                dst_block_ids = fragment3_remote_blocks,
                dp_client_id = dp_client_id,
            )
            transfer_graph.add_transfer_op(op_h2remote)
            transfer_graph.add_dependency(op_h2remote.op_id, op_d2h.op_id)

        cpu_node_to_unlock = self.cpu_cache_engine.insert(sequence_meta,
                                                          fragment12_cpu_blocks,
                                                          is_ready=False,
                                                          match_result=cpu_matched_result)
        ssd_node_to_unlock = None
        if put_to_ssd:
            ssd_node_to_unlock = self.ssd_cache_engine.insert(sequence_meta,
                                                            fragment2_ssd_blocks,
                                                            is_ready=False,
                                                            match_result=ssd_matched_result)
        remote_node_to_unlock = None
        if put_to_remote:
            remote_node_to_unlock = self.remote_cache_engine.insert(sequence_meta,
                                                                    fragment3_remote_blocks,
                                                                    is_ready=False,
                                                                    match_result=remote_matched_result)
        node_to_unlock = {}
        if cpu_node_to_unlock is not None:
            node_to_unlock[DeviceType.CPU] = (cpu_node_to_unlock, cpu_node_to_unlock.size())
        if ssd_node_to_unlock is not None:
            node_to_unlock[DeviceType.SSD] = (ssd_node_to_unlock, ssd_node_to_unlock.size())
        if remote_node_to_unlock is not None:
            node_to_unlock[DeviceType.REMOTE] = (remote_node_to_unlock, remote_node_to_unlock.size())

        skipped_gpu_blocks = len(cpu_matched_blocks)
        return (
            transfer_graph, finished_ops_ids, node_to_unlock, {}, {},
            len(fragment12_gpu_blocks), skipped_gpu_blocks  # op_node_to_ready: {}
        )

    def _put_impl_local(self,
            request_id: int,
            sequence_meta: SequenceMeta,
            block_mask_start: int,
            block_mask_end: int,
            gpu_block_ids: np.ndarray,
            temp_cache_strategy: CacheStrategy,
            dp_client_id: int) \
                -> Tuple[TransferOpGraph, List[int], Dict, Dict, Dict, int, int]:
        """
        transfer pattern:

        GPU:   (skipped)  | fragment1      | fragment2      | (uncompleted block)
                                ↓                ↓
        CPU: (cpu cached) | fragment1(new) | fragment2(new) |
                                                 ↓
        SSD:          (ssd cached)         | fragment2(new) |

        """
        enable_gpu = not temp_cache_strategy.ignore_gpu
        enable_cpu = self.cache_config.enable_cpu
        enable_ssd = self.cache_config.enable_ssd and not temp_cache_strategy.ignore_ssd
        enable_gds = self.cache_config.enable_gds and not temp_cache_strategy.ignore_gds
        assert enable_gpu
        assert enable_cpu
        assert self.cpu_cache_engine is not None

        if self.index_accel:
            cpu_matched_result, ssd_matched_result = self.match_local_accel(sequence_meta,
                                                                            temp_cache_strategy=temp_cache_strategy,
                                                                            is_put=True)
        else:
            cpu_matched_result, ssd_matched_result = self.match_local(sequence_meta,
                                                                      temp_cache_strategy=temp_cache_strategy,
                                                                      is_put=True)
        cpu_matched_blocks = cpu_matched_result.physical_blocks[
            :cpu_matched_result.num_matched_blocks][block_mask_start:block_mask_end]
        ssd_matched_blocks = ssd_matched_result.physical_blocks[
            :ssd_matched_result.num_matched_blocks][block_mask_start:block_mask_end]

        #if len(cpu_matched_blocks) > len(ssd_matched_blocks):
        #    print(f"[PUT_LOCAL] CPU matched blocks are greater than SSD matched blocks, skipping")
        #    return self._empty_put_return(request_id)


        num_skipped_blocks = len(cpu_matched_blocks)
        fragment12_num_blocks = len(gpu_block_ids) - num_skipped_blocks
        if fragment12_num_blocks == 0:
            return self._empty_put_return(request_id)
        fragment2_num_blocks = len(gpu_block_ids) - len(ssd_matched_blocks)
        if not enable_ssd:
            fragment2_num_blocks = 0

        fragment12_gpu_blocks = gpu_block_ids[num_skipped_blocks:]

        fragment12_cpu_blocks = self.cpu_cache_engine.take(
            num_required_blocks=fragment12_num_blocks,
            protected_node = cpu_matched_result.last_node,
            strict=False
        )

        if enable_ssd:
            fragment2_ssd_blocks = self.ssd_cache_engine.take(
                num_required_blocks=fragment2_num_blocks,
                protected_node = ssd_matched_result.last_node,
                strict=False
            )
        else:
            fragment2_ssd_blocks = np.array([], dtype=np.int64)

        if len(fragment12_cpu_blocks) < fragment12_num_blocks or \
            len(fragment2_ssd_blocks) < fragment2_num_blocks:
            print(f"[WARNING] PUT request {request_id} FAILED: CPU={len(fragment12_cpu_blocks)}/{fragment12_num_blocks}, SSD={len(fragment2_ssd_blocks)}/{fragment2_num_blocks}")
            self.cpu_cache_engine.recycle(fragment12_cpu_blocks)
            if enable_ssd:
                self.ssd_cache_engine.recycle(fragment2_ssd_blocks)
            return self._empty_put_return(request_id)

        transfer_graph = TransferOpGraph()
        finished_ops_ids = []
        op_node_to_ready = {}

        op_d2h = TransferOp(
            graph_id = transfer_graph.graph_id,
            transfer_type = TransferType.D2H,
            src_block_ids = fragment12_gpu_blocks,
            dst_block_ids = fragment12_cpu_blocks,
            dp_client_id = dp_client_id,
        )
        transfer_graph.add_transfer_op(op_d2h)
        finished_ops_ids.append(op_d2h.op_id)

        # Phase D-2: tag self-SD + clone for each peer SD.
        self._maybe_attach_multi_sd_d2h_ops(transfer_graph, op_d2h)

        if fragment2_num_blocks > 0:
            if len(fragment12_cpu_blocks) < fragment2_num_blocks:
                flexkv_logger.warning(f"fragment12_cpu_blocks: {len(fragment12_cpu_blocks)}, "
                                      f"fragment2_num_blocks: {fragment2_num_blocks}, "
                                      f"cpu match blocks are bigger than SSD match blocks number. "
                                      f"This should not often happen if CPU cache size is smaller than SSD cache size.")
                num_needed_from_cpu_matched = fragment2_num_blocks - len(fragment12_cpu_blocks)
                fragment2_cpu_blocks = np.concatenate([cpu_matched_blocks[-num_needed_from_cpu_matched:], \
                    fragment12_cpu_blocks])
            else:
                fragment2_cpu_blocks = fragment12_cpu_blocks[-fragment2_num_blocks:]
            op_h2disk = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.H2DISK,
                src_block_ids = fragment2_cpu_blocks,
                dst_block_ids = fragment2_ssd_blocks,
                dp_client_id = dp_client_id,
            )
            transfer_graph.add_transfer_op(op_h2disk)

            transfer_graph.add_dependency(op_h2disk.op_id, op_d2h.op_id)

        """insert and lock"""
        cpu_node_to_unlock = self.cpu_cache_engine.insert(sequence_meta,
                                                          fragment12_cpu_blocks,
                                                          is_ready=False,
                                                          match_result=cpu_matched_result)
        op_node_to_ready[op_d2h.op_id] = (DeviceType.CPU, cpu_node_to_unlock, cpu_node_to_unlock.size())
        ssd_node_to_unlock = None
        if len(fragment2_ssd_blocks) > 0:
            ssd_node_to_unlock = self.ssd_cache_engine.insert(sequence_meta,
                                                            fragment2_ssd_blocks,
                                                            is_ready=False,
                                                            match_result=ssd_matched_result)
            op_node_to_ready[op_h2disk.op_id] = (DeviceType.SSD, ssd_node_to_unlock, ssd_node_to_unlock.size())
        node_to_unlock = {}
        if cpu_node_to_unlock is not None:
            node_to_unlock[DeviceType.CPU] = (cpu_node_to_unlock, cpu_node_to_unlock.size())
        if ssd_node_to_unlock is not None:
            node_to_unlock[DeviceType.SSD] = (ssd_node_to_unlock, ssd_node_to_unlock.size())

        skipped_gpu_blocks = len(cpu_matched_blocks)
        return (
            transfer_graph, finished_ops_ids, node_to_unlock, op_node_to_ready, {},
            len(fragment12_gpu_blocks), skipped_gpu_blocks
        )

    def _transfer_callback(self,
                           node_to_unlock: Dict[DeviceType, Tuple[RadixNode, int]],
                           buffer_to_free: Optional[Dict[DeviceType, np.ndarray]] = None,
                           is_put: bool = False,
                           sd_notify_kwargs: Optional[Dict] = None) -> None:
        if DeviceType.CPU in node_to_unlock:
            assert self.cpu_cache_engine is not None
            cpu_node = node_to_unlock[DeviceType.CPU][0]
            self.cpu_cache_engine.unlock(cpu_node)
            self.cpu_cache_engine.set_ready(cpu_node, True, node_to_unlock[DeviceType.CPU][1])
            if is_put and self.cache_config.enable_p2p_cpu:
                self.cpu_cache_engine.local_index.insert_and_publish(cpu_node)
        if DeviceType.SSD in node_to_unlock:
            assert self.ssd_cache_engine is not None
            ssd_node = node_to_unlock[DeviceType.SSD][0]
            self.ssd_cache_engine.unlock(ssd_node)
            self.ssd_cache_engine.set_ready(ssd_node, True, node_to_unlock[DeviceType.SSD][1])
            if is_put and self.cache_config.enable_p2p_ssd:
                self.ssd_cache_engine.local_index.insert_and_publish(node_to_unlock[DeviceType.SSD][0])
        if DeviceType.REMOTE in node_to_unlock:
            assert self.remote_cache_engine is not None
            self.remote_cache_engine.unlock(node_to_unlock[DeviceType.REMOTE][0])
            self.remote_cache_engine.set_ready(
                node_to_unlock[DeviceType.REMOTE][0], True, node_to_unlock[DeviceType.REMOTE][1]
            )
            if is_put and self.enable_kv_sharing:
                self.remote_cache_engine.insert_and_publish(node_to_unlock[DeviceType.REMOTE][0])
        if buffer_to_free is not None:
            if DeviceType.CPU in buffer_to_free:
                assert self.cpu_cache_engine is not None
                self.cpu_cache_engine.recycle(buffer_to_free[DeviceType.CPU])
            if DeviceType.SSD in buffer_to_free:
                assert self.ssd_cache_engine is not None
                self.ssd_cache_engine.recycle(buffer_to_free[DeviceType.SSD])
            if DeviceType.REMOTE in buffer_to_free:
                assert self.remote_cache_engine is not None
                self.remote_cache_engine.recycle(buffer_to_free[DeviceType.REMOTE])

        # §2.1 dist_reuse PUT-path glue: once every cache level has
        # unlocked + published, announce the new prefix to our own
        # AggregateRadixTree (self-SD ACK).  Kept after the
        # insert_and_publish calls above so the block meta is
        # guaranteed visible in Redis before the in-memory ack fires.
        # No-op when dist_reuse isn't attached.
        if is_put and sd_notify_kwargs:
            try:
                self._notify_sd_ready_on_put(**sd_notify_kwargs)
            except Exception:
                # Absolute must-not-raise: the callback runs on the
                # transfer worker's completion path.
                pass

    def _op_callback(self, device_type: DeviceType, node_to_ready: RadixNode, ready_length: int) -> None:
        if device_type == DeviceType.CPU:
            assert self.cpu_cache_engine is not None
            self.cpu_cache_engine.set_ready(node_to_ready, True, ready_length)
        elif device_type == DeviceType.SSD:
            assert self.ssd_cache_engine is not None
            self.ssd_cache_engine.set_ready(node_to_ready, True, ready_length)
        elif device_type == DeviceType.REMOTE:
            assert self.remote_cache_engine is not None
            self.remote_cache_engine.set_ready(node_to_ready, True, ready_length)

    # ==================================================================
    # Dist-reuse integration API (Batch F — cache_engine ↔ MasterCoordinator)
    # ==================================================================
    def attach_dist_reuse(self, master_coord) -> None:
        """Wire this cache engine to the ``MasterCoordinator`` that the
        ``KVTaskManager`` built after the Remote ready handshake completed.

        Phase D-4 (proposal_unify_with_graph_dispatch_2026-05-15.md):
        cross-SD coordination flows through the unified
        ``TransferOpGraph`` dispatch path with per-op
        ``target_node_ids`` filtering on each Remote, which replaced
        the previous ``CoordinationCoordinator`` plumbing entirely.
        The deprecated ``coord_dispatcher`` parameter was removed in
        the same cleanup pass that deleted the legacy
        ``coord_query`` / ``coord_get`` / ``coord_put`` ZMQ protocol.

        Args:
            master_coord: A
                :class:`flexkv.common.dist_reuse.MasterCoordinator` owning
                the per-instance ``AggregateRadixTree`` + refcount book-
                keeping + Layer-1 ``FailureDetector``.
        """
        self._master_coord = master_coord

        # Broadcast the refcount guard to every subengine so that each
        # subengine's ``RadixTreeIndex.evict`` (or the accel equivalent,
        # once plumbed on the C++ side) can honour
        # ``MasterCoordinator.is_evictable``.  See §2.2 in
        # docs/dist_reuse/implementation_gap_2026-05-11.md.
        self._broadcast_evict_guard(self.is_evictable)

        # Wire the Layer-1 FailureDetector's on_peer_lost callback to
        # invalidate aggregate-radix entries from the dying peer.  Do
        # this lazily (only if coordinator owns a detector).
        if master_coord is not None and hasattr(master_coord, "set_peer_lost_hook"):
            master_coord.set_peer_lost_hook(self._on_peer_lost)

    def detach_dist_reuse(self) -> None:
        """Drop references to the coordinator — invoked from shutdown."""
        self._master_coord = None
        # Remove the guard — eviction falls back to the legacy
        # behaviour that only looks at ``lock_cnt``.
        self._broadcast_evict_guard(None)

    def _maybe_attach_multi_sd_d2h_ops(
        self,
        transfer_graph,
        op_d2h,
    ) -> None:
        """Phase D-2 (proposal_unify_with_graph_dispatch_2026-05-15.md
        §6.3): when the instance has multiple SDs, mirror the master's
        D2H op into the per-peer-SD form so the broadcast graph
        dispatch on each Remote handle picks up the right slice.

        Specifically:

        * Stamp ``target_node_ids=[self_node_id]`` on the master's own
          ``op_d2h`` so the master in-process handle (and only that
          handle) executes it.  Each peer Remote's ``_handle_submit``
          will drop this op via the ``target_node_ids`` filter.
        * For every peer SD known to the MasterCoordinator (with a
          finished ``RemoteReadyMsg``), append a clone of ``op_d2h``
          with ``target_node_ids=[peer_node_id]``.  Peer Remote's
          ``_handle_submit`` keeps only its own clone.  The peer's
          ``CompletedOp(sd_key, contributing_node_id)`` then routes
          back through the master polling thread's
          ``_completion_sink`` → :meth:`_on_peer_sd_completed_op`.

        Single-SD / dist_reuse-disabled paths leave ``op_d2h``
        untouched (``target_node_ids=None`` ⇒ no filter, identical to
        legacy behaviour).
        """
        if self._master_coord is None:
            return
        try:
            sd_to_nid = self._master_coord.get_sd_to_nid_map()
        except Exception:
            return
        if not sd_to_nid:
            return  # bootstrap not finished yet — leave legacy behaviour.

        try:
            self_sd_str = self._master_coord.self_sd.serialize()
        except Exception:
            return
        self_node_id = sd_to_nid.get(self_sd_str, -1)
        if self_node_id < 0:
            return

        # Tag master's own D2H op.
        try:
            op_d2h.target_node_ids = [int(self_node_id)]
        except Exception:
            return

        # Append a clone per peer SD.
        try:
            from flexkv.common.transfer import TransferOp, TransferType
        except Exception:
            return
        for sd_str, peer_nid in sd_to_nid.items():
            if sd_str == self_sd_str:
                continue
            try:
                peer_op = TransferOp(
                    graph_id=transfer_graph.graph_id,
                    transfer_type=TransferType.D2H,
                    src_block_ids=op_d2h.src_block_ids,
                    dst_block_ids=op_d2h.dst_block_ids,
                    dp_client_id=op_d2h.dp_client_id,
                    target_node_ids=[int(peer_nid)],
                )
                transfer_graph.add_transfer_op(peer_op)
            except Exception as e:  # pragma: no cover
                try:
                    from flexkv.common.debug import flexkv_logger
                    flexkv_logger.warning(
                        f"[DistReuse:D-2] failed to append peer-SD D2H op "
                        f"for sd={sd_str} (nid={peer_nid}): {e}"
                    )
                except Exception:
                    pass

    def _maybe_attach_multi_sd_peerh2h_ops(
        self,
        transfer_graph,
        op_peerh2h,
        sequence_meta,
        prefix_terminal_block_idx: int,
    ) -> List:
        """Phase D-3 (proposal_unify_with_graph_dispatch_2026-05-15.md
        §6.4): mirror of :meth:`_maybe_attach_multi_sd_d2h_ops` for the
        GET-path PEERH2H op when the instance owns multiple SDs.

        Why this is necessary
        ---------------------
        The single-SD PEERH2H op pulls one peer instance's CPU slice
        through mooncake on the *Master's* SD.  In a multi-SD instance
        every other SD must also pull its own slice from the same peer
        instance — otherwise the GPU would see only the Master-SD
        layer/head shard and PP / cross-node TP would be incomplete.

        The legacy single-SD op as constructed by the caller already
        targets the peer instance correctly on the Master's own SD
        (``src_block_node_ids = cpu_matched_result.matched_node_ids``,
        which is the peer instance's master-SD ``distributed_node_id``).
        Phase D-3 transforms it into N ops, one per SD:

        * Stamp ``target_node_ids = [self_node_id]`` on the master's
          op so only the Master's local TransferEngine executes it
          (``_filter_graph_inplace_by_target_node_ids`` on the in-proc
          handle drops it from peer Remotes, and the per-Remote
          ``_filter_graph_by_target_node_ids`` drops it from peers
          on the wire).
        * For every peer SD known to be ``fully_ready`` (per
          :meth:`AggregateRadixTree.match_fully_ready`), append a
          clone with ``target_node_ids = [peer_sd_node_id]`` and
          ``src_block_node_ids = [peer_instance_node_on_that_sd]``
          fetched from ``ReadyEntry.ready_sds``.  The clone runs on
          that peer SD's Remote node and pulls *its* layer/head shard
          from the same peer instance through that peer SD's mooncake
          server.

        Returns the list of clone ops added (empty list when single-SD
        / dist_reuse off / aggregate not fully ready / bootstrap not
        finished).  Caller is expected to add a graph dependency
        ``op_h2d → clone`` for every returned clone so the H2D waits
        for *all* SDs to land their slice before issuing the H2D copy.

        Single-SD / dist_reuse-disabled paths leave ``op_peerh2h``
        untouched (``target_node_ids=None`` ⇒ no filter, identical to
        legacy behaviour).  No new ZMQ messages or sync primitives
        are introduced — completion of each clone flows back through
        the existing ``CompletedOp`` → ``_completion_sink`` route
        already wired up by Phase D-2.
        """
        clones: List = []
        if self._master_coord is None:
            return clones
        try:
            sd_to_nid = self._master_coord.get_sd_to_nid_map()
        except Exception:
            return clones
        if not sd_to_nid:
            return clones  # bootstrap not finished yet — leave legacy behaviour.
        try:
            self_sd_str = self._master_coord.self_sd.serialize()
        except Exception:
            return clones
        self_node_id = sd_to_nid.get(self_sd_str, -1)
        if self_node_id < 0:
            return clones

        # Single-SD instance — no clones to make; do not even tag the
        # master op (legacy code path stays bit-identical).
        if len(sd_to_nid) <= 1:
            return clones

        # Look up the per-SD ``contributing peer node_id`` map from the
        # AggregateRadixTree.  ``ready_sds[sd_key] = peer_instance's
        # node_id on that SD``.  This is populated by D-2's PUT-path
        # ``mark_sd_ready`` calls fired from each peer SD's CompletedOp
        # ack.  If we somehow get here without a fully-ready entry
        # (race with eviction, leak, etc.), fall back to legacy and
        # let ``_sharing_domain_gate_get`` reject the GET downstream.
        try:
            sequence_meta.gen_hashes()
        except Exception:
            return clones
        if (prefix_terminal_block_idx < 0 or
            prefix_terminal_block_idx >= sequence_meta.block_hashes.shape[0]):
            return clones
        try:
            prefix_hash = int(
                sequence_meta.block_hashes[prefix_terminal_block_idx].item()
            )
        except Exception:
            return clones
        try:
            entry = self._master_coord.match_fully_ready(prefix_hash)
        except Exception:
            return clones
        if entry is None or not entry.ready_sds:
            # Aggregate not fully ready → ``_sharing_domain_gate_get``
            # will reject this GET anyway; do not pollute the graph
            # with peer-SD clones that would never resolve.
            return clones

        # Tag master's own PEERH2H op so only the Master executes it.
        try:
            op_peerh2h.target_node_ids = [int(self_node_id)]
        except Exception:
            return clones

        # Append a clone per peer SD, pointing at the same peer
        # instance's slice on that SD.
        try:
            from flexkv.common.transfer import TransferOp, TransferType
        except Exception:
            return clones

        # Cache shape needed for src_block_node_ids: an array of length
        # ``len(src_block_ids)`` filled with the peer's node_id on that
        # SD.  ``cpu_matched_result.matched_node_ids`` (master SD) is
        # already in this shape; we mirror it for each peer SD.
        n_blocks = len(op_peerh2h.src_block_ids)
        for sd_str, peer_sd_nid in sd_to_nid.items():
            if sd_str == self_sd_str:
                continue
            # Peer instance's node_id on *this* peer SD (as recorded
            # by D-2's ack path).  -1 sentinel means we never got a
            # confirming ack from that SD; skip — gate will reject.
            try:
                peer_instance_nid_on_sd = int(entry.ready_sds.get(sd_str, -1))
            except Exception:
                peer_instance_nid_on_sd = -1
            if peer_instance_nid_on_sd < 0:
                continue
            try:
                peer_src_node_ids = np.full(
                    n_blocks, int(peer_instance_nid_on_sd), dtype=np.int64,
                )
                clone = TransferOp(
                    graph_id=transfer_graph.graph_id,
                    transfer_type=TransferType.PEERH2H,
                    src_block_ids=op_peerh2h.src_block_ids,
                    dst_block_ids=op_peerh2h.dst_block_ids,
                    dp_client_id=op_peerh2h.dp_client_id,
                    remote_node_ids=peer_src_node_ids,
                    src_block_node_ids=peer_src_node_ids,
                    target_node_ids=[int(peer_sd_nid)],
                )
                transfer_graph.add_transfer_op(clone)
                clones.append(clone)
            except Exception as e:  # pragma: no cover
                try:
                    from flexkv.common.debug import flexkv_logger
                    flexkv_logger.warning(
                        f"[DistReuse:D-3] failed to append peer-SD PEERH2H op "
                        f"for sd={sd_str} (sd_nid={peer_sd_nid}, "
                        f"peer_inst_nid={peer_instance_nid_on_sd}): {e}"
                    )
                except Exception:
                    pass
        return clones

    def _broadcast_evict_guard(self, fn: Optional[Callable[[int], bool]]) -> None:
        """Install ``fn`` as the refcount guard on every owned subengine.

        Subengines that don't (yet) support a guard expose a no-op
        ``set_evict_guard`` (see ``CacheEngineAccel``).  Legacy engines
        without the method at all are simply skipped — their evict
        behaviour is unchanged.
        """
        for engine in (
            self.cpu_cache_engine,
            self.ssd_cache_engine,
            self.remote_cache_engine,
        ):
            if engine is None:
                continue
            setter = getattr(engine, "set_evict_guard", None)
            if callable(setter):
                try:
                    setter(fn)
                except Exception:
                    # Defensive — a buggy subengine must not wedge
                    # attach/detach for the rest of the system.
                    pass

    def _on_peer_lost(self, peer_instance_id: str) -> None:
        """FailureDetector callback.  Best-effort — MUST NOT raise
        (the callback runs on the detector's polling thread)."""
        if self._master_coord is None:
            return
        try:
            self._master_coord.invalidate_by_peer_instance(peer_instance_id)
        except Exception as e:  # pragma: no cover — defensive
            try:
                from flexkv.common.debug import flexkv_logger
                flexkv_logger.warning(
                    f"[DistReuse] invalidate_by_peer_instance({peer_instance_id}) raised: {e}"
                )
            except Exception:
                pass

    # ---- refcount guard for evict paths --------------------------------
    def is_evictable(self, block_id: int) -> bool:
        """Evict path check: refcount>0 blocks participating in an in-flight
        coord GET must NOT be evicted.  Defaults to True when dist-reuse
        is off (legacy behaviour)."""
        if self._master_coord is None:
            return True
        try:
            return bool(self._master_coord.is_evictable(int(block_id)))
        except Exception:
            return True

    # ---- hooks Master transfer_callback calls when a PUT lands --------
    def _notify_master_sd_ready(
        self,
        prefix_hash: int,
        block_ids: list,
    ) -> None:
        """Phase D-2 (proposal_unify_with_graph_dispatch_2026-05-15.md §6.3):
        announce that the Master's own SD finished publishing a prefix.

        The Master's self-SD ack is recorded in the ``AggregateRadixTree``
        immediately.  In multi-SD deployments the **peer-SD acks arrive
        asynchronously** through the graph-dispatch path: each peer SD's
        ``TransferManagerOnRemote`` runs the per-SD D2H op (filtered into
        its own slice by ``target_node_ids``) and ships back a
        ``CompletedOp(sd_key, contributing_node_id, success=True)``.  The
        Master's ``TransferManagerMultiNodeHandle._completion_sink`` then
        invokes :meth:`_on_peer_sd_completed_op` which calls
        ``mark_sd_ready(peer_sd, node_id=...)``.

        That replaces the old ``coord_put`` broadcast-and-collect pattern
        with a single mechanism (graph dispatch) shared with cross-machine
        TP / PP — see proposal §2.

        For ``total_sd_count == 1`` (the common single-SD shape) this
        method is byte-identical to legacy: only the self-SD mark fires.
        """
        if self._master_coord is None:
            return

        # Self-SD mark — always first, always unconditional.  Pass
        # node_id so the GET-side cross-instance reuse path knows which
        # node holds the master SD's slice.  Best-effort lookup;
        # default to -1 if the master coord doesn't expose it yet.
        try:
            self_node_id = int(getattr(self._master_coord, "self_node_id", -1))
        except Exception:
            self_node_id = -1
        try:
            self._master_coord.mark_sd_ready(
                prefix_hash=int(prefix_hash),
                sd_key_str=self._master_coord.self_sd.serialize(),
                block_ids=list(block_ids) if block_ids is not None else [],
                node_id=self_node_id,
            )
        except Exception as e:  # pragma: no cover
            try:
                from flexkv.common.debug import flexkv_logger
                flexkv_logger.warning(f"[DistReuse] mark_sd_ready raised: {e}")
            except Exception:
                pass

        # Phase D-2: register a pending PUT batch for the
        # ``_completion_sink`` to consume when peer SD CompletedOps
        # arrive.  Keyed by ``prefix_hash`` because the CompletedOp
        # carries no batch identity beyond ``graph_id`` — and a single
        # graph may carry several PUT prefixes (e.g. a merged batch
        # graph).  We use ``prefix_hash`` as the natural identifier
        # because ``mark_sd_ready`` keys on it.  See
        # :meth:`_on_peer_sd_completed_op`.
        try:
            total_sd = int(getattr(self._master_coord, "total_sd_count", 1))
        except Exception:
            total_sd = 1
        if total_sd <= 1:
            return  # No peer SDs to wait for.

        # Stash (prefix_hash, block_ids) keyed by the per-(prefix_hash,
        # peer_sd) tuple that ``_on_peer_sd_completed_op`` will look up.
        # We do not store ``contributing_peer`` here — that comes back
        # on the CompletedOp's ``contributing_node_id`` field and the
        # peer's instance_id can be reverse-looked-up via the
        # MasterCoordinator if needed.
        try:
            block_ids_list = [int(b) for b in (block_ids or [])]
        except Exception:
            block_ids_list = []
        try:
            with self._pending_put_lock:
                self._pending_put_batches[int(prefix_hash)] = block_ids_list
        except AttributeError:
            # Lock not yet initialized (legacy code path that constructs
            # GlobalCacheEngine without going through __init__'s
            # initialization of _pending_put_*).  Initialize lazily and
            # retry once.
            self._pending_put_lock = __import__("threading").Lock()
            self._pending_put_batches: Dict[int, list] = {}
            with self._pending_put_lock:
                self._pending_put_batches[int(prefix_hash)] = block_ids_list

    def _on_peer_sd_completed_op(self, completed_op) -> None:
        """Phase D-2 (proposal §3.5 / §6.3): completion-sink handler.

        Invoked on the master's polling thread for every ``CompletedOp``
        that arrives with a non-empty ``sd_key`` and ``success=True``.
        Each such CompletedOp signals "peer SD ``sd_key`` (held by
        ``contributing_node_id``) finished its share of some PUT batch".

        We map the ``CompletedOp`` back to the prefix_hash via the
        ``_pending_put_batches`` registry populated by
        :meth:`_notify_master_sd_ready`.  When the prefix is fully
        ready (all peer SDs have acked) the entry naturally falls out
        on the next eviction or stays as a no-op for further PUTs.

        The CompletedOp carries no prefix_hash directly — the legacy
        plan was to store ``graph_id → prefix_hash`` but that requires
        threading through the kvtask boundary.  We take a simpler
        route: when only one PUT batch is in flight per peer (the
        common case), there's exactly one ``_pending_put_batches``
        entry and it's the right one.  When multiple PUT batches are
        in flight we mark every pending prefix that the peer hasn't
        acked yet — overhead is O(in-flight batches × peer SDs) and
        bounded by the kvtask scheduler's window.

        NOTE: this is best-effort; absent a graph_id → prefix_hash
        registry, false positives ("mark prefix X ready on SD Y when
        actually a different batch finished") are possible if multiple
        PUTs to the same SD run concurrently with overlapping
        prefixes.  In practice the kvtask scheduler serializes a
        single PUT at a time per request so collisions are rare.
        Phase D-3 will add a ``graph_id → prefix_hash`` registry to
        eliminate the ambiguity.
        """
        if self._master_coord is None:
            return
        sd_key = getattr(completed_op, "sd_key", "") or ""
        if not sd_key:
            return  # Not a peer-SD CompletedOp — ignore.
        if sd_key == self._master_coord.self_sd.serialize():
            # Master's own CompletedOp loops back through the same sink
            # in some test harnesses — self-SD is already marked by
            # _notify_master_sd_ready, ignore.
            return
        if not getattr(completed_op, "success", True):
            # Failed op — let FailureReportMsg handle invalidation.
            return

        node_id = int(getattr(completed_op, "contributing_node_id", -1))
        try:
            with self._pending_put_lock:
                # Mark every still-pending prefix as ready on this SD.
                # The aggregate radix's ``mark_sd_ready`` is idempotent
                # so repeated calls for the same (prefix_hash, sd_key)
                # are harmless.
                pending_snapshot = list(self._pending_put_batches.items())
        except AttributeError:
            return

        for prefix_hash, block_ids_list in pending_snapshot:
            try:
                self._master_coord.mark_sd_ready(
                    prefix_hash=int(prefix_hash),
                    sd_key_str=sd_key,
                    block_ids=block_ids_list,
                    node_id=node_id,
                )
            except Exception as e:  # pragma: no cover
                try:
                    from flexkv.common.debug import flexkv_logger
                    flexkv_logger.debug(
                        f"[DistReuse] _on_peer_sd_completed_op: "
                        f"mark_sd_ready({sd_key}, prefix={prefix_hash}) "
                        f"raised: {e}"
                    )
                except Exception:
                    pass

    # ------------------------------------------------------------------
    # GET-path glue (§2.1 of docs/dist_reuse/implementation_gap_*.md)
    # ------------------------------------------------------------------
    def _sharing_domain_gate_get(
        self,
        *,
        sequence_meta,
        return_mask,
        block_start_idx: int,
        num_gpu_blocks_to_transfer: int,
    ) -> bool:
        """Cross-SD barrier for a GET about to reuse cached blocks.

        **Contract** (design doc §4.4 / §5.1):

        * Single-SD instance (``PP == 1 and tp_node_count == 1``) —
          this is the dominant production shape today.  No coordination
          needed; return ``True`` immediately.  Zero regression for
          deployments that stay on ``enable_p2p_cpu`` only.

        * Multi-SD instance (``PP > 1`` OR ``tp_node_count > 1``) —
          every peer SD of the same instance must ACK that it also
          holds the prefix.  The in-memory ``AggregateRadixTree``
          ``fully_ready`` bit is what we gate on.  It is populated by
          two paths:

          1. **Self-SD PUT** (local ACK).  :meth:`_notify_master_sd_ready`
             runs on the transfer_callback after D2H + Redis publish
             on the Master's own SD.
          2. **Peer-SD PUT** (remote ACK).  Each peer SD's
             ``TransferManagerOnRemote`` runs the per-SD D2H clone op
             (filtered into its own slice by ``target_node_ids``) and
             ships back a ``CompletedOp(sd_key,
             contributing_node_id, success=True)`` via the
             ``TransferManagerMultiNodeHandle._completion_sink``,
             which routes into :meth:`_on_peer_sd_completed_op` →
             ``mark_sd_ready``.

          Together these two paths flip ``fully_ready`` True for the
          prefix, at which point a subsequent GET clears this gate.

        * ``dist_reuse`` not attached (``has_dist_reuse`` is False) —
          no-op, behave like the legacy path.

        Return True to allow reuse; False to force the caller into a
        cache-miss fallback.

        Why a local ``fully_ready`` check rather than firing a
        per-GET cross-SD query:

        - Design-doc §4.4 favours PUT-driven propagation of the
          aggregate state over GET-driven queries to keep the
          per-request latency close to the existing single-SD path.
          (An earlier draft proposed an on-demand ``coord_query``
          round-trip; that protocol was dropped in Phase D-4 in
          favour of the PUT-driven aggregate radix.)
        - For ``total_sd_count == 1`` the prefix is trivially fully
          ready once we inserted it locally.  No round-trip cost.
        - For ``total_sd_count > 1`` with no peer SD acks yet (e.g.
          single-node dev setup), the gate still enforces the
          contract defensively: if we don't have a positive
          ``fully_ready`` signal, we refuse to reuse.  This is
          consistent with design §5.1 "any miss → fallback to
          prefill".
        """
        if not self.has_dist_reuse:
            return True
        if self._master_coord is None:
            return True

        # Single-SD instance — degenerate case, no coordination needed.
        try:
            total_sd = int(getattr(self._master_coord, "total_sd_count", 1))
        except Exception:
            total_sd = 1
        if total_sd <= 1:
            return True

        # Multi-SD instance — require the aggregate radix to have a
        # fully-ready entry for the prefix we're about to reuse.
        if num_gpu_blocks_to_transfer <= 0:
            return True
        try:
            sequence_meta.gen_hashes()
        except Exception:
            # Defensive: if we can't hash, we can't gate — allow
            # through and rely on data-plane failure closure.
            return True

        # The prefix we're about to reuse starts at ``block_start_idx``
        # and covers ``num_gpu_blocks_to_transfer`` blocks.  Check the
        # aggregate-radix ``fully_ready`` bit for the *last* block in
        # the reuse range — design §5.1 requires *all* blocks in the
        # reused prefix to be fully ready, but in practice the
        # aggregate radix stores prefixes by their terminal block's
        # hash, so we check the last block and rely on the tree's
        # invariant (parent ready ⇒ ancestors ready).
        terminal_block_idx = block_start_idx + num_gpu_blocks_to_transfer - 1
        if terminal_block_idx >= sequence_meta.block_hashes.shape[0]:
            return True
        try:
            prefix_hash = int(sequence_meta.block_hashes[terminal_block_idx].item())
        except Exception:
            return True

        try:
            entry = self._master_coord.match_fully_ready(prefix_hash)
        except Exception:
            return True  # never let a buggy aggregate wedge the GET

        # Fully-ready: let the reuse proceed.
        if entry is not None:
            return True

        # Not fully-ready: reject the reuse.  Caller converts to
        # empty-return, upstream re-runs prefill.
        try:
            from flexkv.common.debug import flexkv_logger
            flexkv_logger.debug(
                f"[DistReuse] sharing-domain gate rejected prefix_hash={prefix_hash} "
                f"(total_sd={total_sd}, fully_ready=no)"
            )
        except Exception:
            pass
        return False

    def _notify_sd_ready_on_put(
        self,
        *,
        sequence_meta,
        inserted_block_ids,
        block_start_idx: int,
        num_blocks_inserted: int,
    ) -> None:
        """PUT-path hook (§4.4 design doc): mark the newly-inserted
        prefix as ready on *this* SD (self-SD ACK) and register a
        pending PUT batch so the graph-dispatch
        ``_completion_sink`` can mark every peer SD ready when their
        per-SD D2H clones complete.  Cross-SD coordination is carried
        on the same ``TransferOpGraph`` the master broadcasts via
        ``_launch_task`` — there is no separate coord protocol
        message (Phase D-4).

        This is idempotent and safe to call from the PUT completion
        callback — in the degenerate single-SD case it still does the
        self-SD mark, which makes ``_sharing_domain_gate_get`` return
        True on the same prefix next time.

        Best-effort — never raises.
        """
        if not self.has_dist_reuse:
            return
        if self._master_coord is None:
            return
        if num_blocks_inserted <= 0:
            return
        try:
            sequence_meta.gen_hashes()
        except Exception:
            return
        terminal_idx = block_start_idx + num_blocks_inserted - 1
        if terminal_idx < 0 or terminal_idx >= sequence_meta.block_hashes.shape[0]:
            return
        try:
            prefix_hash = int(sequence_meta.block_hashes[terminal_idx].item())
        except Exception:
            return

        block_ids_list = []
        try:
            if inserted_block_ids is not None:
                block_ids_list = [int(b) for b in inserted_block_ids]
        except Exception:
            block_ids_list = []

        try:
            self._notify_master_sd_ready(
                prefix_hash=prefix_hash,
                block_ids=block_ids_list,
            )
        except Exception as e:
            try:
                from flexkv.common.debug import flexkv_logger
                flexkv_logger.warning(f"[DistReuse] _notify_sd_ready_on_put failed: {e}")
            except Exception:
                pass

    # Phase D-4: _coord_get_cross_sd / _coord_get_cleanup_on_failure / ingest_coord_ack
    # were deleted (proposal_unify_with_graph_dispatch_2026-05-15.md §附录 A).
    # Cross-SD GET coordination is now expressed as multi-target PEERH2H ops on a
    # single TransferOpGraph broadcast through the existing _launch_task path.

    @property
    def has_dist_reuse(self) -> bool:
        """True when the engine is wired to a live ``MasterCoordinator``."""
        return self._master_coord is not None



    @nvtx.annotate("Match Prefix Accel", color="yellow")
    def match_local_accel(self,
                        sequence_meta: SequenceMeta,
                        temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
                        is_put: bool = False,
                        gpu_matched_blocks: int = 0) \
                            -> Tuple[MatchResultAccel, MatchResultAccel]:
        #from flexkv.common.debug import flexkv_logger
        cpu_matched_result = MatchResultAccel()
        ssd_matched_result = MatchResultAccel()
        if self.cpu_cache_engine:
            if not self.cache_config.enable_p2p_cpu:
                cpu_matched_result = self.cpu_cache_engine.match(sequence_meta)
            else:
                #flexkv_logger.info(f"[MATCH DEBUG] CPU P2P enabled, calling match_all() instead of match_local()")
                if is_put:
                    cpu_matched_result = self.cpu_cache_engine.match_local(sequence_meta)
                else:
                    cpu_matched_result = self.cpu_cache_engine.match_all(sequence_meta, gpu_matched_blocks)
        if temp_cache_strategy.ignore_ssd:
            return cpu_matched_result, ssd_matched_result
        #TODO: we assume that ssd and gds are not enabled at the same time
        if self.ssd_cache_engine:
            if not self.cache_config.enable_p2p_ssd:
                ssd_matched_result = self.ssd_cache_engine.match(sequence_meta)
            else:
                #flexkv_logger.info(f"[MATCH DEBUG] SSD P2P enabled, calling match_all() instead of match_local()")
                if is_put:
                    ssd_matched_result = self.ssd_cache_engine.match_local(sequence_meta)
                else:
                    ssd_matched_result = self.ssd_cache_engine.match_all(sequence_meta, gpu_matched_blocks)

        return cpu_matched_result, ssd_matched_result

    @nvtx.annotate("Match Prefix", color="yellow")
    def match_local(self,
                    sequence_meta: SequenceMeta,
                    temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
                    is_put: bool = False) \
                        -> Tuple[MatchResult, MatchResult]:
        cpu_matched_result = MatchResult()
        ssd_matched_result = MatchResult()
        if self.cpu_cache_engine:
            cpu_matched_result = self.cpu_cache_engine.match(sequence_meta)
        if self.ssd_cache_engine and not temp_cache_strategy.ignore_ssd:
            ssd_matched_result = self.ssd_cache_engine.match(sequence_meta)

        return cpu_matched_result, ssd_matched_result

    @nvtx.annotate("Match All Prefix accel", color="yellow")
    def match_all_accel(self,
                        sequence_meta: SequenceMeta,
                        temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
                        is_get: bool = True) \
                            -> Tuple[MatchResultAccel, MatchResultAccel, MatchResultAccel]:
        cpu_matched_result = MatchResultAccel()
        ssd_matched_result = MatchResultAccel()
        remote_matched_result = MatchResultAccel()
        if self.cpu_cache_engine:
            cpu_matched_result = self.cpu_cache_engine.match(sequence_meta)
        if self.ssd_cache_engine and not temp_cache_strategy.ignore_ssd:
            ssd_matched_result = self.ssd_cache_engine.match(sequence_meta)
        if self.remote_cache_engine and not temp_cache_strategy.ignore_remote:
            if self.enable_kv_sharing:
                if is_get:
                    remote_matched_result = self.remote_cache_engine.match_all(sequence_meta)
                else:
                    remote_matched_result = self.remote_cache_engine.match_local(sequence_meta)
            else:
                remote_matched_result = self.remote_cache_engine.match(sequence_meta)

        return cpu_matched_result, ssd_matched_result, remote_matched_result

    @nvtx.annotate("Match All Prefix", color="yellow")
    def match_all(self,
                  sequence_meta: SequenceMeta,
                  temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY) \
                      -> Tuple[MatchResult, MatchResult, MatchResult]:
        cpu_matched_result = MatchResult()
        ssd_matched_result = MatchResult()
        remote_matched_result = MatchResult()
        if self.cpu_cache_engine:
            cpu_matched_result = self.cpu_cache_engine.match(sequence_meta)
        if self.ssd_cache_engine and not temp_cache_strategy.ignore_ssd:
            ssd_matched_result = self.ssd_cache_engine.match(sequence_meta)
        if self.remote_cache_engine and not temp_cache_strategy.ignore_remote:
            remote_matched_result = self.remote_cache_engine.match(sequence_meta)

        return cpu_matched_result, ssd_matched_result, remote_matched_result

    def _check_input(self,
                      token_ids: np.ndarray,
                      token_mask: np.ndarray,
                      slot_mapping: np.ndarray) -> None:
        assert token_ids.dtype == np.int64
        # assert token_mask.dtype == np.bool_, f"token_mask.dtype={token_mask.dtype}"
        assert slot_mapping.dtype == np.int64
        assert token_ids.ndim == 1
        assert token_mask.ndim == 1
        assert slot_mapping.ndim == 1
        assert token_ids.size == token_mask.size, f"token_ids.size={token_ids.size}, token_mask.size={token_mask.size}"
        assert slot_mapping.size == token_mask.sum(), \
            f"slot_mapping.size={slot_mapping.size}, token_mask.sum()={token_mask.sum()}"

    @staticmethod
    def slot_mapping_to_block_ids(slot_mapping: np.ndarray, tokens_per_block: int) -> np.ndarray:
        block_ids: np.ndarray = slot_mapping[::tokens_per_block] // tokens_per_block
        return block_ids

    def _get_block_range(self,
                         token_mask: np.ndarray) -> Tuple[int, int]:
        mask_idx = np.where(token_mask)[0]
        if len(mask_idx) == 0:
            return 0, 0
        start_idx = mask_idx[0].item() // self.tokens_per_block
        end_idx = mask_idx[-1].item() // self.tokens_per_block
        return start_idx, end_idx + 1
