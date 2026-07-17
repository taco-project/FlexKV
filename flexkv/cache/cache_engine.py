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
from flexkv.cache.swa_cache_engine import SWAOpConstructor
from flexkv.common.block import SequenceMeta
from flexkv.common.config import CacheConfig, ModelConfig, GLOBAL_CONFIG_FROM_ENV
from flexkv.common.transfer import (
    DeviceType,
    TransferOpGraph,
    TransferOp,
    TransferType,
    add_virtual_op_for_multiple_finished_ops,
)
from flexkv.common.debug import flexkv_logger, summarize_id_tensor
from flexkv.common.type import MatchResultAccel
from flexkv.integration.dynamo.collector import KVEventCollector
from flexkv.metrics import FlexKVMetricsCollector, init_global_collector, get_global_collector

DEVICE_TYPE: List[str] = ['CPU', 'GPU', 'SSD', 'REMOTE']
_VALID_EVICTION_POLICIES = {'lru', 'lfu', 'slru', 'fifo', 'mru', 'filo'}


@dataclass
class GetTransferPlan:
    transfer_graph: TransferOpGraph
    finished_ops_ids: List[int]
    node_to_unlock: Dict[DeviceType, Tuple[object, int]]
    op_callback_dict: Dict[int, Callable]
    buffer_to_free: Dict[DeviceType, np.ndarray]
    num_gpu_blocks_to_transfer: int

    @classmethod
    def empty(cls) -> "GetTransferPlan":
        return cls(
            transfer_graph=TransferOpGraph.create_empty_graph(),
            finished_ops_ids=[],
            node_to_unlock={},
            op_callback_dict={},
            buffer_to_free={},
            num_gpu_blocks_to_transfer=0,
        )


@dataclass
class PutTransferPlan:
    transfer_graph: TransferOpGraph
    finished_ops_ids: List[int]
    node_to_unlock: Dict[DeviceType, Tuple[object, int]]
    op_callback_dict: Dict[int, Callable]
    buffer_to_free: Dict[DeviceType, np.ndarray]
    num_gpu_blocks_to_transfer: int
    skipped_gpu_blocks: int

    @classmethod
    def empty(cls) -> "PutTransferPlan":
        return cls(
            transfer_graph=TransferOpGraph.create_empty_graph(),
            finished_ops_ids=[],
            node_to_unlock={},
            op_callback_dict={},
            buffer_to_free={},
            num_gpu_blocks_to_transfer=0,
            skipped_gpu_blocks=0,
        )


@dataclass
class SWAReadSource:
    hit_blocks: int = 0
    host_slot: int = -1
    node: Optional[object] = None
    device_type: Optional[DeviceType] = None
    engine: Optional[object] = None

    @property
    def found(self) -> bool:
        return self.hit_blocks > 0 and self.host_slot >= 0 and self.node is not None


@dataclass(frozen=True)
class SWAReadReservation:
    """Pinned SWA source plus any transient CPU staging slot and graph op."""
    source: SWAReadSource
    staging_slot: int
    h2d_id: int


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
                 metrics_collector = None,
                 protected_threshold = 2,
                 swa_config: Optional["SWAPoolConfig"] = None):
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
        if not isinstance(protected_threshold, int) or protected_threshold < 1:
            raise ValueError(f"Invalid protected_threshold: {protected_threshold}. "
                              f"protected_threshold must be an integer >= 1")

        self.device_type = device_type

        self.index = CRadixTreeIndex(tokens_per_block, num_total_blocks, hit_reward_seconds, eviction_policy,
                                     protected_threshold)

        self.mempool = Mempool(num_total_blocks=num_total_blocks)

        self.tokens_per_block = tokens_per_block
        self.num_total_blocks = num_total_blocks
        self.evict_ratio = evict_ratio
        self.evict_start_threshold = evict_start_threshold

        self.event_collector = event_collector
        self._metrics_collector = metrics_collector

        # SWA (Sliding Window Attention) — NODE-MOUNTED on the Full-KV radix
        # tree (hicache / sglang style), NOT a standalone index. The radix nodes
        # carry the SWA slot / tombstone / lock (see csrc/radix_tree.h and
        # flexkv/cache/radixtree.py); this engine only owns the SWA host-pool
        # (slot bytes + free-list) and the slot alloc/free/drain plumbing. SWA
        # and Full eviction are UNIFIED through the one tree so the two pools
        # never drift. Thisengine owns SWA initialization for its tier; init_swa()
        #  remains public for tests and explicit embedding.
        self.swa_pool = None
        tier_swa_config = (swa_config.for_cache_tier(device_type)
                           if swa_config is not None else None)
        if tier_swa_config is not None:
            self.init_swa(tier_swa_config)

    def init_swa(self, swa_config: "SWAPoolConfig") -> None:
        """Initialize the SWA host pool for node-mounted SWA on this engine."""
        from flexkv.swa.swa_host_pool import SWAHostPool
        self.swa_pool = SWAHostPool(swa_config)

    @property
    def swa_enabled(self) -> bool:
        return self.swa_pool is not None

    def _alloc_swa_slot(self, protected_node=None) -> int:
        """Allocate one SWA slot; evict SWA-LRU once when the pool is full."""
        if self.swa_pool is None:
            return -1
        slot = self.swa_pool.allocate()
        if slot is not None:
            return slot
        # can not allocate SWA slot, evict SWA-LRU once
        if protected_node is not None:
            self.lock_node(protected_node)
        try:
            self._evict_swa_slots(1)
        finally:
            if protected_node is not None:
                self.unlock(protected_node)
        slot = self.swa_pool.allocate()
        return slot if slot is not None else -1

    def _free_swa_slot(self, slot: int) -> None:
        """Return one detached SWA slot to this tier's pool."""
        self.swa_pool.free(int(slot))

    def _drain_unmounted_swa_slots(self) -> None:
        """Return slots detached by radix-tree structural changes to the pool."""
        if self.swa_pool is None:
            return
        for slot in self.index.drain_freed_swa_slots():
            self._free_swa_slot(slot)

    def _pin_swa_node(self, node) -> None:
        self.index.lock(node)
        try:
            node.inc_swa_lock_ref()
        except Exception:
            self.index.unlock(node)
            raise

    def _evict_swa_slots(self, num_swa_evicted: int) -> int:
        """Evict node-mounted SWA slots through the C++ radix tree."""
        if self.swa_pool is None:
            return 0
        evicted_full = torch.zeros(0, dtype=torch.int64)
        num_freed = self.index.evict_swa(evicted_full, num_swa_evicted)
        if evicted_full.numel() > 0:
            self.mempool.recycle_blocks(evicted_full.numpy())
        self._drain_unmounted_swa_slots()
        return num_freed

    def reset(self) -> None:
        self.index.reset()
        self.mempool.reset()
        # The tree reset bulk-deletes all nodes (their SWA slots are not
        # buffered), so re-arm the SWA pool as fully free to avoid a leak.
        if self.swa_pool is not None:
            self.swa_pool.reset()

    def match(self, sequence_meta: SequenceMeta) -> MatchResultAccel:
        sequence_meta.gen_hashes()
        match_result = self.index.match_prefix(torch.from_numpy(sequence_meta.block_hashes).to(torch.int64),
                                              sequence_meta.num_blocks, True)
        # physical blocks (torch.Tensor -> numpy, zero-copy on CPU)
        phys = match_result.physical_blocks.cpu().numpy()
        # optional block_node_ids
        try:
            bnis = getattr(match_result, "block_node_ids", None)
            if isinstance(bnis, torch.Tensor) and bnis.numel() > 0:
                bnids_np = bnis.cpu().numpy()
            else:
                bnids_np = None
        except Exception:
            bnids_np = None
        return MatchResultAccel(
            num_ready_matched_blocks=match_result.num_ready_matched_blocks,
            num_matched_blocks=match_result.num_matched_blocks,
            last_ready_node=match_result.last_ready_node,
            last_node=match_result.last_node,
            last_node_matched_length=match_result.last_node_matched_length,
            physical_blocks=phys,
            block_node_ids=bnids_np,
            matched_pos="remote" if self.device_type == DeviceType.REMOTE else "local",
            # SWA node-mount: carry the SWA hit found on the SAME forward pass so
            # the SWA-aware get can reuse it (no second match_prefix walk).
            last_swa_node=getattr(match_result, "last_swa_node", None),
            swa_hit_blocks=int(getattr(match_result, "swa_hit_blocks", 0) or 0),
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
                target_blocks = torch.zeros(evict_block_num, dtype=torch.int64)
                evicted_block_hashes = torch.zeros(evict_block_num, dtype=torch.int64)
                # evict() resizes both tensors in-place to the actual freed count
                # (which may EXCEED evict_block_num when the I2 tombstone cascade
                # frees ancestors) and returns that count. Trust it, don't assume
                # evict_block_num.
                num_evicted = self.index.evict(target_blocks, evicted_block_hashes, evict_block_num)
                if target_blocks.numel() != num_evicted:
                    target_blocks.resize_(num_evicted)
                    evicted_block_hashes.resize_(num_evicted)
                target_blocks = target_blocks.numpy()
                self.mempool.recycle_blocks(target_blocks)

                # SWA node-mount: full eviction may have connected-freed SWA
                # slots (record_freed_swa_slot in split/evict). Return them to the
                # SWA host pool so the two pools stay in lock-step (I1). No-op when
                # SWA is disabled.
                self._drain_unmounted_swa_slots()

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
        self._drain_unmounted_swa_slots()

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
                 metrics_collector = None,
                 protected_threshold = 2,
                 swa_config: Optional["SWAPoolConfig"] = None):
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
        if not isinstance(protected_threshold, int) or protected_threshold < 1:
            raise ValueError(f"Invalid protected_threshold: {protected_threshold}. "
                              f"protected_threshold must be an integer >= 1")

        self.device_type = device_type

        self.index = RadixTreeIndex(tokens_per_block=tokens_per_block, hit_reward_seconds=hit_reward_seconds, eviction_policy=eviction_policy,
                                       protected_threshold=protected_threshold)

        self.mempool = Mempool(num_total_blocks=num_total_blocks)

        self.tokens_per_block = tokens_per_block
        self.num_total_blocks = num_total_blocks
        self.evict_ratio = evict_ratio
        self.evict_start_threshold = evict_start_threshold

        self.event_collector = event_collector
        self._metrics_collector = metrics_collector

        # Legacy Python mirror. Keep the SWA helpers local to this class; the
        # C++ CacheEngineAccel path is the maintained path.
        self.swa_pool = None
        self.tier_swa_config = (swa_config.for_cache_tier(device_type)
                           if swa_config is not None else None)
        if self.tier_swa_config is not None:
            self.init_swa(self.tier_swa_config)

    def init_swa(self, swa_config: "SWAPoolConfig") -> None:
        """Initialize the SWA host pool for node-mounted SWA on this engine."""
        from flexkv.swa.swa_host_pool import SWAHostPool
        self.swa_pool = SWAHostPool(swa_config)

    @property
    def swa_enabled(self) -> bool:
        return self.tier_swa_config is not None and self.tier_swa_config.enabled \
               and self.swa_pool is not None

    def _alloc_swa_slot(self, protected_node=None) -> int:
        """Allocate one SWA slot; evict SWA-LRU once when the pool is full."""
        if self.swa_pool is None:
            return -1
        slot = self.swa_pool.allocate()
        if slot is not None:
            return slot
        if protected_node is not None:
            self.lock_node(protected_node)
        try:
            self._evict_swa_slots(1)
        finally:
            if protected_node is not None:
                self.unlock(protected_node)
        slot = self.swa_pool.allocate()
        return slot if slot is not None else -1

    def _free_swa_slot(self, slot: int) -> None:
        """Return one detached SWA slot to this tier's pool."""
        self.swa_pool.free(int(slot))

    def _drain_unmounted_swa_slots(self) -> None:
        """Return slots detached by radix-tree structural changes to the pool."""
        if self.swa_pool is None:
            return
        for slot in self.index.drain_freed_swa_slots():
            self._free_swa_slot(slot)

    def _pin_swa_node(self, node) -> None:
        self.index.lock(node)
        try:
            node.inc_swa_lock_ref()
        except Exception:
            self.index.unlock(node)
            raise

    def _evict_swa_slots(self, num_swa_evicted: int) -> int:
        """Evict node-mounted SWA slots through the Python radix tree."""
        if self.swa_pool is None:
            return 0
        evicted_full, num_freed = self.index.evict_swa(num_swa_evicted)
        if evicted_full.size > 0:
            self.mempool.recycle_blocks(evicted_full)
        self._drain_unmounted_swa_slots()
        return num_freed

    def reset(self) -> None:
        self.index.reset()
        self.mempool.reset()
        if self.swa_pool is not None:
            self.swa_pool.reset()

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
                evicted_blocks, evicted_block_hashes = self.index.evict(evict_block_num)
                self.mempool.recycle_blocks(evicted_blocks)

                # SWA node-mount: return connected-freed SWA slots to the pool (I1).
                self._drain_unmounted_swa_slots()

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
        self._drain_unmounted_swa_slots()

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
        self.protected_threshold = GLOBAL_CONFIG_FROM_ENV.slru_protected_threshold

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
                self.cpu_cache_engine = CacheEngineAccel(
                    device_type=DeviceType.CPU,
                    num_total_blocks=cache_config.num_cpu_blocks,
                    tokens_per_block=cache_config.tokens_per_block,
                    evict_ratio=self.evict_ratio,
                    hit_reward_seconds=self.hit_reward_seconds,
                    evict_start_threshold=self.evict_start_threshold,
                    eviction_policy=self.eviction_policy,
                    event_collector=event_collector,
                    metrics_collector=self._metrics_collector,
                    protected_threshold=self.protected_threshold,
                    swa_config=cache_config.swa,
                )
            else:
                self.cpu_cache_engine = CacheEngine(
                    device_type=DeviceType.CPU,
                    num_total_blocks=cache_config.num_cpu_blocks,
                    tokens_per_block=cache_config.tokens_per_block,
                    evict_ratio=self.evict_ratio,
                    hit_reward_seconds=self.hit_reward_seconds,
                    evict_start_threshold=self.evict_start_threshold,
                    eviction_policy=self.eviction_policy,
                    event_collector=event_collector,
                    metrics_collector=self._metrics_collector,
                    protected_threshold=self.protected_threshold,
                    swa_config=cache_config.swa,
                )
            self.cache_engines[DeviceType.CPU] = self.cpu_cache_engine
        if cache_config.enable_ssd:
            if cache_config.enable_p2p_ssd:
                self.ssd_cache_engine = HierarchyLRCacheEngine.from_cache_config(cache_config, self.node_id, DeviceType.SSD, meta=self.redis_meta)
            elif self.index_accel:
                self.ssd_cache_engine = CacheEngineAccel(
                    device_type=DeviceType.SSD,
                    num_total_blocks=cache_config.num_ssd_blocks,
                    tokens_per_block=cache_config.tokens_per_block,
                    evict_ratio=self.evict_ratio,
                    hit_reward_seconds=self.hit_reward_seconds,
                    evict_start_threshold=self.evict_start_threshold,
                    eviction_policy=self.eviction_policy,
                    event_collector=event_collector,
                    metrics_collector=self._metrics_collector,
                    protected_threshold=self.protected_threshold,
                    swa_config=cache_config.swa,
                )
            else:
                self.ssd_cache_engine = CacheEngine(
                    device_type=DeviceType.SSD,
                    num_total_blocks=cache_config.num_ssd_blocks,
                    tokens_per_block=cache_config.tokens_per_block,
                    evict_ratio=self.evict_ratio,
                    hit_reward_seconds=self.hit_reward_seconds,
                    evict_start_threshold=self.evict_start_threshold,
                    eviction_policy=self.eviction_policy,
                    event_collector=event_collector,
                    metrics_collector=self._metrics_collector,
                    protected_threshold=self.protected_threshold,
                    swa_config=cache_config.swa,
                )
            self.cache_engines[DeviceType.SSD] = self.ssd_cache_engine
        if cache_config.enable_remote:
            if cache_config.enable_kv_sharing:
                # Build PCFSCacheEngine from CacheConfig directly (replacing RemotePCFSCacheEngine) TODO
                self.remote_cache_engine = HierarchyLRCacheEngine.from_cache_config(cache_config, self.node_id, DeviceType.REMOTE, meta=self.redis_meta)
            elif self.index_accel:
                self.remote_cache_engine = CacheEngineAccel(
                    device_type=DeviceType.REMOTE,
                    num_total_blocks=cache_config.num_remote_blocks,
                    tokens_per_block=cache_config.tokens_per_block,
                    evict_ratio=self.evict_ratio,
                    hit_reward_seconds=self.hit_reward_seconds,
                    evict_start_threshold=self.evict_start_threshold,
                    eviction_policy=self.eviction_policy,
                    event_collector=None,
                    metrics_collector=self._metrics_collector,
                    protected_threshold=self.protected_threshold,
                    swa_config=cache_config.swa,
                )
            else:
                self.remote_cache_engine = CacheEngine(
                    device_type=DeviceType.REMOTE,
                    num_total_blocks=cache_config.num_remote_blocks,
                    tokens_per_block=cache_config.tokens_per_block,
                    evict_ratio=self.evict_ratio,
                    hit_reward_seconds=self.hit_reward_seconds,
                    evict_start_threshold=self.evict_start_threshold,
                    eviction_policy=self.eviction_policy,
                    event_collector=None,
                    metrics_collector=self._metrics_collector,
                    protected_threshold=self.protected_threshold,
                    swa_config=cache_config.swa,
                )
            self.cache_engines[DeviceType.REMOTE] = self.remote_cache_engine

        # SWA peer-op builder. Per-tier match/slot resolution is fused into the
        # Full-KV get/put implementations; this helper only appends SWA ops.
        self.swa_op_constructor = SWAOpConstructor(self)

        #TODO move this to kvmanager.start()
        self.start()

        self._empty_get_return: Callable[[int], GetTransferPlan] = \
            lambda request_id: GetTransferPlan.empty()
        self._empty_put_return: Callable[[int], PutTransferPlan] = \
            lambda request_id: PutTransferPlan.empty()

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

    def get(self,
            request_id: int,
            token_ids: np.ndarray,
            token_mask: np.ndarray,
            slot_mapping: np.ndarray,
            dp_client_id: int,
            temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
            namespace: Optional[List[str]] = None,
            swa_aware: bool = False) \
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
        # block_end_idx is the block just past the LAST True in token_mask. On the
        # plain path the caller marks every non-resident token up to the aligned
        # end, so this equals aligned_length // tokens_per_block. On the SWA-aware
        # path (swa_aware=True) _get_impl_* clamps the window to usable = min(full,
        # swa) after matching, which can end before the aligned length. So the
        # invariant is <= (can never exceed the aligned length), not ==. Nothing
        # below uses aligned_length; all downstream sizing keys off block_end_idx.
        assert block_end_idx <= aligned_length // self.tokens_per_block
        gpu_block_ids = self.slot_mapping_to_block_ids(slot_mapping,
                                                       self.tokens_per_block)[:block_end_idx-block_start_idx]

        sequence_meta = SequenceMeta(token_ids=aligned_token_ids,
                                     tokens_per_block=self.cache_config.tokens_per_block,
                                     namespace=namespace)

        if not self.cache_config.enable_remote or temp_cache_strategy.ignore_remote:
            # from this entrance, we will also handle the case of peer_cpu and peer_ssd
            plan = self._get_impl_local(
                request_id,
                sequence_meta,
                block_start_idx,
                block_end_idx,
                gpu_block_ids,
                temp_cache_strategy,
                dp_client_id,
                swa_aware=swa_aware,
            )
        else:
            #TODO pcfs will be supported later
            plan = self._get_impl_global(
                request_id,
                sequence_meta,
                block_start_idx,
                block_end_idx,
                gpu_block_ids,
                temp_cache_strategy,
                dp_client_id,
                swa_aware=swa_aware,
            )

        transfer_graph, task_end_op_id = add_virtual_op_for_multiple_finished_ops(
            plan.transfer_graph,
            plan.finished_ops_ids,
            dp_client_id,
            )

        return_mask = np.zeros_like(token_mask, dtype=np.bool_)
        return_mask[block_start_idx* self.tokens_per_block:
                    (block_start_idx + plan.num_gpu_blocks_to_transfer) * self.tokens_per_block] = True

        # if layer_num // layer_granularity != 1:
        #     transfer_graph, finished_ops_ids = convert_read_graph_to_layer_wise_graph(transfer_graph=transfer_graph,
        #                                                                         finished_ops_ids=finished_ops_ids,
        #                                                                         layer_num=layer_num,
        #                                                                         layer_granularity=layer_granularity)

        for device_type in plan.node_to_unlock:
            self.cache_engines[device_type].lock_node(plan.node_to_unlock[device_type][0])

        callback = partial(self._transfer_callback,
                           node_to_unlock=plan.node_to_unlock,
                           buffer_to_free=plan.buffer_to_free)

        op_callback_dict = plan.op_callback_dict

        # Update mempool metrics after GET operation
        if self._metrics_collector is not None:
            self._update_mempool_metrics()

        return transfer_graph, return_mask, callback, op_callback_dict, task_end_op_id

    def _build_op_callback_dict(self, op_node_to_ready: Dict) -> Dict[int, Callable]:
        op_callback_dict = {}
        for op_id, (device_type, node_to_ready, ready_length) in op_node_to_ready.items():
            op_callback_dict[op_id] = partial(self._op_callback,
                                              device_type=device_type,
                                              node_to_ready=node_to_ready,
                                              ready_length=ready_length)
        return op_callback_dict

    @staticmethod
    def _append_op_callback(op_callback_dict: Dict[int, Callable],
                            op_id: int,
                            callback: Callable) -> None:
        """Append ``callback`` without overwriting another completion action."""
        previous = op_callback_dict.get(op_id)
        if previous is None:
            op_callback_dict[op_id] = callback
            return

        def combined_callback() -> None:
            previous()
            callback()

        op_callback_dict[op_id] = combined_callback

    def _publish_swa_put_slot(self,
                              device_type: DeviceType,
                              node,
                              slot: int) -> None:
        """Make a reserved PUT slot readable after its tier transfer completes.

        The slot id is allocated before graph construction so the data plane can
        address it, but it is deliberately not mounted on the radix node until
        this callback.  Full-KV and SWA publication therefore remain independent:
        either transfer may finish first without exposing unfilled SWA bytes.
        """
        assert node is not None
        assert slot >= 0
        engine = self.cache_engines[device_type]
        engine.index.set_swa(node, int(slot))

    def _fail_put_before_insert(
            self,
            request_id: int,
            reason: str,
            cpu_blocks: np.ndarray,
            cpu_swa_slot: int = -1,
            ssd_blocks: Optional[np.ndarray] = None,
            ssd_swa_slot: int = -1,
            remote_blocks: Optional[np.ndarray] = None,
            remote_swa_slot: int = -1) -> PutTransferPlan:
        flexkv_logger.warning(
            "[FlexKV-SWA] PUT request failed before radix insert; "
            f"request_id={request_id}, reason={reason}, "
            f"cpu_blocks={len(cpu_blocks)}, ssd_blocks={0 if ssd_blocks is None else len(ssd_blocks)}, "
            f"remote_blocks={0 if remote_blocks is None else len(remote_blocks)}, "
            f"cpu_swa_slot={cpu_swa_slot}, ssd_swa_slot={ssd_swa_slot}, "
            f"remote_swa_slot={remote_swa_slot}"
        )
        if cpu_swa_slot >= 0:
            self.cpu_cache_engine._free_swa_slot(cpu_swa_slot)
        if ssd_swa_slot >= 0:
            self.ssd_cache_engine._free_swa_slot(ssd_swa_slot)
        if remote_swa_slot >= 0:
            self.remote_cache_engine._free_swa_slot(remote_swa_slot)
        self.cpu_cache_engine.recycle(cpu_blocks)
        if ssd_blocks is not None:
            self.ssd_cache_engine.recycle(ssd_blocks)
        if remote_blocks is not None:
            self.remote_cache_engine.recycle(remote_blocks)
        return self._empty_put_return(request_id)

    def _get_impl_global(self,
            request_id: int,
            sequence_meta: SequenceMeta,
            block_mask_start: int,
            block_mask_end: int,
            gpu_block_ids: np.ndarray,
            temp_cache_strategy: CacheStrategy,
            dp_client_id: int,
            swa_aware: bool = False) \
                 -> GetTransferPlan:
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
        transfer_graph = TransferOpGraph()
        swa_reservation: Optional[SWAReadReservation] = None
        if swa_aware:
            block_mask_end, swa_read_source = self._select_swa_read_source(
                block_mask_start,
                block_mask_end,
                {DeviceType.CPU: cpu_matched_result,
                 DeviceType.SSD: ssd_matched_result,
                 DeviceType.REMOTE: remote_matched_result},
            )
            protected_cpu_node = (
                cpu_matched_result.last_ready_node
                if cpu_matched_result.num_ready_matched_blocks > block_mask_start
                else None
            )
            if enable_gpu:
                swa_reservation = self._reserve_swa_read_source(
                    transfer_graph, swa_read_source, protected_cpu_node, dp_client_id)
            if swa_read_source.found and swa_reservation is None:
                block_mask_end = block_mask_start
            if (enable_gpu and swa_read_source.found and swa_reservation is None
                    and self._metrics_collector is not None):
                self._metrics_collector.record_allocation_failure("global")
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
            self._release_swa_read_reservation(swa_reservation)
            # All cache levels missed - record miss for all requested blocks
            if self._metrics_collector is not None:
                total_query_blocks = block_mask_end - block_mask_start
                if total_query_blocks > 0:
                    self._metrics_collector.record_cache_miss(total_query_blocks)
            return self._empty_get_return(request_id)
        assert fragment123_num_blocks <= len(gpu_block_ids)

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
            try:
                fragment23_cpu_blocks = self.cpu_cache_engine.take(
                    num_required_blocks=num_extra_required_blocks,
                    protected_node=cpu_matched_result.last_node,
                    strict=True
                )
            except RuntimeError:
                self._release_swa_read_reservation(swa_reservation)
                if self._metrics_collector is not None:
                    self._metrics_collector.record_allocation_failure("global")
                return self._empty_get_return(request_id)
            if len(fragment23_cpu_blocks) < num_extra_required_blocks:
                self.cpu_cache_engine.recycle(fragment23_cpu_blocks)
                self._release_swa_read_reservation(swa_reservation)
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
        num_gpu_blocks_to_transfer = len(fragment123_gpu_blocks) if enable_gpu else 0
        op_callback_dict = {}
        if swa_reservation is not None:
            assert num_gpu_blocks_to_transfer > 0
            finished_ops_ids.append(swa_reservation.h2d_id)
            op_callback_dict[swa_reservation.h2d_id] = partial(
                self._swa_release_load_lock,
                node=swa_reservation.source.node,
                staging_slot=swa_reservation.staging_slot,
                engine=swa_reservation.source.engine,
            )

        return GetTransferPlan(
            transfer_graph=transfer_graph,
            finished_ops_ids=finished_ops_ids,
            node_to_unlock=node_to_unlock,
            op_callback_dict=op_callback_dict,
            buffer_to_free=buffer_to_free,
            num_gpu_blocks_to_transfer=num_gpu_blocks_to_transfer,
        )

    def _get_impl_local(self,
                        request_id: int,
                        sequence_meta: SequenceMeta,
                        block_mask_start: int,
                        block_mask_end: int,
                        gpu_block_ids: np.ndarray,
                        temp_cache_strategy: CacheStrategy,
                        dp_client_id: int,
                        swa_aware: bool = False) \
                            -> GetTransferPlan:
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

        transfer_graph = TransferOpGraph()
        swa_reservation: Optional[SWAReadReservation] = None
        if swa_aware:
            block_mask_end, swa_read_source = self._select_swa_read_source(
                block_mask_start,
                block_mask_end,
                {DeviceType.CPU: cpu_matched_result,
                 DeviceType.SSD: ssd_matched_result},
            )
            protected_cpu_node = (
                cpu_matched_result.last_ready_node
                if cpu_matched_result.num_ready_matched_blocks > block_mask_start
                else None
            )
            if enable_gpu:
                swa_reservation = self._reserve_swa_read_source(
                    transfer_graph, swa_read_source, protected_cpu_node, dp_client_id)
            if swa_read_source.found and swa_reservation is None:
                block_mask_end = block_mask_start
            if (enable_gpu and swa_read_source.found and swa_reservation is None
                    and self._metrics_collector is not None):
                self._metrics_collector.record_allocation_failure("local")

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
            self._release_swa_read_reservation(swa_reservation)
            # All cache levels missed - record miss for all requested blocks
            if self._metrics_collector is not None:
                total_query_blocks = block_mask_end - block_mask_start
                if total_query_blocks > 0:
                    self._metrics_collector.record_cache_miss(total_query_blocks)
            nvtx.end_range(nvtx_range)
            return self._empty_get_return(request_id)
        assert fragment12_num_blocks <= len(gpu_block_ids)

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
            self._release_swa_read_reservation(swa_reservation)
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
            finished_ops_ids.append(op_h2d.op_id)

        node_to_unlock = {}
        if cpu_node_to_unlock is not None:
            node_to_unlock[DeviceType.CPU] = (cpu_node_to_unlock, cpu_node_to_unlock.size())
        if ssd_node_to_unlock is not None:
            node_to_unlock[DeviceType.SSD] = (ssd_node_to_unlock, ssd_node_to_unlock.size())
        buffer_to_free = {DeviceType.CPU: cpu_blocks_to_free}
        num_gpu_blocks_to_transfer = len(fragment12_gpu_blocks) if enable_gpu else 0
        op_callback_dict = self._build_op_callback_dict(op_node_to_ready)

        if swa_reservation is not None:
            assert num_gpu_blocks_to_transfer > 0
            finished_ops_ids.append(swa_reservation.h2d_id)
            op_callback_dict[swa_reservation.h2d_id] = partial(
                self._swa_release_load_lock,
                node=swa_reservation.source.node,
                staging_slot=swa_reservation.staging_slot,
                engine=swa_reservation.source.engine,
            )
        nvtx.end_range(nvtx_range)
        return GetTransferPlan(
            transfer_graph=transfer_graph,
            finished_ops_ids=finished_ops_ids,
            node_to_unlock=node_to_unlock,
            op_callback_dict=op_callback_dict,
            buffer_to_free=buffer_to_free,
            num_gpu_blocks_to_transfer=num_gpu_blocks_to_transfer,
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
            plan = self._put_impl_local(
                request_id,
                sequence_meta,
                block_start_idx,
                block_end_idx,
                gpu_block_ids,
                temp_cache_strategy,
                dp_client_id,
            )
        else:
            plan = self._put_impl_global(
                request_id,
                sequence_meta,
                block_start_idx,
                block_end_idx,
                gpu_block_ids,
                temp_cache_strategy,
                dp_client_id,
            )

        transfer_graph, task_end_op_id = add_virtual_op_for_multiple_finished_ops(
            plan.transfer_graph,
            plan.finished_ops_ids,
            dp_client_id,
        )
        return_mask = np.zeros_like(token_mask, dtype=np.bool_)
        return_mask[(block_start_idx + plan.skipped_gpu_blocks)* self.tokens_per_block:
                    (block_start_idx + plan.skipped_gpu_blocks + plan.num_gpu_blocks_to_transfer) * self.tokens_per_block] = True

        for device_type in plan.node_to_unlock:
            self.cache_engines[device_type].lock_node(plan.node_to_unlock[device_type][0])

        callback = partial(self._transfer_callback,
                           node_to_unlock=plan.node_to_unlock,
                           buffer_to_free=plan.buffer_to_free,
                           is_put=True)

        op_callback_dict = plan.op_callback_dict

        # Update mempool metrics after PUT operation
        if self._metrics_collector is not None:
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
                -> PutTransferPlan:
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

        cpu_swa_slot = -1
        ssd_swa_slot = -1
        remote_swa_slot = -1

        if self.swa_op_constructor.enabled:
            cpu_swa_slot = self.cpu_cache_engine._alloc_swa_slot(
                cpu_matched_result.last_node)
            if cpu_swa_slot >= 0 and put_to_ssd:
                ssd_swa_slot = self.ssd_cache_engine._alloc_swa_slot(
                    ssd_matched_result.last_node)
            if (cpu_swa_slot >= 0 and
                    (not put_to_ssd or ssd_swa_slot >= 0) and
                    put_to_remote):
                remote_swa_slot = self.remote_cache_engine._alloc_swa_slot(
                    remote_matched_result.last_node)
            if (cpu_swa_slot < 0 or
                    (put_to_ssd and ssd_swa_slot < 0) or
                    (put_to_remote and remote_swa_slot < 0)):
                return self._fail_put_before_insert(
                    request_id=request_id,
                    reason="swa_slot_alloc_failed",
                    cpu_blocks=fragment12_cpu_blocks,
                    cpu_swa_slot=cpu_swa_slot,
                    ssd_blocks=fragment2_ssd_blocks if put_to_ssd else None,
                    ssd_swa_slot=ssd_swa_slot,
                    remote_blocks=fragment3_remote_blocks if put_to_remote else None,
                    remote_swa_slot=remote_swa_slot,
                )

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
        flexkv_logger.info(
            "[FlexKV-SEGV-DEBUG] cache_engine create D2H op (global_put) "
            f"request_id={request_id}, op_id={op_d2h.op_id}, "
            f"graph_id={transfer_graph.graph_id}, dp_client_id={dp_client_id}, "
            f"fragment12_num_blocks={fragment12_num_blocks}, "
            f"fragment2_num_blocks={fragment2_num_blocks}, "
            f"fragment3_num_blocks={fragment3_num_blocks}, "
            f"{summarize_id_tensor('gpu_src', fragment12_gpu_blocks)}, "
            f"{summarize_id_tensor('cpu_dst', fragment12_cpu_blocks)}"
        )
        transfer_graph.add_transfer_op(op_d2h)
        finished_ops_ids.append(op_d2h.op_id)

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
                fragment3_cpu_blocks = np.concatenate([cpu_matched_blocks[-extra_num_cpu_blocks:],
                                                       fragment12_cpu_blocks])
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

        if cpu_swa_slot >= 0:
            empty = np.array([], dtype=np.int64)
            swa_ops = self.swa_op_constructor.build_put_chain(
                transfer_graph,
                gpu_slot_ids=self._SWA_GPU_PLACEHOLDER.copy(),
                cpu_slot_ids=np.array([cpu_swa_slot], dtype=np.int64),
                ssd_slot_ids=(np.array([ssd_swa_slot], dtype=np.int64)
                              if ssd_swa_slot >= 0 else empty),
                remote_slot_ids=(np.array([remote_swa_slot], dtype=np.int64)
                                 if remote_swa_slot >= 0 else empty),
                dp_client_id=dp_client_id,
                return_op_ids=True,
            )
            assert swa_ops.d2h_id is not None
            if put_to_ssd:
                assert swa_ops.h2disk_id is not None
            if put_to_remote:
                assert swa_ops.h2remote_id is not None
            finished_ops_ids.append(swa_ops.d2h_id)

        cpu_node_to_unlock = self.cpu_cache_engine.insert(
            sequence_meta,
            fragment12_cpu_blocks,
            is_ready=False,
            match_result=cpu_matched_result,
        )
        op_node_to_ready[op_d2h.op_id] = (
            DeviceType.CPU, cpu_node_to_unlock, cpu_node_to_unlock.size())
        ssd_node_to_unlock = None
        if put_to_ssd:
            ssd_node_to_unlock = self.ssd_cache_engine.insert(
                sequence_meta,
                fragment2_ssd_blocks,
                is_ready=False,
                match_result=ssd_matched_result,
            )
            op_node_to_ready[op_h2disk.op_id] = (
                DeviceType.SSD, ssd_node_to_unlock, ssd_node_to_unlock.size())
        remote_node_to_unlock = None
        if put_to_remote:
            remote_node_to_unlock = self.remote_cache_engine.insert(
                sequence_meta,
                fragment3_remote_blocks,
                is_ready=False,
                match_result=remote_matched_result,
            )
            op_node_to_ready[op_h2remote.op_id] = (
                DeviceType.REMOTE,
                remote_node_to_unlock,
                remote_node_to_unlock.size(),
            )
        node_to_unlock = {}
        if cpu_node_to_unlock is not None:
            node_to_unlock[DeviceType.CPU] = (cpu_node_to_unlock, cpu_node_to_unlock.size())
        if ssd_node_to_unlock is not None:
            node_to_unlock[DeviceType.SSD] = (ssd_node_to_unlock, ssd_node_to_unlock.size())
        if remote_node_to_unlock is not None:
            node_to_unlock[DeviceType.REMOTE] = (remote_node_to_unlock, remote_node_to_unlock.size())

        op_callback_dict = self._build_op_callback_dict(op_node_to_ready)
        if cpu_swa_slot >= 0:
            self._append_op_callback(
                op_callback_dict,
                swa_ops.d2h_id,
                partial(self._publish_swa_put_slot,
                        DeviceType.CPU, cpu_node_to_unlock, cpu_swa_slot),
            )
        if ssd_swa_slot >= 0:
            self._append_op_callback(
                op_callback_dict,
                swa_ops.h2disk_id,
                partial(self._publish_swa_put_slot,
                        DeviceType.SSD, ssd_node_to_unlock, ssd_swa_slot),
            )
        if remote_swa_slot >= 0:
            self._append_op_callback(
                op_callback_dict,
                swa_ops.h2remote_id,
                partial(self._publish_swa_put_slot,
                        DeviceType.REMOTE, remote_node_to_unlock, remote_swa_slot),
            )
        skipped_gpu_blocks = len(cpu_matched_blocks)
        return PutTransferPlan(
            transfer_graph=transfer_graph,
            finished_ops_ids=finished_ops_ids,
            node_to_unlock=node_to_unlock,
            op_callback_dict=op_callback_dict,
            buffer_to_free={},
            num_gpu_blocks_to_transfer=len(fragment12_gpu_blocks),
            skipped_gpu_blocks=skipped_gpu_blocks,
        )

    def _put_impl_local(self,
            request_id: int,
            sequence_meta: SequenceMeta,
            block_mask_start: int,
            block_mask_end: int,
            gpu_block_ids: np.ndarray,
            temp_cache_strategy: CacheStrategy,
            dp_client_id: int) \
                -> PutTransferPlan:
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

        cpu_swa_slot = -1
        ssd_swa_slot = -1

        if self.swa_op_constructor.enabled:
            cpu_swa_slot = self.cpu_cache_engine._alloc_swa_slot(
                cpu_matched_result.last_node)
            if cpu_swa_slot >= 0 and fragment2_num_blocks > 0:
                ssd_swa_slot = self.ssd_cache_engine._alloc_swa_slot(
                    ssd_matched_result.last_node)
            if (cpu_swa_slot < 0 or
                    (fragment2_num_blocks > 0 and ssd_swa_slot < 0)):
                return self._fail_put_before_insert(
                    request_id=request_id,
                    reason="swa_slot_alloc_failed",
                    cpu_blocks=fragment12_cpu_blocks,
                    cpu_swa_slot=cpu_swa_slot,
                    ssd_blocks=fragment2_ssd_blocks if enable_ssd else None,
                    ssd_swa_slot=ssd_swa_slot,
                )

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
        flexkv_logger.info(
            "[FlexKV-SEGV-DEBUG] cache_engine create D2H op (local_put) "
            f"request_id={request_id}, op_id={op_d2h.op_id}, "
            f"graph_id={transfer_graph.graph_id}, dp_client_id={dp_client_id}, "
            f"fragment12_num_blocks={fragment12_num_blocks}, "
            f"fragment2_num_blocks={fragment2_num_blocks}, "
            f"{summarize_id_tensor('gpu_src', fragment12_gpu_blocks)}, "
            f"{summarize_id_tensor('cpu_dst', fragment12_cpu_blocks)}"
        )
        transfer_graph.add_transfer_op(op_d2h)
        finished_ops_ids.append(op_d2h.op_id)

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

        if cpu_swa_slot >= 0:
            empty = np.array([], dtype=np.int64)
            swa_ops = self.swa_op_constructor.build_put_chain(
                transfer_graph,
                gpu_slot_ids=self._SWA_GPU_PLACEHOLDER.copy(),
                cpu_slot_ids=np.array([cpu_swa_slot], dtype=np.int64),
                ssd_slot_ids=(np.array([ssd_swa_slot], dtype=np.int64)
                              if ssd_swa_slot >= 0 else empty),
                remote_slot_ids=empty,
                dp_client_id=dp_client_id,
                return_op_ids=True,
            )
            assert swa_ops.d2h_id is not None
            if fragment2_num_blocks > 0:
                assert swa_ops.h2disk_id is not None
            finished_ops_ids.append(swa_ops.d2h_id)

        """insert and lock"""
        cpu_node_to_unlock = self.cpu_cache_engine.insert(
            sequence_meta,
            fragment12_cpu_blocks,
            is_ready=False,
            match_result=cpu_matched_result,
        )
        op_node_to_ready[op_d2h.op_id] = (DeviceType.CPU, cpu_node_to_unlock, cpu_node_to_unlock.size())
        ssd_node_to_unlock = None
        if len(fragment2_ssd_blocks) > 0:
            ssd_node_to_unlock = self.ssd_cache_engine.insert(
                sequence_meta,
                fragment2_ssd_blocks,
                is_ready=False,
                match_result=ssd_matched_result,
            )
            op_node_to_ready[op_h2disk.op_id] = (DeviceType.SSD, ssd_node_to_unlock, ssd_node_to_unlock.size())
        node_to_unlock = {}
        if cpu_node_to_unlock is not None:
            node_to_unlock[DeviceType.CPU] = (cpu_node_to_unlock, cpu_node_to_unlock.size())
        if ssd_node_to_unlock is not None:
            node_to_unlock[DeviceType.SSD] = (ssd_node_to_unlock, ssd_node_to_unlock.size())

        op_callback_dict = self._build_op_callback_dict(op_node_to_ready)
        if cpu_swa_slot >= 0:
            self._append_op_callback(
                op_callback_dict,
                swa_ops.d2h_id,
                partial(self._publish_swa_put_slot,
                        DeviceType.CPU, cpu_node_to_unlock, cpu_swa_slot),
            )
        if ssd_swa_slot >= 0:
            self._append_op_callback(
                op_callback_dict,
                swa_ops.h2disk_id,
                partial(self._publish_swa_put_slot,
                        DeviceType.SSD, ssd_node_to_unlock, ssd_swa_slot),
            )
        skipped_gpu_blocks = len(cpu_matched_blocks)
        return PutTransferPlan(
            transfer_graph=transfer_graph,
            finished_ops_ids=finished_ops_ids,
            node_to_unlock=node_to_unlock,
            op_callback_dict=op_callback_dict,
            buffer_to_free={},
            num_gpu_blocks_to_transfer=len(fragment12_gpu_blocks),
            skipped_gpu_blocks=skipped_gpu_blocks,
        )

    def _transfer_callback(self,
                           node_to_unlock: Dict[DeviceType, Tuple[RadixNode, int]],
                           buffer_to_free: Optional[Dict[DeviceType, np.ndarray]] = None,
                           is_put: bool = False) -> None:
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

    @nvtx.annotate("Match Prefix Accel", color="yellow")
    def match_local_accel(self,
                        sequence_meta: SequenceMeta,
                        temp_cache_strategy: CacheStrategy = DEFAULT_CACHE_STRATEGY,
                        is_put: bool = False,
                        gpu_matched_blocks: int = 0) \
                            -> Tuple[MatchResultAccel, MatchResultAccel]:
        #from flexkv.common.debug import flexkv_logger, summarize_id_tensor
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

    def _select_swa_read_source(
        self,
        block_mask_start: int,
        block_mask_end: int,
        tier_match_results: Dict[DeviceType, object],
    ) -> Tuple[int, SWAReadSource]:
        """Return the largest usable SWA-aware Full-KV end and its exact SWA source."""
        if not self.swa_op_constructor.enabled or not tier_match_results:
            return block_mask_start, SWAReadSource()

        candidates: List[Tuple[int, DeviceType, object]] = []
        for device_type, match_result in tier_match_results.items():
            if match_result is None:
                continue

            swa_hit = int(match_result.swa_hit_blocks)
            if swa_hit <= block_mask_start:
                continue

            if swa_hit > block_mask_end:
                # The radix match covers the complete token sequence, while the
                # request mask may stop earlier. A snapshot for a deeper trailing
                # window cannot serve this request window; try another tier.
                continue
            assert match_result.last_swa_node is not None
            candidates.append((swa_hit, device_type, match_result))

        for usable_end, device_type, match_result in sorted(
            candidates,
            key=lambda item: item[0],
            reverse=True,
        ):
            engine = self.cache_engines.get(device_type)
            if engine is None or not getattr(engine, "swa_enabled", False):
                continue

            source_node = match_result.last_swa_node
            source_slot = int(source_node.swa_host_slot)
            assert source_slot >= 0

            return usable_end, SWAReadSource(
                hit_blocks=usable_end,
                host_slot=source_slot,
                node=source_node,
                device_type=device_type,
                engine=engine,
            )

        return block_mask_start, SWAReadSource()

    def _reserve_swa_read_source(
        self,
        graph: TransferOpGraph,
        source: SWAReadSource,
        protected_cpu_node,
        dp_client_id: int,
    ) -> Optional[SWAReadReservation]:
        """Pin a source and build its SWA load chain before committing a Full hit.

        Non-CPU sources need a transient CPU SWA staging slot. Allocation may
        evict through the CPU radix, so protect the CPU Full-KV node referenced by
        this GET. Returning ``None`` means the caller must report no cache hit;
        Full-only restore is invalid for an SWA-aware GET.
        """
        assert self.cpu_cache_engine is not None
        if not source.found:
            return None

        source.engine._pin_swa_node(source.node)
        staging_slot = -1
        cpu_swa_slots = np.array([source.host_slot], dtype=np.int64)
        ssd_swa_slots = np.array([], dtype=np.int64)
        remote_swa_slots = np.array([], dtype=np.int64)

        if source.device_type != DeviceType.CPU:
            staging_slot = self.cpu_cache_engine._alloc_swa_slot(
                protected_node=protected_cpu_node)
            if staging_slot < 0:
                self._swa_release_load_lock(
                    node=source.node, engine=source.engine)
                flexkv_logger.warning(
                    "[FlexKV-SWA] GET staging allocation failed; "
                    f"source={source.device_type}, hit_blocks={source.hit_blocks}"
                )
                return None
            cpu_swa_slots = np.array([staging_slot], dtype=np.int64)
            source_slots = np.array([source.host_slot], dtype=np.int64)
            if source.device_type == DeviceType.SSD:
                ssd_swa_slots = source_slots
            else:
                remote_swa_slots = source_slots

        h2d_id = self.swa_op_constructor.build_get_chain(
            graph,
            gpu_slot_ids=self._SWA_GPU_PLACEHOLDER.copy(),
            cpu_slot_ids=cpu_swa_slots,
            ssd_slot_ids=ssd_swa_slots,
            remote_slot_ids=remote_swa_slots,
            dp_client_id=dp_client_id,
        )
        if h2d_id is None:
            self._swa_release_load_lock(
                node=source.node,
                staging_slot=staging_slot,
                engine=source.engine,
            )
            return None

        return SWAReadReservation(
            source=source,
            staging_slot=staging_slot,
            h2d_id=h2d_id,
        )

    def _release_swa_read_reservation(
        self, reservation: Optional[SWAReadReservation]) -> None:
        if reservation is None:
            return
        self._swa_release_load_lock(
            node=reservation.source.node,
            staging_slot=reservation.staging_slot,
            engine=reservation.source.engine,
        )

    # The GPU-side SWA slot is a size-1 placeholder here (window == one page ==
    # one slot on DSv4). It is rebound late from the request's swa_slot_mapping
    # via TransferOpGraph.set_swa_gpu_blocks() in launch, mirroring the Full-KV
    # GPU late-bind.

    _SWA_GPU_PLACEHOLDER = np.array([0], dtype=np.int64)

    def _swa_release_load_lock(self, node, staging_slot: int = -1, engine=None) -> None:
        """SWA H2D completion callback: release the source pin and free any
        transient CPU staging slot.

        For a CPU-sourced load, ``node`` is the matched CPU SWA node and its pin
        is dropped with the plain dec (dec_swa_lock_ref, NOT dec_swa_lock_only):
        the loaded window stays cached for future reuse. For a staged
        (SSD/REMOTE) source, ``node`` is the source-tier node (same pin release)
        and ``staging_slot`` is the transient CPU SWA slot used as the DISK2H/
        REMOTE2H destination — it is unmounted (not a cached entry), so free it
        back to the CPU SWA pool. No-op on parts that are absent."""
        try:
            if node is not None and getattr(node, "swa_lock_ref", 0) > 0:
                node.dec_swa_lock_ref()
                if engine is not None:
                    engine.index.unlock(node)
                elif hasattr(node, "unlock"):
                    node.unlock()
                else:
                    node.lock_cnt -= 1
        except Exception:  # noqa: BLE001 — never let a callback crash the loop
            pass
        try:
            if staging_slot is not None and staging_slot >= 0:
                cpu_engine = self.cpu_cache_engine
                if cpu_engine is not None:
                    cpu_engine._free_swa_slot(int(staging_slot))
        except Exception:  # noqa: BLE001
            pass

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

    def swa_slot_mapping_to_slot_ids(self, swa_slot_mapping: np.ndarray) -> np.ndarray:
        """Convert an SWA slot_mapping into page-granular SWA pool slot ids."""
        window = self.tokens_per_block
        sm = np.asarray(swa_slot_mapping, dtype=np.int64)
        return sm[::window] // window

    def _get_block_range(self,
                         token_mask: np.ndarray) -> Tuple[int, int]:
        mask_idx = np.where(token_mask)[0]
        if len(mask_idx) == 0:
            return 0, 0
        start_idx = mask_idx[0].item() // self.tokens_per_block
        end_idx = mask_idx[-1].item() // self.tokens_per_block
        return start_idx, end_idx + 1
