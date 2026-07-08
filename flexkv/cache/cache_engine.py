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
from typing import List, Tuple, Optional, Dict, Callable, Union
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
from flexkv.cache.swa_cache_engine import SWACacheManager
from flexkv.common.block import SequenceMeta
from flexkv.common.config import CacheConfig, ModelConfig, GLOBAL_CONFIG_FROM_ENV
from flexkv.common.transfer import (
    DeviceType, TransferOpGraph, TransferOp, TransferType
)
from flexkv.common.debug import flexkv_logger, summarize_id_tensor
from flexkv.common.type import MatchResultAccel
from flexkv.integration.dynamo.collector import KVEventCollector
from flexkv.metrics import FlexKVMetricsCollector, init_global_collector, get_global_collector

DEVICE_TYPE: List[str] = ['CPU', 'GPU', 'SSD', 'REMOTE']
_VALID_EVICTION_POLICIES = {'lru', 'lfu', 'fifo', 'mru', 'filo'}


@dataclass
class FullKVGetPlan:
    transfer_graph: TransferOpGraph
    finished_ops_ids: List[int]
    node_to_unlock: Dict[DeviceType, Tuple[object, int]]
    op_node_to_ready: Dict[int, Tuple[DeviceType, object, int]]
    buffer_to_free: Dict[DeviceType, np.ndarray]
    num_gpu_blocks_to_transfer: int
    tier_match_results: Dict[DeviceType, object] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> "FullKVGetPlan":
        return cls(
            transfer_graph=TransferOpGraph.create_empty_graph(),
            finished_ops_ids=[],
            node_to_unlock={},
            op_node_to_ready={},
            buffer_to_free={},
            num_gpu_blocks_to_transfer=0,
            tier_match_results={},
        )


@dataclass
class SWAGetSource:
    hit_blocks: int = 0
    source_tier: Optional[DeviceType] = None
    gpu_slots: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    cpu_slots: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    ssd_slots: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    remote_slots: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))
    lock_node: Optional[object] = None
    staging_slot: int = -1

    @classmethod
    def empty(cls) -> "SWAGetSource":
        return cls()


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

        self.device_type = device_type

        self.index = CRadixTreeIndex(tokens_per_block, num_total_blocks, hit_reward_seconds, eviction_policy)

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
        # never drift (see deployments/swa_design/08_节点挂载SWA架构.md). This
        # engine owns SWA initialization for its tier; init_swa() remains public
        # for tests and explicit embedding.
        self.swa_pool = None
        tier_swa_config = (swa_config.for_cache_tier(device_type)
                           if swa_config is not None else None)
        if tier_swa_config is not None:
            self.init_swa(tier_swa_config)

    def init_swa(self, swa_config: "SWAPoolConfig") -> None:
        """Initialize the SWA host pool for node-mounted SWA on this engine.

        Node-mount: the radix tree nodes hold the SWA state; this pool only
        supplies slot bytes + a free-list. SWA is page-granular, so the physical
        SWA page size must match ``tokens_per_block``.
        """
        from flexkv.swa.swa_host_pool import SWAHostPool
        self.swa_pool = SWAHostPool(swa_config)

    @property
    def swa_enabled(self) -> bool:
        return self.swa_pool is not None

    def swa_alloc_slot(self) -> int:
        """Allocate a free SWA host-pool slot, evicting SWA-LRU when full.

        Returns the slot id, or -1 when the pool is full and every SWA entry is
        locked (cannot make space). Drains any slots freed by the SWA eviction
        back to the pool first.
        """
        if self.swa_pool is None:
            return -1
        slot = self.swa_pool.allocate()
        if slot is not None:
            return slot
        # Pool full: evict one SWA (node-mounted, watermark-driven) then retry.
        self._evict_swa(1)
        slot = self.swa_pool.allocate()
        return slot if slot is not None else -1

    def _drain_swa_slots(self) -> None:
        """Return SWA slots freed by tree structural changes to the host pool."""
        if self.swa_pool is None:
            return
        for slot in self.index.drain_freed_swa_slots():
            if slot is not None and slot >= 0:
                self.swa_pool.free(int(slot))

    def _evict_swa(self, num_swa_evicted: int) -> int:
        """SWA-only eviction on the node-mounted tree, then drain freed slots and
        recycle any full blocks freed by leaf deletions. Returns # SWA freed."""
        if self.swa_pool is None:
            return 0
        evicted_full = torch.zeros(0, dtype=torch.int64)
        num_freed = self.index.evict_swa(evicted_full, num_swa_evicted)
        # Full blocks freed by leaf/tombstone deletions go back to the mempool.
        if evicted_full.numel() > 0:
            self.mempool.recycle_blocks(evicted_full.numpy())
        self._drain_swa_slots()
        return num_freed

    def set_swa(self, node: "CRadixNode", slot: int) -> None:
        """Mount an SWA slot on ``node``'s trailing page (store side)."""
        self.index.set_swa(node, int(slot))

    def _probe_swa_match(self, sequence_meta: SequenceMeta,
                         upper_bound_blocks: int):
        """Run a clamped SWA probe without updating Full-KV cache heat."""
        sequence_meta.gen_hashes()
        num_blocks = min(int(upper_bound_blocks), sequence_meta.num_blocks)
        if num_blocks <= 0:
            return None
        block_hashes = torch.from_numpy(
            sequence_meta.block_hashes[:num_blocks]).to(torch.int64)
        return self.index.match_prefix(block_hashes, num_blocks, False)

    def _resolve_swa_match(self,
                           sequence_meta: SequenceMeta,
                           upper_bound_blocks: int,
                           match_result=None,
                           lock_for_load: bool = False,
                           ) -> Tuple[int, int, Optional["CRadixNode"]]:
        """Resolve the SWA hit from an existing Full-KV match or a clamped probe."""
        if self.swa_pool is None or upper_bound_blocks <= 0:
            return 0, -1, None
        upper_bound_blocks = int(upper_bound_blocks)

        mr = match_result
        swa_node = getattr(mr, "last_swa_node", None) if mr is not None else None
        swa_hit = int(getattr(mr, "swa_hit_blocks", 0) or 0) if mr is not None else 0

        if swa_node is not None and swa_hit > upper_bound_blocks:
            # The reused Full-KV result saw an SWA node past this tier's usable
            # bound. Re-probe the clamped prefix to find the deepest legal SWA.
            mr = None

        if mr is None:
            mr = self._probe_swa_match(sequence_meta, upper_bound_blocks)
            swa_node = getattr(mr, "last_swa_node", None) if mr is not None else None
            swa_hit = int(getattr(mr, "swa_hit_blocks", 0) or 0) if mr is not None else 0

        if swa_node is None or swa_hit <= 0 or swa_hit > upper_bound_blocks:
            return 0, -1, None
        slot = int(swa_node.swa_host_slot)
        if slot < 0:
            return 0, -1, None

        # Read-hit heat: both reused results and clamped probes represent an actual
        # SWA reuse, so refresh the SWA-LRU independently of Full-KV heat.
        self.index.promote_swa(swa_node)
        if lock_for_load:
            swa_node.inc_swa_lock_ref()
        return swa_hit, slot, swa_node

    def match_swa(self,
                  sequence_meta: SequenceMeta,
                  upper_bound_blocks: int,
                  lock_for_load: bool = False,
                  match_result=None,
                  return_node: bool = False,
                  ) -> Union[Tuple[int, int], Tuple[int, int, Optional["CRadixNode"]]]:
        """Match the longest reusable trailing-SWA prefix within an upper bound.

        This is the canonical SWA match entry point. Pass ``match_result`` when a
        Full-KV ``match()`` result is already available; otherwise this performs a
        clamped, match-only probe. When ``lock_for_load`` is true, the matched SWA
        node is pinned until the SWA H2D completion callback releases it, and the
        node handle can be returned as the third tuple item.

        Returns ``(swa_hit_blocks, slot_id)`` by default, or
        ``(swa_hit_blocks, slot_id, node)`` when ``return_node`` or
        ``lock_for_load`` is true. The node is pinned only when
        ``lock_for_load`` is true. Miss is ``0, -1[, None]``.
        """
        include_node = return_node or lock_for_load
        swa_hit, slot, node = self._resolve_swa_match(
            sequence_meta, upper_bound_blocks, match_result=match_result,
            lock_for_load=lock_for_load)
        if include_node:
            return swa_hit, slot, node
        return swa_hit, slot

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
                self._drain_swa_slots()

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
                 metrics_collector = None,
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

        self.device_type = device_type

        self.index = RadixTreeIndex(tokens_per_block=tokens_per_block, hit_reward_seconds=hit_reward_seconds, eviction_policy=eviction_policy)

        self.mempool = Mempool(num_total_blocks=num_total_blocks)

        self.tokens_per_block = tokens_per_block
        self.num_total_blocks = num_total_blocks
        self.evict_ratio = evict_ratio
        self.evict_start_threshold = evict_start_threshold

        self.event_collector = event_collector
        self._metrics_collector = metrics_collector

        # Node-mounted SWA host pool (non-accel Python RadixTreeIndex mirror).
        # See CacheEngineAccel for the semantics.
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

    def swa_alloc_slot(self) -> int:
        if self.swa_pool is None:
            return -1
        slot = self.swa_pool.allocate()
        if slot is not None:
            return slot
        self._evict_swa(1)
        slot = self.swa_pool.allocate()
        return slot if slot is not None else -1

    def _drain_swa_slots(self) -> None:
        if self.swa_pool is None:
            return
        for slot in self.index.drain_freed_swa_slots():
            if slot is not None and slot >= 0:
                self.swa_pool.free(int(slot))

    def _evict_swa(self, num_swa_evicted: int) -> int:
        if self.swa_pool is None:
            return 0
        evicted_full, num_freed = self.index.evict_swa(num_swa_evicted)
        if evicted_full.size > 0:
            self.mempool.recycle_blocks(evicted_full)
        self._drain_swa_slots()
        return num_freed

    def set_swa(self, node: "RadixNode", slot: int) -> None:
        self.index.set_swa(node, int(slot))

    def _probe_swa_match(self, sequence_meta: SequenceMeta,
                         upper_bound_blocks: int):
        """Run a clamped SWA probe without updating Full-KV cache heat."""
        sequence_meta.gen_hashes()
        num_blocks = min(int(upper_bound_blocks), sequence_meta.num_blocks)
        if num_blocks <= 0:
            return None
        clamped = SequenceMeta(
            token_ids=sequence_meta.token_ids[:num_blocks * self.tokens_per_block],
            tokens_per_block=self.tokens_per_block,
            namespace=getattr(sequence_meta, "_namespace", None),
        )
        return self.index.match_prefix(clamped, update_cache_info=False)

    def _resolve_swa_match(self,
                           sequence_meta: SequenceMeta,
                           upper_bound_blocks: int,
                           match_result=None,
                           lock_for_load: bool = False,
                           ) -> Tuple[int, int, Optional["RadixNode"]]:
        """Resolve the SWA hit from an existing Full-KV match or a clamped probe."""
        if self.swa_pool is None or upper_bound_blocks <= 0:
            return 0, -1, None
        upper_bound_blocks = int(upper_bound_blocks)

        mr = match_result
        swa_node = getattr(mr, "last_swa_node", None) if mr is not None else None
        swa_hit = int(getattr(mr, "swa_hit_blocks", 0) or 0) if mr is not None else 0

        if swa_node is not None and swa_hit > upper_bound_blocks:
            mr = None

        if mr is None:
            mr = self._probe_swa_match(sequence_meta, upper_bound_blocks)
            swa_node = getattr(mr, "last_swa_node", None) if mr is not None else None
            swa_hit = int(getattr(mr, "swa_hit_blocks", 0) or 0) if mr is not None else 0

        if swa_node is None or swa_hit <= 0 or swa_hit > upper_bound_blocks:
            return 0, -1, None
        slot = int(swa_node.swa_host_slot)
        if slot < 0:
            return 0, -1, None

        self.index.promote_swa(swa_node)
        if lock_for_load:
            swa_node.inc_swa_lock_ref()
        return swa_hit, slot, swa_node

    def match_swa(self,
                  sequence_meta: SequenceMeta,
                  upper_bound_blocks: int,
                  lock_for_load: bool = False,
                  match_result=None,
                  return_node: bool = False,
                  ) -> Union[Tuple[int, int], Tuple[int, int, Optional["RadixNode"]]]:
        """Canonical node-mounted SWA match on the Python RadixTreeIndex mirror.

        Pass ``match_result`` to reuse an existing Full-KV match; otherwise this
        runs a clamped match-only probe. Returns ``(hit, slot)`` by default, or
        ``(hit, slot, node)`` when ``return_node`` or ``lock_for_load`` is true.
        The node is pinned only when ``lock_for_load`` is true. Miss is
        ``0, -1[, None]``.
        """
        include_node = return_node or lock_for_load
        swa_hit, slot, node = self._resolve_swa_match(
            sequence_meta, upper_bound_blocks, match_result=match_result,
            lock_for_load=lock_for_load)
        if include_node:
            return swa_hit, slot, node
        return swa_hit, slot

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
                self._drain_swa_slots()

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
                                                self._metrics_collector,
                                                swa_config=cache_config.swa)
            else:
                self.cpu_cache_engine = CacheEngine(DeviceType.CPU,
                                                cache_config.num_cpu_blocks,
                                                cache_config.tokens_per_block,
                                                self.evict_ratio,
                                                self.hit_reward_seconds,
                                                self.evict_start_threshold,
                                                self.eviction_policy,
                                                event_collector,
                                                self._metrics_collector,
                                                swa_config=cache_config.swa)
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
                                                self._metrics_collector,
                                                swa_config=cache_config.swa)
            else:
                self.ssd_cache_engine = CacheEngine(DeviceType.SSD,
                                                cache_config.num_ssd_blocks,
                                                cache_config.tokens_per_block,
                                                self.evict_ratio,
                                                self.hit_reward_seconds,
                                                self.evict_start_threshold,
                                                self.eviction_policy,
                                                event_collector,
                                                self._metrics_collector,
                                                swa_config=cache_config.swa)
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
                                                   self._metrics_collector,
                                                   swa_config=cache_config.swa)
            else:
                self.remote_cache_engine = CacheEngine(DeviceType.REMOTE,
                                                   cache_config.num_remote_blocks,
                                                   cache_config.tokens_per_block,
                                                   self.evict_ratio,
                                                   self.hit_reward_seconds,
                                                   self.evict_start_threshold,
                                                   self.eviction_policy,
                                                   None,
                                                   self._metrics_collector,
                                                   swa_config=cache_config.swa)
            self.cache_engines[DeviceType.REMOTE] = self.remote_cache_engine

        # SWA control plane: multi-tier match + peer-op graph construction. Built
        # after the per-tier cache engines (and their swa_pool) exist; reaches
        # them via this GlobalCacheEngine. Gated by cache_config.enable_swa_transfer.
        self.swa_cache = SWACacheManager(self)

        #TODO move this to kvmanager.start()
        self.start()

        # Empty on a full miss.
        self._empty_get_return: Callable[[int], FullKVGetPlan] = \
            lambda request_id: FullKVGetPlan.empty()
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
            full_plan = self._get_impl_local(
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
            full_plan = self._get_impl_global(
                request_id,
                sequence_meta,
                block_start_idx,
                block_end_idx,
                gpu_block_ids,
                temp_cache_strategy,
                dp_client_id,
                swa_aware=swa_aware,
            )

        # SWA peer-op: build the SWA load chain into the SAME graph as the full-KV
        # ops and append its terminal SWA H2D to finished_ops_ids, so the VIRTUAL
        # barrier waits for full AND SWA. SWA ops carry is_swa=True (routed to the
        # SWA worker) and use SWA-pool slot ids: the CPU slot is the node-mounted
        # radix match, the GPU slot is a placeholder bound late at launch
        # (graph.set_swa_gpu_blocks). build_get_chain is a no-op unless SWA
        # transfer is on and a tier has a live SWA hit. num_full_hit = the ready
        # full-KV prefix resident after this get (the SWA hit's upper bound).
        num_full_hit = block_start_idx + full_plan.num_gpu_blocks_to_transfer
        swa_source = self._resolve_swa_get_source(
            request_id, sequence_meta, block_start_idx, block_end_idx,
            num_full_hit, tier_match_results=full_plan.tier_match_results)
        swa_h2d_id = self.swa_cache.build_get_chain(
            full_plan.transfer_graph,
            gpu_slot_ids=swa_source.gpu_slots,
            cpu_slot_ids=swa_source.cpu_slots,
            ssd_slot_ids=swa_source.ssd_slots,
            remote_slot_ids=swa_source.remote_slots,
            dp_client_id=dp_client_id,
        )
        if swa_h2d_id is not None:
            full_plan.finished_ops_ids.append(swa_h2d_id)
        transfer_graph, task_end_op_id = add_virtual_op_for_multiple_finished_ops(
            full_plan.transfer_graph,
            full_plan.finished_ops_ids,
            dp_client_id,
            )

        return_mask = np.zeros_like(token_mask, dtype=np.bool_)
        return_mask[block_start_idx* self.tokens_per_block:
                    (block_start_idx + full_plan.num_gpu_blocks_to_transfer) * self.tokens_per_block] = True

        # if layer_num // layer_granularity != 1:
        #     transfer_graph, finished_ops_ids = convert_read_graph_to_layer_wise_graph(transfer_graph=transfer_graph,
        #                                                                         finished_ops_ids=finished_ops_ids,
        #                                                                         layer_num=layer_num,
        #                                                                         layer_granularity=layer_granularity)

        for device_type in full_plan.node_to_unlock:
            self.cache_engines[device_type].lock_node(full_plan.node_to_unlock[device_type][0])

        callback = partial(self._transfer_callback,
                           node_to_unlock=full_plan.node_to_unlock,
                           buffer_to_free=full_plan.buffer_to_free)

        op_callback_dict = {} # dict, op_id -> callback
        for op_id in full_plan.op_node_to_ready:
            op_callback_dict[op_id] = partial(self._op_callback,
                                              device_type=full_plan.op_node_to_ready[op_id][0],
                                              node_to_ready=full_plan.op_node_to_ready[op_id][1],
                                              ready_length=full_plan.op_node_to_ready[op_id][2])

        # SWA load lock release: _resolve_swa_get_source pinned the matched SWA node
        # (lock_for_load) so it can't be evicted before the SWA H2D reads it. When
        # that H2D completes, release the pin. Keyed on the SWA H2D op so it fires
        # exactly once on completion, alongside the full-KV op callbacks.
        # The SWA H2D completion callback releases the source-tier pin and, for a
        # staged (SSD/REMOTE) source, frees the transient CPU staging slot. Keyed
        # on the SWA H2D op so it fires exactly once, alongside the full-KV ops.
        if swa_h2d_id is not None and (swa_source.lock_node is not None
                                       or swa_source.staging_slot >= 0):
            op_callback_dict[swa_h2d_id] = partial(
                self._swa_release_load_lock, node=swa_source.lock_node,
                staging_slot=swa_source.staging_slot)

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
            dp_client_id: int,
            swa_aware: bool = False) \
                 -> FullKVGetPlan:
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
        # SWA-aware: clamp the Full-KV transfer to usable = min(full, swa) read
        # from this same match. See _get_impl_local for rationale.
        if swa_aware:
            block_mask_end = self._clamp_end_to_swa(
                block_mask_start, block_mask_end,
                {DeviceType.CPU: cpu_matched_result,
                 DeviceType.SSD: ssd_matched_result,
                 DeviceType.REMOTE: remote_matched_result})
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
        # Trailing dict: per-tier Full-KV match results, so SWA slot resolution
        # reuses them instead of re-walking the tree.
        return FullKVGetPlan(
            transfer_graph=transfer_graph,
            finished_ops_ids=finished_ops_ids,
            node_to_unlock=node_to_unlock,
            op_node_to_ready={},
            buffer_to_free=buffer_to_free,
            num_gpu_blocks_to_transfer=len(fragment123_gpu_blocks) if enable_gpu else 0,
            tier_match_results={
                DeviceType.CPU: cpu_matched_result,
                DeviceType.SSD: ssd_matched_result,
                DeviceType.REMOTE: remote_matched_result,
            },
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
                            -> FullKVGetPlan:
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

        # SWA-aware: clamp the Full-KV transfer to usable = min(full, swa). The
        # SWA window past `usable` has no reusable KV, so loading Full past it
        # would feed stale KV to the SWA-layer attention. The SWA hit is read from
        # this same match (no re-walk); block_mask_end is the sole knob — every
        # fragment below slices [block_mask_start:block_mask_end].
        if swa_aware:
            block_mask_end = self._clamp_end_to_swa(
                block_mask_start, block_mask_end,
                {DeviceType.CPU: cpu_matched_result,
                 DeviceType.SSD: ssd_matched_result})

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
        nvtx.end_range(nvtx_range)
        # Trailing dict: per-tier Full-KV match results, so SWA slot resolution
        # reuses them instead of re-walking the tree.
        return FullKVGetPlan(
            transfer_graph=transfer_graph,
            finished_ops_ids=finished_ops_ids,
            node_to_unlock=node_to_unlock,
            op_node_to_ready=op_node_to_ready,
            buffer_to_free=buffer_to_free,
            num_gpu_blocks_to_transfer=len(fragment12_gpu_blocks) if enable_gpu else 0,
            tier_match_results={
                DeviceType.CPU: cpu_matched_result,
                DeviceType.SSD: ssd_matched_result,
            },
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

        # SWA peer-op (data plane): build the SWA store chain into THIS graph
        # alongside the full-KV ops, then append its SWA D2H op to
        # finished_ops_ids (joins the VIRTUAL barrier alongside the full-KV D2H).
        # The SWA H2DISK/H2REMOTE write-through ops depend on the SWA D2H but are
        # fire-and-forget (not reported), exactly like the full-KV path. is_swa=True
        # routes them to the SWA worker; SWA-pool slot ids. build_put_chain returns
        # None (no-op) unless enable_swa_transfer is on. The CPU slot is a freshly
        # allocated SWA-pool slot mounted on the stored tail node (node-mounted
        # set_swa); the GPU slot is a placeholder bound LATE from the request's
        # swa_slot_mapping. The stored tail node is node_to_unlock[CPU] (the new
        # ready prefix's deepest node); its trailing page is the SWA window.
        swa_gpu_slots, swa_cpu_slots, swa_ssd_slots, swa_remote_slots = \
            self._swa_put_slots(request_id, sequence_meta, block_start_idx,
                                block_end_idx, node_to_unlock)
        swa_d2h_id = self.swa_cache.build_put_chain(
            transfer_graph,
            gpu_slot_ids=swa_gpu_slots,
            cpu_slot_ids=swa_cpu_slots,
            ssd_slot_ids=swa_ssd_slots,
            remote_slot_ids=swa_remote_slots,
            dp_client_id=dp_client_id,
        )
        if swa_d2h_id is not None:
            finished_ops_ids.append(swa_d2h_id)
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

        callback = partial(self._transfer_callback,
                           node_to_unlock=node_to_unlock,
                           buffer_to_free=buffer_to_free,
                           is_put=True)

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

        # (SWA store chain is built once in the put() wrapper via
        # self.swa_cache.build_put_chain, peer to this full-KV graph.)

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

    def _swa_hit_from_match_results(self, tier_match_results: Dict) -> int:
        """Best (max) SWA hit across tiers, read from already-computed Full-KV
        match results — no tree walk, no promote, no pin.

        Each tier's match_prefix returned ``swa_hit_blocks`` on the same pass that
        produced its Full-KV hit, already <= that tier's Full-KV hit (SWA subset
        of Full). Returns 0 when SWA is disabled or no tier has a live SWA hit.
        Read-only: the promote/pin happens later in _resolve_swa_get_source.
        """
        if not self.swa_cache.enabled or not tier_match_results:
            return 0
        best = 0
        for mr in tier_match_results.values():
            hit = int(getattr(mr, "swa_hit_blocks", 0) or 0) if mr is not None else 0
            if hit > best:
                best = hit
        return best

    def _clamp_end_to_swa(self, block_mask_start: int, block_mask_end: int,
                          tier_match_results: Dict) -> int:
        """Clamp block_mask_end to usable = min(full_hit, swa_hit).

        All three are absolute block counts from prefix block 0. ``block_mask_end``
        is the Full-KV query-window end (the full hit); ``swa_hit`` comes from the
        same-pass match results. When there is no SWA hit, usable collapses to
        block_mask_start (empty Full-KV window) — loading Full past the reusable
        SWA window would feed stale KV to the SWA-layer attention. Never widens the
        window and never drops below block_mask_start.
        """
        swa_hit = self._swa_hit_from_match_results(tier_match_results)
        usable = min(block_mask_end, swa_hit)
        return max(block_mask_start, usable)

    # --- SWA slot sources for the get()/put() build chains --------------------
    # Resolve the per-tier SWA-pool slot ids for the current request so the SWA
    # build chains can wire ops into the transfer graph. GET reads the matched
    # node's swa_host_slot (node-mounted radix); PUT allocates a fresh slot and
    # mounts it on the stored tail node (set_swa). The GPU-side slot is a
    # size-1 PLACEHOLDER here (window == one page == one slot on DSv4); it is
    # rebound LATE from the request's swa_slot_mapping via
    # TransferOpGraph.set_swa_gpu_blocks() in launch (mirror of the full-KV
    # late-bind). All are no-ops (empty) unless swa_cache.enabled (i.e.
    # enable_swa_transfer AND the CPU tier has an SWA host pool).
    # See deployments/swa_design/08_节点挂载SWA架构.md §10.

    _SWA_GPU_PLACEHOLDER = np.array([0], dtype=np.int64)

    def _empty_swa_put_slots(self):
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty, empty

    def _resolve_swa_get_source(self, request_id, sequence_meta, block_start_idx,
                                block_end_idx, full_hit_blocks,
                                tier_match_results=None) -> SWAGetSource:
        """Resolve the complete SWA GET source plan across tiers.

        Mirrors full-KV multi-tier staging: the trailing SWA window (page
        granular, one slot) is sourced from the highest-priority tier that holds
        it (CPU > SSD > REMOTE), matching full-KV's loader preference.

        * CPU source: the matched CPU SWA slot feeds the SWA H2D directly (no
          staging), and the CPU node is pinned (released on H2D completion).
        * SSD/REMOTE source: there is no CPU-resident window, so a TRANSIENT CPU
          SWA staging slot is allocated as the DISK2H/REMOTE2H destination and
          the SWA H2D source (mirrors full-KV's fragment23 CPU staging blocks).
          The source-tier node is pinned; the staging slot is freed on H2D
          completion (it is unmounted — not a cached entry).

        ``tier_match_results`` (DeviceType -> MatchResult), when provided, is the
        Full-KV match already computed for this get; each tier resolves its SWA
        slot via the canonical ``match_swa(..., match_result=..., lock_for_load)``
        path. Without a match result, ``match_swa`` performs a fresh clamped probe.

        Returns an ``SWAGetSource`` carrying both graph slots and lifecycle
        metadata. ``cpu_slots`` is always the SWA H2D source (matched CPU slot OR
        staging slot); ``ssd_slots``/``remote_slots`` are non-empty only for a
        staged source; ``staging_slot`` is the transient CPU slot to free on H2D
        completion (-1 when CPU-sourced). Empty when SWA is disabled or no tier
        holds the window."""
        if not self.swa_cache.enabled or full_hit_blocks <= 0:
            return SWAGetSource.empty()
        cpu_engine = self.cpu_cache_engine
        if cpu_engine is None or not getattr(cpu_engine, "swa_enabled", False):
            return SWAGetSource.empty()
        gpu = self._SWA_GPU_PLACEHOLDER.copy()
        empty = np.array([], dtype=np.int64)

        def _match_locked(engine, device_type):
            """Resolve (hit, slot, node) for a tier, pinned for load."""
            mr = tier_match_results.get(device_type) if tier_match_results else None
            return engine.match_swa(
                sequence_meta, upper_bound_blocks=full_hit_blocks,
                lock_for_load=True, match_result=mr, return_node=True)

        # 1) CPU tier first (preferred, no staging). The matched CPU SWA node is
        #    pinned; the H2D completion callback releases the pin.
        swa_hit, cpu_slot, lock_node = \
            _match_locked(cpu_engine, DeviceType.CPU)
        if swa_hit > 0 and cpu_slot >= 0:
            cpu = np.array([cpu_slot], dtype=np.int64)
            return SWAGetSource(
                hit_blocks=swa_hit,
                source_tier=DeviceType.CPU,
                gpu_slots=gpu,
                cpu_slots=cpu,
                ssd_slots=empty,
                remote_slots=empty,
                lock_node=lock_node,
                staging_slot=-1,
            )

        # 2) Fall back to SSD then REMOTE: source tier stages into a transient
        #    CPU SWA slot (DISK2H/REMOTE2H -> CPU) that the SWA H2D then reads.
        for device_type in (DeviceType.SSD, DeviceType.REMOTE):
            engine = self.cache_engines.get(device_type)
            if engine is None or not getattr(engine, "swa_enabled", False):
                continue
            tier_hit, tier_slot, tier_node = \
                _match_locked(engine, device_type)
            if tier_hit <= 0 or tier_slot < 0:
                continue
            # Transient CPU staging slot (unmounted; freed on H2D completion).
            staging_slot = cpu_engine.swa_alloc_slot()
            if staging_slot < 0:
                # No CPU room to stage: release the just-taken source pin and
                # skip SWA reuse for this get (full-KV load is unaffected).
                if tier_node is not None and getattr(tier_node, "swa_lock_ref", 0) > 0:
                    tier_node.dec_swa_lock_ref()
                cpu_engine._drain_swa_slots()
                return SWAGetSource.empty()
            cpu = np.array([staging_slot], dtype=np.int64)
            tier = np.array([tier_slot], dtype=np.int64)
            ssd = tier if device_type == DeviceType.SSD else empty
            remote = tier if device_type == DeviceType.REMOTE else empty
            return SWAGetSource(
                hit_blocks=tier_hit,
                source_tier=device_type,
                gpu_slots=gpu,
                cpu_slots=cpu,
                ssd_slots=ssd,
                remote_slots=remote,
                lock_node=tier_node,
                staging_slot=staging_slot,
            )

        return SWAGetSource.empty()

    def _swa_put_slots(self, request_id, sequence_meta, block_start_idx,
                       block_end_idx, node_to_unlock):
        """Resolve SWA-pool slots for a PUT (store), across all stored tiers.

        Mirrors full-KV write-through: the SWA D2H lands the window in the CPU
        SWA pool, and — for every tier full-KV ALSO stored to (CPU/SSD/REMOTE,
        i.e. present in ``node_to_unlock``) — allocate that tier's SWA slot and
        mount it on that tier's store node (set_swa). build_put_chain then emits
        the SWA H2DISK / H2REMOTE write-through ops (CPU SWA slot -> tier slot),
        fire-and-forget, exactly like the full-KV H2DISK/H2REMOTE. Gating on
        ``node_to_unlock`` keeps SWA a subset of Full PER TIER: SWA is written to
        a tier iff full-KV was.

        Returns ``(gpu_placeholder, cpu=[slot], ssd, remote)``. The CPU
        slot is required (the D2H target); ssd/remote are non-empty only when
        that tier both stored full-KV and has an SWA host pool. All empty when
        SWA is disabled, there is no CPU store node, or the CPU pool is full and
        fully locked (full-KV store is unaffected)."""
        empty = np.array([], dtype=np.int64)
        cpu_store_node = (node_to_unlock[DeviceType.CPU][0]
                          if node_to_unlock and DeviceType.CPU in node_to_unlock
                          else None)
        if not self.swa_cache.enabled or cpu_store_node is None:
            return self._empty_swa_put_slots()
        cpu_engine = self.cpu_cache_engine
        if cpu_engine is None or not getattr(cpu_engine, "swa_enabled", False):
            return self._empty_swa_put_slots()
        slot = cpu_engine.swa_alloc_slot()
        if slot < 0:
            return self._empty_swa_put_slots()
        cpu_engine.set_swa(cpu_store_node, slot)
        cpu_engine._drain_swa_slots()
        cpu = np.array([slot], dtype=np.int64)
        gpu = self._SWA_GPU_PLACEHOLDER.copy()

        # Write-through to a lower tier iff full-KV stored there AND that tier
        # has an SWA host pool. One slot per tier (page-granular window). A tier
        # whose SWA pool is full-and-locked simply skips write-through (its slot
        # stays empty) — the CPU SWA store is unaffected.
        def _tier_slot(device_type):
            if device_type not in node_to_unlock:
                return empty
            engine = self.cache_engines.get(device_type)
            if engine is None or not getattr(engine, "swa_enabled", False):
                return empty
            tier_slot = engine.swa_alloc_slot()
            if tier_slot < 0:
                return empty
            engine.set_swa(node_to_unlock[device_type][0], tier_slot)
            engine._drain_swa_slots()
            return np.array([tier_slot], dtype=np.int64)

        ssd = _tier_slot(DeviceType.SSD)
        remote = _tier_slot(DeviceType.REMOTE)
        return gpu, cpu, ssd, remote


    def _swa_release_load_lock(self, node, staging_slot: int = -1) -> None:
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
        except Exception:  # noqa: BLE001 — never let a callback crash the loop
            pass
        try:
            if staging_slot is not None and staging_slot >= 0:
                cpu_engine = self.cpu_cache_engine
                if cpu_engine is not None and cpu_engine.swa_pool is not None:
                    cpu_engine.swa_pool.free(int(staging_slot))
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
