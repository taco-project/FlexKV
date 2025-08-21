import threading
import time
from functools import partial
from queue import Queue
from typing import List, Tuple, Optional, Dict, Callable
from dataclasses import dataclass

import numpy as np
import torch
from flexkv.c_ext import CRadixNode, CRadixTreeIndex, CMatchResult

from flexkv.cache.mempool import Mempool
from flexkv.cache.radixtree import RadixTreeIndex, RadixNode, MatchResult
from flexkv.cache.transfer_pattern import add_virtal_op_for_mutiple_finished_ops
from flexkv.common.block import SequenceMeta
from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.common.exceptions import InvalidConfigError, NotEnoughSpaceError
from flexkv.common.transfer import (
    DeviceType, TransferOpGraph, TransferOp, TransferType
)
from flexkv.common.debug import flexkv_logger
@dataclass
class MatchResultAccel:
    num_ready_matched_blocks: int = 0
    num_matched_blocks: int = 0
    last_ready_node: Optional['CRadixNode'] = None
    last_node: Optional['CRadixNode'] = None
    last_node_matched_length: int = 0
    physical_blocks: torch.Tensor = torch.empty(0, dtype=torch.int64)

    def __post_init__(self) -> None:
        assert self.physical_blocks.ndim == 1
        assert self.physical_blocks.dtype == torch.int64

class CacheEngineAccel:
    def __init__(self,
                 device_type: DeviceType,
                 num_total_blocks: int,
                 tokens_per_block: int,
                 evict_ratio: float):
        if not isinstance(device_type, DeviceType):
            raise InvalidConfigError(f"Unknown device type: {device_type}")
        if num_total_blocks <= 0:
            raise InvalidConfigError(f"Invalid num_total_blocks: {num_total_blocks}")
        if tokens_per_block <= 0 or (tokens_per_block & (tokens_per_block - 1)) != 0:
            raise InvalidConfigError(f"Invalid tokens_per_block: {tokens_per_block}, "
                              f"tokens_per_block must be a power of 2")

        self.device_type = device_type

        self.index = CRadixTreeIndex(tokens_per_block, num_total_blocks)

        self.mempool = Mempool(num_total_blocks=num_total_blocks)

        self.tokens_per_block = tokens_per_block
        self.num_total_blocks = num_total_blocks
        self.evict_ratio = evict_ratio

    def reset(self) -> None:
        self.index.reset()
        self.mempool.reset()

    def match(self, sequence_meta: SequenceMeta) -> MatchResultAccel:
        sequence_meta.gen_hashes()
        match_result = self.index.match_prefix(torch.from_numpy(sequence_meta.block_hashes).to(torch.int64),
                                              sequence_meta.num_blocks, True)
        return MatchResultAccel(match_result.num_ready_matched_blocks, match_result.num_matched_blocks,
                            match_result.last_ready_node, match_result.last_node,
                            match_result.last_node_matched_length,
                            torch.tensor(match_result.physical_blocks, dtype=torch.int64))

    def insert(self,
               sequence_meta: SequenceMeta,
               physical_block_ids: torch.Tensor,
               num_insert_blocks: int = -1,
               is_ready: bool = True,
               match_result: Optional[MatchResultAccel] = None) -> Optional[CRadixNode]:
        sequence_meta.gen_hashes()
        if match_result is None:
          return self.index.insert(physical_block_ids,
                                 torch.from_numpy(sequence_meta.block_hashes).to(torch.int64),
                                 sequence_meta.num_blocks,
                                 num_insert_blocks,
                                 is_ready)
        else:
          return self.index.insert(physical_block_ids,
                                 torch.from_numpy(sequence_meta.block_hashes).to(torch.int64),
                                 sequence_meta.num_blocks,
                                 num_insert_blocks,
                                 is_ready,
                                 match_result.last_node,
                                 match_result.num_matched_blocks,
                                 match_result.last_node_matched_length)

    def lock_node(self, node: CRadixNode) -> None:
        self.index.lock(node)

    def cleanup(self, node: CRadixNode, cleanup_length: int) -> None:
        self.index.unlock(node)
        self.index.set_ready(node, True, cleanup_length)

    def take(self,
             num_required_blocks: int,
             protected_node: Optional[CRadixNode] = None,
             strict: bool = True) -> torch.Tensor:
        if num_required_blocks > self.mempool.num_free_blocks:
            if protected_node is not None:
                self.index.lock(protected_node)
            evict_block_num = max(num_required_blocks - self.mempool.num_free_blocks, int(self.mempool.num_total_blocks * self.evict_ratio))
            target_blocks = torch.zeros(evict_block_num, dtype=torch.int64)
            num_evicted = self.index.evict(target_blocks, evict_block_num)
            if num_evicted != evict_block_num:
                target_blocks.resize_(num_evicted)
            self.mempool.recycle_blocks(target_blocks)

            if protected_node is not None:
                self.index.unlock(protected_node)
        if strict and num_required_blocks > self.mempool.num_free_blocks:
            raise NotEnoughSpaceError("Not enough free blocks to take, ",
                                      required=num_required_blocks,
                                      available=self.mempool.num_free_blocks)
        num_allocated_blocks = min(num_required_blocks, self.mempool.num_free_blocks)
        return self.mempool.allocate_blocks(num_allocated_blocks)

    def recycle(self, physical_blocks: torch.Tensor) -> None:
        self.mempool.recycle_blocks(physical_blocks)

class CacheEngine:
    def __init__(self,
                 device_type: DeviceType,
                 num_total_blocks: int,
                 tokens_per_block: int,
                 evict_ratio: float):
        if not isinstance(device_type, DeviceType):
            raise InvalidConfigError(f"Unknown device type: {device_type}")
        if num_total_blocks <= 0:
            raise InvalidConfigError(f"Invalid num_total_blocks: {num_total_blocks}")
        if tokens_per_block <= 0 or (tokens_per_block & (tokens_per_block - 1)) != 0:
            raise InvalidConfigError(f"Invalid tokens_per_block: {tokens_per_block}, "
                              f"tokens_per_block must be a power of 2")

        self.device_type = device_type

        self.index = RadixTreeIndex(tokens_per_block=tokens_per_block)

        self.mempool = Mempool(num_total_blocks=num_total_blocks)

        self.tokens_per_block = tokens_per_block
        self.num_total_blocks = num_total_blocks
        self.evict_ratio = evict_ratio

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
        return self.index.insert(sequence_meta,
                                 physical_block_ids,
                                 num_insert_blocks=num_insert_blocks,
                                 is_ready=is_ready,
                                 match_result=match_result)

    def lock_node(self, node: RadixNode) -> None:
        self.index.lock(node)

    def cleanup(self, node: RadixNode, cleanup_length: int) -> None:
        self.index.unlock(node)
        self.index.set_ready(node, True, cleanup_length)

    def take(self,
             num_required_blocks: int,
             protected_node: Optional[RadixNode] = None,
             strict: bool = True) -> np.ndarray:
        if num_required_blocks > self.mempool.num_free_blocks:
            if protected_node is not None:
                self.index.lock(protected_node)
            evict_block_num = max(num_required_blocks - self.mempool.num_free_blocks, int(self.mempool.num_total_blocks * self.evict_ratio))
            self.mempool.recycle_blocks(
                self.index.evict(evict_block_num)
            )
            if protected_node is not None:
                self.index.unlock(protected_node)
        if strict and num_required_blocks > self.mempool.num_free_blocks:
            raise NotEnoughSpaceError("Not enough free blocks to take, ",
                                      required=num_required_blocks,
                                      available=self.mempool.num_free_blocks)
        num_allocated_blocks = min(num_required_blocks, self.mempool.num_free_blocks)
        return self.mempool.allocate_blocks(num_allocated_blocks)

    def recycle(self, physical_blocks: np.ndarray) -> None:
        self.mempool.recycle_blocks(physical_blocks)

class GlobalCacheEngine:
    def __init__(self, cache_config: CacheConfig, model_config: ModelConfig):
        self.cache_config = cache_config
        self.model_config = model_config
        self.tokens_per_block = cache_config.tokens_per_block

        self.cpu_cache_engine = None
        self.ssd_cache_engine = None
        self.remote_cache_engine = None

        self.cache_engines = {}

        if cache_config.enable_cpu:
            if cache_config.index_accel:
                self.cpu_cache_engine = CacheEngineAccel(DeviceType.CPU,
                                                cache_config.num_cpu_blocks,
                                                cache_config.tokens_per_block,
                                                cache_config.evict_ratio)
            else:
                self.cpu_cache_engine = CacheEngine(DeviceType.CPU,
                                                cache_config.num_cpu_blocks,
                                                cache_config.tokens_per_block,
                                                cache_config.evict_ratio)
            self.cache_engines[DeviceType.CPU] = self.cpu_cache_engine
        if cache_config.enable_ssd:
            if cache_config.index_accel:
                self.ssd_cache_engine = CacheEngineAccel(DeviceType.SSD,
                                                cache_config.num_ssd_blocks,
                                                cache_config.tokens_per_block,
                                                cache_config.evict_ratio)
            else:
                self.ssd_cache_engine = CacheEngine(DeviceType.SSD,
                                                cache_config.num_ssd_blocks,
                                                cache_config.tokens_per_block,
                                                cache_config.evict_ratio)
            self.cache_engines[DeviceType.SSD] = self.ssd_cache_engine
        if cache_config.enable_remote:
            if cache_config.index_accel:
                self.remote_cache_engine = CacheEngineAccel(DeviceType.REMOTE,
                                                   cache_config.num_remote_blocks,
                                                   cache_config.tokens_per_block,
                                                   cache_config.evict_ratio)
            else:
                self.remote_cache_engine = CacheEngine(DeviceType.REMOTE,
                                                   cache_config.num_remote_blocks,
                                                   cache_config.tokens_per_block,
                                                   cache_config.evict_ratio)
            self.cache_engines[DeviceType.REMOTE] = self.remote_cache_engine

        self._empty_get_return: Callable[[int], Tuple[TransferOpGraph, List[int], Dict, Dict, int]] = \
            lambda request_id: (TransferOpGraph.create_empty_graph(), [], {}, {}, 0)
        self._empty_put_return: Callable[[int], Tuple[TransferOpGraph, List[int], Dict, Dict, int, int]] = \
            lambda request_id: (TransferOpGraph.create_empty_graph(), [], {}, {}, 0, 0)

    def reset(self) -> None:
        if self.cpu_cache_engine:
            self.cpu_cache_engine.reset()
        if self.ssd_cache_engine:
            self.ssd_cache_engine.reset()
        if self.remote_cache_engine:
            self.remote_cache_engine.reset()

    def get(self,
            request_id: int,
            token_ids: np.ndarray,
            token_mask: np.ndarray,
            slot_mapping: np.ndarray,
            layer_num: int = -1,
            layer_granularity: int = -1,
            dp_id: int = 0) -> Tuple[TransferOpGraph, np.ndarray, Callable, int]:
        self._check_input(token_ids, token_mask, slot_mapping)

        if layer_num == -1:
            layer_num = self.model_config.num_layers
        if layer_granularity == -1:
            layer_granularity = layer_num

        if layer_num != layer_granularity:
            flexkv_logger.error(f"Layerwise transfer is not supported yet, "
                                f"layer_num: {layer_num}, layer_granularity: {layer_granularity}")
            raise NotImplementedError(f"Layerwise transfer is not supported yet, "
                                      f"layer_num: {layer_num}, layer_granularity: {layer_granularity}")

        # ignore the last incomplete block
        aligned_length = (token_ids.shape[0] // self.tokens_per_block) * self.tokens_per_block
        aligned_token_ids = token_ids[:aligned_length]
        token_mask[aligned_length:] = False

        block_start_idx, block_end_idx = self._get_block_range(token_mask)
        assert block_end_idx == aligned_length // self.tokens_per_block
        gpu_block_ids = self.slot_mapping_to_block_ids(slot_mapping,
                                                       self.tokens_per_block)[:block_end_idx-block_start_idx]

        sequence_meta = SequenceMeta(token_ids=aligned_token_ids,
                                     tokens_per_block=self.cache_config.tokens_per_block)

        if not self.cache_config.enable_remote:
            transfer_graph, finished_ops_ids, node_to_unlock, buffer_to_free, num_gpu_blocks_to_transfer = \
                self._get_impl_local(
                request_id,
                sequence_meta,
                block_start_idx,
                block_end_idx,
                gpu_block_ids,
                layer_num
            )
        else:
            transfer_graph, finished_ops_ids, node_to_unlock, buffer_to_free, num_gpu_blocks_to_transfer = \
                self._get_impl_global(
                request_id,
                sequence_meta,
                block_start_idx,
                block_end_idx,
                gpu_block_ids,
                layer_num
            )

        transfer_graph, task_end_op_id = add_virtal_op_for_mutiple_finished_ops(
            transfer_graph,
            finished_ops_ids
            )

        return_mask = np.zeros_like(token_mask, dtype=np.bool_)
        return_mask[block_start_idx* self.tokens_per_block:
                    (block_start_idx + num_gpu_blocks_to_transfer) * self.tokens_per_block] = True

        # if layer_num // layer_granularity != 1:
        #     transfer_graph, finished_ops_ids = convert_read_graph_to_layer_wise_graph(transfer_graph=transfer_graph,
        #                                                                         finished_ops_ids=finished_ops_ids,
        #                                                                         layer_num=layer_num,
        #                                                                         layer_granularity=layer_granularity)
        transfer_graph.bind_to_dp_group(dp_id)

        for device_type in node_to_unlock:
            self.cache_engines[device_type].lock_node(node_to_unlock[device_type][0])

        callback = partial(self._transfer_callback,
                           node_to_unlock=node_to_unlock,
                           buffer_to_free=buffer_to_free)

        return transfer_graph, return_mask, callback, task_end_op_id

    def _get_impl_global(self,
            request_id: int,
            sequence_meta: SequenceMeta,
            block_mask_start: int,
            block_mask_end: int,
            gpu_block_ids: np.ndarray,
            layer_num: int) -> Tuple[TransferOpGraph, List[int], Dict, Dict, int]:
        """
        transfer pattern:

        GPU: (gpu cached) | fragment1 | fragment2      | fragment3      | (need compute)
                               ↑          ↑               ↑
        CPU:     ...      | fragment1 | fragment2(new) | fragment3(new) ← (from REMOTE)
                                          ↑               ↓
        SSD:     ...      | fragment1 | fragment2      | fragment3(new)

        """
        assert self.cache_config.enable_cpu and self.cache_config.enable_remote
        assert self.cpu_cache_engine is not None
        assert self.remote_cache_engine is not None

        cpu_matched_result, ssd_matched_result, remote_matched_result = self.match_all(sequence_meta)

        cpu_matched_blocks = cpu_matched_result.physical_blocks[
            :cpu_matched_result.num_ready_matched_blocks][block_mask_start:block_mask_end]
        ssd_matched_blocks = ssd_matched_result.physical_blocks[
            :ssd_matched_result.num_ready_matched_blocks][block_mask_start:block_mask_end]
        remote_matched_blocks = remote_matched_result.physical_blocks[
            :remote_matched_result.num_ready_matched_blocks][block_mask_start:block_mask_end]

        fragment123_num_blocks = max(len(cpu_matched_blocks), len(ssd_matched_blocks), len(remote_matched_blocks))
        #early return if no blocks to transfer
        if fragment123_num_blocks == 0:
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
        op_disk2h = None
        if fragment2_num_blocks > 0:
            op_disk2h = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.DISK2H,
                src_block_ids = fragment2_ssd_blocks,
                dst_block_ids = fragment123_cpu_blocks[fragment1_num_blocks:fragment12_num_blocks],
                layer_id = 0,
                layer_granularity = layer_num
            )
            transfer_graph.add_transfer_op(op_disk2h)

        op_remote2h = None
        if fragment3_num_blocks > 0:
            op_remote2h = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.REMOTE2H,
                src_block_ids = fragment3_remote_blocks,
                dst_block_ids = fragment123_cpu_blocks[-fragment3_num_blocks:],
                layer_id = 0,
                layer_granularity = layer_num
            )
            transfer_graph.add_transfer_op(op_remote2h)

        # prepare ssd blocks to transfer
        write_ssd_blocks_from_remote = False
        if (self.cache_config.enable_ssd and
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
                    layer_id = 0,
                    layer_granularity = layer_num
                )
                transfer_graph.add_transfer_op(op_h2disk)
                transfer_graph.add_dependency(op_h2disk.op_id, op_remote2h.op_id)

                ssd_node_to_unlock = self.ssd_cache_engine.insert(sequence_meta,
                                                                fragment3_ssd_blocks,
                                                                num_insert_blocks=fragment123_num_blocks + \
                                                                    block_mask_start,
                                                                is_ready=False,
                                                                match_result=ssd_matched_result)

        op_h2d = TransferOp(
            graph_id = transfer_graph.graph_id,
            transfer_type = TransferType.H2D,
            src_block_ids = fragment123_cpu_blocks,
            dst_block_ids = fragment123_gpu_blocks,
            layer_id = 0,
            layer_granularity = layer_num
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
        return transfer_graph, finished_ops_ids, node_to_unlock, buffer_to_free, len(fragment123_gpu_blocks)

    def _get_impl_local(self,
                        request_id: int,
                        sequence_meta: SequenceMeta,
                        block_mask_start: int,
                        block_mask_end: int,
                        gpu_block_ids: np.ndarray,
                        layer_num: int) -> Tuple[TransferOpGraph, List[int], Dict, Dict, int]:
        """
        transfer pattern:

        GPU: (gpu cached) | fragment1 | fragment2      | (need compute)
                               ↑          ↑
        CPU:     ...      | fragment1 | fragment2(new) | (uncached)
                                          ↑
        SSD:     ...      | fragment1 | fragment2      | (uncached)

        """
        assert self.cache_config.enable_cpu
        assert self.cpu_cache_engine is not None

        if self.cache_config.index_accel:
            cpu_matched_result, ssd_matched_result = self.match_local_accel(sequence_meta)
        else:
            cpu_matched_result, ssd_matched_result = self.match_local(sequence_meta)

        # tailor the blocks to assure:
        # the blocks are needed by the mask & the blocks are ready
        cpu_matched_blocks = cpu_matched_result.physical_blocks[:cpu_matched_result.num_ready_matched_blocks]
        cpu_matched_blocks = cpu_matched_blocks[block_mask_start:block_mask_end]
        # if ssd disabled, len(ssd_physical_blocks) is 0
        ssd_matched_blocks = ssd_matched_result.physical_blocks[:ssd_matched_result.num_ready_matched_blocks]
        ssd_matched_blocks = ssd_matched_blocks[block_mask_start:block_mask_end]

        fragment12_num_blocks = max(len(cpu_matched_blocks), len(ssd_matched_blocks))
        fragment1_num_blocks = len(cpu_matched_blocks)
        fragment2_num_blocks = max(len(ssd_matched_blocks) - len(cpu_matched_blocks), 0)
        #early return if no blocks to transfer
        if fragment12_num_blocks == 0:
            return self._empty_get_return(request_id)
        assert fragment12_num_blocks <= len(gpu_block_ids)

        transfer_graph = TransferOpGraph()
        finished_ops_ids = []

        fragment12_gpu_blocks = gpu_block_ids[:fragment12_num_blocks]
        fragment2_ssd_blocks = ssd_matched_blocks[-fragment2_num_blocks:]
        fragment1_cpu_blocks = cpu_matched_blocks[:fragment1_num_blocks]

        cpu_node_to_unlock = cpu_matched_result.last_ready_node
        ssd_node_to_unlock = ssd_matched_result.last_ready_node

        # prepare cpu blocks to transfer
        cpu_blocks_to_free = np.array([], dtype=np.int64)
        op_disk2h = None
        fragment2_cpu_blocks = None
        if fragment2_num_blocks > 0:
            fragment2_cpu_blocks = self.cpu_cache_engine.take(
                num_required_blocks=fragment2_num_blocks,
                protected_node=cpu_matched_result.last_node,
                strict=False
            )
            if len(fragment2_cpu_blocks) < fragment2_num_blocks:
                # NOTE: not enough space to allocate, skip the request
                # there might be a better way to handle this
                self.cpu_cache_engine.recycle(fragment2_cpu_blocks)
                return self._empty_get_return(request_id)

            op_disk2h = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.DISK2H,
                src_block_ids = fragment2_ssd_blocks,
                dst_block_ids = fragment2_cpu_blocks,
                layer_id = 0,
                layer_granularity = layer_num
            )
            transfer_graph.add_transfer_op(op_disk2h)
            # we only insert the buffer blocks to cpu cache engine only:
            # 1. the cpu cache engine satisfies prefix cache after insertion
            # 2. the sequence is all ready blocks
            if (cpu_matched_result.num_ready_matched_blocks >= block_mask_start and
                cpu_matched_result.num_ready_matched_blocks == cpu_matched_result.num_matched_blocks):
                cpu_node_to_unlock = self.cpu_cache_engine.insert(sequence_meta,
                                                                  fragment2_cpu_blocks,
                                                                  num_insert_blocks=fragment12_num_blocks + \
                                                                    block_mask_start,
                                                                  is_ready=False,
                                                                  match_result=cpu_matched_result)
            else:
                cpu_blocks_to_free = fragment2_cpu_blocks
        if fragment2_cpu_blocks is not None:
            fragment12_cpu_blocks = np.concatenate([fragment1_cpu_blocks, fragment2_cpu_blocks])
        else:
            fragment12_cpu_blocks = fragment1_cpu_blocks
        op_h2d = TransferOp(
            graph_id = transfer_graph.graph_id,
            transfer_type = TransferType.H2D,
            src_block_ids = fragment12_cpu_blocks,
            dst_block_ids = fragment12_gpu_blocks,
            layer_id = 0,
            layer_granularity = layer_num
        )
        transfer_graph.add_transfer_op(op_h2d)
        if op_disk2h is not None:
            transfer_graph.add_dependency(op_h2d.op_id, op_disk2h.op_id)
        finished_ops_ids.append(op_h2d.op_id)

        node_to_unlock = {}
        if cpu_node_to_unlock is not None:
            node_to_unlock[DeviceType.CPU] = (cpu_node_to_unlock, cpu_node_to_unlock.size())
        if ssd_node_to_unlock is not None:
            node_to_unlock[DeviceType.SSD] = (ssd_node_to_unlock, ssd_node_to_unlock.size())
        buffer_to_free = {DeviceType.CPU: cpu_blocks_to_free}

        return transfer_graph, finished_ops_ids, node_to_unlock, buffer_to_free, len(fragment12_gpu_blocks)

    def put(self,
            request_id: int,
            token_ids: np.ndarray,
            token_mask: np.ndarray,
            slot_mapping: np.ndarray,
            layer_num : int = -1,
            dp_id: int = 0) -> Tuple[TransferOpGraph, np.ndarray, Callable, int]:
        self._check_input(token_ids, token_mask, slot_mapping)

        if layer_num == -1:
            layer_num = self.model_config.num_layers
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
                                     tokens_per_block=self.cache_config.tokens_per_block)

        if not self.cache_config.enable_remote:
            (transfer_graph, finished_ops_ids, node_to_unlock,
             buffer_to_free, num_gpu_blocks_to_transfer, skipped_gpu_blocks) = \
                self._put_impl_local(
                    request_id,
                    sequence_meta,
                    block_start_idx,
                    block_end_idx,
                    gpu_block_ids,
                    layer_num
                )
        else:
            (transfer_graph, finished_ops_ids, node_to_unlock,
             buffer_to_free, num_gpu_blocks_to_transfer, skipped_gpu_blocks) = \
                self._put_impl_global(
                    request_id,
                    sequence_meta,
                    block_start_idx,
                    block_end_idx,
                    gpu_block_ids,
                    layer_num
                )

        transfer_graph, task_end_op_id = add_virtal_op_for_mutiple_finished_ops(
            transfer_graph,
            finished_ops_ids
        )

        return_mask = np.zeros_like(token_mask, dtype=np.bool_)
        return_mask[(block_start_idx + skipped_gpu_blocks)* self.tokens_per_block:
                    (block_start_idx + skipped_gpu_blocks + num_gpu_blocks_to_transfer) * self.tokens_per_block] = True
        transfer_graph.bind_to_dp_group(dp_id)

        for device_type in node_to_unlock:
            self.cache_engines[device_type].lock_node(node_to_unlock[device_type][0])

        callback = partial(self._transfer_callback,
                           node_to_unlock=node_to_unlock,
                           buffer_to_free=buffer_to_free)

        return transfer_graph, return_mask, callback, task_end_op_id

    def _put_impl_global(self,
            request_id: int,
            sequence_meta: SequenceMeta,
            block_mask_start: int,
            block_mask_end: int,
            gpu_block_ids: np.ndarray,
            layer_num : int) -> Tuple[TransferOpGraph, List[int], Dict, Dict, int, int]:
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
        assert self.cache_config.enable_cpu and self.cache_config.enable_remote
        assert self.cpu_cache_engine is not None
        assert self.remote_cache_engine is not None

        if self.cache_config.index_accel:
            cpu_matched_result, ssd_matched_result, remote_matched_result = self.match_all_accel(sequence_meta)
        else:
            cpu_matched_result, ssd_matched_result, remote_matched_result = self.match_all(sequence_meta)
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
        if not self.cache_config.enable_ssd:
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
        if self.cache_config.enable_ssd and fragment2_num_blocks > 0:
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
            layer_id = 0,
            layer_granularity = layer_num
        )
        transfer_graph.add_transfer_op(op_d2h)
        finished_ops_ids.append(op_d2h.op_id)

        if put_to_ssd:
            fragment2_cpu_blocks = fragment12_cpu_blocks[-fragment2_num_blocks:]
            op_h2disk = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.H2DISK,
                src_block_ids = fragment2_cpu_blocks,
                dst_block_ids = fragment2_ssd_blocks,
                layer_id = 0,
                layer_granularity = layer_num
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
                layer_id = 0,
                layer_granularity = layer_num
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
        return transfer_graph, finished_ops_ids, node_to_unlock, {}, len(fragment12_gpu_blocks), skipped_gpu_blocks

    def _put_impl_local(self,
            request_id: int,
            sequence_meta: SequenceMeta,
            block_mask_start: int,
            block_mask_end: int,
            gpu_block_ids: np.ndarray,
            layer_num : int) -> Tuple[TransferOpGraph, List[int], Dict, Dict, int, int]:
        """
        transfer pattern:

        GPU:   (skipped)  | fragment1      | fragment2      | (uncompleted block)
                                ↓                ↓
        CPU: (cpu cached) | fragment1(new) | fragment2(new) |
                                                 ↓
        SSD:          (ssd cached)         | fragment2(new) |

        """
        assert self.cache_config.enable_cpu
        assert self.cpu_cache_engine is not None
        # assert self.ssd_cache_engine is not None

        if  self.cache_config.index_accel:
            cpu_matched_result, ssd_matched_result = self.match_local_accel(sequence_meta)
        else:
            cpu_matched_result, ssd_matched_result = self.match_local(sequence_meta)
        cpu_matched_blocks = cpu_matched_result.physical_blocks[
            :cpu_matched_result.num_matched_blocks][block_mask_start:block_mask_end]
        ssd_matched_blocks = ssd_matched_result.physical_blocks[
            :ssd_matched_result.num_matched_blocks][block_mask_start:block_mask_end]

        num_skipped_blocks = len(cpu_matched_blocks)
        fragment12_num_blocks = len(gpu_block_ids) - num_skipped_blocks
        if fragment12_num_blocks == 0:
            return self._empty_put_return(request_id)
        fragment2_num_blocks = len(gpu_block_ids) - len(ssd_matched_blocks)
        if not self.cache_config.enable_ssd:
            fragment2_num_blocks = 0

        fragment12_gpu_blocks = gpu_block_ids[num_skipped_blocks:]

        fragment12_cpu_blocks = self.cpu_cache_engine.take(
            num_required_blocks=fragment12_num_blocks,
            protected_node = cpu_matched_result.last_node,
            strict=False
        )
        if self.cache_config.enable_ssd:
            fragment2_ssd_blocks = self.ssd_cache_engine.take(
                num_required_blocks=fragment2_num_blocks,
                protected_node = ssd_matched_result.last_node,
                strict=False
            )
        else:
            fragment2_ssd_blocks = np.array([], dtype=np.int64)
        if len(fragment12_cpu_blocks) < fragment12_num_blocks or \
            len(fragment2_ssd_blocks) < fragment2_num_blocks:
            self.cpu_cache_engine.recycle(fragment12_cpu_blocks)
            if self.cache_config.enable_ssd:
                self.ssd_cache_engine.recycle(fragment2_ssd_blocks)
            return self._empty_put_return(request_id)

        transfer_graph = TransferOpGraph()
        finished_ops_ids = []

        op_d2h = TransferOp(
            graph_id = transfer_graph.graph_id,
            transfer_type = TransferType.D2H,
            src_block_ids = fragment12_gpu_blocks,
            dst_block_ids = fragment12_cpu_blocks,
            layer_id = 0,
            layer_granularity = layer_num
        )
        transfer_graph.add_transfer_op(op_d2h)
        finished_ops_ids.append(op_d2h.op_id)

        if fragment2_num_blocks > 0:
            fragment2_cpu_blocks = fragment12_cpu_blocks[-fragment2_num_blocks:]
            op_h2disk = TransferOp(
                graph_id = transfer_graph.graph_id,
                transfer_type = TransferType.H2DISK,
                src_block_ids = fragment2_cpu_blocks,
                dst_block_ids = fragment2_ssd_blocks,
                layer_id = 0,
                layer_granularity = layer_num
            )
            transfer_graph.add_transfer_op(op_h2disk)

            transfer_graph.add_dependency(op_h2disk.op_id, op_d2h.op_id)

        """insert and lock"""
        cpu_node_to_unlock = self.cpu_cache_engine.insert(sequence_meta,
                                                          fragment12_cpu_blocks,
                                                          is_ready=False,
                                                          match_result=cpu_matched_result)
        ssd_node_to_unlock = None
        if len(fragment2_ssd_blocks) > 0:
            ssd_node_to_unlock = self.ssd_cache_engine.insert(sequence_meta,
                                                            fragment2_ssd_blocks,
                                                            is_ready=False,
                                                            match_result=ssd_matched_result)
        node_to_unlock = {}
        if cpu_node_to_unlock is not None:
            node_to_unlock[DeviceType.CPU] = (cpu_node_to_unlock, cpu_node_to_unlock.size())
        if ssd_node_to_unlock is not None:
            node_to_unlock[DeviceType.SSD] = (ssd_node_to_unlock, ssd_node_to_unlock.size())

        skipped_gpu_blocks = len(cpu_matched_blocks)
        return transfer_graph, finished_ops_ids, node_to_unlock, {}, len(fragment12_gpu_blocks), skipped_gpu_blocks

    def _transfer_callback(self,
                           node_to_unlock: Dict[DeviceType, Tuple[RadixNode, int]],
                           buffer_to_free: Optional[Dict[DeviceType, np.ndarray]] = None) -> None:
        if DeviceType.CPU in node_to_unlock:
            assert self.cpu_cache_engine is not None
            self.cpu_cache_engine.cleanup(node_to_unlock[DeviceType.CPU][0], node_to_unlock[DeviceType.CPU][1])
        if DeviceType.SSD in node_to_unlock:
            assert self.ssd_cache_engine is not None
            self.ssd_cache_engine.cleanup(node_to_unlock[DeviceType.SSD][0], node_to_unlock[DeviceType.SSD][1])
        if DeviceType.REMOTE in node_to_unlock:
            assert self.remote_cache_engine is not None
            self.remote_cache_engine.cleanup(node_to_unlock[DeviceType.REMOTE][0], node_to_unlock[DeviceType.REMOTE][1])
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

    def match_local_accel(self, sequence_meta: SequenceMeta) -> Tuple[MatchResultAccel, MatchResultAccel]:
        cpu_matched_result = MatchResultAccel()
        ssd_matched_result = MatchResultAccel()
        if self.cpu_cache_engine:
            cpu_matched_result = self.cpu_cache_engine.match(sequence_meta)
        if self.ssd_cache_engine:
            ssd_matched_result = self.ssd_cache_engine.match(sequence_meta)

        return cpu_matched_result, ssd_matched_result

    def match_local(self, sequence_meta: SequenceMeta) -> Tuple[MatchResult, MatchResult]:
        cpu_matched_result = MatchResult()
        ssd_matched_result = MatchResult()
        if self.cpu_cache_engine:
            cpu_matched_result = self.cpu_cache_engine.match(sequence_meta)
        if self.ssd_cache_engine:
            ssd_matched_result = self.ssd_cache_engine.match(sequence_meta)

        return cpu_matched_result, ssd_matched_result

    def match_all_accel(self,
                        sequence_meta: SequenceMeta) -> Tuple[MatchResultAccel, MatchResultAccel, MatchResultAccel]:
        cpu_matched_result = MatchResultAccel()
        ssd_matched_result = MatchResultAccel()
        remote_matched_result = MatchResultAccel()
        # TODO: avoid redundant match?
        if self.cpu_cache_engine:
            cpu_matched_result = self.cpu_cache_engine.match(sequence_meta)
        if self.ssd_cache_engine:
            ssd_matched_result = self.ssd_cache_engine.match(sequence_meta)
        if self.remote_cache_engine:
            remote_matched_result = self.remote_cache_engine.match(sequence_meta)

        return cpu_matched_result, ssd_matched_result, remote_matched_result

    def match_all(self, sequence_meta: SequenceMeta) -> Tuple[MatchResult, MatchResult, MatchResult]:
        cpu_matched_result = MatchResult()
        ssd_matched_result = MatchResult()
        remote_matched_result = MatchResult()
        # TODO: avoid redundant match?
        if self.cpu_cache_engine:
            cpu_matched_result = self.cpu_cache_engine.match(sequence_meta)
        if self.ssd_cache_engine:
            ssd_matched_result = self.ssd_cache_engine.match(sequence_meta)
        if self.remote_cache_engine:
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
