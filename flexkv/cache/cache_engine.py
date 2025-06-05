from typing import List, Tuple, Optional, Dict, Callable
from queue import Queue
import torch
import time
import threading
from functools import partial

from flexkv.common.transfer import DeviceType, TransferOpGraph
from flexkv.cache.radixtree import RadixTreeIndex, RadixNode, MatchResult
from flexkv.cache.mempool import Mempool
from flexkv.common.block import SequenceMeta
from flexkv.common.config import CacheConfig, ModelConfig
from flexkv.cache.transfer_pattern import (
    create_read_graph_cpu_storage, create_read_graph_cpu_ssd_remote, convert_read_graph_to_layer_wise_graph,
    create_write_graph_cpu_storage, create_write_graph_cpu_ssd_remote
)
from flexkv.common.request import cacheEngineRequestType, cacheEngineRequest


class CacheEngine:
    def __init__(self,
                 device_type: DeviceType,
                 num_total_blocks: int,
                 tokens_per_block: int):
        self.device_type = device_type

        self.index = RadixTreeIndex(tokens_per_block=tokens_per_block)

        self.mempool = Mempool(num_total_blocks=num_total_blocks)

    def reset(self):
        self.index.reset()
        self.mempool.reset()

    def match(self, sequence_meta: SequenceMeta) -> MatchResult:
        match_result = self.index.match_prefix(sequence_meta,
                                              update_cache_info=True)
        return match_result

    def insert(self,
               sequence_meta: SequenceMeta,
               physical_block_ids: torch.Tensor,
               num_insert_blocks: int = -1,
               is_ready: bool = True,
               match_result: Optional[MatchResult] = None) -> RadixNode:
        return self.index.insert(sequence_meta,
                                 physical_block_ids,
                                 num_insert_blocks=num_insert_blocks,
                                 is_ready=is_ready,
                                 match_result=match_result)

    def lock_node(self, node: RadixNode) -> None:
        self.index.lock(node)

    def cleanup(self, node: RadixNode) -> None:
        self.index.unlock(node)
        self.index.set_ready(node, True)

    def take(self,
             num_required_blocks: int,
             protected_node: Optional[RadixNode] = None,
             strict: bool = True) -> torch.Tensor:
        if num_required_blocks > self.mempool.num_free_blocks:
            if protected_node is not None:
                self.index.lock(protected_node)
            self.mempool.recycle_blocks(
                self.index.evict(num_required_blocks - self.mempool.num_free_blocks)
            )
            if protected_node is not None:
                self.index.unlock(protected_node)
        if strict and num_required_blocks > self.mempool.num_free_blocks:
            raise ValueError(f"Not enough free blocks to take, "
                             f"required: {num_required_blocks}, "
                             f"free: {self.mempool.num_free_blocks}")
        return self.mempool.allocate_blocks(num_required_blocks)

    def recycle(self, physical_blocks: torch.Tensor) -> None:
        self.mempool.recycle_blocks(physical_blocks)

class GlobalCacheEngine:
    def __init__(self, cache_config: CacheConfig, model_config: ModelConfig):
        self.cache_config = cache_config
        self.model_config = model_config
        self.tokens_per_block = cache_config.tokens_per_block

        self.cpu_cache_engine = None
        self.ssd_cache_engine = None
        self.remote_cache_engine = None

        if cache_config.enable_cpu:
            self.cpu_cache_engine = CacheEngine(DeviceType.CPU,
                                                cache_config.num_cpu_blocks,
                                                cache_config.tokens_per_block)
        if cache_config.enable_ssd:
            self.ssd_cache_engine = CacheEngine(DeviceType.SSD,
                                                cache_config.num_ssd_blocks,
                                                cache_config.tokens_per_block)
        if cache_config.enable_remote:
            #NOTE here we use ssd to replace the remote file system
            # such as CFS for single node test
            self.remote_cache_engine = CacheEngine(DeviceType.SSD,
                                                   cache_config.num_remote_blocks,
                                                   cache_config.tokens_per_block)

    def reset(self):
        if self.cpu_cache_engine:
            self.cpu_cache_engine.reset()
        if self.ssd_cache_engine:
            self.ssd_cache_engine.reset()
        if self.remote_cache_engine:
            self.remote_cache_engine.reset()

    def get(self,
            token_ids: torch.Tensor,
            token_mask: torch.Tensor,
            slot_mapping: torch.Tensor,
            layer_num: int = -1,
            layer_granularity: int = -1) -> Tuple[TransferOpGraph, torch.Tensor, Callable, List[int]]:
        if not self.cache_config.enable_remote and not self.cache_config.enable_ssd:
            return self._get_only_cpu(token_ids, token_mask, slot_mapping, layer_num, layer_granularity)

        self._check_input(token_ids, token_mask, slot_mapping)

        if layer_num == -1:
            layer_num = self.model_config.num_layers
        if layer_granularity == -1:
            layer_granularity = layer_num

        # ignore the last incomplete block
        aligned_length = (token_ids.shape[0] // self.tokens_per_block) * self.tokens_per_block
        aligned_token_ids = token_ids[:aligned_length]
        # TODO: support get the last incomplete block
        token_mask[aligned_length:] = False
        start_idx, end_idx = self._get_block_range(token_mask)

        assert end_idx == aligned_length // self.tokens_per_block

        #NOTE: the slot_mapping starts from token that need to be fetched
        gpu_block_mapping = self._slot_to_block_mapping(slot_mapping)[:end_idx-start_idx]

        sequence_meta = SequenceMeta(token_ids=aligned_token_ids,
                                     tokens_per_block=self.cache_config.tokens_per_block)

        cpu_matched_result, ssd_matched_result = self.match_local(sequence_meta)
        remote_enabled = self.need_remote_cache(cpu_matched_result, ssd_matched_result, end_idx - start_idx)
        if remote_enabled:
            #TODO: perhaps we need to lock the last node of remote cache engine
            # at the same time we match it in real remote cache engine
            remote_matched_result = self.remote_cache_engine.match(sequence_meta)

        # tailor the blocks to assure:
        # the blocks are needed by the mask & the blocks are ready
        cpu_physical_blocks = cpu_matched_result.physical_blocks[:cpu_matched_result.num_ready_matched_blocks]
        cpu_physical_blocks = cpu_physical_blocks[start_idx:end_idx]
        ssd_physical_blocks = ssd_matched_result.physical_blocks[:ssd_matched_result.num_ready_matched_blocks]
        ssd_physical_blocks = ssd_physical_blocks[start_idx:end_idx]
        if remote_enabled:
            remote_physical_blocks = remote_matched_result.physical_blocks[
                :remote_matched_result.num_ready_matched_blocks
            ]
            remote_physical_blocks = remote_physical_blocks[start_idx:end_idx]
            # remote cache has less results than local cache
            if len(remote_physical_blocks) <= len(ssd_physical_blocks):
                remote_enabled = False
                remote_physical_blocks = []
        else:
            remote_physical_blocks = []

        num_transfer_blocks = max(len(cpu_physical_blocks), len(ssd_physical_blocks), len(remote_physical_blocks))
        assert num_transfer_blocks <= len(gpu_block_mapping)


        gpu_blocks_to_transfer = gpu_block_mapping[:num_transfer_blocks]
        cpu_blocks_to_transfer = cpu_physical_blocks
        ssd_blocks_to_transfer = ssd_physical_blocks[len(cpu_physical_blocks):]
        remote_blocks_to_transfer = remote_physical_blocks[len(ssd_physical_blocks):]

        cpu_node_to_unlock = cpu_matched_result.last_ready_node
        ssd_node_to_unlock = ssd_matched_result.last_ready_node
        if remote_enabled:
            remote_node_to_unlock = remote_matched_result.last_ready_node
            self.remote_cache_engine.lock_node(remote_node_to_unlock)
        else:
            remote_node_to_unlock = None

        # prepare cpu blocks to transfer
        cpu_blocks_to_free = []
        if len(cpu_physical_blocks) < num_transfer_blocks:
            extra_cpu_blocks = self.cpu_cache_engine.take(
                num_required_blocks=num_transfer_blocks - len(cpu_physical_blocks),
                protected_node=cpu_matched_result.last_node,
                strict=True
            )
            cpu_blocks_to_transfer = torch.cat([cpu_blocks_to_transfer, extra_cpu_blocks])
            # we only insert the buffer blocks to cpu cache engine only:
            # 1. the cpu cache engine satisfies prefix cache after insertion
            # 2. the sequence is all ready blocks
            if (cpu_matched_result.num_ready_matched_blocks >= start_idx and
                cpu_matched_result.num_ready_matched_blocks == cpu_matched_result.num_matched_blocks):
                cpu_node_to_unlock = self.cpu_cache_engine.insert(sequence_meta,
                                                                  extra_cpu_blocks,
                                                                  num_insert_blocks=num_transfer_blocks,
                                                                  is_ready=False,
                                                                  match_result=cpu_matched_result)
            else:
                cpu_blocks_to_free = extra_cpu_blocks
        self.cpu_cache_engine.lock_node(cpu_node_to_unlock)

        # prepare ssd blocks to transfer
        write_ssd_blocks_from_remote = True # this can be a parameter
        if (remote_enabled and
            len(ssd_physical_blocks) < num_transfer_blocks and
            ssd_matched_result.num_ready_matched_blocks >= start_idx and
            ssd_matched_result.num_ready_matched_blocks == ssd_matched_result.num_matched_blocks):
            # only when the above all are satisfied, we load data back from cpu to ssd
            write_ssd_blocks_from_remote = True
            extra_ssd_blocks = self.ssd_cache_engine.take(
                num_required_blocks=num_transfer_blocks - len(ssd_physical_blocks),
                protected_node=ssd_matched_result.last_node,
                strict=True
            )
            ssd_blocks_to_transfer = torch.cat([ssd_blocks_to_transfer, extra_ssd_blocks])
            ssd_node_to_unlock = self.ssd_cache_engine.insert(sequence_meta,
                                                            extra_ssd_blocks,
                                                            num_insert_blocks=num_transfer_blocks,
                                                            is_ready=False,
                                                            match_result=ssd_matched_result)
        self.ssd_cache_engine.lock_node(ssd_node_to_unlock)
        if remote_enabled and len(remote_blocks_to_transfer) > 0:
            transfer_graph, finished_ops_ids = create_read_graph_cpu_ssd_remote(gpu_blocks=gpu_blocks_to_transfer,
                                                                                cpu_blocks=cpu_blocks_to_transfer,
                                                                                ssd_blocks=ssd_blocks_to_transfer,
                                                                                remote_blocks=remote_blocks_to_transfer,
                                                                                gpu_device_id=0,
                                                                                layer_num=layer_num,
                                                                                write_back_to_ssd=write_ssd_blocks_from_remote)
        else:
            transfer_graph, finished_ops_ids = create_read_graph_cpu_storage(gpu_blocks=gpu_blocks_to_transfer,
                                                                            cpu_blocks=cpu_blocks_to_transfer,
                                                                            ssd_blocks=ssd_blocks_to_transfer,
                                                                            gpu_device_id=0,
                                                                            layer_num=layer_num)

        # NOTE: for now in build transfer graph, we assume that cpu works as a cache for ssd
        if layer_num // layer_granularity != 1:
            transfer_graph, finished_ops_ids = convert_read_graph_to_layer_wise_graph(transfer_graph=transfer_graph,
                                                                                    finished_ops_ids=finished_ops_ids,
                                                                                    layer_num=layer_num,
                                                                                    layer_granularity=layer_granularity)
        # print(f"IN GET FUNCTION IN CACHE ENGINE:")
        # transfer_graph.print_op_map()
        return_mask = torch.zeros_like(token_mask)
        return_mask[start_idx* self.tokens_per_block:
                    (start_idx + len(gpu_blocks_to_transfer)) * self.tokens_per_block] = True

        node_to_unlock = {DeviceType.CPU: cpu_node_to_unlock,
                          DeviceType.SSD: ssd_node_to_unlock}
        if remote_enabled:
            node_to_unlock[DeviceType.REMOTE] = remote_node_to_unlock
        buffer_to_free = {DeviceType.CPU: cpu_blocks_to_free}

        callback = partial(self._transfer_callback,
                           node_to_unlock=node_to_unlock,
                           buffer_to_free=buffer_to_free)

        return transfer_graph, return_mask, callback, finished_ops_ids

    def put(self,
            token_ids: torch.Tensor,
            token_mask: torch.Tensor,
            slot_mapping: torch.Tensor,
            layer_num : int = -1) -> Tuple[TransferOpGraph, torch.Tensor, Callable, List[int]]:
        if not self.cache_config.enable_remote and not self.cache_config.enable_ssd:
            return self._put_only_cpu(token_ids, token_mask, slot_mapping, layer_num)

        self._check_input(token_ids, token_mask, slot_mapping)

        if layer_num == -1:
            layer_num = self.model_config.num_layers

        # ignore the last incomplete block
        aligned_length = (token_ids.shape[0] // self.tokens_per_block) * self.tokens_per_block
        aligned_token_ids = token_ids[:aligned_length]
        # TODO: support put the last incomplete block
        token_mask[aligned_length:] = False
        start_idx, end_idx = self._get_block_range(token_mask)

        # the mask should has a prefix of True
        assert start_idx == 0

        gpu_block_mapping = self._slot_to_block_mapping(slot_mapping)[:end_idx-start_idx]

        sequence_meta = SequenceMeta(token_ids=aligned_token_ids,
                                     tokens_per_block=self.cache_config.tokens_per_block)

        cpu_matched_result, ssd_matched_result = self.match_local(sequence_meta)
        cpu_matched_blocks = cpu_matched_result.physical_blocks[start_idx:end_idx]
        ssd_matched_blocks = ssd_matched_result.physical_blocks[start_idx:end_idx]
        remote_matched_blocks = []
        if self.cache_config.enable_remote:
            #TODO: perhaps we need to lock the last node of remote cache engine
            # at the same time we match it in real remote cache engine
            remote_matched_result = self.remote_cache_engine.match(sequence_meta)

        # in fact, we still face the problem that some of the matched cpu blocks is not ready
        # but we need to copy them into ssd or remote. We have the assumption that the not ready blocks
        # will be ready after the copy of new blocks in this put operation from gpu to cpu is done.
        # this should be good.

        # cpu works as a cache for ssd
        assert len(cpu_matched_blocks) <= len(ssd_matched_blocks)

        gpu_blocks_to_transfer = gpu_block_mapping[len(cpu_matched_blocks):]
        cpu_blocks_to_transfer = self.cpu_cache_engine.take(
            num_required_blocks=len(gpu_block_mapping) - len(cpu_matched_blocks),
            protected_node = cpu_matched_result.last_node,
            strict=True
        )
        ssd_blocks_to_transfer = self.ssd_cache_engine.take(
            num_required_blocks=len(gpu_block_mapping) - len(ssd_matched_blocks),
            protected_node = ssd_matched_result.last_node,
            strict=True
        )

        cpu_node_to_unlock = self.cpu_cache_engine.insert(sequence_meta,
                                                          cpu_blocks_to_transfer,
                                                          is_ready=False,
                                                          match_result=cpu_matched_result)
        ssd_node_to_unlock = self.ssd_cache_engine.insert(sequence_meta,
                                                          ssd_blocks_to_transfer,
                                                          is_ready=False,
                                                          match_result=ssd_matched_result)

        self.cpu_cache_engine.lock_node(cpu_node_to_unlock)
        self.ssd_cache_engine.lock_node(ssd_node_to_unlock)

        # NOTE the take operation and lock operation should be automic with the match operation for real remote engine

        remote_blocks_to_transfer = torch.tensor([])
        remote_node_to_unlock = None
        if self.cache_config.enable_remote:
            remote_blocks_to_transfer = self.remote_cache_engine.take(
                num_required_blocks=len(gpu_block_mapping) - len(remote_matched_blocks),
                protected_node = remote_matched_result.last_node,
                strict=True
            )
            remote_node_to_unlock = self.remote_cache_engine.insert(sequence_meta,
                                                                    remote_blocks_to_transfer,
                                                                    is_ready=False,
                                                                    match_result=remote_matched_result)
            self.remote_cache_engine.lock_node(remote_node_to_unlock)
            # we need more cpu blocks incase the matched length of remote is less than that of cpu
            if len(remote_blocks_to_transfer) > len(cpu_blocks_to_transfer):
                extra_cpu_blocks_num = len(remote_blocks_to_transfer) - len(cpu_blocks_to_transfer)
                extra_cpu_blocks = cpu_matched_result.physical_blocks[-extra_cpu_blocks_num:]
                cpu_blocks_to_transfer = torch.cat([extra_cpu_blocks, cpu_blocks_to_transfer])

        transfer_graph, finished_ops_ids = create_write_graph_cpu_ssd_remote(gpu_blocks=gpu_blocks_to_transfer,
                                                                            cpu_blocks=cpu_blocks_to_transfer,
                                                                            ssd_blocks=ssd_blocks_to_transfer,
                                                                            remote_blocks=remote_blocks_to_transfer,
                                                                            gpu_device_id = 0,
                                                                            layer_num = layer_num)
        #print(f"IN PUT FUNCTION IN CACHE ENGINE:")
        # transfer_graph.print_op_map()

        return_mask = torch.zeros_like(token_mask)
        return_mask[start_idx* self.tokens_per_block:
                    (start_idx + len(gpu_block_mapping)) * self.tokens_per_block] = True

        node_to_unlock = {DeviceType.CPU: cpu_node_to_unlock,
                          DeviceType.SSD: ssd_node_to_unlock}
        if self.cache_config.enable_remote:
            node_to_unlock[DeviceType.REMOTE] = remote_node_to_unlock

        callback = partial(self._transfer_callback,
                           node_to_unlock=node_to_unlock)
        return transfer_graph, return_mask, callback, finished_ops_ids

    def _get_only_cpu(self,
                      token_ids: torch.Tensor,
                      token_mask: torch.Tensor,
                      slot_mapping: torch.Tensor,
                      layer_num: int = -1,
                      layer_granularity: int = -1) -> Tuple[TransferOpGraph, torch.Tensor, Callable, List[int]]:
        self._check_input(token_ids, token_mask, slot_mapping)

        if layer_num == -1:
            layer_num = self.model_config.num_layers
        if layer_granularity == -1:
            layer_granularity = layer_num

        # ignore the last incomplete block
        aligned_length = (token_ids.shape[0] // self.tokens_per_block) * self.tokens_per_block
        aligned_token_ids = token_ids[:aligned_length]
        # TODO: support get the last incomplete block
        token_mask[aligned_length:] = False
        start_idx, end_idx = self._get_block_range(token_mask)

        assert end_idx == aligned_length // self.tokens_per_block

        #NOTE: the slot_mapping starts from token that need to be fetched
        gpu_block_mapping = self._slot_to_block_mapping(slot_mapping)[:end_idx-start_idx]

        sequence_meta = SequenceMeta(token_ids=aligned_token_ids,
                                     tokens_per_block=self.cache_config.tokens_per_block)

        cpu_matched_result, _ = self.match_local(sequence_meta)

        # tailor the blocks to assure:
        # the blocks are needed by the mask & the blocks are ready
        cpu_physical_blocks = cpu_matched_result.physical_blocks[:cpu_matched_result.num_ready_matched_blocks]
        cpu_physical_blocks = cpu_physical_blocks[start_idx:end_idx]

        num_transfer_blocks = len(cpu_physical_blocks)
        assert num_transfer_blocks <= len(gpu_block_mapping)


        gpu_blocks_to_transfer = gpu_block_mapping[:num_transfer_blocks]
        cpu_blocks_to_transfer = cpu_physical_blocks

        cpu_node_to_unlock = cpu_matched_result.last_ready_node

        # prepare cpu blocks to transfer
        self.cpu_cache_engine.lock_node(cpu_node_to_unlock)

        transfer_graph, finished_ops_ids = create_read_graph_cpu_storage(gpu_blocks=gpu_blocks_to_transfer,
                                                                         cpu_blocks=cpu_blocks_to_transfer,
                                                                         ssd_blocks=torch.tensor([]),
                                                                         gpu_device_id=0,
                                                                         layer_num=layer_num)

        # NOTE: for now in build transfer graph, we assume that cpu works as a cache for ssd
        if layer_num // layer_granularity != 1:
            transfer_graph, finished_ops_ids = convert_read_graph_to_layer_wise_graph(transfer_graph=transfer_graph,
                                                                                    finished_ops_ids=finished_ops_ids,
                                                                                    layer_num=layer_num,
                                                                                    layer_granularity=layer_granularity)
        # print(f"IN GET FUNCTION IN CACHE ENGINE:")
        # transfer_graph.print_op_map()
        return_mask = torch.zeros_like(token_mask)
        return_mask[start_idx* self.tokens_per_block:
                    (start_idx + len(gpu_blocks_to_transfer)) * self.tokens_per_block] = True

        node_to_unlock = {DeviceType.CPU: cpu_node_to_unlock}

        callback = partial(self._transfer_callback,
                           node_to_unlock=node_to_unlock)

        return transfer_graph, return_mask, callback, finished_ops_ids

    def _put_only_cpu(self,
                      token_ids: torch.Tensor,
                      token_mask: torch.Tensor,
                      slot_mapping: torch.Tensor,
                      layer_num: int = -1) -> Tuple[TransferOpGraph, torch.Tensor, Callable, List[int]]:
        self._check_input(token_ids, token_mask, slot_mapping)

        if layer_num == -1:
            layer_num = self.model_config.num_layers

        # ignore the last incomplete block
        aligned_length = (token_ids.shape[0] // self.tokens_per_block) * self.tokens_per_block
        aligned_token_ids = token_ids[:aligned_length]
        # TODO: support put the last incomplete block
        token_mask[aligned_length:] = False
        start_idx, end_idx = self._get_block_range(token_mask)

        # the mask should has a prefix of True
        assert start_idx == 0

        gpu_block_mapping = self._slot_to_block_mapping(slot_mapping)[:end_idx-start_idx]

        sequence_meta = SequenceMeta(token_ids=aligned_token_ids,
                                     tokens_per_block=self.cache_config.tokens_per_block)

        cpu_matched_result, _ = self.match_local(sequence_meta)
        cpu_matched_blocks = cpu_matched_result.physical_blocks[start_idx:end_idx]

        gpu_blocks_to_transfer = gpu_block_mapping[len(cpu_matched_blocks):]
        cpu_blocks_to_transfer = self.cpu_cache_engine.take(
            num_required_blocks=len(gpu_block_mapping) - len(cpu_matched_blocks),
            protected_node = cpu_matched_result.last_node,
            strict=True
        )

        cpu_node_to_unlock = self.cpu_cache_engine.insert(sequence_meta,
                                                          cpu_blocks_to_transfer,
                                                          is_ready=False,
                                                          match_result=cpu_matched_result)

        self.cpu_cache_engine.lock_node(cpu_node_to_unlock)

        transfer_graph, finished_ops_ids = create_write_graph_cpu_ssd_remote(gpu_blocks=gpu_blocks_to_transfer,
                                                                            cpu_blocks=cpu_blocks_to_transfer,
                                                                            ssd_blocks=torch.tensor([]),
                                                                            remote_blocks=torch.tensor([]),
                                                                            gpu_device_id = 0,
                                                                            layer_num = layer_num)
        # print(f"IN PUT FUNCTION IN CACHE ENGINE:")
        # transfer_graph.print_op_map()

        return_mask = torch.zeros_like(token_mask)
        return_mask[start_idx* self.tokens_per_block:
                    (start_idx + len(gpu_block_mapping)) * self.tokens_per_block] = True

        node_to_unlock = {DeviceType.CPU: cpu_node_to_unlock}

        callback = partial(self._transfer_callback,
                           node_to_unlock=node_to_unlock)
        return transfer_graph, return_mask, callback, finished_ops_ids

    def _transfer_callback(self,
                           node_to_unlock: Dict[DeviceType, RadixNode],
                           buffer_to_free: Optional[Dict[DeviceType, torch.Tensor]] = None) -> None:
        if DeviceType.CPU in node_to_unlock:
            self.cpu_cache_engine.cleanup(node_to_unlock[DeviceType.CPU])
        if DeviceType.SSD in node_to_unlock:
            self.ssd_cache_engine.cleanup(node_to_unlock[DeviceType.SSD])
        if DeviceType.REMOTE in node_to_unlock:
            self.remote_cache_engine.cleanup(node_to_unlock[DeviceType.REMOTE])
        if buffer_to_free is not None:
            if DeviceType.CPU in buffer_to_free:
                self.cpu_cache_engine.recycle(buffer_to_free[DeviceType.CPU])
            if DeviceType.SSD in buffer_to_free:
                self.ssd_cache_engine.recycle(buffer_to_free[DeviceType.SSD])
            if DeviceType.REMOTE in buffer_to_free:
                self.remote_cache_engine.recycle(buffer_to_free[DeviceType.REMOTE])

    def need_remote_cache(self,
                          cpu_matched_result: MatchResult,
                          ssd_matched_result: MatchResult,
                          num_required_blocks: int) -> bool:
        #TODO: we need cost function to decide whether to use remote cache
        return self.cache_config.enable_remote

    def match_local(self, sequence_meta: SequenceMeta) -> Tuple[MatchResult, MatchResult]:
        cpu_matched_result = MatchResult()
        ssd_matched_result = MatchResult()
        if self.cpu_cache_engine:
            cpu_matched_result = self.cpu_cache_engine.match(sequence_meta)
        if self.ssd_cache_engine:
            ssd_matched_result = self.ssd_cache_engine.match(sequence_meta)

        return cpu_matched_result, ssd_matched_result

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
                      token_ids: torch.Tensor,
                      token_mask: torch.Tensor,
                      slot_mapping: torch.Tensor) -> None:
        assert token_ids.ndim == 1
        assert token_mask.ndim == 1
        assert slot_mapping.ndim == 1
        assert len(token_ids) == len(token_mask)
        assert len(slot_mapping) == token_mask.sum().item()

    def _slot_to_block_mapping(self,
                              slot_mapping: torch.Tensor) -> torch.Tensor:
        block_mapping = slot_mapping[::self.tokens_per_block] // self.tokens_per_block
        return block_mapping

    def _get_block_range(self,
                         token_mask: torch.Tensor) -> Tuple[int, int]:
        mask_idx = torch.where(token_mask)[0]
        start_idx = mask_idx[0].item() // self.tokens_per_block
        end_idx = mask_idx[-1].item() // self.tokens_per_block
        return start_idx, end_idx + 1
