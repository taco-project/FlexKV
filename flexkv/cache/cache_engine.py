from typing import List, Tuple, Optional, Dict
from queue import Queue
import torch
import time
import threading
from functools import partial

from flexkv.common.transfer import DeviceType, TransferOpGraph
from flexkv.cache.index import TokenToBlockIndex
from flexkv.cache.mempool import Mempool
from flexkv.common.block import BlockMeta, SequenceMeta, BlockStatus
from flexkv.common.config import CacheConfig
from flexkv.cache.transfer_pattern import create_read_transfer_graph, create_write_transfer_graph
from flexkv.common.request import cacheEngineRequestType, cacheEngineRequest

class CacheEngine:
    def __init__(self,
                 device_type: DeviceType,
                 num_total_blocks: int,
                 tokens_per_block: int):
        self.device_type = device_type

        self.index = TokenToBlockIndex(tokens_per_block=tokens_per_block)

        self.mempool = Mempool(num_total_blocks=num_total_blocks)

    def reset(self):
        self.index.reset()
        self.mempool.reset()

    def match(self, sequence_meta: SequenceMeta) -> torch.Tensor:
        physical_block_ids = self.index.match_prefix(sequence_meta,
                                              update_cache_info=True)
        return physical_block_ids

    def match_length(self, sequence_meta: SequenceMeta) -> int:
        return self.index.match_length(sequence_meta)

    def insert(self,
               sequence_meta: SequenceMeta,
               physical_block_ids: torch.Tensor,
               match_length: int = -1,
               is_ready: bool = True,
               locked: bool = False,
               as_buffer: bool = False) -> None:
        if match_length == -1:
            # in insert, we can use the default maximum_status for match
            match_length = self.index.match_length(sequence_meta=sequence_meta)
        self.index.insert(sequence_meta, physical_block_ids, match_length, is_ready, locked, as_buffer)

    def lock_blocks(self, block_ids: torch.Tensor) -> None:
        self.index.lock(block_ids)

    def unlock_blocks(self, block_ids: torch.Tensor) -> None:
        self.index.unlock(block_ids)

    def set_ready(self, block_ids: torch.Tensor, is_ready: bool = True) -> None:
        self.index.set_ready(block_ids, is_ready)

    def take(self,
             num_required_blocks: int,
             protected_blocks: Optional[torch.Tensor] = None,
             strict: bool = True) -> List[int]:
        if num_required_blocks > self.mempool.num_free_blocks:
            if protected_blocks is not None:
                self.lock_blocks(protected_blocks)
            self.mempool.recycle_blocks(
                self.index.evict(num_required_blocks - self.mempool.num_free_blocks)
            )
            if protected_blocks is not None:
                self.unlock_blocks(protected_blocks)
        if strict and num_required_blocks > self.mempool.num_free_blocks:
            raise ValueError("Not enough free blocks to take")
        return self.mempool.allocate_blocks(num_required_blocks)

    def cleanup(self, block_ids: torch.Tensor, as_buffer: bool = False) -> None:
        if as_buffer:
            # TODO: for now we ignore the buffer blocks because we don't have them now
            # actually we should return those blocks to the mempool
            raise NotImplementedError("Buffer cleanup is not implemented")
        else:
            self.set_ready(block_ids)
            self.unlock_blocks(block_ids)

class GlobalCacheEngine:
    def __init__(self, cache_config: CacheConfig):
        self.cache_config = cache_config
        self.tokens_per_block = cache_config.tokens_per_block

        self.cpu_cache_engine = None
        self.ssd_cache_engine = None
        self.remote_cache_engine = None

        if not cache_config.enable_cpu and not cache_config.enable_ssd:
            raise ValueError("Either enable_cpu or enable_ssd must be True")

        if not cache_config.enable_cpu and not cache_config.use_gds:
            raise ValueError("use_gds must be True if enable_cpu is False")

        if cache_config.enable_cpu:
            self.cpu_cache_engine = CacheEngine(DeviceType.CPU,
                                                cache_config.num_cpu_blocks,
                                                cache_config.tokens_per_block)
        if cache_config.enable_ssd:
            self.ssd_cache_engine = CacheEngine(DeviceType.SSD,
                                                cache_config.num_ssd_blocks,
                                                cache_config.tokens_per_block)
        if cache_config.enable_remote:
            raise NotImplementedError("Remote cache is not implemented")

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
            slot_mapping: torch.Tensor) -> Tuple[TransferOpGraph, torch.Tensor]:
        self._check_input(token_ids, token_mask, slot_mapping)

        # ignore the last incomplete block
        aligned_length = (token_ids.shape[0] // self.tokens_per_block) * self.tokens_per_block
        aligned_token_ids = token_ids[:aligned_length]
        # TODO: support get the last incomplete block
        token_mask[aligned_length:] = False
        start_idx, end_idx = self._get_block_range(token_mask)
        #NOTE: the slot_mapping starts from token that need to be fetched
        gpu_block_mapping = self._slot_to_block_mapping(slot_mapping)[:end_idx-start_idx]

        sequence_meta = SequenceMeta(token_ids=aligned_token_ids,
                                     tokens_per_block=self.cache_config.tokens_per_block)

        # TODO(very important): in some cases, real cpu_matched_blocks < start_idx, this will cause bugs
        cpu_matched_blocks, ssd_matched_blocks, _ = self.match(sequence_meta, start_idx, end_idx)

        num_transfer_blocks = max(len(cpu_matched_blocks), len(ssd_matched_blocks))
        assert num_transfer_blocks <= len(gpu_block_mapping)

        gpu_blocks_to_transfer = gpu_block_mapping[:num_transfer_blocks]
        cpu_blocks_to_transfer = cpu_matched_blocks
        ssd_blocks_to_transfer = ssd_matched_blocks[len(cpu_matched_blocks):]

        self.cpu_cache_engine.lock_blocks(cpu_blocks_to_transfer)
        self.ssd_cache_engine.lock_blocks(ssd_blocks_to_transfer)

        if len(cpu_matched_blocks) < len(ssd_matched_blocks):
            extra_cpu_blocks = self.cpu_cache_engine.take(
                num_required_blocks=len(ssd_matched_blocks) - len(cpu_matched_blocks),
                protected_blocks=cpu_matched_blocks[-1:],
                strict=True
            )
            cpu_blocks_to_transfer = torch.cat([cpu_blocks_to_transfer, extra_cpu_blocks])
            # here we still put the buffer in the cpu index, so no buffer is needed
            self.cpu_cache_engine.insert(sequence_meta,
                                         extra_cpu_blocks,
                                         len(cpu_matched_blocks) + start_idx,
                                         is_ready=False,
                                         locked=True,
                                         as_buffer=False)

        # NOTE: for now in build transfer graph, we assume that cpu works as a cache for ssd
        transfer_graph = create_read_transfer_graph(ssd_blocks_to_transfer,
                                                    cpu_blocks_to_transfer,
                                                    gpu_blocks_to_transfer)

        return_mask = torch.zeros_like(token_mask)
        return_mask[start_idx* self.tokens_per_block:
                    (start_idx + len(gpu_blocks_to_transfer)) * self.tokens_per_block] = True

        callback = partial(self._transfer_callback,
                           block_ids_to_unlock={DeviceType.CPU: cpu_blocks_to_transfer,
                                                DeviceType.SSD: ssd_blocks_to_transfer})

        return transfer_graph, return_mask, callback

    def put(self,
            token_ids: torch.Tensor,
            token_mask: torch.Tensor,
            slot_mapping: torch.Tensor) -> Tuple[TransferOpGraph, torch.Tensor]:
        self._check_input(token_ids, token_mask, slot_mapping)

        # ignore the last incomplete block
        aligned_length = (token_ids.shape[0] // self.tokens_per_block) * self.tokens_per_block
        aligned_token_ids = token_ids[:aligned_length]
        # TODO: support put the last incomplete block
        token_mask[aligned_length:] = False
        # WARNING the start_idx and end_idx is token level, not block level
        start_idx, end_idx = self._get_block_range(token_mask)

        # the mask should has a prefix of True
        assert start_idx == 0

        gpu_block_mapping = self._slot_to_block_mapping(slot_mapping)[:end_idx-start_idx]

        sequence_meta = SequenceMeta(token_ids=aligned_token_ids,
                                     tokens_per_block=self.cache_config.tokens_per_block)

        cpu_matched_blocks, ssd_matched_blocks, _ = self.match(sequence_meta, start_idx, end_idx)

        # cpu works as a cache for ssd
        assert len(cpu_matched_blocks) <= len(ssd_matched_blocks)

        gpu_blocks_to_transfer = gpu_block_mapping[len(cpu_matched_blocks):]
        # lock the last block of cpu and ssd to avoid eviction in this turn
        cpu_blocks_to_transfer = self.cpu_cache_engine.take(
            num_required_blocks=len(gpu_block_mapping) - len(cpu_matched_blocks),
            protected_blocks = cpu_matched_blocks[-1:],
            strict=True
        )
        ssd_blocks_to_transfer = self.ssd_cache_engine.take(
            num_required_blocks=len(gpu_block_mapping) - len(ssd_matched_blocks),
            protected_blocks = ssd_matched_blocks[-1:],
            strict=True
        )

        self.cpu_cache_engine.insert(sequence_meta,
                                     cpu_blocks_to_transfer,
                                     len(cpu_matched_blocks) + start_idx,  # start_idx == 0
                                     is_ready=False,
                                     locked=True,
                                     as_buffer=False)
        self.ssd_cache_engine.insert(sequence_meta,
                                     ssd_blocks_to_transfer,
                                     len(ssd_matched_blocks) + start_idx,  # start_idx == 0
                                     is_ready=False,
                                     locked=True,
                                     as_buffer=False)

        transfer_graph = create_write_transfer_graph(ssd_blocks_to_transfer,
                                                    cpu_blocks_to_transfer,
                                                    gpu_blocks_to_transfer)

        return_mask = torch.zeros_like(token_mask)
        return_mask[start_idx* self.tokens_per_block:
                    (start_idx + len(gpu_block_mapping)) * self.tokens_per_block] = True

        callback = partial(self._transfer_callback,
                           block_ids_to_unlock={DeviceType.CPU: cpu_blocks_to_transfer,
                                                DeviceType.SSD: ssd_blocks_to_transfer})
        return transfer_graph, return_mask, callback

    def _transfer_callback(self,
                           block_ids_to_unlock: Dict[DeviceType, torch.Tensor]):
        if DeviceType.CPU in block_ids_to_unlock:
            self.cpu_cache_engine.cleanup(block_ids_to_unlock[DeviceType.CPU], as_buffer=False)
        if DeviceType.SSD in block_ids_to_unlock:
            self.ssd_cache_engine.cleanup(block_ids_to_unlock[DeviceType.SSD], as_buffer=False)

    def match(self, sequence_meta: SequenceMeta,
              start_idx: int,
              end_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cpu_matched_blocks = torch.empty(0, dtype=torch.int64)
        ssd_matched_blocks = torch.empty(0, dtype=torch.int64)
        remote_matched_blocks = torch.empty(0, dtype=torch.int64)
        # TODO: avoid redundant match?
        if self.cpu_cache_engine:
            cpu_matched_blocks = self.cpu_cache_engine.match(sequence_meta)[start_idx:end_idx]
        if self.ssd_cache_engine:
            ssd_matched_blocks = self.ssd_cache_engine.match(sequence_meta)[start_idx:end_idx]
        if self.remote_cache_engine:
            remote_matched_blocks = self.remote_cache_engine.match(sequence_meta)[start_idx:end_idx]

        return cpu_matched_blocks, ssd_matched_blocks, remote_matched_blocks

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
        block_mapping = slot_mapping // self.tokens_per_block
        block_mapping = torch.unique(block_mapping)
        return block_mapping

    def _get_block_range(self,
                         token_mask: torch.Tensor) -> Tuple[int, int]:
        mask_idx = torch.where(token_mask)[0]
        start_idx = mask_idx[0].item() // self.tokens_per_block
        end_idx = mask_idx[-1].item() // self.tokens_per_block
        return start_idx, end_idx + 1
