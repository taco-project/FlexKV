from typing import List, Tuple

import torch

from flexkv.common.transfer import DeviceType, TransferOpGraph
from flexkv.cache.index import TokenToBlockIndex
from flexkv.cache.mempool import Mempool
from flexkv.common.block import BlockMeta, SequenceMeta
from flexkv.common.config import CacheConfig
from flexkv.cache.transfer_pattern import create_read_transfer_graph, create_write_transfer_graph

class CacheEngine:
    def __init__(self,
                 device_type: DeviceType,
                 num_total_blocks: int,
                 tokens_per_block: int):
        self.device_type = device_type

        self.index = TokenToBlockIndex(tokens_per_block=tokens_per_block)

        self.mempool = Mempool(num_total_blocks=num_total_blocks)

    def match(self, sequence_meta: SequenceMeta) -> torch.Tensor:
        physical_block_ids = self.index.match_prefix(sequence_meta,
                                              update_cache_info=True,
                                              lock_blocks=True)
        return physical_block_ids

    def match_length(self, sequence_meta: SequenceMeta) -> int:
        return self.index.match_length(sequence_meta)

    def insert(self, sequence_meta: SequenceMeta) -> None:
        self.index.insert(sequence_meta)

    def take(self, num_required_blocks: int) -> List[int]:
        if num_required_blocks > self.mempool.num_free_blocks:
            self.mempool.recycle_blocks(
                self.index.evict(
                    num_required_blocks - self.mempool.num_free_blocks
                )
            )
        if num_required_blocks > self.mempool.num_free_blocks:
            raise ValueError("Not enough free blocks to take")
        return self.mempool.allocate_blocks(num_required_blocks)

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

    def get(self,
            token_ids: torch.Tensor,
            token_mask: torch.Tensor,
            slot_mapping: torch.Tensor) -> TransferOpGraph:
        self._check_input(token_ids, token_mask, slot_mapping)

        # ignore the last incomplete block
        aligned_length = (token_ids.shape[0] // self.tokens_per_block) * self.tokens_per_block
        aligned_token_ids = token_ids[:aligned_length]
        start_idx, end_idx = self._get_block_range(token_mask)

        gpu_block_mapping = self.slot_to_block_mapping(slot_mapping)[:end_idx-start_idx]

        sequence_meta = SequenceMeta(token_ids=aligned_token_ids,
                                     tokens_per_block=self.cache_config.tokens_per_block)

        cpu_matched_blocks, ssd_matched_blocks, _ = self.match(sequence_meta, start_idx, end_idx)

        num_transfer_blocks = max(len(cpu_matched_blocks), len(ssd_matched_blocks))
        assert num_transfer_blocks <= len(gpu_block_mapping)

        gpu_blocks_to_transfer = gpu_block_mapping[:num_transfer_blocks]
        cpu_blocks_to_transfer = cpu_matched_blocks
        ssd_blocks_to_transfer = ssd_matched_blocks[len(cpu_matched_blocks):]

        if len(cpu_matched_blocks) < len(ssd_matched_blocks):
            cpu_blocks_to_transfer = torch.cat([cpu_blocks_to_transfer,
                                                self.cpu_cache_engine.take(
                                                    len(ssd_matched_blocks) - len(cpu_matched_blocks)
                                                )])

        transfer_graph = create_read_transfer_graph(ssd_blocks_to_transfer,
                                                    cpu_blocks_to_transfer,
                                                    gpu_blocks_to_transfer)

        return_mask = torch.zeros_like(token_mask)
        return_mask[start_idx* self.tokens_per_block:
                    (start_idx + len(gpu_blocks_to_transfer)) * self.tokens_per_block] = True

        return transfer_graph, return_mask

    def put(self,
            token_ids: torch.Tensor,
            token_mask: torch.Tensor,
            slot_mapping: torch.Tensor) -> TransferOpGraph:
        self._check_input(token_ids, token_mask, slot_mapping)

        # ignore the last incomplete block
        aligned_length = (token_ids.shape[0] // self.tokens_per_block) * self.tokens_per_block
        aligned_token_ids = token_ids[:aligned_length]
        start_idx, end_idx = self._get_block_range(token_mask)

        # the mask should has a prefix of True
        assert start_idx == 0

        gpu_block_mapping = self.slot_to_block_mapping(slot_mapping)[:end_idx-start_idx]

        sequence_meta = SequenceMeta(token_ids=aligned_token_ids,
                                     tokens_per_block=self.cache_config.tokens_per_block)

        cpu_matched_blocks, ssd_matched_blocks, _ = self.match(sequence_meta, start_idx, end_idx)

        # cpu works as a cache for ssd
        assert len(cpu_matched_blocks) <= len(ssd_matched_blocks)

        gpu_blocks_to_transfer = gpu_block_mapping[len(cpu_matched_blocks):]
        cpu_blocks_to_transfer = self.cpu_cache_engine.take(len(gpu_block_mapping) - len(cpu_matched_blocks))
        ssd_blocks_to_transfer = self.ssd_cache_engine.take(len(gpu_block_mapping) - len(ssd_matched_blocks))

        transfer_graph = create_write_transfer_graph(ssd_blocks_to_transfer,
                                                    cpu_blocks_to_transfer,
                                                    gpu_blocks_to_transfer)

        return_mask = torch.zeros_like(token_mask)
        return_mask[start_idx* self.tokens_per_block:
                    (start_idx + len(gpu_block_mapping)) * self.tokens_per_block] = True

        return transfer_graph, return_mask

    def match(self, sequence_meta: SequenceMeta,
              start_idx: int,
              end_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cpu_match_results = []
        ssd_match_results = []
        remote_match_results = []

        if self.cpu_cache_engine:
            cpu_match_results = self.cpu_cache_engine.match(sequence_meta)[start_idx:end_idx]
        if self.ssd_cache_engine:
            ssd_match_results = self.ssd_cache_engine.match(sequence_meta)[start_idx:end_idx]
        if self.remote_cache_engine:
            remote_match_results = self.remote_cache_engine.match(sequence_meta)[start_idx:end_idx]

        return cpu_match_results, ssd_match_results, remote_match_results

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
        start_idx = mask_idx[0].item()
        end_idx = mask_idx[-1].item()
        return start_idx, end_idx + 1
