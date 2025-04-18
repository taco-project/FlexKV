from typing import List, Tuple

import torch

from flexkv.common.transfer import DeviceType
from flexkv.cache.index import TokenToBlockIndex
from flexkv.cache.mempool import Mempool
from flexkv.common.block import BlockMeta, SequenceMeta
from flexkv.common.config import CacheConfig


class CacheEngine:
    def __init__(self,
                 device_type: DeviceType,
                 num_total_blocks: int,
                 tokens_per_block: int):
        self.device_type = device_type

        self.index = TokenToBlockIndex(tokens_per_block=tokens_per_block)

        self.mempool = Mempool(num_total_blocks=num_total_blocks)

    def match(self, sequence_meta: SequenceMeta) -> List[BlockMeta]:
        block_metas = self.index.match_prefix(sequence_meta,
                                              update_cache_info=True,
                                              lock_blocks=True)
        return block_metas

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

    def get(self, token_ids: torch.Tensor) -> List[BlockMeta]:
        sequence_meta = SequenceMeta(token_ids=token_ids,
                                     tokens_per_block=self.cache_config.tokens_per_block)

        cpu_match, ssd_match, remote_match = self.match(sequence_meta)

        ...


    def put(self, token_ids: torch.Tensor):
        sequence_meta = SequenceMeta(token_ids=token_ids,
                                     tokens_per_block=self.cache_config.tokens_per_block)

        cpu_match, ssd_match, remote_match = self.match(sequence_meta)

        ...

    def match(self, sequence_meta: SequenceMeta) \
        -> Tuple[List[BlockMeta], List[BlockMeta], List[BlockMeta]]:
        cpu_match_results = []
        ssd_match_results = []
        remote_match_results = []

        if self.cpu_cache_engine:
            cpu_match_results = self.cpu_cache_engine.match(sequence_meta)
        if self.ssd_cache_engine:
            ssd_match_results = self.ssd_cache_engine.match(sequence_meta)
        if self.remote_cache_engine:
            remote_match_results = self.remote_cache_engine.match(sequence_meta)

        return cpu_match_results, ssd_match_results, remote_match_results
