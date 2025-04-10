import abc
from typing import List
from enum import Enum, auto

from flexkv.core.block import BlockMeta


class EvictionPolicy(Enum):
    LRU = auto()


class Evictor(abc.ABC):
    @abc.abstractmethod
    def evict(self, candidate_blocks: List[BlockMeta]) -> List[int]:
        raise NotImplementedError


class LRUEvictor(Evictor):
    def evict(
        self,
        candidate_blocks: List[BlockMeta],
        evict_num: int,
    ) -> List[BlockMeta]:
        if evict_num > len(candidate_blocks):
            return candidate_blocks
        sorted_blocks = sorted(
            candidate_blocks,
            key=lambda x: x.last_access_time,
        )
        return sorted_blocks[:evict_num]


def create_evictor(policy: EvictionPolicy) -> Evictor:
    if policy == EvictionPolicy.LRU:
        return LRUEvictor()
    else:
        raise ValueError(f"Unsupported eviction policy: {policy}")
