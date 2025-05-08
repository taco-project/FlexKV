from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, NewType, Optional

import numpy as np
import torch

from flexkv.common.hash_utils import HashType, gen_hashes, get_hash_size


class BlockStatus(Enum):
    UNREGISTERED = 0
    AVAILABLE = 1
    LOCKED = 2
    IN_GET = 3
    IN_PUT = 4


@dataclass
class BlockMeta:

    hash: HashType

    prev_hash: Optional[HashType]

    physical_block_id: int = -1

    last_access_time: float = 0

    lock_cnt: int = 0

    child_cnt: int = 0

    status: BlockStatus = BlockStatus.UNREGISTERED

    def __lt__(self, other: "BlockMeta") -> bool:
        return self.last_access_time < other.last_access_time

    def __str__(self):
        return (
            f"BlockMeta(hash={self.hash}, "
            f"prev_hash={self.prev_hash if self.prev_hash else None}, "
            f"physical_block_id={self.physical_block_id}, "
            f"last_access_time={self.last_access_time}, "
            f"lock_cnt={self.lock_cnt}, "
            f"child_cnt={self.child_cnt}, "
            f"status={self.status})"
        )

    def __repr__(self):
        return self.__str__()


@dataclass
class SequenceMeta:

    token_ids: torch.Tensor

    tokens_per_block: int

    block_hashes: Optional[torch.Tensor] = None

    _has_hashes: bool = False

    def __init__(self, token_ids: torch.Tensor, tokens_per_block: int):
        self.token_ids = token_ids
        self.tokens_per_block = tokens_per_block

    def __post_init__(self):
        assert self.token_ids.ndim == 1
        assert self.tokens_per_block > 0

    @property
    def num_blocks(self) -> int:
        return len(self.token_ids) // self.tokens_per_block

    @property
    def length(self) -> int:
        return len(self.token_ids)

    def has_hashes(self) -> bool:
        return self._has_hashes

    def get_hash(self, block_id: int) -> HashType:
        assert self._has_hashes
        return self.block_hashes[block_id].item()

    def gen_hashes(self) -> None:
        if self._has_hashes:
            return
        assert self.token_ids.ndim == 1
        self.block_hashes = gen_hashes(self.token_ids, self.tokens_per_block)
        assert self.block_hashes.ndim == 1
        assert self.block_hashes.numel() == self.num_blocks
        assert self.block_hashes.element_size() == get_hash_size()
        self._has_hashes = True
