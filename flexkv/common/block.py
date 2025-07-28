from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, NewType, Optional

import numpy as np
import torch

from flexkv.common.hash_utils import HashType, gen_hashes, get_hash_size, hash_tensor


@dataclass
class SequenceMeta:

    token_ids: torch.Tensor

    tokens_per_block: int

    block_hashes: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    _has_hashes: bool = False

    def __init__(self, token_ids: torch.Tensor, tokens_per_block: int):
        self.token_ids = token_ids
        self.tokens_per_block = tokens_per_block

    def __post_init__(self) -> None:
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

    def get_hash(self, block_id: int) -> Optional[HashType]:
        if block_id >= self.num_blocks:
            return None
        if self._has_hashes:
            return HashType(int(self.block_hashes[block_id].item()))
        else:
            return hash_tensor(self.token_ids[:(block_id+1)*self.tokens_per_block])

    def gen_hashes(self) -> None:
        if self._has_hashes:
            return
        assert self.token_ids.ndim == 1
        self.block_hashes = gen_hashes(self.token_ids, self.tokens_per_block).numpy()
        assert self.block_hashes.ndim == 1
        assert self.block_hashes.size == self.num_blocks
        assert self.block_hashes.itemsize == get_hash_size()
        self._has_hashes = True
