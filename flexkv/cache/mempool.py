from collections import deque
from typing import List

import numpy as np


class Mempool:
    def __init__(
        self,
        num_total_blocks: int,
    ):
        assert num_total_blocks > 0
        self.num_total_blocks = num_total_blocks

        self._free_mask = np.ones(self.num_total_blocks, dtype=np.bool_)
        self._num_free = num_total_blocks
        self._free_ids = self._free_mask.nonzero()[0]
        self._free_ids_offset = 0

    def reset(self) -> None:
        self._free_mask.fill(True)
        self._num_free = self.num_total_blocks
        self._free_ids = self._free_mask.nonzero()[0]
        self._free_ids_offset = 0

    def allocate_blocks(self, num: int) -> np.ndarray:
        if num < 0:
            raise ValueError(f"num must be greater than 0, but got {num}")
        if num > self._num_free:
            raise ValueError(f"Not enough free blocks, required: {num}, available: {self._num_free}")

        if num > len(self._free_ids) - self._free_ids_offset:
            self._update_free_ids()

        free_ids = self._free_ids[self._free_ids_offset:self._free_ids_offset+num]
        self._free_ids_offset += num

        self._free_mask[free_ids] = False
        self._num_free -= num
        return free_ids

    def recycle_blocks(self, block_ids: np.ndarray) -> None:
        if block_ids.ndim != 1 or block_ids.dtype != np.int64:
            raise ValueError("block_ids must be a 1D tensor of int64")
        
        block_ids = np.unique(block_ids)

        already_free = self._free_mask[block_ids]
        if already_free.any():
            raise ValueError(f"block_ids {block_ids[already_free]} are already free")
        self._free_mask[block_ids] = True
        self._num_free += len(block_ids)

    def _update_free_ids(self) -> None:
        self._free_ids = self._free_mask.nonzero()[0]
        self._free_ids_offset = 0

    @property
    def num_free_blocks(self) -> int:
        return self._num_free

    @property
    def num_used_blocks(self) -> int:
        return self.num_total_blocks - self._num_free
