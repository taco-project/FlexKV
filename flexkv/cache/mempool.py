from collections import deque
from typing import List

import torch


class Mempool:
    def __init__(
        self,
        num_total_blocks: int,
    ):
        assert num_total_blocks > 0
        self.num_total_blocks = num_total_blocks

        self._free_mask = torch.ones(self.num_total_blocks, dtype=torch.bool)
        self._num_free = num_total_blocks
        self._free_ids = self._free_mask.nonzero()
        self._free_ids_offset = 0

    def reset(self) -> None:
        self._free_mask.fill_(True)
        self._num_free = self.num_total_blocks
        self._free_ids = self._free_mask.nonzero()
        self._free_ids_offset = 0

    def allocate_blocks(self, num: int) -> torch.Tensor:
        assert num > 0
        assert num <= self._num_free, "Not enough free blocks"

        if num > len(self._free_ids):
            self._update_free_ids()

        free_ids = self._free_ids[self._free_ids_offset:self._free_ids_offset+num]
        self._free_ids_offset += num

        self._free_mask[free_ids] = False
        self._num_free -= num
        return free_ids

    def recycle_blocks(self, block_ids: torch.Tensor) -> None:
        assert not self._free_mask[block_ids].any()
        self._free_mask[block_ids] = True
        self._num_free += len(block_ids)

    def _update_free_ids(self) -> None:
        self._free_ids = self._free_mask.nonzero()
        self._free_ids_offset = 0

    @property
    def num_free_blocks(self) -> int:
        return self._num_free

    @property
    def num_used_blocks(self) -> int:
        return self.num_total_blocks - self._num_free
