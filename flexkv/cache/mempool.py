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
        self.free_ids = deque(range(self.num_total_blocks))
        self.usage_mark = torch.zeros(self.num_total_blocks, dtype=torch.bool)

    def allocate_blocks(self, num: int) -> List[int]:
        if num > self.num_total_blocks:
            raise ValueError("Exceed max num of blocks")
        if self.num_free_blocks < num:
            raise ValueError("Not enough free blocks")
        ids = []
        for _ in range(num):
            free_id = self.free_ids.popleft()
            assert not self.usage_mark[free_id]
            self.usage_mark[free_id] = True
            ids.append(free_id)
        return ids

    def free_blocks(self, block_ids: List[int]) -> None:
        for block_id in block_ids:
            assert block_id < self.num_total_blocks and block_id >= 0
            assert self.usage_mark[block_id]
            self.usage_mark[block_id] = False
        self.free_ids.extend(block_ids)

    def reset(self) -> None:
        self.free_ids.clear()
        self.usage_mark.zero_()

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_ids)

    @property
    def num_used_blocks(self) -> int:
        return self.num_total_blocks - self.num_free_blocks
