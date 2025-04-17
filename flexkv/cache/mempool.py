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
        self.marks = torch.zeros(self.num_total_blocks, dtype=torch.bool)

    def allocate_blocks(self, num: int) -> torch.Tensor:
        assert num < self.num_total_blocks, "Exceed max num of blocks"
        assert num <= self.num_free_blocks, "Not enough free blocks"
        ids = []
        for _ in range(num):
            free_id = self.free_ids.popleft()
            assert not self.marks[free_id]
            self.marks[free_id] = True
            ids.append(free_id)
        return torch.tensor(ids, dtype=torch.int32)

    def free_blocks(self, block_ids: torch.Tensor) -> None:
        for block_id in block_ids:
            assert block_id < self.num_total_blocks and block_id >= 0
            assert self.marks[block_id]
            self.marks[block_id] = False
        self.free_ids.extend(block_ids.tolist())

    def reset(self) -> None:
        self.free_ids.clear()
        self.marks.zero_()

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_ids)

    @property
    def num_used_blocks(self) -> int:
        return self.num_total_blocks - self.num_free_blocks
