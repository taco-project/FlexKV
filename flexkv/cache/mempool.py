from collections import deque
from typing import List, Tuple

import numpy as np

class Mempool:
    def __init__(
        self,
        num_total_blocks: int,
        block_dedup: bool = False
    ):
        assert num_total_blocks > 0
        self.num_total_blocks = num_total_blocks
        self.block_dedup = block_dedup
        self.block_dedup_count = 0

        self._ref_counts = np.zeros(self.num_total_blocks, dtype=np.int32)
        self._num_free = num_total_blocks
        self._free_ids = np.where(self._ref_counts == 0)[0]
        self._free_ids_offset = 0
        if block_dedup:
            self._hash_id_map = dict()
            self._block_hashes = np.zeros(self.num_total_blocks, dtype=np.int64)

    def reset(self) -> None:
        self._ref_counts.fill(0)
        self._num_free = self.num_total_blocks
        self._free_ids = np.where(self._ref_counts == 0)[0]
        self._free_ids_offset = 0
        if self.block_dedup:
            self._hash_id_map.clear()
            self._block_hashes.fill(0)

    def allocate_blocks(self, num: int, block_hashes: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        if num < 0:
            raise ValueError(f"num must be greater than 0, but got {num}")
        if num > self._num_free:
            raise ValueError("Not enough free blocks, required:{num}, available:{self._num_free}")

        if num > len(self._free_ids) - self._free_ids_offset:
            self._update_free_ids()

        if self.block_dedup and block_hashes is not None:
            bidx = 0
            for hkey in block_hashes:
                bnum = self._hash_id_map.get(hkey, -1)
                if bnum == -1:
                    bnum = self.__free_ids[self.__free_ids_offset]
                    free_ids[bidx] = bnum
                    free_ref_counts[bidx] = 1
                    self._free_ids_offset += 1
                    self._num_free -= 1
                    self._ref_counts[bnum] = 1
                    self._block_hashes[bnum] = hkey
                    self._hash_id_map[heky] = bnum
                    self.block_dedup_count += 1
                else:
                    self._ref_counts[bnum] += 1
                    free_ids[bidx] = bnum
                    free_ref_counts[bidx] = self._ref_counts[bnum]
                bidx += 1
            return free_ids, free_ref_counts
        else:
            free_ids = self._free_ids[self._free_ids_offset:self._free_ids_offset+num]
            self._free_ids_offset += num

            self._ref_counts[free_ids] = 1
            self._num_free -= num
            return free_ids, None

    def recycle_blocks(self, block_ids: np.ndarray) -> None:
        if block_ids.ndim != 1 or block_ids.dtype != np.int64:
            raise ValueError("block_ids must be a 1D tensor of int64")
        if np.any(self._ref_counts[block_ids] == 0):
            free_ids = block_ids[self._ref_counts[block_ids]]
            raise ValueError(f"Cannot recycle free block_ids repeatedly: {free_ids}")

        self._ref_counts[block_ids] -= 1
        if self.block_dedup:
            free_ids = block_ids[np.where(self._ref_counts[block_ids] == 0)[0]]
            self._num_free += len(free_ids)
            for hkey in self._block_hashes[free_ids]:
                self._hash_id_map.pop(hkey, None)
        else:
            self._num_free += len(block_ids)

    def _update_free_ids(self) -> None:
        self._free_ids = np.where(self._ref_counts == 0)[0]
        self._free_ids_offset = 0

    @property
    def num_free_blocks(self) -> int:
        return self._num_free

    @property
    def num_used_blocks(self) -> int:
        return self.num_total_blocks - self._num_free
