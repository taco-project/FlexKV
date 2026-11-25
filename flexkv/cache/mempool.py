from dataclasses import dataclass
from operator import index
from typing import Optional, Sequence, Tuple

import numpy as np


@dataclass
class _FreeRegion:
    start: int
    end: int
    node: int
    free_ids: np.ndarray
    offset: int
    num_free: int


class Mempool:
    def __init__(
        self,
        num_total_blocks: int,
        numa_block_ranges: Optional[Sequence[Tuple[int, int, int]]] = None,
    ):
        assert num_total_blocks > 0
        self.num_total_blocks = num_total_blocks
        self._regions: list[_FreeRegion] = []
        if numa_block_ranges is not None and len(numa_block_ranges) >= 2:
            try:
                ranges = sorted((index(lo), index(hi), index(node))
                                for lo, hi, node in numa_block_ranges)
            except (TypeError, ValueError) as exc:
                raise ValueError("NUMA block ranges must contain integer start, end and node") from exc
            start = 0
            for lo, hi, node in ranges:
                if lo != start or not lo < hi <= num_total_blocks or node < 0:
                    raise ValueError("NUMA block ranges must partition the mempool")
                self._regions.append(_FreeRegion(lo, hi, node, np.empty(0, dtype=np.int64), 0, 0))
                start = hi
            if start != num_total_blocks:
                raise ValueError("NUMA block ranges must partition the mempool")
            self._region_ends = np.array([region.end for region in self._regions], dtype=np.int64)

        self._free_mask = np.empty(num_total_blocks, dtype=np.bool_)
        self.reset()

    def reset(self) -> None:
        self._free_mask.fill(True)
        self._num_free = self.num_total_blocks
        self._rr_cursor = 0
        if self._regions:
            for region in self._regions:
                region.free_ids = np.arange(region.start, region.end, dtype=np.int64)
                region.offset = 0
                region.num_free = region.end - region.start
        else:
            self._update_free_ids()

    def allocate_blocks(self, num: int, preferred_node: Optional[int] = None) -> np.ndarray:
        if num < 0:
            raise ValueError(f"num must be greater than 0, but got {num}")
        if num > self._num_free:
            raise ValueError(f"Not enough free blocks, required: {num}, available: {self._num_free}")

        if self._regions:
            free_ids = self._allocate_numa_blocks(num, preferred_node)
        else:
            if num > len(self._free_ids) - self._free_ids_offset:
                self._update_free_ids()
            free_ids = self._free_ids[self._free_ids_offset:self._free_ids_offset + num]
            self._free_ids_offset += num

        self._free_mask[free_ids] = False
        self._num_free -= num
        return free_ids

    def _allocate_numa_blocks(self, num: int, preferred_node: Optional[int]) -> np.ndarray:
        if num == 0:
            return np.empty(0, dtype=np.int64)

        cursor = self._rr_cursor
        regions = self._regions[cursor:] + self._regions[:cursor]
        preferred = [region for region in regions if region.node == preferred_node]
        if preferred:
            # Fill the preferred node before putting any blocks on other nodes.
            regions = preferred + [region for region in regions if region.node != preferred_node]
            counts = []
            remaining = num
            for region in regions:
                count = min(remaining, region.num_free)
                counts.append(count)
                remaining -= count
            advance_cursor = sum(counts[:len(preferred)]) < num
        else:
            counts = self._balanced_counts(num, regions)
            advance_cursor = True

        allocations = [(region, count) for region, count in zip(regions, counts, strict=True) if count]
        parts = []
        for region, count in allocations:
            if count > len(region.free_ids) - region.offset:
                region.free_ids = self._free_mask[region.start:region.end].nonzero()[0] + region.start
                region.offset = 0
            parts.append(region.free_ids[region.offset:region.offset + count])
        free_ids = parts[0] if len(parts) == 1 else np.concatenate(parts)
        for region, count in allocations:
            region.offset += count
            region.num_free -= count
        if advance_cursor:
            self._rr_cursor = (cursor + 1) % len(regions)
        return free_ids

    @staticmethod
    def _balanced_counts(num: int, regions: Sequence[_FreeRegion]) -> list[int]:
        """Find an equal share after removing regions that cannot supply it."""
        remaining, active = num, len(regions)
        for capacity in sorted(region.num_free for region in regions):
            if capacity * active >= remaining:
                break
            remaining -= capacity
            active -= 1
        share, extra = divmod(remaining, active)
        counts = []
        for region in regions:
            count = min(region.num_free, share)
            if extra and region.num_free > share:
                count += 1
                extra -= 1
            counts.append(count)
        return counts

    def recycle_blocks(self, block_ids: np.ndarray) -> None:
        if block_ids.ndim != 1 or block_ids.dtype != np.int64:
            raise ValueError("block_ids must be a 1D tensor of int64")
        if len(block_ids) == 0:
            return
        if np.any(block_ids < 0) or np.any(block_ids >= self.num_total_blocks):
            raise ValueError("block_ids must be within the range of [0, num_total_blocks)")
        block_ids = np.unique(block_ids)

        already_free = self._free_mask[block_ids]
        if already_free.any():
            raise ValueError(f"block_ids {block_ids[already_free]} are already free")
        self._free_mask[block_ids] = True
        self._num_free += len(block_ids)
        if self._regions:
            offsets = np.searchsorted(block_ids, self._region_ends)
            for region, count in zip(self._regions, np.diff(offsets, prepend=0), strict=True):
                region.num_free += int(count)

    def _update_free_ids(self) -> None:
        self._free_ids = self._free_mask.nonzero()[0]
        self._free_ids_offset = 0

    @property
    def num_free_blocks(self) -> int:
        return self._num_free

    @property
    def num_used_blocks(self) -> int:
        return self.num_total_blocks - self._num_free
