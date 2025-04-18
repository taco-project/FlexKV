import time
import heapq
import torch
from typing import Dict, List, Union, Tuple

from flexkv.common.block import BlockMeta, BlockStatus, SequenceMeta
from flexkv.common.hash_utils import HashType, hash_tensor
from flexkv.c_ext import get_hash_size
from flexkv.c_ext import get_prefix_block_ids
from flexkv.c_ext import get_block_ids_from_hashes
from flexkv.c_ext import index_batch_insert

class TokenToBlockIndex:
    def __init__(self, tokens_per_block: int, max_num_blocks: int = 1000000):
        self.index: Dict[HashType, int] = {}

        self.tokens_per_block = tokens_per_block

        self.leaf_blocks: Dict[HashType, int] = {}

        self.max_num_blocks = max_num_blocks

        self._available_block_ids = torch.ones(max_num_blocks, dtype=torch.bool)
        self._prev_id = torch.full((max_num_blocks,), -1, dtype=torch.int64)
        self._physical_block_id = torch.full((max_num_blocks,), -1, dtype=torch.int64)
        self._last_access_time = torch.zeros(max_num_blocks, dtype=torch.float64)
        self._lock_cnt = torch.zeros(max_num_blocks, dtype=torch.int32)
        self._child_cnt = torch.zeros(max_num_blocks, dtype=torch.int32)
        self._status = torch.zeros(max_num_blocks, dtype=torch.int32)

    def reset(self)->None:
        self.index.clear()
        self.leaf_blocks.clear()

        self._available_block_ids = torch.ones(self.max_num_blocks, dtype=torch.bool)
        self._prev_id = torch.full((self.max_num_blocks,), -1, dtype=torch.int64)
        self._physical_block_id = torch.full((self.max_num_blocks,), -1, dtype=torch.int64)
        self._last_access_time.zero_()
        self._lock_cnt.zero_()
        self._child_cnt.zero_()
        self._status.zero_()

    def match_prefix(self,
                    sequence: SequenceMeta,
                    update_cache_info: bool = True,
                    lock_blocks: bool = True) -> torch.Tensor:
        if sequence.has_hashes():
            prefix_block_ids = self._match_hashes_impl(sequence)
        else:
            prefix_block_ids = self._match_token_ids_impl(sequence)
        if update_cache_info:
            current_time = time.time()
            self._last_access_time[prefix_block_ids] = current_time
        if lock_blocks:
            self._lock_cnt[prefix_block_ids] += 1
        return prefix_block_ids

    def match_length(self,
                    sequence: SequenceMeta) -> int:
        last_index, _ = self._search_last_cached_block(sequence)
        return last_index + 1

    # TODO: optimize this
    def _match_hashes_impl(self, sequence: SequenceMeta) -> torch.Tensor:
        cached_prefix_length = self.match_length(sequence)
        prefix_block_ids = get_block_ids_from_hashes(sequence.block_hashes[:cached_prefix_length],
                                                        self.index)
        return prefix_block_ids

    def _match_token_ids_impl(self,
                              sequence_meta: SequenceMeta) -> torch.Tensor:
        last_index, last_block_id = self._search_last_cached_block(sequence_meta)
        prefix_block_ids = get_prefix_block_ids(last_index, last_block_id, self._prev_id)
        return prefix_block_ids

    def insert(self,
               sequence_meta: SequenceMeta,
               match_length: int,
               physical_block_ids: torch.Tensor) -> None:
        assert match_length >= 0 and match_length <= sequence_meta.num_blocks
        assert physical_block_ids.ndim == 1
        assert len(physical_block_ids) == sequence_meta.num_blocks - match_length

        if match_length == sequence_meta.num_blocks:
            return

        sequence_meta.gen_hashes()

        last_cached_index = match_length - 1
        first_index = match_length
        assert sequence_meta.block_hashes[first_index] not in self.index
        last_cached_block_id = -1
        if match_length > 0:
            assert sequence_meta.block_hashes[last_cached_index] in self.index
            last_cached_block_id = self.index[sequence_meta.block_hashes[last_cached_index]]
            last_cached_block_hash = sequence_meta.block_hashes[last_cached_index]
            self.leaf_blocks.pop(last_cached_block_hash, None)
            self._child_cnt[last_cached_block_id] += 1

        inserted_block_ids = physical_block_ids
        assert self._available_block_ids[inserted_block_ids].all()
        self._available_block_ids[inserted_block_ids] = False

        self._prev_id[inserted_block_ids[1:]] = inserted_block_ids[:-1]

        self._prev_id[inserted_block_ids[0]] = last_cached_block_id
        self._physical_block_id[inserted_block_ids] = physical_block_ids
        self._last_access_time[inserted_block_ids] = time.time()
        self._lock_cnt[inserted_block_ids] = 0
        self._child_cnt[inserted_block_ids[:-1]] = 1
        self._child_cnt[inserted_block_ids[-1]] = 0
        self._status[inserted_block_ids] = 0

        # self.index.update({
        #     sequence_meta.block_hashes[i]: inserted_block_ids[i - first_index]
        #     for i in range(first_index, sequence_meta.num_blocks)
        # })
        self._batch_insert(sequence_meta.block_hashes[first_index:],
                           inserted_block_ids[first_index:])

        self.leaf_blocks[sequence_meta.block_hashes[sequence_meta.num_blocks - 1]] = inserted_block_ids[-1]

    def _search_last_cached_block(self,
                                  sequence_meta: SequenceMeta
                                  ) -> Tuple[int, int]:
        left, right = 0, sequence_meta.num_blocks - 1
        last_index = -1
        last_block_id = -1
        while left <= right:
            mid = (left + right) // 2
            hash_key = None
            if sequence_meta.has_hashes():
                hash_key = sequence_meta.block_hashes[mid]
            else:
                hash_key = hash_tensor(sequence_meta.token_ids[
                    0:(mid+1)*self.tokens_per_block
                ])
            if hash_key in self.index:
                last_index = mid
                last_block_id = self.index[hash_key]
                left = mid + 1
            else:
                right = mid - 1
        return last_index, last_block_id

    def evict(self, num_evicted: int) -> List[BlockMeta]:
        candidates = torch.tensor([block_id for block_id in self.leaf_blocks.values() \
                                  if self._evicted(block_id)], dtype=torch.int32)
        evicted_block_metas = []
        heapq.heapify(candidates)
        for _ in range(num_evicted):
            if len(candidates) == 0:
                break
            block_meta = heapq.heappop(candidates)
            self.index.pop(block_meta.hash)
            self.leaf_blocks.pop(block_meta.hash)
            block_meta.status = BlockStatus.UNREGISTERED
            if block_meta.prev_hash:
                prev_block_meta = self.index[block_meta.prev_hash]
                prev_block_meta.child_cnt -= 1
                if prev_block_meta.child_cnt == 0:
                    self.leaf_blocks[prev_block_meta.hash] = prev_block_meta
                if self._evicted(prev_block_meta):
                    heapq.heappush(candidates, prev_block_meta)
            evicted_block_metas.append(block_meta)
        return evicted_block_metas

    def _evicted(self, block_meta: BlockMeta) -> bool:
        return block_meta.status == BlockStatus.AVAILABLE and \
            block_meta.child_cnt == 0

    def _batch_insert(self,
                      hashes: List[HashType],
                      block_ids: torch.Tensor) -> None:
        index_batch_insert(hashes,
                          block_ids,
                          self.index)

    @property
    def total_cached_blocks(self) -> int:
        return len(self.index)

if __name__ == "__main__":
    tokens_per_block = 2
    index = TokenToBlockIndex(tokens_per_block=tokens_per_block)
    print(f"init index, tokens_per_block = {tokens_per_block}")

    token_ids1 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    token_ids2 = torch.tensor([1, 2, 3, 4, 15, 16, 17, 18])

    seq1 = SequenceMeta(token_ids=token_ids1, tokens_per_block=tokens_per_block)
    seq2 = SequenceMeta(token_ids=token_ids2, tokens_per_block=tokens_per_block)

    index.insert(seq1, 0, torch.tensor([0, 1, 2, 3], dtype=torch.int64))
    print(f"insert seq1 = {seq1.token_ids}, "
          f"total cached blocks = {index.total_cached_blocks}")
    seq2_match_length = index.match_length(seq2)
    assert seq2_match_length == 2
    index.insert(seq2, seq2_match_length, torch.tensor([8, 9], dtype=torch.int64))
    print(f"insert seq2 = {seq2.token_ids}, "
          f"total cached blocks = {index.total_cached_blocks}")

    seq3 = SequenceMeta(token_ids=torch.tensor([1,2,3,4,0,0]),
                        tokens_per_block=tokens_per_block)
    match_result = index.match_prefix(seq3)
    print(f"match {seq3.token_ids}, num cached blocks: {len(match_result)}")

    evicted_blocks = index.evict(3)
    print(f"evict {len(evicted_blocks)} blocks, "
          f"total cached blocks = {index.total_cached_blocks}")

    match_result = index.match_prefix(seq1)
    print(f"match {seq1.token_ids}, num cached blocks: {len(match_result)}")
    match_result = index.match_prefix(seq2)
    print(f"match {seq2.token_ids}, num cached blocks: {len(match_result)}")

    evicted_blocks = index.evict(10)
    print(f"evict {len(evicted_blocks)} blocks, "
          f"total cached blocks = {index.total_cached_blocks}")

    match_result = index.match_prefix(seq1)
    print(f"match {seq1.token_ids}, num cached blocks: {len(match_result)}")
    match_result = index.match_prefix(seq2)
    print(f"match {seq2.token_ids}, num cached blocks: {len(match_result)}")
