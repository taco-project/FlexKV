import heapq
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch

from flexkv.c_ext import (Hasher, find_n_liner_parents_for_eviction,
                          get_block_ids_from_hashes, get_hash_size,
                          get_prefix_block_ids, index_batch_insert,
                          index_batch_remove)
from flexkv.common.block import BlockMeta, BlockStatus, SequenceMeta
from flexkv.common.hash_utils import HashType, hash_tensor


@dataclass
class EvictionCandidate:
    hash: HashType
    block_id: int
    last_access_time: float

    def __lt__(self, other: 'EvictionCandidate') -> bool:
        return self.last_access_time < other.last_access_time

class TokenToBlockIndex:
    def __init__(self, tokens_per_block: int, max_num_blocks: int = 1000000):
        self.index: Dict[HashType, int] = {}

        self.tokens_per_block = tokens_per_block

        self.leaf_blocks: Dict[HashType, int] = {}

        self.max_num_blocks = max_num_blocks

        self.reset()

    def reset(self)->None:
        self.index.clear()
        self.leaf_blocks.clear()

        max_num_blocks = self.max_num_blocks
        self._available_block_ids = torch.ones(max_num_blocks, dtype=torch.bool)
        self._prev_id = torch.full((max_num_blocks,), -1, dtype=torch.int64)
        self._last_access_time = torch.zeros(max_num_blocks, dtype=torch.float64)
        self._lock_cnt = torch.zeros(max_num_blocks, dtype=torch.int32)
        self._child_cnt = torch.zeros(max_num_blocks, dtype=torch.int32)
        self._ready = torch.zeros(max_num_blocks, dtype=torch.bool)
        self._hash_values = torch.zeros((max_num_blocks, get_hash_size()), dtype=torch.uint8)

    def match_prefix(self,
                    sequence: SequenceMeta,
                    update_cache_info: bool = True) -> torch.Tensor:
        if sequence.has_hashes():
            prefix_block_ids = self._match_hashes_impl(sequence)
        else:
            prefix_block_ids = self._match_token_ids_impl(sequence)
        ready_mask = self._ready[prefix_block_ids]
        ready_block_ids = prefix_block_ids[ready_mask]

        if update_cache_info:
            current_time = time.time()
            self._last_access_time[ready_block_ids] = current_time
        return ready_block_ids

    def match_length(self,
                    sequence: SequenceMeta) -> int:
        """get the length of the longest prefix that is cached"""
        last_index, _ = self._search_last_cached_block(sequence)
        return last_index + 1

    # TODO: optimize this
    def _match_hashes_impl(self,
                           sequence: SequenceMeta) -> torch.Tensor:
        """match the prefix of the sequence with the cached hashes"""
        cached_prefix_length = self.match_length(sequence)
        prefix_block_ids = get_block_ids_from_hashes(sequence.block_hashes[:cached_prefix_length],
                                                        self.index)
        return prefix_block_ids

    def _match_token_ids_impl(self,
                              sequence_meta: SequenceMeta) -> torch.Tensor:
        """match the prefix of the sequence with the cached token ids"""
        last_index, last_block_id = self._search_last_cached_block(sequence_meta)
        prefix_block_ids = get_prefix_block_ids(last_index, last_block_id, self._prev_id)
        return prefix_block_ids

    def insert(self,
               sequence_meta: SequenceMeta,
               physical_block_ids: torch.Tensor,
               match_length: int,
               insert_length: int = -1,
               is_ready: bool = True,
               locked: bool = False) -> None:
        if insert_length == -1:
            insert_length = sequence_meta.num_blocks
        assert 0 <= match_length <= sequence_meta.num_blocks
        assert 0 <= insert_length <= sequence_meta.num_blocks
        assert match_length <= insert_length

        assert physical_block_ids.ndim == 1
        assert len(physical_block_ids) == insert_length - match_length

        if match_length == insert_length:
            return

        sequence_meta.gen_hashes()

        last_cached_index = match_length - 1
        first_index = match_length
        assert bytes(sequence_meta.block_hashes[first_index]) not in self.index
        last_cached_block_id = -1
        if match_length > 0:
            assert bytes(sequence_meta.block_hashes[last_cached_index]) in self.index
            last_cached_block_id = self.index[bytes(sequence_meta.block_hashes[last_cached_index])]
            last_cached_block_hash = bytes(sequence_meta.block_hashes[last_cached_index])
            self.leaf_blocks.pop(last_cached_block_hash, None)
            self._child_cnt[last_cached_block_id] += 1

        inserted_block_ids = physical_block_ids
        #TODO maintain the available block ids
        assert self._available_block_ids[inserted_block_ids].all()
        self._available_block_ids[inserted_block_ids] = False

        self._prev_id[inserted_block_ids[1:]] = inserted_block_ids[:-1]
        self._prev_id[inserted_block_ids[0]] = last_cached_block_id

        self._last_access_time[inserted_block_ids] = time.time()
        self._lock_cnt[inserted_block_ids] = locked
        self._ready[inserted_block_ids] = is_ready
        self._child_cnt[inserted_block_ids[:-1]] = 1
        self._child_cnt[inserted_block_ids[-1]] = 0

        self._hash_values[inserted_block_ids] = sequence_meta.block_hashes[first_index:
                                                                           first_index+insert_length-match_length]

        self._batch_insert(sequence_meta.block_hashes[first_index:first_index+insert_length-match_length],
                           inserted_block_ids)
        self.leaf_blocks[bytes(sequence_meta.block_hashes[insert_length-1])] = inserted_block_ids[-1].item()

    def _search_last_cached_block(self,
                                  sequence_meta: SequenceMeta) -> Tuple[int, int]:
        left, right = 0, sequence_meta.num_blocks - 1
        last_index = -1
        last_block_id = -1
        hasher = Hasher()
        hash_res = torch.zeros(get_hash_size(), dtype=torch.uint8)
        hash_start_pos = 0
        while left <= right:
            mid = (left + right) // 2
            hash_key = None
            if sequence_meta.has_hashes():
                hash_key = bytes(sequence_meta.block_hashes[mid])
            else:
                hasher.hash_out(sequence_meta.token_ids[
                    hash_start_pos*self.tokens_per_block:(mid+1)*self.tokens_per_block
                ], hash_res)
                hash_key = bytes(hash_res)
            if hash_key in self.index:
                last_index = mid
                last_block_id = self.index[hash_key]
                left = mid + 1
                hash_start_pos = mid + 1
            else:
                right = mid - 1
                hasher.reset()
                hash_start_pos = 0
        return last_index, last_block_id

    def evict(self, num_evicted: int) -> torch.Tensor:
        leaves = self.leaf_blocks.items()
        _last_access_time = self._last_access_time.numpy()
        _lock_cnt = self._lock_cnt.numpy()
        _ready = self._ready.numpy()
        candidates = [
            EvictionCandidate(block_hash, block_id, _last_access_time[block_id])
            for block_hash, block_id in leaves
            if _lock_cnt[block_id] == 0 and _ready[block_id] == 1
        ]
        heapq.heapify(candidates)
        results = torch.tensor([], dtype=torch.int64)
        evicted_hashes = None
        while len(results) < num_evicted:
            if len(candidates) == 0:
                break
            candidate = heapq.heappop(candidates)
            candidate_id = candidate.block_id
            # update the leaf blocks, although in this evcition,
            # we don't need to access the leaf blocks
            assert candidate.hash in self.leaf_blocks, \
                f"Candidate {candidate.hash} not in leaf blocks"
            self.leaf_blocks.pop(candidate.hash)
            # get a chain of parent blocks of the selected block, so that
            # we can evict with larger granularity, and keep the cache
            # more likely to hit
            # preallocate the memory for the evicted block ids
            max_last_access_time = float('inf')
            if len(candidates) > 0:
                max_last_access_time = candidates[0].last_access_time
            evicted_block_ids = find_n_liner_parents_for_eviction(candidate_id,
                                                                self._prev_id,
                                                                self._lock_cnt,
                                                                self._child_cnt,
                                                                self._ready,
                                                                self._last_access_time,
                                                                max_last_access_time,
                                                                num_evicted - len(results))
            # deal with the parent block of the evicted block
            parent_block_id = self._prev_id[evicted_block_ids[-1]].item()
            if parent_block_id != -1: # not the root block
                self._child_cnt[parent_block_id] -= 1
                if (self._child_cnt[parent_block_id] == 0 and
                    self._ready[parent_block_id] and
                    self._lock_cnt[parent_block_id] == 0):
                    # not locked and available, add as new leaf and new candidate
                    hash_key = bytes(self._hash_values[parent_block_id])
                    self.leaf_blocks[hash_key] = parent_block_id
                    heapq.heappush(candidates, EvictionCandidate(hash_key,
                                                                 parent_block_id,
                                                                 self._last_access_time[parent_block_id]))
            if evicted_hashes is None:
                evicted_hashes = self._hash_values[evicted_block_ids]
                results = evicted_block_ids
            else:
                evicted_hashes = torch.cat([evicted_hashes, self._hash_values[evicted_block_ids]])
                results = torch.cat([results, evicted_block_ids])
        self._batch_remove(evicted_hashes)
        self._available_block_ids[results] = True
        return results

    def lock(self, block_ids: torch.Tensor) -> None:
        self._lock_cnt[block_ids] += 1
        assert (self._lock_cnt[block_ids] > 0).all()

    def unlock(self, block_ids: torch.Tensor) -> None:
        self._lock_cnt[block_ids] -= 1
        assert (self._lock_cnt[block_ids] >= 0).all()

    def set_ready(self, block_ids: torch.Tensor, is_ready: bool = True) -> None:
        self._ready[block_ids] = is_ready

    def _batch_insert(self,
                      hashes: torch.Tensor,
                      block_ids: torch.Tensor) -> None:
        index_batch_insert(hashes,
                          block_ids,
                          self.index)

    def _batch_remove(self, hashes: torch.Tensor) -> None:
        index_batch_remove(hashes,
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

    index.insert(seq1, 0, torch.tensor([0, 1, 2, 3], dtype=torch.int64), is_ready=True)
    print(f"insert seq1 = {seq1.token_ids}, "
          f"total cached blocks = {index.total_cached_blocks}")
    seq2_match_length = index.match_length(seq2)
    assert seq2_match_length == 2
    index.insert(seq2, seq2_match_length, torch.tensor([8, 9], dtype=torch.int64), is_ready=True)
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
