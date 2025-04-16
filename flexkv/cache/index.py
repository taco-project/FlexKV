import time
import heapq
import torch
from typing import Dict, List, Union, Tuple

from flexkv.common.block import BlockMeta, BlockStatus, SequenceMeta
from flexkv.common.hash_utils import HashType, hash_tensor

class TokenToBlockIndex:
    def __init__(self, tokens_per_block: int):
        self.index: Dict[HashType, BlockMeta] = {}

        self.tokens_per_block = tokens_per_block

        self.leaf_blocks: Dict[HashType, List[BlockMeta]] = {}

    def reset(self)->None:
        self.index.clear()
        self.leaf_blocks.clear()

    def match_prefix(self,
                    sequence: SequenceMeta,
                    update_cache_info: bool = True,
                    lock_blocks: bool = True) -> List[BlockMeta]:
        if sequence.has_hashes:
            prefix_blocks = self._match_hashes_impl(sequence)
        else:
            prefix_blocks = self._match_token_ids_impl(sequence)

        if update_cache_info:
            current_time = time.time()
            for block_meta in prefix_blocks:
                block_meta.last_access_time = current_time
        if lock_blocks:
            for block_meta in prefix_blocks:
                block_meta.lock_cnt += 1
        return prefix_blocks

    def match_length(self,
                    sequence: SequenceMeta) -> int:
        last_block_index, _ = self._search_last_cached_block(sequence)
        return last_block_index + 1

    def _match_hashes_impl(self, sequence: SequenceMeta) -> List[BlockMeta]:
        prefix_block_metas = []
        for hash in sequence.block_hashes:
            if hash in self.index:
                prefix_block_metas.append(self.index[hash])
        return prefix_block_metas

    def _match_token_ids_impl(self,
                              sequence_meta: SequenceMeta) -> List[BlockMeta]:
        _, last_block_hash = self._search_last_cached_block(sequence_meta)
        prefix_block_metas = []
        while last_block_hash is not None:
            prefix_block_metas.append(self.index[last_block_hash])
            last_block_hash = self.index[last_block_hash].prev_hash
        return prefix_block_metas[::-1]

    def insert(self, sequence_meta: SequenceMeta)->None:
        sequence_meta.gen_hashes()

        prefix_index, _ = self._search_last_cached_block(sequence_meta)

        for i in range(prefix_index + 1, sequence_meta.num_blocks):
            block_hash = sequence_meta.block_hashes[i]
            if block_hash in self.index:
                continue
            prev_hash = sequence_meta.block_hashes[i-1] if i > 0 else None
            # TODO: get physical block id from params
            physical_block_id = -1
            block_meta = BlockMeta(
                hash=block_hash,
                prev_hash=prev_hash,
                physical_block_id=physical_block_id,
                status=BlockStatus.AVAILABLE,
                child_cnt = 1
            )
            if i == len(sequence_meta.block_hashes) - 1:
                block_meta.child_cnt = 0
                self.leaf_blocks[block_hash] = block_meta
            self.index[block_hash] = block_meta

        if prefix_index >= 0 and \
            prefix_index < len(sequence_meta.block_hashes) - 1:
            prefix_hash = sequence_meta.block_hashes[prefix_index]
            self.leaf_blocks.pop(prefix_hash, None)
            self.index[prefix_hash].child_cnt += 1

    def _search_last_cached_block(self,
                                  sequence_meta: SequenceMeta
                                  ) -> Tuple[int, HashType]:
        left, right = 0, sequence_meta.num_blocks - 1
        last_block_index = -1
        last_block_hash = None
        while left <= right:
            mid = (left + right) // 2
            hash_key = None
            if sequence_meta.has_hashes:
                hash_key = sequence_meta.block_hashes[mid]
            else:
                hash_key = hash_tensor(sequence_meta.token_ids[
                    0:(mid+1)*self.tokens_per_block
                ])
            if hash_key in self.index:
                last_block_index = mid
                last_block_hash = hash_key
                left = mid + 1
            else:
                right = mid - 1
        return last_block_index, last_block_hash

    def evict(self, num_evicted: int) -> List[BlockMeta]:
        candidates = [block_meta for block_meta in self.leaf_blocks.values() \
                      if self._evicted(block_meta)]
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

    index.insert(seq1)
    print(f"insert seq1 = {seq1.token_ids}, "
          f"total cached blocks = {index.total_cached_blocks}")
    index.insert(seq2)
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
