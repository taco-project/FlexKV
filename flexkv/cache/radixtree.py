import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union, Optional

import torch
import numpy as np

from flexkv.c_ext import (find_n_liner_parents_for_eviction,
                          get_block_ids_from_hashes, get_prefix_block_ids)
from flexkv.common.block import SequenceMeta
from flexkv.common.hash_utils import HashType, Hasher


@dataclass
class MatchResult:
    num_matched_blocks: int = 0
    last_node: Optional['RadixNode'] = None
    last_node_matched_length: int = 0
    physical_blocks: torch.Tensor = torch.empty(0, dtype=torch.int64)

@dataclass
class RadixNode:
    content_hash: np.ndarray
    content_physical: np.ndarray

    is_ready: bool
    lock_cnt: int
    last_access_time: float

    parent: Optional['RadixNode'] = None
    children: Dict[HashType, 'RadixNode'] = field(default_factory=dict)

    def __post_init__(self):
        assert self.content_hash.ndim == 1
        assert self.content_physical.ndim == 1
        assert self.content_hash.size == self.content_physical.size

    def __lt__(self, other: 'RadixNode') -> bool:
        return self.last_access_time < other.last_access_time

    def size(self) -> int:
        return self.content_hash.size

    def head_hash(self) -> HashType:
        return self.content_hash[0] if self.size() > 0 else None

    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def is_root(self) -> bool:
        return self.parent is None

    def evictable(self) -> bool:
        return not self.is_root() and self.is_leaf() and not self.in_use()

    def in_use(self) -> bool:
        return self.lock_cnt > 0

    def split(self, length: int) -> 'RadixNode':
        assert length < self.size()
        assert length > 0
        assert not self.in_use()
        new_node = RadixNode(
            content_hash=self.content_hash[length:],
            content_physical=self.content_physical[length:],
            is_ready=self.is_ready,
            lock_cnt=self.lock_cnt,
            last_access_time=self.last_access_time,
            parent=self,
            children=self.children
        )
        self.content_hash = self.content_hash[:length]
        self.content_physical = self.content_physical[:length]
        self.children = {}
        self.children[new_node.head_hash()] = new_node
        return new_node

    def shrink(self, length: int) -> np.ndarray:
        assert length < self.size()
        assert length > 0
        assert self.is_leaf()
        assert not self.in_use()
        physical_blocks = self.content_physical[length:]
        self.content_hash = self.content_hash[:length]
        self.content_physical = self.content_physical[:length]
        return physical_blocks

class RadixTreeIndex:
    def __init__(self, tokens_per_block: int, max_num_blocks: int = 1000000):
        self.root_node = RadixNode(content_hash=np.array([]),
                                   content_physical=np.array([]),
                                   is_ready=True,
                                   lock_cnt=0,
                                   last_access_time=time.time())

        self.tokens_per_block = tokens_per_block

        self.leaf_nodes: Dict[HashType, RadixNode] = {}

        self.max_num_blocks = max_num_blocks

        self.reset()

    def reset(self)->None:
        self.root_node = RadixNode(content_hash=np.array([]),
                                   content_physical=np.array([]),
                                   is_ready=True,
                                   lock_cnt=0,
                                   last_access_time=time.time())
        self.leaf_nodes.clear()

    def match_prefix(self,
                    sequence: SequenceMeta,
                    update_cache_info: bool = True) -> MatchResult:
        sequence.gen_hashes()
        current_node = self.root_node
        prefix_blocks_num = 0
        last_node_matched_length = 0
        physical_blocks = np.array([], dtype=np.int64)
        while prefix_blocks_num < sequence.num_blocks:
            if update_cache_info:
                current_node.last_access_time = time.time()
            child_hash = sequence.get_hash(prefix_blocks_num + current_node.size())  # exceed
            if child_hash in current_node.children:
                prefix_blocks_num += current_node.size()
                physical_blocks = np.concatenate([physical_blocks, current_node.content_physical])
                current_node = current_node.children[child_hash]
            else:
                if not current_node.is_root():
                    cmp_length = min(current_node.size(), sequence.num_blocks - prefix_blocks_num)
                    node_hash = current_node.content_hash[:cmp_length]
                    seq_hash = sequence.block_hashes[prefix_blocks_num:prefix_blocks_num+cmp_length].numpy()
                    matched_length = (node_hash == seq_hash).sum().item()
                    physical_blocks = np.concatenate([physical_blocks, current_node.content_physical[:matched_length]])
                else:
                    matched_length = 0
                last_node_matched_length = matched_length
                prefix_blocks_num += matched_length
                break
        return MatchResult(num_matched_blocks=prefix_blocks_num,
                           last_node=current_node,
                           last_node_matched_length=last_node_matched_length,
                           physical_blocks=torch.as_tensor(physical_blocks, dtype=torch.int64))

    def num_matched_blocks(self,
                    sequence: SequenceMeta) -> int:
        match_result = self.match_prefix(sequence)
        return match_result.num_matched_blocks

    def insert(self,
               sequence_meta: SequenceMeta,
               physical_block_ids: torch.Tensor,
               num_insert_blocks: int = -1,
               is_ready: bool = True) -> RadixNode:
        if num_insert_blocks == -1:
            num_insert_blocks = sequence_meta.num_blocks
        assert 0 <= num_insert_blocks <= sequence_meta.num_blocks

        assert physical_block_ids.ndim == 1

        sequence_meta.gen_hashes()
        match_result = self.match_prefix(sequence_meta)
        num_matched_blocks = match_result.num_matched_blocks
        last_node = match_result.last_node
        last_node_matched_length = match_result.last_node_matched_length

        assert len(physical_block_ids) == num_insert_blocks - num_matched_blocks, \
            f"num_insert_blocks = {num_insert_blocks}, " \
            f"num_matched_blocks = {num_matched_blocks}, " \
            f"len(physical_block_ids) = {len(physical_block_ids)}"

        if num_matched_blocks >= num_insert_blocks:
            return

        new_node = RadixNode(
            content_hash=sequence_meta.block_hashes[num_matched_blocks:num_insert_blocks].numpy(),
            content_physical=physical_block_ids.numpy(),
            is_ready=is_ready,
            lock_cnt=0,
            last_access_time=time.time()
        )
        self.leaf_nodes[new_node.head_hash()] = new_node
        if last_node.head_hash() in self.leaf_nodes:
            self.leaf_nodes.pop(last_node.head_hash())

        if last_node_matched_length != last_node.size() and last_node_matched_length != 0:
            splited_node = last_node.split(last_node_matched_length)
            if splited_node.is_leaf():
                self.leaf_nodes[splited_node.head_hash()] = splited_node

        new_node.parent = last_node
        last_node.children[new_node.head_hash()] = new_node
        return new_node

    def evict(self, num_evicted: int) -> torch.Tensor:
        candidates = []
        for node in self.leaf_nodes.values():
            if node.evictable():
                candidates.append(node)
        heapq.heapify(candidates)
        evicted_blocks = np.array([], dtype=np.int64)
        while len(evicted_blocks) < num_evicted and candidates:
            node = heapq.heappop(candidates)
            if node.size() > num_evicted - len(evicted_blocks):
                physical_blocks = node.shrink(num_evicted - len(evicted_blocks))
            else:
                node.parent.children.pop(node.head_hash())
                self.leaf_nodes.pop(node.head_hash(), None)
                if node.parent.is_leaf():
                    self.leaf_nodes[node.parent.head_hash()] = node.parent
                if node.parent.evictable():
                    heapq.heappush(candidates, node.parent)
                physical_blocks = node.content_physical
                del node
            evicted_blocks = np.concatenate([evicted_blocks, physical_blocks])
        return torch.as_tensor(evicted_blocks, dtype=torch.int64)

    def lock(self, node: RadixNode) -> None:
        node.lock_cnt += 1
        assert node.lock_cnt > 0

    def unlock(self, node: RadixNode) -> None:
        node.lock_cnt -= 1
        assert node.lock_cnt >= 0

    def set_ready(self, node: RadixNode, is_ready: bool = True) -> None:
        node.is_ready = is_ready

    def total_cached_blocks(self) -> int:
        total_cached_blocks = 0
        queue = [self.root_node]
        while queue:
            node = queue.pop(0)
            total_cached_blocks += node.size()
            queue.extend(node.children.values())
        return total_cached_blocks

if __name__ == "__main__":
    tokens_per_block = 2
    index = RadixTreeIndex(tokens_per_block=tokens_per_block)
    print(f"init index, tokens_per_block = {tokens_per_block}")

    token_ids1 = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8])
    token_ids2 = torch.tensor([1, 2, 3, 4, 15, 16, 17, 18])

    seq1 = SequenceMeta(token_ids=token_ids1, tokens_per_block=tokens_per_block)
    seq2 = SequenceMeta(token_ids=token_ids2, tokens_per_block=tokens_per_block)

    index.insert(seq1, torch.tensor([0, 1, 2, 3], dtype=torch.int64), is_ready=True)
    print(f"insert seq1 = {seq1.token_ids}, "
          f"total cached blocks = {index.total_cached_blocks()}")
    seq2_matched_blocks = index.num_matched_blocks(seq2)
    assert seq2_matched_blocks == 2
    index.insert(seq2, torch.tensor([8, 9], dtype=torch.int64), num_matched_blocks=2, is_ready=True)
    print(f"insert seq2 = {seq2.token_ids}, "
          f"total cached blocks = {index.total_cached_blocks()}")

    seq3 = SequenceMeta(token_ids=torch.tensor([1,2,3,4,0,0]),
                        tokens_per_block=tokens_per_block)
    match_result = index.num_matched_blocks(seq3)
    print(f"match {seq3.token_ids}, num cached blocks: {match_result}")

    evicted_blocks = index.evict(3)
    print(f"evict {len(evicted_blocks)} blocks, "
          f"total cached blocks = {index.total_cached_blocks()}")

    match_result = index.num_matched_blocks(seq1)
    print(f"match {seq1.token_ids}, num cached blocks: {match_result}")
    match_result = index.num_matched_blocks(seq2)
    print(f"match {seq2.token_ids}, num cached blocks: {match_result}")

    evicted_blocks = index.evict(10)
    print(f"evict {len(evicted_blocks)} blocks, "
          f"total cached blocks = {index.total_cached_blocks()}")

    match_result = index.num_matched_blocks(seq1)
    print(f"match {seq1.token_ids}, num cached blocks: {match_result}")
    match_result = index.num_matched_blocks(seq2)
    print(f"match {seq2.token_ids}, num cached blocks: {match_result}")
