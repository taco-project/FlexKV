# SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import heapq
import time
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch

from flexkv.common.block import SequenceMeta
from flexkv.common.hash_utils import HashType, Hasher
from flexkv.common.exceptions import LogicError
from flexkv.common.transfer import DeviceType


@dataclass
class MatchResult:
    num_ready_matched_blocks: int = 0
    num_matched_blocks: int = 0
    matched_pos: str = "local"
    last_ready_node: Optional['RadixNode'] = None
    last_node: Optional['RadixNode'] = None
    last_node_matched_length: int = 0
    physical_blocks: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    def __post_init__(self) -> None:
        assert self.physical_blocks.ndim == 1
        assert self.physical_blocks.dtype == np.int64

    def is_empty(self) -> bool:
        return self.num_matched_blocks == 0

@dataclass
class RadixNode:
    block_hashes: np.ndarray
    physical_blocks: np.ndarray

    is_ready: bool
    lock_cnt: int
    last_access_time: float

    parent: Optional['RadixNode'] = None
    children: Dict[Optional[HashType], 'RadixNode'] = field(default_factory=dict)

    def __post_init__(self) -> None:
        assert self.block_hashes.ndim == 1
        assert self.physical_blocks.ndim == 1
        assert self.block_hashes.size == self.physical_blocks.size

    def __lt__(self, other: 'RadixNode') -> bool:
        return self.last_access_time < other.last_access_time

    def size(self) -> int:
        return self.block_hashes.size

    def head_hash(self) -> HashType:
        return HashType(int(self.block_hashes[0])) if self.size() > 0 else HashType(0)

    def num_children(self) -> int:
        return len(self.children)

    def is_leaf(self) -> bool:
        return self.num_children() == 0

    def is_root(self) -> bool:
        return self.parent is None

    def evictable(self) -> bool:
        return not self.is_root() and self.is_leaf() and not self.in_use()

    def in_use(self) -> bool:
        return self.lock_cnt > 0 or not self.is_ready

    def split(self, prefix_length: int) -> 'RadixNode':
        assert prefix_length < self.size()
        assert prefix_length > 0
        assert self.parent is not None
        # assert not self.in_use()
        # new_node is parent of self
        new_node = RadixNode(
            block_hashes=self.block_hashes[:prefix_length],
            physical_blocks=self.physical_blocks[:prefix_length],
            is_ready=self.is_ready,
            lock_cnt=0,  # Note: only lock near-leaf node
            last_access_time=self.last_access_time,
        )
        self.block_hashes = self.block_hashes[prefix_length:]
        self.physical_blocks = self.physical_blocks[prefix_length:]

        self.parent.children[new_node.head_hash()] = new_node
        new_node.parent = self.parent
        self.parent = new_node
        new_node.children[self.head_hash()] = self
        return new_node

    def shrink(self, length: int) -> np.ndarray:
        assert length < self.size()
        assert length > 0
        assert self.is_leaf()
        assert not self.in_use()
        remaining_length = self.size() - length
        physical_blocks = self.physical_blocks[remaining_length:]
        self.block_hashes = self.block_hashes[:remaining_length]
        self.physical_blocks = self.physical_blocks[:remaining_length]
        return physical_blocks

    def merge_child(self) -> None:  # ignore status
        assert self.num_children() == 1
        child = list(self.children.values())[0]
        self.block_hashes = np.concatenate([self.block_hashes, child.block_hashes])
        self.physical_blocks = np.concatenate([self.physical_blocks, child.physical_blocks])
        self.last_access_time = max(self.last_access_time, child.last_access_time)
        self.children.clear()

class RadixTreeIndex:
    def __init__(self, tokens_per_block: int, max_num_blocks: int = 1000000):
        self.root_node: RadixNode = RadixNode(block_hashes=np.array([], dtype=np.int64),
                                              physical_blocks=np.array([], dtype=np.int64),
                                              is_ready=True,
                                              lock_cnt=0,
                                              last_access_time=time.time())

        self.tokens_per_block = tokens_per_block

        self.leaf_nodes: Dict[HashType, RadixNode] = {}

        self.max_num_blocks = max_num_blocks

    def reset(self) -> None:
        self.root_node = RadixNode(block_hashes=np.array([], dtype=np.int64),
                                   physical_blocks=np.array([], dtype=np.int64),
                                   is_ready=True,
                                   lock_cnt=0,
                                   last_access_time=time.time())
        self.leaf_nodes.clear()

    def is_empty(self) -> bool:
        return len(self.leaf_nodes) == 0

    def match_prefix(self,
                    sequence: SequenceMeta,
                    update_cache_info: bool = True) -> MatchResult:
        sequence.gen_hashes()
        current_node = self.root_node
        last_ready_node = self.root_node
        prefix_blocks_num = 0
        ready_prefix_blocks_num = 0
        last_node_matched_length = 0
        physical_blocks = np.array([], dtype=np.int64)
        while prefix_blocks_num < sequence.num_blocks:
            if update_cache_info:
                current_node.last_access_time = time.time()
            child_hash = sequence.get_hash(prefix_blocks_num + current_node.size())
            if child_hash in current_node.children:
                if current_node.is_ready:
                    last_ready_node = current_node
                    ready_prefix_blocks_num += current_node.size()
                prefix_blocks_num += current_node.size()
                physical_blocks = np.concatenate([physical_blocks, current_node.physical_blocks])
                current_node = current_node.children[child_hash]
            else:
                if not current_node.is_root():
                    cmp_length = min(current_node.size(), sequence.num_blocks - prefix_blocks_num)
                    left = 0
                    right = cmp_length
                    while left < right:
                        mid = (left + right) // 2
                        if current_node.block_hashes[mid] == sequence.get_hash(prefix_blocks_num+mid):
                            left = mid + 1
                        else:
                            right = mid
                    matched_length = left
                    physical_blocks = np.concatenate([physical_blocks, current_node.physical_blocks[:matched_length]])
                else:
                    matched_length = 0
                if current_node.is_ready:
                    last_ready_node = current_node
                    ready_prefix_blocks_num += matched_length
                last_node_matched_length = matched_length
                prefix_blocks_num += matched_length
                break
        return MatchResult(num_matched_blocks=prefix_blocks_num,
                           num_ready_matched_blocks=ready_prefix_blocks_num,
                           last_ready_node=last_ready_node,
                           last_node=current_node,
                           last_node_matched_length=last_node_matched_length,
                           physical_blocks=physical_blocks)

    def num_matched_blocks(self,
                    sequence: SequenceMeta) -> int:
        match_result = self.match_prefix(sequence)
        return match_result.num_matched_blocks

    def insert(self,
               sequence_meta: SequenceMeta,
               physical_block_ids: np.ndarray,
               num_insert_blocks: int = -1,
               is_ready: bool = True,
               match_result: Optional[MatchResult] = None) -> Optional[RadixNode]:
        if num_insert_blocks == -1:
            num_insert_blocks = sequence_meta.num_blocks
        assert 0 <= num_insert_blocks <= sequence_meta.num_blocks

        assert physical_block_ids.ndim == 1
        assert physical_block_ids.dtype == np.int64

        sequence_meta.gen_hashes()
        if match_result is None:
            match_result = self.match_prefix(sequence_meta)
        num_matched_blocks = match_result.num_matched_blocks
        last_node = match_result.last_node
        assert last_node is not None
        last_node_matched_length = match_result.last_node_matched_length
        assert last_node_matched_length != 0 or last_node.is_root()

        assert len(physical_block_ids) == num_insert_blocks - num_matched_blocks, \
            f"num_insert_blocks = {num_insert_blocks}, " \
            f"num_matched_blocks = {num_matched_blocks}, " \
            f"len(physical_block_ids) = {len(physical_block_ids)}"

        if num_matched_blocks >= num_insert_blocks:
            # not insert any new blocks
            return None

        new_node = RadixNode(
            block_hashes=sequence_meta.block_hashes[num_matched_blocks:num_insert_blocks],
            physical_blocks=physical_block_ids,
            is_ready=is_ready,
            lock_cnt=0,
            last_access_time=time.time()
        )

        last_node_leaf = last_node.is_leaf() and not last_node.is_root()
        if last_node_leaf:
            self.leaf_nodes.pop(last_node.head_hash(), None)

        if last_node_matched_length < last_node.size():
            last_node.split(last_node_matched_length)
            if last_node_leaf:
                self.leaf_nodes[last_node.head_hash()] = last_node
            last_node = last_node.parent
            assert last_node is not None

        new_node.parent = last_node
        last_node.children[new_node.head_hash()] = new_node
        self.leaf_nodes[new_node.head_hash()] = new_node

        return new_node

    def evict(self, num_evicted: int) -> np.ndarray:
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
                assert node.parent is not None  # node is not root
                node.parent.children.pop(node.head_hash())
                self.leaf_nodes.pop(node.head_hash(), None)
                if node.parent.is_leaf():
                    self.leaf_nodes[node.parent.head_hash()] = node.parent
                if node.parent.evictable():
                    heapq.heappush(candidates, node.parent)
                physical_blocks = node.physical_blocks
                node.parent = None
            evicted_blocks = np.concatenate([evicted_blocks, physical_blocks])
        return evicted_blocks

    def lock(self, node: RadixNode) -> None:
        if node.lock_cnt < 0:
            raise LogicError("before lock, lock_cnt < 0")
        node.lock_cnt += 1

    def unlock(self, node: RadixNode) -> None:
        if node.lock_cnt <= 0:
            raise LogicError("before unlock, lock_cnt <= 0")
        node.lock_cnt -= 1

    def set_ready(self, node: RadixNode, is_ready: bool = True, ready_length: int = -1) -> None:
        node.is_ready = is_ready
        if ready_length > 0:
            ready_length -= node.size()
            num_node = 1
            while ready_length > 0:
                if node.parent is None:
                    raise LogicError("node is None in set_ready")
                else:
                    node = node.parent
                ready_length -= node.size()
                node.is_ready = True
                num_node += 1
            assert ready_length == 0

    def total_cached_blocks(self) -> int:
        total_cached_blocks = 0
        queue = [self.root_node]
        while queue:
            node = queue.pop(0)
            total_cached_blocks += node.size()
            queue.extend(node.children.values())
        return total_cached_blocks

    def total_node_num(self) -> int:  # include root node
        total_node_num = -1  # exclude root node
        queue = [self.root_node]
        while queue:
            node = queue.pop(0)
            total_node_num += 1
            queue.extend(node.children.values())
        return total_node_num

    def total_ready_blocks(self) -> int:
        total_ready_blocks = 0
        queue = [self.root_node]
        while queue:
            node = queue.pop(0)
            if node.is_ready:
                total_ready_blocks += node.size()
            queue.extend(node.children.values())
        return total_ready_blocks

    def total_unready_blocks(self) -> int:
        return self.total_cached_blocks() - self.total_ready_blocks()

if __name__ == "__main__":
    tokens_per_block = 2
    index = RadixTreeIndex(tokens_per_block=tokens_per_block)
    print(f"init index, tokens_per_block = {tokens_per_block}")

    token_ids1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int64)
    token_ids2 = np.array([1, 2, 3, 4, 15, 16, 17, 18], dtype=np.int64)

    seq1 = SequenceMeta(token_ids=token_ids1, tokens_per_block=tokens_per_block)
    seq2 = SequenceMeta(token_ids=token_ids2, tokens_per_block=tokens_per_block)

    index.insert(seq1, np.array([0, 1, 2, 3], dtype=np.int64), is_ready=True)
    print(f"insert seq1 = {seq1.token_ids}, "
          f"total cached blocks = {index.total_cached_blocks()}")
    seq2_matched_blocks = index.num_matched_blocks(seq2)
    assert seq2_matched_blocks == 2
    index.insert(seq2, np.array([8, 9], dtype=np.int64), is_ready=True)
    print(f"insert seq2 = {seq2.token_ids}, "
          f"total cached blocks = {index.total_cached_blocks()}")

    seq3 = SequenceMeta(token_ids=np.array([1,2,3,4,0,0], dtype=np.int64),
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
