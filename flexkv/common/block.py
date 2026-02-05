from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, NewType, Optional

import numpy as np
import torch

from flexkv.common.hash_utils import HashType, Hasher, gen_hashes, get_hash_size


def _get_namespace_hash_key(namespace: Optional[List[str]]) -> Optional[np.ndarray]:
    if not namespace or len(namespace) == 0:
        return None
    
    namespace_key = ":".join(namespace)
    namespace_bytes = namespace_key.encode('utf-8')
    namespace_array = np.frombuffer(namespace_bytes, dtype=np.uint8).astype(np.int64)
    return namespace_array


def hash_token(token_ids: np.ndarray, namespace: Optional[List[str]]) -> HashType:
    hasher = Hasher()
    hasher.reset()

    if namespace:
        namespace_key = _get_namespace_hash_key(namespace)
        if namespace_key is not None:
            hasher.update(namespace_key)

    hasher.update(token_ids)

    return HashType(hasher.digest())

@dataclass
class SequenceMeta:

    token_ids: np.ndarray

    tokens_per_block: int

    block_hashes: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    _has_hashes: bool = False

    namespace_id: Optional[int] = None

    def __init__(self, token_ids: np.ndarray, tokens_per_block: int, namespace: Optional[List[str]] = None):
        assert token_ids.ndim == 1
        assert tokens_per_block > 0
        
        self.token_ids = token_ids
        self.tokens_per_block = tokens_per_block
        self.namespace_id = None
        self.block_hashes = np.array([], dtype=np.int64)
        self._has_hashes = False

        self._namespace = namespace

        self.gen_hashes()

    @property
    def num_blocks(self) -> int:
        return len(self.token_ids) // self.tokens_per_block

    @property
    def length(self) -> int:
        return len(self.token_ids)

    def has_hashes(self) -> bool:
        return self._has_hashes

    def _create_initialized_hasher(self) -> Hasher:
        hasher = Hasher()
        hasher.reset()
        if self._namespace and len(self._namespace) > 0:
            namespace_key = _get_namespace_hash_key(self._namespace)
            if namespace_key is not None:
                hasher.update(namespace_key)
                self.namespace_id = int(hasher.digest())
        return hasher

    def get_hash(self, block_id: int) -> Optional[HashType]:
        if block_id >= self.num_blocks:
            return None
        assert self._has_hashes, "Hashes should be generated during initialization"
        return HashType(int(self.block_hashes[block_id].item()))

    def gen_hashes(self) -> None:
        if self._has_hashes:
            return
        assert self.token_ids.ndim == 1

        hasher = self._create_initialized_hasher()

        self.block_hashes = gen_hashes(self.token_ids, self.tokens_per_block, hasher)

        assert self.block_hashes.ndim == 1
        assert self.block_hashes.size == self.num_blocks
        assert self.block_hashes.itemsize == get_hash_size()
        self._has_hashes = True
