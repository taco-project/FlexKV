from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, NewType, Optional

import numpy as np
import torch

from flexkv.common.hash_utils import HashType, Hasher, gen_hashes, get_hash_size


def caller_hashes_required() -> bool:
    from flexkv.common.config import GLOBAL_CONFIG_FROM_ENV
    return bool(GLOBAL_CONFIG_FROM_ENV.hash_scheme)


def _get_namespace_hash_key(namespace: Optional[List[str]]) -> Optional[np.ndarray]:
    if not namespace or len(namespace) == 0:
        return None
    
    # Use null character as delimiter to prevent ambiguity
    # e.g. ["a:b", "c"] → "a:b\x00c" != ["a", "b:c"] → "a\x00b:c"
    namespace_key = "\x00".join(namespace)
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


def format_block_hash(value: Optional[int]) -> str:
    """Format signed or unsigned 64-bit hashes as fixed-width hex."""
    if value is None:
        return "-"
    return f"0x{int(value) & ((1 << 64) - 1):016x}"


@dataclass
class SequenceMeta:

    token_ids: np.ndarray

    tokens_per_block: int

    block_hashes: np.ndarray = field(default_factory=lambda: np.array([], dtype=np.int64))

    _has_hashes: bool = False

    namespace_id: Optional[int] = None

    def __init__(self, token_ids: np.ndarray, tokens_per_block: int, namespace: Optional[List[str]] = None,
                 *, block_hashes: Optional[np.ndarray] = None):
        assert token_ids.ndim == 1
        assert tokens_per_block > 0
        
        self.token_ids = token_ids
        self.tokens_per_block = tokens_per_block
        self.namespace_id = None
        self.block_hashes = np.array([], dtype=np.int64)
        self._has_hashes = False

        self._namespace = namespace

        if block_hashes is not None:
            self._adopt_hashes(block_hashes)
        elif caller_hashes_required():
            raise ValueError("FLEXKV_HASH_SCHEME is set but the request carries no block_hashes")
        else:
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

    def _adopt_hashes(self, block_hashes: np.ndarray) -> None:
        # Already prefix-chained (namespace included) by the caller.
        hashes = np.ascontiguousarray(np.asarray(block_hashes).reshape(-1))
        if hashes.dtype not in (np.int64, np.uint64):
            raise TypeError(f"block_hashes must be int64 or uint64, got {hashes.dtype}")
        if hashes.size < self.num_blocks:
            raise ValueError(
                f"block_hashes has {hashes.size} entries for {self.num_blocks} blocks")
        self.block_hashes = hashes[:self.num_blocks].view(np.uint64)
        self._has_hashes = True

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
