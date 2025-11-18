import time
from typing import NewType, Optional

import numpy as np
import torch

from flexkv import c_ext


HashType = NewType('HashType', int)

def get_hash_size() -> int:
    return int(c_ext.get_hash_size())

class Hasher:
    def __init__(self) -> None:
        self.hasher = c_ext.Hasher()

    def reset(self) -> None:
        self.hasher.reset()

    def update(self, array: np.ndarray) -> None:
        self.hasher.update(torch.from_numpy(array))

    def digest(self) -> HashType:
        return HashType(self.hasher.digest())

_HASHER = Hasher()

def hash_array(array: np.ndarray) -> HashType:
    _HASHER.reset()
    _HASHER.update(array)
    return HashType(_HASHER.digest())

def hash_array_with_prefix(array: np.ndarray, prefix: int) -> HashType:
    """Hash array with a prefix value (e.g., device type) to avoid collisions."""
    _HASHER.reset()
    # Add prefix as a single-element array to the hash
    _HASHER.update(np.array([prefix], dtype=np.int64))
    _HASHER.update(array)
    return HashType(_HASHER.digest())

def gen_hashes(token_ids: np.ndarray, tokens_per_block: int, hasher: Optional[Hasher] = None) -> np.ndarray:
    block_hashes = np.zeros(token_ids.size // tokens_per_block, dtype=np.uint64)
    if hasher is None:
        hasher = Hasher()
    c_ext.gen_hashes(hasher.hasher, torch.from_numpy(token_ids), tokens_per_block, torch.from_numpy(block_hashes))
    return block_hashes

if __name__ == "__main__":
    np.random.seed(0)
    token_ids = np.random.randint(0, 10000, (1000, ), dtype=np.int64)
    print(f"token ids length: {token_ids.shape[0]}")
    result = hash_array(token_ids)
    start = time.time()
    for i in range(1):
        result = hash_array(token_ids)
    end = time.time()
    print(f"array hash: {result}, average time: {(end - start)*1000/5}ms")
    # start = time.time()
    # result2 = gen_hashes(token_ids, 16)
    # end = time.time()
    # print(f"block hashes: {result2}, time: {(end - start)*1000}ms")
