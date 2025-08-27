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
        self.hasher.update(array)

    def digest(self) -> HashType:
        return HashType(self.hasher.digest())

def hash_array(array: np.ndarray) -> HashType:
    hasher = Hasher()
    hasher.update(array)
    return HashType(hasher.digest())

def gen_hashes(token_ids: np.ndarray, tokens_per_block: int, hasher: Optional[Hasher] = None) -> np.ndarray:
    block_hashes = np.zeros(token_ids.size // tokens_per_block, dtype=np.uint64)
    if hasher is None:
        hasher = Hasher()
    c_ext.gen_hashes(hasher.hasher, torch.from_numpy(token_ids), tokens_per_block, torch.from_numpy(block_hashes))
    return block_hashes

if __name__ == "__main__":
    np.random.seed(0)
    token_ids = np.random.randint(0, 10000, (32000, ), dtype=np.int64)
    print(f"token ids length: {token_ids.shape[0]}")
    start = time.time()
    result = hash_array(token_ids)
    end = time.time()
    print(f"array hash: {result}, time: {end - start}s")
    start = time.time()
    result2 = gen_hashes(token_ids, 16)
    end = time.time()
    print(f"block hashes: {result2}, time: {end - start}s")
