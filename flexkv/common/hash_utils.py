from typing import NewType, Optional

import torch

from flexkv import c_ext

HashType = NewType('HashType', int)

def get_hash_size() -> int:
    return c_ext.get_hash_size()

class Hasher:
    def __init__(self):
        self.hasher = c_ext.Hasher()

    def reset(self) -> None:
        self.hasher.reset()

    def update(self, tensor: torch.Tensor) -> None:
        self.hasher.update(tensor)

    def digest(self) -> HashType:
        return HashType(self.hasher.digest())

def hash_tensor(tensor: torch.Tensor) -> HashType:
    hasher = Hasher()
    hasher.update(tensor)
    return hasher.digest()

def gen_hashes(token_ids: torch.Tensor, tokens_per_block: int, hasher: Optional[Hasher] = None) -> torch.Tensor:
    block_hashes = torch.zeros(token_ids.numel() // tokens_per_block, dtype=torch.uint64)
    if hasher is None:
        hasher = Hasher()
    c_ext.gen_hashes(hasher.hasher, token_ids, tokens_per_block, block_hashes)
    return block_hashes

if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    token_ids = torch.randint(0, 10000, (32000, ), dtype=torch.int64)
    print(f"token ids length: {token_ids.shape[0]}")
    start = time.time()
    result = hash_tensor(token_ids)
    end = time.time()
    print(f"tensor hash: {result}, time: {end - start}s")
    start = time.time()
    result = gen_hashes(token_ids, 16)
    end = time.time()
    print(f"block hashes: {result}, time: {end - start}s")
