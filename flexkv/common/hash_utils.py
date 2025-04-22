from typing import NewType

import torch

from flexkv import c_ext

HashType = NewType('HashType', bytes)

def get_hash_size() -> int:
    return c_ext.get_hash_size()

def hash_tensor(tensor: torch.Tensor) -> torch.Tensor:
    assert tensor.ndim == 1
    result = torch.zeros(get_hash_size(), dtype=torch.uint8)
    c_ext.hash_tensor(tensor, result)
    return result

if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    token_ids = torch.randint(0, 10000, (32000, ), dtype=torch.int64)
    print(f"token ids length: {token_ids.shape[0]}")
    start = time.time()
    result = hash_tensor(token_ids)
    print(f"time: {time.time() - start}s")
    print(f"hash: {result}")
