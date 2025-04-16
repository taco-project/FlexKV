from typing import NewType

import torch

from flexkv.c_ext import get_hash_size
from flexkv.c_ext import hash_tensor as _hash_tensor_cpp

HashType = NewType('HashType', bytes)

def hash_tensor(tensor: torch.Tensor) -> HashType:
    assert tensor.ndim == 1
    result = torch.zeros(get_hash_size(), dtype=torch.uint8)
    _hash_tensor_cpp(tensor, result)
    return result.numpy().tobytes()

if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    token_ids = torch.randint(0, 10000, (32000, ))
    tokens_per_block = 1
    print(f"token ids length: {token_ids.shape[0]}, "
          f"tokens_per_block: {tokens_per_block}")
    start = time.time()
    result = hash_tensor(token_ids)
    print(f"time: {time.time() - start}s")
    print(f"hash: {result}")
