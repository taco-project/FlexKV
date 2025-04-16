import hashlib
from typing import NewType

import torch

HashType = NewType('HashType', str)
HashObject = NewType('HashObject', hashlib._hashlib.HASH)

def hash_ctx_init() -> HashObject:
    return hashlib.md5()

def hash_ctx_update(hash_ctx: HashObject, tensor: torch.Tensor) -> None:
    hash_ctx.update(tensor.numpy().tobytes())

def hash_ctx_finalize(hash_ctx: HashObject) -> HashType:
    return hash_ctx.hexdigest()

def hash_tensor(tensor: torch.Tensor) -> HashType:
    assert tensor.ndim == 1
    return hashlib.md5(tensor.numpy().tobytes()).hexdigest()

if __name__ == "__main__":
    import time
    torch.manual_seed(0)
    token_ids = torch.randint(0, 10000, (32000, ))
    tokens_per_block = 1
    print(f"token ids length: {token_ids.shape[0]}, "
          f"tokens_per_block: {tokens_per_block}")
    start = time.time()
    hash_ctx = hash_ctx_init()
    for i in range(0, token_ids.shape[0], tokens_per_block):
        hash_ctx_update(hash_ctx, token_ids[i:i+tokens_per_block])
    hash_str = hash_ctx_finalize(hash_ctx)
    print(f"time: {time.time() - start}s")
    print(f"hash: {hash_str}")
