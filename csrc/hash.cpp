#include <chrono>
#include <cstring>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>
#include <xxhash.h>

#include <torch/extension.h>

#include "cache_utils.h"

namespace flexkv {

int get_hash_size() { return sizeof(HashType); }

Hasher::Hasher() {
  xxhasher = XXH64_createState();
  if (xxhasher == nullptr) {
    throw std::runtime_error("Failed to create XXH64 state");
  }
  reset();
}

Hasher::~Hasher() {
  if (xxhasher != nullptr) {
    XXH64_freeState(xxhasher);
  }
}

void Hasher::reset() { XXH64_reset(xxhasher, 0); }

HashType Hasher::digest() { return XXH64_digest(xxhasher); }

Hasher &Hasher::update(const torch::Tensor &input) {
  assert(input.ndim() == 1);
  XXH64_update(xxhasher, input.data_ptr(),
               input.numel() * input.element_size());
  return *this;
}

Hasher &Hasher::update(const void *input, size_t size) {
  XXH64_update(xxhasher, input, size);
  return *this;
}

void gen_hashes(Hasher &hasher, const torch::Tensor &token_ids,
                int tokens_per_block, torch::Tensor &block_hashes) {
  assert(token_ids.ndim() == 1);
  assert(block_hashes.ndim() == 1);
  assert(token_ids.numel() / tokens_per_block == block_hashes.numel());

  assert(token_ids.element_size() == sizeof(int64_t));
  assert(block_hashes.element_size() == sizeof(HashType));

  int num_blocks = block_hashes.numel();
  int64_t *token_ids_ptr = token_ids.data_ptr<int64_t>();
  HashType *block_hashes_ptr =
      reinterpret_cast<HashType *>(block_hashes.data_ptr());

  for (int i = 0; i < num_blocks; i++) {
    hasher.update(token_ids_ptr + i * tokens_per_block,
                  tokens_per_block * sizeof(int64_t));
    block_hashes_ptr[i] = hasher.digest();
  }
}

} // namespace flexkv
