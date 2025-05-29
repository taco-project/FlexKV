#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <xxhash.h>
namespace py = pybind11;

namespace flexkv {

using HashType = XXH64_hash_t;

int get_hash_size();

class Hasher {
private:
  XXH64_state_t *xxhasher;

public:
  Hasher();
  ~Hasher();
  void reset();
  Hasher &update(const torch::Tensor &input);
  Hasher &update(const void *input, size_t size);
  HashType digest();
};

void gen_hashes(Hasher &hasher, const torch::Tensor &token_ids,
                int tokens_per_block, torch::Tensor &block_hashes);

} // namespace flexkv
