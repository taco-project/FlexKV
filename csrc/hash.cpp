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

int get_hash_size() { return sizeof(XXH64_hash_t); }

void hash_tensor(const torch::Tensor &tensor, torch::Tensor &result) {
  assert(tensor.ndim() == 1);

  *reinterpret_cast<XXH64_hash_t*>(result.data_ptr()) =
      XXH64(tensor.data_ptr(), tensor.numel() * tensor.element_size(), 0);
}

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

void Hasher::reset() {
    XXH64_reset(xxhasher, 0);
}

XXH64_hash_t Hasher::hash(const torch::Tensor &input) {
    assert(input.ndim() == 1);
    XXH64_update(xxhasher, input.data_ptr(), input.numel() * input.element_size());
    return XXH64_digest(xxhasher);
}

void Hasher::hash_out(const torch::Tensor &input, torch::Tensor &result) {
    assert(input.ndim() == 1);

    XXH64_update(xxhasher, input.data_ptr(), input.numel() * input.element_size());
    XXH64_hash_t hash_value = XXH64_digest(xxhasher);

    *reinterpret_cast<XXH64_hash_t*>(result.data_ptr()) = hash_value;
}

} // namespace flexkv
