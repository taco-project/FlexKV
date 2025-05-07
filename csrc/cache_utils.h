#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <xxhash.h>
namespace py = pybind11;

namespace flexkv {

int get_hash_size();

void hash_tensor(const torch::Tensor &tensor, torch::Tensor &result);

class Hasher {
private:
    XXH64_state_t* xxhasher;

public:
    Hasher();
    ~Hasher();
    void reset();
    XXH64_hash_t hash(const torch::Tensor &input);
    void hash_out(const torch::Tensor &input, torch::Tensor &result);
};

torch::Tensor get_prefix_block_ids(int64_t last_block_index,
                                   int64_t last_block_id,
                                   const torch::Tensor &prev_block_ids);

torch::Tensor find_n_liner_parents_for_eviction(
    int64_t block_id, const torch::Tensor &prev_block_ids,
    const torch::Tensor &lock_cnt, const torch::Tensor &child_cnt,
    const torch::Tensor &ready, const torch::Tensor &last_access_time,
    double max_last_access_time, int max_num_evicted);

torch::Tensor get_block_ids_from_hashes(const torch::Tensor &hashes,
                                        const py::dict &hash_to_block_id);

void index_batch_insert(const torch::Tensor &hashes,
                        const torch::Tensor &block_ids,
                        py::dict &hash_to_block_id);

void index_batch_remove(const torch::Tensor &hashes,
                        py::dict &hash_to_block_id);

} // namespace flexkv
