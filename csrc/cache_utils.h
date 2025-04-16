#pragma once

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

namespace flexkv {

int get_hash_size();

void hash_tensor(const torch::Tensor &tensor, torch::Tensor &result);

torch::Tensor get_prefix_block_ids(int last_block_index, int last_block_id,
                                   const torch::Tensor &prev_block_ids);

torch::Tensor get_block_ids_from_hashes(const torch::Tensor &hashes,
                                        const py::dict &hash_to_block_id);
} // namespace flexkv
