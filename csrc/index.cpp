#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "cache_utils.h"

namespace py = pybind11;

namespace flexkv {

torch::Tensor get_prefix_block_ids(int64_t last_block_index,
                                   int64_t last_block_id,
                                   const torch::Tensor &prev_block_ids) {
  assert(prev_block_ids.ndim() == 1);
  assert(prev_block_ids.type() == torch::kInt64);

  torch::Tensor result = torch::zeros({last_block_index + 1}, torch::kInt64);

  int64_t *prev_block_ids_ptr = prev_block_ids.data_ptr<int64_t>();
  int64_t *result_ptr = result.data_ptr<int64_t>();
  for (int i = last_block_index; i >= 0; i--) {
    result_ptr[i] = last_block_id;
    last_block_id = prev_block_ids_ptr[last_block_id];
  }

  return result;
}

torch::Tensor get_block_ids_from_hashes(const py::list &hashes,
                                        const py::dict &hash_to_block_id) {
  assert(hashes.size() > 0);

  int num_hashes = hashes.size();
  int hash_size = get_hash_size();

  torch::Tensor result = torch::zeros({num_hashes}, torch::kInt64);
  int64_t *result_ptr = result.data_ptr<int64_t>();

  for (int i = 0; i < num_hashes; i++) {
    py::bytes hash_bytes = hashes[i].cast<py::bytes>();
    result_ptr[i] = hash_to_block_id[hash_bytes].cast<int64_t>();
  }

  return result;
}

void index_batch_insert(const py::list &hashes, const torch::Tensor &block_ids,
                        py::dict &hash_to_block_id) {
  assert(block_ids.ndim() == 1);
  assert(block_ids.type() == torch::kInt64);
  assert(hashes.size() == block_ids.size());

  int num_hashes = hashes.size();
  int64_t *block_ids_ptr = block_ids.data_ptr<int64_t>();
  for (int i = 0; i < num_hashes; i++) {
    py::bytes hash_bytes = hashes[i].cast<py::bytes>();
    hash_to_block_id[hash_bytes] = block_ids_ptr[i];
  }
}

} // namespace flexkv
