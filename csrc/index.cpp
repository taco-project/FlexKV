#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "cache_utils.h"

namespace py = pybind11;

namespace flexkv {

torch::Tensor get_prefix_block_ids(int last_block_index, int last_block_id,
                                   const torch::Tensor &prev_block_ids) {
  assert(prev_block_ids.ndim() == 1);
  assert(prev_block_ids.type() == torch::kInt32);

  torch::Tensor result = torch::zeros({last_block_index + 1}, torch::kInt32);

  int32_t *prev_block_ids_ptr = prev_block_ids.data_ptr<int32_t>();
  int32_t *result_ptr = result.data_ptr<int32_t>();
  for (int i = last_block_index; i >= 0; i--) {
    result_ptr[i] = last_block_id;
    last_block_id = prev_block_ids_ptr[last_block_id];
  }

  return result;
}

torch::Tensor get_block_ids_from_hashes(const torch::Tensor &hashes,
                                        const py::dict &hash_to_block_id) {
  assert(hashes.ndim() == 2);
  assert(hashes.contiguous());

  int num_rows = hashes.size(0);
  int num_cols = hashes.size(1);
  const uint8_t *hashes_ptr = hashes.data_ptr<uint8_t>();

  assert(num_cols * hashes.element_size() == get_hash_size());

  torch::Tensor result = torch::zeros({num_rows}, torch::kInt32);
  int32_t *result_ptr = result.data_ptr<int32_t>();

  for (int i = 0; i < num_rows; i++) {
    const uint8_t *row_ptr = hashes_ptr + i * num_cols * hashes.element_size();
    auto row_bytes = py::bytes(reinterpret_cast<const char *>(row_ptr),
                               num_cols * hashes.element_size());
    result_ptr[i] = hash_to_block_id[row_bytes].cast<int32_t>();
  }

  return result;
}

} // namespace flexkv
