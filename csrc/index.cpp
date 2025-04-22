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

int32_t find_n_liner_parents_for_eviction(int64_t block_id, 
                                          const torch::Tensor &prev_block_ids,
                                          const torch::Tensor &lock_cnt,
                                          const torch::Tensor &child_cnt,
                                          const torch::Tensor &status,
                                          torch::Tensor &result) {
  int32_t result_num = 0;
  int n = result.size(0);
  int64_t *prev_block_ids_ptr = prev_block_ids.data_ptr<int64_t>();
  int64_t *result_ptr = result.data_ptr<int64_t>();
  int32_t *child_cnt_ptr = child_cnt.data_ptr<int32_t>();
  int32_t *status_ptr = status.data_ptr<int32_t>();
  int32_t *lock_cnt_ptr = lock_cnt.data_ptr<int32_t>();

  result_ptr[result_num++] = block_id;
  block_id = prev_block_ids_ptr[block_id];
  while (result_num <= n 
         && block_id != -1
         && status_ptr[block_id] == 1
         && child_cnt_ptr[block_id] == 1
         && lock_cnt_ptr[block_id] == 0) {
    result_ptr[result_num++] = block_id;
    block_id = prev_block_ids_ptr[block_id];
  }
  return result_num;
}

torch::Tensor get_block_ids_from_hashes(const torch::Tensor &hashes,
                                        const py::dict &hash_to_block_id) {
  assert(hashes.ndim() == 2);
  assert(hashes.type() == torch::kUInt8);

  int num_hashes = hashes.size(0);
  int hash_size = get_hash_size();

  torch::Tensor result = torch::zeros({num_hashes}, torch::kInt64);
  int64_t *result_ptr = result.data_ptr<int64_t>();
  uint8_t *hashes_ptr = hashes.data_ptr<uint8_t>();

  for (int i = 0; i < num_hashes; i++) {
    py::bytes hash_bytes(reinterpret_cast<char*>(hashes_ptr + i * hash_size), hash_size);
    result_ptr[i] = hash_to_block_id[hash_bytes].cast<int64_t>();
  }

  return result;
}

void index_batch_insert(const torch::Tensor &hashes, const torch::Tensor &block_ids,
                        py::dict &hash_to_block_id) {
  assert(block_ids.ndim() == 1);
  assert(block_ids.type() == torch::kInt64);
  assert(hashes.ndim() == 2);
  assert(hashes.type() == torch::kUInt8);

  int num_hashes = hashes.size(0);
  int hash_size = hashes.size(1);
  
  int64_t *block_ids_ptr = block_ids.data_ptr<int64_t>();
  uint8_t *hashes_ptr = hashes.data_ptr<uint8_t>();
  for (int i = 0; i < num_hashes; i++) {
    py::bytes hash_bytes(reinterpret_cast<char*>(hashes_ptr + i * hash_size), hash_size);
    hash_to_block_id[hash_bytes] = block_ids_ptr[i];
  }
}

void index_batch_remove(const torch::Tensor &hashes, py::dict &hash_to_block_id) {
  assert(hashes.ndim() == 2);
  assert(hashes.type() == torch::kUInt8);

  int num_hashes = hashes.size(0);
  int hash_size = hashes.size(1);
  
  uint8_t *hashes_ptr = hashes.data_ptr<uint8_t>();
  for (int i = 0; i < num_hashes; i++) {
    py::bytes hash_bytes(reinterpret_cast<char*>(hashes_ptr + i * hash_size), hash_size);
    hash_to_block_id.attr("__delitem__")(hash_bytes);
  }
}
} // namespace flexkv
