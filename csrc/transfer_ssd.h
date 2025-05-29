#pragma once
#include <torch/extension.h>
#include <vector>

namespace flexkv {

void transfer_kv_blocks_ssd(
    const std::vector<std::string> &filenames,
    const torch::Tensor &cpu_layer_id_list,
    const torch::Tensor &cpu_layer_ptrs_tensor,
    const torch::Tensor &ssd_block_ids, const torch::Tensor &cpu_block_ids,
    int64_t cpu_kv_stride_in_bytes, int64_t ssd_layer_stride_in_bytes,
    int64_t ssd_block_stride_in_bytes, int64_t ssd_kv_stride_in_bytes,
    int64_t block_size_in_bytes, int64_t total_layers, bool is_read,
    int round_robin, bool use_mmap = false, int num_threads_per_file = 8);

} // namespace flexkv
