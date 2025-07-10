#pragma once

#include <atomic>
#include <condition_variable>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <thread>
#include <torch/extension.h>
#include <vector>

namespace flexkv {

class TPTransferThreadGroup {
public:
  TPTransferThreadGroup(
      int num_gpus, const std::vector<std::vector<torch::Tensor>> &gpu_blocks,
      torch::Tensor &cpu_blocks, int dp_group_id);
  ~TPTransferThreadGroup();

  void tp_group_transfer(const torch::Tensor &gpu_block_id_tensor,
                         const int64_t gpu_kv_stride_in_bytes,
                         const int64_t gpu_block_stride_in_bytes,
                         const int64_t gpu_chunk_size_in_bytes,
                         const torch::Tensor &cpu_block_id_tensor,
                         const int64_t cpu_kv_stride_in_bytes,
                         const int64_t cpu_layer_stride_in_bytes,
                         const int64_t cpu_block_stride_in_bytes,
                         const int64_t cpu_chunk_size_in_bytes,
                         const int transfer_sms, const bool is_host_to_device,
                         const bool use_ce_transfer, const int layer_id,
                         const int layer_granularity, const bool is_mla);

private:
  int num_gpus_;
  int dp_group_id_;
  void **gpu_blocks_;
  void *cpu_blocks_;
  std::vector<std::thread> threads_;
  std::vector<cudaStream_t> streams_;
};

} // namespace flexkv
