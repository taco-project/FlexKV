#pragma once

#include <atomic>
#include <condition_variable>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <torch/extension.h>
#include <vector>

#include "gtensor_handler.cuh"
#include "transfer.cuh"
#include "transfer_ssd.h"

namespace flexkv {

class LayerwiseTransferGroup {
public:
  LayerwiseTransferGroup(
      int num_gpus, const std::vector<std::vector<torch::Tensor>> &gpu_blocks,
      torch::Tensor &cpu_blocks,
      std::map<int, std::vector<std::string>> &ssd_files, int dp_group_id,
      int num_layers, torch::Tensor &gpu_kv_strides_tensor,
      torch::Tensor &gpu_block_strides_tensor,
      torch::Tensor &gpu_layer_strides_tensor,
      torch::Tensor &gpu_chunk_sizes_tensor, int iouring_entries,
      int iouring_flags);

  ~LayerwiseTransferGroup();

  // Layerwise transfer: SSD->CPU + CPU->GPU in one call
  void layerwise_transfer(
      const torch::Tensor
          &ssd_block_ids, // SSD source block ids (for disk2host)
      const torch::Tensor
          &cpu_block_ids_d2h, // CPU dest block ids (for disk2host)
      const int64_t ssd_layer_stride_in_bytes,
      const int64_t ssd_kv_stride_in_bytes, const int num_blocks_per_file,
      const int round_robin, const int num_threads_per_device,
      const torch::Tensor
          &gpu_block_id_tensor, // GPU dest block ids (for host2device)
      const torch::Tensor
          &cpu_block_id_tensor, // CPU source block ids (for host2device)
      const int64_t cpu_kv_stride_in_bytes,
      const int64_t cpu_layer_stride_in_bytes,
      const int64_t cpu_block_stride_in_bytes,
      const int64_t cpu_chunk_size_in_bytes, const int transfer_sms,
      const bool use_ce_transfer, const int layer_id,
      const int layer_granularity, const bool is_mla);

private:
  using Task = std::function<void()>;
  std::future<void> enqueue_for_gpu(int gpu_idx, Task task);

  int num_gpus_;
  int dp_group_id_;
  void **gpu_blocks_;
  void *cpu_blocks_;
  int num_tensors_per_gpu_;
  int64_t *gpu_kv_strides_in_bytes_;
  int64_t *gpu_block_strides_in_bytes_;
  int64_t *gpu_layer_strides_in_bytes_;
  int64_t *gpu_chunk_sizes_in_bytes_;

  BackendType backend_type_;
  std::vector<GTensorHandler> gpu_tensor_handlers_;

  std::vector<std::thread> threads_;
  std::vector<cudaStream_t> streams_;

  std::vector<std::queue<Task>> queues_;
  std::vector<std::mutex> mtxs_;
  std::vector<std::condition_variable> cvs_;
  std::atomic<bool> stop_pool_;

  // SSD IO context
  bool enable_ssd_;
  std::unique_ptr<SSDIOCTX> ioctx_;
};

} // namespace flexkv
