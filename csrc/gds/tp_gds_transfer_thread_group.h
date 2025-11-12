#pragma once

#include <atomic>
#include <condition_variable>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <thread>
#include <torch/extension.h>
#include <vector>
#include <string>
#include <map>
#include <queue>
#include <functional>
#include <future>

// Forward declaration
class GDSManager;

namespace flexkv {

class TPGDSTransferThreadGroup {
public:
  TPGDSTransferThreadGroup(
      int num_gpus, 
      const std::vector<std::vector<torch::Tensor>> &gpu_blocks,
      std::map<int, std::vector<std::string>> &ssd_files, 
      int dp_group_id,
      int num_layers,
      torch::Tensor &gpu_kv_strides_tensor,
      torch::Tensor &gpu_block_strides_tensor,
      torch::Tensor &gpu_chunk_sizes_tensor);
  ~TPGDSTransferThreadGroup();

  void tp_group_transfer(
      const torch::Tensor &gpu_block_id_tensor,
      const torch::Tensor &ssd_block_id_tensor,
      const int64_t ssd_layer_stride_in_bytes,
      const int64_t ssd_kv_stride_in_bytes,
      const int64_t ssd_block_stride_in_bytes,
      const int64_t ssd_chunk_size_in_bytes,
      const int64_t num_blocks_per_file,
      const bool is_read,  // true for SSD->GPU, false for GPU->SSD
      const int layer_id,
      const int layer_granularity, 
      const bool is_mla);

private:
  using Task = std::function<void()>;
  std::future<void> enqueue_for_gpu(int gpu_idx, Task task);

  int num_gpus_;
  int dp_group_id_;
  void **gpu_blocks_;
  
  int64_t *gpu_kv_strides_in_bytes_;
  int64_t *gpu_block_strides_in_bytes_;
  int64_t *gpu_chunk_sizes_in_bytes_;
  
  std::vector<GDSManager*> gds_managers_;
  std::vector<std::thread> threads_;
  std::vector<cudaStream_t> streams_;

  std::vector<std::queue<Task>> queues_;
  std::vector<std::mutex> mtxs_;
  std::vector<std::condition_variable> cvs_;
  std::atomic<bool> stop_pool_;
};

} // namespace flexkv 