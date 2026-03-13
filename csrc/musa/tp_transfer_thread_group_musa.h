/*
 * MUSA TPTransferThreadGroup - mirrors csrc/tp_transfer_thread_group.h
 * Uses musaStream_t and musa* APIs. Build when MUSA SDK and transfer_musa are available.
 */
#pragma once

#include <atomic>
#include <condition_variable>
#include <musa_runtime.h>
#include <memory>
#include <mutex>
#include <thread>
#include <torch/extension.h>
#include <vector>
#include <queue>
#include <functional>
#include <future>
#include <string>
#include "transfer_musa.muh"
#include "gtensor_handler_musa.h"

namespace flexkv {

class TPTransferThreadGroupMusa {
 public:
  TPTransferThreadGroupMusa(
      int num_gpus, const std::vector<std::vector<torch::Tensor>>& gpu_blocks,
      torch::Tensor& cpu_blocks, int dp_group_id,
      int num_layers,
      torch::Tensor& gpu_kv_strides_tensor,
      torch::Tensor& gpu_block_strides_tensor,
      torch::Tensor& gpu_layer_strides_tensor,
      torch::Tensor& gpu_chunk_sizes_tensor);
  ~TPTransferThreadGroupMusa();

  void tp_group_transfer(const torch::Tensor& gpu_block_id_tensor,
                         const torch::Tensor& cpu_block_id_tensor,
                         const int64_t cpu_kv_stride_in_bytes,
                         const int64_t cpu_layer_stride_in_bytes,
                         const int64_t cpu_block_stride_in_bytes,
                         const int64_t cpu_tp_stride_in_bytes,
                         const int transfer_sms, const bool is_host_to_device,
                         const bool use_ce_transfer, const int layer_id,
                         const int layer_granularity, const bool is_mla);

 private:
  using Task = std::function<void()>;
  std::future<void> enqueue_for_gpu(int gpu_idx, Task task);

  int num_gpus_;
  int dp_group_id_;
  void** gpu_blocks_;
  void* cpu_blocks_;
  int num_tensors_per_gpu_;
  int64_t* gpu_kv_strides_in_bytes_;
  int64_t* gpu_block_strides_in_bytes_;
  int64_t* gpu_layer_strides_in_bytes_;
  int64_t* gpu_chunk_sizes_in_bytes_;

  BackendType backend_type_;
  std::vector<GTensorHandler> gpu_tensor_handlers_;

  std::vector<std::thread> threads_;
  std::vector<musaStream_t> streams_;

  std::vector<std::queue<Task>> queues_;
  std::vector<std::mutex> mtxs_;
  std::vector<std::condition_variable> cvs_;
  std::atomic<bool> stop_pool_;
};

}  // namespace flexkv
