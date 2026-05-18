/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Vendor-scoped header for TPTransferThreadGroup. Moved from
 * csrc/tp_transfer_thread_group.h during the GPU backend abstraction
 * refactor (P3).
 */
#pragma once

#include "gtensor_handler.cuh"
#include "transfer.cuh"
#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <torch/extension.h>
#include <vector>

namespace flexkv {

class TPTransferThreadGroup {
public:
  TPTransferThreadGroup(int num_gpus,
                        const std::vector<int64_t> &gpu_block_ptrs_flat,
                        int num_tensors_per_gpu, int64_t cpu_blocks_ptr,
                        int dp_group_id, int num_layers,
                        const std::vector<int64_t> &gpu_kv_strides_in_bytes,
                        const std::vector<int64_t> &gpu_block_strides_in_bytes,
                        const std::vector<int64_t> &gpu_layer_strides_in_bytes,
                        const std::vector<int64_t> &gpu_chunk_sizes_in_bytes,
                        const std::vector<int64_t> &gpu_device_ids);

  ~TPTransferThreadGroup();

  void tp_group_transfer(const torch::Tensor &gpu_block_id_tensor,
                         const torch::Tensor &cpu_block_id_tensor,
                         const int64_t cpu_kv_stride_in_bytes,
                         const int64_t cpu_layer_stride_in_bytes,
                         const int64_t cpu_block_stride_in_bytes,
                         const int64_t cpu_tp_stride_in_bytes,
                         const int transfer_num_cta,
                         const bool is_host_to_device,
                         const bool use_ce_transfer, const int layer_id,
                         const int layer_granularity, const bool is_mla);

private:
  using Task = std::function<void()>;
  std::future<void> enqueue_for_gpu(int gpu_idx, Task task);

  int num_gpus_;
  int dp_group_id_;
  std::vector<int> gpu_device_ids_;
  void **gpu_blocks_;
  void *cpu_blocks_;
  int num_tensors_per_gpu_;
  int64_t *gpu_kv_strides_in_bytes_;
  int64_t *gpu_block_strides_in_bytes_;
  int64_t *gpu_layer_strides_in_bytes_;
  int64_t *gpu_chunk_sizes_in_bytes_;

  // Simplified: just one vector of handlers, runtime backend type selection
  BackendType backend_type_;
  std::vector<GTensorHandler> gpu_tensor_handlers_;

  std::vector<std::thread> threads_;
  std::vector<gpu_stream_t> streams_;

  std::vector<std::queue<Task>> queues_;
  std::vector<std::mutex> mtxs_;
  std::vector<std::condition_variable> cvs_;
  std::atomic<bool> stop_pool_;
};

} // namespace flexkv
