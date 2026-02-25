/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <atomic>
#include <condition_variable>
#include <cuda_runtime.h>
#include <memory>
#include <mutex>
#include <thread>
#include <torch/extension.h>
#include <vector>
#include <condition_variable>
#include <queue>
#include <functional>
#include <future>
#include <string>
#include "transfer.cuh"
#include "gtensor_handler.cuh"

namespace flexkv {
  
class TPTransferThreadGroup {
public:
  TPTransferThreadGroup(
      int num_gpus,
      const std::vector<int64_t> &gpu_block_ptrs_flat,
      int num_tensors_per_gpu,
      int64_t cpu_blocks_ptr,
      int dp_group_id,
      int num_layers,
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
                         const int transfer_sms, const bool is_host_to_device,
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
  std::vector<cudaStream_t> streams_;

  std::vector<std::queue<Task>> queues_;
  std::vector<std::mutex> mtxs_;
  std::vector<std::condition_variable> cvs_;
  std::atomic<bool> stop_pool_;
};

} // namespace flexkv
