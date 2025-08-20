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
  using Task = std::function<void()>;
  std::future<void> enqueue_for_gpu(int gpu_idx, Task task);

  int num_gpus_;
  int dp_group_id_;
  void **gpu_blocks_;
  void *cpu_blocks_;
  std::vector<std::thread> threads_;
  std::vector<cudaStream_t> streams_;

  std::vector<std::queue<Task>> queues_;
  std::vector<std::mutex> mtxs_;
  std::vector<std::condition_variable> cvs_;
  std::atomic<bool> stop_pool_;
};

} // namespace flexkv
