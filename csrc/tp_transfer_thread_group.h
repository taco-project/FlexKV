/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
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

#include "device_thread_pool.h"
#include "gtensor_handler.cuh"
#include "transfer.cuh"
#include "ce_transfer.h"
#include <cuda_runtime.h>
#include <functional>
#include <future>
#include <memory>
#include <string>
#include <torch/extension.h>
#include <vector>

namespace flexkv {

#ifdef FLEXKV_ENABLE_NVCOMP
struct NvcompTPState;
#endif

class TPTransferThreadGroup {
public:
  TPTransferThreadGroup(int num_gpus,
                        const std::vector<int64_t> &gpu_block_ptrs_flat,
                        int num_tensors_per_gpu, int64_t cpu_blocks_ptr,
                        int num_layers,
                        const std::vector<int64_t> &gpu_kv_strides_in_bytes,
                        const std::vector<int64_t> &gpu_block_strides_in_bytes,
                        const std::vector<int64_t> &gpu_layer_strides_in_bytes,
                        const std::vector<int64_t> &gpu_chunk_sizes_in_bytes,
                        const std::vector<int64_t> &gpu_device_ids,
                        bool enable_nvcomp = false,
                        int nvcomp_batch_size = 0,
                        int nvcomp_data_type = 0,
                        CETransferConfig ce_config = CETransferConfig{});

  ~TPTransferThreadGroup();

  void update_gpu_block_ptrs(
      const std::vector<int64_t> &gpu_block_ptrs_flat);

  // sync=true (the default) keeps the historical contract: the call returns
  // only once every rank's copy has landed. sync=false launches onto each
  // rank's stream and returns as soon as the launches are issued -- the caller
  // must then call wait_all_streams() before touching the data. Multi-group
  // callers want the latter: with sync=true group N+1 cannot even be launched
  // while group N drains, which serializes what the per-rank streams could
  // otherwise overlap.
  void tp_group_transfer(const torch::Tensor &gpu_block_id_tensor,
                         const torch::Tensor &cpu_block_id_tensor,
                         const int64_t cpu_kv_stride_in_bytes,
                         const int64_t cpu_layer_stride_in_bytes,
                         const int64_t cpu_block_stride_in_bytes,
                         const int64_t cpu_tp_stride_in_bytes,
                         const int transfer_num_cta,
                         const bool is_host_to_device,
                         const bool use_ce_transfer, const int layer_id,
                         const int layer_granularity, const int kv_dim,
                         const int num_kv_heads,
                         const std::string &kv_shared_across_ranks_mode = "sharded",
                         const int designated_rank = 0,
                         const bool sync = true);

  // Block until every rank's stream has drained. Only needed after a
  // tp_group_transfer(sync=false); a no-op otherwise.
  void wait_all_streams();

#ifdef FLEXKV_ENABLE_NVCOMP

  void init_nvcomp(int nvcomp_batch_size, int nvcomp_data_type);
  void destroy_nvcomp_state();
  void ensure_nvcomp_initialized();

  size_t tp_group_transfer_ans(const torch::Tensor &gpu_block_id_tensor,
                               const torch::Tensor &cpu_block_id_tensor,
                               const int64_t cpu_kv_stride_in_bytes,
                               const int64_t cpu_layer_stride_in_bytes,
                               const int64_t cpu_block_stride_in_bytes,
                               const int64_t cpu_tp_stride_in_bytes,
                               const int transfer_num_cta,
                               const bool is_host_to_device,
                               const bool use_ce_transfer, const int layer_id,
                               const int layer_granularity, const int kv_dim,
                               const int num_kv_heads,
                               const int64_t cpu_size_table_tp_ptr,
                               const int64_t cpu_size_table_tp_rank_stride,
                               const int64_t cpu_size_table_block_stride,
                               const int64_t cpu_size_table_layer_stride);
#endif

private:
  using Task = std::function<void()>;
  std::future<void> enqueue_for_gpu(int gpu_idx, Task task);

  // One thread + one stream per rank, cudaSetDevice applied once at start-up.
  // Was five hand-rolled members here (threads_, queues_, mtxs_, cvs_,
  // stop_pool_) plus a destructor that had to join before destroying streams;
  // the same five lived in TPGDSTransferThreadGroup and the since-deleted
  // LayerwiseTransferGroup,
  // and the copies drifted -- this one leaked every stream it created until
  // that was fixed in only this copy.
  std::unique_ptr<DeviceThreadPool> pool_;

  int num_gpus_;
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

  CETransferConfig ce_config_;

  // Mirrors pool_->stream(i); kept because nvcomp_ans_tp.cpp indexes
  // streams_[i] directly from inside a pool task.
  std::vector<cudaStream_t> streams_;

  // rank_rotate mode: request-level round-robin counter, incremented each D2H call.
  int rotate_counter_ = 0;

#ifdef FLEXKV_ENABLE_NVCOMP
  std::unique_ptr<NvcompTPState> nvcomp_state_;
#endif
};

} // namespace flexkv
