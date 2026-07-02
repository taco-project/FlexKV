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
#include "tp_transfer_thread_group.h"
#include "transfer.cuh"
#ifdef FLEXKV_ENABLE_NVCOMP
#include "compression/ans/nvcomp_ans_tp.h"
#endif
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <stdexcept>
#include <type_traits>

namespace flexkv {

TPTransferThreadGroup::TPTransferThreadGroup(
    int num_gpus, const std::vector<int64_t> &gpu_block_ptrs_flat,
    int num_tensors_per_gpu, int64_t cpu_blocks_ptr,
    int num_layers, const std::vector<int64_t> &gpu_kv_strides_in_bytes,
    const std::vector<int64_t> &gpu_block_strides_in_bytes,
    const std::vector<int64_t> &gpu_layer_strides_in_bytes,
    const std::vector<int64_t> &gpu_chunk_sizes_in_bytes,
    const std::vector<int64_t> &gpu_device_ids,
    bool enable_nvcomp, int nvcomp_batch_size, int nvcomp_data_type) {
  const c10::cuda::CUDAGuard restore_device_on_exit(c10::cuda::current_device());

  num_gpus_ = num_gpus;
  num_tensors_per_gpu_ = num_tensors_per_gpu;

  gpu_kv_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_block_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_layer_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_chunk_sizes_in_bytes_ = new int64_t[num_gpus];
  for (int i = 0; i < num_gpus; i++) {
    gpu_kv_strides_in_bytes_[i] = gpu_kv_strides_in_bytes[i];
    gpu_block_strides_in_bytes_[i] = gpu_block_strides_in_bytes[i];
    gpu_layer_strides_in_bytes_[i] = gpu_layer_strides_in_bytes[i];
    gpu_chunk_sizes_in_bytes_[i] = gpu_chunk_sizes_in_bytes[i];
  }

  queues_.resize(num_gpus_);
  mtxs_ = std::vector<std::mutex>(num_gpus_);
  cvs_ = std::vector<std::condition_variable>(num_gpus_);

  cudaError_t malloc_err = cudaMallocHost(
      (void **)&gpu_blocks_, num_gpus_ * num_tensors_per_gpu_ * sizeof(void *));
  if (malloc_err != cudaSuccess) {
    throw std::runtime_error(std::string("cudaMallocHost failed: ") +
                             cudaGetErrorString(malloc_err));
  }
  for (size_t i = 0; i < gpu_block_ptrs_flat.size(); ++i) {
    gpu_blocks_[i] = reinterpret_cast<void *>(gpu_block_ptrs_flat[i]);
  }

  if (num_tensors_per_gpu_ == 1) {
    backend_type_ = BackendType::TRTLLM;
  } else if (num_tensors_per_gpu_ == num_layers) {
    backend_type_ = BackendType::VLLM;
  } else if (num_tensors_per_gpu_ == num_layers * 2) {
    backend_type_ = BackendType::SGLANG;
  } else {
    throw std::runtime_error("Unsupported GPU block type: " +
                             std::to_string(num_tensors_per_gpu_));
  }

  gpu_tensor_handlers_.reserve(num_gpus_);
  for (int i = 0; i < num_gpus_; i++) {
    int64_t **gpu_blocks_ptr =
        reinterpret_cast<int64_t **>(gpu_blocks_ + i * num_tensors_per_gpu_);
    gpu_tensor_handlers_.emplace_back(
        backend_type_, gpu_blocks_ptr, num_layers, gpu_kv_strides_in_bytes_[i],
        gpu_block_strides_in_bytes_[i], gpu_layer_strides_in_bytes_[i]);
  }

  cpu_blocks_ = reinterpret_cast<void *>(cpu_blocks_ptr);

  gpu_device_ids_.resize(num_gpus_);
  for (int i = 0; i < num_gpus_; ++i) {
    gpu_device_ids_[i] = static_cast<int>(gpu_device_ids[i]);
  }

  streams_.resize(num_gpus_);
  for (int i = 0; i < num_gpus_; i += 1) {
    cudaError_t err = cudaSetDevice(gpu_device_ids_[i]);
    if (err != cudaSuccess)
      throw std::runtime_error(std::string("cudaSetDevice failed: ") +
                               cudaGetErrorString(err));
    err = cudaStreamCreate(&streams_[i]);
    if (err != cudaSuccess)
      throw std::runtime_error(std::string("cudaStreamCreate failed: ") +
                               cudaGetErrorString(err));
  }
  // create the thread pool
  stop_pool_ = false;
  for (int i = 0; i < num_gpus_; ++i) {
    threads_.emplace_back([this, i]() {
      int device_id = gpu_device_ids_[i];
      cudaSetDevice(device_id); // only once

      while (true) {
        Task task;
        {
          std::unique_lock<std::mutex> lk(mtxs_[i]);
          cvs_[i].wait(lk, [&] { return stop_pool_ || !queues_[i].empty(); });
          if (stop_pool_ && queues_[i].empty())
            return;

          task = std::move(queues_[i].front());
          queues_[i].pop();
        }
        task(); //
      }
    });
  }

#ifdef FLEXKV_ENABLE_NVCOMP
  if (enable_nvcomp) {
    init_nvcomp(nvcomp_batch_size, nvcomp_data_type);
  }
#endif

}

TPTransferThreadGroup::~TPTransferThreadGroup() {
  const c10::cuda::CUDAGuard restore_device_on_exit(c10::cuda::current_device());

  stop_pool_ = true;
  for (auto &cv : cvs_)
    cv.notify_all();
  for (auto &t : threads_)
    if (t.joinable())
      t.join();

  cudaFreeHost(gpu_blocks_);

#ifdef FLEXKV_ENABLE_NVCOMP
  destroy_nvcomp_state();
#endif

  gpu_tensor_handlers_.clear();
  delete[] gpu_kv_strides_in_bytes_;
  delete[] gpu_block_strides_in_bytes_;
  delete[] gpu_layer_strides_in_bytes_;
  delete[] gpu_chunk_sizes_in_bytes_;
}

std::future<void> TPTransferThreadGroup::enqueue_for_gpu(int gpu_idx,
                                                         Task task) {
  auto pkg = std::make_shared<std::packaged_task<void()>>(std::move(task));
  auto fut = pkg->get_future();
  {
    std::lock_guard<std::mutex> lk(mtxs_[gpu_idx]);
    queues_[gpu_idx].emplace([pkg] { (*pkg)(); });
  }
  cvs_[gpu_idx].notify_one();
  return fut;
}

void TPTransferThreadGroup::tp_group_transfer(
    const torch::Tensor &gpu_block_id_tensor,
    const torch::Tensor &cpu_block_id_tensor,
    const int64_t cpu_kv_stride_in_bytes,
    const int64_t cpu_layer_stride_in_bytes,
    const int64_t cpu_block_stride_in_bytes,
    const int64_t cpu_tp_stride_in_bytes, const int transfer_num_cta,
    const bool is_host_to_device, const bool use_ce_transfer,
    const int layer_id, const int layer_granularity, const bool is_mla,
    const std::string &mla_d2h_mode) {

  std::atomic<bool> failed{false};
  std::string error_msg;
  // threads_.clear();
  // threads_.reserve(num_gpus_);

  // Barrier sync_point(num_gpus_);
  std::vector<std::future<void>> futures;
  futures.reserve(num_gpus_);

  // Validate mla_d2h_mode parameter (only meaningful for MLA)
  std::string mode = mla_d2h_mode;
  if (is_mla && mode != "sharded" && mode != "all_write" && mode != "rank0_only") {
    fprintf(stderr, "[FlexKV] Warning: Invalid mla_d2h_mode='%s', using default 'sharded'\n",
            mode.c_str());
    mode = "sharded";
  }

  // In sharded D2H mode, chunk_size is divided by num_gpus_ and used as both
  // the per-rank transfer size and the stride between ranks. If chunk_size
  // is not divisible by num_gpus_, the integer division drops trailing bytes,
  // leaving a hole in the assembled KV on CPU.
  // All ranks share the same chunk_size (MLA = identical KV), so check [0] once.
  if (is_mla && !is_host_to_device && mode == "sharded" && num_gpus_ > 1) {
    if (gpu_chunk_sizes_in_bytes_[0] % num_gpus_ != 0) {
      throw std::runtime_error(
          "sharded MLA D2H mode requires gpu_chunk_size divisible by "
          "num_gpus, but chunk_size=" +
          std::to_string(gpu_chunk_sizes_in_bytes_[0]) + " and num_gpus=" +
          std::to_string(num_gpus_) + ". Use 'all_write' or 'rank0_only' "
          "mode, or adjust head_dim/tokens_per_block so chunk_size is "
          "divisible.");
    }
  }

  for (int i = 0; i < num_gpus_; ++i) {
    // For rank0_only mode in D2H: only rank 0 performs transfer
    if (is_mla && !is_host_to_device && mode == "rank0_only" && i != 0) {
      // Skip D2H transfer for non-rank0 GPUs
      futures.emplace_back(enqueue_for_gpu(i, [i]() {
        // Empty task - non-rank0 GPUs do nothing in rank0_only D2H mode
      }));
      continue;
    }

    futures.emplace_back(enqueue_for_gpu(i, [&, i]() {
      try {
        int num_blocks = gpu_block_id_tensor.numel();

        int64_t *gpu_block_ids =
            static_cast<int64_t *>(gpu_block_id_tensor.data_ptr());
        int64_t *cpu_block_ids =
            static_cast<int64_t *>(cpu_block_id_tensor.data_ptr());
        void *cpu_ptr = cpu_blocks_;
        int64_t cpu_startoff_inside_chunks = 0;
        int64_t gpu_startoff_inside_chunks = 0;
        int64_t chunk_size = gpu_chunk_sizes_in_bytes_[i];

        if (is_mla) {
          // MLA offset logic — inlined (was shared via mla_utils.h)
          if (mode == "sharded") {
            if (!is_host_to_device) {
              int64_t shard = gpu_chunk_sizes_in_bytes_[i] / num_gpus_;
              cpu_startoff_inside_chunks = i * shard;
              gpu_startoff_inside_chunks = i * shard;
              chunk_size = shard;
            } else {
              cpu_startoff_inside_chunks = 0;
              gpu_startoff_inside_chunks = 0;
              chunk_size = gpu_chunk_sizes_in_bytes_[i];
            }
          } else if (mode == "all_write") {
            // Each rank's complete KV occupies num_blocks blocks on CPU
            // ([GPU0][GPU1]...[GPUN]). Rank i's region starts at i * num_blocks
            // blocks from the base. Use cpu_block_stride (not gpu_chunk_size)
            // because BLOCKFIRST's block_stride includes all layers+kv_dims,
            // while LAYERFIRST's block_stride == chunk_size (same result).
            cpu_startoff_inside_chunks = i * num_blocks * cpu_block_stride_in_bytes;
            gpu_startoff_inside_chunks = 0;
            chunk_size = gpu_chunk_sizes_in_bytes_[i];
          } else if (mode == "rank0_only") {
            // D2H: only rank 0 writes (non-rank0 handled by outer continue)
            // H2D: all GPUs read from offset 0
            cpu_startoff_inside_chunks = 0;
            gpu_startoff_inside_chunks = 0;
            chunk_size = gpu_chunk_sizes_in_bytes_[i];
          }
        } else {
          // Non-MLA scenario: use default logic
          cpu_startoff_inside_chunks = i * cpu_tp_stride_in_bytes;
          gpu_startoff_inside_chunks = 0;
          chunk_size = gpu_chunk_sizes_in_bytes_[i];
        }
        
        // Dispatch to the appropriate template based on backend type
        switch (backend_type_) {
        case BackendType::VLLM:
          flexkv::transfer_kv_blocks<BackendType::VLLM>(
              num_blocks, layer_id, layer_granularity, gpu_block_ids,
              gpu_tensor_handlers_[i], gpu_startoff_inside_chunks,
              cpu_block_ids, cpu_ptr, cpu_kv_stride_in_bytes,
              cpu_layer_stride_in_bytes, cpu_block_stride_in_bytes,
              cpu_startoff_inside_chunks, chunk_size, streams_[i],
              transfer_num_cta, is_host_to_device, use_ce_transfer, is_mla);
          break;
        case BackendType::TRTLLM:
          flexkv::transfer_kv_blocks<BackendType::TRTLLM>(
              num_blocks, layer_id, layer_granularity, gpu_block_ids,
              gpu_tensor_handlers_[i], gpu_startoff_inside_chunks,
              cpu_block_ids, cpu_ptr, cpu_kv_stride_in_bytes,
              cpu_layer_stride_in_bytes, cpu_block_stride_in_bytes,
              cpu_startoff_inside_chunks, chunk_size, streams_[i],
              transfer_num_cta, is_host_to_device, use_ce_transfer, is_mla);
          break;
        case BackendType::SGLANG:
          flexkv::transfer_kv_blocks<BackendType::SGLANG>(
              num_blocks, layer_id, layer_granularity, gpu_block_ids,
              gpu_tensor_handlers_[i], gpu_startoff_inside_chunks,
              cpu_block_ids, cpu_ptr, cpu_kv_stride_in_bytes,
              cpu_layer_stride_in_bytes, cpu_block_stride_in_bytes,
              cpu_startoff_inside_chunks, chunk_size, streams_[i],
              transfer_num_cta, is_host_to_device, use_ce_transfer, is_mla);
          break;
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
          failed = true;
          error_msg = cudaGetErrorString(err);
        }
      } catch (const std::exception &e) {
        failed = true;
        error_msg = e.what();
      }
    }));
  }

  for (auto &f : futures) {
    f.get();
  }

  if (failed) {
    throw std::runtime_error("tp_group_transfer failed: " + error_msg);
  }
}

} // namespace flexkv
