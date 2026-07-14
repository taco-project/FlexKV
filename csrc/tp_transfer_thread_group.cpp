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
#include "transfer_debug.h"

#include <chrono>
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

  bool enable_sharded_d2h = is_mla && !is_host_to_device;
  const int num_blocks = gpu_block_id_tensor.numel();
  const auto xfer_t0 = std::chrono::steady_clock::now();

  // FLEXKV_D2H_LOG(
  //     "tp_group_transfer ENTER backend=%s num_gpus=%d blocks=%d "
  //     "h2d=%d ce=%d mla=%d sharded_d2h=%d layer_id=%d layer_gran=%d cta=%d "
  //     "cpu_strides(kv/layer/block/tp)=%lld/%lld/%lld/%lld cpu_base=%p",
  //     backend_name(backend_type_), num_gpus_, num_blocks,
  //     is_host_to_device ? 1 : 0, use_ce_transfer ? 1 : 0, is_mla ? 1 : 0,
  //     enable_sharded_d2h ? 1 : 0, layer_id, layer_granularity, transfer_num_cta,
  //     static_cast<long long>(cpu_kv_stride_in_bytes),
  //     static_cast<long long>(cpu_layer_stride_in_bytes),
  //     static_cast<long long>(cpu_block_stride_in_bytes),
  //     static_cast<long long>(cpu_tp_stride_in_bytes), cpu_blocks_);

  if (num_blocks > 0) {
    const int64_t *gpu_ids =
        static_cast<const int64_t *>(gpu_block_id_tensor.data_ptr());
    const int64_t *cpu_ids =
        static_cast<const int64_t *>(cpu_block_id_tensor.data_ptr());
    // FLEXKV_D2H_LOG(
    //     "tp_group_transfer block_ids gpu=[%lld,%lld] cpu=[%lld,%lld]",
    //     static_cast<long long>(min_block_id(gpu_ids, num_blocks)),
    //     static_cast<long long>(max_block_id(gpu_ids, num_blocks)),
    //     static_cast<long long>(min_block_id(cpu_ids, num_blocks)),
    //     static_cast<long long>(max_block_id(cpu_ids, num_blocks)));
    // log_pointer_attributes("gpu_block_ids", gpu_ids);
    // log_pointer_attributes("cpu_block_ids", cpu_ids);
    // log_pointer_attributes("cpu_base", cpu_blocks_);
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
      D2hDebugGpuScope gpu_scope(i);
      try {
        int64_t *gpu_block_ids =
            static_cast<int64_t *>(gpu_block_id_tensor.data_ptr());
        int64_t *cpu_block_ids =
            static_cast<int64_t *>(cpu_block_id_tensor.data_ptr());
        void *cpu_ptr = cpu_blocks_;
        int64_t cpu_startoff_inside_chunks = 0;
        if (enable_sharded_d2h)
          cpu_startoff_inside_chunks =
              i * gpu_chunk_sizes_in_bytes_[i] / num_gpus_;
        else if (!is_mla)
          cpu_startoff_inside_chunks = i * cpu_tp_stride_in_bytes;
        int64_t gpu_startoff_inside_chunks =
            enable_sharded_d2h ? i * gpu_chunk_sizes_in_bytes_[i] / num_gpus_
                               : 0;
        // we assume that the chunk size is the same for all gpus,
        // even if they have different number of gpu_blocks
        int64_t chunk_size = enable_sharded_d2h
                                 ? gpu_chunk_sizes_in_bytes_[i] / num_gpus_
                                 : gpu_chunk_sizes_in_bytes_[i];

        // FLEXKV_D2H_LOG(
        //     "tp_group_transfer gpu=%d dev=%d chunk=%lld gpu_off=%lld "
        //     "cpu_off=%lld stream=%p",
        //     i, gpu_device_ids_[i], static_cast<long long>(chunk_size),
        //     static_cast<long long>(gpu_startoff_inside_chunks),
        //     static_cast<long long>(cpu_startoff_inside_chunks),
        //     static_cast<void *>(streams_[i]));

        const auto gpu_t0 = std::chrono::steady_clock::now();
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

        const auto gpu_ms =
            std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - gpu_t0)
                .count();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
          failed = true;
          error_msg = std::string("gpu=") + std::to_string(i) + " dev=" +
                      std::to_string(gpu_device_ids_[i]) + " " +
                      cudaGetErrorString(err);
          // FLEXKV_D2H_LOG("tp_group_transfer gpu=%d FAIL elapsed_ms=%lld err=%s",
          //                i, static_cast<long long>(gpu_ms),
          //                cudaGetErrorString(err));
        } else {
          // FLEXKV_D2H_LOG("tp_group_transfer gpu=%d thread done elapsed_ms=%lld",
          //                i, static_cast<long long>(gpu_ms));
        }
      } catch (const std::exception &e) {
        failed = true;
        error_msg = std::string("gpu=") + std::to_string(i) + " exception: " +
                    e.what();
        // FLEXKV_D2H_LOG("tp_group_transfer gpu=%d exception: %s", i, e.what());
      }
    }));
  }

  // FLEXKV_D2H_LOG("tp_group_transfer waiting for %d gpu futures", num_gpus_);
  for (int i = 0; i < static_cast<int>(futures.size()); ++i) {
    const auto wait_t0 = std::chrono::steady_clock::now();
    futures[i].get();
    const auto wait_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - wait_t0)
            .count();
    // FLEXKV_D2H_LOG("tp_group_transfer future gpu=%d joined wait_ms=%lld", i,
    //                static_cast<long long>(wait_ms));
  }

  const auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                            std::chrono::steady_clock::now() - xfer_t0)
                            .count();

  if (failed) {
    // FLEXKV_D2H_LOG("tp_group_transfer LEAVE FAIL total_ms=%lld err=%s",
    //                static_cast<long long>(total_ms), error_msg.c_str());
    throw std::runtime_error("tp_group_transfer failed: " + error_msg);
  }

  // FLEXKV_D2H_LOG("tp_group_transfer LEAVE OK total_ms=%lld",
  //                static_cast<long long>(total_ms));
}

} // namespace flexkv
