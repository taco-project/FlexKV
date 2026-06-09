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
  // Save caller's current device so per-GPU cudaSetDevice loops don't leak
  // a device change back to the caller's thread.
  int saved_device = -1;
  cudaGetDevice(&saved_device);

  num_gpus_ = num_gpus;
  num_tensors_per_gpu_ = num_tensors_per_gpu;

#ifdef FLEXKV_ENABLE_NVCOMP
  if (enable_nvcomp) {
    if (nvcomp_batch_size <= 0) {
      throw std::invalid_argument(
          "TPTransferThreadGroup: nvcomp_batch_size must be positive");
    }
    nvcomp_batch_size_ = nvcomp_batch_size;
    nvcomp_data_type_ = nvcomp_data_type;
  }
#else
  (void)enable_nvcomp;
  (void)nvcomp_batch_size;
  (void)nvcomp_data_type;
#endif

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

  if (saved_device >= 0) cudaSetDevice(saved_device);
}

TPTransferThreadGroup::~TPTransferThreadGroup() {
  int saved_device = -1;
  cudaGetDevice(&saved_device);

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

  if (saved_device >= 0) cudaSetDevice(saved_device);
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
    const int layer_id, const int layer_granularity, const bool is_mla) {

  std::atomic<bool> failed{false};
  std::string error_msg;
  // threads_.clear();
  // threads_.reserve(num_gpus_);

  // Barrier sync_point(num_gpus_);
  std::vector<std::future<void>> futures;
  futures.reserve(num_gpus_);

  bool enable_sharded_d2h = is_mla && !is_host_to_device;

  for (int i = 0; i < num_gpus_; ++i) {
    futures.emplace_back(enqueue_for_gpu(i, [&, i]() {
      try {
        int num_blocks = gpu_block_id_tensor.numel();

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

#ifdef FLEXKV_ENABLE_NVCOMP
void TPTransferThreadGroup::ensure_nvcomp_initialized() {
  if (nvcomp_ready_) {
    return;
  }
  if (nvcomp_batch_size_ <= 0) {
    throw std::runtime_error(
        "TPTransferThreadGroup: nvcomp config missing; construct with "
        "enable_nvcomp=True and nvcomp batch/data type before calling "
        "tp_group_transfer_ans");
  }
  init_nvcomp(nvcomp_batch_size_, nvcomp_data_type_);
}

void TPTransferThreadGroup::destroy_nvcomp_state() {
  int saved_device = -1;
  cudaGetDevice(&saved_device);

  for (int i = 0; i < static_cast<int>(ans_contexts_.size()); i++) {
    if (ans_contexts_[i] != nullptr) {
      cudaSetDevice(gpu_device_ids_[i]);
      delete ans_contexts_[i];
      ans_contexts_[i] = nullptr;
    }
  }
  ans_contexts_.clear();

  if (owned_gpu_block_ids_ != nullptr || owned_cpu_block_ids_ != nullptr) {
    for (int i = 0; i < num_gpus_; i++) {
      if (owned_gpu_block_ids_ != nullptr && owned_gpu_block_ids_[i] != nullptr)
        cudaFreeHost(owned_gpu_block_ids_[i]);
      if (owned_cpu_block_ids_ != nullptr && owned_cpu_block_ids_[i] != nullptr)
        cudaFreeHost(owned_cpu_block_ids_[i]);
    }
    delete[] owned_gpu_block_ids_;
    delete[] owned_cpu_block_ids_;
    delete[] owned_block_id_capacity_;
    owned_gpu_block_ids_ = nullptr;
    owned_cpu_block_ids_ = nullptr;
    owned_block_id_capacity_ = nullptr;
  }

  nvcomp_ready_ = false;
  if (saved_device >= 0) cudaSetDevice(saved_device);
}

void TPTransferThreadGroup::init_nvcomp(int nvcomp_batch_size,
                                        int nvcomp_data_type) {
  if (nvcomp_batch_size <= 0) {
    throw std::invalid_argument(
        "TPTransferThreadGroup: nvcomp_batch_size must be positive");
  }
  if (nvcomp_ready_ && nvcomp_batch_size_ == nvcomp_batch_size &&
      nvcomp_data_type_ == nvcomp_data_type) {
    return;
  }
  nvcomp_batch_size_ = nvcomp_batch_size;
  nvcomp_data_type_ = nvcomp_data_type;

  int saved_device = -1;
  cudaGetDevice(&saved_device);

  try {
    destroy_nvcomp_state();
    ans_contexts_.resize(num_gpus_, nullptr);
    owned_gpu_block_ids_ = new int64_t *[num_gpus_]();
    owned_cpu_block_ids_ = new int64_t *[num_gpus_]();
    owned_block_id_capacity_ = new int64_t[num_gpus_]();
    for (int i = 0; i < num_gpus_; i++) {
      cudaError_t err = cudaSetDevice(gpu_device_ids_[i]);
      if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("cudaSetDevice failed for nvcomp ctx: ") +
            cudaGetErrorString(err));
      ans_contexts_[i] = new ANSTransferContext();
      // Non-MLA ranks use per-rank chunks. MLA ranks use the full canonical
      // chunk because MLA KV is replicated across TP ranks, not head-sharded.
      ans_ctx_create(ans_contexts_[i], (size_t)nvcomp_batch_size,
                     (size_t)gpu_chunk_sizes_in_bytes_[i], nvcomp_data_type);
    }
    nvcomp_ready_ = true;
  } catch (...) {
    destroy_nvcomp_state();
    if (saved_device >= 0) cudaSetDevice(saved_device);
    throw;
  }

  if (saved_device >= 0) cudaSetDevice(saved_device);
}

size_t TPTransferThreadGroup::tp_group_transfer_ans(
    const torch::Tensor &gpu_block_id_tensor,
    const torch::Tensor &cpu_block_id_tensor,
    const int64_t cpu_kv_stride_in_bytes,
    const int64_t cpu_layer_stride_in_bytes,
    const int64_t cpu_block_stride_in_bytes,
    const int64_t cpu_tp_stride_in_bytes, const int transfer_num_cta,
    const bool is_host_to_device, const bool use_ce_transfer,
    const int layer_id, const int layer_granularity, const bool is_mla,
    const int64_t cpu_size_table_tp_ptr,
    const int64_t cpu_size_table_tp_rank_stride,
    const int64_t cpu_size_table_block_stride,
    const int64_t cpu_size_table_layer_stride) {
  (void)transfer_num_cta;
  (void)use_ce_transfer;

  ensure_nvcomp_initialized();
  if (cpu_size_table_tp_ptr == 0 ||
      (!is_mla && cpu_size_table_tp_rank_stride == 0) ||
      cpu_size_table_block_stride == 0 ||
      cpu_size_table_layer_stride == 0) {
    throw std::runtime_error(
        "TPTransferThreadGroup: nvcomp TP requires a non-null "
        "cpu_size_table/cpu_size_table_tp pointer and non-zero required "
        "size-table strides.");
  }

  // Accumulates compressed payload bytes across all TP ranks' ans_* calls so
  // Python can compute the per-op compression ratio (uncomp / wire).
  std::atomic<size_t> total_compressed_bytes{0};
  std::atomic<bool> failed{false};
  std::mutex error_mutex;
  std::string error_msg;
  auto record_error = [&](const std::string& msg) {
    std::lock_guard<std::mutex> lock(error_mutex);
    if (!failed.exchange(true)) {
      error_msg = msg;
    }
  };
  std::vector<std::future<void>> futures;
  futures.reserve(num_gpus_);

  for (int i = 0; i < num_gpus_; ++i) {
    futures.emplace_back(enqueue_for_gpu(i, [&, i]() {
      try {
        int num_blocks = gpu_block_id_tensor.numel();

        int64_t *gpu_block_ids =
            static_cast<int64_t *>(gpu_block_id_tensor.data_ptr());
        int64_t *cpu_block_ids =
            static_cast<int64_t *>(cpu_block_id_tensor.data_ptr());
        void *cpu_ptr = cpu_blocks_;

        auto dispatch_nvcomp = [&](auto backend_tag, int cur_num_blocks,
                                   int64_t *cur_gpu_block_ids,
                                   int64_t *cur_cpu_block_ids,
                                   void *cur_cpu_ptr,
                                   int64_t cur_cpu_kv_stride,
                                   int64_t cur_cpu_layer_stride,
                                   int64_t cur_cpu_block_stride,
                                   int64_t cur_chunk_size,
                                   uint32_t *cur_size_table_base) {
          constexpr BackendType BT = decltype(backend_tag)::value;
          size_t comp = 0;
          if (is_host_to_device) {
            comp = transfer_kv_blocks_ans_decomp<BT>(
                ans_contexts_[i], cur_num_blocks, layer_id, layer_granularity,
                cur_gpu_block_ids, gpu_tensor_handlers_[i], cur_cpu_block_ids,
                cur_cpu_ptr, cur_cpu_kv_stride, cur_cpu_layer_stride,
                cur_cpu_block_stride, cur_chunk_size, is_mla,
                cur_size_table_base, cpu_size_table_block_stride,
                cpu_size_table_layer_stride, streams_[i]);
          } else {
            comp = transfer_kv_blocks_ans_comp<BT>(
                ans_contexts_[i], cur_num_blocks, layer_id, layer_granularity,
                cur_gpu_block_ids, gpu_tensor_handlers_[i], cur_cpu_block_ids,
                cur_cpu_ptr, cur_cpu_kv_stride, cur_cpu_layer_stride,
                cur_cpu_block_stride, cur_chunk_size, is_mla,
                cur_size_table_base, cpu_size_table_block_stride,
                cpu_size_table_layer_stride, streams_[i]);
          }
          // Per-rank accumulation so sum-across-ranks == system total compressed
          // bytes. MHA: each rank holds a unique 1/N slice. MLA D2H: owner-
          // sharded, each rank handles its own blocks. MLA H2D: every rank reads
          // the same canonical table and returns the identical full sum, so only
          // rank 0 contributes to avoid over-counting by N.
          const bool skip_accumulate = is_mla && is_host_to_device && i != 0;
          if (!skip_accumulate) {
            total_compressed_bytes.fetch_add(comp, std::memory_order_relaxed);
          }
        };

        auto run_dispatch = [&](int cur_num_blocks, int64_t *cur_gpu_block_ids,
                                int64_t *cur_cpu_block_ids, void *cur_cpu_ptr,
                                int64_t cur_cpu_kv_stride,
                                int64_t cur_cpu_layer_stride,
                                int64_t cur_cpu_block_stride,
                                int64_t cur_chunk_size,
                                uint32_t *cur_size_table_base) {
          switch (backend_type_) {
          case BackendType::VLLM:
            dispatch_nvcomp(
                std::integral_constant<BackendType, BackendType::VLLM>{},
                cur_num_blocks, cur_gpu_block_ids, cur_cpu_block_ids,
                cur_cpu_ptr, cur_cpu_kv_stride, cur_cpu_layer_stride,
                cur_cpu_block_stride, cur_chunk_size, cur_size_table_base);
            break;
          case BackendType::TRTLLM:
            dispatch_nvcomp(
                std::integral_constant<BackendType, BackendType::TRTLLM>{},
                cur_num_blocks, cur_gpu_block_ids, cur_cpu_block_ids,
                cur_cpu_ptr, cur_cpu_kv_stride, cur_cpu_layer_stride,
                cur_cpu_block_stride, cur_chunk_size, cur_size_table_base);
            break;
          case BackendType::SGLANG:
            dispatch_nvcomp(
                std::integral_constant<BackendType, BackendType::SGLANG>{},
                cur_num_blocks, cur_gpu_block_ids, cur_cpu_block_ids,
                cur_cpu_ptr, cur_cpu_kv_stride, cur_cpu_layer_stride,
                cur_cpu_block_stride, cur_chunk_size, cur_size_table_base);
            break;
          }
        };

        if (is_mla) {
          // MLA KV is replicated across TP ranks. One canonical compressed full
          // chunk lives in the size table. D2H is distributed by cpu_block_id
          // owner so all ranks contribute without splitting the chunk; H2D fans out — every rank reads
          // the same canonical table.
          uint32_t *canonical_size_table_base =
              reinterpret_cast<uint32_t *>(cpu_size_table_tp_ptr);

          if (is_host_to_device) {
            run_dispatch(num_blocks, gpu_block_ids, cpu_block_ids, cpu_ptr,
                         cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes,
                         cpu_block_stride_in_bytes,
                         gpu_chunk_sizes_in_bytes_[i],
                         canonical_size_table_base);
          } else {
            // Allocate enough pinned scratch for the block-id list owned by
            // this rank during MLA D2H owner-sharding.
            if (num_blocks > owned_block_id_capacity_[i]) {
              int64_t *new_gpu_ids = nullptr;
              cudaError_t err = cudaMallocHost(
                  reinterpret_cast<void **>(&new_gpu_ids),
                  static_cast<size_t>(num_blocks) * sizeof(int64_t));
              if (err != cudaSuccess) {
                throw std::runtime_error(
                    std::string("owned_gpu_block_ids: cudaMallocHost failed: ") +
                    cudaGetErrorString(err));
              }

              int64_t *new_cpu_ids = nullptr;
              err = cudaMallocHost(reinterpret_cast<void **>(&new_cpu_ids),
                                   static_cast<size_t>(num_blocks) *
                                       sizeof(int64_t));
              if (err != cudaSuccess) {
                cudaFreeHost(new_gpu_ids);
                throw std::runtime_error(
                    std::string("owned_cpu_block_ids: cudaMallocHost failed: ") +
                    cudaGetErrorString(err));
              }

              if (owned_gpu_block_ids_[i] != nullptr)
                cudaFreeHost(owned_gpu_block_ids_[i]);
              if (owned_cpu_block_ids_[i] != nullptr)
                cudaFreeHost(owned_cpu_block_ids_[i]);
              owned_gpu_block_ids_[i] = new_gpu_ids;
              owned_cpu_block_ids_[i] = new_cpu_ids;
              owned_block_id_capacity_[i] = num_blocks;
            }

            int owned_blocks = 0;
            for (int b = 0; b < num_blocks; b++) {
              int64_t owner = cpu_block_ids[b] % num_gpus_;
              if (owner < 0) owner += num_gpus_;
              if (owner != i) continue;
              owned_gpu_block_ids_[i][owned_blocks] = gpu_block_ids[b];
              owned_cpu_block_ids_[i][owned_blocks] = cpu_block_ids[b];
              owned_blocks++;
            }
            if (owned_blocks > 0) {
              run_dispatch(owned_blocks, owned_gpu_block_ids_[i],
                           owned_cpu_block_ids_[i], cpu_ptr,
                           cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes,
                           cpu_block_stride_in_bytes,
                           gpu_chunk_sizes_in_bytes_[i],
                           canonical_size_table_base);
            }
          }
        } else {
          // MHA: each rank uses a non-TP ANSTransferContext on its per-rank
          // slice of the CPU buffer. The table is
          // [tp_size, num_cpu_blocks, num_layers, kv_dim] uint32; each rank
          // gets a 3-D slice the non-TP kernels treat like a regular table.
          void *cpu_ptr_offset =
              static_cast<uint8_t *>(cpu_ptr) + i * cpu_tp_stride_in_bytes;
          uint32_t *rank_size_table_base =
              reinterpret_cast<uint32_t *>(cpu_size_table_tp_ptr) +
              (int64_t)i * cpu_size_table_tp_rank_stride;

          run_dispatch(num_blocks, gpu_block_ids, cpu_block_ids, cpu_ptr_offset,
                       cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes,
                       cpu_block_stride_in_bytes, gpu_chunk_sizes_in_bytes_[i],
                       rank_size_table_base);
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
          record_error(cudaGetErrorString(err));
        }
      } catch (const std::exception &e) {
        record_error(e.what());
      }
    }));
  }

  for (auto &f : futures) {
    f.get();
  }

  if (failed) {
    throw std::runtime_error("tp_group_transfer_ans failed: " + error_msg);
  }
  return total_compressed_bytes.load(std::memory_order_relaxed);
}
#endif // FLEXKV_ENABLE_NVCOMP

} // namespace flexkv