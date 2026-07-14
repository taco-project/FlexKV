/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */
#include "compression/ans/nvcomp_ans_tp.h"
#include "tp_transfer_thread_group.h"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <atomic>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace flexkv {

void TPTransferThreadGroup::ensure_nvcomp_initialized() {
  if (nvcomp_state_ && nvcomp_state_->ready) {
    return;
  }
  if (!nvcomp_state_ || nvcomp_state_->batch_size <= 0) {
    throw std::runtime_error(
        "TPTransferThreadGroup: nvcomp config missing; construct with "
        "enable_nvcomp=True and nvcomp batch/data type before calling "
        "tp_group_transfer_ans");
  }
  init_nvcomp(nvcomp_state_->batch_size, nvcomp_state_->data_type);
}

void TPTransferThreadGroup::destroy_nvcomp_state() {
  if (!nvcomp_state_) {
    return;
  }

  const c10::cuda::CUDAGuard restore_device_on_exit(c10::cuda::current_device());

  for (int i = 0; i < static_cast<int>(nvcomp_state_->ans_contexts.size()); i++) {
    if (nvcomp_state_->ans_contexts[i] != nullptr) {
      cudaSetDevice(gpu_device_ids_[i]);
      delete nvcomp_state_->ans_contexts[i];
      nvcomp_state_->ans_contexts[i] = nullptr;
    }
  }
  nvcomp_state_->ans_contexts.clear();

  if (nvcomp_state_->owned_gpu_block_ids != nullptr || nvcomp_state_->owned_cpu_block_ids != nullptr) {
    for (int i = 0; i < num_gpus_; i++) {
      if (nvcomp_state_->owned_gpu_block_ids != nullptr && nvcomp_state_->owned_gpu_block_ids[i] != nullptr)
        cudaFreeHost(nvcomp_state_->owned_gpu_block_ids[i]);
      if (nvcomp_state_->owned_cpu_block_ids != nullptr && nvcomp_state_->owned_cpu_block_ids[i] != nullptr)
        cudaFreeHost(nvcomp_state_->owned_cpu_block_ids[i]);
    }
    delete[] nvcomp_state_->owned_gpu_block_ids;
    delete[] nvcomp_state_->owned_cpu_block_ids;
    delete[] nvcomp_state_->owned_block_id_capacity;
    nvcomp_state_->owned_gpu_block_ids = nullptr;
    nvcomp_state_->owned_cpu_block_ids = nullptr;
    nvcomp_state_->owned_block_id_capacity = nullptr;
  }

  nvcomp_state_->ready = false;
}

void TPTransferThreadGroup::init_nvcomp(int nvcomp_batch_size,
                                        int nvcomp_data_type) {
  if (nvcomp_batch_size <= 0) {
    throw std::invalid_argument(
        "TPTransferThreadGroup: nvcomp_batch_size must be positive");
  }
  if (!nvcomp_state_) {
    nvcomp_state_ = std::make_unique<NvcompTPState>();
  }
  if (nvcomp_state_->ready && nvcomp_state_->batch_size == nvcomp_batch_size &&
      nvcomp_state_->data_type == nvcomp_data_type) {
    return;
  }
  nvcomp_state_->batch_size = nvcomp_batch_size;
  nvcomp_state_->data_type = nvcomp_data_type;

  const c10::cuda::CUDAGuard restore_device_on_exit(c10::cuda::current_device());

  try {
    destroy_nvcomp_state();
    nvcomp_state_->ans_contexts.resize(num_gpus_, nullptr);
    nvcomp_state_->owned_gpu_block_ids = new int64_t *[num_gpus_]();
    nvcomp_state_->owned_cpu_block_ids = new int64_t *[num_gpus_]();
    nvcomp_state_->owned_block_id_capacity = new int64_t[num_gpus_]();
    for (int i = 0; i < num_gpus_; i++) {
      cudaError_t err = cudaSetDevice(gpu_device_ids_[i]);
      if (err != cudaSuccess)
        throw std::runtime_error(
            std::string("cudaSetDevice failed for nvcomp ctx: ") +
            cudaGetErrorString(err));
      nvcomp_state_->ans_contexts[i] = new ANSTransferContext();
      // Non-MLA ranks use per-rank chunks. MLA ranks use the full canonical
      // chunk because MLA KV is replicated across TP ranks, not head-sharded.
      ans_ctx_create(nvcomp_state_->ans_contexts[i], (size_t)nvcomp_batch_size,
                     (size_t)gpu_chunk_sizes_in_bytes_[i], nvcomp_data_type);
    }
    nvcomp_state_->ready = true;
  } catch (...) {
    destroy_nvcomp_state();
    throw;
  }
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
                nvcomp_state_->ans_contexts[i], cur_num_blocks, layer_id, layer_granularity,
                cur_gpu_block_ids, gpu_tensor_handlers_[i], cur_cpu_block_ids,
                cur_cpu_ptr, cur_cpu_kv_stride, cur_cpu_layer_stride,
                cur_cpu_block_stride, cur_chunk_size, is_mla,
                cur_size_table_base, cpu_size_table_block_stride,
                cpu_size_table_layer_stride, streams_[i]);
          } else {
            comp = transfer_kv_blocks_ans_comp<BT>(
                nvcomp_state_->ans_contexts[i], cur_num_blocks, layer_id, layer_granularity,
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
            if (num_blocks > nvcomp_state_->owned_block_id_capacity[i]) {
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

              if (nvcomp_state_->owned_gpu_block_ids[i] != nullptr)
                cudaFreeHost(nvcomp_state_->owned_gpu_block_ids[i]);
              if (nvcomp_state_->owned_cpu_block_ids[i] != nullptr)
                cudaFreeHost(nvcomp_state_->owned_cpu_block_ids[i]);
              nvcomp_state_->owned_gpu_block_ids[i] = new_gpu_ids;
              nvcomp_state_->owned_cpu_block_ids[i] = new_cpu_ids;
              nvcomp_state_->owned_block_id_capacity[i] = num_blocks;
            }

            int owned_blocks = 0;
            for (int b = 0; b < num_blocks; b++) {
              int64_t owner = cpu_block_ids[b] % num_gpus_;
              if (owner < 0) owner += num_gpus_;
              if (owner != i) continue;
              nvcomp_state_->owned_gpu_block_ids[i][owned_blocks] = gpu_block_ids[b];
              nvcomp_state_->owned_cpu_block_ids[i][owned_blocks] = cpu_block_ids[b];
              owned_blocks++;
            }
            if (owned_blocks > 0) {
              run_dispatch(owned_blocks, nvcomp_state_->owned_gpu_block_ids[i],
                           nvcomp_state_->owned_cpu_block_ids[i], cpu_ptr,
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

} // namespace flexkv
