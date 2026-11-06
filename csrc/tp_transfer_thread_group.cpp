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
#include "logging.h"
#include "transfer.cuh"
#include "transfer_backend.h"
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
    bool enable_nvcomp, int nvcomp_batch_size, int nvcomp_data_type,
    CETransferConfig ce_config)
    : ce_config_(ce_config) {
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

  // Threads and streams: one per rank, device bound once. The pool owns the
  // streams and destroys them after joining its threads.
  pool_ = std::make_unique<DeviceThreadPool>(gpu_device_ids_);
  streams_.resize(num_gpus_);
  for (int i = 0; i < num_gpus_; ++i) {
    streams_[i] = pool_->stream(i);
  }

#ifdef FLEXKV_ENABLE_NVCOMP
  if (enable_nvcomp) {
    init_nvcomp(nvcomp_batch_size, nvcomp_data_type);
  }
#endif

}

void TPTransferThreadGroup::update_gpu_block_ptrs(
    const std::vector<int64_t> &gpu_block_ptrs_flat) {
  const size_t expected =
      static_cast<size_t>(num_gpus_) * num_tensors_per_gpu_;
  if (gpu_block_ptrs_flat.size() != expected) {
    throw std::invalid_argument("GPU pointer count does not match transfer group");
  }
  for (int i = 0; i < num_gpus_; ++i) {
    cudaError_t err = cudaSetDevice(gpu_device_ids_[i]);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaSetDevice failed: ") +
                               cudaGetErrorString(err));
    }
    err = cudaStreamSynchronize(streams_[i]);
    if (err != cudaSuccess) {
      throw std::runtime_error(std::string("cudaStreamSynchronize failed: ") +
                               cudaGetErrorString(err));
    }
  }
  for (size_t i = 0; i < expected; ++i) {
    gpu_blocks_[i] = reinterpret_cast<void *>(gpu_block_ptrs_flat[i]);
  }
}

TPTransferThreadGroup::~TPTransferThreadGroup() {
  const c10::cuda::CUDAGuard restore_device_on_exit(c10::cuda::current_device());
  // CUDAGuard only restores c10's view of the current device; the raw
  // cudaSetDevice calls below bypass that cache, so snapshot and restore the
  // driver-level device explicitly as well.
  // Destroying the pool joins its threads, then syncs and destroys every
  // stream. It must happen *before* the pinned buffers are freed below: an
  // in-flight copy may still be reading gpu_blocks_. streams_ only mirrors
  // the pool's handles, so it is dangling from here on.
  pool_.reset();
  streams_.clear();
  // Swallow any error raised by the teardown above so it is not mistaken for
  // a transfer failure by the next cudaGetLastError() caller.
  cudaGetLastError();

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
  return pool_->enqueue(gpu_idx, std::move(task));
}

void TPTransferThreadGroup::tp_group_transfer(
    const torch::Tensor &gpu_block_id_tensor,
    const torch::Tensor &cpu_block_id_tensor,
    const int64_t cpu_kv_stride_in_bytes,
    const int64_t cpu_layer_stride_in_bytes,
    const int64_t cpu_block_stride_in_bytes,
    const int64_t cpu_tp_stride_in_bytes,     const int transfer_num_cta,
    const bool is_host_to_device, const bool use_ce_transfer,
    const int layer_id, const int layer_granularity, const int kv_dim,
    const int num_kv_heads,
    const std::string &kv_shared_across_ranks_mode,
    const int designated_rank, const bool sync) {

  std::atomic<bool> failed{false};
  // error_msg is written from up to num_gpus_ worker threads concurrently.
  // Guard it: an unsynchronized std::string assignment from several threads
  // is a data race (torn read / double free of the heap buffer), not merely
  // a "last writer wins" ambiguity. The first message is kept because it is
  // the one closest to the root cause; later ranks usually just report the
  // downstream fallout.
  std::mutex error_mtx;
  std::string error_msg;
  auto record_error = [&](const std::string &msg) {
    std::lock_guard<std::mutex> lk(error_mtx);
    failed = true;
    if (error_msg.empty()) {
      error_msg = msg;
    }
  };
  std::vector<std::future<void>> futures;
  futures.reserve(num_gpus_);

  // Validate kv_shared_across_ranks_mode parameter (only meaningful for shared KV)
  std::string mode = kv_shared_across_ranks_mode;
  if (num_kv_heads == 1 && mode != "sharded" && mode != "all_write" && mode != "rank0_only"
      && mode != "layer_parallel" && mode != "rank_rotate") {
    FLEXKV_LOG_WARNING(
        "operation=transfer_config act=fallback status=degraded "
        "field=kv_shared_across_ranks_mode value=\"%s\" fallback=sharded",
        mode.c_str());
    mode = "sharded";
  }

  // In sharded D2H mode, chunk_size is divided by num_gpus_ and used as both
  // the per-rank transfer size and the stride between ranks. If chunk_size
  // is not divisible by num_gpus_, the integer division drops trailing bytes,
  // leaving a hole in the assembled KV on CPU.
  // All ranks share the same chunk_size (num_kv_heads==1 = identical KV), so check [0] once.
  if (num_kv_heads == 1 && !is_host_to_device && mode == "sharded" && num_gpus_ > 1) {
    if (gpu_chunk_sizes_in_bytes_[0] % num_gpus_ != 0) {
      throw std::runtime_error(
          "sharded kv_shared_across_ranks D2H mode requires gpu_chunk_size divisible by "
          "num_gpus, but chunk_size=" +
          std::to_string(gpu_chunk_sizes_in_bytes_[0]) + " and num_gpus=" +
          std::to_string(num_gpus_) + ". Use 'all_write' or 'rank0_only' "
          "mode, or adjust head_dim/tokens_per_block so chunk_size is "
          "divisible.");
    }
  }

  // rank_rotate: resolve designated rank from round-robin counter, treat as rank0_only.
  int eff_designated_rank = designated_rank;
  if (num_kv_heads == 1 && !is_host_to_device && mode == "rank_rotate") {
    eff_designated_rank = rotate_counter_;
    rotate_counter_ = (rotate_counter_ + 1) % num_gpus_;
  }

  for (int i = 0; i < num_gpus_; ++i) {
    // For rank0_only / rank_rotate mode in D2H: only the designated rank performs transfer
    if (num_kv_heads == 1 && !is_host_to_device && (mode == "rank0_only" || mode == "rank_rotate")
        && i != eff_designated_rank) {
      // Skip D2H transfer for non-designated GPUs
      futures.emplace_back(enqueue_for_gpu(i, [i]() {
        // Empty task - non-designated GPUs do nothing in rank0_only D2H mode
      }));
      continue;
    }

    // round_robin D2H: skip ranks with 0 layers (layer_granularity < num_gpus_)
    if (num_kv_heads == 1 && !is_host_to_device && mode == "layer_parallel") {
      int L_rotate = layer_granularity, N_rotate = num_gpus_;
      int layers_per_rank_rotate = L_rotate / N_rotate;
      int remainder_rotate = L_rotate % N_rotate;
      int my_count_rotate = (i < remainder_rotate) ? (layers_per_rank_rotate + 1)
                                           : layers_per_rank_rotate;
      if (my_count_rotate == 0) {
        futures.emplace_back(enqueue_for_gpu(i, [i]() {}));
        continue;
      }
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

        if (num_kv_heads > 1) {
          cpu_startoff_inside_chunks = i * cpu_tp_stride_in_bytes;
        } else if (mode == "sharded" && !is_host_to_device) {
          // sharded D2H: per-rank shard
          int64_t shard = gpu_chunk_sizes_in_bytes_[i] / num_gpus_;
          cpu_startoff_inside_chunks = i * shard;
          gpu_startoff_inside_chunks = i * shard;
          chunk_size = shard;
        } else if (mode == "all_write") {
          // per-rank full-KV region
          cpu_startoff_inside_chunks = i * num_blocks * cpu_block_stride_in_bytes;
        }
        
        // Effective layer range: round_robin assigns subset; else full (layer_id, layer_granularity).
        int eff_start_layer = layer_id;
        int eff_num_layers = layer_granularity;
        if (num_kv_heads == 1 && !is_host_to_device && mode == "layer_parallel") {
          int L_rotate = layer_granularity, N_rotate = num_gpus_;
          int layers_per_rank_rotate = L_rotate / N_rotate;
          int remainder_rotate = L_rotate % N_rotate;
          int my_start_rotate;
          if (i < remainder_rotate) {
            my_start_rotate = i * (layers_per_rank_rotate + 1);
          } else {
            my_start_rotate = remainder_rotate * (layers_per_rank_rotate + 1) +
                          (i - remainder_rotate) * layers_per_rank_rotate;
          }
          eff_start_layer = layer_id + my_start_rotate;
          eff_num_layers = (i < remainder_rotate) ? (layers_per_rank_rotate + 1)
                                              : layers_per_rank_rotate;
        }

        // One region's arguments, this rank's slice. The three-arm switch on
        // backend_type_ that used to be written out here lives once in
        // transfer_backend.cpp; this call site now says only *what* to move
        // and *by which mechanism*.
        RegionTransferArgs args;
        args.num_blocks = num_blocks;
        args.gpu_block_ids = gpu_block_ids;
        args.cpu_block_ids = cpu_block_ids;
        args.start_layer_id = eff_start_layer;
        args.num_layers = eff_num_layers;
        args.kv_dim = kv_dim;
        args.chunk_size_in_bytes = chunk_size;
        args.is_host_to_device = is_host_to_device;
        args.tensor_kind = backend_type_;
        args.gpu_tensor_handler = gpu_tensor_handlers_[i];
        args.gpu_block_stride_in_bytes = gpu_block_strides_in_bytes_[i];
        args.gpu_startoff_inside_chunks = gpu_startoff_inside_chunks;
        args.cpu_ptr = cpu_ptr;
        args.cpu_kv_stride_in_bytes = cpu_kv_stride_in_bytes;
        args.cpu_layer_stride_in_bytes = cpu_layer_stride_in_bytes;
        args.cpu_block_stride_in_bytes = cpu_block_stride_in_bytes;
        args.cpu_startoff_inside_chunks = cpu_startoff_inside_chunks;
        args.transfer_num_cta = transfer_num_cta;

        // A backend never synchronizes, so sync=true is honoured here rather
        // than inside transfer_kv_blocks. Same observable contract: this
        // returns only once the rank's copy has landed.
        launch_region(TransferBackendKind::AUTO, use_ce_transfer, args,
                      streams_[i], ce_config_);
        if (sync) {
          cudaError_t sync_err = cudaStreamSynchronize(streams_[i]);
          if (sync_err != cudaSuccess) {
            record_error(std::string("rank ") + std::to_string(i) +
                         ": stream sync: " + cudaGetErrorString(sync_err));
          }
        }

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
          record_error(std::string("rank ") + std::to_string(i) + ": " +
                       cudaGetErrorString(err));
        }
      } catch (const std::exception &e) {
        record_error(std::string("rank ") + std::to_string(i) + ": " + e.what());
      } catch (...) {
        // A non-std::exception throw previously escaped the packaged_task and
        // resurfaced at f.get() as an opaque failure with `failed` still
        // false, so the caller saw success.
        record_error(std::string("rank ") + std::to_string(i) +
                     ": unknown exception");
      }
    }));
  }

  // Every future must be waited on, even after one of them reports a
  // failure: returning early would let the lambdas' captured references
  // (block-id tensors, error_msg) die while worker threads still read them.
  for (auto &f : futures) {
    try {
      f.get();
    } catch (const std::exception &e) {
      record_error(std::string("future: ") + e.what());
    } catch (...) {
      record_error("future: unknown exception");
    }
  }

  if (failed) {
    std::string msg;
    {
      std::lock_guard<std::mutex> lk(error_mtx);
      msg = error_msg;
    }
    throw std::runtime_error("tp_group_transfer failed: " + msg);
  }
}

void TPTransferThreadGroup::wait_all_streams() {
  // Drains from the pool threads rather than the caller: each pool thread has
  // already had cudaSetDevice() applied once at start-up, so this needs no
  // device juggling on the calling thread and cannot disturb its current
  // device. With sync=false, a copy failure only surfaces here.
  pool_->sync_all_streams();
}

} // namespace flexkv
