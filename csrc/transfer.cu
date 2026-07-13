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
#include <cuda_runtime.h>
#include <torch/extension.h>

#include "monitoring/metrics_manager.h"
#include "transfer.cuh"
#include "transfer_debug.h"

namespace flexkv {

#define FLOAT4_PTR(ptr) reinterpret_cast<float4 *>(ptr)

constexpr int kFloat4AlignBytes = 16;
constexpr int kInt64AlignBytes = 8;

static bool use_float4_kernel_path(int64_t chunk_size_in_bytes,
                                   int64_t gpu_startoff_inside_chunks,
                                   int64_t cpu_startoff_inside_chunks) {
  return (chunk_size_in_bytes % kFloat4AlignBytes == 0) &&
         (gpu_startoff_inside_chunks % kFloat4AlignBytes == 0) &&
         (cpu_startoff_inside_chunks % kFloat4AlignBytes == 0);
}

// 8-byte (int64) copy path for MLA-sharded D2H where per-TP shard offsets are
// 8-aligned but not 16-aligned (e.g. DSv4 bytes_per_page_padded=37440, TP=8).
template <BackendType Type>
__global__ void transfer_kv_blocks_kernel_8b(
    int num_blocks, int start_layer_id, int num_layers, int64_t *gpu_block_ids,
    GTensorHandler gpu_handler, int64_t gpu_startoff_inside_chunks,
    int64_t *cpu_block_ids, int64_t *cpu_ptr, int64_t cpu_kv_stride,
    int64_t cpu_layer_stride, int64_t cpu_block_stride,
    int64_t cpu_startoff_inside_chunks, int64_t copy_size, bool is_mla,
    bool is_host_to_device) {
  int kv_dim = is_mla ? 1 : 2;
  int num_chunks = num_layers * kv_dim * num_blocks;

  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int warps_per_block = blockDim.x / 32;
  int total_warps = gridDim.x * warps_per_block;

  for (int chunk_idx = blockIdx.x * warps_per_block + warp_id;
       chunk_idx < num_chunks; chunk_idx += total_warps) {
    int layer_idx = start_layer_id + chunk_idx / (num_blocks * kv_dim);
    int kv_idx = (chunk_idx % (num_blocks * kv_dim)) / num_blocks;
    int gpu_block_idx = gpu_block_ids[chunk_idx % num_blocks];
    int cpu_block_idx = cpu_block_ids[chunk_idx % num_blocks];

    int64_t *cpu_chunk_ptr =
        cpu_ptr + layer_idx * cpu_layer_stride + kv_idx * cpu_kv_stride +
        cpu_block_idx * cpu_block_stride + cpu_startoff_inside_chunks;

    int64_t *gpu_ptr =
        ptr_at<Type>(gpu_handler, layer_idx, kv_idx, gpu_block_idx);
    int64_t *gpu_chunk_ptr =
        reinterpret_cast<int64_t *>(gpu_ptr) + gpu_startoff_inside_chunks;

    int64_t *src_chunk_ptr = is_host_to_device ? cpu_chunk_ptr : gpu_chunk_ptr;
    int64_t *dst_chunk_ptr = is_host_to_device ? gpu_chunk_ptr : cpu_chunk_ptr;

    // Use explicit PTX ld/st (same as float4 path) so D2H can write pinned host
    // memory from device; plain C++ stores may fault on unmapped host pointers.
    for (int64_t idx = lane_id; idx < copy_size; idx += 32) {
      int64_t element;
      asm volatile("ld.global.nc.u64 %0, [%1];"
                   : "=l"(element)
                   : "l"(&src_chunk_ptr[idx])
                   : "memory");
      asm volatile("st.global.cg.u64 [%0], %1;"
                   :: "l"(&dst_chunk_ptr[idx]), "l"(element)
                   : "memory");
    }
  }
}

// Templated CUDA kernel - backend type determined at compile time
template <BackendType Type>
__global__ void transfer_kv_blocks_kernel(
    int num_blocks, int start_layer_id, int num_layers, int64_t *gpu_block_ids,
    GTensorHandler gpu_handler, int64_t gpu_startoff_inside_chunks,
    int64_t *cpu_block_ids, int64_t *cpu_ptr, int64_t cpu_kv_stride,
    int64_t cpu_layer_stride, int64_t cpu_block_stride,
    int64_t cpu_startoff_inside_chunks, int64_t copy_size, bool is_mla,
    bool is_host_to_device) {
  int kv_dim = is_mla ? 1 : 2;
  int num_chunks = num_layers * kv_dim * num_blocks;
  int64_t copy_size_in_float4 = copy_size * sizeof(int64_t) / sizeof(float4);

  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int warps_per_block = blockDim.x / 32;
  int total_warps = gridDim.x * warps_per_block;

  for (int chunk_idx = blockIdx.x * warps_per_block + warp_id;
       chunk_idx < num_chunks; chunk_idx += total_warps) {
    int layer_idx = start_layer_id + chunk_idx / (num_blocks * kv_dim);
    int kv_idx = (chunk_idx % (num_blocks * kv_dim)) / num_blocks;
    int gpu_block_idx = gpu_block_ids[chunk_idx % num_blocks];
    int cpu_block_idx = cpu_block_ids[chunk_idx % num_blocks];

    int64_t *cpu_chunk_ptr =
        cpu_ptr + layer_idx * cpu_layer_stride + kv_idx * cpu_kv_stride +
        cpu_block_idx * cpu_block_stride + cpu_startoff_inside_chunks;

    // Use template specialization to compute gpu pointer
    int64_t *gpu_ptr =
        ptr_at<Type>(gpu_handler, layer_idx, kv_idx, gpu_block_idx);
    int64_t *gpu_chunk_ptr =
        reinterpret_cast<int64_t *>(gpu_ptr) + gpu_startoff_inside_chunks;

    int64_t *src_chunk_ptr = is_host_to_device ? cpu_chunk_ptr : gpu_chunk_ptr;
    int64_t *dst_chunk_ptr = is_host_to_device ? gpu_chunk_ptr : cpu_chunk_ptr;

    for (int64_t idx = lane_id; idx < copy_size_in_float4; idx += 32) {
      float4 element;
      asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3},[%4];"
                   : "=f"(element.x), "=f"(element.y), "=f"(element.z),
                     "=f"(element.w)
                   : "l"(&FLOAT4_PTR(src_chunk_ptr)[idx])
                   : "memory");
      asm volatile("st.global.cg.v4.f32 [%0],{%1,%2,%3,%4};" ::"l"(
                       &FLOAT4_PTR(dst_chunk_ptr)[idx]),
                   "f"(element.x), "f"(element.y), "f"(element.z),
                   "f"(element.w)
                   : "memory");
    }
  }
}

// Templated host function
template <BackendType Type>
void transfer_kv_blocks(
    int num_blocks, int start_layer_id, int num_layers, int64_t *gpu_block_ids,
    GTensorHandler gpu_tensor_handler, int64_t gpu_startoff_inside_chunks,
    int64_t *cpu_block_ids, void *cpu_ptr, int64_t cpu_kv_stride_in_bytes,
    int64_t cpu_layer_stride_in_bytes, int64_t cpu_block_stride_in_bytes,
    int64_t cpu_startoff_inside_chunks, int64_t chunk_size_in_bytes,
    cudaStream_t stream, int transfer_num_cta, bool is_host_to_device,
    bool use_ce_transfer, bool is_mla, bool sync) {

  int block_size = 1024;

  int block_count = transfer_num_cta;

  int64_t *cpu_ptr_int64 = reinterpret_cast<int64_t *>(cpu_ptr);
  int64_t cpu_kv_stride_int64 = cpu_kv_stride_in_bytes / sizeof(int64_t);
  int64_t cpu_block_stride_int64 = cpu_block_stride_in_bytes / sizeof(int64_t);
  int64_t cpu_layer_stride_int64 = cpu_layer_stride_in_bytes / sizeof(int64_t);
  int64_t cpu_startoff_inside_chunks_int64 =
      cpu_startoff_inside_chunks / sizeof(int64_t);
  int64_t gpu_startoff_inside_chunks_int64 =
      gpu_startoff_inside_chunks / sizeof(int64_t);
  int64_t chunk_size_in_int64 = chunk_size_in_bytes / sizeof(int64_t);

  dim3 blockDim(block_size);
  dim3 gridDim(block_count);

  const int kv_dim = is_mla ? 1 : 2;
  const int gpu_idx = d2h_debug_gpu_index;
  // const int64_t gpu_blk_min = min_block_id(gpu_block_ids, num_blocks);
  // const int64_t gpu_blk_max = max_block_id(gpu_block_ids, num_blocks);
  // const int64_t cpu_blk_min = min_block_id(cpu_block_ids, num_blocks);
  // const int64_t cpu_blk_max = max_block_id(cpu_block_ids, num_blocks);
  // const int64_t est_cpu_end_bytes =
  //     (static_cast<int64_t>(start_layer_id) + num_layers) *
  //         cpu_layer_stride_in_bytes +
  //     cpu_blk_max * cpu_block_stride_in_bytes + cpu_startoff_inside_chunks +
  //     chunk_size_in_bytes;

  // FLEXKV_D2H_LOG(
  //     "transfer_kv_blocks ENTER gpu=%d backend=%s ce=%d h2d=%d mla=%d "
  //     "blocks=%d layers=[%d,%d) chunk=%lld gpu_off=%lld cpu_off=%lld "
  //     "cpu_strides(kv/layer/block)=%lld/%lld/%lld gpu_blk=[%lld,%lld] "
  //     "cpu_blk=[%lld,%lld] est_cpu_end=%lld sync=%d",
  //     gpu_idx, backend_name(Type), use_ce_transfer ? 1 : 0,
  //     is_host_to_device ? 1 : 0, is_mla ? 1 : 0, num_blocks, start_layer_id,
  //     start_layer_id + num_layers, static_cast<long long>(chunk_size_in_bytes),
  //     static_cast<long long>(gpu_startoff_inside_chunks),
  //     static_cast<long long>(cpu_startoff_inside_chunks),
  //     static_cast<long long>(cpu_kv_stride_in_bytes),
  //     static_cast<long long>(cpu_layer_stride_in_bytes),
  //     static_cast<long long>(cpu_block_stride_in_bytes),
  //     static_cast<long long>(gpu_blk_min), static_cast<long long>(gpu_blk_max),
  //     static_cast<long long>(cpu_blk_min), static_cast<long long>(cpu_blk_max),
  //     static_cast<long long>(est_cpu_end_bytes), sync ? 1 : 0);

  // log_pointer_attributes("cpu_base", cpu_ptr);
  // log_pointer_attributes("gpu_block_ids", gpu_block_ids);
  // log_pointer_attributes("cpu_block_ids", cpu_block_ids);

  // CE transfer mode (Copy Engine using cudaMemcpyAsync)
  if (use_ce_transfer) {
    const int total_memcpy =
        num_layers * kv_dim * num_blocks;
    int memcpy_idx = 0;
    int fail_layer = -1;
    int fail_kv = -1;
    int fail_block = -1;
    int64_t fail_gpu_blk = -1;
    int64_t fail_cpu_blk = -1;
    void *fail_src = nullptr;
    void *fail_dst = nullptr;
    cudaError_t first_async_err = cudaSuccess;

    for (int i = 0; i < num_layers; i++) {
      for (int j = 0; j < kv_dim; j++) {
        for (int k = 0; k < num_blocks; k++) {
          int64_t gpu_block_idx = gpu_block_ids[k];
          int64_t cpu_block_idx = cpu_block_ids[k];

          int64_t *cpu_chunk_ptr =
              cpu_ptr_int64 + (i + start_layer_id) * cpu_layer_stride_int64 +
              j * cpu_kv_stride_int64 + cpu_block_idx * cpu_block_stride_int64 +
              cpu_startoff_inside_chunks_int64;

          int64_t *gpu_ptr = ptr_at<Type>(gpu_tensor_handler,
                                          i + start_layer_id, j, gpu_block_idx);
          int64_t *gpu_chunk_ptr = reinterpret_cast<int64_t *>(gpu_ptr) +
                                   gpu_startoff_inside_chunks_int64;

          const void *src =
              is_host_to_device ? static_cast<const void *>(cpu_chunk_ptr)
                                : static_cast<const void *>(gpu_chunk_ptr);
          void *dst = is_host_to_device
                          ? static_cast<void *>(gpu_chunk_ptr)
                          : static_cast<void *>(cpu_chunk_ptr);
          const cudaMemcpyKind kind = is_host_to_device ? cudaMemcpyHostToDevice
                                                        : cudaMemcpyDeviceToHost;

          if (d2h_debug_enabled() &&
              (memcpy_idx == 0 || memcpy_idx == total_memcpy / 2 ||
               memcpy_idx == total_memcpy - 1)) {
            // FLEXKV_D2H_LOG(
            //     "CE sample gpu=%d idx=%d/%d layer=%d kv=%d blk=%d "
            //     "gpu_blk=%lld cpu_blk=%lld src=%p dst=%p bytes=%lld",
            //     gpu_idx, memcpy_idx, total_memcpy, i + start_layer_id, j, k,
            //     static_cast<long long>(gpu_block_idx),
            //     static_cast<long long>(cpu_block_idx), src, dst,
            //     static_cast<long long>(chunk_size_in_bytes));
          }

          cudaError_t memcpy_err =
              cudaMemcpyAsync(dst, src, chunk_size_in_bytes, kind, stream);
          if (memcpy_err != cudaSuccess && first_async_err == cudaSuccess) {
            first_async_err = memcpy_err;
            fail_layer = i + start_layer_id;
            fail_kv = j;
            fail_block = k;
            fail_gpu_blk = gpu_block_idx;
            fail_cpu_blk = cpu_block_idx;
            fail_src = const_cast<void *>(src);
            fail_dst = dst;
          }

          FLEXKV_GPU_CPU_TRANSFER(is_host_to_device, chunk_size_in_bytes);
          memcpy_idx++;
        }
      }
    }

    // FLEXKV_D2H_LOG("CE submitted gpu=%d memcpy_count=%d first_async_err=%s",
    //                gpu_idx, total_memcpy,
    //                cudaGetErrorString(first_async_err));

    if (first_async_err != cudaSuccess) {
      // FLEXKV_D2H_LOG(
      //     "CE memcpyAsync FAIL gpu=%d layer=%d kv=%d blk_idx=%d "
      //     "gpu_blk=%lld cpu_blk=%lld src=%p dst=%p err=%s",
      //     gpu_idx, fail_layer, fail_kv, fail_block,
      //     static_cast<long long>(fail_gpu_blk),
      //     static_cast<long long>(fail_cpu_blk), fail_src, fail_dst,
      //     cudaGetErrorString(first_async_err));
    }
  } else {
    const bool float4_path = use_float4_kernel_path(
        chunk_size_in_bytes, gpu_startoff_inside_chunks,
        cpu_startoff_inside_chunks);

    // FLEXKV_D2H_LOG(
    //     "kernel path gpu=%d float4=%d align(chunk/gpu/cpu)=%lld/%lld/%lld "
    //     "cta=%d",
    //     gpu_idx, float4_path ? 1 : 0,
    //     static_cast<long long>(chunk_size_in_bytes % kFloat4AlignBytes),
    //     static_cast<long long>(gpu_startoff_inside_chunks % kFloat4AlignBytes),
    //     static_cast<long long>(cpu_startoff_inside_chunks % kFloat4AlignBytes),
    //     block_count);

    if (float4_path) {
      transfer_kv_blocks_kernel<Type><<<gridDim, blockDim, 0, stream>>>(
          num_blocks, start_layer_id, num_layers, gpu_block_ids,
          gpu_tensor_handler, gpu_startoff_inside_chunks_int64, cpu_block_ids,
          cpu_ptr_int64, cpu_kv_stride_int64, cpu_layer_stride_int64,
          cpu_block_stride_int64, cpu_startoff_inside_chunks_int64,
          chunk_size_in_int64, is_mla, is_host_to_device);
    } else {
      transfer_kv_blocks_kernel_8b<Type><<<gridDim, blockDim, 0, stream>>>(
          num_blocks, start_layer_id, num_layers, gpu_block_ids,
          gpu_tensor_handler, gpu_startoff_inside_chunks_int64, cpu_block_ids,
          cpu_ptr_int64, cpu_kv_stride_int64, cpu_layer_stride_int64,
          cpu_block_stride_int64, cpu_startoff_inside_chunks_int64,
          chunk_size_in_int64, is_mla, is_host_to_device);
    }

    cudaError_t launch_err = cudaGetLastError();
    if (launch_err != cudaSuccess) {
      // FLEXKV_D2H_LOG("kernel launch FAIL gpu=%d err=%s", gpu_idx,
      //                cudaGetErrorString(launch_err));
    } else {
      // FLEXKV_D2H_LOG("kernel launched gpu=%d variant=%s", gpu_idx,
      //                float4_path ? "float4" : "8b");
    }

    int64_t actual_chunk_bytes = float4_path
        ? (chunk_size_in_int64 * sizeof(int64_t) / sizeof(float4)) *
              sizeof(float4)
        : chunk_size_in_bytes;
    FLEXKV_GPU_CPU_TRANSFER(
        is_host_to_device,
        actual_chunk_bytes * static_cast<int64_t>(num_layers) *
            static_cast<int64_t>(kv_dim) * static_cast<int64_t>(num_blocks));
  }

  if (sync) {
    const auto sync_t0 = std::chrono::steady_clock::now();
    // FLEXKV_D2H_LOG("stream sync BEGIN gpu=%d ce=%d stream=%p", gpu_idx,
    //                use_ce_transfer ? 1 : 0, static_cast<void *>(stream));
    cudaError_t sync_err = cudaStreamSynchronize(stream);
    const auto sync_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::steady_clock::now() - sync_t0)
                             .count();
    cudaError_t post_sync_err = cudaGetLastError();

    if (sync_err != cudaSuccess) {
      // FLEXKV_D2H_LOG(
      //     "stream sync FAIL gpu=%d ce=%d elapsed_ms=%lld sync_err=%s post_err=%s",
      //     gpu_idx, use_ce_transfer ? 1 : 0, static_cast<long long>(sync_ms),
      //     cudaGetErrorString(sync_err), cudaGetErrorString(post_sync_err));
    } else if (post_sync_err != cudaSuccess) {
      // FLEXKV_D2H_LOG(
      //     "stream sync OK but pending err gpu=%d ce=%d elapsed_ms=%lld err=%s",
      //     gpu_idx, use_ce_transfer ? 1 : 0, static_cast<long long>(sync_ms),
      //     cudaGetErrorString(post_sync_err));
    } else {
      // FLEXKV_D2H_LOG("stream sync END gpu=%d ce=%d elapsed_ms=%lld ok", gpu_idx,
      //                use_ce_transfer ? 1 : 0, static_cast<long long>(sync_ms));
    }
  }

  // FLEXKV_D2H_LOG("transfer_kv_blocks LEAVE gpu=%d ce=%d", gpu_idx,
  //                use_ce_transfer ? 1 : 0);
}

// Explicit template instantiations
template void transfer_kv_blocks<BackendType::VLLM>(int, int, int, int64_t *,
                                                    GTensorHandler, int64_t,
                                                    int64_t *, void *, int64_t,
                                                    int64_t, int64_t, int64_t,
                                                    int64_t, cudaStream_t, int,
                                                    bool, bool, bool, bool);

template void transfer_kv_blocks<BackendType::TRTLLM>(
    int, int, int, int64_t *, GTensorHandler, int64_t, int64_t *, void *,
    int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t, int, bool, bool,
    bool, bool);

template void transfer_kv_blocks<BackendType::SGLANG>(
    int, int, int, int64_t *, GTensorHandler, int64_t, int64_t *, void *,
    int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t, int, bool, bool,
    bool, bool);

} // namespace flexkv
