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

#include "transfer.cuh"

namespace flexkv {

#define FLOAT4_PTR(ptr) reinterpret_cast<float4 *>(ptr)

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

  // 计算warp信息
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int warps_per_block = blockDim.x / 32;
  int total_warps = gridDim.x * warps_per_block;
  
  // 每个warp处理一个chunk
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

    // warp内的线程协作拷贝数据
    for (int64_t idx = lane_id; idx < copy_size_in_float4; idx += 32) {
      float4 element;
      asm volatile("ld.global.nc.v4.f32 {%0,%1,%2,%3},[%4];" 
                   : "=f"(element.x), "=f"(element.y), "=f"(element.z), "=f"(element.w)
                   : "l"(&FLOAT4_PTR(src_chunk_ptr)[idx]) 
                   : "memory");
      asm volatile("st.global.cg.v4.f32 [%0],{%1,%2,%3,%4};" 
                   :: "l"(&FLOAT4_PTR(dst_chunk_ptr)[idx]), 
                      "f"(element.x), "f"(element.y), "f"(element.z), "f"(element.w)
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
    cudaStream_t stream, int transfer_sms, bool is_host_to_device,
    bool use_ce_transfer, bool is_mla, bool sync) {

  int block_size = 1024;
  static int max_blocks_per_sm = -1;
  if (max_blocks_per_sm == -1) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, transfer_kv_blocks_kernel<Type>, block_size, 0);
  }

  // if (transfer_sms == -1) {
  //   transfer_sms = 4;
  // }

  int block_count = transfer_sms; // transfer_sms * max_blocks_per_sm;

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

  // CE transfer mode (Copy Engine using cudaMemcpyAsync)
  if (use_ce_transfer) {
    int kv_dim = is_mla ? 1 : 2;
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

          if (is_host_to_device) {
            cudaMemcpyAsync(gpu_chunk_ptr, cpu_chunk_ptr, chunk_size_in_bytes,
                            cudaMemcpyHostToDevice, stream);
          } else {
            cudaMemcpyAsync(cpu_chunk_ptr, gpu_chunk_ptr, chunk_size_in_bytes,
                            cudaMemcpyDeviceToHost, stream);
          }
        }
      }
    }
  } else {
    // Custom kernel transfer
    transfer_kv_blocks_kernel<Type><<<gridDim, blockDim, 0, stream>>>(
        num_blocks, start_layer_id, num_layers, gpu_block_ids,
        gpu_tensor_handler, gpu_startoff_inside_chunks_int64, cpu_block_ids,
        cpu_ptr_int64, cpu_kv_stride_int64, cpu_layer_stride_int64,
        cpu_block_stride_int64, cpu_startoff_inside_chunks_int64,
        chunk_size_in_int64, is_mla, is_host_to_device);
  }
  if (sync) {
    cudaStreamSynchronize(stream);
  }
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
