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
#include <cooperative_groups.h>

#include "transfer.cuh"

// nvCOMP Device API headers
#include <nvcomp/device/backend_common.hpp>
#include <nvcomp/device/operators.hpp>
#include <nvcomp/device/backend_api.hpp>
#include <nvcomp/device/detail/lz4/compress_device.cuh>
#include <nvcomp/device/detail/lz4/decompress_device.cuh>
#include <nvcomp/device/user_api.hpp>

namespace cg = cooperative_groups;

namespace flexkv {

#define FLOAT4_PTR(ptr) reinterpret_cast<float4 *>(ptr)

// Templated CUDA kernel - backend type determined at compile time (original non-compressed version)
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

// Compressed transfer kernel - D2H compresses, H2D decompresses
template <BackendType Type>
__global__ void transfer_kv_blocks_compressed_kernel(
    int num_blocks, int start_layer_id, int num_layers, int64_t *gpu_block_ids,
    GTensorHandler gpu_handler, int64_t gpu_startoff_inside_chunks,
    int64_t *cpu_block_ids, int64_t *cpu_ptr, int64_t cpu_kv_stride,
    int64_t cpu_layer_stride, int64_t cpu_block_stride,
    int64_t cpu_startoff_inside_chunks, int64_t chunk_size_in_bytes, bool is_mla,
    bool is_host_to_device, uint8_t *compress_tmp_buffer) {
  
  using namespace nvcomp::device;
  using namespace nvcomp::device::detail;
  
  // Use nvCOMP's WarpGroup type
  using NvcompWarpGroup = nvcomp::device::detail::WarpGroup;
  
  int kv_dim = is_mla ? 1 : 2;
  int num_chunks = num_layers * kv_dim * num_blocks;

  // 计算warp信息
  int warp_id = threadIdx.x / 32;
  int lane_id = threadIdx.x % 32;
  int warps_per_block = blockDim.x / 32;
  int global_warp_id = blockIdx.x * warps_per_block + warp_id;
  int total_warps = gridDim.x * warps_per_block;
  
  // Get cooperative group warp handle - must match nvCOMP's WarpGroup type
  auto block = cg::this_thread_block();
  NvcompWarpGroup warp = cg::tiled_partition<32>(block);
  
  // Calculate shared memory and temp buffer per warp
  extern __shared__ uint8_t shared_mem[];
  
  // Decompress需要的共享内存大小
  constexpr size_t decomp_shmem_per_warp = ShmemSizeGroup<nvcomp_algo::lz4, nvcomp_direction::decompress>::execute();
  uint8_t* my_shmem = shared_mem + warp_id * decomp_shmem_per_warp;
  
  // Compress需要的临时缓冲区（哈希表）大小
  constexpr size_t comp_tmp_per_warp = TmpSizeGroup<nvcomp_grouptype::warp, nvcomp_algo::lz4, nvcomp_direction::compress>::execute(MAX_UNCOMP_CHUNK_SIZE, nvcomp_datatype::uint8);
  uint8_t* my_tmp = compress_tmp_buffer + global_warp_id * comp_tmp_per_warp;
  
  // 每个warp处理一个chunk
  for (int chunk_idx = global_warp_id; chunk_idx < num_chunks; chunk_idx += total_warps) {
    int layer_idx = start_layer_id + chunk_idx / (num_blocks * kv_dim);
    int kv_idx = (chunk_idx % (num_blocks * kv_dim)) / num_blocks;
    int gpu_block_idx = gpu_block_ids[chunk_idx % num_blocks];
    int cpu_block_idx = cpu_block_ids[chunk_idx % num_blocks];

    // CPU chunk pointer (存储压缩数据)
    uint8_t *cpu_chunk_ptr = reinterpret_cast<uint8_t*>(
        cpu_ptr + layer_idx * cpu_layer_stride + kv_idx * cpu_kv_stride +
        cpu_block_idx * cpu_block_stride + cpu_startoff_inside_chunks);

    // GPU chunk pointer (原始数据)
    int64_t *gpu_ptr = ptr_at<Type>(gpu_handler, layer_idx, kv_idx, gpu_block_idx);
    uint8_t *gpu_chunk_ptr = reinterpret_cast<uint8_t*>(
        reinterpret_cast<int64_t *>(gpu_ptr) + gpu_startoff_inside_chunks);

    if (is_host_to_device) {
      // H2D: CPU compressed data -> GPU decompressed data
      // 读取压缩大小 (存储在 CPU chunk 的前 4 字节)
      uint32_t comp_size = *reinterpret_cast<uint32_t*>(cpu_chunk_ptr);
      
      // 压缩数据在头部之后
      const uint8_t* comp_data = cpu_chunk_ptr + COMPRESS_HEADER_SIZE;
      
      // 解压到GPU - 使用正确的 API 签名
      size_t decomp_size = chunk_size_in_bytes;
      Decompress<NvcompWarpGroup, nvcomp_datatype::uint8, nvcomp_algo::lz4>().execute(
          comp_data,
          gpu_chunk_ptr,
          static_cast<size_t>(comp_size),
          &decomp_size,
          my_shmem,
          nullptr,  // LZ4解压不需要tmp buffer
          warp);
    } else {
      // D2H: GPU raw data -> CPU compressed data
      // 压缩数据存储在头部之后
      uint8_t* comp_data = cpu_chunk_ptr + COMPRESS_HEADER_SIZE;
      
      size_t comp_size;
      Compress<NvcompWarpGroup, nvcomp_datatype::uint8, nvcomp_algo::lz4>().execute(
          gpu_chunk_ptr,
          comp_data,
          chunk_size_in_bytes,
          &comp_size,
          nullptr,  // LZ4压缩不需要shared memory
          my_tmp,
          MAX_UNCOMP_CHUNK_SIZE,
          warp);
      
      // 写入压缩大小到头部 (只有 lane 0 写)
      if (lane_id == 0) {
        *reinterpret_cast<uint32_t*>(cpu_chunk_ptr) = static_cast<uint32_t>(comp_size);
      }
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
    bool use_ce_transfer, bool is_mla, bool sync,
    bool enable_compression, void* compress_tmp_buffer) {

  int block_size = 1024;
  static int max_blocks_per_sm = -1;
  if (max_blocks_per_sm == -1) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, transfer_kv_blocks_kernel<Type>, block_size, 0);
  }

  int block_count = transfer_sms;

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

  // CE transfer mode (Copy Engine using cudaMemcpyAsync) - doesn't support compression
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
  } else if (enable_compression && compress_tmp_buffer != nullptr) {
    // Compressed transfer using nvCOMP LZ4
    using namespace nvcomp::device::detail;
    
    // Calculate shared memory size for decompression
    constexpr size_t decomp_shmem_per_warp = 
        nvcomp::device::detail::ShmemSizeGroup<nvcomp::device::nvcomp_algo::lz4, 
                                                nvcomp::device::nvcomp_direction::decompress>::execute();
    int warps_per_block = block_size / 32;
    size_t shmem_size = warps_per_block * decomp_shmem_per_warp;
    
    transfer_kv_blocks_compressed_kernel<Type><<<gridDim, blockDim, shmem_size, stream>>>(
        num_blocks, start_layer_id, num_layers, gpu_block_ids,
        gpu_tensor_handler, gpu_startoff_inside_chunks_int64, cpu_block_ids,
        cpu_ptr_int64, cpu_kv_stride_int64, cpu_layer_stride_int64,
        cpu_block_stride_int64, cpu_startoff_inside_chunks_int64,
        chunk_size_in_bytes, is_mla, is_host_to_device,
        reinterpret_cast<uint8_t*>(compress_tmp_buffer));
  } else {
    // Custom kernel transfer (non-compressed)
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

// Helper function to get required temp buffer size for compression
size_t get_compress_tmp_buffer_size(int num_warps) {
  using namespace nvcomp::device::detail;
  constexpr size_t tmp_per_warp = TmpSizeGroup<nvcomp::device::nvcomp_grouptype::warp, 
                                                nvcomp::device::nvcomp_algo::lz4, 
                                                nvcomp::device::nvcomp_direction::compress>::execute(
      MAX_UNCOMP_CHUNK_SIZE, nvcomp::device::nvcomp_datatype::uint8);
  return num_warps * tmp_per_warp;
}

// Explicit template instantiations
template void transfer_kv_blocks<BackendType::VLLM>(int, int, int, int64_t *,
                                                    GTensorHandler, int64_t,
                                                    int64_t *, void *, int64_t,
                                                    int64_t, int64_t, int64_t,
                                                    int64_t, cudaStream_t, int,
                                                    bool, bool, bool, bool,
                                                    bool, void*);

template void transfer_kv_blocks<BackendType::TRTLLM>(
    int, int, int, int64_t *, GTensorHandler, int64_t, int64_t *, void *,
    int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t, int, bool, bool,
    bool, bool, bool, void*);

template void transfer_kv_blocks<BackendType::SGLANG>(
    int, int, int, int64_t *, GTensorHandler, int64_t, int64_t *, void *,
    int64_t, int64_t, int64_t, int64_t, int64_t, cudaStream_t, int, bool, bool,
    bool, bool, bool, void*);

} // namespace flexkv
