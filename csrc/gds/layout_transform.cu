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
#include <cuda_runtime.h>
#include <cstdio>
#include "layout_transform.cuh"

namespace flexkv {

#define FLOAT4_PTR(ptr) reinterpret_cast<float4*>(ptr)

template<BackendType Type>
__global__ void layout_transform_kernel(
    int64_t* buffer_base,
    int64_t buffer_layer_stride,
    int64_t buffer_kv_stride,
    int64_t buffer_block_stride,
    int64_t chunk_size,
    GTensorHandler gpu_handler,
    int64_t* gpu_block_ids,
    int num_blocks,
    int num_layers,
    bool is_mla,
    bool buffer_to_target
) {
    int kv_dim = is_mla ? 1 : 2;
    int num_chunks = num_layers * kv_dim * num_blocks;
    int64_t chunk_size_in_float4 = chunk_size * sizeof(int64_t) / sizeof(float4);
    
    for (int chunk_idx = blockIdx.x; chunk_idx < num_chunks; chunk_idx += gridDim.x) {
        int block_local_idx = chunk_idx % num_blocks;
        int layer_idx = chunk_idx / (num_blocks * kv_dim);
        int kv_idx = (chunk_idx % (num_blocks * kv_dim)) / num_blocks;
        
        int64_t gpu_block_idx = gpu_block_ids[block_local_idx];
        
        // Calculate buffer pointer
        int64_t* buffer_ptr = buffer_base + 
                               block_local_idx * buffer_block_stride +
                               layer_idx * buffer_layer_stride +
                               kv_idx * buffer_kv_stride;
        
        // Calculate target GPU pointer
        int64_t* gpu_ptr = ptr_at<Type>(gpu_handler, layer_idx, kv_idx, gpu_block_idx);
        
        int64_t* src_ptr = buffer_to_target ? buffer_ptr : gpu_ptr;
        int64_t* dst_ptr = buffer_to_target ? gpu_ptr : buffer_ptr;
        
        for (int64_t idx = threadIdx.x; idx < chunk_size_in_float4; idx += blockDim.x) {
            float4 element = FLOAT4_PTR(src_ptr)[idx];
            FLOAT4_PTR(dst_ptr)[idx] = element;
        }
    }
}

template<BackendType Type>
void launch_layout_transform_kernel(
    int64_t* buffer_base,
    int64_t buffer_layer_stride,
    int64_t buffer_kv_stride,
    int64_t buffer_block_stride,
    int64_t chunk_size,
    GTensorHandler gpu_handler,
    int64_t* gpu_block_ids,
    int num_blocks,
    int num_layers,
    bool is_mla,
    bool buffer_to_target,
    cudaStream_t stream
) {
    if (num_blocks == 0 || num_layers == 0) return;
    
    int block_size = 128; // TODO: warp level 
    int kv_dim = is_mla ? 1 : 2;
    int num_chunks = num_layers * kv_dim * num_blocks;
    
    int device_id;
    cudaGetDevice(&device_id);
    
    int max_blocks_per_sm = 1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, layout_transform_kernel<Type>, block_size, 0);

    int available_blocks_per_sm = 4;
    
    int grid_size = std::min(num_chunks, available_blocks_per_sm * max_blocks_per_sm);
    
    layout_transform_kernel<Type><<<grid_size, block_size, 0, stream>>>(
        buffer_base,
        buffer_layer_stride,
        buffer_kv_stride,
        buffer_block_stride,
        chunk_size,
        gpu_handler,
        gpu_block_ids,
        num_blocks,
        num_layers,
        is_mla,
        buffer_to_target
    );

    // 捕获并打印 kernel 启动或执行错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "layout_transform_kernel launch failed: %s\n", cudaGetErrorString(err));
        return;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "layout_transform_kernel execution failed: %s\n", cudaGetErrorString(err));
        return;
    }
}

// Explicit template instantiations
template void launch_layout_transform_kernel<BackendType::VLLM>(
    int64_t*, int64_t, int64_t, int64_t, int64_t, GTensorHandler, int64_t*,
    int, int, bool, bool, cudaStream_t);

template void launch_layout_transform_kernel<BackendType::TRTLLM>(
    int64_t*, int64_t, int64_t, int64_t, int64_t, GTensorHandler, int64_t*,
    int, int, bool, bool, cudaStream_t);

template void launch_layout_transform_kernel<BackendType::SGLANG>(
    int64_t*, int64_t, int64_t, int64_t, int64_t, GTensorHandler, int64_t*,
    int, int, bool, bool, cudaStream_t);

} // namespace flexkv

