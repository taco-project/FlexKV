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

#include "gtensor_handler.cuh"
#include <cuda_runtime.h>

namespace flexkv {

// Template function for transfer, specialized for each backend type
template <BackendType Type>
void transfer_kv_blocks(
    int num_blocks, int start_layer_id, int num_layers, int64_t *gpu_block_ids,
    GTensorHandler gpu_tensor_handler, // Pass by value!
    int64_t gpu_startoff_inside_chunks, int64_t *cpu_block_ids, void *cpu_ptr,
    int64_t cpu_kv_stride_in_bytes, int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_block_stride_in_bytes, int64_t cpu_startoff_inside_chunks,
    int64_t chunk_size_in_bytes, cudaStream_t stream, int transfer_num_cta,
    bool is_host_to_device, bool use_ce_transfer, bool is_mla,
    bool sync = true);

} // namespace flexkv

#ifdef FLEXKV_ENABLE_NVCOMP
#include <cstddef>
#include <cstdint>
#include <vector>

#include "nvcomp/ans.h"

namespace flexkv {

struct ANSTransferContext {
    ANSTransferContext() = default;
    ~ANSTransferContext();

    ANSTransferContext(const ANSTransferContext&) = delete;
    ANSTransferContext& operator=(const ANSTransferContext&) = delete;

    // Chunk geometry — fixed at ctx_create, used to size every buffer below.
    bool initialized = false;       // owns live CUDA resources after ans_ctx_create succeeds
    int device_id = -1;             // CUDA device where the context resources were allocated
    size_t max_num_chunks = 0;      // max chunks this ctx is sized for (buffer capacity ceiling)
    size_t max_chunk_size = 0;      // uncompressed bytes per chunk
    size_t max_comp_chunk_bytes = 0;// max compressed bytes per chunk, rounded up to 16B (staging slot stride)
    size_t comp_temp_bytes = 0;     // nvcomp compression scratch size
    size_t decomp_temp_bytes = 0;   // nvcomp decompression scratch size

    nvcompBatchedANSOpts_t opts{};  // nvcomp ANS options (data_type: float16 for bf16/fp16, uint8 for fp8)

    // GPU buffers — compression (D2H: compress on GPU, then copy payload to CPU)
    void*    d_comp_temp = nullptr;           // nvcomp compression scratch (comp_temp_bytes)
    uint8_t* d_comp_staging_base = nullptr;   // single contiguous alloc backing both staging slots (2 * total)
    uint8_t* d_comp_staging[2] = {nullptr, nullptr}; // double-buffered staging: compressed chunks land here before CPU write
    void**   d_uncomp_ptrs = nullptr;         // device array: ptr to each chunk's uncompressed GPU source
    size_t*  d_uncomp_sizes = nullptr;        // device array: uncompressed size per chunk (all = max_chunk_size)
    void**   d_comp_ptrs[2] = {nullptr, nullptr}; // per-slot device array: ptr into staging for each compressed chunk
    size_t*  d_comp_sizes[2] = {nullptr, nullptr}; // per-slot device array: compressed size per chunk (compress output / decompress input)
    int*     d_overflow = nullptr;            // device counter: incremented when a compressed chunk exceeds the CPU slot during D2H write

    // GPU buffers — decompression (H2D: copy payload from CPU, then decompress on GPU)
    void*           d_decomp_temp = nullptr;        // nvcomp decompression scratch (decomp_temp_bytes)
    void**          d_decomp_ptrs[2] = {nullptr, nullptr}; // per-slot device array: ptr to each chunk's decompressed GPU destination
    size_t*         d_decomp_buf_sizes[2] = {nullptr, nullptr}; // per-slot device array: output buffer capacity per chunk (= max_chunk_size)
    size_t*         d_decomp_act_sizes = nullptr;   // device array: actual decompressed size per chunk (nvcomp output)
    nvcompStatus_t* d_statuses = nullptr;           // device array: per-chunk nvcomp decompress status code

    // Host scratch — staged on host then copied H2D once during ctx_create
    std::vector<void*>  h_ptr_scratch;    // builds d_comp_ptrs (staging slot pointers)
    std::vector<size_t> h_size_scratch;   // builds the uncompressed/decompress-buffer size arrays

    // Kernel launch config
    int write_cpu_grid = 0;         // grid size for the staging->CPU write kernel (from occupancy API)
    int read_cpu_grid = 0;          // grid size for the CPU->staging read kernel (from occupancy API)
    int transfer_sms = 0;           // SMs budgeted for the CPU payload read/write kernels (default 4)

    // Double-buffer pipeline — overlaps nvcomp (de)compress with the CPU payload copy
    cudaStream_t cpu_transfer_stream = nullptr;   // high-priority stream running the CPU read/write kernels
    cudaEvent_t  compress_done[2] = {nullptr, nullptr}; // per-slot: compress (D2H) / CPU-read (H2D) finished
    cudaEvent_t  slot_done[2] = {nullptr, nullptr}; // per-slot: slot free for reuse (CPU-write done / decompress done)
};

void ans_ctx_create(ANSTransferContext* ctx, size_t max_num_chunks,
                    size_t max_chunk_size, int data_type,
                    int transfer_sms = -1);

void ans_ctx_destroy(ANSTransferContext* ctx);

// Returns the total compressed payload bytes for the chunks touched by this call.
template<BackendType Type>
size_t transfer_kv_blocks_ans_comp(
    ANSTransferContext* ctx,
    int num_blocks, int start_layer_id, int num_layers,
    int64_t* gpu_block_ids,
    GTensorHandler gpu_handler,
    int64_t* cpu_block_ids, void* cpu_ptr,
    int64_t cpu_kv_stride_in_bytes, int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_block_stride_in_bytes,
    int64_t chunk_size_in_bytes,
    bool is_mla,
    uint32_t* cpu_size_table_base,
    int64_t   cpu_size_table_block_stride,
    int64_t   cpu_size_table_layer_stride,
    cudaStream_t stream);

// Returns the total compressed payload bytes for the chunks touched by this call.
template<BackendType Type>
size_t transfer_kv_blocks_ans_decomp(
    ANSTransferContext* ctx,
    int num_blocks, int start_layer_id, int num_layers,
    int64_t* gpu_block_ids,
    GTensorHandler gpu_handler,
    int64_t* cpu_block_ids, void* cpu_ptr,
    int64_t cpu_kv_stride_in_bytes, int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_block_stride_in_bytes,
    int64_t chunk_size_in_bytes,
    bool is_mla,
    uint32_t* cpu_size_table_base,
    int64_t   cpu_size_table_block_stride,
    int64_t   cpu_size_table_layer_stride,
    cudaStream_t stream);

} // namespace flexkv

#endif // FLEXKV_ENABLE_NVCOMP
