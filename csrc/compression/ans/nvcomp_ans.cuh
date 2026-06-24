/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "transfer.cuh"

#include "nvcomp/ans.h"

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <vector>

namespace flexkv {

struct ANSTransferContext {
  ANSTransferContext() = default;
  ~ANSTransferContext();

  ANSTransferContext(const ANSTransferContext&) = delete;
  ANSTransferContext& operator=(const ANSTransferContext&) = delete;

  // Chunk geometry -- fixed at ctx_create, used to size every buffer below.
  bool initialized = false;
  int device_id = -1;
  size_t max_num_chunks = 0;
  size_t max_chunk_size = 0;
  size_t max_comp_chunk_bytes = 0;
  size_t comp_temp_bytes = 0;
  size_t decomp_temp_bytes = 0;

  nvcompBatchedANSOpts_t opts{};

  // GPU buffers -- compression (D2H: compress on GPU, then copy payload to CPU)
  void* d_comp_temp = nullptr;
  uint8_t* d_comp_staging_base = nullptr;
  uint8_t* d_comp_staging[2] = {nullptr, nullptr};
  void** d_uncomp_ptrs = nullptr;
  size_t* d_uncomp_sizes = nullptr;
  void** d_comp_ptrs[2] = {nullptr, nullptr};
  size_t* d_comp_sizes[2] = {nullptr, nullptr};
  int* d_overflow = nullptr;

  // GPU buffers -- decompression (H2D: copy payload from CPU, then decompress)
  void* d_decomp_temp = nullptr;
  void** d_decomp_ptrs[2] = {nullptr, nullptr};
  size_t* d_decomp_buf_sizes[2] = {nullptr, nullptr};
  size_t* d_decomp_act_sizes = nullptr;
  nvcompStatus_t* d_statuses = nullptr;

  // Host scratch -- staged on host then copied H2D once during ctx_create.
  std::vector<void*> h_ptr_scratch;
  std::vector<size_t> h_size_scratch;

  // Kernel launch config.
  int write_cpu_grid = 0;
  int read_cpu_grid = 0;
  int transfer_sms = 0;

  // Double-buffer pipeline.
  cudaStream_t cpu_transfer_stream = nullptr;
  cudaEvent_t compress_done[2] = {nullptr, nullptr};
  cudaEvent_t slot_done[2] = {nullptr, nullptr};
};

void ans_ctx_create(ANSTransferContext* ctx, size_t max_num_chunks,
                    size_t max_chunk_size, int data_type,
                    int transfer_sms = -1);

void ans_ctx_destroy(ANSTransferContext* ctx);

template <BackendType Type>
size_t transfer_kv_blocks_ans_comp(
    ANSTransferContext* ctx,
    int num_blocks,
    int start_layer_id,
    int num_layers,
    int64_t* gpu_block_ids,
    GTensorHandler gpu_handler,
    int64_t* cpu_block_ids,
    void* cpu_ptr,
    int64_t cpu_kv_stride_in_bytes,
    int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_block_stride_in_bytes,
    int64_t chunk_size_in_bytes,
    bool is_mla,
    uint32_t* cpu_size_table_base,
    int64_t cpu_size_table_block_stride,
    int64_t cpu_size_table_layer_stride,
    cudaStream_t stream);

template <BackendType Type>
size_t transfer_kv_blocks_ans_decomp(
    ANSTransferContext* ctx,
    int num_blocks,
    int start_layer_id,
    int num_layers,
    int64_t* gpu_block_ids,
    GTensorHandler gpu_handler,
    int64_t* cpu_block_ids,
    void* cpu_ptr,
    int64_t cpu_kv_stride_in_bytes,
    int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_block_stride_in_bytes,
    int64_t chunk_size_in_bytes,
    bool is_mla,
    uint32_t* cpu_size_table_base,
    int64_t cpu_size_table_block_stride,
    int64_t cpu_size_table_layer_stride,
    cudaStream_t stream);

} // namespace flexkv
