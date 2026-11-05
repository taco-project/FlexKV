/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "transfer.cuh"

#include <cstdint>
#include <cuda_runtime.h>

namespace flexkv {

static constexpr int STAGING_TRANSFER_KERNEL_BLOCK_SIZE = 1024;

template <bool is_write_cpu>
__global__ void staging_transfer_kernel(
    uint8_t* __restrict__ d_comp_staging,
    size_t staging_stride,
    size_t* __restrict__ d_comp_sizes,
    uint8_t* __restrict__ cpu_ptr,
    size_t chunk_capacity,
    int* __restrict__ d_overflow,
    int64_t cpu_kv_stride,
    int64_t cpu_layer_stride,
    int64_t cpu_block_stride,
    const int64_t* __restrict__ cpu_block_ids,
    uint32_t* __restrict__ cpu_size_table_base,
    int64_t cpu_size_table_block_stride,
    int64_t cpu_size_table_layer_stride,
    int start_layer_id,
    int kv_dim,
    int num_blocks,
    int batch_start,
    int bsz) {
  const int lane = threadIdx.x & 31;
  const int warp_id = threadIdx.x >> 5;
  const int warps_per_block = blockDim.x >> 5;
  const size_t global_warp = (size_t)blockIdx.x * warps_per_block + warp_id;
  const size_t total_warps = (size_t)gridDim.x * warps_per_block;

  for (size_t i = global_warp; i < (size_t)bsz; i += total_warps) {
    int g = batch_start + i;
    int layer = g / (kv_dim * num_blocks);
    int kv = (g % (kv_dim * num_blocks)) / num_blocks;
    int b = g % num_blocks;

    uint8_t* chunk_base =
        cpu_ptr +
        (int64_t)(layer + start_layer_id) * cpu_layer_stride +
        (int64_t)kv * cpu_kv_stride +
        cpu_block_ids[b] * cpu_block_stride;

    uint32_t* table_entry =
        cpu_size_table_base +
        cpu_block_ids[b] * cpu_size_table_block_stride +
        (int64_t)(start_layer_id + layer) * cpu_size_table_layer_stride +
        (int64_t)kv;

    size_t sz;
    if constexpr (is_write_cpu) {
      sz = d_comp_sizes[i];
      if (sz > chunk_capacity) {
        if (lane == 0) {
          atomicAdd(d_overflow, 1);
          *table_entry = 0;
        }
        continue;
      }
      if (lane == 0) {
        *table_entry = static_cast<uint32_t>(sz);
      }
    } else {
      sz = static_cast<size_t>(*table_entry);
      // TODO(nvcomp-guard): validate H2D size-table entries before copying.
      // A stale/corrupt size of 0 or > staging_stride can feed invalid
      // compressed payloads to nvcomp or overrun staging.
      if (lane == 0) {
        d_comp_sizes[i] = sz;
      }
    }

    uint8_t* staging = d_comp_staging + (size_t)i * staging_stride;
    uint8_t* cpu_data = chunk_base;
    const float4* src =
        reinterpret_cast<const float4*>(is_write_cpu ? staging : cpu_data);
    float4* dst = reinterpret_cast<float4*>(is_write_cpu ? cpu_data : staging);

    int64_t n_f4 = sz / sizeof(float4);
    for (int64_t j = lane; j < n_f4; j += 32) {
      dst[j] = __ldg(&src[j]);
    }

    size_t tail = n_f4 * sizeof(float4);
    const uint8_t* src_tail = reinterpret_cast<const uint8_t*>(src) + tail;
    uint8_t* dst_tail = reinterpret_cast<uint8_t*>(dst) + tail;
    for (size_t j = lane; j < sz - tail; j += 32) {
      dst_tail[j] = src_tail[j];
    }
  }
}

template <BackendType Type>
__global__ void build_gpu_chunk_ptrs_kernel(
    void** __restrict__ d_uncomp_ptrs,
    GTensorHandler gpu_handler,
    const int64_t* __restrict__ gpu_block_ids,
    int start_layer_id,
    int kv_dim,
    int num_blocks,
    int batch_start,
    int bsz) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < bsz;
       i += gridDim.x * blockDim.x) {
    int g = batch_start + i;
    int layer = g / (kv_dim * num_blocks);
    int kv = (g % (kv_dim * num_blocks)) / num_blocks;
    int b = g % num_blocks;
    d_uncomp_ptrs[i] = static_cast<void*>(
        ptr_at<Type>(gpu_handler, start_layer_id + layer, kv, gpu_block_ids[b]));
  }
}

} // namespace flexkv
