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
          // Record transfer metrics after each cudaMemcpyAsync submission
          // Direction convention (from GPU perspective):
          //   - is_host_to_device=true  -> read (CPU->GPU, data flows INTO GPU)
          //   - is_host_to_device=false -> write (GPU->CPU, data flows OUT of
          //   GPU)
          FLEXKV_GPU_CPU_TRANSFER(is_host_to_device, chunk_size_in_bytes);
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

    // Record transfer metrics after kernel launch (cannot record inside kernel)
    // Total bytes = actual_chunk_bytes * num_layers * kv_dim * num_blocks
    // Note: Kernel transfers in float4 units, so we calculate aligned bytes to
    // match Direction convention (from GPU perspective):
    //   - is_host_to_device=true  -> read (CPU->GPU, data flows INTO GPU)
    //   - is_host_to_device=false -> write (GPU->CPU, data flows OUT of GPU)
    int kv_dim = is_mla ? 1 : 2;
    // Calculate actual bytes transferred (aligned to float4, matching kernel
    // logic)
    int64_t actual_chunk_bytes =
        (chunk_size_in_int64 * sizeof(int64_t) / sizeof(float4)) *
        sizeof(float4);
    FLEXKV_GPU_CPU_TRANSFER(
        is_host_to_device,
        actual_chunk_bytes * static_cast<int64_t>(num_layers) *
            static_cast<int64_t>(kv_dim) * static_cast<int64_t>(num_blocks));
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

#ifdef FLEXKV_ENABLE_NVCOMP
#include <cstdio>
#include <algorithm>
#include <stdexcept>
#include <string>

namespace flexkv {

#define ANS_NVCOMP_CHECK(call)                                          \
  do {                                                                  \
    nvcompStatus_t _s = (call);                                         \
    if (_s != nvcompSuccess) {                                          \
      fprintf(stderr, "[nvcomp] error %d at %s:%d\n",                   \
              (int)_s, __FILE__, __LINE__);                             \
      throw std::runtime_error("nvcomp ANS error");                     \
    }                                                                   \
  } while (0)

#define CUDA_CHECK(call)                                            \
  do {                                                                  \
    cudaError_t _e = (call);                                            \
    if (_e != cudaSuccess) {                                            \
      fprintf(stderr, "[nvcomp] CUDA error: %s at %s:%d\n",            \
              cudaGetErrorString(_e), __FILE__, __LINE__);              \
      throw std::runtime_error(cudaGetErrorString(_e));                 \
    }                                                                   \
  } while (0)

static const int ANS_KERNEL_BLOCK_SIZE = 1024;

template<bool is_write_cpu>
__global__ void ans_transfer_kernel(
    uint8_t* __restrict__ d_comp_staging,
    size_t staging_stride,
    size_t* __restrict__ d_comp_sizes,      // D2H input; H2D output
    uint8_t* __restrict__ cpu_ptr,
    size_t chunk_capacity,
    int* __restrict__ d_overflow,
    int64_t cpu_kv_stride,
    int64_t cpu_layer_stride,
    int64_t cpu_block_stride,
    const int64_t* __restrict__ cpu_block_ids,
    uint32_t* __restrict__ cpu_size_table_base,
    int64_t   cpu_size_table_block_stride,
    int64_t   cpu_size_table_layer_stride,
    int start_layer_id,
    int kv_dim, int num_blocks,
    int batch_start, int bsz)
{
    // Warp-per-chunk: each warp independently handles one chunk.
    // 4 warps per CTA process 4 chunks simultaneously, hiding per-chunk
    // PCIe latency warmup across warps.
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
            cpu_ptr
            + (int64_t)(layer + start_layer_id) * cpu_layer_stride
            + (int64_t)kv * cpu_kv_stride
            + cpu_block_ids[b] * cpu_block_stride;

        // External size table entry pointer (uint32_t, raw bytes).
        uint32_t* table_entry =
            cpu_size_table_base
            + cpu_block_ids[b]                          * cpu_size_table_block_stride
            + (int64_t)(start_layer_id + layer)         * cpu_size_table_layer_stride
            + (int64_t)kv;

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
            if (lane == 0)
                *table_entry = static_cast<uint32_t>(sz);
        } else {
            sz = static_cast<size_t>(*table_entry);
            // TODO(nvcomp-guard): validate H2D size-table entries before
            // copying. A stale/corrupt size of 0 or > staging_stride can feed
            // invalid compressed payloads to nvcomp or overrun staging.
            if (lane == 0)
                d_comp_sizes[i] = sz;
        }

        uint8_t* staging = d_comp_staging + (size_t)i * staging_stride;
        uint8_t* cpu_data = chunk_base;  // payload at offset 0 (no inline header)
        const float4* src = reinterpret_cast<const float4*>(is_write_cpu ? staging : cpu_data);
        float4* dst = reinterpret_cast<float4*>(is_write_cpu ? cpu_data : staging);

        int64_t n_f4 = sz / sizeof(float4);
        for (int64_t j = lane; j < n_f4; j += 32)
            dst[j] = __ldg(&src[j]);

        size_t tail = n_f4 * sizeof(float4);
        const uint8_t* src_tail = reinterpret_cast<const uint8_t*>(src) + tail;
        uint8_t* dst_tail = reinterpret_cast<uint8_t*>(dst) + tail;
        for (size_t j = lane; j < sz - tail; j += 32)
            dst_tail[j] = src_tail[j];
    }
}

template<BackendType Type>
__global__ void ans_build_gpu_chunk_ptrs_kernel(
    void** __restrict__ d_uncomp_ptrs,
    GTensorHandler gpu_handler,
    const int64_t* __restrict__ gpu_block_ids,
    int start_layer_id, int kv_dim, int num_blocks,
    int batch_start, int bsz)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < bsz; i += gridDim.x * blockDim.x) {
        int g = batch_start + i;
        int layer = g / (kv_dim * num_blocks);
        int kv = (g % (kv_dim * num_blocks)) / num_blocks;
        int b = g % num_blocks;
        d_uncomp_ptrs[i] = static_cast<void*>(
            ptr_at<Type>(gpu_handler, start_layer_id + layer, kv, gpu_block_ids[b]));
    }
}

void ans_ctx_create(ANSTransferContext* ctx, size_t max_num_chunks,
                    size_t max_chunk_size, int data_type,
                    int transfer_sms) {
  if (ctx == nullptr) {
    throw std::invalid_argument("ans_ctx_create: ctx must be non-null");
  }
  ans_ctx_destroy(ctx);

  if (transfer_sms == -1) {
    transfer_sms = 4;
  }
  if (transfer_sms <= 0) {
    throw std::invalid_argument(
        "ans_ctx_create: transfer_sms must be positive or -1 for the default");
  }
  ctx->transfer_sms = transfer_sms;
  if (max_num_chunks == 0 || max_chunk_size == 0) {
    throw std::invalid_argument(
        "ans_ctx_create: max_num_chunks and max_chunk_size must be greater "
        "than zero");
  }

  CUDA_CHECK(cudaGetDevice(&ctx->device_id));
  ctx->max_num_chunks = max_num_chunks;
  ctx->max_chunk_size = max_chunk_size;

  try {
    ctx->opts = nvcompBatchedANSDefaultOpts;
    // data_type: 0 = FLOAT16 (bf16/fp16), 1 = UCHAR/UINT8 (fp8)
    ctx->opts.data_type = (data_type == 0) ? float16 : uint8;

    const size_t max_total = max_num_chunks * max_chunk_size;

    ANS_NVCOMP_CHECK(nvcompBatchedANSCompressGetMaxOutputChunkSize(
        max_chunk_size, ctx->opts, &ctx->max_comp_chunk_bytes));
    // Round up to 16-byte alignment so the CPU read/write kernel can use
    // float4 (16-byte) loads/stores on d_comp_staging + i * max_comp_chunk_bytes.
    // nvcomp only guarantees 8-byte alignment for output chunk pointers.
    ctx->max_comp_chunk_bytes = (ctx->max_comp_chunk_bytes + 15) & ~size_t(15);
    ANS_NVCOMP_CHECK(nvcompBatchedANSCompressGetTempSizeEx(
        max_num_chunks, max_chunk_size, ctx->opts,
        &ctx->comp_temp_bytes, max_total));
    ANS_NVCOMP_CHECK(nvcompBatchedANSDecompressGetTempSizeEx(
        max_num_chunks, max_chunk_size,
        &ctx->decomp_temp_bytes, max_total));

    const size_t comp_staging_total = max_num_chunks * ctx->max_comp_chunk_bytes;
    const size_t ptr_bytes  = max_num_chunks * sizeof(void*);
    const size_t size_bytes = max_num_chunks * sizeof(size_t);

    // GPU compression buffers (double-buffered where needed for D2H pipeline)
    CUDA_CHECK(cudaMalloc(&ctx->d_comp_temp,       ctx->comp_temp_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_comp_staging_base, 2 * comp_staging_total));
    ctx->d_comp_staging[0] = ctx->d_comp_staging_base;
    ctx->d_comp_staging[1] = ctx->d_comp_staging_base + comp_staging_total;
    CUDA_CHECK(cudaMalloc(&ctx->d_uncomp_ptrs,     ptr_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_uncomp_sizes,    size_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_comp_ptrs[0],    ptr_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_comp_ptrs[1],    ptr_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_comp_sizes[0],   size_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_comp_sizes[1],   size_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_overflow,        sizeof(int)));

    // GPU decompression buffers (double-buffered for H2D pipeline)
    CUDA_CHECK(cudaMalloc(&ctx->d_decomp_temp,         ctx->decomp_temp_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_decomp_ptrs[0],      ptr_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_decomp_ptrs[1],      ptr_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf_sizes[0], size_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf_sizes[1], size_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_decomp_act_sizes,    size_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_statuses,
        max_num_chunks * sizeof(nvcompStatus_t)));

    ctx->h_ptr_scratch.resize(max_num_chunks);
    ctx->h_size_scratch.resize(max_num_chunks);

    // Pre-fill d_comp_ptrs for both slots
    for (int slot = 0; slot < 2; slot++) {
      for (size_t i = 0; i < max_num_chunks; i++)
        ctx->h_ptr_scratch[i] = ctx->d_comp_staging[slot] + i * ctx->max_comp_chunk_bytes;
      CUDA_CHECK(cudaMemcpy(ctx->d_comp_ptrs[slot], ctx->h_ptr_scratch.data(),
                                 ptr_bytes, cudaMemcpyHostToDevice));
    }

    // Pre-fill size arrays: all chunks have the same uncompressed size
    for (size_t i = 0; i < max_num_chunks; i++)
      ctx->h_size_scratch[i] = max_chunk_size;
    CUDA_CHECK(cudaMemcpy(ctx->d_uncomp_sizes, ctx->h_size_scratch.data(),
                               size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_decomp_buf_sizes[0], ctx->h_size_scratch.data(),
                               size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_decomp_buf_sizes[1], ctx->h_size_scratch.data(),
                               size_bytes, cudaMemcpyHostToDevice));

    // Create a high-priority stream for CPU payload read/write kernels so they
    // can run as soon as compress/decompress dependencies are satisfied.
    {
      int least_priority, greatest_priority;
      CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
      CUDA_CHECK(cudaStreamCreateWithPriority(
          &ctx->cpu_transfer_stream, cudaStreamNonBlocking, greatest_priority));
    }
    for (int i = 0; i < 2; i++) {
      CUDA_CHECK(cudaEventCreateWithFlags(&ctx->compress_done[i], cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&ctx->slot_done[i],  cudaEventDisableTiming));
    }

    // Compute kernel grid sizes via occupancy API
    {
      int write_cpu_bpsm = 0, read_cpu_bpsm = 0;
      CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &write_cpu_bpsm, ans_transfer_kernel<true>, ANS_KERNEL_BLOCK_SIZE, 0));
      CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &read_cpu_bpsm, ans_transfer_kernel<false>, ANS_KERNEL_BLOCK_SIZE, 0));
      ctx->write_cpu_grid = ctx->transfer_sms * std::max(write_cpu_bpsm, 1);
      ctx->read_cpu_grid  = ctx->transfer_sms * std::max(read_cpu_bpsm, 1);
    }
    ctx->initialized = true;
  } catch (...) {
    ans_ctx_destroy(ctx);
    throw;
  }
}

void ans_ctx_destroy(ANSTransferContext* ctx) {
  if (ctx == nullptr) {
    return;
  }

  int saved_device = -1;
  cudaGetDevice(&saved_device);
  if (ctx->device_id >= 0) {
    cudaSetDevice(ctx->device_id);
  }

  if (ctx->d_comp_temp != nullptr) cudaFree(ctx->d_comp_temp);
  if (ctx->d_comp_staging_base != nullptr) cudaFree(ctx->d_comp_staging_base);
  for (int i = 0; i < 2; i++) {
    if (ctx->d_comp_ptrs[i] != nullptr) cudaFree(ctx->d_comp_ptrs[i]);
    if (ctx->d_comp_sizes[i] != nullptr) cudaFree(ctx->d_comp_sizes[i]);
    if (ctx->compress_done[i] != nullptr) cudaEventDestroy(ctx->compress_done[i]);
    if (ctx->slot_done[i] != nullptr) cudaEventDestroy(ctx->slot_done[i]);
  }
  if (ctx->d_overflow != nullptr) cudaFree(ctx->d_overflow);
  if (ctx->cpu_transfer_stream != nullptr) cudaStreamDestroy(ctx->cpu_transfer_stream);
  if (ctx->d_uncomp_ptrs != nullptr) cudaFree(ctx->d_uncomp_ptrs);
  if (ctx->d_uncomp_sizes != nullptr) cudaFree(ctx->d_uncomp_sizes);
  if (ctx->d_decomp_temp != nullptr) cudaFree(ctx->d_decomp_temp);
  for (int i = 0; i < 2; i++) {
    if (ctx->d_decomp_ptrs[i] != nullptr) cudaFree(ctx->d_decomp_ptrs[i]);
    if (ctx->d_decomp_buf_sizes[i] != nullptr) cudaFree(ctx->d_decomp_buf_sizes[i]);
  }
  if (ctx->d_decomp_act_sizes != nullptr) cudaFree(ctx->d_decomp_act_sizes);
  if (ctx->d_statuses != nullptr) cudaFree(ctx->d_statuses);

  ctx->initialized = false;
  ctx->device_id = -1;
  ctx->max_num_chunks = 0;
  ctx->max_chunk_size = 0;
  ctx->max_comp_chunk_bytes = 0;
  ctx->comp_temp_bytes = 0;
  ctx->decomp_temp_bytes = 0;
  ctx->opts = {};
  ctx->d_comp_temp = nullptr;
  ctx->d_comp_staging_base = nullptr;
  ctx->d_uncomp_ptrs = nullptr;
  ctx->d_uncomp_sizes = nullptr;
  ctx->d_overflow = nullptr;
  ctx->d_decomp_temp = nullptr;
  ctx->d_decomp_act_sizes = nullptr;
  ctx->d_statuses = nullptr;
  ctx->cpu_transfer_stream = nullptr;
  ctx->write_cpu_grid = 0;
  ctx->read_cpu_grid = 0;
  ctx->transfer_sms = 0;
  for (int i = 0; i < 2; i++) {
    ctx->d_comp_staging[i] = nullptr;
    ctx->d_comp_ptrs[i] = nullptr;
    ctx->d_comp_sizes[i] = nullptr;
    ctx->d_decomp_ptrs[i] = nullptr;
    ctx->d_decomp_buf_sizes[i] = nullptr;
    ctx->compress_done[i] = nullptr;
    ctx->slot_done[i] = nullptr;
  }
  ctx->h_ptr_scratch.clear();
  ctx->h_size_scratch.clear();

  if (saved_device >= 0) {
    cudaSetDevice(saved_device);
  }
}

ANSTransferContext::~ANSTransferContext() {
  ans_ctx_destroy(this);
}

static void sync_streams(ANSTransferContext* ctx, cudaStream_t stream) {
  CUDA_CHECK(cudaStreamSynchronize(stream));
  CUDA_CHECK(cudaStreamSynchronize(ctx->cpu_transfer_stream));
}

static size_t sum_compressed_bytes_from_size_table(
    const uint32_t* cpu_size_table_base,
    const int64_t* cpu_block_ids,
    int num_blocks,
    int start_layer_id,
    int num_layers,
    int kv_dim,
    int64_t cpu_size_table_block_stride,
    int64_t cpu_size_table_layer_stride) {
  size_t total_comp = 0;
  const int total_chunks = num_layers * kv_dim * num_blocks;
  for (int g = 0; g < total_chunks; g++) {
    int layer = g / (kv_dim * num_blocks);
    int kv = (g % (kv_dim * num_blocks)) / num_blocks;
    int b = g % num_blocks;
    const uint32_t* entry =
        cpu_size_table_base
        + cpu_block_ids[b]                  * cpu_size_table_block_stride
        + (int64_t)(start_layer_id + layer) * cpu_size_table_layer_stride
        + (int64_t)kv;
    total_comp += static_cast<size_t>(*entry);
  }
  return total_comp;
}

static void require_initialized_ans_ctx(const ANSTransferContext* ctx,
                                        const char* caller) {
  if (ctx == nullptr || !ctx->initialized) {
    throw std::invalid_argument(std::string(caller) +
                                ": ANSTransferContext is not initialized");
  }
}

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
    cudaStream_t stream) {

  require_initialized_ans_ctx(ctx, "transfer_kv_blocks_ans_comp");

  const int kv_dim = is_mla ? 1 : 2;
  const int total_chunks = num_layers * kv_dim * num_blocks;
  const int batch_cap = static_cast<int>(ctx->max_num_chunks);
  const int num_batches = (total_chunks + batch_cap - 1) / batch_cap;

  if (chunk_size_in_bytes <= 0 ||
      static_cast<size_t>(chunk_size_in_bytes) != ctx->max_chunk_size) {
    throw std::invalid_argument(
        "transfer_kv_blocks_ans_comp: chunk_size_in_bytes must equal "
        "ctx->max_chunk_size");
  }

  CUDA_CHECK(cudaMemset(ctx->d_overflow, 0, sizeof(int)));

  for (int bi = 0; bi < num_batches; bi++) {
    const int bs  = bi * batch_cap;
    const int bsz = std::min(batch_cap, total_chunks - bs);
    const int cur = bi % 2;

    if (bi >= 2)
      CUDA_CHECK(cudaStreamWaitEvent(stream, ctx->slot_done[cur], 0));

    { // compress on GPU
      int threads = 256;
      int blocks = std::min((bsz + threads - 1) / threads, ctx->transfer_sms);
      ans_build_gpu_chunk_ptrs_kernel<Type><<<blocks, threads, 0, stream>>>(
          ctx->d_uncomp_ptrs, gpu_handler, gpu_block_ids,
          start_layer_id, kv_dim, num_blocks, bs, bsz);

      ANS_NVCOMP_CHECK(nvcompBatchedANSCompressAsync(
          (const void* const*)ctx->d_uncomp_ptrs,
          ctx->d_uncomp_sizes,
          chunk_size_in_bytes,
          bsz,
          ctx->d_comp_temp,
          ctx->comp_temp_bytes,
          ctx->d_comp_ptrs[cur],
          ctx->d_comp_sizes[cur],
          ctx->opts,
          stream));
      CUDA_CHECK(cudaEventRecord(ctx->compress_done[cur], stream));
    }

    { // write compressed payload to CPU
      CUDA_CHECK(cudaStreamWaitEvent(ctx->cpu_transfer_stream, ctx->compress_done[cur], 0));
      int grid = std::min(bsz, ctx->write_cpu_grid);
      ans_transfer_kernel<true><<<grid, ANS_KERNEL_BLOCK_SIZE, 0, ctx->cpu_transfer_stream>>>(
          ctx->d_comp_staging[cur], ctx->max_comp_chunk_bytes, ctx->d_comp_sizes[cur],
          static_cast<uint8_t*>(cpu_ptr),
          static_cast<size_t>(chunk_size_in_bytes), ctx->d_overflow,
          cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes, cpu_block_stride_in_bytes,
          cpu_block_ids,
          cpu_size_table_base, cpu_size_table_block_stride, cpu_size_table_layer_stride,
          start_layer_id, kv_dim, num_blocks, bs, bsz);
      CUDA_CHECK(cudaEventRecord(ctx->slot_done[cur], ctx->cpu_transfer_stream));
    }
  }

  sync_streams(ctx, stream);
  int overflow = 0;
  CUDA_CHECK(cudaMemcpy(&overflow, ctx->d_overflow, sizeof(int),
                            cudaMemcpyDeviceToHost));
  // TODO(nvcomp-guard): keep CPU-slot overflow protection before reporting sizes.
  if (overflow != 0) {
    throw std::runtime_error(
        "nvcomp compressed payload exceeded the CPU chunk slot; "
        "increase chunk size, use more compressible data, or disable nvcomp "
        "for this layout");
  }

  // TODO: can be removed in the future. Now only for log CR.
  size_t grand_total_comp = sum_compressed_bytes_from_size_table(
      cpu_size_table_base, cpu_block_ids, num_blocks, start_layer_id,
      num_layers, kv_dim, cpu_size_table_block_stride,
      cpu_size_table_layer_stride);
  return grand_total_comp;
}

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
    cudaStream_t stream) {

  require_initialized_ans_ctx(ctx, "transfer_kv_blocks_ans_decomp");

  const int kv_dim = is_mla ? 1 : 2;
  const int total_chunks = num_layers * kv_dim * num_blocks;
  const int batch_cap = static_cast<int>(ctx->max_num_chunks);
  const int num_batches = (total_chunks + batch_cap - 1) / batch_cap;

  if (chunk_size_in_bytes <= 0 ||
      static_cast<size_t>(chunk_size_in_bytes) != ctx->max_chunk_size) {
    throw std::invalid_argument(
        "transfer_kv_blocks_ans_decomp: chunk_size_in_bytes must equal "
        "ctx->max_chunk_size");
  }

  for (int bi = 0; bi < num_batches; bi++) {
    const int bs  = bi * batch_cap;
    const int bsz = std::min(batch_cap, total_chunks - bs);
    const int cur = bi % 2;

    if (bi >= 2)
      CUDA_CHECK(cudaStreamWaitEvent(ctx->cpu_transfer_stream, ctx->slot_done[cur], 0));

    { // build decompressed-destination pointers
      int threads = 256;
      int blocks = std::min((bsz + threads - 1) / threads, ctx->transfer_sms);
      ans_build_gpu_chunk_ptrs_kernel<Type><<<blocks, threads, 0, stream>>>(
          ctx->d_decomp_ptrs[cur], gpu_handler, gpu_block_ids,
          start_layer_id, kv_dim, num_blocks, bs, bsz);
    }

    { // read compressed payload from CPU
      int grid = std::min(bsz, ctx->read_cpu_grid);
      ans_transfer_kernel<false><<<grid, ANS_KERNEL_BLOCK_SIZE, 0, ctx->cpu_transfer_stream>>>(
          ctx->d_comp_staging[cur], ctx->max_comp_chunk_bytes, ctx->d_comp_sizes[cur],
          const_cast<uint8_t*>(static_cast<const uint8_t*>(cpu_ptr)),
          static_cast<size_t>(chunk_size_in_bytes), nullptr,
          cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes, cpu_block_stride_in_bytes,
          cpu_block_ids,
          cpu_size_table_base, cpu_size_table_block_stride, cpu_size_table_layer_stride,
          start_layer_id, kv_dim, num_blocks, bs, bsz);
    }

    CUDA_CHECK(cudaEventRecord(ctx->compress_done[cur], ctx->cpu_transfer_stream));

    { // decompress on GPU
      CUDA_CHECK(cudaStreamWaitEvent(stream, ctx->compress_done[cur], 0));
      ANS_NVCOMP_CHECK(nvcompBatchedANSDecompressAsync(
          (const void* const*)ctx->d_comp_ptrs[cur],
          ctx->d_comp_sizes[cur],
          ctx->d_decomp_buf_sizes[cur],
          ctx->d_decomp_act_sizes,
          bsz,
          ctx->d_decomp_temp,
          ctx->decomp_temp_bytes,
          ctx->d_decomp_ptrs[cur],
          ctx->d_statuses,
          stream));
      CUDA_CHECK(cudaEventRecord(ctx->slot_done[cur], stream));
    }
  }

  sync_streams(ctx, stream);

  // TODO: can be removed in the future. Now only for log CR.
  size_t total_comp = sum_compressed_bytes_from_size_table(
      cpu_size_table_base, cpu_block_ids, num_blocks, start_layer_id,
      num_layers, kv_dim, cpu_size_table_block_stride,
      cpu_size_table_layer_stride);
  return total_comp;
}

#define ANS_INSTANTIATE(Type)                                                  \
  template size_t transfer_kv_blocks_ans_comp<Type>(                                  \
      ANSTransferContext*, int, int, int, int64_t*, GTensorHandler,            \
      int64_t*, void*, int64_t, int64_t, int64_t, int64_t, bool,              \
      uint32_t*, int64_t, int64_t,                                            \
      cudaStream_t);                                                           \
  template size_t transfer_kv_blocks_ans_decomp<Type>(                                \
      ANSTransferContext*, int, int, int, int64_t*, GTensorHandler,            \
      int64_t*, void*, int64_t, int64_t, int64_t, int64_t, bool,              \
      uint32_t*, int64_t, int64_t,                                            \
      cudaStream_t);

ANS_INSTANTIATE(BackendType::VLLM)
ANS_INSTANTIATE(BackendType::TRTLLM)
ANS_INSTANTIATE(BackendType::SGLANG)
#undef ANS_INSTANTIATE

// Undef the file-local error-check macros so they can't leak past this TU
// (CUDA_CHECK in particular is a collision-prone name).
#undef CUDA_CHECK
#undef ANS_NVCOMP_CHECK

} // namespace flexkv

#endif // FLEXKV_ENABLE_NVCOMP
