/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */
#include "compression/ans/nvcomp_ans.cuh"
#include "compression/common/staging_transfer.cuh"

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAFunctions.h>
#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace flexkv {

#define ANS_NVCOMP_CHECK(call)                                             \
  do {                                                                     \
    nvcompStatus_t _s = (call);                                            \
    if (_s != nvcompSuccess) {                                             \
      fprintf(stderr, "[nvcomp] error %d at %s:%d\n", (int)_s, __FILE__,   \
              __LINE__);                                                   \
      throw std::runtime_error("nvcomp ANS error");                        \
    }                                                                      \
  } while (0)

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t _e = (call);                                               \
    if (_e != cudaSuccess) {                                               \
      fprintf(stderr, "[nvcomp] CUDA error: %s at %s:%d\n",                \
              cudaGetErrorString(_e), __FILE__, __LINE__);                 \
      throw std::runtime_error(cudaGetErrorString(_e));                    \
    }                                                                      \
  } while (0)

void ans_ctx_create(ANSTransferContext* ctx, size_t max_num_chunks,
                    size_t max_chunk_size, int data_type, int transfer_sms) {
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
        max_num_chunks, max_chunk_size, ctx->opts, &ctx->comp_temp_bytes,
        max_total));
    ANS_NVCOMP_CHECK(nvcompBatchedANSDecompressGetTempSizeEx(
        max_num_chunks, max_chunk_size, &ctx->decomp_temp_bytes, max_total));

    const size_t comp_staging_total =
        max_num_chunks * ctx->max_comp_chunk_bytes;
    const size_t ptr_bytes = max_num_chunks * sizeof(void*);
    const size_t size_bytes = max_num_chunks * sizeof(size_t);

    // GPU compression buffers (double-buffered where needed for D2H pipeline)
    CUDA_CHECK(cudaMalloc(&ctx->d_comp_temp, ctx->comp_temp_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_comp_staging_base, 2 * comp_staging_total));
    ctx->d_comp_staging[0] = ctx->d_comp_staging_base;
    ctx->d_comp_staging[1] = ctx->d_comp_staging_base + comp_staging_total;
    CUDA_CHECK(cudaMalloc(&ctx->d_uncomp_ptrs, ptr_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_uncomp_sizes, size_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_comp_ptrs[0], ptr_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_comp_ptrs[1], ptr_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_comp_sizes[0], size_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_comp_sizes[1], size_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_overflow, sizeof(int)));

    // GPU decompression buffers (double-buffered for H2D pipeline)
    CUDA_CHECK(cudaMalloc(&ctx->d_decomp_temp, ctx->decomp_temp_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_decomp_ptrs[0], ptr_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_decomp_ptrs[1], ptr_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf_sizes[0], size_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf_sizes[1], size_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_decomp_act_sizes, size_bytes));
    CUDA_CHECK(cudaMalloc(&ctx->d_statuses,
                          max_num_chunks * sizeof(nvcompStatus_t)));

    ctx->h_ptr_scratch.resize(max_num_chunks);
    ctx->h_size_scratch.resize(max_num_chunks);

    // Pre-fill d_comp_ptrs for both slots.
    for (int slot = 0; slot < 2; slot++) {
      for (size_t i = 0; i < max_num_chunks; i++) {
        ctx->h_ptr_scratch[i] =
            ctx->d_comp_staging[slot] + i * ctx->max_comp_chunk_bytes;
      }
      CUDA_CHECK(cudaMemcpy(ctx->d_comp_ptrs[slot], ctx->h_ptr_scratch.data(),
                            ptr_bytes, cudaMemcpyHostToDevice));
    }

    // Pre-fill size arrays: all chunks have the same uncompressed size.
    for (size_t i = 0; i < max_num_chunks; i++) {
      ctx->h_size_scratch[i] = max_chunk_size;
    }
    CUDA_CHECK(cudaMemcpy(ctx->d_uncomp_sizes, ctx->h_size_scratch.data(),
                          size_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_decomp_buf_sizes[0],
                          ctx->h_size_scratch.data(), size_bytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx->d_decomp_buf_sizes[1],
                          ctx->h_size_scratch.data(), size_bytes,
                          cudaMemcpyHostToDevice));

    // Create a high-priority stream for CPU payload read/write kernels so they
    // can run as soon as compress/decompress dependencies are satisfied.
    {
      int least_priority, greatest_priority;
      CUDA_CHECK(
          cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
      CUDA_CHECK(cudaStreamCreateWithPriority(
          &ctx->cpu_transfer_stream, cudaStreamNonBlocking, greatest_priority));
    }
    for (int i = 0; i < 2; i++) {
      CUDA_CHECK(
          cudaEventCreateWithFlags(&ctx->compress_done[i], cudaEventDisableTiming));
      CUDA_CHECK(
          cudaEventCreateWithFlags(&ctx->slot_done[i], cudaEventDisableTiming));
    }

    // Compute kernel grid sizes via occupancy API.
    {
      int write_cpu_bpsm = 0, read_cpu_bpsm = 0;
      CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &write_cpu_bpsm, staging_transfer_kernel<true>,
          STAGING_TRANSFER_KERNEL_BLOCK_SIZE, 0));
      CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &read_cpu_bpsm, staging_transfer_kernel<false>,
          STAGING_TRANSFER_KERNEL_BLOCK_SIZE, 0));
      ctx->write_cpu_grid = ctx->transfer_sms * std::max(write_cpu_bpsm, 1);
      ctx->read_cpu_grid = ctx->transfer_sms * std::max(read_cpu_bpsm, 1);
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

  const c10::cuda::CUDAGuard restore_device_on_exit(c10::cuda::current_device());
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
  if (ctx->cpu_transfer_stream != nullptr)
    cudaStreamDestroy(ctx->cpu_transfer_stream);
  if (ctx->d_uncomp_ptrs != nullptr) cudaFree(ctx->d_uncomp_ptrs);
  if (ctx->d_uncomp_sizes != nullptr) cudaFree(ctx->d_uncomp_sizes);
  if (ctx->d_decomp_temp != nullptr) cudaFree(ctx->d_decomp_temp);
  for (int i = 0; i < 2; i++) {
    if (ctx->d_decomp_ptrs[i] != nullptr) cudaFree(ctx->d_decomp_ptrs[i]);
    if (ctx->d_decomp_buf_sizes[i] != nullptr)
      cudaFree(ctx->d_decomp_buf_sizes[i]);
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
        cpu_size_table_base +
        cpu_block_ids[b] * cpu_size_table_block_stride +
        (int64_t)(start_layer_id + layer) * cpu_size_table_layer_stride +
        (int64_t)kv;
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
    const int bs = bi * batch_cap;
    const int bsz = std::min(batch_cap, total_chunks - bs);
    const int cur = bi % 2;

    if (bi >= 2) {
      CUDA_CHECK(cudaStreamWaitEvent(stream, ctx->slot_done[cur], 0));
    }

    {
      int threads = 256;
      int blocks = std::min((bsz + threads - 1) / threads, ctx->transfer_sms);
      build_gpu_chunk_ptrs_kernel<Type><<<blocks, threads, 0, stream>>>(
          ctx->d_uncomp_ptrs, gpu_handler, gpu_block_ids, start_layer_id,
          kv_dim, num_blocks, bs, bsz);

      ANS_NVCOMP_CHECK(nvcompBatchedANSCompressAsync(
          (const void* const*)ctx->d_uncomp_ptrs, ctx->d_uncomp_sizes,
          chunk_size_in_bytes, bsz, ctx->d_comp_temp, ctx->comp_temp_bytes,
          ctx->d_comp_ptrs[cur], ctx->d_comp_sizes[cur], ctx->opts, stream));
      CUDA_CHECK(cudaEventRecord(ctx->compress_done[cur], stream));
    }

    {
      CUDA_CHECK(cudaStreamWaitEvent(ctx->cpu_transfer_stream,
                                     ctx->compress_done[cur], 0));
      int grid = std::min(bsz, ctx->write_cpu_grid);
      staging_transfer_kernel<true>
          <<<grid, STAGING_TRANSFER_KERNEL_BLOCK_SIZE, 0,
             ctx->cpu_transfer_stream>>>(
              ctx->d_comp_staging[cur], ctx->max_comp_chunk_bytes,
              ctx->d_comp_sizes[cur], static_cast<uint8_t*>(cpu_ptr),
              static_cast<size_t>(chunk_size_in_bytes), ctx->d_overflow,
              cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes,
              cpu_block_stride_in_bytes, cpu_block_ids, cpu_size_table_base,
              cpu_size_table_block_stride, cpu_size_table_layer_stride,
              start_layer_id, kv_dim, num_blocks, bs, bsz);
      CUDA_CHECK(cudaEventRecord(ctx->slot_done[cur],
                                 ctx->cpu_transfer_stream));
    }
  }

  sync_streams(ctx, stream);
  int overflow = 0;
  CUDA_CHECK(
      cudaMemcpy(&overflow, ctx->d_overflow, sizeof(int), cudaMemcpyDeviceToHost));
  // TODO(nvcomp-guard): keep CPU-slot overflow protection before reporting sizes.
  if (overflow != 0) {
    throw std::runtime_error(
        "nvcomp compressed payload exceeded the CPU chunk slot; "
        "increase chunk size, use more compressible data, or disable nvcomp "
        "for this layout");
  }

  // TODO: can be removed in the future. Now only for log CR.
  return sum_compressed_bytes_from_size_table(
      cpu_size_table_base, cpu_block_ids, num_blocks, start_layer_id,
      num_layers, kv_dim, cpu_size_table_block_stride,
      cpu_size_table_layer_stride);
}

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
    const int bs = bi * batch_cap;
    const int bsz = std::min(batch_cap, total_chunks - bs);
    const int cur = bi % 2;

    if (bi >= 2) {
      CUDA_CHECK(cudaStreamWaitEvent(ctx->cpu_transfer_stream,
                                     ctx->slot_done[cur], 0));
    }

    {
      int threads = 256;
      int blocks = std::min((bsz + threads - 1) / threads, ctx->transfer_sms);
      build_gpu_chunk_ptrs_kernel<Type><<<blocks, threads, 0, stream>>>(
          ctx->d_decomp_ptrs[cur], gpu_handler, gpu_block_ids,
          start_layer_id, kv_dim, num_blocks, bs, bsz);
    }

    {
      int grid = std::min(bsz, ctx->read_cpu_grid);
      staging_transfer_kernel<false>
          <<<grid, STAGING_TRANSFER_KERNEL_BLOCK_SIZE, 0,
             ctx->cpu_transfer_stream>>>(
              ctx->d_comp_staging[cur], ctx->max_comp_chunk_bytes,
              ctx->d_comp_sizes[cur],
              const_cast<uint8_t*>(static_cast<const uint8_t*>(cpu_ptr)),
              static_cast<size_t>(chunk_size_in_bytes), nullptr,
              cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes,
              cpu_block_stride_in_bytes, cpu_block_ids, cpu_size_table_base,
              cpu_size_table_block_stride, cpu_size_table_layer_stride,
              start_layer_id, kv_dim, num_blocks, bs, bsz);
    }

    CUDA_CHECK(cudaEventRecord(ctx->compress_done[cur],
                               ctx->cpu_transfer_stream));

    {
      CUDA_CHECK(cudaStreamWaitEvent(stream, ctx->compress_done[cur], 0));
      ANS_NVCOMP_CHECK(nvcompBatchedANSDecompressAsync(
          (const void* const*)ctx->d_comp_ptrs[cur], ctx->d_comp_sizes[cur],
          ctx->d_decomp_buf_sizes[cur], ctx->d_decomp_act_sizes, bsz,
          ctx->d_decomp_temp, ctx->decomp_temp_bytes, ctx->d_decomp_ptrs[cur],
          ctx->d_statuses, stream));
      CUDA_CHECK(cudaEventRecord(ctx->slot_done[cur], stream));
    }
  }

  sync_streams(ctx, stream);

  // TODO: can be removed in the future. Now only for log CR.
  return sum_compressed_bytes_from_size_table(
      cpu_size_table_base, cpu_block_ids, num_blocks, start_layer_id,
      num_layers, kv_dim, cpu_size_table_block_stride,
      cpu_size_table_layer_stride);
}

#define ANS_INSTANTIATE(Type)                                               \
  template size_t transfer_kv_blocks_ans_comp<Type>(                         \
      ANSTransferContext*, int, int, int, int64_t*, GTensorHandler, int64_t*, \
      void*, int64_t, int64_t, int64_t, int64_t, bool, uint32_t*, int64_t,    \
      int64_t, cudaStream_t);                                                \
  template size_t transfer_kv_blocks_ans_decomp<Type>(                       \
      ANSTransferContext*, int, int, int, int64_t*, GTensorHandler, int64_t*, \
      void*, int64_t, int64_t, int64_t, int64_t, bool, uint32_t*, int64_t,    \
      int64_t, cudaStream_t);

ANS_INSTANTIATE(BackendType::VLLM)
ANS_INSTANTIATE(BackendType::TRTLLM)
ANS_INSTANTIATE(BackendType::SGLANG)
#undef ANS_INSTANTIATE

#undef CUDA_CHECK
#undef ANS_NVCOMP_CHECK

} // namespace flexkv
