#ifdef FLEXKV_ENABLE_NVCOMP

#include "ans_transfer.h"

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <stdexcept>
#include <nvtx3/nvToolsExt.h>

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

#define ANS_CUDA_CHECK(call)                                            \
  do {                                                                  \
    cudaError_t _e = (call);                                            \
    if (_e != cudaSuccess) {                                            \
      fprintf(stderr, "[nvcomp] CUDA error: %s at %s:%d\n",            \
              cudaGetErrorString(_e), __FILE__, __LINE__);              \
      throw std::runtime_error(cudaGetErrorString(_e));                 \
    }                                                                   \
  } while (0)

struct NvtxScope {
    NvtxScope(const char* name) { nvtxRangePush(name); }
    ~NvtxScope() { nvtxRangePop(); }
    NvtxScope(const NvtxScope&) = delete;
    NvtxScope& operator=(const NvtxScope&) = delete;
};

struct GreenScope {
    nvcompGreenContext_t* ctx_;
    GreenScope(nvcompGreenContext_t* ctx) : ctx_(ctx->valid ? ctx : nullptr) {
        if (ctx_) nvcompGreenContextPush(ctx_);
    }
    ~GreenScope() { if (ctx_) nvcompGreenContextPop(ctx_); }
    GreenScope(const GreenScope&) = delete;
    GreenScope& operator=(const GreenScope&) = delete;
};

static const int ANS_KERNEL_BLOCK_SIZE = 128;

__global__ void ans_d2h_scatter_kernel(
    const uint8_t* __restrict__ d_comp_staging,
    size_t staging_stride,
    const size_t* __restrict__ d_comp_sizes,
    uint8_t* __restrict__ cpu_ptr,
    int64_t cpu_kv_stride,
    int64_t cpu_layer_stride,
    int64_t cpu_block_stride,
    const int64_t* __restrict__ cpu_block_ids,
    int start_layer_id,
    int kv_dim, int num_blocks,
    int batch_start, int bsz,
    int64_t* __restrict__ h_comp_sizes_out)
{
    for (int i = blockIdx.x; i < bsz; i += gridDim.x) {
        int g = batch_start + i;
        int layer = g / (kv_dim * num_blocks);
        int kv = (g % (kv_dim * num_blocks)) / num_blocks;
        int b = g % num_blocks;

        size_t sz = d_comp_sizes[i];
        const float4* src = reinterpret_cast<const float4*>(
            d_comp_staging + (size_t)i * staging_stride);
        float4* dst = reinterpret_cast<float4*>(
            cpu_ptr
            + (int64_t)(layer + start_layer_id) * cpu_layer_stride
            + (int64_t)kv * cpu_kv_stride
            + cpu_block_ids[b] * cpu_block_stride);

        int64_t n_f4 = sz / sizeof(float4);
        for (int64_t j = threadIdx.x; j < n_f4; j += blockDim.x)
            dst[j] = __ldg(&src[j]);

        size_t tail = n_f4 * sizeof(float4);
        const uint8_t* src_tail = reinterpret_cast<const uint8_t*>(src) + tail;
        uint8_t* dst_tail = reinterpret_cast<uint8_t*>(dst) + tail;
        for (size_t j = threadIdx.x; j < sz - tail; j += blockDim.x)
            dst_tail[j] = src_tail[j];

        if (threadIdx.x == 0)
            h_comp_sizes_out[g] = static_cast<int64_t>(sz);
    }
}

__global__ void ans_h2d_gather_kernel(
    uint8_t* __restrict__ d_comp_staging,
    size_t staging_stride,
    const size_t* __restrict__ d_comp_sizes,
    const uint8_t* __restrict__ cpu_ptr,
    int64_t cpu_kv_stride,
    int64_t cpu_layer_stride,
    int64_t cpu_block_stride,
    const int64_t* __restrict__ cpu_block_ids,
    int start_layer_id,
    int kv_dim, int num_blocks,
    int batch_start, int bsz)
{
    for (int i = blockIdx.x; i < bsz; i += gridDim.x) {
        int g = batch_start + i;
        int layer = g / (kv_dim * num_blocks);
        int kv = (g % (kv_dim * num_blocks)) / num_blocks;
        int b = g % num_blocks;

        size_t sz = d_comp_sizes[i];
        float4* dst = reinterpret_cast<float4*>(
            d_comp_staging + (size_t)i * staging_stride);
        const float4* src = reinterpret_cast<const float4*>(
            cpu_ptr
            + (int64_t)(layer + start_layer_id) * cpu_layer_stride
            + (int64_t)kv * cpu_kv_stride
            + cpu_block_ids[b] * cpu_block_stride);

        int64_t n_f4 = sz / sizeof(float4);
        for (int64_t j = threadIdx.x; j < n_f4; j += blockDim.x)
            dst[j] = __ldg(&src[j]);

        size_t tail = n_f4 * sizeof(float4);
        const uint8_t* src_tail = reinterpret_cast<const uint8_t*>(src) + tail;
        uint8_t* dst_tail = reinterpret_cast<uint8_t*>(dst) + tail;
        for (size_t j = threadIdx.x; j < sz - tail; j += blockDim.x)
            dst_tail[j] = src_tail[j];
    }
}

// GPU kernel to gather compressed sizes from 2D pinned metadata buffer into
// d_comp_sizes, replacing Python-side meta_slice + cudaMemcpyAsync.
// comp_sizes_meta layout: [num_layers * kv_dim, num_cpu_blocks] (pinned host)
__global__ void ans_gather_comp_sizes_kernel(
    size_t* __restrict__ d_comp_sizes_out,
    const int64_t* __restrict__ comp_sizes_meta,
    int meta_stride,
    const int64_t* __restrict__ cpu_block_ids,
    int kv_dim, int num_blocks,
    int start_layer_id,
    int batch_start, int bsz)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < bsz; i += gridDim.x * blockDim.x) {
        int g = batch_start + i;
        int meta_row = start_layer_id * kv_dim + g / num_blocks;
        int b = g % num_blocks;
        d_comp_sizes_out[i] = static_cast<size_t>(
            comp_sizes_meta[meta_row * meta_stride + cpu_block_ids[b]]);
    }
}

template<BackendType Type>
__global__ void ans_build_ptrs_kernel(
    void** __restrict__ d_uncomp_ptrs,
    GTensorHandler gpu_handler,
    const int64_t* __restrict__ gpu_block_ids,
    int kv_dim, int num_blocks,
    int batch_start, int bsz)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < bsz; i += gridDim.x * blockDim.x) {
        int g = batch_start + i;
        int layer = g / (kv_dim * num_blocks);
        int kv = (g % (kv_dim * num_blocks)) / num_blocks;
        int b = g % num_blocks;
        d_uncomp_ptrs[i] = static_cast<void*>(
            ptr_at<Type>(gpu_handler, layer, kv, gpu_block_ids[b]));
    }
}


// ---------------------------------------------------------------------------
// Context lifecycle
// ---------------------------------------------------------------------------
void ans_ctx_create(ANSTransferContext* ctx, size_t max_num_chunks,
                    size_t max_chunk_size, int data_type, int log_level,
                    int pipeline_batch_size, int transfer_sms) {
  if (transfer_sms == -1) {
    transfer_sms = 4;
  }
  ctx->transfer_sms = transfer_sms;
  // Auto-compute max_num_chunks when caller passes 0: use the minimum value
  // that achieves the best compression ratio (lowest num_ctas_per_chunk).
  if (max_num_chunks == 0 && max_chunk_size > 0) {
    max_num_chunks = static_cast<size_t>(
        compute_auto_pipeline_batch(max_chunk_size, 0));
  }

  ctx->max_num_chunks = max_num_chunks;
  ctx->max_chunk_size = max_chunk_size;
  ctx->log_level      = log_level;

  ctx->comp_opts = nvcompBatchedANSCompressDefaultOpts;
  ctx->comp_opts.data_type = NVCOMP_TYPE_FLOAT16; // hardcoded
  ctx->decomp_opts = nvcompBatchedANSDecompressDefaultOpts;
  ctx->decomp_opts.max_uncompressed_chunk_size = ctx->max_chunk_size;

  const size_t max_total = max_num_chunks * max_chunk_size;

  ANS_NVCOMP_CHECK(nvcompBatchedANSCompressGetMaxOutputChunkSize(
      max_chunk_size, ctx->comp_opts, &ctx->max_comp_chunk_bytes));
  ANS_NVCOMP_CHECK(nvcompBatchedANSCompressGetTempSizeAsync(
      max_num_chunks, max_chunk_size, ctx->comp_opts,
      &ctx->comp_temp_bytes, max_total));
  ANS_NVCOMP_CHECK(nvcompBatchedANSDecompressGetTempSizeAsync(
      max_num_chunks, max_chunk_size, ctx->decomp_opts,
      &ctx->decomp_temp_bytes, max_total));

  const size_t comp_staging_total = max_num_chunks * ctx->max_comp_chunk_bytes;
  const size_t ptr_bytes  = max_num_chunks * sizeof(void*);
  const size_t size_bytes = max_num_chunks * sizeof(size_t);

  // GPU compression buffers (double-buffered where needed for D2H pipeline)
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_comp_temp,       ctx->comp_temp_bytes));
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_comp_staging_base, 2 * comp_staging_total));
  ctx->d_comp_staging[0] = ctx->d_comp_staging_base;
  ctx->d_comp_staging[1] = ctx->d_comp_staging_base + comp_staging_total;
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_uncomp_ptrs,     ptr_bytes));
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_uncomp_sizes,    size_bytes));
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_comp_ptrs[0],    ptr_bytes));
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_comp_ptrs[1],    ptr_bytes));
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_comp_sizes[0],   size_bytes));
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_comp_sizes[1],   size_bytes));

  // GPU decompression buffers (double-buffered for H2D pipeline)
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_temp,         ctx->decomp_temp_bytes));
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_ptrs[0],      ptr_bytes));
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_ptrs[1],      ptr_bytes));
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf_sizes[0], size_bytes));
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf_sizes[1], size_bytes));
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_act_sizes,    size_bytes));
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_statuses,
      max_num_chunks * sizeof(nvcompStatus_t)));

  ANS_CUDA_CHECK(cudaMallocHost(&ctx->h_comp_sizes[0], size_bytes));
  ANS_CUDA_CHECK(cudaMallocHost(&ctx->h_comp_sizes[1], size_bytes));

  ctx->h_ptr_scratch.resize(max_num_chunks);
  ctx->h_size_scratch.resize(max_num_chunks);

  // Pre-fill d_comp_ptrs for both slots
  for (int slot = 0; slot < 2; slot++) {
    for (size_t i = 0; i < max_num_chunks; i++)
      ctx->h_ptr_scratch[i] = ctx->d_comp_staging[slot] + i * ctx->max_comp_chunk_bytes;
    ANS_CUDA_CHECK(cudaMemcpy(ctx->d_comp_ptrs[slot], ctx->h_ptr_scratch.data(),
                               ptr_bytes, cudaMemcpyHostToDevice));
  }

  // Pre-fill size arrays: all chunks have the same uncompressed size
  for (size_t i = 0; i < max_num_chunks; i++)
    ctx->h_size_scratch[i] = max_chunk_size;
  ANS_CUDA_CHECK(cudaMemcpy(ctx->d_uncomp_sizes, ctx->h_size_scratch.data(),
                             size_bytes, cudaMemcpyHostToDevice));
  ANS_CUDA_CHECK(cudaMemcpy(ctx->d_decomp_buf_sizes[0], ctx->h_size_scratch.data(),
                             size_bytes, cudaMemcpyHostToDevice));
  ANS_CUDA_CHECK(cudaMemcpy(ctx->d_decomp_buf_sizes[1], ctx->h_size_scratch.data(),
                             size_bytes, cudaMemcpyHostToDevice));

  // Create scatter stream with highest priority so gather/scatter blocks
  // are scheduled ahead of decompress/compress blocks when SMs become free.
  {
    int least_priority, greatest_priority;
    cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority);
    ANS_CUDA_CHECK(cudaStreamCreateWithPriority(
        &ctx->scatter_stream, cudaStreamNonBlocking, greatest_priority));
  }
  for (int i = 0; i < 2; i++) {
    ANS_CUDA_CHECK(cudaEventCreateWithFlags(&ctx->compress_done[i], cudaEventDisableTiming));
    ANS_CUDA_CHECK(cudaEventCreateWithFlags(&ctx->scatter_done[i],  cudaEventDisableTiming));
  }

  // Compute kernel grid sizes via occupancy API (matching baseline transfer_kv_blocks_kernel)
  {
    int scatter_bpsm = 0, gather_bpsm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &scatter_bpsm, ans_d2h_scatter_kernel, ANS_KERNEL_BLOCK_SIZE, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &gather_bpsm, ans_h2d_gather_kernel, ANS_KERNEL_BLOCK_SIZE, 0);
    ctx->scatter_grid = ctx->transfer_sms * std::max(scatter_bpsm, 1);
    ctx->gather_grid  = ctx->transfer_sms * std::max(gather_bpsm, 1);
  }

  ctx->green_ctx = NVCOMP_GREEN_CONTEXT_INIT;
  {
    const char* env = std::getenv("FLEXKV_NVCOMP_GREEN_SMS");
    if (env && nvcompGreenContextIsSupported()) {
      int green_sms = std::atoi(env);
      if (green_sms > 0) {
        nvcompStatus_t gs = nvcompGreenContextCreate(&ctx->green_ctx, green_sms);
        if (gs == nvcompSuccess) {
          ctx->pipeline_batch_size = static_cast<int>(max_num_chunks);
          if (log_level >= 1)
            printf("[nvcomp] Green Context: requested %d SMs, allocated %d / %d, "
                   "pipeline_batch_size=%d (no split)\n",
                   green_sms, ctx->green_ctx.num_sms,
                   ctx->green_ctx.total_device_sms,
                   ctx->pipeline_batch_size);
        } else {
          fprintf(stderr, "[nvcomp] Green Context creation failed (status=%d), "
                  "falling back to default scheduling\n", (int)gs);
        }
      }
    }
  }

  // Fallback: auto-tune pipeline_batch_size when green context is unavailable.
  // Split batches so compress/decompress occupies fewer SMs, leaving room for
  // scatter/gather to overlap on the remaining SMs.
  if (!ctx->green_ctx.valid) {
    if (pipeline_batch_size > 0) {
      ctx->pipeline_batch_size = std::min(
          pipeline_batch_size, static_cast<int>(max_num_chunks));
    } else if (max_num_chunks > 0 && max_chunk_size > 0) {
      ctx->pipeline_batch_size = compute_auto_pipeline_batch(
          max_chunk_size, max_num_chunks);
    } else {
      ctx->pipeline_batch_size = static_cast<int>(max_num_chunks);
    }
  }

  if (log_level >= 1) {
    int ctas_log = compute_num_ctas_per_chunk(
        max_chunk_size, ctx->pipeline_batch_size);
    size_t total_gpu = ctx->comp_temp_bytes + 2 * comp_staging_total +
                       ctx->decomp_temp_bytes +
                       4 * ptr_bytes + 5 * size_bytes +
                       2 * ptr_bytes + 2 * size_bytes +
                       max_num_chunks * sizeof(nvcompStatus_t);
    size_t total_host = size_bytes;
    printf("[nvcomp] ANSTransferContext created: max_chunks=%zu, chunk_size=%zu, "
           "max_comp_chunk=%zu, extra_gpu=%.2f MB, extra_host=%.2f MB, "
           "scatter_grid=%d, gather_grid=%d, pipeline_batch_size=%d%s, "
           "num_ctas_per_chunk=%d, blocks_per_batch=%d\n",
           max_num_chunks, max_chunk_size, ctx->max_comp_chunk_bytes,
           total_gpu / (1024.0 * 1024.0),
           total_host / (1024.0 * 1024.0),
           ctx->scatter_grid, ctx->gather_grid,
           ctx->pipeline_batch_size,
           ctx->green_ctx.valid ? "(green, no split)" :
               (pipeline_batch_size > 0) ? "" : "(auto)",
           ctas_log,
           ctx->pipeline_batch_size * ctas_log);
  }
}

void ans_ctx_destroy(ANSTransferContext* ctx) {
  nvcompGreenContextDestroy(&ctx->green_ctx);
  cudaFree(ctx->d_comp_temp);
  cudaFree(ctx->d_comp_staging_base);
  for (int i = 0; i < 2; i++) {
    cudaFree(ctx->d_comp_ptrs[i]);
    cudaFree(ctx->d_comp_sizes[i]);
    cudaEventDestroy(ctx->compress_done[i]);
    cudaEventDestroy(ctx->scatter_done[i]);
  }
  cudaStreamDestroy(ctx->scatter_stream);
  cudaFree(ctx->d_uncomp_ptrs);
  cudaFree(ctx->d_uncomp_sizes);
  cudaFree(ctx->d_decomp_temp);
  cudaFree(ctx->d_decomp_ptrs[0]);
  cudaFree(ctx->d_decomp_ptrs[1]);
  cudaFree(ctx->d_decomp_buf_sizes[0]);
  cudaFree(ctx->d_decomp_buf_sizes[1]);
  cudaFree(ctx->d_decomp_act_sizes);
  cudaFree(ctx->d_statuses);
  cudaFreeHost(ctx->h_comp_sizes[0]);
  cudaFreeHost(ctx->h_comp_sizes[1]);
}

static void ans_sync_streams(ANSTransferContext* ctx, cudaStream_t stream) {
  if (ctx->green_ctx.valid)
    ANS_CUDA_CHECK(cudaStreamSynchronize(ctx->green_ctx.stream));
  else
    ANS_CUDA_CHECK(cudaStreamSynchronize(stream));
  ANS_CUDA_CHECK(cudaStreamSynchronize(ctx->scatter_stream));
}

template<BackendType Type>
void ans_compress_and_d2h(
    ANSTransferContext* ctx,
    int num_blocks, int start_layer_id, int num_layers,
    int64_t* gpu_block_ids,
    GTensorHandler gpu_handler,
    int64_t* cpu_block_ids, void* cpu_ptr,
    int64_t cpu_kv_stride_in_bytes, int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_block_stride_in_bytes,
    int64_t chunk_size_in_bytes,
    bool is_mla,
    int64_t* h_comp_sizes_out,
    cudaStream_t stream) {

  const int kv_dim = is_mla ? 1 : 2;
  const int total_chunks = num_layers * kv_dim * num_blocks;
  const int batch_cap = ctx->pipeline_batch_size;
  const int num_batches = (total_chunks + batch_cap - 1) / batch_cap;

  assert(static_cast<size_t>(chunk_size_in_bytes) <= ctx->max_chunk_size);

  cudaStream_t comp_stream = ctx->green_ctx.valid ? ctx->green_ctx.stream : stream;

  nvtxRangePush("ANS:D2H_total");

  for (int bi = 0; bi < num_batches; bi++) {
    const int bs  = bi * batch_cap;
    const int bsz = std::min(batch_cap, total_chunks - bs);
    const int cur = bi % 2;

    if (bi >= 2)
      ANS_CUDA_CHECK(cudaStreamWaitEvent(comp_stream, ctx->scatter_done[cur], 0));

    { NvtxScope _nvtx("ANS:D2H:compress");
      GreenScope _green(&ctx->green_ctx);

      int threads = 256;
      int blocks = std::min((bsz + threads - 1) / threads, ctx->transfer_sms);
      ans_build_ptrs_kernel<Type><<<blocks, threads, 0, comp_stream>>>(
          ctx->d_uncomp_ptrs, gpu_handler, gpu_block_ids,
          kv_dim, num_blocks, bs, bsz);

      ANS_NVCOMP_CHECK(nvcompBatchedANSCompressAsync(
          (const void* const*)ctx->d_uncomp_ptrs,
          ctx->d_uncomp_sizes,
          chunk_size_in_bytes,
          bsz,
          ctx->d_comp_temp,
          ctx->comp_temp_bytes,
          ctx->d_comp_ptrs[cur],
          ctx->d_comp_sizes[cur],
          ctx->comp_opts,
          nullptr,
          comp_stream));
      ANS_CUDA_CHECK(cudaEventRecord(ctx->compress_done[cur], comp_stream));
    }

    { NvtxScope _nvtx("ANS:D2H:scatter");
      ANS_CUDA_CHECK(cudaStreamWaitEvent(ctx->scatter_stream, ctx->compress_done[cur], 0));
      int grid = std::min(bsz, ctx->scatter_grid);
      ans_d2h_scatter_kernel<<<grid, ANS_KERNEL_BLOCK_SIZE, 0, ctx->scatter_stream>>>(
          ctx->d_comp_staging[cur], ctx->max_comp_chunk_bytes, ctx->d_comp_sizes[cur],
          static_cast<uint8_t*>(cpu_ptr),
          cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes, cpu_block_stride_in_bytes,
          cpu_block_ids, start_layer_id, kv_dim, num_blocks, bs, bsz,
          h_comp_sizes_out);
      ANS_CUDA_CHECK(cudaEventRecord(ctx->scatter_done[cur], ctx->scatter_stream));
    }
  }

  ans_sync_streams(ctx, stream);
  nvtxRangePop(); // D2H_total

  if (ctx->log_level >= 1) {
    size_t grand_total_comp = 0;
    for (int i = 0; i < total_chunks; i++)
      grand_total_comp += static_cast<size_t>(h_comp_sizes_out[i]);
    size_t grand_total_uncomp = static_cast<size_t>(total_chunks) * chunk_size_in_bytes;
    printf("[FlexKV ans_compress_and_d2h] D2H: %d chunks in %d batch(es), %.2f MB -> %.2f MB (ratio %.2fx)\n",
           total_chunks, num_batches,
           grand_total_uncomp / (1024.0 * 1024.0),
           grand_total_comp   / (1024.0 * 1024.0),
           (double)grand_total_uncomp / grand_total_comp);
  }
}

template<BackendType Type>
void ans_h2d_and_decompress(
    ANSTransferContext* ctx,
    int num_blocks, int start_layer_id, int num_layers,
    int64_t* gpu_block_ids,
    GTensorHandler gpu_handler,
    int64_t* cpu_block_ids, void* cpu_ptr,
    int64_t cpu_kv_stride_in_bytes, int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_block_stride_in_bytes,
    int64_t chunk_size_in_bytes,
    bool is_mla,
    const int64_t* comp_sizes_meta, int meta_stride,
    cudaStream_t stream) {

  const int kv_dim = is_mla ? 1 : 2;
  const int total_chunks = num_layers * kv_dim * num_blocks;
  const int batch_cap = ctx->pipeline_batch_size;
  const int num_batches = (total_chunks + batch_cap - 1) / batch_cap;

  assert(static_cast<size_t>(chunk_size_in_bytes) <= ctx->max_chunk_size);
  static_assert(sizeof(size_t) == sizeof(int64_t), "size_t must be 64-bit");

  cudaStream_t decomp_stream = ctx->green_ctx.valid ? ctx->green_ctx.stream : stream;

  nvtxRangePush("ANS:H2D_total");

  for (int bi = 0; bi < num_batches; bi++) {
    const int bs  = bi * batch_cap;
    const int bsz = std::min(batch_cap, total_chunks - bs);
    const int cur = bi % 2;

    if (bi >= 2)
      ANS_CUDA_CHECK(cudaStreamWaitEvent(ctx->scatter_stream, ctx->scatter_done[cur], 0));

    { NvtxScope _nvtx("ANS:H2D:gather_comp_sizes");
      int threads = 256;
      int blocks = std::min((bsz + threads - 1) / threads, ctx->transfer_sms);
      ans_gather_comp_sizes_kernel<<<blocks, threads, 0, ctx->scatter_stream>>>(
          ctx->d_comp_sizes[cur], comp_sizes_meta, meta_stride,
          cpu_block_ids, kv_dim, num_blocks, start_layer_id, bs, bsz);
    }

    { NvtxScope _nvtx("ANS:H2D:build_ptrs");
      GreenScope _green(&ctx->green_ctx);
      int threads = 256;
      int blocks = std::min((bsz + threads - 1) / threads, ctx->transfer_sms);
      ans_build_ptrs_kernel<Type><<<blocks, threads, 0, decomp_stream>>>(
          ctx->d_decomp_ptrs[cur], gpu_handler, gpu_block_ids,
          kv_dim, num_blocks, bs, bsz);
    }

    { NvtxScope _nvtx("ANS:H2D:gather_kernel");
      int grid = std::min(bsz, ctx->gather_grid);
      ans_h2d_gather_kernel<<<grid, ANS_KERNEL_BLOCK_SIZE, 0, ctx->scatter_stream>>>(
          ctx->d_comp_staging[cur], ctx->max_comp_chunk_bytes, ctx->d_comp_sizes[cur],
          static_cast<const uint8_t*>(cpu_ptr),
          cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes, cpu_block_stride_in_bytes,
          cpu_block_ids, start_layer_id, kv_dim, num_blocks, bs, bsz);
    }

    ANS_CUDA_CHECK(cudaEventRecord(ctx->compress_done[cur], ctx->scatter_stream));

    { NvtxScope _nvtx("ANS:H2D:decompress");
      GreenScope _green(&ctx->green_ctx);
      ANS_CUDA_CHECK(cudaStreamWaitEvent(decomp_stream, ctx->compress_done[cur], 0));
      ANS_NVCOMP_CHECK(nvcompBatchedANSDecompressAsync(
          (const void* const*)ctx->d_comp_ptrs[cur],
          ctx->d_comp_sizes[cur],
          ctx->d_decomp_buf_sizes[cur],
          ctx->d_decomp_act_sizes,
          bsz,
          ctx->d_decomp_temp,
          ctx->decomp_temp_bytes,
          ctx->d_decomp_ptrs[cur],
          ctx->decomp_opts,
          ctx->d_statuses,
          decomp_stream));
      ANS_CUDA_CHECK(cudaEventRecord(ctx->scatter_done[cur], decomp_stream));
    }
  }

  ans_sync_streams(ctx, stream);
  nvtxRangePop(); // H2D_total

  if (ctx->log_level >= 1) {
    size_t grand_total_comp = 0;
    for (int g = 0; g < total_chunks; g++) {
      int meta_row = start_layer_id * kv_dim + g / num_blocks;
      int b = g % num_blocks;
      grand_total_comp += static_cast<size_t>(
          comp_sizes_meta[meta_row * meta_stride + cpu_block_ids[b]]);
    }
    size_t total_uncomp = static_cast<size_t>(total_chunks) * chunk_size_in_bytes;
    printf("[FlexKV ans_h2d_and_decompress] H2D: %d chunks in %d batch(es), %.2f MB compressed -> %.2f MB (ratio %.2fx)\n",
           total_chunks, num_batches,
           grand_total_comp / (1024.0 * 1024.0),
           total_uncomp     / (1024.0 * 1024.0),
           (double)total_uncomp / grand_total_comp);
  }
}

// Explicit template instantiations
template void ans_compress_and_d2h<BackendType::VLLM>(
    ANSTransferContext*, int, int, int, int64_t*, GTensorHandler,
    int64_t*, void*, int64_t, int64_t, int64_t, int64_t, bool,
    int64_t*, cudaStream_t);
template void ans_compress_and_d2h<BackendType::TRTLLM>(
    ANSTransferContext*, int, int, int, int64_t*, GTensorHandler,
    int64_t*, void*, int64_t, int64_t, int64_t, int64_t, bool,
    int64_t*, cudaStream_t);
template void ans_compress_and_d2h<BackendType::SGLANG>(
    ANSTransferContext*, int, int, int, int64_t*, GTensorHandler,
    int64_t*, void*, int64_t, int64_t, int64_t, int64_t, bool,
    int64_t*, cudaStream_t);

template void ans_h2d_and_decompress<BackendType::VLLM>(
    ANSTransferContext*, int, int, int, int64_t*, GTensorHandler,
    int64_t*, void*, int64_t, int64_t, int64_t, int64_t, bool,
    const int64_t*, int, cudaStream_t);
template void ans_h2d_and_decompress<BackendType::TRTLLM>(
    ANSTransferContext*, int, int, int, int64_t*, GTensorHandler,
    int64_t*, void*, int64_t, int64_t, int64_t, int64_t, bool,
    const int64_t*, int, cudaStream_t);
template void ans_h2d_and_decompress<BackendType::SGLANG>(
    ANSTransferContext*, int, int, int, int64_t*, GTensorHandler,
    int64_t*, void*, int64_t, int64_t, int64_t, int64_t, bool,
    const int64_t*, int, cudaStream_t);

} // namespace flexkv

#endif // FLEXKV_ENABLE_NVCOMP
