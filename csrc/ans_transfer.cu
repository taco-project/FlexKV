#ifdef FLEXKV_ENABLE_NVCOMP

#include "ans_transfer.h"

#include <cassert>
#include <cstdio>
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

// ---------------------------------------------------------------------------
// GPU kernels: direct scatter D2H / gather H2D via PCIe
// GPU threads read/write CPU pinned memory directly, eliminating intermediate
// pack/unpack + cudaMemcpy + CPU scatter/gather stages.
// ---------------------------------------------------------------------------

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
    int batch_start, int bsz)
{
    for (int i = blockIdx.x; i < bsz; i += gridDim.x) {
        int g = batch_start + i;
        int layer = g / (kv_dim * num_blocks);
        int kv = (g % (kv_dim * num_blocks)) / num_blocks;
        int b = g % num_blocks;

        size_t sz = d_comp_sizes[i];
        const uint8_t* src = d_comp_staging + (size_t)i * staging_stride;
        uint8_t* dst = cpu_ptr
                     + (int64_t)(layer + start_layer_id) * cpu_layer_stride
                     + (int64_t)kv * cpu_kv_stride
                     + cpu_block_ids[b] * cpu_block_stride;

        size_t n_f4 = sz / sizeof(float4);
        for (size_t j = threadIdx.x; j < n_f4; j += blockDim.x)
            reinterpret_cast<float4*>(dst)[j] =
                reinterpret_cast<const float4*>(src)[j];
        size_t tail = n_f4 * sizeof(float4);
        for (size_t j = tail + threadIdx.x; j < sz; j += blockDim.x)
            dst[j] = src[j];
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
        uint8_t* dst = d_comp_staging + (size_t)i * staging_stride;
        const uint8_t* src = cpu_ptr
                           + (int64_t)(layer + start_layer_id) * cpu_layer_stride
                           + (int64_t)kv * cpu_kv_stride
                           + cpu_block_ids[b] * cpu_block_stride;

        size_t n_f4 = sz / sizeof(float4);
        for (size_t j = threadIdx.x; j < n_f4; j += blockDim.x)
            reinterpret_cast<float4*>(dst)[j] =
                reinterpret_cast<const float4*>(src)[j];
        size_t tail = n_f4 * sizeof(float4);
        for (size_t j = tail + threadIdx.x; j < sz; j += blockDim.x)
            dst[j] = src[j];
    }
}

static const int ANS_TRANSFER_SMS = 8;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static inline void decode_chunk_idx(int flat, int kv_dim, int num_blocks,
                                     int& layer, int& kv, int& b) {
    layer = flat / (kv_dim * num_blocks);
    int rem = flat % (kv_dim * num_blocks);
    kv = rem / num_blocks;
    b = rem % num_blocks;
}

static inline uint8_t* cpu_chunk_addr(
    void* cpu_ptr, int layer, int start_layer_id, int kv, int64_t cpu_block_id,
    int64_t layer_stride, int64_t kv_stride, int64_t block_stride)
{
    return static_cast<uint8_t*>(cpu_ptr) +
           static_cast<size_t>(layer + start_layer_id) * layer_stride +
           static_cast<size_t>(kv) * kv_stride +
           static_cast<size_t>(cpu_block_id) * block_stride;
}

// ---------------------------------------------------------------------------
// Context lifecycle
// ---------------------------------------------------------------------------

void ans_ctx_create(ANSTransferContext* ctx, size_t max_num_chunks,
                    size_t max_chunk_size, int data_type, int log_level) {
  ctx->max_num_chunks = max_num_chunks;
  ctx->max_chunk_size = max_chunk_size;
  ctx->log_level      = log_level;

  ctx->comp_opts = nvcompBatchedANSCompressDefaultOpts;
  switch (data_type) {
    case 0:  ctx->comp_opts.data_type = NVCOMP_TYPE_FLOAT16; break;
    case 1:  ctx->comp_opts.data_type = NVCOMP_TYPE_FLOAT16; break;
    default: ctx->comp_opts.data_type = NVCOMP_TYPE_FLOAT16; break;
  }
  ctx->decomp_opts = nvcompBatchedANSDecompressDefaultOpts;

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

  // GPU compression buffers
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_comp_temp,    ctx->comp_temp_bytes));
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_comp_staging, comp_staging_total));
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_uncomp_ptrs,  ptr_bytes));
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_uncomp_sizes, size_bytes));
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_comp_ptrs,    ptr_bytes));
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_comp_sizes,   size_bytes));

  // GPU decompression buffers
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_temp,      ctx->decomp_temp_bytes));
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_ptrs,      ptr_bytes));
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_buf_sizes, size_bytes));
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_decomp_act_sizes, size_bytes));
  ANS_CUDA_CHECK(cudaMalloc(&ctx->d_statuses,
      max_num_chunks * sizeof(nvcompStatus_t)));

  // Host pinned buffer for compressed sizes metadata
  ANS_CUDA_CHECK(cudaMallocHost(&ctx->h_comp_sizes,   size_bytes));

  // Host scratch vectors
  ctx->h_ptr_scratch.resize(max_num_chunks);
  ctx->h_size_scratch.resize(max_num_chunks);

  // Pre-fill d_comp_ptrs: each points into d_comp_staging at stride
  for (size_t i = 0; i < max_num_chunks; i++)
    ctx->h_ptr_scratch[i] = ctx->d_comp_staging + i * ctx->max_comp_chunk_bytes;
  ANS_CUDA_CHECK(cudaMemcpy(ctx->d_comp_ptrs, ctx->h_ptr_scratch.data(),
                             ptr_bytes, cudaMemcpyHostToDevice));

  if (log_level >= 1) {
    size_t total_gpu = ctx->comp_temp_bytes + comp_staging_total +
                       ctx->decomp_temp_bytes +
                       4 * ptr_bytes + 5 * size_bytes +
                       max_num_chunks * sizeof(nvcompStatus_t);
    size_t total_host = size_bytes;
    printf("[nvcomp] ANSTransferContext created: max_chunks=%zu, chunk_size=%zu, "
           "max_comp_chunk=%zu, extra_gpu=%.2f MB, extra_host=%.2f MB\n",
           max_num_chunks, max_chunk_size, ctx->max_comp_chunk_bytes,
           total_gpu / (1024.0 * 1024.0),
           total_host / (1024.0 * 1024.0));
  }
}

void ans_ctx_destroy(ANSTransferContext* ctx) {
  cudaFree(ctx->d_comp_temp);
  cudaFree(ctx->d_comp_staging);
  cudaFree(ctx->d_uncomp_ptrs);
  cudaFree(ctx->d_uncomp_sizes);
  cudaFree(ctx->d_comp_ptrs);
  cudaFree(ctx->d_comp_sizes);
  cudaFree(ctx->d_decomp_temp);
  cudaFree(ctx->d_decomp_ptrs);
  cudaFree(ctx->d_decomp_buf_sizes);
  cudaFree(ctx->d_decomp_act_sizes);
  cudaFree(ctx->d_statuses);
  cudaFreeHost(ctx->h_comp_sizes);
}

// ---------------------------------------------------------------------------
// D2H: compress on GPU → pack → single D2H → CPU scatter
// Internally batched by ctx->max_num_chunks.
// ---------------------------------------------------------------------------

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
  const int batch_cap = static_cast<int>(ctx->max_num_chunks);

  assert(static_cast<size_t>(chunk_size_in_bytes) <= ctx->max_chunk_size);

  size_t grand_total_comp = 0;
  size_t grand_total_uncomp = 0;

  nvtxRangePush("ANS:D2H_total");

  for (int bs = 0; bs < total_chunks; bs += batch_cap) {
    const int bsz = std::min(batch_cap, total_chunks - bs);
    const size_t ptr_bytes  = bsz * sizeof(void*);
    const size_t size_bytes = bsz * sizeof(size_t);

    // 1. Build GPU source pointer array: given gpu block ids, get the gpu address and upload to GPU
    // TODO: build_uncomp_ptrs_kernel
    nvtxRangePush("ANS:D2H:build_ptrs");
    for (int i = 0; i < bsz; i++) {
      int layer, kv, b;
      decode_chunk_idx(bs + i, kv_dim, num_blocks, layer, kv, b);
      ctx->h_ptr_scratch[i] = static_cast<void*>(
          ptr_at<Type>(gpu_handler, layer, kv, gpu_block_ids[b]));
    }
    ANS_CUDA_CHECK(cudaMemcpyAsync(ctx->d_uncomp_ptrs, ctx->h_ptr_scratch.data(),
                                    ptr_bytes, cudaMemcpyHostToDevice, stream));
    nvtxRangePop();

    nvtxRangePush("ANS:D2H:set_uniform_chunk_sizes");
    // 2. Set uniform chunk sizes
    for (int i = 0; i < bsz; i++)
      ctx->h_size_scratch[i] = static_cast<size_t>(chunk_size_in_bytes);
    ANS_CUDA_CHECK(cudaMemcpyAsync(ctx->d_uncomp_sizes, ctx->h_size_scratch.data(),
                                    size_bytes, cudaMemcpyHostToDevice, stream));
    ANS_CUDA_CHECK(cudaStreamSynchronize(stream));
    nvtxRangePop();

    // 3. ANS compress
    nvtxRangePush("ANS:D2H:compress");
    ANS_NVCOMP_CHECK(nvcompBatchedANSCompressAsync(
        (const void* const*)ctx->d_uncomp_ptrs,
        ctx->d_uncomp_sizes,
        chunk_size_in_bytes,
        bsz,
        ctx->d_comp_temp,
        ctx->comp_temp_bytes,
        ctx->d_comp_ptrs,
        ctx->d_comp_sizes,
        ctx->comp_opts,
        nullptr,
        stream));
    ANS_CUDA_CHECK(cudaStreamSynchronize(stream));
    nvtxRangePop();

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    // 4. Read back compressed sizes
    nvtxRangePush("ANS:D2H:read_sizes");
    ANS_CUDA_CHECK(cudaMemcpyAsync(ctx->h_comp_sizes, ctx->d_comp_sizes,
                                    size_bytes, cudaMemcpyDeviceToHost, stream));
    ANS_CUDA_CHECK(cudaStreamSynchronize(stream));
    nvtxRangePop();

    // 5. Fallback check: if any chunk doesn't compress smaller, abort
    for (int i = 0; i < bsz; i++) {
      if (ctx->h_comp_sizes[i] >= static_cast<size_t>(chunk_size_in_bytes)) {
        if (ctx->log_level >= 1) {
          printf("[nvcomp] D2H batch %d: chunk %d comp_size=%zu >= chunk_size=%lld, "
                 "signaling fallback\n",
                 bs / batch_cap, i, ctx->h_comp_sizes[i],
                 (long long)chunk_size_in_bytes);
        }
        h_comp_sizes_out[0] = -1;
        nvtxRangePop(); // D2H_total
        return;
      }
    }

    // 6. GPU scatter kernel: d_comp_staging → CPU block slots (direct PCIe write)
    nvtxRangePush("ANS:D2H:scatter_kernel");
    {
      int grid = std::min(bsz, ANS_TRANSFER_SMS);
      ans_d2h_scatter_kernel<<<grid, 256, 0, stream>>>(
          ctx->d_comp_staging, ctx->max_comp_chunk_bytes, ctx->d_comp_sizes,
          static_cast<uint8_t*>(cpu_ptr),
          cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes, cpu_block_stride_in_bytes,
          cpu_block_ids, start_layer_id, kv_dim, num_blocks, bs, bsz);
      ANS_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    nvtxRangePop();

    // 7. Record compressed sizes on host (already in h_comp_sizes from step 4)
    for (int i = 0; i < bsz; i++) {
      h_comp_sizes_out[bs + i] = static_cast<int64_t>(ctx->h_comp_sizes[i]);
      grand_total_comp += ctx->h_comp_sizes[i];
    }
    grand_total_uncomp += static_cast<size_t>(bsz) * chunk_size_in_bytes;
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
  }

  nvtxRangePop(); // D2H_total

  if (ctx->log_level >= 1) {
    int num_batches = (total_chunks + batch_cap - 1) / batch_cap;
    printf("[nvcomp] D2H: %d chunks in %d batch(es), %.2f MB -> %.2f MB (ratio %.2fx)\n",
           total_chunks, num_batches,
           grand_total_uncomp / (1024.0 * 1024.0),
           grand_total_comp   / (1024.0 * 1024.0),
           (double)grand_total_uncomp / grand_total_comp);
  }
}

// ---------------------------------------------------------------------------
// H2D: GPU gather kernel (CPU pinned → d_comp_staging via PCIe) → decompress
// Internally batched by ctx->max_num_chunks.
// Mixed compressed/uncompressed chunks handled via slow path.
// ---------------------------------------------------------------------------

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
    const int64_t* h_comp_sizes_in,
    cudaStream_t stream) {

  const int kv_dim = is_mla ? 1 : 2;
  const int total_chunks = num_layers * kv_dim * num_blocks;
  const int batch_cap = static_cast<int>(ctx->max_num_chunks);

  assert(static_cast<size_t>(chunk_size_in_bytes) <= ctx->max_chunk_size);

  size_t grand_total_comp = 0;

  nvtxRangePush("ANS:H2D_total");

  for (int bs = 0; bs < total_chunks; bs += batch_cap) {
    const int bsz = std::min(batch_cap, total_chunks - bs);

    // Check if all chunks in this batch are compressed
    bool all_compressed = true;
    for (int i = 0; i < bsz; i++) {
      if (h_comp_sizes_in[bs + i] >= chunk_size_in_bytes) {
        all_compressed = false;
        break;
      }
    }

    if (!all_compressed) {
      // --- Slow path: mixed compressed/uncompressed ---
      nvtxRangePush("ANS:H2D:slow_path");
      int comp_slot = 0;
      for (int i = 0; i < bsz; i++) {
        int flat = bs + i;
        int layer, kv, b;
        decode_chunk_idx(flat, kv_dim, num_blocks, layer, kv, b);
        size_t comp_size = static_cast<size_t>(h_comp_sizes_in[flat]);

        uint8_t* cpu_src = cpu_chunk_addr(cpu_ptr, layer, start_layer_id, kv,
                                           cpu_block_ids[b],
                                           cpu_layer_stride_in_bytes,
                                           cpu_kv_stride_in_bytes,
                                           cpu_block_stride_in_bytes);

        if (comp_size >= static_cast<size_t>(chunk_size_in_bytes)) {
          int64_t* gpu_dst = ptr_at<Type>(gpu_handler, layer, kv, gpu_block_ids[b]);
          ANS_CUDA_CHECK(cudaMemcpyAsync(gpu_dst, cpu_src, chunk_size_in_bytes,
                                          cudaMemcpyHostToDevice, stream));
        } else {
          uint8_t* staging = ctx->d_comp_staging +
                             comp_slot * ctx->max_comp_chunk_bytes;
          ANS_CUDA_CHECK(cudaMemcpyAsync(staging, cpu_src, comp_size,
                                          cudaMemcpyHostToDevice, stream));
          ctx->h_comp_sizes[comp_slot] = comp_size;
          ctx->h_ptr_scratch[comp_slot] = static_cast<void*>(
              ptr_at<Type>(gpu_handler, layer, kv, gpu_block_ids[b]));
          ctx->h_size_scratch[comp_slot] = static_cast<size_t>(chunk_size_in_bytes);
          grand_total_comp += comp_size;
          comp_slot++;
        }
      }

      if (comp_slot > 0) {
        const size_t sub_ptr_bytes  = comp_slot * sizeof(void*);
        const size_t sub_size_bytes = comp_slot * sizeof(size_t);

        ANS_CUDA_CHECK(cudaMemcpyAsync(ctx->d_comp_sizes, ctx->h_comp_sizes,
                                        sub_size_bytes, cudaMemcpyHostToDevice, stream));
        ANS_CUDA_CHECK(cudaMemcpyAsync(ctx->d_decomp_ptrs, ctx->h_ptr_scratch.data(),
                                        sub_ptr_bytes, cudaMemcpyHostToDevice, stream));
        ANS_CUDA_CHECK(cudaMemcpyAsync(ctx->d_decomp_buf_sizes, ctx->h_size_scratch.data(),
                                        sub_size_bytes, cudaMemcpyHostToDevice, stream));
        ANS_CUDA_CHECK(cudaStreamSynchronize(stream));

        ANS_NVCOMP_CHECK(nvcompBatchedANSDecompressAsync(
            (const void* const*)ctx->d_comp_ptrs,
            ctx->d_comp_sizes,
            ctx->d_decomp_buf_sizes,
            ctx->d_decomp_act_sizes,
            comp_slot,
            ctx->d_decomp_temp,
            ctx->decomp_temp_bytes,
            ctx->d_decomp_ptrs,
            ctx->decomp_opts,
            ctx->d_statuses,
            stream));
        ANS_CUDA_CHECK(cudaStreamSynchronize(stream));
      } else {
        ANS_CUDA_CHECK(cudaStreamSynchronize(stream));
      }
      nvtxRangePop();
      continue;
    }

    // --- Fast path: all compressed ---

    // 1. Upload comp_sizes to GPU + gather kernel (CPU → d_comp_staging via PCIe)
    nvtxRangePush("ANS:H2D:gather_kernel");
    {
      const size_t size_bytes = bsz * sizeof(size_t);
      for (int i = 0; i < bsz; i++) {
        ctx->h_comp_sizes[i] = static_cast<size_t>(h_comp_sizes_in[bs + i]);
        grand_total_comp += ctx->h_comp_sizes[i];
      }
      ANS_CUDA_CHECK(cudaMemcpyAsync(ctx->d_comp_sizes, ctx->h_comp_sizes,
                                      size_bytes, cudaMemcpyHostToDevice, stream));

      int grid = std::min(bsz, ANS_TRANSFER_SMS);
      ans_h2d_gather_kernel<<<grid, 256, 0, stream>>>(
          ctx->d_comp_staging, ctx->max_comp_chunk_bytes, ctx->d_comp_sizes,
          static_cast<const uint8_t*>(cpu_ptr),
          cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes, cpu_block_stride_in_bytes,
          cpu_block_ids, start_layer_id, kv_dim, num_blocks, bs, bsz);
    }
    nvtxRangePop();

    // 2. Build decompress output pointers (CPU, overlaps with gather kernel)
    nvtxRangePush("ANS:H2D:build_ptrs");
    for (int i = 0; i < bsz; i++) {
      int layer, kv, b;
      decode_chunk_idx(bs + i, kv_dim, num_blocks, layer, kv, b);
      ctx->h_ptr_scratch[i] = static_cast<void*>(
          ptr_at<Type>(gpu_handler, layer, kv, gpu_block_ids[b]));
      ctx->h_size_scratch[i] = static_cast<size_t>(chunk_size_in_bytes);
    }
    {
      const size_t size_bytes = bsz * sizeof(size_t);
      const size_t ptr_bytes = bsz * sizeof(void*);
      ANS_CUDA_CHECK(cudaMemcpyAsync(ctx->d_decomp_ptrs, ctx->h_ptr_scratch.data(),
                                      ptr_bytes, cudaMemcpyHostToDevice, stream));
      ANS_CUDA_CHECK(cudaMemcpyAsync(ctx->d_decomp_buf_sizes, ctx->h_size_scratch.data(),
                                      size_bytes, cudaMemcpyHostToDevice, stream));
      ANS_CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    nvtxRangePop();

    // 3. ANS decompress
    nvtxRangePush("ANS:H2D:decompress");
    ANS_NVCOMP_CHECK(nvcompBatchedANSDecompressAsync(
        (const void* const*)ctx->d_comp_ptrs,
        ctx->d_comp_sizes,
        ctx->d_decomp_buf_sizes,
        ctx->d_decomp_act_sizes,
        bsz,
        ctx->d_decomp_temp,
        ctx->decomp_temp_bytes,
        ctx->d_decomp_ptrs,
        ctx->decomp_opts,
        ctx->d_statuses,
        stream));
    ANS_CUDA_CHECK(cudaStreamSynchronize(stream));
    nvtxRangePop();

    if (ctx->log_level >= 2) {
      std::vector<nvcompStatus_t> h_statuses(bsz);
      ANS_CUDA_CHECK(cudaMemcpy(h_statuses.data(), ctx->d_statuses,
          bsz * sizeof(nvcompStatus_t), cudaMemcpyDeviceToHost));
      for (int i = 0; i < bsz; i++) {
        if (h_statuses[i] != nvcompSuccess)
          fprintf(stderr, "[nvcomp] ERROR: batch chunk %d decompress status = %d\n",
                  i, (int)h_statuses[i]);
      }
    }
  }

  nvtxRangePop(); // H2D_total

  if (ctx->log_level >= 1) {
    size_t total_uncomp = static_cast<size_t>(total_chunks) * chunk_size_in_bytes;
    int num_batches = (total_chunks + batch_cap - 1) / batch_cap;
    printf("[nvcomp] H2D: %d chunks in %d batch(es), %.2f MB compressed -> %.2f MB (ratio %.2fx)\n",
           total_chunks, num_batches,
           grand_total_comp / (1024.0 * 1024.0),
           total_uncomp     / (1024.0 * 1024.0),
           (double)total_uncomp / grand_total_comp);
  }
}

// ---------------------------------------------------------------------------
// Explicit template instantiations
// ---------------------------------------------------------------------------

#define INSTANTIATE_ANS(TYPE) \
  template void ans_compress_and_d2h<TYPE>( \
      ANSTransferContext*, int, int, int, int64_t*, GTensorHandler, \
      int64_t*, void*, int64_t, int64_t, int64_t, int64_t, bool, \
      int64_t*, cudaStream_t); \
  template void ans_h2d_and_decompress<TYPE>( \
      ANSTransferContext*, int, int, int, int64_t*, GTensorHandler, \
      int64_t*, void*, int64_t, int64_t, int64_t, int64_t, bool, \
      const int64_t*, cudaStream_t);

INSTANTIATE_ANS(BackendType::VLLM)
INSTANTIATE_ANS(BackendType::TRTLLM)
INSTANTIATE_ANS(BackendType::SGLANG)

#undef INSTANTIATE_ANS

} // namespace flexkv

#endif // FLEXKV_ENABLE_NVCOMP
