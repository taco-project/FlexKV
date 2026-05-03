#pragma once

#ifdef FLEXKV_ENABLE_NVCOMP

#include <cuda_runtime.h>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "nvcomp/ans.h"
#include "nvcomp/green_context.h"
#include "gtensor_handler.cuh"

namespace flexkv {

static constexpr int COMP_HEADER_SIZE = 16;

struct ANSTransferContext {
    // Chunk geometry
    size_t max_num_chunks;
    size_t max_chunk_size;          // uncompressed chunk bytes
    size_t max_comp_chunk_bytes;    // max compressed chunk bytes (16-byte aligned)
    size_t comp_temp_bytes;
    size_t decomp_temp_bytes;

    nvcompBatchedANSCompressOpts_t   comp_opts;
    nvcompBatchedANSDecompressOpts_t decomp_opts;

    // GPU buffers — compression
    void*    d_comp_temp;
    uint8_t* d_comp_staging_base;   // contiguous alloc backing both slots
    uint8_t* d_comp_staging[2];     // double-buffered staging
    void**   d_uncomp_ptrs;
    size_t*  d_uncomp_sizes;
    void**   d_comp_ptrs[2];
    size_t*  d_comp_sizes[2];

    // GPU buffers — decompression
    void*           d_decomp_temp;
    void**          d_decomp_ptrs[2];
    size_t*         d_decomp_buf_sizes[2];
    size_t*         d_decomp_act_sizes;
    nvcompStatus_t* d_statuses;

    // Host scratch (for ctx_create pointer setup)
    std::vector<void*>  h_ptr_scratch;
    std::vector<size_t> h_size_scratch;

    // Kernel launch config
    int scatter_grid;
    int gather_grid;
    int transfer_sms;

    // Double-buffer pipeline
    cudaStream_t scatter_stream;
    cudaEvent_t  compress_done[2];
    cudaEvent_t  scatter_done[2];
    int pipeline_batch_size;
    nvcompGreenContext_t green_ctx;

    int log_level;
};

// Wrapper around nvcompBatchedANSGetSubChunkingConfig.
// Returns num_ctas_per_chunk for a given (chunk_size, batch_size, device).
inline int compute_num_ctas_per_chunk(size_t max_chunk_size, int batch_size) {
  int num_ctas = 1, max_sub_chunk_size = 0;
  nvcompBatchedANSGetSubChunkingConfig(
      max_chunk_size, static_cast<size_t>(batch_size), 0,
      &num_ctas, &max_sub_chunk_size);
  return num_ctas;
}

// Find the optimal pipeline_batch_size for minimal SM occupancy while
// preserving the best possible compression ratio.
inline int compute_auto_pipeline_batch(
    size_t max_chunk_size, size_t max_num_chunks)
{
  int device_id, num_sms, max_threads_per_sm;
  cudaGetDevice(&device_id);
  cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, device_id);
  cudaDeviceGetAttribute(&max_threads_per_sm,
                         cudaDevAttrMaxThreadsPerMultiProcessor, device_id);

  int large_batch = num_sms * (max_threads_per_sm / 32);
  int best_ctas = compute_num_ctas_per_chunk(max_chunk_size, large_batch);

  int lo = 64, hi = large_batch;
  while (lo < hi) {
    int mid = lo + (hi - lo) / 2;
    if (compute_num_ctas_per_chunk(max_chunk_size, mid) <= best_ctas)
      hi = mid;
    else
      lo = mid + 1;
  }
  int threshold = lo;

  if (max_num_chunks == 0)
    return threshold;
  return std::min(static_cast<int>(max_num_chunks), threshold);
}

void ans_ctx_create(ANSTransferContext* ctx, size_t max_num_chunks,
                    size_t max_chunk_size, int data_type, int log_level,
                    int pipeline_batch_size = 0, int transfer_sms = -1);

void ans_ctx_destroy(ANSTransferContext* ctx);

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
    cudaStream_t stream);

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
    cudaStream_t stream);

} // namespace flexkv

#endif // FLEXKV_ENABLE_NVCOMP
