#pragma once

#ifdef FLEXKV_ENABLE_NVCOMP

#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "nvcomp/ans.h"
#include "gtensor_handler.cuh"

namespace flexkv {

struct ANSTransferContext {
    size_t max_num_chunks;      // fixed batch budget (e.g. 4096)
    size_t max_chunk_size;
    size_t max_comp_chunk_bytes;
    size_t comp_temp_bytes;
    size_t decomp_temp_bytes;

    nvcompBatchedANSCompressOpts_t   comp_opts;
    nvcompBatchedANSDecompressOpts_t decomp_opts;

    // GPU buffers: compression I/O (sized for max_num_chunks)
    void*    d_comp_temp;
    uint8_t* d_comp_staging_base; // single contiguous allocation for both slots
    uint8_t* d_comp_staging[2];   // double-buffered, pointing into d_comp_staging_base
    void**   d_uncomp_ptrs;
    size_t*  d_uncomp_sizes;
    void**   d_comp_ptrs[2];     // double-buffered (point into respective d_comp_staging)
    size_t*  d_comp_sizes[2];    // double-buffered

    // Double-buffer pipeline resources (shared by D2H and H2D)
    // D2H: scatter_stream runs scatter, compress_done/scatter_done sync
    // H2D: scatter_stream runs gather, compress_done/scatter_done sync
    cudaStream_t scatter_stream;
    cudaEvent_t  compress_done[2];
    cudaEvent_t  scatter_done[2];

    // GPU buffers: decompression I/O
    void*           d_decomp_temp;
    void**          d_decomp_ptrs[2];       // double-buffered for H2D pipeline
    size_t*         d_decomp_buf_sizes[2];  // double-buffered, pre-filled
    size_t*         d_decomp_act_sizes;
    nvcompStatus_t* d_statuses;

    // Host pinned buffer for compressed sizes metadata (double-buffered for H2D pipeline)
    size_t*  h_comp_sizes[2];

    // Host scratch (reused across calls)
    std::vector<void*>  h_ptr_scratch;
    std::vector<size_t> h_size_scratch;

    // Kernel launch config (computed once via occupancy API)
    int scatter_grid;   // grid size for scatter/gather kernels
    int gather_grid;

    int pipeline_batch_size;  // chunks per pipeline stage (≤ max_num_chunks)

    int log_level;
};

void ans_ctx_create(ANSTransferContext* ctx, size_t max_num_chunks,
                    size_t max_chunk_size, int data_type, int log_level,
                    int pipeline_batch_size = 0);

void ans_ctx_destroy(ANSTransferContext* ctx);

// D2H: compress on GPU → GPU scatter kernel to CPU pinned (internally batched)
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
    cudaStream_t stream);

// H2D: GPU gather kernel from CPU pinned → decompress on GPU (internally batched)
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
    cudaStream_t stream);

} // namespace flexkv

#endif // FLEXKV_ENABLE_NVCOMP
