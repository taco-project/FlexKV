#pragma once

#include <cuda_runtime.h>

namespace flexkv {

void transfer_kv_blocks(
    int num_blocks, int start_layer_id, int num_layers, int64_t *gpu_block_ids,
    void **gpu_layer_ptrs, int64_t gpu_kv_stride_in_bytes,
    int64_t gpu_block_stride_in_bytes, int64_t *cpu_block_ids, void *cpu_ptr,
    int64_t cpu_kv_stride_in_bytes, int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_block_stride_in_bytes, int64_t cpu_startoff_inside_chunks,
    int64_t chunk_size_in_bytes, cudaStream_t stream, int transfer_sms,
    bool is_host_to_device, bool use_ce_transfer, bool is_mla);

} // namespace flexkv
