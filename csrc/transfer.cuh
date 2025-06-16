#pragma once

#include <cuda_runtime.h>

namespace flexkv {

void transfer_kv_layers(int num_blocks, int num_layers, int64_t *dst_block_ids,
                        void **dst_layer_ptrs, int64_t dst_kv_stride_in_bytes,
                        int64_t dst_chunk_stride_in_bytes,
                        int64_t dst_startoff_inside_chunks,
                        int64_t *src_block_ids, void **src_layer_ptrs,
                        int64_t src_kv_stride_in_bytes,
                        int64_t src_chunk_stride_in_bytes,
                        int64_t src_startoff_inside_chunks,
                        int64_t chunk_size_in_bytes, cudaStream_t stream,
                        int transfer_sms, bool is_host_to_device,
                        bool use_ce_transfer, bool is_mla);

} // namespace flexkv
