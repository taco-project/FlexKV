/*
 * Vendor-scoped layout_transform header. Moved here from
 * csrc/gds/layout_transform.cuh during P3.
 *
 * GDS is NVIDIA-only (cuFile has no ROCm counterpart), so we keep the
 * native cudaStream_t signature here.
 */
#pragma once

#include <cuda_runtime.h>
#include "../gtensor_handler.cuh"

namespace flexkv {

template<BackendType Type>
void launch_layout_transform_kernel(
    int64_t* buffer_base,
    int64_t buffer_layer_stride,
    int64_t buffer_kv_stride,
    int64_t buffer_block_stride,
    int64_t chunk_size,
    GTensorHandler gpu_handler,
    int64_t* gpu_block_ids,
    int num_blocks,
    int num_layers,
    bool is_mla,
    bool buffer_to_target,
    cudaStream_t stream);

} // namespace flexkv
