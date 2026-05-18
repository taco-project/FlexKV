/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Vendor-scoped header for transfer kernels. Moved here from
 * csrc/transfer.cuh during the GPU backend abstraction refactor (P3).
 */
#pragma once

#include "gtensor_handler.cuh"

namespace flexkv {

// Template function for transfer, specialized for each LLM backend type.
template <BackendType Type>
void transfer_kv_blocks(
    int num_blocks, int start_layer_id, int num_layers, int64_t *gpu_block_ids,
    GTensorHandler gpu_tensor_handler, // Pass by value!
    int64_t gpu_startoff_inside_chunks, int64_t *cpu_block_ids, void *cpu_ptr,
    int64_t cpu_kv_stride_in_bytes, int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_block_stride_in_bytes, int64_t cpu_startoff_inside_chunks,
    int64_t chunk_size_in_bytes, gpu_stream_t stream, int transfer_num_cta,
    bool is_host_to_device, bool use_ce_transfer, bool is_mla);

} // namespace flexkv
