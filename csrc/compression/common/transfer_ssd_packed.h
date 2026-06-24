/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "transfer_ssd.h"

#include <cstdint>
#include <string>
#include <torch/extension.h>

namespace flexkv {

int64_t transfer_kv_blocks_ssd_packed(
    SSDIOCTX &ioctx, const torch::Tensor &cpu_layer_id_list,
    int64_t cpu_tensor_ptr, const torch::Tensor &ssd_block_ids,
    const torch::Tensor &cpu_block_ids, int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes, int64_t chunk_size_in_bytes,
    int64_t block_stride_in_bytes, bool is_read, int num_blocks_per_file,
    int round_robin, int num_threads_per_device, bool is_mla,
    const std::string &layout_type, int total_layers,
    uint32_t* cpu_size_table_base, int64_t cpu_size_table_block_stride,
    int64_t cpu_size_table_layer_stride, uint32_t* ssd_size_table_base,
    int64_t ssd_size_table_block_stride,
    int64_t ssd_size_table_layer_stride, int tp_size = 1,
    int64_t cpu_tp_rank_stride_in_bytes = 0,
    int64_t cpu_size_table_rank_stride = 0,
    int64_t ssd_size_table_rank_stride = 0);

} // namespace flexkv
