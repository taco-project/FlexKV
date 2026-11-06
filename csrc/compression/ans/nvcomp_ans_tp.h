/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "compression/ans/nvcomp_ans.cuh"

#include <cstdint>
#include <vector>

namespace flexkv {

struct NvcompTPState {
  bool ready = false;
  int batch_size = 0;
  int data_type = 0;
  std::vector<ANSTransferContext*> ans_contexts;

  // Per-rank scratch for MLA D2H owner-sharding (block b is owned by rank
  // cpu_block_ids[b] % num_gpus). Grown on demand by tp_group_transfer_ans().
  int64_t** owned_gpu_block_ids = nullptr;
  int64_t** owned_cpu_block_ids = nullptr;
  int64_t* owned_block_id_capacity = nullptr;
};

} // namespace flexkv
