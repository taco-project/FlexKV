/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
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
    int64_t buffer_chunk_offset,
    int64_t chunk_size,
    GTensorHandler gpu_handler,
    int64_t* gpu_block_ids,
    int num_blocks,
    int num_layers,
    bool is_mla,
    bool buffer_to_target,
    cudaStream_t stream);

} // namespace flexkv

