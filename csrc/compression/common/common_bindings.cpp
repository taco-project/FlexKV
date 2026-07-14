/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */
#include "compression/common/transfer_ssd_packed.h"

#include <pybind11/pybind11.h>
#include <torch/extension.h>

namespace py = pybind11;

namespace flexkv {

void register_common_compression_bindings(py::module_& m) {
  m.def("transfer_kv_blocks_ssd_packed",
        [](flexkv::SSDIOCTX &ioctx, const torch::Tensor &cpu_layer_id_list,
           int64_t cpu_tensor_ptr, const torch::Tensor &ssd_block_ids,
           const torch::Tensor &cpu_block_ids,
           int64_t cpu_layer_stride_in_bytes,
           int64_t cpu_kv_stride_in_bytes, int64_t chunk_size_in_bytes,
           int64_t block_stride_in_bytes, bool is_read,
           int num_blocks_per_file, const std::string &layout_type,
           int total_layers, int64_t cpu_size_table_ptr,
           int64_t cpu_size_table_block_stride,
           int64_t cpu_size_table_layer_stride, int64_t ssd_size_table_ptr,
           int64_t ssd_size_table_block_stride,
           int64_t ssd_size_table_layer_stride, int round_robin,
           int num_threads_per_device, bool is_mla, int tp_size,
           int64_t cpu_tp_rank_stride_in_bytes,
           int64_t cpu_size_table_rank_stride,
           int64_t ssd_size_table_rank_stride) {
          return flexkv::transfer_kv_blocks_ssd_packed(
              ioctx, cpu_layer_id_list, cpu_tensor_ptr, ssd_block_ids,
              cpu_block_ids, cpu_layer_stride_in_bytes,
              cpu_kv_stride_in_bytes, chunk_size_in_bytes,
              block_stride_in_bytes, is_read, num_blocks_per_file,
              round_robin, num_threads_per_device, is_mla, layout_type,
              total_layers, reinterpret_cast<uint32_t*>(cpu_size_table_ptr),
              cpu_size_table_block_stride, cpu_size_table_layer_stride,
              reinterpret_cast<uint32_t*>(ssd_size_table_ptr),
              ssd_size_table_block_stride, ssd_size_table_layer_stride,
              tp_size, cpu_tp_rank_stride_in_bytes,
              cpu_size_table_rank_stride, ssd_size_table_rank_stride);
        },
        "Transfer compressed KV blocks between SSD and CPU using "
        "one packed compressed range per block.",
        py::arg("ioctx"), py::arg("cpu_layer_id_list"),
        py::arg("cpu_tensor_ptr"), py::arg("ssd_block_ids"),
        py::arg("cpu_block_ids"), py::arg("cpu_layer_stride_in_bytes"),
        py::arg("cpu_kv_stride_in_bytes"), py::arg("chunk_size_in_bytes"),
        py::arg("block_stride_in_bytes"), py::arg("is_read"),
        py::arg("num_blocks_per_file"), py::arg("layout_type"),
        py::arg("total_layers"), py::arg("cpu_size_table_ptr"),
        py::arg("cpu_size_table_block_stride"),
        py::arg("cpu_size_table_layer_stride"),
        py::arg("ssd_size_table_ptr"),
        py::arg("ssd_size_table_block_stride"),
        py::arg("ssd_size_table_layer_stride"), py::arg("round_robin") = 1,
        py::arg("num_threads_per_device") = 16, py::arg("is_mla") = false,
        py::arg("tp_size") = 1, py::arg("cpu_tp_rank_stride_in_bytes") = 0,
        py::arg("cpu_size_table_rank_stride") = 0,
        py::arg("ssd_size_table_rank_stride") = 0);
}

} // namespace flexkv
