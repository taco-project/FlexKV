/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */
#include "compression/ans/nvcomp_ans.cuh"

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <memory>
#include <pybind11/pybind11.h>
#include <stdexcept>
#include <string>
#include <torch/extension.h>

namespace py = pybind11;

namespace flexkv {

static size_t transfer_kv_blocks_ans_binding(
    bool is_d2h,
    flexkv::ANSTransferContext& ctx,
    torch::Tensor &gpu_block_id_tensor,
    torch::Tensor &gpu_tensor_ptrs_tensor,
    int64_t gpu_kv_stride_in_bytes,
    int64_t gpu_block_stride_in_bytes,
    int64_t gpu_layer_stride_in_bytes,
    torch::Tensor &cpu_block_id_tensor,
    torch::Tensor &cpu_tensor,
    int64_t cpu_kv_stride_in_bytes,
    int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_block_stride_in_bytes,
    int64_t chunk_size_in_bytes,
    int start_layer_id,
    int num_layers,
    bool is_mla,
    int gpu_block_type,
    int64_t cpu_size_table_ptr,
    int64_t cpu_size_table_block_stride,
    int64_t cpu_size_table_layer_stride) {
  TORCH_CHECK(gpu_block_id_tensor.dtype() == torch::kInt64,
              "gpu_block_id_tensor must be int64");
  TORCH_CHECK(cpu_block_id_tensor.dtype() == torch::kInt64,
              "cpu_block_id_tensor must be int64");
  TORCH_CHECK(gpu_tensor_ptrs_tensor.dtype() == torch::kInt64,
              "gpu_tensor_ptrs_tensor must be int64");

  int num_blocks = gpu_block_id_tensor.numel();
  int64_t *gpu_block_ids = static_cast<int64_t*>(gpu_block_id_tensor.data_ptr());
  void **gpu_tensor_ptrs =
      static_cast<void**>(gpu_tensor_ptrs_tensor.data_ptr());
  int64_t *cpu_block_ids = static_cast<int64_t*>(cpu_block_id_tensor.data_ptr());
  void *cpu_ptr = static_cast<void*>(cpu_tensor.data_ptr());
  uint32_t *cpu_size_table = reinterpret_cast<uint32_t*>(cpu_size_table_ptr);
  // TODO(nvcomp-guard): external size table is required for true compressed lengths.
  if (cpu_size_table == nullptr) {
    throw std::runtime_error(
        "transfer_kv_blocks_ans_binding: cpu_size_table_ptr must be non-null (the "
        "external size table is the sole source of nvcomp compressed size)");
  }
  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  flexkv::BackendType bt;
  if (gpu_block_type == 0) {
    bt = flexkv::BackendType::VLLM;
  } else if (gpu_block_type == 1) {
    bt = flexkv::BackendType::TRTLLM;
  } else if (gpu_block_type == 2) {
    bt = flexkv::BackendType::SGLANG;
  } else {
    throw std::runtime_error("Unsupported gpu_block_type: " +
                             std::to_string(gpu_block_type));
  }

  flexkv::GTensorHandler handler(
      bt, reinterpret_cast<int64_t**>(gpu_tensor_ptrs), num_layers,
      gpu_kv_stride_in_bytes, gpu_block_stride_in_bytes,
      gpu_layer_stride_in_bytes);

  size_t compressed_bytes = 0;

#define ANS_DISPATCH(func, Type)                                             \
  compressed_bytes = flexkv::func<Type>(                                      \
      &ctx, num_blocks, start_layer_id, num_layers, gpu_block_ids, handler,   \
      cpu_block_ids, cpu_ptr, cpu_kv_stride_in_bytes,                         \
      cpu_layer_stride_in_bytes, cpu_block_stride_in_bytes,                   \
      chunk_size_in_bytes, is_mla, cpu_size_table,                            \
      cpu_size_table_block_stride, cpu_size_table_layer_stride, stream)
#define ANS_SWITCH(func)                                                     \
  switch (bt) {                                                              \
    case flexkv::BackendType::VLLM:                                          \
      ANS_DISPATCH(func, flexkv::BackendType::VLLM);                         \
      break;                                                                 \
    case flexkv::BackendType::TRTLLM:                                        \
      ANS_DISPATCH(func, flexkv::BackendType::TRTLLM);                       \
      break;                                                                 \
    case flexkv::BackendType::SGLANG:                                        \
      ANS_DISPATCH(func, flexkv::BackendType::SGLANG);                       \
      break;                                                                 \
  }
  if (is_d2h) {
    ANS_SWITCH(transfer_kv_blocks_ans_comp);
  } else {
    ANS_SWITCH(transfer_kv_blocks_ans_decomp);
  }
#undef ANS_SWITCH
#undef ANS_DISPATCH

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
  return compressed_bytes;
}

static size_t transfer_kv_blocks_ans_comp_binding(
    flexkv::ANSTransferContext& ctx,
    torch::Tensor &gpu_block_id_tensor,
    torch::Tensor &gpu_tensor_ptrs_tensor,
    int64_t gpu_kv_stride_in_bytes,
    int64_t gpu_block_stride_in_bytes,
    int64_t gpu_layer_stride_in_bytes,
    torch::Tensor &cpu_block_id_tensor,
    torch::Tensor &cpu_tensor,
    int64_t cpu_kv_stride_in_bytes,
    int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_block_stride_in_bytes,
    int64_t chunk_size_in_bytes,
    int start_layer_id,
    int num_layers,
    bool is_mla,
    int gpu_block_type,
    int64_t cpu_size_table_ptr,
    int64_t cpu_size_table_block_stride,
    int64_t cpu_size_table_layer_stride) {
  return transfer_kv_blocks_ans_binding(
      true, ctx, gpu_block_id_tensor, gpu_tensor_ptrs_tensor,
      gpu_kv_stride_in_bytes, gpu_block_stride_in_bytes,
      gpu_layer_stride_in_bytes, cpu_block_id_tensor, cpu_tensor,
      cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes,
      cpu_block_stride_in_bytes, chunk_size_in_bytes, start_layer_id,
      num_layers, is_mla, gpu_block_type, cpu_size_table_ptr,
      cpu_size_table_block_stride, cpu_size_table_layer_stride);
}

static size_t transfer_kv_blocks_ans_decomp_binding(
    flexkv::ANSTransferContext& ctx,
    torch::Tensor &gpu_block_id_tensor,
    torch::Tensor &gpu_tensor_ptrs_tensor,
    int64_t gpu_kv_stride_in_bytes,
    int64_t gpu_block_stride_in_bytes,
    int64_t gpu_layer_stride_in_bytes,
    torch::Tensor &cpu_block_id_tensor,
    torch::Tensor &cpu_tensor,
    int64_t cpu_kv_stride_in_bytes,
    int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_block_stride_in_bytes,
    int64_t chunk_size_in_bytes,
    int start_layer_id,
    int num_layers,
    bool is_mla,
    int gpu_block_type,
    int64_t cpu_size_table_ptr,
    int64_t cpu_size_table_block_stride,
    int64_t cpu_size_table_layer_stride) {
  return transfer_kv_blocks_ans_binding(
      false, ctx, gpu_block_id_tensor, gpu_tensor_ptrs_tensor,
      gpu_kv_stride_in_bytes, gpu_block_stride_in_bytes,
      gpu_layer_stride_in_bytes, cpu_block_id_tensor, cpu_tensor,
      cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes,
      cpu_block_stride_in_bytes, chunk_size_in_bytes, start_layer_id,
      num_layers, is_mla, gpu_block_type, cpu_size_table_ptr,
      cpu_size_table_block_stride, cpu_size_table_layer_stride);
}

void register_ans_bindings(py::module_& m) {
  py::class_<flexkv::ANSTransferContext>(m, "ANSTransferContext")
      .def(py::init([](size_t max_num_chunks, size_t max_chunk_size,
                       int data_type, int transfer_sms) {
             auto ctx = std::make_unique<flexkv::ANSTransferContext>();
             flexkv::ans_ctx_create(ctx.get(), max_num_chunks, max_chunk_size,
                                    data_type, transfer_sms);
             return ctx;
           }),
           py::arg("max_num_chunks"), py::arg("max_chunk_size"),
           py::arg("data_type") = 0, py::arg("transfer_sms") = -1)
      .def("destroy", [](flexkv::ANSTransferContext& ctx) {
        flexkv::ans_ctx_destroy(&ctx);
      })
      .def_readonly("max_num_chunks",
                    &flexkv::ANSTransferContext::max_num_chunks)
      .def_readonly("max_chunk_size",
                    &flexkv::ANSTransferContext::max_chunk_size)
      .def_readonly("max_comp_chunk_bytes",
                    &flexkv::ANSTransferContext::max_comp_chunk_bytes);

  m.def("transfer_kv_blocks_ans_comp", &transfer_kv_blocks_ans_comp_binding,
        "ANS compress on GPU then D2H transfer", py::arg("ctx"),
        py::arg("gpu_block_id_tensor"), py::arg("gpu_tensor_ptrs_tensor"),
        py::arg("gpu_kv_stride_in_bytes"),
        py::arg("gpu_block_stride_in_bytes"),
        py::arg("gpu_layer_stride_in_bytes"), py::arg("cpu_block_id_tensor"),
        py::arg("cpu_tensor"), py::arg("cpu_kv_stride_in_bytes"),
        py::arg("cpu_layer_stride_in_bytes"),
        py::arg("cpu_block_stride_in_bytes"), py::arg("chunk_size_in_bytes"),
        py::arg("start_layer_id"), py::arg("num_layers"), py::arg("is_mla"),
        py::arg("gpu_block_type"), py::arg("cpu_size_table_ptr"),
        py::arg("cpu_size_table_block_stride"),
        py::arg("cpu_size_table_layer_stride"));
  m.def("transfer_kv_blocks_ans_decomp", &transfer_kv_blocks_ans_decomp_binding,
        "H2D transfer then ANS decompress on GPU", py::arg("ctx"),
        py::arg("gpu_block_id_tensor"), py::arg("gpu_tensor_ptrs_tensor"),
        py::arg("gpu_kv_stride_in_bytes"),
        py::arg("gpu_block_stride_in_bytes"),
        py::arg("gpu_layer_stride_in_bytes"), py::arg("cpu_block_id_tensor"),
        py::arg("cpu_tensor"), py::arg("cpu_kv_stride_in_bytes"),
        py::arg("cpu_layer_stride_in_bytes"),
        py::arg("cpu_block_stride_in_bytes"), py::arg("chunk_size_in_bytes"),
        py::arg("start_layer_id"), py::arg("num_layers"), py::arg("is_mla"),
        py::arg("gpu_block_type"), py::arg("cpu_size_table_ptr"),
        py::arg("cpu_size_table_block_stride"),
        py::arg("cpu_size_table_layer_stride"));
}

} // namespace flexkv
