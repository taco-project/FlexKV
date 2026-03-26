/*
 * csrc/gpu_backend/musa/gpu_transfer_bindings.h
 *
 * MUSA backend — GPU-specific binding functions and pybind11 registrations.
 *
 * Included by csrc/bindings.cpp when FLEXKV_BACKEND_MUSA is defined.
 * Exposes the same surface as the NVIDIA header so the rest of bindings.cpp
 * can use the same REGISTER_GPU_TRANSFER_BINDINGS(m) call site.
 *
 * When FLEXKV_HAS_MUSA_SDK is defined (full SDK present):
 *   - Calls real MUSA transfer kernel (transfer_musa.muh)
 *   - Exposes TPTransferThreadGroupMusa
 *   - Exposes GDSManagerMusa + TPGDSTransferThreadGroupMusa (FLEXKV_ENABLE_GDS)
 * Without the SDK:
 *   - Provides a no-op stub for transfer_kv_blocks (for build/dispatch testing)
 *   - m.attr("HAS_MUSA_SDK") = false
 */
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <map>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#ifdef FLEXKV_HAS_MUSA_SDK
#include <musa_runtime.h>
#include "gpu_backend/musa/transfer_musa.muh"
#include "gpu_backend/musa/tp_transfer_thread_group_musa.h"
#  ifdef FLEXKV_ENABLE_GDS
#    include "gpu_backend/musa/gds/gds_manager_musa.h"
#    include "gpu_backend/musa/gds/tp_gds_transfer_thread_group_musa.h"
#  endif
#  if __has_include(<torch_musa/csrc/core/MUSAStream.h>)
#    include <torch_musa/csrc/core/MUSAStream.h>
static musaStream_t _get_current_musa_stream() {
    return c10::musa::getCurrentMUSAStream(-1).stream();
}
#  else
static musaStream_t _get_current_musa_stream() { return (musaStream_t)0; }
#  endif
#else
#  include "gpu_backend/musa/gtensor_handler_musa.h"
#endif

namespace py = pybind11;

// ---------------------------------------------------------------------------
// GPU–CPU transfer (MUSA kernel or stub)
// ---------------------------------------------------------------------------

static void transfer_kv_blocks_binding(
    torch::Tensor &gpu_block_id_tensor, torch::Tensor &gpu_tensor_ptrs_tensor,
    int64_t gpu_kv_stride_in_bytes, int64_t gpu_block_stride_in_bytes,
    int64_t gpu_layer_stride_in_bytes, torch::Tensor &cpu_block_id_tensor,
    torch::Tensor &cpu_tensor, int64_t cpu_kv_stride_in_bytes,
    int64_t cpu_layer_stride_in_bytes, int64_t cpu_block_stride_in_bytes,
    int64_t chunk_size_in_bytes, int start_layer_id, int num_layers,
    int transfer_num_cta = 4, bool is_host_to_device = true,
    bool use_ce_transfer = false, bool is_mla = false, int gpu_block_type = 0) {

#ifdef FLEXKV_HAS_MUSA_SDK
  int num_blocks = gpu_block_id_tensor.numel();
  int64_t *gpu_block_ids   = static_cast<int64_t *>(gpu_block_id_tensor.data_ptr());
  void   **gpu_tensor_ptrs  = static_cast<void **>(gpu_tensor_ptrs_tensor.data_ptr());
  int64_t *cpu_block_ids   = static_cast<int64_t *>(cpu_block_id_tensor.data_ptr());
  void    *cpu_ptr          = static_cast<void *>(cpu_tensor.data_ptr());

  musaStream_t stream = _get_current_musa_stream();

  flexkv::BackendType backend_type;
  if      (gpu_block_type == 0) backend_type = flexkv::BackendType::VLLM;
  else if (gpu_block_type == 1) backend_type = flexkv::BackendType::TRTLLM;
  else if (gpu_block_type == 2) backend_type = flexkv::BackendType::SGLANG;
  else throw std::runtime_error("Unsupported gpu_block_type: " + std::to_string(gpu_block_type));

  flexkv::GTensorHandler handler(
      backend_type, reinterpret_cast<int64_t **>(gpu_tensor_ptrs), num_layers,
      gpu_kv_stride_in_bytes, gpu_block_stride_in_bytes, gpu_layer_stride_in_bytes);

  switch (backend_type) {
    case flexkv::BackendType::VLLM:
      flexkv::transfer_kv_blocks<flexkv::BackendType::VLLM>(
          num_blocks, start_layer_id, num_layers, gpu_block_ids, handler, 0,
          cpu_block_ids, cpu_ptr, cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes,
          cpu_block_stride_in_bytes, 0, chunk_size_in_bytes, stream,
          transfer_num_cta, is_host_to_device, use_ce_transfer, is_mla);
      break;
    case flexkv::BackendType::TRTLLM:
      flexkv::transfer_kv_blocks<flexkv::BackendType::TRTLLM>(
          num_blocks, start_layer_id, num_layers, gpu_block_ids, handler, 0,
          cpu_block_ids, cpu_ptr, cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes,
          cpu_block_stride_in_bytes, 0, chunk_size_in_bytes, stream,
          transfer_num_cta, is_host_to_device, use_ce_transfer, is_mla);
      break;
    case flexkv::BackendType::SGLANG:
      flexkv::transfer_kv_blocks<flexkv::BackendType::SGLANG>(
          num_blocks, start_layer_id, num_layers, gpu_block_ids, handler, 0,
          cpu_block_ids, cpu_ptr, cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes,
          cpu_block_stride_in_bytes, 0, chunk_size_in_bytes, stream,
          transfer_num_cta, is_host_to_device, use_ce_transfer, is_mla);
      break;
  }

  musaError_t err = musaGetLastError();
  if (err != musaSuccess)
    throw std::runtime_error(std::string("MUSA transfer error: ") + musaGetErrorString(err));
#else
  // Stub: no-op when MUSA SDK is absent (useful for build/dispatch path testing)
  (void)gpu_block_id_tensor;  (void)gpu_tensor_ptrs_tensor;
  (void)gpu_kv_stride_in_bytes; (void)gpu_block_stride_in_bytes;
  (void)gpu_layer_stride_in_bytes; (void)cpu_block_id_tensor;
  (void)cpu_tensor; (void)cpu_kv_stride_in_bytes;
  (void)cpu_layer_stride_in_bytes; (void)cpu_block_stride_in_bytes;
  (void)chunk_size_in_bytes; (void)start_layer_id;
  (void)num_layers; (void)transfer_num_cta;
  (void)is_host_to_device; (void)use_ce_transfer;
  (void)is_mla; (void)gpu_block_type;
#endif
}

// ---------------------------------------------------------------------------
// GDS (muFile) — GPU↔SSD
// ---------------------------------------------------------------------------

#if defined(FLEXKV_HAS_MUSA_SDK) && defined(FLEXKV_ENABLE_GDS)

static void transfer_kv_blocks_gds_binding(
    GDSManagerMusa &gds_manager, const torch::Tensor &gpu_layer_id_list,
    const torch::Tensor &gpu_layer_ptrs_tensor,
    const torch::Tensor &ssd_block_ids, const torch::Tensor &gpu_block_ids,
    int64_t gpu_kv_stride_in_bytes, int64_t gpu_block_stride_in_bytes,
    int64_t gpu_layer_stride_in_bytes, int64_t ssd_layer_stride_in_bytes,
    int64_t ssd_block_stride_in_bytes, int64_t ssd_kv_stride_in_bytes,
    int64_t block_size_in_bytes, int64_t ssd_copy_off_inside_chunks,
    int num_blocks_per_file, int64_t total_layers, bool is_read,
    bool verbose = false, bool is_mla = false,
    int gpu_block_type = 0, int gpu_device_id = 0) {

  TORCH_CHECK(gpu_layer_ptrs_tensor.dtype() == torch::kInt64, "gpu_layer_ptrs must be int64");
  TORCH_CHECK(ssd_block_ids.dtype()        == torch::kInt64, "ssd_block_ids must be int64");
  TORCH_CHECK(gpu_block_ids.dtype()        == torch::kInt64, "gpu_block_ids must be int64");
  TORCH_CHECK(gpu_layer_id_list.dtype()    == torch::kInt32, "gpu_layer_id_list must be int32");

  flexkv::BackendType backend_type;
  if      (gpu_block_type == 0) backend_type = flexkv::BackendType::VLLM;
  else if (gpu_block_type == 1) backend_type = flexkv::BackendType::TRTLLM;
  else if (gpu_block_type == 2) backend_type = flexkv::BackendType::SGLANG;
  else throw std::runtime_error("Unsupported gpu_block_type: " + std::to_string(gpu_block_type));

  void **gpu_tensor_ptrs = static_cast<void **>(gpu_layer_ptrs_tensor.data_ptr());
  flexkv::GTensorHandler handler(
      backend_type, reinterpret_cast<int64_t **>(gpu_tensor_ptrs), total_layers,
      gpu_kv_stride_in_bytes, gpu_block_stride_in_bytes, gpu_layer_stride_in_bytes);

  switch (backend_type) {
    case flexkv::BackendType::VLLM:
      flexkv::transfer_kv_blocks_gds_musa<flexkv::BackendType::VLLM>(
          gds_manager, gpu_layer_id_list, handler, ssd_block_ids, gpu_block_ids,
          ssd_layer_stride_in_bytes, ssd_block_stride_in_bytes, ssd_kv_stride_in_bytes,
          block_size_in_bytes, ssd_copy_off_inside_chunks, ssd_block_stride_in_bytes,
          gpu_device_id, num_blocks_per_file, total_layers, is_read, verbose, is_mla);
      break;
    case flexkv::BackendType::TRTLLM:
      flexkv::transfer_kv_blocks_gds_musa<flexkv::BackendType::TRTLLM>(
          gds_manager, gpu_layer_id_list, handler, ssd_block_ids, gpu_block_ids,
          ssd_layer_stride_in_bytes, ssd_block_stride_in_bytes, ssd_kv_stride_in_bytes,
          block_size_in_bytes, ssd_copy_off_inside_chunks, ssd_block_stride_in_bytes,
          gpu_device_id, num_blocks_per_file, total_layers, is_read, verbose, is_mla);
      break;
    case flexkv::BackendType::SGLANG:
      flexkv::transfer_kv_blocks_gds_musa<flexkv::BackendType::SGLANG>(
          gds_manager, gpu_layer_id_list, handler, ssd_block_ids, gpu_block_ids,
          ssd_layer_stride_in_bytes, ssd_block_stride_in_bytes, ssd_kv_stride_in_bytes,
          block_size_in_bytes, ssd_copy_off_inside_chunks, ssd_block_stride_in_bytes,
          gpu_device_id, num_blocks_per_file, total_layers, is_read, verbose, is_mla);
      break;
  }
}

#endif // FLEXKV_HAS_MUSA_SDK && FLEXKV_ENABLE_GDS

// ---------------------------------------------------------------------------
// Macro: register everything into pybind11 module m
// ---------------------------------------------------------------------------

#ifdef FLEXKV_HAS_MUSA_SDK
#define DO_REGISTER_MUSA_SDK_BINDINGS(m)                                                        \
  (m).attr("HAS_MUSA_SDK") = true;                                                              \
  py::class_<flexkv::TPTransferThreadGroupMusa>((m), "TPTransferThreadGroupMusa")               \
      .def(py::init<int,                                                                          \
                    const std::vector<std::vector<torch::Tensor>> &,                              \
                    torch::Tensor &, int, int,                                                    \
                    torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &>(),       \
           py::arg("num_gpus"), py::arg("gpu_blocks"), py::arg("cpu_blocks"),                    \
           py::arg("dp_group_id"), py::arg("num_layers"),                                        \
           py::arg("gpu_kv_strides_tensor"), py::arg("gpu_block_strides_tensor"),                \
           py::arg("gpu_layer_strides_tensor"), py::arg("gpu_chunk_sizes_tensor"))               \
      .def("tp_group_transfer",                                                                   \
           &flexkv::TPTransferThreadGroupMusa::tp_group_transfer,                               \
           py::arg("gpu_block_id_tensor"), py::arg("cpu_block_id_tensor"),                      \
           py::arg("cpu_kv_stride_in_bytes"), py::arg("cpu_layer_stride_in_bytes"),             \
           py::arg("cpu_block_stride_in_bytes"), py::arg("cpu_tp_stride_in_bytes"),             \
           py::arg("transfer_sms"), py::arg("is_host_to_device"),                               \
           py::arg("use_ce_transfer"), py::arg("layer_id"),                                      \
           py::arg("layer_granularity"), py::arg("is_mla"));                                    \
  DO_REGISTER_MUSA_GDS_BINDINGS(m)
#else
#define DO_REGISTER_MUSA_SDK_BINDINGS(m)  (m).attr("HAS_MUSA_SDK") = false;
#endif

#if defined(FLEXKV_HAS_MUSA_SDK) && defined(FLEXKV_ENABLE_GDS)
#define DO_REGISTER_MUSA_GDS_BINDINGS(m)                                                        \
  (m).def("transfer_kv_blocks_gds", &transfer_kv_blocks_gds_binding,                           \
      "Transfer KV blocks between GPU and GDS storage (MUSA/muFile)",                          \
      py::arg("gds_manager"), py::arg("gpu_layer_id_list"),                                     \
      py::arg("gpu_layer_ptrs_tensor"), py::arg("ssd_block_ids"),                               \
      py::arg("gpu_block_ids"), py::arg("gpu_kv_stride_in_bytes"),                              \
      py::arg("gpu_block_stride_in_bytes"), py::arg("gpu_layer_stride_in_bytes"),               \
      py::arg("ssd_layer_stride_in_bytes"), py::arg("ssd_block_stride_in_bytes"),               \
      py::arg("ssd_kv_stride_in_bytes"), py::arg("block_size_in_bytes"),                        \
      py::arg("ssd_copy_off_inside_chunks"), py::arg("num_blocks_per_file"),                    \
      py::arg("total_layers"), py::arg("is_read"),                                              \
      py::arg("verbose") = false, py::arg("is_mla") = false,                                   \
      py::arg("gpu_block_type") = 0, py::arg("gpu_device_id") = 0);                            \
  py::class_<GDSManagerMusa>((m), "GDSManagerMusa")                                             \
      .def(py::init<std::map<int, std::vector<std::string>> &, int, int>(),                     \
           py::arg("ssd_files"), py::arg("num_devices"), py::arg("round_robin") = 1)            \
      .def("is_ready",        &GDSManagerMusa::is_ready)                                        \
      .def("synchronize",     &GDSManagerMusa::synchronize)                                     \
      .def("get_last_error",  &GDSManagerMusa::get_last_error);                                 \
  py::class_<flexkv::TPGDSTransferThreadGroupMusa>((m), "TPGDSTransferThreadGroupMusa")         \
      .def(py::init<int,                                                                          \
                    const std::vector<std::vector<torch::Tensor>> &,                              \
                    std::map<int, std::vector<std::string>> &, int, int,                         \
                    torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &>(),       \
           py::arg("num_gpus"), py::arg("gpu_blocks"), py::arg("ssd_files"),                    \
           py::arg("dp_group_id"), py::arg("num_layers"),                                        \
           py::arg("gpu_kv_strides_tensor"), py::arg("gpu_block_strides_tensor"),                \
           py::arg("gpu_layer_strides_tensor"), py::arg("gpu_chunk_sizes_tensor"))               \
      .def("tp_group_transfer",                                                                   \
           &flexkv::TPGDSTransferThreadGroupMusa::tp_group_transfer,                            \
           py::arg("gpu_block_id_tensor"), py::arg("ssd_block_id_tensor"),                      \
           py::arg("ssd_layer_stride_in_bytes"), py::arg("ssd_kv_stride_in_bytes"),             \
           py::arg("ssd_block_stride_in_bytes"), py::arg("ssd_tp_stride_in_bytes"),             \
           py::arg("num_blocks_per_file"), py::arg("is_read"),                                  \
           py::arg("layer_id"), py::arg("layer_granularity"), py::arg("is_mla"));
#else
#define DO_REGISTER_MUSA_GDS_BINDINGS(m)  // no-op
#endif

#define REGISTER_GPU_TRANSFER_BINDINGS(m)                                                       \
  (m).def("transfer_kv_blocks", &transfer_kv_blocks_binding,                                    \
      "Transfer multi-layer KV-cache between CPU and GPU (MUSA)",                               \
      py::arg("gpu_block_id_tensor"), py::arg("gpu_tensor_ptrs_tensor"),                        \
      py::arg("gpu_kv_stride_in_bytes"), py::arg("gpu_block_stride_in_bytes"),                  \
      py::arg("gpu_layer_stride_in_bytes"), py::arg("cpu_block_id_tensor"),                     \
      py::arg("cpu_tensor"), py::arg("cpu_kv_stride_in_bytes"),                                 \
      py::arg("cpu_layer_stride_in_bytes"), py::arg("cpu_block_stride_in_bytes"),               \
      py::arg("chunk_size_in_bytes"), py::arg("start_layer_id"), py::arg("num_layers"),         \
      py::arg("transfer_num_cta") = 4, py::arg("is_host_to_device") = true,                    \
      py::arg("use_ce_transfer") = false, py::arg("is_mla") = false,                           \
      py::arg("gpu_block_type") = 0);                                                           \
  DO_REGISTER_MUSA_SDK_BINDINGS(m)
