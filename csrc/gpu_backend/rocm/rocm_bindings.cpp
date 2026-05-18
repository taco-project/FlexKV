// SPDX-License-Identifier: Apache-2.0
//
// AMD ROCm / HIP pybind11 bindings for ``flexkv.c_ext``.
//
// Mirrors ``nvidia_bindings.cpp`` but:
//   * Uses HIP runtime / ATen-HIP headers.
//   * Includes the hipified kernels from ``csrc/gpu_backend/rocm/.hipified/``
//     (produced at build time by ``build_backends/rocm_builder.py``).
//   * Has NO GDS path — cuFile has no ROCm equivalent today, so the SSD
//     direct path is left to the GPU-agnostic ``transfer_kv_blocks_ssd``
//     binding registered by ``csrc/bindings.cpp``.
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/cuda/CUDAContext.h>  // ROCm PyTorch keeps the "cuda" namespace.
#include <hip/hip_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "../backend_bindings.h"
// transfer.cuh is the hipified copy under csrc/gpu_backend/rocm/.hipified/,
// which rocm_builder adds to the include path.
#include "transfer.cuh"
// tp_transfer_thread_group.h lives under the NVIDIA tree but its content is
// already cross-vendor (gpu_* macros from gpu_backend/gpu_types.h). We
// reference it through ``csrc/gpu_backend`` which is on the include path.
#include "nvidia/tp_transfer_thread_group.h"

namespace py = pybind11;

namespace flexkv {
namespace gpu_backend {
namespace rocm_detail {

static void transfer_kv_blocks_binding(
    torch::Tensor& gpu_block_id_tensor, torch::Tensor& gpu_tensor_ptrs_tensor,
    int64_t gpu_kv_stride_in_bytes, int64_t gpu_block_stride_in_bytes,
    int64_t gpu_layer_stride_in_bytes, torch::Tensor& cpu_block_id_tensor,
    torch::Tensor& cpu_tensor, int64_t cpu_kv_stride_in_bytes,
    int64_t cpu_layer_stride_in_bytes, int64_t cpu_block_stride_in_bytes,
    int64_t chunk_size_in_bytes, int start_layer_id, int num_layers,
    int transfer_num_cta = 4, bool is_host_to_device = true,
    bool use_ce_transfer = false, bool is_mla = false, int gpu_block_type = 0) {
  int num_blocks = gpu_block_id_tensor.numel();

  int64_t* gpu_block_ids =
      static_cast<int64_t*>(gpu_block_id_tensor.data_ptr());
  void** gpu_tensor_ptrs =
      static_cast<void**>(gpu_tensor_ptrs_tensor.data_ptr());
  int64_t* cpu_block_ids =
      static_cast<int64_t*>(cpu_block_id_tensor.data_ptr());
  void* cpu_ptr = static_cast<void*>(cpu_tensor.data_ptr());

  // ROCm PyTorch reuses ``at::cuda::getCurrentCUDAStream`` — the underlying
  // type is hipStream_t (ABI-compatible with cudaStream_t after hipify).
  hipStream_t stream = at::cuda::getCurrentCUDAStream();

  flexkv::BackendType backend_type;
  if (gpu_block_type == 0) {
    backend_type = flexkv::BackendType::VLLM;
  } else if (gpu_block_type == 1) {
    backend_type = flexkv::BackendType::TRTLLM;
  } else if (gpu_block_type == 2) {
    backend_type = flexkv::BackendType::SGLANG;
  } else {
    throw std::runtime_error("Unsupported gpu_block_type: " +
                             std::to_string(gpu_block_type));
  }

  flexkv::GTensorHandler handler(
      backend_type, reinterpret_cast<int64_t**>(gpu_tensor_ptrs), num_layers,
      gpu_kv_stride_in_bytes, gpu_block_stride_in_bytes,
      gpu_layer_stride_in_bytes);

  switch (backend_type) {
    case flexkv::BackendType::VLLM:
      flexkv::transfer_kv_blocks<flexkv::BackendType::VLLM>(
          num_blocks, start_layer_id, num_layers, gpu_block_ids, handler, 0,
          cpu_block_ids, cpu_ptr, cpu_kv_stride_in_bytes,
          cpu_layer_stride_in_bytes, cpu_block_stride_in_bytes, 0,
          chunk_size_in_bytes, stream, transfer_num_cta, is_host_to_device,
          use_ce_transfer, is_mla);
      break;
    case flexkv::BackendType::TRTLLM:
      flexkv::transfer_kv_blocks<flexkv::BackendType::TRTLLM>(
          num_blocks, start_layer_id, num_layers, gpu_block_ids, handler, 0,
          cpu_block_ids, cpu_ptr, cpu_kv_stride_in_bytes,
          cpu_layer_stride_in_bytes, cpu_block_stride_in_bytes, 0,
          chunk_size_in_bytes, stream, transfer_num_cta, is_host_to_device,
          use_ce_transfer, is_mla);
      break;
    case flexkv::BackendType::SGLANG:
      flexkv::transfer_kv_blocks<flexkv::BackendType::SGLANG>(
          num_blocks, start_layer_id, num_layers, gpu_block_ids, handler, 0,
          cpu_block_ids, cpu_ptr, cpu_kv_stride_in_bytes,
          cpu_layer_stride_in_bytes, cpu_block_stride_in_bytes, 0,
          chunk_size_in_bytes, stream, transfer_num_cta, is_host_to_device,
          use_ce_transfer, is_mla);
      break;
  }

  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    throw std::runtime_error(hipGetErrorString(err));
  }
}

}  // namespace rocm_detail

void register_rocm_bindings(py::module_& m) {
  using namespace rocm_detail;

  m.def("transfer_kv_blocks", &transfer_kv_blocks_binding,
        "Transfer multi-layer KV-cache between CPU and GPU",
        py::arg("gpu_block_id_tensor"), py::arg("gpu_tensor_ptrs_tensor"),
        py::arg("gpu_kv_stride_in_bytes"), py::arg("gpu_block_stride_in_bytes"),
        py::arg("gpu_layer_stride_in_bytes"), py::arg("cpu_block_id_tensor"),
        py::arg("cpu_tensor"), py::arg("cpu_kv_stride_in_bytes"),
        py::arg("cpu_layer_stride_in_bytes"),
        py::arg("cpu_block_stride_in_bytes"), py::arg("chunk_size_in_bytes"),
        py::arg("start_layer_id"), py::arg("num_layers"),
        py::arg("transfer_num_cta") = 4, py::arg("is_host_to_device") = true,
        py::arg("use_ce_transfer") = false, py::arg("is_mla") = false,
        py::arg("gpu_block_type") = 0);

  py::class_<flexkv::TPTransferThreadGroup>(m, "TPTransferThreadGroup")
      .def(py::init<int, const std::vector<int64_t>&, int, int64_t, int, int,
                    const std::vector<int64_t>&, const std::vector<int64_t>&,
                    const std::vector<int64_t>&, const std::vector<int64_t>&,
                    const std::vector<int64_t>&>(),
           py::arg("num_gpus"), py::arg("gpu_block_ptrs_flat"),
           py::arg("num_tensors_per_gpu"), py::arg("cpu_blocks_ptr"),
           py::arg("dp_group_id"), py::arg("num_layers"),
           py::arg("gpu_kv_strides_in_bytes"),
           py::arg("gpu_block_strides_in_bytes"),
           py::arg("gpu_layer_strides_in_bytes"),
           py::arg("gpu_chunk_sizes_in_bytes"), py::arg("gpu_device_ids"))
      .def("tp_group_transfer",
           &flexkv::TPTransferThreadGroup::tp_group_transfer,
           py::arg("gpu_block_id_tensor"), py::arg("cpu_block_id_tensor"),
           py::arg("cpu_kv_stride_in_bytes"),
           py::arg("cpu_layer_stride_in_bytes"),
           py::arg("cpu_block_stride_in_bytes"),
           py::arg("cpu_tp_stride_in_bytes"), py::arg("transfer_num_cta"),
           py::arg("is_host_to_device"), py::arg("use_ce_transfer"),
           py::arg("layer_id"), py::arg("layer_granularity"),
           py::arg("is_mla"));

  // No GDS / TPGDSTransferThreadGroup binding on ROCm — cuFile has no
  // counterpart; the GPU-agnostic ``transfer_kv_blocks_ssd`` registered by
  // csrc/bindings.cpp covers the SSD path via POSIX/io_uring.
}

}  // namespace gpu_backend
}  // namespace flexkv
