// SPDX-License-Identifier: Apache-2.0
//
// NVIDIA / CUDA pybind11 bindings for ``flexkv.c_ext``.
//
// Everything that depends on cuda_runtime.h, ATen/cuda, NVTX or the cuFile
// (GDS) runtime lives here. ``csrc/bindings.cpp`` only calls
// ``register_nvidia_bindings(m)`` via the abstraction declared in
// ``csrc/gpu_backend/backend_bindings.h``; it does not include any of the
// headers below.
//
// Layout mirrors the legacy bindings.cpp 1:1 so behavior is unchanged.
#include <cstddef>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sys/stat.h>
#include <torch/extension.h>
#include <unistd.h>

#include "../backend_bindings.h"
#include "transfer.cuh"
#include "tp_transfer_thread_group.h"

#ifdef FLEXKV_ENABLE_GDS
#include "gds/gds_manager.h"
#include "gds/tp_gds_transfer_thread_group.h"
#endif

namespace py = pybind11;

namespace flexkv {
namespace gpu_backend {
namespace nvidia_detail {

// ---------------------------------------------------------------------------
// transfer_kv_blocks: GPU<->CPU layered KV-cache copy via CE / SM kernel.
// ---------------------------------------------------------------------------
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
  void** gpu_tensor_ptrs = static_cast<void**>(
      gpu_tensor_ptrs_tensor.data_ptr());  // must be contiguous
  int64_t* cpu_block_ids =
      static_cast<int64_t*>(cpu_block_id_tensor.data_ptr());
  void* cpu_ptr = static_cast<void*>(cpu_tensor.data_ptr());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

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

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

#ifdef FLEXKV_ENABLE_GDS
// ---------------------------------------------------------------------------
// GDS: cuFile-backed direct path between GPU memory and SSD.
// ---------------------------------------------------------------------------
static void transfer_kv_blocks_gds_binding(
    GDSManager& gds_manager, const torch::Tensor& gpu_layer_id_list,
    const torch::Tensor& gpu_layer_ptrs_tensor,
    const torch::Tensor& ssd_block_ids, const torch::Tensor& gpu_block_ids,
    int64_t gpu_kv_stride_in_bytes, int64_t gpu_block_stride_in_bytes,
    int64_t gpu_layer_stride_in_bytes, int64_t ssd_layer_stride_in_bytes,
    int64_t ssd_block_stride_in_bytes, int64_t ssd_kv_stride_in_bytes,
    int64_t block_size_in_bytes, int64_t ssd_copy_off_inside_chunks,
    int num_blocks_per_file, int64_t total_layers, bool is_read,
    bool verbose = false, bool is_mla = false, int gpu_block_type = 0,
    int gpu_device_id = 0) {
  TORCH_CHECK(gpu_layer_ptrs_tensor.dtype() == torch::kInt64,
              "gpu_layer_ptrs must be int64");
  TORCH_CHECK(ssd_block_ids.dtype() == torch::kInt64,
              "ssd_block_ids must be int64");
  TORCH_CHECK(gpu_block_ids.dtype() == torch::kInt64,
              "gpu_block_ids must be int64");
  TORCH_CHECK(gpu_layer_id_list.dtype() == torch::kInt32,
              "gpu_layer_id_list must be int32");

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

  void** gpu_tensor_ptrs =
      static_cast<void**>(gpu_layer_ptrs_tensor.data_ptr());
  flexkv::GTensorHandler handler(
      backend_type, reinterpret_cast<int64_t**>(gpu_tensor_ptrs), total_layers,
      gpu_kv_stride_in_bytes, gpu_block_stride_in_bytes,
      gpu_layer_stride_in_bytes);

  switch (backend_type) {
    case flexkv::BackendType::VLLM:
      flexkv::transfer_kv_blocks_gds<flexkv::BackendType::VLLM>(
          gds_manager, gpu_layer_id_list, handler, ssd_block_ids,
          gpu_block_ids, ssd_layer_stride_in_bytes, ssd_block_stride_in_bytes,
          ssd_kv_stride_in_bytes, block_size_in_bytes,
          ssd_copy_off_inside_chunks, ssd_block_stride_in_bytes, gpu_device_id,
          num_blocks_per_file, total_layers, is_read, verbose, is_mla);
      break;
    case flexkv::BackendType::TRTLLM:
      flexkv::transfer_kv_blocks_gds<flexkv::BackendType::TRTLLM>(
          gds_manager, gpu_layer_id_list, handler, ssd_block_ids,
          gpu_block_ids, ssd_layer_stride_in_bytes, ssd_block_stride_in_bytes,
          ssd_kv_stride_in_bytes, block_size_in_bytes,
          ssd_copy_off_inside_chunks, ssd_block_stride_in_bytes, gpu_device_id,
          num_blocks_per_file, total_layers, is_read, verbose, is_mla);
      break;
    case flexkv::BackendType::SGLANG:
      flexkv::transfer_kv_blocks_gds<flexkv::BackendType::SGLANG>(
          gds_manager, gpu_layer_id_list, handler, ssd_block_ids,
          gpu_block_ids, ssd_layer_stride_in_bytes, ssd_block_stride_in_bytes,
          ssd_kv_stride_in_bytes, block_size_in_bytes,
          ssd_copy_off_inside_chunks, ssd_block_stride_in_bytes, gpu_device_id,
          num_blocks_per_file, total_layers, is_read, verbose, is_mla);
      break;
  }
}

static py::list gds_batch_write_binding(GDSManager& manager,
                                        py::list operations_list) {
  size_t batch_size = operations_list.size();
  std::vector<BatchWriteOp> operations(batch_size);
  std::vector<ssize_t> results(batch_size);

  for (size_t i = 0; i < batch_size; ++i) {
    py::dict op_dict = operations_list[i].cast<py::dict>();
    operations[i].filename = op_dict["filename"].cast<std::string>().c_str();
    operations[i].gpu_data =
        op_dict["gpu_data"].cast<torch::Tensor>().data_ptr();
    operations[i].size = op_dict["size"].cast<size_t>();
    operations[i].file_offset = op_dict["file_offset"].cast<size_t>();
    operations[i].result = &results[i];
  }

  int batch_id = manager.batch_write(operations.data(), batch_size);

  py::list result_list;
  result_list.append(batch_id);
  for (size_t i = 0; i < batch_size; ++i) {
    result_list.append(results[i]);
  }
  return result_list;
}

static py::list gds_batch_read_binding(GDSManager& manager,
                                       py::list operations_list) {
  size_t batch_size = operations_list.size();
  std::vector<BatchReadOp> operations(batch_size);
  std::vector<ssize_t> results(batch_size);

  for (size_t i = 0; i < batch_size; ++i) {
    py::dict op_dict = operations_list[i].cast<py::dict>();
    operations[i].filename = op_dict["filename"].cast<std::string>().c_str();
    operations[i].gpu_buffer =
        op_dict["gpu_buffer"].cast<torch::Tensor>().data_ptr();
    operations[i].size = op_dict["size"].cast<size_t>();
    operations[i].file_offset = op_dict["file_offset"].cast<size_t>();
    operations[i].result = &results[i];
  }

  int batch_id = manager.batch_read(operations.data(), batch_size);

  py::list result_list;
  result_list.append(batch_id);
  for (size_t i = 0; i < batch_size; ++i) {
    result_list.append(results[i]);
  }
  return result_list;
}

static ssize_t gds_write_binding(GDSManager& manager,
                                 const std::string& filename,
                                 torch::Tensor gpu_data,
                                 size_t file_offset = 0) {
  return manager.write(filename.c_str(), gpu_data.data_ptr(),
                       gpu_data.numel() * gpu_data.element_size(),
                       file_offset);
}

static ssize_t gds_read_binding(GDSManager& manager,
                                const std::string& filename,
                                torch::Tensor gpu_buffer,
                                size_t file_offset = 0) {
  return manager.read(filename.c_str(), gpu_buffer.data_ptr(),
                      gpu_buffer.numel() * gpu_buffer.element_size(),
                      file_offset);
}

static ssize_t gds_write_async_binding(GDSManager& manager,
                                       const std::string& filename,
                                       torch::Tensor gpu_data,
                                       size_t file_offset = 0) {
  return manager.write_async(filename.c_str(), gpu_data.data_ptr(),
                             gpu_data.numel() * gpu_data.element_size(),
                             file_offset);
}

static ssize_t gds_read_async_binding(GDSManager& manager,
                                      const std::string& filename,
                                      torch::Tensor gpu_buffer,
                                      size_t file_offset = 0) {
  return manager.read_async(filename.c_str(), gpu_buffer.data_ptr(),
                            gpu_buffer.numel() * gpu_buffer.element_size(),
                            file_offset);
}

static bool create_gds_file_binding(GDSManager& manager,
                                    const std::string& filename,
                                    size_t file_size) {
  int fd = open(filename.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
  if (fd < 0) {
    return false;
  }
  if (ftruncate(fd, file_size) != 0) {
    close(fd);
    return false;
  }
  fsync(fd);
  close(fd);
  return manager.add_file(filename.c_str());
}
#endif  // FLEXKV_ENABLE_GDS

}  // namespace nvidia_detail

// ---------------------------------------------------------------------------
// register_nvidia_bindings: invoked from csrc/bindings.cpp.
// ---------------------------------------------------------------------------
void register_nvidia_bindings(py::module_& m) {
  using namespace nvidia_detail;

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

#ifdef FLEXKV_ENABLE_GDS
  m.def("transfer_kv_blocks_gds", &transfer_kv_blocks_gds_binding,
        "Transfer KV blocks between GPU and GDS storage",
        py::arg("gds_manager"), py::arg("gpu_layer_id_list"),
        py::arg("gpu_layer_ptrs_tensor"), py::arg("ssd_block_ids"),
        py::arg("gpu_block_ids"), py::arg("gpu_kv_stride_in_bytes"),
        py::arg("gpu_block_stride_in_bytes"),
        py::arg("gpu_layer_stride_in_bytes"),
        py::arg("ssd_layer_stride_in_bytes"),
        py::arg("ssd_block_stride_in_bytes"), py::arg("ssd_kv_stride_in_bytes"),
        py::arg("block_size_in_bytes"), py::arg("ssd_copy_off_inside_chunks"),
        py::arg("num_blocks_per_file"), py::arg("total_layers"),
        py::arg("is_read"), py::arg("verbose") = false,
        py::arg("is_mla") = false, py::arg("gpu_block_type") = 0,
        py::arg("gpu_device_id") = 0);

  py::class_<flexkv::TPGDSTransferThreadGroup>(m, "TPGDSTransferThreadGroup")
      .def(py::init<int, const std::vector<int64_t>&, int,
                    std::map<int, std::vector<std::string>>&, int, int,
                    const std::vector<int64_t>&, const std::vector<int64_t>&,
                    const std::vector<int64_t>&, const std::vector<int64_t>&,
                    const std::vector<int64_t>&>(),
           py::arg("num_gpus"), py::arg("gpu_block_ptrs_flat"),
           py::arg("num_tensors_per_gpu"), py::arg("ssd_files"),
           py::arg("dp_group_id"), py::arg("num_layers"),
           py::arg("gpu_kv_strides_in_bytes"),
           py::arg("gpu_block_strides_in_bytes"),
           py::arg("gpu_layer_strides_in_bytes"),
           py::arg("gpu_chunk_sizes_in_bytes"), py::arg("gpu_device_ids"))
      .def("tp_group_transfer",
           &flexkv::TPGDSTransferThreadGroup::tp_group_transfer,
           py::arg("gpu_block_id_tensor"), py::arg("ssd_block_id_tensor"),
           py::arg("ssd_layer_stride_in_bytes"),
           py::arg("ssd_kv_stride_in_bytes"),
           py::arg("ssd_block_stride_in_bytes"),
           py::arg("ssd_tp_stride_in_bytes"), py::arg("num_blocks_per_file"),
           py::arg("is_read"), py::arg("layer_id"),
           py::arg("layer_granularity"), py::arg("is_mla"));

  py::class_<GDSManager>(m, "GDSManager")
      .def(py::init<std::map<int, std::vector<std::string>>&, int, int>(),
           "Initialize GDS Manager with device-organized files",
           py::arg("ssd_files"), py::arg("num_devices"),
           py::arg("round_robin") = 1)
      .def("is_ready", &GDSManager::is_ready,
           "Check if GDS manager is ready for operations")
      .def("get_last_error", &GDSManager::get_last_error,
           "Get the last error message")
      .def("add_file", &GDSManager::add_file,
           "Add and register a file with GDS (creates with O_DIRECT)",
           py::arg("filename"))
      .def("remove_file", &GDSManager::remove_file,
           "Remove and unregister a file from GDS", py::arg("filename"))
      .def("write", &gds_write_binding, "Write data from GPU memory to file",
           py::arg("filename"), py::arg("gpu_data"),
           py::arg("file_offset") = 0)
      .def("read", &gds_read_binding, "Read data from file to GPU memory",
           py::arg("filename"), py::arg("gpu_buffer"),
           py::arg("file_offset") = 0)
      .def("write_async", &gds_write_async_binding,
           "Write data from GPU memory to file asynchronously",
           py::arg("filename"), py::arg("gpu_data"),
           py::arg("file_offset") = 0)
      .def("read_async", &gds_read_async_binding,
           "Read data from file to GPU memory asynchronously",
           py::arg("filename"), py::arg("gpu_buffer"),
           py::arg("file_offset") = 0)
      .def("batch_write", &gds_batch_write_binding, "Batch write operations",
           py::arg("operations"))
      .def("batch_read", &gds_batch_read_binding, "Batch read operations",
           py::arg("operations"))
      .def("batch_synchronize", &GDSManager::batch_synchronize,
           "Wait for batch operations to complete", py::arg("batch_id"))
      .def("synchronize", &GDSManager::synchronize,
           "Synchronize all internal CUDA streams")
      .def("get_file_count", &GDSManager::get_file_count,
           "Get number of files currently managed")
      .def("get_num_devices", &GDSManager::get_num_devices,
           "Get number of devices")
      .def("get_num_files_per_device", &GDSManager::get_num_files_per_device,
           "Get number of files per device")
      .def("get_round_robin", &GDSManager::get_round_robin,
           "Get round-robin granularity")
      .def("get_file_paths", &GDSManager::get_file_paths,
           "Get file paths for a specific device", py::arg("device_id"))
      .def("create_gds_file", &create_gds_file_binding,
           "Create and register a GDS file with specified size",
           py::arg("filename"), py::arg("file_size"));
#endif  // FLEXKV_ENABLE_GDS
}

}  // namespace gpu_backend
}  // namespace flexkv
