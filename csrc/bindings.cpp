#include <vector>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <nvToolsExt.h>
#include <pybind11/pybind11.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <torch/extension.h>
#include <unistd.h>

#include "cache_utils.h"
#include "ipc_memhandle.h"
#include "transfer.cuh"
#include "transfer_ssd.h"

namespace py = pybind11;

template <typename ptrType> ptrType get_tensor_ptr(torch::Tensor &tensor) {
  if (tensor.is_cuda()) {
    return static_cast<ptrType>(tensor.data_ptr());
  }
  if (tensor.is_pinned()) {
    ptrType host_ptr = static_cast<ptrType>(tensor.data_ptr());
    ptrType device_ptr;
    cudaHostGetDevicePointer((void **)&device_ptr,
                             static_cast<void *>(host_ptr), 0);
    return device_ptr;
  }
  throw std::invalid_argument("Tensor is not pinned or cuda");
}

void transfer_kv_layers_binding(
    torch::Tensor &dst_block_id_tensor, torch::Tensor &dst_layer_ptrs_tensor,
    int64_t dst_kv_stride_in_bytes, int64_t dst_chunk_stride_in_bytes,
    torch::Tensor &src_block_id_tensor, torch::Tensor &src_layer_ptrs_tensor,
    int64_t src_kv_stride_in_bytes, int64_t src_chunk_stride_in_bytes,
    int64_t chunk_size_in_bytes, int transfer_sms = -1,
    bool is_host_to_device = true, bool use_ce_transfer = false) {
  int num_blocks = dst_block_id_tensor.numel();
  int num_layers = dst_layer_ptrs_tensor.numel();

  int64_t *dst_block_ids = get_tensor_ptr<int64_t *>(dst_block_id_tensor);
  void **dst_layer_ptrs = get_tensor_ptr<void **>(dst_layer_ptrs_tensor);
  int64_t *src_block_ids = get_tensor_ptr<int64_t *>(src_block_id_tensor);
  void **src_layer_ptrs = get_tensor_ptr<void **>(src_layer_ptrs_tensor);

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  flexkv::transfer_kv_layers(
      num_blocks, num_layers, dst_block_ids, dst_layer_ptrs,
      dst_kv_stride_in_bytes, dst_chunk_stride_in_bytes, src_block_ids,
      src_layer_ptrs, src_kv_stride_in_bytes, src_chunk_stride_in_bytes,
      chunk_size_in_bytes, stream, transfer_sms, is_host_to_device,
      use_ce_transfer);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

void transfer_kv_blocks_ssd(
    const std::string &filename, const torch::Tensor &cpu_layer_id_list,
    const torch::Tensor &cpu_layer_ptrs_tensor,
    const torch::Tensor &ssd_block_ids, const torch::Tensor &cpu_block_ids,
    int64_t cpu_kv_stride_in_bytes, int64_t ssd_layer_stride_in_bytes,
    int64_t ssd_block_stride_in_bytes, int64_t ssd_kv_stride_in_bytes,
    int64_t block_size_in_bytes, int64_t total_layers, bool is_read, bool verbose = false) {
  TORCH_CHECK(cpu_layer_ptrs_tensor.dtype() == torch::kInt64,
              "cpu_layer_ptrs must be int64");
  TORCH_CHECK(ssd_block_ids.dtype() == torch::kInt64,
              "ssd_block_ids must be int64");
  TORCH_CHECK(cpu_block_ids.dtype() == torch::kInt64,
              "cpu_block_ids must be int64");

  transfer_kv_blocks_ssd_mmap_multi_thread(
      filename, cpu_layer_id_list, cpu_layer_ptrs_tensor, ssd_block_ids,
      cpu_block_ids, cpu_kv_stride_in_bytes, ssd_layer_stride_in_bytes,
      ssd_block_stride_in_bytes, ssd_kv_stride_in_bytes, block_size_in_bytes,
      total_layers, is_read, verbose);
}

PYBIND11_MODULE(c_ext, m) {
  m.def("transfer_kv_layers", &transfer_kv_layers_binding,
        "Transfer multi-layer KV-cache between CPU and GPU");
  m.def("transfer_kv_blocks_ssd", &transfer_kv_blocks_ssd,
        "Transfer KV blocks between SSD and CPU memory", py::arg("filename"),
        py::arg("cpu_layer_id_list"), py::arg("cpu_layer_ptrs_tensor"),
        py::arg("ssd_block_ids"), py::arg("cpu_block_ids"),
        py::arg("cpu_kv_stride_in_bytes"), py::arg("ssd_layer_stride_in_bytes"),
        py::arg("ssd_block_stride_in_bytes"), py::arg("ssd_kv_stride_in_bytes"),
        py::arg("block_size_in_bytes"), py::arg("total_layers"), py::arg("is_read"),
        py::arg("verbose") = false);
  m.def("export_handle", &export_memory_handle);
  m.def("import_handle", &import_memory_handle);
  m.def("get_hash_size", &flexkv::get_hash_size,
        "Get the size of the hash result");
  m.def("gen_hashes", &flexkv::gen_hashes, "Generate hashes for a tensor",
        py::arg("hasher"), py::arg("token_ids"), py::arg("tokens_per_block"),
        py::arg("block_hashes"));
  m.def("get_prefix_block_ids", &flexkv::get_prefix_block_ids,
        "Get prefix block ids", py::arg("last_block_index"),
        py::arg("last_block_id"), py::arg("prev_block_ids"));
  m.def("find_n_liner_parents_for_eviction",
        &flexkv::find_n_liner_parents_for_eviction,
        "Find n-liner parents for eviction", py::arg("block_id"),
        py::arg("prev_block_ids"), py::arg("lock_cnt"), py::arg("child_cnt"),
        py::arg("ready"), py::arg("last_access_time"),
        py::arg("max_last_access_time"), py::arg("max_num_evicted"));
  m.def("get_block_ids_from_hashes", &flexkv::get_block_ids_from_hashes,
        "Get block ids from hashes", py::arg("hashes"),
        py::arg("hash_to_block_id"));

  // Add Hasher class binding
  py::class_<flexkv::Hasher>(m, "Hasher")
      .def(py::init<>())
      .def("reset", &flexkv::Hasher::reset)
      .def("update",
           py::overload_cast<const torch::Tensor &>(&flexkv::Hasher::update),
           "Update the hasher with a tensor", py::arg("input"))
      .def("update",
           py::overload_cast<const void *, size_t>(&flexkv::Hasher::update),
           "Update the hasher with pointer and size", py::arg("input"),
           py::arg("size"))
      .def("digest", &flexkv::Hasher::digest, "Return the hash value");
}
