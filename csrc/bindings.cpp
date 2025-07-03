#include <cstddef>
#include <cstdint>
#include <stdexcept>
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
#include "pcfs/pcfs.h"
#include "tp_transfer_thread_group.h"
#include "transfer.cuh"
#include "transfer_ssd.h"

namespace py = pybind11;

void transfer_kv_blocks_binding(
    torch::Tensor &gpu_block_id_tensor, torch::Tensor &gpu_layer_ptrs_tensor,
    int64_t gpu_kv_stride_in_bytes, int64_t gpu_block_stride_in_bytes,
    torch::Tensor &cpu_block_id_tensor, torch::Tensor &cpu_tensor,
    int64_t cpu_kv_stride_in_bytes, int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_block_stride_in_bytes, int64_t chunk_size_in_bytes,
    int start_layer_id, int transfer_sms = -1, bool is_host_to_device = true,
    bool use_ce_transfer = false, bool is_mla = false) {
  int num_blocks = gpu_block_id_tensor.numel();
  int num_layers = gpu_layer_ptrs_tensor.numel();

  int64_t *gpu_block_ids =
      static_cast<int64_t *>(gpu_block_id_tensor.data_ptr());
  void **gpu_layer_ptrs = static_cast<void **>(
      gpu_layer_ptrs_tensor.data_ptr()); // must be contiguous
  int64_t *cpu_block_ids =
      static_cast<int64_t *>(cpu_block_id_tensor.data_ptr());
  void *cpu_ptr = static_cast<void *>(cpu_tensor.data_ptr());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  flexkv::transfer_kv_blocks(
      num_blocks, start_layer_id, num_layers, gpu_block_ids, gpu_layer_ptrs,
      gpu_kv_stride_in_bytes, gpu_block_stride_in_bytes, cpu_block_ids, cpu_ptr,
      cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes,
      cpu_block_stride_in_bytes, 0, chunk_size_in_bytes, stream, transfer_sms,
      is_host_to_device, use_ce_transfer, is_mla);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

void transfer_kv_blocks_ssd_binding(
    flexkv::IOUring &iouring,
    const py::dict &filepath_py_dict, const torch::Tensor &cpu_layer_id_list,
    int64_t cpu_tensor_ptr, const torch::Tensor &ssd_block_ids,
    const torch::Tensor &cpu_block_ids, int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes, int64_t ssd_layer_stride_in_bytes,
    int64_t ssd_kv_stride_in_bytes, int64_t chunk_size_in_bytes,
    int64_t block_stride_in_bytes, bool is_read, int num_blocks_per_file,
    int round_robin = 1, int num_threads_per_device = 8, bool is_mla = false) {
  TORCH_CHECK(ssd_block_ids.dtype() == torch::kInt64,
              "ssd_block_ids must be int64");
  TORCH_CHECK(cpu_block_ids.dtype() == torch::kInt64,
              "cpu_block_ids must be int64");

  std::vector<std::vector<std::string>>
      filepaths; // num_devices * num_files_per_device
  for (const auto &item : filepath_py_dict) {
    std::vector<std::string> filepath_tmp;
    py::list filepath_py_list = item.second.cast<py::list>();
    for (const auto &filepath_py : filepath_py_list) {
      filepath_tmp.push_back(filepath_py.cast<std::string>());
    }
    filepaths.emplace_back(filepath_tmp);
  }

  flexkv::transfer_kv_blocks_ssd(
      iouring,
      filepaths, cpu_layer_id_list, cpu_tensor_ptr, ssd_block_ids,
      cpu_block_ids, cpu_layer_stride_in_bytes, cpu_kv_stride_in_bytes,
      ssd_layer_stride_in_bytes, ssd_kv_stride_in_bytes, chunk_size_in_bytes,
      block_stride_in_bytes, is_read, num_blocks_per_file, round_robin,
      num_threads_per_device, is_mla);
}
#ifdef FLEXKV_ENABLE_CFS
void transfer_kv_blocks_remote(
    const py::list &file_nodeid_list, const torch::Tensor &cpu_layer_id_list,
    const torch::Tensor &cpu_layer_ptrs_tensor,
    const torch::Tensor &remote_block_ids, const torch::Tensor &cpu_block_ids,
    int64_t cpu_kv_stride_in_bytes, int64_t remote_layer_stride_in_bytes,
    int64_t remote_block_stride_in_bytes, int64_t remote_kv_stride_in_bytes,
    int64_t block_size_in_bytes, int64_t total_layers, bool is_read, int partition_block_type,
    int round_robin, int64_t num_remote_blocks_per_file, bool use_mmap = false, int num_threads_per_file = 8, bool is_mla = false) {
  TORCH_CHECK(cpu_layer_ptrs_tensor.dtype() == torch::kInt64,
              "cpu_layer_ptrs must be int64");
  TORCH_CHECK(remote_block_ids.dtype() == torch::kInt64,
              "remote_block_ids must be int64");
  TORCH_CHECK(cpu_block_ids.dtype() == torch::kInt64,
              "cpu_block_ids must be int64");
  std::vector<std::uint64_t> file_nodeids;
  for (const auto &file_nodeid : file_nodeid_list) {
    file_nodeids.push_back(file_nodeid.cast<std::uint64_t>());
  }
  flexkv::transfer_kv_blocks_cfs_mmap_multi_thread(
      file_nodeids, cpu_layer_id_list, cpu_layer_ptrs_tensor, remote_block_ids,
      cpu_block_ids, cpu_kv_stride_in_bytes, remote_layer_stride_in_bytes,
      remote_block_stride_in_bytes, remote_kv_stride_in_bytes, block_size_in_bytes,
      total_layers, is_read, partition_block_type, round_robin, num_remote_blocks_per_file, use_mmap, num_threads_per_file, is_mla);
}
#endif

PYBIND11_MODULE(c_ext, m) {
  m.def("transfer_kv_blocks", &transfer_kv_blocks_binding,
        "Transfer multi-layer KV-cache between CPU and GPU");
  m.def("transfer_kv_blocks_ssd", &transfer_kv_blocks_ssd_binding,
        "Transfer KV blocks between SSD and CPU memory",
	py::arg("iouring"),
        py::arg("filename_list"), py::arg("cpu_layer_id_list"),
        py::arg("cpu_tensor_ptr"), py::arg("ssd_block_ids"),
        py::arg("cpu_block_ids"), py::arg("cpu_layer_stride_in_bytes"),
        py::arg("cpu_kv_stride_in_bytes"), py::arg("ssd_layer_stride_in_bytes"),
        py::arg("ssd_kv_stride_in_bytes"), py::arg("chunk_size_in_bytes"),
        py::arg("block_stride_in_bytes"), py::arg("is_read"),
        py::arg("num_blocks_per_file"), py::arg("round_robin") = 1,
        py::arg("num_threads_per_device") = 16, py::arg("is_mla") = false);
#ifdef FLEXKV_ENABLE_CFS
  m.def("transfer_kv_blocks_remote", &transfer_kv_blocks_remote,
        "Transfer KV blocks between remote and CPU memory", py::arg("file_nodeid_list"),
        py::arg("cpu_layer_id_list"), py::arg("cpu_layer_ptrs_tensor"),
        py::arg("remote_block_ids"), py::arg("cpu_block_ids"),
        py::arg("cpu_kv_stride_in_bytes"), py::arg("remote_layer_stride_in_bytes"),
        py::arg("remote_block_stride_in_bytes"), py::arg("remote_kv_stride_in_bytes"),
        py::arg("block_size_in_bytes"), py::arg("total_layers"),
        py::arg("is_read"), py::arg("partition_block_type"), py::arg("round_robin"), py::arg("num_remote_blocks_per_file"), py::arg("use_mmap") = false,
        py::arg("num_threads_per_file") = 16, py::arg("is_mla") = false);
#endif
  m.def("get_hash_size", &flexkv::get_hash_size,
        "Get the size of the hash result");
  m.def("gen_hashes", &flexkv::gen_hashes, "Generate hashes for a tensor",
        py::arg("hasher"), py::arg("token_ids"), py::arg("tokens_per_block"),
        py::arg("block_hashes"));

  py::class_<flexkv::IOUring>(m, "IOUring")
      .def(py::init<int, int>());

  py::class_<flexkv::TPTransferThreadGroup>(m, "TPTransferThreadGroup")
      .def(py::init<int, const std::vector<std::vector<torch::Tensor>> &,
                    torch::Tensor &, int>())
      .def("tp_group_transfer",
           &flexkv::TPTransferThreadGroup::tp_group_transfer,
           py::arg("gpu_block_id_tensor"), py::arg("gpu_kv_stride_in_bytes"),
           py::arg("gpu_block_stride_in_bytes"),
           py::arg("gpu_chunk_size_in_bytes"), py::arg("cpu_block_id_tensor"),
           py::arg("cpu_kv_stride_in_bytes"),
           py::arg("cpu_layer_stride_in_bytes"),
           py::arg("cpu_block_stride_in_bytes"),
           py::arg("cpu_chunk_size_in_bytes"), py::arg("transfer_sms"),
           py::arg("is_host_to_device"), py::arg("use_ce_transfer"),
           py::arg("layer_id"), py::arg("layer_granularity"),
           py::arg("is_mla"));

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
#ifdef FLEXKV_ENABLE_CFS
  py::class_<flexkv::Pcfs>(m, "Pcfs")
      .def(py::init<const std::string &, uint32_t, const std::string &, bool,
                    const uint64_t>())
      .def("init", &flexkv::Pcfs::init)
      .def("destroy", &flexkv::Pcfs::destroy)
      .def("lookup_or_create_file", &flexkv::Pcfs::lookup_or_create_file,py::arg("filename"), py::arg("file_size"),py::arg("need_create"),py::call_guard<py::gil_scoped_release>())
      .def("open", &flexkv::Pcfs::open)
      .def("close", &flexkv::Pcfs::close)
      .def("write", &flexkv::Pcfs::write)
      .def("read", &flexkv::Pcfs::read);
  // .def("mkdir", &flexkv::Pcfs::mkdir)
  // .def("lookup", &flexkv::Pcfs::lookup);
  m.def("set_pcfs_instance", &flexkv::set_pcfs_instance,
        "Set the global Pcfs instance from a pointer", py::arg("pcfs"));

  m.def("call_pcfs_read", &flexkv::call_pcfs_read, "Call Pcfs::read from C++",
        py::arg("file_nodeid"), py::arg("offset"), py::arg("buffer"),
        py::arg("size"), py::arg("thread_id"));

  m.def("call_pcfs_write", &flexkv::call_pcfs_write,
        "Call Pcfs::write from C++", py::arg("file_nodeid"), py::arg("offset"),
        py::arg("buffer"), py::arg("size"), py::arg("thread_id"));
#endif
}
