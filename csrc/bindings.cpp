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
#include "transfer.cuh"
#include "transfer_ssd.h"
#include "pcfs/pcfs.h"
#include "tp_transfer_thread_group.h"

namespace py = pybind11;

void transfer_kv_layers_binding(
    torch::Tensor &dst_block_id_tensor, torch::Tensor &dst_layer_ptrs_tensor,
    int64_t dst_kv_stride_in_bytes, int64_t dst_chunk_stride_in_bytes,
    torch::Tensor &src_block_id_tensor, torch::Tensor &src_layer_ptrs_tensor,
    int64_t src_kv_stride_in_bytes, int64_t src_chunk_stride_in_bytes,
    int64_t chunk_size_in_bytes, int transfer_sms = -1,
    bool is_host_to_device = true, bool use_ce_transfer = false, bool is_mla = false) {
  int num_blocks = dst_block_id_tensor.numel();
  int num_layers = dst_layer_ptrs_tensor.numel();

  int64_t *dst_block_ids =
      static_cast<int64_t *>(dst_block_id_tensor.data_ptr());
  void **dst_layer_ptrs = static_cast<void **>(
      dst_layer_ptrs_tensor.data_ptr()); // must be contiguous
  int64_t *src_block_ids =
      static_cast<int64_t *>(src_block_id_tensor.data_ptr());
  void **src_layer_ptrs =
      static_cast<void **>(src_layer_ptrs_tensor.data_ptr());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  flexkv::transfer_kv_layers(
      num_blocks, num_layers, dst_block_ids, dst_layer_ptrs,
      dst_kv_stride_in_bytes, dst_chunk_stride_in_bytes, 0, src_block_ids,
      src_layer_ptrs, src_kv_stride_in_bytes, src_chunk_stride_in_bytes, 0,
      chunk_size_in_bytes, stream, transfer_sms, is_host_to_device,
      use_ce_transfer, is_mla);
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

void transfer_kv_blocks_ssd_binding(
    const py::list &filename_list, const torch::Tensor &cpu_layer_id_list,
    const torch::Tensor &cpu_layer_ptrs_tensor,
    const torch::Tensor &ssd_block_ids, const torch::Tensor &cpu_block_ids,
    int64_t cpu_kv_stride_in_bytes, int64_t ssd_layer_stride_in_bytes,
    int64_t ssd_block_stride_in_bytes, int64_t ssd_kv_stride_in_bytes,
    int64_t block_size_in_bytes, int64_t total_layers, bool is_read,
    int round_robin, bool use_mmap = false, int num_threads_per_file = 8,
    bool is_mla = false) {
  TORCH_CHECK(cpu_layer_ptrs_tensor.dtype() == torch::kInt64,
              "cpu_layer_ptrs must be int64");
  TORCH_CHECK(ssd_block_ids.dtype() == torch::kInt64,
              "ssd_block_ids must be int64");
  TORCH_CHECK(cpu_block_ids.dtype() == torch::kInt64,
              "cpu_block_ids must be int64");

  std::vector<std::string> filenames;
  for (const auto &filename : filename_list) {
    filenames.push_back(filename.cast<std::string>());
  }
  flexkv::transfer_kv_blocks_ssd(
      filenames, cpu_layer_id_list, cpu_layer_ptrs_tensor, ssd_block_ids,
      cpu_block_ids, cpu_kv_stride_in_bytes, ssd_layer_stride_in_bytes,
      ssd_block_stride_in_bytes, ssd_kv_stride_in_bytes, block_size_in_bytes,
      total_layers, is_read, round_robin, use_mmap, num_threads_per_file, is_mla);
}
#ifdef FLEXKV_ENABLE_CFS
void transfer_kv_blocks_remote(
    const py::list &file_nodeid_list, const torch::Tensor &cpu_layer_id_list,
    const torch::Tensor &cpu_layer_ptrs_tensor,
    const torch::Tensor &remote_block_ids, const torch::Tensor &cpu_block_ids,
    int64_t cpu_kv_stride_in_bytes, int64_t remote_layer_stride_in_bytes,
    int64_t remote_block_stride_in_bytes, int64_t remote_kv_stride_in_bytes,
    int64_t block_size_in_bytes, int64_t total_layers, bool is_read,
    int round_robin, bool use_mmap = false, int num_threads_per_file = 8,
    bool is_mla = false) {
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
      total_layers, is_read, round_robin, use_mmap, num_threads_per_file, is_mla);
}
#endif

PYBIND11_MODULE(c_ext, m) {
  m.def("transfer_kv_layers", &transfer_kv_layers_binding,
        "Transfer multi-layer KV-cache between CPU and GPU");
  m.def("transfer_kv_blocks_ssd", &transfer_kv_blocks_ssd_binding,
        "Transfer KV blocks between SSD and CPU memory",
        py::arg("filename_list"), py::arg("cpu_layer_id_list"),
        py::arg("cpu_layer_ptrs_tensor"), py::arg("ssd_block_ids"),
        py::arg("cpu_block_ids"), py::arg("cpu_kv_stride_in_bytes"),
        py::arg("ssd_layer_stride_in_bytes"),
        py::arg("ssd_block_stride_in_bytes"), py::arg("ssd_kv_stride_in_bytes"),
        py::arg("block_size_in_bytes"), py::arg("total_layers"),
        py::arg("is_read"), py::arg("round_robin"), py::arg("use_mmap") = false,
        py::arg("num_threads_per_file") = 8, py::arg("is_mla") = false);
#ifdef FLEXKV_ENABLE_CFS
  m.def("transfer_kv_blocks_remote", &transfer_kv_blocks_remote,
        "Transfer KV blocks between remote and CPU memory", py::arg("file_nodeid_list"),
        py::arg("cpu_layer_id_list"), py::arg("cpu_layer_ptrs_tensor"),
        py::arg("remote_block_ids"), py::arg("cpu_block_ids"),
        py::arg("cpu_kv_stride_in_bytes"), py::arg("remote_layer_stride_in_bytes"),
        py::arg("remote_block_stride_in_bytes"), py::arg("remote_kv_stride_in_bytes"),
        py::arg("block_size_in_bytes"), py::arg("total_layers"),
        py::arg("is_read"), py::arg("round_robin"), py::arg("use_mmap") = false,
        py::arg("num_threads_per_file") = 16, py::arg("is_mla") = false);
#endif
  m.def("get_hash_size", &flexkv::get_hash_size,
        "Get the size of the hash result");
  m.def("gen_hashes", &flexkv::gen_hashes, "Generate hashes for a tensor",
        py::arg("hasher"), py::arg("token_ids"), py::arg("tokens_per_block"),
        py::arg("block_hashes"));

  py::class_<flexkv::TPTransferThreadGroup>(m, "TPTransferThreadGroup")
      .def(py::init<int, const std::vector<std::vector<torch::Tensor>>&, const std::vector<torch::Tensor>&, int>())
      .def("tp_group_transfer", &flexkv::TPTransferThreadGroup::tp_group_transfer,
            py::arg("dst_block_id_tensors"),
            py::arg("dst_kv_stride_in_bytes"),
            py::arg("dst_chunk_stride_in_bytes"),
            py::arg("dst_chunk_size_in_bytes"),
            py::arg("src_block_id_tensors"),
            py::arg("src_kv_stride_in_bytes"),
            py::arg("src_chunk_stride_in_bytes"), 
            py::arg("src_chunk_size_in_bytes"),
            py::arg("transfer_sms"),
            py::arg("is_host_to_device"),
            py::arg("use_ce_transfer"),
            py::arg("layer_id"),
            py::arg("layer_granularity"),
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
      .def(py::init<const std::string&, uint32_t, const std::string&, bool, const uint64_t>())
      .def("init", &flexkv::Pcfs::init)
      .def("destroy", &flexkv::Pcfs::destroy)
      .def("lookup_or_create_file", &flexkv::Pcfs::lookup_or_create_file,py::arg("filename"), py::arg("file_size"),py::call_guard<py::gil_scoped_release>())
      .def("open", &flexkv::Pcfs::open)
      .def("close", &flexkv::Pcfs::close)
      .def("write", &flexkv::Pcfs::write)
      .def("read", &flexkv::Pcfs::read);
      // .def("mkdir", &flexkv::Pcfs::mkdir)
      // .def("lookup", &flexkv::Pcfs::lookup);
  m.def("set_pcfs_instance", &flexkv::set_pcfs_instance,
        "Set the global Pcfs instance from a pointer",
        py::arg("pcfs"));

  m.def("call_pcfs_read", &flexkv::call_pcfs_read,
        "Call Pcfs::read from C++",
        py::arg("file_nodeid"), py::arg("offset"), py::arg("buffer"), py::arg("size"), py::arg("thread_id"));

  m.def("call_pcfs_write", &flexkv::call_pcfs_write,
        "Call Pcfs::write from C++",
        py::arg("file_nodeid"), py::arg("offset"), py::arg("buffer"), py::arg("size"), py::arg("thread_id"));
#endif
}
