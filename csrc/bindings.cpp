#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <map>

#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include "transfer.cuh"
#include <fcntl.h>
#include <nvtx3/nvToolsExt.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <torch/extension.h>
#include <unistd.h>

#include "cache_utils.h"
#include "pcfs/pcfs.h"
#include "tp_transfer_thread_group.h"
#include "gds/tp_gds_transfer_thread_group.h"
#include "transfer.cuh"
#include "transfer_ssd.h"
#include "radix_tree.h"
#include "gds/gds_manager.h"
#include "local_radix_tree.h"
#include "distributed_radix_tree.h"
#include "redis_meta_channel.h"
#include "block_meta.h"
#include "lock_free_q.h"
#include "lease_meta_mempool.h"
#include "fiemap_extent.h"
#include "nvme_connect.h"
#include <deque>

namespace py = pybind11;

void transfer_kv_blocks_binding(
    torch::Tensor &gpu_block_id_tensor, torch::Tensor &gpu_tensor_ptrs_tensor,
    int64_t gpu_kv_stride_in_bytes, int64_t gpu_block_stride_in_bytes, int64_t gpu_layer_stride_in_bytes,
    torch::Tensor &cpu_block_id_tensor, torch::Tensor &cpu_tensor,
    int64_t cpu_kv_stride_in_bytes, int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_block_stride_in_bytes, int64_t chunk_size_in_bytes,
    int start_layer_id, int num_layers, int transfer_sms = -1, bool is_host_to_device = true,
    bool use_ce_transfer = false, bool is_mla = false, int gpu_block_type = 0) {
  int num_blocks = gpu_block_id_tensor.numel();

  int64_t *gpu_block_ids =
      static_cast<int64_t *>(gpu_block_id_tensor.data_ptr());
  void **gpu_tensor_ptrs = static_cast<void **>(
      gpu_tensor_ptrs_tensor.data_ptr()); // must be contiguous
  int64_t *cpu_block_ids =
      static_cast<int64_t *>(cpu_block_id_tensor.data_ptr());
  void *cpu_ptr = static_cast<void *>(cpu_tensor.data_ptr());

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();
  
  // Determine backend type from gpu_block_type parameter
  flexkv::BackendType backend_type;
  if (gpu_block_type == 0) {
    backend_type = flexkv::BackendType::VLLM;
  } else if (gpu_block_type == 1) {
    backend_type = flexkv::BackendType::TRTLLM;
  } else if (gpu_block_type == 2) {
    backend_type = flexkv::BackendType::SGLANG;
  } else {
    throw std::runtime_error("Unsupported gpu_block_type: " + std::to_string(gpu_block_type));
  }
  
  // Create GTensorHandler
  flexkv::GTensorHandler handler(
      backend_type,
      reinterpret_cast<int64_t**>(gpu_tensor_ptrs),
      num_layers,
      gpu_kv_stride_in_bytes,
      gpu_block_stride_in_bytes,
      gpu_layer_stride_in_bytes
  );
  
  // Dispatch to appropriate template instantiation
  switch (backend_type) {
    case flexkv::BackendType::VLLM:
      flexkv::transfer_kv_blocks<flexkv::BackendType::VLLM>(
          num_blocks, start_layer_id, num_layers, gpu_block_ids, handler, 0,
          cpu_block_ids, cpu_ptr, cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes,
          cpu_block_stride_in_bytes, 0, chunk_size_in_bytes, stream, transfer_sms,
          is_host_to_device, use_ce_transfer, is_mla);
      break;
    case flexkv::BackendType::TRTLLM:
      flexkv::transfer_kv_blocks<flexkv::BackendType::TRTLLM>(
          num_blocks, start_layer_id, num_layers, gpu_block_ids, handler, 0,
          cpu_block_ids, cpu_ptr, cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes,
          cpu_block_stride_in_bytes, 0, chunk_size_in_bytes, stream, transfer_sms,
          is_host_to_device, use_ce_transfer, is_mla);
      break;
    case flexkv::BackendType::SGLANG:
      flexkv::transfer_kv_blocks<flexkv::BackendType::SGLANG>(
          num_blocks, start_layer_id, num_layers, gpu_block_ids, handler, 0,
          cpu_block_ids, cpu_ptr, cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes,
          cpu_block_stride_in_bytes, 0, chunk_size_in_bytes, stream, transfer_sms,
          is_host_to_device, use_ce_transfer, is_mla);
      break;
  }
  
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    throw std::runtime_error(cudaGetErrorString(err));
  }
}

void transfer_kv_blocks_ssd_binding(
    flexkv::SSDIOCTX &ioctx,
    const torch::Tensor &cpu_layer_id_list, int64_t cpu_tensor_ptr,
    const torch::Tensor &ssd_block_ids, const torch::Tensor &cpu_block_ids,
    int64_t cpu_layer_stride_in_bytes, int64_t cpu_kv_stride_in_bytes,
    int64_t ssd_layer_stride_in_bytes, int64_t ssd_kv_stride_in_bytes,
    int64_t chunk_size_in_bytes, int64_t block_stride_in_bytes, bool is_read,
    int num_blocks_per_file, int round_robin = 1,
    int num_threads_per_device = 8, bool is_mla = false) {
  TORCH_CHECK(ssd_block_ids.dtype() == torch::kInt64,
              "ssd_block_ids must be int64");
  TORCH_CHECK(cpu_block_ids.dtype() == torch::kInt64,
              "cpu_block_ids must be int64");

  flexkv::transfer_kv_blocks_ssd(
      ioctx, cpu_layer_id_list, cpu_tensor_ptr, ssd_block_ids,
      cpu_block_ids, cpu_layer_stride_in_bytes, cpu_kv_stride_in_bytes,
      ssd_layer_stride_in_bytes, ssd_kv_stride_in_bytes, chunk_size_in_bytes,
      block_stride_in_bytes, is_read, num_blocks_per_file, round_robin,
      num_threads_per_device, is_mla);
}

#ifdef FLEXKV_ENABLE_CFS
void transfer_kv_blocks_remote(
    const py::list &file_nodeid_list, const torch::Tensor &cpu_layer_id_list,
    int64_t cpu_tensor_ptr, const torch::Tensor &remote_block_ids,
    const torch::Tensor &cpu_block_ids, int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes, int64_t remote_layer_stride_in_bytes,
    int64_t remote_block_stride_in_bytes, int64_t remote_kv_stride_in_bytes,
    int64_t block_size_in_bytes, int64_t total_layers, bool is_read,
    int partition_block_type, int round_robin,
    int64_t num_remote_blocks_per_file, bool use_mmap = false,
    int num_threads_per_file = 8, bool is_mla = false) {
  TORCH_CHECK(remote_block_ids.dtype() == torch::kInt64,
              "remote_block_ids must be int64");
  TORCH_CHECK(cpu_block_ids.dtype() == torch::kInt64,
              "cpu_block_ids must be int64");
  std::vector<std::uint64_t> file_nodeids;
  for (const auto &file_nodeid : file_nodeid_list) {
    file_nodeids.push_back(file_nodeid.cast<std::uint64_t>());
  }
  flexkv::transfer_kv_blocks_cfs_mmap_multi_thread(
      file_nodeids, cpu_layer_id_list, cpu_tensor_ptr, remote_block_ids,
      cpu_block_ids, cpu_layer_stride_in_bytes, cpu_kv_stride_in_bytes,
      remote_layer_stride_in_bytes, remote_block_stride_in_bytes,
      remote_kv_stride_in_bytes, block_size_in_bytes, total_layers, is_read,
      partition_block_type, round_robin, num_remote_blocks_per_file, use_mmap,
      num_threads_per_file, is_mla);
}

void shared_transfer_kv_blocks_remote_read_binding(
    const py::list &file_nodeid_list,
    const py::list &cfs_blocks_partition_list,
    const py::list &cpu_blocks_partition_list,
    const torch::Tensor &cpu_layer_id_list,
    int64_t cpu_tensor_ptr,
    int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes,
    int64_t cfs_layer_stride_in_bytes,
    int64_t cfs_block_stride_in_bytes,
    int64_t cfs_kv_stride_in_bytes,
    int64_t block_size_in_bytes,
    int64_t total_layers,
    bool is_mla = false,
    int num_threads_per_file = 8) {
    
    // 转换 file_nodeids
    std::vector<std::uint64_t> file_nodeids;
    for (const auto &file_nodeid : file_nodeid_list) {
        file_nodeids.push_back(file_nodeid.cast<std::uint64_t>());
    }
    
    // 转换 cfs_blocks_partition
    std::vector<std::vector<int64_t>> cfs_blocks_partition;
    for (const auto &block_list : cfs_blocks_partition_list) {
        std::vector<int64_t> blocks;
        for (const auto &block_id : block_list) {
            blocks.push_back(block_id.cast<int64_t>());
        }
        cfs_blocks_partition.push_back(std::move(blocks));
    }
    
    // 转换 cpu_blocks_partition
    std::vector<std::vector<int64_t>> cpu_blocks_partition;
    for (const auto &block_list : cpu_blocks_partition_list) {
        std::vector<int64_t> blocks;
        for (const auto &block_id : block_list) {
            blocks.push_back(block_id.cast<int64_t>());
        }
        cpu_blocks_partition.push_back(std::move(blocks));
    }
    
    // 调用 C++ 实现
    flexkv::shared_transfer_kv_blocks_remote_read(
        file_nodeids, cfs_blocks_partition, cpu_blocks_partition, cpu_layer_id_list,
        cpu_tensor_ptr, cpu_layer_stride_in_bytes, cpu_kv_stride_in_bytes,
        cfs_layer_stride_in_bytes, cfs_block_stride_in_bytes, cfs_kv_stride_in_bytes,
        block_size_in_bytes, total_layers, is_mla, num_threads_per_file);
}
#endif

void transfer_kv_blocks_gds_binding(
    GDSManager& gds_manager,
    const torch::Tensor& gpu_layer_id_list,
    const torch::Tensor& gpu_layer_ptrs_tensor,
    const torch::Tensor& gds_block_ids,
    const torch::Tensor& gpu_block_ids,
    int64_t gpu_kv_stride_in_bytes,
    int64_t gpu_block_stride_in_bytes,
    int64_t gpu_layer_stride_in_bytes,
    int64_t gds_layer_stride_in_bytes,
    int64_t gds_block_stride_in_bytes,
    int64_t gds_kv_stride_in_bytes,
    int64_t block_size_in_bytes,
    int64_t gds_copy_off_inside_chunks,
    int num_blocks_per_file,
    int64_t total_layers,
    bool is_read,
    bool verbose = false,
    bool is_mla = false,
    int gpu_block_type = 0
) {
    TORCH_CHECK(gpu_layer_ptrs_tensor.dtype() == torch::kInt64,
                "gpu_layer_ptrs must be int64");
    TORCH_CHECK(gds_block_ids.dtype() == torch::kInt64,
                "gds_block_ids must be int64");
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
        throw std::runtime_error("Unsupported gpu_block_type: " + std::to_string(gpu_block_type));
    }
    
    // Create GTensorHandler
    void **gpu_tensor_ptrs = static_cast<void **>(gpu_layer_ptrs_tensor.data_ptr());
    flexkv::GTensorHandler handler(
        backend_type,
        reinterpret_cast<int64_t**>(gpu_tensor_ptrs),
        total_layers,
        gpu_kv_stride_in_bytes,
        gpu_block_stride_in_bytes,
        gpu_layer_stride_in_bytes
    );
    
    switch (backend_type) {
        case flexkv::BackendType::VLLM:
            flexkv::transfer_kv_blocks_gds<flexkv::BackendType::VLLM>(
                gds_manager, gpu_layer_id_list, handler, gds_block_ids, gpu_block_ids,
                gds_layer_stride_in_bytes, gds_block_stride_in_bytes, gds_kv_stride_in_bytes,
                block_size_in_bytes, gds_copy_off_inside_chunks, num_blocks_per_file,
                total_layers, is_read, verbose, is_mla);
            break;
        case flexkv::BackendType::TRTLLM:
            flexkv::transfer_kv_blocks_gds<flexkv::BackendType::TRTLLM>(
                gds_manager, gpu_layer_id_list, handler, gds_block_ids, gpu_block_ids,
                gds_layer_stride_in_bytes, gds_block_stride_in_bytes, gds_kv_stride_in_bytes,
                block_size_in_bytes, gds_copy_off_inside_chunks, num_blocks_per_file,
                total_layers, is_read, verbose, is_mla);
            break;
        case flexkv::BackendType::SGLANG:
            flexkv::transfer_kv_blocks_gds<flexkv::BackendType::SGLANG>(
                gds_manager, gpu_layer_id_list, handler, gds_block_ids, gpu_block_ids,
                gds_layer_stride_in_bytes, gds_block_stride_in_bytes, gds_kv_stride_in_bytes,
                block_size_in_bytes, gds_copy_off_inside_chunks, num_blocks_per_file,
                total_layers, is_read, verbose, is_mla);
            break;
    }
}

// GDS Manager Python bindings
py::list gds_batch_write_binding(GDSManager& manager, 
                                 py::list operations_list) {
    size_t batch_size = operations_list.size();
    std::vector<BatchWriteOp> operations(batch_size);
    std::vector<ssize_t> results(batch_size);
    
    for (size_t i = 0; i < batch_size; ++i) {
        py::dict op_dict = operations_list[i].cast<py::dict>();
        operations[i].filename = op_dict["filename"].cast<std::string>().c_str();
        operations[i].gpu_data = op_dict["gpu_data"].cast<torch::Tensor>().data_ptr();
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

py::list gds_batch_read_binding(GDSManager& manager, 
                                py::list operations_list) {
    size_t batch_size = operations_list.size();
    std::vector<BatchReadOp> operations(batch_size);
    std::vector<ssize_t> results(batch_size);
    
    for (size_t i = 0; i < batch_size; ++i) {
        py::dict op_dict = operations_list[i].cast<py::dict>();
        operations[i].filename = op_dict["filename"].cast<std::string>().c_str();
        operations[i].gpu_buffer = op_dict["gpu_buffer"].cast<torch::Tensor>().data_ptr();
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

ssize_t gds_write_binding(GDSManager& manager, 
                         const std::string& filename,
                         torch::Tensor gpu_data,
                         size_t file_offset = 0) {
    return manager.write(filename.c_str(), gpu_data.data_ptr(), 
                        gpu_data.numel() * gpu_data.element_size(), file_offset);
}

ssize_t gds_read_binding(GDSManager& manager,
                        const std::string& filename, 
                        torch::Tensor gpu_buffer,
                        size_t file_offset = 0) {
    return manager.read(filename.c_str(), gpu_buffer.data_ptr(),
                       gpu_buffer.numel() * gpu_buffer.element_size(), file_offset);
}

ssize_t gds_write_async_binding(GDSManager& manager,
                               const std::string& filename,
                               torch::Tensor gpu_data,
                               size_t file_offset = 0) {
    return manager.write_async(filename.c_str(), gpu_data.data_ptr(),
                              gpu_data.numel() * gpu_data.element_size(), file_offset);
}

ssize_t gds_read_async_binding(GDSManager& manager,
                              const std::string& filename,
                              torch::Tensor gpu_buffer, 
                              size_t file_offset = 0) {
    return manager.read_async(filename.c_str(), gpu_buffer.data_ptr(),
                             gpu_buffer.numel() * gpu_buffer.element_size(), file_offset);
}

// Helper function to create and initialize a GDS file with specified size
bool create_gds_file_binding(GDSManager& manager, 
                             const std::string& filename, 
                             size_t file_size) {
    // First create/truncate the file to the desired size
    int fd = open(filename.c_str(), O_CREAT | O_RDWR | O_TRUNC, 0644);
    if (fd < 0) {
        return false;
    }
    
    // Pre-allocate the file to the specified size
    if (ftruncate(fd, file_size) != 0) {
        close(fd);
        return false;
    }
    
    // Ensure data is written to disk
    fsync(fd);
    close(fd);
    
    // Now add the file to GDS manager (this will open it with O_DIRECT and register with cuFile)
    return manager.add_file(filename.c_str());
}

PYBIND11_MODULE(c_ext, m) {
  m.def("transfer_kv_blocks", &transfer_kv_blocks_binding,
        "Transfer multi-layer KV-cache between CPU and GPU",
        py::arg("gpu_block_id_tensor"), py::arg("gpu_tensor_ptrs_tensor"),
        py::arg("gpu_kv_stride_in_bytes"), py::arg("gpu_block_stride_in_bytes"),
        py::arg("gpu_layer_stride_in_bytes"), py::arg("cpu_block_id_tensor"),
        py::arg("cpu_tensor"), py::arg("cpu_kv_stride_in_bytes"),
        py::arg("cpu_layer_stride_in_bytes"), py::arg("cpu_block_stride_in_bytes"),
        py::arg("chunk_size_in_bytes"), py::arg("start_layer_id"),
        py::arg("num_layers"), py::arg("transfer_sms") = -1,
        py::arg("is_host_to_device") = true, py::arg("use_ce_transfer") = false,
        py::arg("is_mla") = false, py::arg("gpu_block_type") = 0);
  m.def("transfer_kv_blocks_ssd", &transfer_kv_blocks_ssd_binding,
        "Transfer KV blocks between SSD and CPU memory",
        py::arg("ioctx"), py::arg("cpu_layer_id_list"),
        py::arg("cpu_tensor_ptr"), py::arg("ssd_block_ids"),
        py::arg("cpu_block_ids"), py::arg("cpu_layer_stride_in_bytes"),
        py::arg("cpu_kv_stride_in_bytes"), py::arg("ssd_layer_stride_in_bytes"),
        py::arg("ssd_kv_stride_in_bytes"), py::arg("chunk_size_in_bytes"),
        py::arg("block_stride_in_bytes"), py::arg("is_read"),
        py::arg("num_blocks_per_file"), py::arg("round_robin") = 1,
        py::arg("num_threads_per_device") = 16, py::arg("is_mla") = false);

#ifdef FLEXKV_ENABLE_CFS
  m.def("transfer_kv_blocks_remote", &transfer_kv_blocks_remote,
        "Transfer KV blocks between remote and CPU memory",
        py::arg("file_nodeid_list"), py::arg("cpu_layer_id_list"),
        py::arg("cpu_tensor_ptr"), py::arg("remote_block_ids"),
        py::arg("cpu_block_ids"), py::arg("cpu_layer_stride_in_bytes"),
        py::arg("cpu_kv_stride_in_bytes"),
        py::arg("remote_layer_stride_in_bytes"),
        py::arg("remote_block_stride_in_bytes"),
        py::arg("remote_kv_stride_in_bytes"), py::arg("block_size_in_bytes"),
        py::arg("total_layers"), py::arg("is_read"),
        py::arg("partition_block_type"), py::arg("round_robin"),
        py::arg("num_remote_blocks_per_file"), py::arg("use_mmap") = false,
        py::arg("num_threads_per_file") = 16, py::arg("is_mla") = false);
#endif
  m.def("transfer_kv_blocks_gds", &transfer_kv_blocks_gds_binding,
        "Transfer KV blocks between GPU and GDS storage", py::arg("gds_manager"),
        py::arg("gpu_layer_id_list"), py::arg("gpu_layer_ptrs_tensor"),
        py::arg("gds_block_ids"), py::arg("gpu_block_ids"),
        py::arg("gpu_kv_stride_in_bytes"), py::arg("gpu_block_stride_in_bytes"),
        py::arg("gpu_layer_stride_in_bytes"),
        py::arg("gds_layer_stride_in_bytes"), py::arg("gds_block_stride_in_bytes"),
        py::arg("gds_kv_stride_in_bytes"), py::arg("block_size_in_bytes"),
        py::arg("gds_copy_off_inside_chunks"),
        py::arg("num_blocks_per_file"), py::arg("total_layers"), 
        py::arg("is_read"), py::arg("verbose") = false, py::arg("is_mla") = false,
        py::arg("gpu_block_type") = 0);
  m.def("get_hash_size", &flexkv::get_hash_size,
        "Get the size of the hash result");
  m.def("gen_hashes", &flexkv::gen_hashes, "Generate hashes for a tensor",
        py::arg("hasher"), py::arg("token_ids"), py::arg("tokens_per_block"),
        py::arg("block_hashes"));
  m.def("get_fm_extents", &flexkv::get_fm_extents,
        "Get FIEMAP extents of a file",
        py::arg("fd"), py::arg("max_extents") = 256,
        py::call_guard<py::gil_scoped_release>());
  m.def("nvme_connect", &flexkv::nvme_connect,
        "Connect NVMe-oF targets",
        py::arg("nvmets"));

  py::class_<flexkv::SSDIOCTX>(m, "SSDIOCTX")
      .def(py::init<std::map<int, std::vector<std::string>> &, int, int, int>());

  py::class_<flexkv::TPTransferThreadGroup>(m, "TPTransferThreadGroup")
      .def(py::init<int, const std::vector<std::vector<torch::Tensor>> &,
                    torch::Tensor &, int, int, torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &>(),
           py::arg("num_gpus"), py::arg("gpu_blocks"), py::arg("cpu_blocks"),
           py::arg("dp_group_id"), py::arg("num_layers"),
           py::arg("gpu_kv_strides_tensor"), py::arg("gpu_block_strides_tensor"),
           py::arg("gpu_layer_strides_tensor"), py::arg("gpu_chunk_sizes_tensor"))
      .def("tp_group_transfer",
           &flexkv::TPTransferThreadGroup::tp_group_transfer,
           py::arg("gpu_block_id_tensor"), py::arg("cpu_block_id_tensor"),
           py::arg("cpu_kv_stride_in_bytes"),
           py::arg("cpu_layer_stride_in_bytes"),
           py::arg("cpu_block_stride_in_bytes"),
           py::arg("cpu_chunk_size_in_bytes"), py::arg("transfer_sms"),
           py::arg("is_host_to_device"), py::arg("use_ce_transfer"),
           py::arg("layer_id"), py::arg("layer_granularity"),
           py::arg("is_mla"));

  py::class_<flexkv::TPGDSTransferThreadGroup>(m, "TPGDSTransferThreadGroup")
      .def(py::init<int, const std::vector<std::vector<torch::Tensor>> &,
                    std::map<int, std::vector<std::string>> &, int, int,
                    torch::Tensor &, torch::Tensor &, torch::Tensor &, torch::Tensor &>(),
           py::arg("num_gpus"), py::arg("gpu_blocks"), py::arg("ssd_files"),
           py::arg("dp_group_id"), py::arg("num_layers"),
           py::arg("gpu_kv_strides_tensor"), py::arg("gpu_block_strides_tensor"),
           py::arg("gpu_layer_strides_tensor"), py::arg("gpu_chunk_sizes_tensor"))
      .def("tp_group_transfer",
           &flexkv::TPGDSTransferThreadGroup::tp_group_transfer,
           py::arg("gpu_block_id_tensor"), py::arg("ssd_block_id_tensor"),
           py::arg("ssd_layer_stride_in_bytes"),
           py::arg("ssd_kv_stride_in_bytes"), py::arg("ssd_block_stride_in_bytes"),
           py::arg("ssd_chunk_size_in_bytes"), py::arg("num_blocks_per_file"),
           py::arg("is_read"), py::arg("layer_id"), py::arg("layer_granularity"),
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
      .def("lookup_or_create_file", &flexkv::Pcfs::lookup_or_create_file,
           py::arg("filename"), py::arg("file_size"), py::arg("need_create"),
           py::call_guard<py::gil_scoped_release>())
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

  m.def("shared_transfer_kv_blocks_remote_read", 
        &shared_transfer_kv_blocks_remote_read_binding,
        "Shared transfer KV blocks from remote PCFS to CPU memory",
        py::arg("file_nodeid_list"),
        py::arg("cfs_blocks_partition_list"),
        py::arg("cpu_blocks_partition_list"),
        py::arg("cpu_layer_id_list"),
        py::arg("cpu_tensor_ptr"),
        py::arg("cpu_layer_stride_in_bytes"),
        py::arg("cpu_kv_stride_in_bytes"),
        py::arg("cfs_layer_stride_in_bytes"),
        py::arg("cfs_block_stride_in_bytes"),
        py::arg("cfs_kv_stride_in_bytes"),
        py::arg("block_size_in_bytes"),
        py::arg("total_layers"),
        py::arg("is_mla") = false,
        py::arg("num_threads_per_file") = 8);
#endif

  py::class_<flexkv::CRadixTreeIndex>(m, "CRadixTreeIndex")
      .def(py::init<int, unsigned int, int>(),
           py::arg("tokens_per_block"),
           py::arg("max_num_blocks") = 1000000,
           py::arg("hit_reward_seconds") = 0)
      .def("is_empty", &flexkv::CRadixTreeIndex::is_empty)
      .def("reset", &flexkv::CRadixTreeIndex::reset)
      .def("lock", &flexkv::CRadixTreeIndex::lock, py::arg("node"))
      .def("unlock", &flexkv::CRadixTreeIndex::unlock, py::arg("node"))
      .def("set_ready", &flexkv::CRadixTreeIndex::set_ready,
           py::arg("node"), py::arg("ready"), py::arg("ready_length"))
      .def("insert", &flexkv::CRadixTreeIndex::insert, py::return_value_policy::reference,
          py::arg("physical_block_ids"), py::arg("block_hashes"), py::arg("num_blocks"),
          py::arg("num_insert_blocks"), py::arg("ready") = true, py::arg("node") = nullptr,
          py::arg("num_matched_blocks") = -1, py::arg("last_node_matched_length") = -1,
          py::call_guard<py::gil_scoped_release>())
      .def("evict", &flexkv::CRadixTreeIndex::evict, py::arg("evicted_blocks"), py::arg("num_evicted"),
           py::call_guard<py::gil_scoped_release>())
      .def("total_cached_blocks", &flexkv::CRadixTreeIndex::total_cached_blocks)
      .def("total_unready_blocks", &flexkv::CRadixTreeIndex::total_unready_blocks)
      .def("total_ready_blocks", &flexkv::CRadixTreeIndex::total_ready_blocks)
      .def("match_prefix", &flexkv::CRadixTreeIndex::match_prefix,
           py::arg("block_hashes"), py::arg("num_blocks"), py::arg("update_cache_info"),
           py::call_guard<py::gil_scoped_release>());

  py::class_<flexkv::CRadixNode>(m, "CRadixNode")
      .def(py::init<flexkv::CRadixTreeIndex *, bool, int>())
      .def(py::init<flexkv::CRadixTreeIndex *, bool, int, bool>())
      .def("size", &flexkv::CRadixNode::size)
      .def("has_block_node_ids", &flexkv::CRadixNode::has_block_node_ids);

  py::class_<flexkv::CMatchResult, std::shared_ptr<flexkv::CMatchResult>>(m, "CMatchResult")
      .def(py::init<int, int, int, flexkv::CRadixNode *, flexkv::CRadixNode *, torch::Tensor, torch::Tensor>())
      .def_readonly("last_ready_node", &flexkv::CMatchResult::last_ready_node)
      .def_readonly("last_node", &flexkv::CMatchResult::last_node)
      .def_readonly("physical_blocks", &flexkv::CMatchResult::physical_blocks)
      .def_readonly("block_node_ids", &flexkv::CMatchResult::block_node_ids)
      .def_readonly("num_ready_matched_blocks", &flexkv::CMatchResult::num_ready_matched_blocks)
      .def_readonly("num_matched_blocks", &flexkv::CMatchResult::num_matched_blocks)
      .def_readonly("last_node_matched_length", &flexkv::CMatchResult::last_node_matched_length);
  // Add GDS Manager class binding
  py::class_<GDSManager>(m, "GDSManager")
      .def(py::init<std::map<int, std::vector<std::string>>&, int, int>(),
           "Initialize GDS Manager with device-organized files",
           py::arg("ssd_files"), py::arg("num_devices"), py::arg("round_robin") = 1)
      .def("is_ready", &GDSManager::is_ready,
           "Check if GDS manager is ready for operations")
      .def("get_last_error", &GDSManager::get_last_error,
           "Get the last error message")
      .def("add_file", &GDSManager::add_file,
           "Add and register a file with GDS (creates with O_DIRECT)", py::arg("filename"))
      .def("remove_file", &GDSManager::remove_file,
           "Remove and unregister a file from GDS", py::arg("filename"))
      .def("write", &gds_write_binding,
           "Write data from GPU memory to file", 
           py::arg("filename"), py::arg("gpu_data"), py::arg("file_offset") = 0)
      .def("read", &gds_read_binding,
           "Read data from file to GPU memory",
           py::arg("filename"), py::arg("gpu_buffer"), py::arg("file_offset") = 0)
      .def("write_async", &gds_write_async_binding,
           "Write data from GPU memory to file asynchronously",
           py::arg("filename"), py::arg("gpu_data"), py::arg("file_offset") = 0)
      .def("read_async", &gds_read_async_binding,
           "Read data from file to GPU memory asynchronously",
           py::arg("filename"), py::arg("gpu_buffer"), py::arg("file_offset") = 0)
      .def("batch_write", &gds_batch_write_binding,
           "Batch write operations", py::arg("operations"))
      .def("batch_read", &gds_batch_read_binding,
           "Batch read operations", py::arg("operations"))
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
           "Get file paths for a specific device",
           py::arg("device_id"))
      .def("create_gds_file", &create_gds_file_binding,
            "Create and register a GDS file with specified size", 
            py::arg("filename"), py::arg("file_size"));

  // BlockMeta binding
  py::class_<flexkv::BlockMeta>(m, "BlockMeta")
      .def(py::init<>())
      .def_readwrite("ph", &flexkv::BlockMeta::ph)
      .def_readwrite("pb", &flexkv::BlockMeta::pb)
      .def_readwrite("nid", &flexkv::BlockMeta::nid)
      .def_readwrite("hash", &flexkv::BlockMeta::hash)
      .def_readwrite("lt", &flexkv::BlockMeta::lt)
      .def_readwrite("state", &flexkv::BlockMeta::state);

  // Expose a simple LockFreeQueue<int> to Python as IntQueue
  py::class_<flexkv::LockFreeQueue<int>>(m, "IntQueue")
      .def(py::init<>())
      .def("push", [](flexkv::LockFreeQueue<int> &q, int value) {
        q.push(value);
      }, py::arg("value"))
      .def("pop", [](flexkv::LockFreeQueue<int> &q) {
        int value = 0;
        bool ok = q.pop(value);
        return py::make_tuple(ok, value);
      });

  // RedisMetaChannel binding
  py::class_<flexkv::RedisMetaChannel>(m, "RedisMetaChannel")
      .def(py::init<const std::string&, int, uint32_t, const std::string&, const std::string&, const std::string&>(),
           py::arg("host"), py::arg("port"), py::arg("node_id"), py::arg("local_ip"), py::arg("blocks_key") = std::string("blocks"), py::arg("password") = std::string(""))
      .def("connect", &flexkv::RedisMetaChannel::connect)
      .def("get_node_id", &flexkv::RedisMetaChannel::get_node_id)
      .def("get_local_ip", &flexkv::RedisMetaChannel::get_local_ip)
      .def("make_block_key", &flexkv::RedisMetaChannel::make_block_key, py::arg("node_id"), py::arg("hash"))
      .def("publish_one", [](flexkv::RedisMetaChannel &ch, const flexkv::BlockMeta &m){ return ch.publish(m); })
      .def("publish_batch", [](flexkv::RedisMetaChannel &ch, const std::vector<flexkv::BlockMeta> &metas, size_t batch_size){ return ch.publish(metas, batch_size); }, py::arg("metas"), py::arg("batch_size")=100)
      .def("load", [](flexkv::RedisMetaChannel &ch, size_t max_items){ std::vector<flexkv::BlockMeta> out; ch.load(out, max_items); return out; }, py::arg("max_items"))
      .def("renew_node_leases", py::overload_cast<uint32_t, uint64_t, size_t>(&flexkv::RedisMetaChannel::renew_node_leases), py::arg("node_id"), py::arg("new_lt"), py::arg("batch_size")=200)
      .def("renew_node_leases_with_hashes", [](flexkv::RedisMetaChannel &ch, uint32_t node_id, uint64_t new_lt, const std::vector<int64_t> &hashes, size_t batch_size){ std::list<int64_t> l(hashes.begin(), hashes.end()); return ch.renew_node_leases(node_id, new_lt, l, batch_size); }, py::arg("node_id"), py::arg("new_lt"), py::arg("hashes"), py::arg("batch_size")=200)
      .def("list_keys", [](flexkv::RedisMetaChannel &ch, const std::string &pattern){ std::vector<std::string> keys; ch.list_keys(pattern, keys); return keys; }, py::arg("pattern"))
      .def("list_node_keys", [](flexkv::RedisMetaChannel &ch){ std::vector<std::string> keys; ch.list_node_keys(keys); return keys; })
      .def("list_block_keys", [](flexkv::RedisMetaChannel &ch, uint32_t node_id){ std::vector<std::string> keys; ch.list_block_keys(node_id, keys); return keys; }, py::arg("node_id"))
      .def("hmget_field_for_keys", [](flexkv::RedisMetaChannel &ch, const std::vector<std::string> &keys, const std::string &field){ std::vector<std::string> values; ch.hmget_field_for_keys(keys, field, values); return values; }, py::arg("keys"), py::arg("field"))
      .def("hmget_two_fields_for_keys", [](flexkv::RedisMetaChannel &ch, const std::vector<std::string> &keys, const std::string &f1, const std::string &f2){ std::vector<std::pair<std::string,std::string>> out; ch.hmget_two_fields_for_keys(keys, f1, f2, out); return out; }, py::arg("keys"), py::arg("field1"), py::arg("field2"))
      .def("load_metas_by_keys", [](flexkv::RedisMetaChannel &ch, const std::vector<std::string> &keys){ std::vector<flexkv::BlockMeta> out; ch.load_metas_by_keys(keys, out); return out; }, py::arg("keys"))
      .def("update_block_state_batch", [](flexkv::RedisMetaChannel &ch, uint32_t node_id, const std::vector<int64_t> &hashes, int state, size_t batch_size){ std::deque<int64_t> dq(hashes.begin(), hashes.end()); return ch.update_block_state_batch(node_id, &dq, state, batch_size); }, py::arg("node_id"), py::arg("hashes"), py::arg("state"), py::arg("batch_size")=200)
      .def("delete_blockmeta_batch", [](flexkv::RedisMetaChannel &ch, uint32_t node_id, const std::vector<int64_t> &hashes, size_t batch_size){ std::deque<int64_t> dq(hashes.begin(), hashes.end()); return ch.delete_blockmeta_batch(node_id, &dq, batch_size); }, py::arg("node_id"), py::arg("hashes"), py::arg("batch_size")=200);

  // LocalRadixTree bindings (derived from CRadixTreeIndex)
  py::class_<flexkv::LocalRadixTree, flexkv::CRadixTreeIndex>(m, "LocalRadixTree")
      .def(py::init<int, unsigned int, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>(),
           py::arg("tokens_per_block"),
           py::arg("max_num_blocks") = 1000000u,
           py::arg("lease_ttl_ms") = 100000,
           py::arg("renew_lease_ms") = 0,
           py::arg("refresh_batch_size") = 256,
           py::arg("idle_sleep_ms") = 10,
           py::arg("safety_ttl_ms") = 100,
           py::arg("swap_block_threshold") = 1024,
           py::arg("hit_reward_seconds") = 0)
      .def("set_meta_channel", &flexkv::LocalRadixTree::set_meta_channel, py::arg("channel"))
      .def("start", &flexkv::LocalRadixTree::start, py::arg("channel"))
      .def("stop", &flexkv::LocalRadixTree::stop)
      .def("insert_and_publish", &flexkv::LocalRadixTree::insert_and_publish, py::arg("node"))
      // Mirror CRadixTreeIndex APIs explicitly on LocalRadixTree
      .def("match_prefix", &flexkv::LocalRadixTree::match_prefix,
           py::arg("block_hashes"), py::arg("num_blocks"), py::arg("update_cache_info") = true,
           py::call_guard<py::gil_scoped_release>())
      .def("total_unready_blocks", &flexkv::LocalRadixTree::total_unready_blocks)
      .def("total_ready_blocks", &flexkv::LocalRadixTree::total_ready_blocks)
      .def("total_cached_blocks", &flexkv::LocalRadixTree::total_cached_blocks)
      .def("total_node_num", &flexkv::LocalRadixTree::total_node_num)
      .def("reset", &flexkv::LocalRadixTree::reset)
      .def("is_root", &flexkv::LocalRadixTree::is_root, py::arg("node"))
      .def("remove_node", &flexkv::LocalRadixTree::remove_node, py::arg("node"))
      .def("remove_leaf", &flexkv::LocalRadixTree::remove_leaf, py::arg("node"))
      .def("add_node", &flexkv::LocalRadixTree::add_node, py::arg("node"))
      .def("add_leaf", &flexkv::LocalRadixTree::add_leaf, py::arg("node"))
      .def("lock", &flexkv::LocalRadixTree::lock, py::arg("node"))
      .def("unlock", &flexkv::LocalRadixTree::unlock, py::arg("node"))
      .def("is_empty", &flexkv::LocalRadixTree::is_empty)
      .def("inc_node_count", &flexkv::LocalRadixTree::inc_node_count)
      .def("dec_node_count", &flexkv::LocalRadixTree::dec_node_count)
      .def("set_ready", &flexkv::LocalRadixTree::set_ready, py::arg("node"), py::arg("ready"), py::arg("ready_length") = -1);
      
      // Utilities
      m.attr("LocalRadixTree").cast<py::class_<flexkv::LocalRadixTree, flexkv::CRadixTreeIndex>>()
        .def("drain_pending_queues", &flexkv::LocalRadixTree::drain_pending_queues);

  // DistributedRadixTree bindings (remote reference tree manager)
  py::class_<flexkv::DistributedRadixTree>(m, "DistributedRadixTree")
      .def(py::init<int, unsigned int, uint32_t, size_t, uint32_t, uint32_t, uint32_t, uint32_t>(),
           py::arg("tokens_per_block"),
           py::arg("max_num_blocks"),
           py::arg("node_id"),
           py::arg("refresh_batch_size") = 128,
           py::arg("rebuild_interval_ms") = 1000,
           py::arg("idle_sleep_ms") = 10,
           py::arg("lease_renew_ms") = 5000,
           py::arg("hit_reward_seconds") = 0)
      .def("start", &flexkv::DistributedRadixTree::start, py::arg("channel"))
      .def("stop", &flexkv::DistributedRadixTree::stop)
      .def("remote_tree_refresh", &flexkv::DistributedRadixTree::remote_tree_refresh, py::return_value_policy::reference)
      .def("match_prefix", &flexkv::DistributedRadixTree::match_prefix,
           py::arg("block_hashes"), py::arg("num_blocks"), py::arg("update_cache_info") = true,
           py::call_guard<py::gil_scoped_release>())
      .def("lock", &flexkv::DistributedRadixTree::lock, py::arg("node"))
      .def("unlock", &flexkv::DistributedRadixTree::unlock, py::arg("node"))
      .def("is_empty", &flexkv::DistributedRadixTree::is_empty)
      .def("set_ready", &flexkv::DistributedRadixTree::set_ready, py::arg("node"), py::arg("ready") = true, py::arg("ready_length") = -1);

  // RefRadixTree bindings (for type information)
  py::class_<flexkv::RefRadixTree, flexkv::CRadixTreeIndex>(m, "RefRadixTree")
      .def(py::init<int, unsigned int, uint32_t, uint32_t, flexkv::LockFreeQueue<flexkv::CRadixNode*>*, flexkv::LeaseMetaMemPool*>(),
           py::arg("tokens_per_block"),
           py::arg("max_num_blocks") = 1000000u,
           py::arg("lease_renew_ms") = 5000,
           py::arg("hit_reward_seconds") = 0,
           py::arg("renew_lease_queue") = nullptr,
           py::arg("lt_pool") = nullptr)
      .def("dec_ref_cnt", &flexkv::RefRadixTree::dec_ref_cnt)
      .def("inc_ref_cnt", &flexkv::RefRadixTree::inc_ref_cnt);

  py::class_<flexkv::FileExtent>(m, "FileExtent")
      .def(py::init<>())
      .def_readonly("logical", &flexkv::FileExtent::logical)
      .def_readonly("physical", &flexkv::FileExtent::physical)
      .def_readonly("length", &flexkv::FileExtent::length);
}
