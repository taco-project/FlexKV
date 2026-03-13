/*
 * MUSA bindings for FlexKV — mirrors csrc/bindings.cpp for CUDA.
 *
 * When built with MUSA SDK (FLEXKV_HAS_MUSA_SDK defined), calls the real
 * MUSA transfer kernel via transfer_musa.muh and exposes TPTransferThreadGroupMusa.
 * Without the SDK, provides a C++-only stub so the extension can still be
 * imported for testing the dispatch/build path.
 *
 * SSD transfer (io_uring/pread/pwrite) and CFS remote transfer are pure CPU
 * operations shared with the CUDA extension — they are included here so that
 * a MUSA-only build can work end-to-end.
 *
 * GDS uses GDSManagerMusa (muFile) when FLEXKV_ENABLE_GDS and MUSA SDK are
 * both available.
 */
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>
#include <map>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "../transfer_ssd.h"
#include "../cache_utils.h"
#include "../radix_tree.h"

#ifdef FLEXKV_ENABLE_CFS
#include "../pcfs/pcfs.h"
#endif

#ifdef FLEXKV_HAS_MUSA_SDK
#include <musa_runtime.h>
#include "transfer_musa.muh"
#include "tp_transfer_thread_group_musa.h"

#ifdef FLEXKV_ENABLE_GDS
#include "gds/gds_manager_musa.h"
#include "gds/tp_gds_transfer_thread_group_musa.h"
#endif

#if __has_include(<torch_musa/csrc/core/MUSAStream.h>)
#include <torch_musa/csrc/core/MUSAStream.h>
static musaStream_t get_current_musa_stream() {
    return c10::musa::getCurrentMUSAStream(-1).stream();
}
#else
static musaStream_t get_current_musa_stream() {
    return (musaStream_t)0;  /* default stream when torch_musa headers not available */
}
#endif
#else
#include "gtensor_handler_musa.h"
#endif

namespace py = pybind11;

// ---------------------------------------------------------------------------
// GPU-CPU transfer (MUSA kernel or stub)
// ---------------------------------------------------------------------------

void transfer_kv_blocks_binding_musa(
    torch::Tensor& gpu_block_id_tensor,
    torch::Tensor& gpu_tensor_ptrs_tensor,
    int64_t gpu_kv_stride_in_bytes,
    int64_t gpu_block_stride_in_bytes,
    int64_t gpu_layer_stride_in_bytes,
    torch::Tensor& cpu_block_id_tensor,
    torch::Tensor& cpu_tensor,
    int64_t cpu_kv_stride_in_bytes,
    int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_block_stride_in_bytes,
    int64_t chunk_size_in_bytes,
    int start_layer_id,
    int num_layers,
    int transfer_sms,
    bool is_host_to_device,
    bool use_ce_transfer,
    bool is_mla,
    int gpu_block_type) {

#ifdef FLEXKV_HAS_MUSA_SDK
    int num_blocks = gpu_block_id_tensor.numel();
    int64_t* gpu_block_ids = static_cast<int64_t*>(gpu_block_id_tensor.data_ptr());
    void** gpu_tensor_ptrs = static_cast<void**>(gpu_tensor_ptrs_tensor.data_ptr());
    int64_t* cpu_block_ids = static_cast<int64_t*>(cpu_block_id_tensor.data_ptr());
    void* cpu_ptr = static_cast<void*>(cpu_tensor.data_ptr());

    musaStream_t stream = get_current_musa_stream();

    flexkv::BackendType backend_type;
    if (gpu_block_type == 0)
        backend_type = flexkv::BackendType::VLLM;
    else if (gpu_block_type == 1)
        backend_type = flexkv::BackendType::TRTLLM;
    else if (gpu_block_type == 2)
        backend_type = flexkv::BackendType::SGLANG;
    else
        throw std::runtime_error("Unsupported gpu_block_type: " + std::to_string(gpu_block_type));

    flexkv::GTensorHandler handler(
        backend_type,
        reinterpret_cast<int64_t**>(gpu_tensor_ptrs),
        num_layers,
        gpu_kv_stride_in_bytes,
        gpu_block_stride_in_bytes,
        gpu_layer_stride_in_bytes);

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

    musaError_t err = musaGetLastError();
    if (err != musaSuccess)
        throw std::runtime_error(std::string("MUSA transfer error: ") + musaGetErrorString(err));
#else
    (void)gpu_block_id_tensor; (void)gpu_tensor_ptrs_tensor;
    (void)gpu_kv_stride_in_bytes; (void)gpu_block_stride_in_bytes;
    (void)gpu_layer_stride_in_bytes; (void)cpu_block_id_tensor;
    (void)cpu_tensor; (void)cpu_kv_stride_in_bytes;
    (void)cpu_layer_stride_in_bytes; (void)cpu_block_stride_in_bytes;
    (void)chunk_size_in_bytes; (void)start_layer_id;
    (void)num_layers; (void)transfer_sms;
    (void)is_host_to_device; (void)use_ce_transfer;
    (void)is_mla; (void)gpu_block_type;
#endif
}

// ---------------------------------------------------------------------------
// SSD transfer binding (io_uring / pread-pwrite — pure CPU, shared with CUDA)
// ---------------------------------------------------------------------------

void transfer_kv_blocks_ssd_binding_musa(
    flexkv::SSDIOCTX& ioctx,
    const torch::Tensor& cpu_layer_id_list, int64_t cpu_tensor_ptr,
    const torch::Tensor& ssd_block_ids, const torch::Tensor& cpu_block_ids,
    int64_t cpu_layer_stride_in_bytes, int64_t cpu_kv_stride_in_bytes,
    int64_t ssd_layer_stride_in_bytes, int64_t ssd_kv_stride_in_bytes,
    int64_t chunk_size_in_bytes, int64_t block_stride_in_bytes,
    bool is_read, int num_blocks_per_file, int round_robin = 1,
    int num_threads_per_device = 16, bool is_mla = false) {

    TORCH_CHECK(ssd_block_ids.dtype() == torch::kInt64, "ssd_block_ids must be int64");
    TORCH_CHECK(cpu_block_ids.dtype() == torch::kInt64, "cpu_block_ids must be int64");

    flexkv::transfer_kv_blocks_ssd(
        ioctx, cpu_layer_id_list, cpu_tensor_ptr, ssd_block_ids,
        cpu_block_ids, cpu_layer_stride_in_bytes, cpu_kv_stride_in_bytes,
        ssd_layer_stride_in_bytes, ssd_kv_stride_in_bytes, chunk_size_in_bytes,
        block_stride_in_bytes, is_read, num_blocks_per_file, round_robin,
        num_threads_per_device, is_mla);
}

// ---------------------------------------------------------------------------
// CFS remote transfer binding (pure CPU, shared with CUDA)
// ---------------------------------------------------------------------------

#ifdef FLEXKV_ENABLE_CFS
void transfer_kv_blocks_remote_musa(
    const py::list& file_nodeid_list, const torch::Tensor& cpu_layer_id_list,
    int64_t cpu_tensor_ptr, const torch::Tensor& remote_block_ids,
    const torch::Tensor& cpu_block_ids, int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes, int64_t remote_layer_stride_in_bytes,
    int64_t remote_block_stride_in_bytes, int64_t remote_kv_stride_in_bytes,
    int64_t block_size_in_bytes, int64_t total_layers, bool is_read,
    int partition_block_type, int round_robin,
    int64_t num_remote_blocks_per_file, bool use_mmap = false,
    int num_threads_per_file = 8, bool is_mla = false) {

    TORCH_CHECK(remote_block_ids.dtype() == torch::kInt64, "remote_block_ids must be int64");
    TORCH_CHECK(cpu_block_ids.dtype() == torch::kInt64, "cpu_block_ids must be int64");

    std::vector<std::uint64_t> file_nodeids;
    for (const auto& file_nodeid : file_nodeid_list) {
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
#endif

// ---------------------------------------------------------------------------
// GDS MUSA transfer binding (muFile — requires MUSA SDK)
// ---------------------------------------------------------------------------

#if defined(FLEXKV_HAS_MUSA_SDK) && defined(FLEXKV_ENABLE_GDS)
void transfer_kv_blocks_gds_binding_musa(
    GDSManagerMusa& gds_manager,
    const torch::Tensor& gpu_layer_id_list,
    const torch::Tensor& gpu_layer_ptrs_tensor,
    const torch::Tensor& ssd_block_ids,
    const torch::Tensor& gpu_block_ids,
    int64_t gpu_kv_stride_in_bytes,
    int64_t gpu_block_stride_in_bytes,
    int64_t gpu_layer_stride_in_bytes,
    int64_t ssd_layer_stride_in_bytes,
    int64_t ssd_block_stride_in_bytes,
    int64_t ssd_kv_stride_in_bytes,
    int64_t block_size_in_bytes,
    int64_t ssd_copy_off_inside_chunks,
    int num_blocks_per_file,
    int64_t total_layers,
    bool is_read,
    bool verbose = false,
    bool is_mla = false,
    int gpu_block_type = 0,
    int gpu_device_id = 0) {

    TORCH_CHECK(gpu_layer_ptrs_tensor.dtype() == torch::kInt64, "gpu_layer_ptrs must be int64");
    TORCH_CHECK(ssd_block_ids.dtype() == torch::kInt64, "ssd_block_ids must be int64");
    TORCH_CHECK(gpu_block_ids.dtype() == torch::kInt64, "gpu_block_ids must be int64");
    TORCH_CHECK(gpu_layer_id_list.dtype() == torch::kInt32, "gpu_layer_id_list must be int32");

    flexkv::BackendType backend_type;
    if (gpu_block_type == 0) backend_type = flexkv::BackendType::VLLM;
    else if (gpu_block_type == 1) backend_type = flexkv::BackendType::TRTLLM;
    else if (gpu_block_type == 2) backend_type = flexkv::BackendType::SGLANG;
    else throw std::runtime_error("Unsupported gpu_block_type: " + std::to_string(gpu_block_type));

    void** gpu_tensor_ptrs = static_cast<void**>(gpu_layer_ptrs_tensor.data_ptr());
    flexkv::GTensorHandler handler(
        backend_type,
        reinterpret_cast<int64_t**>(gpu_tensor_ptrs),
        total_layers,
        gpu_kv_stride_in_bytes,
        gpu_block_stride_in_bytes,
        gpu_layer_stride_in_bytes);

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
#endif

// ===========================================================================
// Module definition
// ===========================================================================

PYBIND11_MODULE(c_ext_musa, m) {
    // --- GPU-CPU transfer ---
    m.def("transfer_kv_blocks", &transfer_kv_blocks_binding_musa,
          "Transfer multi-layer KV-cache between CPU and GPU (MUSA)",
          py::arg("gpu_block_id_tensor"), py::arg("gpu_tensor_ptrs_tensor"),
          py::arg("gpu_kv_stride_in_bytes"), py::arg("gpu_block_stride_in_bytes"),
          py::arg("gpu_layer_stride_in_bytes"), py::arg("cpu_block_id_tensor"),
          py::arg("cpu_tensor"), py::arg("cpu_kv_stride_in_bytes"),
          py::arg("cpu_layer_stride_in_bytes"), py::arg("cpu_block_stride_in_bytes"),
          py::arg("chunk_size_in_bytes"), py::arg("start_layer_id"),
          py::arg("num_layers"), py::arg("transfer_sms") = -1,
          py::arg("is_host_to_device") = true, py::arg("use_ce_transfer") = false,
          py::arg("is_mla") = false, py::arg("gpu_block_type") = 0);

    // --- SSD transfer (io_uring / pread-pwrite) ---
    m.def("transfer_kv_blocks_ssd", &transfer_kv_blocks_ssd_binding_musa,
          "Transfer KV blocks between SSD and CPU memory (shared with CUDA)",
          py::arg("ioctx"), py::arg("cpu_layer_id_list"),
          py::arg("cpu_tensor_ptr"), py::arg("ssd_block_ids"),
          py::arg("cpu_block_ids"), py::arg("cpu_layer_stride_in_bytes"),
          py::arg("cpu_kv_stride_in_bytes"), py::arg("ssd_layer_stride_in_bytes"),
          py::arg("ssd_kv_stride_in_bytes"), py::arg("chunk_size_in_bytes"),
          py::arg("block_stride_in_bytes"), py::arg("is_read"),
          py::arg("num_blocks_per_file"), py::arg("round_robin") = 1,
          py::arg("num_threads_per_device") = 16, py::arg("is_mla") = false);

    // --- SSD I/O context ---
    py::class_<flexkv::SSDIOCTX>(m, "SSDIOCTX")
        .def(py::init<std::map<int, std::vector<std::string>>&, int, int, int>());

    // --- Hash utilities ---
    m.def("get_hash_size", &flexkv::get_hash_size, "Get the size of the hash result");
    m.def("gen_hashes", &flexkv::gen_hashes, "Generate hashes for a tensor",
          py::arg("hasher"), py::arg("token_ids"), py::arg("tokens_per_block"),
          py::arg("block_hashes"));

    py::class_<flexkv::Hasher>(m, "Hasher")
        .def(py::init<>())
        .def("reset", &flexkv::Hasher::reset)
        .def("update",
             py::overload_cast<const torch::Tensor&>(&flexkv::Hasher::update),
             "Update the hasher with a tensor", py::arg("input"))
        .def("update",
             py::overload_cast<const void*, size_t>(&flexkv::Hasher::update),
             "Update the hasher with pointer and size", py::arg("input"),
             py::arg("size"))
        .def("digest", &flexkv::Hasher::digest, "Return the hash value");

    // --- Radix tree index (mirrors CUDA bindings) ---
    py::class_<flexkv::CRadixTreeIndex>(m, "CRadixTreeIndex")
        .def(py::init([](int tokens_per_block, unsigned int max_num_blocks, int hit_reward_seconds, std::string eviction_policy) {
             auto policy = flexkv::parse_eviction_policy(eviction_policy);
             return new flexkv::CRadixTreeIndex(tokens_per_block, max_num_blocks, hit_reward_seconds, policy);
        }),
             py::arg("tokens_per_block"),
             py::arg("max_num_blocks") = 1000000,
             py::arg("hit_reward_seconds") = 0,
             py::arg("eviction_policy") = "lru")
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
        .def("evict",
             py::overload_cast<torch::Tensor &, int>(&flexkv::CRadixTreeIndex::evict),
             py::arg("evicted_blocks"), py::arg("num_evicted"),
             py::call_guard<py::gil_scoped_release>())
        .def("evict",
             py::overload_cast<torch::Tensor &, torch::Tensor &, int>(&flexkv::CRadixTreeIndex::evict),
             py::arg("evicted_blocks"), py::arg("evicted_block_hashes"), py::arg("num_evicted"),
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
        .def("has_block_node_ids", &flexkv::CRadixNode::has_block_node_ids)
        .def_property_readonly("parent", &flexkv::CRadixNode::get_parent, py::return_value_policy::reference);

    py::class_<flexkv::CMatchResult, std::shared_ptr<flexkv::CMatchResult>>(m, "CMatchResult")
        .def(py::init<int, int, int, flexkv::CRadixNode *, flexkv::CRadixNode *, torch::Tensor, torch::Tensor>())
        .def_readonly("last_ready_node", &flexkv::CMatchResult::last_ready_node)
        .def_readonly("last_node", &flexkv::CMatchResult::last_node)
        .def_readonly("physical_blocks", &flexkv::CMatchResult::physical_blocks)
        .def_readonly("block_node_ids", &flexkv::CMatchResult::block_node_ids)
        .def_readonly("num_ready_matched_blocks", &flexkv::CMatchResult::num_ready_matched_blocks)
        .def_readonly("num_matched_blocks", &flexkv::CMatchResult::num_matched_blocks)
        .def_readonly("last_node_matched_length", &flexkv::CMatchResult::last_node_matched_length);

    // --- CFS remote transfer ---
#ifdef FLEXKV_ENABLE_CFS
    m.def("transfer_kv_blocks_remote", &transfer_kv_blocks_remote_musa,
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

    py::class_<flexkv::Pcfs>(m, "Pcfs")
        .def(py::init<const std::string&, uint32_t, const std::string&, bool,
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

    m.def("set_pcfs_instance", &flexkv::set_pcfs_instance,
          "Set the global Pcfs instance from a pointer", py::arg("pcfs"));
    m.def("call_pcfs_read", &flexkv::call_pcfs_read, "Call Pcfs::read from C++",
          py::arg("file_nodeid"), py::arg("offset"), py::arg("buffer"),
          py::arg("size"), py::arg("thread_id"));
    m.def("call_pcfs_write", &flexkv::call_pcfs_write, "Call Pcfs::write from C++",
          py::arg("file_nodeid"), py::arg("offset"), py::arg("buffer"),
          py::arg("size"), py::arg("thread_id"));
#endif

    // --- GPU-CPU TP transfer ---
#ifdef FLEXKV_HAS_MUSA_SDK
    m.attr("HAS_MUSA_SDK") = true;

    py::class_<flexkv::TPTransferThreadGroupMusa>(m, "TPTransferThreadGroupMusa")
        .def(py::init<int, const std::vector<std::vector<torch::Tensor>>&,
                      torch::Tensor&, int, int,
                      torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&>(),
             py::arg("num_gpus"), py::arg("gpu_blocks"), py::arg("cpu_blocks"),
             py::arg("dp_group_id"), py::arg("num_layers"),
             py::arg("gpu_kv_strides_tensor"), py::arg("gpu_block_strides_tensor"),
             py::arg("gpu_layer_strides_tensor"), py::arg("gpu_chunk_sizes_tensor"))
        .def("tp_group_transfer",
             &flexkv::TPTransferThreadGroupMusa::tp_group_transfer,
             py::arg("gpu_block_id_tensor"), py::arg("cpu_block_id_tensor"),
             py::arg("cpu_kv_stride_in_bytes"),
             py::arg("cpu_layer_stride_in_bytes"),
             py::arg("cpu_block_stride_in_bytes"),
             py::arg("cpu_tp_stride_in_bytes"), py::arg("transfer_sms"),
             py::arg("is_host_to_device"), py::arg("use_ce_transfer"),
             py::arg("layer_id"), py::arg("layer_granularity"),
             py::arg("is_mla"));

    // --- GDS (muFile) ---
#ifdef FLEXKV_ENABLE_GDS
    m.def("transfer_kv_blocks_gds", &transfer_kv_blocks_gds_binding_musa,
          "Transfer KV blocks between GPU and GDS storage (MUSA/muFile)",
          py::arg("gds_manager"), py::arg("gpu_layer_id_list"),
          py::arg("gpu_layer_ptrs_tensor"), py::arg("ssd_block_ids"),
          py::arg("gpu_block_ids"), py::arg("gpu_kv_stride_in_bytes"),
          py::arg("gpu_block_stride_in_bytes"), py::arg("gpu_layer_stride_in_bytes"),
          py::arg("ssd_layer_stride_in_bytes"), py::arg("ssd_block_stride_in_bytes"),
          py::arg("ssd_kv_stride_in_bytes"), py::arg("block_size_in_bytes"),
          py::arg("ssd_copy_off_inside_chunks"), py::arg("num_blocks_per_file"),
          py::arg("total_layers"), py::arg("is_read"),
          py::arg("verbose") = false, py::arg("is_mla") = false,
          py::arg("gpu_block_type") = 0, py::arg("gpu_device_id") = 0);

    py::class_<GDSManagerMusa>(m, "GDSManagerMusa")
        .def(py::init<std::map<int, std::vector<std::string>>&, int, int>(),
             py::arg("ssd_files"), py::arg("num_devices"), py::arg("round_robin") = 1)
        .def("is_ready", &GDSManagerMusa::is_ready)
        .def("synchronize", &GDSManagerMusa::synchronize)
        .def("get_last_error", &GDSManagerMusa::get_last_error);

    py::class_<flexkv::TPGDSTransferThreadGroupMusa>(m, "TPGDSTransferThreadGroupMusa")
        .def(py::init<int, const std::vector<std::vector<torch::Tensor>>&,
                      std::map<int, std::vector<std::string>>&, int, int,
                      torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&>(),
             py::arg("num_gpus"), py::arg("gpu_blocks"), py::arg("ssd_files"),
             py::arg("dp_group_id"), py::arg("num_layers"),
             py::arg("gpu_kv_strides_tensor"), py::arg("gpu_block_strides_tensor"),
             py::arg("gpu_layer_strides_tensor"), py::arg("gpu_chunk_sizes_tensor"))
        .def("tp_group_transfer",
             &flexkv::TPGDSTransferThreadGroupMusa::tp_group_transfer,
             py::arg("gpu_block_id_tensor"), py::arg("ssd_block_id_tensor"),
             py::arg("ssd_layer_stride_in_bytes"),
             py::arg("ssd_kv_stride_in_bytes"), py::arg("ssd_block_stride_in_bytes"),
             py::arg("ssd_tp_stride_in_bytes"), py::arg("num_blocks_per_file"),
             py::arg("is_read"), py::arg("layer_id"), py::arg("layer_granularity"),
             py::arg("is_mla"));
#endif  // FLEXKV_ENABLE_GDS

#else
    m.attr("HAS_MUSA_SDK") = false;
#endif  // FLEXKV_HAS_MUSA_SDK
}

