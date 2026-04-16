/*
 * csrc/bindings.cpp  —  FlexKV unified C extension entry point
 *
 * This file exposes a single Python extension module "flexkv.c_ext" regardless
 * of the GPU backend selected at build time.
 *
 * Layout
 * ──────
 * ① Platform-independent includes (SSD IO, CFS, P2P, radix tree, hasher, …)
 * ② GPU-vendor-specific includes via a single backend selector block:
 *       FLEXKV_BACKEND_MUSA  → gpu_backend/musa/gpu_transfer_bindings.h
 *       (default / NVIDIA)   → gpu_backend/nvidia/gpu_transfer_bindings.h
 *    Each header defines:
 *       static void transfer_kv_blocks_binding(...)
 *       static void transfer_kv_blocks_gds_binding(...)   [if GDS enabled]
 *       #define REGISTER_GPU_TRANSFER_BINDINGS(m)         [pybind11 registrations]
 * ③ PYBIND11_MODULE(c_ext, m)  — single module name for all backends.
 *    All GPU-specific bindings are injected via REGISTER_GPU_TRANSFER_BINDINGS(m).
 *    All platform-independent bindings are registered inline below.
 *
 * Adding a new GPU vendor
 * ───────────────────────
 * 1. Create csrc/gpu_backend/<vendor>/gpu_transfer_bindings.h following the
 *    same interface (transfer_kv_blocks_binding + REGISTER_GPU_TRANSFER_BINDINGS).
 * 2. Add an #elif branch in section ② below.
 * 3. No other changes to this file are needed.
 */

// ── ① Platform-independent headers ──────────────────────────────────────────
#include <cstddef>
#include <cstdint>
#include <deque>
#include <fcntl.h>
#include <map>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include "cache_utils.h"
#include "monitoring/metrics_manager.h"
#include "radix_tree.h"
#include "transfer_ssd.h"

#ifdef FLEXKV_ENABLE_CFS
#include "pcfs/pcfs.h"
#endif

#ifdef FLEXKV_ENABLE_P2P
#include "dist/block_meta.h"
#include "dist/distributed_radix_tree.h"
#include "dist/lease_meta_mempool.h"
#include "dist/local_radix_tree.h"
#include "dist/lock_free_q.h"
#include "dist/redis_meta_channel.h"
#endif

// ── ② GPU-vendor backend selector ───────────────────────────────────────────
//
// Each backend header must define:
//   static void transfer_kv_blocks_binding(...)
//   #define REGISTER_GPU_TRANSFER_BINDINGS(m)
//
// Add new vendors here with additional #elif blocks.
//
#if defined(FLEXKV_BACKEND_MUSA)
#  include "gpu_backend/musa/gpu_transfer_bindings.h"
#else
// Default: NVIDIA CUDA
#  include "gpu_backend/nvidia/gpu_transfer_bindings.h"
#endif

// ── CFS remote transfer helpers (platform-independent, referenced below) ────
namespace py = pybind11;

#ifdef FLEXKV_ENABLE_CFS
static void transfer_kv_blocks_remote(
    const py::list &file_nodeid_list, const torch::Tensor &cpu_layer_id_list,
    int64_t cpu_tensor_ptr, const torch::Tensor &remote_block_ids,
    const torch::Tensor &cpu_block_ids, int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes, int64_t remote_layer_stride_in_bytes,
    int64_t remote_block_stride_in_bytes, int64_t remote_kv_stride_in_bytes,
    int64_t block_size_in_bytes, int64_t total_layers, bool is_read,
    int partition_block_type, int round_robin,
    int64_t num_remote_blocks_per_file, bool use_mmap = false,
    int num_threads_per_file = 8, bool is_mla = false) {
  TORCH_CHECK(remote_block_ids.dtype() == torch::kInt64, "remote_block_ids must be int64");
  TORCH_CHECK(cpu_block_ids.dtype()    == torch::kInt64, "cpu_block_ids must be int64");
  std::vector<std::uint64_t> file_nodeids;
  for (const auto &fid : file_nodeid_list)
    file_nodeids.push_back(fid.cast<std::uint64_t>());
  flexkv::transfer_kv_blocks_cfs_mmap_multi_thread(
      file_nodeids, cpu_layer_id_list, cpu_tensor_ptr, remote_block_ids,
      cpu_block_ids, cpu_layer_stride_in_bytes, cpu_kv_stride_in_bytes,
      remote_layer_stride_in_bytes, remote_block_stride_in_bytes,
      remote_kv_stride_in_bytes, block_size_in_bytes, total_layers, is_read,
      partition_block_type, round_robin, num_remote_blocks_per_file, use_mmap,
      num_threads_per_file, is_mla);
}

static void shared_transfer_kv_blocks_remote_read_binding(
    const py::list &file_nodeid_list,
    const py::list &cfs_blocks_partition_list,
    const py::list &cpu_blocks_partition_list,
    const torch::Tensor &cpu_layer_id_list, int64_t cpu_tensor_ptr,
    int64_t cpu_layer_stride_in_bytes, int64_t cpu_kv_stride_in_bytes,
    int64_t cfs_layer_stride_in_bytes, int64_t cfs_block_stride_in_bytes,
    int64_t cfs_kv_stride_in_bytes, int64_t block_size_in_bytes,
    int64_t total_layers, bool is_mla = false, int num_threads_per_file = 8) {
  std::vector<std::uint64_t> file_nodeids;
  for (const auto &fid : file_nodeid_list)
    file_nodeids.push_back(fid.cast<std::uint64_t>());
  std::vector<std::vector<int64_t>> cfs_blocks_partition;
  for (const auto &bl : cfs_blocks_partition_list) {
    std::vector<int64_t> v;
    for (const auto &b : bl) v.push_back(b.cast<int64_t>());
    cfs_blocks_partition.push_back(std::move(v));
  }
  std::vector<std::vector<int64_t>> cpu_blocks_partition;
  for (const auto &bl : cpu_blocks_partition_list) {
    std::vector<int64_t> v;
    for (const auto &b : bl) v.push_back(b.cast<int64_t>());
    cpu_blocks_partition.push_back(std::move(v));
  }
  flexkv::shared_transfer_kv_blocks_remote_read(
      file_nodeids, cfs_blocks_partition, cpu_blocks_partition,
      cpu_layer_id_list, cpu_tensor_ptr, cpu_layer_stride_in_bytes,
      cpu_kv_stride_in_bytes, cfs_layer_stride_in_bytes,
      cfs_block_stride_in_bytes, cfs_kv_stride_in_bytes, block_size_in_bytes,
      total_layers, is_mla, num_threads_per_file);
}
#endif // FLEXKV_ENABLE_CFS

static void transfer_kv_blocks_ssd_binding(
    flexkv::SSDIOCTX &ioctx, const torch::Tensor &cpu_layer_id_list,
    int64_t cpu_tensor_ptr, const torch::Tensor &ssd_block_ids,
    const torch::Tensor &cpu_block_ids, int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes, int64_t ssd_layer_stride_in_bytes,
    int64_t ssd_kv_stride_in_bytes, int64_t chunk_size_in_bytes,
    int64_t block_stride_in_bytes, bool is_read, int num_blocks_per_file,
    int round_robin = 1, int num_threads_per_device = 8, bool is_mla = false) {
  TORCH_CHECK(ssd_block_ids.dtype() == torch::kInt64, "ssd_block_ids must be int64");
  TORCH_CHECK(cpu_block_ids.dtype() == torch::kInt64, "cpu_block_ids must be int64");
  flexkv::transfer_kv_blocks_ssd(
      ioctx, cpu_layer_id_list, cpu_tensor_ptr, ssd_block_ids, cpu_block_ids,
      cpu_layer_stride_in_bytes, cpu_kv_stride_in_bytes,
      ssd_layer_stride_in_bytes, ssd_kv_stride_in_bytes, chunk_size_in_bytes,
      block_stride_in_bytes, is_read, num_blocks_per_file, round_robin,
      num_threads_per_device, is_mla);
}

// ── ③ Module definition ──────────────────────────────────────────────────────

PYBIND11_MODULE(c_ext, m) {
  // ── Metrics ──────────────────────────────────────────────────────────────
  m.def("configure_cpp_metrics",
        [](bool enabled, int port) {
          flexkv::monitoring::MetricsManager::Instance().Configure(enabled, port);
        },
        "Configure C++ metrics from Python",
        py::arg("enabled"), py::arg("port"));

  // ── GPU-vendor-specific bindings (transfer_kv_blocks, GDS, TP groups) ───
  REGISTER_GPU_TRANSFER_BINDINGS(m);

  // ── SSD transfer (io_uring / pread-pwrite) ───────────────────────────────
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

  // ── SSDIOCTX ─────────────────────────────────────────────────────────────
  py::class_<flexkv::SSDIOCTX>(m, "SSDIOCTX")
      .def(py::init<std::map<int, std::vector<std::string>> &, int, int, int>());

  // ── Hash utilities ────────────────────────────────────────────────────────
  m.def("get_hash_size", &flexkv::get_hash_size, "Get the size of the hash result");
  m.def("gen_hashes", &flexkv::gen_hashes, "Generate hashes for a tensor",
        py::arg("hasher"), py::arg("token_ids"), py::arg("tokens_per_block"),
        py::arg("block_hashes"));

  py::class_<flexkv::Hasher>(m, "Hasher")
      .def(py::init<>())
      .def("reset", &flexkv::Hasher::reset)
      .def("update",
           py::overload_cast<const torch::Tensor &>(&flexkv::Hasher::update),
           py::arg("input"))
      .def("update",
           py::overload_cast<const void *, size_t>(&flexkv::Hasher::update),
           py::arg("input"), py::arg("size"))
      .def("digest", &flexkv::Hasher::digest);

  // ── CFS remote transfer ───────────────────────────────────────────────────
#ifdef FLEXKV_ENABLE_CFS
  m.def("transfer_kv_blocks_remote", &transfer_kv_blocks_remote,
        "Transfer KV blocks between remote CFS and CPU memory",
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
      .def(py::init<const std::string &, uint32_t, const std::string &, bool,
                    const uint64_t>())
      .def("init",     &flexkv::Pcfs::init)
      .def("destroy",  &flexkv::Pcfs::destroy)
      .def("lookup_or_create_file", &flexkv::Pcfs::lookup_or_create_file,
           py::arg("filename"), py::arg("file_size"), py::arg("need_create"),
           py::call_guard<py::gil_scoped_release>())
      .def("open",  &flexkv::Pcfs::open)
      .def("close", &flexkv::Pcfs::close)
      .def("write", &flexkv::Pcfs::write)
      .def("read",  &flexkv::Pcfs::read);

  m.def("set_pcfs_instance", &flexkv::set_pcfs_instance,
        py::arg("pcfs"));
  m.def("call_pcfs_read",  &flexkv::call_pcfs_read,
        py::arg("file_nodeid"), py::arg("offset"), py::arg("buffer"),
        py::arg("size"), py::arg("thread_id"));
  m.def("call_pcfs_write", &flexkv::call_pcfs_write,
        py::arg("file_nodeid"), py::arg("offset"), py::arg("buffer"),
        py::arg("size"), py::arg("thread_id"));
  m.def("shared_transfer_kv_blocks_remote_read",
        &shared_transfer_kv_blocks_remote_read_binding,
        py::arg("file_nodeid_list"), py::arg("cfs_blocks_partition_list"),
        py::arg("cpu_blocks_partition_list"), py::arg("cpu_layer_id_list"),
        py::arg("cpu_tensor_ptr"), py::arg("cpu_layer_stride_in_bytes"),
        py::arg("cpu_kv_stride_in_bytes"), py::arg("cfs_layer_stride_in_bytes"),
        py::arg("cfs_block_stride_in_bytes"), py::arg("cfs_kv_stride_in_bytes"),
        py::arg("block_size_in_bytes"), py::arg("total_layers"),
        py::arg("is_mla") = false, py::arg("num_threads_per_file") = 8);
#endif // FLEXKV_ENABLE_CFS

  // ── Radix tree (CRadixTreeIndex / CRadixNode / CMatchResult) ─────────────
  py::class_<flexkv::CRadixTreeIndex>(m, "CRadixTreeIndex")
      .def(py::init([](int tokens_per_block, unsigned int max_num_blocks,
                       int hit_reward_seconds, std::string eviction_policy) {
             auto policy = flexkv::parse_eviction_policy(eviction_policy);
             return new flexkv::CRadixTreeIndex(
                 tokens_per_block, max_num_blocks, hit_reward_seconds, policy);
           }),
           py::arg("tokens_per_block"), py::arg("max_num_blocks") = 1000000,
           py::arg("hit_reward_seconds") = 0,
           py::arg("eviction_policy") = "lru")
      .def("is_empty", &flexkv::CRadixTreeIndex::is_empty)
      .def("reset",    &flexkv::CRadixTreeIndex::reset)
      .def("lock",     &flexkv::CRadixTreeIndex::lock,   py::arg("node"))
      .def("unlock",   &flexkv::CRadixTreeIndex::unlock, py::arg("node"))
      .def("set_ready",&flexkv::CRadixTreeIndex::set_ready,
           py::arg("node"), py::arg("ready"), py::arg("ready_length"))
      .def("insert",   &flexkv::CRadixTreeIndex::insert,
           py::return_value_policy::reference,
           py::arg("physical_block_ids"), py::arg("block_hashes"),
           py::arg("num_blocks"), py::arg("num_insert_blocks"),
           py::arg("ready") = true, py::arg("node") = nullptr,
           py::arg("num_matched_blocks") = -1,
           py::arg("last_node_matched_length") = -1,
           py::call_guard<py::gil_scoped_release>())
      .def("evict",
           py::overload_cast<torch::Tensor &, int>(
               &flexkv::CRadixTreeIndex::evict),
           py::arg("evicted_blocks"), py::arg("num_evicted"),
           py::call_guard<py::gil_scoped_release>())
      .def("evict",
           py::overload_cast<torch::Tensor &, torch::Tensor &, int>(
               &flexkv::CRadixTreeIndex::evict),
           py::arg("evicted_blocks"), py::arg("evicted_block_hashes"),
           py::arg("num_evicted"),
           py::call_guard<py::gil_scoped_release>())
      .def("total_cached_blocks",   &flexkv::CRadixTreeIndex::total_cached_blocks)
      .def("total_unready_blocks",  &flexkv::CRadixTreeIndex::total_unready_blocks)
      .def("total_ready_blocks",    &flexkv::CRadixTreeIndex::total_ready_blocks)
      .def("match_prefix",          &flexkv::CRadixTreeIndex::match_prefix,
           py::arg("block_hashes"), py::arg("num_blocks"),
           py::arg("update_cache_info"),
           py::call_guard<py::gil_scoped_release>());

  py::class_<flexkv::CRadixNode>(m, "CRadixNode")
      .def(py::init<flexkv::CRadixTreeIndex *, bool, int>())
      .def(py::init<flexkv::CRadixTreeIndex *, bool, int, bool>())
      .def("size", &flexkv::CRadixNode::size)
      .def("has_block_node_ids", &flexkv::CRadixNode::has_block_node_ids)
      .def_property_readonly("parent", &flexkv::CRadixNode::get_parent,
                             py::return_value_policy::reference);

  py::class_<flexkv::CMatchResult,
             std::shared_ptr<flexkv::CMatchResult>>(m, "CMatchResult")
      .def(py::init<int, int, int, flexkv::CRadixNode *, flexkv::CRadixNode *,
                    torch::Tensor, torch::Tensor>())
      .def_readonly("last_ready_node",            &flexkv::CMatchResult::last_ready_node)
      .def_readonly("last_node",                  &flexkv::CMatchResult::last_node)
      .def_readonly("physical_blocks",            &flexkv::CMatchResult::physical_blocks)
      .def_readonly("block_node_ids",             &flexkv::CMatchResult::block_node_ids)
      .def_readonly("num_ready_matched_blocks",   &flexkv::CMatchResult::num_ready_matched_blocks)
      .def_readonly("num_matched_blocks",         &flexkv::CMatchResult::num_matched_blocks)
      .def_readonly("last_node_matched_length",   &flexkv::CMatchResult::last_node_matched_length);

  // ── Distributed / P2P (BlockMeta, RedisMetaChannel, Local/Dist/RefRadixTree)
#ifdef FLEXKV_ENABLE_P2P
  py::class_<flexkv::BlockMeta>(m, "BlockMeta")
      .def(py::init<>())
      .def_readwrite("ph",    &flexkv::BlockMeta::ph)
      .def_readwrite("pb",    &flexkv::BlockMeta::pb)
      .def_readwrite("nid",   &flexkv::BlockMeta::nid)
      .def_readwrite("hash",  &flexkv::BlockMeta::hash)
      .def_readwrite("lt",    &flexkv::BlockMeta::lt)
      .def_readwrite("state", &flexkv::BlockMeta::state);

  py::class_<flexkv::LockFreeQueue<int>>(m, "IntQueue")
      .def(py::init<>())
      .def("push", [](flexkv::LockFreeQueue<int> &q, int v) { q.push(v); },
           py::arg("value"))
      .def("pop", [](flexkv::LockFreeQueue<int> &q) {
        int v = 0; bool ok = q.pop(v);
        return py::make_tuple(ok, v);
      });

  py::class_<flexkv::RedisMetaChannel>(m, "RedisMetaChannel")
      .def(py::init<const std::string &, int, uint32_t, const std::string &,
                    const std::string &, const std::string &>(),
           py::arg("host"), py::arg("port"), py::arg("node_id"),
           py::arg("local_ip"), py::arg("blocks_key") = std::string("blocks"),
           py::arg("password") = std::string(""))
      .def("connect",      &flexkv::RedisMetaChannel::connect)
      .def("get_node_id",  &flexkv::RedisMetaChannel::get_node_id)
      .def("get_local_ip", &flexkv::RedisMetaChannel::get_local_ip)
      .def("make_block_key",&flexkv::RedisMetaChannel::make_block_key,
           py::arg("node_id"), py::arg("hash"))
      .def("publish_one",
           [](flexkv::RedisMetaChannel &ch, const flexkv::BlockMeta &meta) {
             return ch.publish(meta);
           })
      .def("publish_batch",
           [](flexkv::RedisMetaChannel &ch,
              const std::vector<flexkv::BlockMeta> &metas, size_t batch_size) {
             return ch.publish(metas, batch_size);
           }, py::arg("metas"), py::arg("batch_size") = 100)
      .def("load",
           [](flexkv::RedisMetaChannel &ch, size_t max_items) {
             std::vector<flexkv::BlockMeta> out;
             ch.load(out, max_items);
             return out;
           }, py::arg("max_items"))
      .def("renew_node_leases",
           py::overload_cast<uint32_t, uint64_t, size_t>(
               &flexkv::RedisMetaChannel::renew_node_leases),
           py::arg("node_id"), py::arg("new_lt"), py::arg("batch_size") = 200)
      .def("renew_node_leases_with_hashes",
           [](flexkv::RedisMetaChannel &ch, uint32_t node_id, uint64_t new_lt,
              const std::vector<int64_t> &hashes, size_t batch_size) {
             std::list<int64_t> l(hashes.begin(), hashes.end());
             return ch.renew_node_leases(node_id, new_lt, l, batch_size);
           }, py::arg("node_id"), py::arg("new_lt"), py::arg("hashes"),
           py::arg("batch_size") = 200)
      .def("list_keys",
           [](flexkv::RedisMetaChannel &ch, const std::string &pattern) {
             std::vector<std::string> keys;
             ch.list_keys(pattern, keys);
             return keys;
           }, py::arg("pattern"))
      .def("list_node_keys",
           [](flexkv::RedisMetaChannel &ch) {
             std::vector<std::string> keys;
             ch.list_node_keys(keys);
             return keys;
           })
      .def("list_block_keys",
           [](flexkv::RedisMetaChannel &ch, uint32_t node_id) {
             std::vector<std::string> keys;
             ch.list_block_keys(node_id, keys);
             return keys;
           }, py::arg("node_id"))
      .def("hmget_field_for_keys",
           [](flexkv::RedisMetaChannel &ch,
              const std::vector<std::string> &keys, const std::string &field) {
             std::vector<std::string> values;
             ch.hmget_field_for_keys(keys, field, values);
             return values;
           }, py::arg("keys"), py::arg("field"))
      .def("hmget_two_fields_for_keys",
           [](flexkv::RedisMetaChannel &ch,
              const std::vector<std::string> &keys,
              const std::string &f1, const std::string &f2) {
             std::vector<std::pair<std::string, std::string>> out;
             ch.hmget_two_fields_for_keys(keys, f1, f2, out);
             return out;
           }, py::arg("keys"), py::arg("field1"), py::arg("field2"))
      .def("load_metas_by_keys",
           [](flexkv::RedisMetaChannel &ch,
              const std::vector<std::string> &keys) {
             std::vector<flexkv::BlockMeta> out;
             ch.load_metas_by_keys(keys, out);
             return out;
           }, py::arg("keys"))
      .def("update_block_state_batch",
           [](flexkv::RedisMetaChannel &ch, uint32_t node_id,
              const std::vector<int64_t> &hashes, int state, size_t batch_size) {
             std::deque<int64_t> dq(hashes.begin(), hashes.end());
             return ch.update_block_state_batch(node_id, &dq, state, batch_size);
           }, py::arg("node_id"), py::arg("hashes"), py::arg("state"),
           py::arg("batch_size") = 200)
      .def("delete_blockmeta_batch",
           [](flexkv::RedisMetaChannel &ch, uint32_t node_id,
              const std::vector<int64_t> &hashes, size_t batch_size) {
             std::deque<int64_t> dq(hashes.begin(), hashes.end());
             return ch.delete_blockmeta_batch(node_id, &dq, batch_size);
           }, py::arg("node_id"), py::arg("hashes"), py::arg("batch_size") = 200);

  py::class_<flexkv::LocalRadixTree, flexkv::CRadixTreeIndex>(m, "LocalRadixTree")
      .def(py::init<int, unsigned int, uint32_t, uint32_t, uint32_t, uint32_t,
                    uint32_t, uint32_t, uint32_t, std::string>(),
           py::arg("tokens_per_block"), py::arg("max_num_blocks") = 1000000u,
           py::arg("lease_ttl_ms") = 100000, py::arg("renew_lease_ms") = 0,
           py::arg("refresh_batch_size") = 256, py::arg("idle_sleep_ms") = 10,
           py::arg("safety_ttl_ms") = 100,
           py::arg("swap_block_threshold") = 1024,
           py::arg("hit_reward_seconds") = 0,
           py::arg("eviction_policy") = "lru")
      .def("set_meta_channel",   &flexkv::LocalRadixTree::set_meta_channel, py::arg("channel"))
      .def("start",              &flexkv::LocalRadixTree::start,  py::arg("channel"))
      .def("stop",               &flexkv::LocalRadixTree::stop)
      .def("insert_and_publish", &flexkv::LocalRadixTree::insert_and_publish, py::arg("node"))
      .def("insert",             &flexkv::LocalRadixTree::insert,
           py::return_value_policy::reference,
           py::arg("physical_block_ids"), py::arg("block_hashes"),
           py::arg("num_blocks"), py::arg("num_insert_blocks"),
           py::arg("ready") = true, py::arg("node") = nullptr,
           py::arg("num_matched_blocks") = -1,
           py::arg("last_node_matched_length") = -1,
           py::call_guard<py::gil_scoped_release>())
      .def("evict",
           py::overload_cast<torch::Tensor &, int>(&flexkv::LocalRadixTree::evict),
           py::arg("evicted_blocks"), py::arg("num_evicted"),
           py::call_guard<py::gil_scoped_release>())
      .def("evict",
           py::overload_cast<torch::Tensor &, torch::Tensor &, int>(
               &flexkv::LocalRadixTree::evict),
           py::arg("evicted_blocks"), py::arg("evicted_block_hashes"),
           py::arg("num_evicted"),
           py::call_guard<py::gil_scoped_release>())
      .def("match_prefix",         &flexkv::LocalRadixTree::match_prefix,
           py::arg("block_hashes"), py::arg("num_blocks"),
           py::arg("update_cache_info") = true,
           py::call_guard<py::gil_scoped_release>())
      .def("total_unready_blocks", &flexkv::LocalRadixTree::total_unready_blocks)
      .def("total_ready_blocks",   &flexkv::LocalRadixTree::total_ready_blocks)
      .def("total_cached_blocks",  &flexkv::LocalRadixTree::total_cached_blocks)
      .def("total_node_num",       &flexkv::LocalRadixTree::total_node_num)
      .def("reset",                &flexkv::LocalRadixTree::reset)
      .def("is_root",              &flexkv::LocalRadixTree::is_root, py::arg("node"))
      .def("remove_node",          &flexkv::LocalRadixTree::remove_node, py::arg("node"))
      .def("remove_leaf",          &flexkv::LocalRadixTree::remove_leaf, py::arg("node"))
      .def("add_node",             &flexkv::LocalRadixTree::add_node,    py::arg("node"))
      .def("add_leaf",             &flexkv::LocalRadixTree::add_leaf,    py::arg("node"))
      .def("lock",                 &flexkv::LocalRadixTree::lock,        py::arg("node"))
      .def("unlock",               &flexkv::LocalRadixTree::unlock,      py::arg("node"))
      .def("is_empty",             &flexkv::LocalRadixTree::is_empty)
      .def("inc_node_count",       &flexkv::LocalRadixTree::inc_node_count)
      .def("dec_node_count",       &flexkv::LocalRadixTree::dec_node_count)
      .def("set_ready",            &flexkv::LocalRadixTree::set_ready,
           py::arg("node"), py::arg("ready"), py::arg("ready_length") = -1)
      .def("drain_pending_queues", &flexkv::LocalRadixTree::drain_pending_queues);

  py::class_<flexkv::DistributedRadixTree>(m, "DistributedRadixTree")
      .def(py::init<int, unsigned int, uint32_t, size_t, uint32_t, uint32_t,
                    uint32_t, uint32_t>(),
           py::arg("tokens_per_block"), py::arg("max_num_blocks"),
           py::arg("node_id"), py::arg("refresh_batch_size") = 128,
           py::arg("rebuild_interval_ms") = 1000, py::arg("idle_sleep_ms") = 10,
           py::arg("lease_renew_ms") = 5000, py::arg("hit_reward_seconds") = 0)
      .def("start",               &flexkv::DistributedRadixTree::start, py::arg("channel"))
      .def("stop",                &flexkv::DistributedRadixTree::stop)
      .def("remote_tree_refresh", &flexkv::DistributedRadixTree::remote_tree_refresh,
           py::return_value_policy::reference)
      .def("match_prefix",        &flexkv::DistributedRadixTree::match_prefix,
           py::arg("block_hashes"), py::arg("num_blocks"),
           py::arg("update_cache_info") = true,
           py::call_guard<py::gil_scoped_release>())
      .def("lock",      &flexkv::DistributedRadixTree::lock,      py::arg("node"))
      .def("unlock",    &flexkv::DistributedRadixTree::unlock,    py::arg("node"))
      .def("is_empty",  &flexkv::DistributedRadixTree::is_empty)
      .def("set_ready", &flexkv::DistributedRadixTree::set_ready,
           py::arg("node"), py::arg("ready") = true, py::arg("ready_length") = -1);

  py::class_<flexkv::RefRadixTree, flexkv::CRadixTreeIndex>(m, "RefRadixTree")
      .def(py::init<int, unsigned int, uint32_t, uint32_t,
                    flexkv::LockFreeQueue<flexkv::QueuedNode> *,
                    flexkv::LeaseMetaMemPool *, uint64_t>(),
           py::arg("tokens_per_block"), py::arg("max_num_blocks") = 1000000u,
           py::arg("lease_renew_ms") = 5000, py::arg("hit_reward_seconds") = 0,
           py::arg("renew_lease_queue") = nullptr, py::arg("lt_pool") = nullptr,
           py::arg("generation") = 0)
      .def("dec_ref_cnt",    &flexkv::RefRadixTree::dec_ref_cnt)
      .def("inc_ref_cnt",    &flexkv::RefRadixTree::inc_ref_cnt)
      .def("get_generation", &flexkv::RefRadixTree::get_generation);
#endif // FLEXKV_ENABLE_P2P
}
