#include "compression/common/transfer_ssd_packed.h"
#include "compression/common/packed_ssd.h"

#include <algorithm>
#include <atomic>
#include <future>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace flexkv {

static void partition_and_remap_blocks_by_device_packed(
    const int64_t *cpu_block_ids, const int64_t *ssd_block_ids, int num_blocks,
    int num_devices, int round_robin,
    std::vector<std::vector<int>> &cpu_blocks_partition,
    std::vector<std::vector<int>> &ssd_blocks_partition,
    std::vector<std::vector<int>> &ssd_orig_blocks_partition) {
  for (int i = 0; i < num_blocks; i++) {
    int64_t ssd_block_id = ssd_block_ids[i];
    int64_t cpu_block_id = cpu_block_ids[i];
    int device_id = (ssd_block_id / round_robin) % num_devices;
    int block_id_in_device =
        ((ssd_block_id / round_robin) / num_devices) * round_robin +
        (ssd_block_id % round_robin);
    ssd_blocks_partition[device_id].push_back(block_id_in_device);
    cpu_blocks_partition[device_id].push_back(cpu_block_id);
    ssd_orig_blocks_partition[device_id].push_back(
        static_cast<int>(ssd_block_id));
  }
}

static void validate_size_table_args(
    uint32_t *cpu_size_table_base,
    uint32_t *ssd_size_table_base,
    int tp_size,
    int64_t cpu_tp_rank_stride_in_bytes,
    int64_t cpu_size_table_rank_stride,
    int64_t ssd_size_table_rank_stride) {
  const std::string prefix = "transfer_kv_blocks_ssd_packed: ";

  if (cpu_size_table_base == nullptr || ssd_size_table_base == nullptr) {
    throw std::runtime_error(prefix + "size tables are mandatory");
  }
  if (tp_size <= 0) {
    throw std::runtime_error(prefix + "tp_size must be > 0");
  }
  if (tp_size > 1 &&
      (cpu_tp_rank_stride_in_bytes <= 0 ||
       cpu_size_table_rank_stride <= 0 ||
       ssd_size_table_rank_stride <= 0)) {
    throw std::runtime_error(
        prefix + "TP calls require non-zero rank strides");
  }
}

static int64_t transfer_kv_blocks_ssd_packed_impl(
    SSDIOCTX &ioctx, const torch::Tensor &cpu_layer_id_list,
    int64_t cpu_tensor_ptr, const torch::Tensor &ssd_block_ids,
    const torch::Tensor &cpu_block_ids,
    int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes,
    int64_t chunk_size_in_bytes,
    int64_t block_stride_in_bytes,
    bool is_read, int num_blocks_per_file,
    int round_robin,
    int num_threads_per_device,
    bool is_mla,
    // --- nvcomp packed-specific ---
    bool blockfirst, // true = BLOCKFIRST, false = LAYERFIRST
    int total_layers,
    uint32_t* cpu_size_table_base,
    int64_t cpu_size_table_block_stride,
    int64_t cpu_size_table_layer_stride,
    uint32_t* ssd_size_table_base,
    int64_t ssd_size_table_block_stride,
    int64_t ssd_size_table_layer_stride,
    int tp_size,
    int64_t cpu_tp_rank_stride_in_bytes,
    int64_t cpu_size_table_rank_stride,
    int64_t ssd_size_table_rank_stride) {
  validate_size_table_args(
      cpu_size_table_base, ssd_size_table_base, tp_size,
      cpu_tp_rank_stride_in_bytes, cpu_size_table_rank_stride,
      ssd_size_table_rank_stride);

  const int num_devices = ioctx.get_num_devices();
  const int num_files_per_device = ioctx.get_num_files_per_device();

  const int64_t *ssd_block_id_ptr = ssd_block_ids.data_ptr<int64_t>();
  const int64_t *cpu_block_id_ptr = cpu_block_ids.data_ptr<int64_t>();

  const int num_blocks = ssd_block_ids.size(0);
  const int num_layers = cpu_layer_id_list.size(0);
  if (num_layers == 0) return 0;
  const int32_t *cpu_layer_id_list_ptr = cpu_layer_id_list.data_ptr<int32_t>();
  const int start_layer = cpu_layer_id_list_ptr[0];
  const int end_layer = start_layer + num_layers;

  // TODO(nvcomp-guard): packed SSD requires a transfer starting at layer 0;
  // LAYERFIRST additionally requires the full layer range (its on-disk block
  // slot is sized for all layers).
  const char *layout_name = blockfirst ? "blockfirst" : "layerfirst";
  if (start_layer != 0 || (!blockfirst && num_layers != total_layers)) {
    throw std::runtime_error(
        std::string("transfer_kv_blocks_ssd_packed: ") + layout_name +
        (blockfirst ? " requires layer_id == 0"
                    : " requires a full-layer transfer starting at layer 0"));
  }

  const int kv_dim = is_mla ? 1 : 2;
  // On-disk slot size per block.
  // BLOCKFIRST's block_stride already spans the whole block;
  // LAYERFIRST's block_stride is per-layer/kv, so scale it up.
  const int64_t disk_block_stride_in_bytes =
      blockfirst ? block_stride_in_bytes
                 : static_cast<int64_t>(total_layers) * kv_dim *
                       block_stride_in_bytes;

  // BLOCKFIRST nests rank outermost,
  // LAYERFIRST nests layer outermost -- each matches its CPU memory layout so
  // the per-block gather/scatter walks CPU memory sequentially.
  std::vector<PackedCoord> span_order;
  span_order.reserve(static_cast<size_t>(tp_size) *
                     static_cast<size_t>(end_layer - start_layer) *
                     static_cast<size_t>(kv_dim));
  if (blockfirst) {
    for (int rank = 0; rank < tp_size; rank++)
      for (int lid = start_layer; lid < end_layer; lid++)
        for (int kv = 0; kv < kv_dim; kv++)
          span_order.push_back({rank, lid, kv});
  } else {
    for (int lid = start_layer; lid < end_layer; lid++)
      for (int kv = 0; kv < kv_dim; kv++)
        for (int rank = 0; rank < tp_size; rank++)
          span_order.push_back({rank, lid, kv});
  }

  auto &direct_fds = ioctx.get_fds(is_read, true);    // O_DIRECT
  auto &buffered_fds = ioctx.get_fds(is_read, false); // buffered

  std::vector<std::vector<int>> cpu_blocks_partition(num_devices);
  std::vector<std::vector<int>> ssd_blocks_partition(num_devices);
  std::vector<std::vector<int>> ssd_orig_blocks_partition(num_devices);
  partition_and_remap_blocks_by_device_packed(
      cpu_block_id_ptr, ssd_block_id_ptr, num_blocks, num_devices, round_robin,
      cpu_blocks_partition, ssd_blocks_partition, ssd_orig_blocks_partition);

  // Compressed payload bytes summed across all threads/devices for this op
  // (per-block packed_bytes). Returned so Python can log the real transferred
  // size, mirroring tp_group_transfer_ans.
  std::atomic<int64_t> total_packed_bytes{0};
  std::vector<std::thread> threads;
  std::vector<std::future<std::exception_ptr>> futures;
  for (int t = 0; t < num_threads_per_device; t++) {
    for (int d = 0; d < num_devices; d++) {
      int num_transfer_blocks = cpu_blocks_partition[d].size();
      int num_blocks_per_thread =
          (num_transfer_blocks + num_threads_per_device - 1) /
          num_threads_per_device;
      int start_block = t * num_blocks_per_thread;
      int end_block =
          std::min(start_block + num_blocks_per_thread, num_transfer_blocks);
      if (start_block >= end_block) continue;

      std::promise<std::exception_ptr> prom;
      futures.push_back(prom.get_future());
      threads.emplace_back(
          [d, &total_packed_bytes, &direct_fds, &buffered_fds, &cpu_blocks_partition,
           &ssd_blocks_partition, &ssd_orig_blocks_partition, &span_order,
           start_block, end_block, cpu_tensor_ptr, cpu_layer_stride_in_bytes,
           cpu_kv_stride_in_bytes, block_stride_in_bytes,
           disk_block_stride_in_bytes, cpu_tp_rank_stride_in_bytes,
           chunk_size_in_bytes, num_files_per_device, is_read,
           cpu_size_table_base, cpu_size_table_rank_stride,
           cpu_size_table_block_stride, cpu_size_table_layer_stride,
           ssd_size_table_base, ssd_size_table_rank_stride,
           ssd_size_table_block_stride, ssd_size_table_layer_stride,
           layout_name, prom = std::move(prom)]() mutable {
            try {
              total_packed_bytes.fetch_add(
                  transfer_packed_thread_impl(
                      direct_fds[d], buffered_fds[d], cpu_blocks_partition[d],
                      ssd_blocks_partition[d], ssd_orig_blocks_partition[d],
                      span_order, start_block, end_block, cpu_tensor_ptr,
                      cpu_layer_stride_in_bytes, cpu_kv_stride_in_bytes,
                      block_stride_in_bytes, disk_block_stride_in_bytes,
                      cpu_tp_rank_stride_in_bytes, chunk_size_in_bytes,
                      num_files_per_device, is_read, cpu_size_table_base,
                      cpu_size_table_rank_stride, cpu_size_table_block_stride,
                      cpu_size_table_layer_stride, ssd_size_table_base,
                      ssd_size_table_rank_stride, ssd_size_table_block_stride,
                      ssd_size_table_layer_stride, layout_name),
                  std::memory_order_relaxed);
              prom.set_value(nullptr);
            } catch (...) {
              prom.set_value(std::current_exception());
            }
          });
    }
  }

  for (auto &th : threads) th.join();
  for (auto &f : futures) {
    if (auto e = f.get()) std::rethrow_exception(e);
  }
  return total_packed_bytes.load(std::memory_order_relaxed);
}

int64_t transfer_kv_blocks_ssd_packed(
    SSDIOCTX &ioctx, const torch::Tensor &cpu_layer_id_list,
    int64_t cpu_tensor_ptr, const torch::Tensor &ssd_block_ids,
    const torch::Tensor &cpu_block_ids,
    int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes,
    int64_t chunk_size_in_bytes,
    int64_t block_stride_in_bytes,
    bool is_read, int num_blocks_per_file,
    int round_robin,
    int num_threads_per_device,
    bool is_mla,
    // --- nvcomp packed-specific ---
    const std::string &layout_type,
    int total_layers,
    uint32_t* cpu_size_table_base,
    int64_t cpu_size_table_block_stride,
    int64_t cpu_size_table_layer_stride,
    uint32_t* ssd_size_table_base,
    int64_t ssd_size_table_block_stride,
    int64_t ssd_size_table_layer_stride,
    int tp_size,
    int64_t cpu_tp_rank_stride_in_bytes,
    int64_t cpu_size_table_rank_stride,
    int64_t ssd_size_table_rank_stride) {
  bool blockfirst;
  if (layout_type == "BLOCKFIRST") {
    blockfirst = true;
  } else if (layout_type == "LAYERFIRST") {
    blockfirst = false;
  } else {
    throw std::runtime_error(
        "transfer_kv_blocks_ssd_packed: unsupported layout_type: " +
        layout_type);
  }
  return transfer_kv_blocks_ssd_packed_impl(
      ioctx, cpu_layer_id_list, cpu_tensor_ptr, ssd_block_ids, cpu_block_ids,
      cpu_layer_stride_in_bytes, cpu_kv_stride_in_bytes, chunk_size_in_bytes,
      block_stride_in_bytes, is_read, num_blocks_per_file, round_robin,
      num_threads_per_device, is_mla, blockfirst, total_layers,
      cpu_size_table_base, cpu_size_table_block_stride,
      cpu_size_table_layer_stride, ssd_size_table_base,
      ssd_size_table_block_stride, ssd_size_table_layer_stride, tp_size,
      cpu_tp_rank_stride_in_bytes, cpu_size_table_rank_stride,
      ssd_size_table_rank_stride);
}

} // namespace flexkv
