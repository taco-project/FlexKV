#pragma once

#include <cuda_runtime.h>
#include <fcntl.h>
#include <map>
#include <memory>
#include <nvtx3/nvToolsExt.h>
#include <string>
#include <sys/eventfd.h>
#include <torch/extension.h>
#include <unistd.h>
#include <vector>

#include "gtensor_handler.cuh"
#include "transfer.cuh"
#include "transfer_ssd.h"

namespace flexkv {

// One LayerGroup's CPU/SSD/GPU transfer parameters (multi-group mode only).
// In single-group mode the legacy member fields are used instead.
struct GroupParams {
  // SSD <-> CPU strides (in bytes)
  int num_layers;           // group's local layer count
  int64_t cpu_offset_bytes; // start of group's region inside CPU block
  int64_t ssd_offset_bytes; // start of group's region inside SSD block
  int64_t cpu_layer_stride;
  int64_t cpu_kv_stride;
  int64_t ssd_layer_stride;
  int64_t ssd_kv_stride;
  int64_t chunk_size; // group's chunk size (bytes)

  // CPU -> GPU strides (in bytes), TP-divided
  int64_t h2d_cpu_kv_stride;
  int64_t h2d_cpu_layer_stride;
  int64_t cpu_block_stride; // bytes-per-block (full block, all TP ranks)
  int64_t cpu_tp_stride;    // bytes per TP rank within a block

  // Per-GPU GPU-side strides (size = num_gpus_)
  std::vector<int64_t> gpu_kv_strides;
  std::vector<int64_t> gpu_block_strides;
  std::vector<int64_t> gpu_layer_strides;
  std::vector<int64_t> gpu_chunk_sizes;

  // GPU tensor pointers (cudaMallocHost'd, num_gpus_ * num_tensors_per_gpu)
  void **gpu_blocks_flat;
  int num_tensors_per_gpu;
  BackendType backend_type;
  std::vector<GTensorHandler> gpu_tensor_handlers;
};

class LayerwiseTransferGroup {
public:
  // Single-group constructor (legacy: uniform num_kv_heads/head_size/dtype).
  LayerwiseTransferGroup(
      int num_gpus, const std::vector<std::vector<torch::Tensor>> &gpu_blocks,
      torch::Tensor &cpu_blocks,
      std::map<int, std::vector<std::string>> &ssd_files, int num_layers,
      torch::Tensor &gpu_kv_strides_tensor,
      torch::Tensor &gpu_block_strides_tensor,
      torch::Tensor &gpu_layer_strides_tensor,
      torch::Tensor &gpu_chunk_sizes_tensor, int iouring_entries,
      int iouring_flags, torch::Tensor &layer_eventfds_tensor, int tp_size);

  // Multi-group constructor. ``gpu_blocks_per_group[gi][d]`` is the GPU-side
  // tensor list for group ``gi`` on device ``d``. ``csr_*`` encode the
  // 1:N mapping from original layer id to (group_idx, local_layer_id) members
  // (see ``flexkv.common.config.LayerMemberMap``).
  //
  // Per-group flat int64 arrays (size = num_groups) carry the per-group
  // strides; per-(group, gpu) flat int64 arrays (size = num_groups * num_gpus)
  // carry the per-GPU strides. ``layer_eventfds_tensor`` has shape
  // ``[num_counters, tp_size, num_original_layers]``.
  LayerwiseTransferGroup(
      int num_gpus,
      const std::vector<std::vector<std::vector<torch::Tensor>>>
          &gpu_blocks_per_group,
      torch::Tensor &cpu_blocks,
      std::map<int, std::vector<std::string>> &ssd_files,
      int num_original_layers, const std::vector<int> &csr_offsets,
      const std::vector<int> &csr_group_idx,
      const std::vector<int> &csr_local_id,
      const std::vector<int> &group_num_layers,
      const std::vector<int64_t> &group_cpu_offset_bytes,
      const std::vector<int64_t> &group_ssd_offset_bytes,
      const std::vector<int64_t> &group_cpu_layer_strides,
      const std::vector<int64_t> &group_cpu_kv_strides,
      const std::vector<int64_t> &group_ssd_layer_strides,
      const std::vector<int64_t> &group_ssd_kv_strides,
      const std::vector<int64_t> &group_chunk_sizes,
      const std::vector<int64_t> &group_h2d_cpu_kv_strides,
      const std::vector<int64_t> &group_h2d_cpu_layer_strides,
      const std::vector<int64_t> &group_cpu_block_strides,
      const std::vector<int64_t> &group_cpu_tp_strides,
      const std::vector<int64_t> &group_gpu_kv_strides,
      const std::vector<int64_t> &group_gpu_block_strides,
      const std::vector<int64_t> &group_gpu_layer_strides,
      const std::vector<int64_t> &group_gpu_chunk_sizes, int iouring_entries,
      int iouring_flags, torch::Tensor &layer_eventfds_tensor, int tp_size);

  ~LayerwiseTransferGroup();

  // Single-group layerwise transfer: SSD->CPU (all layers) + CPU->GPU
  // (per layer_granularity batch).
  void layerwise_transfer(
      const torch::Tensor
          &ssd_block_ids, // SSD source block ids (for disk2host)
      const torch::Tensor
          &cpu_block_ids_d2h, // CPU dest block ids (for disk2host)
      const int64_t ssd_layer_stride_in_bytes,
      const int64_t ssd_kv_stride_in_bytes, const int num_blocks_per_file,
      const int round_robin, const int num_threads_per_device,
      const torch::Tensor
          &gpu_block_id_tensor, // GPU dest block ids (for host2device)
      const torch::Tensor
          &cpu_block_id_tensor, // CPU source block ids (for host2device)
      const int64_t cpu_kv_stride_in_bytes,
      const int64_t cpu_layer_stride_in_bytes,
      const int64_t cpu_block_stride_in_bytes,
      const int64_t cpu_chunk_size_in_bytes,
      const int64_t h2d_cpu_kv_stride_in_bytes,
      const int64_t h2d_cpu_layer_stride_in_bytes,
      const int64_t cpu_tp_stride_in_bytes, const int transfer_cta_num,
      const bool use_ce_transfer, const int num_layers,
      const int layer_granularity, const bool is_mla, const int counter_id = 0);

  // Multi-group layerwise transfer: SSD->CPU per group, CPU->GPU per original
  // layer (expanding the CSR to fire one transfer kernel per group member).
  // ``layer_granularity`` is implicitly 1: each original layer fires its own
  // eventfd as soon as ALL its members on ALL GPUs finish.
  void layerwise_transfer_multi_group(
      const torch::Tensor &ssd_block_ids,
      const torch::Tensor &cpu_block_ids_d2h, const int num_blocks_per_file,
      const int round_robin, const int num_threads_per_device,
      const torch::Tensor &gpu_block_id_tensor,
      const torch::Tensor &cpu_block_id_tensor, const int transfer_cta_num,
      const bool use_ce_transfer, const bool is_mla, const int counter_id = 0);

private:
  int num_gpus_;
  // Single-group GPU pointer table (multi-group: nullptr; per-group tables
  // live in ``groups_[gi].gpu_blocks_flat``).
  void **gpu_blocks_;
  void *cpu_blocks_;
  int num_tensors_per_gpu_;
  // Single-group GPU strides (multi-group: nullptr).
  int64_t *gpu_kv_strides_in_bytes_;
  int64_t *gpu_block_strides_in_bytes_;
  int64_t *gpu_layer_strides_in_bytes_;
  int64_t *gpu_chunk_sizes_in_bytes_;

  BackendType backend_type_;
  std::vector<GTensorHandler> gpu_tensor_handlers_;

  std::vector<int> gpu_device_ids_;
  std::vector<cudaStream_t> streams_;
  std::vector<cudaEvent_t> events_;

  // SSD IO context
  bool enable_ssd_;
  std::unique_ptr<SSDIOCTX> ioctx_;

  // Layer eventfds for notification
  // Shape: [num_counters, tp_size, num_layers]
  bool enable_eventfd_;
  int tp_size_;
  int num_counters_;
  int num_layers_; // single-group: model layers; multi-group: original layers
  std::vector<int> layer_eventfds_; // Flat array
  int current_counter_id_; // Current counter set index for this transfer

  // ---- Multi-group state ----
  bool has_multi_group_;
  std::vector<GroupParams> groups_;
  // CSR mapping: for original layer ``i``,
  // members are at csr_offsets_[i] .. csr_offsets_[i+1] - 1, encoded by
  // (csr_group_idx_[m], csr_local_id_[m]).
  std::vector<int> csr_offsets_;
  std::vector<int> csr_group_idx_;
  std::vector<int> csr_local_id_;
  int num_original_layers_;

  // Single-group: ``expected_count = num_gpus_``.
  // Multi-group: ``expected_count = members_this_layer * num_gpus_``.
  void layer_done_callback(int start_layer, int layers_this_batch,
                           int expected_count,
                           nvtxRangeId_t *current_range_id_ptr,
                           bool is_last_batch, const char *next_range_name,
                           nvtxRangeId_t *next_range_id_ptr,
                           int callbacks_per_gpu = 1);
};

} // namespace flexkv
