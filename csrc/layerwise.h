#pragma once

#include <atomic>
#include <cuda_runtime.h>
#include <fcntl.h>
#include <map>
#include <memory>
#include <nvtx3/nvToolsExt.h>
#include <string>
#include <sys/eventfd.h>
#include <thread>
#include <torch/extension.h>
#include <unistd.h>
#include <vector>

#include "gtensor_handler.cuh"
#include "transfer.cuh"
#include "ce_transfer.h"
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
  void **gpu_blocks_flat = nullptr;
  int num_tensors_per_gpu = 0;
  BackendType backend_type = BackendType::VLLM;
  std::vector<GTensorHandler> gpu_tensor_handlers;
  // Keep imported CUDA IPC / shared tensors alive for the lifetime of this
  // group. ``gpu_blocks_flat`` only caches ``data_ptr()``; dropping the
  // Tensor objects closes the IPC mapping and leaves dangling device pointers.
  std::vector<std::vector<torch::Tensor>> gpu_tensors;
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
      int iouring_flags, torch::Tensor &layer_eventfds_tensor, int tp_size,
      // ---- SWA fields ----
      bool has_swa = false,
      const std::vector<std::vector<torch::Tensor>> &swa_gpu_blocks =
          std::vector<std::vector<torch::Tensor>>(),
      torch::Tensor swa_cpu_blocks = torch::Tensor(),
      std::map<int, std::vector<std::string>> swa_ssd_files =
          std::map<int, std::vector<std::string>>(),
      torch::Tensor swa_gpu_kv_strides_tensor = torch::Tensor(),
      torch::Tensor swa_gpu_block_strides_tensor = torch::Tensor(),
      torch::Tensor swa_gpu_layer_strides_tensor = torch::Tensor(),
      torch::Tensor swa_gpu_chunk_sizes_tensor = torch::Tensor(),
      CETransferConfig ce_config = CETransferConfig{});

  // Multi-group constructor. ``gpu_blocks_per_group[gi][d]`` is the GPU-side
  // tensor list for group ``gi`` on device ``d``. ``layer_members`` encodes
  // the 1:N mapping from original layer id to (group_idx, local_layer_id)
  // members directly: ``layer_members[orig]`` is the list of
  // ``(group_idx, local_layer_id)`` pairs for that original layer
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
      int num_original_layers,
      const std::vector<std::vector<std::pair<int, int>>> &layer_members,
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
      int iouring_flags, torch::Tensor &layer_eventfds_tensor, int tp_size,
      // ---- SWA sidecar (orthogonal to layer_groups) ----
      bool has_swa = false,
      const std::vector<std::vector<torch::Tensor>> &swa_gpu_blocks =
          std::vector<std::vector<torch::Tensor>>(),
      torch::Tensor swa_cpu_blocks = torch::Tensor(),
      std::map<int, std::vector<std::string>> swa_ssd_files =
          std::map<int, std::vector<std::string>>(),
      torch::Tensor swa_gpu_kv_strides_tensor = torch::Tensor(),
      torch::Tensor swa_gpu_block_strides_tensor = torch::Tensor(),
      torch::Tensor swa_gpu_layer_strides_tensor = torch::Tensor(),
      torch::Tensor swa_gpu_chunk_sizes_tensor = torch::Tensor(),
      CETransferConfig ce_config = CETransferConfig{});

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
      const int layer_granularity, const bool is_mla, const int counter_id = 0,
      // ---- SWA per-call ids + strides (fused into same layer loop) ----
      const torch::Tensor &swa_h2d_src = torch::Tensor(),
      const torch::Tensor &swa_h2d_dst = torch::Tensor(),
      const torch::Tensor &swa_disk2h_src = torch::Tensor(),
      const torch::Tensor &swa_disk2h_dst = torch::Tensor(),
      const int64_t swa_cpu_kv_stride_in_bytes = 0,
      const int64_t swa_cpu_layer_stride_in_bytes = 0,
      const int64_t swa_cpu_block_stride_in_bytes = 0,
      const int64_t swa_cpu_chunk_size_in_bytes = 0,
      const int64_t swa_h2d_cpu_kv_stride_in_bytes = 0,
      const int64_t swa_h2d_cpu_layer_stride_in_bytes = 0,
      const int64_t swa_cpu_tp_stride_in_bytes = 0,
      const int64_t swa_ssd_layer_stride_in_bytes = 0,
      const int64_t swa_ssd_kv_stride_in_bytes = 0,
      const int swa_num_blocks_per_file = 0,
      // ---- MLA D2H mode (#192) + polling notification (#199) ----
      const std::string &mla_d2h_mode = "sharded",
      const std::string &notify_mode = "hostfunc");

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
      const bool use_ce_transfer, const bool is_mla, const int counter_id = 0,
      // ---- SWA per-call ids + strides (fused into same per-orig loop) ----
      const torch::Tensor &swa_h2d_src = torch::Tensor(),
      const torch::Tensor &swa_h2d_dst = torch::Tensor(),
      const torch::Tensor &swa_disk2h_src = torch::Tensor(),
      const torch::Tensor &swa_disk2h_dst = torch::Tensor(),
      const int64_t swa_cpu_kv_stride_in_bytes = 0,
      const int64_t swa_cpu_layer_stride_in_bytes = 0,
      const int64_t swa_cpu_block_stride_in_bytes = 0,
      const int64_t swa_cpu_chunk_size_in_bytes = 0,
      const int64_t swa_h2d_cpu_kv_stride_in_bytes = 0,
      const int64_t swa_h2d_cpu_layer_stride_in_bytes = 0,
      const int64_t swa_cpu_tp_stride_in_bytes = 0,
      const int64_t swa_ssd_layer_stride_in_bytes = 0,
      const int64_t swa_ssd_kv_stride_in_bytes = 0,
      const int swa_num_blocks_per_file = 0,
      const std::string &mla_d2h_mode = "sharded",
      const std::string &notify_mode = "hostfunc");

  // Bind heterogeneous SWA/state sidecars (SWA KV + compress states) onto the
  // same LayerwiseTransferGroup.  Mutually exclusive with the uniform
  // ``init_swa_sidecar_`` path.  Call after construction when
  // ``swa.multi_group`` is enabled so GET fuses main-KV + SWA/state into one
  // LAYERWISE op.
  void init_swa_multi_group(
      const std::vector<std::vector<std::vector<torch::Tensor>>>
          &swa_gpu_blocks_per_group,
      torch::Tensor swa_cpu_blocks,
      std::map<int, std::vector<std::string>> swa_ssd_files,
      const std::vector<std::vector<std::pair<int, int>>> &swa_layer_members,
      const std::vector<int> &swa_group_num_layers,
      const std::vector<int64_t> &swa_group_cpu_offset_bytes,
      const std::vector<int64_t> &swa_group_ssd_offset_bytes,
      const std::vector<int64_t> &swa_group_cpu_layer_strides,
      const std::vector<int64_t> &swa_group_cpu_kv_strides,
      const std::vector<int64_t> &swa_group_ssd_layer_strides,
      const std::vector<int64_t> &swa_group_ssd_kv_strides,
      const std::vector<int64_t> &swa_group_chunk_sizes,
      const std::vector<int64_t> &swa_group_h2d_cpu_kv_strides,
      const std::vector<int64_t> &swa_group_h2d_cpu_layer_strides,
      const std::vector<int64_t> &swa_group_cpu_block_strides,
      const std::vector<int64_t> &swa_group_cpu_tp_strides,
      const std::vector<int64_t> &swa_group_gpu_kv_strides,
      const std::vector<int64_t> &swa_group_gpu_block_strides,
      const std::vector<int64_t> &swa_group_gpu_layer_strides,
      const std::vector<int64_t> &swa_group_gpu_chunk_sizes,
      int iouring_entries = 512, int iouring_flags = 0);

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

  // ---- SWA dedicated pool state (sidecar; uniform OR multi-group) ----
  bool has_swa_ = false;
  bool has_swa_multi_group_ = false;
  // Uniform SWA (legacy LAYERFIRST single layout)
  void **swa_gpu_blocks_ = nullptr;    // flat [num_gpus * num_tensors_per_gpu]
  void *swa_cpu_blocks_ = nullptr;
  int swa_num_tensors_per_gpu_ = 0;
  int64_t *swa_gpu_kv_strides_in_bytes_ = nullptr;
  int64_t *swa_gpu_block_strides_in_bytes_ = nullptr;
  int64_t *swa_gpu_layer_strides_in_bytes_ = nullptr;
  int64_t *swa_gpu_chunk_sizes_in_bytes_ = nullptr;
  std::vector<GTensorHandler> swa_gpu_tensor_handlers_;
  BackendType swa_backend_type_;
  bool swa_enable_ssd_ = false;
  std::unique_ptr<SSDIOCTX> swa_ioctx_;
  // Heterogeneous SWA/state multi-group (BLOCKFIRST byte-flat host block)
  std::vector<GroupParams> swa_groups_;
  std::vector<std::vector<std::pair<int, int>>> swa_layer_members_;

  // Layer eventfds for notification
  // Shape: [num_counters, tp_size, num_layers]
  bool enable_eventfd_;
  int tp_size_;
  int num_counters_;
  int num_layers_; // single-group: model layers; multi-group: original layers
  std::vector<int> layer_eventfds_; // Flat array
  int current_counter_id_; // Current counter set index for this transfer
  CETransferConfig ce_config_;

  // ---- Multi-group state ----
  bool has_multi_group_;
  std::vector<GroupParams> groups_;
  // Dense mapping: ``layer_members_[orig]`` is the list of
  // (group_idx, local_layer_id) pairs participating in original layer ``orig``.
  std::vector<std::vector<std::pair<int, int>>> layer_members_;
  int num_original_layers_;

  // Single-group: ``expected_count = num_gpus_``.
  // Multi-group: ``expected_count = slots_per_gpu * num_gpus_`` where
  // ``slots_per_gpu = main_members + swa_slots`` (uniform SWA: 0/1;
  // SWA multi-group: ``swa_layer_members_[orig].size()``).
  void layer_done_callback(int start_layer, int layers_this_batch,
                           int expected_count,
                           nvtxRangeId_t *current_range_id_ptr,
                           bool is_last_batch, const char *next_range_name,
                           nvtxRangeId_t *next_range_id_ptr,
                           int callbacks_per_gpu = 1);

  void init_swa_sidecar_(
      bool has_swa,
      const std::vector<std::vector<torch::Tensor>> &swa_gpu_blocks,
      torch::Tensor swa_cpu_blocks,
      std::map<int, std::vector<std::string>> &swa_ssd_files,
      torch::Tensor swa_gpu_kv_strides_tensor,
      torch::Tensor swa_gpu_block_strides_tensor,
      torch::Tensor swa_gpu_layer_strides_tensor,
      torch::Tensor swa_gpu_chunk_sizes_tensor, int num_layers,
      int iouring_entries, int iouring_flags);

  void launch_swa_h2d_layer_(
      int start_layer, int layers_this_batch, int num_blocks,
      int64_t *swa_gpu_block_ids, int64_t *swa_cpu_block_ids,
      int64_t swa_h2d_cpu_kv_stride_in_bytes,
      int64_t swa_h2d_cpu_layer_stride_in_bytes,
      int64_t swa_cpu_block_stride_in_bytes, int transfer_cta_num,
      bool use_ce_transfer);

  // Per-original-layer H2D for heterogeneous SWA/state groups.
  void launch_swa_mg_h2d_layer_(
      int orig_layer, int num_blocks, int64_t *swa_gpu_block_ids,
      int64_t *swa_cpu_block_ids, int transfer_cta_num, bool use_ce_transfer,
      bool is_mla, const std::string &mla_d2h_mode);

  int swa_slots_for_orig_(int orig_layer, bool swa_active) const;

  // ===== Event polling notification (#199) =====
  // Used when notify_mode == "polling".
  // The polling thread queries per-batch CUDA events and writes eventfds
  // as soon as each batch completes on all GPUs, instead of relying on
  // cudaLaunchHostFunc callbacks.
  enum class NotifyMode { HOSTFUNC, POLLING };
  NotifyMode notify_mode_ = NotifyMode::HOSTFUNC;

  struct PollBatchInfo {
    int start_layer;
    int layers_this_batch;
    std::vector<cudaEvent_t> per_gpu_events;
    bool notified = false;
  };
  std::vector<PollBatchInfo> poll_batches_;
  std::atomic<bool> poll_stop_{false};
  std::atomic<int> poll_next_batch_{0};
  std::thread poll_thread_;

  void notify_layer_batch(int start_layer, int layers_this_batch);
  void event_polling_loop();
  void stop_polling_();
};

} // namespace flexkv
