#pragma once

#include <cuda_runtime.h>
#include <fcntl.h>
#include <map>
#include <memory>
#include <string>
#include <torch/extension.h>
#include <vector>
#include <sys/eventfd.h>
#include <unistd.h>
#include <nvtx3/nvToolsExt.h>

#include "gtensor_handler.cuh"
#include "transfer.cuh"
#include "transfer_ssd.h"

namespace flexkv {

class LayerwiseTransferGroup {
public:
  LayerwiseTransferGroup(
      int num_gpus, const std::vector<std::vector<torch::Tensor>> &gpu_blocks,
      torch::Tensor &cpu_blocks,
      std::map<int, std::vector<std::string>> &ssd_files, int dp_group_id,
      int num_layers, torch::Tensor &gpu_kv_strides_tensor,
      torch::Tensor &gpu_block_strides_tensor,
      torch::Tensor &gpu_layer_strides_tensor,
      torch::Tensor &gpu_chunk_sizes_tensor, int iouring_entries,
      int iouring_flags, torch::Tensor &layer_eventfds_tensor, int tp_size);

  ~LayerwiseTransferGroup();

  // Layerwise transfer: SSD->CPU + CPU->GPU
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
      const int64_t cpu_chunk_size_in_bytes, const int transfer_cta_num,
      const bool use_ce_transfer, const int num_layers,
      const int layer_granularity, const bool is_mla,
      const int counter_id = 0);  // Counter set index for triple buffering

private:
  int num_gpus_;
  int dp_group_id_;
  void **gpu_blocks_;
  void *cpu_blocks_;
  int num_tensors_per_gpu_;
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
  // Shape: [tp_size, num_counters, num_layers]
  bool enable_eventfd_;
  int tp_size_;
  int num_counters_;
  int num_layers_;
  std::vector<int> layer_eventfds_;  // Flat array
  int current_counter_id_;  // Current counter set index for this transfer

  void layer_done_callback(int start_layer, int layers_this_batch,
                           nvtxRangeId_t *current_range_id_ptr,
                           bool is_last_batch,
                           const char *next_range_name,
                           nvtxRangeId_t *next_range_id_ptr);
};

} // namespace flexkv
