#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <torch/extension.h>
#include <vector>
#include <string>
#include <map>
#include <functional>
#include <future>
#include "../device_thread_pool.h"
#include "../gtensor_handler.cuh"

// Forward declaration
class GDSManager;

namespace flexkv {

class TPGDSTransferThreadGroup {
public:
  TPGDSTransferThreadGroup(
      int num_gpus,
      const std::vector<int64_t> &gpu_block_ptrs_flat,
      int num_tensors_per_gpu,
      std::map<int, std::vector<std::string>> &ssd_files, 
      int num_layers,
      const std::vector<int64_t> &gpu_kv_strides_in_bytes,
      const std::vector<int64_t> &gpu_block_strides_in_bytes,
      const std::vector<int64_t> &gpu_layer_strides_in_bytes,
      const std::vector<int64_t> &gpu_chunk_sizes_in_bytes,
      const std::vector<int64_t> &gpu_device_ids);
  ~TPGDSTransferThreadGroup();

  void tp_group_transfer(
      const torch::Tensor &gpu_block_id_tensor,
      const torch::Tensor &ssd_block_id_tensor,
      const int64_t ssd_layer_stride_in_bytes,
      const int64_t ssd_kv_stride_in_bytes,
      const int64_t ssd_block_stride_in_bytes,
      const int64_t ssd_chunk_size_in_bytes,
      const int64_t num_blocks_per_file,
      const bool is_read,  // true for SSD->GPU, false for GPU->SSD
      const int layer_id,
      const int layer_granularity,
      const int kv_dim,
      const int num_kv_heads = 1);

private:
  using Task = std::function<void()>;
  std::future<void> enqueue_for_gpu(int gpu_idx, Task task);

  int num_gpus_;
  std::vector<int> gpu_device_ids_;
  void **gpu_blocks_;
  int num_tensors_per_gpu_;
  
  int64_t *gpu_kv_strides_in_bytes_;
  int64_t *gpu_block_strides_in_bytes_;
  int64_t *gpu_layer_strides_in_bytes_;
  int64_t *gpu_chunk_sizes_in_bytes_;
  
  BackendType backend_type_;
  std::vector<GTensorHandler> gpu_tensor_handlers_;
  
  std::vector<GDSManager*> gds_managers_;

  // Was a hand-rolled pool here too (threads_, streams_, queues_, mtxs_,
  // cvs_, stop_pool_) -- the third copy of the same six members. See
  // device_thread_pool.h for why the copies were a problem: they drifted, and
  // this one leaked nothing only because its destructor happened to be the
  // one that got fixed.
  std::unique_ptr<DeviceThreadPool> pool_;
};

} // namespace flexkv 