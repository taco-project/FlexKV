#include "tp_transfer_thread_group.h"
#include "transfer.cuh"
#include <stdexcept>

namespace flexkv {

TPTransferThreadGroup::TPTransferThreadGroup(
    int num_gpus, const std::vector<std::vector<torch::Tensor>> &gpu_blocks,
    torch::Tensor &cpu_blocks, int dp_group_id) {
  num_gpus_ = num_gpus;
  int num_layers = gpu_blocks[0].size();
  cudaMallocHost((void **)&gpu_blocks_,
                 num_gpus_ * num_layers * sizeof(void *));
  for (int i = 0; i < num_gpus_; ++i) {
    for (int j = 0; j < num_layers; ++j) {
      gpu_blocks_[i * num_layers + j] = gpu_blocks[i][j].data_ptr();
    }
  }

  cpu_blocks_ = cpu_blocks.data_ptr();

  dp_group_id_ = dp_group_id;
  streams_.resize(num_gpus_);
  for (int i = 0; i < num_gpus_; i += 1) {
    cudaSetDevice(dp_group_id * num_gpus_ + i);
    cudaStreamCreate(&streams_[i]);
  }
}

TPTransferThreadGroup::~TPTransferThreadGroup() {}

void TPTransferThreadGroup::tp_group_transfer(
    const torch::Tensor &gpu_block_id_tensor,
    const int64_t gpu_kv_stride_in_bytes,
    const int64_t gpu_block_stride_in_bytes,
    const int64_t gpu_chunk_size_in_bytes,
    const torch::Tensor &cpu_block_id_tensor,
    const int64_t cpu_kv_stride_in_bytes,
    const int64_t cpu_layer_stride_in_bytes,
    const int64_t cpu_block_stride_in_bytes,
    const int64_t cpu_chunk_size_in_bytes, const int transfer_sms,
    const bool is_host_to_device, const bool use_ce_transfer,
    const int layer_id, const int layer_granularity, const bool is_mla) {

  std::atomic<bool> failed{false};
  std::string error_msg;
  threads_.clear();
  threads_.reserve(num_gpus_);

  for (int i = 0; i < num_gpus_; ++i) {
    threads_.emplace_back([&, i]() {
      try {
        int num_blocks = gpu_block_id_tensor.numel();
        int num_layers = layer_granularity;

        int64_t *gpu_block_ids =
            static_cast<int64_t *>(gpu_block_id_tensor.data_ptr());
        int64_t *cpu_block_ids =
            static_cast<int64_t *>(cpu_block_id_tensor.data_ptr());
        void **gpu_layer_ptrs =
            static_cast<void **>(gpu_blocks_ + i * num_layers + layer_id);
        void *cpu_ptr = cpu_blocks_;
        int64_t cpu_startoff_inside_chunks =
            is_mla ? 0 : i * gpu_chunk_size_in_bytes;
        cudaSetDevice(dp_group_id_ * num_gpus_ + i);
        flexkv::transfer_kv_blocks(
            num_blocks, layer_id, layer_granularity, gpu_block_ids,
            gpu_layer_ptrs, gpu_kv_stride_in_bytes, gpu_block_stride_in_bytes,
            cpu_block_ids, cpu_ptr, cpu_kv_stride_in_bytes,
            cpu_layer_stride_in_bytes, cpu_block_stride_in_bytes,
            cpu_startoff_inside_chunks, gpu_chunk_size_in_bytes, streams_[i],
            transfer_sms, is_host_to_device, use_ce_transfer, is_mla);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
          failed = true;
          error_msg = cudaGetErrorString(err);
        }
      } catch (const std::exception &e) {
        failed = true;
        error_msg = e.what();
      }
    });
  }

  for (auto &t : threads_) {
    if (t.joinable())
      t.join();
  }

  if (failed) {
    throw std::runtime_error("tp_group_transfer failed: " + error_msg);
  }
}

} // namespace flexkv
