#pragma once

#include <vector>
#include <thread>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <cuda_runtime.h>
#include <torch/extension.h>

namespace flexkv {

class TPTransferThreadGroup {
public:
    TPTransferThreadGroup(int num_gpus, 
                         const std::vector<std::vector<torch::Tensor>>& gpu_blocks,
                         const std::vector<torch::Tensor>& cpu_blocks,
                         int dp_group_id);
    ~TPTransferThreadGroup();

    void tp_group_transfer(
        const torch::Tensor& dst_block_id_tensors,
        const int64_t dst_kv_stride_in_bytes,
        const int64_t dst_chunk_stride_in_bytes,
        const int64_t dst_chunk_size_in_bytes,
        const torch::Tensor& src_block_id_tensors,
        const int64_t src_kv_stride_in_bytes,
        const int64_t src_chunk_stride_in_bytes,
        const int64_t src_chunk_size_in_bytes,
        const int transfer_sms,
        const bool is_host_to_device,
        const bool use_ce_transfer,
        const int layer_id,
        const int layer_granularity,
        const bool is_mla
    );

private:
    int num_gpus_;
    int dp_group_id_;
    void** gpu_blocks_;
    void** cpu_blocks_;
    std::vector<std::thread> threads_;
    std::vector<cudaStream_t> streams_;
};

} // namespace flexkv