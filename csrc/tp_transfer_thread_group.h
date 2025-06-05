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
    TPTransferThreadGroup(int num_gpus, const std::vector<std::vector<torch::Tensor>>& gpu_blocks);
    ~TPTransferThreadGroup();

    void tp_group_transfer(
        const std::vector<torch::Tensor>& dst_block_id_tensors,
        const std::vector<int64_t>& dst_kv_stride_in_bytes,
        const std::vector<int64_t>& dst_chunk_stride_in_bytes,
        const std::vector<int64_t>& dst_chunk_size_in_bytes,
        const std::vector<torch::Tensor>& src_block_id_tensors,
        const std::vector<int64_t>& src_kv_stride_in_bytes,
        const std::vector<int64_t>& src_chunk_stride_in_bytes,
        const std::vector<int64_t>& src_chunk_size_in_bytes,
        const std::vector<int>& transfer_sms,
        const std::vector<bool>& is_host_to_device,
        const std::vector<bool>& use_ce_transfer,
        int layer_id,
        int layer_granularity
    );

private:
    int num_gpus_;
    void** gpu_blocks_;
    std::vector<std::thread> threads_;
};

} // namespace flexkv