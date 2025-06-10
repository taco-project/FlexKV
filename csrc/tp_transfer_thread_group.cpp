#include "tp_transfer_thread_group.h"
#include "transfer.cuh"
#include <stdexcept>

namespace flexkv {

TPTransferThreadGroup::TPTransferThreadGroup(int num_gpus, 
                                             const std::vector<std::vector<torch::Tensor>>& gpu_blocks,
                                             const std::vector<torch::Tensor>& cpu_blocks,
                                             int dp_group_id)
{
    num_gpus_ = num_gpus;
    int num_layers = gpu_blocks[0].size(); 
    cudaMallocHost((void **)&gpu_blocks_, num_gpus_ * num_layers * sizeof(void*));
    for(int i = 0; i < num_gpus_; ++i) {
        for(int j = 0; j < num_layers; ++j) {
            gpu_blocks_[i * num_layers + j] = gpu_blocks[i][j].data_ptr();
        }
    }
    cudaMallocHost((void **)&cpu_blocks_, num_layers * sizeof(void*));
    for(int j = 0; j < num_layers; ++j) {
        cpu_blocks_[j] = cpu_blocks[j].data_ptr();
    }
    dp_group_id_ = dp_group_id;
    streams_.resize(num_gpus_);
    for(int i = 0; i < num_gpus_; i += 1) {
        cudaSetDevice(dp_group_id * num_gpus_ + i);
        cudaStreamCreate(&streams_[i]);
    }
}

TPTransferThreadGroup::~TPTransferThreadGroup() {}

void TPTransferThreadGroup::tp_group_transfer(
    const torch::Tensor& dst_block_id_tensor,
    const int64_t dst_kv_stride_in_bytes,
    const int64_t dst_chunk_stride_in_bytes,
    const int64_t dst_chunk_size_in_bytes,
    const torch::Tensor& src_block_id_tensor,
    const int64_t src_kv_stride_in_bytes,
    const int64_t src_chunk_stride_in_bytes,
    const int64_t src_chunk_size_in_bytes,
    const int transfer_sms,
    const bool is_host_to_device,
    const bool use_ce_transfer,
    const int layer_id,
    const int layer_granularity
) {

    std::atomic<bool> failed{false};
    std::string error_msg;
    threads_.clear();
    threads_.reserve(num_gpus_);

    for (int i = 0; i < num_gpus_; ++i) {
        threads_.emplace_back([&, i]() {
            try {
                int num_blocks = dst_block_id_tensor.numel();
                int num_layers = layer_granularity;

                int64_t* dst_block_ids = static_cast<int64_t*>(dst_block_id_tensor.data_ptr());
                int64_t* src_block_ids = static_cast<int64_t*>(src_block_id_tensor.data_ptr());
                void** dst_layer_ptrs;
                void** src_layer_ptrs;
                int64_t dst_startoff_inside_chunks;
                int64_t src_startoff_inside_chunks;
                int64_t copy_size_in_bytes;
                if (is_host_to_device) {
                    dst_layer_ptrs = static_cast<void**>(gpu_blocks_ + i * num_layers + layer_id);
                    src_layer_ptrs = static_cast<void**>(cpu_blocks_ + layer_id);
                    dst_startoff_inside_chunks = 0;
                    src_startoff_inside_chunks = i * dst_chunk_size_in_bytes;
                    copy_size_in_bytes = dst_chunk_size_in_bytes;
                } else {
                    dst_layer_ptrs = static_cast<void**>(cpu_blocks_ + layer_id);
                    src_layer_ptrs = static_cast<void**>(gpu_blocks_ + i * num_layers + layer_id);
                    dst_startoff_inside_chunks = i * src_chunk_size_in_bytes;
                    src_startoff_inside_chunks = 0;
                    copy_size_in_bytes = src_chunk_size_in_bytes;
                }
                cudaSetDevice(dp_group_id_ * num_gpus_ + i);
                flexkv::transfer_kv_layers(
                    num_blocks, layer_granularity, dst_block_ids, dst_layer_ptrs,
                    dst_kv_stride_in_bytes, dst_chunk_stride_in_bytes,
                    dst_startoff_inside_chunks, src_block_ids, src_layer_ptrs,
                    src_kv_stride_in_bytes, src_chunk_stride_in_bytes,
                    src_startoff_inside_chunks,
                    copy_size_in_bytes, streams_[i], transfer_sms,
                    is_host_to_device, use_ce_transfer
                );
                cudaError_t err = cudaGetLastError();
                if (err != cudaSuccess) {
                    failed = true;
                    error_msg = cudaGetErrorString(err);
                }
            } catch (const std::exception& e) {
                failed = true;
                error_msg = e.what();
            }
        });
    }

    for (auto& t : threads_) {
        if (t.joinable()) t.join();
    }

    if (failed) {
        throw std::runtime_error("tp_group_transfer failed: " + error_msg);
    }
}

} // namespace flexkv