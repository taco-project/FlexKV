#include "tp_transfer_thread_group.h"
#include "transfer.cuh"
#include <stdexcept>

namespace flexkv {

TPTransferThreadGroup::TPTransferThreadGroup(int num_gpus, 
                                             const std::vector<std::vector<torch::Tensor>>& gpu_blocks)
{
    num_gpus_ = num_gpus;
    int num_layers = gpu_blocks[0].size(); 
    cudaMallocHost((void **)&gpu_blocks_, num_gpus_ * num_layers * sizeof(void*));
    for(int i = 0; i < num_gpus_; ++i) {
        for(int j = 0; j < num_layers; ++j) {
            gpu_blocks_[i * num_layers + j] = gpu_blocks[i][j].data_ptr();
        }
    }
}

TPTransferThreadGroup::~TPTransferThreadGroup() {}

void TPTransferThreadGroup::tp_group_transfer(
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
) {

    std::atomic<bool> failed{false};
    std::string error_msg;
    threads_.clear();
    threads_.reserve(num_gpus_);

    for (int i = 0; i < num_gpus_; ++i) {
        threads_.emplace_back([&, i]() {
            try {
                int num_blocks = dst_block_id_tensors[i].numel();
                int num_layers = layer_granularity;
                // TODO 
                int64_t* dst_block_ids = static_cast<int64_t*>(dst_block_id_tensors[i].data_ptr());
                void** dst_layer_ptrs = static_cast<void**>(gpu_blocks_ + i * num_layers * sizeof(void*));
                int64_t* src_block_ids = static_cast<int64_t*>(src_block_id_tensors[i].data_ptr());
                void** src_layer_ptrs = static_cast<void**>(gpu_blocks_ + i * num_layers * sizeof(void*));

                // 设置当前线程的CUDA device
                at::cuda::CUDAGuard device_guard(dst_block_id_tensors[i].device().index());

                cudaStream_t stream = at::cuda::getCurrentCUDAStream();

                flexkv::transfer_kv_layers(
                    num_blocks, num_layers, dst_block_ids, dst_layer_ptrs,
                    dst_kv_stride_in_bytes[i], dst_chunk_stride_in_bytes[i],
                    src_block_ids, src_layer_ptrs,
                    src_kv_stride_in_bytes[i], src_chunk_stride_in_bytes[i],
                    chunk_size_in_bytes[i], stream, transfer_sms[i],
                    is_host_to_device[i], use_ce_transfer[i]
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