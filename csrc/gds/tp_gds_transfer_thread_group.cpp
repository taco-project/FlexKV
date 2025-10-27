#include "tp_gds_transfer_thread_group.h"
#include "gds_manager.h"
#include <stdexcept>

namespace flexkv {

TPGDSTransferThreadGroup::TPGDSTransferThreadGroup(
    int num_gpus, 
    const std::vector<std::vector<torch::Tensor>> &gpu_blocks,
    const std::vector<std::string> &gds_file_paths, 
    int dp_group_id) {
  
  num_gpus_ = num_gpus;
  dp_group_id_ = dp_group_id;
  gds_file_paths_ = gds_file_paths;
  
  // Prepare GPU blocks pointers
  int num_layers = gpu_blocks[0].size();
  cudaMallocHost((void **)&gpu_blocks_, num_gpus_ * num_layers * sizeof(void *));
  
  for (int i = 0; i < num_gpus_; ++i) {
    for (int j = 0; j < num_layers; ++j) {
      gpu_blocks_[i * num_layers + j] = gpu_blocks[i][j].data_ptr();
    }
  }

  // Create GDS managers for each GPU thread
  gds_managers_.resize(num_gpus_);
  for (int i = 0; i < num_gpus_; ++i) {
    gds_managers_[i] = new GDSManager(gds_file_paths_);
    if (!gds_managers_[i]->is_ready()) {
      throw std::runtime_error("Failed to initialize GDS Manager for GPU " + std::to_string(i) + 
                              ": " + gds_managers_[i]->get_last_error());
    }
  }

  queues_.resize(num_gpus_);
  mtxs_   = std::vector<std::mutex>(num_gpus_);
  cvs_    = std::vector<std::condition_variable>(num_gpus_);

  // Create CUDA streams for each GPU
  streams_.resize(num_gpus_);
  for (int i = 0; i < num_gpus_; ++i) {
    cudaSetDevice(dp_group_id_ * num_gpus_ + i);
    cudaStreamCreate(&streams_[i]);
  }

  // Create the thread pool
  stop_pool_ = false;
  for (int i = 0; i < num_gpus_; ++i) {
    threads_.emplace_back([this, i]() {
      int device_id = dp_group_id_ * num_gpus_ + i;
      cudaSetDevice(device_id);  // only once

      while (true) {
        Task task;
        {
          std::unique_lock<std::mutex> lk(mtxs_[i]);
          cvs_[i].wait(lk, [&]{ return stop_pool_ || !queues_[i].empty(); });
          if (stop_pool_ && queues_[i].empty()) return;

          task = std::move(queues_[i].front());
          queues_[i].pop();
        }
        task();  // 
      }
    });
  }
}

TPGDSTransferThreadGroup::~TPGDSTransferThreadGroup() {
  stop_pool_ = true;
  for (auto& cv : cvs_) cv.notify_all();
  for (auto& t : threads_) if (t.joinable()) t.join();

  // Clean up GDS managers
  for (auto* manager : gds_managers_) {
    delete manager;
  }
  
  // Clean up CUDA streams
  for (int i = 0; i < streams_.size(); ++i) {
    cudaSetDevice(dp_group_id_ * num_gpus_ + i);
    cudaStreamDestroy(streams_[i]);
  }
  
  // Clean up GPU blocks pointers
  if (gpu_blocks_) {
    cudaFreeHost(gpu_blocks_);
  }
}

std::future<void> TPGDSTransferThreadGroup::enqueue_for_gpu(int gpu_idx, Task task) {
  auto pkg = std::make_shared<std::packaged_task<void()>>(std::move(task));
  auto fut = pkg->get_future();
  {
      std::lock_guard<std::mutex> lk(mtxs_[gpu_idx]);
      queues_[gpu_idx].emplace([pkg]{ (*pkg)(); });
  }
  cvs_[gpu_idx].notify_one();
  return fut;
}

void TPGDSTransferThreadGroup::tp_group_transfer(
    const torch::Tensor &gpu_block_id_tensor,
    const torch::Tensor &gds_block_id_tensor,
    const int64_t gpu_kv_stride_in_bytes,
    const int64_t gpu_block_stride_in_bytes,
    const int64_t gpu_chunk_size_in_bytes,
    const int64_t gds_layer_stride_in_bytes,
    const int64_t gds_kv_stride_in_bytes,
    const int64_t gds_block_stride_in_bytes,
    const int64_t gds_chunk_size_in_bytes,
    const int64_t num_blocks_per_file,
    const bool is_read,
    const int layer_id,
    const int layer_granularity, 
    const bool is_mla) {

  std::atomic<bool> failed{false};
  std::string error_msg;
  std::vector<std::future<void>> futures;
  futures.reserve(num_gpus_);

  for (int i = 0; i < num_gpus_; ++i) {
    futures.emplace_back(enqueue_for_gpu(i, [&, i]() {
      try {
        // Prepare layer ID list for this specific layer range
        torch::Tensor layer_id_list = torch::arange(layer_id, layer_id + layer_granularity, 
                                                    torch::TensorOptions().dtype(torch::kInt32));
        
        // Prepare GPU layer pointers for this GPU and layer range
        void **gpu_layer_ptrs = static_cast<void **>(gpu_blocks_ + i * layer_granularity + layer_id);
        torch::Tensor gpu_layer_ptrs_tensor = torch::from_blob(
            gpu_layer_ptrs, {layer_granularity}, torch::TensorOptions().dtype(torch::kInt64));
        
        int64_t gds_copy_off_inside_chunks = 0;
        if (!is_mla && num_gpus_ > 1) {
          gds_copy_off_inside_chunks = i * gpu_chunk_size_in_bytes;
        }
        
        // Call the transfer_kv_blocks_gds function with multi-file support
        transfer_kv_blocks_gds(
            *gds_managers_[i],              // GDS manager for this GPU
            gds_file_paths_,                // GDS file paths list
            layer_id_list,                  // Layer IDs to process
            gpu_layer_ptrs_tensor,          // GPU layer pointers
            gds_block_id_tensor,            // GDS block IDs (adjusted for TP)
            gpu_block_id_tensor,            // GPU block IDs
            gpu_kv_stride_in_bytes,         // GPU K-V stride
            gds_layer_stride_in_bytes,      // GDS layer stride
            gds_block_stride_in_bytes,      // GDS block stride
            gds_kv_stride_in_bytes,         // GDS K-V stride
            gpu_chunk_size_in_bytes,        // Block size
            gds_copy_off_inside_chunks,     // GDS copy off inside chunks
            num_blocks_per_file,            // Blocks per file
            layer_granularity,              // Total layers
            is_read,                        // Read or write
            false,                          // Verbose logging
            is_mla                          // MLA
        );
        
        // Check for CUDA errors
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
          failed = true;
          error_msg = cudaGetErrorString(err);
        }
      } catch (const std::exception &e) {
        failed = true;
        error_msg = e.what();
      }
    }));
  }

  for (auto &f : futures) {
    f.get();
  }

  if (failed) {
    throw std::runtime_error("tp_gds_group_transfer failed: " + error_msg);
  }
}

} // namespace flexkv 