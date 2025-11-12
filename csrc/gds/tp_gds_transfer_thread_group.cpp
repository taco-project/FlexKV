#include "tp_gds_transfer_thread_group.h"
#include "gds_manager.h"
#include <stdexcept>

namespace flexkv {

TPGDSTransferThreadGroup::TPGDSTransferThreadGroup(
    int num_gpus, 
    const std::vector<std::vector<torch::Tensor>> &gpu_blocks,
    std::map<int, std::vector<std::string>> &ssd_files, 
    int dp_group_id,
    int num_layers,
    torch::Tensor &gpu_kv_strides_tensor,
    torch::Tensor &gpu_block_strides_tensor,
    torch::Tensor &gpu_chunk_sizes_tensor) {
  
  num_gpus_ = num_gpus;
  dp_group_id_ = dp_group_id;
  
  // per-GPU layout parameters
  gpu_kv_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_block_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_chunk_sizes_in_bytes_ = new int64_t[num_gpus];
  
  int64_t* kv_strides_ptr = gpu_kv_strides_tensor.data_ptr<int64_t>();
  int64_t* block_strides_ptr = gpu_block_strides_tensor.data_ptr<int64_t>();
  int64_t* chunk_sizes_ptr = gpu_chunk_sizes_tensor.data_ptr<int64_t>();
  
  for (int i = 0; i < num_gpus; i++) {
    gpu_kv_strides_in_bytes_[i] = kv_strides_ptr[i];
    gpu_block_strides_in_bytes_[i] = block_strides_ptr[i];
    gpu_chunk_sizes_in_bytes_[i] = chunk_sizes_ptr[i];
  }
  
  // Prepare GPU blocks pointers
  cudaMallocHost((void **)&gpu_blocks_, num_gpus_ * num_layers * sizeof(void *));
  
  for (int i = 0; i < num_gpus_; ++i) {
    for (int j = 0; j < num_layers; ++j) {
      gpu_blocks_[i * num_layers + j] = gpu_blocks[i][j].data_ptr();
    }
  }

  // Create GDS managers for each GPU thread
  gds_managers_.resize(num_gpus_);
  for (int i = 0; i < num_gpus_; ++i) {
    gds_managers_[i] = new GDSManager(ssd_files, ssd_files.size(), 1);
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
  
  // Clean up per-GPU layout parameters
  delete[] gpu_kv_strides_in_bytes_;
  delete[] gpu_block_strides_in_bytes_;
  delete[] gpu_chunk_sizes_in_bytes_;
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
        
        int64_t gds_copy_off_inside_chunks;
        int64_t gpu_chunk_size_in_bytes = gpu_chunk_sizes_in_bytes_[i];
        
        if (is_mla) {
          if (!is_read) {
            gds_copy_off_inside_chunks = i * gpu_chunk_size_in_bytes;
          } else {
            gds_copy_off_inside_chunks = 0;
          }
        } else {
          gds_copy_off_inside_chunks = i * gpu_chunk_size_in_bytes;
        }

        int64_t chunk_size = is_mla && !is_read ? gpu_chunk_size_in_bytes / num_gpus_ : gpu_chunk_size_in_bytes;
        
        // Call the transfer_kv_blocks_gds function with multi-file support
        transfer_kv_blocks_gds(
            *gds_managers_[i],              // GDS manager for this GPU
            layer_id_list,                  // Layer IDs to process
            gpu_layer_ptrs_tensor,          // GPU layer pointers
            gds_block_id_tensor,            // GDS block IDs (adjusted for TP)
            gpu_block_id_tensor,            // GPU block IDs
            gpu_kv_strides_in_bytes_[i],    // GPU K-V stride
            gpu_block_strides_in_bytes_[i], // GPU block stride
            gds_layer_stride_in_bytes,      // GDS layer stride
            gds_block_stride_in_bytes,      // GDS block stride
            gds_kv_stride_in_bytes,         // GDS K-V stride
            chunk_size,                     // Chunk size
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