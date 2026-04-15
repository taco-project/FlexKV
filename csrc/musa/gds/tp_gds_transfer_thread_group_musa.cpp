/*
 * MUSA TP GDS Transfer Thread Group — mirrors csrc/gds/tp_gds_transfer_thread_group.cpp.
 * Uses musa* APIs and GDSManagerMusa. Compile/link with MUSA SDK and muFile.
 */
#include "tp_gds_transfer_thread_group_musa.h"
#include "gds_manager_musa.h"
#include <stdexcept>

namespace flexkv {

TPGDSTransferThreadGroupMusa::TPGDSTransferThreadGroupMusa(
    int num_gpus,
    const std::vector<std::vector<torch::Tensor>>& gpu_blocks,
    std::map<int, std::vector<std::string>>& ssd_files,
    int dp_group_id,
    int num_layers,
    torch::Tensor& gpu_kv_strides_tensor,
    torch::Tensor& gpu_block_strides_tensor,
    torch::Tensor& gpu_layer_strides_tensor,
    torch::Tensor& gpu_chunk_sizes_tensor) {

  num_gpus_ = num_gpus;
  dp_group_id_ = dp_group_id;

  gpu_kv_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_block_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_layer_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_chunk_sizes_in_bytes_ = new int64_t[num_gpus];

  int64_t* kv_strides_ptr = gpu_kv_strides_tensor.data_ptr<int64_t>();
  int64_t* block_strides_ptr = gpu_block_strides_tensor.data_ptr<int64_t>();
  int64_t* layer_strides_ptr = gpu_layer_strides_tensor.data_ptr<int64_t>();
  int64_t* chunk_sizes_ptr = gpu_chunk_sizes_tensor.data_ptr<int64_t>();

  for (int i = 0; i < num_gpus; i++) {
    gpu_kv_strides_in_bytes_[i] = kv_strides_ptr[i];
    gpu_block_strides_in_bytes_[i] = block_strides_ptr[i];
    gpu_layer_strides_in_bytes_[i] = layer_strides_ptr[i];
    gpu_chunk_sizes_in_bytes_[i] = chunk_sizes_ptr[i];
  }

  num_tensors_per_gpu_ = gpu_blocks[0].size();
  musaMallocHost((void**)&gpu_blocks_, num_gpus_ * num_tensors_per_gpu_ * sizeof(void*));

  for (int i = 0; i < num_gpus_; ++i) {
    for (int j = 0; j < num_tensors_per_gpu_; ++j) {
      gpu_blocks_[i * num_tensors_per_gpu_ + j] = gpu_blocks[i][j].data_ptr();
    }
  }

  if (num_tensors_per_gpu_ == 1) {
    backend_type_ = BackendType::TRTLLM;
  } else if (num_tensors_per_gpu_ == num_layers) {
    backend_type_ = BackendType::VLLM;
  } else if (num_tensors_per_gpu_ == num_layers * 2) {
    backend_type_ = BackendType::SGLANG;
  } else {
    throw std::runtime_error("Unsupported GPU block type: " + std::to_string(num_tensors_per_gpu_));
  }

  gpu_tensor_handlers_.reserve(num_gpus_);
  for (int i = 0; i < num_gpus_; i++) {
    int64_t** gpu_blocks_ptr = reinterpret_cast<int64_t**>(gpu_blocks_ + i * num_tensors_per_gpu_);
    gpu_tensor_handlers_.emplace_back(
        backend_type_, gpu_blocks_ptr, num_layers,
        gpu_kv_strides_in_bytes_[i], gpu_block_strides_in_bytes_[i],
        gpu_layer_strides_in_bytes_[i]);
  }

  gds_managers_.resize(num_gpus_);
  for (int i = 0; i < num_gpus_; ++i) {
    gds_managers_[i] = new GDSManagerMusa(ssd_files, ssd_files.size(), 1);
    if (!gds_managers_[i]->is_ready()) {
      throw std::runtime_error("Failed to initialize GDS Manager (MUSA) for GPU " + std::to_string(i) +
                              ": " + gds_managers_[i]->get_last_error());
    }
  }

  queues_.resize(num_gpus_);
  mtxs_ = std::vector<std::mutex>(num_gpus_);
  cvs_ = std::vector<std::condition_variable>(num_gpus_);

  streams_.resize(num_gpus_);
  for (int i = 0; i < num_gpus_; ++i) {
    musaSetDevice(dp_group_id_ * num_gpus_ + i);
    musaStreamCreate(&streams_[i]);
  }

  stop_pool_ = false;
  for (int i = 0; i < num_gpus_; ++i) {
    threads_.emplace_back([this, i]() {
      int device_id = dp_group_id_ * num_gpus_ + i;
      musaSetDevice(device_id);

      while (true) {
        Task task;
        {
          std::unique_lock<std::mutex> lk(mtxs_[i]);
          cvs_[i].wait(lk, [&] { return stop_pool_ || !queues_[i].empty(); });
          if (stop_pool_ && queues_[i].empty()) return;
          task = std::move(queues_[i].front());
          queues_[i].pop();
        }
        task();
      }
    });
  }
}

TPGDSTransferThreadGroupMusa::~TPGDSTransferThreadGroupMusa() {
  stop_pool_ = true;
  for (auto& cv : cvs_) cv.notify_all();
  for (auto& t : threads_) if (t.joinable()) t.join();

  for (auto* manager : gds_managers_) {
    delete manager;
  }

  for (size_t i = 0; i < streams_.size(); ++i) {
    musaSetDevice(dp_group_id_ * num_gpus_ + static_cast<int>(i));
    musaStreamDestroy(streams_[i]);
  }

  if (gpu_blocks_) {
    musaFreeHost(gpu_blocks_);
  }

  gpu_tensor_handlers_.clear();
  delete[] gpu_kv_strides_in_bytes_;
  delete[] gpu_block_strides_in_bytes_;
  delete[] gpu_layer_strides_in_bytes_;
  delete[] gpu_chunk_sizes_in_bytes_;
}

std::future<void> TPGDSTransferThreadGroupMusa::enqueue_for_gpu(int gpu_idx, Task task) {
  auto pkg = std::make_shared<std::packaged_task<void()>>(std::move(task));
  auto fut = pkg->get_future();
  {
    std::lock_guard<std::mutex> lk(mtxs_[gpu_idx]);
    queues_[gpu_idx].emplace([pkg] { (*pkg)(); });
  }
  cvs_[gpu_idx].notify_one();
  return fut;
}

void TPGDSTransferThreadGroupMusa::tp_group_transfer(
    const torch::Tensor& gpu_block_id_tensor,
    const torch::Tensor& ssd_block_id_tensor,
    const int64_t ssd_layer_stride_in_bytes,
    const int64_t ssd_kv_stride_in_bytes,
    const int64_t ssd_block_stride_in_bytes,
    const int64_t ssd_tp_stride_in_bytes,
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
        torch::Tensor layer_id_list = torch::arange(layer_id, layer_id + layer_granularity,
                                                    torch::TensorOptions().dtype(torch::kInt32));
        int64_t ssd_copy_off_inside_chunks;
        int64_t gpu_chunk_size_in_bytes = gpu_chunk_sizes_in_bytes_[i];
        if (is_mla) {
            ssd_copy_off_inside_chunks = 0;
        } else {
          ssd_copy_off_inside_chunks = i * ssd_tp_stride_in_bytes;
        }

        int64_t chunk_size = gpu_chunk_size_in_bytes;
        switch (backend_type_) {
          case BackendType::VLLM:
            flexkv::transfer_kv_blocks_gds_musa<BackendType::VLLM>(
                *gds_managers_[i], layer_id_list, gpu_tensor_handlers_[i],
                ssd_block_id_tensor, gpu_block_id_tensor, ssd_layer_stride_in_bytes,
                ssd_block_stride_in_bytes, ssd_kv_stride_in_bytes, chunk_size,
                ssd_copy_off_inside_chunks, ssd_tp_stride_in_bytes, i, num_blocks_per_file, layer_granularity,
                is_read, false, is_mla);
            break;
          case BackendType::TRTLLM:
            flexkv::transfer_kv_blocks_gds_musa<BackendType::TRTLLM>(
                *gds_managers_[i], layer_id_list, gpu_tensor_handlers_[i],
                ssd_block_id_tensor, gpu_block_id_tensor, ssd_layer_stride_in_bytes,
                ssd_block_stride_in_bytes, ssd_kv_stride_in_bytes, chunk_size,
                ssd_copy_off_inside_chunks, ssd_tp_stride_in_bytes, i, num_blocks_per_file, layer_granularity,
                is_read, false, is_mla);
            break;
          case BackendType::SGLANG:
            flexkv::transfer_kv_blocks_gds_musa<BackendType::SGLANG>(
                *gds_managers_[i], layer_id_list, gpu_tensor_handlers_[i],
                ssd_block_id_tensor, gpu_block_id_tensor, ssd_layer_stride_in_bytes,
                ssd_block_stride_in_bytes, ssd_kv_stride_in_bytes, chunk_size,
                ssd_copy_off_inside_chunks, ssd_tp_stride_in_bytes, i, num_blocks_per_file, layer_granularity,
                is_read, false, is_mla);
            break;
        }

        musaError_t err = musaGetLastError();
        if (err != musaSuccess) {
          failed = true;
          error_msg = musaGetErrorString(err);
        }
      } catch (const std::exception& e) {
        failed = true;
        error_msg = e.what();
      }
    }));
  }

  for (auto& f : futures) {
    f.get();
  }

  if (failed) {
    throw std::runtime_error("tp_gds_group_transfer (MUSA) failed: " + error_msg);
  }
}

} // namespace flexkv
