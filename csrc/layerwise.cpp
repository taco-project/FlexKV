#include "layerwise.h"
#include <fcntl.h>
#include <stdexcept>

namespace flexkv {

LayerwiseTransferGroup::LayerwiseTransferGroup(
    int num_gpus, const std::vector<std::vector<torch::Tensor>> &gpu_blocks,
    torch::Tensor &cpu_blocks,
    std::map<int, std::vector<std::string>> &ssd_files, int dp_group_id,
    int num_layers, torch::Tensor &gpu_kv_strides_tensor,
    torch::Tensor &gpu_block_strides_tensor,
    torch::Tensor &gpu_layer_strides_tensor,
    torch::Tensor &gpu_chunk_sizes_tensor, int iouring_entries,
    int iouring_flags) {

  num_gpus_ = num_gpus;

  gpu_kv_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_block_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_layer_strides_in_bytes_ = new int64_t[num_gpus];
  gpu_chunk_sizes_in_bytes_ = new int64_t[num_gpus];

  int64_t *kv_strides_ptr = gpu_kv_strides_tensor.data_ptr<int64_t>();
  int64_t *block_strides_ptr = gpu_block_strides_tensor.data_ptr<int64_t>();
  int64_t *layer_strides_ptr = gpu_layer_strides_tensor.data_ptr<int64_t>();
  int64_t *chunk_sizes_ptr = gpu_chunk_sizes_tensor.data_ptr<int64_t>();

  for (int i = 0; i < num_gpus; i++) {
    gpu_kv_strides_in_bytes_[i] = kv_strides_ptr[i];
    gpu_block_strides_in_bytes_[i] = block_strides_ptr[i];
    gpu_chunk_sizes_in_bytes_[i] = chunk_sizes_ptr[i];
    gpu_layer_strides_in_bytes_[i] = layer_strides_ptr[i];
  }

  queues_.resize(num_gpus_);
  mtxs_ = std::vector<std::mutex>(num_gpus_);
  cvs_ = std::vector<std::condition_variable>(num_gpus_);

  num_tensors_per_gpu_ = gpu_blocks[0].size();
  cudaMallocHost((void **)&gpu_blocks_,
                 num_gpus_ * num_tensors_per_gpu_ * sizeof(void *));
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
    throw std::runtime_error("Unsupported GPU block type: " +
                             std::to_string(num_tensors_per_gpu_));
  }

  gpu_tensor_handlers_.reserve(num_gpus_);
  for (int i = 0; i < num_gpus_; i++) {
    int64_t **gpu_blocks_ptr =
        reinterpret_cast<int64_t **>(gpu_blocks_ + i * num_tensors_per_gpu_);
    gpu_tensor_handlers_.emplace_back(
        backend_type_, gpu_blocks_ptr, num_layers, gpu_kv_strides_in_bytes_[i],
        gpu_block_strides_in_bytes_[i], gpu_layer_strides_in_bytes_[i]);
  }

  cpu_blocks_ = cpu_blocks.data_ptr();

  dp_group_id_ = dp_group_id;
  streams_.resize(num_gpus_);
  for (int i = 0; i < num_gpus_; i += 1) {
    cudaSetDevice(dp_group_id * num_gpus_ + i);
    cudaStreamCreate(&streams_[i]);
  }

  // Initialize SSD IO context if ssd_files is not empty
  enable_ssd_ = !ssd_files.empty();
  if (enable_ssd_) {
    ioctx_ = std::make_unique<SSDIOCTX>(ssd_files, ssd_files.size(),
                                        iouring_entries, iouring_flags);
  }

  // Create the thread pool
  stop_pool_ = false;
  for (int i = 0; i < num_gpus_; ++i) {
    threads_.emplace_back([this, i]() {
      int device_id = dp_group_id_ * num_gpus_ + i;
      cudaSetDevice(device_id);

      while (true) {
        Task task;
        {
          std::unique_lock<std::mutex> lk(mtxs_[i]);
          cvs_[i].wait(lk, [&] { return stop_pool_ || !queues_[i].empty(); });
          if (stop_pool_ && queues_[i].empty())
            return;

          task = std::move(queues_[i].front());
          queues_[i].pop();
        }
        task();
      }
    });
  }
}

LayerwiseTransferGroup::~LayerwiseTransferGroup() {
  stop_pool_ = true;
  for (auto &cv : cvs_)
    cv.notify_all();
  for (auto &t : threads_)
    if (t.joinable())
      t.join();

  cudaFreeHost(gpu_blocks_);

  gpu_tensor_handlers_.clear();
  delete[] gpu_kv_strides_in_bytes_;
  delete[] gpu_block_strides_in_bytes_;
  delete[] gpu_layer_strides_in_bytes_;
  delete[] gpu_chunk_sizes_in_bytes_;
}

std::future<void> LayerwiseTransferGroup::enqueue_for_gpu(int gpu_idx,
                                                          Task task) {
  auto pkg = std::make_shared<std::packaged_task<void()>>(std::move(task));
  auto fut = pkg->get_future();
  {
    std::lock_guard<std::mutex> lk(mtxs_[gpu_idx]);
    queues_[gpu_idx].emplace([pkg] { (*pkg)(); });
  }
  cvs_[gpu_idx].notify_one();
  return fut;
}

void LayerwiseTransferGroup::layerwise_transfer(
    const torch::Tensor &ssd_block_ids, const torch::Tensor &cpu_block_ids_d2h,
    const int64_t ssd_layer_stride_in_bytes,
    const int64_t ssd_kv_stride_in_bytes, const int num_blocks_per_file,
    const int round_robin, const int num_threads_per_device,
    const torch::Tensor &gpu_block_id_tensor,
    const torch::Tensor &cpu_block_id_tensor,
    const int64_t cpu_kv_stride_in_bytes,
    const int64_t cpu_layer_stride_in_bytes,
    const int64_t cpu_block_stride_in_bytes,
    const int64_t cpu_chunk_size_in_bytes, const int transfer_sms,
    const bool use_ce_transfer, const int layer_id, const int layer_granularity,
    const bool is_mla) {

  // Step 1: SSD -> CPU transfer (if ssd_block_ids is not empty)
  if (enable_ssd_ && ssd_block_ids.numel() > 0) {
    torch::Tensor layer_id_list =
        torch::arange(layer_id, layer_id + layer_granularity,
                      torch::TensorOptions().dtype(torch::kInt32));
    transfer_kv_blocks_ssd(
        *ioctx_, layer_id_list, reinterpret_cast<int64_t>(cpu_blocks_),
        ssd_block_ids, cpu_block_ids_d2h, cpu_layer_stride_in_bytes,
        cpu_kv_stride_in_bytes, ssd_layer_stride_in_bytes,
        ssd_kv_stride_in_bytes, cpu_chunk_size_in_bytes,
        cpu_block_stride_in_bytes,
        true, // is_read: SSD -> CPU
        num_blocks_per_file, round_robin, num_threads_per_device, is_mla);
  }

  // Step 2: CPU -> GPU transfer
  std::atomic<bool> failed{false};
  std::string error_msg;
  std::vector<std::future<void>> futures;
  futures.reserve(num_gpus_);

  for (int i = 0; i < num_gpus_; ++i) {
    futures.emplace_back(enqueue_for_gpu(i, [&, i]() {
      try {
        int num_blocks = gpu_block_id_tensor.numel();

        int64_t *gpu_block_ids =
            static_cast<int64_t *>(gpu_block_id_tensor.data_ptr());
        int64_t *cpu_block_ids =
            static_cast<int64_t *>(cpu_block_id_tensor.data_ptr());
        void *cpu_ptr = cpu_blocks_;
        int64_t cpu_startoff_inside_chunks = i * gpu_chunk_sizes_in_bytes_[i];
        if (is_mla) {
          cpu_startoff_inside_chunks = 0;
        }
        int64_t gpu_startoff_inside_chunks = 0;
        int64_t chunk_size = gpu_chunk_sizes_in_bytes_[i];

        // Dispatch to the appropriate template based on backend type
        switch (backend_type_) {
        case BackendType::VLLM:
          flexkv::transfer_kv_blocks<BackendType::VLLM>(
              num_blocks, layer_id, layer_granularity, gpu_block_ids,
              gpu_tensor_handlers_[i], gpu_startoff_inside_chunks,
              cpu_block_ids, cpu_ptr, cpu_kv_stride_in_bytes,
              cpu_layer_stride_in_bytes, cpu_block_stride_in_bytes,
              cpu_startoff_inside_chunks, chunk_size, streams_[i], transfer_sms,
              true, use_ce_transfer, is_mla);
          break;
        case BackendType::TRTLLM:
          flexkv::transfer_kv_blocks<BackendType::TRTLLM>(
              num_blocks, layer_id, layer_granularity, gpu_block_ids,
              gpu_tensor_handlers_[i], gpu_startoff_inside_chunks,
              cpu_block_ids, cpu_ptr, cpu_kv_stride_in_bytes,
              cpu_layer_stride_in_bytes, cpu_block_stride_in_bytes,
              cpu_startoff_inside_chunks, chunk_size, streams_[i], transfer_sms,
              true, use_ce_transfer, is_mla);
          break;
        case BackendType::SGLANG:
          flexkv::transfer_kv_blocks<BackendType::SGLANG>(
              num_blocks, layer_id, layer_granularity, gpu_block_ids,
              gpu_tensor_handlers_[i], gpu_startoff_inside_chunks,
              cpu_block_ids, cpu_ptr, cpu_kv_stride_in_bytes,
              cpu_layer_stride_in_bytes, cpu_block_stride_in_bytes,
              cpu_startoff_inside_chunks, chunk_size, streams_[i], transfer_sms,
              true, use_ce_transfer, is_mla);
          break;
        }

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
    throw std::runtime_error("layerwise_transfer failed: " + error_msg);
  }
}

} // namespace flexkv
