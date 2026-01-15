#include "layerwise.h"
#include <atomic>
#include <cstdio>
#include <fcntl.h>
#include <stdexcept>
#include <sys/eventfd.h>
#include <unistd.h>

namespace flexkv {

struct LayerCallbackData {
  int start_layer;
  int layers_this_batch;
  int num_gpus;
  std::atomic<int> *counter;
  // Eventfd info for notification
  bool enable_eventfd;
  int tp_size;
  int num_layers;
  int *layer_eventfds;  // Pointer to eventfds array for current counter set
};

static void CUDART_CB layer_done_host_callback(void *userData) {
  LayerCallbackData *data = static_cast<LayerCallbackData *>(userData);
  int completed = data->counter->fetch_add(1) + 1;
  if (completed == data->num_gpus) {
    // Notify via eventfd when all GPUs complete this layer batch
    if (data->enable_eventfd && data->layer_eventfds != nullptr) {
      // Signal each tp_rank's eventfd for completed layers
      for (int layer = data->start_layer; 
           layer < data->start_layer + data->layers_this_batch; ++layer) {
        for (int tp_rank = 0; tp_rank < data->tp_size; ++tp_rank) {
          int fd = data->layer_eventfds[tp_rank * data->num_layers + layer];
          if (fd >= 0) {
            // Write 2 to support both get_key_buffer and get_value_buffer waits
            uint64_t val = 2;
            ssize_t ret = write(fd, &val, sizeof(val));
            // if (ret == sizeof(val)) {
            //   fprintf(stderr, "[LayerwiseTransfer] eventfd_write SUCCESS: tp_rank=%d, layer=%d, fd=%d, val=%lu\n", 
            //          tp_rank, layer, fd, val);
            // } else {
            //   fprintf(stderr, 
            //       "[LayerwiseTransfer] Warning: eventfd_write failed for "
            //       "tp_rank %d, layer %d, fd %d, errno=%d\n", tp_rank, layer, fd, errno);
            // }
            // fflush(stderr);
          }
        }
      }
    }
    // else {
    //   fprintf(stderr, "[LayerwiseTransfer] WARNING: eventfd disabled or null! enable=%d, ptr=%p\n",
    //           data->enable_eventfd, (void*)data->layer_eventfds);
    //   fflush(stderr);
    // }    
    
    // fprintf(stderr,
    //     "[LayerwiseTransfer] All %d GPUs: Layers [%d, %d) transfer completed\n",
    //     data->num_gpus, data->start_layer,
    //     data->start_layer + data->layers_this_batch);
    delete data->counter;
  }
  delete data;
}

LayerwiseTransferGroup::LayerwiseTransferGroup(
    int num_gpus, const std::vector<std::vector<torch::Tensor>> &gpu_blocks,
    torch::Tensor &cpu_blocks,
    std::map<int, std::vector<std::string>> &ssd_files, int dp_group_id,
    int num_layers, torch::Tensor &gpu_kv_strides_tensor,
    torch::Tensor &gpu_block_strides_tensor,
    torch::Tensor &gpu_layer_strides_tensor,
    torch::Tensor &gpu_chunk_sizes_tensor, int iouring_entries,
    int iouring_flags, torch::Tensor &layer_eventfds_tensor, int tp_size) {

  num_gpus_ = num_gpus;
  num_layers_ = num_layers;
  tp_size_ = tp_size;
  current_counter_id_ = 0;

  // Initialize eventfds
  enable_eventfd_ = (layer_eventfds_tensor.numel() > 0);
  if (enable_eventfd_) {
    // layer_eventfds_tensor layout: [num_counters, tp_size, num_layers]
    // Index formula: counter_id * tp_size * num_layers + tp_rank * num_layers + layer
    int total_fds = layer_eventfds_tensor.numel();
    num_counters_ = total_fds / (tp_size * num_layers);
    
    int32_t *fds_ptr = layer_eventfds_tensor.data_ptr<int32_t>();
    layer_eventfds_.assign(fds_ptr, fds_ptr + total_fds);
    
    printf("[LayerwiseTransferGroup] Initialized with eventfds: "
           "tp_size=%d, num_counters=%d, num_layers=%d, total_fds=%d\n",
           tp_size_, num_counters_, num_layers_, total_fds);
  } else {
    num_counters_ = 0;
    printf("[LayerwiseTransferGroup] Initialized without eventfds\n");
  }

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

  // Create CUDA streams for each GPU
  streams_.resize(num_gpus_);
  events_.resize(num_gpus_);
  for (int i = 0; i < num_gpus_; i++) {
    cudaSetDevice(dp_group_id * num_gpus_ + i);
    cudaStreamCreate(&streams_[i]);
    cudaEventCreate(&events_[i]);
  }

  // Initialize SSD IO context if ssd_files is not empty
  enable_ssd_ = !ssd_files.empty();
  if (enable_ssd_) {
    ioctx_ = std::make_unique<SSDIOCTX>(ssd_files, ssd_files.size(),
                                        iouring_entries, iouring_flags);
  }
}

LayerwiseTransferGroup::~LayerwiseTransferGroup() {
  for (int i = 0; i < num_gpus_; i++) {
    cudaSetDevice(dp_group_id_ * num_gpus_ + i);
    cudaStreamDestroy(streams_[i]);
    cudaEventDestroy(events_[i]);
  }

  cudaFreeHost(gpu_blocks_);

  gpu_tensor_handlers_.clear();
  delete[] gpu_kv_strides_in_bytes_;
  delete[] gpu_block_strides_in_bytes_;
  delete[] gpu_layer_strides_in_bytes_;
  delete[] gpu_chunk_sizes_in_bytes_;
}

void LayerwiseTransferGroup::layer_done_callback(int start_layer,
                                                 int layers_this_batch) {
  std::atomic<int> *counter = new std::atomic<int>(0);
  
  // Get eventfd pointer for current counter set
  int *eventfds_ptr = nullptr;
  if (enable_eventfd_ && num_counters_ > 0) {
    // Offset into layer_eventfds_ for current counter set
    int offset = current_counter_id_ * tp_size_ * num_layers_;
    eventfds_ptr = layer_eventfds_.data() + offset;
  }
  
  for (int i = 0; i < num_gpus_; ++i) {
    LayerCallbackData *data = new LayerCallbackData{
        start_layer, layers_this_batch, num_gpus_, counter,
        enable_eventfd_, tp_size_, num_layers_, eventfds_ptr};
    cudaLaunchHostFunc(streams_[i], layer_done_host_callback, data);
  }
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
    const bool use_ce_transfer, const int num_layers,
    const int layer_granularity, const bool is_mla,
    const int counter_id) {

  // Set current counter ID for eventfd notification
  current_counter_id_ = counter_id;

  int num_blocks = gpu_block_id_tensor.numel();
  int64_t *gpu_block_ids =
      static_cast<int64_t *>(gpu_block_id_tensor.data_ptr());
  int64_t *cpu_block_ids =
      static_cast<int64_t *>(cpu_block_id_tensor.data_ptr());
  void *cpu_ptr = cpu_blocks_;

  for (int start_layer = 0; start_layer < num_layers;
       start_layer += layer_granularity) {
    int layers_this_batch =
        std::min(layer_granularity, num_layers - start_layer);
    // Step 1: SSD -> CPU transfer
    if (enable_ssd_ && ssd_block_ids.numel() > 0) {
      torch::Tensor layer_id_list =
          torch::arange(start_layer, start_layer + layers_this_batch,
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
    for (int i = 0; i < num_gpus_; ++i) {
      cudaSetDevice(dp_group_id_ * num_gpus_ + i);

      int64_t cpu_startoff_inside_chunks = i * gpu_chunk_sizes_in_bytes_[i];
      if (is_mla) {
        cpu_startoff_inside_chunks = 0;
      }
      int64_t gpu_startoff_inside_chunks = 0;
      int64_t chunk_size = gpu_chunk_sizes_in_bytes_[i];

      switch (backend_type_) {
      case BackendType::VLLM:
        flexkv::transfer_kv_blocks<BackendType::VLLM>(
            num_blocks, start_layer, layers_this_batch, gpu_block_ids,
            gpu_tensor_handlers_[i], gpu_startoff_inside_chunks, cpu_block_ids,
            cpu_ptr, cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes,
            cpu_block_stride_in_bytes, cpu_startoff_inside_chunks, chunk_size,
            streams_[i], transfer_sms, true, use_ce_transfer, is_mla, false);
        break;
      case BackendType::TRTLLM:
        flexkv::transfer_kv_blocks<BackendType::TRTLLM>(
            num_blocks, start_layer, layers_this_batch, gpu_block_ids,
            gpu_tensor_handlers_[i], gpu_startoff_inside_chunks, cpu_block_ids,
            cpu_ptr, cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes,
            cpu_block_stride_in_bytes, cpu_startoff_inside_chunks, chunk_size,
            streams_[i], transfer_sms, true, use_ce_transfer, is_mla, false);
        break;
      case BackendType::SGLANG:
        flexkv::transfer_kv_blocks<BackendType::SGLANG>(
            num_blocks, start_layer, layers_this_batch, gpu_block_ids,
            gpu_tensor_handlers_[i], gpu_startoff_inside_chunks, cpu_block_ids,
            cpu_ptr, cpu_kv_stride_in_bytes, cpu_layer_stride_in_bytes,
            cpu_block_stride_in_bytes, cpu_startoff_inside_chunks, chunk_size,
            streams_[i], transfer_sms, true, use_ce_transfer, is_mla, false);
        break;
      }
    }

    layer_done_callback(start_layer, layers_this_batch);
  }
  for (int i = 0; i < num_gpus_; ++i) {
    cudaError_t err = cudaStreamSynchronize(streams_[i]);
    if (err != cudaSuccess) {
      throw std::runtime_error("layerwise_transfer failed on GPU " +
                               std::to_string(i) + ": " +
                               cudaGetErrorString(err));
    }
  }
}

} // namespace flexkv
