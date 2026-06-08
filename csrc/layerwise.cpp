#include "layerwise.h"
#include <atomic>
#include <cstdio>
#include <fcntl.h>
#include <nvtx3/nvToolsExt.h>
#include <stdexcept>
#include <sys/eventfd.h>
#include <unistd.h>

namespace flexkv {

struct LayerCallbackData {
  int start_layer;
  int layers_this_batch;
  // Total number of GPU/member callbacks that must complete before
  // the eventfd for this layer fires. Single-group: num_gpus_.
  // Multi-group: members_this_layer * num_gpus_.
  int expected_count;
  std::atomic<int> *counter;
  // Eventfd info for notification
  bool enable_eventfd;
  int tp_size;
  int num_layers;
  int *layer_eventfds; // Pointer to eventfds array for current counter set
  // NVTX range id for CPU->GPU transfer
  nvtxRangeId_t *current_range_id_ptr; // Pointer to current layer's range ID
  bool is_last_batch;                  // Whether this is the last batch
  char next_range_name[64]; // Name for next layer's range (if not last batch)
  nvtxRangeId_t *next_range_id_ptr; // Pointer to next layer's range ID storage
};

static void CUDART_CB layer_done_host_callback(void *userData) {
  LayerCallbackData *data = static_cast<LayerCallbackData *>(userData);
  int completed = data->counter->fetch_add(1) + 1;
  if (completed == data->expected_count) {
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
          }
        }
      }
    }
    // End current NVTX range when all GPUs complete
    if (data->current_range_id_ptr != nullptr &&
        *data->current_range_id_ptr != 0) {
      nvtxRangeEnd(*data->current_range_id_ptr);
    }
    // Start next layer's NVTX range (so it begins right after current layer
    // ends)
    if (!data->is_last_batch && data->next_range_id_ptr != nullptr) {
      *data->next_range_id_ptr = nvtxRangeStartA(data->next_range_name);
    }
    delete data->counter;
  }
  delete data;
}

LayerwiseTransferGroup::LayerwiseTransferGroup(
    int num_gpus, const std::vector<std::vector<torch::Tensor>> &gpu_blocks,
    torch::Tensor &cpu_blocks,
    std::map<int, std::vector<std::string>> &ssd_files, int num_layers,
    torch::Tensor &gpu_kv_strides_tensor,
    torch::Tensor &gpu_block_strides_tensor,
    torch::Tensor &gpu_layer_strides_tensor,
    torch::Tensor &gpu_chunk_sizes_tensor, int iouring_entries,
    int iouring_flags, torch::Tensor &layer_eventfds_tensor, int tp_size) {

  num_gpus_ = num_gpus;
  num_layers_ = num_layers;
  tp_size_ = tp_size;
  current_counter_id_ = 0;
  has_multi_group_ = false;
  num_original_layers_ = num_layers;

  // Initialize eventfds
  enable_eventfd_ = (layer_eventfds_tensor.numel() > 0);
  if (enable_eventfd_) {
    // layer_eventfds_tensor layout: [num_counters, tp_size, num_layers]
    // Index formula: counter_id * tp_size * num_layers + tp_rank * num_layers +
    // layer
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

  // Get GPU device IDs from tensors (like tp_transfer_thread_group.cpp)
  gpu_device_ids_.resize(num_gpus_);
  for (int i = 0; i < num_gpus_; ++i) {
    gpu_device_ids_[i] = gpu_blocks[i][0].device().index();
  }

  // Create CUDA streams for each GPU
  streams_.resize(num_gpus_);
  events_.resize(num_gpus_);

  // Get highest priority (lowest value)
  int leastPriority, greatestPriority;
  cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);

  for (int i = 0; i < num_gpus_; i++) {
    cudaSetDevice(gpu_device_ids_[i]);
    cudaStreamCreateWithPriority(&streams_[i], cudaStreamNonBlocking,
                                 greatestPriority);
    cudaEventCreate(&events_[i]);
  }

  // Initialize SSD IO context if ssd_files is not empty
  enable_ssd_ = !ssd_files.empty();
  if (enable_ssd_) {
    ioctx_ = std::make_unique<SSDIOCTX>(ssd_files, ssd_files.size(),
                                        iouring_entries, iouring_flags);
  }
}

LayerwiseTransferGroup::LayerwiseTransferGroup(
    int num_gpus,
    const std::vector<std::vector<std::vector<torch::Tensor>>>
        &gpu_blocks_per_group,
    torch::Tensor &cpu_blocks,
    std::map<int, std::vector<std::string>> &ssd_files, int num_original_layers,
    const std::vector<std::vector<std::pair<int, int>>> &layer_members,
    const std::vector<int> &group_num_layers,
    const std::vector<int64_t> &group_cpu_offset_bytes,
    const std::vector<int64_t> &group_ssd_offset_bytes,
    const std::vector<int64_t> &group_cpu_layer_strides,
    const std::vector<int64_t> &group_cpu_kv_strides,
    const std::vector<int64_t> &group_ssd_layer_strides,
    const std::vector<int64_t> &group_ssd_kv_strides,
    const std::vector<int64_t> &group_chunk_sizes,
    const std::vector<int64_t> &group_h2d_cpu_kv_strides,
    const std::vector<int64_t> &group_h2d_cpu_layer_strides,
    const std::vector<int64_t> &group_cpu_block_strides,
    const std::vector<int64_t> &group_cpu_tp_strides,
    const std::vector<int64_t> &group_gpu_kv_strides,
    const std::vector<int64_t> &group_gpu_block_strides,
    const std::vector<int64_t> &group_gpu_layer_strides,
    const std::vector<int64_t> &group_gpu_chunk_sizes, int iouring_entries,
    int iouring_flags, torch::Tensor &layer_eventfds_tensor, int tp_size) {

  num_gpus_ = num_gpus;
  num_layers_ = num_original_layers;
  num_original_layers_ = num_original_layers;
  tp_size_ = tp_size;
  current_counter_id_ = 0;
  has_multi_group_ = true;

  // Legacy single-group GPU members are unused in multi-group mode.
  num_tensors_per_gpu_ = 0;
  gpu_blocks_ = nullptr;
  gpu_kv_strides_in_bytes_ = nullptr;
  gpu_block_strides_in_bytes_ = nullptr;
  gpu_layer_strides_in_bytes_ = nullptr;
  gpu_chunk_sizes_in_bytes_ = nullptr;

  // Initialize eventfds (shape [num_counters, tp_size, num_original_layers]).
  enable_eventfd_ = (layer_eventfds_tensor.numel() > 0);
  if (enable_eventfd_) {
    int total_fds = layer_eventfds_tensor.numel();
    num_counters_ = total_fds / (tp_size * num_original_layers);

    int32_t *fds_ptr = layer_eventfds_tensor.data_ptr<int32_t>();
    layer_eventfds_.assign(fds_ptr, fds_ptr + total_fds);

    printf(
        "[LayerwiseTransferGroup multi-group] Initialized with eventfds: "
        "tp_size=%d, num_counters=%d, num_original_layers=%d, total_fds=%d\n",
        tp_size_, num_counters_, num_original_layers_, total_fds);
  } else {
    num_counters_ = 0;
    printf(
        "[LayerwiseTransferGroup multi-group] Initialized without eventfds\n");
  }

  layer_members_ = layer_members;

  if (static_cast<int>(layer_members_.size()) != num_original_layers_) {
    throw std::runtime_error(
        "[LayerwiseTransferGroup multi-group] layer_members must have length "
        "num_original_layers, got " +
        std::to_string(layer_members_.size()) + " vs expected " +
        std::to_string(num_original_layers_));
  }

  int num_groups = static_cast<int>(group_num_layers.size());
  if (static_cast<int>(gpu_blocks_per_group.size()) != num_groups) {
    throw std::runtime_error(
        "[LayerwiseTransferGroup multi-group] gpu_blocks_per_group size " +
        std::to_string(gpu_blocks_per_group.size()) +
        " does not match num_groups " + std::to_string(num_groups));
  }

  groups_.resize(num_groups);
  for (int gi = 0; gi < num_groups; ++gi) {
    GroupParams &gp = groups_[gi];
    gp.num_layers = group_num_layers[gi];
    gp.cpu_offset_bytes = group_cpu_offset_bytes[gi];
    gp.ssd_offset_bytes = group_ssd_offset_bytes[gi];
    gp.cpu_layer_stride = group_cpu_layer_strides[gi];
    gp.cpu_kv_stride = group_cpu_kv_strides[gi];
    gp.ssd_layer_stride = group_ssd_layer_strides[gi];
    gp.ssd_kv_stride = group_ssd_kv_strides[gi];
    gp.chunk_size = group_chunk_sizes[gi];
    gp.h2d_cpu_kv_stride = group_h2d_cpu_kv_strides[gi];
    gp.h2d_cpu_layer_stride = group_h2d_cpu_layer_strides[gi];
    gp.cpu_block_stride = group_cpu_block_strides[gi];
    gp.cpu_tp_stride = group_cpu_tp_strides[gi];

    gp.gpu_kv_strides.resize(num_gpus);
    gp.gpu_block_strides.resize(num_gpus);
    gp.gpu_layer_strides.resize(num_gpus);
    gp.gpu_chunk_sizes.resize(num_gpus);
    for (int d = 0; d < num_gpus; ++d) {
      gp.gpu_kv_strides[d] = group_gpu_kv_strides[gi * num_gpus + d];
      gp.gpu_block_strides[d] = group_gpu_block_strides[gi * num_gpus + d];
      gp.gpu_layer_strides[d] = group_gpu_layer_strides[gi * num_gpus + d];
      gp.gpu_chunk_sizes[d] = group_gpu_chunk_sizes[gi * num_gpus + d];
    }

    if (static_cast<int>(gpu_blocks_per_group[gi].size()) != num_gpus) {
      throw std::runtime_error(
          "[LayerwiseTransferGroup multi-group] gpu_blocks_per_group[" +
          std::to_string(gi) + "].size() != num_gpus");
    }
    gp.num_tensors_per_gpu =
        static_cast<int>(gpu_blocks_per_group[gi][0].size());
    cudaMallocHost((void **)&gp.gpu_blocks_flat,
                   num_gpus * gp.num_tensors_per_gpu * sizeof(void *));
    for (int d = 0; d < num_gpus; ++d) {
      if (static_cast<int>(gpu_blocks_per_group[gi][d].size()) !=
          gp.num_tensors_per_gpu) {
        throw std::runtime_error(
            "[LayerwiseTransferGroup multi-group] gpu_blocks_per_group[" +
            std::to_string(gi) + "][" + std::to_string(d) +
            "] tensor count mismatch");
      }
      for (int t = 0; t < gp.num_tensors_per_gpu; ++t) {
        gp.gpu_blocks_flat[d * gp.num_tensors_per_gpu + t] =
            gpu_blocks_per_group[gi][d][t].data_ptr();
      }
    }

    if (gp.num_tensors_per_gpu == 1) {
      gp.backend_type = BackendType::TRTLLM;
    } else if (gp.num_tensors_per_gpu == gp.num_layers) {
      gp.backend_type = BackendType::VLLM;
    } else if (gp.num_tensors_per_gpu == gp.num_layers * 2) {
      gp.backend_type = BackendType::SGLANG;
    } else {
      throw std::runtime_error(
          "[LayerwiseTransferGroup multi-group] group " + std::to_string(gi) +
          " has unsupported tensors_per_gpu=" +
          std::to_string(gp.num_tensors_per_gpu) +
          " for num_layers=" + std::to_string(gp.num_layers));
    }

    gp.gpu_tensor_handlers.reserve(num_gpus);
    for (int d = 0; d < num_gpus; ++d) {
      int64_t **gpu_blocks_ptr = reinterpret_cast<int64_t **>(
          gp.gpu_blocks_flat + d * gp.num_tensors_per_gpu);
      gp.gpu_tensor_handlers.emplace_back(
          gp.backend_type, gpu_blocks_ptr, gp.num_layers, gp.gpu_kv_strides[d],
          gp.gpu_block_strides[d], gp.gpu_layer_strides[d]);
    }
  }

  cpu_blocks_ = cpu_blocks.data_ptr();

  // GPU device IDs (from group 0 device 0's first tensor).
  gpu_device_ids_.resize(num_gpus_);
  for (int d = 0; d < num_gpus_; ++d) {
    gpu_device_ids_[d] = gpu_blocks_per_group[0][d][0].device().index();
  }

  streams_.resize(num_gpus_);
  events_.resize(num_gpus_);
  int leastPriority, greatestPriority;
  cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
  for (int d = 0; d < num_gpus_; ++d) {
    cudaSetDevice(gpu_device_ids_[d]);
    cudaStreamCreateWithPriority(&streams_[d], cudaStreamNonBlocking,
                                 greatestPriority);
    cudaEventCreate(&events_[d]);
  }

  enable_ssd_ = !ssd_files.empty();
  if (enable_ssd_) {
    ioctx_ = std::make_unique<SSDIOCTX>(ssd_files, ssd_files.size(),
                                        iouring_entries, iouring_flags);
  }
}

LayerwiseTransferGroup::~LayerwiseTransferGroup() {
  for (int i = 0; i < num_gpus_; i++) {
    cudaSetDevice(gpu_device_ids_[i]);
    cudaStreamDestroy(streams_[i]);
    cudaEventDestroy(events_[i]);
  }

  if (gpu_blocks_ != nullptr) {
    cudaFreeHost(gpu_blocks_);
  }
  for (auto &gp : groups_) {
    if (gp.gpu_blocks_flat != nullptr) {
      cudaFreeHost(gp.gpu_blocks_flat);
      gp.gpu_blocks_flat = nullptr;
    }
  }

  gpu_tensor_handlers_.clear();
  delete[] gpu_kv_strides_in_bytes_;
  delete[] gpu_block_strides_in_bytes_;
  delete[] gpu_layer_strides_in_bytes_;
  delete[] gpu_chunk_sizes_in_bytes_;
}

void LayerwiseTransferGroup::layer_done_callback(
    int start_layer, int layers_this_batch, int expected_count,
    nvtxRangeId_t *current_range_id_ptr, bool is_last_batch,
    const char *next_range_name, nvtxRangeId_t *next_range_id_ptr,
    int callbacks_per_gpu) {
  std::atomic<int> *counter = new std::atomic<int>(0);

  // Get eventfd pointer for current counter set
  int *eventfds_ptr = nullptr;
  if (enable_eventfd_ && num_counters_ > 0) {
    int offset = current_counter_id_ * tp_size_ * num_layers_;
    eventfds_ptr = layer_eventfds_.data() + offset;
  }

  for (int rep = 0; rep < callbacks_per_gpu; ++rep) {
    for (int i = 0; i < num_gpus_; ++i) {
      LayerCallbackData *data = new LayerCallbackData{start_layer,
                                                      layers_this_batch,
                                                      expected_count,
                                                      counter,
                                                      enable_eventfd_,
                                                      tp_size_,
                                                      num_layers_,
                                                      eventfds_ptr,
                                                      current_range_id_ptr,
                                                      is_last_batch,
                                                      {0},
                                                      next_range_id_ptr};
      if (next_range_name != nullptr) {
        snprintf(data->next_range_name, sizeof(data->next_range_name), "%s",
                 next_range_name);
      }
      cudaLaunchHostFunc(streams_[i], layer_done_host_callback, data);
    }
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
    const int64_t cpu_chunk_size_in_bytes,
    const int64_t h2d_cpu_kv_stride_in_bytes,
    const int64_t h2d_cpu_layer_stride_in_bytes,
    const int64_t cpu_tp_stride_in_bytes, const int transfer_cta_num,
    const bool use_ce_transfer, const int num_layers,
    const int layer_granularity, const bool is_mla, const int counter_id) {

  if (has_multi_group_) {
    throw std::runtime_error(
        "[LayerwiseTransferGroup] layerwise_transfer() invoked on a "
        "multi-group instance; use layerwise_transfer_multi_group() instead.");
  }

  // Set current counter ID for eventfd notification
  current_counter_id_ = counter_id;

  int num_blocks = gpu_block_id_tensor.numel();
  int64_t *gpu_block_ids =
      static_cast<int64_t *>(gpu_block_id_tensor.data_ptr());
  int64_t *cpu_block_ids =
      static_cast<int64_t *>(cpu_block_id_tensor.data_ptr());
  void *cpu_ptr = cpu_blocks_;

  // Create CUDA events for timing each layer batch (on GPU 0)
  int num_batches = (num_layers + layer_granularity - 1) / layer_granularity;
  std::vector<cudaEvent_t> timing_events(num_batches + 1); // +1 for start event
  std::vector<int> batch_start_layers(num_batches);
  std::vector<int> batch_layers_count(num_batches);

  cudaSetDevice(gpu_device_ids_[0]);
  for (int i = 0; i <= num_batches; ++i) {
    cudaEventCreate(&timing_events[i]);
  }

  // Record start event
  cudaEventRecord(timing_events[0], streams_[0]);

  // Allocate storage for NVTX range IDs (one per batch)
  std::vector<nvtxRangeId_t> h2d_range_ids(num_batches, 0);
  // Pre-generate all range names with data size info
  std::vector<std::string> h2d_range_names(num_batches);
  for (int b = 0; b < num_batches; ++b) {
    int sl = b * layer_granularity;
    int ltb = std::min(layer_granularity, num_layers - sl);
    // Calculate data size for this batch: chunk_size * 2 (K+V) * layers *
    // num_blocks
    int64_t bytes_this_batch = 0;
    for (int g = 0; g < num_gpus_; ++g) {
      bytes_this_batch += gpu_chunk_sizes_in_bytes_[g] * 2 * ltb * num_blocks;
    }
    char name[256];
    snprintf(name, sizeof(name), "CPU->GPU Layer[%d,%d) %.2fMB", sl, sl + ltb,
             bytes_this_batch / (1024.0 * 1024.0));
    h2d_range_names[b] = name;
  }

  // Start the first batch's NVTX range in main thread
  if (num_batches > 0) {
    h2d_range_ids[0] = nvtxRangeStartA(h2d_range_names[0].c_str());
  }

  // Step 0: SSD -> CPU transfer for ALL layers at once (before layerwise loop).
  // This is required because the CPU memory uses TP-divided layout where each
  // rank's data occupies a contiguous region [rank*tp_stride,
  // (rank+1)*tp_stride). Per-layer-batch SSD reads with full strides would land
  // at wrong CPU positions for TP > 1.
  if (enable_ssd_ && ssd_block_ids.numel() > 0) {
    int num_ssd_blocks = ssd_block_ids.numel();
    int64_t ssd_bytes =
        cpu_chunk_size_in_bytes * 2 * num_layers * num_ssd_blocks;
    double ssd_mb = ssd_bytes / (1024.0 * 1024.0);
    char ssd_range_name[128];
    snprintf(ssd_range_name, sizeof(ssd_range_name),
             "SSD->CPU AllLayers[0,%d) %.2fMB", num_layers, ssd_mb);
    nvtxRangePushA(ssd_range_name);

    torch::Tensor all_layer_ids = torch::arange(
        0, num_layers, torch::TensorOptions().dtype(torch::kInt32));
    transfer_kv_blocks_ssd(
        *ioctx_, all_layer_ids, reinterpret_cast<int64_t>(cpu_blocks_),
        ssd_block_ids, cpu_block_ids_d2h, cpu_layer_stride_in_bytes,
        cpu_kv_stride_in_bytes, ssd_layer_stride_in_bytes,
        ssd_kv_stride_in_bytes, cpu_chunk_size_in_bytes,
        cpu_block_stride_in_bytes,
        true, // is_read: SSD -> CPU
        num_blocks_per_file, round_robin, num_threads_per_device, is_mla);

    nvtxRangePop();
  }

  int batch_idx = 0;
  for (int start_layer = 0; start_layer < num_layers;
       start_layer += layer_granularity) {
    int layers_this_batch =
        std::min(layer_granularity, num_layers - start_layer);

    batch_start_layers[batch_idx] = start_layer;
    batch_layers_count[batch_idx] = layers_this_batch;

    // Step 1: CPU -> GPU transfer
    // NVTX range for this batch was already started (by main thread for first
    // batch, or by previous batch's callback for subsequent batches)

    for (int i = 0; i < num_gpus_; ++i) {
      cudaSetDevice(gpu_device_ids_[i]);
      int64_t cpu_startoff_inside_chunks = i * cpu_tp_stride_in_bytes;
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
            cpu_ptr, h2d_cpu_kv_stride_in_bytes, h2d_cpu_layer_stride_in_bytes,
            cpu_block_stride_in_bytes, cpu_startoff_inside_chunks, chunk_size,
            streams_[i], transfer_cta_num, true, use_ce_transfer, is_mla,
            false);
        break;
      case BackendType::TRTLLM:
        flexkv::transfer_kv_blocks<BackendType::TRTLLM>(
            num_blocks, start_layer, layers_this_batch, gpu_block_ids,
            gpu_tensor_handlers_[i], gpu_startoff_inside_chunks, cpu_block_ids,
            cpu_ptr, h2d_cpu_kv_stride_in_bytes, h2d_cpu_layer_stride_in_bytes,
            cpu_block_stride_in_bytes, cpu_startoff_inside_chunks, chunk_size,
            streams_[i], transfer_cta_num, true, use_ce_transfer, is_mla,
            false);
        break;
      case BackendType::SGLANG:
        flexkv::transfer_kv_blocks<BackendType::SGLANG>(
            num_blocks, start_layer, layers_this_batch, gpu_block_ids,
            gpu_tensor_handlers_[i], gpu_startoff_inside_chunks, cpu_block_ids,
            cpu_ptr, h2d_cpu_kv_stride_in_bytes, h2d_cpu_layer_stride_in_bytes,
            cpu_block_stride_in_bytes, cpu_startoff_inside_chunks, chunk_size,
            streams_[i], transfer_cta_num, true, use_ce_transfer, is_mla,
            false);
        break;
      }
    }

    // Record event after this batch on GPU 0
    cudaSetDevice(gpu_device_ids_[0]);
    cudaEventRecord(timing_events[batch_idx + 1], streams_[0]);

    // NVTX: current range ends in callback, next range starts in callback
    bool is_last_batch = (batch_idx == num_batches - 1);
    const char *next_name =
        is_last_batch ? nullptr : h2d_range_names[batch_idx + 1].c_str();
    nvtxRangeId_t *next_id_ptr =
        is_last_batch ? nullptr : &h2d_range_ids[batch_idx + 1];

    layer_done_callback(start_layer, layers_this_batch, num_gpus_,
                        &h2d_range_ids[batch_idx], is_last_batch, next_name,
                        next_id_ptr);
    batch_idx++;
  }
  for (int i = 0; i < num_gpus_; ++i) {
    cudaError_t err = cudaStreamSynchronize(streams_[i]);
    if (err != cudaSuccess) {
      throw std::runtime_error("layerwise_transfer failed on GPU " +
                               std::to_string(i) + ": " +
                               cudaGetErrorString(err));
    }
  }

  // Calculate and print timing for each layer batch
  // chunk_size per GPU * num_gpus * 2 (K+V) * layers_this_batch * num_blocks
  // fprintf(stderr, "\n[LayerwiseTransfer] CPU->GPU Transfer Timing
  // (num_blocks=%d):\n", num_blocks);
  float total_time_ms = 0.0f;
  int64_t total_bytes = 0;

  for (int i = 0; i < num_batches; ++i) {
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, timing_events[i], timing_events[i + 1]);

    // Calculate bytes transferred for this batch
    // For each GPU: chunk_size * 2 (K+V) * layers * num_blocks
    int64_t bytes_this_batch = 0;
    for (int g = 0; g < num_gpus_; ++g) {
      bytes_this_batch +=
          gpu_chunk_sizes_in_bytes_[g] * 2 * batch_layers_count[i] * num_blocks;
    }

    double bandwidth_gbps =
        (bytes_this_batch / (1024.0 * 1024.0 * 1024.0)) / (elapsed_ms / 1000.0);

    // fprintf(stderr, "  Layers [%d, %d): time=%.3f ms, size=%.2f MB,
    // bandwidth=%.2f GB/s\n",
    //         batch_start_layers[i],
    //         batch_start_layers[i] + batch_layers_count[i],
    //         elapsed_ms,
    //         bytes_this_batch / (1024.0 * 1024.0),
    //         bandwidth_gbps);

    total_time_ms += elapsed_ms;
    total_bytes += bytes_this_batch;
  }

  double total_bandwidth_gbps =
      (total_bytes / (1024.0 * 1024.0 * 1024.0)) / (total_time_ms / 1000.0);
  // fprintf(stderr, "  Total: time=%.3f ms, size=%.2f MB, avg_bandwidth=%.2f
  // GB/s\n\n",
  //         total_time_ms, total_bytes / (1024.0 * 1024.0),
  //         total_bandwidth_gbps);
  // fflush(stderr);

  // Cleanup timing events
  cudaSetDevice(gpu_device_ids_[0]);
  for (int i = 0; i <= num_batches; ++i) {
    cudaEventDestroy(timing_events[i]);
  }
}

void LayerwiseTransferGroup::layerwise_transfer_multi_group(
    const torch::Tensor &ssd_block_ids, const torch::Tensor &cpu_block_ids_d2h,
    const int num_blocks_per_file, const int round_robin,
    const int num_threads_per_device, const torch::Tensor &gpu_block_id_tensor,
    const torch::Tensor &cpu_block_id_tensor, const int transfer_cta_num,
    const bool use_ce_transfer, const bool is_mla, const int counter_id) {

  if (!has_multi_group_) {
    throw std::runtime_error(
        "[LayerwiseTransferGroup] layerwise_transfer_multi_group() invoked on "
        "a single-group instance; use layerwise_transfer() instead.");
  }

  current_counter_id_ = counter_id;

  int num_blocks = gpu_block_id_tensor.numel();
  int64_t *gpu_block_ids =
      static_cast<int64_t *>(gpu_block_id_tensor.data_ptr());
  int64_t *cpu_block_ids =
      static_cast<int64_t *>(cpu_block_id_tensor.data_ptr());

  // Step 0: SSD -> CPU. CPU and SSD share an identical per-block byte layout
  // (multi-group BLOCKFIRST), so each block is transferred as one opaque blob.
  // Collapses the prior num_groups * tp_size calls into a single call and
  // removes the sub-4KiB IO hazard for groups with small per-chunk sizes
  // (e.g., DSv4 indexer at compress_ratio=128). The kernel's slow path with
  // num_layers=1, layer_stride=chunk_size=block_stride and is_mla=true issues
  // exactly one pread/pwrite of block_stride bytes per block.
  if (enable_ssd_ && ssd_block_ids.numel() > 0) {
    const int64_t block_stride = groups_[0].cpu_block_stride;
    char ssd_range_name[128];
    snprintf(ssd_range_name, sizeof(ssd_range_name),
             "SSD->CPU MultiGroup Blocks (%lld blocks, %.2fMB each)",
             static_cast<long long>(ssd_block_ids.numel()),
             block_stride / (1024.0 * 1024.0));
    nvtxRangePushA(ssd_range_name);

    torch::Tensor one_layer_id =
        torch::zeros({1}, torch::TensorOptions().dtype(torch::kInt32));
    transfer_kv_blocks_ssd(
        *ioctx_, one_layer_id, reinterpret_cast<int64_t>(cpu_blocks_),
        ssd_block_ids, cpu_block_ids_d2h,
        /*cpu_layer_stride_in_bytes=*/block_stride,
        /*cpu_kv_stride_in_bytes=*/0,
        /*ssd_layer_stride_in_bytes=*/block_stride,
        /*ssd_kv_stride_in_bytes=*/0,
        /*chunk_size_in_bytes=*/block_stride,
        /*block_stride_in_bytes=*/block_stride,
        /*is_read=*/true, num_blocks_per_file, round_robin,
        num_threads_per_device, /*is_mla=*/true, /*ssd_copy_offset=*/0);
    nvtxRangePop();
  }

  // Uncached layers (e.g. DSv4 layers 0/1/MTP with compress_ratio==0 at the
  // model level) carry empty member lists: no data to transfer, so no kernels
  // are launched, no eventfd callback is registered, and no NVTX range opened.
  //
  // HOWEVER, the model's forward still waits per-layer on these eventfds for
  // any path that reads a per-layer buffer regardless of compress_ratio. In
  // particular DSv4-Flash's SWA reload path calls
  // ``get_swa_key_buffer_radix(layer_id) -> wait_layer_transfer(layer_id) ->
  // eventfd_read`` for EVERY layer (the SWA pool spans all layers, including
  // the ratio==0 dense layers). If we never post the skipped layers' eventfds,
  // the consumer blocks forever -> scheduler watchdog timeout. So we post a
  // "no-op satisfied" signal (the same value 2 the real callback writes) for
  // every skipped layer immediately: they have no in-flight transfer, so the
  // data they cover (none) is trivially "ready". Done before the active-layer
  // loop launches kernels so the consumer can never observe an unposted gap.
  std::vector<int> active_origs;
  active_origs.reserve(num_original_layers_);
  if (enable_eventfd_ && num_counters_ > 0 && !layer_eventfds_.empty()) {
    int offset = current_counter_id_ * tp_size_ * num_layers_;
    int *eventfds_ptr = layer_eventfds_.data() + offset;
    for (int orig = 0; orig < num_original_layers_; ++orig) {
      if (!layer_members_[orig].empty()) {
        continue;
      }
      for (int tp_rank = 0; tp_rank < tp_size_; ++tp_rank) {
        int fd = eventfds_ptr[tp_rank * num_layers_ + orig];
        if (fd >= 0) {
          // Match layer_done_host_callback: write 2 to satisfy both
          // get_key_buffer and get_value_buffer waits for this layer.
          uint64_t val = 2;
          ssize_t ret = write(fd, &val, sizeof(val));
          (void)ret;
        }
      }
    }
  }
  for (int orig = 0; orig < num_original_layers_; ++orig) {
    if (!layer_members_[orig].empty()) {
      active_origs.push_back(orig);
    }
  }

  // NVTX ranges per original layer (one slot per orig so callbacks can index
  // by orig regardless of how the active subset chains together).
  std::vector<nvtxRangeId_t> h2d_range_ids(num_original_layers_, 0);
  std::vector<std::string> h2d_range_names(num_original_layers_);
  for (int orig : active_origs) {
    char name[160];
    snprintf(name, sizeof(name), "CPU->GPU OrigLayer[%d] members=%zu", orig,
             layer_members_[orig].size());
    h2d_range_names[orig] = name;
  }
  if (!active_origs.empty()) {
    int first = active_origs.front();
    h2d_range_ids[first] = nvtxRangeStartA(h2d_range_names[first].c_str());
  }

  // Main loop: iterate the *active* original layers only. For each, fan out N
  // members' transfer kernels (each on every GPU), then register a single
  // eventfd callback that fires after members_this_layer * num_gpus_ GPU
  // callbacks land. The "next" NVTX range chains to the next active orig, so
  // gaps from uncached layers do not leave unclosed ranges.
  for (size_t ai = 0; ai < active_origs.size(); ++ai) {
    int orig = active_origs[ai];
    const auto &members = layer_members_[orig];
    int members_this_layer = static_cast<int>(members.size());

    for (const auto &member : members) {
      int gi = member.first;
      int local_id = member.second;
      const GroupParams &gp = groups_[gi];

      for (int d = 0; d < num_gpus_; ++d) {
        cudaSetDevice(gpu_device_ids_[d]);
        int64_t cpu_startoff_inside_chunks = d * gp.cpu_tp_stride;
        if (is_mla) {
          cpu_startoff_inside_chunks = 0;
        }
        int64_t gpu_startoff_inside_chunks = 0;
        int64_t chunk_size = gp.gpu_chunk_sizes[d];
        void *cpu_ptr_for_group =
            static_cast<char *>(cpu_blocks_) + gp.cpu_offset_bytes;

        switch (gp.backend_type) {
        case BackendType::VLLM:
          flexkv::transfer_kv_blocks<BackendType::VLLM>(
              num_blocks, local_id, 1, gpu_block_ids, gp.gpu_tensor_handlers[d],
              gpu_startoff_inside_chunks, cpu_block_ids, cpu_ptr_for_group,
              gp.h2d_cpu_kv_stride, gp.h2d_cpu_layer_stride,
              gp.cpu_block_stride, cpu_startoff_inside_chunks, chunk_size,
              streams_[d], transfer_cta_num, true, use_ce_transfer, is_mla,
              false);
          break;
        case BackendType::TRTLLM:
          flexkv::transfer_kv_blocks<BackendType::TRTLLM>(
              num_blocks, local_id, 1, gpu_block_ids, gp.gpu_tensor_handlers[d],
              gpu_startoff_inside_chunks, cpu_block_ids, cpu_ptr_for_group,
              gp.h2d_cpu_kv_stride, gp.h2d_cpu_layer_stride,
              gp.cpu_block_stride, cpu_startoff_inside_chunks, chunk_size,
              streams_[d], transfer_cta_num, true, use_ce_transfer, is_mla,
              false);
          break;
        case BackendType::SGLANG:
          flexkv::transfer_kv_blocks<BackendType::SGLANG>(
              num_blocks, local_id, 1, gpu_block_ids, gp.gpu_tensor_handlers[d],
              gpu_startoff_inside_chunks, cpu_block_ids, cpu_ptr_for_group,
              gp.h2d_cpu_kv_stride, gp.h2d_cpu_layer_stride,
              gp.cpu_block_stride, cpu_startoff_inside_chunks, chunk_size,
              streams_[d], transfer_cta_num, true, use_ce_transfer, is_mla,
              false);
          break;
        }
      }
    }

    bool is_last_active = (ai + 1 == active_origs.size());
    int next_orig = is_last_active ? -1 : active_origs[ai + 1];
    const char *next_name =
        is_last_active ? nullptr : h2d_range_names[next_orig].c_str();
    nvtxRangeId_t *next_id_ptr =
        is_last_active ? nullptr : &h2d_range_ids[next_orig];

    layer_done_callback(/*start_layer=*/orig, /*layers_this_batch=*/1,
                        /*expected_count=*/members_this_layer * num_gpus_,
                        &h2d_range_ids[orig], is_last_active, next_name,
                        next_id_ptr,
                        /*callbacks_per_gpu=*/members_this_layer);
  }

  for (int d = 0; d < num_gpus_; ++d) {
    cudaError_t err = cudaStreamSynchronize(streams_[d]);
    if (err != cudaSuccess) {
      throw std::runtime_error("layerwise_transfer_multi_group failed on GPU " +
                               std::to_string(d) + ": " +
                               cudaGetErrorString(err));
    }
  }
}

} // namespace flexkv
