/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */
#include "layer_notify.h"

#include "logging.h"

#include <chrono>
#include <cstdint>
#include <stdexcept>
#include <unistd.h>

namespace flexkv {

LayerNotifyMode parse_layer_notify_mode(const std::string &s) {
  return s == "polling" ? LayerNotifyMode::POLLING : LayerNotifyMode::HOSTFUNC;
}

LayerEventfdTable::LayerEventfdTable(const torch::Tensor &fds_tensor,
                                     int tp_size, int num_layers)
    : tp_size_(tp_size), num_layers_(num_layers) {
  if (!fds_tensor.defined() || fds_tensor.numel() == 0 || tp_size <= 0 ||
      num_layers <= 0) {
    return;
  }
  const int per_counter = tp_size * num_layers;
  if (fds_tensor.numel() % per_counter != 0) {
    throw std::invalid_argument(
        "layer eventfd tensor has " + std::to_string(fds_tensor.numel()) +
        " entries, not a multiple of tp_size*num_layers=" +
        std::to_string(per_counter));
  }
  const torch::Tensor cpu_fds = fds_tensor.to(torch::kCPU).contiguous();
  const int32_t *src = cpu_fds.data_ptr<int32_t>();
  fds_.assign(src, src + cpu_fds.numel());
  num_counters_ = static_cast<int>(fds_.size()) / per_counter;
  enabled_ = true;
}

void LayerEventfdTable::post(int counter_id, int layer) const {
  if (!enabled_ || layer < 0 || layer >= num_layers_) {
    return;
  }
  if (counter_id < 0 || counter_id >= num_counters_) {
    return;
  }
  const int *base = fds_.data() + counter_id * tp_size_ * num_layers_;
  for (int tp_rank = 0; tp_rank < tp_size_; ++tp_rank) {
    const int fd = base[tp_rank * num_layers_ + layer];
    if (fd >= 0) {
      // One semaphore token per layer per transfer. sglang consumes exactly
      // one; writing any other value desynchronizes its accounting silently.
      uint64_t val = 1;
      ssize_t ret = write(fd, &val, sizeof(val));
      (void)ret;
    }
  }
}

namespace {

// What a hostfunc callback needs. Heap-allocated per (layer, rank) and freed
// by the callback itself: the driver gives us no other hook to free it, and
// the callback cannot call any CUDA API (including cudaFree).
struct HostFuncData {
  const LayerEventfdTable *table;
  int counter_id;
  int layer;
  std::atomic<int> *counter; // shared by the ranks of one layer
  int expected;              // ranks that must arrive before the post
};

void CUDART_CB layer_done_host_callback(void *user_data) {
  auto *d = static_cast<HostFuncData *>(user_data);
  const int arrived = d->counter->fetch_add(1, std::memory_order_acq_rel) + 1;
  if (arrived == d->expected) {
    d->table->post(d->counter_id, d->layer);
    delete d->counter;
  }
  delete d;
}

} // namespace

LayerNotifier::~LayerNotifier() {
  stop_polling();
  destroy_events();
}

void LayerNotifier::reset(LayerNotifyMode mode,
                          const std::vector<int> &device_ids, int counter_id) {
  stop_polling();
  destroy_events();
  mode_ = mode;
  device_ids_ = device_ids;
  counter_id_ = counter_id;
  batches_.clear();
  batch_of_layer_.assign(table_.num_layers() > 0 ? table_.num_layers() : 0, -1);
  poll_stop_.store(false, std::memory_order_release);
  poll_next_.store(0, std::memory_order_release);
  poll_done_.store(false, std::memory_order_release);
  poll_failed_.store(false, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lk(poll_error_mtx_);
    poll_error_msg_.clear();
  }
}

void LayerNotifier::post_empty(int layer) {
  table_.post(counter_id_, layer);
}

void LayerNotifier::begin_layer(int layer) {
  if (layer < 0) {
    return;
  }
  if (layer >= static_cast<int>(batch_of_layer_.size())) {
    batch_of_layer_.resize(layer + 1, -1);
  }
  if (batch_of_layer_[layer] >= 0) {
    return; // already declared
  }
  LayerBatch b;
  b.layer = layer;
  if (mode_ == LayerNotifyMode::POLLING) {
    b.per_rank_events.resize(device_ids_.size(), nullptr);
    for (size_t r = 0; r < device_ids_.size(); ++r) {
      cudaSetDevice(device_ids_[r]);
      cudaEventCreateWithFlags(&b.per_rank_events[r], cudaEventDisableTiming);
    }
  } else if (table_.enabled()) {
    // One counter shared by this layer's ranks; the last callback frees it.
    b.hostfunc_counter = new std::atomic<int>(0);
  }
  batch_of_layer_[layer] = static_cast<int>(batches_.size());
  batches_.push_back(std::move(b));
}

void LayerNotifier::record(int layer, int rank, cudaStream_t stream) {
  if (!table_.enabled()) {
    return; // nobody to tell; do not pay for events or callbacks
  }
  if (layer < 0 || layer >= static_cast<int>(batch_of_layer_.size()) ||
      batch_of_layer_[layer] < 0) {
    throw std::logic_error("LayerNotifier::record for layer " +
                           std::to_string(layer) +
                           " without a matching begin_layer");
  }
  LayerBatch &b = batches_[batch_of_layer_[layer]];
  if (mode_ == LayerNotifyMode::POLLING) {
    cudaEventRecord(b.per_rank_events[rank], stream);
    return;
  }
  // HOSTFUNC: one callback per rank; whichever arrives last posts the fd and
  // frees the shared counter. The counter is heap-allocated (not a member)
  // because the callback may outlive this LayerNotifier's next reset().
  auto *d = new HostFuncData{&table_, counter_id_, layer, b.hostfunc_counter,
                             static_cast<int>(device_ids_.size())};
  cudaLaunchHostFunc(stream, layer_done_host_callback, d);
}

void LayerNotifier::arm() {
  if (!table_.enabled() || mode_ != LayerNotifyMode::POLLING ||
      batches_.empty()) {
    return;
  }
  poll_thread_ = std::thread(&LayerNotifier::polling_loop, this);
}

void LayerNotifier::polling_loop() {
  int prev_device = 0;
  cudaGetDevice(&prev_device);

  while (!poll_stop_.load(std::memory_order_acquire)) {
    const int next = poll_next_.load(std::memory_order_acquire);
    if (next >= static_cast<int>(batches_.size())) {
      break;
    }
    LayerBatch &b = batches_[next];
    if (b.notified) {
      poll_next_.fetch_add(1, std::memory_order_acq_rel);
      continue;
    }

    bool all_done = true;
    bool hard_error = false;
    for (size_t r = 0; r < device_ids_.size() && all_done; ++r) {
      cudaSetDevice(device_ids_[r]);
      const cudaError_t err = cudaEventQuery(b.per_rank_events[r]);
      if (err == cudaErrorNotReady) {
        all_done = false;
      } else if (err != cudaSuccess) {
        // Distinguishing this from "not ready" is the whole point: a real
        // error used to leave the loop spinning forever on a layer that
        // would never complete, with the consumer blocked on its eventfd.
        all_done = false;
        hard_error = true;
        std::lock_guard<std::mutex> lk(poll_error_mtx_);
        if (poll_error_msg_.empty()) {
          poll_error_msg_ = std::string("cudaEventQuery failed on GPU ") +
                            std::to_string(device_ids_[r]) + " for layer " +
                            std::to_string(b.layer) + ": " +
                            cudaGetErrorString(err);
        }
      }
    }

    if (hard_error) {
      poll_failed_.store(true, std::memory_order_release);
      poll_stop_.store(true, std::memory_order_release);
      break;
    }
    if (all_done) {
      b.notified = true;
      table_.post(counter_id_, b.layer);
      poll_next_.fetch_add(1, std::memory_order_acq_rel);
    } else {
      // yield() burned a whole core for the duration of every transfer; a
      // short sleep keeps notification latency far under per-layer compute.
      std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
  }

  cudaSetDevice(prev_device);
  poll_done_.store(true, std::memory_order_release);
}

bool LayerNotifier::wait(double timeout_s, std::string *error_out) {
  if (mode_ != LayerNotifyMode::POLLING || !poll_thread_.joinable()) {
    return !poll_failed_.load(std::memory_order_acquire);
  }
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::duration<double>(timeout_s);
  while (!poll_done_.load(std::memory_order_acquire)) {
    if (std::chrono::steady_clock::now() >= deadline) {
      if (error_out != nullptr) {
        *error_out = "per-layer notification did not drain within " +
                     std::to_string(timeout_s) + "s";
      }
      // Leave the thread running: it still references the batch events, and
      // the next reset()/destructor joins it. Returning false lets the caller
      // fail the op rather than hang.
      return false;
    }
    std::this_thread::sleep_for(std::chrono::microseconds(200));
  }
  poll_thread_.join();
  if (poll_failed_.load(std::memory_order_acquire)) {
    if (error_out != nullptr) {
      std::lock_guard<std::mutex> lk(poll_error_mtx_);
      *error_out = poll_error_msg_;
    }
    return false;
  }
  return true;
}

void LayerNotifier::stop_polling() {
  poll_stop_.store(true, std::memory_order_release);
  if (poll_thread_.joinable()) {
    poll_thread_.join();
  }
}

void LayerNotifier::destroy_events() {
  if (device_ids_.empty()) {
    batches_.clear();
    return;
  }
  int prev_device = 0;
  cudaGetDevice(&prev_device);
  for (LayerBatch &b : batches_) {
    for (size_t r = 0; r < b.per_rank_events.size(); ++r) {
      if (b.per_rank_events[r] != nullptr) {
        cudaSetDevice(device_ids_[r < device_ids_.size() ? r : 0]);
        cudaEventDestroy(b.per_rank_events[r]);
      }
    }
    b.per_rank_events.clear();
  }
  cudaSetDevice(prev_device);
  batches_.clear();
}

} // namespace flexkv
