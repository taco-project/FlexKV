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
  shutdown_polling();
  destroy_events();
  release_event_pool();
}

void LayerNotifier::reset(LayerNotifyMode mode,
                          const std::vector<int> &device_ids, int counter_id) {
  quiesce_polling();
  // Not destroy_events(): the events are pooled and outlive the transfer.
  // Only the per-transfer bookkeeping below is rewound. An event that was
  // recorded last time is in the "completed" state, and cudaEventRecord
  // overwrites it, so there is nothing to clear.
  if (device_ids != pool_device_ids_) {
    // Different rank set (or the first transfer): the pool's events were
    // created on the old devices, so they cannot be recorded on the new ones.
    release_event_pool();
  }
  mode_ = mode;
  device_ids_ = device_ids;
  counter_id_ = counter_id;
  batches_.clear();
  batch_of_layer_.assign(table_.num_layers() > 0 ? table_.num_layers() : 0, -1);
  poll_stop_.store(false, std::memory_order_release);
  poll_next_.store(0, std::memory_order_release);
  poll_failed_.store(false, std::memory_order_release);
  {
    std::lock_guard<std::mutex> lk(poll_error_mtx_);
    poll_error_msg_.clear();
  }
}

void LayerNotifier::post_empty(int layer) {
  table_.post(counter_id_, layer);
}

cudaEvent_t LayerNotifier::event_for(int layer, int rank) {
  const int ranks = static_cast<int>(device_ids_.size());
  if (pool_ranks_ != ranks || layer >= pool_layers_) {
    // Grow to cover this layer, keeping what is already created. A rank-count
    // change cannot be a grow -- reset() released the pool for that case.
    const int new_layers = std::max(layer + 1, pool_layers_ * 2);
    std::vector<cudaEvent_t> grown(static_cast<size_t>(new_layers) * ranks,
                                   nullptr);
    for (int l = 0; l < pool_layers_; ++l) {
      for (int r = 0; r < pool_ranks_ && r < ranks; ++r) {
        grown[static_cast<size_t>(l) * ranks + r] =
            event_pool_[static_cast<size_t>(l) * pool_ranks_ + r];
      }
    }
    event_pool_.swap(grown);
    pool_layers_ = new_layers;
    pool_ranks_ = ranks;
    pool_device_ids_ = device_ids_;
  }
  cudaEvent_t &slot = event_pool_[static_cast<size_t>(layer) * ranks + rank];
  if (slot == nullptr) {
    cudaSetDevice(device_ids_[rank]);
    cudaEventCreateWithFlags(&slot, cudaEventDisableTiming);
  }
  return slot;
}

void LayerNotifier::release_event_pool() {
  if (!pool_device_ids_.empty()) {
    int prev_device = 0;
    cudaGetDevice(&prev_device);
    for (int l = 0; l < pool_layers_; ++l) {
      for (int r = 0; r < pool_ranks_; ++r) {
        cudaEvent_t e = event_pool_[static_cast<size_t>(l) * pool_ranks_ + r];
        if (e != nullptr) {
          cudaSetDevice(pool_device_ids_[r]);
          cudaEventDestroy(e);
        }
      }
    }
    cudaSetDevice(prev_device);
  }
  event_pool_.clear();
  pool_layers_ = 0;
  pool_ranks_ = 0;
  pool_device_ids_.clear();
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
      b.per_rank_events[r] = event_for(layer, static_cast<int>(r));
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
  {
    std::lock_guard<std::mutex> lk(poll_mtx_);
    if (!poll_thread_.joinable()) {
      poll_thread_ = std::thread(&LayerNotifier::polling_loop, this);
    }
    poll_active_ = true;
  }
  poll_cv_.notify_one();
}

void LayerNotifier::polling_loop() {
  // No device to save and restore: this thread is ours for the object's whole
  // life, and run_polling_round() sets the device it needs on every query.
  for (;;) {
    {
      // Park until arm() hands over a round, or the destructor asks us out.
      std::unique_lock<std::mutex> lk(poll_mtx_);
      poll_cv_.wait(lk, [&] { return poll_active_ || poll_exit_; });
      if (poll_exit_) {
        break;
      }
    }
    run_polling_round();
    {
      std::lock_guard<std::mutex> lk(poll_mtx_);
      poll_active_ = false;
    }
    poll_cv_.notify_all();
  }
}

void LayerNotifier::run_polling_round() {
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
}

bool LayerNotifier::wait(double timeout_s, std::string *error_out) {
  if (mode_ != LayerNotifyMode::POLLING || !poll_thread_.joinable()) {
    return !poll_failed_.load(std::memory_order_acquire);
  }
  {
    std::unique_lock<std::mutex> lk(poll_mtx_);
    const bool drained =
        poll_cv_.wait_for(lk, std::chrono::duration<double>(timeout_s),
                          [&] { return !poll_active_; });
    if (!drained) {
      if (error_out != nullptr) {
        *error_out = "per-layer notification did not drain within " +
                     std::to_string(timeout_s) + "s";
      }
      // Leave the round running: it still references the batch events, and
      // the next reset() quiesces it. Returning false lets the caller fail
      // the op rather than hang.
      return false;
    }
  }
  if (poll_failed_.load(std::memory_order_acquire)) {
    if (error_out != nullptr) {
      std::lock_guard<std::mutex> lk(poll_error_mtx_);
      *error_out = poll_error_msg_;
    }
    return false;
  }
  return true;
}

void LayerNotifier::quiesce_polling() {
  if (!poll_thread_.joinable()) {
    return;
  }
  poll_stop_.store(true, std::memory_order_release);
  std::unique_lock<std::mutex> lk(poll_mtx_);
  poll_cv_.wait(lk, [&] { return !poll_active_; });
}

void LayerNotifier::shutdown_polling() {
  if (!poll_thread_.joinable()) {
    return;
  }
  poll_stop_.store(true, std::memory_order_release);
  {
    std::unique_lock<std::mutex> lk(poll_mtx_);
    poll_cv_.wait(lk, [&] { return !poll_active_; });
    poll_exit_ = true;
  }
  poll_cv_.notify_all();
  poll_thread_.join();
}

void LayerNotifier::destroy_events() {
  // The events themselves belong to event_pool_ and are reused across
  // transfers; a LayerBatch only borrows them. So this drops the borrowed
  // handles and the batch list -- release_event_pool() is what frees.
  for (LayerBatch &b : batches_) {
    b.per_rank_events.clear();
  }
  batches_.clear();
}

} // namespace flexkv
