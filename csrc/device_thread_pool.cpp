/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */
#include "device_thread_pool.h"

#include <stdexcept>

namespace flexkv {

DeviceThreadPool::DeviceThreadPool(const std::vector<int> &device_ids)
    : device_ids_(device_ids) {
  const int n = static_cast<int>(device_ids_.size());
  // Snapshot the driver-level current device: the cudaSetDevice calls below
  // are raw, so nothing else restores it for us.
  int prev_device = 0;
  cudaGetDevice(&prev_device);

  queues_.resize(n);
  mtxs_ = std::vector<std::mutex>(n);
  cvs_ = std::vector<std::condition_variable>(n);
  streams_.assign(n, nullptr);

  for (int i = 0; i < n; ++i) {
    cudaError_t err = cudaSetDevice(device_ids_[i]);
    if (err == cudaSuccess) {
      err = cudaStreamCreate(&streams_[i]);
    }
    if (err != cudaSuccess) {
      // Destroy what we already made: the destructor will not run for a ctor
      // that throws.
      for (int j = 0; j < i; ++j) {
        if (streams_[j] != nullptr &&
            cudaSetDevice(device_ids_[j]) == cudaSuccess) {
          cudaStreamDestroy(streams_[j]);
        }
      }
      cudaSetDevice(prev_device);
      cudaGetLastError();
      throw std::runtime_error(
          std::string("DeviceThreadPool: device setup failed for device ") +
          std::to_string(device_ids_[i]) + ": " + cudaGetErrorString(err));
    }
  }
  cudaSetDevice(prev_device);

  for (int i = 0; i < n; ++i) {
    threads_.emplace_back([this, i]() {
      // Once, for the life of the thread. Every task that runs here inherits
      // the right device without touching it.
      cudaSetDevice(device_ids_[i]);
      while (true) {
        Task task;
        {
          std::unique_lock<std::mutex> lk(mtxs_[i]);
          cvs_[i].wait(lk, [&] { return stop_ || !queues_[i].empty(); });
          if (stop_ && queues_[i].empty()) {
            return;
          }
          task = std::move(queues_[i].front());
          queues_[i].pop();
        }
        task();
      }
    });
  }
}

void DeviceThreadPool::shutdown() {
  if (stop_.exchange(true)) {
    return;
  }
  for (auto &cv : cvs_) {
    cv.notify_all();
  }
  for (auto &t : threads_) {
    if (t.joinable()) {
      t.join();
    }
  }
}

DeviceThreadPool::~DeviceThreadPool() {
  int prev_device = 0;
  cudaGetDevice(&prev_device);

  // Join first: a live worker thread could still be enqueuing onto a stream
  // we are about to destroy.
  shutdown();

  for (size_t i = 0; i < streams_.size(); ++i) {
    if (streams_[i] == nullptr) {
      continue;
    }
    if (cudaSetDevice(device_ids_[i]) != cudaSuccess) {
      continue; // device gone (driver shutdown); nothing safe left to do
    }
    cudaStreamSynchronize(streams_[i]);
    cudaStreamDestroy(streams_[i]);
    streams_[i] = nullptr;
  }
  cudaSetDevice(prev_device);
  // Swallow teardown errors so the next cudaGetLastError() caller does not
  // mistake them for a transfer failure.
  cudaGetLastError();
}

std::future<void> DeviceThreadPool::enqueue(int idx, Task task) {
  auto pkg = std::make_shared<std::packaged_task<void()>>(std::move(task));
  auto fut = pkg->get_future();
  {
    std::lock_guard<std::mutex> lk(mtxs_[idx]);
    queues_[idx].emplace([pkg] { (*pkg)(); });
  }
  cvs_[idx].notify_one();
  return fut;
}

void DeviceThreadPool::run_on_all(const std::function<void(int)> &body,
                                  const char *what) {
  std::mutex error_mtx;
  std::string error_msg;
  auto record = [&](const std::string &msg) {
    std::lock_guard<std::mutex> lk(error_mtx);
    if (error_msg.empty()) {
      error_msg = msg;
    }
  };

  std::vector<std::future<void>> futures;
  futures.reserve(device_ids_.size());
  for (int i = 0; i < size(); ++i) {
    futures.emplace_back(enqueue(i, [&, i]() {
      try {
        body(i);
      } catch (const std::exception &e) {
        record(std::string("rank ") + std::to_string(i) + ": " + e.what());
      } catch (...) {
        // A non-std::exception throw would otherwise escape the
        // packaged_task and resurface at f.get() as an opaque failure.
        record(std::string("rank ") + std::to_string(i) +
               ": unknown exception");
      }
    }));
  }
  for (auto &f : futures) {
    try {
      f.get();
    } catch (const std::exception &e) {
      record(std::string("future: ") + e.what());
    } catch (...) {
      record("future: unknown exception");
    }
  }
  if (!error_msg.empty()) {
    throw std::runtime_error(std::string(what) + " failed: " + error_msg);
  }
}

void DeviceThreadPool::sync_all_streams() {
  run_on_all(
      [this](int i) {
        cudaError_t err = cudaStreamSynchronize(streams_[i]);
        if (err != cudaSuccess) {
          throw std::runtime_error(cudaGetErrorString(err));
        }
      },
      "wait_all_streams");
}

} // namespace flexkv
