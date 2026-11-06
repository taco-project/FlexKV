/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * One thread + one stream per device, with cudaSetDevice applied once.
 *
 * TPTransferThreadGroup, TPGDSTransferThreadGroup and the (now deleted)
 * LayerwiseTransferGroup each grew their own copy of this: a vector of
 * threads, a vector of queues,
 * a vector of mutexes, a vector of condition_variables, a stop flag, and a
 * destructor that has to join before destroying streams.  The copies drifted
 * -- one of them leaked every stream it created for the process lifetime
 * until that was fixed in only that copy.
 *
 * The invariant this exists to hold: a device's work runs on a thread that
 * called cudaSetDevice(device) exactly once at start-up.  Doing it per task
 * is both slower and a correctness trap, because a task that throws between
 * set and restore leaves the calling thread's current device wrong.
 */
#pragma once

#include <atomic>
#include <condition_variable>
#include <cuda_runtime.h>
#include <functional>
#include <future>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

namespace flexkv {

class DeviceThreadPool {
public:
  using Task = std::function<void()>;

  // Creates one thread and one stream per entry in ``device_ids``.  Throws if
  // any cudaSetDevice/cudaStreamCreate fails, after cleaning up what it had
  // already created.
  explicit DeviceThreadPool(const std::vector<int> &device_ids);
  ~DeviceThreadPool();

  DeviceThreadPool(const DeviceThreadPool &) = delete;
  DeviceThreadPool &operator=(const DeviceThreadPool &) = delete;

  int size() const { return static_cast<int>(device_ids_.size()); }
  int device_id(int idx) const { return device_ids_[idx]; }
  cudaStream_t stream(int idx) const { return streams_[idx]; }

  std::future<void> enqueue(int idx, Task task);

  // Run ``body(idx)`` on every device's thread and wait for all of them.
  // Collects the first error rather than the last: the first is the one
  // closest to the root cause, later ranks usually report downstream fallout.
  // Every future is waited on even after a failure -- returning early would
  // let a lambda's captured references die while a worker thread still reads
  // them.  Throws std::runtime_error prefixed with ``what`` if any failed.
  void run_on_all(const std::function<void(int)> &body, const char *what);

  // cudaStreamSynchronize on every device, from that device's own thread.
  void sync_all_streams();

private:
  std::vector<int> device_ids_;
  std::vector<cudaStream_t> streams_;
  std::vector<std::thread> threads_;
  std::vector<std::queue<Task>> queues_;
  std::vector<std::mutex> mtxs_;
  std::vector<std::condition_variable> cvs_;
  std::atomic<bool> stop_{false};

  void shutdown();
};

} // namespace flexkv
