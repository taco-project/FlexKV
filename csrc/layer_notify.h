/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * "Layer L has landed" -- the notification half of layerwise, on its own.
 *
 * ``layerwise.cpp`` fuses three things: a launch shape (one H2D batch per
 * layer), a stride table (single- vs multi-group), and this -- telling the
 * consumer, through the eventfds it handed over, that layer L is readable.
 * Only the third is a contract; the other two are RegionBatchGroup's job
 * already.  Split out, ``LayerNotifier`` is ~150 lines that any launcher can
 * drive, and "layerwise" stops being a worker class and becomes an argument.
 *
 * Two ways to learn a layer finished, both preserved from layerwise.cpp
 * because both are load-bearing (tests parametrize over them):
 *
 * ``HOSTFUNC``
 *     cudaLaunchHostFunc after the layer's last launch on each rank.  The
 *     driver's callback thread posts the fd.  Simple, but the callback cannot
 *     call any CUDA API and adds a launch to every rank per layer.
 *
 * ``POLLING``
 *     A cudaEvent per (layer, rank) plus one thread doing cudaEventQuery.
 *     Costs a thread; wins when the callback thread is contended.
 *
 * The eventfd write is ``uint64_t 1`` per tp_rank per layer -- a semaphore
 * post, one token per layer per transfer.  sglang consumes exactly one, so
 * writing 2 (or writing once for a two-layer batch) silently corrupts its
 * accounting rather than failing; that is why PER_LAYER pins granularity to 1.
 */
#pragma once

#include <atomic>
#include <condition_variable>
#include <cuda_runtime.h>
#include <mutex>
#include <string>
#include <thread>
#include <torch/extension.h>
#include <vector>

namespace flexkv {

enum class LayerNotifyMode { HOSTFUNC, POLLING };

// Parse the Python-side string; anything unrecognised is HOSTFUNC, which is
// the historical default.
LayerNotifyMode parse_layer_notify_mode(const std::string &s);

// The consumer's eventfd table: [num_counters, tp_size, num_layers], flat.
// Owns nothing but ints -- the fds themselves belong to the consumer process
// and were duplicated into ours by SCM_RIGHTS.
class LayerEventfdTable {
public:
  LayerEventfdTable() = default;
  // ``fds_tensor`` empty means "no consumer wants per-layer notification";
  // every post() is then a no-op and enabled() is false.
  LayerEventfdTable(const torch::Tensor &fds_tensor, int tp_size,
                    int num_layers);

  bool enabled() const { return enabled_; }
  int num_counters() const { return num_counters_; }
  int num_layers() const { return num_layers_; }

  // Post one semaphore token to every tp_rank's fd for ``layer``.
  void post(int counter_id, int layer) const;

private:
  bool enabled_ = false;
  int tp_size_ = 0;
  int num_layers_ = 0;
  int num_counters_ = 0;
  std::vector<int> fds_;
};

// Drives one transfer's worth of per-layer notification.
//
// Lifecycle, once per transfer:
//   reset(mode, device_ids, counter_id)
//   begin_layer(L) / record(L, rank, stream) ... for each launched layer
//   arm()                       -- starts the polling thread, if polling
//   wait(timeout_s, &err)       -- blocks until every layer has been posted
//
// A layer with no work at all is posted by ``post_empty(L)`` at reset time:
// the consumer is waiting on every layer's fd regardless of whether this
// model has state for it, so skipping the post hangs it.
class LayerNotifier {
public:
  LayerNotifier() = default;
  ~LayerNotifier();

  LayerNotifier(const LayerNotifier &) = delete;
  LayerNotifier &operator=(const LayerNotifier &) = delete;

  void set_table(LayerEventfdTable table) { table_ = std::move(table); }
  const LayerEventfdTable &table() const { return table_; }
  bool enabled() const { return table_.enabled(); }

  // Drain and release the previous transfer's state, then start a new one.
  // Safe to call when nothing is in flight.
  void reset(LayerNotifyMode mode, const std::vector<int> &device_ids,
             int counter_id);

  // Post a layer that has no work, immediately and from the calling thread.
  void post_empty(int layer);

  // Declare that ``layer`` has launches coming on every rank. Must be called
  // before the first record() for that layer, in the order the layers will
  // complete (polling walks them in this order).
  void begin_layer(int layer);

  // Record the completion marker for ``layer`` on ``rank``'s stream, after
  // that layer's last launch on that stream.
  void record(int layer, int rank, cudaStream_t stream);

  // Start the polling thread (no-op in HOSTFUNC mode, where the driver's
  // callbacks are already armed by record()).
  void arm();

  // Block until every declared layer has been posted. Returns false on a
  // hard CUDA error or timeout, filling ``error_out`` when non-null.
  //
  // In HOSTFUNC mode there is nothing to wait for here: the caller
  // synchronizes the streams itself, which is strictly stronger.
  bool wait(double timeout_s, std::string *error_out);

private:
  struct LayerBatch {
    int layer = 0;
    std::vector<cudaEvent_t> per_rank_events; // POLLING only
    bool notified = false;                    // POLLING only
    // HOSTFUNC only: arrival counter shared by this layer's per-rank
    // callbacks. Owned by the callbacks, not by this struct -- a callback can
    // still be pending when the next reset() clears batches_, so freeing it
    // here would be a use-after-free in the driver's callback thread.
    std::atomic<int> *hostfunc_counter = nullptr;
  };

  // A cudaEvent is a driver object, not a value: creating one per (layer,
  // rank) per transfer costs ~64 create+destroy pairs on a 64-layer model and
  // buys nothing, because the *shape* of what we wait on never changes across
  // transfers -- only which layers happen to carry work. So the events live
  // here, indexed [layer * num_ranks + rank], and reset() rewinds the
  // bookkeeping instead of freeing them. Grown on demand; released only in
  // the destructor and when the rank set itself changes.
  cudaEvent_t event_for(int layer, int rank);
  void release_event_pool();

  // Outer park/wake loop, run by poll_thread_ for the object's whole life.
  void polling_loop();
  // One transfer's worth of sweeping: walk batches_ in order, posting each
  // layer as its per-rank events complete. Returns when the last layer has
  // been posted, on a hard CUDA error, or when poll_stop_ is set.
  void run_polling_round();
  // Bring the current polling round to a stop and wait for the thread to
  // leave it. The thread itself stays alive for the next transfer.
  void quiesce_polling();
  // Permanently stop and join the polling thread. Destructor only.
  void shutdown_polling();
  void destroy_events();

  LayerEventfdTable table_;
  LayerNotifyMode mode_ = LayerNotifyMode::HOSTFUNC;
  std::vector<int> device_ids_;
  int counter_id_ = 0;

  std::vector<LayerBatch> batches_;
  // batch index for a layer, so record() is O(1) without assuming the caller
  // records in the same order it began.
  std::vector<int> batch_of_layer_;

  // Persistent [layer * pool_ranks_ + rank] event pool; see event_for().
  std::vector<cudaEvent_t> event_pool_;
  int pool_layers_ = 0;
  int pool_ranks_ = 0;
  std::vector<int> pool_device_ids_; // the devices event_pool_ was built on

  std::atomic<bool> poll_stop_{false};
  std::atomic<int> poll_next_{0};
  std::atomic<bool> poll_failed_{false};
  std::mutex poll_error_mtx_;
  std::string poll_error_msg_;

  // The polling thread is created once and parked between transfers: creating
  // and joining a thread per transfer is ~40us of kernel work on the critical
  // path, for a thread whose body is identical every time. ``poll_active_``
  // is the handshake -- arm() sets it and notifies, the thread clears it when
  // the last layer has been posted (or on error), and quiesce_polling() waits
  // for that. ``poll_exit_`` is the one-way shutdown flag.
  std::thread poll_thread_;
  std::mutex poll_mtx_;
  std::condition_variable poll_cv_;
  bool poll_active_ = false;
  bool poll_exit_ = false;
};

} // namespace flexkv
