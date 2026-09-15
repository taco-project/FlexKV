/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Many regions, one fan-out.
 *
 * A "region" is one contiguous slice of KV that shares a stride table: full
 * attention, SWA, a DSv4 indexer, a linear-attention state.  A modern model
 * has several per layer, and today each one is a separate round trip:
 * ``GPUCPUTransferWorker._transfer_impl`` loops over
 * ``tp_group_transfer_groups`` and calls ``tp_group_transfer`` per group.
 * Every iteration crosses the Python/C++ boundary, fans out to num_gpus
 * threads, and joins them again.  For a four-region layer that is four
 * fan-out/join pairs and four GIL round trips to move data that could have
 * gone onto the same stream back to back.
 *
 * Worse, the loop is where the model's shape leaks upward: Python has to know
 * that DSv4 has an indexer, that this model has SWA, that those need
 * different strides.  That is the "one worker per model" complaint --
 * a region list is the thing that varies, so a region list is what should be
 * passed, not a control-flow shape baked into a worker class.
 *
 * RegionBatchGroup registers the regions once (their pointers and strides do
 * not change between transfers) and then takes a batch of *requests* -- which
 * blocks, which layers, which direction -- and issues all of them in one
 * fan-out.  Inside each rank's thread the regions launch in order onto that
 * rank's stream, so region N+1's launch overlaps region N's copy.
 *
 * Each request names its own backend, so "copy engine for the big main-KV
 * region, SM kernel for the tiny indexer" becomes a per-region field rather
 * than a process-wide env var.
 */
#pragma once

#include "ce_transfer.h"
#include "device_thread_pool.h"
#include "gtensor_handler.cuh"
#include "layer_notify.h"
#include "transfer_backend.h"

#include <memory>
#include <string>
#include <torch/extension.h>
#include <vector>

namespace flexkv {

// Static geometry of one region.  Registered once at construction; nothing
// here changes between transfers.
struct RegionDesc {
  std::string name; // "full", "swa", "indexer", ... -- for errors and traces

  // ---- host side ----
  int64_t cpu_ptr = 0; // already offset to this region's base
  int64_t cpu_kv_stride_in_bytes = 0;
  int64_t cpu_layer_stride_in_bytes = 0;
  int64_t cpu_block_stride_in_bytes = 0;
  int64_t cpu_tp_stride_in_bytes = 0;

  // ---- device side, one entry per rank ----
  std::vector<int64_t> gpu_block_ptrs_flat; // [num_gpus * num_tensors_per_gpu]
  int num_tensors_per_gpu = 0;
  std::vector<int64_t> gpu_kv_strides_in_bytes;
  std::vector<int64_t> gpu_block_strides_in_bytes;
  std::vector<int64_t> gpu_layer_strides_in_bytes;
  std::vector<int64_t> gpu_chunk_sizes_in_bytes;

  // ---- shape ----
  int num_layers = 0; // layers *in this region*, not in the model
  int kv_dim = 1;
  int num_kv_heads = 1;
};

// How the ranks of a TP group divide up a *rank-shared* region -- one where
// every rank holds identical bytes (num_kv_heads == 1, e.g. MLA).  Head-sharded
// regions ignore this entirely: there the split is the head split.
//
// Only D2H is affected.  On H2D every rank needs the whole thing regardless of
// who wrote it, so every mode reads the same bytes.
enum class RankShareMode {
  SHARDED = 0,       // each rank writes its 1/N slice of the chunk
  ALL_WRITE,         // every rank writes a full copy into its own CPU region
  RANK0_ONLY,        // only the designated rank writes
  LAYER_PARALLEL,    // ranks split the layer range
  RANK_ROTATE,       // rank0_only with a round-robin designated rank
};

// Parse the Python-side string.  Unknown values degrade to SHARDED with a
// warning, which is the behaviour TPTransferThreadGroup has always had.
RankShareMode parse_rank_share_mode(const std::string &s);

// One transfer of one region.  This is what changes per call.
struct RegionRequest {
  int region_index = 0; // into the registered RegionDesc list
  torch::Tensor gpu_block_id_tensor;
  torch::Tensor cpu_block_id_tensor;
  int layer_id = 0;          // first layer, local to the region
  int layer_granularity = 0; // 0 = all layers of the region
  bool is_host_to_device = false;
  int transfer_num_cta = 4;
  TransferBackendKind backend = TransferBackendKind::AUTO;
  // Only consulted when backend == AUTO, where it reproduces the historical
  // use_ce_transfer decision.
  bool use_ce_transfer = false;
  // Rank-shared regions only (the region's own num_kv_heads == 1).  Carried
  // per request rather than per region because the mode is a policy the caller
  // picks, not geometry: the same region can be written rank0_only on one
  // transfer and layer_parallel on the next.
  RankShareMode rank_share_mode = RankShareMode::SHARDED;
  int designated_rank = 0; // RANK0_ONLY; RANK_ROTATE resolves its own
  // Which *original model* layer this request completes, or -1 for "not part
  // of a per-layer milestone".  Only consulted by submit_layerwise(): the
  // notifier posts layer L's eventfd once every request tagged L has landed
  // on every participating rank.
  //
  // It is a separate field from ``layer_id`` because they answer different
  // questions.  layer_id is an offset into this region's own layer range; a
  // model layer can be covered by several regions (main KV + SWA + indexer),
  // each with its own local numbering, and it is the *model* layer the
  // consumer waits on.
  int milestone_layer = -1;
};

class RegionBatchGroup {
public:
  // ``gpu_device_ids`` has one entry per rank; every region must supply
  // per-rank arrays of that same length.
  RegionBatchGroup(const std::vector<int64_t> &gpu_device_ids,
                   const std::vector<RegionDesc> &regions,
                   CETransferConfig ce_config = CETransferConfig{});
  ~RegionBatchGroup();

  RegionBatchGroup(const RegionBatchGroup &) = delete;
  RegionBatchGroup &operator=(const RegionBatchGroup &) = delete;

  int num_regions() const { return static_cast<int>(regions_.size()); }
  int num_gpus() const { return pool_->size(); }

  // Issue every request.  One fan-out across ranks; within a rank the
  // requests are launched in the order given, onto that rank's single stream,
  // which is what makes them overlap instead of serializing on a join.
  //
  // sync=true drains before returning (the safe default and the historical
  // contract).  sync=false returns after the launches; the caller must call
  // wait_all_streams() before touching the data.
  void submit(const std::vector<RegionRequest> &requests, bool sync = true);

  // submit() plus per-layer notification -- what used to be a whole worker
  // class (LayerwiseTransferWorker + csrc/layerwise.cpp).
  //
  // Requests are grouped by ``milestone_layer`` and launched in ascending
  // layer order, so layer L's copies are all on the stream before layer L+1's.
  // After the last request of layer L on a rank, that rank records a
  // completion marker; when every rank's marker for L has fired, the
  // consumer's eventfd for L is posted -- which is what lets it start layer
  // L's attention while L+1 is still in flight.
  //
  // ``empty_layers`` are model layers the consumer waits on but this model has
  // no state for; they are posted immediately. Skipping them hangs the
  // consumer, which does not know they are empty.
  //
  // Returns after the launches; the caller must wait_layer_completion() before
  // recycling the source blocks. Requests with milestone_layer < 0 are
  // launched first, before any milestone layer, and are covered by the first
  // layer's marker.
  void submit_layerwise(const std::vector<RegionRequest> &requests,
                        const std::vector<int> &empty_layers, int counter_id);

  // Block until every milestone layer of the last submit_layerwise() has been
  // posted and every stream has drained. False on timeout or CUDA error.
  bool wait_layer_completion(double timeout_s, std::string *error_out);

  // Hand over the consumer's eventfd table, shaped
  // [num_counters, tp_size, num_layers]. An empty tensor means the consumer
  // wants no per-layer notification: submit_layerwise() then degrades to
  // launch-in-layer-order with a single drain, which is exactly
  // CompletionContract.WHOLE.
  void set_layer_eventfds(const torch::Tensor &fds_tensor, int tp_size,
                          int num_layers,
                          const std::string &notify_mode = "hostfunc");

  bool layer_notification_enabled() const { return notifier_.enabled(); }

  void wait_all_streams();

  // Re-point one region's device tensors after a GPU hot remap.  Drains
  // first: the old pointers may still be in flight.
  void update_region_gpu_ptrs(int region_index,
                              const std::vector<int64_t> &gpu_block_ptrs_flat);

private:
  // Per-region device-side state, resolved once at construction.
  struct RegionState {
    RegionDesc desc;
    BackendType tensor_kind = BackendType::VLLM;
    void **gpu_blocks = nullptr; // pinned, [num_gpus * num_tensors_per_gpu]
    std::vector<GTensorHandler> handlers; // one per rank
  };

  // The rank-local slice of one request, i.e. exactly what a backend needs.
  RegionTransferArgs build_args(const RegionRequest &req, int rank) const;

  // Whether ``rank`` participates in this request at all.  Only rank-shared
  // D2H can answer false (rank0_only's non-designated ranks, layer_parallel's
  // ranks that drew zero layers).
  bool rank_participates(const RegionRequest &req, int rank) const;

  // Shared prologue of submit() and submit_layerwise(): validate every
  // request, then resolve RANK_ROTATE on the caller's thread (reading the
  // cursor inside the per-rank lambdas would let two ranks disagree on who
  // was designated). Returns the list to launch -- ``requests_in`` itself
  // when no rotation was needed, otherwise ``scratch``.
  const std::vector<RegionRequest> &
  prepare(const std::vector<RegionRequest> &requests_in,
          std::vector<RegionRequest> &scratch);

  std::unique_ptr<DeviceThreadPool> pool_;
  std::vector<RegionState> regions_;
  CETransferConfig ce_config_;
  LayerNotifier notifier_;
  LayerNotifyMode layer_notify_mode_ = LayerNotifyMode::HOSTFUNC;
  std::vector<int> device_ids_;
  // RANK_ROTATE's round-robin cursor.  Advanced once per submit() per
  // rotating request, on the caller's thread before the fan-out, so every
  // rank in a fan-out agrees on who was designated.
  int rotate_counter_ = 0;
};

} // namespace flexkv
