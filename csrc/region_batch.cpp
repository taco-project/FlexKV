/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */
#include "region_batch.h"

#include "logging.h"
#include <algorithm>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <stdexcept>

namespace flexkv {

namespace {

// Same rule as TPTransferThreadGroup: the count of tensors a rank exposes
// tells you which attention backend allocated them.  Kept as a free function
// so both classes read from one statement of the rule.
BackendType infer_tensor_kind(int num_tensors_per_gpu, int num_layers,
                              const std::string &region_name) {
  if (num_tensors_per_gpu == 1) {
    return BackendType::TRTLLM;
  }
  if (num_tensors_per_gpu == num_layers) {
    return BackendType::VLLM;
  }
  if (num_tensors_per_gpu == num_layers * 2) {
    return BackendType::SGLANG;
  }
  throw std::runtime_error(
      "region '" + region_name + "': cannot infer tensor layout from " +
      std::to_string(num_tensors_per_gpu) + " tensors per GPU with " +
      std::to_string(num_layers) + " layers");
}

void check_per_rank_size(const std::vector<int64_t> &v, size_t expected,
                         const char *field, const std::string &region_name) {
  if (v.size() != expected) {
    throw std::invalid_argument(
        "region '" + region_name + "': " + field + " has " +
        std::to_string(v.size()) + " entries, expected " +
        std::to_string(expected) + " (one per rank)");
  }
}

// The layer range rank ``r`` of ``n`` draws from ``total`` layers, remainder
// spread over the low ranks.  Returned as (start_offset, count); count == 0
// means this rank has nothing to do.  One statement of the split so the
// "does this rank participate" test and the "which layers" computation cannot
// disagree -- in TPTransferThreadGroup they were two copies of the same four
// lines, 40 lines apart.
std::pair<int, int> layer_slice(int total, int n, int r) {
  const int per = total / n;
  const int rem = total % n;
  const int count = (r < rem) ? per + 1 : per;
  const int start = (r < rem) ? r * (per + 1)
                              : rem * (per + 1) + (r - rem) * per;
  return {start, count};
}

} // namespace

RankShareMode parse_rank_share_mode(const std::string &s) {
  if (s == "sharded") return RankShareMode::SHARDED;
  if (s == "all_write") return RankShareMode::ALL_WRITE;
  if (s == "rank0_only") return RankShareMode::RANK0_ONLY;
  if (s == "layer_parallel") return RankShareMode::LAYER_PARALLEL;
  if (s == "rank_rotate") return RankShareMode::RANK_ROTATE;
  FLEXKV_LOG_WARNING(
      "operation=transfer_config act=fallback status=degraded "
      "field=kv_shared_across_ranks_mode value=\"%s\" fallback=sharded",
      s.c_str());
  return RankShareMode::SHARDED;
}

RegionBatchGroup::RegionBatchGroup(const std::vector<int64_t> &gpu_device_ids,
                                   const std::vector<RegionDesc> &regions,
                                   CETransferConfig ce_config)
    : ce_config_(ce_config) {
  if (gpu_device_ids.empty()) {
    throw std::invalid_argument("RegionBatchGroup: no devices");
  }
  if (regions.empty()) {
    throw std::invalid_argument("RegionBatchGroup: no regions");
  }
  const c10::cuda::CUDAGuard restore_device_on_exit(c10::cuda::current_device());

  std::vector<int> device_ids;
  device_ids.reserve(gpu_device_ids.size());
  for (int64_t d : gpu_device_ids) {
    device_ids.push_back(static_cast<int>(d));
  }
  const size_t n_ranks = device_ids.size();

  // Validate everything *before* allocating anything: a throw halfway through
  // would leak the pinned buffers of the regions already built, since the
  // destructor does not run for a constructor that throws.
  for (const RegionDesc &d : regions) {
    check_per_rank_size(d.gpu_kv_strides_in_bytes, n_ranks,
                        "gpu_kv_strides_in_bytes", d.name);
    check_per_rank_size(d.gpu_block_strides_in_bytes, n_ranks,
                        "gpu_block_strides_in_bytes", d.name);
    check_per_rank_size(d.gpu_layer_strides_in_bytes, n_ranks,
                        "gpu_layer_strides_in_bytes", d.name);
    check_per_rank_size(d.gpu_chunk_sizes_in_bytes, n_ranks,
                        "gpu_chunk_sizes_in_bytes", d.name);
    if (d.num_tensors_per_gpu <= 0) {
      throw std::invalid_argument("region '" + d.name +
                                  "': num_tensors_per_gpu must be positive");
    }
    const size_t expected_ptrs = n_ranks * d.num_tensors_per_gpu;
    if (d.gpu_block_ptrs_flat.size() != expected_ptrs) {
      throw std::invalid_argument(
          "region '" + d.name + "': gpu_block_ptrs_flat has " +
          std::to_string(d.gpu_block_ptrs_flat.size()) + " entries, expected " +
          std::to_string(expected_ptrs));
    }
    // Throws for an unrecognised count; do it here so the error names the
    // region rather than surfacing later from inside a worker thread.
    (void)infer_tensor_kind(d.num_tensors_per_gpu, d.num_layers, d.name);
  }

  pool_ = std::make_unique<DeviceThreadPool>(device_ids);
  device_ids_ = device_ids;

  regions_.reserve(regions.size());
  for (const RegionDesc &d : regions) {
    RegionState st;
    st.desc = d;
    st.tensor_kind = infer_tensor_kind(d.num_tensors_per_gpu, d.num_layers,
                                       d.name);

    const size_t n_ptrs = n_ranks * d.num_tensors_per_gpu;
    // Pinned so the pointer table itself is not a page-fault source for a
    // kernel that dereferences it. Mirrors TPTransferThreadGroup.
    cudaError_t err =
        cudaMallocHost((void **)&st.gpu_blocks, n_ptrs * sizeof(void *));
    if (err != cudaSuccess) {
      // Free the regions built so far; ~RegionBatchGroup will not run.
      for (RegionState &done : regions_) {
        cudaFreeHost(done.gpu_blocks);
      }
      regions_.clear();
      throw std::runtime_error(std::string("cudaMallocHost failed: ") +
                               cudaGetErrorString(err));
    }
    for (size_t i = 0; i < n_ptrs; ++i) {
      st.gpu_blocks[i] = reinterpret_cast<void *>(d.gpu_block_ptrs_flat[i]);
    }

    st.handlers.reserve(n_ranks);
    for (size_t r = 0; r < n_ranks; ++r) {
      int64_t **rank_ptrs = reinterpret_cast<int64_t **>(
          st.gpu_blocks + r * d.num_tensors_per_gpu);
      st.handlers.emplace_back(st.tensor_kind, rank_ptrs, d.num_layers,
                               d.gpu_kv_strides_in_bytes[r],
                               d.gpu_block_strides_in_bytes[r],
                               d.gpu_layer_strides_in_bytes[r]);
    }
    regions_.push_back(std::move(st));
  }

  FLEXKV_LOG_DEBUG("operation=region_batch_init regions=%zu ranks=%zu",
                   regions_.size(), n_ranks);
}

RegionBatchGroup::~RegionBatchGroup() {
  // Destroy the pool first: joining its threads guarantees nobody is still
  // reading the pointer tables freed below.
  pool_.reset();
  for (RegionState &st : regions_) {
    if (st.gpu_blocks != nullptr) {
      cudaFreeHost(st.gpu_blocks);
      st.gpu_blocks = nullptr;
    }
  }
  cudaGetLastError();
}

RegionTransferArgs RegionBatchGroup::build_args(const RegionRequest &req,
                                                int rank) const {
  const RegionState &st = regions_[req.region_index];
  const RegionDesc &d = st.desc;

  RegionTransferArgs a;
  a.num_blocks = static_cast<int>(req.gpu_block_id_tensor.numel());
  a.gpu_block_ids =
      static_cast<int64_t *>(req.gpu_block_id_tensor.data_ptr());
  a.cpu_block_ids =
      static_cast<int64_t *>(req.cpu_block_id_tensor.data_ptr());
  a.start_layer_id = req.layer_id;
  a.num_layers =
      req.layer_granularity > 0 ? req.layer_granularity : d.num_layers;
  a.kv_dim = d.kv_dim;
  a.chunk_size_in_bytes = d.gpu_chunk_sizes_in_bytes[rank];
  a.is_host_to_device = req.is_host_to_device;

  a.tensor_kind = st.tensor_kind;
  a.gpu_tensor_handler = st.handlers[rank];
  a.gpu_block_stride_in_bytes = d.gpu_block_strides_in_bytes[rank];
  a.gpu_startoff_inside_chunks = 0;

  a.cpu_ptr = reinterpret_cast<void *>(d.cpu_ptr);
  a.cpu_kv_stride_in_bytes = d.cpu_kv_stride_in_bytes;
  a.cpu_layer_stride_in_bytes = d.cpu_layer_stride_in_bytes;
  a.cpu_block_stride_in_bytes = d.cpu_block_stride_in_bytes;
  a.cpu_startoff_inside_chunks = 0;

  const int n_ranks = pool_->size();
  if (d.num_kv_heads > 1) {
    // Head-sharded KV: each rank owns a slice of the heads, so it writes into
    // its own slot inside the block.  The rank-share modes do not apply --
    // there is nothing shared to divide.
    a.cpu_startoff_inside_chunks = rank * d.cpu_tp_stride_in_bytes;
  } else if (!req.is_host_to_device && n_ranks > 1) {
    // Rank-shared KV on the way out: every rank holds the same bytes, so the
    // group gets to choose who writes what.  H2D is unaffected -- each rank
    // needs the whole thing no matter who produced it.
    switch (req.rank_share_mode) {
    case RankShareMode::SHARDED: {
      // Each rank writes its 1/N slice of the chunk, gpu and cpu offsets
      // moving together.
      const int64_t shard = d.gpu_chunk_sizes_in_bytes[rank] / n_ranks;
      a.cpu_startoff_inside_chunks = rank * shard;
      a.gpu_startoff_inside_chunks = rank * shard;
      a.chunk_size_in_bytes = shard;
      break;
    }
    case RankShareMode::ALL_WRITE:
      // Every rank writes a full copy, into its own stretch of the CPU pool.
      a.cpu_startoff_inside_chunks =
          static_cast<int64_t>(rank) * a.num_blocks * d.cpu_block_stride_in_bytes;
      break;
    case RankShareMode::LAYER_PARALLEL: {
      const auto [start, count] = layer_slice(a.num_layers, n_ranks, rank);
      a.start_layer_id = req.layer_id + start;
      a.num_layers = count;
      break;
    }
    case RankShareMode::RANK0_ONLY:
    case RankShareMode::RANK_ROTATE:
      // Non-designated ranks never reach here (rank_participates filtered
      // them out); the designated one writes the whole thing at offset 0.
      break;
    }
  }

  a.transfer_num_cta = req.transfer_num_cta;
  return a;
}

bool RegionBatchGroup::rank_participates(const RegionRequest &req,
                                         int rank) const {
  const RegionDesc &d = regions_[req.region_index].desc;
  const int n_ranks = pool_->size();
  if (d.num_kv_heads > 1 || req.is_host_to_device || n_ranks <= 1) {
    return true;
  }
  switch (req.rank_share_mode) {
  case RankShareMode::RANK0_ONLY:
  case RankShareMode::RANK_ROTATE:
    return rank == req.designated_rank;
  case RankShareMode::LAYER_PARALLEL: {
    const int total =
        req.layer_granularity > 0 ? req.layer_granularity : d.num_layers;
    return layer_slice(total, n_ranks, rank).second > 0;
  }
  default:
    return true;
  }
}

const std::vector<RegionRequest> &
RegionBatchGroup::prepare(const std::vector<RegionRequest> &requests_in,
                          std::vector<RegionRequest> &scratch) {
  const int n_ranks = pool_->size();
  for (const RegionRequest &req : requests_in) {
    if (req.region_index < 0 || req.region_index >= num_regions()) {
      throw std::out_of_range("RegionRequest.region_index " +
                              std::to_string(req.region_index) +
                              " out of range [0, " +
                              std::to_string(num_regions()) + ")");
    }
    const RegionDesc &d = regions_[req.region_index].desc;
    if (req.gpu_block_id_tensor.numel() != req.cpu_block_id_tensor.numel()) {
      throw std::invalid_argument(
          "region '" + d.name + "': gpu/cpu block id counts differ (" +
          std::to_string(req.gpu_block_id_tensor.numel()) + " vs " +
          std::to_string(req.cpu_block_id_tensor.numel()) + ")");
    }
    const int last =
        req.layer_id +
        (req.layer_granularity > 0 ? req.layer_granularity : d.num_layers);
    if (req.layer_id < 0 || last > d.num_layers) {
      throw std::out_of_range(
          "region '" + d.name + "': layer range [" +
          std::to_string(req.layer_id) + ", " + std::to_string(last) +
          ") outside the region's " + std::to_string(d.num_layers) +
          " layers");
    }
    // Sharded D2H divides the chunk by rank count with integer division; a
    // non-divisible chunk silently drops the trailing bytes, leaving a hole in
    // the assembled KV on the host. Fail loudly instead, naming the way out.
    if (d.num_kv_heads == 1 && !req.is_host_to_device && n_ranks > 1 &&
        req.rank_share_mode == RankShareMode::SHARDED &&
        d.gpu_chunk_sizes_in_bytes[0] % n_ranks != 0) {
      throw std::runtime_error(
          "region '" + d.name +
          "': sharded kv_shared_across_ranks D2H requires gpu_chunk_size "
          "divisible by num_gpus, but chunk_size=" +
          std::to_string(d.gpu_chunk_sizes_in_bytes[0]) +
          " and num_gpus=" + std::to_string(n_ranks) +
          ". Use 'all_write' or 'rank0_only' mode, or adjust "
          "head_dim/tokens_per_block so chunk_size is divisible.");
    }
  }

  // Resolve RANK_ROTATE here, on the caller's thread, before the fan-out:
  // reading the counter inside the per-rank lambdas would let two ranks see
  // different designated ranks for the same request.
  bool needs_rotation = false;
  for (const RegionRequest &req : requests_in) {
    const RegionDesc &d = regions_[req.region_index].desc;
    if (d.num_kv_heads == 1 && !req.is_host_to_device && n_ranks > 1 &&
        req.rank_share_mode == RankShareMode::RANK_ROTATE) {
      needs_rotation = true;
      break;
    }
  }
  if (!needs_rotation) {
    return requests_in;
  }
  scratch = requests_in;
  for (RegionRequest &req : scratch) {
    const RegionDesc &d = regions_[req.region_index].desc;
    if (d.num_kv_heads == 1 && !req.is_host_to_device && n_ranks > 1 &&
        req.rank_share_mode == RankShareMode::RANK_ROTATE) {
      req.designated_rank = rotate_counter_;
      rotate_counter_ = (rotate_counter_ + 1) % n_ranks;
    }
  }
  return scratch;
}

void RegionBatchGroup::submit(const std::vector<RegionRequest> &requests_in,
                              bool sync) {
  if (requests_in.empty()) {
    return;
  }
  std::vector<RegionRequest> scratch;
  const std::vector<RegionRequest> *requests = &prepare(requests_in, scratch);

  // One fan-out for the whole batch. Each rank walks the request list in
  // order, launching onto its own stream; nothing joins in between, so
  // request N+1's launch overlaps request N's copy.
  pool_->run_on_all(
      [&](int rank) {
        for (const RegionRequest &req : *requests) {
          if (!rank_participates(req, rank)) {
            continue;
          }
          RegionTransferArgs args = build_args(req, rank);
          launch_region(req.backend, req.use_ce_transfer, args,
                        pool_->stream(rank), ce_config_);
          cudaError_t err = cudaGetLastError();
          if (err != cudaSuccess) {
            throw std::runtime_error(
                "region '" + regions_[req.region_index].desc.name +
                "': " + cudaGetErrorString(err));
          }
        }
      },
      "submit_region_batch");

  if (sync) {
    wait_all_streams();
  }
}

void RegionBatchGroup::set_layer_eventfds(const torch::Tensor &fds_tensor,
                                          int tp_size, int num_layers,
                                          const std::string &notify_mode) {
  notifier_.set_table(LayerEventfdTable(fds_tensor, tp_size, num_layers));
  layer_notify_mode_ = parse_layer_notify_mode(notify_mode);
}

void RegionBatchGroup::submit_layerwise(
    const std::vector<RegionRequest> &requests_in,
    const std::vector<int> &empty_layers, int counter_id) {
  notifier_.reset(layer_notify_mode_, device_ids_, counter_id);

  // A layer with no state in this model still has a consumer waiting on its
  // fd. Post before launching anything: the consumer can proceed past it
  // immediately, and there is nothing to wait for.
  for (int layer : empty_layers) {
    notifier_.post_empty(layer);
  }
  if (requests_in.empty()) {
    return;
  }

  std::vector<RegionRequest> scratch;
  const std::vector<RegionRequest> &requests = prepare(requests_in, scratch);

  // Group by milestone layer, ascending, preserving the caller's order within
  // a layer. Untagged requests (milestone_layer < 0) go first: they are not a
  // milestone of their own, so the first tagged layer's marker covers them.
  std::vector<int> order(requests.size());
  for (size_t i = 0; i < order.size(); ++i) {
    order[i] = static_cast<int>(i);
  }
  std::stable_sort(order.begin(), order.end(), [&](int a, int b) {
    return requests[a].milestone_layer < requests[b].milestone_layer;
  });

  // Declare the layers up front and in launch order: the polling loop walks
  // batches_ in the order begin_layer() was called, and it must match the
  // order the markers will actually fire in.
  for (int idx : order) {
    const int layer = requests[idx].milestone_layer;
    if (layer >= 0) {
      notifier_.begin_layer(layer);
    }
  }

  pool_->run_on_all(
      [&](int rank) {
        int open_layer = -1; // milestone whose marker is not yet recorded
        for (int idx : order) {
          const RegionRequest &req = requests[idx];
          const int layer = req.milestone_layer;
          if (layer != open_layer && open_layer >= 0) {
            // Every request of open_layer is on the stream; the marker
            // recorded here fires once they have all completed.
            notifier_.record(open_layer, rank, pool_->stream(rank));
          }
          if (layer >= 0) {
            open_layer = layer;
          }
          if (!rank_participates(req, rank)) {
            continue;
          }
          RegionTransferArgs args = build_args(req, rank);
          launch_region(req.backend, req.use_ce_transfer, args,
                        pool_->stream(rank), ce_config_);
          cudaError_t err = cudaGetLastError();
          if (err != cudaSuccess) {
            throw std::runtime_error(
                "region '" + regions_[req.region_index].desc.name +
                "': " + cudaGetErrorString(err));
          }
        }
        if (open_layer >= 0) {
          notifier_.record(open_layer, rank, pool_->stream(rank));
        }
      },
      "submit_region_batch_layerwise");

  notifier_.arm();
}

bool RegionBatchGroup::wait_layer_completion(double timeout_s,
                                             std::string *error_out) {
  if (!notifier_.wait(timeout_s, error_out)) {
    return false;
  }
  // Markers only cover work recorded before them. Draining the streams is
  // what makes "op complete" also mean "no DMA is still reading the source
  // blocks the caller is about to recycle".
  try {
    pool_->sync_all_streams();
  } catch (const std::exception &e) {
    if (error_out != nullptr) {
      *error_out = e.what();
    }
    return false;
  }
  return true;
}

void RegionBatchGroup::wait_all_streams() { pool_->sync_all_streams(); }

void RegionBatchGroup::update_region_gpu_ptrs(
    int region_index, const std::vector<int64_t> &gpu_block_ptrs_flat) {
  if (region_index < 0 || region_index >= num_regions()) {
    throw std::out_of_range("region_index out of range");
  }
  RegionState &st = regions_[region_index];
  const size_t expected =
      static_cast<size_t>(pool_->size()) * st.desc.num_tensors_per_gpu;
  if (gpu_block_ptrs_flat.size() != expected) {
    throw std::invalid_argument(
        "GPU pointer count does not match region '" + st.desc.name + "'");
  }
  // An in-flight copy may still be reading the old pointers.
  wait_all_streams();
  for (size_t i = 0; i < expected; ++i) {
    st.gpu_blocks[i] = reinterpret_cast<void *>(gpu_block_ptrs_flat[i]);
  }
  // handlers_ hold `gpu_blocks + rank * n` -- the table address, not the
  // pointers in it -- so they stay valid across this rewrite.
}

} // namespace flexkv
