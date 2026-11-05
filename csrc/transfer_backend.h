/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * One region's worth of transfer arguments, and the backends that can move it.
 *
 * Before this file, "which mechanism moves the bytes" was a bool
 * (``use_ce_transfer``) threaded through every call site, and "which tensor
 * layout" was a three-arm switch on ``BackendType`` repeated at every one of
 * them -- once in tp_transfer_thread_group.cpp, four times in layerwise.cpp,
 * three times in bindings.cpp.  Adding a mechanism meant widening the bool
 * into an enum at every site; adding a call site meant copying the switch.
 *
 * Here the switch lives once, inside ``launch``, and a mechanism is an object
 * with a name.  That is what lets NIXL/Mooncake stop being *workers* (a
 * process, a queue, an op protocol) and become what they actually are: another
 * way to move one region's bytes.
 *
 * A backend never synchronizes.  It launches onto the stream it is handed and
 * returns; the caller decides when to drain.  This is what makes region
 * batching possible at all -- N regions go onto one stream back to back and
 * are joined once, instead of N fan-out/join round trips.
 */
#pragma once

#include "ce_transfer.h"
#include "gtensor_handler.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <type_traits>

namespace flexkv {

// Turn a runtime BackendType into a compile-time one, once.
//
// The three-arm switch existed because ``transfer_kv_blocks`` and
// ``transfer_kv_blocks_gds`` take the layout as a template parameter, so every
// call site had to spell out all three instantiations.  This says that in one
// place: ``with_tensor_kind(k, [&](auto tag) { f<decltype(tag)::value>(...); })``
// instantiates the same three arms without the call site restating them, and a
// fourth layout is one arm here rather than one arm per site.
template <BackendType K>
using TensorKindTag = std::integral_constant<BackendType, K>;

template <typename F>
auto with_tensor_kind(BackendType kind, F &&fn) {
  switch (kind) {
  case BackendType::VLLM:
    return fn(TensorKindTag<BackendType::VLLM>{});
  case BackendType::TRTLLM:
    return fn(TensorKindTag<BackendType::TRTLLM>{});
  case BackendType::SGLANG:
    return fn(TensorKindTag<BackendType::SGLANG>{});
  }
  throw std::invalid_argument("unknown BackendType (device tensor layout)");
}

// How to move the bytes.  Orthogonal to ``BackendType``, which says how the
// *device tensor* is laid out (vLLM per-layer, TRT-LLM single, SGLang k/v
// split) -- one names the mechanism, the other the memory.
enum class TransferBackendKind : int {
  // Let resolve_backend() pick.  Preserves the historical behaviour where the
  // caller's use_ce_transfer flag decided, so AUTO + the flag is a no-op
  // rename rather than a policy change.
  AUTO = 0,
  // The custom float4/int64 CUDA copy kernel.  Costs SMs, wins on scattered
  // block ids and small chunks.
  SM_KERNEL = 1,
  // cudaMemcpyAsync via the copy engines, with the CE path analysis
  // (CONTIG/SEGMENT/GATHER) in ce_transfer.cu.  Costs no SMs.
  COPY_ENGINE = 2,
  // Reserved for Phase 6: these exist today as whole Transfer *Workers*.
  // Declared here so the enum is the single list of mechanisms even while
  // two of them are still implemented one level up.
  GDS = 3,
  NIXL = 4,
  MOONCAKE = 5,
};

const char *to_string(TransferBackendKind kind);

// Everything one region needs on one rank.  Deliberately a flat POD: it is
// built inside the per-rank worker thread from the rank's slice of a
// RegionSet, so it must be cheap to construct and must not own anything.
struct RegionTransferArgs {
  // ---- what to move ----
  int num_blocks = 0;
  int64_t *gpu_block_ids = nullptr;
  int64_t *cpu_block_ids = nullptr;
  int start_layer_id = 0;
  int num_layers = 0;
  int kv_dim = 1;
  int64_t chunk_size_in_bytes = 0;
  bool is_host_to_device = false;

  // ---- device side ----
  BackendType tensor_kind = BackendType::VLLM;
  GTensorHandler gpu_tensor_handler;
  int64_t gpu_block_stride_in_bytes = 0;
  int64_t gpu_startoff_inside_chunks = 0;

  // ---- host side ----
  void *cpu_ptr = nullptr;
  int64_t cpu_kv_stride_in_bytes = 0;
  int64_t cpu_layer_stride_in_bytes = 0;
  int64_t cpu_block_stride_in_bytes = 0;
  int64_t cpu_startoff_inside_chunks = 0;

  // ---- tuning ----
  int transfer_num_cta = 4;
};

class TransferBackend {
public:
  virtual ~TransferBackend() = default;
  virtual const char *name() const = 0;
  virtual TransferBackendKind kind() const = 0;

  // Launch onto ``stream`` and return.  Never synchronizes: draining is the
  // caller's job, because only the caller knows how many regions are still
  // coming.
  virtual void launch(const RegionTransferArgs &args, cudaStream_t stream,
                      const CETransferConfig &ce_config) const = 0;
};

// Backends are stateless singletons; the reference outlives any caller.
// Throws for the kinds still implemented as workers (GDS/NIXL/MOONCAKE) --
// they are reachable through this enum but not yet through this interface.
const TransferBackend &get_backend(TransferBackendKind kind);

// AUTO -> a concrete kind.  ``use_ce_transfer`` is the historical caller flag;
// passing it here keeps the old decision in one place instead of at every
// call site, so a smarter policy (chunk size, contiguity, SM pressure) has
// exactly one function to grow into.
TransferBackendKind resolve_backend(TransferBackendKind requested,
                                    bool use_ce_transfer);

// Convenience: resolve + launch.  The common case at every call site.
void launch_region(TransferBackendKind kind, bool use_ce_transfer,
                   const RegionTransferArgs &args, cudaStream_t stream,
                   const CETransferConfig &ce_config);

} // namespace flexkv
