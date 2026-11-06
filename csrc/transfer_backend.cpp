/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * The three-arm BackendType switch, written once.
 */
#include "transfer_backend.h"
#include "transfer.cuh"
#include <stdexcept>

namespace flexkv {

const char *to_string(TransferBackendKind kind) {
  switch (kind) {
  case TransferBackendKind::AUTO: return "auto";
  case TransferBackendKind::SM_KERNEL: return "sm_kernel";
  case TransferBackendKind::COPY_ENGINE: return "copy_engine";
  case TransferBackendKind::GDS: return "gds";
  case TransferBackendKind::NIXL: return "nixl";
  case TransferBackendKind::MOONCAKE: return "mooncake";
  }
  return "unknown";
}

namespace {

// Both host-device backends funnel into transfer_kv_blocks; they differ only
// in the use_ce_transfer argument.  That is the whole distinction today, and
// keeping them as two named objects rather than one object with a bool is
// what lets a caller say which mechanism it wants instead of restating the
// implementation detail.
void dispatch_by_tensor_kind(const RegionTransferArgs &a, bool use_ce,
                             cudaStream_t stream,
                             const CETransferConfig &ce_config) {
  // sync=false unconditionally: a TransferBackend launches and returns.
  // The historical sync=true lived here because the only caller drained
  // immediately; region batching needs the launches to stack up.
  with_tensor_kind(a.tensor_kind, [&](auto tag) {
    transfer_kv_blocks<decltype(tag)::value>(
        a.num_blocks, a.start_layer_id, a.num_layers, a.gpu_block_ids,
        a.gpu_tensor_handler, a.gpu_startoff_inside_chunks, a.cpu_block_ids,
        a.cpu_ptr, a.cpu_kv_stride_in_bytes, a.cpu_layer_stride_in_bytes,
        a.cpu_block_stride_in_bytes, a.cpu_startoff_inside_chunks,
        a.chunk_size_in_bytes, stream, a.transfer_num_cta,
        a.is_host_to_device, use_ce, a.kv_dim, a.gpu_block_stride_in_bytes,
        /*sync=*/false, ce_config);
  });
}

class SMKernelBackend final : public TransferBackend {
public:
  const char *name() const override { return "sm_kernel"; }
  TransferBackendKind kind() const override {
    return TransferBackendKind::SM_KERNEL;
  }
  void launch(const RegionTransferArgs &args, cudaStream_t stream,
              const CETransferConfig &ce_config) const override {
    dispatch_by_tensor_kind(args, /*use_ce=*/false, stream, ce_config);
  }
};

class CopyEngineBackend final : public TransferBackend {
public:
  const char *name() const override { return "copy_engine"; }
  TransferBackendKind kind() const override {
    return TransferBackendKind::COPY_ENGINE;
  }
  void launch(const RegionTransferArgs &args, cudaStream_t stream,
              const CETransferConfig &ce_config) const override {
    dispatch_by_tensor_kind(args, /*use_ce=*/true, stream, ce_config);
  }
};

const SMKernelBackend kSMKernel;
const CopyEngineBackend kCopyEngine;

} // namespace

const TransferBackend &get_backend(TransferBackendKind kind) {
  switch (kind) {
  case TransferBackendKind::SM_KERNEL:
    return kSMKernel;
  case TransferBackendKind::COPY_ENGINE:
    return kCopyEngine;
  case TransferBackendKind::AUTO:
    throw std::invalid_argument(
        "get_backend(AUTO): call resolve_backend() first");
  default:
    // GDS/NIXL/MOONCAKE are still Transfer Workers (Phase 6). The enum lists
    // them so there is one inventory of mechanisms; this throw is the marker
    // for the ones that have not moved yet.
    throw std::invalid_argument(
        std::string("transfer backend not available through this interface "
                    "(still implemented as a worker): ") +
        to_string(kind));
  }
}

TransferBackendKind resolve_backend(TransferBackendKind requested,
                                    bool use_ce_transfer) {
  if (requested != TransferBackendKind::AUTO) {
    return requested;
  }
  return use_ce_transfer ? TransferBackendKind::COPY_ENGINE
                         : TransferBackendKind::SM_KERNEL;
}

void launch_region(TransferBackendKind kind, bool use_ce_transfer,
                   const RegionTransferArgs &args, cudaStream_t stream,
                   const CETransferConfig &ce_config) {
  get_backend(resolve_backend(kind, use_ce_transfer))
      .launch(args, stream, ce_config);
}

} // namespace flexkv
