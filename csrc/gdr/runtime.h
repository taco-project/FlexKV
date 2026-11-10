// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "gdr/config.h"
#include "gdr/transport.h"
#include "transfer_backend.h"
#include <memory>
#include <map>
#include <functional>
namespace flexkv::gdr {
// Owned by RegionBatchGroup. Drains before its descriptors/pinned pools die.
class Context {
 public:
  Context(const Config &config, const std::vector<int> &devices);
  ~Context();
  TransferBackendKind select(int rank, const RegionTransferArgs &args,
                             TransferBackendKind original);
  // Pure pre-fanout decision: all ranks must have live registrations.
  bool prefer_layer_parallel(const RegionTransferArgs &full_chunk_args) const;
  // A homogeneous forced-GDR batch can stay on its dedicated thread.
  bool can_run_batch(int rank, bool h2d) const;
  void run_batch(int rank, cudaStream_t stream, const std::function<void()> &body);
  void launch(int rank, const RegionTransferArgs &args, cudaStream_t stream);
  // Explicit owner registration for same-process callers/tests only.
  void register_local(int rank, void *ptr, size_t bytes);
  // Register a complete, long-lived host KV pool before payload submission.
  void register_host(int rank, void *ptr, size_t bytes);
  std::vector<uint32_t> prepare_remote(int rank, const std::string &nic,
                                       const std::string &endpoint);
  std::string endpoint(int rank) const;
  int read_depth(int rank) const;
  void connect_remote(int rank, const std::vector<uint32_t> &qpns,
                      const std::vector<RemoteRegion> &regions);
  void fail_init(int rank, const std::string &reason);
  // Join host submissions, including RDMA completion and write visibility.
  void fence(int rank, cudaStream_t stream);
  void drain();
  std::map<std::string, uint64_t> statistics() const;
 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};
} // namespace flexkv::gdr
