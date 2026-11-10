// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>
namespace flexkv::gdr {
struct Span { void *host; void *gpu; size_t bytes; };
struct RemoteRegion { uintptr_t local; size_t bytes; uint64_t remote; uint32_t key; };
struct TransportProbe {
  uint64_t resolve_ns=0, build_ns=0, doorbell_ns=0, post_total_ns=0;
  uint64_t credit_ns=0, drain_ns=0, credit_waits=0, drain_waits=0;
  uint64_t poll_calls=0, empty_polls=0, cqes=0, wqes=0, posts=0;
  uint64_t mr_lookup_ns=0, mr_register_ns=0, mr_drain_ns=0;
  uint64_t mr_calls=0, mr_hits=0, mr_registrations=0;
};
struct H2DCompletionProbe {
  uint64_t calls=0, total_ns=0, payload_drain_ns=0;
  uint64_t readback_calls=0, readback_ns=0, device_sync_calls=0, device_sync_ns=0;
  uint64_t native_flush_calls=0, native_flush_ns=0;
};
struct TransportStats { TransportProbe probe; H2DCompletionProbe h2d_completion; uint64_t wqes=0, posts=0, extended_posts=0, completions=0, post_ns=0, cq_ns=0, read_capacity=0, flush_reads=0; };
// One owner thread per transport; no process-global engines or borrowed MRs.
// Register allocations in their owner process; imported aliases use remote mappings.
class Transport {
 public:
  Transport(int device, int qps, const std::string &nic, bool timing,
            const std::string &required_endpoint = "", int read_depth = 16);
  ~Transport();
  Transport(const Transport&) = delete;
  Transport& operator=(const Transport&) = delete;
  void register_device(void *base, size_t bytes);
  void register_host(void *base, size_t bytes);
  // Completion includes RDMA write visibility before any consuming CUDA launch.
  // The consumer usually runs in the allocation owner process, so visibility
  // must not depend on anything scoped to this process's CUDA contexts.
  // Once posting starts, uncertain completion terminates the worker: no retry.
  void post(const std::vector<Span> &spans, bool h2d);
  void finish();
  void set_probe(bool enabled); // owner-thread only; D2H stage collection, no logging
  void set_completion_probe(bool enabled); // H2D finish stages; no extra fences
  void transfer(const std::vector<Span> &spans, bool h2d);
  uint64_t gpu_address(void *ptr, size_t bytes);
  uint32_t device_key(void *ptr, size_t bytes) const;
  const std::string &endpoint() const;
  int read_depth() const;
  std::vector<uint32_t> prepare_remote();
  uint32_t accept(uint32_t sender);
  void release(uint32_t target);
  void connect_remote(const std::vector<uint32_t> &targets,
                      const std::vector<RemoteRegion> &regions);
  const std::string &nic() const;
  TransportStats stats() const;
  static bool compiled();
 private:
  struct Impl;
  std::unique_ptr<Impl> p_;
};
} // namespace flexkv::gdr
