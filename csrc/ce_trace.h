#pragma once

#include <cstdint>

namespace flexkv {
struct GTensorHandler;
struct CETransferConfig;
struct CEAnalysis;
enum class CEPath : int;

// Disabled by default. All logging failures are diagnostic only.
bool ce_trace_enabled() noexcept;
void ce_trace_set_enabled(bool enabled) noexcept;
// Drain this logger's private queue, without shutting down other loggers.
uint64_t ce_trace_shutdown() noexcept;
void ce_trace_log(int tensor_kind, const GTensorHandler &gpu, int num_blocks,
                  int start_layer, int num_layers, int kv_dim,
                  int64_t chunk_bytes, int64_t gpu_block_stride,
                  int64_t cpu_kv_stride, int64_t cpu_layer_stride,
                  int64_t cpu_block_stride, int64_t gpu_offset,
                  int64_t cpu_offset, int transfer_num_cta, bool h2d,
                  bool use_ce, CEPath path, const CETransferConfig &config,
                  const CEAnalysis *analysis, const int64_t *gpu_ids,
                  const int64_t *cpu_ids, int device) noexcept;
} // namespace flexkv
