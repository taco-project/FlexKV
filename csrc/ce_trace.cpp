#include "ce_trace.h"
#include "ce_transfer.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <unistd.h>

#include <spdlog/async_logger.h>
#include <spdlog/details/thread_pool.h>
#include <spdlog/sinks/basic_file_sink.h>

namespace flexkv {
namespace {
bool configured() {
  const char *value = std::getenv("FLEXKV_CE_TRACE");
  return value != nullptr && std::string(value) == "1";
}
std::atomic<bool> &enabled() {
  static std::atomic<bool> value{configured()};
  return value;
}
struct TraceState {
  std::mutex mutex;
  std::shared_ptr<spdlog::details::thread_pool> pool;
  std::shared_ptr<spdlog::logger> logger;
  uint64_t sequence = 0;
  uint64_t dropped = 0;
  int max_blocks = 256;
  bool stopped = false;

  void open() {
    if (logger || stopped)
      return;
    const char *base = std::getenv("FLEXKV_CE_TRACE_FILE");
    const std::string path = std::string(base ? base : "flexkv_ce_trace") +
                             "." + std::to_string(getpid()) + ".jsonl";
    if (const char *limit = std::getenv("FLEXKV_CE_TRACE_MAX_BLOCKS")) {
      char *end = nullptr;
      const long parsed = std::strtol(limit, &end, 10);
      if (end != limit && *end == '\0' && parsed >= 0 && parsed <= 1000000)
        max_blocks = static_cast<int>(parsed);
    }
    auto sink =
        std::make_shared<spdlog::sinks::basic_file_sink_mt>(path, false);
    // Private pool: never initialize or shut down spdlog's global pool.
    pool = std::make_shared<spdlog::details::thread_pool>(4096, 1);
    logger = std::make_shared<spdlog::async_logger>(
        "flexkv_ce_trace", sink, pool,
        spdlog::async_overflow_policy::overrun_oldest);
    logger->set_error_handler([](const std::string &message) {
      enabled().store(false, std::memory_order_relaxed);
      std::fprintf(stderr, "[FLEXKV] CE trace disabled: %s\n", message.c_str());
    });
    logger->set_pattern("%v");
    logger->flush_on(spdlog::level::info);
  }
};
TraceState &state() {
  static TraceState value;
  return value;
}
void ids(std::ostream &out, const int64_t *values, int count) {
  out << '[';
  for (int i = 0; i < count; ++i) {
    if (i)
      out << ',';
    out << values[i];
  }
  out << ']';
}
} // namespace

bool ce_trace_enabled() noexcept {
  return enabled().load(std::memory_order_relaxed);
}
void ce_trace_set_enabled(bool on) noexcept {
  enabled().store(on, std::memory_order_relaxed);
}

uint64_t ce_trace_shutdown() noexcept {
  ce_trace_set_enabled(false);
  try {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    s.stopped = true;
    if (s.pool)
      s.dropped += s.pool->overrun_counter();
    s.logger.reset();
    s.pool.reset(); // joins the private writer after queued records drain
    if (s.dropped)
      std::fprintf(stderr, "[FLEXKV] CE trace dropped %llu records\n",
                   static_cast<unsigned long long>(s.dropped));
    return s.dropped;
  } catch (...) {
    return 0;
  }
}

void ce_trace_log(int tensor_kind, const GTensorHandler &gpu, int num_blocks,
                  int start_layer, int num_layers, int kv_dim,
                  int64_t chunk_bytes, int64_t gpu_block_stride,
                  int64_t cpu_kv_stride, int64_t cpu_layer_stride,
                  int64_t cpu_block_stride, int64_t gpu_offset,
                  int64_t cpu_offset, int transfer_num_cta, bool h2d,
                  bool use_ce, CEPath path, const CETransferConfig &config,
                  const CEAnalysis *analysis, const int64_t *gpu_ids,
                  const int64_t *cpu_ids, int device) noexcept {
  if (!ce_trace_enabled() || num_blocks <= 0 || num_layers <= 0)
    return;
  try {
    auto &s = state();
    std::lock_guard<std::mutex> lock(s.mutex);
    if (s.stopped)
      return;
    s.open();
    const int count =
        s.max_blocks == 0 ? num_blocks : std::min(num_blocks, s.max_blocks);
    const auto now = std::chrono::system_clock::now().time_since_epoch();
    std::ostringstream out;
    out << std::boolalpha
        << "{\"schema_version\":1,\"event\":\"transfer_launch\",\"trace_id\":"
        << s.sequence++ << ",\"pid\":" << getpid() << ",\"tid\":"
        << std::hash<std::thread::id>{}(std::this_thread::get_id())
        << ",\"ts_ns\":"
        << std::chrono::duration_cast<std::chrono::nanoseconds>(now).count()
        << ",\"device\":" << device << ",\"tensor_kind\":" << tensor_kind
        << ",\"transfer_backend\":\"" << (use_ce ? "copy_engine" : "sm_kernel")
        << '"' << ",\"direction\":\"" << (h2d ? "H2D" : "D2H") << '"'
        << ",\"num_blocks\":" << num_blocks
        << ",\"start_layer_id\":" << start_layer
        << ",\"num_layers\":" << num_layers
        << ",\"total_num_layers\":" << gpu.num_layers
        << ",\"kv_dim\":" << kv_dim
        << ",\"chunk_size_in_bytes\":" << chunk_bytes
        << ",\"transfer_num_cta\":" << transfer_num_cta
        << ",\"strides\":{\"gpu_kv\":" << gpu.gpu_kv_stride * 8
        << ",\"gpu_block\":" << gpu_block_stride
        << ",\"gpu_layer\":" << gpu.gpu_layer_stride * 8
        << ",\"cpu_kv\":" << cpu_kv_stride
        << ",\"cpu_block\":" << cpu_block_stride
        << ",\"cpu_layer\":" << cpu_layer_stride << '}'
        << ",\"offsets\":{\"gpu\":" << gpu_offset << ",\"cpu\":" << cpu_offset
        << '}'
        << ",\"ce_config\":{\"segment_threshold\":" << config.segment_threshold
        << ",\"path_opt_enabled\":" << config.path_opt_enabled
        << ",\"force_path\":" << config.force_path
        << ",\"enable_memcpy2d\":" << config.enable_memcpy2d
        << ",\"is_blockfirst\":" << config.is_blockfirst
        << ",\"num_kv_heads\":" << config.num_kv_heads
        << ",\"gather_threads\":" << config.gather_threads
        << ",\"gather_nt\":" << config.gather_nt << '}';
    out << ",\"ce_path_id\":";
    if (use_ce)
      out << static_cast<int>(path);
    else
      out << "null";
    if (analysis) {
      out << ",\"analysis\":{\"gpu_log_contig\":" << analysis->gpu_log_contig
          << ",\"cpu_log_contig\":" << analysis->cpu_log_contig
          << ",\"gpu_phys_contig\":" << analysis->gpu_phys_contig
          << ",\"cpu_phys_contig\":" << analysis->cpu_phys_contig
          << ",\"num_segments\":" << analysis->num_segments << '}';
    }
    out << ",\"gpu_block_ids\":";
    ids(out, gpu_ids, count);
    out << ",\"cpu_block_ids\":";
    ids(out, cpu_ids, count);
    out << ",\"block_ids_truncated\":" << (count != num_blocks)
        << ",\"dropped_before\":" << s.pool->overrun_counter() << '}';
    s.logger->info(out.str());
  } catch (const std::exception &e) {
    ce_trace_set_enabled(false);
    std::fprintf(stderr, "[FLEXKV] CE trace disabled: %s\n", e.what());
  } catch (...) {
    ce_trace_set_enabled(false);
    std::fprintf(stderr, "[FLEXKV] CE trace disabled: logging failure\n");
  }
}
} // namespace flexkv
