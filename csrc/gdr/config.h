// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace flexkv::gdr {
enum class Path { Original, CE, Kernel, GDR, Auto };
enum class Selector { AvgBytes, Segments };
struct DirectionConfig {
  Path path = Path::Auto;
  Selector selector = Selector::AvgBytes;
  bool fallback_on_init_failure = true;
  uint64_t gdr_when_ce_avg_bytes_below = 512 * 1024;
  uint64_t gdr_max_block_fragment_bytes = 128 * 1024;
  int segment_threshold = 8;
};
struct Config {
  bool enabled = false;
  int qp_count = 4;
  int read_depth = 16; // 0: largest supported power of two, capped at 128
  bool use_dedicated_gdr_thread = true;
  bool d2h_auto_layer_parallel = true;
  DirectionConfig h2d, d2h;
  bool summary = false, timing = false, probe = false;
  std::vector<std::string> nic_names; // empty: nearest PCIe NIC per device
};
struct Shape {
  uint64_t blocks = 0, segments = 0, fragment_bytes = 0;
};
// Separate policies intentionally: D2H can evolve independently of H2D.
inline bool choose_h2d_gdr(const Shape &s, const DirectionConfig &c) {
  if (c.path == Path::GDR) return true;
  if (c.path != Path::Auto || !s.blocks || !s.segments ||
      !s.fragment_bytes || s.fragment_bytes > c.gdr_max_block_fragment_bytes) return false;
  if (c.selector == Selector::Segments) return s.segments >= uint64_t(c.segment_threshold);
  return static_cast<long double>(s.fragment_bytes) * s.blocks / s.segments <
         c.gdr_when_ce_avg_bytes_below;
}
inline bool choose_d2h_gdr(const Shape &s, const DirectionConfig &c) {
  if (c.path == Path::GDR) return true;
  if (c.path != Path::Auto || !s.blocks || !s.segments ||
      !s.fragment_bytes || s.fragment_bytes > c.gdr_max_block_fragment_bytes) return false;
  if (c.selector == Selector::Segments) return s.segments >= uint64_t(c.segment_threshold);
  return static_cast<long double>(s.fragment_bytes) * s.blocks / s.segments <
         c.gdr_when_ce_avg_bytes_below;
}
inline void validate(const Config &c) {
  if (c.qp_count < 1 || c.qp_count > 16) throw std::invalid_argument("gdr.qp_count must be 1..16");
  if(c.read_depth < 0 || c.read_depth > 128 || (c.read_depth && (c.read_depth & (c.read_depth-1))))
    throw std::invalid_argument("GDR read_depth must be 0 or a power of two <=128");
  for (const auto *d : {&c.h2d, &c.d2h})
    if (!d->gdr_when_ce_avg_bytes_below || !d->gdr_max_block_fragment_bytes || d->segment_threshold < 1)
      throw std::invalid_argument("GDR selection thresholds must be positive");
}
} // namespace flexkv::gdr
