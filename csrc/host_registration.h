#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <map>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <vector>

namespace flexkv {

struct HostRegistration {
  uintptr_t base;
  size_t bytes;
  // Zero means the entire range was registered in one CUDA call.
  size_t chunk_bytes;
};

inline std::map<uintptr_t, HostRegistration> &host_registrations() {
  static std::map<uintptr_t, HostRegistration> ranges;
  return ranges;
}

inline std::mutex &host_registrations_mutex() {
  static std::mutex mutex;
  return mutex;
}

inline void register_host_region(uintptr_t base, size_t bytes,
                                 size_t chunk_bytes) {
  if (!base || !bytes || bytes > std::numeric_limits<uintptr_t>::max() - base)
    throw std::invalid_argument("invalid host registration range");
  std::lock_guard<std::mutex> lock(host_registrations_mutex());
  auto &ranges = host_registrations();
  auto next = ranges.lower_bound(base);
  if ((next != ranges.end() && next->first < base + bytes) ||
      (next != ranges.begin() &&
       std::prev(next)->second.base + std::prev(next)->second.bytes > base))
    throw std::invalid_argument("overlapping host registration ranges");
  ranges.emplace(base, HostRegistration{base, bytes, chunk_bytes});
}

inline void unregister_host_region(uintptr_t base) {
  std::lock_guard<std::mutex> lock(host_registrations_mutex());
  host_registrations().erase(base);
}

inline std::optional<HostRegistration> find_host_registration(uintptr_t ptr) {
  std::lock_guard<std::mutex> lock(host_registrations_mutex());
  const auto &ranges = host_registrations();
  auto next = ranges.upper_bound(ptr);
  if (next == ranges.begin())
    return std::nullopt;
  const auto &range = std::prev(next)->second;
  return ptr - range.base < range.bytes ? std::optional<HostRegistration>(range)
                                        : std::nullopt;
}

struct HostCopySlice {
  size_t host_offset;
  size_t device_offset;
  size_t width;
  size_t height;
  size_t host_pitch;
  size_t device_pitch;
};

// A CUDA host DMA must stay within one cudaHostRegister allocation. Pitched
// rows can skip chunks, and even a single row can cross a chunk boundary.
inline std::vector<HostCopySlice>
plan_registered_host_copy(const HostRegistration &range, uintptr_t host,
                          size_t width, size_t height, size_t host_pitch,
                          size_t device_pitch) {
  if (!width || !height)
    return {};
  if (host_pitch < width || device_pitch < width || host < range.base)
    throw std::invalid_argument("invalid host copy pitch or address");
  const size_t offset = host - range.base;
  const size_t max = std::numeric_limits<size_t>::max();
  if (offset >= range.bytes || width > range.bytes - offset ||
      height - 1 > (range.bytes - offset - width) / host_pitch ||
      height - 1 > (max - width) / device_pitch)
    throw std::invalid_argument("host copy exceeds its registered range");
  if (!range.chunk_bytes)
    return {{0, 0, width, height, host_pitch, device_pitch}};

  std::vector<HostCopySlice> slices;
  size_t row = 0;
  while (row < height) {
    const size_t row_offset = offset + row * host_pitch;
    const size_t chunk_start =
        (row_offset / range.chunk_bytes) * range.chunk_bytes;
    const size_t available =
        std::min(range.chunk_bytes, range.bytes - chunk_start) -
        (row_offset - chunk_start);
    if (width <= available) {
      const size_t rows =
          std::min(height - row, (available - width) / host_pitch + 1);
      slices.push_back({row * host_pitch, row * device_pitch, width, rows,
                        host_pitch, device_pitch});
      row += rows;
      continue;
    }
    size_t column = 0;
    while (column < width) {
      const size_t current = row_offset + column;
      const size_t chunk_remaining =
          range.chunk_bytes - current % range.chunk_bytes;
      const size_t bytes = std::min(width - column, chunk_remaining);
      slices.push_back({row * host_pitch + column, row * device_pitch + column,
                        bytes, 1, bytes, bytes});
      column += bytes;
    }
    ++row;
  }
  return slices;
}

} // namespace flexkv
