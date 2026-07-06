/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */
#include "compression/common/packed_ssd.h"
#include "monitoring/metrics_manager.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace flexkv {

static constexpr int64_t PACKED_DIRECT_IO_ALIGN = 512;
static constexpr int64_t PACKED_DIRECT_IO_BUFFER_ALIGN = 4096;

AlignedDirectIOBuffer::AlignedDirectIOBuffer(int64_t bytes)
    : ptr_(nullptr), bytes_(bytes) {
  if (bytes_ <= 0) {
    throw std::runtime_error(
        "transfer_kv_blocks_ssd_packed: invalid buffer size");
  }
  int rc = posix_memalign(&ptr_,
                          static_cast<size_t>(PACKED_DIRECT_IO_BUFFER_ALIGN),
                          static_cast<size_t>(bytes_));
  if (rc != 0 || ptr_ == nullptr) {
    throw std::runtime_error(
        "transfer_kv_blocks_ssd_packed: posix_memalign failed");
  }
}

AlignedDirectIOBuffer::~AlignedDirectIOBuffer() { free(ptr_); }

void *AlignedDirectIOBuffer::ptr() { return ptr_; }

int64_t AlignedDirectIOBuffer::bytes() const { return bytes_; }

static inline uint32_t checked_comp_bytes(
    uint32_t value, int64_t chunk_size, bool is_read,
    int cpu_block_id, int ssd_block_id, const PackedCoord &coord) {
  const std::string where =
      std::string(is_read ? "DISK2H" : "H2DISK") +
      " cpu_block=" + std::to_string(cpu_block_id) +
      " ssd_block=" + std::to_string(ssd_block_id) +
      " rank=" + std::to_string(coord.rank) +
      " layer=" + std::to_string(coord.lid) +
      " kv=" + std::to_string(coord.kv);
  if (value == 0) {
    throw std::runtime_error(
        "transfer_kv_blocks_ssd_packed: size-table entry is 0 (" +
        where + ")");
  }
  if (static_cast<int64_t>(value) > chunk_size) {
    throw std::runtime_error(
        "transfer_kv_blocks_ssd_packed: compressed payload is larger than "
        "the CPU/SSD chunk slot, value=" + std::to_string(value) +
        " chunk_size=" + std::to_string(chunk_size) + " (" + where + ")");
  }
  return value;
}

void do_transfer_packed_blocks(
    int fd,
    int64_t ssd_off,
    int64_t transfer_bytes,
    bool is_read,
    void *staging_buffer,
    const std::vector<PackedSpan> &spans) {
  char *staging = reinterpret_cast<char *>(staging_buffer);

  // H2DISK gather (host memory only, no I/O yet): the compressed chunks are
  // scattered across the CPU tensor (one per chunk-size slot). memcpy each into
  // the contiguous host staging buffer, then zero-pad the tail up to the
  // 512-aligned transfer length, so the whole block can go out in one write.
  if (!is_read) {
    int64_t cursor = 0;
    for (const auto &span : spans) {
      memcpy(staging + cursor, span.cpu_ptr,
             static_cast<size_t>(span.comp_bytes));
      *span.dst_table_entry = span.comp_bytes;
      cursor += span.comp_bytes;
    }
    if (transfer_bytes > cursor) {
      memset(staging + cursor, 0, static_cast<size_t>(transfer_bytes - cursor));
    }
  }

  // The one block I/O: a single full pread/pwrite of the packed range, looping
  // over partial transfers and retrying on EINTR.
  int64_t done = 0;
  while (done < transfer_bytes) {
    ssize_t rc;
    do {
      rc = is_read
               ? pread(fd, staging + done, transfer_bytes - done, ssd_off + done)
               : pwrite(fd, staging + done, transfer_bytes - done, ssd_off + done);
    } while (rc < 0 && errno == EINTR);
    if (rc <= 0) {
      throw std::runtime_error(
          is_read ? "transfer_kv_blocks_ssd_packed: read failed"
                  : "transfer_kv_blocks_ssd_packed: write failed");
    }
    done += rc;
  }

  // DISK2H scatter (host memory only): the pread above filled the host staging
  // buffer with this block's tightly-packed compressed chunks. Walk the spans in
  // the same order they were packed and, for each chunk: memcpy its comp_bytes
  // from the running staging cursor back to its slot in the CPU tensor
  // (span.cpu_ptr), record comp_bytes into the CPU-side size table
  // (span.dst_table_entry), and advance the cursor tightly by comp_bytes. Any
  // 512-aligned tail left in staging is O_DIRECT padding and is ignored.
  if (is_read) {
    int64_t cursor = 0;
    for (const auto &span : spans) {
      memcpy(span.cpu_ptr, staging + cursor,
             static_cast<size_t>(span.comp_bytes));
      *span.dst_table_entry = span.comp_bytes;
      cursor += span.comp_bytes;
    }
  }

  FLEXKV_CPU_SSD_TRANSFER(is_read, transfer_bytes);
}

// One worker thread of the packed nvcomp SSD path, shared by BLOCKFIRST and
// LAYERFIRST. The two layouts differ only in `span_order` (the on-disk byte
// order of the spans) and `disk_block_stride_in_bytes` (the on-disk slot size
// per block); everything else is identical. It owns a contiguous slice
// [start_block, end_block) of a single device's block list and, for each block:
//   1. resolves which file the block lives in and that file's (direct, buffered)
//      fds  (round-robin: file_index = ssd_block_id % num_files_per_device);
//   2. converts the device-local ssd block id into an in-file block id;
//   3. builds the per-block packed span layout (the rank/layer/kv compressed
//      payloads laid out contiguously in span_order) and issues the single SSD
//      read/write.
// The staging buffer and spans vector are allocated once and reused across all
// blocks this thread processes.
int64_t transfer_packed_thread_impl(
    const std::vector<int> &direct_fd_list,
    const std::vector<int> &buffered_fd_list,
    const std::vector<int> &cpu_block_ids,
    const std::vector<int> &ssd_block_ids_in_device,
    const std::vector<int> &ssd_block_ids_orig,
    const std::vector<PackedCoord> &span_order,
    int start_block, int end_block, int64_t cpu_tensor_ptr,
    int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes,
    int64_t block_stride_in_bytes,
    int64_t disk_block_stride_in_bytes,
    int64_t cpu_tp_rank_stride_in_bytes,
    int64_t chunk_size_in_bytes,
    int num_files_per_device, bool is_read,
    uint32_t *cpu_size_table_base,
    int64_t cpu_size_table_rank_stride,
    int64_t cpu_size_table_block_stride,
    int64_t cpu_size_table_layer_stride,
    uint32_t *ssd_size_table_base,
    int64_t ssd_size_table_rank_stride,
    int64_t ssd_size_table_block_stride,
    int64_t ssd_size_table_layer_stride,
    const char *layout_name) {
  if (end_block <= start_block) return 0;

  // Sum of this thread's compressed payload across its block range. Mirrors
  // tp_group_transfer_ans: the compressed (packed) bytes, excluding O_DIRECT
  // tail padding (transfer_bytes - packed_bytes).
  int64_t thread_packed_bytes = 0;
  AlignedDirectIOBuffer staging(disk_block_stride_in_bytes);
  std::vector<PackedSpan> spans;
  spans.reserve(span_order.size());
  for (int bid = start_block; bid < end_block; bid++) {
    int cpu_block_id = cpu_block_ids[bid];
    int ssd_block_id = ssd_block_ids_in_device[bid];
    int ssd_block_id_orig = ssd_block_ids_orig[bid]; // pre-remap id for size-table indexing
    int file_index = ssd_block_id % num_files_per_device;
    int direct_fd = direct_fd_list[file_index];
    int buffered_fd = buffered_fd_list[file_index];
    ssd_block_id /= num_files_per_device; // block id in single file

    // Build this block's packed layout in memory: one span per rank/layer/kv
    // compressed payload, appended in on-disk (span_order) order.
    spans.clear();
    int64_t packed_bytes = 0;
    for (const PackedCoord &c : span_order) {
      uint32_t *cpu_entry = cpu_size_table_base
          + static_cast<int64_t>(c.rank) * cpu_size_table_rank_stride
          + static_cast<int64_t>(cpu_block_id) * cpu_size_table_block_stride
          + static_cast<int64_t>(c.lid) * cpu_size_table_layer_stride
          + static_cast<int64_t>(c.kv);
      uint32_t *ssd_entry = ssd_size_table_base
          + static_cast<int64_t>(c.rank) * ssd_size_table_rank_stride
          + static_cast<int64_t>(ssd_block_id_orig) * ssd_size_table_block_stride
          + static_cast<int64_t>(c.lid) * ssd_size_table_layer_stride
          + static_cast<int64_t>(c.kv);

      uint32_t comp_bytes = checked_comp_bytes(
          is_read ? *ssd_entry : *cpu_entry, chunk_size_in_bytes, is_read,
          cpu_block_id, ssd_block_id_orig, c);
      if (is_read) {
        *cpu_entry = comp_bytes;
      } else {
        *ssd_entry = comp_bytes;
      }

      void *cpu_ptr = reinterpret_cast<char *>(cpu_tensor_ptr)
          + static_cast<int64_t>(cpu_block_id) * block_stride_in_bytes
          + static_cast<int64_t>(c.rank) * cpu_tp_rank_stride_in_bytes
          + static_cast<int64_t>(c.lid) * cpu_layer_stride_in_bytes
          + static_cast<int64_t>(c.kv) * cpu_kv_stride_in_bytes;
      spans.push_back({
          cpu_ptr,
          comp_bytes,
          is_read ? cpu_entry : ssd_entry,
      });
      packed_bytes += comp_bytes;
    }

    // Spans are packed tightly (no per-chunk padding); only the single block
    // I/O needs alignment. O_DIRECT requires the offset and transfer length to
    // be 512-aligned, so round the packed total up (the tail is zero-padded)
    // when the block qualifies; buffered I/O has no such constraint and writes
    // the exact total.
    int64_t ssd_off =
        static_cast<int64_t>(ssd_block_id) * disk_block_stride_in_bytes;
    int64_t direct_io_bytes =
        (packed_bytes + PACKED_DIRECT_IO_ALIGN - 1) & ~(PACKED_DIRECT_IO_ALIGN - 1);
    bool use_direct_io =
        (ssd_off % PACKED_DIRECT_IO_ALIGN == 0) &&
        direct_io_bytes <= disk_block_stride_in_bytes &&
        direct_io_bytes <= staging.bytes();
    int64_t transfer_bytes = use_direct_io ? direct_io_bytes : packed_bytes;
    if (transfer_bytes > disk_block_stride_in_bytes ||
        transfer_bytes > staging.bytes()) {
      throw std::runtime_error(
          std::string("transfer_kv_blocks_ssd_packed ") + layout_name +
          ": packed block exceeds raw block slot");
    }

    // Execute the SSD transfer once for this block. H2DISK gathers all spans
    // into the staging buffer then pwrite()s the packed range; DISK2H pread()s
    // the packed range then scatters each span back into its CPU slot.
    do_transfer_packed_blocks(
        use_direct_io ? direct_fd : buffered_fd, ssd_off, transfer_bytes,
        is_read, staging.ptr(), spans);
    thread_packed_bytes += packed_bytes;
  }
  return thread_packed_bytes;
}

} // namespace flexkv
