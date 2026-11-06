/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cstdint>
#include <vector>

namespace flexkv {

class AlignedDirectIOBuffer {
public:
  explicit AlignedDirectIOBuffer(int64_t bytes);
  ~AlignedDirectIOBuffer();

  void *ptr();
  int64_t bytes() const;

private:
  void *ptr_;
  int64_t bytes_;
};

// Describes one compressed chunk inside a block's packed on-disk layout, i.e.
// how to scatter/gather it between the CPU slot and the contiguous staging
// buffer during a single SSD read/write.
struct PackedSpan {
  void *cpu_ptr;             // this chunk's address in the CPU tensor
  uint32_t comp_bytes;       // compressed payload size: the memcpy length, the
                             // value recorded in the size table, and (spans are
                             // packed contiguously) its on-disk footprint
  uint32_t *dst_table_entry; // size-table slot to write comp_bytes into
};

// One (tp-rank, layer, kv) coordinate of a compressed chunk in a block's packed
// layout. The order of these in a list defines the on-disk byte order of the
// spans (BLOCKFIRST nests rank outermost, LAYERFIRST nests layer outermost).
struct PackedCoord {
  int rank;
  int lid;
  int kv;
};

void do_transfer_packed_blocks(
    int fd, int64_t ssd_off, int64_t transfer_bytes, bool is_read,
    void *staging_buffer, const std::vector<PackedSpan> &spans);

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
    const char *layout_name);

} // namespace flexkv
