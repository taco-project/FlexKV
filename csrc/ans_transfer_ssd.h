#pragma once

#ifdef FLEXKV_ENABLE_NVCOMP

#include <cstdint>
#include <torch/extension.h>
#include "transfer_ssd.h"

namespace flexkv {

static constexpr int64_t ANS_COMP_HEADER_SIZE = 16;
static constexpr int64_t ANS_DIRECT_IO_ALIGN = 512;

// CPU<->SSD transfer for nvcomp-compressed blocks.
//
// compressed_io=false (default):
//   H2DISK writes full chunk_size, DISK2H reads full chunk_size.
//   Behavior matches transfer_kv_blocks_ssd (except no block-first path).
//   Compatible with SSD/Remote/P2P that expect uncompressed-size blocks.
//
// compressed_io=true:
//   H2DISK writes only header + compressed data (saves ~30% write bandwidth).
//   DISK2H two-phase reads: 512B header first, then remaining compressed data.
void ans_transfer_kv_blocks_ssd(
    SSDIOCTX &ioctx, const torch::Tensor &cpu_layer_id_list,
    int64_t cpu_tensor_ptr, const torch::Tensor &ssd_block_ids,
    const torch::Tensor &cpu_block_ids, int64_t cpu_layer_stride_in_bytes,
    int64_t cpu_kv_stride_in_bytes, int64_t ssd_layer_stride_in_bytes,
    int64_t ssd_kv_stride_in_bytes, int64_t chunk_size_in_bytes,
    int64_t block_stride_in_bytes, bool is_read, int num_blocks_per_file,
    int round_robin = 1, int num_threads_per_device = 16, bool is_mla = false,
    bool compressed_io = false);

} // namespace flexkv

#endif // FLEXKV_ENABLE_NVCOMP
