/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * CE adaptive transfer: host-side ce_analysis + multi-path execution.
 */
#pragma once

#include "gtensor_handler.cuh"
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

namespace flexkv {

// ---- CE transfer config ----

struct CETransferConfig {
  int64_t segment_threshold = 8;
  // PER_BLOCK (false) vs adaptive paths (true)
  bool path_opt_enabled = true;
  // -1 auto; 0-4 force path (tests only)
  int force_path = -1;
  // cudaMemcpy2DAsync fast path: NVIDIA on, P800 off
  bool enable_memcpy2d = false;
  // CPU layout: BLOCKFIRST vs LAYERFIRST
  bool is_blockfirst = false;
  // model uses MLA (kv_dim=1)
  bool is_mla = false;
};
enum class CEPath : int {
  PER_BLOCK = -1,       // baseline (path_opt_enabled == false)
  CONTIG_DIRECT = 0,    // contiguous source -> direct memcpy
  SEGMENT_DIRECT = 1,   // segmented source -> direct per-segment memcpy
  SEGMENT_SCATTER = 2,  // segmented source -> staging + CPU scatter
  GATHER_SCATTER = 3,   // GPU gather -> staging + CPU scatter
  GATHER_DIRECT = 4,    // GPU gather + D2D transform -> direct memcpy (BF, non-sharded only)
};

// ---- Analysis structs ----

struct CESegment {
  int start_k;    // start block idx
  int nr_blocks;  // block count
};

struct CEAnalysis {
  bool gpu_log_contig;   // gpu_block_ids[k+1] == gpu_block_ids[k]+1
  bool cpu_log_contig;   // cpu_block_ids[k+1] == cpu_block_ids[k]+1
  bool cpu_phys_contig;  // cpu_block_stride == chunk_size (LAYERFIRST + non-sharded)
  bool gpu_phys_contig;  // gpu_block_stride == chunk_size (non-sharded D2H)
  int num_segments;
  std::vector<CESegment> segments;
};

// ---- Analysis & path selection ----

CEAnalysis analyze_ce_transfer(
    const int64_t *gpu_block_ids, const int64_t *cpu_block_ids,
    int num_blocks, int64_t cpu_block_stride_in_bytes,
    int64_t chunk_size_in_bytes, int64_t gpu_block_stride_in_bytes);

CEPath choose_path(const CEAnalysis &ce_analysis, const CETransferConfig &ce_config,
                   int64_t chunk_size_in_bytes = 0,
                   bool is_host_to_device = false,
                   bool is_full_block = false);

void *get_cached_host_buffer(size_t size);
void *get_cached_device_buffer(size_t size, int slot = 0);

// ---- PER_BLOCK: one memcpy/block, slowest, always-correct ----
template <BackendType Type>
void ce_transfer_per_block(
    int num_blocks, int start_layer_id, int num_layers, int kv_dim,
    int64_t *gpu_block_ids, GTensorHandler gpu_tensor_handler,
    int64_t gpu_startoff_inside_chunks_int64,
    int64_t *cpu_block_ids, int64_t *cpu_ptr_int64,
    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
    int64_t cpu_block_stride_int64,
    int64_t cpu_startoff_inside_chunks_int64, int64_t chunk_size_in_bytes,
    cudaStream_t stream, bool is_host_to_device);

// ---- CONTIG_DIRECT: one big memcpy (both sides contig) ----
template <BackendType Type>
void ce_transfer_contig_direct(
    int num_blocks, int start_layer_id, int num_layers, int kv_dim,
    int64_t *gpu_block_ids, GTensorHandler gpu_tensor_handler,
    int64_t gpu_startoff_inside_chunks_int64,
    int64_t *cpu_block_ids, int64_t *cpu_ptr_int64,
    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
    int64_t cpu_block_stride_int64,
    int64_t cpu_startoff_inside_chunks_int64, int64_t chunk_size_in_bytes,
    cudaStream_t stream, bool is_host_to_device);

// ---- SEGMENT_DIRECT: per-run memcpy, no staging ----
template <BackendType Type>
void ce_transfer_segment_direct(
    int num_blocks, int start_layer_id, int num_layers, int kv_dim,
    int64_t *gpu_block_ids, GTensorHandler gpu_tensor_handler,
    int64_t gpu_startoff_inside_chunks_int64,
    int64_t *cpu_block_ids, int64_t *cpu_ptr_int64,
    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
    int64_t cpu_block_stride_int64,
    int64_t cpu_startoff_inside_chunks_int64, int64_t chunk_size_in_bytes,
    cudaStream_t stream, bool is_host_to_device,
    const CEAnalysis &ce_analysis, const CETransferConfig &ce_config);

// ---- SEGMENT_SCATTER: staging + merged-seg memcpy + CPU scatter (gpu contiguous) ----
template <BackendType Type>
void ce_transfer_segment_scatter(
    int num_blocks, int start_layer_id, int num_layers, int kv_dim,
    int64_t *gpu_block_ids, GTensorHandler gpu_tensor_handler,
    int64_t gpu_startoff_inside_chunks_int64,
    int64_t *cpu_block_ids, int64_t *cpu_ptr_int64,
    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
    int64_t cpu_block_stride_int64,
    int64_t cpu_startoff_inside_chunks_int64, int64_t chunk_size_in_bytes,
    cudaStream_t stream, bool is_host_to_device,
    const CEAnalysis &ce_analysis, const CETransferConfig &ce_config);

// ---- GATHER_SCATTER: GPU gather/scatter through staging (sharded D2H, many segs) ----
template <BackendType Type>
void ce_transfer_gather_scatter(
    int num_blocks, int start_layer_id, int num_layers, int kv_dim,
    int64_t *gpu_block_ids, GTensorHandler gpu_tensor_handler,
    int64_t gpu_startoff_inside_chunks_int64,
    int64_t *cpu_block_ids, int64_t *cpu_ptr_int64,
    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
    int64_t cpu_block_stride_int64,
    int64_t cpu_startoff_inside_chunks_int64, int64_t chunk_size_in_bytes,
    cudaStream_t stream, bool is_host_to_device,
    const CEAnalysis &ce_analysis, const CETransferConfig &ce_config);

// ---- GATHER_DIRECT: BF + !cpu_phys_contig + gpu contiguous; D2D transpose + per-seg copy ----
//     sharded D2H misplaces shards -> GATHER_SCATTER instead.
template <BackendType Type>
void ce_transfer_gather_direct(
    int num_blocks, int start_layer_id, int num_layers, int kv_dim,
    int64_t *gpu_block_ids, GTensorHandler gpu_tensor_handler,
    int64_t gpu_startoff_inside_chunks_int64,
    int64_t *cpu_block_ids, int64_t *cpu_ptr_int64,
    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
    int64_t cpu_block_stride_int64,
    int64_t cpu_startoff_inside_chunks_int64, int64_t chunk_size_in_bytes,
    cudaStream_t stream, bool is_host_to_device,
    const CEAnalysis &ce_analysis, const CETransferConfig &ce_config);

void scatter_to_cpu(const void *staging_buf, int64_t *cpu_ptr_int64,
                    int64_t *cpu_block_ids, int num_blocks,
                    int64_t cpu_block_stride_int64,
                    int64_t cpu_startoff_inside_chunks_int64,
                    int64_t chunk_size_in_bytes, int layer_idx, int kv_idx,
                    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
                    int start_layer_id, bool cpu_phys_contig);

void gather_from_cpu(void *staging_buf, const int64_t *cpu_ptr_int64,
                     const int64_t *cpu_block_ids, int num_blocks,
                     int64_t cpu_block_stride_int64,
                     int64_t cpu_startoff_inside_chunks_int64,
                     int64_t chunk_size_in_bytes, int layer_idx, int kv_idx,
                     int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
                     int start_layer_id, bool cpu_phys_contig);

} // namespace flexkv
