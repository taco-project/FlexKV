/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * CE adaptive transfer: host-side analysis + multi-path execution.
 * Extracted from transfer.cu for maintainability and future extension
 * (e.g. multi-process scatter).
 */
#pragma once

#include "gtensor_handler.cuh"
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <vector>

namespace flexkv {

// ============================================================================
// CE transfer configuration (passed from Python GLOBAL_CONFIG_FROM_ENV)
// ============================================================================

struct CETransferConfig {
  int64_t segment_threshold = 8;
  bool use_pingpong = true;
  bool path_opt_enabled = true;  // Enable Path 0/1/2 (false = baseline)
};

// ============================================================================
// Analysis structs
// ============================================================================

struct CESegment {
  int start_k;    // start block index in the id arrays
  int run_len;    // number of blocks in this segment
};

struct CEAnalysis {
  bool src_log_contig;   // gpu_block_ids[k+1] == gpu_block_ids[k]+1
  bool dst_log_contig;   // cpu_block_ids[k+1] == cpu_block_ids[k]+1
  bool dst_phys_contig;  // cpu_block_stride == chunk_size (LAYERFIRST + non-sharded)
  bool src_phys_contig;  // gpu_block_stride == chunk_size (non-sharded D2H)
  int num_segments;
  std::vector<CESegment> segments;
};

// ============================================================================
// Analysis & path selection
// ============================================================================

CEAnalysis analyze_ce_transfer(
    const int64_t *gpu_block_ids, const int64_t *cpu_block_ids,
    int num_blocks, int64_t cpu_block_stride_in_bytes,
    int64_t chunk_size_in_bytes, int64_t gpu_block_stride_in_bytes);

int choose_path(const CEAnalysis &a, const CETransferConfig &cfg);

// ============================================================================
// Cached staging buffers & events
// ============================================================================

void *get_cached_hugepage_buffer(size_t size);
cudaEvent_t *get_cached_pingpong_events();

// ============================================================================
// Baseline: per-block memcpy (no optimization)
template <BackendType Type>
void ce_path_baseline(
    int num_blocks, int start_layer_id, int num_layers, int kv_dim,
    int64_t *gpu_block_ids, GTensorHandler gpu_tensor_handler,
    int64_t gpu_startoff_inside_chunks_int64,
    int64_t *cpu_block_ids, int64_t *cpu_ptr_int64,
    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
    int64_t cpu_block_stride_int64,
    int64_t cpu_startoff_inside_chunks_int64, int64_t chunk_size_in_bytes,
    cudaStream_t stream, bool is_host_to_device);

// Path 0: single large memcpy per (layer, kv_dim)
//    Requires: src_log_contig && dst_log_contig && dst_phys_contig
// ============================================================================

template <BackendType Type>
void ce_path0_single_memcpy(
    int num_blocks, int start_layer_id, int num_layers, int kv_dim,
    int64_t *gpu_block_ids, GTensorHandler gpu_tensor_handler,
    int64_t gpu_startoff_inside_chunks_int64,
    int64_t *cpu_block_ids, int64_t *cpu_ptr_int64,
    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
    int64_t cpu_block_stride_int64,
    int64_t cpu_startoff_inside_chunks_int64, int64_t chunk_size_in_bytes,
    cudaStream_t stream, bool is_host_to_device);

// ============================================================================
// Path 1: per-segment memcpy (few segments <= threshold)
//    dst_phys_contig (LF + non-sharded): direct segment memcpy
//    !dst_phys_contig (BF or sharded D2H): staging + CPU scatter
//      src_phys_contig: continuous segment memcpy GPU<->staging
//      !src_phys_contig: per-block memcpy GPU<->staging (sharded D2H only)
// ============================================================================

template <BackendType Type>
void ce_path1_segment_memcpy(
    int num_blocks, int start_layer_id, int num_layers, int kv_dim,
    int64_t *gpu_block_ids, GTensorHandler gpu_tensor_handler,
    int64_t gpu_startoff_inside_chunks_int64,
    int64_t *cpu_block_ids, int64_t *cpu_ptr_int64,
    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
    int64_t cpu_block_stride_int64,
    int64_t cpu_startoff_inside_chunks_int64, int64_t chunk_size_in_bytes,
    cudaStream_t stream, bool is_host_to_device,
    const CEAnalysis &analysis, const CETransferConfig &cfg);

// ============================================================================
// Path 2: gather/scatter pipeline (many segments > threshold)
//    Requires: src_phys_contig (GPU block stride == chunk_size)
//    D2H: GPU index_select gather -> D2H staging -> CPU scatter
//    H2D: CPU gather -> H2D staging -> GPU index_copy_ scatter
// ============================================================================

template <BackendType Type>
void ce_path2_gather_scatter(
    int num_blocks, int start_layer_id, int num_layers, int kv_dim,
    int64_t *gpu_block_ids, GTensorHandler gpu_tensor_handler,
    int64_t gpu_startoff_inside_chunks_int64,
    int64_t *cpu_block_ids, int64_t *cpu_ptr_int64,
    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
    int64_t cpu_block_stride_int64,
    int64_t cpu_startoff_inside_chunks_int64, int64_t chunk_size_in_bytes,
    cudaStream_t stream, bool is_host_to_device,
    const CEAnalysis &analysis, const CETransferConfig &cfg);

} // namespace flexkv
