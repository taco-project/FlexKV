/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * CE adaptive transfer implementation: host-side analysis + multi-path
 * execution. Extracted from transfer.cu.
 *
 * Three paths selected by analyzing block-id contiguity:
 *
 *   Path 0: Both gpu/cpu block ids fully contiguous AND cpu physical stride
 *           == chunk_size (LAYERFIRST). Single large cudaMemcpyAsync per
 *           (layer, kv_dim) — optimal, O(1) API calls.
 *
 *   Path 1: Few segments (<= threshold). Per-segment cudaMemcpyAsync.
 *           - LAYERFIRST (phys_contig): direct memcpy to final dst.
 *           - BLOCKFIRST (!phys_contig): staging buffer + CPU scatter with
 *             optional ping-pong overlap.
 *
 *   Path 2: Many segments (> threshold). GPU index_select gather + single
 *           D2H/H2D + CPU scatter (D2H) or CPU gather + H2D + GPU
 *           index_copy_ scatter (H2D). Uses staging buffer + ping-pong.
 *
 * Configuration is passed via CETransferConfig struct (from Python
 * GLOBAL_CONFIG_FROM_ENV), NOT read from environment variables directly.
 */
#include "ce_transfer.h"

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstring>
#include <utility>

#include "monitoring/metrics_manager.h"

// FLEXKV_GPU_CPU_TRANSFER: metrics hook for transfer byte accounting.
// Defined as no-op when monitoring is disabled (FLEXKV_ENABLE_METRICS=0).
#ifndef FLEXKV_GPU_CPU_TRANSFER
#define FLEXKV_GPU_CPU_TRANSFER(is_h2d, size)
#endif

namespace flexkv {

// ---- Segment computation ----

CEAnalysis analyze_ce_transfer(
    const int64_t *gpu_block_ids, const int64_t *cpu_block_ids,
    int num_blocks, int64_t cpu_block_stride_in_bytes,
    int64_t chunk_size_in_bytes, int64_t gpu_block_stride_in_bytes) {
  CEAnalysis a;
  a.src_log_contig = true;
  a.dst_log_contig = true;
  a.dst_phys_contig = (cpu_block_stride_in_bytes == chunk_size_in_bytes);
  a.src_phys_contig = (gpu_block_stride_in_bytes == 0 ||
                       gpu_block_stride_in_bytes == chunk_size_in_bytes);
  a.num_segments = 0;

  if (num_blocks == 0) return a;

  a.num_segments = 1;
  int seg_start = 0;
  for (int k = 1; k < num_blocks; ++k) {
    bool src_step = (gpu_block_ids[k] == gpu_block_ids[k - 1] + 1);
    bool dst_step = (cpu_block_ids[k] == cpu_block_ids[k - 1] + 1);
    if (!src_step) a.src_log_contig = false;
    if (!dst_step) a.dst_log_contig = false;
    if (!src_step || !dst_step) {
      a.segments.push_back({seg_start, k - seg_start});
      seg_start = k;
      a.num_segments++;
    }
  }
  a.segments.push_back({seg_start, num_blocks - seg_start});
  return a;
}

// ---- Path selection ----

int choose_path(const CEAnalysis &a, const CETransferConfig &cfg) {
  // Path 0: logical + physical contiguity on both sides
  if (a.src_log_contig && a.dst_log_contig && a.dst_phys_contig) return 0;
  // Path 2 (index_select) requires src_phys_contig (GPU block stride == chunk_size).
  // Sharded D2H has chunk=shard, stride=gpu_chunk -> !src_phys_contig -> skip Path 2.
  if (!a.src_phys_contig) return 1;
  // Path 1: few segments
  if (a.num_segments <= cfg.segment_threshold) return 1;
  // Path 2: many segments
  return 2;
}

// ---- Cached hugepage staging buffer ----

void *get_cached_hugepage_buffer(size_t size) {
  thread_local void *buf = nullptr;
  thread_local size_t buf_size = 0;
  if (size > buf_size) {
    if (buf) {
      cudaFreeHost(buf);
    }
    cudaMallocHost(&buf, size, cudaHostAllocDefault);
    buf_size = size;
  }
  return buf;
}

// ---- Cached ping-pong events ----

cudaEvent_t *get_cached_pingpong_events() {
  thread_local cudaEvent_t events[2] = {nullptr, nullptr};
  if (events[0] == nullptr) {
    cudaEventCreateWithFlags(&events[0], cudaEventDisableTiming);
    cudaEventCreateWithFlags(&events[1], cudaEventDisableTiming);
  }
  return events;
}


// ============================================================================
// Baseline: per-block memcpy (no segment merging, no index_select, no ping-pong)
//   Used as microbenchmark baseline to quantify optimization gains.
//   Correct for all modes (sharded, BF, etc.) but slowest — O(num_blocks) API calls.
// ============================================================================
template <BackendType Type>
void ce_path_baseline(
    int num_blocks, int start_layer_id, int num_layers, int kv_dim,
    int64_t *gpu_block_ids, GTensorHandler gpu_tensor_handler,
    int64_t gpu_startoff_inside_chunks_int64,
    int64_t *cpu_block_ids, int64_t *cpu_ptr_int64,
    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
    int64_t cpu_block_stride_int64,
    int64_t cpu_startoff_inside_chunks_int64, int64_t chunk_size_in_bytes,
    cudaStream_t stream, bool is_host_to_device) {
  cudaMemcpyKind kind = is_host_to_device ? cudaMemcpyHostToDevice
                                          : cudaMemcpyDeviceToHost;
  for (int i = 0; i < num_layers; i++) {
    for (int j = 0; j < kv_dim; j++) {
      int64_t *cpu_base =
          cpu_ptr_int64 + (i + start_layer_id) * cpu_layer_stride_int64 +
          j * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
      for (int b = 0; b < num_blocks; b++) {
        int64_t *gpu_ptr = ptr_at<Type>(gpu_tensor_handler,
                                        i + start_layer_id, j,
                                        gpu_block_ids[b]);
        int64_t *gpu_ptr_off =
            reinterpret_cast<int64_t *>(gpu_ptr) +
            gpu_startoff_inside_chunks_int64;
        int64_t *cpu_ptr_b =
            cpu_base + cpu_block_ids[b] * cpu_block_stride_int64;
        void *dst = is_host_to_device ? (void *)gpu_ptr_off : (void *)cpu_ptr_b;
        void *src = is_host_to_device ? (void *)cpu_ptr_b : (void *)gpu_ptr_off;
        cudaMemcpyAsync(dst, src, chunk_size_in_bytes, kind, stream);
        FLEXKV_GPU_CPU_TRANSFER(is_host_to_device, chunk_size_in_bytes);
      }
    }
  }
}

// ============================================================================
// Path 0: single large memcpy per (layer, kv_dim)
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
    cudaStream_t stream, bool is_host_to_device) {
  int64_t big_size = chunk_size_in_bytes * num_blocks;
  cudaMemcpyKind kind = is_host_to_device ? cudaMemcpyHostToDevice
                                          : cudaMemcpyDeviceToHost;
  for (int i = 0; i < num_layers; i++) {
    for (int j = 0; j < kv_dim; j++) {
      int64_t *cpu_chunk_ptr =
          cpu_ptr_int64 + (i + start_layer_id) * cpu_layer_stride_int64 +
          j * cpu_kv_stride_int64 +
          cpu_block_ids[0] * cpu_block_stride_int64 +
          cpu_startoff_inside_chunks_int64;
      int64_t *gpu_ptr = ptr_at<Type>(gpu_tensor_handler, i + start_layer_id,
                                      j, gpu_block_ids[0]);
      int64_t *gpu_chunk_ptr = reinterpret_cast<int64_t *>(gpu_ptr) +
                               gpu_startoff_inside_chunks_int64;
      void *dst = is_host_to_device ? (void *)gpu_chunk_ptr
                                    : (void *)cpu_chunk_ptr;
      void *src = is_host_to_device ? (void *)cpu_chunk_ptr
                                    : (void *)gpu_chunk_ptr;
      cudaMemcpyAsync(dst, src, big_size, kind, stream);
      FLEXKV_GPU_CPU_TRANSFER(is_host_to_device, big_size);
    }
  }
}

// ============================================================================
// Path 1: per-segment memcpy (few segments)
//    LAYERFIRST: direct segment memcpy
//    BLOCKFIRST: staging buffer + CPU scatter + optional ping-pong
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
    const CEAnalysis &analysis, const CETransferConfig &cfg) {
  cudaMemcpyKind kind = is_host_to_device ? cudaMemcpyHostToDevice
                                          : cudaMemcpyDeviceToHost;
  bool use_staging = !analysis.dst_phys_contig;  // BLOCKFIRST
  bool use_pingpong = use_staging && cfg.use_pingpong;

  if (!use_staging) {
    // ---- LAYERFIRST: direct segment memcpy ----
    for (int i = 0; i < num_layers; i++) {
      for (int j = 0; j < kv_dim; j++) {
        for (const auto &seg : analysis.segments) {
          int64_t seg_size = (int64_t)seg.run_len * chunk_size_in_bytes;
          int64_t *cpu_ptr =
              cpu_ptr_int64 + (i + start_layer_id) * cpu_layer_stride_int64 +
              j * cpu_kv_stride_int64 +
              cpu_block_ids[seg.start_k] * cpu_block_stride_int64 +
              cpu_startoff_inside_chunks_int64;
          int64_t *gpu_ptr = ptr_at<Type>(gpu_tensor_handler,
                                          i + start_layer_id, j,
                                          gpu_block_ids[seg.start_k]);
          int64_t *gpu_ptr_off =
              reinterpret_cast<int64_t *>(gpu_ptr) +
              gpu_startoff_inside_chunks_int64;
          void *dst = is_host_to_device ? (void *)gpu_ptr_off
                                        : (void *)cpu_ptr;
          void *src = is_host_to_device ? (void *)cpu_ptr
                                        : (void *)gpu_ptr_off;
          cudaMemcpyAsync(dst, src, seg_size, kind, stream);
          FLEXKV_GPU_CPU_TRANSFER(is_host_to_device, seg_size);
        }
      }
    }
  } else {
    // ---- BLOCKFIRST: staging buffer + CPU scatter/gather ----
    size_t layer_buf_size = (size_t)num_blocks * chunk_size_in_bytes;
    bool need_pp = use_pingpong;

    void *host_base = get_cached_hugepage_buffer(need_pp ? layer_buf_size * 2
                                                         : layer_buf_size);
    void *host_bufs[2] = {
        host_base,
        need_pp ? (char *)host_base + layer_buf_size : nullptr};
    cudaEvent_t *pp_events = need_pp ? get_cached_pingpong_events() : nullptr;

    const int64_t total_iters = (int64_t)num_layers * kv_dim;
    for (int64_t it = 0; it < total_iters; ++it) {
      int i = (int)(it / kv_dim);
      int j = (int)(it % kv_dim);
      int idx = need_pp ? (int)(it & 1) : 0;
      int prev_idx = idx ^ 1;
      void *buf = host_bufs[idx];

      if (!is_host_to_device) {
        // ---- D2H ----
        // D2H all segments into staging
        int64_t seg_offset = 0;
        if (analysis.src_phys_contig) {
          // GPU blocks contiguous: continuous segment memcpy
          for (const auto &seg : analysis.segments) {
            int64_t seg_size = (int64_t)seg.run_len * chunk_size_in_bytes;
            int64_t *gpu_ptr = ptr_at<Type>(gpu_tensor_handler,
                                            i + start_layer_id, j,
                                            gpu_block_ids[seg.start_k]);
            int64_t *gpu_ptr_off =
                reinterpret_cast<int64_t *>(gpu_ptr) +
                gpu_startoff_inside_chunks_int64;
            cudaMemcpyAsync((char *)buf + seg_offset, gpu_ptr_off, seg_size,
                            cudaMemcpyDeviceToHost, stream);
            FLEXKV_GPU_CPU_TRANSFER(false, seg_size);
            seg_offset += seg_size;
          }
        } else {
          // GPU blocks not contiguous (sharded D2H): per-block memcpy
          for (const auto &seg : analysis.segments) {
            for (int b = 0; b < seg.run_len; ++b) {
              int64_t *gpu_ptr = ptr_at<Type>(gpu_tensor_handler,
                                              i + start_layer_id, j,
                                              gpu_block_ids[seg.start_k + b]);
              int64_t *gpu_ptr_off =
                  reinterpret_cast<int64_t *>(gpu_ptr) +
                  gpu_startoff_inside_chunks_int64;
              cudaMemcpyAsync((char *)buf + seg_offset, gpu_ptr_off,
                              chunk_size_in_bytes,
                              cudaMemcpyDeviceToHost, stream);
              FLEXKV_GPU_CPU_TRANSFER(false, chunk_size_in_bytes);
              seg_offset += chunk_size_in_bytes;
            }
          }
        }
        if (need_pp) {
          cudaEventRecord(pp_events[idx], stream);
          // CPU scatter previous layer
          if (it >= 1) {
            cudaEventSynchronize(pp_events[prev_idx]);
            int pi = (int)((it - 1) / kv_dim);
            int pj = (int)((it - 1) % kv_dim);
            // scatter from host_bufs[prev_idx] to strided dst
            int64_t *cpu_base =
                cpu_ptr_int64 + pi * cpu_layer_stride_int64 +
                pj * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
            int64_t off = 0;
            for (const auto &seg : analysis.segments) {
              for (int b = 0; b < seg.run_len; ++b) {
                int64_t cb = cpu_block_ids[seg.start_k + b];
                int64_t *dst = cpu_base + cb * cpu_block_stride_int64;
                memcpy(dst, (char *)host_bufs[prev_idx] + off,
                       chunk_size_in_bytes);
                off += chunk_size_in_bytes;
              }
            }
          }
        } else {
          cudaStreamSynchronize(stream);
          // scatter current layer
          int64_t *cpu_base =
              cpu_ptr_int64 + i * cpu_layer_stride_int64 +
              j * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
          int64_t off = 0;
          for (const auto &seg : analysis.segments) {
            for (int b = 0; b < seg.run_len; ++b) {
              int64_t cb = cpu_block_ids[seg.start_k + b];
              int64_t *dst = cpu_base + cb * cpu_block_stride_int64;
              memcpy(dst, (char *)buf + off, chunk_size_in_bytes);
              off += chunk_size_in_bytes;
            }
          }
        }
      } else {
        // ---- H2D ----
        // CPU gather from strided src into staging
        int64_t *cpu_base =
            cpu_ptr_int64 + i * cpu_layer_stride_int64 +
            j * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
        int64_t off = 0;
        for (const auto &seg : analysis.segments) {
          for (int b = 0; b < seg.run_len; ++b) {
            int64_t cb = cpu_block_ids[seg.start_k + b];
            int64_t *src = cpu_base + cb * cpu_block_stride_int64;
            memcpy((char *)buf + off, src, chunk_size_in_bytes);
            off += chunk_size_in_bytes;
          }
        }
        // H2D all segments from staging
        off = 0;
        if (analysis.src_phys_contig) {
          // GPU blocks contiguous: continuous segment memcpy
          for (const auto &seg : analysis.segments) {
            int64_t seg_size = (int64_t)seg.run_len * chunk_size_in_bytes;
            int64_t *gpu_ptr = ptr_at<Type>(gpu_tensor_handler,
                                            i + start_layer_id, j,
                                            gpu_block_ids[seg.start_k]);
            int64_t *gpu_ptr_off =
                reinterpret_cast<int64_t *>(gpu_ptr) +
                gpu_startoff_inside_chunks_int64;
            cudaMemcpyAsync(gpu_ptr_off, (char *)buf + off, seg_size,
                            cudaMemcpyHostToDevice, stream);
            FLEXKV_GPU_CPU_TRANSFER(true, seg_size);
            off += seg_size;
          }
        } else {
          // GPU blocks not contiguous (sharded D2H): per-block memcpy
          for (const auto &seg : analysis.segments) {
            for (int b = 0; b < seg.run_len; ++b) {
              int64_t *gpu_ptr = ptr_at<Type>(gpu_tensor_handler,
                                              i + start_layer_id, j,
                                              gpu_block_ids[seg.start_k + b]);
              int64_t *gpu_ptr_off =
                  reinterpret_cast<int64_t *>(gpu_ptr) +
                  gpu_startoff_inside_chunks_int64;
              cudaMemcpyAsync(gpu_ptr_off, (char *)buf + off,
                              chunk_size_in_bytes,
                              cudaMemcpyHostToDevice, stream);
              FLEXKV_GPU_CPU_TRANSFER(true, chunk_size_in_bytes);
              off += chunk_size_in_bytes;
            }
          }
        }
        if (need_pp) {
          cudaEventRecord(pp_events[idx], stream);
          if (it >= 1) {
            cudaEventSynchronize(pp_events[prev_idx]);
          }
        }
      }
    }
    // Drain last ping-pong slot (D2H)
    if (!is_host_to_device && need_pp && total_iters >= 1) {
      int64_t last = total_iters - 1;
      int last_idx = (int)(last & 1);
      cudaEventSynchronize(pp_events[last_idx]);
      int li = (int)(last / kv_dim);
      int lj = (int)(last % kv_dim);
      int64_t *cpu_base = cpu_ptr_int64 + li * cpu_layer_stride_int64 +
          lj * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
      int64_t off = 0;
      for (const auto &seg : analysis.segments) {
        for (int b = 0; b < seg.run_len; ++b) {
          int64_t cb = cpu_block_ids[seg.start_k + b];
          int64_t *dst = cpu_base + cb * cpu_block_stride_int64;
          memcpy(dst, (char *)host_bufs[last_idx] + off,
                 chunk_size_in_bytes);
          off += chunk_size_in_bytes;
        }
      }
    }
    // Drain last ping-pong slot (H2D)
    if (is_host_to_device && need_pp && total_iters >= 1) {
      int64_t last = total_iters - 1;
      int last_idx = (int)(last & 1);
      cudaEventSynchronize(pp_events[last_idx]);
    }
  }
}

// ============================================================================
// Path 2: gather/scatter pipeline
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
    const CEAnalysis &analysis, const CETransferConfig &cfg) {
  TORCH_CHECK(chunk_size_in_bytes % sizeof(int64_t) == 0,
              "Path 2 requires chunk_size_in_bytes % 8 == 0");
  const int64_t elems_per_block = chunk_size_in_bytes / sizeof(int64_t);
  const int64_t buffer_size = (int64_t)num_blocks * chunk_size_in_bytes;
  bool use_pingpong = cfg.use_pingpong;

  // Bind ATen to our cuda stream
  int cur_dev = 0;
  cudaGetDevice(&cur_dev);
  c10::cuda::CUDAStream aten_stream =
      c10::cuda::getStreamFromExternal(stream, cur_dev);
  c10::cuda::CUDAStreamGuard stream_guard(aten_stream);

  // Find max indices for tensor views
  int64_t max_gpu_id = 0, max_cpu_id = 0;
  for (int k = 0; k < num_blocks; ++k) {
    if (gpu_block_ids[k] > max_gpu_id) max_gpu_id = gpu_block_ids[k];
    if (cpu_block_ids[k] > max_cpu_id) max_cpu_id = cpu_block_ids[k];
  }

  auto i64_cuda = at::TensorOptions().dtype(at::kLong).device(at::kCUDA);
  auto i64_cpu = at::TensorOptions().dtype(at::kLong).device(at::kCPU);

  // Transfer block ids to GPU (for index_select / index_copy_)
  at::Tensor gpu_ids_cpu = at::from_blob(
      const_cast<int64_t *>(gpu_block_ids), {num_blocks}, i64_cpu);
  at::Tensor gpu_ids_cuda = gpu_ids_cpu.to(at::kCUDA, /*non_blocking=*/true);
  at::Tensor cpu_ids_cpu = at::from_blob(
      const_cast<int64_t *>(cpu_block_ids), {num_blocks}, i64_cpu);
  at::Tensor dst_ids_cuda =
      (is_host_to_device)
          ? cpu_ids_cpu.to(at::kCUDA, /*non_blocking=*/true)
          : at::Tensor();

  // Allocate ping-pong device buffers
  bool need_dev_buf =
      (!is_host_to_device && !analysis.src_log_contig) ||
      (is_host_to_device && !analysis.dst_log_contig);
  at::Tensor dev_buf[2];
  if (need_dev_buf) {
    dev_buf[0] = at::empty({num_blocks, elems_per_block}, i64_cuda);
    dev_buf[1] = at::empty({num_blocks, elems_per_block}, i64_cuda);
  }

  // Allocate ping-pong host staging buffers
  bool need_host_buf =
      !is_host_to_device ||  // D2H: always stage then scatter
      (is_host_to_device && !analysis.src_log_contig);

  at::Tensor host_buf[2];
  if (need_host_buf) {
    auto pinned = at::TensorOptions().dtype(at::kLong).device(at::kCPU)
                      .pinned_memory(true);
    host_buf[0] = at::empty({num_blocks, elems_per_block}, pinned);
    if (use_pingpong) {
      host_buf[1] = at::empty({num_blocks, elems_per_block}, pinned);
    }
  }

  cudaEvent_t *pp_events = (need_host_buf && use_pingpong)
                                ? get_cached_pingpong_events()
                                : nullptr;

  const int64_t total_iters = (int64_t)num_layers * kv_dim;
  for (int64_t it = 0; it < total_iters; ++it) {
    int i = (int)(it / kv_dim);
    int j = (int)(it % kv_dim);
    int idx = use_pingpong ? (int)(it & 1) : 0;
    int prev_idx = idx ^ 1;

    int64_t *gpu_layer_kv_base =
        ptr_at<Type>(gpu_tensor_handler, i + start_layer_id, j, 0);

    if (!is_host_to_device) {
      // ============ D2H ============
      // Step 1: GPU gather (if src non-contig)
      const int64_t *d2h_src;
      if (analysis.src_log_contig) {
        d2h_src = reinterpret_cast<int64_t *>(gpu_layer_kv_base) +
                  gpu_startoff_inside_chunks_int64 +
                  gpu_block_ids[0] * (chunk_size_in_bytes / sizeof(int64_t));
      } else {
        at::Tensor src_view = at::from_blob(
            gpu_layer_kv_base, {max_gpu_id + 1, elems_per_block}, i64_cuda);
        at::index_select_out(dev_buf[idx], src_view, 0, gpu_ids_cuda);
        d2h_src = reinterpret_cast<int64_t *>(dev_buf[idx].data_ptr());
      }

      // Step 2: D2H into staging
      void *dst_ptr = need_host_buf ? host_buf[idx].data_ptr()
                                   : (cpu_ptr_int64 +
                                      (i + start_layer_id) * cpu_layer_stride_int64 +
                                      j * cpu_kv_stride_int64 +
                                      cpu_block_ids[0] * cpu_block_stride_int64 +
                                      cpu_startoff_inside_chunks_int64);
      cudaMemcpyAsync(dst_ptr, d2h_src, buffer_size,
                      cudaMemcpyDeviceToHost, stream);
      FLEXKV_GPU_CPU_TRANSFER(false, buffer_size);

      if (pp_events) {
        cudaEventRecord(pp_events[idx], stream);
        // Step 3: CPU scatter previous slot
        if (it >= 1) {
          cudaEventSynchronize(pp_events[prev_idx]);
          int pi = (int)((it - 1) / kv_dim);
          int pj = (int)((it - 1) % kv_dim);
          int64_t *cpu_base = cpu_ptr_int64 + pi * cpu_layer_stride_int64 +
              pj * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
          for (int k = 0; k < num_blocks; ++k) {
            int64_t cb = cpu_block_ids[k];
            memcpy(cpu_base + cb * cpu_block_stride_int64,
                   (char *)host_buf[prev_idx].data_ptr() +
                       (int64_t)k * chunk_size_in_bytes,
                   chunk_size_in_bytes);
          }
        }
      } else if (need_host_buf) {
        cudaStreamSynchronize(stream);
        // scatter current
        int64_t *cpu_base = cpu_ptr_int64 + i * cpu_layer_stride_int64 +
            j * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
        for (int k = 0; k < num_blocks; ++k) {
          int64_t cb = cpu_block_ids[k];
          memcpy(cpu_base + cb * cpu_block_stride_int64,
                 (char *)host_buf[idx].data_ptr() +
                     (int64_t)k * chunk_size_in_bytes,
                 chunk_size_in_bytes);
        }
      }
    } else {
      // ============ H2D ============
      // Step 1: CPU gather (if src non-contig)
      const void *h2d_src;
      if (analysis.src_log_contig && analysis.dst_phys_contig) {
        // Direct from cpu_ptr
        h2d_src = cpu_ptr_int64 +
                  (i + start_layer_id) * cpu_layer_stride_int64 +
                  j * cpu_kv_stride_int64 +
                  cpu_block_ids[0] * cpu_block_stride_int64 +
                  cpu_startoff_inside_chunks_int64;
      } else {
        if (pp_events && it >= 1) {
          cudaEventSynchronize(pp_events[idx]);
        }
        // gather into staging
        int64_t *cpu_base = cpu_ptr_int64 + i * cpu_layer_stride_int64 +
            j * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
        for (int k = 0; k < num_blocks; ++k) {
          int64_t cb = cpu_block_ids[k];
          memcpy((char *)host_buf[idx].data_ptr() +
                     (int64_t)k * chunk_size_in_bytes,
                 cpu_base + cb * cpu_block_stride_int64,
                 chunk_size_in_bytes);
        }
        h2d_src = host_buf[idx].data_ptr();
      }

      // Step 2: H2D
      void *h2d_dst;
      if (analysis.dst_log_contig) {
        h2d_dst = reinterpret_cast<int64_t *>(gpu_layer_kv_base) +
                  gpu_startoff_inside_chunks_int64 +
                  gpu_block_ids[0] * (chunk_size_in_bytes / sizeof(int64_t));
      } else {
        h2d_dst = dev_buf[idx].data_ptr();
      }
      cudaMemcpyAsync(h2d_dst, h2d_src, buffer_size,
                      cudaMemcpyHostToDevice, stream);
      FLEXKV_GPU_CPU_TRANSFER(true, buffer_size);

      if (pp_events) {
        cudaEventRecord(pp_events[idx], stream);
      }

      // Step 3: GPU scatter (if dst non-contig)
      if (!analysis.dst_log_contig) {
        at::Tensor dst_view = at::from_blob(
            gpu_layer_kv_base, {max_gpu_id + 1, elems_per_block}, i64_cuda);
        dst_view.index_copy_(0, dst_ids_cuda, dev_buf[idx]);
      }
    }
  }

  // Drain last D2H scatter
  if (!is_host_to_device && pp_events && total_iters >= 1) {
    int64_t last = total_iters - 1;
    int last_idx = (int)(last & 1);
    cudaEventSynchronize(pp_events[last_idx]);
    int li = (int)(last / kv_dim);
    int lj = (int)(last % kv_dim);
    int64_t *cpu_base = cpu_ptr_int64 + li * cpu_layer_stride_int64 +
        lj * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
    for (int k = 0; k < num_blocks; ++k) {
      int64_t cb = cpu_block_ids[k];
      memcpy(cpu_base + cb * cpu_block_stride_int64,
             (char *)host_buf[last_idx].data_ptr() +
                 (int64_t)k * chunk_size_in_bytes,
             chunk_size_in_bytes);
    }
  }

  // Drain last H2D
  if (is_host_to_device && pp_events && total_iters >= 1) {
    int64_t last = total_iters - 1;
    int last_idx = (int)(last & 1);
    cudaEventSynchronize(pp_events[last_idx]);
  }

  // Sync if GPU scatter ran on non-default stream
  if (is_host_to_device && !analysis.dst_log_contig) {
    cudaStreamSynchronize(stream);
  }
}

// ---- Explicit template instantiations ----

template void ce_path0_single_memcpy<BackendType::VLLM>(
    int, int, int, int, int64_t *, GTensorHandler, int64_t,
    int64_t *, int64_t *, int64_t, int64_t, int64_t, int64_t, int64_t,
    cudaStream_t, bool);
template void ce_path0_single_memcpy<BackendType::TRTLLM>(
    int, int, int, int, int64_t *, GTensorHandler, int64_t,
    int64_t *, int64_t *, int64_t, int64_t, int64_t, int64_t, int64_t,
    cudaStream_t, bool);
template void ce_path_baseline<BackendType::VLLM>(
    int, int, int, int, int64_t*, GTensorHandler, int64_t,
    int64_t*, int64_t*, int64_t, int64_t, int64_t, int64_t, int64_t,
    cudaStream_t, bool);
template void ce_path_baseline<BackendType::TRTLLM>(
    int, int, int, int, int64_t*, GTensorHandler, int64_t,
    int64_t*, int64_t*, int64_t, int64_t, int64_t, int64_t, int64_t,
    cudaStream_t, bool);
template void ce_path_baseline<BackendType::SGLANG>(
    int, int, int, int, int64_t*, GTensorHandler, int64_t,
    int64_t*, int64_t*, int64_t, int64_t, int64_t, int64_t, int64_t,
    cudaStream_t, bool);

template void ce_path0_single_memcpy<BackendType::SGLANG>(
    int, int, int, int, int64_t *, GTensorHandler, int64_t,
    int64_t *, int64_t *, int64_t, int64_t, int64_t, int64_t, int64_t,
    cudaStream_t, bool);

template void ce_path1_segment_memcpy<BackendType::VLLM>(
    int, int, int, int, int64_t *, GTensorHandler, int64_t,
    int64_t *, int64_t *, int64_t, int64_t, int64_t, int64_t, int64_t,
    cudaStream_t, bool, const CEAnalysis &, const CETransferConfig &);
template void ce_path1_segment_memcpy<BackendType::TRTLLM>(
    int, int, int, int, int64_t *, GTensorHandler, int64_t,
    int64_t *, int64_t *, int64_t, int64_t, int64_t, int64_t, int64_t,
    cudaStream_t, bool, const CEAnalysis &, const CETransferConfig &);
template void ce_path1_segment_memcpy<BackendType::SGLANG>(
    int, int, int, int, int64_t *, GTensorHandler, int64_t,
    int64_t *, int64_t *, int64_t, int64_t, int64_t, int64_t, int64_t,
    cudaStream_t, bool, const CEAnalysis &, const CETransferConfig &);

template void ce_path2_gather_scatter<BackendType::VLLM>(
    int, int, int, int, int64_t *, GTensorHandler, int64_t,
    int64_t *, int64_t *, int64_t, int64_t, int64_t, int64_t, int64_t,
    cudaStream_t, bool, const CEAnalysis &, const CETransferConfig &);
template void ce_path2_gather_scatter<BackendType::TRTLLM>(
    int, int, int, int, int64_t *, GTensorHandler, int64_t,
    int64_t *, int64_t *, int64_t, int64_t, int64_t, int64_t, int64_t,
    cudaStream_t, bool, const CEAnalysis &, const CETransferConfig &);
template void ce_path2_gather_scatter<BackendType::SGLANG>(
    int, int, int, int, int64_t *, GTensorHandler, int64_t,
    int64_t *, int64_t *, int64_t, int64_t, int64_t, int64_t, int64_t,
    cudaStream_t, bool, const CEAnalysis &, const CETransferConfig &);

} // namespace flexkv
