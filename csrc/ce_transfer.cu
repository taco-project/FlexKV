/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * CE adaptive transfer: host-side ce_analysis + five-path multi-path execution
 * (CONTIG_DIRECT/SEGMENT_DIRECT/SEGMENT_SCATTER/GATHER_SCATTER/GATHER_DIRECT),
 * selected by choose_path(). See ce_transfer.h and docs for the path taxonomy.
 */
#include "ce_transfer.h"

#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <cstring>
#include <utility>
#include <unordered_map>
#include <array>
#include <algorithm>
#include <condition_variable>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

#include "monitoring/metrics_manager.h"

// metrics no-op when monitoring disabled
#ifndef FLEXKV_GPU_CPU_TRANSFER
#define FLEXKV_GPU_CPU_TRANSFER(is_h2d, size)
#endif

namespace flexkv {

// ---- Segment computation ----

CEAnalysis analyze_ce_transfer(
    const int64_t *gpu_block_ids, const int64_t *cpu_block_ids,
    int num_blocks, int64_t cpu_block_stride_in_bytes,
    int64_t chunk_size_in_bytes, int64_t gpu_block_stride_in_bytes) {
  CEAnalysis ce_analysis;
  ce_analysis.gpu_log_contig = true;
  ce_analysis.cpu_log_contig = true;
  ce_analysis.cpu_phys_contig = (cpu_block_stride_in_bytes == chunk_size_in_bytes);
  ce_analysis.gpu_phys_contig = (gpu_block_stride_in_bytes == 0 ||
                              gpu_block_stride_in_bytes == chunk_size_in_bytes);
  ce_analysis.num_segments = 0;

  if (num_blocks == 0) return ce_analysis;

  ce_analysis.num_segments = 1;
  int seg_start = 0;
  for (int k = 1; k < num_blocks; ++k) {
    bool src_step = (gpu_block_ids[k] == gpu_block_ids[k - 1] + 1);
    bool dst_step = (cpu_block_ids[k] == cpu_block_ids[k - 1] + 1);
    if (!src_step) ce_analysis.gpu_log_contig = false;
    if (!dst_step) ce_analysis.cpu_log_contig = false;
    if (!src_step || !dst_step) {
      ce_analysis.segments.push_back({seg_start, k - seg_start});
      seg_start = k;
      ce_analysis.num_segments++;
    }
  }
  ce_analysis.segments.push_back({seg_start, num_blocks - seg_start});
  return ce_analysis;
}

// ---- Path selection ----

// sharded D2H -> GATHER_SCATTER (SEGMENT_SCATTER misplaces shards)
CEPath choose_path(const CEAnalysis &ce_analysis, const CETransferConfig &ce_config,
                   int64_t chunk_size_in_bytes,
                   bool is_host_to_device, bool is_full_block) {
  // GATHER_DIRECT: BF + !cpu_phys_contig + gpu contiguous
  // bfirst+MLA+D2H+!full_block -> SEGMENT_SCATTER
  if (ce_config.is_blockfirst && !ce_analysis.cpu_phys_contig && ce_analysis.gpu_phys_contig) {
    // D2H only
    if (!is_host_to_device && ce_config.is_mla && !is_full_block)
      return CEPath::SEGMENT_SCATTER;  // bfirst MLA layer_parallel D2H
    return CEPath::GATHER_DIRECT;
  }

  // CONTIG_DIRECT: both sides contig -> one big memcpy
  if (ce_analysis.gpu_log_contig && ce_analysis.cpu_log_contig && ce_analysis.cpu_phys_contig && ce_analysis.gpu_phys_contig)
    return CEPath::CONTIG_DIRECT;

  // Sharded D2H -> GATHER_SCATTER (SEGMENT_SCATTER misplaces shards).
  if (!ce_analysis.gpu_phys_contig)
    return CEPath::GATHER_SCATTER;

  // LAYERFIRST non-MLA uses SEGMENT_SCATTER (strided is head-dim, D2D can't help).
  if (ce_analysis.num_segments <= ce_config.segment_threshold) {
    return ce_analysis.cpu_phys_contig ? CEPath::SEGMENT_DIRECT
                             : CEPath::SEGMENT_SCATTER;
  }
  // Many scattered segments -> GATHER_SCATTER (needs 8-aligned chunk, else S_SCT).
  if (chunk_size_in_bytes > 0 && chunk_size_in_bytes % sizeof(int64_t) != 0)
    return CEPath::SEGMENT_SCATTER;
  return CEPath::GATHER_SCATTER;
}

// Cached host staging buffer (per-device)

struct HostStagingBuf {
  void *buf = nullptr;
  size_t size = 0;

  ~HostStagingBuf() {
    if (buf) {
      cudaFreeHost(buf);
      buf = nullptr;
    }
  }
};

// Device staging buffer cache (per-device)
struct DeviceStagingBuf {
  void *buf = nullptr;
  size_t size = 0;

  ~DeviceStagingBuf() {
    if (buf) {
      cudaFree(buf);
      buf = nullptr;
    }
  }
};

void *get_cached_host_buffer(size_t size) {
  int dev = 0;
  cudaGetDevice(&dev);
  thread_local std::unordered_map<int, HostStagingBuf> cache;
  HostStagingBuf &b = cache[dev];
  if (size > b.size) {
    if (b.buf) {
      cudaFreeHost(b.buf);
    }
    TORCH_CHECK(cudaSuccess == cudaMallocHost(&b.buf, size, cudaHostAllocDefault),
                "cudaMallocHost failed for cached host buffer");
    b.size = size;
  }
  return b.buf;
}

// Cached device buffer (per-device, slot-keyed). null on cudaMalloc failure -> PER_BLOCK.
void *get_cached_device_buffer(size_t size, int slot) {
  int dev = 0;
  cudaGetDevice(&dev);
  thread_local std::unordered_map<int, std::array<DeviceStagingBuf, 3>> cache;
  DeviceStagingBuf &b = cache[dev][slot];
  if (size > b.size) {
    if (b.buf) {
      cudaFree(b.buf);
      b.buf = nullptr;
      b.size = 0;
    }
    // cudaMalloc failed -> PER_BLOCK
    if (cudaSuccess != cudaMalloc(&b.buf, size)) {
      cudaGetLastError();
      return nullptr;
    }
    b.size = size;
  }
  return b.buf;
}

// Cached CUDA event pair (per-device)
struct CachedEventPair {
  cudaEvent_t ev[2] = {nullptr, nullptr};
  bool created = false;
};

cudaEvent_t *get_cached_event_pair(bool need, bool &created) {
  if (!need) return nullptr;
  thread_local std::unordered_map<int, CachedEventPair> cache;
  int dev = 0;
  cudaGetDevice(&dev);
  CachedEventPair &e = cache[dev];
  if (!e.created) {
    TORCH_CHECK(cudaSuccess == cudaEventCreateWithFlags(&e.ev[0], cudaEventDisableTiming),
                "cudaEventCreateWithFlags failed for cached event[0]");
    TORCH_CHECK(cudaSuccess == cudaEventCreateWithFlags(&e.ev[1], cudaEventDisableTiming),
                "cudaEventCreateWithFlags failed for cached event[1]");
    e.created = true;
  }
  created = true;
  return e.ev;
}


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

// device buffer failure -> fall back to PER_BLOCK.

// ---- CONTIG_DIRECT: single large memcpy ----

template <BackendType Type>
void ce_transfer_contig_direct(
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
    const CEAnalysis &ce_analysis, const CETransferConfig &ce_config) {
  (void)ce_config;  // ping-pong / staging config unused: this path never stages.
  cudaMemcpyKind kind = is_host_to_device ? cudaMemcpyHostToDevice
                                          : cudaMemcpyDeviceToHost;
  for (int i = 0; i < num_layers; i++) {
    for (int j = 0; j < kv_dim; j++) {
      for (const auto &seg : ce_analysis.segments) {
        int64_t seg_size = (int64_t)seg.nr_blocks * chunk_size_in_bytes;
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
}

// ---- Ping-pong: D2H double-buffer (SEGMENT_SCATTER, GATHER_SCATTER); not in direct paths ----

// ---- Parallel CPU gather/scatter with NT store ----
namespace {

enum class CopyImpl { SCALAR, AVX2, AVX512 };

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#define FLEXKV_X86 1
#else
#define FLEXKV_X86 0
#endif

#if FLEXKV_X86
__attribute__((target("avx512f"))) void nt_copy_avx512(char *d,
                                                          const char *s,
                                                          size_t n) {
  size_t i = 0;
  size_t head =
      static_cast<size_t>((64 - (reinterpret_cast<uintptr_t>(d) & 63)) & 63);
  if (head > n)
    head = n;
  if (head) {
    std::memcpy(d, s, head);
    i = head;
  }
  for (; i + 64 <= n; i += 64) {
    __m512i v = _mm512_loadu_si512(reinterpret_cast<const void *>(s + i));
    _mm512_stream_si512(reinterpret_cast<__m512i *>(d + i), v);
  }
  if (i < n)
    std::memcpy(d + i, s + i, n - i);
}

__attribute__((target("avx2"))) void nt_copy_avx2(char *d, const char *s,
                                                     size_t n) {
  size_t i = 0;
  size_t head =
      static_cast<size_t>((32 - (reinterpret_cast<uintptr_t>(d) & 31)) & 31);
  if (head > n)
    head = n;
  if (head) {
    std::memcpy(d, s, head);
    i = head;
  }
  for (; i + 32 <= n; i += 32) {
    __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(s + i));
    _mm256_stream_si256(reinterpret_cast<__m256i *>(d + i), v);
  }
  if (i < n)
    std::memcpy(d + i, s + i, n - i);
}
#endif

CopyImpl pick_copy_impl(bool nt_enabled) {
#if FLEXKV_X86
  if (!nt_enabled)
    return CopyImpl::SCALAR;
  if (__builtin_cpu_supports("avx512f"))
    return CopyImpl::AVX512;
  if (__builtin_cpu_supports("avx2"))
    return CopyImpl::AVX2;
  return CopyImpl::SCALAR;
#else
  (void)nt_enabled;
  return CopyImpl::SCALAR;
#endif
}

inline void nt_copy_block(char *d, const char *s, size_t n, CopyImpl impl) {
#if FLEXKV_X86
  switch (impl) {
  case CopyImpl::AVX512:
    nt_copy_avx512(d, s, n);
    return;
  case CopyImpl::AVX2:
    nt_copy_avx2(d, s, n);
    return;
  default:
    std::memcpy(d, s, n);
    return;
  }
#else
  (void)impl;
  std::memcpy(d, s, n);
#endif
}

inline void sfence_if_nt(CopyImpl impl) {
#if FLEXKV_X86
  if (impl != CopyImpl::SCALAR)
    _mm_sfence();
#else
  (void)impl;
#endif
}

// Persistent per-GPU-thread CPU thread pool. Caller participates as one worker;
// (N-1) background threads handle the rest. cv-based wakeup (not busy-spin).
// thread_local: each GPU dispatch thread gets its own pool — zero cross-GPU contention.
class CopyPool {
public:
  // thread_local: each GPU dispatch thread gets its own pool. No cross-GPU contention.
  // threads param only used on first init; thread_local persists afterwards.
  static CopyPool &instance(int threads) {
    thread_local CopyPool p(threads);
    return p;
  }
  int worker_count() const { return static_cast<int>(workers_.size()); }
  void run(int total, const std::function<void(int, int)> &body) {
    const int nw = static_cast<int>(workers_.size());
    const int participants = nw + 1;
    const int chunk = (total + participants - 1) / participants;
    for (int w = 0; w < nw; ++w) {
      const int start = (w + 1) * chunk;
      const int end = std::min(start + chunk, total);
      if (start >= total) {
        workers_[w]->mark_done();
        continue;
      }
      workers_[w]->submit(&body, start, end);
    }
    const int main_end = std::min(chunk, total);
    if (main_end > 0)
      body(0, main_end);
    for (int w = 0; w < nw; ++w)
      workers_[w]->wait();
  }

private:
  struct Worker {
    std::thread th;
    std::mutex m;
    std::condition_variable cv;
    std::condition_variable done_cv;
    const std::function<void(int, int)> *fn = nullptr;
    int start = 0, end = 0;
    bool has_job = false;
    bool done = true;
    bool stop = false;
    Worker() { th = std::thread([this] { loop(); }); }
    ~Worker() {
      {
        std::lock_guard<std::mutex> lk(m);
        stop = true;
      }
      cv.notify_one();
      if (th.joinable())
        th.join();
    }
    void submit(const std::function<void(int, int)> *f, int s, int e) {
      {
        std::lock_guard<std::mutex> lk(m);
        fn = f;
        start = s;
        end = e;
        has_job = true;
        done = false;
      }
      cv.notify_one();
    }
    void mark_done() {
      std::lock_guard<std::mutex> lk(m);
      done = true;
    }
    void wait() {
      std::unique_lock<std::mutex> lk(m);
      done_cv.wait(lk, [this] { return done; });
    }
    void loop() {
      for (;;) {
        const std::function<void(int, int)> *f = nullptr;
        int s = 0, e = 0;
        {
          std::unique_lock<std::mutex> lk(m);
          cv.wait(lk, [this] { return has_job || stop; });
          if (stop && !has_job)
            return;
          f = fn;
          s = start;
          e = end;
          has_job = false;
        }
        try {
          (*f)(s, e);
        } catch (...) {
        }
        {
          std::lock_guard<std::mutex> lk(m);
          done = true;
        }
        done_cv.notify_one();
      }
    }
  };
  std::vector<std::unique_ptr<Worker>> workers_;
  explicit CopyPool(int total_threads) {
    for (int i = 0; i < total_threads - 1; ++i)
      workers_.push_back(std::make_unique<Worker>());
  }
};

constexpr int kPrefetchDist = 16;

} // namespace

// ============================================================================
// scatter_to_cpu: staging buf → strided CPU dst.
//   LAYERFIRST (cpu_phys_contig): merge consecutive block_ids into one copy.
//   BLOCKFIRST: per-block parallel scatter with NT stores via thread pool.
// ============================================================================
void scatter_to_cpu(const void *staging_buf, int64_t *cpu_ptr_int64,
                    int64_t *cpu_block_ids, int num_blocks,
                    int64_t cpu_block_stride_int64,
                    int64_t cpu_startoff_inside_chunks_int64,
                    int64_t chunk_size_in_bytes, int layer_idx, int kv_idx,
                    int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
                    int start_layer_id, bool cpu_phys_contig,
                    int gather_threads, bool gather_nt) {
  int64_t *cpu_base = cpu_ptr_int64 +
      (layer_idx + start_layer_id) * cpu_layer_stride_int64 +
      kv_idx * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
  const char *src = static_cast<const char *>(staging_buf);

  if (cpu_phys_contig) {
    // LAYERFIRST: merge consecutive block_ids into one copy. NT store avoids cache pollution.
    const CopyImpl impl = pick_copy_impl(gather_nt);
    int64_t k = 0;
    while (k < num_blocks) {
      int64_t run_start = k;
      while (k + 1 < num_blocks &&
             cpu_block_ids[k + 1] == cpu_block_ids[k] + 1) {
        ++k;
      }
      int64_t nr_blocks = k - run_start + 1;
      int64_t cb = cpu_block_ids[run_start];
      int64_t run_bytes = nr_blocks * chunk_size_in_bytes;
      nt_copy_block((char *)(cpu_base + cb * cpu_block_stride_int64),
                  src + (int64_t)run_start * chunk_size_in_bytes,
                  (size_t)run_bytes, impl);
      ++k;
    }
    sfence_if_nt(impl);
    return;
  }

  // BLOCKFIRST: per-block parallel NT scatter.
  const CopyImpl impl = pick_copy_impl(gather_nt);
  auto body = [&](int begin, int end) {
    for (int kk = begin; kk < end; ++kk) {
      if (kk + kPrefetchDist < end)
        __builtin_prefetch(
            cpu_base + cpu_block_ids[kk + kPrefetchDist] * cpu_block_stride_int64,
            1 /*write*/, 0 /*non-temporal*/);
      int64_t cb = cpu_block_ids[kk];
      nt_copy_block((char *)(cpu_base + cb * cpu_block_stride_int64),
                  src + (int64_t)kk * chunk_size_in_bytes,
                  (size_t)chunk_size_in_bytes, impl);
    }
    sfence_if_nt(impl);
  };
  if (gather_threads <= 0 || num_blocks < 8 ||
      CopyPool::instance(gather_threads).worker_count() == 0) {
    body(0, num_blocks);
  } else {
    CopyPool::instance(gather_threads).run(num_blocks, body);
  }
}

// ============================================================================
// gather_from_cpu: strided CPU src → contiguous staging buf.
//   H2D counterpart of scatter_to_cpu. Same LAYERFIRST/BLOCKFIRST logic.
// ============================================================================
void gather_from_cpu(void *staging_buf, const int64_t *cpu_ptr_int64,
                     const int64_t *cpu_block_ids, int num_blocks,
                     int64_t cpu_block_stride_int64,
                     int64_t cpu_startoff_inside_chunks_int64,
                     int64_t chunk_size_in_bytes, int layer_idx, int kv_idx,
                     int64_t cpu_kv_stride_int64, int64_t cpu_layer_stride_int64,
                     int start_layer_id, bool cpu_phys_contig,
                     int gather_threads, bool gather_nt) {
  const int64_t *cpu_base = cpu_ptr_int64 +
      (layer_idx + start_layer_id) * cpu_layer_stride_int64 +
      kv_idx * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
  char *dst = static_cast<char *>(staging_buf);

  if (cpu_phys_contig) {
    // LAYERFIRST: merge consecutive block_ids into one copy. NT store on staging buf.
    const CopyImpl impl = pick_copy_impl(gather_nt);
    int64_t k = 0;
    while (k < num_blocks) {
      int64_t run_start = k;
      while (k + 1 < num_blocks &&
             cpu_block_ids[k + 1] == cpu_block_ids[k] + 1) {
        ++k;
      }
      int64_t nr_blocks = k - run_start + 1;
      int64_t cb = cpu_block_ids[run_start];
      int64_t run_bytes = nr_blocks * chunk_size_in_bytes;
      nt_copy_block(dst + (int64_t)run_start * chunk_size_in_bytes,
                  (const char *)(cpu_base + cb * cpu_block_stride_int64),
                  (size_t)run_bytes, impl);
      ++k;
    }
    sfence_if_nt(impl);
    return;
  }

  // BLOCKFIRST: per-block parallel NT gather.
  const CopyImpl impl = pick_copy_impl(gather_nt);
  auto body = [&](int begin, int end) {
    for (int kk = begin; kk < end; ++kk) {
      if (kk + kPrefetchDist < end)
        __builtin_prefetch(
            cpu_base + cpu_block_ids[kk + kPrefetchDist] * cpu_block_stride_int64,
            0 /*read*/, 0 /*non-temporal*/);
      int64_t cb = cpu_block_ids[kk];
      nt_copy_block(dst + (int64_t)kk * chunk_size_in_bytes,
                  (const char *)(cpu_base + cb * cpu_block_stride_int64),
                  (size_t)chunk_size_in_bytes, impl);
    }
    sfence_if_nt(impl);
  };
  if (gather_threads <= 0 || num_blocks < 8 ||
      CopyPool::instance(gather_threads).worker_count() == 0) {
    body(0, num_blocks);
  } else {
    CopyPool::instance(gather_threads).run(num_blocks, body);
  }
}

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
    const CEAnalysis &ce_analysis, const CETransferConfig &ce_config) {

  // ---- memcpy2d branch (bidirectional) ----
  // cudaMemcpy2DAsync per segment; fast NVIDIA, slow/unsupported elsewhere.
  // Scattered blocks: 50x slower -> fall through to staging.
  if (ce_config.enable_memcpy2d && ce_analysis.num_segments <= ce_config.segment_threshold) {
    cudaMemcpyKind kind = is_host_to_device ? cudaMemcpyHostToDevice
                                            : cudaMemcpyDeviceToHost;
    const int64_t total_iters = (int64_t)num_layers * kv_dim;
    for (int64_t it = 0; it < total_iters; ++it) {
      int i = (int)(it / kv_dim);
      int j = (int)(it % kv_dim);
      for (const auto &seg : ce_analysis.segments) {
        // GPU block stride = pointer diff of two adjacent blocks
        int64_t *gpu_ptr_first = ptr_at<Type>(gpu_tensor_handler,
                                              i + start_layer_id, j,
                                              gpu_block_ids[seg.start_k]);
        int64_t *gpu_ptr_next = ptr_at<Type>(gpu_tensor_handler,
                                             i + start_layer_id, j,
                                             gpu_block_ids[seg.start_k] + 1);
        size_t gpu_pitch = (size_t)((char *)gpu_ptr_next - (char *)gpu_ptr_first);
        void *gpu_ptr = (char *)gpu_ptr_first +
            gpu_startoff_inside_chunks_int64 * sizeof(int64_t);
        // CPU pointer: strided, first block in segment.
        int64_t *cpu_base = cpu_ptr_int64 +
            (i + start_layer_id) * cpu_layer_stride_int64 +
            j * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
        void *cpu_ptr = cpu_base + cpu_block_ids[seg.start_k] * cpu_block_stride_int64;
        size_t cpu_pitch = (size_t)cpu_block_stride_int64 * sizeof(int64_t);

        // Select src/dst/pitch by direction.
        void *dst = is_host_to_device ? gpu_ptr : cpu_ptr;
        void *src = is_host_to_device ? cpu_ptr : gpu_ptr;
        size_t dpitch = is_host_to_device ? gpu_pitch : cpu_pitch;
        size_t spitch = is_host_to_device ? cpu_pitch : gpu_pitch;

        cudaMemcpy2DAsync(dst, dpitch, src, spitch,
                          chunk_size_in_bytes, seg.nr_blocks, kind, stream);
        FLEXKV_GPU_CPU_TRANSFER(is_host_to_device, chunk_size_in_bytes * seg.nr_blocks);
      }
    }
    cudaStreamSynchronize(stream);
    return;
  }

  // ---- staging buffer + CPU scatter/gather (D2H ping-pong) ----
  size_t layer_buf_size = (size_t)num_blocks * chunk_size_in_bytes;
  bool need_pingpong = !is_host_to_device;

  void *host_base = get_cached_host_buffer(need_pingpong ? layer_buf_size * 2
                                                       : layer_buf_size);
  void *host_bufs[2] = {
      host_base,
      need_pingpong ? (char *)host_base + layer_buf_size : nullptr};
  // Cached ping-pong events (per-device)
  bool events_created = false;
  cudaEvent_t *pingpong_events = get_cached_event_pair(need_pingpong, events_created);

  const int64_t total_iters = (int64_t)num_layers * kv_dim;
  for (int64_t it = 0; it < total_iters; ++it) {
    int i = (int)(it / kv_dim);
    int j = (int)(it % kv_dim);
    int idx = need_pingpong ? (int)(it & 1) : 0;
    int prev_idx = idx ^ 1;
    void *buf = host_bufs[idx];

    if (!is_host_to_device) {
      // ---- D2H: all segments into staging ----
      int64_t seg_offset = 0;
      for (const auto &seg : ce_analysis.segments) {
        int64_t seg_size = (int64_t)seg.nr_blocks * chunk_size_in_bytes;
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
      if (need_pingpong) {
        cudaEventRecord(pingpong_events[idx], stream);
        // CPU scatter previous layer
        if (it >= 1) {
          cudaEventSynchronize(pingpong_events[prev_idx]);
          int pi = (int)((it - 1) / kv_dim);
          int pj = (int)((it - 1) % kv_dim);
          // scatter from host_bufs[prev_idx] to strided dst
          scatter_to_cpu(host_bufs[prev_idx], cpu_ptr_int64,
                         cpu_block_ids, num_blocks,
                         cpu_block_stride_int64,
                         cpu_startoff_inside_chunks_int64,
                         chunk_size_in_bytes, pi, pj,
                         cpu_kv_stride_int64, cpu_layer_stride_int64,
                         start_layer_id, ce_analysis.cpu_phys_contig,
                         ce_config.gather_threads, ce_config.gather_nt);
        }
      } else {
        cudaStreamSynchronize(stream);
        // scatter current layer
        scatter_to_cpu(buf, cpu_ptr_int64,
                       cpu_block_ids, num_blocks,
                       cpu_block_stride_int64,
                       cpu_startoff_inside_chunks_int64,
                       chunk_size_in_bytes, i, j,
                       cpu_kv_stride_int64, cpu_layer_stride_int64,
                       start_layer_id, ce_analysis.cpu_phys_contig,
                       ce_config.gather_threads, ce_config.gather_nt);
      }
    } else {
      // ---- H2D: gather all -> H2D all -> drain (no ping-pong) ----
      gather_from_cpu(buf, cpu_ptr_int64,
                      cpu_block_ids, num_blocks,
                      cpu_block_stride_int64,
                      cpu_startoff_inside_chunks_int64,
                      chunk_size_in_bytes, i, j,
                      cpu_kv_stride_int64, cpu_layer_stride_int64,
                      start_layer_id, ce_analysis.cpu_phys_contig,
                      ce_config.gather_threads, ce_config.gather_nt);
      // H2D segments from staging
      int64_t off = 0;
      for (const auto &seg : ce_analysis.segments) {
        int64_t seg_size = (int64_t)seg.nr_blocks * chunk_size_in_bytes;
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
      // Drain so next iter's gather can overwrite buf safely.
      cudaStreamSynchronize(stream);
    }
  }
  // Drain last ping-pong slot (D2H)
  if (!is_host_to_device && need_pingpong && total_iters >= 1) {
    int64_t last = total_iters - 1;
    int last_idx = (int)(last & 1);
    cudaEventSynchronize(pingpong_events[last_idx]);
    int li = (int)(last / kv_dim);
    int lj = (int)(last % kv_dim);
    scatter_to_cpu(host_bufs[last_idx], cpu_ptr_int64,
                   cpu_block_ids, num_blocks,
                   cpu_block_stride_int64,
                   cpu_startoff_inside_chunks_int64,
                   chunk_size_in_bytes, li, lj,
                   cpu_kv_stride_int64, cpu_layer_stride_int64,
                   start_layer_id, ce_analysis.cpu_phys_contig,
                   ce_config.gather_threads, ce_config.gather_nt);
  }
  // Events cached, not destroyed (all work already synced).
}

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
    const CEAnalysis &ce_analysis, const CETransferConfig &ce_config) {
  TORCH_CHECK(chunk_size_in_bytes % sizeof(int64_t) == 0,
              "GATHER_SCATTER requires chunk_size_in_bytes % 8 == 0 "
              "(its index_select/index_copy_ view needs an int64-aligned chunk)");
  const int64_t elems_per_block = chunk_size_in_bytes / sizeof(int64_t);
  // staging buffer size = num_blocks * chunk_size
  const size_t buf_bytes = (size_t)num_blocks * (size_t)chunk_size_in_bytes;

  // Bind ATen to our cuda stream
  int cur_dev = 0;
  cudaGetDevice(&cur_dev);
  c10::cuda::CUDAStream aten_stream =
      c10::cuda::getStreamFromExternal(stream, cur_dev);
  c10::cuda::CUDAStreamGuard stream_guard(aten_stream);

  // Find max GPU block index for tensor views
  int64_t max_gpu_id = 0;
  for (int k = 0; k < num_blocks; ++k) {
    if (gpu_block_ids[k] > max_gpu_id) max_gpu_id = gpu_block_ids[k];
  }

  // Pin ATen views to cur_dev (thread iterates GPUs)
  auto i64_cuda = at::TensorOptions().dtype(at::kLong)
                      .device(at::kCUDA, cur_dev);

  // Block ids -> GPU (cached; safe to return without draining)
  const size_t ids_bytes = (size_t)num_blocks * sizeof(int64_t);
  void *gpu_ids_raw = nullptr;
  at::Tensor gpu_ids_cuda;
  void *dst_ids_raw = nullptr;
  at::Tensor dst_ids_cuda;
  if (!ce_analysis.gpu_log_contig || !ce_analysis.gpu_phys_contig) {
    gpu_ids_raw = get_cached_device_buffer(ids_bytes);
    if (!gpu_ids_raw) { ce_transfer_per_block<Type>(num_blocks, start_layer_id, num_layers, kv_dim, gpu_block_ids, gpu_tensor_handler, gpu_startoff_inside_chunks_int64, cpu_block_ids, cpu_ptr_int64, cpu_kv_stride_int64, cpu_layer_stride_int64, cpu_block_stride_int64, cpu_startoff_inside_chunks_int64, chunk_size_in_bytes, stream, is_host_to_device); return; }
    cudaMemcpyAsync(gpu_ids_raw, gpu_block_ids, ids_bytes,
                    cudaMemcpyHostToDevice, stream);
    gpu_ids_cuda = at::from_blob(gpu_ids_raw, {num_blocks}, i64_cuda);

    if (is_host_to_device) {
      dst_ids_raw = get_cached_device_buffer(ids_bytes, 1);  // slot=1
      if (!dst_ids_raw) { ce_transfer_per_block<Type>(num_blocks, start_layer_id, num_layers, kv_dim, gpu_block_ids, gpu_tensor_handler, gpu_startoff_inside_chunks_int64, cpu_block_ids, cpu_ptr_int64, cpu_kv_stride_int64, cpu_layer_stride_int64, cpu_block_stride_int64, cpu_startoff_inside_chunks_int64, chunk_size_in_bytes, stream, is_host_to_device); return; }
      cudaMemcpyAsync(dst_ids_raw, gpu_block_ids, ids_bytes,
                      cudaMemcpyHostToDevice, stream);
      dst_ids_cuda = at::from_blob(dst_ids_raw, {num_blocks}, i64_cuda);
    }
  }

  // Raw cudaMalloc + from_blob (ATen cache unsafe under external stream).
  // Device buffer when GPU blocks non-contiguous.
  bool need_dev_buf = !ce_analysis.gpu_log_contig || !ce_analysis.gpu_phys_contig;
  void *dev_raw[2] = {nullptr, nullptr};
  at::Tensor dev_buf[2];
  if (need_dev_buf) {
    bool need_two = !is_host_to_device;  // D2H ping-pong only
    size_t dev_alloc = need_two ? buf_bytes * 2 : buf_bytes;
    void *dev_base = get_cached_device_buffer(dev_alloc, 2);  // slot=2
    if (!dev_base) { ce_transfer_per_block<Type>(num_blocks, start_layer_id, num_layers, kv_dim, gpu_block_ids, gpu_tensor_handler, gpu_startoff_inside_chunks_int64, cpu_block_ids, cpu_ptr_int64, cpu_kv_stride_int64, cpu_layer_stride_int64, cpu_block_stride_int64, cpu_startoff_inside_chunks_int64, chunk_size_in_bytes, stream, is_host_to_device); return; }
    dev_raw[0] = dev_base;
    dev_buf[0] = at::from_blob(dev_raw[0], {num_blocks, elems_per_block},
                               i64_cuda);
    if (need_two) {
      dev_raw[1] = (char *)dev_base + buf_bytes;
      dev_buf[1] = at::from_blob(dev_raw[1], {num_blocks, elems_per_block},
                                 i64_cuda);
    }
  }

  // Host staging: D2H always; H2D only when CPU src non-contiguous.
  bool need_host_buf =
      !is_host_to_device ||  // D2H: stage + scatter
      (is_host_to_device &&
       !(ce_analysis.cpu_log_contig && ce_analysis.cpu_phys_contig));  // H2D: gather

  // memcpy2d: no host staging; keep host_buf when scattered (segfault otherwise).
  if (ce_config.enable_memcpy2d && ce_analysis.num_segments <= ce_config.segment_threshold) {
    need_host_buf = false;
  }

  // D2H ping-pong overlaps CPU scatter with GPU D2H.
  bool need_pingpong_host = need_host_buf && !is_host_to_device;

  void *host_buf[2] = {nullptr, nullptr};
  if (need_host_buf) {
    // 2x for ping-pong (two halves), 1x otherwise.
    size_t host_alloc = need_pingpong_host ? buf_bytes * 2 : buf_bytes;
    void *host_base = get_cached_host_buffer(host_alloc);
    host_buf[0] = host_base;
    if (need_pingpong_host) {
      host_buf[1] = (char *)host_base + buf_bytes;
    }
  }

  // Cached ping-pong events (per-device)
  bool events_created = false;
  cudaEvent_t *pingpong_events = get_cached_event_pair(need_pingpong_host, events_created);

  const int64_t total_iters = (int64_t)num_layers * kv_dim;

  // memcpy2d branch: cudaMemcpy2DAsync per segment (bypasses host staging/scatter).
  if (ce_config.enable_memcpy2d && ce_analysis.num_segments <= ce_config.segment_threshold) {
    cudaMemcpyKind kind = is_host_to_device ? cudaMemcpyHostToDevice
                                            : cudaMemcpyDeviceToHost;
    for (int64_t it = 0; it < total_iters; ++it) {
      int i = (int)(it / kv_dim);
      int j = (int)(it % kv_dim);

      // GPU block stride (pitch) in bytes.
      int64_t *gpu_ptr_block0 =
          ptr_at<Type>(gpu_tensor_handler, i + start_layer_id, j, 0);
      int64_t *gpu_ptr_block1 =
          ptr_at<Type>(gpu_tensor_handler, i + start_layer_id, j, 1);
      int64_t gpu_block_stride_bytes =
          (int64_t)((char *)gpu_ptr_block1 - (char *)gpu_ptr_block0);
      int64_t *gpu_layer_kv_base =
          gpu_ptr_block0 + gpu_startoff_inside_chunks_int64;

      // CPU base for this (layer, kv).
      int64_t *cpu_base = cpu_ptr_int64 +
          (i + start_layer_id) * cpu_layer_stride_int64 +
          j * cpu_kv_stride_int64 + cpu_startoff_inside_chunks_int64;
      size_t cpu_pitch = (size_t)cpu_block_stride_int64 * sizeof(int64_t);

      bool gpu_contig = ce_analysis.gpu_log_contig && ce_analysis.gpu_phys_contig;

      if (!is_host_to_device) {
        // ---- D2H ----
        // Step 1: GPU gather (if needed)
        const int64_t *d2h_src;
        size_t dev_pitch;
        if (gpu_contig) {
          d2h_src = gpu_layer_kv_base +
                    gpu_block_ids[0] * (gpu_block_stride_bytes / sizeof(int64_t));
          dev_pitch = (size_t)gpu_block_stride_bytes;
        } else {
          at::Tensor src_view = at::from_blob(
              gpu_layer_kv_base, {max_gpu_id + 1, elems_per_block},
              {(int64_t)(gpu_block_stride_bytes / sizeof(int64_t)), 1}, i64_cuda);
          at::index_select_out(dev_buf[0], src_view, 0, gpu_ids_cuda);
          d2h_src = reinterpret_cast<int64_t *>(dev_buf[0].data_ptr());
          dev_pitch = (size_t)chunk_size_in_bytes;
        }

        // Step 2: memcpy2d dev_buf/GPU -> CPU (strided, per segment)
        for (const auto &seg : ce_analysis.segments) {
          int64_t cb = cpu_block_ids[seg.start_k];
          void *cpu_dst = cpu_base + cb * cpu_block_stride_int64;
          void *src = (char *)d2h_src +
                      (int64_t)seg.start_k * chunk_size_in_bytes;
          cudaMemcpy2DAsync(cpu_dst, cpu_pitch, src, dev_pitch,
                            chunk_size_in_bytes, (size_t)seg.nr_blocks,
                            kind, stream);
          FLEXKV_GPU_CPU_TRANSFER(false, chunk_size_in_bytes * seg.nr_blocks);
        }
        cudaStreamSynchronize(stream);
      } else {
        // ---- H2D ----
        // Step 1: memcpy2d CPU -> dev_buf/GPU (strided, per segment)
        int64_t *h2d_dst;
        size_t dev_pitch;
        if (gpu_contig) {
          h2d_dst = gpu_layer_kv_base +
                    gpu_block_ids[0] * (gpu_block_stride_bytes / sizeof(int64_t));
          dev_pitch = (size_t)gpu_block_stride_bytes;
        } else {
          h2d_dst = reinterpret_cast<int64_t *>(dev_buf[0].data_ptr());
          dev_pitch = (size_t)chunk_size_in_bytes;
        }

        for (const auto &seg : ce_analysis.segments) {
          int64_t cb = cpu_block_ids[seg.start_k];
          const int64_t *cpu_src = cpu_base + cb * cpu_block_stride_int64;
          void *dst = (char *)h2d_dst +
                      (int64_t)seg.start_k * chunk_size_in_bytes;
          cudaMemcpy2DAsync(dst, dev_pitch, cpu_src, cpu_pitch,
                            chunk_size_in_bytes, (size_t)seg.nr_blocks,
                            kind, stream);
          FLEXKV_GPU_CPU_TRANSFER(true, chunk_size_in_bytes * seg.nr_blocks);
        }
        cudaStreamSynchronize(stream);

        // Step 2: GPU scatter (if needed)
        if (!gpu_contig) {
          at::Tensor dst_view = at::from_blob(
              gpu_layer_kv_base, {max_gpu_id + 1, elems_per_block},
              {(int64_t)(gpu_block_stride_bytes / sizeof(int64_t)), 1}, i64_cuda);
          dst_view.index_copy_(0, dst_ids_cuda, dev_buf[0]);
        }
      }
    }
    cudaStreamSynchronize(stream);
    gpu_ids_cuda.reset();
    dst_ids_cuda.reset();
    dev_buf[0].reset();
    dev_buf[1].reset();
    return;
  }

  for (int64_t it = 0; it < total_iters; ++it) {
    int i = (int)(it / kv_dim);
    int j = (int)(it % kv_dim);
    int idx = pingpong_events ? (int)(it & 1) : 0;
    int prev_idx = idx ^ 1;

    // GPU block stride (pitch) in int64 elems.
    int64_t *gpu_ptr_block0 =
        ptr_at<Type>(gpu_tensor_handler, i + start_layer_id, j, 0);
    int64_t *gpu_ptr_block1 =
        ptr_at<Type>(gpu_tensor_handler, i + start_layer_id, j, 1);
    int64_t gpu_block_stride_elems =
        (int64_t)((char *)gpu_ptr_block1 - (char *)gpu_ptr_block0) /
        sizeof(int64_t);
    int64_t *gpu_layer_kv_base =
        gpu_ptr_block0 + gpu_startoff_inside_chunks_int64;

    if (!is_host_to_device) {
      // ---- D2H ----
      // Step 1: GPU gather (if src non-contig — logical or physical)
      const int64_t *d2h_src;
      if (ce_analysis.gpu_log_contig && ce_analysis.gpu_phys_contig) {
        d2h_src = gpu_layer_kv_base +
                  gpu_block_ids[0] * gpu_block_stride_elems;
      } else {
        at::Tensor src_view = at::from_blob(
            gpu_layer_kv_base, {max_gpu_id + 1, elems_per_block},
            {gpu_block_stride_elems, 1}, i64_cuda);
        at::index_select_out(dev_buf[idx], src_view, 0, gpu_ids_cuda);
        d2h_src = reinterpret_cast<int64_t *>(dev_buf[idx].data_ptr());
      }

      // Step 2: D2H into staging
      void *dst_ptr = need_host_buf ? host_buf[idx]
                                   : (void *)(cpu_ptr_int64 +
                                      (i + start_layer_id) * cpu_layer_stride_int64 +
                                      j * cpu_kv_stride_int64 +
                                      cpu_block_ids[0] * cpu_block_stride_int64 +
                                      cpu_startoff_inside_chunks_int64);
      cudaMemcpyAsync(dst_ptr, d2h_src, buf_bytes,
                      cudaMemcpyDeviceToHost, stream);
      FLEXKV_GPU_CPU_TRANSFER(false, buf_bytes);

      if (pingpong_events) {
        cudaEventRecord(pingpong_events[idx], stream);
        // Step 3: CPU scatter previous slot
        if (it >= 1) {
          cudaEventSynchronize(pingpong_events[prev_idx]);
          int pi = (int)((it - 1) / kv_dim);
          int pj = (int)((it - 1) % kv_dim);
          scatter_to_cpu(host_buf[prev_idx], cpu_ptr_int64,
                         cpu_block_ids, num_blocks,
                         cpu_block_stride_int64,
                         cpu_startoff_inside_chunks_int64,
                         chunk_size_in_bytes, pi, pj,
                         cpu_kv_stride_int64, cpu_layer_stride_int64,
                         start_layer_id, ce_analysis.cpu_phys_contig,
                         ce_config.gather_threads, ce_config.gather_nt);
        }
      } else if (need_host_buf) {
        cudaStreamSynchronize(stream);
        // scatter current
        scatter_to_cpu(host_buf[idx], cpu_ptr_int64,
                       cpu_block_ids, num_blocks,
                       cpu_block_stride_int64,
                       cpu_startoff_inside_chunks_int64,
                       chunk_size_in_bytes, i, j,
                       cpu_kv_stride_int64, cpu_layer_stride_int64,
                       start_layer_id, ce_analysis.cpu_phys_contig,
                       ce_config.gather_threads, ce_config.gather_nt);
      }
    } else {
      // ---- H2D ----
      // Step 1: CPU gather (if CPU src non-contig)
      const void *h2d_src;
      if (ce_analysis.cpu_log_contig && ce_analysis.cpu_phys_contig) {
        // CPU src is contiguous — direct from cpu_ptr, no staging needed.
        h2d_src = cpu_ptr_int64 +
                  (i + start_layer_id) * cpu_layer_stride_int64 +
                  j * cpu_kv_stride_int64 +
                  cpu_block_ids[0] * cpu_block_stride_int64 +
                  cpu_startoff_inside_chunks_int64;
      } else {
        // gather into staging (H2D: no ping-pong, idx always 0)
        gather_from_cpu(host_buf[idx], cpu_ptr_int64,
                        cpu_block_ids, num_blocks,
                        cpu_block_stride_int64,
                        cpu_startoff_inside_chunks_int64,
                        chunk_size_in_bytes, i, j,
                        cpu_kv_stride_int64, cpu_layer_stride_int64,
                        start_layer_id, ce_analysis.cpu_phys_contig,
                        ce_config.gather_threads, ce_config.gather_nt);
        h2d_src = host_buf[idx];
      }

      // Step 2: H2D — GPU dst contiguity (logical + physical)
      void *h2d_dst;
      if (ce_analysis.gpu_log_contig && ce_analysis.gpu_phys_contig) {
        h2d_dst = gpu_layer_kv_base +
                  gpu_block_ids[0] * gpu_block_stride_elems;
      } else {
        h2d_dst = dev_buf[idx].data_ptr();
      }
      cudaMemcpyAsync(h2d_dst, h2d_src, buf_bytes,
                      cudaMemcpyHostToDevice, stream);
      FLEXKV_GPU_CPU_TRANSFER(true, buf_bytes);

      // Step 3: GPU scatter (if GPU dst non-contig — logical or physical)
      if (!ce_analysis.gpu_log_contig || !ce_analysis.gpu_phys_contig) {
        at::Tensor dst_view = at::from_blob(
            gpu_layer_kv_base, {max_gpu_id + 1, elems_per_block},
            {gpu_block_stride_elems, 1}, i64_cuda);
        dst_view.index_copy_(0, dst_ids_cuda, dev_buf[idx]);
      }

      // H2D: no ping-pong (idx=0); drain so staging is safe to reuse.
      if (need_host_buf || need_dev_buf) {
        cudaStreamSynchronize(stream);
      }
    }
  }

  // Drain last D2H scatter
  if (!is_host_to_device && pingpong_events && total_iters >= 1) {
    int64_t last = total_iters - 1;
    int last_idx = (int)(last & 1);
    cudaEventSynchronize(pingpong_events[last_idx]);
    int li = (int)(last / kv_dim);
    int lj = (int)(last % kv_dim);
    scatter_to_cpu(host_buf[last_idx], cpu_ptr_int64,
                   cpu_block_ids, num_blocks,
                   cpu_block_stride_int64,
                   cpu_startoff_inside_chunks_int64,
                   chunk_size_in_bytes, li, lj,
                   cpu_kv_stride_int64, cpu_layer_stride_int64,
                   start_layer_id, ce_analysis.cpu_phys_contig,
                   ce_config.gather_threads, ce_config.gather_nt);
  }

  // Drain last H2D
  if (is_host_to_device && pingpong_events && total_iters >= 1) {
    int64_t last = total_iters - 1;
    int last_idx = (int)(last & 1);
    cudaEventSynchronize(pingpong_events[last_idx]);
  }

  // Final drain (id tensors freed; cached bufs survive)
  cudaStreamSynchronize(stream);

  // Events cached, not destroyed here (all work already synced).

  // Release from_blob views (underlying buffers cached).
  gpu_ids_cuda.reset();
  dst_ids_cuda.reset();
  dev_buf[0].reset();
  dev_buf[1].reset();
}

// GATHER_DIRECT: BF non-sharded (MLA/MHA). D2D transpose then per-segment copy.

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
    const CEAnalysis &ce_analysis, const CETransferConfig &ce_config) {
  TORCH_CHECK(chunk_size_in_bytes % sizeof(int64_t) == 0,
              "GATHER_DIRECT requires chunk_size % 8 == 0");
  const int64_t elems_per_block = chunk_size_in_bytes / sizeof(int64_t);
  const size_t buf_bytes = (size_t)num_blocks * (size_t)chunk_size_in_bytes;
  const int64_t total_iters = (int64_t)num_layers * kv_dim;
  // Device staging: [num_blocks, total_iters, elems_per_block] (BLOCKFIRST)
  const size_t total_dev_bytes = buf_bytes * (size_t)total_iters;

  // Bind ATen to our cuda stream
  int cur_dev = 0;
  cudaGetDevice(&cur_dev);
  c10::cuda::CUDAStream aten_stream =
      c10::cuda::getStreamFromExternal(stream, cur_dev);
  c10::cuda::CUDAStreamGuard stream_guard(aten_stream);
  auto i64_cuda = at::TensorOptions().dtype(at::kLong)
                      .device(at::kCUDA, cur_dev);

  // Find max GPU block index for tensor views
  int64_t max_gpu_id = 0;
  for (int k = 0; k < num_blocks; ++k)
    if (gpu_block_ids[k] > max_gpu_id) max_gpu_id = gpu_block_ids[k];

  // block ids -> GPU (index_select when !gpu_log_contig)
  const size_t ids_bytes = (size_t)num_blocks * sizeof(int64_t);
  void *gpu_ids_raw = nullptr;
  at::Tensor gpu_ids_cuda;
  if (!ce_analysis.gpu_log_contig) {
    gpu_ids_raw = get_cached_device_buffer(ids_bytes);
    if (!gpu_ids_raw) { ce_transfer_per_block<Type>(num_blocks, start_layer_id, num_layers, kv_dim, gpu_block_ids, gpu_tensor_handler, gpu_startoff_inside_chunks_int64, cpu_block_ids, cpu_ptr_int64, cpu_kv_stride_int64, cpu_layer_stride_int64, cpu_block_stride_int64, cpu_startoff_inside_chunks_int64, chunk_size_in_bytes, stream, is_host_to_device); return; }
    cudaMemcpyAsync(gpu_ids_raw, gpu_block_ids, ids_bytes,
                    cudaMemcpyHostToDevice, stream);
    gpu_ids_cuda = at::from_blob(gpu_ids_raw, {num_blocks}, i64_cuda);
  }

  // Device staging buffer: [num_blocks, total_iters, elems_per_block] contiguous
  void *dev_staging = get_cached_device_buffer(total_dev_bytes, 2);
  if (!dev_staging) { ce_transfer_per_block<Type>(num_blocks, start_layer_id, num_layers, kv_dim, gpu_block_ids, gpu_tensor_handler, gpu_startoff_inside_chunks_int64, cpu_block_ids, cpu_ptr_int64, cpu_kv_stride_int64, cpu_layer_stride_int64, cpu_block_stride_int64, cpu_startoff_inside_chunks_int64, chunk_size_in_bytes, stream, is_host_to_device); return; }
  at::Tensor dev_staging_view = at::from_blob(
      dev_staging, {num_blocks, total_iters, elems_per_block}, i64_cuda);

  if (!is_host_to_device) {
    // ---- D2H ----
    // Step 1: D2D transpose — gather each (layer, kv) into staging
    for (int64_t it = 0; it < total_iters; ++it) {
      int i = (int)(it / kv_dim);
      int j = (int)(it % kv_dim);
      int64_t *gpu_ptr_block0 =
          ptr_at<Type>(gpu_tensor_handler, i + start_layer_id, j, 0);
      int64_t *gpu_ptr_block1 =
          ptr_at<Type>(gpu_tensor_handler, i + start_layer_id, j, 1);
      // GPU block stride (pitch) in int64 elems; sharded D2H needs explicit stride.
      int64_t gpu_block_stride_elems =
          (int64_t)((char *)gpu_ptr_block1 - (char *)gpu_ptr_block0) /
          sizeof(int64_t);
      // Add gpu_startoff to land on this rank's shard.
      int64_t *gpu_layer_kv_base =
          gpu_ptr_block0 + gpu_startoff_inside_chunks_int64;
      at::Tensor src_view = at::from_blob(
          gpu_layer_kv_base, {max_gpu_id + 1, elems_per_block},
          {gpu_block_stride_elems, 1}, i64_cuda);
      // Gather into staging[:, it, :] via index_select_out (no temp).
      auto staging_slice = dev_staging_view.select(1, it);
      if (ce_analysis.gpu_log_contig) {
        staging_slice.copy_(src_view.narrow(0, gpu_block_ids[0], num_blocks));
      } else {
        at::index_select_out(staging_slice, src_view, 0, gpu_ids_cuda);
      }
    }

    // Step 2: D2H per-segment (staging layout matches CPU BLOCKFIRST).
    int64_t block_bytes = total_iters * chunk_size_in_bytes;
    // Full-block (all layers at once): contiguous copy; else per-block scatter.
    bool full_block = (block_bytes == cpu_block_stride_int64 * sizeof(int64_t));
    if (full_block) {
      for (const auto &seg : ce_analysis.segments) {
        int64_t seg_start_block = cpu_block_ids[seg.start_k];
        int64_t seg_bytes = (int64_t)seg.nr_blocks * block_bytes;
        int64_t *cpu_dst = cpu_ptr_int64 +
            (seg_start_block * cpu_block_stride_int64) +
            cpu_startoff_inside_chunks_int64;
        void *src = (char *)dev_staging +
            (int64_t)seg.start_k * total_iters * chunk_size_in_bytes;
        cudaMemcpyAsync(cpu_dst, src, seg_bytes,
                        cudaMemcpyDeviceToHost, stream);
        FLEXKV_GPU_CPU_TRANSFER(false, seg_bytes);
      }
    } else {
      // Per-layer batch (layer_parallel): L/N layers contiguous per block.
      // memcpy2d: 2D copy (fast NVIDIA); else per-block cudaMemcpy (portable).
      size_t width = (size_t)total_iters * chunk_size_in_bytes;
      if (ce_config.enable_memcpy2d) {
        size_t spitch = (size_t)cpu_block_stride_int64 * sizeof(int64_t);
        for (const auto &seg : ce_analysis.segments) {
          int64_t seg_start_block = cpu_block_ids[seg.start_k];
          int64_t *cpu_dst = cpu_ptr_int64 +
              seg_start_block * cpu_block_stride_int64 +
              start_layer_id * cpu_layer_stride_int64 +
              cpu_startoff_inside_chunks_int64;
          void *src = (char *)dev_staging +
              (int64_t)seg.start_k * total_iters * chunk_size_in_bytes;
          cudaMemcpy2DAsync(cpu_dst, spitch, src, width,
                            width, (size_t)seg.nr_blocks,
                            cudaMemcpyDeviceToHost, stream);
          FLEXKV_GPU_CPU_TRANSFER(false, width * seg.nr_blocks);
        }
      } else {
        for (int b = 0; b < num_blocks; ++b) {
          int64_t cb = cpu_block_ids[b];
          int64_t *cpu_dst = cpu_ptr_int64 +
              cb * cpu_block_stride_int64 +
              start_layer_id * cpu_layer_stride_int64 +
              cpu_startoff_inside_chunks_int64;
          void *src = (char *)dev_staging +
              (int64_t)b * total_iters * chunk_size_in_bytes;
          cudaMemcpyAsync(cpu_dst, src, width,
                          cudaMemcpyDeviceToHost, stream);
          FLEXKV_GPU_CPU_TRANSFER(false, width);
        }
      }
    }
    cudaStreamSynchronize(stream);

  } else {
    // ---- H2D ----
    // Step 1: H2D per-segment CPU BLOCKFIRST -> dev_staging (reverse of D2H).
    int64_t block_bytes = total_iters * chunk_size_in_bytes;
    // Fast contiguous path only when block_bytes == cpu_block_stride; else per-block.
    bool full_block = (block_bytes == cpu_block_stride_int64 * sizeof(int64_t));
    if (full_block) {
      for (const auto &seg : ce_analysis.segments) {
        int64_t seg_start_block = cpu_block_ids[seg.start_k];
        int64_t seg_bytes = (int64_t)seg.nr_blocks * block_bytes;
        int64_t *cpu_src = cpu_ptr_int64 +
            (seg_start_block * cpu_block_stride_int64) +
            cpu_startoff_inside_chunks_int64;
        void *dst = (char *)dev_staging +
            (int64_t)seg.start_k * total_iters * chunk_size_in_bytes;
        cudaMemcpyAsync(dst, cpu_src, seg_bytes,
                        cudaMemcpyHostToDevice, stream);
        FLEXKV_GPU_CPU_TRANSFER(true, seg_bytes);
      }
    } else {
      // Per-layer batch (layer_parallel): symmetric to D2H.
      size_t width = (size_t)total_iters * chunk_size_in_bytes;
      if (ce_config.enable_memcpy2d) {
        size_t spitch = (size_t)cpu_block_stride_int64 * sizeof(int64_t);
        for (const auto &seg : ce_analysis.segments) {
          int64_t seg_start_block = cpu_block_ids[seg.start_k];
          const int64_t *cpu_src = cpu_ptr_int64 +
              seg_start_block * cpu_block_stride_int64 +
              start_layer_id * cpu_layer_stride_int64 +
              cpu_startoff_inside_chunks_int64;
          void *dst = (char *)dev_staging +
              (int64_t)seg.start_k * total_iters * chunk_size_in_bytes;
          cudaMemcpy2DAsync(dst, width, cpu_src, spitch,
                            width, (size_t)seg.nr_blocks,
                            cudaMemcpyHostToDevice, stream);
          FLEXKV_GPU_CPU_TRANSFER(true, width * seg.nr_blocks);
        }
      } else {
        for (int b = 0; b < num_blocks; ++b) {
          int64_t cb = cpu_block_ids[b];
          const int64_t *cpu_src = cpu_ptr_int64 +
              cb * cpu_block_stride_int64 +
              start_layer_id * cpu_layer_stride_int64 +
              cpu_startoff_inside_chunks_int64;
          void *dst = (char *)dev_staging +
              (int64_t)b * total_iters * chunk_size_in_bytes;
          cudaMemcpyAsync(dst, cpu_src, width,
                          cudaMemcpyHostToDevice, stream);
          FLEXKV_GPU_CPU_TRANSFER(true, width);
        }
      }
    }

    // Step 2: D2D reverse transpose (staging -> GPU LAYERFIRST).
    for (int64_t it = 0; it < total_iters; ++it) {
      int i = (int)(it / kv_dim);
      int j = (int)(it % kv_dim);
      int64_t *gpu_ptr_block0 =
          ptr_at<Type>(gpu_tensor_handler, i + start_layer_id, j, 0);
      int64_t *gpu_ptr_block1 =
          ptr_at<Type>(gpu_tensor_handler, i + start_layer_id, j, 1);
      // GPU block stride (pitch) in int64 elements (see D2H branch).
      int64_t gpu_block_stride_elems =
          (int64_t)((char *)gpu_ptr_block1 - (char *)gpu_ptr_block0) /
          sizeof(int64_t);
      // Add gpu_startoff to land on this rank's shard.
      int64_t *gpu_layer_kv_base =
          gpu_ptr_block0 + gpu_startoff_inside_chunks_int64;
      at::Tensor dst_view = at::from_blob(
          gpu_layer_kv_base, {max_gpu_id + 1, elems_per_block},
          {gpu_block_stride_elems, 1}, i64_cuda);
      at::Tensor src_slice = dev_staging_view.select(1, it);
      if (ce_analysis.gpu_log_contig) {
        dst_view.narrow(0, gpu_block_ids[0], num_blocks).copy_(src_slice);
      } else {
        dst_view.index_copy_(0, gpu_ids_cuda, src_slice);
      }
    }
    cudaStreamSynchronize(stream);
  }

  // Release from_blob views (buffers cached).
  gpu_ids_cuda.reset();
  dev_staging_view.reset();
}

// Explicit template instantiations (NOSTG / STG signature groups).

#define FLEXKV_INST_NOSTG(FN, BK)                                            \
  template void FN<BackendType::BK>(                                         \
      int, int, int, int, int64_t *, GTensorHandler, int64_t,                \
      int64_t *, int64_t *, int64_t, int64_t, int64_t, int64_t, int64_t,     \
      cudaStream_t, bool);

#define FLEXKV_INST_STG(FN, BK)                                              \
  template void FN<BackendType::BK>(                                         \
      int, int, int, int, int64_t *, GTensorHandler, int64_t,                \
      int64_t *, int64_t *, int64_t, int64_t, int64_t, int64_t, int64_t,     \
      cudaStream_t, bool, const CEAnalysis &, const CETransferConfig &);

#define FLEXKV_INST_STG_SYNC(FN, BK)                                         \
  template void FN<BackendType::BK>(                                         \
      int, int, int, int, int64_t *, GTensorHandler, int64_t,                \
      int64_t *, int64_t *, int64_t, int64_t, int64_t, int64_t, int64_t,     \
      cudaStream_t, bool, const CEAnalysis &, const CETransferConfig &, bool);

#define FLEXKV_INST_ALL_BACKENDS(MACRO, FN)                                  \
  MACRO(FN, VLLM) MACRO(FN, TRTLLM) MACRO(FN, SGLANG)

FLEXKV_INST_ALL_BACKENDS(FLEXKV_INST_NOSTG, ce_transfer_per_block)
FLEXKV_INST_ALL_BACKENDS(FLEXKV_INST_NOSTG, ce_transfer_contig_direct)
FLEXKV_INST_ALL_BACKENDS(FLEXKV_INST_STG, ce_transfer_segment_direct)
FLEXKV_INST_ALL_BACKENDS(FLEXKV_INST_STG, ce_transfer_segment_scatter)
FLEXKV_INST_ALL_BACKENDS(FLEXKV_INST_STG, ce_transfer_gather_scatter)
FLEXKV_INST_ALL_BACKENDS(FLEXKV_INST_STG, ce_transfer_gather_direct)

#undef FLEXKV_INST_NOSTG
#undef FLEXKV_INST_STG
#undef FLEXKV_INST_STG_SYNC
#undef FLEXKV_INST_ALL_BACKENDS

} // namespace flexkv