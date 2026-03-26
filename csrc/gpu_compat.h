/*
 * gpu_compat.h — GPU runtime compatibility header for FlexKV.
 * Provides a thin abstraction over CUDA / MUSA runtime APIs used in bindings.cpp.
 *
 * Usage:
 *   #include "gpu_compat.h"
 *   ... use flexkv::gpu_stream_t, flexkv::gpu_get_last_error(), etc.
 */
#pragma once

#ifdef FLEXKV_BACKEND_MUSA
// ─── MUSA backend ────────────────────────────────────────────────────────────
#include <musa_runtime.h>
#include <ATen/musa/MUSAContext.h>
namespace flexkv {
using gpu_stream_t  = musaStream_t;
using gpu_error_t   = musaError_t;
constexpr gpu_error_t GPU_SUCCESS = musaSuccess;
inline gpu_stream_t   gpu_current_stream() { return at::musa::getCurrentMUSAStream(); }
inline gpu_error_t    gpu_get_last_error() { return musaGetLastError(); }
inline const char*    gpu_get_error_string(gpu_error_t e) { return musaGetErrorString(e); }
}  // namespace flexkv

#else
// ─── CUDA backend (NVIDIA, default) ──────────────────────────────────────────
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
namespace flexkv {
using gpu_stream_t  = cudaStream_t;
using gpu_error_t   = cudaError_t;
constexpr gpu_error_t GPU_SUCCESS = cudaSuccess;
inline gpu_stream_t   gpu_current_stream() { return at::cuda::getCurrentCUDAStream(); }
inline gpu_error_t    gpu_get_last_error() { return cudaGetLastError(); }
inline const char*    gpu_get_error_string(gpu_error_t e) { return cudaGetErrorString(e); }
}  // namespace flexkv
#endif
