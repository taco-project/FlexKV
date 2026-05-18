/*
 * SPDX-FileCopyrightText: Copyright (c) <2025> NVIDIA CORPORATION & AFFILIATES.
 * All rights reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Cross-vendor GPU type aliases.
 *
 * Exactly one of FLEXKV_BACKEND_NVIDIA / FLEXKV_BACKEND_ROCM /
 * FLEXKV_BACKEND_MUSA must be defined by the build system. When none is
 * defined we fall back to host-only stubs (compiles, but no real GPU).
 *
 * This header is intentionally header-only so any source file that
 * touches GPU runtime APIs can use ``gpu*`` symbols regardless of the
 * vendor it is being compiled for.
 */
#pragma once

#if defined(FLEXKV_BACKEND_NVIDIA)
  #include <cuda_runtime.h>

  using gpu_stream_t = cudaStream_t;
  using gpuError_t   = cudaError_t;

  #define gpuMallocHost      cudaMallocHost
  #define gpuFreeHost        cudaFreeHost
  #define gpuSetDevice       cudaSetDevice
  #define gpuGetDevice       cudaGetDevice
  #define gpuStreamCreate    cudaStreamCreate
  #define gpuStreamDestroy   cudaStreamDestroy
  #define gpuStreamSynchronize cudaStreamSynchronize
  #define gpuMemcpyAsync     cudaMemcpyAsync
  #define gpuMemcpyHostToDevice cudaMemcpyHostToDevice
  #define gpuMemcpyDeviceToHost cudaMemcpyDeviceToHost
  #define gpuMemcpyDeviceToDevice cudaMemcpyDeviceToDevice
  #define gpuGetLastError    cudaGetLastError
  #define gpuGetErrorString  cudaGetErrorString
  #define gpuSuccess         cudaSuccess

#elif defined(FLEXKV_BACKEND_ROCM)
  #include <hip/hip_runtime.h>

  using gpu_stream_t = hipStream_t;
  using gpuError_t   = hipError_t;

  #define gpuMallocHost      hipHostMalloc
  #define gpuFreeHost        hipHostFree
  #define gpuSetDevice       hipSetDevice
  #define gpuGetDevice       hipGetDevice
  #define gpuStreamCreate    hipStreamCreate
  #define gpuStreamDestroy   hipStreamDestroy
  #define gpuStreamSynchronize hipStreamSynchronize
  #define gpuMemcpyAsync     hipMemcpyAsync
  #define gpuMemcpyHostToDevice hipMemcpyHostToDevice
  #define gpuMemcpyDeviceToHost hipMemcpyDeviceToHost
  #define gpuMemcpyDeviceToDevice hipMemcpyDeviceToDevice
  #define gpuGetLastError    hipGetLastError
  #define gpuGetErrorString  hipGetErrorString
  #define gpuSuccess         hipSuccess

#elif defined(FLEXKV_BACKEND_MUSA)
  #include <musa_runtime.h>

  using gpu_stream_t = musaStream_t;
  using gpuError_t   = musaError_t;

  #define gpuMallocHost      musaMallocHost
  #define gpuFreeHost        musaFreeHost
  #define gpuSetDevice       musaSetDevice
  #define gpuGetDevice       musaGetDevice
  #define gpuStreamCreate    musaStreamCreate
  #define gpuStreamDestroy   musaStreamDestroy
  #define gpuStreamSynchronize musaStreamSynchronize
  #define gpuMemcpyAsync     musaMemcpyAsync
  #define gpuMemcpyHostToDevice musaMemcpyHostToDevice
  #define gpuMemcpyDeviceToHost musaMemcpyDeviceToHost
  #define gpuMemcpyDeviceToDevice musaMemcpyDeviceToDevice
  #define gpuGetLastError    musaGetLastError
  #define gpuGetErrorString  musaGetErrorString
  #define gpuSuccess         musaSuccess

#else
  /* Host-only fallback (Generic / CPU build). */
  #include <cstdint>
  using gpu_stream_t = void *;
  using gpuError_t   = int;

  #ifndef __host__
    #define __host__
  #endif
  #ifndef __device__
    #define __device__
  #endif

  #define gpuSuccess 0

#endif
