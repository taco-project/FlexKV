#pragma once

#include "host_registration.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace flexkv {

inline void check_host_dma_status(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(std::string("CE host DMA failed: ") +
                             cudaGetErrorString(status));
}

inline void checked_host_memcpy2d_async(void *dst, size_t dpitch,
                                        const void *src, size_t spitch,
                                        size_t width, size_t height,
                                        cudaMemcpyKind kind,
                                        cudaStream_t stream) {
  const bool h2d = kind == cudaMemcpyHostToDevice;
  const bool d2h = kind == cudaMemcpyDeviceToHost;
  const uintptr_t host = reinterpret_cast<uintptr_t>(h2d ? src : dst);
  const auto range = (h2d || d2h) ? find_host_registration(host) : std::nullopt;
  if (!range) {
    check_host_dma_status(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width,
                                            height, kind, stream));
    return;
  }
  const size_t host_pitch = h2d ? spitch : dpitch;
  const size_t device_pitch = h2d ? dpitch : spitch;
  for (const auto &slice : plan_registered_host_copy(
           *range, host, width, height, host_pitch, device_pitch)) {
    void *part_dst = static_cast<char *>(dst) +
                     (h2d ? slice.device_offset : slice.host_offset);
    const void *part_src = static_cast<const char *>(src) +
                           (h2d ? slice.host_offset : slice.device_offset);
    if (slice.height == 1) {
      check_host_dma_status(
          cudaMemcpyAsync(part_dst, part_src, slice.width, kind, stream));
    } else {
      check_host_dma_status(cudaMemcpy2DAsync(
          part_dst, h2d ? slice.device_pitch : slice.host_pitch, part_src,
          h2d ? slice.host_pitch : slice.device_pitch, slice.width,
          slice.height, kind, stream));
    }
  }
}

inline void checked_host_memcpy_async(void *dst, const void *src, size_t bytes,
                                      cudaMemcpyKind kind,
                                      cudaStream_t stream) {
  const bool h2d = kind == cudaMemcpyHostToDevice;
  const bool d2h = kind == cudaMemcpyDeviceToHost;
  const uintptr_t host = reinterpret_cast<uintptr_t>(h2d ? src : dst);
  const auto range = (h2d || d2h) ? find_host_registration(host) : std::nullopt;
  if (!range) {
    check_host_dma_status(cudaMemcpyAsync(dst, src, bytes, kind, stream));
    return;
  }
  for (const auto &slice :
       plan_registered_host_copy(*range, host, bytes, 1, bytes, bytes)) {
    void *part_dst = static_cast<char *>(dst) +
                     (h2d ? slice.device_offset : slice.host_offset);
    const void *part_src = static_cast<const char *>(src) +
                           (h2d ? slice.host_offset : slice.device_offset);
    check_host_dma_status(
        cudaMemcpyAsync(part_dst, part_src, slice.width, kind, stream));
  }
}

} // namespace flexkv
