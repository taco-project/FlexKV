#pragma once

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#include "gtensor_handler.cuh"

namespace flexkv {

inline bool d2h_debug_enabled() {
  static int enabled = -1;
  if (enabled < 0) {
    const char *env = std::getenv("FLEXKV_D2H_DEBUG");
    // Default ON; set FLEXKV_D2H_DEBUG=0 to disable verbose C++ D2H logs.
    enabled = (env != nullptr && std::strcmp(env, "0") == 0) ? 0 : 1;
  }
  return enabled != 0;
}

inline thread_local int d2h_debug_gpu_index = -1;

struct D2hDebugGpuScope {
  explicit D2hDebugGpuScope(int gpu_index) { d2h_debug_gpu_index = gpu_index; }
  ~D2hDebugGpuScope() { d2h_debug_gpu_index = -1; }
};

#define FLEXKV_D2H_LOG(...)                                                    \
  do {                                                                         \
    if (flexkv::d2h_debug_enabled()) {                                         \
      fprintf(stderr, "[FlexKV-D2H-DEBUG] ");                                  \
      fprintf(stderr, __VA_ARGS__);                                            \
      fprintf(stderr, "\n");                                                   \
      fflush(stderr);                                                          \
    }                                                                          \
  } while (0)

inline const char *backend_name(BackendType type) {
  switch (type) {
  case BackendType::VLLM:
    return "VLLM";
  case BackendType::TRTLLM:
    return "TRTLLM";
  case BackendType::SGLANG:
    return "SGLANG";
  }
  return "UNKNOWN";
}

inline void log_pointer_attributes(const char *name, const void *ptr) {
  if (!d2h_debug_enabled()) {
    return;
  }
  cudaPointerAttributes attr;
  std::memset(&attr, 0, sizeof(attr));
#if CUDART_VERSION >= 10000
  attr.type = cudaMemoryTypeUnregistered;
#endif
  cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
  if (err == cudaSuccess) {
#if CUDART_VERSION >= 10000
    FLEXKV_D2H_LOG("ptr %-16s %p type=%d device=%d", name, ptr,
                   static_cast<int>(attr.type), attr.device);
#else
    FLEXKV_D2H_LOG("ptr %-16s %p legacy-attr-ok", name, ptr);
#endif
  } else {
    FLEXKV_D2H_LOG("ptr %-16s %p attr_err=%s", name, ptr,
                   cudaGetErrorString(err));
  }
}

inline int64_t min_block_id(const int64_t *ids, int num_blocks) {
  if (num_blocks <= 0) {
    return 0;
  }
  int64_t v = ids[0];
  for (int i = 1; i < num_blocks; ++i) {
    if (ids[i] < v) {
      v = ids[i];
    }
  }
  return v;
}

inline int64_t max_block_id(const int64_t *ids, int num_blocks) {
  if (num_blocks <= 0) {
    return 0;
  }
  int64_t v = ids[0];
  for (int i = 1; i < num_blocks; ++i) {
    if (ids[i] > v) {
      v = ids[i];
    }
  }
  return v;
}

} // namespace flexkv
