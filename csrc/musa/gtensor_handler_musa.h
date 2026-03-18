/*
 * GTensorHandler for MUSA - backend-agnostic GPU tensor descriptor.
 * Mirrors csrc/gtensor_handler.cuh with musa_runtime.h (no CUDA dependency).
 */
#pragma once

#include <musa_runtime.h>
#include <cstdint>

namespace flexkv {

enum class BackendType {
    VLLM,
    TRTLLM,
    SGLANG
};

struct GTensorHandler {
    BackendType type;
    int64_t **gpu_tensor_ptrs;
    int64_t num_layers;
    int64_t gpu_kv_stride;
    int64_t gpu_block_stride;
    int64_t gpu_layer_stride;

    __host__ __device__
    GTensorHandler()
        : type(BackendType::VLLM),
          gpu_tensor_ptrs(nullptr),
          num_layers(0),
          gpu_kv_stride(0),
          gpu_block_stride(0),
          gpu_layer_stride(0) {}

    __host__ __device__
    GTensorHandler(BackendType type,
                   int64_t **gpu_tensor_ptrs,
                   int64_t num_layers,
                   int64_t gpu_kv_stride_in_bytes,
                   int64_t gpu_block_stride_in_bytes,
                   int64_t gpu_layer_stride_in_bytes)
        : type(type),
          gpu_tensor_ptrs(gpu_tensor_ptrs),
          num_layers(num_layers),
          gpu_kv_stride(gpu_kv_stride_in_bytes / sizeof(int64_t)),
          gpu_block_stride(gpu_block_stride_in_bytes / sizeof(int64_t)),
          gpu_layer_stride(gpu_layer_stride_in_bytes / sizeof(int64_t)) {}
};

template<BackendType Type>
__device__ __host__ inline
int64_t* ptr_at(const GTensorHandler& handler,
                int64_t layer_idx, int64_t kv_idx, int64_t block_idx);

template<>
__device__ __host__ inline
int64_t* ptr_at<BackendType::VLLM>(const GTensorHandler& handler,
                                   int64_t layer_idx, int64_t kv_idx, int64_t block_idx) {
    return handler.gpu_tensor_ptrs[layer_idx] +
           kv_idx * handler.gpu_kv_stride +
           block_idx * handler.gpu_block_stride;
}

template<>
__device__ __host__ inline
int64_t* ptr_at<BackendType::TRTLLM>(const GTensorHandler& handler,
                                     int64_t layer_idx, int64_t kv_idx, int64_t block_idx) {
    return handler.gpu_tensor_ptrs[0] +
           block_idx * handler.gpu_block_stride +
           layer_idx * handler.gpu_layer_stride +
           kv_idx * handler.gpu_kv_stride;
}

template<>
__device__ __host__ inline
int64_t* ptr_at<BackendType::SGLANG>(const GTensorHandler& handler,
                                     int64_t layer_idx, int64_t kv_idx, int64_t block_idx) {
    return handler.gpu_tensor_ptrs[kv_idx * handler.num_layers + layer_idx] +
           block_idx * handler.gpu_block_stride;
}

}  // namespace flexkv
