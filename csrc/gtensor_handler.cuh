#pragma once
#include <cuda_runtime.h>

namespace flexkv {

// Backend type enumeration
enum class BackendType {
    VLLM,
    TRTLLM,
    SGLANG
};

// Simplified GTensorHandler - no inheritance, just a data structure
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

// Template specialization for different backends
// Forward declaration
template<BackendType Type>
__device__ __host__ inline
int64_t* ptr_at(const GTensorHandler& handler, 
               int64_t layer_idx, int64_t kv_idx, int64_t block_idx);

// vLLM specialization
template<>
__device__ __host__ inline
int64_t* ptr_at<BackendType::VLLM>(const GTensorHandler& handler,
                                    int64_t layer_idx, int64_t kv_idx, int64_t block_idx) {
    return handler.gpu_tensor_ptrs[layer_idx] + 
           kv_idx * handler.gpu_kv_stride + 
           block_idx * handler.gpu_block_stride;
}

// TRT-LLM specialization
template<>
__device__ __host__ inline
int64_t* ptr_at<BackendType::TRTLLM>(const GTensorHandler& handler,
                                      int64_t layer_idx, int64_t kv_idx, int64_t block_idx) {
    return handler.gpu_tensor_ptrs[0] + 
           block_idx * handler.gpu_block_stride + 
           layer_idx * handler.gpu_layer_stride + 
           kv_idx * handler.gpu_kv_stride;
}

// SGLang specialization
template<>
__device__ __host__ inline
int64_t* ptr_at<BackendType::SGLANG>(const GTensorHandler& handler,
                                      int64_t layer_idx, int64_t kv_idx, int64_t block_idx) {
    return handler.gpu_tensor_ptrs[kv_idx * handler.num_layers + layer_idx] + 
           block_idx * handler.gpu_block_stride;
}

} // namespace flexkv