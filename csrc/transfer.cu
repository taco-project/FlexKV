#include <cuda_runtime.h>
#include <torch/extension.h>

#include "transfer.cuh"

namespace flexkv {

#define FLOAT4_PTR(ptr) reinterpret_cast<float4 *>(ptr)

__global__ void transfer_kv_layers_kernel(
    int num_blocks, int num_layers, int64_t *dst_block_ids,
    int64_t **dst_layer_ptrs, int64_t dst_kv_stride, int64_t dst_chunk_stride,
    int64_t *src_block_ids, int64_t **src_layer_ptrs, int64_t src_kv_stride,
    int64_t src_chunk_stride, int64_t chunk_size) {
  int num_chunks = num_layers * 2 * num_blocks;
  int64_t chunk_size_in_float4 = chunk_size * sizeof(int64_t) / sizeof(float4);

  for (int chunk_idx = blockIdx.x; chunk_idx < num_chunks;
       chunk_idx += gridDim.x) {
    int lay_idx = chunk_idx / (num_blocks * 2);
    int kv_idx = (chunk_idx % (num_blocks * 2)) / num_blocks;
    int dst_block_idx = dst_block_ids[chunk_idx % num_blocks];
    int src_block_idx = src_block_ids[chunk_idx % num_blocks];

    int64_t *dst_layer_kv_ptr =
        dst_layer_ptrs[lay_idx] + kv_idx * dst_kv_stride;
    int64_t *src_layer_kv_ptr =
        src_layer_ptrs[lay_idx] + kv_idx * src_kv_stride;

    int64_t *dst_chunk_ptr =
        dst_layer_kv_ptr + dst_block_idx * dst_chunk_stride;
    int64_t *src_chunk_ptr =
        src_layer_kv_ptr + src_block_idx * src_chunk_stride;

    for (int64_t idx = threadIdx.x; idx < chunk_size_in_float4;
         idx += blockDim.x) {
      float4 element = __ldg(&FLOAT4_PTR(src_chunk_ptr)[idx]);
      FLOAT4_PTR(dst_chunk_ptr)[idx] = element;
    }
  }
}

void transfer_kv_layers(int num_blocks, int num_layers, int64_t *dst_block_ids,
                        void **dst_layer_ptrs, int64_t dst_kv_stride_in_bytes,
                        int64_t dst_chunk_stride_in_bytes,
                        int64_t *src_block_ids, void **src_layer_ptrs,
                        int64_t src_kv_stride_in_bytes,
                        int64_t src_chunk_stride_in_bytes,
                        int64_t chunk_size_in_bytes, cudaStream_t stream,
                        int transfer_sms, bool is_host_to_device,
                        bool use_ce_transfer) {
  int block_size = 128;
  static int max_blocks_per_sm = -1;
  if (max_blocks_per_sm == -1) {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &max_blocks_per_sm, transfer_kv_layers_kernel, block_size, 0);
  }

  if (transfer_sms == -1) {
    transfer_sms = 4;
  }

  int block_count = transfer_sms * max_blocks_per_sm;

  int64_t **dst_layer_ptrs_int64 = reinterpret_cast<int64_t **>(dst_layer_ptrs);
  int64_t **src_layer_ptrs_int64 = reinterpret_cast<int64_t **>(src_layer_ptrs);
  int64_t dst_kv_stride_int64 = dst_kv_stride_in_bytes / sizeof(int64_t);
  int64_t src_kv_stride_int64 = src_kv_stride_in_bytes / sizeof(int64_t);
  int64_t dst_chunk_stride_int64 = dst_chunk_stride_in_bytes / sizeof(int64_t);
  int64_t src_chunk_stride_int64 = src_chunk_stride_in_bytes / sizeof(int64_t);
  int64_t chunk_size_in_int64 = chunk_size_in_bytes / sizeof(int64_t);

  dim3 blockDim(block_size);
  dim3 gridDim(block_count);
  if (use_ce_transfer) {
    for (int i = 0; i < num_layers; i++) {
      for (int j = 0; j < 2; j++) {
        for (int k = 0; k < num_blocks; k++) {
          int64_t *dst_layer_kv_ptr =
              dst_layer_ptrs_int64[i] + j * dst_kv_stride_int64;
          int64_t *src_layer_kv_ptr =
              src_layer_ptrs_int64[i] + j * src_kv_stride_int64;
          int64_t dst_block_idx = dst_block_ids[k];
          int64_t src_block_idx = src_block_ids[k];
          int64_t *dst_chunk_ptr =
              dst_layer_kv_ptr + dst_block_idx * dst_chunk_stride_int64;
          int64_t *src_chunk_ptr =
              src_layer_kv_ptr + src_block_idx * src_chunk_stride_int64;

          if (is_host_to_device) {
            cudaMemcpyAsync(dst_chunk_ptr, src_chunk_ptr, chunk_size_in_bytes,
                            cudaMemcpyHostToDevice, stream);
          } else {
            cudaMemcpyAsync(dst_chunk_ptr, src_chunk_ptr, chunk_size_in_bytes,
                            cudaMemcpyDeviceToHost, stream);
          }
        }
      }
    }
  } else {
    transfer_kv_layers_kernel<<<gridDim, blockDim, 0, stream>>>(
        num_blocks, num_layers, dst_block_ids, dst_layer_ptrs_int64,
        dst_kv_stride_int64, dst_chunk_stride_int64, src_block_ids,
        src_layer_ptrs_int64, src_kv_stride_int64, src_chunk_stride_int64,
        chunk_size_in_int64);
  }
}

} // namespace flexkv
