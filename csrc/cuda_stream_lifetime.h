#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace flexkv {

// A recoverable transfer failure must mean no DMA can still touch its blocks.
// If CUDA cannot establish that boundary, do not return an ordinary failure
// that lets the caller recycle them. Terminate this transfer process instead.
inline void drain_stream_or_abort(cudaStream_t stream, const char *what) {
  cudaError_t status = cudaStreamSynchronize(stream);
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s: cannot establish CUDA stream quiescence: %s\n",
                 what, cudaGetErrorString(status));
    std::fflush(stderr);
    std::abort();
  }
  // The original exception retains the failure. Clear the thread-local CUDA
  // error after a successful drain so it cannot poison a later retry's check.
  cudaGetLastError();
}

} // namespace flexkv
