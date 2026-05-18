/*
 * Compatibility shim. The real header has moved to
 * csrc/gpu_backend/nvidia/gtensor_handler.cuh during the GPU backend
 * abstraction refactor (P3). This shim is kept so that any third-party
 * code that still includes the legacy path keeps compiling.
 */
#pragma once
#include "gpu_backend/nvidia/gtensor_handler.cuh"
