/*
 * This file has moved to csrc/gpu_backend/nvidia/transfer.cu during the
 * GPU backend abstraction refactor (P3). It is no longer compiled by
 * setup.py; the build_backends/cuda_builder.py picks the new path
 * automatically.
 *
 * If you see this error, you are likely invoking nvcc directly on the
 * legacy path. Switch to ``pip install -e .`` (or the new path) instead.
 */
#error "csrc/transfer.cu has moved to csrc/gpu_backend/nvidia/transfer.cu"
