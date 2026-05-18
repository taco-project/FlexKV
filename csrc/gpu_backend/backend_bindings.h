// SPDX-License-Identifier: Apache-2.0
//
// GPU backend pybind11 registration interface.
//
// Each vendor (NVIDIA / ROCm / MUSA / ...) implements ONE function with the
// signature ``void register_<vendor>_bindings(pybind11::module_& m)`` that
// adds all of its vendor-specific Python bindings (transfer_kv_blocks,
// GDSManager, TPGDSTransferThreadGroup, etc.) to the same ``flexkv.c_ext``
// module.
//
// ``csrc/bindings.cpp`` then dispatches to the right implementation purely
// via the ``FLEXKV_BACKEND_<VENDOR>`` macro chosen by setup.py / the
// build-backends layer, so it does NOT itself include any vendor header.
//
// The TPTransferThreadGroup binding is also routed through this file even
// though its C++ implementation is already cross-vendor (cf.
// csrc/gpu_backend/nvidia/tp_transfer_thread_group.cpp + gpu_types.h):
// keeping the binding next to the kernel sources keeps each vendor's wheel
// self-contained.
#pragma once

#include <pybind11/pybind11.h>

namespace flexkv {
namespace gpu_backend {

#if defined(FLEXKV_BACKEND_NVIDIA)
void register_nvidia_bindings(pybind11::module_& m);
#endif

#if defined(FLEXKV_BACKEND_ROCM)
void register_rocm_bindings(pybind11::module_& m);
#endif

#if defined(FLEXKV_BACKEND_MUSA)
void register_musa_bindings(pybind11::module_& m);
#endif

inline void register_active_backend_bindings(pybind11::module_& m) {
#if defined(FLEXKV_BACKEND_NVIDIA)
  register_nvidia_bindings(m);
#elif defined(FLEXKV_BACKEND_ROCM)
  register_rocm_bindings(m);
#elif defined(FLEXKV_BACKEND_MUSA)
  register_musa_bindings(m);
#else
  // No GPU backend selected at compile time. ``c_ext`` still exposes the
  // CPU-only bindings (radix tree, hasher, SSD, P2P, ...) registered
  // directly inside csrc/bindings.cpp.
  (void)m;
#endif
}

}  // namespace gpu_backend
}  // namespace flexkv
