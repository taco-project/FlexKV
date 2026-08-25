/*
 * FlexKV NVTX compatibility shim.
 *
 * Purpose
 * -------
 * FlexKV's C++ sources annotate hot paths with NVTX ranges so they show up in
 * Nsight Systems timelines. NVTX is an NVIDIA-only facility: the header
 * ``nvtx3/nvToolsExt.h`` ships with the CUDA toolkit and the symbols are
 * resolved by the NVIDIA driver/profiler at runtime.
 *
 * On CUDA-like accelerators that are not NVIDIA GPUs (e.g. Baidu Kunlun P800,
 * which exposes a partial CUDA runtime through its own toolchain), the NVTX
 * header is absent and there is no profiler to consume the events. Rather than
 * stripping the annotations -- which would fork the sources and make future
 * upstream merges painful -- this header provides a drop-in replacement that
 * compiles to nothing when NVTX is unavailable.
 *
 * Usage
 * -----
 * Replace ``#include <nvtx3/nvToolsExt.h>`` with ``#include "flexkv_nvtx.h"``.
 * All ``nvtxRange*`` call sites stay exactly as they were.
 *
 * Selection
 * ---------
 * The build system decides which path is active:
 *
 *   FLEXKV_ENABLE_NVTX defined   -> real NVTX (include the CUDA header)
 *   FLEXKV_ENABLE_NVTX undefined -> inline no-op stubs
 *
 * ``setup.py`` defines ``FLEXKV_ENABLE_NVTX`` automatically when the NVTX
 * header can be located, and leaves it undefined otherwise. It can be forced
 * either way with the ``FLEXKV_ENABLE_NVTX`` environment variable.
 *
 * The stubs are ``static inline`` so every translation unit gets its own
 * trivially-inlinable copy; with -O3 the calls vanish entirely and no symbols
 * are emitted. ``nvtxRangeId_t`` keeps its real ``uint64_t`` type so that
 * existing declarations such as ``std::vector<nvtxRangeId_t>`` and
 * ``nvtxRangeId_t *`` continue to work untouched.
 */
#pragma once

#if defined(FLEXKV_ENABLE_NVTX)

#include <nvtx3/nvToolsExt.h>

#else // !FLEXKV_ENABLE_NVTX

#include <cstdint>

// Mirror the upstream typedefs so declarations in FlexKV headers (function
// signatures, std::vector<nvtxRangeId_t>, out-parameters) keep compiling.
typedef uint64_t nvtxRangeId_t;
typedef int nvtxRangePushPopLevel_t;

// Sentinel used by upstream NVTX to report a failed range start. FlexKV
// initialises range-id storage to 0, so 0 stays the "no active range" value.
#define NVTX_NO_PUSH_POP_TRACKING ((int)-2)

namespace flexkv {
namespace nvtx_stub {

// A non-zero constant keeps `if (range_id)` style guards behaving as they do
// with real NVTX (where a successful start_range never returns 0).
inline constexpr nvtxRangeId_t kDummyRangeId = 1;

} // namespace nvtx_stub
} // namespace flexkv

static inline nvtxRangeId_t nvtxRangeStartA(const char *message) {
  (void)message;
  return flexkv::nvtx_stub::kDummyRangeId;
}

static inline nvtxRangeId_t nvtxRangeStartW(const wchar_t *message) {
  (void)message;
  return flexkv::nvtx_stub::kDummyRangeId;
}

static inline void nvtxRangeEnd(nvtxRangeId_t id) { (void)id; }

static inline int nvtxRangePushA(const char *message) {
  (void)message;
  return 0;
}

static inline int nvtxRangePushW(const wchar_t *message) {
  (void)message;
  return 0;
}

static inline int nvtxRangePop(void) { return 0; }

static inline void nvtxMarkA(const char *message) { (void)message; }

static inline void nvtxMarkW(const wchar_t *message) { (void)message; }

static inline void nvtxNameOsThreadA(uint32_t threadId, const char *name) {
  (void)threadId;
  (void)name;
}

#endif // FLEXKV_ENABLE_NVTX
