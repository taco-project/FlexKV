#pragma once

#ifndef FLEXKV_ENABLE_NVTX
#define FLEXKV_ENABLE_NVTX 1
#endif

#if FLEXKV_ENABLE_NVTX
#include <nvtx3/nvToolsExt.h>
#else
#include <cstdint>
using nvtxRangeId_t = uint64_t;
using nvtxRangePushPopLevel_t = int;
#define NVTX_NO_PUSH_POP_TRACKING (-2)
static inline nvtxRangeId_t nvtxRangeStartA(const char *) { return 1; }
static inline nvtxRangeId_t nvtxRangeStartW(const wchar_t *) { return 1; }
static inline void nvtxRangeEnd(nvtxRangeId_t) {}
static inline int nvtxRangePushA(const char *) { return 0; }
static inline int nvtxRangePushW(const wchar_t *) { return 0; }
static inline int nvtxRangePop() { return 0; }
static inline void nvtxMarkA(const char *) {}
static inline void nvtxMarkW(const wchar_t *) {}
static inline void nvtxNameOsThreadA(uint32_t, const char *) {}
#endif
