#pragma once
#include <cstdint>
using CUdeviceptr=uintptr_t;
constexpr int CUDA_SUCCESS=0,CU_POINTER_ATTRIBUTE_SYNC_MEMOPS=1;
inline int cuPointerSetAttribute(const void *,int,CUdeviceptr) {return 0;}
