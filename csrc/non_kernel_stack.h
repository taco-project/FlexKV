#pragma once

#ifndef FLEXKV_ENABLE_KERNEL_TRANSFER
#define FLEXKV_ENABLE_KERNEL_TRANSFER 1
#endif

namespace flexkv {

// Host-only builds (FLEXKV_ENABLE_KERNEL_TRANSFER=0) always run the non-kernel stack.
constexpr bool non_kernel_stack_mode() { return !FLEXKV_ENABLE_KERNEL_TRANSFER; }

} // namespace flexkv
