# csrc/gpu_backend/generic/CMakeLists.txt
# Generic CPU fallback — no GPU SDK required

add_library(flexkv_backend_generic STATIC fallback.cpp)
target_compile_options(flexkv_backend_generic PRIVATE -std=c++17 -O2)
