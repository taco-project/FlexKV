# csrc/gpu_backend/nvidia/CMakeLists.txt
# NVIDIA CUDA backend — compiled only when FLEXKV_BACKEND=nvidia

enable_language(CUDA)

find_library(CUFILE_LIB cufile HINTS ${CUDAToolkit_LIBRARY_DIR})
if(CUFILE_LIB)
  add_compile_definitions(FLEXKV_HAVE_GDS)
  message(STATUS "FlexKV: cuFile found → GDS enabled")
else()
  message(STATUS "FlexKV: cuFile not found → GDS disabled")
endif()

set(NVIDIA_SOURCES
  transfer.cu
  layout_transform.cu
  gds_manager.cpp
  tp_transfer_thread_group.cpp
  tp_gds_transfer_thread_group.cpp
  gpu_ral.cpp
)

add_library(flexkv_backend_nvidia STATIC ${NVIDIA_SOURCES})

target_compile_options(flexkv_backend_nvidia PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr -O3 -DCUDA_AVAILABLE -DFLEXKV_BACKEND_NVIDIA>
  $<$<COMPILE_LANGUAGE:CXX>:-std=c++17 -O3 -DCUDA_AVAILABLE -DFLEXKV_BACKEND_NVIDIA>
)

target_link_libraries(flexkv_backend_nvidia
  CUDA::cudart CUDA::cuda_driver
)

if(CUFILE_LIB)
  target_link_libraries(flexkv_backend_nvidia ${CUFILE_LIB})
endif()
