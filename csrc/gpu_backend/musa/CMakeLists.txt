# csrc/gpu_backend/musa/CMakeLists.txt
# 摩尔线程 MUSA backend — 必须含 muFile GDS，需要 MUSA SDK + MUSA_HOME

set(MUSA_HOME $ENV{MUSA_HOME})
if(NOT MUSA_HOME)
  message(FATAL_ERROR "MUSA_HOME must be set for MUSA backend")
endif()

set(MUSA_SOURCES
  bindings_musa.cpp
  tp_transfer_thread_group_musa.cpp
)

if(EXISTS "${MUSA_HOME}/bin/mcc")
  list(APPEND MUSA_SOURCES transfer_musa.mu)
  list(APPEND MUSA_SOURCES gds/gds_manager_musa.cpp)
  list(APPEND MUSA_SOURCES gds/tp_gds_transfer_thread_group_musa.cpp)
  list(APPEND MUSA_SOURCES gds/layout_transform_musa.mu)
  message(STATUS "FlexKV: mcc found at ${MUSA_HOME}/bin/mcc → building with MUSA SDK + muFile GDS")
else()
  message(FATAL_ERROR "mcc compiler not found in ${MUSA_HOME}/bin — cannot build MUSA backend")
endif()

add_library(flexkv_backend_musa STATIC ${MUSA_SOURCES})

target_compile_definitions(flexkv_backend_musa PRIVATE
  FLEXKV_BACKEND_MUSA MUSA_AVAILABLE FLEXKV_STORAGE_MUFILE
)

target_include_directories(flexkv_backend_musa PRIVATE ${MUSA_HOME}/include)
target_link_libraries(flexkv_backend_musa
  -L${MUSA_HOME}/lib -lmusa -lmusart -lmufile
)
