cmake_minimum_required(VERSION 3.10)
project(MainProject VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_BINARY_DIR}/include)

file(MAKE_DIRECTORY ${CMAKE_INSTALL_INCLUDEDIR})

file(GLOB XXHASH_HEADERS "third_party/xxHash/*.h")
file(COPY ${XXHASH_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set(XXHASH_SOURCES
    third_party/xxHash/xxhash.c
)
add_library(xxhash SHARED ${XXHASH_SOURCES})
target_include_directories(xxhash PUBLIC
    third_party/xxHash
    ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES ${XXHASH_HEADERS} DESTINATION include)

# ==================== prometheus-cpp Library ====================
# Option to enable/disable monitoring (env FLEXKV_ENABLE_METRICS=0 or -DFLEXKV_ENABLE_MONITORING=OFF)
set(_FLEXKV_MONITORING_DEFAULT ON)
if(DEFINED ENV{FLEXKV_ENABLE_METRICS})
  if("$ENV{FLEXKV_ENABLE_METRICS}" STREQUAL "0" OR "$ENV{FLEXKV_ENABLE_METRICS}" STREQUAL "OFF"
      OR "$ENV{FLEXKV_ENABLE_METRICS}" STREQUAL "NO" OR "$ENV{FLEXKV_ENABLE_METRICS}" STREQUAL "FALSE")
    set(_FLEXKV_MONITORING_DEFAULT OFF)
  endif()
endif()
option(FLEXKV_ENABLE_MONITORING "Enable Prometheus monitoring support" ${_FLEXKV_MONITORING_DEFAULT})

if(FLEXKV_ENABLE_MONITORING)
    message(STATUS "Prometheus monitoring: ENABLED")
    
    # Enable PIC for all targets (required for linking static libs into shared libs)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    
    # Configure prometheus-cpp build options
    set(ENABLE_PUSH OFF CACHE BOOL "Disable push gateway support")
    set(ENABLE_COMPRESSION OFF CACHE BOOL "Disable compression support")
    set(ENABLE_TESTING OFF CACHE BOOL "Disable prometheus-cpp tests")
    set(GENERATE_PKGCONFIG OFF CACHE BOOL "Disable pkg-config generation")
    
    # Save CMAKE_INSTALL_INCLUDEDIR before adding prometheus-cpp
    set(_SAVED_CMAKE_INSTALL_INCLUDEDIR ${CMAKE_INSTALL_INCLUDEDIR})
    # Reset to default for prometheus-cpp (it expects relative path like "include")
    unset(CMAKE_INSTALL_INCLUDEDIR)
    
    # Add prometheus-cpp subdirectory (EXCLUDE_FROM_ALL prevents install targets)
    add_subdirectory(third_party/prometheus-cpp EXCLUDE_FROM_ALL)
    
    # Force prometheus-cpp libraries to output to our unified lib directory
    set_target_properties(core PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
    set_target_properties(pull PROPERTIES
        ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
    )
    
    # Copy prometheus-cpp headers to unified build/include directory
    # 1. Source headers from core/include and pull/include (copied at configure time)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/prometheus-cpp/core/include/prometheus
         DESTINATION ${CMAKE_BINARY_DIR}/include)
    file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/third_party/prometheus-cpp/pull/include/prometheus
         DESTINATION ${CMAKE_BINARY_DIR}/include)
    
    # 2. Generated headers (core_export.h, pull_export.h) - use custom target
    add_custom_target(copy_prometheus_headers ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_BINARY_DIR}/third_party/prometheus-cpp/core/include/prometheus
            ${CMAKE_BINARY_DIR}/include/prometheus
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            ${CMAKE_BINARY_DIR}/third_party/prometheus-cpp/pull/include/prometheus
            ${CMAKE_BINARY_DIR}/include/prometheus
        DEPENDS core pull
        COMMENT "Copying prometheus-cpp generated headers to build/include"
    )
    
    # Restore CMAKE_INSTALL_INCLUDEDIR for FlexKV
    set(CMAKE_INSTALL_INCLUDEDIR ${_SAVED_CMAKE_INSTALL_INCLUDEDIR})
    
    # ==================== FlexKV Monitoring Library ====================
    set(MONITORING_SOURCES
        csrc/monitoring/metrics_manager.cpp
    )
    
    # Use STATIC library to avoid PIC issues with prometheus-cpp static libs
    add_library(flexkv_monitoring STATIC ${MONITORING_SOURCES})
    target_include_directories(flexkv_monitoring PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/csrc/monitoring
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/prometheus-cpp/core/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/prometheus-cpp/pull/include
    )
    target_link_libraries(flexkv_monitoring PUBLIC
        prometheus-cpp::pull
        prometheus-cpp::core
    )
    
    # Install monitoring headers
    install(FILES csrc/monitoring/metrics_manager.h DESTINATION include/flexkv/monitoring)
    
else()
    message(STATUS "Prometheus monitoring: DISABLED")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
