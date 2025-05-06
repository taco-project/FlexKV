cmake_minimum_required(VERSION 3.10)
project(MainProject VERSION 1.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set build type (if not specified)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Output directory settings
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_INSTALL_INCLUDEDIR ${CMAKE_BINARY_DIR}/include)

# Create include directory in build folder
file(MAKE_DIRECTORY ${CMAKE_INSTALL_INCLUDEDIR})

# Copy header files to build include directory
file(GLOB XXHASH_HEADERS "third_party/xxHash/*.h")
file(COPY ${XXHASH_HEADERS} DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# If you need to manually create build rules for projects under third_party
# For example, create a library for xxhash
set(XXHASH_SOURCES
    third_party/xxHash/xxhash.c
)
add_library(xxhash SHARED ${XXHASH_SOURCES})
target_include_directories(xxhash PUBLIC 
    third_party/xxHash
    ${CMAKE_INSTALL_INCLUDEDIR}
)

# Install headers (optional if you want to support 'make install')
install(FILES ${XXHASH_HEADERS} DESTINATION include)

# Print some information
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}") 