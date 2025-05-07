#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Get absolute path to project root
PROJECT_ROOT=$(pwd)

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with CMake
echo "=== Running CMake configuration ==="
cmake ..

# Build third-party libraries
echo "=== Building third-party libraries ==="
cmake --build .

# Set BUILD_LIB_PATH using absolute path
BUILD_LIB_PATH=$(pwd)/lib
echo "=== Setting BUILD_LIB_PATH to $BUILD_LIB_PATH ==="

# Return to the project root
cd ..

# Install the package with pip
echo "=== Installing package with pip ==="
pip install --no-build-isolation -e .

# Add to LD_LIBRARY_PATH for current session
export LD_LIBRARY_PATH=$BUILD_LIB_PATH:$LD_LIBRARY_PATH
echo "Added $BUILD_LIB_PATH to LD_LIBRARY_PATH for current session"

echo "=== Build and installation completed successfully ==="