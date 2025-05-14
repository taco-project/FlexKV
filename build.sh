#!/bin/bash
set -e

PROJECT_ROOT=$(pwd)

# 安装submodule
git submodule update --init --recursive

mkdir -p build
cd build

echo "=== Running CMake configuration ==="
cmake ..

echo "=== Building third-party libraries ==="
cmake --build .

BUILD_LIB_PATH=$(pwd)/lib
echo "=== Setting BUILD_LIB_PATH to $BUILD_LIB_PATH ==="

cd ..

echo "=== Installing package with pip ==="
pip install --no-build-isolation -e .

export LD_LIBRARY_PATH=$BUILD_LIB_PATH:$LD_LIBRARY_PATH
echo "Added $BUILD_LIB_PATH to LD_LIBRARY_PATH for current session"

echo "=== Build and installation completed successfully ==="
