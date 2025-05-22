#!/bin/bash
set -e

PROJECT_ROOT=$(pwd)
BUILD_TYPE="debug"  # Default to debug build

# Parse command line arguments
for arg in "$@"; do
  case $arg in
    --debug)
      BUILD_TYPE="debug"
      shift
      ;;
    --release)
      BUILD_TYPE="release"
      shift
      ;;
    *)
      # Unknown option
      ;;
  esac
done

echo "=== Building in ${BUILD_TYPE} mode ==="

# Install submodules
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
if [ "$BUILD_TYPE" = "debug" ]; then
  FLEXKV_DEBUG=1 pip install --no-build-isolation -e .
else
  FLEXKV_DEBUG=0 pip install --no-build-isolation -e .
fi

export LD_LIBRARY_PATH=$BUILD_LIB_PATH:$LD_LIBRARY_PATH
echo "Added $BUILD_LIB_PATH to LD_LIBRARY_PATH for current session"

echo "=== Build and installation completed successfully in ${BUILD_TYPE} mode ==="
