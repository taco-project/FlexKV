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

# Set LD_LIBRARY_PATH for immediate use
export LD_LIBRARY_PATH=$BUILD_LIB_PATH:$LD_LIBRARY_PATH
echo "Added $BUILD_LIB_PATH to LD_LIBRARY_PATH for current session"

# Copy shared libraries to package directory for permanent access
echo "=== Copying shared libraries to package directory ==="
PACKAGE_LIB_DIR="flexkv/lib"
mkdir -p $PACKAGE_LIB_DIR

if [ -d "$BUILD_LIB_PATH" ]; then
    for lib_file in $BUILD_LIB_PATH/*.so*; do
        if [ -f "$lib_file" ]; then
            cp "$lib_file" "$PACKAGE_LIB_DIR/"
            echo "Copied $(basename $lib_file) to $PACKAGE_LIB_DIR/"
        fi
    done
else
    echo "Warning: Build lib directory $BUILD_LIB_PATH not found"
fi

echo "=== Build and installation completed successfully in ${BUILD_TYPE} mode ==="
echo "You can now run tests directly without setting LD_LIBRARY_PATH manually"

if [ "$BUILD_TYPE" = "debug" ]; then
  FLEXKV_DEBUG=1 pip install -v --no-build-isolation -e .
elif [ "$BUILD_TYPE" = "release" ]; then
  FLEXKV_DEBUG=0 python setup.py bdist_wheel -v
else
  FLEXKV_DEBUG=0 pip install -v --no-build-isolation -e .
fi
