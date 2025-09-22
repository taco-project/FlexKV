#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 [--cuda|--no-cuda] [install|build|develop]"
  echo "  --cuda       Build with CUDA (default)"
  echo "  --no-cuda    Build without CUDA"
  echo "  install      pip install -v . (default)"
  echo "  build        python -m build"
  echo "  develop      pip install -v -e ."
}

WITH_CUDA=1
ACTION="install"
BUILD_TYPE="debug"

for arg in "$@"; do
  case "$arg" in
    --cuda) WITH_CUDA=1 ;;
    --no-cuda) WITH_CUDA=0 ;;
    --debug) BUILD_TYPE="debug" ;;
    --release) BUILD_TYPE="release" ;;
    install|build|develop) ACTION="$arg" ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $arg"; usage; exit 1 ;;
  esac
done

export FLEXKV_WITH_CUDA="${WITH_CUDA}"

if [[ "${WITH_CUDA}" == "1" ]]; then
  echo "[INFO] Building WITH CUDA"
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "[WARN] nvcc not found in PATH; ensure CUDA toolkit is available"
  fi
else
  echo "[INFO] Building WITHOUT CUDA"
fi

# Defer Python build/install to the end to ensure env vars are set and isolation is disabled

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
  for lib_file in "$BUILD_LIB_PATH"/*.so*; do
    if [ -f "$lib_file" ]; then
      cp "$lib_file" "$PACKAGE_LIB_DIR/"
      echo "Copied $(basename "$lib_file") to $PACKAGE_LIB_DIR/"
    fi
  done
else
  echo "Warning: Build lib directory $BUILD_LIB_PATH not found"
fi

# Drive Python build/install
if [[ "$ACTION" == "install" ]]; then
  FLEXKV_DEBUG=$([[ "$BUILD_TYPE" == "debug" ]] && echo 1 || echo 0) pip install -v --no-build-isolation .
elif [[ "$ACTION" == "build" ]]; then
  python -m build
elif [[ "$ACTION" == "develop" ]]; then
  FLEXKV_DEBUG=$([[ "$BUILD_TYPE" == "debug" ]] && echo 1 || echo 0) pip install -v --no-build-isolation -e .
else
  usage; exit 1
fi
