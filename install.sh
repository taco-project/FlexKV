#!/bin/bash
# =============================================================================
# FlexKV One-Click Install Script
# =============================================================================
# Usage:
#   bash install.sh [OPTIONS]
#
# Options:
#   --venv PATH       Specify virtual environment path (default: ./venv)
#   --no-venv         Skip virtual environment creation, install directly
#   --release         Build in release mode (with Cython compilation)
#   --debug           Build in debug mode (default, no Cython)
#   --enable-metrics  Enable Prometheus monitoring support
#   --enable-p2p      Enable distributed P2P/Redis support (default: enabled)
#   --disable-p2p     Disable distributed P2P/Redis support
#   --enable-gds      Enable GDS support
#   --enable-cfs      Enable CFS support
#   --skip-deps       Skip system dependency installation
#   --clean           Clean all build artifacts and exit
#   -h, --help        Show this help message
# =============================================================================
set -e

# ======================== Default Configuration ========================
VENV_PATH="./venv"
USE_VENV=1
BUILD_TYPE="debug"
ENABLE_METRICS=0
ENABLE_P2P=1
ENABLE_GDS=0
ENABLE_CFS=0
SKIP_DEPS=0
CLEAN_ONLY=0

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# ======================== Helper Functions ========================
info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
success() { echo -e "${GREEN}[OK]${NC} $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC} $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

usage() {
    head -n 17 "$0" | tail -n 14 | sed 's/^# \?//'
    exit 0
}

# ======================== Parse Arguments ========================
while [[ $# -gt 0 ]]; do
    case "$1" in
        --venv)
            VENV_PATH="$2"
            USE_VENV=1
            shift 2
            ;;
        --no-venv)
            USE_VENV=0
            shift
            ;;
        --release)
            BUILD_TYPE="release"
            shift
            ;;
        --debug)
            BUILD_TYPE="debug"
            shift
            ;;
        --enable-metrics)
            ENABLE_METRICS=1
            shift
            ;;
        --enable-p2p)
            ENABLE_P2P=1
            shift
            ;;
        --disable-p2p)
            ENABLE_P2P=0
            shift
            ;;
        --enable-gds)
            ENABLE_GDS=1
            shift
            ;;
        --enable-cfs)
            ENABLE_CFS=1
            shift
            ;;
        --skip-deps)
            SKIP_DEPS=1
            shift
            ;;
        --clean)
            CLEAN_ONLY=1
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            warn "Unknown option: $1"
            shift
            ;;
    esac
done

# ======================== Project Root ========================
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"
info "Project root: $PROJECT_ROOT"

# ======================== Clean Mode ========================
if [ "$CLEAN_ONLY" -eq 1 ]; then
    info "Cleaning all build artifacts..."
    bash build.sh --clean
    if [ -d "$VENV_PATH" ]; then
        rm -rf "$VENV_PATH"
        info "Removed virtual environment: $VENV_PATH"
    fi
    success "Clean completed."
    exit 0
fi

# ======================== Step 1: Check System Dependencies ========================
info "============================================"
info "Step 1: Checking system dependencies"
info "============================================"

check_command() {
    if command -v "$1" &>/dev/null; then
        success "$1 found: $(command -v "$1")"
        return 0
    else
        warn "$1 not found"
        return 1
    fi
}

MISSING_CMDS=()
MISSING_PKGS=()

# Check essential commands
check_command python3 || MISSING_CMDS+=("python3")
check_command cmake   || { MISSING_CMDS+=("cmake"); MISSING_PKGS+=("cmake"); }
check_command git     || { MISSING_CMDS+=("git"); MISSING_PKGS+=("git"); }
check_command gcc     || { MISSING_CMDS+=("gcc"); MISSING_PKGS+=("build-essential"); }
check_command g++     || { MISSING_CMDS+=("g++"); MISSING_PKGS+=("build-essential"); }

# Check python3-venv availability (test with a real temporary venv to catch missing ensurepip)
if [ "$USE_VENV" -eq 1 ]; then
    _VENV_TEST_DIR=$(mktemp -d)
    if ! python3 -m venv "$_VENV_TEST_DIR/test_venv" &>/dev/null 2>&1; then
        rm -rf "$_VENV_TEST_DIR"
        warn "python3-venv not available (ensurepip missing)"
        PY_MINOR=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        MISSING_PKGS+=("python3.${PY_MINOR#3.}-venv" "python3-venv" "python3-full")
    else
        rm -rf "$_VENV_TEST_DIR"
    fi
fi

# Check for liburing-dev (required by setup.py: -luring)
if ! dpkg -s liburing-dev &>/dev/null 2>&1 && ! rpm -q liburing-devel &>/dev/null 2>&1; then
    if [ -f /etc/debian_version ]; then
        warn "liburing-dev not found"
        MISSING_PKGS+=("liburing-dev")
    elif [ -f /etc/redhat-release ]; then
        warn "liburing-devel not found"
        MISSING_PKGS+=("liburing-devel")
    fi
fi

# Check for hiredis if P2P enabled
if [ "$ENABLE_P2P" -eq 1 ]; then
    if ! dpkg -s libhiredis-dev &>/dev/null 2>&1 && ! rpm -q hiredis-devel &>/dev/null 2>&1; then
        if [ -f /etc/debian_version ]; then
            MISSING_PKGS+=("libhiredis-dev")
        elif [ -f /etc/redhat-release ]; then
            MISSING_PKGS+=("hiredis-devel")
        fi
    fi
fi

# Install missing packages
if [ ${#MISSING_PKGS[@]} -gt 0 ] && [ "$SKIP_DEPS" -eq 0 ]; then
    # Deduplicate
    UNIQUE_PKGS=($(echo "${MISSING_PKGS[@]}" | tr ' ' '\n' | sort -u | tr '\n' ' '))
    info "Installing missing packages: ${UNIQUE_PKGS[*]}"

    if command -v apt-get &>/dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y -qq "${UNIQUE_PKGS[@]}"
    elif command -v yum &>/dev/null; then
        sudo yum install -y "${UNIQUE_PKGS[@]}"
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y "${UNIQUE_PKGS[@]}"
    else
        error "Cannot auto-install packages. Please manually install: ${UNIQUE_PKGS[*]}"
    fi
    success "System packages installed."
elif [ ${#MISSING_PKGS[@]} -gt 0 ] && [ "$SKIP_DEPS" -eq 1 ]; then
    warn "Skipping dependency installation (--skip-deps). Missing: ${MISSING_PKGS[*]}"
fi

# Final check for critical commands
for cmd in python3 cmake git gcc g++; do
    command -v "$cmd" &>/dev/null || error "$cmd is still not available. Please install it manually."
done

# Check NVIDIA CUDA toolkit
if ! command -v nvcc &>/dev/null; then
    warn "nvcc not found. CUDA toolkit is required for building FlexKV."
    warn "Please install CUDA toolkit from: https://developer.nvidia.com/cuda-downloads"
    warn "Or load it via: module load cuda"
fi

success "System dependencies check passed."

# ======================== Step 2: Setup Python Virtual Environment ========================
info "============================================"
info "Step 2: Setting up Python environment"
info "============================================"

if [ "$USE_VENV" -eq 1 ]; then
    if [ -d "$VENV_PATH" ] && [ -f "$VENV_PATH/bin/activate" ]; then
        info "Using existing virtual environment: $VENV_PATH"
    else
        info "Creating virtual environment at: $VENV_PATH"
        python3 -m venv "$VENV_PATH"
        success "Virtual environment created."
    fi

    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    success "Virtual environment activated: $(which python3)"

    # Upgrade pip
    info "Upgrading pip..."
    pip install --upgrade pip -q
else
    warn "Skipping virtual environment (--no-venv). Installing to system Python."
    warn "If you encounter 'externally-managed-environment' error, use --venv instead."
fi

# Install Python build dependencies
info "Installing Python build dependencies..."
pip install -q setuptools wheel
if [ "$BUILD_TYPE" = "release" ]; then
    pip install -q "Cython>=3.0.10"
fi

# Check if torch is installed
if ! python3 -c "import torch" &>/dev/null 2>&1; then
    warn "PyTorch not found. Installing PyTorch..."
    warn "If you need a specific CUDA version, please install PyTorch manually first."
    pip install torch
fi
success "Python environment ready."

# ======================== Step 3: Initialize Git Submodules ========================
info "============================================"
info "Step 3: Initializing git submodules"
info "============================================"

if [ "$ENABLE_METRICS" -eq 1 ]; then
    info "Metrics enabled: initializing all submodules (including prometheus-cpp)..."
    git submodule update --init --recursive
else
    info "Metrics disabled: initializing only xxHash submodule..."
    git submodule update --init --recursive third_party/xxHash
fi
success "Git submodules initialized."

# ======================== Step 4: Build C++ Libraries ========================
info "============================================"
info "Step 4: Building C++ libraries (CMake)"
info "============================================"

mkdir -p build
cd build

CMAKE_ARGS=""
if [ "$ENABLE_METRICS" -eq 0 ]; then
    CMAKE_ARGS="-DFLEXKV_ENABLE_MONITORING=OFF"
fi

info "Running CMake configuration..."
cmake .. $CMAKE_ARGS

info "Building C++ libraries..."
cmake --build . -j"$(nproc)"

BUILD_LIB_PATH="$(pwd)/lib"
cd "$PROJECT_ROOT"

# Set LD_LIBRARY_PATH
export LD_LIBRARY_PATH="$BUILD_LIB_PATH:$LD_LIBRARY_PATH"

# Copy shared libraries to package directory
info "Copying shared libraries to package directory..."
PACKAGE_LIB_DIR="flexkv/lib"
mkdir -p "$PACKAGE_LIB_DIR"
if [ -d "$BUILD_LIB_PATH" ]; then
    for lib_file in "$BUILD_LIB_PATH"/*.so*; do
        if [ -f "$lib_file" ]; then
            cp "$lib_file" "$PACKAGE_LIB_DIR/"
        fi
    done
fi
success "C++ libraries built successfully."

# ======================== Step 4.5: Install Python Runtime Dependencies ========================
info "============================================"
info "Step 4.5: Installing Python runtime dependencies"
info "============================================"

# Core runtime dependencies (always needed)
RUNTIME_DEPS="numpy pyzmq psutil nvtx pyyaml expiring-dict"

# Additional dependencies for P2P/distributed mode
if [ "$ENABLE_P2P" -eq 1 ]; then
    RUNTIME_DEPS="$RUNTIME_DEPS redis"
fi

info "Installing runtime dependencies: $RUNTIME_DEPS"
pip install -q $RUNTIME_DEPS
success "Python runtime dependencies installed."

# ======================== Step 5: Install Python Package ========================
info "============================================"
info "Step 5: Installing FlexKV Python package"
info "============================================"

# Set environment variables for build
export FLEXKV_ENABLE_METRICS="$ENABLE_METRICS"
export FLEXKV_ENABLE_P2P="$ENABLE_P2P"
export FLEXKV_ENABLE_GDS="$ENABLE_GDS"
export FLEXKV_ENABLE_CFS="$ENABLE_CFS"

if [ "$BUILD_TYPE" = "debug" ]; then
    export FLEXKV_DEBUG=1
    info "Installing in debug mode (editable, no Cython)..."
    pip install -v --no-build-isolation -e .
elif [ "$BUILD_TYPE" = "release" ]; then
    export FLEXKV_DEBUG=0
    info "Building release wheel..."
    python3 setup.py bdist_wheel -v
    # Install the built wheel
    WHEEL_FILE=$(ls dist/flexkv-*.whl 2>/dev/null | head -n 1)
    if [ -n "$WHEEL_FILE" ]; then
        pip install "$WHEEL_FILE"
    else
        error "Wheel file not found in dist/"
    fi
fi
success "FlexKV Python package installed."

# ======================== Step 6: Verify Installation ========================
info "============================================"
info "Step 6: Verifying installation"
info "============================================"

python3 -c "
import flexkv
print('FlexKV imported successfully')
try:
    print(f'Version: {flexkv.__version__}')
except AttributeError:
    pass
try:
    from flexkv import c_ext
    print('C extension loaded successfully')
except ImportError as e:
    print(f'Warning: C extension not loaded: {e}')
" && success "FlexKV installation verified!" || warn "Verification had warnings, see above."

# ======================== Summary ========================
echo ""
info "============================================"
success "FlexKV installation completed!"
info "============================================"
echo ""
info "Build type:       $BUILD_TYPE"
info "Metrics:          $([ $ENABLE_METRICS -eq 1 ] && echo 'Enabled' || echo 'Disabled')"
info "P2P/Redis:        $([ $ENABLE_P2P -eq 1 ] && echo 'Enabled' || echo 'Disabled')"
info "GDS:              $([ $ENABLE_GDS -eq 1 ] && echo 'Enabled' || echo 'Disabled')"
info "CFS:              $([ $ENABLE_CFS -eq 1 ] && echo 'Enabled' || echo 'Disabled')"

if [ "$USE_VENV" -eq 1 ]; then
    VENV_ABS_PATH="$(cd "$VENV_PATH" && pwd)"
    echo ""
    info "Virtual environment: $VENV_ABS_PATH"
    info "To activate it in a new terminal, run:"
    echo ""
    echo "    source $VENV_ABS_PATH/bin/activate"
    echo ""
fi
