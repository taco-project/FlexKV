#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Configure with CMake
echo "=== Running CMake configuration ==="
cmake ..

# Build third-party libraries
echo "=== Building third-party libraries ==="
cmake --build .

# Return to the project root
cd ..

# Install the package with pip
echo "=== Installing package with pip ==="
pip install --no-build-isolation -e .

echo "=== Build and installation completed successfully ==="