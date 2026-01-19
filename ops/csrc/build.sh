#!/bin/bash
# Build script for miniinfer_ext CMake project

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
BUILD_TYPE="${1:-Release}"

echo "=========================================="
echo "Building MiniInfer Extension"
echo "Build Type: ${BUILD_TYPE}"
echo "=========================================="

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with CMake
echo ""
echo ">>> Configuring CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DBUILD_SHARED_LIBS=ON

# Build
echo ""
echo ">>> Building..."
cmake --build . --config ${BUILD_TYPE} -j$(nproc)

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "Python extension: ${SCRIPT_DIR}/miniinfer_ext/_ext.so"
echo "=========================================="
