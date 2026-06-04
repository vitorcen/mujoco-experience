#!/bin/bash

# MuJoCo Build Script

set -e  # Exit immediately if a command exits with a non-zero status

echo "🏗️  MuJoCo Experience Build Setup"
echo "==================================="

# 0. Activate Conda Environment

CURRENT_ENV=$(echo $CONDA_DEFAULT_ENV)
TARGET_ENV="mujoco"
CONDA_SH=""

# Try to find conda.sh to source it
LOCATIONS=(
    "$HOME/miniconda3/etc/profile.d/conda.sh"
    "$HOME/anaconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
    "/usr/local/miniconda3/etc/profile.d/conda.sh"
)

# If conda command works, try to get base path from it
if command -v conda &> /dev/null; then
    CONDA_BASE_PATH=$(conda info --base 2>/dev/null)
    if [ -n "$CONDA_BASE_PATH" ]; then
        LOCATIONS=("$CONDA_BASE_PATH/etc/profile.d/conda.sh" "${LOCATIONS[@]}")
    fi
fi

# Find the first existing conda.sh
for loc in "${LOCATIONS[@]}"; do
    if [ -f "$loc" ]; then
        CONDA_SH="$loc"
        break
    fi
done

# Source conda.sh if found
if [ -n "$CONDA_SH" ]; then
    # Use . instead of source for better compatibility
    . "$CONDA_SH"
fi

# Check if we need to switch environment
if [ "$CURRENT_ENV" != "$TARGET_ENV" ]; then
    if command -v conda &> /dev/null; then
        # Check if environment exists
        ENV_LIST=$(conda env list | awk '{print $1}')
        
        if echo "$ENV_LIST" | grep -q "^${TARGET_ENV}$"; then
             echo "🔄 Switching to '$TARGET_ENV' environment..."
             conda activate $TARGET_ENV
             if [ $? -eq 0 ]; then
                echo "✅ Activated '$TARGET_ENV'."
             else
                echo "⚠️  Failed to activate '$TARGET_ENV'. Proceeding with current environment..."
             fi
        else
             echo "⚠️  Conda environment '$TARGET_ENV' not found. You might want to run ./init.sh first."
             echo "   Proceeding with current environment..."
        fi
    else
        echo "⚠️  Conda not found or not initialized. Proceeding with current environment..."
    fi
else
    echo "✅ Already in '$TARGET_ENV' environment."
fi

echo "-----------------------------------"

# 1. Check Prerequisites

if ! command -v cmake &> /dev/null; then
    echo "❌ Error: 'cmake' is not installed."
    echo "   Please install it via: sudo apt-get install cmake"
    exit 1
fi

if ! command -v git &> /dev/null; then
    echo "❌ Error: 'git' is not installed."
    exit 1
fi

# 2. Update Submodules

echo "📦 Checking submodules..."
# Check if mujoco dir exists and is not empty
if [ ! -d "dependencies/mujoco" ] || [ -z "$(ls -A dependencies/mujoco)" ]; then
    echo "   Initializing mujoco submodule..."
    git submodule update --init --recursive
else
    # Check if CMakeLists.txt exists, if not, it's likely an incomplete checkout
    if [ ! -f "dependencies/mujoco/CMakeLists.txt" ]; then
        echo "   Updating mujoco submodule..."
        git submodule update --init --recursive
    else
        echo "✅ MuJoCo submodule appears populated."
    fi
fi

# 3. Configure CMake

BUILD_DIR="build"
BUILD_TYPE="Release"

echo "-----------------------------------"
echo "⚙️  Configuring CMake ($BUILD_TYPE)..."

mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Run CMake
# We use the parent directory's CMakeLists.txt
cmake .. -DCMAKE_BUILD_TYPE=$BUILD_TYPE

# 4. Build

echo "-----------------------------------"
echo "🔨 Building project..."

# Get number of cores for parallel build
NUM_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
echo "   Using $NUM_CORES cores..."

cmake --build . --config $BUILD_TYPE --parallel $NUM_CORES

echo "-----------------------------------"
echo "🎉 Build complete!"
echo ""
echo "Executable locations:"
echo "   Simulate:  $BUILD_DIR/dependencies/mujoco/bin/simulate"
echo "   Test libs: $BUILD_DIR/dependencies/mujoco/lib/"
echo ""
echo "To run the simulator (Examples):"
echo "   ./$BUILD_DIR/dependencies/mujoco/bin/simulate ./dependencies/mujoco/model/humanoid/humanoid.xml"
echo "   ./$BUILD_DIR/dependencies/mujoco/bin/simulate ./dependencies/mujoco/model/car/car.xml"
echo "   ./$BUILD_DIR/dependencies/mujoco/bin/simulate ./dependencies/mujoco/model/cards/cards.xml"
echo "   ./$BUILD_DIR/dependencies/mujoco/bin/simulate ./dependencies/mujoco/model/flex/flag.xml"
echo "   ./$BUILD_DIR/dependencies/mujoco/bin/simulate ./dependencies/mujoco/model/mug/mug.xml"
echo ""
