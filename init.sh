#!/bin/bash

# MuJoCo Experience Initialization Script

echo "🏗️  MuJoCo Experience Environment Setup"
echo "======================================="

# 1. Check/Activate Conda Environment

CURRENT_ENV=$(echo $CONDA_DEFAULT_ENV)
TARGET_ENV="mujoco"

# Function to attempt to locate and source conda
source_conda() {
    # Try common locations
    LOCATIONS=(
        "$HOME/miniconda3/etc/profile.d/conda.sh"
        "$HOME/anaconda3/etc/profile.d/conda.sh"
        "/opt/conda/etc/profile.d/conda.sh"
        "/usr/local/miniconda3/etc/profile.d/conda.sh"
    )
    
    # Try to find conda base via command if available
    if command -v conda &> /dev/null; then
        CONDA_BASE=$(conda info --base 2>/dev/null)
        if [ -n "$CONDA_BASE" ]; then
            LOCATIONS=("$CONDA_BASE/etc/profile.d/conda.sh" "${LOCATIONS[@]}")
        fi
    fi

    for loc in "${LOCATIONS[@]}"; do
        if [ -f "$loc" ]; then
            source "$loc"
            return 0
        fi
    done
    return 1
}

if command -v conda &> /dev/null; then
    # Check if environment exists
    ENV_LIST=$(conda env list | awk '{print $1}')
    
    if echo "$ENV_LIST" | grep -q "^${TARGET_ENV}$"; then
        echo "✅ Conda environment '$TARGET_ENV' found."
    else
        echo "⚠️  Conda environment '$TARGET_ENV' not found. Creating it..."
        conda create -n $TARGET_ENV python=3.10 -y
        echo "✅ Created environment '$TARGET_ENV'."
    fi

    # Activate if not already active
    if [ "$CURRENT_ENV" != "$TARGET_ENV" ]; then
        echo "🔄 Switching to '$TARGET_ENV' environment..."
        source_conda
        conda activate $TARGET_ENV
        
        if [ $? -eq 0 ]; then
            echo "✅ Activated '$TARGET_ENV'."
        else
            echo "❌ Failed to activate '$TARGET_ENV'. Please run 'conda activate $TARGET_ENV' manually after this script."
            echo "   Proceeding with installation commands using 'conda run'..."
            USE_CONDA_RUN=true
        fi
    else
        echo "✅ Already in '$TARGET_ENV' environment."
    fi
else
    echo "❌ Conda command not found. Please install Anaconda or Miniconda first."
    exit 1
fi

echo "---------------------------------------"

# 2. Check and Install Python Dependencies

echo "📦 Checking Python dependencies..."

# Helper to run pip in the correct environment
run_pip() {
    if [ "$USE_CONDA_RUN" = true ]; then
        conda run -n $TARGET_ENV pip "$@"
    else
        pip "$@"
    fi
}

# Helper to check import in the correct environment
check_import() {
    local module=$1
    if [ "$USE_CONDA_RUN" = true ]; then
        conda run -n $TARGET_ENV python -c "import $module" 2>/dev/null
    else
        python -c "import $module" 2>/dev/null
    fi
    return $?
}

# MuJoCo Python Bindings
if ! check_import "mujoco"; then
    echo "⚠️  'mujoco' not found. Installing..."
    run_pip install mujoco
else
    echo "✅ 'mujoco' is installed."
fi

# MuJoCo Python Viewer (for interactive visualization in Python)
if ! check_import "mujoco_viewer" && ! check_import "mujoco.viewer"; then
    # Note: mujoco 2.3.6+ has native viewer in mujoco.viewer, but we might want the standalone package too
    # or just rely on native. Native is better.
    # But for compatibility with some older examples, maybe install mujoco-python-viewer?
    # Let's stick to native mujoco which is installed above. 
    # But let's check for numpy explicitly as it's essential.
    echo "ℹ️  Using native 'mujoco.viewer' included in mujoco package."
else
    echo "✅ 'mujoco.viewer' is available."
fi

# NumPy
if ! check_import "numpy"; then
    echo "⚠️  'numpy' not found. Installing..."
    run_pip install numpy
else
    echo "✅ 'numpy' is installed."
fi

# ImageIO (often used for saving videos)
if ! check_import "imageio"; then
    echo "⚠️  'imageio' not found. Installing..."
    run_pip install imageio
else
    echo "✅ 'imageio' is installed."
fi

# Matplotlib (for plotting)
if ! check_import "matplotlib"; then
    echo "⚠️  'matplotlib' not found. Installing..."
    run_pip install matplotlib
else
    echo "✅ 'matplotlib' is installed."
fi

# Transformers & Pillow & Torch (for VLA Demo)
if ! check_import "torch"; then
     echo "⚠️  'torch' not found. Installing..."
     run_pip install torch torchvision torchaudio
fi

if ! check_import "transformers"; then
    echo "⚠️  'transformers' not found. Installing..."
    run_pip install transformers pillow
fi

echo "---------------------------------------"

# 3. Check System Dependencies (Optional)

echo "🛠️  Checking system tools..."

if command -v cmake &> /dev/null; then
    echo "✅ 'cmake' is installed ($(cmake --version | head -n1))."
else
    echo "⚠️  'cmake' not found. Required if you want to build C++ examples."
    echo "   Install via: sudo apt-get install cmake"
fi

if command -v ffmpeg &> /dev/null; then
    echo "✅ 'ffmpeg' is installed."
else
    echo "⚠️  'ffmpeg' not found. Recommended for video rendering."
    echo "   Install via: sudo apt-get install ffmpeg"
fi

echo "---------------------------------------"
echo "🎉 Setup complete!"
echo ""
echo "To verify installation, try running:"
if [ "$USE_CONDA_RUN" = true ]; then
    echo "  conda activate $TARGET_ENV"
fi
echo "  python -c 'import mujoco; print(f\"MuJoCo version: {mujoco.__version__}\")'"
echo ""
