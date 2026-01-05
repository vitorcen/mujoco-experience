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
        conda run -n $TARGET_ENV python -m pip "$@"
    else
        python -m pip "$@"
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
     run_pip install -U torch torchvision torchaudio
fi

echo "📦 Checking VLA (OpenVLA) dependencies..."
# OpenVLA 对部分依赖版本有硬性要求，尤其是 timm：
# - timm 必须满足 >=0.9.10 且 <1.0.0（timm 1.x 会直接报错）
# 同时，OpenVLA 的 remote-code 对 transformers 的新版本也可能不兼容。
# 为了“能跑起来”，这里固定到一个已知更稳的组合。

echo "🔧 Ensuring compatible versions: timm>=0.9.10,<1.0.0 ; transformers==4.40.1 ; tokenizers==0.19.1"
run_pip install -U \
  "timm>=0.9.10,<1.0.0" \
  "transformers==4.40.1" \
  "tokenizers==0.19.1" \
  "accelerate>=0.26.0" \
  "protobuf" \
  "sentencepiece" \
  "pillow"

# bitsandbytes：用于 4-bit 量化（可选，但建议安装）
run_pip install -U "bitsandbytes>=0.43.0"

if ! check_import "diffusers"; then
    echo "⚠️  'diffusers' not found. Installing..."
    run_pip install -U diffusers
fi

# PyQuaternion (for DeepMimic)
if ! check_import "pyquaternion"; then
    echo "⚠️  'pyquaternion' not found. Installing..."
    run_pip install pyquaternion
fi

# Jupyter & IPython Kernel (for .ipynb notebooks)
echo "📓 Installing Jupyter & IPython Kernel..."
run_pip install jupyter ipykernel

# Configure Jupyter Kernel
echo "🔗 Registering Jupyter Kernel..."
if [ "$USE_CONDA_RUN" = true ]; then
    conda run -n $TARGET_ENV python -m ipykernel install --user --name=$TARGET_ENV --display-name "Python ($TARGET_ENV)" > /dev/null 2>&1 || true
else
    python -m ipykernel install --user --name=$TARGET_ENV --display-name "Python ($TARGET_ENV)" > /dev/null 2>&1 || true
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

if ! ldconfig -p | grep -q libOSMesa; then
     echo "⚠️  'libOSMesa' not found. Recommended for software rendering (headless)."
     echo "   Install via: sudo apt-get install libosmesa6 libosmesa6-dev"
fi

if ! ldconfig -p | grep -q libglapi; then
     echo "⚠️  'libgl1-mesa-dri' (Mesa drivers) might be missing."
     echo "   Install via: sudo apt-get install libgl1-mesa-dri libglx-mesa0 mesa-utils"
fi

echo "---------------------------------------"
echo "🖥️  Checking MuJoCo OpenGL backends (GLX/EGL/OSMesa)..."

# 说明：
# - 你遇到的 `Xlib: extension "NV-GLX" missing` 属于 GLX/显示环境问题，会导致 mujoco.viewer 无法启动。
# - `MUJOCO_GL=osmesa` 需要系统里有 libOSMesa；否则会在 import 阶段报错。
# - `MUJOCO_GL=egl` 通常更适合无显示环境/服务器做离屏渲染（录视频）。

if [ "$USE_CONDA_RUN" = true ]; then
    conda run -n $TARGET_ENV python - <<'PY'
import ctypes.util
print("OSMesa ->", ctypes.util.find_library("OSMesa"))
print("EGL    ->", ctypes.util.find_library("EGL"))
print("GLX    ->", ctypes.util.find_library("GLX"))
PY
else
    python - <<'PY'
import ctypes.util
print("OSMesa ->", ctypes.util.find_library("OSMesa"))
print("EGL    ->", ctypes.util.find_library("EGL"))
print("GLX    ->", ctypes.util.find_library("GLX"))
PY
fi

echo ""
echo "如果你需要在无桌面环境渲染/录视频，推荐 EGL："
echo "  MUJOCO_GL=egl python scripts/go2_terrain_demo.py --headless --record out.mp4"
echo ""
echo "如果你一定要用 OSMesa（软件渲染），需要系统安装 libOSMesa："
echo "  sudo apt-get update && sudo apt-get install -y libosmesa6 libosmesa6-dev"
echo ""
echo "如果你要在桌面开 viewer，但遇到 GLX 问题，可尝试安装 Mesa GLX（软件 GLX）："
echo "  sudo apt-get update && sudo apt-get install -y libgl1-mesa-dri libglx-mesa0 mesa-utils"
echo "  __GLX_VENDOR_LIBRARY_NAME=mesa LIBGL_ALWAYS_SOFTWARE=1 python scripts/go2_terrain_demo.py"

echo "---------------------------------------"
echo "🎉 Setup complete!"
echo ""
echo "To verify installation, try running:"
if [ "$USE_CONDA_RUN" = true ]; then
    echo "  conda activate $TARGET_ENV"
fi
echo "  python -c 'import mujoco; print(f\"MuJoCo version: {mujoco.__version__}\")'"
echo ""
