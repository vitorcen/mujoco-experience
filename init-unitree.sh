#!/bin/bash
# Unitree MuJoCo Simulator Installation Script
# Auto-generated installation script for unitree_mujoco project

set -e  # Exit on error

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SUDO_PASS="Abc.123"
MUJOCO_VERSION="3.2.6"
UNITREE_SDK2_DIR="/opt/unitree_robotics"
MUJOCO_DIR="$HOME/.mujoco"
PROJECT_DIR="$HOME/work/mujoco-experience/unitree_mujoco"
RL_GYM_DIR="$HOME/work/mujoco-experience/unitree_rl_gym"
RSL_RL_DIR="$HOME/work/mujoco-experience/rsl_rl"

echo_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

sudo_cmd() {
    echo "$SUDO_PASS" | sudo -S "$@"
}

# Step 1: Install system dependencies
echo_info "Installing system dependencies..."
sudo_cmd apt update
sudo_cmd apt install -y \
    libyaml-cpp-dev \
    libspdlog-dev \
    libboost-all-dev \
    libglfw3-dev \
    git \
    cmake \
    build-essential

# Step 2: Install unitree_sdk2
if [ -d "$UNITREE_SDK2_DIR" ]; then
    echo_warn "unitree_sdk2 already exists at $UNITREE_SDK2_DIR, skipping..."
else
    echo_info "Cloning and building unitree_sdk2..."
    cd /tmp
    rm -rf unitree_sdk2
    git clone https://github.com/unitreerobotics/unitree_sdk2.git
    cd unitree_sdk2
    mkdir -p build && cd build
    cmake .. -DCMAKE_INSTALL_PREFIX=$UNITREE_SDK2_DIR
    make -j$(nproc)
    sudo_cmd make install
    echo_info "unitree_sdk2 installed to $UNITREE_SDK2_DIR"
fi

# Step 3: Download and setup MuJoCo
if [ -d "$MUJOCO_DIR/mujoco-$MUJOCO_VERSION" ]; then
    echo_warn "MuJoCo already exists at $MUJOCO_DIR/mujoco-$MUJOCO_VERSION, skipping..."
else
    echo_info "Downloading MuJoCo $MUJOCO_VERSION..."
    mkdir -p "$MUJOCO_DIR"
    cd "$MUJOCO_DIR"

    MUJOCO_URL="https://github.com/google-deepmind/mujoco/releases/download/${MUJOCO_VERSION}/mujoco-${MUJOCO_VERSION}-linux-x86_64.tar.gz"
    wget -O mujoco.tar.gz "$MUJOCO_URL"
    tar -xzf mujoco.tar.gz
    rm mujoco.tar.gz
    echo_info "MuJoCo extracted to $MUJOCO_DIR/mujoco-$MUJOCO_VERSION"
fi

# Create symlink
echo_info "Creating MuJoCo symlink..."
cd "$PROJECT_DIR/simulate"
if [ -L "mujoco" ]; then
    rm mujoco
fi
ln -s "$MUJOCO_DIR/mujoco-$MUJOCO_VERSION" mujoco

# Step 4: Build unitree_mujoco simulator
echo_info "Building unitree_mujoco simulator..."
cd "$PROJECT_DIR/simulate"
rm -rf build
mkdir build && cd build
cmake ..
make -j$(nproc)

echo_info "Build completed successfully!"

# Step 4.5: Install Python SDK (unitree_sdk2py) - OPTIONAL
echo_warn "Skipping Python SDK installation due to cyclonedds version incompatibility"
echo_info "System cyclonedds (0.10.4) is incompatible with latest cyclonedds-python"
echo_info "Python example requires manual setup. Use C++ examples instead."
echo_info ""
echo_info "If you need Python SDK, try:"
echo_info "  1. Build CycloneDDS from source (latest version)"
echo_info "  2. Or use Docker/Conda environment"
echo_info ""

# Step 5: Build example program
echo_info "Building stand_go2 example..."
cd "$PROJECT_DIR/example/cpp"
rm -rf build
mkdir build && cd build
cmake ..
make -j$(nproc)

# Step 6: Install RL dependencies (rsl_rl)
if [ -d "$RSL_RL_DIR" ]; then
    echo_warn "rsl_rl already exists at $RSL_RL_DIR, skipping..."
else
    echo_info "Installing rsl_rl (RL algorithm library)..."
    cd "$HOME/work/mujoco-experience"
    git clone https://github.com/leggedrobotics/rsl_rl.git
    cd rsl_rl
    git checkout v1.0.2
    pip3 install -e .
    echo_info "rsl_rl v1.0.2 installed"
fi

# Step 7: Setup unitree_rl_gym (Reinforcement Learning environment)
if [ -d "$RL_GYM_DIR" ]; then
    echo_info "unitree_rl_gym exists, verifying submodules..."
    cd "$HOME/work/mujoco-experience"
    git submodule update --init --recursive
else
    echo_warn "unitree_rl_gym not found. Run: git submodule update --init --recursive"
fi

echo_info ""
echo_info "========================================="
echo_info "Installation Complete!"
echo_info "========================================="
echo_info ""
echo_info "To run the simulator:"
echo_info "  cd $PROJECT_DIR/simulate/build"
echo_info "  ./unitree_mujoco -r go2 -s scene_terrain.xml"
echo_info ""
echo_info "To run example (in another terminal):"
echo_info "  cd $PROJECT_DIR/example/cpp/build"
echo_info "  ./stand_go2"
echo_info ""
echo_info "See readme_zh.md for more usage examples."
