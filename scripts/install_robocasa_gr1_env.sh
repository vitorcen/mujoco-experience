#!/usr/bin/env bash
# Install RoboCasa **GR1 Tabletop** conda env "robocasa_gr1" — idempotent, non-interactive.
#
# This is a SEPARATE env from the kitchen `robocasa` / `robocasa_gr00t` envs because:
#   - The GR1 tabletop repo (robocasa/robocasa-gr1-tabletop-tasks) also installs a
#     package literally named `robocasa` (v0.2.0, pins numpy==1.26.4 / mujoco==3.2.6).
#     It WOULD collide with the kitchen robocasa package if put in the same env.
#   - Its closed-loop eval needs the *current* Isaac-GR00T main, which ships
#     examples/robocasa-gr1-tabletop-tasks/scripts/{inference_service,simulation_service}.py
#     + gr00t.eval.wrappers.video_recording_wrapper. Our existing older
#     dependencies/Isaac-GR00T checkout (kept for N1.7 kitchen training) lacks them,
#     so we clone a fresh tree at dependencies/Isaac-GR00T-gr1.
#
# Follows the upstream GR1 README recipe (Isaac-GR00T + robosuite master + the repo),
# all in one python=3.10 env, then downloads the DigitalCousin tabletop assets.
#
# Run from any CWD inside the repo; we cd to git root.

set -euo pipefail

ENV_NAME="${ROBOCASA_GR1_ENV_NAME:-robocasa_gr1}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

GR00T_DIR="dependencies/Isaac-GR00T-gr1"
GR1_DIR="dependencies/robocasa-gr1-tabletop-tasks"
GR00T_REMOTE="https://github.com/NVIDIA/Isaac-GR00T.git"
GR1_REMOTE="https://github.com/robocasa/robocasa-gr1-tabletop-tasks.git"

echo ">>> REPO_ROOT=$REPO_ROOT  ENV_NAME=$ENV_NAME"

# ---------- 1. locate conda -------------------------------------------------
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "$CONDA_BIN" ]]; then
  echo "ERROR: conda not found in PATH or \$CONDA_EXE" >&2
  exit 1
fi
CONDA_BASE="$("$CONDA_BIN" info --base)"
# shellcheck disable=SC1091
source "$CONDA_BASE/etc/profile.d/conda.sh"

# ---------- 2. create env if missing ----------------------------------------
ENV_DIR="$CONDA_BASE/envs/$ENV_NAME"
if conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo ">>> conda env [$ENV_NAME] exists, skip create"
else
  echo ">>> creating conda env [$ENV_NAME] python=3.10 ..."
  conda create -c conda-forge -n "$ENV_NAME" python=3.10 -y
fi
conda activate "$ENV_NAME"
python -m pip install --upgrade pip setuptools wheel >/dev/null

# ---------- 3. clone repos ---------------------------------------------------
if [[ ! -f "$GR00T_DIR/setup.py" && ! -f "$GR00T_DIR/pyproject.toml" ]]; then
  echo ">>> cloning Isaac-GR00T (current main) -> $GR00T_DIR ..."
  git clone --depth 1 "$GR00T_REMOTE" "$GR00T_DIR"
else
  echo ">>> $GR00T_DIR present, skip clone"
fi
if [[ ! -f "$GR1_DIR/setup.py" ]]; then
  echo ">>> cloning robocasa-gr1-tabletop-tasks -> $GR1_DIR ..."
  git clone --depth 1 "$GR1_REMOTE" "$GR1_DIR"
else
  echo ">>> $GR1_DIR present, skip clone"
fi

# ---------- 4. install Isaac-GR00T -------------------------------------------
# Order matters. gr00t's pyproject pins `flash-attn==2.7.4.post1` as a HARD dep
# but sources its prebuilt wheel via `[tool.uv.sources]` — a uv-only mechanism
# that **pip does not understand**. Under pip, `-e .[base]` therefore tries to
# COMPILE flash-attn from source and dies ("Failed to build flash-attn when
# getting requirements to build wheel"). The earlier segfault was the same step
# choking on a fat one-shot resolve.
#
# Fix: install in dependency order with PREBUILT wheels, no compilation:
#   1. torch 2.5.1+cu124 (the version gr00t's torchcodec==0.4.0 era expects)
#   2. flash-attn 2.7.4.post1 — the torch2.5/cu12/cp310/abiFALSE prebuilt wheel,
#      matching our torch exactly. Downloads ~190MB, unzips, NO nvcc compile,
#      NO GPU use — safe to run alongside a training job.
#   3. gr00t[base] — now sees flash-attn satisfied and skips it.
FA_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

if ! python -c "import torch" 2>/dev/null; then
  echo ">>> installing torch==2.5.1 torchvision==0.20.1 (prebuilt, heavy deps first) ..."
  pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1
fi

# flash-attn prebuilt wheel (matches torch 2.5 / cu12 / cp310 / abiFALSE).
# This is NOT a from-source build — it's a binary wheel, no GPU, ~1 min.
if ! python -c "import flash_attn" 2>/dev/null; then
  echo ">>> installing flash-attn 2.7.4.post1 (prebuilt torch2.5 wheel, no compile) ..."
  pip install --no-cache-dir "$FA_WHEEL" || \
    echo "WARN: flash-attn prebuilt wheel failed — gr00t[base] install will then try to compile it"
fi

if ! python -c "import gr00t" 2>/dev/null; then
  # gr00t is a uv-native package: its pyproject lists deployment-only packages
  # (tensorrt-cu12, tensorrt-cu13, deepspeed, onnx, onnxscript, triton) as HARD
  # deps whose prebuilt wheels come from [tool.uv.sources]. pip can't read that,
  # so `-e .[base]` tries to source-build tensorrt-cu12 and dies. None of those
  # are needed for download / preview / closed-loop eval (§1-§5) — they're for
  # ONNX/TensorRT export and multi-GPU training. So we install the RUNTIME deps
  # (everything except those + already-installed torch/flash-attn), then install
  # the gr00t package itself with --no-deps.
  echo ">>> extracting gr00t runtime deps (skipping tensorrt/deepspeed/onnx/triton) ..."
  python -c "import tomllib" 2>/dev/null || pip install --no-cache-dir -q tomli
  REQ="$(mktemp)"
  python - "$GR00T_DIR/pyproject.toml" > "$REQ" <<'PY'
import sys, re
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib
data = tomllib.load(open(sys.argv[1], "rb"))
SKIP = ("tensorrt", "deepspeed", "onnx", "onnxscript", "triton",
        "flash-attn", "flash_attn", "torch", "torchvision")
for dep in data["project"]["dependencies"]:
    name = re.split(r"[<>=!~ \[;]", dep.strip())[0].lower()
    if name in SKIP:
        continue
    # keep any environment marker (e.g. torchcodec platform guard) intact
    print(dep)
PY
  echo ">>> installing gr00t runtime deps ..."
  pip install --no-cache-dir -r "$REQ"
  echo ">>> pip install --no-deps -e $GR00T_DIR (package body only) ..."
  pip install --no-cache-dir --no-deps -e "$GR00T_DIR"
  rm -f "$REQ"
else
  echo ">>> gr00t importable, skip install"
fi

# ---------- 5. install robosuite (v1.5.1, pinned) ---------------------------
# Upstream setup pins robosuite v1.5.1 (NOT master — master's GR1 controller
# configs reference an unimplemented WHOLE_BODY_MINK_IK class; see kitchen notes).
if ! python -c "import robosuite" 2>/dev/null; then
  echo ">>> installing robosuite v1.5.1 (pinned) ..."
  pip install --no-cache-dir "git+https://github.com/ARISE-Initiative/robosuite.git@v1.5.1"
fi

# ---------- 6. install the GR1 tabletop repo (editable) ----------------------
# Provides the `robocasa` (gr1 fork) package + envs + demo_task.py / playback.
if ! python -c "import robocasa, robocasa.environments.tabletop" 2>/dev/null; then
  echo ">>> pip install -e $GR1_DIR ..."
  pip install -e "$GR1_DIR"
fi

# ---------- 7. download tabletop (DigitalCousin) assets ----------------------
# The script uses non-relative imports (from download_groot_assets import ...),
# so it must run from inside its own scripts/ dir.
ASSET_DIR="$GR1_DIR/robocasa/models/assets/objects"
if [[ -d "$ASSET_DIR/sketchfab" && -d "$ASSET_DIR/lightwheel" ]]; then
  echo ">>> tabletop assets present, skip download"
else
  echo ">>> downloading tabletop DigitalCousin assets (-y) ..."
  ( cd "$GR1_DIR/robocasa/scripts" && python download_tabletop_assets.py -y ) || \
    echo "WARN: asset download exited non-zero; rendering (§2) may miss objects"
fi

# ---------- 8. headless-render hint ------------------------------------------
# Offscreen MuJoCo rendering (demo_task.py / eval video) needs an EGL/OSMesa GL.
# Persist MUJOCO_GL=egl in the activate hook so the notebook stays 1-line.
ACT_DIR="$ENV_DIR/etc/conda/activate.d"
mkdir -p "$ACT_DIR"
cat > "$ACT_DIR/robocasa_gr1.sh" <<'EOF'
# auto-generated by scripts/install_robocasa_gr1_env.sh
# Headless offscreen rendering for demo_task.py / simulation eval videos.
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
EOF
echo ">>> wrote activate hook: $ACT_DIR/robocasa_gr1.sh"

echo ""
echo "=============================================="
echo "RoboCasa GR1 Tabletop env ready: conda activate $ENV_NAME"
echo "Quick test:"
echo "  conda run -n $ENV_NAME python -c 'import robocasa, gr00t, robosuite; print(\"ok\")'"
echo "=============================================="
