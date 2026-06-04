#!/usr/bin/env bash
# Install RoboCasa GR1 Tabletop **N1.5** eval env "robocasa_gr1_n15" — idempotent.
#
# WHY a SECOND gr1 env (sibling to robocasa_gr1):
#   The downloaded checkpoint (youliangtan/gr00t-n1.5-robocasa-tabletop-posttrain)
#   is **N1.5**, and the upstream gr1 closed-loop recipe ("recipe A") is N1.5-era:
#       server : Isaac-GR00T  scripts/inference_service.py  --data_config fourier_gr1_arms_waist
#       client : Isaac-GR00T  scripts/simulation_service.py --env_name gr1_unified/...
#   That recipe needs `gr00t.eval.simulation` (which does `import robocasa`) + the gr1
#   robocasa fork **co-installed in ONE env**.
#   - The existing `robocasa_gr1` env was built on the *current* Isaac-GR00T main, which
#     has moved on to the **N1.7** eval stack (gr00t/eval/run_gr00t_server.py + a uv venv
#     + ROBOCASA_GR1_TABLETOP embodiment). It has neither `gr00t.eval.simulation` nor
#     `gr00t.eval.robot`, and would need an N1.7 checkpoint we don't have. Version mismatch.
#   - The kitchen `robocasa_gr00t` env HAS the N1.5 gr00t (dependencies/Isaac-GR00T) but
#     must NOT get the gr1 robocasa fork: that fork is also named `robocasa` and pins
#     numpy==1.26.4 / mujoco==3.2.6 — it would collide with the kitchen robocasa and
#     POISON the authoritative N1.5/N1.7/pi0.5 benchmark env. RED LINE: never touch it.
#
# So: a fresh env that pairs the N1.5 gr00t (this repo's dependencies/Isaac-GR00T, the
# "GR00T N1.5 for RoboCasa" tree) with the gr1 robocasa fork + robosuite v1.5.1.
# Reuses the already-cloned repos and already-downloaded tabletop assets.

set -euo pipefail

ENV_NAME="${ROBOCASA_GR1_N15_ENV_NAME:-robocasa_gr1_n15}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

GR00T_DIR="dependencies/Isaac-GR00T"               # N1.5 tree (has gr00t.eval.simulation)
GR1_DIR="dependencies/robocasa-gr1-tabletop-tasks" # gr1 robocasa fork + assets + envs

echo ">>> REPO_ROOT=$REPO_ROOT  ENV_NAME=$ENV_NAME"

# ---------- 0. sanity: required source trees present ------------------------
[[ -f "$GR00T_DIR/pyproject.toml" ]] || { echo "ERROR: $GR00T_DIR missing (N1.5 Isaac-GR00T)" >&2; exit 1; }
[[ -f "$GR1_DIR/setup.py"        ]] || { echo "ERROR: $GR1_DIR missing (run install_robocasa_gr1_env.sh first)" >&2; exit 1; }
[[ -f "$GR00T_DIR/gr00t/eval/simulation.py" ]] || { echo "ERROR: $GR00T_DIR lacks gr00t/eval/simulation.py — not the N1.5 recipe-A tree" >&2; exit 1; }

# ---------- 1. locate conda -------------------------------------------------
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
[[ -n "$CONDA_BIN" ]] || { echo "ERROR: conda not found in PATH or \$CONDA_EXE" >&2; exit 1; }
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

# ---------- 3. torch + flash-attn (prebuilt wheels, no compile, no GPU) -----
# Same stack as robocasa_gr1 — torch 2.5.1/cu124, flash-attn 2.7.4.post1 matching wheel.
FA_WHEEL="https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

if ! python -c "import torch" 2>/dev/null; then
  echo ">>> installing torch==2.5.1 torchvision==0.20.1 (prebuilt) ..."
  pip install --no-cache-dir torch==2.5.1 torchvision==0.20.1
fi
if ! python -c "import flash_attn" 2>/dev/null; then
  echo ">>> installing flash-attn 2.7.4.post1 (prebuilt torch2.5 wheel, no compile) ..."
  pip install --no-cache-dir "$FA_WHEEL" || \
    echo "WARN: flash-attn prebuilt wheel failed — gr00t install may try to compile it"
fi

# ---------- 4. install N1.5 gr00t (runtime deps only, then --no-deps) -------
# Same trick as the gr1 installer: gr00t's pyproject lists deployment-only packages
# (tensorrt/deepspeed/onnx/onnxscript/triton) as hard deps sourced from [tool.uv.sources]
# that pip can't read. None are needed for closed-loop eval. Install runtime deps minus
# those + already-installed torch/flash-attn, then the package body with --no-deps.
if ! python -c "import gr00t" 2>/dev/null; then
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
    print(dep)
PY
  echo ">>> installing gr00t (N1.5) runtime deps ..."
  pip install --no-cache-dir -r "$REQ"
  echo ">>> pip install --no-deps -e $GR00T_DIR ..."
  pip install --no-cache-dir --no-deps -e "$GR00T_DIR"
  rm -f "$REQ"
else
  echo ">>> gr00t importable, skip install"
fi

# ---------- 5. robosuite v1.5.1 (pinned, same as gr1) -----------------------
if ! python -c "import robosuite" 2>/dev/null; then
  echo ">>> installing robosuite v1.5.1 (pinned) ..."
  pip install --no-cache-dir "git+https://github.com/ARISE-Initiative/robosuite.git@v1.5.1"
fi

# ---------- 6. gr1 tabletop robocasa fork (editable) ------------------------
if ! python -c "import robocasa, robocasa.environments.tabletop" 2>/dev/null; then
  echo ">>> pip install -e $GR1_DIR ..."
  pip install -e "$GR1_DIR"
fi

# ---------- 7. assets (shared with robocasa_gr1, usually already present) ---
ASSET_DIR="$GR1_DIR/robocasa/models/assets/objects"
if [[ -d "$ASSET_DIR/sketchfab" && -d "$ASSET_DIR/lightwheel" ]]; then
  echo ">>> tabletop assets present, skip download"
else
  echo ">>> downloading tabletop DigitalCousin assets (-y) ..."
  ( cd "$GR1_DIR/robocasa/scripts" && python download_tabletop_assets.py -y ) || \
    echo "WARN: asset download exited non-zero; rendering may miss objects"
fi

# ---------- 8. headless-render hint (EGL) -----------------------------------
ACT_DIR="$ENV_DIR/etc/conda/activate.d"
mkdir -p "$ACT_DIR"
cat > "$ACT_DIR/robocasa_gr1_n15.sh" <<'EOF'
# auto-generated by scripts/install_robocasa_gr1_n15_env.sh
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
EOF
echo ">>> wrote activate hook: $ACT_DIR/robocasa_gr1_n15.sh"

echo ""
echo "=============================================="
echo "RoboCasa GR1 Tabletop N1.5 eval env ready: conda activate $ENV_NAME"
echo "Verify:"
echo "  conda run -n $ENV_NAME python -c 'import robocasa, gr00t, robosuite; from gr00t.eval.simulation import SimulationInferenceClient; print(\"ok\")'"
echo "=============================================="
