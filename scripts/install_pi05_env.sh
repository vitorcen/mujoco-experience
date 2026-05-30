#!/usr/bin/env bash
# Install π0.5 inference env (JAX/Orbax/openpi) — idempotent.
#
# Why a separate env: π0.5 ships only as a JAX/Orbax checkpoint, which needs
# openpi + JAX 0.5.3 + Flax 0.10.2 + orbax-checkpoint 0.11.13. These conflict
# with robocasa (numpy 2.2.5 + mujoco 3.3.1). The eval pipeline is two-process:
# this env hosts the inference server, the existing `robocasa` env hosts the
# sim client.
#
# Layout:
#   - clone path:       dependencies/openpi  (Physical-Intelligence/openpi)
#   - uv venv:          dependencies/openpi/.venv  (managed by uv sync)
#   - checkpoint path:  $ROBOCASA_DATA_PATH/checkpoints/pi05_pretrain_human300/...

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

OPENPI_DIR="dependencies/openpi"

export ROBOCASA_DATA_PATH="${ROBOCASA_DATA_PATH:-$HOME/.cache/robocasa}"

# ---------- 1. uv (Python env manager) ---------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  echo ">>> installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# ---------- 2. init openpi submodule + its third_party submodules ------------
if [[ ! -f "$OPENPI_DIR/pyproject.toml" ]]; then
  echo ">>> init submodule $OPENPI_DIR ..."
  GIT_LFS_SKIP_SMUDGE=1 git submodule update --init "$OPENPI_DIR"
fi
if [[ ! -f "$OPENPI_DIR/third_party/aloha/pyproject.toml" ]]; then
  echo ">>> init openpi's third_party submodules (aloha, libero) ..."
  (cd "$OPENPI_DIR" && GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --recursive)
fi

# ---------- 3. uv sync + editable install ------------------------------------
cd "$OPENPI_DIR"
if [[ ! -f ".venv/bin/python" ]]; then
  echo ">>> uv sync openpi env (JAX 0.5.3 cuda12 + flax + orbax, ~5 min) ..."
  GIT_LFS_SKIP_SMUDGE=1 uv sync --no-dev
  echo ">>> uv pip install -e . ..."
  GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
fi

# ---------- 4. smoke check ---------------------------------------------------
if uv run --no-dev python -c "import jax, openpi; assert jax.devices()[0].platform == 'gpu'" 2>/dev/null; then
  echo ">>> openpi env OK (JAX sees GPU)"
else
  echo "WARN: JAX did not detect a GPU. Inference will run on CPU (very slow)."
fi

# zmq for the inference protocol — uv venv may already have it via deps; install if not.
uv pip install --quiet pyzmq 2>/dev/null || true

# pyzmq in robocasa env too (client side)
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -n "$CONDA_BIN" ]]; then
  CONDA_BASE="$("$CONDA_BIN" info --base)"
  # shellcheck disable=SC1091
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  if conda env list | awk '{print $1}' | grep -qx "robocasa"; then
    conda activate robocasa
    pip install --quiet pyzmq 2>/dev/null || true
    conda deactivate
  fi
fi

echo ""
echo "=============================================="
echo "π0.5 inference env ready."
echo "  server env: dependencies/openpi/.venv (uv-managed)"
echo "  client env: robocasa (sim + zmq)"
echo "Next: download the ckpt + run a preview from RoboCasa.ipynb §2.5"
echo "=============================================="
