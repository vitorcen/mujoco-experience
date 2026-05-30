"""Orchestrator: run π0.5 (official RoboCasa ckpt) against a single RoboCasa env.

Server runs in the `openpi-experience` uv env (JAX/Orbax/openpi); client runs
in `robocasa` conda env. Communicate over ZMQ + pickle.

Usage:
    python scripts/robocasa_eval_pi05.py --env-name OpenCabinet --n-episodes 5
"""
import argparse
import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
# Prefer the in-repo submodule + .venv created by scripts/install_pi05_env.sh;
# fall back to the sibling openpi-experience checkout if the local .venv isn't
# populated yet (dev machine where openpi was set up first).
_LOCAL = REPO_ROOT / "dependencies" / "openpi"
_SIBLING = Path("/home/david/work/openpi-experience/openpi")
OPENPI_ROOT = _LOCAL if (_LOCAL / ".venv" / "bin" / "python").exists() else _SIBLING
DEFAULT_CKPT = "checkpoints/pi05_pretrain_human300/multitask_learning/75000"


def _abs_ckpt(rel):
    base = os.environ.get("ROBOCASA_DATA_PATH", os.path.expanduser("~/.cache/robocasa"))
    return os.path.join(base, rel)


def _find_free_port(preferred):
    s = socket.socket()
    try:
        s.bind(("", preferred))
        s.close()
        return preferred
    except OSError:
        s.close()
    s = socket.socket()
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_port_open(host, port, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False


def _conda_cmd(env_name, py_cmd):
    return (
        f"source $(conda info --base)/etc/profile.d/conda.sh && "
        f"conda activate {env_name} && "
        f"export PYTHONDONTWRITEBYTECODE=1 && "
        f"exec python -u {py_cmd}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", default="OpenCabinet")
    ap.add_argument("--split", default="target", choices=["pretrain", "target"])
    ap.add_argument("--n-episodes", type=int, default=2)
    ap.add_argument("--n-action-steps", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT,
                    help="ckpt dir under ROBOCASA_DATA_PATH (or abs path)")
    ap.add_argument("--port", type=int, default=5557)
    ap.add_argument("--server-warmup-s", type=int, default=300,
                    help="how long to wait for JAX to compile + load (~60-120s typical)")
    ap.add_argument("--client-env", default="robocasa")
    ap.add_argument("--results-path", default=None)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--render-warmup-s", type=float, default=5.0)
    ap.add_argument("--discrete-state-input", action="store_true",
                    help="pi05 native default; libero/droid use False. Try if SR=0.")
    args = ap.parse_args()

    ckpt = args.ckpt if os.path.isabs(args.ckpt) else _abs_ckpt(args.ckpt)
    if not os.path.isdir(ckpt):
        sys.exit(f"ckpt dir not found: {ckpt}")

    port = _find_free_port(args.port)
    if port != args.port:
        print(f"[orch] port {args.port} busy, using {port}", flush=True)

    # Server runs via uv inside openpi-experience to inherit the JAX env.
    server_py = (
        f"{REPO_ROOT}/scripts/_pi05_inference_server.py "
        f"--checkpoint-dir {shlex.quote(ckpt)} "
        f"--port {port}"
    )
    if args.discrete_state_input:
        server_py += " --discrete-state-input"
    # XLA autotuner triggers a ptxas SIGSEGV on RTX 4090 + JAX 0.5.3 cuda12
    # wheel during the gemm_fusion autotune pass — disable to get past JIT
    # compile. Costs a few % on throughput but inference is already <200ms.
    server_cmd = ["bash", "-c",
                  f"cd {OPENPI_ROOT} && "
                  f"export XLA_FLAGS=--xla_gpu_autotune_level=0 && "
                  f"exec uv run --no-dev python -u {server_py}"]

    client_py = (
        f"{REPO_ROOT}/scripts/_pi05_eval_client.py "
        f"--env-name {args.env_name} "
        f"--split {args.split} "
        f"--n-episodes {args.n_episodes} "
        f"--n-action-steps {args.n_action_steps} "
        f"--max-steps {args.max_steps} "
        f"--port {port}"
    )
    if args.results_path:
        client_py += f" --results-path {shlex.quote(args.results_path)}"
    if args.render:
        client_py += f" --render --render-warmup-s {args.render_warmup_s}"
    client_cmd = ["bash", "-c", _conda_cmd(args.client_env, client_py)]

    print(f"[orch] starting pi05 server (openpi/JAX) ...", flush=True)
    server = subprocess.Popen(server_cmd, stdout=sys.stdout, stderr=sys.stderr,
                              preexec_fn=os.setsid)

    try:
        if not _wait_port_open("localhost", port, args.server_warmup_s):
            server.terminate()
            sys.exit(f"[orch] server failed to bind port {port} within "
                     f"{args.server_warmup_s}s — check server logs above")
        print(f"[orch] server up on port {port}; starting client ...", flush=True)
        ret = subprocess.call(client_cmd)
    finally:
        print("[orch] tearing down server ...", flush=True)
        try:
            os.killpg(os.getpgid(server.pid), signal.SIGTERM)
            server.wait(timeout=10)
        except Exception:
            try:
                os.killpg(os.getpgid(server.pid), signal.SIGKILL)
            except Exception:
                pass

    sys.exit(ret)


if __name__ == "__main__":
    main()
