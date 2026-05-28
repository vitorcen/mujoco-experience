"""Orchestrator: run a GR00T N1.5 policy against a single RoboCasa env.

Spawns the inference server in the `robocasa_gr00t` conda env and the sim
client in the `robocasa` env, both as subprocesses communicating over ZMQ.

Usage:
    python scripts/robocasa_eval_gr00t.py --env-name CloseFridge --n-episodes 3

Designed to be called via `!python ...` from the notebook (so the calling
python env doesn't need any of the gr00t/robocasa dependencies).
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
DEFAULT_CKPT = "checkpoints/gr00t_n1-5/multitask_learning/checkpoint-120000"


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


def _conda_cmd(env_name, py_cmd):
    """Build a bash command that activates a conda env then runs python."""
    return (
        f"source $(conda info --base)/etc/profile.d/conda.sh && "
        f"conda activate {env_name} && "
        f"export PYTHONDONTWRITEBYTECODE=1 && "
        f"exec python -u {py_cmd}"
    )


def _wait_port_open(host, port, timeout_s):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=1):
                return True
        except OSError:
            time.sleep(1)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", default="CloseFridge")
    ap.add_argument("--split", default="pretrain", choices=["pretrain", "target"])
    ap.add_argument("--n-episodes", type=int, default=2)
    ap.add_argument("--n-action-steps", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--ckpt", default=DEFAULT_CKPT,
                    help="checkpoint dir, relative to ROBOCASA_DATA_PATH")
    ap.add_argument("--data-config", default="panda_omron")
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--server-warmup-s", type=int, default=180,
                    help="how long to wait for the server to bind the port")
    ap.add_argument("--server-env", default="robocasa_gr00t")
    ap.add_argument("--client-env", default="robocasa")
    ap.add_argument("--results-path", default=None)
    ap.add_argument("--render", action="store_true",
                    help="Open a live MuJoCo viewer mirroring the policy rollout.")
    ap.add_argument("--render-warmup-s", type=float, default=5.0,
                    help="With --render, sleep N seconds after viewer opens before "
                         "starting policy (default 5s) so the user can locate the window.")
    args = ap.parse_args()

    ckpt = args.ckpt if os.path.isabs(args.ckpt) else _abs_ckpt(args.ckpt)
    if not os.path.isdir(ckpt):
        sys.exit(f"checkpoint dir not found: {ckpt}\n"
                 f"hint: run the download cell in RoboCasa.ipynb (§5).")

    port = _find_free_port(args.port)
    if port != args.port:
        print(f"[orch] port {args.port} busy, using {port}", flush=True)

    server_py = (
        f"{REPO_ROOT}/scripts/_gr00t_inference_server.py "
        f"--model-path {shlex.quote(ckpt)} "
        f"--data-config {args.data_config} "
        f"--embodiment-tag {args.embodiment_tag} "
        f"--port {port}"
    )
    client_py = (
        f"{REPO_ROOT}/scripts/_gr00t_eval_client.py "
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

    server_cmd = ["bash", "-c", _conda_cmd(args.server_env, server_py)]
    client_cmd = ["bash", "-c", _conda_cmd(args.client_env, client_py)]

    print(f"[orch] starting server in env [{args.server_env}] ...", flush=True)
    server = subprocess.Popen(server_cmd, stdout=sys.stdout, stderr=sys.stderr,
                              preexec_fn=os.setsid)

    try:
        if not _wait_port_open("localhost", port, args.server_warmup_s):
            server.terminate()
            sys.exit(f"[orch] server failed to bind port {port} within "
                     f"{args.server_warmup_s}s — check server logs above")
        print(f"[orch] server is up on port {port}; starting client ...", flush=True)
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
