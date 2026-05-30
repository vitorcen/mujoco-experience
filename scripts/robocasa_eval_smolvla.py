"""Orchestrator: run a SmolVLA policy against a single RoboCasa env.

Spawns the inference server in the `lerobot` conda env and the sim client in
the `robocasa` env. Mirrors robocasa_eval_gr00t.py structure.

Usage:
    python scripts/robocasa_eval_smolvla.py --env-name OpenCabinet --n-episodes 5
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
    ap.add_argument("--env-name", default="OpenCabinet")
    ap.add_argument("--split", default="target", choices=["pretrain", "target"])
    ap.add_argument("--n-episodes", type=int, default=2)
    ap.add_argument("--n-action-steps", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--model-path", default="lerobot/smolvla_robocasa",
                    help="HF repo id or local path")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--server-warmup-s", type=int, default=300,
                    help="how long to wait for the server to bind the port "
                         "(SmolVLM 500M backbone needs ~30s to load)")
    ap.add_argument("--server-env", default="lerobot")
    ap.add_argument("--client-env", default="robocasa")
    ap.add_argument("--results-path", default=None)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--render-warmup-s", type=float, default=5.0)
    args = ap.parse_args()

    port = _find_free_port(args.port)
    if port != args.port:
        print(f"[orch] port {args.port} busy, using {port}", flush=True)

    server_py = (
        f"{REPO_ROOT}/scripts/_smolvla_inference_server.py "
        f"--model-path {shlex.quote(args.model_path)} "
        f"--port {port}"
    )
    client_py = (
        f"{REPO_ROOT}/scripts/_smolvla_eval_client.py "
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
