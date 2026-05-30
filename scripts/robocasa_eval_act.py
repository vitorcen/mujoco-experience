"""Orchestrator: run a locally-trained ACT policy against a single RoboCasa env.

Spawns the ACT inference server in the `lerobot` conda env and the sim client
in the `robocasa` env. The client is scripts/_gr00t_eval_client.py UNCHANGED —
the ZMQ wire protocol is identical between the GR00T and ACT servers, by design.

Usage:
    python scripts/robocasa_eval_act.py --env-name OpenCabinet --n-episodes 50
    python scripts/robocasa_eval_act.py --env-name OpenCabinet --render
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
DEFAULT_CKPT = REPO_ROOT / "robocasa-training" / "checkpoints" / "act_opencabinet"


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
    ap.add_argument("--split", default="target", choices=["pretrain", "target"],
                    help="ACT here was trained on the target split — match it.")
    ap.add_argument("--n-episodes", type=int, default=10)
    ap.add_argument("--n-action-steps", type=int, default=50,
                    help="How many steps from each ACT chunk (chunk_size=100) the client replays before requerying.")
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT),
                    help="ACT ckpt dir (parent of checkpoints/*/pretrained_model, or the pretrained_model dir directly).")
    ap.add_argument("--port", type=int, default=5556,
                    help="Default 5556 to avoid clashing with the GR00T eval port 5555.")
    ap.add_argument("--server-warmup-s", type=int, default=120)
    ap.add_argument("--server-env", default="lerobot")
    ap.add_argument("--client-env", default="robocasa")
    ap.add_argument("--results-path", default=None)
    ap.add_argument("--render", action="store_true",
                    help="Open a live MuJoCo viewer mirroring the policy rollout.")
    ap.add_argument("--render-warmup-s", type=float, default=5.0)
    ap.add_argument("--temporal-ensemble", type=float, default=None,
                    help="Enable ACT temporal ensembling at inference (coeff, e.g. 0.01). "
                         "Forces client n_action_steps=1 + per-episode reset.")
    args = ap.parse_args()

    ckpt = os.path.abspath(args.ckpt)
    if not os.path.isdir(ckpt):
        sys.exit(f"ACT ckpt dir not found: {ckpt}\n"
                 f"hint: run `python robocasa-training/scripts/train_act.py --steps 100000` first.")

    port = _find_free_port(args.port)
    if port != args.port:
        print(f"[orch] port {args.port} busy, using {port}", flush=True)

    # Temporal ensembling: server ensembles per step (returns T=1), so the client
    # must query every step (n_action_steps=1) and reset the server each episode.
    client_n_action_steps = 1 if args.temporal_ensemble is not None else args.n_action_steps

    server_py = (
        f"{REPO_ROOT}/robocasa-training/scripts/serve_act.py "
        f"--model-path {shlex.quote(ckpt)} "
        f"--n-action-steps {args.n_action_steps} "
        f"--port {port}"
    )
    if args.temporal_ensemble is not None:
        server_py += f" --temporal-ensemble {args.temporal_ensemble}"
    client_py = (
        f"{REPO_ROOT}/scripts/_gr00t_eval_client.py "
        f"--env-name {args.env_name} "
        f"--split {args.split} "
        f"--n-episodes {args.n_episodes} "
        f"--n-action-steps {client_n_action_steps} "
        f"--max-steps {args.max_steps} "
        f"--port {port}"
    )
    if args.temporal_ensemble is not None:
        client_py += " --send-reset"
    if args.results_path:
        client_py += f" --results-path {shlex.quote(args.results_path)}"
    if args.render:
        client_py += f" --render --render-warmup-s {args.render_warmup_s}"

    server_cmd = ["bash", "-c", _conda_cmd(args.server_env, server_py)]
    client_cmd = ["bash", "-c", _conda_cmd(args.client_env, client_py)]

    print(f"[orch] starting ACT server in env [{args.server_env}] ...", flush=True)
    server = subprocess.Popen(server_cmd, stdout=sys.stdout, stderr=sys.stderr,
                              preexec_fn=os.setsid)
    try:
        if not _wait_port_open("localhost", port, args.server_warmup_s):
            server.terminate()
            sys.exit(f"[orch] ACT server failed to bind port {port} within "
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
