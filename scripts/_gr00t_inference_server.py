"""GR00T N1.5 ZMQ inference server (runs in `robocasa_gr00t` conda env).

Protocol: pickle-over-ZMQ REP socket.
  request  = {"op": "get_action", "obs": <dict of np.ndarray>}
  response = <dict of np.ndarray>  (T action steps × action_dim per key)
  request  = {"op": "shutdown"} -> {"ok": True}, then exit

Not invoked directly; spawned by scripts/robocasa_eval_gr00t.py.
"""
import argparse
import pickle
import sys

import numpy as np
import zmq


def _to_numpy(d):
    out = {}
    for k, v in d.items():
        if hasattr(v, "cpu"):
            v = v.cpu().numpy()
        elif hasattr(v, "numpy"):
            v = v.numpy()
        out[k] = np.asarray(v)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--data-config", default="panda_omron")
    ap.add_argument("--embodiment-tag", default="new_embodiment")
    ap.add_argument("--denoising-steps", type=int, default=4)
    ap.add_argument("--port", type=int, default=5555)
    args = ap.parse_args()

    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy

    cfg = DATA_CONFIG_MAP[args.data_config]
    print(f"[server] loading policy from {args.model_path} ...", flush=True)
    policy = Gr00tPolicy(
        model_path=args.model_path,
        modality_config=cfg.modality_config(),
        modality_transform=cfg.transform(),
        embodiment_tag=args.embodiment_tag,
        denoising_steps=args.denoising_steps,
    )

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{args.port}")
    print(f"[server] ready on tcp://*:{args.port}", flush=True)

    while True:
        try:
            req = pickle.loads(sock.recv())
        except Exception as e:
            print(f"[server] bad request: {e}", file=sys.stderr, flush=True)
            sock.send(pickle.dumps({"error": str(e)}))
            continue
        op = req.get("op")
        if op == "shutdown":
            sock.send(pickle.dumps({"ok": True}))
            print("[server] shutting down", flush=True)
            break
        if op != "get_action":
            sock.send(pickle.dumps({"error": f"unknown op {op}"}))
            continue
        try:
            action_dict = policy.get_action(req["obs"])
            sock.send(pickle.dumps(_to_numpy(action_dict)))
        except Exception as e:
            import traceback
            traceback.print_exc()
            sock.send(pickle.dumps({"error": str(e)}))


if __name__ == "__main__":
    main()
