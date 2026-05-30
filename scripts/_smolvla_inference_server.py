"""SmolVLA ZMQ inference server (runs in `lerobot` conda env).

Loads `lerobot/smolvla_robocasa` and serves 12-dim action chunks over ZMQ REP.

Protocol mirrors _gr00t_inference_server.py:
  request  = {"op": "get_action", "obs": <dict>} -> response dict with key "action"
                   shape (T, 12) numpy float32
  request  = {"op": "shutdown"} -> {"ok": True}, then exit

Expected obs (built by _smolvla_eval_client.py):
  observation.images.robot0_agentview_left   : (3,256,256) float32 in [0,1]
  observation.images.robot0_agentview_right  : (3,256,256) float32 in [0,1]
  observation.images.robot0_eye_in_hand      : (3,256,256) float32 in [0,1]
  observation.state                          : (16,) float32 (raw robocasa state)
  task                                       : str (language instruction)

Not invoked directly; spawned by scripts/robocasa_eval_smolvla.py.
"""
import argparse
import pickle
import sys

import numpy as np
import torch
import zmq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", default="lerobot/smolvla_robocasa")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.processor.pipeline import DataProcessorPipeline

    print(f"[server] loading policy from {args.model_path} ...", flush=True)
    policy = SmolVLAPolicy.from_pretrained(args.model_path).to(args.device)
    policy.eval()
    preproc = DataProcessorPipeline.from_pretrained(
        args.model_path, config_filename="policy_preprocessor.json"
    )
    postproc = DataProcessorPipeline.from_pretrained(
        args.model_path, config_filename="policy_postprocessor.json"
    )
    print("[server] policy ready", flush=True)

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{args.port}")
    print(f"[server] listening on tcp://*:{args.port}", flush=True)

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
            obs = req["obs"]
            # Convert all ndarray fields to tensors. Preprocessor expects single-frame
            # (no batch dim) per key — AddBatchDimensionProcessorStep adds it.
            sample = {}
            for k, v in obs.items():
                if isinstance(v, np.ndarray):
                    sample[k] = torch.from_numpy(v.copy())
                else:
                    sample[k] = v
            batch = preproc(sample)
            chunk = policy.predict_action_chunk(batch)  # (1, T, 12) normalized
            chunk = postproc({"action": chunk})["action"]  # unnormalized (1, T, 12)
            chunk_np = chunk.detach().cpu().numpy()[0]  # (T, 12) float32
            sock.send(pickle.dumps({"action": chunk_np}))
        except Exception as e:
            import traceback
            traceback.print_exc()
            sock.send(pickle.dumps({"error": str(e)}))


if __name__ == "__main__":
    main()
