"""π0.5 ZMQ inference server (runs in `openpi` JAX env).

Loads the official `robocasa/robocasa365_checkpoints/pi05_pretrain_human300`
Orbax checkpoint and serves 12-dim action chunks over ZMQ REP.

The official RoboCasa pi05 checkpoint has no upstream openpi data config, so we
write one inline matching the team's training conventions (decoded from
assets/norm_stats.json):

  state 16-dim order: eef_pos_rel(3) + eef_rot_rel(4) + base_pos(3) +
                      base_rot(4) + gripper_qpos(2)
  action 12-dim order: eef_pos(3) + eef_rot(3) + gripper(1) +
                       base_motion(4) + control_mode(1)

3 cameras → pi0 image slots (best-guess mapping, swap on the client side if SR=0):
  robot0_agentview_left  -> base_0_rgb
  robot0_agentview_right -> left_wrist_0_rgb
  robot0_eye_in_hand     -> right_wrist_0_rgb

Protocol mirrors _smolvla_inference_server.py:
  request  = {"op": "get_action", "obs": <dict>} -> {"action": (T, 12) float32}
  request  = {"op": "shutdown"} -> {"ok": True}, then exit

Not invoked directly; spawned by scripts/robocasa_eval_pi05.py.
"""
import argparse
import dataclasses
import os
import pathlib
import pickle
import sys
import time

import einops
import numpy as np
import zmq

# openpi imports get resolved via the openpi-experience uv env.
import jax.numpy as jnp
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.policies.policy_config as policy_config
import openpi.training.config as _config
import openpi.transforms as transforms


def _parse_image(image):
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class RoboCasaInputs(transforms.DataTransformFn):
    """RoboCasa env obs dict -> openpi pi0/pi05 internal format.

    Input keys (from _pi05_eval_client.py):
      observation/state                       : (16,) float32 — concatenated
      observation/agentview_left              : (H,W,3) or (3,H,W) image
      observation/agentview_right             : same
      observation/eye_in_hand                 : same
      prompt                                  : str
    """
    model_type: _model.ModelType

    def __call__(self, data):
        base   = _parse_image(data["observation/agentview_left"])
        wleft  = _parse_image(data["observation/agentview_right"])
        wright = _parse_image(data["observation/eye_in_hand"])
        out = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base,
                "left_wrist_0_rgb": wleft,
                "right_wrist_0_rgb": wright,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            out["actions"] = data["actions"]
        if "prompt" in data:
            out["prompt"] = data["prompt"]
        return out


@dataclasses.dataclass(frozen=True)
class RoboCasaOutputs(transforms.DataTransformFn):
    """Slice the (T, 32) padded action chunk back to (T, 12) for robocasa.
    """
    def __call__(self, data):
        return {"actions": np.asarray(data["actions"][:, :12])}


@dataclasses.dataclass(frozen=True)
class RoboCasaDataConfig(_config.DataConfigFactory):
    """Custom DataConfigFactory wiring RoboCasaInputs/Outputs."""

    def create(self, assets_dirs, model_config):
        repack_transform = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation/agentview_left":  "observation.images.robot0_agentview_left",
                        "observation/agentview_right": "observation.images.robot0_agentview_right",
                        "observation/eye_in_hand":     "observation.images.robot0_eye_in_hand",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "task",
                    }
                )
            ]
        )
        data_transforms = transforms.Group(
            inputs=[RoboCasaInputs(model_type=model_config.model_type)],
            outputs=[RoboCasaOutputs()],
        )
        model_transforms = _config.ModelTransformFactory()(model_config)
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            # Robocasa pi05 norm_stats.json only carries mean/std (q01/q99 = null),
            # so force z-score normalization. Default create_base_config sets True
            # for any non-PI0 model_type which would crash on the missing quantiles.
            use_quantile_norm=False,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True,
                    help="path to pi05_pretrain_human300/multitask_learning/75000")
    ap.add_argument("--port", type=int, default=5557)
    ap.add_argument("--action-horizon", type=int, default=50)
    ap.add_argument("--default-prompt", default="")
    ap.add_argument("--discrete-state-input", action="store_true",
                    help="match pi05's native default (state as discrete tokens). "
                         "Try this if continuous state gives 0%% SR.")
    args = ap.parse_args()

    ckpt_dir = pathlib.Path(args.checkpoint_dir).expanduser().resolve()
    if not ckpt_dir.exists():
        sys.exit(f"checkpoint dir not found: {ckpt_dir}")
    if not (ckpt_dir / "params" / "_METADATA").exists():
        sys.exit(f"missing params/_METADATA — checkpoint not in Orbax format under {ckpt_dir}")

    print(f"[server] building pi05 train_config (custom robocasa data) ...", flush=True)
    train_config = _config.TrainConfig(
        name="pi05_robocasa_inference",
        model=pi0_config.Pi0Config(
            pi05=True,
            action_horizon=args.action_horizon,
            # pi05 default is True (state injected as language tokens). pi05_libero
            # overrides to False (continuous state). Without RoboCasa team's training
            # config we expose both via --discrete-state-input.
            discrete_state_input=args.discrete_state_input,
        ),
        data=RoboCasaDataConfig(
            # repo_id is required by DataConfigFactory.create_base_config (it
            # uses it as fallback asset_id) — robocasa365 dataset uses this
            # exact ckpt's asset directory, so point it there.
            repo_id="robocasa/robocasa365_checkpoints",
            base_config=_config.DataConfig(prompt_from_task=True),
            assets=_config.AssetsConfig(
                assets_dir=str(ckpt_dir / "assets"),
                asset_id=".",  # norm_stats.json sits directly under assets/
            ),
        ),
        # weights come from the ckpt's params/ dir at load time, so leave
        # weight_loader at default (NoOpWeightLoader) — TrainConfig requires it.
    )
    print(f"[server] loading policy from {ckpt_dir} ...", flush=True)
    # Inference-time repack: data_config.repack_transforms only run during
    # training (policy.py wires only the explicit `repack_transforms` kwarg here
    # into the policy's input pipeline). So we pass the same repack again.
    inference_repack = transforms.Group(
        inputs=[
            transforms.RepackTransform(
                {
                    "observation/agentview_left":  "observation.images.robot0_agentview_left",
                    "observation/agentview_right": "observation.images.robot0_agentview_right",
                    "observation/eye_in_hand":     "observation.images.robot0_eye_in_hand",
                    "observation/state": "observation.state",
                    "prompt": "task",
                }
            )
        ]
    )
    policy = policy_config.create_trained_policy(
        train_config,
        ckpt_dir,
        repack_transforms=inference_repack,
        default_prompt=args.default_prompt or None,
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
            result = policy.infer(obs)
            # policy.infer returns dict with "actions" key, shape (T, action_dim_padded)
            chunk = np.asarray(result["actions"])  # already sliced to (T, 12) by RoboCasaOutputs
            sock.send(pickle.dumps({"action": chunk.astype(np.float32)}))
        except Exception as e:
            import traceback
            traceback.print_exc()
            sock.send(pickle.dumps({"error": str(e)}))


if __name__ == "__main__":
    main()
