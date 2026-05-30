"""RoboCasa sim client that drives a SmolVLA policy via ZMQ (runs in `robocasa` env).

The policy's training dataset (`lerobot/robocasa_target_human_unified`) was built
directly from RoboCasa env's native obs/action — 12-dim action and 16-dim state
match the env spec exactly. Only camera-key rename and HWC→CHW float conversion
are needed before sending obs to the server.

Camera rename (per train_config.json):
  video.robot0_agentview_left  -> observation.images.robot0_agentview_left
  video.robot0_agentview_right -> observation.images.robot0_agentview_right
  video.robot0_eye_in_hand     -> observation.images.robot0_eye_in_hand
The server's preprocessor renames again to camera1/camera2/camera3.

State concat (dataset order, gives 16-dim):
  base_position(3) + base_rotation(4) + eef_pos_rel(3) + eef_rot_rel(4) + gripper_qpos(2)

Success metric: read `info["success"]` per step (set by robocasa gym wrapper).
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import zmq

import robocasa  # noqa: F401  registers gym envs
import gymnasium as gym


# Robocasa env -> lerobot-dataset obs key mapping.
CAMERA_KEY_MAP = {
    "video.robot0_agentview_left":  "observation.images.robot0_agentview_left",
    "video.robot0_agentview_right": "observation.images.robot0_agentview_right",
    "video.robot0_eye_in_hand":     "observation.images.robot0_eye_in_hand",
}

# Dataset state order (verified against normalizer stats means in
# policy_preprocessor_step_5_normalizer_processor.safetensors).
STATE_KEYS = (
    "state.base_position",       # 3
    "state.base_rotation",       # 4
    "state.end_effector_position_relative",  # 3
    "state.end_effector_rotation_relative",  # 4
    "state.gripper_qpos",        # 2
)


def flat_action_to_robocasa_dict(flat_action):
    """Split SmolVLA's 12-dim flat action into the dict robocasa GymEnv expects.

    Layout matches lerobot/envs/robocasa.py::convert_action — this is the
    canonical inverse of how the training dataset was flattened.
    """
    return {
        "action.base_motion": flat_action[0:4],
        "action.control_mode": flat_action[4:5],
        "action.end_effector_position": flat_action[5:8],
        "action.end_effector_rotation": flat_action[8:11],
        "action.gripper_close": flat_action[11:12],
    }


def _build_policy_obs(obs, task_description):
    """Convert RoboCasa env obs dict to the format the SmolVLA server expects."""
    out = {}
    for src, dst in CAMERA_KEY_MAP.items():
        img = obs[src]  # HWC uint8
        # HWC uint8 [0,255] -> CHW float32 [0,1]
        img = np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1))
        out[dst] = img
    state_parts = [np.asarray(obs[k], dtype=np.float32).ravel() for k in STATE_KEYS]
    state = np.concatenate(state_parts)
    assert state.shape == (16,), f"state dim {state.shape} != 16"
    out["observation.state"] = state
    out["task"] = task_description
    return out


class PolicyClient:
    def __init__(self, host, port, timeout_s=180):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_s * 1000)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_s * 1000)
        self.sock.connect(f"tcp://{host}:{port}")

    def get_action_chunk(self, obs):
        self.sock.send(pickle.dumps({"op": "get_action", "obs": obs}))
        reply = pickle.loads(self.sock.recv())
        if "error" in reply:
            raise RuntimeError(f"server error: {reply['error']}")
        return reply["action"]  # (T, 12) float32

    def close(self):
        self.sock.close()
        self.ctx.term()


def _aim_viewer_at_agentview(env, viewer):
    """Re-aim passive viewer at robot0_agentview_center; see _gr00t_eval_client.py."""
    inner = env.unwrapped.env
    cam_id = inner.sim.model.camera_name2id("robot0_agentview_center")
    cam_pos = inner.sim.data.cam_xpos[cam_id].copy()
    cam_mat = inner.sim.data.cam_xmat[cam_id].reshape(3, 3)
    view_dir = -cam_mat[:, 2]
    LOOK_AHEAD = 1.5
    lookat = cam_pos + view_dir * LOOK_AHEAD
    delta = cam_pos - lookat
    distance = float(np.linalg.norm(delta))
    viewer.cam.lookat[:] = lookat
    viewer.cam.distance = distance
    viewer.cam.azimuth = float(np.degrees(np.arctan2(-delta[1], -delta[0])))
    viewer.cam.elevation = float(-np.degrees(np.arcsin(delta[2] / distance)))
    viewer.sync()


def _attach_passive_viewer(env):
    import mujoco
    inner = env.unwrapped.env
    viewer = mujoco.viewer.launch_passive(
        inner.sim.model._model,
        inner.sim.data._data,
        show_left_ui=False,
        show_right_ui=False,
    )
    viewer.opt.geomgroup[0] = 0  # hide untextured collision proxy
    _aim_viewer_at_agentview(env, viewer)
    return viewer


def run_episode(env, client, n_action_steps, max_steps, viewer=None, initial_obs=None, task_desc=None):
    if initial_obs is None:
        obs, _ = env.reset()
    else:
        obs = initial_obs
    if viewer is not None:
        _aim_viewer_at_agentview(env, viewer)
    inner = env.unwrapped.env
    eef_site = inner.sim.model.site_name2id("gripper0_right_grip_site")
    eef_origin = inner.sim.data.site_xpos[eef_site].copy()
    base_origin = inner.sim.data.qpos[:3].copy()
    arm_qpos_idx = slice(4, 11)
    arm_qpos_origin = inner.sim.data.qpos[arm_qpos_idx].copy()

    if task_desc is None:
        task_desc = obs["annotation.human.task_description"]
    success = False
    steps = 0
    while steps < max_steps:
        policy_obs = _build_policy_obs(obs, task_desc)
        chunk = client.get_action_chunk(policy_obs)  # (T_chunk, 12)
        T = min(n_action_steps, chunk.shape[0], max_steps - steps)
        for t in range(T):
            flat = np.asarray(chunk[t], dtype=np.float32)
            action = flat_action_to_robocasa_dict(flat)
            obs, _r, term, trunc, info = env.step(action)
            steps += 1
            if viewer is not None and viewer.is_running():
                viewer.sync()
            if steps % 25 == 0:
                eef_now = inner.sim.data.site_xpos[eef_site]
                base_now = inner.sim.data.qpos[:3]
                arm_now = inner.sim.data.qpos[arm_qpos_idx]
                eef_mm = float(np.linalg.norm(eef_now - eef_origin) * 1000)
                base_mm = float(np.linalg.norm(base_now - base_origin) * 1000)
                arm_deg = float(np.linalg.norm(arm_now - arm_qpos_origin) * 180 / np.pi)
                print(f"[client] step{steps}: |Δeef|={eef_mm:.0f}mm "
                      f"|Δbase|={base_mm:.0f}mm |Δarm|={arm_deg:.1f}° from start", flush=True)
            if info.get("success", False):
                success = True
            if term or trunc or success:
                return success, steps
    return success, steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", required=True)
    ap.add_argument("--split", default="target", choices=["pretrain", "target"])
    ap.add_argument("--n-episodes", type=int, default=2)
    ap.add_argument("--n-action-steps", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--results-path", default=None)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--render-warmup-s", type=float, default=5.0)
    args = ap.parse_args()

    full_env_name = f"robocasa/{args.env_name}"
    print(f"[client] making env {full_env_name} (split={args.split})", flush=True)
    env = gym.make(full_env_name, split=args.split, enable_render=True)

    print("[client] warm-up reset ...", flush=True)
    initial_obs, _ = env.reset()

    viewer = None
    if args.render:
        print("[client] attaching passive MuJoCo viewer ...", flush=True)
        viewer = _attach_passive_viewer(env)
        warmup = max(0.0, args.render_warmup_s)
        if warmup > 0:
            print(f"[client] viewer opened; policy starts in {warmup:.0f}s ...", flush=True)
            t_end = time.time() + warmup
            while time.time() < t_end:
                viewer.sync()
                time.sleep(0.05)

    client = PolicyClient(args.host, args.port)

    results = []
    t0 = time.time()
    for ep in range(args.n_episodes):
        try:
            ep_initial = initial_obs if ep == 0 else None
            ok, steps = run_episode(env, client, args.n_action_steps,
                                    args.max_steps, viewer=viewer,
                                    initial_obs=ep_initial,
                                    task_desc=args.env_name)
        except Exception as e:
            import traceback
            print(f"[client] episode {ep} crashed: {e}", flush=True)
            traceback.print_exc()
            ok, steps = False, 0
        results.append({"episode": ep, "success": bool(ok), "steps": steps})
        rate = float(np.mean([r["success"] for r in results]))
        print(f"[client] EP {ep}: success={ok} steps={steps} | rate={rate:.2f}", flush=True)

    dt = time.time() - t0
    summary = {
        "env_name": args.env_name,
        "split": args.split,
        "n_episodes": args.n_episodes,
        "success_rate": float(np.mean([r["success"] for r in results])),
        "wall_time_s": dt,
        "results": results,
    }
    print("\n========== RESULTS ==========")
    print(json.dumps(summary, indent=2))

    if args.results_path:
        os.makedirs(os.path.dirname(args.results_path) or ".", exist_ok=True)
        with open(args.results_path, "w") as f:
            json.dump(summary, f, indent=2)

    if viewer is not None:
        viewer.close()
    env.close()
    client.close()


if __name__ == "__main__":
    main()
