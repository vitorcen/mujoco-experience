"""RoboCasa sim client that drives a GR00T policy via ZMQ (runs in `robocasa` env).

For each rollout:
  reset env -> loop env.step using action chunks pulled from the server.
GR00T outputs an action *chunk* of shape (T, action_dim) per dict key; we replay
those T steps before querying again.

Success metric: read `info["success"]` (set by robocasa's gym wrapper from
`env._check_success()`). Reported as success rate over N episodes.

If --render is set, a passive MuJoCo viewer attaches to the same sim state and
mirrors policy actions live (offscreen camera obs still feeds the policy).

Not invoked directly; spawned by scripts/robocasa_eval_gr00t.py.
"""
import argparse
import json
import os
import pickle
import sys
import time

import numpy as np
import zmq

# Import robocasa to register gym envs. Stub trick from robocasa_demo.py is
# unnecessary here — the orchestrator clears bytecode and sets
# PYTHONDONTWRITEBYTECODE=1 to dodge the PEP 659 quickened-opcode bug.
import robocasa  # noqa: F401
import gymnasium as gym


def _add_time_dim(obs):
    """robocasa gym wrapper returns single-frame obs: state.* (D,), video.* (H,W,C).
    GR00T's transform pipeline expects a leading T=state_horizon dimension on every
    modality (T=1 for PandaOmron). Without it, ConcatTransform mis-attributes the
    later batch dim to T and GR00TTransform.apply_single runs (skipping collate),
    so PIL.Image objects leak into model input and crash at `is_floating_point`.
    """
    out = {}
    for k, v in obs.items():
        if isinstance(v, np.ndarray):
            out[k] = v[None]  # prepend T=1
        elif isinstance(v, str):
            out[k] = [v]
        else:
            out[k] = v
    return out


class PolicyClient:
    def __init__(self, host, port, timeout_s=120):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_s * 1000)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_s * 1000)
        self.sock.connect(f"tcp://{host}:{port}")

    def get_action_chunk(self, obs):
        self.sock.send(pickle.dumps({"op": "get_action", "obs": _add_time_dim(obs)}))
        reply = pickle.loads(self.sock.recv())
        if "error" in reply:
            raise RuntimeError(f"server error: {reply['error']}")
        return reply

    def close(self):
        self.sock.close()
        self.ctx.term()


def _slice_step(action_chunk, t):
    """Pick step t out of an (T, D) action chunk for each key, return env-step dict."""
    out = {}
    for k, v in action_chunk.items():
        v = np.asarray(v)
        if v.ndim >= 1 and v.shape[0] > t:
            out[k] = v[t]
        else:
            out[k] = v
    return out


def _aim_viewer_at_agentview(env, viewer):
    """Point the passive viewer at robot0_agentview_center.

    robocasa drops the robot ~10m from world origin and randomizes the kitchen
    layout per reset, so the camera must be re-aimed every episode or the user
    sees an empty wall while the robot acts off-screen.
    """
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
    """Open a passive MuJoCo viewer on the env's sim and aim at the robot.

    Caller must ensure env.reset() has run at least once before this — robocasa
    only loads kitchen textures lazily via the offscreen renderer during
    obs construction, and the passive viewer's GL context only picks up
    textures that exist at context-creation time.
    """
    import mujoco
    inner = env.unwrapped.env
    viewer = mujoco.viewer.launch_passive(
        inner.sim.model._model,
        inner.sim.data._data,
        show_left_ui=False,
        show_right_ui=False,
    )
    # robocasa puts collision-proxy geoms (untextured) in geomgroup 0 and the
    # textured visual geoms in geomgroup 1. Without hiding group 0 the viewer
    # renders the flat-colour collision layer on top of the visuals. This is
    # exactly what robosuite's MjviewerRenderer does internally.
    viewer.opt.geomgroup[0] = 0
    _aim_viewer_at_agentview(env, viewer)
    return viewer


def run_episode(env, client, n_action_steps, max_steps, viewer=None, initial_obs=None):
    if initial_obs is None:
        obs, _ = env.reset()
    else:
        # Reuse the warm-up reset's obs for episode 0 — avoids burning an
        # extra random init that drifts episode 0 onto a different layout
        # than callers had pre-warm-up (codex spotted this regression).
        obs = initial_obs
    if viewer is not None:
        # robocasa re-randomizes layout & robot pose on every reset, so the
        # camera must be re-aimed or we end up staring into a wall.
        _aim_viewer_at_agentview(env, viewer)
    inner = env.unwrapped.env
    eef_site = inner.sim.model.site_name2id("gripper0_right_grip_site")
    eef_origin = inner.sim.data.site_xpos[eef_site].copy()
    base_origin = inner.sim.data.qpos[:3].copy()
    # Track ARM joint motion directly (qpos[4:11] for PandaOmron 7-DOF arm).
    # |Δeef| includes base translation, so it's not enough to prove the arm
    # itself is moving — joint qpos is the ground truth.
    arm_qpos_idx = slice(4, 11)
    arm_qpos_origin = inner.sim.data.qpos[arm_qpos_idx].copy()
    success = False
    steps = 0
    while steps < max_steps:
        chunk = client.get_action_chunk(obs)
        T = min(n_action_steps, max_steps - steps)
        for t in range(T):
            action = _slice_step(chunk, t)
            obs, _r, term, trunc, info = env.step(action)
            steps += 1
            if viewer is not None and viewer.is_running():
                viewer.sync()
            # Periodic motion heartbeat so the user can confirm the robot is
            # actually moving even if the viewer window isn't visible.
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
    ap.add_argument("--env-name", required=True, help="e.g. CloseFridge, OpenCabinet")
    ap.add_argument("--split", default="pretrain", choices=["pretrain", "target"])
    ap.add_argument("--n-episodes", type=int, default=2)
    ap.add_argument("--n-action-steps", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=400)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5555)
    ap.add_argument("--results-path", default=None)
    ap.add_argument("--render", action="store_true",
                    help="Open a passive MuJoCo viewer mirroring policy rollouts.")
    ap.add_argument("--render-warmup-s", type=float, default=5.0,
                    help="With --render, pause this many seconds after the viewer "
                         "opens before starting the policy, so the user can locate "
                         "the window and see the initial scene before motion starts.")
    args = ap.parse_args()

    full_env_name = f"robocasa/{args.env_name}"
    print(f"[client] making env {full_env_name} (split={args.split})", flush=True)
    env = gym.make(full_env_name, split=args.split, enable_render=True)

    # Force a reset BEFORE attaching the passive viewer so robosuite's offscreen
    # renderer initialises its GL context + uploads textures first. If we open
    # the passive viewer first, its GL context comes up before robocasa has
    # generated kitchen textures, and the viewer ends up rendering flat colours.
    # We keep the obs and feed it into episode 0 so we don't burn an extra
    # random init (episode 0 would otherwise see different layout/objects).
    print("[client] warm-up reset (loads textures before viewer init) ...", flush=True)
    initial_obs, _ = env.reset()

    viewer = None
    if args.render:
        print("[client] attaching passive MuJoCo viewer ...", flush=True)
        viewer = _attach_passive_viewer(env)
        warmup = max(0.0, args.render_warmup_s)
        if warmup > 0:
            print(f"[client] ============================================", flush=True)
            print(f"[client] >>> MuJoCo viewer opened. Bring it to focus.  <<<", flush=True)
            print(f"[client] >>> Policy starts in {warmup:.0f}s ...           <<<", flush=True)
            print(f"[client] ============================================", flush=True)
            # The viewer renders the static scene in its own thread; we just
            # sleep (with periodic sync to keep it responsive) so the user has
            # time to locate the window before motion starts.
            t_end = time.time() + warmup
            while time.time() < t_end:
                viewer.sync()
                time.sleep(0.05)
            print(f"[client] starting policy now.", flush=True)

    client = PolicyClient(args.host, args.port)

    results = []
    t0 = time.time()
    for ep in range(args.n_episodes):
        try:
            ep_initial_obs = initial_obs if ep == 0 else None
            ok, steps = run_episode(env, client, args.n_action_steps,
                                    args.max_steps, viewer=viewer,
                                    initial_obs=ep_initial_obs)
        except Exception as e:
            print(f"[client] episode {ep} crashed: {e}", flush=True)
            ok, steps = False, 0
        results.append({"episode": ep, "success": bool(ok), "steps": steps})
        rate = np.mean([r["success"] for r in results])
        print(f"[client] EP {ep}: success={ok} steps={steps} | cumulative_rate={rate:.2f}", flush=True)

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
        print(f"[client] saved to {args.results_path}")

    if viewer is not None:
        viewer.close()
    env.close()
    client.close()


if __name__ == "__main__":
    main()
