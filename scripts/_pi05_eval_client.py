"""RoboCasa sim client driving the π0.5 policy via ZMQ (runs in `robocasa` env).

Mirrors _smolvla_eval_client.py but adapted to the π0.5 / official RoboCasa
ckpt's input/output conventions (decoded from assets/norm_stats.json):

  state 16-dim order (this ckpt):
    eef_pos_rel(3) + eef_rot_rel(4) + base_pos(3) + base_rot(4) + gripper_qpos(2)

  action 12-dim order (this ckpt):
    eef_pos(3) + eef_rot(3) + gripper(1) + base_motion(4) + control_mode(1)

Differs from SmolVLA's lerobot/robocasa_target_human_unified ordering, which
groups by (base, ee, gripper) and uses convert_action's layout.

PROCESS ISOLATION (2026-05-30): like _gr00t_eval_client.py, this runs in two
modes. The DRIVER (default) spawns one fresh WORKER subprocess per episode and
never imports robocasa/mujoco itself, so it stays pristine. Reason: a C-extension
(most likely MuJoCo's GL/EGL) corrupts the heap across repeated robosuite
env.reset() churn, and CPython 3.11's PEP 659 adaptive interpreter surfaces that
as bizarre bytecode errors (unknown opcode / 'method' not iterable / NoneType not
iterable in pure-Python yaml/xml). One episode per process = corruption can never
accumulate; a worker that segfaults costs only its own episode.
"""
import argparse
import json
import os
import pickle
import signal
import subprocess
import sys
import time

import numpy as np
import zmq


def _lazy_env_imports():
    """Import the heavy (corruption-prone) sim stack. Worker process only."""
    import robocasa  # noqa: F401  (registers gym envs)
    import gymnasium as gym
    return gym


# State concat in the order this ckpt expects (from norm_stats.json analysis).
STATE_KEYS = (
    "state.end_effector_position_relative",  # 3
    "state.end_effector_rotation_relative",  # 4
    "state.base_position",                   # 3
    "state.base_rotation",                   # 4
    "state.gripper_qpos",                    # 2
)


def flat_action_to_robocasa_dict(flat):
    """Split π0.5's 12-dim flat action into the dict robocasa GymEnv expects.

    Layout per norm_stats.json (eef_pos / eef_rot / gripper / base_motion /
    control_mode) — this is different from lerobot's canonical convert_action
    layout used by SmolVLA, but matches the official RoboCasa pi05 ckpt.
    """
    return {
        "action.end_effector_position": flat[0:3],
        "action.end_effector_rotation": flat[3:6],
        "action.gripper_close":         flat[6:7],
        "action.base_motion":           flat[7:11],
        "action.control_mode":          flat[11:12],
    }


def _build_policy_obs(obs, task_description):
    """RoboCasa env obs -> the keys the pi05 server expects."""
    out = {
        # Keep HWC uint8 — _parse_image on server converts as needed.
        "observation.images.robot0_agentview_left":  obs["video.robot0_agentview_left"],
        "observation.images.robot0_agentview_right": obs["video.robot0_agentview_right"],
        "observation.images.robot0_eye_in_hand":     obs["video.robot0_eye_in_hand"],
        "task": task_description,
    }
    state_parts = [np.asarray(obs[k], dtype=np.float32).ravel() for k in STATE_KEYS]
    state = np.concatenate(state_parts)
    assert state.shape == (16,), f"state dim {state.shape} != 16"
    out["observation.state"] = state
    return out


class PolicyClient:
    def __init__(self, host, port, timeout_s=300):
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
    viewer.opt.geomgroup[0] = 0
    _aim_viewer_at_agentview(env, viewer)
    return viewer


def run_episode(env, client, n_action_steps, max_steps, viewer=None,
                initial_obs=None, task_desc=None):
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
        chunk = client.get_action_chunk(policy_obs)
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
    ap.add_argument("--port", type=int, default=5557)
    ap.add_argument("--results-path", default=None)
    ap.add_argument("--render", action="store_true")
    ap.add_argument("--render-warmup-s", type=float, default=5.0)
    ap.add_argument("--task-mode", default="natural-language",
                    choices=["natural-language", "env-name"],
                    help="natural-language = use obs['annotation.human.task_description'] "
                         "(e.g. 'Open the cabinet door.'). env-name = use env_name "
                         "('OpenCabinet'). pi05 ckpt was likely trained on natural lang.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Base RNG seed. The driver gives episode N seed (base+N) "
                         "so every policy faces the identical reproducible scene "
                         "sequence — a FAIR benchmark. Unset = old random behaviour.")
    # --- process-isolation knobs (internal; set by the driver, not the caller) ---
    ap.add_argument("--episode", type=int, default=None,
                    help="WORKER MODE: run exactly this one episode and exit. "
                         "When omitted, runs in DRIVER MODE (spawns one isolated "
                         "worker subprocess per episode — see module docstring).")
    ap.add_argument("--episode-result-path", default=None,
                    help="WORKER MODE: where to write this episode's {success,steps} JSON.")
    args = ap.parse_args()

    if args.episode is None:
        return _driver_main(args)
    return _worker_main(args)


def _worker_main(args):
    """Run exactly ONE episode in this fresh process, then exit."""
    gym = _lazy_env_imports()
    ep = args.episode

    full_env_name = f"robocasa/{args.env_name}"
    print(f"[worker ep{ep}] making env {full_env_name} (split={args.split})", flush=True)
    env = gym.make(full_env_name, split=args.split, enable_render=True)

    print(f"[worker ep{ep}] warm-up reset (seed={args.seed}) ...", flush=True)
    initial_obs, _ = env.reset(seed=args.seed) if args.seed is not None else env.reset()

    viewer = None
    if args.render:
        print(f"[worker ep{ep}] attaching passive MuJoCo viewer ...", flush=True)
        viewer = _attach_passive_viewer(env)
        warmup = max(0.0, args.render_warmup_s)
        if warmup > 0:
            print(f"[worker ep{ep}] viewer opened; policy starts in {warmup:.0f}s ...", flush=True)
            t_end = time.time() + warmup
            while time.time() < t_end:
                viewer.sync()
                time.sleep(0.05)

    client = PolicyClient(args.host, args.port)
    task = args.env_name if args.task_mode == "env-name" else None
    try:
        ok, steps = run_episode(env, client, args.n_action_steps,
                                args.max_steps, viewer=viewer,
                                initial_obs=initial_obs, task_desc=task)
    except Exception as e:
        import traceback
        print(f"[worker ep{ep}] episode crashed: {e}", flush=True)
        traceback.print_exc()
        ok, steps = False, 0

    print(f"[worker ep{ep}] success={ok} steps={steps}", flush=True)
    if args.episode_result_path:
        with open(args.episode_result_path, "w") as f:
            json.dump({"episode": ep, "success": bool(ok), "steps": int(steps)}, f)

    if viewer is not None:
        viewer.close()
    env.close()
    client.close()


def _driver_main(args):
    """Spawn one isolated worker subprocess per episode, collect results, write
    the same summary JSON the single-process client used to. This process never
    imports robocasa/mujoco, so it stays pristine no matter how badly a worker
    corrupts its heap; a crash/timeout → sim-DNF (steps=0) and we continue."""
    results = []
    t0 = time.time()

    def _write_summary():
        summary = {
            "env_name": args.env_name,
            "split": args.split,
            "n_episodes": args.n_episodes,
            "n_completed": len(results),
            "success_rate": float(np.mean([r["success"] for r in results])) if results else 0.0,
            "wall_time_s": time.time() - t0,
            "results": results,
        }
        if args.results_path:
            os.makedirs(os.path.dirname(args.results_path) or ".", exist_ok=True)
            tmp = args.results_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(summary, f, indent=2)
            os.replace(tmp, args.results_path)
        return summary

    ep_timeout_s = int(os.environ.get("EP_TIMEOUT_S", "900"))
    seed_base = args.seed if args.seed is not None else int(os.environ.get("SEED_BASE", "0"))
    res_dir = os.path.dirname(os.path.abspath(args.results_path)) if args.results_path else "/tmp"
    os.makedirs(res_dir, exist_ok=True)

    base_cmd = [sys.executable, "-u", os.path.abspath(__file__),
                "--env-name", args.env_name, "--split", args.split,
                "--n-action-steps", str(args.n_action_steps),
                "--max-steps", str(args.max_steps),
                "--host", args.host, "--port", str(args.port),
                "--task-mode", args.task_mode, "--n-episodes", "1"]
    if args.render:
        base_cmd += ["--render", "--render-warmup-s", str(args.render_warmup_s)]

    print(f"[driver] {args.n_episodes} episodes, one isolated worker each "
          f"(env={args.env_name} split={args.split} port={args.port})", flush=True)

    max_ep_tries = int(os.environ.get("EVAL_EP_RETRIES", "3"))

    def _spawn_worker(ep, ep_res_path):
        """One worker subprocess for episode `ep`. Returns (rec, rc): rec is the
        result dict, or None if the worker crashed/timed out before writing one."""
        if os.path.exists(ep_res_path):
            os.remove(ep_res_path)
        cmd = base_cmd + ["--episode", str(ep), "--episode-result-path", ep_res_path,
                          "--seed", str(seed_base + ep)]
        proc = subprocess.Popen(cmd, start_new_session=True)
        try:
            rc = proc.wait(timeout=ep_timeout_s)
        except subprocess.TimeoutExpired:
            print(f"[driver] episode {ep} exceeded {ep_timeout_s}s — killing worker group",
                  flush=True)
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                proc.kill()
            proc.wait()
            rc = -1
        rec = None
        if os.path.exists(ep_res_path):
            try:
                rec = json.load(open(ep_res_path))
            except Exception:
                rec = None
            finally:
                try:
                    os.remove(ep_res_path)
                except OSError:
                    pass
        return rec, rc

    for ep in range(args.n_episodes):
        ep_res_path = os.path.join(res_dir, f".ep_pi05_{ep}_{os.getpid()}.json")
        # Same seed = same scene. A sim-DNF (worker wrote no result) is retried up
        # to EVAL_EP_RETRIES times on the IDENTICAL scene; an honest model failure
        # (steps>0, success=False) is a REAL outcome and is never retried.
        rec = None
        for attempt in range(1, max_ep_tries + 1):
            tag = f"episode {ep+1}/{args.n_episodes} (seed {seed_base+ep})"
            if attempt > 1:
                tag += f" [retry {attempt}/{max_ep_tries}]"
            print(f"\n[driver] ===== {tag} =====", flush=True)
            rec, rc = _spawn_worker(ep, ep_res_path)
            if rec is not None:
                break
            tail = (" — retrying same scene" if attempt < max_ep_tries
                    else " — giving up, recording sim-DNF")
            print(f"[driver] episode {ep} attempt {attempt}/{max_ep_tries} "
                  f"no result (exit={rc}){tail}", flush=True)
        if rec is None:
            rec = {"episode": ep, "success": False, "steps": 0}
        results.append(rec)
        rate = np.mean([r["success"] for r in results])
        print(f"[driver] EP {ep}: success={rec['success']} steps={rec['steps']} "
              f"| cumulative_rate={rate:.2f}", flush=True)
        _write_summary()

    summary = _write_summary()
    print("\n========== RESULTS ==========")
    print(json.dumps(summary, indent=2))
    if args.results_path:
        print(f"[driver] saved to {args.results_path}")


if __name__ == "__main__":
    main()
