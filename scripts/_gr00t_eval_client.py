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
import signal
import subprocess
import sys
import time

import numpy as np
import zmq

# NOTE: robocasa / gymnasium / mujoco are imported LAZILY (see _lazy_env_imports),
# NOT at module top. Root cause of the old "random crash" benchmark failures
# (unknown opcode / line -1 / 'method' object is not iterable in pure-Python
# yaml/xml during env.reset()): a C-extension (most likely MuJoCo's GL/EGL)
# corrupts the heap across repeated robosuite env.reset() churn, and CPython
# 3.11's PEP 659 adaptive specializing interpreter surfaces that corruption as
# bizarre bytecode errors. PYTHONDONTWRITEBYTECODE=1 does NOT help (PEP 659
# quickening is in-memory, unrelated to .pyc). The fix is process isolation:
# the DRIVER never touches robocasa/mujoco (stays pristine) and spawns one fresh
# WORKER subprocess per episode, so corruption can never accumulate or leak
# across episodes. A worker that segfaults/opcode-crashes costs only its own
# episode — the driver records it and moves on.


def _lazy_env_imports():
    """Import the heavy (corruption-prone) sim stack. Worker process only."""
    import robocasa  # noqa: F401  (registers gym envs)
    import gymnasium as gym
    return gym


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
    def __init__(self, host, port, timeout_s=None, send_reset=False):
        # Env override for slow policies (e.g. NF4 DreamZero first chunk does a one-time
        # CPU UMT5 encode + ~85s denoise > default 120s). Default unchanged -> GR00T/pi0.5
        # benchmarks behave identically.
        if timeout_s is None:
            timeout_s = int(os.environ.get("POLICY_TIMEOUT_S", "120"))
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.setsockopt(zmq.RCVTIMEO, timeout_s * 1000)
        self.sock.setsockopt(zmq.SNDTIMEO, timeout_s * 1000)
        self.sock.connect(f"tcp://{host}:{port}")
        # When True, send a {"op":"reset"} at each episode start so a stateful
        # server (e.g. ACT with temporal ensembling) clears per-episode state.
        # Off by default so the GR00T server — which doesn't implement "reset" —
        # is unaffected.
        self.send_reset = send_reset

    def maybe_reset(self):
        if not self.send_reset:
            return
        self.sock.send(pickle.dumps({"op": "reset"}))
        reply = pickle.loads(self.sock.recv())
        if "error" in reply:
            raise RuntimeError(f"server reset error: {reply['error']}")

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
    # Tell a stateful server a new episode is starting (no-op unless send_reset).
    client.maybe_reset()
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
                # Optional per-step delay so a human can actually SEE the motion: a
                # 16-step chunk otherwise replays in <1 s then freezes while the next
                # chunk computes. Env-gated, default 0 -> headless benchmarks unaffected.
                _delay = float(os.environ.get("RENDER_STEP_DELAY_S", "0") or 0)
                if _delay > 0:
                    time.sleep(_delay)
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
    ap.add_argument("--send-reset", action="store_true",
                    help="Send a {op:reset} to the server at each episode start "
                         "(for stateful servers like ACT temporal ensembling). "
                         "Leave off for the GR00T server which has no reset op.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Base RNG seed. The driver gives episode N the seed "
                         "(base+N), so every policy faces the identical, "
                         "reproducible scene sequence — a FAIR benchmark. "
                         "robocasa's reset(seed) fully determines the scene "
                         "(layout/style + object & robot placement). Unset = "
                         "old random behaviour (unfair across policies).")
    # --- process-isolation knobs (internal; set by the driver, not the caller) ---
    ap.add_argument("--episode", type=int, default=None,
                    help="WORKER MODE: run exactly this one episode and exit. "
                         "When omitted, runs in DRIVER MODE (spawns one isolated "
                         "worker subprocess per episode — see module docstring).")
    ap.add_argument("--episode-result-path", default=None,
                    help="WORKER MODE: where to write this episode's {success,steps} "
                         "JSON. The driver reads it back after the worker exits.")
    args = ap.parse_args()

    if args.episode is None:
        return _driver_main(args)
    return _worker_main(args)

def _worker_main(args):
    """Run exactly ONE episode in this fresh process, then exit. All the
    corruption-prone work (robocasa import, env build, repeated reset churn) is
    confined here, so a crash takes only this episode down — not the whole run."""
    gym = _lazy_env_imports()
    ep = args.episode

    full_env_name = f"robocasa/{args.env_name}"
    print(f"[worker ep{ep}] making env {full_env_name} (split={args.split})", flush=True)
    env = gym.make(full_env_name, split=args.split, enable_render=True)

    # Force a reset BEFORE attaching the passive viewer so robosuite's offscreen
    # renderer initialises its GL context + uploads textures first. If we open
    # the passive viewer first, its GL context comes up before robocasa has
    # generated kitchen textures, and the viewer ends up rendering flat colours.
    # We feed this obs into the episode so we don't burn an extra random init.
    print(f"[worker ep{ep}] warm-up reset (seed={args.seed}) ...", flush=True)
    initial_obs, _ = env.reset(seed=args.seed) if args.seed is not None else env.reset()

    viewer = None
    if args.render:
        print(f"[worker ep{ep}] attaching passive MuJoCo viewer ...", flush=True)
        viewer = _attach_passive_viewer(env)
        warmup = max(0.0, args.render_warmup_s)
        if warmup > 0:
            print(f"[worker ep{ep}] ============================================", flush=True)
            print(f"[worker ep{ep}] >>> MuJoCo viewer opened. Bring it to focus. <<<", flush=True)
            print(f"[worker ep{ep}] >>> Policy starts in {warmup:.0f}s ...          <<<", flush=True)
            print(f"[worker ep{ep}] ============================================", flush=True)
            t_end = time.time() + warmup
            while time.time() < t_end:
                viewer.sync()
                time.sleep(0.05)
            print(f"[worker ep{ep}] starting policy now.", flush=True)

    client = PolicyClient(args.host, args.port, send_reset=args.send_reset)
    try:
        ok, steps = run_episode(env, client, args.n_action_steps,
                                args.max_steps, viewer=viewer,
                                initial_obs=initial_obs)
    except Exception as e:
        print(f"[worker ep{ep}] episode crashed: {e}", flush=True)
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
    """Spawn one isolated worker subprocess per episode, collect results, and
    write the same summary JSON the single-process client used to write.

    This process NEVER imports robocasa/mujoco, so it stays pristine no matter
    how badly a worker corrupts its own heap. A worker that segfaults or dies
    with a PEP-659 bytecode error simply leaves no result file → recorded as a
    sim-DNF (steps=0) and we move to the next episode."""
    results = []
    t0 = time.time()

    def _write_summary():
        if not args.results_path:
            return None
        summary = {
            "env_name": args.env_name,
            "split": args.split,
            "n_episodes": args.n_episodes,
            "n_completed": len(results),
            "success_rate": float(np.mean([r["success"] for r in results])) if results else 0.0,
            "wall_time_s": time.time() - t0,
            "results": results,
        }
        os.makedirs(os.path.dirname(args.results_path) or ".", exist_ok=True)
        tmp = args.results_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(summary, f, indent=2)
        os.replace(tmp, args.results_path)
        return summary

    # Hard per-episode wall-clock cap: a worker that hangs (e.g. MuJoCo GL stall)
    # must not block the whole benchmark. On timeout we kill the worker's whole
    # process group (it owns its server connection only; the persistent policy
    # server is the orchestrator's child, untouched) — this is what prevents the
    # 1.5h-zombie-client situation. Tune via EP_TIMEOUT_S.
    ep_timeout_s = int(os.environ.get("EP_TIMEOUT_S", "900"))
    # Fair benchmark by default: episode N gets seed (base+N), so every policy
    # faces the IDENTICAL reproducible scene sequence. Override base via --seed or
    # SEED_BASE. (robocasa target-split OpenCabinet has 1 fixed layout (4,6); the
    # per-reset object/robot placement is what the seed pins down.)
    seed_base = args.seed if args.seed is not None else int(os.environ.get("SEED_BASE", "0"))

    # Per-episode result handoff file lives next to the results JSON (or /tmp).
    res_dir = os.path.dirname(os.path.abspath(args.results_path)) if args.results_path else "/tmp"
    os.makedirs(res_dir, exist_ok=True)

    base_cmd = [sys.executable, "-u", os.path.abspath(__file__),
                "--env-name", args.env_name, "--split", args.split,
                "--n-action-steps", str(args.n_action_steps),
                "--max-steps", str(args.max_steps),
                "--host", args.host, "--port", str(args.port),
                "--n-episodes", "1"]
    if args.send_reset:
        base_cmd.append("--send-reset")
    if args.render:
        base_cmd += ["--render", "--render-warmup-s", str(args.render_warmup_s)]

    print(f"[driver] {args.n_episodes} episodes, one isolated worker each "
          f"(env={args.env_name} split={args.split} port={args.port})", flush=True)

    max_ep_tries = int(os.environ.get("EVAL_EP_RETRIES", "3"))

    def _spawn_worker(ep, ep_res_path):
        """One worker subprocess for episode `ep`. Returns (rec, rc): rec is the
        result dict, or None if the worker crashed/timed out before writing one.
        start_new_session → worker is its own process group, so a timeout kill
        takes down the worker AND any sim/GL child. stdio inherited → live logs."""
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
        ep_res_path = os.path.join(res_dir, f".ep_{ep}_{os.getpid()}.json")
        # Same seed = same scene. A sim-DNF (worker wrote no result: segfault /
        # PEP-659 opcode / timeout) is retried up to EVAL_EP_RETRIES times on the
        # IDENTICAL scene — robocasa's crash is probabilistic, so a fresh process
        # usually gets through. An honest model failure (steps>0, success=False)
        # is a REAL outcome and is never retried.
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

    summary = _write_summary() or {
        "env_name": args.env_name, "split": args.split,
        "n_episodes": args.n_episodes, "n_completed": len(results),
        "success_rate": float(np.mean([r["success"] for r in results])) if results else 0.0,
        "wall_time_s": time.time() - t0, "results": results,
    }
    print("\n========== RESULTS ==========")
    print(json.dumps(summary, indent=2))
    if args.results_path:
        print(f"[driver] saved to {args.results_path}")


if __name__ == "__main__":
    main()
