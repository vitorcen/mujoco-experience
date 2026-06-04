# Live on-screen GUI rollout for RoboCasa GR1 Tabletop (recipe A, N1.5).
#
# The upstream gr00t.eval.simulation pipeline only renders OFFSCREEN (-> mp4). This
# client adds a native MuJoCo passive viewer so you can WATCH the GR1 robot move in
# real time while the policy drives it closed-loop.
#
# How it works (same trick as the OpenCabinet GUI demo):
#   - robosuite/robocasa is MuJoCo-based. We build the SAME single env the offscreen
#     eval uses (SyncVectorEnv n_envs=1 -> MultiStepWrapper -> RoboCasaEnv), reusing the
#     proven obs/action formatting, then grab the env's raw mjModel/mjData and open
#     `mujoco.viewer.launch_passive` on them.
#   - MUJOCO_GL=egl still drives the OFFSCREEN camera obs fed to the policy; the passive
#     viewer uses its own GLFW window. Two GL contexts, one shared physics state.
#   - MultiStepWrapper.step() calls the base env's step() once per sim step; we wrap that
#     base step to viewer.sync() after each, so motion is smooth (not 16-step jumps).
#
# Talks to the SAME ZMQ policy server as the headless eval (inference_service.py --server).
import argparse
import os
import time

import numpy as np

# Registers the 197 gr1_unified/* gym envs (a bare `import robocasa` does NOT).
import robocasa.utils.gym_utils  # noqa: F401
import mujoco
import mujoco.viewer

from gr00t.eval import simulation as _sim
from gr00t.eval.simulation import (
    MultiStepConfig,
    SimulationConfig,
    SimulationInferenceClient,
    VideoConfig,
)

# The N1.5 `_create_single_env` is the kitchen variant: it passes `split=` to gym.make,
# which the gr1 env creator rejects. Strip it (gr1-tabletop variant omits split).
_orig_gym_make = _sim.gym.make
def _gym_make_no_split(env_id, **kwargs):
    kwargs.pop("split", None)
    return _orig_gym_make(env_id, **kwargs)
_sim.gym.make = _gym_make_no_split


def _raw_mujoco(base_env):
    """Reach the raw mjModel/mjData behind a RoboCasaEnv gym env."""
    rs = base_env.env  # RoboCasaEnv.env is the underlying robosuite env
    return rs.sim.model._model, rs.sim.data._data


def main():
    ap = argparse.ArgumentParser(description="Live GUI rollout for GR1 tabletop (recipe A)")
    ap.add_argument("--env_name", required=True)
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--n_episodes", type=int, default=3)
    ap.add_argument("--max_episode_steps", type=int, default=720)
    ap.add_argument("--n_action_steps", type=int, default=16)
    ap.add_argument("--step_delay", type=float, default=0.02,
                    help="seconds to sleep after each sim step so motion is watchable")
    ap.add_argument("--camera", default="egoview",
                    help="model camera to view from (default egoview = robot head / policy view; "
                         "others: robot0_behindhead, robot0_agentview_center, robot0_frontview, ...). "
                         "Pass 'free' to keep the default free camera.")
    a = ap.parse_args()

    client = SimulationInferenceClient(host=a.host, port=a.port)
    print("modality configs:", list(client.get_modality_config().keys()))

    # n_envs=1 -> SyncVectorEnv runs in-process, so we can reach the inner env. video_dir
    # None -> no recording wrapper (live view only). Reuse the proven setup_environment.
    config = SimulationConfig(
        env_name=a.env_name, n_episodes=a.n_episodes, n_envs=1,
        video=VideoConfig(video_dir=None),
        multistep=MultiStepConfig(n_action_steps=a.n_action_steps,
                                  max_episode_steps=a.max_episode_steps),
    )
    vec = client.setup_environment(config)
    msw = vec.envs[0]            # MultiStepWrapper
    base = msw.unwrapped         # RoboCasaEnv gym env (innermost)
    model, data = _raw_mujoco(base)

    print(">>> opening MuJoCo passive viewer window (DISPLAY=%s) ..." % os.environ.get("DISPLAY"))
    viewer = mujoco.viewer.launch_passive(model, data)

    # The default free camera spawns outside the scene (you'd see the room from afar).
    # Lock the view to a model camera — default `egoview` = the robot's head camera, i.e.
    # exactly what the policy sees. `--camera free` keeps the orbit camera.
    if a.camera and a.camera != "free":
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, a.camera)
        if cam_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = cam_id
            print(f">>> viewer locked to camera '{a.camera}' (id {cam_id})")
        else:
            print(f">>> WARN: camera '{a.camera}' not found; keeping free camera")
    viewer.sync()

    # Sync the viewer after every base-env sim step -> smooth on-screen motion.
    _base_step = base.step
    def _step_render(act):
        out = _base_step(act)
        if viewer.is_running():
            viewer.sync()
            if a.step_delay > 0:
                time.sleep(a.step_delay)
        return out
    base.step = _step_render

    successes = []
    try:
        obs, _ = vec.reset()
        if viewer.is_running():
            viewer.sync()
        completed = 0
        cur_success = False
        while completed < a.n_episodes and viewer.is_running():
            actions = client._get_actions_from_server(obs)
            obs, rewards, terminations, truncations, infos = vec.step(actions)
            cur_success |= bool(infos["success"][0][0])
            if terminations[0] or truncations[0]:
                successes.append(cur_success)
                print(f"EP {len(successes)} success: {cur_success}; "
                      f"SR so far: {np.mean(successes):.2f}")
                cur_success = False
                completed += 1
    finally:
        try:
            vec.close()
        except Exception:
            pass
        if viewer.is_running():
            viewer.close()
    if successes:
        print(f"Success rate: {np.mean(successes):.2f} ({sum(successes)}/{len(successes)})")


if __name__ == "__main__":
    main()
