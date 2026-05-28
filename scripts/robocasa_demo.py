"""RoboCasa demo launcher — thin driver invoked from RoboCasa.ipynb.

Subcommands:
  setup          run install_robocasa_env.sh (idempotent)
  status         report env / dataset / asset state
  list           list all demo keys
  launch <key>   spawn the demo in the `robocasa` conda env (real-time viewer)
  kill           kill the running viewer
  run-internal   (private) the body that runs inside the robocasa env

Demo keys look like `scene:<id>`, `task:<id>`, `robot:<id>`. See REGISTRY.
"""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_NAME = os.environ.get("ROBOCASA_ENV_NAME", "robocasa")
PID_FILE = Path("/tmp/robocasa_demo.pid")

# (task_name, robot, layout, style, description)
# layout/style: -1 = random, otherwise 0..9 (see robocasa LayoutType / StyleType)
REGISTRY = {
    "scene:browse": ("Kitchen", "PandaOmron", 1, 1, "Layout 1 + Style 1 (standard kitchen, PandaOmron)"),
    "scene:random": ("Kitchen", "PandaOmron", -1, -1, "Random layout + style"),
    "scene:island":  ("Kitchen", "PandaOmron", 3, 2, "Island kitchen (layout 3 style 2)"),
    "scene:galley":  ("Kitchen", "PandaOmron", 4, 4, "Galley kitchen (layout 4 style 4)"),

    "task:pick_place_cab": ("PickPlaceCounterToCabinet", "PandaOmron", 1, 1, "Pick & place: counter → cabinet"),
    "task:pick_place_sink": ("PickPlaceCounterToSink", "PandaOmron", 1, 1, "Pick & place: counter → sink"),
    "task:open_door":     ("OpenCabinet", "PandaOmron", 1, 1, "Open cabinet door"),
    "task:close_drawer":  ("CloseDrawer", "PandaOmron", 1, 1, "Close drawer"),
    "task:turn_on_stove": ("TurnOnStove", "PandaOmron", 1, 1, "Turn on stove"),
    "task:turn_on_faucet": ("TurnOnSinkFaucet", "PandaOmron", 1, 1, "Turn on sink faucet"),
    "task:microwave":     ("TurnOnMicrowave", "PandaOmron", 1, 1, "Turn on microwave"),
}
# Note: humanoid (GR1) and mobile-base (Tiago) robots require additional
# controllers that robosuite master ships configs for but doesn't actually
# implement (WHOLE_BODY_MINK_IK) — dropped until upstream lands them.


def _conda_run(args, extra_env=None):
    """Run a python command inside the robocasa conda env via `conda run`."""
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    cmd = ["conda", "run", "-n", ENV_NAME, "--no-capture-output", "python"] + list(args)
    return subprocess.run(cmd, env=env)


def cmd_setup(_args):
    script = REPO_ROOT / "scripts" / "install_robocasa_env.sh"
    return subprocess.run(["bash", str(script)]).returncode


def cmd_status(_args):
    print(f"REPO_ROOT      = {REPO_ROOT}")
    print(f"ENV_NAME       = {ENV_NAME}")
    print(f"ROBOCASA_DATA_PATH = {os.environ.get('ROBOCASA_DATA_PATH', '(unset)')}")
    print(f"PID_FILE exists = {PID_FILE.exists()} ({PID_FILE.read_text().strip() if PID_FILE.exists() else ''})")
    print("--- robocasa import check ---")
    r = subprocess.run(
        ["conda", "run", "-n", ENV_NAME, "--no-capture-output", "python", "-c",
         "import robocasa, robosuite, mujoco; "
         "print(f'robocasa  {robocasa.__version__} @ {robocasa.__file__}'); "
         "print(f'robosuite {robosuite.__version__}'); "
         "print(f'mujoco    {mujoco.__version__}'); "
         "import os; "
         "asset_dir = os.path.join(os.path.dirname(robocasa.__file__), 'models/assets'); "
         "print(f'assets present: textures={os.path.isdir(asset_dir+\"/textures\")}  fixtures={os.path.isdir(asset_dir+\"/fixtures\")}  objaverse={os.path.isdir(asset_dir+\"/objects/objaverse\")}')"
        ]
    )
    return r.returncode


def cmd_list(_args):
    print(f"{'KEY':<22} {'TASK':<24} {'ROBOT':<22} LAYOUT STYLE  DESCRIPTION")
    print("-" * 110)
    for k, (task, robot, layout, style, desc) in REGISTRY.items():
        print(f"{k:<22} {task:<24} {robot:<22} {layout:>6} {style:>5}  {desc}")
    return 0


def cmd_kill(_args):
    if not PID_FILE.exists():
        print("no PID file")
        return 0
    pid = int(PID_FILE.read_text().strip())
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
        print(f"sent SIGTERM to pgid of pid {pid}")
    except ProcessLookupError:
        print(f"pid {pid} already gone")
    PID_FILE.unlink(missing_ok=True)
    return 0


def _clear_stale_pyc():
    """Wipe .pyc caches that the Python 3.11 adaptive interpreter sometimes
    leaves in an unloadable state (manifests as `SystemError: unknown opcode`
    or `TypeError: __init__() should return None, not 'type'` mid-import)."""
    import shutil
    targets = [REPO_ROOT / "dependencies" / "robocasa"]
    sp = Path(f"/home/david/miniconda3/envs/{ENV_NAME}/lib/python3.11/site-packages")
    if sp.exists():
        for pkg in ("gymnasium", "gym", "robosuite", "robosuite_models"):
            targets.append(sp / pkg)
    for root in targets:
        if not root.exists():
            continue
        for cache in root.rglob("__pycache__"):
            shutil.rmtree(cache, ignore_errors=True)


def cmd_launch(args):
    key = args.key
    if key not in REGISTRY:
        print(f"unknown demo key: {key}\nUse `list` to see available keys.")
        return 2
    _clear_stale_pyc()
    cmd_kill(None)  # ensure single instance
    task, robot, layout, style, desc = REGISTRY[key]
    print(f"=== {key} : {desc} ===")
    print(f"task={task} robot={robot} layout={layout} style={style}")
    # Spawn inside robocasa env as own process group so kill works.
    # -u to flush prints immediately so the notebook output isn't stuck buffered.
    # PYTHONDONTWRITEBYTECODE: avoid quickened-opcode .pyc poisoning.
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = ["conda", "run", "-n", ENV_NAME, "--no-capture-output", "python", "-u",
           str(REPO_ROOT / "scripts" / "robocasa_demo.py"), "run-internal",
           "--task", task, "--robot", robot,
           "--layout", str(layout), "--style", str(style)]

    # Retry on transient PEP 659 specializer crashes (unknown opcode, corrupted
    # double-linked list, IndexError mid-import, TypeError on type checks).
    # If the child died inside the first 10 s without ever printing "viewer up",
    # restart it. After "viewer up" the user is in the viewer — never retry.
    MAX_TRIES = 3
    for attempt in range(1, MAX_TRIES + 1):
        log_path = Path(f"/tmp/robocasa_demo_{os.getpid()}.log")
        with log_path.open("w") as logf:
            p = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                 start_new_session=True)
            PID_FILE.write_text(str(p.pid))
            viewer_up = False
            t0 = time.time()
            while p.poll() is None:
                time.sleep(0.5)
                # tail-check for "viewer up" marker
                try:
                    if "viewer up" in log_path.read_text():
                        viewer_up = True
                        break
                except FileNotFoundError:
                    pass
                if time.time() - t0 > 30 and not viewer_up:
                    break
            if viewer_up:
                # Stream the log so the user sees what's happening, then wait.
                sys.stdout.write(log_path.read_text())
                sys.stdout.flush()
                p.wait()
                PID_FILE.unlink(missing_ok=True)
                log_path.unlink(missing_ok=True)
                return 0
            # Child exited (or hung) without bringing up viewer — likely PEP 659 crash.
            rc = p.poll() if p.poll() is not None else -1
            if p.poll() is None:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                p.wait()
            log_text = log_path.read_text() if log_path.exists() else ""
            log_path.unlink(missing_ok=True)
            PID_FILE.unlink(missing_ok=True)
            # Any pre-viewer death is treated as transient (Python 3.11 PEP 659
            # specializer is the usual culprit). Cheap to restart — ~5 s.
            if attempt < MAX_TRIES:
                print(f"[attempt {attempt}/{MAX_TRIES}] child died before viewer launched, retrying ...")
                continue
            sys.stdout.write(log_text)
            sys.stdout.flush()
            return rc
    return 1


def cmd_run_internal(args):
    """Runs inside the robocasa conda env. Opens an interactive mujoco viewer."""
    import sys, types
    # robocasa/__init__.py auto-imports robocasa.wrappers.gym_wrapper which then
    # loops `gymnasium.register(...)` over ~700 envs. Under Python 3.11's
    # adaptive interpreter that loop intermittently crashes mid-import with
    # `SystemError: unknown opcode`, `TypeError: called match pattern must be a
    # type`, or `corrupted double-linked list` — classic PEP 659 specialization
    # bugs. We don't use the gym wrapper (we call robosuite.make directly), so
    # we stub the module before importing robocasa.
    stub = types.ModuleType("robocasa.wrappers.gym_wrapper")
    stub.RoboCasaGymEnv = type("RoboCasaGymEnv", (), {})
    sys.modules["robocasa.wrappers.gym_wrapper"] = stub

    import numpy as np
    import robosuite
    import robocasa  # noqa: F401  — registers kitchen envs into robosuite REGISTERED_ENVS
    from robosuite.controllers import load_composite_controller_config

    config = {
        "env_name": args.task,
        "robots": args.robot,
        "controller_configs": load_composite_controller_config(robot=args.robot),
        "translucent_robot": False,
    }
    print(f"[run-internal] robosuite.make({config['env_name']}, robot={args.robot}) ...")
    env = robosuite.make(
        **config,
        has_renderer=True,
        has_offscreen_renderer=False,
        # render_camera=None keeps the viewer in FREE (orbit) mode so the user
        # can drag the mouse to look around. We aim the free camera at the
        # robot below — robosuite's DEFAULT_FREE_CAM aims at world origin which
        # is 14m away from where robocasa drops the robot, hence "in the wall".
        render_camera=None,
        ignore_done=True,
        use_camera_obs=False,
        control_freq=20,
        renderer="mjviewer",
    )
    if args.layout >= 0 and args.style >= 0:
        env.layout_and_style_ids = [[args.layout, args.style]]

    env.reset()
    # Derive the orbit camera's initial pose from robot0_agentview_center —
    # the canonical 3rd-person view the robocasa authors chose for each layout.
    # Different kitchen layouts face the robot different ways; a hard-coded
    # azimuth ends up inside a wall on half of them.
    cam_id = env.sim.model.camera_name2id("robot0_agentview_center")
    cam_pos = env.sim.data.cam_xpos[cam_id].copy()
    cam_mat = env.sim.data.cam_xmat[cam_id].reshape(3, 3)
    view_dir = -cam_mat[:, 2]  # mujoco cameras look down their local -z axis
    LOOK_AHEAD = 1.5
    lookat = cam_pos + view_dir * LOOK_AHEAD
    delta = cam_pos - lookat  # = -view_dir * LOOK_AHEAD
    distance = float(np.linalg.norm(delta))
    # MuJoCo free camera: eye = lookat - (cos(az)cos(el), sin(az)cos(el), sin(el))*dist
    # All three components of (eye - lookat) carry a minus sign w.r.t. spherical
    # coords, so atan2 takes -delta and arcsin takes -delta.z. Verified against
    # robosuite DEFAULT_FREE_CAM (az=180 el=-20 lookat=[0,0,1] dist=2 → eye
    # ≈ (1.88, 0, 1.68), an over-the-table view).
    elevation = float(-np.degrees(np.arcsin(delta[2] / distance)))
    azimuth = float(np.degrees(np.arctan2(-delta[1], -delta[0])))
    env.viewer.camera_config = {
        "lookat": [float(lookat[0]), float(lookat[1]), float(lookat[2])],
        "distance": distance,
        "azimuth": azimuth,
        "elevation": elevation,
    }
    print("[run-internal] viewer up. Random low-amplitude actions. Ctrl-C to quit.")
    low, high = env.action_spec
    try:
        while True:
            action = np.random.uniform(low, high) * 0.0  # idle by default; viewer interactive
            obs, reward, done, info = env.step(action)
            env.render()
    except KeyboardInterrupt:
        print("[run-internal] interrupted")
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("setup")
    sub.add_parser("status")
    sub.add_parser("list")
    sub.add_parser("kill")
    p_launch = sub.add_parser("launch")
    p_launch.add_argument("key")
    p_run = sub.add_parser("run-internal")
    p_run.add_argument("--task", required=True)
    p_run.add_argument("--robot", required=True)
    p_run.add_argument("--layout", type=int, default=-1)
    p_run.add_argument("--style", type=int, default=-1)

    args = p.parse_args()
    if args.cmd == "setup":      return cmd_setup(args)
    if args.cmd == "status":     return cmd_status(args)
    if args.cmd == "list":       return cmd_list(args)
    if args.cmd == "kill":       return cmd_kill(args)
    if args.cmd == "launch":     return cmd_launch(args)
    if args.cmd == "run-internal": return cmd_run_internal(args)
    return 1


if __name__ == "__main__":
    sys.exit(main() or 0)
