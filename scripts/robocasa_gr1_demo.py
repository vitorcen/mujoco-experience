"""RoboCasa **GR1 Tabletop** demo driver — thin driver invoked from RoboCasa-Tabletop.ipynb.

GR1 tabletop is a *separate* line from the kitchen OpenCabinet work: a GR1 humanoid
(Fourier hands + waist) doing 24 tabletop pick-and-place tasks. Different repo
(robocasa/robocasa-gr1-tabletop-tasks), different env (`robocasa_gr1`), different
embodiment (`gr1` / data config `fourier_gr1_arms_waist`).

Subcommands:
  status                 report env / repo / assets / checkpoint / clip state
  list                   list the 24 tabletop tasks (3 naming forms + group)
  clip <TASK>            download ONE ego_view demo mp4 from the HF LeRobot dataset
                         (network only — no conda env, no GPU). Instant scene preview.
  download-ckpt [NAME]   download a checkpoint (network only). NAME in CKPTS (default: youliangtan)
  render <TASK>          [GPU] offscreen-render the scene with random actions -> mp4
  eval <TASK>            [GPU] run the GR00T policy closed-loop -> mp4 + success rate

Naming: a task NAME (e.g. PnPCupToDrawerClose) maps to
  env id          gr1_unified/<NAME>_GR1ArmsAndWaistFourierHands_Env   (demo_task / eval)
  lerobot folder  gr1_unified.<NAME>                                   (HF dataset clips)
"""
import argparse
import os
import signal
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_NAME = os.environ.get("ROBOCASA_GR1_ENV_NAME", "robocasa_gr1")
DATA_DIR = Path(os.environ.get("ROBOCASA_GR1_DATA", os.path.expanduser("~/.cache/robocasa_gr1")))
GR00T_DIR = REPO_ROOT / "dependencies" / "Isaac-GR00T-gr1"
GR1_DIR = REPO_ROOT / "dependencies" / "robocasa-gr1-tabletop-tasks"

# Closed-loop eval is N1.5 "recipe A": the downloaded youliangtan checkpoint is N1.5, so
# it needs the N1.5 gr00t (dependencies/Isaac-GR00T, the "GR00T N1.5 for RoboCasa" tree,
# which ships gr00t.eval.simulation) co-installed with the gr1 robocasa fork — a SEPARATE
# env from `robocasa_gr1` (built on N1.7 main). Build it with install_robocasa_gr1_n15_env.sh.
N15_ENV = os.environ.get("ROBOCASA_GR1_N15_ENV_NAME", "robocasa_gr1_n15")
N15_GR00T_DIR = REPO_ROOT / "dependencies" / "Isaac-GR00T"
SIM_CLIENT = REPO_ROOT / "scripts" / "_gr1_simulation_service.py"
GUI_CLIENT = REPO_ROOT / "scripts" / "_gr1_gui_client.py"

DATASET_REPO = "nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim"
ENV_SUFFIX = "_GR1ArmsAndWaistFourierHands_Env"

# 24 tabletop tasks. (NAME, group, human description)
# group: "core" = 6 PnP-Close tasks (1000 teleop demos each), "novel" = 18 Posttrain split-A.
TASKS = [
    ("PnPCupToDrawerClose",       "core",  "杯 → 抽屉，关抽屉 / cup → drawer, close"),
    ("PnPPotatoToMicrowaveClose", "core",  "土豆 → 微波炉，关门 / potato → microwave, close"),
    ("PnPMilkToMicrowaveClose",   "core",  "牛奶 → 微波炉，关门 / milk → microwave, close"),
    ("PnPBottleToCabinetClose",   "core",  "瓶 → 柜子，关门 / bottle → cabinet, close"),
    ("PnPWineToCabinetClose",     "core",  "红酒 → 柜子，关门 / wine → cabinet, close"),
    ("PnPCanToDrawerClose",       "core",  "罐 → 抽屉，关抽屉 / can → drawer, close"),
    ("PosttrainPnPNovelFromCuttingboardToBasketSplitA",      "novel", "砧板 → 篮子 / cuttingboard → basket"),
    ("PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA","novel", "砧板 → 纸箱 / cuttingboard → box"),
    ("PosttrainPnPNovelFromCuttingboardToPanSplitA",         "novel", "砧板 → 平底锅 / cuttingboard → pan"),
    ("PosttrainPnPNovelFromCuttingboardToPotSplitA",         "novel", "砧板 → 汤锅 / cuttingboard → pot"),
    ("PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA","novel", "砧板 → 多层篮 / cuttingboard → tiered basket"),
    ("PosttrainPnPNovelFromPlacematToBasketSplitA",          "novel", "餐垫 → 篮子 / placemat → basket"),
    ("PosttrainPnPNovelFromPlacematToBowlSplitA",            "novel", "餐垫 → 碗 / placemat → bowl"),
    ("PosttrainPnPNovelFromPlacematToPlateSplitA",           "novel", "餐垫 → 盘子 / placemat → plate"),
    ("PosttrainPnPNovelFromPlacematToTieredshelfSplitA",     "novel", "餐垫 → 多层架 / placemat → tiered shelf"),
    ("PosttrainPnPNovelFromPlateToBowlSplitA",               "novel", "盘 → 碗 / plate → bowl"),
    ("PosttrainPnPNovelFromPlateToCardboardboxSplitA",       "novel", "盘 → 纸箱 / plate → box"),
    ("PosttrainPnPNovelFromPlateToPanSplitA",                "novel", "盘 → 平底锅 / plate → pan"),
    ("PosttrainPnPNovelFromPlateToPlateSplitA",              "novel", "盘 → 盘 / plate → plate"),
    ("PosttrainPnPNovelFromTrayToCardboardboxSplitA",        "novel", "托盘 → 纸箱 / tray → box"),
    ("PosttrainPnPNovelFromTrayToPlateSplitA",               "novel", "托盘 → 盘 / tray → plate"),
    ("PosttrainPnPNovelFromTrayToPotSplitA",                 "novel", "托盘 → 汤锅 / tray → pot"),
    ("PosttrainPnPNovelFromTrayToTieredbasketSplitA",        "novel", "托盘 → 多层篮 / tray → tiered basket"),
    ("PosttrainPnPNovelFromTrayToTieredshelfSplitA",         "novel", "托盘 → 多层架 / tray → tiered shelf"),
]
TASK_NAMES = [t[0] for t in TASKS]

# Checkpoint registry. Each: (repo_id, subdir, allow_patterns or None=all, note)
CKPTS = {
    # ⭐ default: GR00T author (You Liang Tan), N1.5 post-trained on the 24 tabletop tasks.
    "youliangtan": (
        "youliangtan/gr00t-n1.5-robocasa-tabletop-posttrain",
        "gr00t-n1.5-tabletop-posttrain",
        None,
        "N1.5 post-trained, ~7.6GB, paper ~47% (GR00T author). No model card but standard layout.",
    ),
    # base foundation model — zero-shot ~42% on these tasks.
    "base": (
        "nvidia/GR00T-N1.5-3B",
        "GR00T-N1.5-3B",
        None,
        "Base foundation model, zero-shot ~42%.",
    ),
    # N1.6 community SFT (skip the 13GB optimizer.pt — inference doesn't need it).
    "n16": (
        "karthikpythireddi93/gr00t-n16-gr1-tabletop-sft",
        "gr00t-n16-tabletop-sft",
        ["*.safetensors", "*.json", "experiment_cfg/*"],
        "N1.6 community SFT, ~9.7GB (optimizer skipped).",
    ),
}


def env_id(name):
    return f"gr1_unified/{name}{ENV_SUFFIX}"


def lerobot_folder(name):
    return f"gr1_unified.{name}"


def _resolve(name_or_idx):
    """Accept a full NAME, a unique substring, or a 1-based index."""
    s = name_or_idx
    if s.isdigit():
        i = int(s) - 1
        if 0 <= i < len(TASK_NAMES):
            return TASK_NAMES[i]
    if s in TASK_NAMES:
        return s
    hits = [n for n in TASK_NAMES if s.lower() in n.lower()]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        print(f"ambiguous task '{s}', matches: {hits}", file=sys.stderr)
    else:
        print(f"unknown task '{s}'. Use `list` to see the 24 tasks.", file=sys.stderr)
    return None


# ---------------------------------------------------------------------------
def cmd_status(_a):
    print(f"REPO_ROOT          = {REPO_ROOT}")
    print(f"ENV_NAME           = {ENV_NAME}")
    print(f"ROBOCASA_GR1_DATA  = {DATA_DIR}")
    have_env = subprocess.run(["conda", "env", "list"], capture_output=True, text=True).stdout
    print(f"conda env exists   = {ENV_NAME in have_env.split()}")
    print(f"Isaac-GR00T-gr1    = {GR00T_DIR.exists()}  ({GR00T_DIR})")
    print(f"  eval_gr00t_robocasa.py = {(GR00T_DIR / 'examples/robocasa-gr1-tabletop-tasks/eval_gr00t_robocasa.py').exists()}")
    print(f"  inference_service.py   = {(GR00T_DIR / 'scripts/inference_service.py').exists()}")
    print(f"robocasa-gr1 repo  = {GR1_DIR.exists()}  ({GR1_DIR})")
    asset_dir = GR1_DIR / "robocasa/models/assets/objects"
    print(f"  tabletop assets  = sketchfab:{(asset_dir/'sketchfab').exists()} lightwheel:{(asset_dir/'lightwheel').exists()}")
    ck = DATA_DIR / "checkpoints"
    print(f"checkpoints dir    = {ck} (exists={ck.exists()})")
    if ck.exists():
        for d in sorted(ck.iterdir()):
            if d.is_dir():
                print(f"  - {d.name}")
    clips = DATA_DIR / "preview_clips"
    if clips.exists():
        n = len(list(clips.glob("*.mp4")))
        print(f"preview clips      = {n} mp4 in {clips}")
    return 0


def cmd_list(_a):
    print(f"{'#':>2}  {'TASK NAME':<48} {'GROUP':<6} DESCRIPTION")
    print("-" * 100)
    for i, (name, grp, desc) in enumerate(TASKS, 1):
        print(f"{i:>2}  {name:<48} {grp:<6} {desc}")
    print(f"\nenv id form     : gr1_unified/<NAME>{ENV_SUFFIX}")
    print(f"lerobot folder  : gr1_unified.<NAME>   (for `clip`)")
    return 0


def cmd_clip(a):
    """Download ONE ego_view demo mp4 straight from the HF LeRobot dataset.
    No conda env, no GPU — the lightest possible scene preview."""
    name = _resolve(a.task)
    if not name:
        return 2
    from huggingface_hub import hf_hub_download
    ep = a.episode
    rel = (f"LeRobot/{lerobot_folder(name)}/videos/chunk-000/"
           f"observation.images.ego_view/episode_{ep:06d}.mp4")
    out_dir = DATA_DIR / "preview_clips"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f">>> downloading demo clip: {rel}")
    try:
        p = hf_hub_download(repo_id=DATASET_REPO, repo_type="dataset", filename=rel,
                            local_dir=str(out_dir / "_hf"))
    except Exception as e:
        print(f"ERROR: {e}\n(try a different --episode 0..999)", file=sys.stderr)
        return 1
    dst = out_dir / f"{name}.ep{ep:06d}.mp4"
    import shutil
    shutil.copy(p, dst)
    print(f">>> saved: {dst}")
    print(dst)  # last line = path, for the notebook to pick up
    return 0


def cmd_download_ckpt(a):
    key = a.name
    if key not in CKPTS:
        print(f"unknown ckpt '{key}'. Choices: {list(CKPTS)}", file=sys.stderr)
        return 2
    repo_id, subdir, patterns, note = CKPTS[key]
    from huggingface_hub import snapshot_download
    target = DATA_DIR / "checkpoints" / subdir
    target.mkdir(parents=True, exist_ok=True)
    print(f">>> [{key}] {repo_id}\n    {note}\n    -> {target}")
    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=str(target),
                      allow_patterns=patterns, max_workers=a.workers)
    print(f">>> done: {target}")
    print(target)
    return 0


def _conda_run(args, cwd=None, background=False, extra_env=None, env_name=None):
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    cmd = ["conda", "run", "-n", env_name or ENV_NAME, "--no-capture-output", "python", "-u"] + list(args)
    if background:
        return subprocess.Popen(cmd, cwd=cwd, env=env)
    return subprocess.run(cmd, cwd=cwd, env=env)


def cmd_render(a):
    """[GPU] Offscreen-render the scene with random actions -> mp4 (demo_task.py)."""
    name = _resolve(a.task)
    if not name:
        return 2
    out_dir = DATA_DIR / "render"
    out_dir.mkdir(parents=True, exist_ok=True)
    script = GR1_DIR / "robocasa" / "scripts" / "demo_task.py"
    print(f">>> rendering {env_id(name)} (random actions, offscreen) ...")
    # demo_task.py writes mp4 into <cwd>/video/
    r = _conda_run([str(script), env_id(name)], cwd=str(out_dir),
                   extra_env={"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"})
    vids = sorted((out_dir / "video").glob("*.mp4"), key=lambda p: p.stat().st_mtime)
    if vids:
        print(f">>> newest video: {vids[-1]}")
        print(vids[-1])
    return r.returncode


def _kill_group(p):
    """SIGKILL the whole process group of a Popen started with start_new_session=True."""
    if p is None:
        return
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGKILL)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass


def _start_n15_server(ckpt, port, warmup, retries):
    """Launch the N1.5 inference server in robocasa_gr1_n15, poll its log for the ready
    marker, and retry the flaky import/load SIGSEGV (model loads, then a heisenbug
    segfaults ~1-in-3). Returns the Popen once ready, or None if it never came up."""
    import time
    server_py = N15_GR00T_DIR / "scripts" / "inference_service.py"
    srv_log = Path("/tmp/gr1_n15_server.log")
    egl = {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}
    server_cmd = [str(server_py), "--server", "--model_path", ckpt,
                  "--embodiment_tag", "gr1", "--data_config", "fourier_gr1_arms_waist",
                  "--denoising_steps", "4", "--port", str(port)]
    for attempt in range(1, retries + 1):
        print(f">>> [{N15_ENV}] starting N1.5 server (attempt {attempt}/{retries}, "
              f"model={ckpt}, port={port}) ...")
        lf = open(srv_log, "w")
        cmd = ["conda", "run", "-n", N15_ENV, "--no-capture-output", "python", "-u"] + server_cmd
        env = os.environ.copy(); env.update(egl)
        srv = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env,
                               start_new_session=True)
        deadline = time.time() + warmup
        while time.time() < deadline:
            if srv_log.exists() and "Server is ready and listening" in srv_log.read_text(errors="ignore"):
                print(">>> server ready.")
                return srv
            if srv.poll() is not None:  # crashed (segfault) before ready
                print(f">>> server died (exit {srv.returncode}) during load; retrying ...")
                break
            time.sleep(3)
        else:
            print(">>> server didn't become ready before warmup deadline; retrying ...")
        _kill_group(srv)
    return None


def cmd_eval(a):
    """[GPU] Closed-loop policy eval: inference server + sim client -> mp4 + SR.

    N1.5 "recipe A" (matches the youliangtan N1.5 checkpoint), both processes in the
    `robocasa_gr1_n15` env (N1.5 gr00t + gr1 robocasa fork). Build it first with
    `scripts/install_robocasa_gr1_n15_env.sh`. Offscreen render (MUJOCO_GL=egl) -> mp4.
      server : dependencies/Isaac-GR00T/scripts/inference_service.py --server
               --embodiment_tag gr1 --data_config fourier_gr1_arms_waist
      client : scripts/_gr1_simulation_service.py --client --env_name gr1_unified/<task>
    """
    name = _resolve(a.task)
    if not name:
        return 2
    ckpt = a.ckpt or str(DATA_DIR / "checkpoints" / CKPTS["youliangtan"][1])
    if not Path(ckpt).exists():
        print(f"ckpt not found: {ckpt}\nrun `download-ckpt` first.", file=sys.stderr)
        return 1
    server_py = N15_GR00T_DIR / "scripts" / "inference_service.py"
    if not SIM_CLIENT.exists() or not server_py.exists():
        print(f"missing eval scripts: server={server_py} client={SIM_CLIENT}", file=sys.stderr)
        return 1
    port = a.port
    vid = DATA_DIR / "eval_videos"
    vid.mkdir(parents=True, exist_ok=True)
    egl = {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}

    srv = _start_n15_server(ckpt, port, a.server_warmup, a.server_retries)
    if srv is None:
        print("ERROR: server never came up; see /tmp/gr1_n15_server.log", file=sys.stderr)
        return 1
    try:
        client_cmd = [str(SIM_CLIENT), "--client", "--env_name", env_id(name),
                      "--video_dir", str(vid), "--n_envs", str(a.n_envs),
                      "--n_episodes", str(a.n_episodes),
                      "--max_episode_steps", str(a.max_episode_steps),
                      "--n_action_steps", str(a.n_action_steps),
                      "--port", str(port)]
        # The client crashes ~25% of the time at IMPORT (mujoco/robosuite EGL GL-init is
        # flaky under MUJOCO_GL=egl — manifests as SIGSEGV or a bogus AttributeError).
        # That happens before any episode runs, so retrying is safe and idempotent; the
        # server stays up across attempts (no model reload). Once import clears, rollout
        # is stable.
        rc = 1
        for attempt in range(1, a.server_retries + 1):
            print(f">>> running sim client on {env_id(name)} (attempt {attempt}/{a.server_retries}, "
                  f"{a.n_episodes} ep, n_envs={a.n_envs}) ...")
            r = _conda_run(client_cmd, env_name=N15_ENV, extra_env=egl)
            rc = r.returncode
            if rc == 0:
                break
            print(f">>> client exited {rc} (likely flaky EGL import crash); retrying ...")
        print(f">>> videos in: {vid}")
        return rc
    finally:
        _kill_group(srv)


def cmd_gui(a):
    """[GPU+DISPLAY] Live on-screen rollout: same N1.5 server, but a native MuJoCo
    passive-viewer window so you WATCH the GR1 robot move in real time (not a recorded
    mp4). Must run from a session with a display (DISPLAY=:0)."""
    name = _resolve(a.task)
    if not name:
        return 2
    ckpt = a.ckpt or str(DATA_DIR / "checkpoints" / CKPTS["youliangtan"][1])
    if not Path(ckpt).exists():
        print(f"ckpt not found: {ckpt}\nrun `download-ckpt` first.", file=sys.stderr)
        return 1
    if not GUI_CLIENT.exists():
        print(f"missing GUI client: {GUI_CLIENT}", file=sys.stderr)
        return 1
    os.environ.setdefault("DISPLAY", ":0")
    port = a.port
    # Server renders offscreen obs (egl); the viewer opens its own GLFW window.
    egl = {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl", "DISPLAY": os.environ["DISPLAY"]}
    srv = _start_n15_server(ckpt, port, a.server_warmup, a.server_retries)
    if srv is None:
        print("ERROR: server never came up; see /tmp/gr1_n15_server.log", file=sys.stderr)
        return 1
    try:
        gui_cmd = [str(GUI_CLIENT), "--env_name", env_id(name),
                   "--n_episodes", str(a.n_episodes),
                   "--max_episode_steps", str(a.max_episode_steps),
                   "--n_action_steps", str(a.n_action_steps),
                   "--step_delay", str(a.step_delay),
                   "--camera", a.camera,
                   "--port", str(port)]
        # Same flaky-EGL-import retry as cmd_eval (crash happens before the window opens).
        rc = 1
        for attempt in range(1, a.server_retries + 1):
            print(f">>> launching live GUI viewer on {env_id(name)} "
                  f"(attempt {attempt}/{a.server_retries}, DISPLAY={os.environ['DISPLAY']}) ...")
            r = _conda_run(gui_cmd, env_name=N15_ENV, extra_env=egl)
            rc = r.returncode
            if rc == 0:
                break
            print(f">>> GUI client exited {rc} (likely flaky EGL import crash); retrying ...")
        return rc
    finally:
        _kill_group(srv)


def main():
    p = argparse.ArgumentParser(description="RoboCasa GR1 Tabletop demo driver")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("status")
    sub.add_parser("list")
    pc = sub.add_parser("clip"); pc.add_argument("task"); pc.add_argument("--episode", type=int, default=0)
    pd = sub.add_parser("download-ckpt"); pd.add_argument("name", nargs="?", default="youliangtan")
    pd.add_argument("--workers", type=int, default=8)
    pr = sub.add_parser("render"); pr.add_argument("task")
    pe = sub.add_parser("eval"); pe.add_argument("task")
    pe.add_argument("--ckpt", default=None); pe.add_argument("--n-episodes", type=int, default=10)
    pe.add_argument("--n-envs", type=int, default=5); pe.add_argument("--server-warmup", type=int, default=120)
    pe.add_argument("--server-retries", type=int, default=4)
    pe.add_argument("--port", type=int, default=5556)
    pe.add_argument("--max-episode-steps", type=int, default=720)
    pe.add_argument("--n-action-steps", type=int, default=16)
    pg = sub.add_parser("gui"); pg.add_argument("task")
    pg.add_argument("--ckpt", default=None); pg.add_argument("--n-episodes", type=int, default=3)
    pg.add_argument("--server-warmup", type=int, default=120)
    pg.add_argument("--server-retries", type=int, default=4)
    pg.add_argument("--port", type=int, default=5556)
    pg.add_argument("--max-episode-steps", type=int, default=720)
    pg.add_argument("--n-action-steps", type=int, default=16)
    pg.add_argument("--step-delay", type=float, default=0.02,
                    help="seconds to sleep after each sim step so motion is watchable")
    pg.add_argument("--camera", default="egoview",
                    help="view camera (default egoview=robot head/policy view; or robot0_behindhead, "
                         "robot0_agentview_center, robot0_frontview, ...; 'free' = orbit camera)")

    a = p.parse_args()
    return {
        "status": cmd_status, "list": cmd_list, "clip": cmd_clip,
        "download-ckpt": cmd_download_ckpt, "render": cmd_render, "eval": cmd_eval,
        "gui": cmd_gui,
    }[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main() or 0)
