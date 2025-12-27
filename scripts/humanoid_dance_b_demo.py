import time
import os
import mujoco
import mujoco.viewer
import sys
import argparse
import imageio

# Add scripts directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mocap_utils import MocapPlayer

def main():
    parser = argparse.ArgumentParser(description="Humanoid Dance B MoCap Demo")
    parser.add_argument("--headless", action="store_true", help="无 GUI 运行")
    parser.add_argument("--duration", type=float, default=15.0, help="Headless 运行时长")
    parser.add_argument("--record", type=str, help="录制视频文件 (e.g. dance.mp4)")
    parser.add_argument("--fps", type=int, default=30, help="录制帧率")
    parser.add_argument("--width", type=int, default=1280, help="录制宽度")
    parser.add_argument("--height", type=int, default=720, help="录制高度")
    args = parser.parse_args()

    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dm_dir = os.path.join(project_root, "DeepMimic_mujoco", "src", "mujoco")
    xml_path = os.path.join(dm_dir, "humanoid_deepmimic", "envs", "asset", "dp_env_v2.xml")
    motion_path = os.path.join(dm_dir, "motions", "humanoid3d_dance_b.txt")

    if not os.path.exists(xml_path):
        print(f"Error: XML not found at {xml_path}")
        return
    if not os.path.exists(motion_path):
        print(f"Error: Motion not found at {motion_path}")
        return

    # Load Model
    with open(xml_path, 'r') as f:
        xml_content = f.read()
    try:
        model = mujoco.MjModel.from_xml_string(xml_content)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Load Motion
    player = MocapPlayer(motion_path, loop=True)
    print(f"Playing Dance B motion ({len(player.data_config)} frames)...")

    # Run Simulation
    start_time = time.time()
    
    # Headless
    if args.headless:
        print("Running in headless mode...")
        while time.time() - start_time < args.duration:
            elapsed = time.time() - start_time
            qpos = player.get_frame_qpos(elapsed, model)
            if qpos is not None:
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
            time.sleep(0.016)
        print("Done.")
        return

    # Recording
    if args.record:
        print(f"Recording to {args.record}...")
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        frames = []
        duration = player.dt * len(player.data_config)
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            qpos = player.get_frame_qpos(elapsed, model)
            if qpos is not None:
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
            renderer.update_scene(data)
            frames.append(renderer.render())
            time.sleep(1.0/args.fps)
        imageio.mimsave(args.record, frames, fps=args.fps)
        print("Saved.")
        return

    # GUI
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                elapsed = time.time() - start_time
                qpos = player.get_frame_qpos(elapsed, model)
                if qpos is not None:
                    data.qpos[:] = qpos
                    mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.016)
    except Exception as e:
        print(f"Viewer error: {e}")
        print("Try MUJOCO_GL=egl python ... --record output.mp4")

if __name__ == "__main__":
    main()
