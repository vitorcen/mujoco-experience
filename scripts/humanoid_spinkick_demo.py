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
    parser = argparse.ArgumentParser(description="Humanoid Spinkick MoCap Demo")
    parser.add_argument("--headless", action="store_true", help="无 GUI 运行（仅控制台输出）")
    parser.add_argument("--duration", type=float, default=10.0, help="Headless 运行时长（秒）")
    parser.add_argument("--record", type=str, help="录制视频文件（如 spinkick.mp4）")
    parser.add_argument("--fps", type=int, default=30, help="录制帧率")
    parser.add_argument("--width", type=int, default=1280, help="录制宽度")
    parser.add_argument("--height", type=int, default=720, help="录制高度")
    args = parser.parse_args()
    
    # 路径
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dm_dir = os.path.join(project_root, "dependencies", "DeepMimic_mujoco", "src", "mujoco")
    
    # 1. 准备 XML
    xml_path = os.path.join(dm_dir, "humanoid_deepmimic", "envs", "asset", "dp_env_v2.xml")
    
    if not os.path.exists(xml_path):
        print(f"Error: XML file not found at {xml_path}")
        return

    with open(xml_path, 'r') as f:
        xml_content = f.read()
        
    try:
        model = mujoco.MjModel.from_xml_string(xml_content)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. 加载 Motion (Spinkick)
    motion_path = os.path.join(dm_dir, "motions", "humanoid3d_spinkick.txt")
    if not os.path.exists(motion_path):
        print(f"Error: Motion file not found at {motion_path}")
        return

    player = MocapPlayer(motion_path, loop=True)
    print(f"Playing Spinkick motion...")
    
    # 3. Headless 模式
    if args.headless:
        print("Running in headless mode (no GUI)...")
        start_time = time.time()
        
        while time.time() - start_time < args.duration:
            elapsed = time.time() - start_time
            qpos = player.get_frame_qpos(elapsed, model)
            
            if qpos is not None and len(qpos) == len(data.qpos):
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
            
            # 每秒打印一次根位置
            if int(elapsed) != int(elapsed - model.opt.timestep):
                root_pos = data.qpos[:3]
                print(f"[t={elapsed:5.2f}s] root_xyz = {root_pos}")
            
            time.sleep(0.016)
        
        print("Headless simulation finished.")
        return
    
    # 4. 录制模式
    if args.record:
        print(f"Recording video to {args.record}...")
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        frames = []
        start_time = time.time()
        
        duration = player.dt * len(player.data_config) if not player.loop else 10.0
        
        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            qpos = player.get_frame_qpos(elapsed, model)
            
            if qpos is not None and len(qpos) == len(data.qpos):
                data.qpos[:] = qpos
                mujoco.mj_forward(model, data)
            
            renderer.update_scene(data)
            frame = renderer.render()
            frames.append(frame)
            
            time.sleep(1.0 / args.fps)
        
        imageio.mimsave(args.record, frames, fps=args.fps)
        print(f"✅ Video saved: {args.record}")
        return
    
    # 5. GUI 模式
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            start_time = time.time()
            
            while viewer.is_running():
                now = time.time()
                elapsed = now - start_time
                
                qpos = player.get_frame_qpos(elapsed, model)
                
                if qpos is not None and len(qpos) == len(data.qpos):
                    data.qpos[:] = qpos
                    mujoco.mj_forward(model, data)
                
                viewer.sync()
                time.sleep(0.016)
    except Exception as e:
        print(f"❌ Viewer 启动失败: {e}")
        print("\n解决方案:")
        print("  1. Headless 运行: python scripts/humanoid_spinkick_demo.py --headless")
        print("  2. 录制视频: MUJOCO_GL=egl python scripts/humanoid_spinkick_demo.py --record spinkick.mp4")
        print("  3. 检查系统 OpenGL 库: 参考 doc/mujoco_opengl_backend_setup.md")

if __name__ == "__main__":
    main()
