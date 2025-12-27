import time
import os
import mujoco
import mujoco.viewer
import numpy as np
import re
import argparse
import imageio

# ===========================
# 配置区域
# ===========================
# 崎岖地形场景
SCENE_XML_PATH = "unitree_mujoco/unitree_robots/go2/scene_terrain.xml"

# 简易控制器参数
GAIT_FREQ = 2.5
STANDING_HEIGHT = 0.28

class SimpleQuadrupedController:
    """简单的原地踏步/行走控制器 (开环)"""
    def __init__(self):
        self.phase = 0.0
        # Go2 默认关节角 (站立)
        # FR, FL, RR, RL
        self.default_angle = np.array([0.0, 0.8, -1.6] * 4) 
        
    def update(self, vx, wz, dt):
        self.phase += dt * GAIT_FREQ * 2 * np.pi
        
        # 抬腿幅度 (崎岖地形需要抬高一点)
        amp_swing = 0.2 + 0.3 * abs(vx) # 基础抬腿 + 速度增益
        
        # 简单的相位机: 对角步态
        offsets = np.array([0, np.pi, np.pi, 0])
        
        targets = self.default_angle.copy()
        
        for i in range(4):
            idx = i * 3
            p = self.phase + offsets[i]
            
            # 髋部摆动 (前进/转向)
            targets[idx] += 0.0 # 简化，暂不处理 Hip Roll/Yaw 复杂耦合
            
            # 腿部屈伸 (Pitch)
            # sin > 0 时抬腿 (屈膝 + 屈髋)
            s = np.sin(p)
            if s > 0:
                # 摆动相：抬腿
                targets[idx+1] += s * amp_swing       # Thigh
                targets[idx+2] += s * amp_swing * 1.5 # Calf (多屈一点)
            else:
                # 支撑相：稍微用力蹬地 (伸展)
                targets[idx+1] += s * 0.05
                targets[idx+2] += s * 0.05
                
        return targets

def load_scene_with_vfs():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 路径定义
    go2_dir = os.path.join(project_root, "unitree_mujoco", "unitree_robots", "go2")
    scene_path = os.path.join(go2_dir, "scene_terrain.xml")
    
    # 1. 读取主 XML
    with open(scene_path, "r") as f:
        xml_content = f.read()
        
    # 2. 修复相对路径 (../xxx.png -> xxx.png)
    # 这样我们可以把文件扁平化放入 VFS 根目录 (或指定映射)
    #
    # 注意：go2.xml 里设置了 <compiler meshdir="assets" .../>，被 include 进主场景后，
    # MuJoCo 在解析某些资源（包括 hfield 的 file）时会尝试带上 "assets/" 前缀。
    # 运行日志正是 "Error opening file assets/height_field.png"。
    # 因此这里直接把 hfield 的 file 重写到 assets/ 下，配合 VFS 注入同名 key。
    xml_content = xml_content.replace("../height_field.png", "assets/height_field.png")
    xml_content = xml_content.replace("../unitree_hfield.png", "assets/unitree_hfield.png")
    
    # 3. 构建 VFS
    assets = {}
    
    # go2.xml (被 include)
    with open(os.path.join(go2_dir, "go2.xml"), "rb") as f:
        assets["go2.xml"] = f.read()
        
    # 地形贴图/高度图（Go2 目录下就有）
    # 注意：MuJoCo 的 Python VFS 会对文件名做归一化（容易把不同 key 视为同名文件），
    # 如果同时注册 "assets/height_field.png" 和 "height_field.png" 可能触发：
    #   Repeated file name in assets dict: height_field.png
    # 因此这里仅注册与 XML 重写后完全一致的 key（assets/...）。
    for fname in ["height_field.png", "unitree_hfield.png"]:
        p = os.path.join(go2_dir, fname)
        if os.path.exists(p):
            with open(p, "rb") as f:
                blob = f.read()
                assets[f"assets/{fname}"] = blob
                
    # Go2 自身的 mesh assets (在 go2/assets)
    # go2.xml 引用通常是 "assets/xxx.obj"
    go2_assets_dir = os.path.join(go2_dir, "assets")
    if os.path.exists(go2_assets_dir):
        for name in os.listdir(go2_assets_dir):
            p = os.path.join(go2_assets_dir, name)
            if os.path.isfile(p):
                with open(p, "rb") as f:
                    # key 保持 go2.xml 里引用的相对路径
                    assets[f"assets/{name}"] = f.read()
                    
    return mujoco.MjModel.from_xml_string(xml_content, assets=assets)

def main():
    parser = argparse.ArgumentParser(description="Unitree Go2 rough-terrain demo (unitree_mujoco scene).")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="不启动 GUI viewer，仅跑仿真与控制，并周期性打印 base 位姿（适用于无 GLX/远程环境）。",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=20.0,
        help="headless 模式下仿真时长（秒）。默认 20s。",
    )
    parser.add_argument(
        "--print-hz",
        type=float,
        default=2.0,
        help="headless 模式下打印频率（Hz）。默认 2Hz。",
    )
    parser.add_argument(
        "--record",
        type=str,
        default="",
        help="输出 mp4 路径（离屏渲染录制）。需要 MUJOCO_GL=egl（推荐）或 MUJOCO_GL=osmesa 可用。",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="录制视频帧率（Hz）。默认 30。")
    parser.add_argument("--width", type=int, default=640, help="录制视频宽度。默认 640。")
    parser.add_argument("--height", type=int, default=480, help="录制视频高度。默认 480。")
    args = parser.parse_args()

    print("Loading Go2 Terrain Scene...")
    try:
        model = load_scene_with_vfs()
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 初始化控制器
    controller = SimpleQuadrupedController()
    
    # 将 Go2 放到地形上方 (避免卡在盒子里)
    # 地形有些 box 在 z=0.5 左右
    # 找到 freejoint (通常 qpos 0-6)
    # 看看 scene_terrain.xml 里有没有 start pos，或者我们手动设
    if model.nq >= 7:
        # qpos[0:3] = pos
        data.qpos[0] = 0.0 # x
        data.qpos[1] = 0.0 # y
        data.qpos[2] = 0.6 # z (抬高一点)
    
    mujoco.mj_forward(model, data)

    print("Starting simulation... (Go2 walking on rough terrain)")
    
    def step_control_once():
        # 控制
        dt = model.opt.timestep

        # 简单的一直往前走（开环）
        vx = 0.5
        wz = 0.0

        targets = controller.update(vx, wz, dt)

        # 应用 PD (Go2 12 dof)
        # actuator 顺序通常是 FR(3), FL(3), RR(3), RL(3)
        kp = 40
        kd = 2

        # 注意：unitree_mujoco 的 go2.xml actuator 定义顺序可能不同
        # 通常是 FR_hip, FR_thigh, FR_calf, FL_...
        # 我们假设前 12 个 actuator 是腿部
        if len(data.ctrl) >= 12 and model.nq >= (7 + 12) and model.nv >= (6 + 12):
            for i in range(12):
                # qpos: freejoint(7) 后紧接 12 个 hinge
                q = data.qpos[7 + i]
                dq = data.qvel[6 + i]
                des = targets[i]
                tau = kp * (des - q) - kd * dq
                data.ctrl[i] = tau

        mujoco.mj_step(model, data)

    if args.headless:
        # headless：不依赖 glfw / GLX，适合远程/无桌面环境。
        t_end = data.time + float(args.duration)
        print_period = 1.0 / max(1e-6, float(args.print_hz))
        next_print = data.time

        # 可选：离屏渲染录制
        record_path = args.record.strip()
        renderer = None
        frames = []
        if record_path:
            # 重要：录制依赖离屏 OpenGL 上下文。推荐用 EGL：
            #   MUJOCO_GL=egl python scripts/go2_terrain_demo.py --headless --record out.mp4
            try:
                renderer = mujoco.Renderer(model, height=int(args.height), width=int(args.width))
                print(f"Recording enabled: {record_path} @ {args.fps:.1f}fps ({args.width}x{args.height})")
            except Exception as e:
                print(f"❌ 无法创建离屏 Renderer：{e}")
                print("你可能缺少可用的离屏 OpenGL 后端。推荐尝试：")
                print("  MUJOCO_GL=egl    python scripts/go2_terrain_demo.py --headless --record out.mp4")
                print("  MUJOCO_GL=osmesa python scripts/go2_terrain_demo.py --headless --record out.mp4")
                return

        fps = max(1e-6, float(args.fps))
        frame_period = 1.0 / fps
        next_frame = data.time

        while data.time < t_end:
            step_control_once()
            if data.time >= next_print:
                # freejoint: qpos[0:3] = base xyz
                base_xyz = data.qpos[0:3].copy() if model.nq >= 3 else np.zeros(3)
                print(f"[t={data.time:6.2f}s] base_xyz = {base_xyz}")
                next_print += print_period

            if renderer is not None and data.time >= next_frame:
                renderer.update_scene(data)
                pixels = renderer.render()
                frames.append(pixels)
                next_frame += frame_period

        if renderer is not None:
            try:
                # imageio 会根据扩展名选择 writer；mp4 需要 ffmpeg（通常 imageio 会自带/或走系统 ffmpeg）
                imageio.mimwrite(record_path, frames, fps=fps, quality=8)
                print(f"✅ Video written: {record_path} (frames={len(frames)})")
            except Exception as e:
                print(f"❌ 写视频失败：{e}")
                print("你可以先确保系统有 ffmpeg：sudo apt-get install -y ffmpeg")

        print("Headless simulation finished.")
        print("如果你需要可视化：请在有桌面 OpenGL/GLX 的环境运行，或用录屏/离屏渲染方案。")
        return

    # GUI 渲染（依赖 glfw + GLX/OpenGL）
    try:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_start = time.time()
                step_control_once()
                viewer.sync()

                # 保持实时
                time_until_next = model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)
    except Exception as e:
        print(f"Viewer 启动失败：{e}")
        print("你当前环境很可能缺少可用的 GLX/OpenGL（例如远程/容器/驱动不匹配）。")
        print("建议改用 headless 模式验证控制是否在跑：")
        print("  python scripts/go2_terrain_demo.py --headless --duration 20")
        print("如果你想强行尝试 EGL/OSMesa（取决于系统安装情况）：")
        print("  MUJOCO_GL=egl    python scripts/go2_terrain_demo.py --headless")
        print("  MUJOCO_GL=osmesa python scripts/go2_terrain_demo.py --headless")
        return

if __name__ == "__main__":
    main()
