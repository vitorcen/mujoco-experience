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

        # 趴下姿态 (用于起立初始化)
        # 参考官方 stand_go2.py: [0.04, 1.22, -2.44]
        self.prone_angle = np.array([0.0, 1.22, -2.44] * 4)

    def update(self, vx, wz, dt):
        self.phase += dt * GAIT_FREQ * 2 * np.pi

        # Lift amplitude — modest, terrain wants ~0.25 rad knee bend at peak.
        amp_lift = 0.25 + 0.20 * abs(vx)
        # Forward sweep amplitude on the hip-pitch (thigh) joint. Scaled by vx.
        stride = 0.25 * vx
        # Yaw command: differential stride between left/right rows.
        yaw_split = 0.15 * wz

        # Trot pairing: (FR, RL) swing while (FL, RR) stance.
        # Actuator order in go2.xml is FR, FL, RR, RL.
        offsets = np.array([0.0, np.pi, np.pi, 0.0])
        # Per-leg stride sign: front legs reach forward, rear legs push back.
        stride_sign = np.array([+1.0, +1.0, -1.0, -1.0])
        # Yaw sign: right side (FR, RR) vs left side (FL, RL).
        yaw_sign = np.array([+1.0, -1.0, +1.0, -1.0])

        targets = self.default_angle.copy()

        for i in range(4):
            idx = i * 3
            p = self.phase + offsets[i]
            s = np.sin(p)
            c = np.cos(p)

            # Hip pitch (thigh): cosine sweeps the foot fore/aft for propulsion;
            # during swing (s>0) also tuck the leg up by *decreasing* thigh angle
            # (Go2 convention: thigh ~0.8 standing, smaller = leg lifted).
            lift = max(s, 0.0) * amp_lift
            targets[idx + 1] -= c * (stride * stride_sign[i] + yaw_split * yaw_sign[i])
            targets[idx + 1] -= lift

            # Knee (calf): default ~-1.6 (bent). Bending more = more negative.
            # During swing we want extra knee flex; during stance keep it stiff.
            targets[idx + 2] -= lift * 1.5

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
    if model.nq >= 19:
        # 1. 设置基座位置 (x, y, z)
        data.qpos[0] = 0.0
        data.qpos[1] = 0.0
        data.qpos[2] = 0.35 # z (降低高度，准备从趴下开始起立)

        # 2. 设置关节初始角度为“趴下”姿态
        # 这样机器人会先趴在地上，然后慢慢站起来，这是最稳的初始化方式
        data.qpos[7:19] = controller.prone_angle

    mujoco.mj_forward(model, data)

    print("Starting simulation... (Go2 walking on rough terrain)")

    def step_control_once():
        # 控制
        dt = model.opt.timestep
        t = data.time

        # --- Soft Start (平滑起立) ---
        # 模仿官方 stand_go2.py 的逻辑：从趴下姿态插值过渡到站立姿态。
        # 给一段额外的纯站立稳态时间，再起步走。
        warmup_time = 2.0
        stand_hold = 1.0  # seconds of standing still before walking
        target_kp = 80.0

        if t < warmup_time:
            vx = 0.0
            wz = 0.0
            ratio = t / warmup_time
            # Smoothstep for gentler take-off.
            ratio = ratio * ratio * (3.0 - 2.0 * ratio)
            targets = (1 - ratio) * controller.prone_angle + ratio * controller.default_angle
            kp = target_kp * (0.4 + 0.6 * ratio)
        elif t < warmup_time + stand_hold:
            targets = controller.default_angle.copy()
            kp = target_kp
        else:
            vx = 0.4
            wz = 0.0
            targets = controller.update(vx, wz, dt)
            kp = target_kp

        kd = 4.0

        # 应用 PD (Go2 12 dof)
        # actuator 顺序通常是 FR(3), FL(3), RR(3), RL(3)

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
