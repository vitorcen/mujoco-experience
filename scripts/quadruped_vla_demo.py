import time
import os
import mujoco
import mujoco.viewer
import numpy as np
import re
from PIL import Image
import argparse
import importlib.metadata

# ===========================
# 配置区域
# ===========================

# 使用 OpenVLA-7B
# 显存经验值（粗略，和实现/缓存/激活有关）：
# - FP16 全精度推理：显存占用通常约等于“模型权重大小”的 ~2x
# - 4-bit 量化推理：显存占用通常约等于“模型权重大小”的 ~0.5x
MODEL_ID = "openvla/openvla-7b"
LOAD_IN_4BIT = False  # 默认不启用量化（按需用 --load-4bit 开启）

# 场景：Unitree Go2 + 红色球
SCENE_XML_PATH = "scripts/go2_scene.xml"
INSTRUCTION = "Walk to the red ball"

# 步态参数
GAIT_FREQ = 2.5  # Hz
STEP_HEIGHT = 0.08
STANDING_HEIGHT = 0.28

# VLA -> 速度映射参数（OpenVLA 的 action 往往很小，需要放大/限幅/死区）
VLA_INFER_HZ = 2.0          # 推理频率（Hz），越高响应越快但更耗算力
VLA_VX_GAIN = 8.0           # 前进速度增益
VLA_WZ_GAIN = 6.0           # 转向速度增益
VX_MAX = 1.2                # m/s（限幅）
WZ_MAX = 2.0                # rad/s（限幅）
VX_MIN_ABS = 0.12           # m/s（非零时的最小绝对速度，避免“挪不动”）
CMD_SMOOTH_ALPHA = 0.35     # 速度指令 EMA 平滑系数（0~1，越大越跟随，越小越平滑）

# ===========================
# VLA 加载器 (复用之前逻辑)
# ===========================
class OpenVLAController:
    def __init__(self, model_id, load_in_4bit=True):
        self.model = None
        self.processor = None
        # 简单检查 nvidia-smi
        has_gpu = os.system("nvidia-smi > /dev/null 2>&1") == 0
        self.device = "cuda" if has_gpu else "cpu"
        
        print(f"Initializing VLA on {self.device}...")
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
            from transformers import __version__ as transformers_version

            def _from_pretrained_eager(**kwargs):
                # OpenVLA 的 remote code 在某些 transformers 新版本下会因为 SDPA/FlashAttention 自动选择而报
                # "'OpenVLAForActionPrediction' object has no attribute '_supports_sdpa'"
                # 强制使用 eager attention 可以绕开这条路径。
                try:
                    return AutoModelForVision2Seq.from_pretrained(
                        **kwargs,
                        attn_implementation="eager",
                    )
                except TypeError:
                    # 老版本 transformers 不支持 attn_implementation 参数
                    return AutoModelForVision2Seq.from_pretrained(**kwargs)
            
            if self.device == "cuda" and load_in_4bit:
                # bitsandbytes 4bit 量化依赖：缺失/安装损坏时常见报错
                # "No package metadata was found for bitsandbytes"
                try:
                    _ = importlib.metadata.version("bitsandbytes")
                    import bitsandbytes  # noqa: F401
                except Exception as e:
                    raise RuntimeError(
                        "bitsandbytes 不可用，无法启用 4-bit 量化。"
                        "请在正确的环境中执行：pip install -U --force-reinstall bitsandbytes\n"
                        f"原始错误: {e}"
                    )
                from transformers import BitsAndBytesConfig
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                print(f"Loading {model_id} with 4-bit quantization...")
                self.model = _from_pretrained_eager(
                    pretrained_model_name_or_path=model_id,
                    quantization_config=quantization_config,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
            else:
                self.model = _from_pretrained_eager(
                    pretrained_model_name_or_path=model_id,
                    torch_dtype=torch.float16 if self.device=="cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                ).to(self.device)

            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            print("✅ VLA Model loaded!")
        except Exception as e:
            msg = str(e)
            if "_supports_sdpa" in msg:
                print(
                    "❌ VLA Load Failed: OpenVLA remote-code 与当前 transformers 版本不兼容（SDPA 属性缺失）。\n"
                    f"当前 transformers={transformers_version}\n"
                    "建议在 mujoco 环境中降级到较兼容版本，例如：\n"
                    "  pip install -U \"transformers==4.44.2\" \"tokenizers==0.19.1\"\n"
                    "然后重新运行。\n"
                    "（我已在脚本中尝试强制 eager attention，但若仍失败就需要降级。）\n"
                    "Running in MOCK mode."
                )
            else:
                print(f"❌ VLA Load Failed: {e}\nRunning in MOCK mode.")

    def predict(self, image_pil, instruction):
        if self.model is None:
            # Mock: 总是向前走
            return np.array([0.5, 0.0, 0, 0, 0, 0, 0])

        import torch
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(prompt, image_pil, return_tensors="pt")
        # input_ids 等保持 int64；pixel_values 用模型自身 dtype，避免 bf16/half 混用导致 conv2d 报错
        for k, v in list(inputs.items()):
            if not hasattr(v, "to"):
                continue
            inputs[k] = v.to(self.model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=getattr(self.model, "dtype", torch.float16))
        # OpenVLA remote-code 的 predict_action 可能会在 input_ids 末尾补一个 token，
        # 但不会同步更新 attention_mask，导致长度差 1 的报错。
        # 最稳妥的做法：不传 attention_mask（让模型自行处理）。
        if "attention_mask" in inputs:
            del inputs["attention_mask"]
        with torch.inference_mode():
            action = self.model.predict_action(**inputs, unnorm_key="bridge_orig", use_cache=False)
        return action

# ===========================
# 简易四足步态控制器
# ===========================
class SimpleQuadrupedController:
    def __init__(self):
        self.phase = 0.0
        # Go2 关节顺序: FR_hip, FR_thigh, FR_calf, FL_..., RR_..., RL_... (需根据XML确认)
        # 通常是 FR, FL, RR, RL
        self.default_angle = np.array([0.0, 0.8, -1.6] * 4) # 站立姿态
        
    def update(self, vx, wz, dt):
        """
        vx: 前进速度 (-1 ~ 1)
        wz: 转向速度 (-1 ~ 1)
        """
        self.phase += dt * GAIT_FREQ * 2 * np.pi
        
        # 简化的逆运动学模拟 (直接生成关节角度正弦波)
        # 对角腿相位差 pi
        offsets = np.array([0, np.pi, np.pi, 0]) 
        
        # 髋关节 (转向)
        hip_target = np.array([-wz, wz, -wz, wz]) * 0.3
        
        # 腿部摆动 (前进)
        # 用 vx 调制摆动幅度，但为了避免 vx 很小导致“走不动”，引入最小摆动幅度
        vx_clip = float(np.clip(vx, -1.0, 1.0))
        vx_eff = vx_clip
        if abs(vx_eff) > 1e-6 and abs(vx_eff) < 0.18:
            vx_eff = np.sign(vx_eff) * 0.18
        amp = vx_eff * 0.55
        
        targets = self.default_angle.copy()
        
        for i in range(4):
            idx = i * 3
            # Hip (Yaw/Roll depending on config)
            targets[idx] += hip_target[i]
            
            # Thigh & Calf (Pitch) - 简单的正弦摆动模拟迈步
            p = self.phase + offsets[i]
            targets[idx+1] += np.sin(p) * amp
            targets[idx+2] += np.sin(p + 1.0) * amp # 相位差产生抬腿
            
        return targets

# ===========================
# 主程序
# ===========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="强制使用 MOCK，不加载 VLA")
    parser.add_argument("--load-4bit", action="store_true", help="启用 bitsandbytes 4-bit 量化（节省显存）")
    # 向后兼容：旧参数 --no-4bit（现在默认就是 no-4bit）
    parser.add_argument("--no-4bit", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--smoke", action="store_true", help="只跑一次 VLA 推理做自检，然后退出（不启动 viewer）")
    parser.add_argument("--infer-hz", type=float, default=VLA_INFER_HZ, help="VLA 推理频率 (Hz)")
    parser.add_argument("--vx-gain", type=float, default=VLA_VX_GAIN, help="VLA->vx 增益")
    parser.add_argument("--wz-gain", type=float, default=VLA_WZ_GAIN, help="VLA->wz 增益")
    parser.add_argument("--vx-max", type=float, default=VX_MAX, help="vx 限幅 (m/s)")
    parser.add_argument("--wz-max", type=float, default=WZ_MAX, help="wz 限幅 (rad/s)")
    parser.add_argument("--vx-min-abs", type=float, default=VX_MIN_ABS, help="非零时 vx 最小绝对值 (m/s)")
    parser.add_argument("--smooth-alpha", type=float, default=CMD_SMOOTH_ALPHA, help="速度指令 EMA 平滑系数 (0~1)")
    args = parser.parse_args()

    # 运行环境提示（你刚才是在 base 环境跑的）
    conda_env = os.environ.get("CONDA_DEFAULT_ENV", "")
    if conda_env and conda_env != "mujoco":
        print(f"⚠️  当前 conda 环境是 '{conda_env}'，建议使用 'mujoco'：conda activate mujoco")
    if not conda_env:
        print("⚠️  未检测到 conda 环境变量，建议使用项目的 mujoco 环境运行。")

    # 1. 动态生成 Go2 场景 XML
    # 我们需要在前面放一个红球
    go2_xml_path = "mujoco_menagerie/unitree_go2/scene.xml"
    
    # 使用 VFS 加载
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # 读取原始 scene.xml 并插入红球
    # 注意：这里为了简单，我们直接用 vfs 挂载原始文件，并通过字符串修改添加红球
    # 但由于 scene.xml include 了 go2.xml，我们需要把它们都放进 VFS
    
    # 简化：我们直接加载 go2/scene.xml，然后通过 MJCF 运行时添加 geom (不支持)
    # 或者我们生成一个新的 xml 字符串
    
    menagerie_dir = os.path.join(project_root, "mujoco_menagerie")
    go2_dir = os.path.join(menagerie_dir, "unitree_go2")
    
    with open(os.path.join(go2_dir, "scene.xml"), "r") as f:
        scene_content = f.read()
    
    # 插入红球
    red_ball = '<body name="target" pos="2 0 0.5"><geom type="sphere" size="0.2" rgba="1 0 0 1"/></body>'
    scene_content = scene_content.replace('</worldbody>', f'{red_ball}</worldbody>')
    
    # 修改 include 路径为文件名 (配合 VFS)
    scene_content = re.sub(r'<include\s+file="[^"]+"\s*/>', '<include file="go2.xml"/>', scene_content)
    
    # 构建 VFS
    assets = {}
    # go2.xml
    with open(os.path.join(go2_dir, "go2.xml"), "rb") as f:
        assets["go2.xml"] = f.read()
    
    # Assets
    assets_dir = os.path.join(go2_dir, "assets")
    for name in os.listdir(assets_dir):
        fp = os.path.join(assets_dir, name)
        if os.path.isfile(fp):
            with open(fp, "rb") as f:
                blob = f.read()
            assets[f"assets/{name}"] = blob
            # assets[name] = blob # fallback
            
    # 加载
    print("Loading Go2 Model...")
    try:
        model = mujoco.MjModel.from_xml_string(scene_content, assets=assets)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 初始化 VLA
    if args.mock:
        vla = OpenVLAController(MODEL_ID, load_in_4bit=False)
        vla.model = None
        vla.processor = None
        print("⚠️  --mock 已开启：将使用固定前进指令，不会占用大显存。")
    else:
        # args.no_4bit 仅用于兼容旧命令，默认本来就是 False
        load_4bit = bool(args.load_4bit) and (not bool(args.no_4bit))
        vla = OpenVLAController(MODEL_ID, load_in_4bit=load_4bit)
    
    # 初始化步态控制器
    controller = SimpleQuadrupedController()
    
    # 渲染器
    renderer = mujoco.Renderer(model, height=224, width=224)
    # 需要找到相机。Go2 XML 里通常定义了 "front_camera" 或类似
    # 如果没有，就用默认视角
    camera_id = -1
    
    # Smoke test：跑一次推理就退出，避免 interactive viewer 卡住
    if args.smoke:
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera_id)
        rgb = renderer.render()
        action = vla.predict(Image.fromarray(rgb), INSTRUCTION)
        print(f"SMOKE OK. action[0:2]={action[:2]}")
        return

    # 仿真循环
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        last_infer = 0
        
        # 初始动作
        cmd_vx = 0.0
        cmd_wz = 0.0
        
        while viewer.is_running():
            now = time.time()
            dt = model.opt.timestep
            
            # --- VLA 推理 ---
            infer_period = 1.0 / max(0.1, float(args.infer_hz))
            if now - last_infer > infer_period:
                last_infer = now
                renderer.update_scene(data, camera=camera_id)
                rgb = renderer.render()
                
                # 预测
                action = vla.predict(Image.fromarray(rgb), INSTRUCTION)
                
                # 映射: VLA Action (Arm) -> Quadruped Velocity
                # action[0] (x) -> Forward velocity
                # action[1] (y) -> Turn velocity
                # OpenVLA 输出通常在 [-1, 1] 之间 (如果是 delta)
                
                raw_vx = float(action[0]) * float(args.vx_gain)
                raw_wz = float(action[1]) * float(args.wz_gain) * -1.0

                # 限幅
                raw_vx = float(np.clip(raw_vx, -float(args.vx_max), float(args.vx_max)))
                raw_wz = float(np.clip(raw_wz, -float(args.wz_max), float(args.wz_max)))

                # Deadzone / Min velocity to ensure motion when VLA output is weak
                # We also need a lower threshold to avoid noise-induced tiny movements
                if abs(raw_vx) < 0.02: # 降低死区下限，允许微小信号被放大
                     raw_vx = 0.0
                elif abs(raw_vx) < float(args.vx_min_abs):
                    raw_vx = float(np.sign(raw_vx) * float(args.vx_min_abs))

                # EMA 平滑，避免抖动
                alpha = float(np.clip(args.smooth_alpha, 0.0, 1.0))
                cmd_vx = (1 - alpha) * cmd_vx + alpha * raw_vx
                cmd_wz = (1 - alpha) * cmd_wz + alpha * raw_wz
                
                print(f"VLA Command: vx={cmd_vx:.2f}, wz={cmd_wz:.2f}")
            
            # --- 底层控制 (高频) ---
            targets = controller.update(cmd_vx, cmd_wz, dt)
            
            # 应用控制 (Go2 有 12 个 motor)
            # data.ctrl 的顺序需匹配 XML actuator 定义
            # 假设顺序匹配 (FR, FL, RR, RL)
            if len(data.ctrl) >= 12:
                # 简单的 P 控制模拟
                kp = 60
                kd = 3
                for i in range(12):
                    q = data.qpos[7 + i] # qpos 前7个是 freejoint
                    dq = data.qvel[6 + i]
                    des = targets[i]
                    tau = kp * (des - q) - kd * dq
                    data.ctrl[i] = tau
            
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(dt)

if __name__ == "__main__":
    main()
