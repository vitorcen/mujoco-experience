import time
import os
import mujoco
import mujoco.viewer
import numpy as np
import re

# ===========================
# 配置区域
# ===========================

# 使用 OpenVLA-7B (4-bit 量化版以节省显存)
# 注意: 首次运行会自动下载模型 (需 ~5GB 流量)
MODEL_ID = "openvla/openvla-7b" 
LOAD_IN_4BIT = True

# 场景配置 (复用 Panda 的场景)
SCENE_XML_PATH = "scripts/panda_manipulation.xml"
SITE_NAME = "attachment_site"
INSTRUCTION = "Pick up the red cube"

# 控制参数
CONTROL_DT = 0.1  # VLA 推理间隔 (秒)
IK_DT = 0.002     # IK 积分步长 (秒)

# ===========================
# VLA 模型加载器
# ===========================

class OpenVLAController:
    def __init__(self, model_id, load_in_4bit=True):
        self.model = None
        self.processor = None
        self.device = "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"
        
        print(f"Initializing VLA on {self.device}...")
        
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor
            from transformers import __version__ as transformers_version

            def _from_pretrained_eager(**kwargs):
                # 兼容 OpenVLA remote code 在新 transformers 下的 SDPA 检测问题
                try:
                    return AutoModelForVision2Seq.from_pretrained(
                        **kwargs,
                        attn_implementation="eager",
                    )
                except TypeError:
                    return AutoModelForVision2Seq.from_pretrained(**kwargs)
            
            # 只有在有 GPU 时才开启 4-bit 加载
            if self.device == "cuda" and load_in_4bit:
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
                print(f"Loading {model_id} (Full Precision)... Warning: Requires high RAM!")
                self.model = _from_pretrained_eager(
                    pretrained_model_name_or_path=model_id,
                    torch_dtype=torch.float16 if self.device=="cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                ).to(self.device)

            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            print("✅ VLA Model loaded successfully!")
            
        except ImportError as e:
            print(f"❌ Dependencies missing: {e}")
            print("Running in MOCK mode (random actions).")
        except Exception as e:
            msg = str(e)
            if "_supports_sdpa" in msg:
                print(
                    "❌ Model loading failed: OpenVLA remote-code 与当前 transformers 版本不兼容（SDPA 属性缺失）。\n"
                    f"当前 transformers={transformers_version}\n"
                    "建议在 mujoco 环境中降级到较兼容版本，例如：\n"
                    "  pip install -U \"transformers==4.44.2\" \"tokenizers==0.19.1\"\n"
                    "（脚本已尝试强制 eager attention，若仍失败则需要降级。）\n"
                    "Running in MOCK mode."
                )
            else:
                print(f"❌ Model loading failed: {e}")
                print("Running in MOCK mode.")

    def predict(self, image_pil, instruction):
        """
        输入: PIL Image, 文本指令
        输出: 7维动作向量 [dx, dy, dz, dr, dp, dy, gripper]
        """
        if self.model is None:
            # Mock 模式: 缓慢向下移动
            return np.array([0.0, 0.0, -0.01, 0, 0, 0, 1.0])

        import torch
        
        # OpenVLA 的 Prompt 格式
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        
        inputs = self.processor(prompt, image_pil, return_tensors="pt")
        for k, v in list(inputs.items()):
            if not hasattr(v, "to"):
                continue
            inputs[k] = v.to(self.model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype=getattr(self.model, "dtype", torch.float16))
        if "attention_mask" in inputs:
            del inputs["attention_mask"]
        
        with torch.inference_mode():
            # 预测动作
            action = self.model.predict_action(**inputs, unnorm_key="bridge_orig", use_cache=False)
            
        # action 是 numpy array
        return action

# ===========================
# 主程序
# ===========================

def load_panda_scene_with_vfs(xml_path):
    """使用 VFS 加载 Panda 场景，解决路径依赖问题"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    menagerie_dir = os.path.join(project_root, "mujoco_menagerie")
    panda_dir = os.path.join(menagerie_dir, "franka_emika_panda")
    
    # 1. 读取主 XML
    full_xml_path = os.path.join(project_root, xml_path)
    with open(full_xml_path, "r") as f:
        xml_content = f.read()
    
    # 2. 替换 include
    xml_content = re.sub(r'<include\s+file="[^"]+"\s*/>', '<include file="panda.xml"/>', xml_content)
    
    # 3. 准备 VFS
    assets = {}
    with open(os.path.join(panda_dir, "panda.xml"), "rb") as f:
        assets["panda.xml"] = f.read()
        
    assets_dir = os.path.join(panda_dir, "assets")
    for name in os.listdir(assets_dir):
        file_path = os.path.join(assets_dir, name)
        if os.path.isfile(file_path):
            with open(file_path, "rb") as f:
                blob = f.read()
            assets[f"assets/{name}"] = blob
            
    return mujoco.MjModel.from_xml_string(xml_content, assets=assets)

def main():
    # 1. 准备环境
    print("--- Real VLA Inference Demo ---")
    try:
        model = load_panda_scene_with_vfs(SCENE_XML_PATH)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading MuJoCo model: {e}")
        return

    # 获取末端执行器：优先用 site，否则回退到 hand body
    ee_kind = "site"
    ee_id = None
    try:
        ee_id = model.site(SITE_NAME).id
    except Exception:
        ee_kind = "body"
        try:
            ee_id = model.body("hand").id
        except Exception:
            print("Error: Could not find end-effector site or 'hand' body.")
            return

    # 初始化渲染器 (224x224 是 OpenVLA 的标准输入分辨率)
    renderer = mujoco.Renderer(model, height=224, width=224)
    
    # 2. 加载 VLA 模型
    vla = OpenVLAController(MODEL_ID, load_in_4bit=LOAD_IN_4BIT)
    
    # 3. 初始化控制状态
    from PIL import Image
    
    # 目标位姿 (初始设为当前末端位置)
    if ee_kind == "site":
        current_pos = data.site_xpos[ee_id].copy()
    else:
        current_pos = data.xpos[ee_id].copy()
    target_pos = current_pos.copy()
    # 稍微抬高一点作为起始目标
    target_pos[2] += 0.1 
    
    jac = np.zeros((6, model.nv))
    error = np.zeros(6)
    
    # 复位
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        if ee_kind == "site":
            target_pos = data.site_xpos[ee_id].copy()
        else:
            target_pos = data.xpos[ee_id].copy()

    print(f"\nInstruction: '{INSTRUCTION}'")
    print("Starting simulation...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_vla_time = 0
        start_time = time.time()
        
        while viewer.is_running():
            now = time.time()
            sim_t = data.time
            
            # === VLA 推理 (低频) ===
            if now - last_vla_time > CONTROL_DT:
                last_vla_time = now
                
                # 1. 渲染图像
                renderer.update_scene(data)
                rgb = renderer.render()
                image_pil = Image.fromarray(rgb)
                
                # 2. 模型预测
                # action: [dx, dy, dz, dr, dp, dy, gripper] (通常是相对末端的 delta)
                action = vla.predict(image_pil, INSTRUCTION)
                
                # 3. 更新目标位置 (简单的积分)
                # OpenVLA 输出通常比较小，根据需要缩放
                # 这里假设前3维是位置 delta
                scale_pos = 0.05
                target_pos += action[:3] * scale_pos
                
                # 夹爪控制 (action[6] > 0.5 usually means close or open depending on dataset)
                # OpenVLA bridge: 0=close, 1=open? 需要查阅具体 checkpoint 文档
                # 这里假设 >0.5 是闭合
                gripper_target = 0 if action[6] > 0.5 else 255
            
            # === IK 控制 (高频) ===
            
            # 1. 计算误差
            if ee_kind == "site":
                current_pos = data.site_xpos[ee_id]
            else:
                current_pos = data.xpos[ee_id]
            error[:3] = target_pos - current_pos
            # error[3:] = ... (旋转误差忽略，保持朝下)
            
            # 2. 计算雅可比
            if ee_kind == "site":
                mujoco.mj_jacSite(model, data, jac[:3], jac[3:], ee_id)
            else:
                mujoco.mj_jacBody(model, data, jac[:3], jac[3:], ee_id)
            
            # 3. 求解关节速度
            # J * dq = ve -> dq = J_pinv * ve
            # ve = K * error
            dq = np.linalg.lstsq(jac[:3], error[:3] * 5.0, rcond=0.01)[0]
            
            # 4. 积分得到关节位置
            q_current = data.qpos[:7]
            q_target = q_current + dq[:7] * IK_DT
            
            # 5. 设置控制信号
            data.ctrl[:7] = q_target
            # 夹爪：Panda menagerie 通常是最后 1 个 actuator（tendon split），ctrlrange 0..255
            if len(data.ctrl) >= 8:
                try:
                    data.ctrl[7] = gripper_target
                except Exception:
                    pass

            # === 物理步进 ===
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # 保持实时
            time.sleep(model.opt.timestep)

if __name__ == "__main__":
    main()
