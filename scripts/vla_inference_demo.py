import time
import mujoco
import mujoco.viewer
import numpy as np

# 如果想使用 Hugging Face 的模型，需要安装 transformers 和 PIL
# pip install transformers pillow torch
try:
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from PIL import Image
    HAS_VLA = True
except ImportError:
    HAS_VLA = False
    print("Warning: transformers/torch not installed. Running in mock mode.")

# 配置
MODEL_PATH = './mujoco_menagerie/unitree_h1/scene.xml'
# 使用一个小模型或者 Mock
VLA_MODEL_ID = "HuggingFaceTB/SmolVLA-171M" # 假设的模型ID，或者使用真实的 OpenVLA

class VLAController:
    def __init__(self, model_id, device='cpu'):
        self.mock_mode = not HAS_VLA
        if not self.mock_mode:
            print(f"Loading VLA model: {model_id}...")
            # 注意：真实的 SmolVLA 加载需要更多显存和特定代码，这里仅展示加载流程
            # self.processor = AutoProcessor.from_pretrained(model_id)
            # self.model = AutoModelForVision2Seq.from_pretrained(model_id).to(device)
            # 这里为了演示，我们依然使用 Mock 逻辑，但结构保持一致
            self.mock_mode = True 
        
    def predict_action(self, image, instruction):
        """
        模拟 VLA 推理：输入图像和指令，输出动作向量
        """
        # 1. 预处理图像
        # inputs = self.processor(text=instruction, images=image, return_tensors="pt")
        
        # 2. 模型推理
        # generated_ids = self.model.generate(**inputs, max_new_tokens=128)
        # generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # 3. 解析动作为控制信号 (Mock)
        # 这里简单地返回一个随时间变化的正弦波作为动作
        t = time.time()
        # H1 有 19 个自由度，返回 19 维向量
        action = np.ones(19) * np.sin(t * 2) * 20.0 
        return action

def main():
    # 1. 加载 MuJoCo 模型
    print(f"Loading MuJoCo model: {MODEL_PATH}")
    try:
        m = mujoco.MjModel.from_xml_path(MODEL_PATH)
        d = mujoco.MjData(m)
    except ValueError:
        print("Error: Model not found. Did you run 'git submodule update --init mujoco_menagerie'?")
        return

    # 2. 初始化 VLA 控制器
    vla = VLAController(VLA_MODEL_ID)
    instruction = "Walk forward"

    # 3. 初始化渲染器 (用于获取视觉输入)
    renderer = mujoco.Renderer(m, height=480, width=640)

    # 4. 启动 Viewer
    with mujoco.viewer.launch_passive(m, d) as viewer:
        start_time = time.time()
        
        while viewer.is_running():
            step_start = time.time()

            # --- VLA Loop (通常频率低于物理模拟) ---
            
            # A. 获取视觉观测 (Vision Observation)
            renderer.update_scene(d)
            rgb_image = renderer.render()
            # 如果需要显示第一人称视角，可以在 XML 里定义 camera 并在这里指定 camera_id
            
            # B. VLA 推理 (Inference)
            # 将 numpy array 转为 PIL Image (如果模型需要)
            pil_image = Image.fromarray(rgb_image) if HAS_VLA else None
            
            # 获取动作
            action = vla.predict_action(pil_image, instruction)
            
            # C. 执行动作 (Action Execution)
            # 注意：VLA 输出通常是归一化的或特定的空间（如末端执行器），需要重映射
            # 这里直接假设输出对应关节控制
            if len(action) == len(d.ctrl):
                d.ctrl[:] = action
            else:
                # 维度不匹配时，只控制部分或做广播 (Demo Only)
                min_len = min(len(action), len(d.ctrl))
                d.ctrl[:min_len] = action[:min_len]

            # --- Physics Loop ---
            mujoco.mj_step(m, d)

            # 同步 Viewer
            viewer.sync()

            # 保持实时
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
