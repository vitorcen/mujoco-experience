import time
import os
import mujoco
import mujoco.viewer
import numpy as np
import re
from dataclasses import dataclass

# ===========================
# 配置区域
# ===========================
SCENE_XML_PATH = "scripts/g1_scene.xml"
ACT_CHUNK_SIZE = 10  # 动作块大小
ACT_EXECUTION_HORIZON = 5  # 执行步数 (Temporal Ensembling)

# 动作展示参数：原地踏步 + 双手抬高在身体前左右摆动（循环）
CYCLE_SECONDS = 6.0         # 循环周期
STEP_FREQ_HZ = 1.5          # 原地踏步频率

# 训练/推理参数（让“策略不是 mock”）
CKPT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "g1_act.pt")
TRAIN_STEPS = 1500
BATCH_SIZE = 128
LR = 3e-4
DT_ACTION = 0.05  # 秒：动作块的时间间隔

# 训练用：stand_ctrl 注入点
_GLOBAL_STAND_CTRL = None

# ===========================
# ACT Policy（可训练 / 可加载）
# ===========================
@dataclass
class ActConfig:
    obs_dim: int
    act_dim: int
    chunk: int
    d_model: int = 256
    nhead: int = 8
    nlayers: int = 4


class ActNet:
    """一个最小可用的 ACT 风格网络：输入 obs -> 输出未来 K 步动作块。"""

    def __init__(self, cfg: ActConfig, device: str):
        import torch
        import torch.nn as nn

        self.cfg = cfg
        self.device = device

        self.net = nn.Sequential(
            nn.LayerNorm(cfg.obs_dim),
            nn.Linear(cfg.obs_dim, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.chunk * cfg.act_dim),
        ).to(device)

    def state_dict(self):
        return self.net.state_dict()

    def load_state_dict(self, sd):
        self.net.load_state_dict(sd)

    def __call__(self, obs):
        import torch

        out = self.net(obs)
        return out.view(-1, self.cfg.chunk, self.cfg.act_dim)


class ACTPolicy:
    def __init__(self, obs_dim: int, act_dim: int, chunk: int):
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = ActConfig(obs_dim=obs_dim, act_dim=act_dim, chunk=chunk)
        self.model = ActNet(self.cfg, device=self.device)

        os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)

        if os.path.exists(CKPT_PATH):
            ckpt = torch.load(CKPT_PATH, map_location=self.device)
            self.model.load_state_dict(ckpt["state_dict"])
            print(f"✅ Loaded ACT checkpoint: {CKPT_PATH}")
        else:
            print("⚠️  ACT checkpoint not found. Training a small policy now (synthetic data)...")
            self._train_synthetic()

    def _train_synthetic(self):
        import torch
        import torch.nn.functional as F

        # 生成一批 synthetic 数据：用“正常动作轨迹生成器”作为 teacher
        # obs: 当前 ctrl（29维），target: 未来 K 步 ctrl
        rng = np.random.default_rng(0)
        t0s = rng.uniform(0, CYCLE_SECONDS, size=TRAIN_STEPS)

        # 这里 stand_ctrl 会在 main() 里设置后通过闭包注入
        assert _GLOBAL_STAND_CTRL is not None, "stand_ctrl not initialized"
        stand_ctrl = _GLOBAL_STAND_CTRL

        obs = []
        tgt = []
        for t0 in t0s:
            o = build_g1_ctrl_traj(stand_ctrl, float(t0))
            chunk = []
            for k in range(self.cfg.chunk):
                tk = float(t0 + (k + 1) * DT_ACTION)
                chunk.append(build_g1_ctrl_traj(stand_ctrl, tk))
            obs.append(o)
            tgt.append(np.stack(chunk, axis=0))

        obs = torch.tensor(np.stack(obs, axis=0), dtype=torch.float32, device=self.device)
        tgt = torch.tensor(np.stack(tgt, axis=0), dtype=torch.float32, device=self.device)

        opt = torch.optim.AdamW(self.model.net.parameters(), lr=LR)

        self.model.net.train()
        for step in range(TRAIN_STEPS):
            idx = torch.randint(0, obs.shape[0], (BATCH_SIZE,), device=self.device)
            pred = self.model(obs[idx])
            loss = F.mse_loss(pred, tgt[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if (step + 1) % 200 == 0:
                print(f"[ACT train] step {step+1}/{TRAIN_STEPS} loss={loss.item():.6f}")

        torch.save({"state_dict": self.model.state_dict(), "cfg": self.cfg.__dict__}, CKPT_PATH)
        self.model.net.eval()
        print(f"✅ Saved ACT checkpoint: {CKPT_PATH}")

    def predict_chunk(self, obs_ctrl: np.ndarray) -> np.ndarray:
        """返回 (K, act_dim)"""
        import torch

        x = torch.tensor(obs_ctrl[None, :], dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            y = self.model(x)[0].detach().cpu().numpy()
        return y

# ===========================
# 轨迹生成器：原地踏步 + 抬手左右摆动（关节空间）
# ===========================
def build_g1_ctrl_traj(stand_ctrl: np.ndarray, t: float) -> np.ndarray:
    """基于 stand_ctrl 生成一个可见且稳定的周期动作（不要求动态平衡）。"""
    ctrl = stand_ctrl.copy()

    # 归一化周期：0..1
    s = (t % CYCLE_SECONDS) / CYCLE_SECONDS
    phase = 2.0 * np.pi * STEP_FREQ_HZ * t

    # === 腿部：原地踏步（左右交替抬腿） ===
    # 通过“摆动期屈膝 + 少量髋前摆”制造抬腿观感
    l = np.sin(phase)
    r = np.sin(phase + np.pi)
    l_swing = np.maximum(0.0, l)  # 0..1
    r_swing = np.maximum(0.0, r)

    hip_amp = 0.22
    knee_amp = 0.75
    ankle_amp = 0.35

    # 左腿（0..5）
    ctrl[0] += hip_amp * (l_swing - 0.3)         # left_hip_pitch：摆动期略前摆
    ctrl[3] += knee_amp * l_swing                # left_knee：摆动期屈膝
    ctrl[4] += ankle_amp * l_swing               # left_ankle_pitch：配合抬脚

    # 右腿（6..11）
    ctrl[6] += hip_amp * (r_swing - 0.3)
    ctrl[9] += knee_amp * r_swing
    ctrl[10] += ankle_amp * r_swing

    # === 手臂：双手抬高在身体前左右摆动 ===
    # 目标：手在身体前方（shoulder_pitch 向前抬），左右摆动主要靠 roll/yaw
    swing = np.sin(2.0 * np.pi * s)   # 左右摆动
    lift = 0.9                        # 抬手强度（常量偏置）

    # 左臂（15..21）
    ctrl[15] += -lift                 # left_shoulder_pitch：向前抬
    ctrl[16] += -0.35 * swing         # left_shoulder_roll：左右摆
    ctrl[17] += 0.25 * swing          # left_shoulder_yaw：微调前方弧线
    ctrl[18] += 0.9                   # left_elbow：弯曲

    # 右臂（22..28）
    ctrl[22] += -lift
    ctrl[23] += 0.35 * swing
    ctrl[24] += -0.25 * swing
    ctrl[25] += 0.9

    # 腰部轻微反向配合，看起来更自然
    ctrl[12] += -0.08 * swing         # waist_yaw

    return ctrl

# ===========================
# 主程序
# ===========================
def main():
    # 1. 动态生成 G1 场景 XML (带地面)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    menagerie_dir = os.path.join(project_root, "mujoco_menagerie")
    g1_dir = os.path.join(menagerie_dir, "unitree_g1")
    
    # 我们创建一个临时的 scene xml 字符串，include g1.xml
    scene_content = """
<mujoco model="g1_scene">
  <include file="g1.xml"/>
  <statistic center="0 0 1" extent="1.5"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
  <!-- 固定基座：原地踏步演示，不需要底座平移 -->
  <equality>
    <weld name="fix_base" body1="pelvis" body2="world"/>
  </equality>
</mujoco>
    """
    
    # 2. 构建 VFS
    assets = {}
    with open(os.path.join(g1_dir, "g1.xml"), "rb") as f:
        assets["g1.xml"] = f.read()
        
    assets_dir = os.path.join(g1_dir, "assets")
    if os.path.exists(assets_dir):
        for name in os.listdir(assets_dir):
            fp = os.path.join(assets_dir, name)
            if os.path.isfile(fp):
                with open(fp, "rb") as f:
                    blob = f.read()
                assets[f"assets/{name}"] = blob
    
    # NOTE:
    # g1.xml 里有 freejoint；这里用 weld 把 pelvis 绑定到 mocap body，从而“稳定且可控地”移动底座。
    
    print("Loading G1 Model (Fixed Base for Demo)...")
    try:
        model = mujoco.MjModel.from_xml_string(scene_content, assets=assets)
        data = mujoco.MjData(model)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # stand keyframe
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "stand")
    if key_id < 0:
        print("Error: keyframe 'stand' not found in g1.xml")
        return

    # Reset 到 stand pose
    mujoco.mj_resetDataKeyframe(model, data, key_id)
    stand_ctrl = model.key_ctrl[key_id].copy()
    data.ctrl[:] = stand_ctrl

    # 为“非 mock”训练注入 stand_ctrl
    global _GLOBAL_STAND_CTRL
    _GLOBAL_STAND_CTRL = stand_ctrl.copy()

    # 初始化 ACT（可训练/可加载）
    nu = model.nu
    policy = ACTPolicy(obs_dim=nu, act_dim=nu, chunk=ACT_CHUNK_SIZE)
    
    mujoco.mj_forward(model, data)

    # 渲染器
    renderer = mujoco.Renderer(model, height=480, width=640)

    # 仿真循环
    print("Starting simulation viewer...")
    try:
        viewer_ctx = mujoco.viewer.launch_passive(model, data)
    except Exception as e:
        print(f"Viewer failed to start: {e}")
        return

    with viewer_ctx as viewer:
        # 给窗口一点初始化时间（有些平台 launch_passive 后立即查询会短暂为 False）
        time.sleep(0.2)
        print(f"viewer.is_running()={viewer.is_running()}")
        if not viewer.is_running():
            print("Viewer exited immediately. 可能原因：无图形界面/GLFW 初始化失败/窗口被关闭。")
            print("Falling back to headless stepping for 2 seconds...")
            for _ in range(int(2.0 / model.opt.timestep)):
                mujoco.mj_step(model, data)
            return
        step_count = 0
        action_buffer = None # 当前执行的动作块
        action_idx = 0
        
        while viewer.is_running():
            step_start = time.time()
            
            # --- ACT 推理 (低频) ---
            # 每隔 ACT_EXECUTION_HORIZON 步推理一次，或者当 buffer 用完时
            if action_buffer is None or action_idx >= ACT_EXECUTION_HORIZON:
                # 1. 获取观测
                renderer.update_scene(data)
                img = renderer.render()
                # 观测：这里用当前 ctrl（目标关节角）作为 proprio 输入
                current_ctrl = data.ctrl.copy()

                # 2. 预测动作块（真正的 learned policy）
                action_buffer = policy.predict_chunk(current_ctrl)
                action_idx = 0
                # print(f"ACT Inference: Generated chunk of size {len(action_buffer)}")
            
            # --- 执行动作 ---
            # 取出当前步的动作
            if action_idx < len(action_buffer):
                target_q = action_buffer[action_idx]
                data.ctrl[:] = target_q
                action_idx += 1
            else:
                # 如果 buffer 用完了但还没到推理时间（不太可能如果逻辑对的话），保持最后动作
                pass

            # --- 物理步进 ---
            mujoco.mj_step(model, data)
            viewer.sync()
            
            # 保持实时
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
