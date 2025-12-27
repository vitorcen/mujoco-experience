import time
import os
import mujoco
import mujoco.viewer
import numpy as np
import re
from dataclasses import dataclass

# 动作展示参数：原地踏步 + 双手抬高在身体前左右摆动（循环）
CYCLE_SECONDS = 6.0
STEP_FREQ_HZ = 1.5

# 训练/推理参数（让“策略不是 mock”）
CKPT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "g1_diffusion.pt")
TRAIN_STEPS = 2000
BATCH_SIZE = 96
LR = 3e-4
HORIZON = 10         # 生成长度（步）
N_DIFF_STEPS = 20    # 推理采样步数
DT_ACTION = 0.05

# 训练用：stand_ctrl 注入点
_GLOBAL_STAND_CTRL = None

# ===========================
# Diffusion Policy (Mock)
# ===========================
def build_g1_ctrl_traj(stand_ctrl: np.ndarray, t: float) -> np.ndarray:
    """和 ACT 脚本一致：原地踏步 + 双手抬高在身体前左右摆动。"""
    ctrl = stand_ctrl.copy()

    s = (t % CYCLE_SECONDS) / CYCLE_SECONDS
    phase = 2.0 * np.pi * STEP_FREQ_HZ * t

    l = np.sin(phase)
    r = np.sin(phase + np.pi)
    l_swing = np.maximum(0.0, l)
    r_swing = np.maximum(0.0, r)

    hip_amp = 0.22
    knee_amp = 0.75
    ankle_amp = 0.35

    ctrl[0] += hip_amp * (l_swing - 0.3)
    ctrl[3] += knee_amp * l_swing
    ctrl[4] += ankle_amp * l_swing

    ctrl[6] += hip_amp * (r_swing - 0.3)
    ctrl[9] += knee_amp * r_swing
    ctrl[10] += ankle_amp * r_swing

    swing = np.sin(2.0 * np.pi * s)
    lift = 0.9

    ctrl[15] += -lift
    ctrl[16] += -0.35 * swing
    ctrl[17] += 0.25 * swing
    ctrl[18] += 0.9

    ctrl[22] += -lift
    ctrl[23] += 0.35 * swing
    ctrl[24] += -0.25 * swing
    ctrl[25] += 0.9

    ctrl[12] += -0.08 * swing
    return ctrl


@dataclass
class DiffCfg:
    obs_dim: int
    act_dim: int
    horizon: int
    d_model: int = 256


class DiffNet:
    """一个最小 diffusion policy 网络：预测 epsilon（噪声）。"""

    def __init__(self, cfg: DiffCfg, device: str):
        import torch
        import torch.nn as nn

        self.cfg = cfg
        self.device = device

        # 简单时间 embedding（sin/cos）
        self.t_embed = nn.Sequential(
            nn.Linear(2, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
        ).to(device)

        self.net = nn.Sequential(
            nn.LayerNorm(cfg.obs_dim + cfg.horizon * cfg.act_dim + cfg.d_model),
            nn.Linear(cfg.obs_dim + cfg.horizon * cfg.act_dim + cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, cfg.horizon * cfg.act_dim),
        ).to(device)

    def state_dict(self):
        return {"t_embed": self.t_embed.state_dict(), "net": self.net.state_dict()}

    def load_state_dict(self, sd):
        self.t_embed.load_state_dict(sd["t_embed"])
        self.net.load_state_dict(sd["net"])

    def __call__(self, obs, x_noisy, t01):
        import torch

        # t01: [B] in [0,1]
        te = torch.stack([torch.sin(2 * torch.pi * t01), torch.cos(2 * torch.pi * t01)], dim=-1)
        te = self.t_embed(te)
        inp = torch.cat([obs, x_noisy.view(x_noisy.shape[0], -1), te], dim=-1)
        out = self.net(inp).view(-1, self.cfg.horizon, self.cfg.act_dim)
        return out


class DiffusionPolicy:
    def __init__(self, obs_dim: int, act_dim: int):
        import torch

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cfg = DiffCfg(obs_dim=obs_dim, act_dim=act_dim, horizon=HORIZON)
        self.model = DiffNet(self.cfg, device=self.device)

        # 线性 beta schedule（训练/推理都需要）
        self.betas = np.linspace(1e-4, 0.02, N_DIFF_STEPS).astype(np.float32)
        self.alphas = 1.0 - self.betas
        self.alphas_cum = np.cumprod(self.alphas)

        os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
        if os.path.exists(CKPT_PATH):
            ckpt = torch.load(CKPT_PATH, map_location=self.device)
            self.model.load_state_dict(ckpt["state_dict"])
            print(f"✅ Loaded Diffusion checkpoint: {CKPT_PATH}")
        else:
            print("⚠️  Diffusion checkpoint not found. Training a small policy now (synthetic data)...")
            self._train_synthetic()

    def _train_synthetic(self):
        import torch
        import torch.nn.functional as F

        rng = np.random.default_rng(0)
        t0s = rng.uniform(0, CYCLE_SECONDS, size=TRAIN_STEPS)

        assert _GLOBAL_STAND_CTRL is not None, "stand_ctrl not initialized"
        stand_ctrl = _GLOBAL_STAND_CTRL

        obs = []
        traj = []
        for t0 in t0s:
            o = build_g1_ctrl_traj(stand_ctrl, float(t0))
            seq = []
            for k in range(self.cfg.horizon):
                tk = float(t0 + (k + 1) * DT_ACTION)
                seq.append(build_g1_ctrl_traj(stand_ctrl, tk))
            obs.append(o)
            traj.append(np.stack(seq, axis=0))

        obs = torch.tensor(np.stack(obs, axis=0), dtype=torch.float32, device=self.device)
        traj = torch.tensor(np.stack(traj, axis=0), dtype=torch.float32, device=self.device)

        opt = torch.optim.AdamW(list(self.model.t_embed.parameters()) + list(self.model.net.parameters()), lr=LR)

        self.model.net.train()
        for step in range(TRAIN_STEPS):
            idx = torch.randint(0, obs.shape[0], (BATCH_SIZE,), device=self.device)
            o = obs[idx]
            x0 = traj[idx]

            # sample diffusion step
            t_idx = torch.randint(0, N_DIFF_STEPS, (o.shape[0],), device=self.device)
            a_bar = torch.tensor(self.alphas_cum, device=self.device)[t_idx].view(-1, 1, 1)
            eps = torch.randn_like(x0)
            xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * eps

            t01 = t_idx.float() / float(N_DIFF_STEPS - 1)
            eps_pred = self.model(o, xt, t01)
            loss = F.mse_loss(eps_pred, eps)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            if (step + 1) % 250 == 0:
                print(f"[Diff train] step {step+1}/{TRAIN_STEPS} loss={loss.item():.6f}")

        torch.save({"state_dict": self.model.state_dict(), "cfg": self.cfg.__dict__}, CKPT_PATH)
        self.model.net.eval()
        print(f"✅ Saved Diffusion checkpoint: {CKPT_PATH}")

    def sample_action(self, obs_ctrl: np.ndarray) -> np.ndarray:
        """返回 horizon 序列，取第 1 步执行。"""
        import torch

        o = torch.tensor(obs_ctrl[None, :], dtype=torch.float32, device=self.device)
        x = torch.randn((1, self.cfg.horizon, self.cfg.act_dim), device=self.device)

        for i in reversed(range(N_DIFF_STEPS)):
            t01 = torch.tensor([i / float(N_DIFF_STEPS - 1)], device=self.device, dtype=torch.float32)
            with torch.inference_mode():
                eps = self.model(o, x, t01)

            beta = self.betas[i]
            alpha = self.alphas[i]
            a_bar = self.alphas_cum[i]

            beta_t = torch.tensor(beta, device=self.device)
            alpha_t = torch.tensor(alpha, device=self.device)
            a_bar_t = torch.tensor(a_bar, device=self.device)

            # DDPM update
            x = (1 / torch.sqrt(alpha_t)) * (x - (beta_t / torch.sqrt(1 - a_bar_t)) * eps)
            if i > 0:
                x = x + torch.sqrt(beta_t) * torch.randn_like(x)

        return x[0].detach().cpu().numpy()

def main():
    # 1. 加载 G1 (同上)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    menagerie_dir = os.path.join(project_root, "mujoco_menagerie")
    g1_dir = os.path.join(menagerie_dir, "unitree_g1")
    
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
    
    assets = {}
    with open(os.path.join(g1_dir, "g1.xml"), "rb") as f:
        assets["g1.xml"] = f.read()
    assets_dir = os.path.join(g1_dir, "assets")
    if os.path.exists(assets_dir):
        for name in os.listdir(assets_dir):
            if os.path.isfile(os.path.join(assets_dir, name)):
                with open(os.path.join(assets_dir, name), "rb") as f:
                    assets[f"assets/{name}"] = f.read()
    
    # NOTE:
    # 原地踏步：把 pelvis 焊到 world，避免无平衡控制时倒地。
    
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

    mujoco.mj_resetDataKeyframe(model, data, key_id)
    stand_ctrl = model.key_ctrl[key_id].copy()
    data.ctrl[:] = stand_ctrl

    global _GLOBAL_STAND_CTRL
    _GLOBAL_STAND_CTRL = stand_ctrl.copy()

    # 初始化 Policy（可训练/可加载）
    policy = DiffusionPolicy(obs_dim=model.nu, act_dim=model.nu)
    
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=480, width=640)

    # 仿真循环
    print("Starting simulation viewer...")
    try:
        viewer_ctx = mujoco.viewer.launch_passive(model, data)
    except Exception as e:
        print(f"Viewer failed to start: {e}")
        return

    with viewer_ctx as viewer:
        time.sleep(0.2)
        print(f"viewer.is_running()={viewer.is_running()}")
        if not viewer.is_running():
            print("Viewer exited immediately. 可能原因：无图形界面/GLFW 初始化失败/窗口被关闭。")
            print("Falling back to headless stepping for 2 seconds...")
            for _ in range(int(2.0 / model.opt.timestep)):
                mujoco.mj_step(model, data)
            return
        while viewer.is_running():
            step_start = time.time()
            
            # --- Diffusion Policy 推理 ---
            # 通常也是闭环运行，或者 Receding Horizon Control
            # 这里演示逐帧预测
            
            # 1. 观测
            renderer.update_scene(data)
            img = renderer.render()
            current_action = data.ctrl.copy()
            
            # 2. 预测
            # 真正的 diffusion policy：采样一段动作序列，执行第一步
            seq = policy.sample_action(current_action)
            target_action = seq[0]
            
            # 3. 执行
            # G1 是位置控制，直接写入
            data.ctrl[:] = target_action

            mujoco.mj_step(model, data)
            viewer.sync()
            
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
