# SCRIPTS.md - 脚本运行指南

本目录 `scripts/` 包含用于演示、测试和控制 MuJoCo 仿真的 Python 脚本。

> **提示**：运行前请确保已激活环境并更新子模块：
> ```bash
> conda activate mujoco
> git submodule update --init --recursive
> ```

---

## 1. 🤖 AI & VLA 策略 (AI Policies)

本节展示结合视觉语言模型 (VLA)、Transformer (ACT) 和扩散模型 (Diffusion) 的高级控制策略。

### 四足机器人 VLA 导航
**脚本**: `scripts/quadruped_vla_demo.py`
演示使用 OpenVLA 视觉模型指挥 Unitree Go2 机器狗走向红球（跨形态控制）。
```bash
python scripts/quadruped_vla_demo.py
```

### 人形机器人 ACT 策略
**脚本**: `scripts/humanoid_act_demo.py`
演示 ACT (Action Chunking with Transformers) 策略控制 Unitree G1。
```bash
python scripts/humanoid_act_demo.py
```

### 人形机器人 Diffusion Policy
**脚本**: `scripts/humanoid_diffusion_demo.py`
演示 Diffusion Policy 策略控制 Unitree G1。
```bash
python scripts/humanoid_diffusion_demo.py
```

### VLA 机械臂推理
**脚本**: `scripts/vla_inference_demo.py`
演示 OpenVLA 控制机械臂的完整流程。
```bash
python scripts/vla_inference_demo.py
```

---

## 2. 🏃 人形机器人 MoCap (DeepMimic)

本节演示使用 **DeepMimic** 动作捕捉数据驱动人形机器人（直接运行即可预览）。

**基础运动 (Walk, Run, Spinkick)**:
```bash
# 行走 (Walk)
python scripts/humanoid_walk_demo.py

# 跑步 (Run)
python scripts/humanoid_run_demo.py

# 回旋踢 (Spinkick)
python scripts/humanoid_spinkick_demo.py
```

**特技与恢复 (Dance, Backflip, Getup)**:
```bash
# 街舞 (Dance B)
python scripts/humanoid_dance_b_demo.py

# 后空翻 (Backflip)
python scripts/humanoid_backflip_demo.py

# 跌倒爬起 (Stand Up - Faceup & Facedown)
python scripts/humanoid_getup_demo.py
```

---

## 3. 🦾 机械臂操作 (Manipulation)

### Franka Panda 抓取 (IK Control)
**脚本**: `scripts/panda_ik_demo.py`
使用差分逆运动学 (Differential IK) 控制 Panda 机械臂抓取物体。
```bash
python scripts/panda_ik_demo.py
```

### 基础关节控制
简单的关节空间控制演示。
```bash
# Franka Panda 简单控制
python scripts/panda_demo.py

# Franka FR3 控制
python scripts/fr3_demo.py

# UR10e 机械臂控制
python scripts/ur10e_demo.py
```

---

## 4. 🐕 四足机器人 (Quadruped)

### Go2 崎岖地形行走
**脚本**: `scripts/go2_terrain_demo.py`
演示 Unitree Go2 在复杂地形（台阶、障碍）上的行走。
```bash
# GUI 模式
python scripts/go2_terrain_demo.py

# Headless 模式 (无 GUI)
python scripts/go2_terrain_demo.py --headless --duration 20
```

### Boston Dynamics Spot
**脚本**: `scripts/spot_demo.py`
控制 Spot 机器狗进行基础动作演示。
```bash
python scripts/spot_demo.py
```

---

## 5. 🚁 其他机器人 (Other Robots)

### Agility Robotics Cassie
**脚本**: `scripts/cassie_demo.py`
控制 Cassie 双足机器人。
```bash
python scripts/cassie_demo.py
```

### Bitcraze Crazyflie 2
**脚本**: `scripts/crazyflie_demo.py`
控制微型四旋翼无人机。
```bash
python scripts/crazyflie_demo.py
```

### Hello MuJoCo
**脚本**: `scripts/hello_mujoco.py`
最基础的 MuJoCo 仿真循环测试。
```bash
python scripts/hello_mujoco.py
```
