# SCRIPTS.md - 脚本运行指南

本目录 `scripts/` 包含用于演示、测试和控制 MuJoCo 仿真的 Python 脚本。

> **提示**：运行以下脚本前，请确保已激活 Conda 环境：
> ```bash
> conda activate mujoco
> ```
> 并且已初始化 submodule：
> ```bash
> git submodule update --init mujoco_menagerie
> ```

---

## 目录索引

1.  **[quadruped_vla_demo.py](#1-四足机器人-vla-导航演示)**: 🔥 **(创新)** 演示使用 OpenVLA 视觉模型指挥 Unitree Go2 机器狗走向红球（跨形态控制）。
2.  **[humanoid_act_demo.py](#2-人形机器人-act-演示)**: 🔥 演示 ACT (Action Chunking with Transformers) 风格的策略控制 Unitree G1 人形机器人。
3.  **[humanoid_diffusion_demo.py](#3-人形机器人-diffusion-policy-演示)**: 🔥 演示 Diffusion Policy 风格的策略控制 Unitree G1。
4.  **[panda_ik_demo.py](#4-franka-panda-抓取演示-ik-control)**: 🔥 演示使用逆运动学 (IK) 控制 Panda 机械臂抓取桌子上的物体。
5.  **[vla_inference_demo.py](#5-vla-机械臂推理演示)**: 演示 OpenVLA 控制机械臂的完整流程。
3.  **[hello_mujoco.py](#3-hello-mujoco-基础步进)**: 最简 MuJoCo 仿真循环。
4.  **[spot_demo.py](#4-boston-dynamics-spot-控制)**: 波士顿动力 Spot 机器狗控制。
5.  **[cassie_demo.py](#5-agility-robotics-cassie-控制)**: Cassie 双足机器人控制。
6.  **[crazyflie_demo.py](#6-bitcraze-crazyflie-2-控制)**: 微型四旋翼无人机控制。
7.  **[ur10e_demo.py](#7-universal-robots-ur10e-控制)**: UR10e 机械臂控制。
8.  **[panda_demo.py](#8-franka-emika-panda-简单控制)**: Panda 机械臂简单关节控制。
9.  **[fr3_demo.py](#9-franka-emika-fr3-控制)**: Franka FR3 机械臂控制。

---

## 1. 四足机器人 VLA 导航演示

**脚本**: `scripts/quadruped_vla_demo.py`

🔥 **跨形态控制演示**：这是一个创新的 Demo，展示如何将针对机械臂训练的 VLA 模型（OpenVLA）用于指挥四足机器人（Unitree Go2）。
*   **原理**：将 OpenVLA 输出的手臂动作向量（Delta X, Delta Y）重映射为机器狗的（前进速度, 转向速度）。
*   **场景**：机器狗面前有一个红球，VLA 接收图像指令 "Walk to the red ball"，输出动作驱动机器狗靠近。
*   **底层控制**：内置了一个简易的运动学步态生成器 (Kinematic Gait Generator)，将速度指令转化为 12 个关节的动作。

**运行方式**:
```bash
python scripts/quadruped_vla_demo.py
```

---

## 2. 人形机器人 ACT 演示

**脚本**: `scripts/humanoid_act_demo.py`

演示 **ACT (Action Chunking with Transformers)** 策略在 Unitree G1 人形机器人上的应用。ACT 的核心思想是一次推理预测未来的一个动作块（Action Chunk），然后按顺序执行，并利用时间集成（Temporal Ensembling）来平滑动作。此脚本模拟了这一过程。

**运行方式**:
```bash
python scripts/humanoid_act_demo.py
```

---

## 3. 人形机器人 Diffusion Policy 演示

**脚本**: `scripts/humanoid_diffusion_demo.py`

演示 **Diffusion Policy** 在 Unitree G1 上的应用。Diffusion Policy 利用扩散模型生成平滑且多模态的动作序列。此脚本模拟了生成平滑全身动作的过程。

**运行方式**:
```bash
python scripts/humanoid_diffusion_demo.py
```

---

## 4. Franka Panda 抓取演示 (IK Control)

**脚本**: `scripts/panda_ik_demo.py`

🔥 **主流实现演示**：此脚本展示了 MuJoCo 中实现机器人操作的主流方法：
1.  **场景构建**: 使用 XML `<include>` 将 Panda 机器人、桌子、方块（物体）组合在一起。
2.  **差分逆运动学 (Differential IK)**: 使用 `mujoco.mj_jacSite` 实时计算雅可比矩阵，驱动机械臂末端追踪红色方块，并模拟抓取动作。

**运行方式**:
```bash
python scripts/panda_ik_demo.py
```

---

## 2. VLA 视觉-语言-动作模型演示

**脚本**: `scripts/vla_inference_demo.py`

演示 VLA (Visual-Language-Action) 模型（如 OpenVLA, SmolVLA）与 MuJoCo 的集成流程。加载 Unitree H1 人形机器人，使用 `mujoco.Renderer` 获取视觉输入，模拟输出控制信号。

**运行方式**:
```bash
python scripts/vla_inference_demo.py
```

---

## 3. Hello MuJoCo (基础步进)

**脚本**: `scripts/hello_mujoco.py`

最基础的仿真循环，加载 Humanoid 模型并运行仿真。

**运行方式**:
```bash
python scripts/hello_mujoco.py
```

---

## 4. Boston Dynamics Spot 控制

**脚本**: `scripts/spot_demo.py`

控制波士顿动力 Spot 机器狗。让所有关节进行正弦波运动，展示机器狗的动态。

**运行方式**:
```bash
python scripts/spot_demo.py
```

---

## 5. Agility Robotics Cassie 控制

**脚本**: `scripts/cassie_demo.py`

控制 Cassie 双足机器人。Cassie 具有复杂的闭链结构，此脚本演示简单的关节驱动。

**运行方式**:
```bash
python scripts/cassie_demo.py
```

---

## 6. Bitcraze Crazyflie 2 控制

**脚本**: `scripts/crazyflie_demo.py`

控制 Crazyflie 2.0 微型无人机。演示如何控制旋翼推力。

**运行方式**:
```bash
python scripts/crazyflie_demo.py
```

---

## 7. Universal Robots UR10e 控制

**脚本**: `scripts/ur10e_demo.py`

演示 UR10e 机械臂。此脚本可能演示简单的重力补偿或轨迹跟随。

**运行方式**:
```bash
python scripts/ur10e_demo.py
```

---

## 8. Franka Emika Panda 简单控制

**脚本**: `scripts/panda_demo.py`

简单的关节空间控制，让 Panda 机械臂关节摆动。

**运行方式**:
```bash
python scripts/panda_demo.py
```

---

## 9. Franka Emika FR3 控制

**脚本**: `scripts/fr3_demo.py`

控制 Franka FR3 机械臂（Panda 的后继型号）。

**运行方式**:
```bash
python scripts/fr3_demo.py
```
