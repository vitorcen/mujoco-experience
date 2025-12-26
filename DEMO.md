# MuJoCo 演示指南

本指南介绍如何运行 MuJoCo 的各种演示和示例。

> **前提条件**：
> 1. 已完成项目构建（运行 `./build.sh`）。
> 2. 以下所有命令均假设你在项目根目录 `mujoco-experience/` 下执行。
> 3. 第三方模型需要初始化 submodule: `git submodule update --init mujoco_menagerie`

---

## 1. C++ 示例 (Simulate)

`simulate` 是 MuJoCo 的主要可视化工具，支持加载 XML 模型、交互式拖拽、查看物理参数等。

### 常用操作键位
- **鼠标左键拖拽**: 对物体施加外力
- **鼠标右键拖拽**: 旋转视角
- **Scroll**: 缩放视角
- **Space**: 暂停/继续仿真
- **Backspace**: 重置仿真
- **Double Click**: 选择物体 (查看详细信息)
- **F1**: 查看帮助菜单

---

## 2. 第三方机器人 (MuJoCo Menagerie)

[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) 是官方维护的高质量机器人模型库。
以下示例加载的是 `scene.xml`，通常包含地面和环境光照，适合直接观察。

### 🦿 四足机器人 (Quadrupeds)

**Unitree Go1** (宇树科技)
```bash
./build/bin/simulate ./mujoco_menagerie/unitree_go1/scene.xml
```

**Unitree Go2** (宇树科技)
```bash
./build/bin/simulate ./mujoco_menagerie/unitree_go2/scene.xml
```

**ANYbotics ANYmal C**
```bash
./build/bin/simulate ./mujoco_menagerie/anybotics_anymal_c/scene.xml
```

**Boston Dynamics Spot**
```bash
./build/bin/simulate ./mujoco_menagerie/boston_dynamics_spot/scene.xml
```

### 🦾 机械臂 (Manipulators)

**Franka Emika Panda**
```bash
./build/bin/simulate ./mujoco_menagerie/franka_emika_panda/scene.xml
```

**Universal Robots UR5e**
```bash
./build/bin/simulate ./mujoco_menagerie/universal_robots_ur5e/scene.xml
```

**KUKA IIWA 14**
```bash
./build/bin/simulate ./mujoco_menagerie/kuka_iiwa_14/scene.xml
```

**xArm 7** (UFactory)
```bash
./build/bin/simulate ./mujoco_menagerie/ufactory_xarm7/scene.xml
```

### 🤖 人形机器人 (Humanoids)

**Unitree H1**
```bash
./build/bin/simulate ./mujoco_menagerie/unitree_h1/scene.xml
```

**Unitree G1**
```bash
./build/bin/simulate ./mujoco_menagerie/unitree_g1/scene.xml
```

### 🖐️ 灵巧手 (Dexterous Hands)

**Shadow Hand**
```bash
./build/bin/simulate ./mujoco_menagerie/shadow_hand/left_hand.xml
```

**Wonik Allegro Hand**
```bash
./build/bin/simulate ./mujoco_menagerie/wonik_allegro/scene_right.xml
```

---

## 3. 内置模型运行清单

MuJoCo 提供了丰富的内置模型，以下是可直接复制运行的命令：

### 🤖 机器人与载具 (Robots & Vehicles)

**经典人形机器人 (Humanoid)**
```bash
./build/bin/simulate ./mujoco/model/humanoid/humanoid.xml
```

**简易小车 (Car)**
```bash
./build/bin/simulate ./mujoco/model/car/car.xml
```

**肌腱驱动机械臂 (Tendon Arm)**
```bash
./build/bin/simulate ./mujoco/model/tendon_arm/arm26.xml
```

**滑块曲柄机构 (Slider Crank)**
```bash
./build/bin/simulate ./mujoco/model/slider_crank/slider_crank.xml
```

### 🧶 软体与布料 (Soft Bodies & Cloth)

**飘动的旗帜 (Flag)**
```bash
./build/bin/simulate ./mujoco/model/flex/flag.xml
```

**软体果冻 (Jelly)**
```bash
./build/bin/simulate ./mujoco/model/flex/jelly.xml
```

**蹦床 (Trampoline)**
```bash
./build/bin/simulate ./mujoco/model/flex/trampoline.xml
```

**软体兔子 (Bunny)**
```bash
./build/bin/simulate ./mujoco/model/flex/bunny.xml
```

### 🎲 物理互动与接触 (Interactions)

**纸牌屋 (Cards)**
*展示复杂的接触动力学*
```bash
./build/bin/simulate ./mujoco/model/cards/cards.xml
```

**易碎杯子 (Mug)**
*展示复合物体与碰撞*
```bash
./build/bin/simulate ./mujoco/model/mug/mug.xml
```

**3x3x3 魔方 (Cube)**
```bash
./build/bin/simulate ./mujoco/model/cube/cube_3x3x3.xml
```

**吊床 (Hammock)**
```bash
./build/bin/simulate ./mujoco/model/hammock/hammock.xml
```

### ⚙️ 经典物理演示 (Physics Demos)

**牛顿摆 (Newton's Cradle)**
```bash
./build/bin/simulate ./mujoco/model/replicate/newton_cradle.xml
```

**多米诺骨牌 (Dominos)**
```bash
./build/bin/simulate ./mujoco/model/sleep/dominos.xml
```

**气球 (Balloons)**
```bash
./build/bin/simulate ./mujoco/model/balloons/balloons.xml
```

**粒子系统 (Particles)**
```bash
./build/bin/simulate ./mujoco/model/replicate/particle.xml
```

---

## 4. 运行控制与推理 (Control & Inference)

MuJoCo 本身是一个物理引擎，不包含预训练的智能体策略。`simulate` 工具仅用于被动预览和手动交互。
要实现“推理”或控制机器人运动，你需要编写代码（通常是 Python）来计算控制信号并发送给引擎。

### 示例：控制 Unitree Go1 运动

创建一个 Python 脚本 `control_demo.py`，让机器人趴下/站起：

```python
import time
import mujoco
import mujoco.viewer
import numpy as np

# 加载 Unitree Go1 模型
# 注意：确保已经初始化了 mujoco_menagerie submodule
model = mujoco.MjModel.from_xml_path('./mujoco_menagerie/unitree_go1/scene.xml')
data = mujoco.MjData(model)

# 获取关节名称映射 (仅供参考)
# print([model.joint(i).name for i in range(model.njnt)])

# 定义一个简单的控制信号生成器 (正弦波)
def get_control_signal(t):
    # 这里只是简单演示让关节动起来
    # 实际推理会使用神经网络 policy.predict(observation)
    return np.sin(t * 5) * 5  # 简单的震荡信号

with mujoco.viewer.launch_passive(model, data) as viewer:
    start_time = time.time()
    
    # 仿真循环
    while viewer.is_running():
        step_start = time.time()
        current_time = step_start - start_time

        # 生成控制信号并应用到执行器 (actuators)
        # data.ctrl 对应 XML 中定义的 actuators
        # Go1 有 12 个电机
        ctrl_signal = get_control_signal(current_time)
        
        # 将信号应用到所有执行器 (仅作演示)
        data.ctrl[:] = ctrl_signal

        # 物理步进
        mujoco.mj_step(model, data)

        # 同步 Viewer
        viewer.sync()

        # 实时同步
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
```

运行控制脚本：
```bash
python control_demo.py
```

### 进阶：强化学习推理

如果你想运行训练好的强化学习策略（如 RL 训练出的走路策略），通常需要：
1. **策略文件**: 一个训练好的神经网络权重文件（如 `.pt` 或 `.onnx`）。
2. **推理代码**: 加载权重，获取 MuJoCo 的观测 (`data.qpos`, `data.qvel`, `data.sensordata`)，输入网络，获取动作，赋值给 `data.ctrl`。

MuJoCo 官方提供了 **[MJM (MuJoCo MPC)](https://github.com/google-deepmind/mujoco_mpc)** 和 **[MJX (MuJoCo XLA)](https://github.com/google-deepmind/mujoco/tree/main/mjx)** 用于高级控制和学习。
