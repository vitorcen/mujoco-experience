# MuJoCo Experience

Google DeepMind MuJoCo 物理引擎学习与实践项目。MuJoCo (Multi-Joint dynamics with Contact) 是一个高性能的物理仿真引擎，广泛用于机器人学、生物力学和机器学习研究。

<div align="center">
  <img src="doc/assets/g1_29dof_rev.png" alt="Unitree G1 Simulation" width="600"/>
  <p><i>Unitree G1 人形机器人在 MuJoCo 中的仿真</i></p>
</div>

---

## 项目结构

```
mujoco-experience/
├── mujoco/                    # MuJoCo 源码（submodule）
│   ├── model/                 # 内置示例模型 (XML)
│   ├── sample/                # C++ 示例代码
│   ├── simulate/              # 图形化仿真器源码
│   └── python/                # Python 绑定
├── mujoco_menagerie/          # 第三方机器人模型库（submodule）
├── unitree_mujoco/            # Unitree 机器人仿真器（submodule）
│   ├── simulate/              # C++ 仿真器源码
│   ├── example/               # 示例程序（cpp/python/ros2）
│   ├── unitree_robots/        # Unitree 机器人 MJCF 模型
│   └── readme_zh.md           # 中文文档
├── scripts/                   # Python 演示与控制脚本
│   └── vla_inference_demo.py  # VLA 模型推理演示
├── DEMO.md                    # C++ 仿真器运行指南 (Simulate)
├── SCRIPTS.md                 # Python 脚本运行指南 (Scripts)
├── UNITREE.md                 # Unitree 机器人仿真指南
├── build.sh                   # 一键构建脚本
├── init.sh                    # 环境初始化脚本
├── init-unitree.sh            # Unitree 仿真器安装脚本
└── README.md                  # 本文档
```

---

## 系统要求

### 硬件要求
- **CPU**: x86-64 或 ARM64 处理器
- **GPU**: 支持 OpenGL 3.2+ (用于图形渲染)

### 软件要求
- **操作系统**: Linux (Ubuntu 20.04+), macOS, Windows
- **构建工具**:
  - CMake 3.16+
  - C++ 编译器 (GCC/Clang/MSVC)
  - Ninja (推荐) 或 Make
- **Python** (可选，用于 Python 绑定): 3.8+

---

## 安装与构建

### 1. 克隆项目

```bash
git clone <your-repo-url> mujoco-experience
cd mujoco-experience
```

### 2. 初始化环境与依赖

使用 `init.sh` 自动配置 Conda 环境并安装 Python 依赖：

```bash
./init.sh
conda activate mujoco
```

### 3. 构建 C++ 仿真器

使用 `build.sh` 一键构建 `simulate` 可视化工具：

```bash
./build.sh
```

构建完成后，可执行文件位于 `build/bin/simulate`。

---

## 快速体验

### 1. 运行可视化仿真器 (Simulate)

`simulate` 是 MuJoCo 的原生查看器，适合预览模型和手动交互。

**运行内置 Humanoid:**
```bash
./build/bin/simulate ./mujoco/model/humanoid/humanoid.xml
```

**运行 Unitree Go1 机器狗 (Menagerie):**
```bash
# 需先初始化 submodule
git submodule update --init mujoco_menagerie

./build/bin/simulate ./mujoco_menagerie/unitree_go1/scene.xml
```

👉 **更多 XML 模型运行指令请查看 [演示指南 (DEMO.md)](./DEMO.md)**

### 2. Unitree 机器人仿真器 (C++ Based)

基于 Unitree SDK2 的完整仿真环境，支持 Go2, B2, H1, G1 等多款机器人，提供 sim-to-real 开发流程。

**快速安装：**
```bash
./init-unitree.sh
```

**启动仿真示例：**
```bash
# 启动仿真器
./unitree_mujoco/simulate/build/unitree_mujoco -r go2 -s scene_terrain.xml

# 另开终端运行控制程序
./unitree_mujoco/example/cpp/build/stand_go2
```

👉 **详细安装、控制策略、ROS2 集成、常见问题等请查看 [Unitree 仿真指南 (UNITREE.md)](./UNITREE.md)**

### 3. 运行 Python 控制脚本

使用 Python API 进行控制和推理。

**运行 VLA 视觉-语言-动作模型演示:**

```bash
python scripts/vla_inference_demo.py
```

👉 **更多 Python 脚本说明请查看 [脚本指南 (SCRIPTS.md)](./SCRIPTS.md)**

---

## 官方资源

- **官方仓库**: [google-deepmind/mujoco](https://github.com/google-deepmind/mujoco)
- **官方文档**: [MuJoCo Documentation](https://mujoco.readthedocs.io/)
- **模型库**: [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)

---

## 许可证

MuJoCo 使用 Apache License 2.0 许可证。
