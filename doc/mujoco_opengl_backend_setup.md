## MuJoCo 可视化/离屏渲染环境配置（GLX / EGL / OSMesa）

你在运行 `mujoco.viewer` 或设置 `MUJOCO_GL=osmesa` 时遇到的问题，基本都属于 **OpenGL 后端与系统动态库** 的问题，而不是 MJCF 模型或控制策略本身。

常见现象：

- GUI viewer 无法启动，出现 `Xlib:  extension "NV-GLX" missing on display ":0".`
- 设置 `MUJOCO_GL=osmesa` 后，`import mujoco` 直接报错：
  - `AttributeError: 'NoneType' object has no attribute 'glGetError'`

下面给出推荐配置路线。

---

## 1. 先理解 3 种后端分别解决什么问题

- **GLX（默认）**
  - 用于“开窗口”的桌面渲染（GLFW + GLX）
  - 依赖 X11/Wayland 的桌面 OpenGL
  - 远程桌面/容器/ssh -X 经常出问题

- **EGL（推荐用于 headless 离屏渲染）**
  - 适合服务器/无显示环境渲染，能配合 `mujoco.Renderer` 录视频
  - 一般不需要 X11
  - 需要系统提供 `libEGL`

- **OSMesa（软件离屏渲染）**
  - 纯软件渲染（CPU），不依赖 GPU
  - 需要系统提供 `libOSMesa`
  - 如果系统没装 `libOSMesa`，`MUJOCO_GL=osmesa` 会在 import 阶段失败

---

## 2. 你的日志说明了什么？

### 2.1 `NV-GLX missing`（viewer 起不来）

这是 **GLX / 显示环境 / NVIDIA 驱动** 的问题。典型原因：

- 当前 DISPLAY 对应的 X Server 没有 NVidia GLX 扩展
- 驱动/远程显示/容器环境导致 GLX vendor 不一致

这会导致 `mujoco.viewer.launch_passive()` 创建窗口失败，控制循环直接退出，看起来就像“策略没跑”。

### 2.2 `MUJOCO_GL=osmesa` import 报错

一般是系统缺库。可以用 Python 验证：

```bash
python - <<'PY'
import ctypes.util
print("OSMesa ->", ctypes.util.find_library("OSMesa"))
print("EGL    ->", ctypes.util.find_library("EGL"))
print("GLX    ->", ctypes.util.find_library("GLX"))
PY
```

如果 `OSMesa -> None`，就说明系统没有 `libOSMesa`，OSMesa 后端必然不可用。

---

## 3. 推荐方案：用 EGL 做离屏录制（无需 viewer）

本项目 `scripts/go2_terrain_demo.py` 支持 headless + 录制：

```bash
MUJOCO_GL=egl python scripts/go2_terrain_demo.py --headless --duration 20 --record out.mp4
```

如果写视频失败，请确认系统有 ffmpeg：

```bash
sudo apt-get update && sudo apt-get install -y ffmpeg
```

---

## 4. 如果你坚持用 OSMesa（软件渲染）

安装系统依赖：

```bash
sudo apt-get update && sudo apt-get install -y libosmesa6 libosmesa6-dev
```

然后再运行：

```bash
MUJOCO_GL=osmesa python scripts/go2_terrain_demo.py --headless --record out.mp4
```

---

## 5. 如果你想“开窗口”但 GLX 有问题（用 Mesa 软件 GLX 兜底）

安装 Mesa：

```bash
sudo apt-get update && sudo apt-get install -y libgl1-mesa-dri libglx-mesa0 mesa-utils
```

强制使用 Mesa + 软件渲染启动 viewer：

```bash
__GLX_VENDOR_LIBRARY_NAME=mesa LIBGL_ALWAYS_SOFTWARE=1 python scripts/go2_terrain_demo.py
```

这通常能绕过 NVIDIA GLX 的问题，但帧率会比较低（CPU 渲染）。

