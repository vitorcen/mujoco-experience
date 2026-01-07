## MuJoCo Viewer 无法启动（NV-GLX missing）排查与解决（中文说明）

当你运行 MuJoCo Python 脚本使用 `mujoco.viewer.launch_passive()` 时，可能会看到类似错误：

- `Xlib:  extension "NV-GLX" missing on display ":0".`

并表现为：

- 脚本打印了 “Starting simulation...”，但窗口没有出现或一闪而过
- 控制循环实际上没有开始（因为 viewer 创建失败就退出）

---

## 1. 这是什么问题？

`mujoco.viewer` 基于 GLFW 创建 OpenGL 上下文。

出现 `NV-GLX missing` 通常意味着：

- 当前显示环境（X11/Wayland/远程显示）没有可用的 NVIDIA GLX 扩展
- 或者 NVIDIA 驱动/GLX 与当前会话不匹配（常见于远程桌面、容器、ssh -X、混合显卡等）

这不是 MuJoCo 模型本身的问题，而是 **GUI 渲染后端（OpenGL/GLX）的问题**。

---

## 2. 快速验证：用 headless 模式确认“策略是否在跑”

本项目的 `scripts/go2_terrain_demo.py` 支持 `--headless`：

```bash
python scripts/go2_terrain_demo.py --headless --duration 20
```

你会看到周期性输出，例如：

- `[t=  1.00s] base_xyz = [...]`

这能确认：

- 模型加载没问题
- 控制/步进循环在正常运行

---

## 3. 如果你一定要可视化，有哪些选择？

### 选择 A：在“有桌面 OpenGL/GLX”的环境运行

最直接：在本机桌面会话（而非不完整的远程显示）运行脚本。

### 选择 B：EGL / OSMesa（离屏）

如果你只是想“能跑起来、能渲染/录视频”，可以尝试：

```bash
MUJOCO_GL=egl python scripts/go2_terrain_demo.py --headless
```

或：

```bash
MUJOCO_GL=osmesa python scripts/go2_terrain_demo.py --headless
```

注意：

- `egl` 需要系统/驱动提供 EGL（通常与 NVIDIA 驱动相关）
- `osmesa` 走软件渲染，可能需要额外系统包支持（不同发行版包名不同）

### 选择 C：在可视化机器上跑，远程只看结果

如果你的开发环境是服务器/容器，建议在有 GUI 的机器上运行 viewer，
服务器侧只做 headless 仿真并记录日志/视频。

---

## 4. 本项目的建议实践

- **开发调控制**：优先 `--headless`，看输出确认“确实在动”
- **需要看效果**：切到有桌面 OpenGL 的机器运行 viewer，或改成离屏渲染输出视频

