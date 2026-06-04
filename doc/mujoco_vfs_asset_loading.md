## MuJoCo 资源加载路径问题：为什么会找不到文件？

在 MuJoCo 的 MJCF（XML）里，很多资源（例如 `mesh`、`texture`、`hfield`）都会通过 `file="..."` 引用外部文件。当你用 Python 加载模型时，常见的“找不到文件”错误类似：

- `Error opening file assets/height_field.png: No such file or directory`
- `Error opening file .../meshes/xxx.stl: No such file or directory`

根因通常是 **XML 里写的是相对路径**，而你加载 XML 的方式（工作目录、`<include>`、`<compiler>` 里的目录选项）导致 MuJoCo 实际去找文件的路径与你预期不一致。

本项目里我主要用两类策略解决：

- **策略 A：重写 XML 中的相对路径 + 使用 MuJoCo VFS 注入资源（推荐，最稳）**
- **策略 B：保证从“正确的基准目录”加载 XML（不推荐，容易被 include/meshdir 影响）**

下面重点解释策略 A（这是我之前和现在都在用的通用方案）。

---

## 策略 A：XML 重写 + VFS（Virtual File System）注入资源

MuJoCo Python 支持：

```python
mujoco.MjModel.from_xml_string(xml_string, assets=assets_dict)
```

其中 `assets_dict` 是一个“虚拟文件系统”：**key 是 MuJoCo 要打开的文件名（或相对路径），value 是对应文件的 bytes**。

因此，只要你能做到：

- 让 XML 引用的 `file="xxx"` 变成你能控制的 key（例如统一改成 `assets/xxx`）
- 在 `assets_dict` 里提供对应 key 的内容

就不需要依赖真实文件系统的相对路径解析，也不需要改 submodule 文件。

---

## 为什么 `meshdir="assets"` 会“污染”其他资源的路径？

以 `unitree_mujoco` 的 Go2 为例，`go2.xml` 顶部有：

- `<compiler meshdir="assets" ... />`

当主场景 `scene_terrain.xml` 通过 `<include file="go2.xml" />` 把它包含进来后，MuJoCo 在解析资源路径时，可能会对某些资源也带上 `assets/` 前缀（你看到的报错正是 `assets/height_field.png`）。

这会导致：

- 你在 `scene_terrain.xml` 里写了 `file="height_field.png"` 或 `file="../height_field.png"`
- 但 MuJoCo 实际去找 `assets/height_field.png`

如果你的 VFS 没有提供 `assets/height_field.png` 这个 key，就会报 “No such file or directory”。

---

## 本项目的落地做法（Go2 Terrain Demo）

在 `scripts/go2_terrain_demo.py` 我做了三件事：

- **1）重写主场景 XML 的 hfield 路径**
  - 把 `../height_field.png` 直接替换为 `assets/height_field.png`
  - 把 `../unitree_hfield.png` 直接替换为 `assets/unitree_hfield.png`

- **2）把地形图片从正确目录读出来**
  - Go2 的图片实际在：`dependencies/unitree_mujoco/unitree_robots/go2/height_field.png`
  - 不是在 `dependencies/unitree_mujoco/unitree_robots/` 根目录

- **3）在 VFS 里“双路径注册”**
  - 同时注册：
    - `assets/height_field.png`
    - `height_field.png`
  - 这样即使 MuJoCo 在不同情况下用不同的相对路径，也能命中。

---

## 适用范围

这个方案适用于几乎所有“submodule 内含大量 include/相对路径/贴图/网格”的场景，例如：

- `mujoco_menagerie` 的复杂机器人模型（大量 STL/OBJ）
- `unitree_mujoco` 的场景（hfield/texture/mesh 混合）
- 任何包含 `<include>` 且资源路径不稳定的 MJCF

关键原则是：

- **不要改 submodule 文件**
- **在脚本里统一把 XML 的 `file=...` 重写到你能控制的虚拟路径**
- **把所有被引用的资源以 bytes 注入到 `assets` 字典**

