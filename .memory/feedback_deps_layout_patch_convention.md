---
name: feedback_deps_layout_patch_convention
description: 第三方 submodule 统一放 dependencies/;改动用 .patch 不用 cp;自己的 fork 提交到 fork
metadata:
  type: feedback
---

# 依赖布局 + submodule 改动维护约定 (2026-06-04 定)

**所有第三方(非 vitorcen fork)submodule 统一放在 `dependencies/`。** vitorcen 自己 fork 的
(`pi05_minimax_vla`、`robocasa-training`)留在顶层。2026-06-04 把 7 个第三方 submodule
(mujoco / mujoco_menagerie / KungfuBot / unitree_mujoco / unitree_rl_gym / unitree_sdk2_python /
DeepMimic_mujoco)从顶层 `git mv` 进了 `dependencies/`,并把 3 个之前未纳管的 GR00T/gr1 树
(Isaac-GR00T / Isaac-GR00T-gr1 / robocasa-gr1-tabletop-tasks)正式 `submodule add` 纳管
(commit 72df867)。

**Why:** 顶层乱;`dependencies/` 已有 robocasa/openpi,统一隔离第三方代码。注意 `Isaac-GR00T`
名字带 Isaac 但**不依赖 Isaac Sim**,跑的是 RoboCasa=robosuite=MuJoCo,属 MuJoCo 栈依赖
(详见 [[project_robocasa_gr00t_two_tree_deps]])。

**改第三方 submodule 内文 → 用 `.patch`,绝不用 `cp` 整文件覆盖。**
- 约定:`patches/<submodule_name>/NNNN-description.patch`(submodule 根 `git diff` 出来)+
  `patches/apply_patches.sh`(幂等:已应用则反向校验跳过)。与姊妹项目
  `/home/david/work/isaaclab-experience/patches` 同约定。
- **Why:** diff 即文档、`git apply` 上游漂移时报冲突;`cp` 静默覆盖、看不出改了啥、撑大仓库。
  已淘汰旧的 `patch_files/do-patches.sh`(cp 版)。
- 现有 patch:mujoco_menagerie(trs_so_arm100 加 VLA 相机+抓取方块)、unitree_mujoco
  (climb_stairs example)、unitree_rl_gym(terrain/stairs 部署配置)。

**How to apply:**
- 新依赖:第三方进 `dependencies/`,自己长期 own 的 fork 进顶层并提交到 fork。
- 改了 `dependencies/<sm>` 内文:`git -C dependencies/<sm> diff > patches/<sm>/NNNN-*.patch`,
  别 cp。fresh clone 后跑 `./patches/apply_patches.sh` 复原改动。
- submodule 内解压的运行时资产(如 robocasa textures):`.gitmodules` 设
  `submodule.<path>.ignore = untracked` 抑制噪声,别提交资产。
- ⚠️ 移动 submodule 用 `git mv`(自动改 .gitmodules path + 修 worktree `.git` 指针);
  移动后 submodule 的 `.git` 是**文件**(gitdir 指针)不是目录,脚本判断用 `-e` 不是 `-d`。
- 本仓有 **auto-commit hook**,大改动可能被自动 commit(本次移动被自动收成 72df867),
  动手前后查 `git log` 确认状态,别重复提交。
