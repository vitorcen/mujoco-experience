# patches/ — 第三方 submodule 本地改动 (local mods to vendored submodules)

把对 `dependencies/<submodule>` 内文件的修改维护成**可审查的 git patch**,而不是
复制整文件覆盖。约定与姊妹项目 `../isaaclab-experience/patches` 一致。

> Keep changes to third-party submodules as **reviewable git patches**, not full-file
> copies. Same convention as the sibling `isaaclab-experience/patches`.

## 布局 / Layout

```
patches/<submodule_name>/NNNN-description.patch
```

每个 `.patch` 是在 submodule 根目录 `git diff` 出来的(路径形如 `a/foo b/foo`),
所以用 `git -C dependencies/<submodule_name> apply` 应用。

## 用法 / Usage

```bash
./patches/apply_patches.sh        # 幂等:已应用的自动跳过 / idempotent
```

## 为什么 patch 而非 cp / Why patch, not cp

- **diff 即文档** —— 一眼看出改了什么 (the diff *is* the documentation)。
- 上游 pin 漂移时 `git apply` **报冲突**,而 `cp` 会静默覆盖 (conflict-aware vs silent clobber)。
- 仓库不被整份 vendored 文件撑大 (no vendored bloat)。

## 何时改用 fork / When to fork instead

改动大、长期自己维护 → 直接 fork 该仓库、提交到 fork、submodule 指向 fork
(例:`pi05_minimax_vla`、`robocasa-training` 已是 vitorcen fork)。
**小 tweak 用 patch,长期 own 用 fork,绝不用 cp。**

## 现有 patch / Current patches

| submodule | patch | 内容 |
|---|---|---|
| `mujoco_menagerie` | `0001-trs_so_arm100-add-vla-cameras-and-pick-cube` | trs_so_arm100 加 wrist/top 相机 + chip_box_lid/orange_cube 抓取目标 |
| `unitree_mujoco` | `0001-cpp-add-climb_stairs-example` | C++ example 加 climb_stairs(trot 爬楼)+ CMake target |
| `unitree_rl_gym` | `0001-add-terrain-stairs-deploy-configs` | 新增 g1 terrain/stairs 部署配置 + scene_terrain.xml |
