---
name: feedback_watchdog_interleave_earlystop
description: GR00T watchdog 默认中频 interleave + SR 掉头早停;改 max_steps 会毁 LR 调度，必须用回调停 slice
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 8f5534ca-f3c2-49d7-a247-bf54c9209e6f
---

用户指令（2026-05-31）："改成中频 interleave + SR 掉头早停，后续默认这样来"。已实现并设为默认。

**Why:** 单 4090 训练与 eval 不能共存(训练占 ~22.7/24GB)。原 watchdog 单 slice 训到 MAX_STEPS 才评 → 看不到中途 SR 曲线、无法早停(human 微调过拟合快，峰值常在早段)。中频 interleave 在峰值早时净省时间(早停砍掉过拟合尾段，省 > 重启开销)，代价是跑满时约 +9~14% 重启开销。甜点区 1500-2000 步/slice。

**How it works（关键不要踩坑）:**
- **绝不按 slice 降 `--max_steps`** —— 那会让 LR 调度每个 slice 衰减到 ~0 形成锯齿、毁掉训练。正确做法:`max_steps` 永远=最终目标(调度是一条连续曲线)，slice 靠 `scripts/launch_finetune_n17.py` 新增的 `StopAtStepCallback`(env `STOP_AT_STEP`>0 时 `on_step_end` 命中即 `should_save+should_training_stop`)干净停(rc=0)，下个 slice resume 续调度。
- `scripts/watchdog_gr00t.sh` 新 env:`INTERLEAVE_STEPS`(默认1500,=0退回旧"末段评")、`EARLYSTOP_PATIENCE`(默认3,=0关)、`EARLYSTOP_MIN_DELTA`(默认0)。interleave 开时自动把 `EVAL_STEPS_MULTIPLE`/`KEEP_MULTIPLE` 对齐到 `INTERLEAVE_STEPS`(否则边界 ckpt 既不评也不留)。
- 主循环:每 slice 停在 `(base/IL+1)*IL` 边界(cap MAX_STEPS)→ 顶部 eval 边界 ckpt → `check_early_stop`(读 EVAL_LOG 按 step 排序,running-best 后连续 PATIENCE 个未超 best 即停;需 >PATIENCE 点防单点噪声)→ rc=0 时只有 `global_step>=MAX_STEPS` 才算真完成,否则当 slice 边界 continue。
- **顺带修两 bug:** EVAL_MAX_STEPS 默认 400→**1200**(400 砍掉慢成功、骗人,见 [[project_n17_mimicgen_native_mix]] 坑3);eval 循环 `sort -t- -k2` 改 `sed|sort -n`(路径含 mujoco-experience 的 dash 会打乱顺序)。
- 改动用 写新文件+原子 mv 落盘(运行中的 bash 持旧 inode,原地编辑会损坏)。旧版备份 `watchdog_gr00t.sh.orig_20260531`。

**How to apply:** 以后任何 GR00T watchdog 长训默认就是 interleave+早停,无需额外设;要旧行为传 `INTERLEAVE_STEPS=0`。仍需 `EVAL_LOG=<专属路径>`(见 [[project_n17_mimicgen_native_mix]] 坑2)。
