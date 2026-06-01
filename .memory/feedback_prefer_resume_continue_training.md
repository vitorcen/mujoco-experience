---
name: feedback_prefer_resume_continue_training
description: 能续训就续训——扩展已完成的训练用 resume 续 global_step，绝不从头 warm-start 重训
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 95f5bea6-bee9-4707-8fe6-95951b6ac1f3
---

用户指令（2026-05-31）："能续训就续训"。要延长一个已跑完的训练(看更多步会不会继续提升/找 sweep 峰)时,**默认用 resume 续训**(从最新 ckpt 接着 global_step 往上走),而不是从某个 ckpt warm-start 重新计步、重训。

**Why:** resume 保留 optimizer 动量 + LR 调度连续性 + 训练进度,省时且不重走已学过的路;warm-start 重训会重置 global_step+optimizer、重 warmup 扰动好策略(实测 stage-2 warm-start step1000 一度跌到 10% 才回升)。

**How it works:** GR00T `gr00t/experiment/experiment.py:305` 恒 `trainer.train(resume_from_checkpoint=True)` → `get_last_checkpoint(output_dir)`。所以**同一 OUTPUT_DIR 重启 watchdog + 调高 MAX_STEPS,就会从该目录最新 ckpt 续**(BASE_MODEL 仅在 output_dir 空时用)。LR 调度按新 MAX_STEPS 重建、恢复 last_epoch → LR 从 base*(剩余比例) 平滑续(如 15k→30k 扩展,resume 处 LR≈0.5*base)。

**How to apply:**
- 扩展训练:`OUTPUT_DIR=<原目录> MAX_STEPS=<更高> bash watchdog_gr00t.sh`,自动 resume。
- 配合 [[feedback_watchdog_interleave_earlystop]] 的 interleave 实时 eval + 早停,边续边看 SR 曲线找 sweep 峰。
- INTERLEAVE_STEPS 对齐现有 ckpt 网格(现有 1000 倍数就设 1000),否则 KEEP_MULTIPLE 重对齐会误删现有 ckpt。
- 复用同一 EVAL_LOG 避免重评已评 ckpt;但注意早停 no_improve 计数会带历史(必要时调高 PATIENCE 或人工看)。
- 区别于 Recipe B 的"换数据阶段"(那是有意 warm-start 新阶段,如 [[project_n17_mimicgen_native_mix]] 的 34k→纯human);"续训"特指同数据同阶段往后多训。
