---
name: feedback-train-with-watcher
description: 长时训练必须分 slice + 持续 eval + 早停 — 任何 ACT/SL/RL 训练 launcher 不挂 watcher 都是 bug
metadata: 
  node_type: memory
  type: feedback
  originSessionId: 7a08b045-ec5e-41e0-90e1-3c36231eee63
---

任何 > 1 小时的训练任务，launcher **必须**同时分 slice + 每段 eval 出 SR 曲线 + 暴露 abort marker，不能"启动训练就走人等结果"。Nohup 后台无 watcher 是反模式。

**Why**：兄弟项目 humanoid-training 总结的血泪教训（详见 `/home/david/work/humanoid-training/.memory/feedback_train_with_watcher.md`）：
1. DP v0.4.0 训 10h、train_loss 0.554 → 0.011（看起来完美收敛），final eval 0/15 — 1× 10k 步 quick eval 就能 1h 内发现，省 9h
2. DreamerV3 small h1hand 1M 步 6h+ 烧光，ep score 5-9 / success_bar=650 ≈ 0% — 没挂 watcher 没有任何中途信号
3. 关键产物是 **SR vs step 曲线**，不是 final ckpt 本身

**How to apply**（已落地到 `robocasa-training/scripts/train_act.py`）：

1. **分 slice**：默认 `total_steps / 10`，每段保 ckpt + 跑 3-ep eval
2. **状态分类**：DEAD（连续 3 eval < 5%）/ UNDERFIT（最近 vs 之前窗口提升 ≤ 5%）/ OVERFIT（peak 已过 ≥2 eval 且 current < 0.7×peak）/ PROGRESS
3. **早停 marker**：`output_dir/.eval_abort` — touch 它训练在下一 slice 边界退出
4. **DEAD 默认不杀**：`--dead_kills` opt-in，否则只告警；误判杀训练比错过几次 eval 更糟
5. **VRAM 协调**：本机 4090 24G ACT 训练吃 22.75/24 GiB → train + eval 必须 sequential（停训→eval→续训），不能并行。云端独立 GPU 才能并行 watcher。
6. **最终 slice 加大 eval**：默认 `--final-eval-episodes=20` 拿 honest peak number；中间 slice 3 ep 够 DEAD/PROGRESS 区分
7. **CSV 必出**：`logs/sr_curve.csv` 一行一 slice，含 status 列 — 这是 ssh 唯一可视化（不依赖 tensorboard）

**反模式 / Anti-patterns**：
- ❌ `nohup python train.py &` 走人
- ❌ 只看 train loss / KL loss，不在训练中 eval
- ❌ 训完一次性大 eval，发现 0% 才知道白训
- ❌ 把 eval 嵌进训练主循环（拖慢训练 throughput）—— 应该是同进程顺序分段或独立 watcher 进程

兄弟项目实现参考：`/home/david/work/humanoid-training/scripts/train_watcher.py`、`launch_train.sh`、`train_status.sh`。
