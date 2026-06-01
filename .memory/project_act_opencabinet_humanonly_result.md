---
name: project-act-opencabinet-humanonly-result
description: ACT 在 RoboCasa OpenCabinet human-only 500-demo 上训 13 epoch = 0% SR，精度受限（非 bug），下一步 mimicgen 10×数据
metadata: 
  node_type: memory
  type: project
  originSessionId: 7a08b045-ec5e-41e0-90e1-3c36231eee63
---

**结论（2026-05-29）**：在 `robocasa-training` 仓用 LeRobot ACT 训 RoboCasa **OpenCabinet** target/human（500 demos, 184k 帧），从 scratch 训到 **300k steps ≈ 13 epoch**（4090, bs=8, chunk=100, lr=1e-5），**closed-loop success rate = 0%**（10 个干净 episode 全失败：100k 诊断 5ep + 300k 最终 5ep）。

**这不是 bug —— 是数据/精度受限。** 诊断方法（决定性）：open-loop replay test —— 用训练集帧走 serve_act 完全相同的 preproc→policy→postproc，对比预测 action vs 数据集记录值。证据：
- 策略**确实学到了**：gripper 时机误差比仅 0.01（完美），动作平滑
- 但 **eef 精度不够**：300k 时 eef_pos L1≈0.05–0.06、eef_rot L1≈0.04，而可靠操作需 ~0.02（差 2–3×）
- eef_rotation 是最顽固瓶颈（全程降得最多 0.072→0.040 但仍不够；抓门把手最需姿态精度）
- eef_pos 在 ~0.05 plateau，300k 甚至回弹（疑轻微过拟合）
- 排除了 plumbing bug：会话早期 **GR00T ckpt 用同一 client 跑到 70%**，证明 action 字典格式 + env 全对
- 排除了 inference 开环 horizon：n_action_steps=8 和 50 都 0%

**关键数据事实**：OpenCabinet target/human demos **底盘全程静止**（action 前 5 维 base_motion+control_mode 恒为 0）—— 机器人预置在柜前只用手臂。所以 eval 里 Δbase~30mm 是正确的不是 bug。

**下一步杠杆（最强）**：mimicgen 数据。`pretrain/atomic/OpenCabinet` 的 mg demo 有 **8644 episodes**（比 registry 标的 5000 多，11GB tar, v2.0 格式, 3.74M 帧）。10× 数据直接攻 eef 精度短板。约束：全量 precache=2.2TB 不可能，盘只够 ~1500-demo 子集（383GB）。recipe 待定：mimicgen-only vs pretrain-then-finetune（RoboCasa/GR00T 官方用后者）。

**已扩展验证（2026-05-29 同日）——三个杠杆全部 0%**：

1. **mimicgen-only 500**（pretrain/mg 子集，训到 150k）：closed-loop SR 仍 **0%**（50k/100k clean eval）。但精度上 **eef_rot=0.033 完胜 human**（甚至优于 human 300k 的 0.040），eef_pos/grip 持平。→ 数据源能破 rotation 精度瓶颈，但 closed-loop 不 transfer（mimicgen demos 初始分布与 target eval 偏 + 闭环累积误差）。

2. **temporal ensembling**（coeff=0.01，ACT 原生闭环技巧，本仓 serve_act.py 已实现 + client --send-reset + reset op）：human 300k 上 **10/10 = 0%**。动作更平滑但任务仍不完成 → 失败不是 jitter/累积误差，是更深的能力/精度问题。

**最终判定**：ACT(80M, from scratch) 在 RoboCasa OpenCabinet 上，无论数据源（human/mimicgen）、训练时长（13 epoch）、inference（chunk-replay / temporal ensembling），closed-loop SR 都是 **0%**。对照：**GR00T(2B VLA, 大规模预训练)同 eval pipeline 70%**（会话早期验证）。差距是**模型容量 + 预训练规模**，不是 eval/pipeline bug（pipeline 被 GR00T 的 70% 证明正确）。

3. **pretrain(mimicgen-500)-then-finetune(human-500)**（官方 recipe，train_act.py `--init-from` 加载预训练权重）：finetune 从 loss 0.243 起步（远低于 scratch 3.1，预训练权重确实加载），25k/75k clean eval 仍 **0%**。精度（在 human 数据上）：finetune@100k eef_rot=0.040（继承 mimicgen 的 rotation 优势，追平 human 300k），但 eef_pos=0.103（反而比 human-only 差，mimicgen pos 偏置没被纠正）。net 精度混合，SR 仍 0%。

**最终最终判定（所有 4 个杠杆都试尽）**：ACT(80M from scratch) 在 RoboCasa OpenCabinet 上，human-only / mimicgen-only / temporal-ensembling / pretrain-then-finetune **全部 closed-loop SR = 0%**。对照 GR00T(2B VLA, 大规模预训练)同 eval pipeline **70%**。**确认是模型容量 + 预训练规模的能力差距**，不是数据、inference、recipe 或 pipeline 问题。小模型从头训不适合 RoboCasa atomic 操作任务 —— 要 >0% 必须走预训练 VLA 路线（GR00T 已 70%）或大幅加大模型/数据规模。这个 "ACT-scratch 全 0% vs GR00T-VLA 70%" 对照矩阵是 robocasa-training 仓的核心负面结果发现。

**复用资产**：`subset_mimicgen.py`（v2.0→v2.1 子集+补 stats，内置 dtype/image-stats 两修复）、`precache_videos.py` + `frame_cache_patch.py`（避 torchcodec SEGV）、`serve_act.py`（含 TE）、分 slice watcher `train_act.py`。

相关：[[reference-lerobot-torchcodec-fork-segv]]（训练提速）、[[feedback-train-with-watcher]]（分 slice eval）。配套文档 `robocasa-training/doc/training_act.html`。
