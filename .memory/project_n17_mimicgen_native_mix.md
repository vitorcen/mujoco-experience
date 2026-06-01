---
name: project_n17_mimicgen_native_mix
description: MimicGen 增强训练用 GR00T 原生多数据集 mixing，不做物理合并；v2.0 MG 直接可加载
metadata: 
  node_type: memory
  type: project
  originSessionId: 8f5534ca-f3c2-49d7-a247-bf54c9209e6f
---

N1.7 MimicGen 数据增强：**用 GR00T 原生 multi-dataset mixing，不物理合并 parquet**（消除 v2.0→v2.1 格式统一这一整类特殊情况）。

**根因 / Why it works:** `gr00t/data/dataset/factory.py` 对 `config.data.datasets` 里每个 spec 的每个 `dataset_paths` 独立 `generate_stats()`+`generate_rel_stats()` 再建 `ShardedSingleStepDataset`，采样权重 `weight = relative_length × mix_ratio`（`base_config.py:127` 对所有 spec 求和）。每个数据集保留自己的 `meta/tasks.jsonl`+`modality.json`，模型只消费解析后的语言字符串和 state/action 张量。

**所以三类差异全部无关（实测 confirmed）：**
- MG v2.0 缺 `frame_index`、`observation.state`/`action` 是 `list<double>`(非 fixed_size)、`next.reward`/`timestamp` 是 double(非float)、无 `episodes_stats.jsonl` → loader 用 pandas 读、按 `meta/episodes.jsonl`+sharding 重建索引，`generate_stats` 重生成 stats。实测 `ShardedSingleStepDataset` 加载 MG 8644ep = 3322 shards / 3.4M steps 成功。
- `tasks.jsonl` task_index 语义不同（MG 0=doors,2=door；human 0=door,1=doors）→ 无所谓，各自内部一致，模型读字符串。
- `modality.json` MG 多一个 `human.task_name` 映射 → 无害。

**实现:** `scripts/launch_finetune_n17.py` 已加 env 接口（向后兼容，无 env 则与原行为一致）：
- `PRIMARY_MIX_RATIO`（默认1.0）= `--dataset_path`(human) 权重
- `EXTRA_DATASETS` = 逗号分隔 `path=ratio`，追加为额外 mix spec
env 经 watchdog→train_n17.sh(无 env -i，继承)→launcher 透传。

**数据路径:**
- human v2.1: `~/.cache/robocasa/datasets/v1.0/target/atomic/OpenCabinet/20250813/lerobot_old`（500ep，champion 训练集）
- MG v2.0 池: `~/.cache/robocasa/datasets/v1.0/pretrain/atomic/OpenCabinet/20250819/mg/demo/2025-08-20-21-54-43/lerobot`（8644ep，全量参与，比文档 1500 子采样更多样）

首跑 Recipe A 单阶段混合：human=1.0 / MG=3.0（75% MG / 25% human），OUTPUT_DIR=`checkpoints/gr00t_n17_mg_mix`（勿覆盖 human-only `gr00t_n17_opencabinet`），MAX_STEPS=40000，watchdog 分段 eval（KEEP/EVAL_STEPS_MULTIPLE=2000）。详见 [[project_n17_16k_run_state]]。

**✅ 首跑完成（2026-05-31）:** 40000 步一个 cycle 干净跑完零崩（num_workers=0 扛住 40k）。`epoch=1.0`（混合大集只过 1 遍，与 human-only 多遍过 500ep 不同 regime → 早期 ckpt 欠训，peak 大概率在后半段），final loss=0.0644 仍在降（疑未过拟合，>40k 可能还能涨）。

**⚠️ 坑：只剩 28000-40000 的 ckpt（28000,30000,...,40000 + 39250/500/750）。** 根因 = HF Trainer `save_total_limit=10`（train_n17.sh 默认）。训练一个 slice 不崩 → watchdog 自定义"保留 2000 倍数"的 prune 只在 cycle 之间跑、只有一个 cycle → save_total_limit 静默清掉 28000 以下全部。**下次长训要保留全网格须 `SAVE_TOTAL_LIMIT=` 调到 ≥(MAX_STEPS/KEEP_MULTIPLE + 余量)，如 40k/2k→设 25。** 本跑早期 ckpt 不可恢复，但因 epoch=1.0 早期欠训，预计非 peak，可接受。

**坑2：EVAL_LOG `logs/gr00t_n17_ckpts.csv` 与 human-only 旧跑共用 → 含旧行 1000-16000（污染）。** ≥18000 的行才是本 MG-mix 跑（旧跑只到 16000）。authoritative 用 sweep_n17.py 写独立 `benchmark/results/sweep/`，不受污染。**任何新 watchdog 跑务必 `EVAL_LOG=<专属路径>`**，否则 ckpt 步号落在 1000-16000 区间会被当"已评"全跳过、in-loop eval 不跑。

**坑3（重大）：coarse 400 步 eval 在此 regime 完全失效、会骗人。** MG 让策略变慢（成功 mean ~594 步 > 400）。watchdog 粗评(400步)显示 28k-40k≈7%，但 sweep_n17.py **1200 步**权威评：**ckpt-34000=53.3%(8/15,mean594)、ckpt-40000=40.0%(6/15)**。即 from-scratch 75%MG 混合 **34k 见顶 53.3% > champion 50%**、40k 尾部过拟合回落。**判 SR 一律用 1200 步**（matches baselines），别信 400 步粗评。注:53.3% 是 15 轮,baselines 是 30 轮,要进榜需补 34k 30 轮。

**🏆 Stage-2 结果（2026-05-31，重大）:MG 两阶段让 N1.7 反超 N1.5。** 30 轮 ×1200 步 seed-locked 权威评:ckpt-12000=59.3%(16/27)、**ckpt-14000=69.2%(18/26,solid,sweep 峰)**、ckpt-15000 不可靠(同 ckpt 两跑 81% vs 50%,sim-DNF 8-9 个=**该 ckpt 策略本身把 sim 撞崩多**,非环境;12k/14k 仅 3-4 DNF 正常)。**诚实结论:stage-2 峰 ~65-70%(14k),超 human-only 50%、追平/超 N1.5 64%。** 新榜:N1.7-stage2-14k ~69% ≥ N1.5 64% > human-only 50% > from-scratch-mix-34k 53.3% > pi0.5 17.4%。

**坑4（重大）:naive resume 扩展会 LR 跳变冲击崩策略。** 15k(70%)naive resume 续训(max_steps 15k→30k),HF 按新 horizon 重建调度、15k 处 LR 从~0 跳回~5.3e-5,1000 步把策略打崩到 **16000=0%**(真实 eval)。**扩展已收敛 ckpt 必须用低 LR**(给 train_n17.sh 加了 `LEARNING_RATE`/`WARMUP_RATIO` env;Phase2 用 LR=1e-5 续,resume 处 LR≈4e-6 不冲击)。详见 [[feedback_prefer_resume_continue_training]]。

**坑5:coarse 10ep 严重误导。** 12k coarse 70%→30轮 59.3%;14k coarse 60%→30轮 69.2%;15k coarse 70%→30轮 50-81%。10ep CI ±15-30%,定榜必 30 轮(见坑3)。

**✅ Phase2 低LR续训完成→确认 14k 是 sweep 峰。** resume from 15000、LR=1e-5(resume LR≈4e-6 不冲击,16k=50% 未崩,vs naive 1e-4 的 16k=0%——验证 LR 旋钮修复有效)。但续训点 16k=50%、17k=0% 持续退化,interleave 早停 patience=4 正确触发("peak is past")停在 17k。**结论:past 15k 无论 naive 还是低LR 都退化,sweep 峰=ckpt-14000(~69%),多训无益。**

**坑6:eval-server 卡死会留 D-state 僵尸毁全盘。** watchdog 默认 EVAL_POLICY_PORT=5557;某次 server "failed to bind 5557 within 300s" 留下 D-state python(SIGKILL 无效),它毒化所有 /proc 扫描→pkill/pgrep/ps 全 hang→watchdog cleanup_procs 卡死;且占住 5557 让后续 eval 绑不上(sweep 用 5640+ 才能跑)。primary GPU 不能 --gpu-reset,只能 reboot 清。**教训:eval 换非 5557 端口;watchdog 的 pkill -f 对 D-state 脆弱。**

**▶ Phase3 公平 30/30 重跑进行中（用户授权覆盖红线）:** policies.tsv N1.7 行已改 `GR00T-N1.7-MG2stage`→`gr00t_n17_s2_humanft/checkpoint-14000`;run_benchmark.py ROUNDS=30 SEED_BASE=0 EVAL_EP_RETRIES=10 重跑 N1.7-MG2stage+N1.5+pi0.5(都 seed-locked 同 layout 4,6 同款橱柜,retry 凑满30有效)。旧 authoritative 已备份 `benchmark/results/_pre_fair30_backup_20260601/`。

**✅ HF 已发布（2026-06-01）:** wsagi/GR00T-N1.7-RoboCasa-OpenCabinet —— **main=checkpoint-14000(stage-2 best)**+新卡片(公平榜:N1.5 70%>N1.7-MG2stage 53.3%>pi0.5 23.3%,两阶段 MG 配方,诚实说明 N1.5 领先);旧 human-only 11000 在分支 **`human500-ckpt-11000`**;YAML `base_model: nvidia/GR00T-N1.7-3B` + `datasets: [robocasa/robocasa-assets]` 已生效可点上游。

**坑7:HF 发大模型三件套(全踩过)。** (a) safetensors 走 **xet 端点被墙**(0.00B/s),小文件走普通 LFS 正常 → 修=**`HF_HUB_DISABLE_XET=1`** 强制 legacy LFS(~5-11MB/s)。(b) **`hf upload-large-folder`(整目录)对大文件不稳/卡** → 改 **逐文件 `hf upload <repo> <file> <repopath>`**(每文件单独 commit;已 pre-upload 的会秒 commit=内容去重)。(c) **视频在卡片放不出来**=README 用了相对路径 `src="videos/x.mp4"`;HF 渲染器不解析 → 必须用**全 resolve URL** `<video controls src="https://huggingface.co/<repo>/resolve/main/videos/x.mp4">`(参考 wsagi/HumanoidBench-TD-MPC2)。**结论:HF 发大模型 = `HF_HUB_DISABLE_XET=1` + 逐文件 hf upload + 视频用 resolve URL;删旧 ckpt 前先确认新 ckpt 已 commit。**

**最终结论:** MG 两阶段(MG-mix 预训34k→纯human微调14k)把 N1.7 从 human-only ~43-50% 提到公平 53.3%(真实~10点提升),但**未超 N1.5 70%**。之前"N1.7 69%>N1.5 64%"是 exclude-DNF 不均匀偏置+方差的假象;公平 30/30(retry 凑满,0 DNF 剔除)翻盘=N1.5 领先。闭环方差大(±~8-9%/30轮),单轮估计。

**▶ Stage-2 原始（Recipe B 第二阶段，用户指令"基于34k回纯human额外15k内超50%"）:** warm-start from `gr00t_n17_mg_mix/checkpoint-34000`（model_type=Gr00tN1d7→launcher 当 warm，smoke 验证 first loss=0.093 << 冷启1.25），DATASET=纯 human（无 EXTRA_DATASETS），MAX_STEPS=15000，OUTPUT=`checkpoints/gr00t_n17_s2_humanft`，**SAVE_TOTAL_LIMIT=25**(修坑1)、KEEP_MULTIPLE=1000(细网格)、**EVAL_LOG 专属**(修坑2)、**EVAL_MAX_STEPS=1200**(修坑3)。预期早段(2k-6k)冲新 peak 后过拟合,细网格抓峰,跑完 sweep_n17.py(CKPT_DIR=该目录)精扫。
