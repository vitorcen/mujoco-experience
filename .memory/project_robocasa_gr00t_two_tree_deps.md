---
name: project-robocasa-gr00t-two-tree-deps
description: "RoboCasa GR00T 依赖双树架构——N1.5 fork(eval) vs N1.7 上游(训练),各自独立 submodule,N1.7 原生 XYZ_ROT6D 免 fork"
metadata: 
  node_type: memory
  type: project
  originSessionId: 7a08b045-ec5e-41e0-90e1-3c36231eee63
---

**RoboCasa GR00T 的 Isaac-GR00T 依赖是"双树"架构（2026-05-29 定）**，因为没有任何单一仓同时有 N1.7 + RoboCasa 支持：

- **N1.5 + RoboCasa**：`mujoco-experience/dependencies/Isaac-GR00T` = robocasa-benchmark fork（commit 9d7d7a9，pyproject v1.1.0），含 `panda_omron` data config + 手动 rotation_6d。**这是 N1.5/N1.7 eval 用的 `panda_omron` 管线**（`robocasa_gr00t` conda env 的 editable gr00t 指向这里，跑出过 OpenCabinet 70%）。**绝不能动。**
- **N1.7 训练**：`robocasa-training/dependencies/Isaac-GR00T` = NVIDIA 上游 submodule，pin `n1.7-release`(23ace64)，含 `gr00t_n1d7`。**vendored 进 robocasa-training 自己**（不伸进父仓 deps，更不碰外部 isaaclab-experience）。加时用 `--reference` 加速后 `git repack -a -d` + 删 alternates **彻底脱钩**，`git fsck` 通过。

**关键技术点**：NVIDIA N1.7 上游**原生有 `ActionType.EEF` + `ActionFormat.XYZ_ROT6D`**（在 `gr00t/data/types.py` + `state_action/state_action_processor.py`），正是 robocasa-benchmark fork 当年在 N1.5 上手动加的那套 quaternion↔rotation_6d。**所以 N1.7 训练侧不需要 port fork 的任何 RoboCasa 代码**——只需仿 LeIsaac `leisaac_config_n17.py` 注册一个 modality config（panda_omron 的 state16/action12/3cam 键 + action delta_indices=range(40) 对齐 N1.7 的 action_horizon=40）。NVIDIA 上游本身**没有** `panda_omron`/`BimanualPandaGripperDataConfig`/rotation_6d 的旧式 DataConfig（N1.7 重构掉了）。

**两个 N1.7 base 模型本地 HF cache 已存**：`nvidia/GR00T-N1.7-3B`(warm，推荐) + `nvidia/Cosmos-Reason2-2B`(cold backbone)。

**env 已建 + 脚本已移植 + smoke 跑通（2026-05-29）**：
- conda env `gr00t-n17` = clone `fastwam`(torch2.7.1+cu128 + flash-attn2.8.3 + numpy1.26.4 现成,省 flash-attn 编译) + `pip install -e dependencies/Isaac-GR00T --no-deps` + 补 transformers==4.57.3/diffusers/tyro/albumentations/dm-tree。
- 脚本全在 `robocasa-training/scripts/`(vendoring 进本仓): `launch_finetune_n17.py`(改 GR00T_ROOT 指本仓 + use_relative_action=False)、`train_n17.sh`(uv→conda gr00t-n17)、`watchdog_gr00t.sh`(eval 调 eval_gr00t_n17.py 解析 success_rate)、`robocasa_config_n17.py`(新写: panda_omron 键 + horizon40 + 全 action rep=ABSOLUTE 因动作已是 delta)、`serve_gr00t_n17.py`+`eval_gr00t_n17.py`(新写: N1.7 Gr00tPolicy 解包 (action,info),复用父仓 _gr00t_eval_client.py)。
- `train_n17.sh` 已 2-step smoke 跑通(warm-start→数据→前向反向→存 ckpt,loss=1.16)。**两个本机坑(默认已修)**: ① launcher `torch.compile(action_head)` 在 torch2.7.1 inductor 炸(`'aten' has no attribute 'map'`)→ `COMPILE_ACTION_HEAD_DISABLE=1`。② RoboCasa **3 相机**(LeIsaac 2)+ 桌面 gnome 占 0.4GB → micro-batch=2 OOM,small24 默认 **micro-batch=1**(global4/accum4),~4.3s/step。
- 训练数据: `.../OpenCabinet/20250813/**lerobot_old**`(v2.1 带 modality.json;新 `lerobot/` 无 modality.json 不能训)。HF_HUB_OFFLINE=1 会让 transformers4.57 的 Qwen3VL tokenizer mistral-regex 检查报错,训练别设 offline。
- **还缺**: N1.7 eval closed-loop 未用真 ckpt 实跑(serve/orchestrator import 通过);第一个 ckpt 出来先开环 replay 探针验 obs 键对齐再信 SR。

**实跑经验(2026-05-29 watchdog 8000-step run)**:
- **【最重要】随机崩根治 = `DATALOADER_NUM_WORKERS=0` + `PIPELINE_OVERLAP_DISABLE=1` 两个都要**:N1.7 在本 4090 每 ~50-1000 步随机崩,报 corrupted-iterator `ValueError`(`clip_grad_norm_→named_parameters`:`too many values to unpack (expected 0)` / `not enough values to unpack (expected 92, got 2)`,"expected N" 数字乱变 = **async CUDA 内存损坏**非代码 bug)+ 偶发 SIGSEGV。**两个独立成因**:① forked dataloader worker 损坏 CUDA 状态 → num_workers=0;② launcher 的 `non_blocking=True` H2D patch(`_patch_cuda_pipeline_overlap`)→ `PIPELINE_OVERLAP_DISABLE=1`。**归因证据链**:nw4+overlap 崩@444/1057;nw0+overlap 仍崩@1331(worker 不是全部);nw0+**overlap off** 干净过 1750 —— **non_blocking off 是决定性那半**。两者 train_n17.sh 默认都已设。num_workers=0 速度几乎不变(~1.5it/s,数据集小+shard 缓存)。**这是"watchdog 反复崩磨进度甚至零进展死循环" vs "训练顺畅跑"的分水岭。**
- watchdog 崩后**残留训练进程(及其 child)会占满 GPU 22GB → 下一轮启动 CUDA-init 段错误级联失败卡住**。kill 时要按进程组/pkill launch_finetune_n17 + 确认 GPU 排空再启动。`pkill ... 2>/dev/null` 在 harness errexit 下返回非零会中断后续命令,清理脚本要 `|| true`。
- clean resume 本身没问题(HF `resume_from_checkpoint=True` 正确从 ckpt-1000 续、loss 接续);崩的是 num_workers fork,不是 resume。
- **崩溃不可根治(native CUDA 损坏)**:即便 num_workers=0 + non_blocking off + codex 的 stable-clip 补丁(缓存 param list 不再每步走 named_modules,patch=1 确认加载),**仍崩** —— 同样 `expected 92` ValueError **+ 段错误核心转储(SIGSEGV)**。崩点随机(444/1057/1331/1978/2120),"expected N" 变化 = native 内存损坏在 Python 层冒烟。**根因:24GB 4090 低于 N1.7 官方 40GB+ finetune 推荐 + 消费卡无 ECC(静默显存错误)。codex(gpt-5.5)独立同结论。** 配置只降频不根治。**根治解 = 云 GPU(40GB+);本机只能靠 watchdog grind。**
- **watchdog 两个设计修正(用户指出)**:① **重试用"连续无进展 cycle 数"(MAX_STALL),不是全局总数** —— ckpt 一推进就清零,只要在前进就无限 grind,只有连续 N 次(默认 40)无进展才放弃(真卡死)。全局 MAX_RETRIES 只当远端 backstop。② **KEEP_MULTIPLE 必须 = EVAL_STEPS_MULTIPLE(都 1000)** —— 否则 eval sweep 点(1000 倍数)若非 KEEP_MULTIPLE 倍数会被当临时 ckpt 在 watchdog cycle 边界 eval 到它之前剪掉 → 漏 SR 点。SAVE_STEPS=100(密存崩溃恢复)与 KEEP_MULTIPLE=1000(永久 sweep 点)分工。
- **GUI eval 节奏**:不必每个正式 ckpt 都 GUI eval(headless SR 已每 1000 自动评)。GUI 只在:① 早期一次验管线+动作(ckpt-1000 已做,抓出 obs/action 格式 bug);② SR 首次 >0 时;③ 最终最佳 ckpt。欠拟合阶段(SR 0)GUI 看不出新东西且打断脆弱的 run。

**【最终定论 2026-05-29】本机训练不可行——native 内存损坏,卡死在 ckpt-2100,到不了 8000**:
- 多轮重启(run4-10)证据:崩溃签名**每次都不同且诡异** —— `SIGSEGV 核心转储` / `UnboundLocalError: local 'b'` / `TypeError: bad argument type for built-in operation` / `ValueError: ...unpack (expected N)` 且 N 乱跳(0/92)。这是**运行时进程内存被随机 bit-flip 损坏**的指纹,不是代码 bug。
- **关键判别**:从 **eval 验证过的 ckpt-2000**(之前跑过 10 episode 正常)续训,**加载阶段也崩**同样的 UnboundLocalError → **排除 ckpt 损坏、排除 resume 逻辑,坐实运行时损坏**。
- 配置全调尽无效(compile off / num_workers=0 / non_blocking off / codex stable-clip / settle / stall 逻辑)——只降频(首崩 444→~2000)不根治;最后连模型加载/resume 都过不去。
- **【根因更正】不是硬件,是 env 依赖栈不自洽**:用户关键洞察 —— 下载的 **N1.5 checkpoint 在同一台机/同卡上推理完全不报错**(robocasa_gr00t env, torch2.5.1+cu124)。同硬件 N1.5 干净、N1.7 崩 → **排除硬件,是 N1.7 的 gr00t-n17 env 问题**。该 env 是**从 fastwam 克隆 + pip 补装**的,CUDA/库栈和 torch2.7.1+cu128 不自洽(cuDNN/CUDA runtime 等隐性不匹配 → 进程级随机内存损坏,表现为 9 种乱码错跨 pydantic/named_parameters/SIGSEGV/UnboundLocal/TypeError 等)。**flash-attn 2.8.3→2.7.4.post1 单独修不够**(仍崩 pydantic FieldInfo)→ 是整套 env 栈的问题。**正解:用 Isaac-GR00T 自带 uv.lock `uv sync` 建干净 .venv**(= isaaclab 跑通 N1.7 的精确依赖集:torch2.7.1+cu128/flash_attn2.7.4.post1/cudnn90701),而非克隆补丁的 conda env。
- **未试的本机杠杆**:flash-attn 降到 2.7.4.post1 或禁用(eager attn,但 3B 无 flash-attn 省显存→可能 OOM);memtest86 验 RAM。
- **【已解决 2026-05-29】根因 = 坏 env,uv venv 修复,本机 4090 能训,不用上云**:
  - `uv sync` 在 Isaac-GR00T submodule(自带 uv.lock,295KB)建干净 `.venv` —— 精确依赖集 = isaaclab 跑通 N1.7 的那套(torch2.7.1+cu128 / flash_attn2.7.4.post1 / **cudnn 90701**)。
  - **决定性验证**:用 `.venv/bin/python` 从 ckpt-2100 续训,**干净训过 2120 崩点一路到 ckpt-2600 零乱码错**(坏 conda env 在此反复崩)。坐实是 env 依赖栈(cudnn 等)不匹配,**非硬件、非 4090 显存不足**。
  - **train_n17.sh 已改**:优先用 `$GR00T_ROOT/.venv/bin/python`(uv venv),conda 作后备。**eval_gr00t_n17.py 已改**:server 也用 `.venv`(`_server_cmd`),client 仍 robocasa conda env。
  - num_workers=0 / non_blocking off / stable-clip 只是**降频红鲱鱼**(把首崩从 444 推到 2000),真正的 cure 是干净 env。**教训:N1.7 env 必须 `uv sync` from uv.lock,绝不要克隆别的 env + pip 补丁(CUDA/cudnn/flash-attn ABI 不匹配 → 进程随机内存损坏,伪装成硬件故障)。**
  - 诊断法:同卡跑另一个模型(N1.5)若干净 → 是 env 不是硬件。
- **micro-batch=1 + grad-ckpt 在 24GB 完全够**,不需要更大卡。watchdog 的 MAX_STALL=40(连续无进展才停)+ SAVE_STEPS=100 + KEEP_MULTIPLE=EVAL_STEPS_MULTIPLE=1000(sweep 点不被剪)。
- **【训练完成 2026-05-30】run12 干净训完 8000 步**(cycle1 一口气 2600→7300、cycle2 7300→8000,exit=0)。ckpt-8000 在盘。**uv venv 修复彻底确认**。
- **两个剪枝回调冲突**:launcher 的 LossDrivenPruneCallback(LOSS_PRUNE_TOP_K=2)会删掉 KEEP_MULTIPLE 要保的 1000 倍数 sweep ckpt → SR 曲线丢了 3000-6000。已设 `LOSS_PRUNE_DISABLE=1` 默认修复(未来 run 保全 sweep 点)。
- **SR 结果**:能测到的都是 **0**(1000/2000 各 10ep 干净=0;8000 仅 1ep 完成=0;7000/8000 自动 eval NaN)。模型够到柜子但开不了。**但 eval 现在不可靠** —— robocasa env(client,sim)session 跑几小时后**也开始出乱码错**(yaml IndexError/ScalarToken、gym dict-namespace、'choice' UnboundLocal),而 session 开始时它干净(N1.5 70%、早先手动 1000/1800 评跑通)。**= 机器在持续重载下出现内存不稳,波及 robocasa env**(gr00t env 是 uv 修好的;这是另一层 session-degradation,疑边际硬件/热)。**拿可靠 ckpt-8000 SR 需重启机器清状态后再评**。模型已训好存盘,随时可评。

**【最终结果 2026-05-30,重启后干净 eval】ckpt-8000 SR = 40%(2/5),模型成功学会开柜子!** EP2 ✓(389步)、EP4 ✓(162步快速成功);EP0/1/3 ✗ 都卡 400 步上限。用户看 GUI 确认:失败多是"开了一边半、时间不够"(双门柜没在 400 步限内开完)→ **不是能力问题,是 episode 时限不够**。**提 SR 杠杆:加大 --max-steps 给足时间 + 多 episode 拿稳健数字**。**【已验证关键】eval episode 时限是 SR 的决定性因素:demo 真实长度 median 412/p90 533/max 664 步(20fps),而旧 eval 只给 400 步(连 demo 中位数都不到)→ 严重低估。给 1200 步(60s)重评 ckpt-8000:EP0✓214步、EP2✓**1153步**(400 步下必失败!)、EP1✗(robocasa 崩非模型)→ **SR 0.67(2/3),真实跑起来 2/2 全成功**。即 **OpenCabinet 任务需 ~400-1150 步,eval --max-steps 应给 ≥800-1200**。robocasa env 乱码崩(yaml/'k'/'choice'/ScalarToken)跑久了会偶发复现(重启暂清,载多了又来)—— sim 端 flakiness,跳过崩的 episode 即可。

**【更正:不是硬件,是进程串扰 2026-05-30】** 一度怀疑物理内存 bit-flip(干净进程 import robocasa 段错误/unknown opcode/各种乱码错)。**对照实验证伪**:N1.5-only GUI benchmark 干净跑 → viewer 开、EP 全成功、import_crash=0、SR 100%(4/4 起)。**真因 = 多个 eval server 残留 + GPU/CUDA 状态串扰**:我反复 kill 不干净(pkill 自匹配 exit1 触发级联取消 + 孤儿 server 占 GPU),新进程在混乱 CUDA 状态下启动就崩出各种乱码错。**一旦干净单跑(runner `_wait_gpu_clean` 杀孤儿+等显存排空),robocasa env 稳定。不用换内存、不用上云。** 教训:① pkill -f 会匹配自身命令行 → 用 per-PID kill 或 grep 排除自身;② benchmark 策略间必须 _wait_gpu_clean 彻底清孤儿 server。

**benchmark 资产已建好(可移植,健康机器直接用)**:`robocasa-training/benchmark/`:`policies.tsv`(TSV 配 N1.7 自训 / 下载的 N1.5 / 下载的 pi0.5,ckpt 已填对)、`run_benchmark.py`(20轮/策略 + 整次eval重试 RUNNER_RETRIES + 策略间 _wait_gpu_clean + 输出 leaderboard.md 按 SR 排名)、`_gr00t_eval_client.py` 加了 EVAL_EP_RETRIES episode 级重试。pi0.5 ckpt: `~/.cache/robocasa/checkpoints/pi05_pretrain_human300/multitask_learning/75000`。换健康机器 `python benchmark/run_benchmark.py` 即出 Leaderboard。重启确实清掉了 session 累积的内存不稳(robocasa env 重新干净,5 episode 无崩)。**整链闭环成功:克隆 env 坏栈(伪装硬件故障)→ uv.lock 重建 env → 本机 4090 训完 8000 → 40% SR。对照 GR00T-N1.5 multitask 70%(本机单任务 N1.7 finetune 40%,仅 5 ep)。**
- **watchdog 默认 GLOBAL_BATCH 必须=4(micro-batch1)**:之前 watchdog 头部写了 `GLOBAL_BATCH:-8` 会覆盖 train_n17.sh small24 的 `:-4` → micro-batch2 → 第一步就 OOM。已改 watchdog 默认 4。教训:watchdog 别 export 盖过 train 脚本的 profile 默认。
- **micro-batch=1 实测 ~1.48 it/s(~0.67s/step,比估的 4.3s 快很多)**,GPU 稳定 23.4GB(贴边但不崩),8000 step ≈ 1.5h。
- **step-444 一次性随机崩**:`clip_grad_norm_→named_parameters` 报 `ValueError: too many values to unpack (expected 0)` —— 是 4090 上 N1.x 的随机 CUDA 故障(非确定性,cycle2 重训顺利越过 444 到 500 存档)。**watchdog auto-resume 正是为此设计**。唯一坑:cycle1 崩在第一个 save(500)之前 → 没存下 → 从头重训;靠 cycle2 没复现才过关。若随机崩频繁且总在 500 前,需把 SAVE_STEPS 降到 ~250 保证崩前已存。
- **watchdog eval 时机**:只在 cycle 边界(crash-resume 或 MAX_STEPS 完成)跑 eval,**clean run 中途不 eval**。所以 SR 曲线靠"崩溃-resume"机会触发 + 末尾;4090 随机崩频繁时自然会有中途点。要严格定期 eval 需改 watchdog 主动中断训练。

微调方式（回答"是不是 8bit"）：**bf16 选择性全参**——冻 Cosmos VLM(视觉+LLM)，全参训 DiT action head+projector(~560M)，adafactor+grad-ckpt 压 4090(实测 22.4GB/3.5s/step)。**非 8bit、非 LoRA**（paged_adamw_8bit 试过 bnb resume 崩，弃用）。

**【eval 崩溃真正根因 2026-05-30 终定 —— 推翻"进程串扰/硬件"两次误判】**
- 症状:client(robocasa env)在 `env.reset()` 崩在纯 Python yaml/xml 热路径,错乱码:`unknown opcode
  215`/`scanner.py line -1`/`'method' object is not iterable`/`NoneType is not iterable`。**渐进+概率性**
  (pi0.5 前2轮好后2轮崩)。
- **真因(Opus+codex gpt-5.5 一致)**:client 是唯一的 **Python 3.11**(PEP659 自适应特化解释器开启;两个
  server 都 py3.10 无此),单进程跑完全部 N episode、复用 env 跨 reset。robocasa 每次 reset 重建整个 MJCF
  (海量 C 层 ElementTree/PyYAML churn),某 C 扩展(最可能 **MuJoCo GL/EGL**)堆越界,损坏在长寿进程累积,
  PEP659 把它现形成诡异字节码错。N1.7 慢→进程活久→reset 多→比 N1.5 更易崩(N1.5"可靠"是采样运气)。
  `PYTHONDONTWRITEBYTECODE=1` 治不了(quickening 在内存与 .pyc 无关);py3.11 无关特化开关。
- **修复(已验证彻底)**:`scripts/_gr00t_eval_client.py` + `scripts/_pi05_eval_client.py` 重构成
  **driver/worker 双模式** —— driver 绝不 import robocasa/mujoco(永远干净),每 episode spawn 全新 worker
  子进程(`--episode N` + `--seed N` + `--episode-result-path`),worker 跑完 1 轮即退、结果写文件;worker
  崩(段错/opcode/超时)→ driver 记 sim-DNF(steps=0)续跑。driver 加 `EP_TIMEOUT_S=900s` + `os.killpg`
  进程组杀(根治挂 1.5h 的僵尸 client)。**崩溃从"整轮 0/N 垃圾"变"崩一轮只赔一轮",90+ episode 实测无整轮崩。**
- 之前两次误判作废:① "硬件 bit-flip" ② "孤儿 server + GPU/CUDA 串扰"(`_wait_gpu_clean`)。真因是 py3.11
  client 单进程内 native 堆累积损坏,与 GPU/server 无关。但 `_wait_gpu_clean` 仍有用(防策略间显存占用)。

**【公平化 eval 2026-05-30 —— seed 锁场景】**
- 旧:client `env.reset()` 不传 seed → 每轮随机,各 policy 面对不同厨房 = 不公平。
- 机制:`gym_wrapper.reset(seed)` → `env.rng=default_rng(seed)` → `kitchen.py:595 rng.choice
  (layout_and_style_ids)` 选 layout/style(OpenCabinet target 池约 10 种 (1,1)..(10,10))+ rng 驱动
  柜子摆位/门朝向/机器人位姿。**实测 seed=N 完全锁定整场景且可复现,不同 seed 不同场景**(scene-hash 验证)。
- 修复:两 client 加 `--seed`;driver 默认 episode N 传 `base+N`(SEED_BASE 默认 0)→ 所有 policy 按 seed
  0..N-1 面对**完全相同可复现**场景序列 = 公平。run_benchmark 默认公平,无需额外配置。
- 时间:robocasa **20Hz**,`max_steps=1200`=**60s 仿真**;demo median 412步(21s)/max 664步(33s),1200≈最长
  demo 1.8 倍,够。worker 另有 `EP_TIMEOUT_S=900s` 墙钟兜底。

**【权威榜单 2026-05-30,30 轮公平 GUI,seed 0-29,1200步;SR 排除 sim-DNF 后有效轮(make_leaderboard.py 重算)】**
| 名次 | 策略 | SR | 成功/有效 | 成功均步 | sim-DNF |
|---|---|---|---|---|---|
| 1 | GR00T-N1.5-multitask(下载,多任务) | **64.0%** | 16/25 | 410 | 5 |
| 2 | GR00T-N1.7-OpenCabinet(本机自训 8000) | **36.0%** | 9/25 | 615 | 5 |
| 3 | pi0.5-pretrain-human300(下载) | **17.4%** | 4/23 | 495 | 7 |
- 诚实结论:本机自训单任务 N1.7 **36%** > pi0.5 **17%**,但 < N1.5 多任务 **64%**。N1.5 64% 对上早先记忆的
  60-70%(说明那是真实水平非乐观噪声)。N1.7 成功均步 615 远高于 N1.5 410 → 能开但慢、效率低。
- ⚠️ 注意 `run_benchmark.py` 自写的 success_rate 把 sim-DNF 也算进分母(pi0.5 显示 13.3%=4/30)→ **以
  `make_leaderboard.py` 重算为准**(排除 steps==0 崩溃轮)。两个数字差异即 sim-DNF 数。
- **坑(给后人)**:run_benchmark 顺序 N1.7→N1.5→pi0.5;**别在 pi0.5 还在跑时看到中途 leaderboard 就以为完成**
  (我犯过:误杀在跑的 pi0.5 第8轮,`--only pi0.5` 补跑)。判完成只认日志末尾 `[bench] leaderboard saved`
  + 该 policy json 满 30 轮。批量延迟 tool 输出也会让你读到半截数据误报中间 SR,以最终 json 为准。

配套文档 `robocasa-training/doc/training_gr00t_n17.html`。相关：[[project-act-opencabinet-humanonly-result]]（ACT 0% vs GR00T 70% 对照，N1.7 是正确路线的升级）、[[feedback-train-with-watcher]]（边训边 eval watchdog）。
