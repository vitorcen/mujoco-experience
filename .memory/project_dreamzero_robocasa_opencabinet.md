---
name: project_dreamzero_robocasa_opencabinet
description: DreamZero(14B Wan WAM) LoRA 微调 RoboCasa OpenCabinet human500，AutoDL 上跑；脚手架复用 LeIsaac
metadata: 
  node_type: memory
  type: project
  originSessionId: 8f5534ca-f3c2-49d7-a247-bf54c9209e6f
---

**▶ RESUME HERE（2026-06-01 末态）：**
1. **✅ AutoDL 正式 15k 训练已 LIVE(2026-06-01 启动)**。无卡模式坑6已解(控制台切回有卡=RTX PRO 6000 96G)。`setsid bash train_dreamzero_robocasa_lora.sh` 跑起,log=`/root/autodl-tmp/train_robocasa_15k.log`,OUTPUT_DIR=`/root/autodl-tmp/dreamzero_robocasa_opencabinet_lora`。step0 已确认 GPU 54GB/100%、dynamics_loss 0.116/action_loss 0.196(与 smoke 0.636 同量级正常)。6.3s/step,save_steps=2500,save_total_limit=5。**中途每出 checkpoint→拉回本地 4090 NF4 eval 验 SR 提升**(用户要求):一键 `bash robocasa-training/scripts/dreamzero/pull_and_eval_ckpt.sh <STEP> [N_ep=3] [max_steps=1200]`(自动 scp 415MB→起 NF4 server→headless SR)。NF4 单 ep≈1-1.5h,先 3ep 取趋势。死了能续训(同 OUTPUT_DIR 调高 MAX_STEPS 自动 resume,见 [[feedback_prefer_resume_continue_training]])。OMP_NUM_THREADS libgomp 警告无害(torchrun 自设)。
2. **✅ 本地 GUI/sim eval 全跑通**(2026-06-01,steps=32,arm 转 75°/移 22cm)。见 [[project_dreamzero_robocasa_gui_eval]]。scipy heisenbug 是误判;真凶三 bug 已修(CPU prompt 编码防 OOM / POLICY_TIMEOUT_S / action dict 全名查找)。**本地 server 可能还挂在 :5702 占 ~11GB 显存**——训练/eval 前按 GPU PID 杀(`nvidia-smi --query-compute-apps`→`kill -9`)。
3. **未 commit**：`robocasa-training/scripts/dreamzero/`(convert/yaml/train/eval/serve+GUI 三 bug 修) + `scripts/_gr00t_eval_client.py`(加 POLICY_TIMEOUT_S env,默认 120 向后兼容) + `doc/training_dreamzero_opencabinet.html` + 本地 `dreamzero-repo` cotrain robocasa patch。待用户确认 commit。
4. 本机改动:dreamzero env scipy 1.17.1→1.15.3;推理脚本路径用 $HOME。

---

新实验线（2026-05-31 起）：在 AutoDL ≥80GB 单卡上对 **DreamZero**（World-Action Model，Wan2.1-I2V-14B-480P 视频扩散骨干 + UMT5-XXL，arXiv 2602.15922）做 **bf16 LoRA(r=4)** 微调，数据 = RoboCasa `OpenCabinet` human **500 demo**（`~/.cache/robocasa/datasets/v1.0/target/atomic/OpenCabinet/20250813/lerobot_old`，LeRobot v2.1，3 相机，state[16]/action[12]，fps20，h264，758MB）。目标是验证 WAM 范式在 RoboCasa sim 的首个数据点，success=SR>50%（对标 [[project_n17_16k_run_state]] 榜单 N1.5 64%/N1.7 50%/pi0.5 17.4%）。

**关键复用：脚手架已存在** `/home/david/work/isaaclab-experience/LeIsaac/scripts/`：
- `finetune/dreamzero/`：`train_dreamzero_bf16_lora.sh`(实战 SO-101) + `convert_leisaac_to_gear.sh` + `leisaac_relative.yaml` + `reencode_av1_to_h264.sh`
- `cloud/autodl/`：13 步 AutoDL 流程（无卡模式装环境 ¥0.1/h，只训练才切 GPU）；本机 scp 推权重/数据
- `inference/dreamzero/`：NF4 量化推理（4090 可跑推理，训不动）

**RoboCasa 比 LeIsaac 更契合 DreamZero（两个大坑天然不存在）：**
1. 3 相机原生对齐 base 的 num_views=3（LeIsaac 只 2 视角是 schema mismatch）
2. 视频已 h264（省掉 AV1→h264 重编码大坑）

**硬件红线：** 14B Wan+UMT5 训练需 ≥80GB 单卡（A100/H100 80G 或 Pro6000 96G），bf16 LoRA r=4 + ZeRO-2 offload；4090 24G 只能 NF4 推理。放弃 INT8/bnb（源码无 hooks）。LoRA rank 固定 r=4，绝不像 LLM 开 32（视频扩散 LoRA 过 r=16 崩 + 500demo 过拟合）。

**唯一设计风险点：** RoboCasa action 的 eef_pos/rot 可能已是 OSC delta，DreamZero `--relative-action-keys` 再标 relative 会双重差分。首跑只标 `base_motion` relative，eef 当绝对量+q99。维度切分从 dataset `meta/modality.json` 读出：state base_pos[0:3]/base_rot[3:7]/eef_pos[7:10]/eef_rot[10:14]/gripper[14:16]；action base_motion[0:4]/ctrl_mode[4:5]/eef_pos[5:8]/eef_rot[8:11]/gripper_close[11:12]。

**推理慢：** AR 视频扩散 ~3s/次(H100)，~25 次/episode ≈75s/ep。eval 必须 1200 step（400 cap 误判慢成功，见 [[feedback_watchdog_interleave_earlystop]]）。

文档：`robocasa-training/doc/training_dreamzero_opencabinet.html`。**红线同 [[project_n17_16k_run_state]]：绝不重跑 N1.5/pi0.5，绝不碰 authoritative_30round。**

**AutoDL 机器：** `ssh -p 32660 root@connect.westd.seetacloud.com`(密码见会话)。卡=RTX PRO 6000 Blackwell 96GB，数据盘 130G(扩容后)。Wan base 在 `/root/autodl-tmp/wan2.1-i2v-14b-480p`(77G,完整)，数据 `/root/autodl-tmp/opencabinet`，repo `/root/autodl-tmp/dreamzero-repo`，脚本 `/root/autodl-tmp/scripts_robocasa_dreamzero/`。本地三件套存 `robocasa-training/scripts/dreamzero/`。

**✅ SMOKE 通过(2026-05-31)：** 50步 train_loss 0.636，模型上卡 53GB(96G余量足)，6.3s/step，checkpoint-50 存成功(LoRA-only 415M=adapter_model.pt+model.safetensors新头)。拉回本地 `robocasa-training/checkpoints/dreamzero_robocasa_smoke/checkpoint-50`。

**打通过程踩的 4 个坑(都已修)：**
1. **env numpy 损坏**→见 [[reference_autodl_dreamzero_numpy_corruption]]。还降了 pandas 2.3.3→2.2.3、pyarrow 24→21(datasets需>=21)。
2. **task KeyError**：convert 的 `--task-key task` 错(parquet 无此列);build_tasks 还把 tasks.jsonl 写空。修=patch dataset `meta/modality.json` 的 `annotation.task.original_key`→`task_index`(loader 经 tasks.jsonl 数字索引解析文本) + 从本地恢复真实 tasks.jsonl/episodes.jsonl。**不能靠重跑 convert 修**(build_tasks 只把数字列转无用字符串且每次重写)。
3. **`Embodiment ID 13 not supported`**：`groot/vla/model/dreamzero/transform/dreamzero_cotrain.py` 的 collate 给每个 embodiment 硬编码文字 prompt 模板，只支持 AGIBOT/OXE_DROID/GR1/MECKA/XDOF/YAM。修=给 try+except 两条 if-elif 链各加 ROBOCASA_PANDA_OMRON 分支(3视角描述)。备份 `.orig_robocasa_bak`。
4. **僵尸进程骗人**：旧 torchrun crash 后 agent 不退，pkill 没杀净→一直在盯旧 log。教训：relaunch 前 `pkill -9 -f experiment.py;pkill -9 -f torchrun` 并确认 0 进程+log mtime 在涨+python 占 CPU 才信。smoke 用 num_workers=0(去 worker 僵尸+in-process traceback)。

**实测时间：6.3s/step → 15k步≈26h(compute-bound,14B forward,num_workers 帮不大)。** 用户决策：先验 eval 管线再训。

**✅ EVAL 管线已验通(2026-05-31)：** 本地 `robocasa-training/scripts/dreamzero/eval_dreamzero_robocasa.py`(用 dreamzero env)。NF4 Wan14B+robocasa LoRA 在 4090 跑通,峰值 **20.35GB/24GB**,推理 85s(NF4慢)。喂 episode0 真实观测→输出 action(24,12) finite。**机械臂会动**:eef_pos[-2.45,2.06]/eef_rot/gripper 都变化;base_motion 全0(OpenCabinet 底盘不动,合理)。复用 LeIsaac `inference/dreamzero/dreamzero_inference_loader.py`(build_dreamzero_inference_model)+`dreamzero_policy.py`(_build_fake_groot_sim_policy,传 embodiment_str=robocasa_panda_omron),自己构 robocasa batch(3视角 video.robot0_* + state 5键 + annotation.task)。**本地 dreamzero-repo 也打了 cotrain robocasa 分支 patch**(同 AutoDL)。NF4 server 参考 `isaaclab-experience/server/dreamzero_leisaac/server.py`。

**坑5：`save_total_limit` 硬断言 ≥5**(`groot/vla/experiment/base.py:619` "must be >= 5 for standarized evaluation")。改成 4 会启动即 AssertionError 崩。LoRA ckpt 仅 415MB/个,5个=2GB 不占地方,固定用 5。

**坑6：AutoDL 实例被切回无卡模式 → GPU 消失**。特征:`/usr/bin/nvidia-smi` 变 0 字节空文件 + `torch.cuda.is_available()=False` + 训练 SIGKILL(exit -9)。非 OOM/非磁盘。只能在 AutoDL 控制台重切有卡模式,shell 改不了。正式 15k 训练脚本已就绪(save_total_limit=5,num_workers=0,清了 smoke 产物),用户切回 GPU 后 `cd /root/autodl-tmp/scripts_robocasa_dreamzero && setsid bash train_dreamzero_robocasa_lora.sh` 即可。
