---
name: project_n17_16k_run_state
description: "N1.7 OpenCabinet 训练/sweep 已完成,下一步 MimicGen 混训 + HF 发布的状态续接(compact 用)"
metadata: 
  node_type: memory
  type: project
  originSessionId: 572208be-8009-4c07-ace1-6e46a48912cd
---

# N1.7 OpenCabinet — 训练+sweep 完成,下一步 MimicGen 混训 + HF 发布

**状态(2026-05-30):** 16k 训练 + 30轮精扫 **已完成**。最终榜单已出。当前在做 ①MimicGen 方案 doc ②HF 发布(视频转码)③准备 MimicGen 混训。compact 回来后继续训练。

## 最终榜单(已写 benchmark/leaderboard.md)
| # | 策略 | SR(30轮 seed-locked) | 成功/有效 | 成功均步 |
|---|---|---|---|---|
| 1 | N1.5-multitask(下载) | **64.0%** | 16/25 | 410 |
| 2 | N1.7-OpenCabinet 自训 peak=ckpt-11000 | **50.0%** | 13/26 | 627 |
| 3 | pi0.5-pretrain-human300(下载) | **17.4%** | 4/23 | 495 |

- N1.7 sweep: 11000=**50.0%**(峰) / 13000=42.3% / 15000=44.0% / 16000粗测=10%(崩)。
- **过拟合实锤**:loss 8k 后收敛但 SR 11k 见顶后回落。500 human demo 数据瓶颈。
- N1.7 方差大:ckpt-11000 两批30轮 = 50% vs 27.6%,真实区间 ~30-50%,置信区间宽 ±10-15pt。
- **红线**:绝不重跑 N1.5/pi0.5;绝不碰 `benchmark/results/authoritative_30round/`(md5 N1.5=0c0d073465368faec1b007b940babb54 pi0.5=18388112c67ec2c962b8e9c389647753)。

## MimicGen 数据清单(已核实 info.json,纠正旧文档"8644 ep"错误)
| 数据集 | 路径 | ep | frames | modality.json | 可直接训 |
|---|---|---|---|---|---|
| Human(当前用) | `~/.cache/robocasa/datasets/v1.0/target/atomic/OpenCabinet/20250813/lerobot_old` | 500 | 183847 | YES | ✓在用 |
| MG 已转换 | `~/.cache/robocasa/datasets/mimicgen_opencab_500/lerobot_old` | 500 | 164848 | YES | ✓即可 |
| MG 已转换v3.0 | `~/.cache/robocasa/datasets/mimicgen_opencab_500/lerobot` | 500 | 164848 | — | 需核对 |
| MG 预训练池 | `~/.cache/robocasa/datasets/v1.0/pretrain/atomic/OpenCabinet/20250819/mg/demo/2025-08-20-21-54-43/lerobot` | **3000** | 761958 | **NO** | 补 modality.json 后可用 |

- 可用 MG 总量 = 500(即用) + 3000(补 modality.json)= 最多 3500 ep。
- `train_n17.sh` 只吃单个 `DATASET_DIR` 且强制要 `meta/modality.json` → 混训需先合并成一个 lerobot 集。

## MimicGen 混训方案(已写 doc/mimicgen_data_strategy.html,白底/中英/SVG)
1. 复制 modality.json 解锁 3000-ep 池。
2. 写 `scripts/merge_lerobot_mg_human.py`:均匀随机抽 ~1500 MG + 全部 500 human → 合并集(重写 info.json/episodes.jsonl,连续重编 index,拷贝/软链 parquet+videos),输出 `~/.cache/robocasa/datasets/opencab_mix_mg1500_human500/lerobot_old`。
3. 训练:`DATASET_DIR=<合并集> MAX_STEPS=40000 SAVE_STEPS=1000 KEEP_MULTIPLE=2000 bash scripts/watchdog_gr00t.sh`(~40k step≈7.4h,实际8-9h)。
4. 训完用 `benchmark/sweep_n17.py STEPS="..."` 新 sweep 找新峰值(预期后移到 25k-40k)。
5. 对比新峰值 vs 50% vs N1.5 64%,更新 leaderboard.md。
- 预期:乐观 55-65% 追平 N1.5;保守 +5-10pt。**不保证超 64%**(N1.5 是 120k步多任务大模型)。

## HF 发布任务(✅ 素材就绪,待 push)
- 用户要求:① 提交一版到 HF ② 两录屏 webm 转 mp4 嵌 README 直接预览 ③ README 写"下载 OpenCabinet 做 eval 方法"方便别人下 ckpt 运行。
- **✅ 视频转码完成**:`robocasa-training/hf_release/videos/opencabinet-demo-{1,2}.mp4`(h264 1704x992,6.2s/8.0s,813KB/410KB,完整无损)。
  - **坑1**:conda ffmpeg(`/home/david/miniconda3/bin/ffmpeg`)缺 libx264.so.138 用不了;**必须用系统 `/usr/bin/ffmpeg`**(6.1.1)。
  - **坑2(关键)**:源 webm 是 **VP8 分辨率 1705×993 奇数宽**,libx264 报"width not divisible by 2"。**修法 `-vf "scale=trunc(iw/2)*2:trunc(ih/2)*2"`**。之前误判"VP9损坏只救出9s"是错的——视频完好,纯奇数宽度。
  - 源 webm:`/home/david/视频/录屏/GR00T-N1.7-RoboCasa-OpenCabinet-{1,2}.webm`。
- **✅ README 已写** `robocasa-training/hf_release/README.md`:HF model card,嵌 `<video src="videos/...mp4">`,含最终榜单+sweep曲线+训练配方+"Run eval yourself"完整章节(装环境/下assets `python -m robocasa.scripts.download_kitchen_assets`/下ckpt/seed-locked跑eval;eval纯sim不需demo数据)。
- **待办**:① 确认 HF repo 名+账号 ② 是否传权重(peak ckpt-11000,`checkpoints/gr00t_n17_opencabinet/checkpoint-11000`)还是仅 card+video ③ 实际 push(skill hf-publish-model)。**push 前必须问用户,不自动 push。**

## 关键工具纪律(本会话反复踩坑)
- 单卡 4090:训练/eval 不能共存。启动 eval 三条件:ckpt 全 + watchdog 退 + GPU<3000MiB。
- 杀进程用显式 PID;**绝不 pkill -f 含命令自身字符串**(自杀 shell)。
- pkill/kill 退 1 → 级联取消并行调用 → 单工具调用 + 命令结尾 `exit 0`。
- stdout 批量延迟污染 → 写时间戳文件再读;**读 Bash 实际回显的路径,别猜时间戳**。


## ⚠️ HF 上传阻塞(2026-05-30 23:18)— 网络问题,待用户决策
- **repo 已传**:README.md + videos/(2 mp4)+ .gitattributes,**网页已可看 card+视频**。
- **❌ checkpoint-11000 权重(11.7GB)传不上去**:upload-large-folder 卡在 pre-uploaded 3/5,实测网络出口 **0.00MB/s**(5秒0.01MB),进程活着但到 HF 存储端点零吞吐。
  - 排除:hf_transfer(关掉仍卡)、参数、并发。huggingface.co 主站可达(200/0.73s),小文件能传,**仅大文件上传链路断流**。
  - 判定:**机器到 HF 大文件上传端点的网络问题**(疑似墙/限速),非代码可解。已停所有上传进程(NONE_CLEAN),不再无限重试。
- **待用户决策**:① 配代理(HTTPS_PROXY)重试 ② 换网络/时段 ③ 接受仅 card+视频先发,权重之后补。
- 重试命令(配好代理后):cd robocasa-training/hf_release && /home/david/miniconda3/bin/hf upload-large-folder wsagi/GR00T-N1.7-RoboCasa-OpenCabinet . --repo-type=model --num-workers=2 --no-bars(.cache 断点续传)。
- 发布目录已备好:hf_release/checkpoint-11000/(软链,已排除训练态 optimizer/rng/scheduler)。
