---
name: project_gr1_tabletop_preview
description: RoboCasa GR1 Tabletop 下载+预览 notebook 的状态(独立于 OpenCabinet 主线)
metadata: 
  node_type: memory
  type: project
  originSessionId: 572208be-8009-4c07-ace1-6e46a48912cd
---

# RoboCasa GR1 Tabletop — 下载+预览 notebook

**和 OpenCabinet 主线无关的独立预览页。** GR-1 人形(Fourier 手+腰)做 24 桌面 PnP,NVIDIA GR00T N1 论文(2503.14734)官方 benchmark。用户要求:notebook 下载+预览现成场景和 checkpoint,**先下载安装、先不跑**(4090 在跑 MimicGen 混训,GPU 占满,见 [[project_n17_16k_run_state]])。

## 已建文件
- `RoboCasa-Tabletop.ipynb`(18 cells,白底/中英,仿 RoboCasa.ipynb 薄壳模式)
- `scripts/install_robocasa_gr1_env.sh` — 独立 env `robocasa_gr1`(py3.10),clone Isaac-GR00T(main,到 `dependencies/Isaac-GR00T-gr1`)+robosuite master+GR1 repo,下 tabletop 资产。**flash-attn 默认跳过**(BUILD_FLASH_ATTN=1 才装,§5 eval 前再补,避免编译干扰训练)
- `scripts/robocasa_gr1_demo.py` — 驱动:status/list/clip/download-ckpt(无 GPU,已验证)+ render/eval(需 GPU,按上游命令构造)

## 关键事实
- **必须独立 env**:GR1 repo 也装名为 `robocasa` 的包(v0.2.0,pin numpy1.26.4/mujoco3.2.6),和厨房 robocasa 包冲突。
- **checkpoint 有现成的**:⭐`youliangtan/gr00t-n1.5-robocasa-tabletop-posttrain`(7.6GB,GR00T 论文作者 You Liang Tan,N1.5 post-train 24任务,README 报~47%,无 model card 但结构标准可推理)。备选:`nvidia/GR00T-N1.5-3B`基座(zero-shot42%)、`karthikpythireddi93/gr00t-n16-gr1-tabletop-sft`(N1.6社区)。
- **最优预览=数据集 clip**:`nvidia/PhysicalAI-Robotics-GR00T-Teleop-Sim`(55.4GB:HDF5 14G+LeRobot 41.5G,⚠️CC-BY-NC禁商用)的 LeRobot 子集每条 demo 存 ego_view mp4(~0.1-0.8MB),`hf_hub_download` 单文件直接内嵌,**无需 env/GPU**。路径 `LeRobot/gr1_unified.<NAME>/videos/chunk-000/observation.images.ego_view/episode_NNNNNN.mp4`。
- **eval 两套配方(别混)**:**A. N1.5(配 youliangtan ckpt)** server `inference_service.py --data-config fourier_gr1_arms_waist --embodiment-tag gr1`(原 robocasa fork README;驱动 `eval` 走这套)。**B. N1.7 权威**(Isaac-GR00T `examples/robocasa-gr1-tabletop-tasks/README.md`):**uv venv**(`setup_RoboCasaGR1TabletopTasks.sh`,非 conda)+ `ROBOCASA_GR1_TABLETOP` embodiment;server=`gr00t/eval/run_gr00t_server.py --model-path <ck> --embodiment-tag ROBOCASA_GR1_TABLETOP --use-sim-policy-wrapper`;client=`gr00t/eval/rollout_policy.py --n-episodes 10 --policy-client-port 5555 --max-episode-steps 720 --n-action-steps 8 --n-envs 5 --env-name <id>`。配方 B 需 N1.7 `ROBOCASA_GR1_TABLETOP` finetune(HF 暂无现成;**官方 benchmark 均值 44.5%/24 任务**就是它,N1.7 退役了 N1.5 的 gr1_unified tag)。youliangtan 是 N1.5→用 A。
- 命名:NAME→env id `gr1_unified/<NAME>_GR1ArmsAndWaistFourierHands_Env`;lerobot 文件夹 `gr1_unified.<NAME>`。

## 安装踩坑链(已解决,写进 install_robocasa_gr1_env.sh)
**gr00t 是 uv-native 包**:pyproject 声明部署重包为硬依赖但预编译 wheel 走 `[tool.uv.sources]`,**pip 不认** → `pip install -e .[base]` 对这些包逐个源码编译失败。踩坑顺序:① 一次性大 resolve **段错误**;② flash-attn 源码编译失败;③ tensorrt-cu12 源码编译失败。**最终解法**:(a) torch==2.5.1+cu124 先装(gr00t 那套 torchcodec==0.4.0 配 2.5,非 pyproject 写的 2.7.1);(b) flash-attn 用预编译 wheel `flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310` 装(不编译不占 GPU);(c) 从 pyproject 提取运行时依赖、grep -v 掉 `tensorrt/deepspeed/onnx/triton/flash-attn/torch` 装上;(d) `pip install --no-deps -e gr00t` 装本体。**有意跳过 tensorrt-cu12/deepspeed/onnx/triton**(纯 ONNX/TRT 部署+多卡训练,预览/eval 不需要)。

## 状态(2026-05-31 已完成主体)
- ✅ **env robocasa_gr1 装好(已验证 ALL_OK)**:`import gr00t,robocasa,robocasa.environments.tabletop,flash_attn,robosuite,torch` 全过。torch 2.5.1+cu124 / flash-attn 2.7.4.post1 / robosuite 1.5.1 / robocasa(gr1)0.2.0(mujoco 3.2.6/numpy 1.26.4)。
- ✅ youliangtan ckpt 7.6G → `~/.cache/robocasa_gr1/checkpoints/gr00t-n1.5-tabletop-posttrain`
- ✅ 6 个 core clip → `~/.cache/robocasa_gr1/preview_clips/`
- ✅ **tabletop 3D 资产已下好**(status: sketchfab:True lightwheel:True,objects 目录 7.8G)。objaverse 子项(utexas.box.com zip)下载/解压失败,但被 `download_and_extract_zip` 的 `try/except` 吞掉、不阻断,textures/fixtures/generative_textures/lightwheel/sketchfab 都成功。(注:install 脚本第7段只检查 sketchfab+lightwheel 存在即跳过,符合现状。)
- driver status 字段 `eval_gr00t_robocasa.py=False`:§5 实际用上游 `gr00t/eval/rollout_policy.py`(配方B,见上),driver 里那个路径名是旧的,§5 跑前要对齐(§5 需 N1.7 ROBOCASA_GR1_TABLETOP ckpt + GPU,非 youliangtan)。
- **render/§5 eval 没跑(GPU 让给 MimicGen 训练)**。
