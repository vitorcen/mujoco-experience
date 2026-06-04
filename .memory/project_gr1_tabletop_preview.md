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

## 状态(2026-06-02 闭环 eval 已跑通 ✅)
- ✅ **env robocasa_gr1 装好**:torch 2.5.1+cu124 / flash-attn 2.7.4.post1 / robosuite 1.5.1 / robocasa(gr1)0.2.0(mujoco 3.2.6/numpy 1.26.4)。
- ✅ youliangtan ckpt 7.6G → `~/.cache/robocasa_gr1/checkpoints/gr00t-n1.5-tabletop-posttrain`,6 个 core clip,tabletop 3D 资产全下好(sketchfab+lightwheel)。

### 🎯 闭环 eval 跑通(配方A,youliangtan N1.5)— PnPCupToDrawerClose **SR 0.40 (4/10)**,~27s/ep
**根因发现**:`robocasa_gr1` env 是用 **N1.7** Isaac-GR00T main 建的(无 `gr00t.eval.simulation`/`eval.robot`,只有 uv 版 run_gr00t_server),但下载的 youliangtan 是 **N1.5** + 配方A 是 N1.5 → 版本错配,`robocasa_gr1_demo.py eval` 指向的脚本根本不存在。配方A 的 client `simulation_service.py` 上游 main 已删,只在隔壁 `isaaclab-experience/dependencies/Isaac-GR00T-N1.5` 还有。
**解法**:新建独立 env **`robocasa_gr1_n15`**(`scripts/install_robocasa_gr1_n15_env.sh`):本仓 **plain `dependencies/Isaac-GR00T`(N1.5 "GR00T N1.5 for RoboCasa" 树,有 `gr00t.eval.simulation`)+ gr1 robocasa fork + robosuite1.5.1**,共存一个 env。🔴 **不能塞进 `robocasa_gr00t`**(权威 kitchen benchmark env,gr1 fork 的 numpy1.26.4/mujoco3.2.6 pin 会污染)。
- 跑法:`python3 scripts/robocasa_gr1_demo.py eval <TASK> --n-episodes N --n-envs 1`。server=`dependencies/Isaac-GR00T/scripts/inference_service.py --server --model_path <ck> --embodiment_tag gr1 --data_config fourier_gr1_arms_waist`(tyro,**下划线** args),client=vendored `scripts/_gr1_simulation_service.py --client`。两进程同 env,ZMQ port 默认 5556,offscreen mp4 → `~/.cache/robocasa_gr1/eval_videos/`。
- **三个非显然坑(都已修在 demo/client 里)**:
  1. **gr1_unified 没注册** → client 必须 `import robocasa.utils.gym_utils`(注册 197 个 env);plain `simulation.py` 只 bare `import robocasa` 不触发。
  2. **`MUJOCO_GL=egl` 下 robocasa import ~25% flaky**(SIGSEGV 或伪 AttributeError `'str' object has no attribute ...Env`,都是 EGL GL-init 不稳)→ server **轮询 "Server is ready" 标记 + 崩溃重试**,client **import 阶段崩溃也重试**(崩在 import 前,重试幂等;server 不重载)。
  3. **plain `simulation.py` 是 kitchen 变体**`_create_single_env` 传 `split=` → gr1 env creator 不收 → TypeError。client 里 **monkeypatch `_sim.gym.make` 去掉 split**(gr1-tabletop 变体本就不传 split)。n_envs>1 用 spawn 会每 worker 重撞 EGL flaky,故默认 n_envs=1(SyncVectorEnv)。
- 对照:N1.7 官方 benchmark 此任务报 35%,N1.5 youliangtan 40% 在合理区间。
- ✅ 已 VLC 在 DISPLAY=:0 循环播放 `eval_videos/PnPCupToDrawer_ep01_success.mp4` 给用户看。

### 🖥️ 实时 on-screen GUI viewer(`gui` 子命令,2026-06-02 加)
用户要"边推理边看机器人动"(非回放 mp4)。robosuite/robocasa 底层是 MuJoCo → 用原生 `mujoco.viewer.launch_passive` 开窗口。
- 新文件 `scripts/_gr1_gui_client.py` + driver 加 `gui` 子命令。跑法:`DISPLAY=:0 python3 scripts/robocasa_gr1_demo.py gui <TASK> --n-episodes 3 --step-delay 0.03 --camera egoview`。
- **关键招**:复用 `SimulationInferenceClient.setup_environment`(n_envs=1→SyncVectorEnv 进程内,能拿到 inner env),从 `base.env.sim.model._model/.data._data` 取原生 mjModel/mjData → `launch_passive`。**MUJOCO_GL=egl 仍给 policy 喂离屏 camera obs;viewer 自己开 GLFW 窗口**(两个 GL context 共享同一物理状态,和 OpenCabinet GUI 一个套路)。
- **平滑动作**:patch `base.step`(MultiStepWrapper 每 sim 步调它)→ 每步 `viewer.sync()`+`step_delay`,否则 16 步跳一次。
- **相机坑(用户反馈)**:passive viewer 默认自由相机生成在场景外(看到房间外面)。**锁定到 model 相机**:`viewer.cam.type=mjCAMERA_FIXED; viewer.cam.fixedcamid=mj_name2id(...,'egoview')`。该 env 11 个相机,policy 用的是 **`egoview`(id 8,机器人头部第一人称=policy 视角)**;想看机器人全身用 `robot0_behindhead`/`robot0_agentview_center`/`robot0_frontview`。`--camera free` 保留轨道相机。
- GUI client 也吃那个 **flaky EGL import 崩**(~25%),cmd_gui 同样重试(崩在开窗前,幂等)。
- **配方B(N1.7 uv+ROBOCASA_GR1_TABLETOP)仍未跑**:需 N1.7 finetune 权重(HF 无现成),非 youliangtan。
