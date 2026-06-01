---
name: project_dreamzero_robocasa_gui_eval
description: DreamZero RoboCasa 本地 GUI/sim eval 管线全跑通(steps=32);三个叠加 bug 已修(CPU prompt 编码防 OOM/POLICY_TIMEOUT_S/action dict 全名查找)
metadata: 
  node_type: memory
  type: project
  originSessionId: 8f5534ca-f3c2-49d7-a247-bf54c9209e6f
---

DreamZero RoboCasa 本地可视化(GUI sim rollout)管线，属 [[project_dreamzero_robocasa_opencabinet]]。2026-05-31。

**架构(server-client, 两 conda env, ZMQ pickle):**
- server: `dreamzero` env, NF4 Wan14B+robocasa LoRA, ~20GB VRAM/4090。脚本 `robocasa-training/scripts/dreamzero/serve_dreamzero_robocasa.py`(pickle 协议 `{op:get_action/reset/shutdown}`,镜像 `scripts/_gr00t_inference_server.py`)。
- client: `robocasa` env, `scripts/_gr00t_eval_client.py --render`(passive MuJoCo viewer on DISPLAY=:0)。一键脚本 `robocasa-training/scripts/dreamzero/run_gui_demo.sh`。
- 复用 LeIsaac `inference/dreamzero/` 的 `build_dreamzero_inference_model`+`_build_fake_groot_sim_policy`(embodiment_str=robocasa_panda_omron)。NF4 server 模板参考 `isaaclab-experience/server/dreamzero_leisaac/server.py`(那个是 msgpack 协议给 LeIsaac,robocasa 用 pickle)。

**关键契约(实例化 robocasa/OpenCabinet env 确认):**
- env OBS 键→策略键映射:`state.base_position`→`base_pos`,`base_rotation`→`base_rot`,`end_effector_position_relative`→`eef_pos`,`end_effector_rotation_relative`→`eef_rot`,`gripper_qpos`同名;video.robot0_* 同名;`annotation.human.task_description`→`annotation.task`。
- env.step ACTION 键←策略输出:`base_motion`✓,`ctrl_mode`→`control_mode`,`eef_pos`→`end_effector_position`,`eef_rot`→`end_effector_rotation`,`gripper_close`✓。
- client `_add_time_dim` 给每个 obs 加 T=1;`env.step(action_dict)` 直接吃 dict;server 返回 `{action.X:(T,D)}`。

**✅ 2026-06-01 GUI 全跑通(steps=32,机械臂动了)。** 之前以为的 scipy heisenbug 是**误判**:scipy 1.17.1→1.15.3 降级**已修好**那个 `_docscrape.py int += None`(旧崩溃栈 `header = r.read()` 是 1.17.1 的 _docscrape,1.15.3 结构不同)。当时"后台必崩"其实是别的问题被 harness 后台 log race 掩盖了。现在 server import 链干净,10/10 不崩。

**真正挡路的三个叠加 bug(全已修,机械臂不动的真凶是 #3):**
1. **VRAM OOM**:loader 默认 encode_prompt 把 ~11GB UMT5-XXL 搬上 GPU(首次 cache-miss),叠加 DiT 8.7GB + MuJoCo render 3.8GB → 爆 24GB(只差 80MiB)。修=serve 脚本里 override `ah.encode_prompt` 为 **CPU 编码**(UMT5 本就在 CPU,只把 embedding 送 GPU),GPU 峰值 20→18GB,留 6GB 余量。一次性 ~75s CPU 编码后缓存。
2. **ZMQ recv 超时**:client `PolicyClient` 默认 `timeout_s=120`,但带 CPU 编码的首 chunk 要 136s → `Resource temporarily unavailable`(EAGAIN)崩 steps=0。修=`_gr00t_eval_client.py:68` 加 `POLICY_TIMEOUT_S` env 覆盖(默认仍 120,GR00T/pi0.5 基准零影响),GUI 跑用 `POLICY_TIMEOUT_S=600`。
3. **空动作字典(机械臂不动直接原因)**:`sim_policy.py:606 batch.act = unnormalized_action` 是**以全名 `action.eef_pos` 为键的 dict**;serve 原来只 `getattr(result_batch.act, "eef_pos")`(短名)→全 miss→返回 `{}`→client 无动作→steps=0。eval 脚本能动是因为它有 fallback 查全名。修=serve 加 `_act_lookup` 先查 `action.<sub>` 全名再 fallback 短名+支持 dict 访问。验证:`get_action -> [('action.base_motion',(24,4)),('action.control_mode',(24,1)),('action.end_effector_position',(24,3)),('action.end_effector_rotation',(24,3)),('action.gripper_close',(24,1))]`。

**⚠️ "机械臂看不到动"是节奏错觉(2026-06-01 实测澄清)。** 运动是真实且大幅的:`[client] step25: |Δarm|=75.7° |Δeef|=223mm`(关节转 75°、末端移 22cm)。但客户端播放循环 `for t in range(T): env.step; viewer.sync()` **无逐步延时**→ 每个 16 步 chunk 在 **<1 秒瞬间走完**,然后 **冻结 ~60-130s** 算下一 chunk。max_steps=32=只有 2 次亚秒爆发,中间全静止,用户极易全程只看到冻结。**想耐看:** 在 `_gr00t_eval_client.py:193` 循环里 `env.step` 后加 `time.sleep(0.15)`(16 步拉成 ~2.5s 平滑可见),和/或调大 `--max-steps`(64/96 更多轮)。success=False 对 50步 smoke ckpt 是预期(只验管线+可视化,非开柜成功)。

**运行配方(已验证 2026-06-01):** server `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python serve_dreamzero_robocasa.py --ckpt-dir <ckpt> --port 5702`(load ~2.5min,ready 后 GPU 9.7GB);client `DISPLAY=:0 SCIPY_ARRAY_API=0 POLICY_TIMEOUT_S=600 conda run -n robocasa python scripts/_gr00t_eval_client.py --env-name OpenCabinet --split target --port 5702 --n-episodes 1 --max-steps 32 --n-action-steps 16 --render --seed 0`。节奏:首 chunk ~136s(含一次性 CPU 文本编码),之后每 chunk ~58s(CLIP image encode 44s+diffusion 8s 都在 CPU)。50步 smoke ckpt success=False 正常(只验证管线+可视化)。

**robocasa env-build flaky heisenbug**:env 构建偶发 `yaml/reader.py self.index += 1: 'Element' has no attribute index` / `isinstance() arg 2 must be a type` / SIGSEGV(exit -11)。非确定,client 自带 3 次重试(EVAL_EP_RETRIES),通常第 2-3 次过。N1.5/N1.7 基准也是这套重试扛过来的。

**坑(踩过):** ① 启动脚本里 `pkill -f serve_dreamzero_robocasa` 会**误杀自己**(launcher 自己 argv 含该串)→ 任务秒退 exit 1。改用 nvidia-smi compute-apps PID 杀。② setsid+`&` 在 sandbox Bash 里的子进程会被 harness reap→日志文件不生成;server 要用 harness 的 `run_in_background:true`(直接跑 python,跨轮存活)。③ 杀 NF4 server 后显存若没回收,是进程还活着(睡眠态非僵尸),`sudo kill -9 <PID>` 即放;sudo 在 tty-less 后台要 `echo PW | sudo -S -p '' cmd`(`sudo -v` 缓存会失效再 prompt 致 hang)。
