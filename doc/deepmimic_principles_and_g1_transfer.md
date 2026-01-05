# DeepMimic 策略原理与迁移到 Unitree G1 的工程路线

本文用“能落地实现”的角度解释 DeepMimic 的核心思想，并给出把 DeepMimic 风格策略迁移到 **Unitree G1**（MuJoCo 仿真）的一条可操作路线。

---

## 1. DeepMimic 到底在做什么？

DeepMimic 的目标不是“硬编码动作”，而是：

- 给定一段参考动作（MoCap / 动作捕捉轨迹）
- 让一个物理角色（带质量、惯量、接触、摩擦）在仿真中**学会**稳定、逼真地复现该动作
- 并且具备一定的抗扰能力（被推、地面变化等）

它的核心结构可以概括为：

- **参考动作**（Reference Motion / Motion Clip）
- **强化学习策略**（Policy \( \pi_\theta(a|s, \phi) \)）
- **低层控制**（常见是 PD 伺服或 torque + PD）
- **奖励函数**（把“像不像参考动作”量化成 reward）

其中 \(\phi\) 通常是“相位”（phase），表示当前应处于参考动作的哪个时间点（0~1）。

---

## 2. 参考动作（MoCap）如何表示？

DeepMimic 的 motion 文件通常是每帧一条记录，包含：

- 根节点位置与朝向（root pos/quat）
- 各关节的旋转（3DOF 用四元数、1DOF 用标量）
- 每帧持续时间（duration）

工程上你会遇到的关键点：

- **坐标系差异**：很多 MoCap / DeepMimic 数据是 Y-up，而 MuJoCo 是 Z-up，需要做坐标系对齐（本文项目里我们已经在 `scripts/utils/mocap_utils.py` 做了 position/rotation 对齐）。
- **关节拓扑与自由度差异**：参考动作的骨架不一定等于目标机器人骨架（这就是“动作重定向/retargeting”的核心难点）。

---

## 3. DeepMimic 的策略（Policy）在学什么？

DeepMimic 通常把任务定义为“跟踪参考动作”，因此奖励会包含多项“模仿误差”：

- **姿态误差**：关节角/四元数差（pose）
- **关节角速度误差**（vel）
- **根节点位置误差**（root position）
- **根节点朝向误差**（root orientation）
- **末端（手/脚）位置误差**（end-effector）

典型形式（示意）：

\[
r = w_p \exp(-k_p \|q - q^*\|^2) + w_v \exp(-k_v \|\dot{q} - \dot{q}^*\|^2) + \dots
\]

策略网络的输入（观测）常见包括：

- 当前关节角 \(q\)、关节角速度 \(\dot{q}\)
- 根节点姿态/速度（线速度、角速度）
- 参考动作在相位 \(\phi\) 处的目标状态（或相对误差）
- 相位 \(\phi\) 本身（让策略知道“应该做到哪一帧”）

输出（动作）常见两种：

- **目标关节角** \(q_{target}\)（交给 PD 控制器执行）
- **关节力矩** \(\tau\)（更难训练，但更通用）

原 DeepMimic/后续复现中常见训练算法：**TRPO / PPO**（也有人用 SAC 等）。

---

## 4. 为什么 DeepMimic “看起来很稳”？关键在低层控制

很多 DeepMimic 系统采用“策略输出目标角 + PD 伺服”的结构：

- RL 策略负责给出“下一步想要到达的姿态”
- 低层 PD 负责把它变成稳定的力矩，并处理接触带来的高频扰动

PD 常见形式：

\[
\tau = K_p (q_{target} - q) + K_d (\dot{q}_{target} - \dot{q}) + \tau_{ff}
\]

工程上要注意：

- 如果机器人是强耦合/闭链/大惯量（比如真实人形），单纯的“把 MoCap 角度当 qpos 强行写进去”只能算**运动学回放**，不会稳定。
- DeepMimic 的“物理真实感”来自于：策略学会在接触约束下输出可行的目标，并由 PD/torque 实际推动系统。

---

## 5. 如何把 DeepMimic 转到 Unitree G1？

这里有两条路线：**(A) 只做回放/可视化** vs **(B) 真正训练一个 DeepMimic 风格策略**。

### A. 只做 MoCap 回放（你现在项目里已有）

特点：

- 不训练、不需要 reward
- 直接把参考动作映射到 MuJoCo 模型的 `qpos`
- 优点：开发快，验证 motion 文件、坐标系、关节命名非常方便
- 缺点：不是物理控制，遇到接触/扰动不稳，且很容易出现“手脚反”“关节轴不对”的问题

适合：做 Demo、检查动作数据、做 retargeting 之前的可视化对齐。

### B. 训练 DeepMimic 风格策略到 G1（推荐路线）

目标：让 G1 在物理仿真里学会稳定模仿（走、跑、起身、踢等）。

最关键的工作分 6 步：

#### 1) 统一骨架/自由度（Motion Retargeting）

你必须定义一个从“参考骨架”到“G1 关节”的映射：

- 参考动作为“人体骨架”（DeepMimic humanoid）
- G1 有自己的关节集合（肩、髋、膝、踝等，数量/轴定义可能不同）

常见做法：

- **手工映射 + 轴校正**：先保证主要关节（髋、膝、踝、肩、肘）方向一致
- 对不匹配的 DOF：
  - 直接丢弃（比如参考有腰部多自由度，机器人只有一部分）
  - 或者用“最接近”的自由度近似

输出结果应是：给定相位 \(\phi\)，能得到 G1 的参考关节目标 \(q^*_{G1}(\phi)\) 与（可选）\(\dot{q}^*_{G1}(\phi)\)。

#### 2) 定义观测（Observation）

工程上建议最小可用观测集：

- 根节点姿态（quat）与角速度
- 关节角 \(q\)、关节角速度 \(\dot{q}\)
- 相位 \(\phi\)（以及 \(\sin(2\pi\phi), \cos(2\pi\phi)\) 更稳定）
- （可选）参考目标 \(q^*_{G1}(\phi)\) 或误差 \(q - q^*\)

#### 3) 定义动作（Action）与控制接口

强烈建议从“**动作=目标角增量/目标角**”开始：

- action: \(\Delta q\) 或 \(q_{target}\)
- 执行：用 MuJoCo actuator（position/torque）实现 PD 伺服

对 G1 这种人形，直接学 torque 会更难；先用 PD 更稳。

#### 4) 奖励函数（Reward）

DeepMimic 的 reward 核心就是“像参考动作”：

- \(r_{pose}\)：关节角接近 \(q^*\)
- \(r_{vel}\)：关节角速度接近 \(\dot{q}^*\)
- \(r_{root}\)：躯干高度/姿态稳定
- \(r_{end}\)：脚/手末端位置接近参考（可选）
- 任务项：比如“向前走”、起身成功、保持站立等

起步建议：

- 先做 **Stand** / **Idle**（站直保持），再做 Walk/Run
- 从小扰动开始（少随机推力），逐步增加

#### 5) 训练（TRPO/PPO 等）

你提到 TRPO：它的特点是更新更“保守”，在某些连续控制任务上更稳，但实现复杂度比 PPO 高。

工程建议：

- 先用 PPO（实现成熟、调参经验多）跑通；
- 如果你明确要“对齐 DeepMimic 论文”，再用 TRPO。

#### 6) 部署到 G1 模型（MuJoCo / 真机）

部署时要处理：

- **控制频率**：policy 推理频率（如 20~50Hz） vs 仿真步长（如 1~2ms）
- **动作滤波**：低通/动作平滑，避免抖
- **状态估计**：真机时 root 状态来自 IMU + 里程计
- **域随机化**：摩擦、质量、延迟等，缩小 sim2real gap

---

## 6. “Stand up straight (training via TRPO)” 在 G1 上怎么做？

建议把它拆成两个阶段：

1) **Getup imitation（模仿起身）**：参考轨迹来自 getup_faceup / getup_facedown 的 retarget 结果，reward 以 imitation 为主。
2) **Stand stabilization（站立稳定）**：在站立姿态附近加入更强的稳定 reward（躯干 pitch/roll、COM、足底接触等），并加扰动训练。

最终效果是：

- 从倒地状态开始
- 策略输出动作（目标关节角/增量）
- PD 驱动执行
- 成功站直并保持

---

## 7. 本项目里“从 DeepMimic 到 G1”当前处于哪一步？

目前我们已经完成/在做的是：

- ✅ DeepMimic motion 的读取与坐标对齐
- ✅ 动作的运动学回放（MoCap play）

下一步如果要真正做到“DeepMimic 策略迁移到 G1”，需要新增：

- G1 的关节映射与 motion retargeting（输出 \(q^*_{G1}(\phi)\)）
- 一个 RL 训练环境（obs/action/reward/termination）
- 策略训练脚本（PPO/TRPO）
- PD 控制器与执行链路（policy -> target -> PD -> torque）

---

## 8. 常见坑（非常建议提前规避）

- **关节轴定义不一致**：同名关节在不同模型中旋转轴方向可能不同，导致“手脚反、前后反”。
- **根节点坐标系**：root 的 yaw/pitch/roll 对齐不对，会导致整体朝向/摆臂方向看起来怪。
- **接触建模**：脚底几何、摩擦、软化参数会极大影响走/跑学习稳定性。
- **控制饱和与关节限位**：G1 关节限位更严格，参考动作可能不可行，必须 retarget + clamp + penalty。
- **控制频率不匹配**：policy 太慢或太快都会抖；需要明确 policy Hz 与仿真子步。

---

## 9. 你接下来想要哪条落地方向？

如果你要“真正 DeepMimic 风格策略迁移到 G1”，我建议从 **Stand（站立）** 或 **Walk（行走）** 开始做：先把“关节映射 + imitation reward + PD 控制”跑通，再扩展到 Run、Getup、Dance。

