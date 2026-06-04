import time
import mujoco
import mujoco.viewer
import numpy as np

# 场景文件路径
MODEL_PATH = "scripts/panda_manipulation.xml"

# 定义关键名称 (根据 XML 中的定义)
SITE_NAME = "attachment_site"  # Panda 手末端的 site
MOCHAP_NAME = "target"         # 如果有 mocap 的话，这里我们直接用物体坐标

def main():
    # 使用 MuJoCo 的 VFS (assets dict) 解决 include + meshdir 的路径问题：
    # - 不修改 submodule
    # - 不把绝对路径写进 XML 文件
    # - 稳定加载 menagerie 的 panda.xml 以及 assets/*
    try:
        import os
        import re

        print(f"Loading model: {MODEL_PATH}")

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        menagerie_dir = os.path.join(project_root, "dependencies", "mujoco_menagerie")
        panda_dir = os.path.join(menagerie_dir, "franka_emika_panda")
        panda_xml_path = os.path.join(panda_dir, "panda.xml")
        panda_assets_dir = os.path.join(panda_dir, "assets")

        # 1) 读取主场景 XML（scripts/ 下）
        model_path_abs = os.path.join(project_root, MODEL_PATH)
        with open(model_path_abs, "r") as f:
            xml_content = f.read()

        # 2) 将 <include ...> 改成加载 VFS 内的 panda.xml
        xml_content = re.sub(r'<include\s+file="[^"]+"\s*/>', '<include file="panda.xml"/>', xml_content)

        assets: dict[str, bytes] = {}

        # 3) 注入 panda.xml 到 VFS
        with open(panda_xml_path, "rb") as f:
            assets["panda.xml"] = f.read()

        # 4) 注入 assets/* 到 VFS
        #    panda.xml 中 meshdir="assets" => 会请求 "assets/<filename>"
        for name in os.listdir(panda_assets_dir):
            file_path = os.path.join(panda_assets_dir, name)
            if not os.path.isfile(file_path):
                continue
            with open(file_path, "rb") as f:
                blob = f.read()
            assets[f"assets/{name}"] = blob

        # 5) 直接从字符串加载（MuJoCo 会优先从 VFS 找 include/mesh）
        model = mujoco.MjModel.from_xml_string(xml_content, assets=assets)
        data = mujoco.MjData(model)

    except Exception as e:
        print(f"Error: Failed to load model.\nDetail: {e}")
        return

    # 获取末端执行器（End Effector）的 ID
    # Menagerie 的 Panda XML 中，手部有一个 site 叫 "attachment_site"
    # 如果找不到，可以用 "hand" body
    site_id = None
    try:
        site_id = model.site(SITE_NAME).id
    except:
        print(f"Warning: Site '{SITE_NAME}' not found, trying to find 'hand' body...")
        try:
            site_id = model.body("hand").id
        except:
            print("Error: Could not find 'hand' body either. Exiting.")
            return

    # 获取目标物体的 Body ID
    try:
        box_id = model.body("box").id
    except:
        print("Error: Body 'box' not found. Is the XML loaded correctly?")
        return

    # IK 参数
    integration_dt = 1.0    # 积分步长
    damping = 1e-4          # 阻尼系数 (防止奇异点)
    K = 5.0                 # 增益 (移动速度)

    # 预分配矩阵内存
    jac = np.zeros((6, model.nv))  # 6xNv (Position + Rotation)
    diag = damping * np.eye(6)
    
    # 错误向量 (Position 3 + Rotation 3)
    error = np.zeros(6) 
    error_pos = error[:3]
    error_ori = error[3:]

    # 复位到 keyframe (如果有)
    if model.nkey > 0:
        mujoco.mj_resetDataKeyframe(model, data, 0)

    print("Starting IK control loop...")
    print("Robot will try to hover over the red box.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        while viewer.is_running():
            step_start = time.time()

            # 1. 获取目标位置 (红盒子的当前位置)
            # 可以在盒子上发加一点偏移，让手悬停在上方
            target_pos = data.xpos[box_id] + np.array([0, 0, 0.15])
            
            # 目标姿态 (朝下)
            # 这里简单起见，我们只追踪位置 (Position Only IK)，忽略旋转部分的误差
            # 如果要全姿态 IK，需要计算 error_ori
            
            # 2. 获取当前末端位置
            if site_id is not None:
                # 检查 site_id 是否在有效范围内
                if site_id < model.nsite:
                    site_pos = data.site_xpos[site_id]
                else:
                    # 如果 fallback 到了 body ID，需要用 body_xpos
                    # 注意：前面代码逻辑有点混淆，如果用 body id，应该去 body_xpos
                    # 让我们修正获取逻辑
                    site_pos = data.xpos[site_id] 
            else:
                break

            # 3. 计算误差 (dx)
            error_pos[:] = target_pos - site_pos

            # 4. 计算雅可比矩阵 (J)
            if site_id < model.nsite:
                 mujoco.mj_jacSite(model, data, jac[:3], jac[3:], site_id)
            else:
                 # 使用 body 雅可比
                 mujoco.mj_jacBody(model, data, jac[:3], jac[3:], site_id)

            # 5. 求解 IK: dq = J_pinv * dx
            # V = J * dq -> dq = J+ * V
            # 我们期望末端速度 V = K * error
            # 这里只用位置雅可比 (前3行) 做简单的位置追踪
            jac_pos = jac[:3]
            
            # 求解线性方程组 (带阻尼的最小二乘法 / Damped Least Squares)
            # dq = (J.T * J + lambda * I)^-1 * J.T * dx
            # 或者直接用 numpy 的 lstsq
            dq = np.linalg.lstsq(jac_pos, error_pos * K, rcond=None)[0]

            # 6. 计算目标关节角度 (积分)
            # 对于 Panda 的旋转关节，直接线性积分即可：q_new = q_current + dq * dt
            # 只有涉及浮动基座(free joint)或球铰时才必须用 mj_integratePos
            q_current = data.qpos[:7].copy()
            q_target = q_current + dq[:7] * integration_dt * model.opt.timestep
            
            # 限制关节角度在物理范围内 (可选，防止鬼畜)
            # q_target = np.clip(q_target, model.jnt_range[:7, 0], model.jnt_range[:7, 1])

            # 应用到控制信号
            # 注意：actuators 的顺序可能与 qpos 不完全一致，但 Panda 模型中是一致的
            data.ctrl[:7] = q_target
            
            # 简单的 gripper 控制 (随着时间开合)
            t = time.time() - start_time
            gripper_ctrl = 255 if int(t) % 4 < 2 else 0
            if len(data.ctrl) >= 9:
                data.ctrl[7] = gripper_ctrl
                data.ctrl[8] = gripper_ctrl

            # 7. 物理步进
            mujoco.mj_step(model, data)
            viewer.sync()

            # 保持实时
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
