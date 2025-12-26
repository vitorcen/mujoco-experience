import time
import mujoco
import mujoco.viewer
import numpy as np

# 模型路径
MODEL_PATH = './mujoco_menagerie/boston_dynamics_spot/scene.xml'

def main():
    print(f"Loading Spot model: {MODEL_PATH}")
    try:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data = mujoco.MjData(model)
    except ValueError:
        print("Error: Model not found. Did you run 'git submodule update --init mujoco_menagerie'?")
        return

    # Spot 通常有 12 个关节电机
    # FL_hip, FL_upper, FL_lower, FR_hip... 等
    # 我们让它做一个简单的"蹲起"动作或摆动
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        while viewer.is_running():
            step_start = time.time()
            t = step_start - start_time

            # 生成控制信号
            # 简单的正弦波，让腿部关节摆动
            # ctrl 维度取决于 actuators 数量
            action = np.sin(t * 3.0) * 0.5 
            
            # 将信号应用到所有执行器
            # 注意：实际控制需要针对每个关节设计 PID 或逆运动学
            data.ctrl[:] = action

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
