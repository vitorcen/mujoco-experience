import time
import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = './mujoco_menagerie/universal_robots_ur10e/scene.xml'

def main():
    print(f"Loading UR10e model: {MODEL_PATH}")
    try:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data = mujoco.MjData(model)
    except ValueError:
        print("Error: Model not found. Did you run 'git submodule update --init mujoco_menagerie'?")
        return

    # UR10e 机械臂，通常有 6 个自由度
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        
        # 重置到一个非零姿态，避免奇异点
        # data.qpos[:] = [0, -1.57, 1.57, -1.57, -1.57, 0] # 经典 home pose
        
        while viewer.is_running():
            step_start = time.time()
            t = step_start - start_time

            # 让所有关节缓慢正弦摆动
            target_pos = np.sin(t * 1.0) * 0.5
            
            # 这里我们直接发送位置控制信号 (假设 actuators 是 position control)
            # 或者是 torque control，具体取决于 XML 定义
            # 如果是 torque control，这样写机器人会软塌塌的或者乱动
            # Menagerie 的 UR10e scene.xml 默认 actuator 可能是 position 或 integrated position
            data.ctrl[:] = target_pos

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
