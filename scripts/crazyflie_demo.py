import time
import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = './mujoco_menagerie/bitcraze_crazyflie_2/scene.xml'

def main():
    print(f"Loading Crazyflie model: {MODEL_PATH}")
    try:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data = mujoco.MjData(model)
    except ValueError:
        print("Error: Model not found. Did you run 'git submodule update --init mujoco_menagerie'?")
        return

    # Crazyflie 有 4 个旋翼电机
    # 控制信号通常对应推力 (thrust)
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        while viewer.is_running():
            step_start = time.time()
            t = step_start - start_time

            # 尝试起飞：给出足够的推力
            # 注意：不同模型的 actuator 定义不同，有些是 0-1，有些是具体力值
            # 这里给一个缓慢增加的推力测试
            thrust = 0.3 + np.sin(t) * 0.1 
            thrust = max(0, min(1.0, thrust)) # Clamp

            data.ctrl[:] = thrust

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
