import time
import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = './dependencies/mujoco_menagerie/franka_emika_panda/scene.xml'

def main():
    print(f"Loading Panda model: {MODEL_PATH}")
    try:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data = mujoco.MjData(model)
    except ValueError:
        print("Error: Model not found. Did you run 'git submodule update --init dependencies/mujoco_menagerie'?")
        return

    # Franka Panda 有 7 个关节 + 2 个 gripper 指头
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        while viewer.is_running():
            step_start = time.time()
            t = step_start - start_time

            # 简单的测试信号
            action = np.zeros(len(data.ctrl))
            # 让前几个关节动一动
            action[0] = np.sin(t) * 0.5
            action[1] = np.cos(t) * 0.5
            
            # 尝试控制 gripper (通常是最后两个 actuator)
            if len(action) >= 2:
                action[-1] = 255 if int(t) % 2 == 0 else 0 # 模拟开关
                action[-2] = 255 if int(t) % 2 == 0 else 0

            data.ctrl[:] = action

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
