import time
import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = './mujoco_menagerie/franka_fr3/scene.xml'

def main():
    print(f"Loading FR3 model: {MODEL_PATH}")
    try:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data = mujoco.MjData(model)
    except ValueError:
        print("Error: Model not found. Did you run 'git submodule update --init mujoco_menagerie'?")
        return

    # FR3 与 Panda 结构类似
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        while viewer.is_running():
            step_start = time.time()
            t = step_start - start_time

            # 简单的测试信号
            action = np.zeros(len(data.ctrl))
            # 简单的波浪动作
            for i in range(min(7, len(action))):
                 action[i] = np.sin(t + i*0.5) * 0.3

            data.ctrl[:] = action

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
