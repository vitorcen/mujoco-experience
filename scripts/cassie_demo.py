import time
import mujoco
import mujoco.viewer
import numpy as np

MODEL_PATH = './mujoco_menagerie/agility_cassie/scene.xml'

def main():
    print(f"Loading Cassie model: {MODEL_PATH}")
    try:
        model = mujoco.MjModel.from_xml_path(MODEL_PATH)
        data = mujoco.MjData(model)
    except ValueError:
        print("Error: Model not found. Did you run 'git submodule update --init mujoco_menagerie'?")
        return

    # Cassie 是双足机器人，具有较多的被动关节和闭链结构
    
    with mujoco.viewer.launch_passive(model, data) as viewer:
        start_time = time.time()
        while viewer.is_running():
            step_start = time.time()
            t = step_start - start_time

            # 简单的测试信号
            ctrl_val = np.sin(t * 2.0) * 0.2
            data.ctrl[:] = ctrl_val

            mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()
