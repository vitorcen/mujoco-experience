"""Unitree Go2 real RL locomotion policy demo (no VLA).

We load the public ONNX policy `diasAiMaster/unitree-go2-velocity-flat` (a
PPO/RSL_RL-style velocity-commanded MLP policy trained on flat ground) and
deploy it on the Menagerie Go2 in MuJoCo. The policy is ~700 KB, 45-D obs ->
12-D joint-position delta, runs in microseconds on CPU.

Why not VLA: OpenVLA outputs 7-D end-effector deltas trained on table-top
manipulation. There is no sane mapping from that to a quadruped's 12 joints.
A purpose-trained locomotion policy is the correct tool here.

Obs layout (45 = 3+3+3+12+12+12), all scale=1.0:
  base_ang_vel(3) | projected_gravity(3) | vel_cmd(3) |
  joint_pos - default(12) | joint_vel(12) | last_action(12)

Action (12) -> q_target = action * 0.5 + default_joint_pos.
Apply per-joint PD torques at sim freq (deploy.yaml gains).

Joint order in policy == sim XML order [FL, FR, RL, RR] x (hip, thigh, calf).
The deploy.yaml's `joint_ids_map` is for real Unitree hardware which uses a
different order; we don't need it in MuJoCo.
"""
import os
import sys
import time
import argparse
import numpy as np
import yaml
import mujoco
import mujoco.viewer
import re

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)

POLICY_REPO = "diasAiMaster/unitree-go2-velocity-flat"

# Default joint pos (FL, FR, RL, RR) x (hip, thigh, calf). Matches deploy.yaml.
DEFAULT_QPOS = np.array(
    [-0.1, 0.9, -1.8,  0.1, 0.9, -1.8,
     -0.1, 0.9, -1.8,  0.1, 0.9, -1.8], dtype=np.float32
)
ACTION_SCALE = 0.5
# Per-joint PD gains from deploy.yaml. Pattern: [hip=20, thigh=20, calf=40] per leg.
KP = np.array([20, 20, 40, 20, 20, 40, 20, 20, 40, 20, 20, 40], dtype=np.float32)
KD = np.array([1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1, 2], dtype=np.float32)

POLICY_DT = 0.02  # 50 Hz policy update, decimation 4 over 0.005 sim dt


def download_policy():
    from huggingface_hub import snapshot_download
    p = snapshot_download(POLICY_REPO,
                          allow_patterns=["policy.onnx", "policy.onnx.data", "params/*"])
    return p


def build_scene_with_ball():
    """Load Go2 scene with a red target ball, via VFS to flatten includes."""
    go2_dir = os.path.join(_PROJECT_ROOT, "mujoco_menagerie", "unitree_go2")
    with open(os.path.join(go2_dir, "scene.xml")) as f:
        scene_xml = f.read()
    # Place a red target ball in front of the dog.
    target = '<body name="target" pos="2 0 0.5"><geom type="sphere" size="0.2" rgba="1 0.1 0.1 1"/></body>'
    scene_xml = scene_xml.replace("</worldbody>", f"{target}</worldbody>")
    # Flatten the include path so MuJoCo finds go2.xml via the VFS.
    scene_xml = re.sub(r'<include\s+file="[^"]+"\s*/>', '<include file="go2.xml"/>', scene_xml)

    assets = {"go2.xml": open(os.path.join(go2_dir, "go2.xml"), "rb").read()}
    assets_dir = os.path.join(go2_dir, "assets")
    for name in os.listdir(assets_dir):
        fp = os.path.join(assets_dir, name)
        if os.path.isfile(fp):
            assets[f"assets/{name}"] = open(fp, "rb").read()
    return mujoco.MjModel.from_xml_string(scene_xml, assets=assets)


def quat_rotate_inverse(quat_wxyz, vec):
    """Rotate vec by the inverse of quat (world -> body frame). MuJoCo quat is [w, x, y, z]."""
    w, x, y, z = quat_wxyz
    # Build R^T = R(q^{-1}). Faster: use mju_rotVecQuat with conjugate.
    qc = np.array([w, -x, -y, -z], dtype=np.float64)
    out = np.zeros(3, dtype=np.float64)
    mujoco.mju_rotVecQuat(out, vec.astype(np.float64), qc)
    return out


class PolicyRunner:
    def __init__(self, onnx_path):
        import onnxruntime as ort
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.last_action = np.zeros(12, dtype=np.float32)

    def step(self, base_ang_vel, projected_gravity, vel_cmd, q_rel, qd):
        obs = np.concatenate([
            base_ang_vel.astype(np.float32),
            projected_gravity.astype(np.float32),
            vel_cmd.astype(np.float32),
            q_rel.astype(np.float32),
            qd.astype(np.float32),
            self.last_action,
        ])[None, :]
        action = self.sess.run(None, {self.in_name: obs})[0].flatten()
        self.last_action = action.copy()
        return action


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vx", type=float, default=0.6, help="lin_vel_x command (m/s)")
    parser.add_argument("--vy", type=float, default=0.0, help="lin_vel_y command (m/s)")
    parser.add_argument("--wz", type=float, default=0.0, help="ang_vel_z command (rad/s)")
    parser.add_argument("--smoke", action="store_true", help="one inference + a few steps then exit")
    args = parser.parse_args()

    print("--- Go2 RL locomotion (real ONNX policy) ---")
    print(f"Velocity command: vx={args.vx}, vy={args.vy}, wz={args.wz}")

    print("[1/3] Downloading policy...")
    p = download_policy()
    onnx_path = os.path.join(p, "policy.onnx")
    print(f"      {onnx_path}")

    print("[2/3] Loading scene...")
    model = build_scene_with_ball()
    data = mujoco.MjData(model)
    # Drop the robot at a safe height and at its default pose.
    data.qpos[0:3] = [0.0, 0.0, 0.42]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:19] = DEFAULT_QPOS
    mujoco.mj_forward(model, data)
    print(f"      nq={model.nq}, nu={model.nu}, dt={model.opt.timestep}")
    assert model.nu == 12, "expected 12 torque actuators"

    print("[3/3] Loading ONNX policy...")
    runner = PolicyRunner(onnx_path)
    vel_cmd = np.array([args.vx, args.vy, args.wz], dtype=np.float32)

    def compute_obs():
        # qpos: [pos(3), quat(4), joints(12)]; qvel: [linv(3), angv(3, local frame), jointv(12)]
        quat = data.qpos[3:7].copy()
        ang_vel = data.qvel[3:6].copy()
        # gravity vector in world frame is (0,0,-1); project into base frame.
        gravity_b = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        q_rel = data.qpos[7:19].astype(np.float32) - DEFAULT_QPOS
        qd = data.qvel[6:18].astype(np.float32)
        return ang_vel, gravity_b, q_rel, qd

    def policy_step():
        ang_vel, grav, q_rel, qd = compute_obs()
        action = runner.step(ang_vel, grav, vel_cmd, q_rel, qd)
        return action

    if args.smoke:
        action = policy_step()
        q_target = action * ACTION_SCALE + DEFAULT_QPOS
        tau = KP * (q_target - data.qpos[7:19]) - KD * data.qvel[6:18]
        print(f"SMOKE: action[:4]={np.round(action[:4],3).tolist()}")
        print(f"       q_target[:4]={np.round(q_target[:4],3).tolist()}")
        print(f"       tau[:4]={np.round(tau[:4],2).tolist()}")
        # Run 100 steps to confirm no NaNs.
        for _ in range(100):
            action = policy_step()
            q_target = action * ACTION_SCALE + DEFAULT_QPOS
            for _ in range(4):
                tau = KP * (q_target - data.qpos[7:19]) - KD * data.qvel[6:18]
                data.ctrl[:12] = tau
                mujoco.mj_step(model, data)
        print(f"       after 100 policy steps: base_z={data.qpos[2]:.3f}, base_lin_x_vel={data.qvel[0]:.3f}")
        return

    print("\nLaunching viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [data.qpos[0], data.qpos[1], 0.3]
        viewer.cam.distance = 2.5
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0

        last_policy_t = -1.0
        action = np.zeros(12, dtype=np.float32)
        q_target = DEFAULT_QPOS.copy()

        while viewer.is_running():
            t = data.time
            if t - last_policy_t >= POLICY_DT - 1e-6:
                last_policy_t = t
                action = policy_step()
                q_target = action * ACTION_SCALE + DEFAULT_QPOS

            tau = KP * (q_target - data.qpos[7:19]) - KD * data.qvel[6:18]
            data.ctrl[:12] = tau
            mujoco.mj_step(model, data)
            viewer.cam.lookat[0] = data.qpos[0]
            viewer.cam.lookat[1] = data.qpos[1]
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
