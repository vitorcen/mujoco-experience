import json
import numpy as np
import mujoco
import math

# ===========================
# Helper Functions (Ported from DeepMimic)
# ===========================

BODY_JOINTS_IN_DP_ORDER = ["chest", "neck", "right_hip", "right_knee", \
                        "right_ankle", "right_shoulder", "right_elbow", "left_hip", \
                        "left_knee", "left_ankle", "left_shoulder", "left_elbow"]

DOF_DEF = {"root": 3, "chest": 3, "neck": 3, "right_shoulder": 3, \
           "right_elbow": 1, "right_wrist": 0, "left_shoulder": 3, "left_elbow": 1, \
           "left_wrist": 0, "right_hip": 3, "right_knee": 1, "right_ankle": 3, \
           "left_hip": 3, "left_knee": 1, "left_ankle": 3}

def quaternion_to_euler(q, axes='rxyz'):
    """
    Convert quaternion [w, x, y, z] to Euler angles (radians).
    """
    w, x, y, z = q
    
    # Standard conversion for Intrinsic X-Y-Z Euler
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return [roll_x, pitch_y, yaw_z]

def q_mult(q1, q2):
    """Quaternion multiplication."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return np.array([w, x, y, z])

def align_rotation(rot):
    """
    Align DeepMimic rotation (Y-up) to MuJoCo (Z-up).
    DeepMimic 仓库里采用的是“坐标系变换（change of basis）”：
      q_out = q_left * q_in * q_right
    其中 q_left 表示绕 X 轴 -90°，q_right 表示绕 X 轴 +90°（q_left 的逆）。
    """
    # rot 的格式是 [w, x, y, z]
    q_left = np.array([0.7071067811865476, -0.7071067811865476, 0.0, 0.0], dtype=np.float64)
    q_right = np.array([0.7071067811865476, 0.7071067811865476, 0.0, 0.0], dtype=np.float64)
    return q_mult(q_mult(q_left, rot), q_right)

def align_position(pos):
    """
    Align DeepMimic position (Y-up) to MuJoCo (Z-up).
    DeepMimic: x, y(up), z
    MuJoCo: x, -z, y (after axis swap)
    """
    x, y, z = pos
    return np.array([x, -z, y])


class MocapPlayer:
    """
    通用的 DeepMimic 动作捕捉数据播放器。
    加载 DeepMimic JSON 格式的动作文件，并提供 MuJoCo qpos 映射。
    """
    def __init__(self, motion_file, loop=True):
        self.data_config = []  # 存储每帧的关节状态字典
        self.dt = 0.016  # 典型帧间隔（用于展示/回退）
        self.loop = loop
        self._frame_dt = None          # 每帧持续时间（float数组）
        self._cum_time = None          # 累积时间（单调递增，最后为总时长）
        self._total_duration = None    # 总时长（秒）
        self.load_motion(motion_file)
        
    def load_motion(self, filepath):
        """加载 DeepMimic JSON 格式的动作文件"""
        print(f"Loading motion file: {filepath}")
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        if 'Loop' in data:
            self.loop = (data['Loop'] == 'wrap')
            
        if 'Frames' in data:
            frames = data['Frames']
            # DeepMimic 的 motion 每帧第 0 维是 duration，但部分文件最后一帧会给 0，
            # 如果像以前那样每帧覆盖 self.dt，会导致最终 dt=0 -> 除零。
            durations = []
            for frame in frames:
                if not frame:
                    continue
                d = float(frame[0])
                if d > 1e-9:
                    durations.append(d)

            # 选择一个稳健的 dt 回退：优先用正 duration 的中位数，否则用 1/60
            if durations:
                dt_guess = float(np.median(np.array(durations, dtype=np.float64)))
                if not np.isfinite(dt_guess) or dt_guess <= 1e-9:
                    dt_guess = 1.0 / 60.0
            else:
                dt_guess = 1.0 / 60.0
            self.dt = dt_guess

            frame_dt = []
            for frame in frames:
                # 帧格式: [duration, px, py, pz, rw, rx, ry, rz, ...]
                d = float(frame[0]) if frame else 0.0
                if not np.isfinite(d) or d <= 1e-9:
                    d = dt_guess
                frame_dt.append(d)
                
                # Root Position
                root_pos = align_position(np.array(frame[1:4]))
                
                # Root Rotation
                root_rot = align_rotation(np.array(frame[4:8]))  # [w,x,y,z]
                
                state_dict = {}
                state_dict['root'] = list(root_pos) + list(root_rot)
                
                curr_idx = 8
                for joint in BODY_JOINTS_IN_DP_ORDER:
                    dof = DOF_DEF[joint]
                    if dof == 3:  # 3DOF joint (quaternion -> euler)
                        rot = align_rotation(np.array(frame[curr_idx : curr_idx+4]))
                        curr_idx += 4
                        euler = quaternion_to_euler(rot)
                        state_dict[joint] = euler
                    elif dof == 1:  # 1DOF joint (scalar)
                        val = frame[curr_idx]
                        curr_idx += 1
                        state_dict[joint] = [val]
                    elif dof == 0:  # Fixed joint
                         state_dict[joint] = []
                         
                self.data_config.append(state_dict)

            self._frame_dt = np.array(frame_dt, dtype=np.float64)
            self._cum_time = np.cumsum(self._frame_dt)
            self._total_duration = float(self._cum_time[-1]) if len(self._cum_time) else 0.0
                
        print(
            f"✅ Loaded {len(self.data_config)} frames "
            f"(dt≈{self.dt:.6f}s, total≈{self._total_duration:.3f}s)"
        )
                
    def get_frame_qpos(self, time_sec, model):
        """
        根据时间获取对应的 qpos 向量。
        """
        if not self.data_config:
            return None

        # 使用“累积时间轴 + 二分查找”选帧，避免 dt=0 或不定 dt 的除法问题
        if self._cum_time is None or self._total_duration is None or self._total_duration <= 0:
            # 极端回退：按固定 dt 估算
            dt = self.dt if self.dt > 1e-9 else 1.0 / 60.0
            frame_idx = int(time_sec / dt)
            frame_idx = frame_idx % len(self.data_config) if self.loop else min(frame_idx, len(self.data_config) - 1)
        else:
            t = float(time_sec)
            if self.loop:
                t = t % self._total_duration
            else:
                t = min(max(t, 0.0), self._total_duration - 1e-12)

            frame_idx = int(np.searchsorted(self._cum_time, t, side="right"))
            if frame_idx >= len(self.data_config):
                frame_idx = len(self.data_config) - 1
            
        state = self.data_config[frame_idx]
        qpos = np.zeros(model.nq)
        
        # 1. Root (Free Joint) - 前 7 DOFs
        qpos[0:7] = state['root']
        
        # 2. 其他关节：通过名称映射
        qaddr = 7  # Root 后的起始地址
        
        for i in range(1, model.njnt):  # 跳过 root joint (index 0)
            jname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if not jname:
                qaddr += 1
                continue
            
            # DeepMimic 的关节命名：chest_x, chest_y, chest_z, right_elbow 等
            base_name = jname
            axis_idx = 0
            
            if jname.endswith("_x"):
                base_name = jname[:-2]
                axis_idx = 0
            elif jname.endswith("_y"):
                base_name = jname[:-2]
                axis_idx = 1
            elif jname.endswith("_z"):
                base_name = jname[:-2]
                axis_idx = 2
                
            if base_name in state:
                vals = state[base_name]
                if axis_idx < len(vals):
                    qpos[qaddr] = vals[axis_idx]
            
            qaddr += 1
            
        return qpos
