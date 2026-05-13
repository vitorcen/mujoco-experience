"""Octo-Small × WidowX × Bridge-style scene.

Why this combo:
- Octo is multi-embodiment but trained heavily on bridge_dataset (WidowX 250s).
- Same physical embodiment + tabletop scene used by OpenVLA, so we reuse
  widowx_bridge_scene.xml.
- Octo's interface differs from OpenVLA: it takes a dict observation with
  image_primary (+ optional image_wrist, proprio), language instruction, and
  returns an *action chunk* of shape (pred_horizon, action_dim).

If `octo` is not installed, we fall back to a small mock so the visual pipeline
(scene + camera + IK + viewer) still demonstrates correctly. Real install:

    pip install octo-models @ git+https://github.com/octo-models/octo.git@main

And weights come from `rail-berkeley/octo-small-1.5`.
"""
import time
import os
import re
import argparse
import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image

SCENE_XML_PATH = "scripts/widowx_bridge_scene.xml"
INSTRUCTION = "pick up the red block"

EE_BODY = "wx250s/gripper_link"
ARM_NQ = 6
GRIPPER_OPEN = 0.037
GRIPPER_CLOSED = 0.015

CONTROL_DT = 0.1
IK_DT = 0.002


def _patch_jax_for_octo():
    """Octo's source uses jax.random.KeyArray which was removed in jax >= 0.4.30.
    We monkey-patch it back to jax.Array (PRNG keys are just arrays in modern jax)
    so we don't have to fork or edit the installed octo package."""
    try:
        import jax
        if not hasattr(jax.random, "KeyArray"):
            jax.random.KeyArray = jax.Array
    except Exception:
        pass


class OctoController:
    """Wraps Octo-Small-1.5 to expose a (image_256, instruction)->7D action interface.

    Octo small-1.5 expects:
      - image_primary at 256x256, window of T=2 frames
      - image_wrist at 128x128, T=2 (we pad with zeros since we don't render a wrist cam)
      - pad_mask_dict to flag which modalities are real
      - language task pre-created via model.create_tasks

    Action chunk: shape (1, pred_horizon, 7); we consume the first step each call.
    Bridge action layout: [dx, dy, dz, droll, dpitch, dyaw, gripper].
    """
    WINDOW = 2
    PRIMARY_SIZE = 256
    WRIST_SIZE = 128

    def __init__(self, model_id="hf://rail-berkeley/octo-small-1.5"):
        self.model = None
        self.task = None
        self._rng_seed = 0
        self._history = []  # most-recent-last list of (256,256,3) uint8
        try:
            _patch_jax_for_octo()
            from octo.model.octo_model import OctoModel
            self.model = OctoModel.load_pretrained(model_id)
            self.task = self.model.create_tasks(texts=[INSTRUCTION])
            print(f"[OK] Octo loaded from {model_id}")
        except Exception as e:
            print(f"[WARN] Octo unavailable ({type(e).__name__}: {e}).\n"
                  f"        Running in MOCK mode. To enable real inference:\n"
                  f"        pip install git+https://github.com/octo-models/octo.git")

    def _build_obs(self, frame_uint8):
        """Maintain a 2-frame window and pack into the dict Octo expects."""
        # Octo's docs: pad shorter windows by repeating the current frame.
        if not self._history:
            self._history = [frame_uint8.copy()] * self.WINDOW
        else:
            self._history = self._history[1:] + [frame_uint8.copy()]
        primary = np.stack(self._history, axis=0)[None]  # (1, T=2, 256, 256, 3)
        wrist = np.zeros((1, self.WINDOW, self.WRIST_SIZE, self.WRIST_SIZE, 3),
                          dtype=np.uint8)
        return {
            "image_primary": primary,
            "image_wrist": wrist,
            "timestep_pad_mask": np.ones((1, self.WINDOW), dtype=bool),
            "timestep": np.arange(self.WINDOW, dtype=np.int32)[None],
            "task_completed": np.zeros((1, self.WINDOW, 4), dtype=bool),
            "pad_mask_dict": {
                "image_primary": np.ones((1, self.WINDOW), dtype=bool),
                "image_wrist": np.zeros((1, self.WINDOW), dtype=bool),  # no wrist cam
                "timestep": np.ones((1, self.WINDOW), dtype=bool),
                "task_completed": np.zeros((1, self.WINDOW), dtype=bool),
            },
        }

    def predict(self, image_rgb_uint8, proprio=None):
        if self.model is None:
            return np.array([0.0, 0.0, -0.01, 0.0, 0.0, 0.0, 0.0])
        import jax
        obs = self._build_obs(image_rgb_uint8)
        self._rng_seed += 1
        actions = self.model.sample_actions(
            obs, self.task, rng=jax.random.PRNGKey(self._rng_seed)
        )  # (B=1, pred_horizon, action_dim=7)
        # Unnormalize with the model's bridge_dataset stats (already baked in).
        action = np.asarray(actions[0, 0])
        return action


def load_scene_with_vfs(xml_path):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wx_dir = os.path.join(project_root, "mujoco_menagerie", "trossen_wx250s")
    with open(os.path.join(project_root, xml_path), "r") as f:
        xml = f.read()
    xml = re.sub(r'<include\s+file="[^"]+wx250s\.xml"\s*/>', '<include file="wx250s.xml"/>', xml)
    assets = {"wx250s.xml": open(os.path.join(wx_dir, "wx250s.xml"), "rb").read()}
    assets_dir = os.path.join(wx_dir, "assets")
    for name in os.listdir(assets_dir):
        fp = os.path.join(assets_dir, name)
        if os.path.isfile(fp):
            with open(fp, "rb") as f:
                assets[f"assets/{name}"] = f.read()
    return mujoco.MjModel.from_xml_string(xml, assets=assets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    print("--- Octo-Small × WidowX × Bridge demo ---")
    model = load_scene_with_vfs(SCENE_XML_PATH)
    data = mujoco.MjData(model)
    ee_id = model.body(EE_BODY).id

    # Octo-Small-1.5 expects 256x256 primary observations (not 224 like OpenVLA).
    renderer = mujoco.Renderer(model, height=256, width=256)
    vla_camera = "vla_view"

    if args.mock:
        octo = OctoController.__new__(OctoController)
        octo.model, octo.task = None, None
    else:
        octo = OctoController()

    try:
        mujoco.mj_resetDataKeyframe(model, data, model.key("bridge_home").id)
    except Exception:
        pass
    mujoco.mj_forward(model, data)

    target_pos = data.body(EE_BODY).xpos.copy()
    gripper_target = GRIPPER_OPEN
    jac = np.zeros((6, model.nv))

    if args.smoke:
        renderer.update_scene(data, camera=vla_camera)
        rgb = renderer.render()
        action = octo.predict(rgb)
        print(f"SMOKE OK. action={np.round(np.asarray(action, dtype=float), 4).tolist()}")
        return

    print(f"\nInstruction: {INSTRUCTION!r}")
    print("Launching viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Same opening pose as widowx_bridge_demo (shares the same scene file).
        viewer.cam.lookat[:] = [0.15, 0.0, 0.12]
        viewer.cam.distance = 0.85
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0
        last_t = 0.0
        while viewer.is_running():
            now = time.time()
            if now - last_t > CONTROL_DT:
                last_t = now
                renderer.update_scene(data, camera=vla_camera)
                rgb = renderer.render()
                proprio = np.concatenate([data.qpos[:ARM_NQ], [data.qpos[ARM_NQ]]])
                action = np.asarray(octo.predict(rgb, proprio=proprio), dtype=np.float64).flatten()
                target_pos += action[:3]
                gripper_target = GRIPPER_OPEN if action[6] > 0.5 else GRIPPER_CLOSED

            current_pos = data.body(EE_BODY).xpos
            err_pos = (target_pos - current_pos) * 5.0
            mujoco.mj_jacBody(model, data, jac[:3], jac[3:], ee_id)
            dq, *_ = np.linalg.lstsq(jac[:3], err_pos, rcond=0.01)
            data.ctrl[:ARM_NQ] = data.qpos[:ARM_NQ] + dq[:ARM_NQ] * IK_DT
            data.ctrl[ARM_NQ] = gripper_target

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
