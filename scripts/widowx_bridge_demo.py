"""OpenVLA × WidowX 250s × Bridge V2 style scene.

Why this matches the training distribution:
- WidowX 250s is the exact physical robot used to collect BridgeData V2.
- Tabletop with small graspable objects (red block / corn shape / blue cup).
- Fixed 3rd-person camera at the Bridge-typical front-right angle.
- 224x224 RGB rendering (OpenVLA's input resolution).
- `unnorm_key="bridge_orig"` so action deltas come out in WidowX physical units.
"""
import time
import os
import re
import argparse
import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image

# ---- config ----
MODEL_ID = "openvla/openvla-7b"
LOAD_IN_4BIT = False

SCENE_XML_PATH = "scripts/widowx_bridge_scene.xml"
INSTRUCTION = "Pick up the red block"

EE_BODY = "wx250s/gripper_link"
ARM_NQ = 6            # waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
GRIPPER_OPEN = 0.037
GRIPPER_CLOSED = 0.015

CONTROL_DT = 0.1
IK_DT = 0.002


# ---- OpenVLA loader (shared pattern with vla_inference_demo.py) ----
class OpenVLAController:
    def __init__(self, model_id, load_in_4bit=False):
        self.model = None
        self.processor = None
        self.device = "cuda" if os.system("nvidia-smi > /dev/null 2>&1") == 0 else "cpu"
        print(f"Initializing VLA on {self.device}...")
        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor

            def _from_pretrained_eager(**kwargs):
                try:
                    return AutoModelForVision2Seq.from_pretrained(
                        **kwargs, attn_implementation="eager"
                    )
                except TypeError:
                    return AutoModelForVision2Seq.from_pretrained(**kwargs)

            if self.device == "cuda" and load_in_4bit:
                from transformers import BitsAndBytesConfig
                quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
                self.model = _from_pretrained_eager(
                    pretrained_model_name_or_path=model_id,
                    quantization_config=quant,
                    device_map="auto",
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                )
            else:
                self.model = _from_pretrained_eager(
                    pretrained_model_name_or_path=model_id,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    low_cpu_mem_usage=True,
                    trust_remote_code=True,
                ).to(self.device)
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
            print("[OK] OpenVLA loaded.")
        except Exception as e:
            print(f"[WARN] OpenVLA load failed: {e}\n        Running in MOCK mode (constant downward action).")

    def predict(self, image_pil, instruction):
        if self.model is None:
            # Mock: drift down + close gripper (so the visual pipeline still demonstrates motion)
            return np.array([0.0, 0.0, -0.01, 0.0, 0.0, 0.0, 0.0])
        import torch
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(prompt, image_pil, return_tensors="pt")
        for k, v in list(inputs.items()):
            if hasattr(v, "to"):
                inputs[k] = v.to(self.model.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(
                dtype=getattr(self.model, "dtype", torch.float16)
            )
        if "attention_mask" in inputs:
            del inputs["attention_mask"]
        with torch.inference_mode():
            return self.model.predict_action(**inputs, unnorm_key="bridge_orig", use_cache=False)


# ---- Scene loader via VFS ----
def load_scene_with_vfs(xml_path):
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    wx_dir = os.path.join(project_root, "mujoco_menagerie", "trossen_wx250s")

    with open(os.path.join(project_root, xml_path), "r") as f:
        xml = f.read()
    # Rewrite the relative include to a VFS-local name
    xml = re.sub(r'<include\s+file="[^"]+wx250s\.xml"\s*/>', '<include file="wx250s.xml"/>', xml)

    assets = {}
    with open(os.path.join(wx_dir, "wx250s.xml"), "rb") as f:
        assets["wx250s.xml"] = f.read()
    assets_dir = os.path.join(wx_dir, "assets")
    for name in os.listdir(assets_dir):
        fp = os.path.join(assets_dir, name)
        if os.path.isfile(fp):
            with open(fp, "rb") as f:
                blob = f.read()
            assets[f"assets/{name}"] = blob
    return mujoco.MjModel.from_xml_string(xml, assets=assets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Skip model load, use mock actions")
    parser.add_argument("--smoke", action="store_true", help="One inference then exit")
    args = parser.parse_args()

    print("--- OpenVLA × WidowX (Bridge-style) demo ---")
    model = load_scene_with_vfs(SCENE_XML_PATH)
    data = mujoco.MjData(model)

    ee_id = model.body(EE_BODY).id

    renderer = mujoco.Renderer(model, height=224, width=224)
    vla_camera = "vla_view"
    try:
        model.camera(vla_camera).id
    except Exception:
        vla_camera = -1

    if args.mock:
        vla = OpenVLAController.__new__(OpenVLAController)
        vla.model, vla.processor = None, None
        vla.device = "cpu"
        print("[mock] using constant action")
    else:
        vla = OpenVLAController(MODEL_ID, LOAD_IN_4BIT)

    # Reset to 'bridge_home' (defined in widowx_bridge_scene.xml); wx250s.xml has its own 'home'
    # that only covers 8 qpos, so picking that one would drop our 3 free-joint objects to origin.
    try:
        key_id = model.key("bridge_home").id
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    except Exception:
        pass
    mujoco.mj_forward(model, data)

    target_pos = data.body(EE_BODY).xpos.copy()
    gripper_target = GRIPPER_OPEN
    jac = np.zeros((6, model.nv))

    if args.smoke:
        renderer.update_scene(data, camera=vla_camera)
        rgb = renderer.render()
        action = vla.predict(Image.fromarray(rgb), INSTRUCTION)
        print(f"SMOKE OK. action={np.round(np.asarray(action, dtype=float), 4).tolist()}")
        return

    print(f"\nInstruction: {INSTRUCTION!r}")
    print("Launching viewer...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Explicit opening camera so the user immediately sees arm + table + objects.
        # Free-look still works (mouse drag/scroll); this just sets the initial pose.
        viewer.cam.lookat[:] = [0.15, 0.0, 0.12]
        viewer.cam.distance = 0.85
        viewer.cam.azimuth = 135.0
        viewer.cam.elevation = -20.0
        last_vla = 0.0
        while viewer.is_running():
            now = time.time()
            if now - last_vla > CONTROL_DT:
                last_vla = now
                renderer.update_scene(data, camera=vla_camera)
                rgb = renderer.render()
                action = np.asarray(vla.predict(Image.fromarray(rgb), INSTRUCTION),
                                    dtype=np.float64).flatten()
                # Bridge unnorm returns physical meters per action step -> apply directly.
                target_pos += action[:3]
                gripper_target = GRIPPER_OPEN if action[6] > 0.5 else GRIPPER_CLOSED

            # IK: minimize position error on EE body
            current_pos = data.body(EE_BODY).xpos
            err_pos = (target_pos - current_pos) * 5.0
            mujoco.mj_jacBody(model, data, jac[:3], jac[3:], ee_id)
            dq, *_ = np.linalg.lstsq(jac[:3], err_pos, rcond=0.01)
            q_target = data.qpos[:ARM_NQ] + dq[:ARM_NQ] * IK_DT
            data.ctrl[:ARM_NQ] = q_target
            data.ctrl[ARM_NQ] = gripper_target  # gripper actuator

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
