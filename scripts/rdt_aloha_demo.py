"""RDT-1B × ALOHA bimanual demo (REAL inference).

Loads RDT-1B via the official thu-ml/RoboticsDiffusionTransformer code that we
cloned into rdt_src/. Key design notes:

- RDT-1B was trained for bimanual ALOHA in mind; native 14-D action layout is
  [right_arm(6), right_gripper, left_arm(6), left_gripper]. Our ALOHA scene's
  actuator order is [left_arm(6), left_gripper, right_arm(6), right_gripper] —
  we just swap halves to translate.

- step() expects 6 images in order [ext_{t-1}, right_wrist_{t-1}, left_wrist_{t-1},
  ext_t, right_wrist_t, left_wrist_t]. Our scene only renders the external view;
  agilex_model handles None entries by substituting a neutral background image.

- Language conditioning normally requires a T5-XXL embedding (11B params). To
  keep this demo self-contained we feed zero embeddings so the policy runs in
  "unconditioned" mode. The action_chunk it produces will still be a smooth
  bimanual trajectory; just not language-driven.
"""
import time
import os
import sys
import argparse
import numpy as np
import yaml
import mujoco
import mujoco.viewer
from PIL import Image

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_HERE)
_RDT_PATH = os.path.join(_PROJECT_ROOT, "rdt_src")
sys.path.insert(0, _RDT_PATH)

SCENE_XML_PATH = "scripts/aloha_rdt_scene.xml"
INSTRUCTION = "Pick up the red cube with the left arm and the blue cube with the right arm"

CONTROL_DT = 0.5  # diffusion is slow; predict every 0.5s


class RDTController:
    def __init__(self, weights_path=None):
        self.policy = None
        if not weights_path or not os.path.isfile(weights_path):
            print(f"[info] RDT weights not at {weights_path} — running mock bimanual wave.")
            return
        try:
            import torch
            from scripts.agilex_model import create_model
            with open(os.path.join(_RDT_PATH, "configs/base.yaml")) as f:
                cfg = yaml.safe_load(f)
            print(f"[RDT] loading {weights_path} (~2.3 GB)...", flush=True)
            # agilex_model's load_pretrained_weights only accepts .pt/.safetensors,
            # but the HF release is `pytorch_model.bin`. Build the model first without
            # weights, then manually load the bin file (it's a plain torch state dict).
            self.policy = create_model(
                args=cfg,
                dtype=torch.bfloat16,
                pretrained=None,
                pretrained_vision_encoder_name_or_path="google/siglip-so400m-patch14-384",
            )
            print("[RDT] loading bin weights into policy...", flush=True)
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
            sd = checkpoint["module"] if isinstance(checkpoint, dict) and "module" in checkpoint else checkpoint
            missing, unexpected = self.policy.policy.load_state_dict(sd, strict=False)
            print(f"[RDT] weight load: missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
            # Re-cast (load_state_dict can leave tensors in fp32)
            self.policy.policy = self.policy.policy.to("cuda", dtype=torch.bfloat16)
            # text_embeds shape: (B=1, T=1, lang_token_dim=4096). Zero = unconditioned.
            self.text_embeds = torch.zeros((1, 1, cfg["model"]["lang_token_dim"]),
                                            device="cuda", dtype=torch.bfloat16)
            self._image_history = []  # last external-view PIL image(s)
            self._cached_chunk = None
            self._chunk_step = 0
            print("[RDT] policy ready.", flush=True)
        except Exception as e:
            print(f"[WARN] RDT load failed: {type(e).__name__}: {e}")
            print("        Falling back to mock bimanual wave.")
            self.policy = None

    def predict(self, ext_image_pil, joint_state_14):
        """ALOHA 14-D state in -> ALOHA 14-D joint target out.
        Returns the same layout as actuators (left_arm,left_grip,right_arm,right_grip)."""
        if self.policy is None:
            t = time.time()
            s, c = np.sin(t * 0.4), np.cos(t * 0.4)
            left = np.array([0.4*s, -0.96+0.2*c, 1.16, 0.0, -0.3+0.2*s, 0.0, 0.037])
            right = np.array([-0.4*s, -0.96+0.2*c, 1.16, 0.0, -0.3-0.2*s, 0.0, 0.037])
            return np.concatenate([left, right])

        import torch
        # Maintain 2-frame ext-view history
        if not self._image_history:
            self._image_history = [ext_image_pil, ext_image_pil]
        else:
            self._image_history = [self._image_history[-1], ext_image_pil]
        # 6-image list: [ext_{t-1}, rw_{t-1}, lw_{t-1}, ext_t, rw_t, lw_t]
        # We don't have wrist cameras — pass None so step() fills with neutral bg.
        images = [self._image_history[0], None, None,
                  self._image_history[1], None, None]

        # ALOHA actuator order -> RDT/agilex order (swap halves)
        # ALOHA:  [LA(6), LG,  RA(6), RG]
        # Agilex: [RA(6), RG,  LA(6), LG]
        ja = np.asarray(joint_state_14, dtype=np.float32)
        agilex_joints = np.concatenate([ja[7:14], ja[0:7]])  # 14
        proprio = torch.tensor(agilex_joints).unsqueeze(0)   # (1, 14)

        # Reuse cached chunk for chunk_size-1 steps to amortize diffusion cost
        if self._cached_chunk is None or self._chunk_step >= self._cached_chunk.shape[1] - 1:
            with torch.no_grad():
                self._cached_chunk = self.policy.step(proprio, images, self.text_embeds)
            self._chunk_step = 0
        else:
            self._chunk_step += 1

        action_agilex = self._cached_chunk[0, self._chunk_step].cpu().numpy()  # (14,)
        # Swap halves back: agilex [RA,RG,LA,LG] -> ALOHA [LA,LG,RA,RG]
        action_aloha = np.concatenate([action_agilex[7:14], action_agilex[0:7]])
        return action_aloha


def load_scene_with_vfs(xml_path):
    aloha_dir = os.path.join(_PROJECT_ROOT, "dependencies", "mujoco_menagerie", "aloha")
    with open(os.path.join(_PROJECT_ROOT, xml_path)) as f:
        scene_xml = f.read()
    assets = {"aloha.xml": open(os.path.join(aloha_dir, "aloha.xml"), "rb").read()}
    for sibling in ("joint_position_actuators.xml", "filtered_cartesian_actuators.xml",
                    "keyframe_ctrl.xml", "keyframe_no_act.xml"):
        sp = os.path.join(aloha_dir, sibling)
        if os.path.exists(sp):
            assets[sibling] = open(sp, "rb").read()
    assets_dir = os.path.join(aloha_dir, "assets")
    for name in os.listdir(assets_dir):
        fp = os.path.join(assets_dir, name)
        if os.path.isfile(fp):
            assets[f"assets/{name}"] = open(fp, "rb").read()
    return mujoco.MjModel.from_xml_string(scene_xml, assets=assets)


def find_rdt_weights():
    """Find a locally cached RDT-1B checkpoint or /tmp download."""
    for cand in [
        "/tmp/rdt_1b_pytorch_model.bin",
        os.path.expanduser("~/.cache/huggingface/hub/models--robotics-diffusion-transformer--rdt-1b/snapshots/eb09036cc64ca4945051acbd1bd581d30a1d7711/pytorch_model.bin"),
    ]:
        if os.path.isfile(cand) and os.path.getsize(cand) >= 2_000_000_000:
            return cand
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--weights", default=None,
                        help="Path to RDT-1B pytorch_model.bin. If omitted, auto-detect.")
    parser.add_argument("--mock", action="store_true", help="skip model, mock only")
    args = parser.parse_args()

    print("--- RDT-1B × ALOHA (bimanual) demo ---")
    model = load_scene_with_vfs(SCENE_XML_PATH)
    data = mujoco.MjData(model)
    print(f"loaded: nq={model.nq}, nu={model.nu}, ncam={model.ncam}")

    renderer = mujoco.Renderer(model, height=224, width=224)

    weights = args.weights or (None if args.mock else find_rdt_weights())
    rdt = RDTController(weights)

    try:
        mujoco.mj_resetDataKeyframe(model, data, model.key("rdt_home").id)
    except Exception:
        pass
    mujoco.mj_forward(model, data)

    def render_ext():
        renderer.update_scene(data, camera="vla_view")
        return Image.fromarray(renderer.render())

    if args.smoke:
        img = render_ext()
        state14 = np.array([data.ctrl[i] if i < model.nu else 0.0 for i in range(14)])
        action = rdt.predict(img, state14)
        print(f"SMOKE OK. action_left  = {np.round(action[:7], 3).tolist()}")
        print(f"          action_right = {np.round(action[7:], 3).tolist()}")
        return

    print(f"\nInstruction: {INSTRUCTION!r}")
    print("Launching viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.0, -0.07, 0.08]
        viewer.cam.distance = 1.0
        viewer.cam.azimuth = 115.0
        viewer.cam.elevation = -28.0

        last_t = 0.0
        target_ctrl = data.ctrl.copy()
        while viewer.is_running():
            now = time.time()
            if now - last_t > CONTROL_DT:
                last_t = now
                img = render_ext()
                state14 = np.concatenate([data.qpos[:6], [data.ctrl[6]],
                                          data.qpos[8:14], [data.ctrl[13]]])
                action = np.asarray(rdt.predict(img, state14), dtype=np.float64).flatten()
                target_ctrl[:7] = action[:7]
                target_ctrl[7:14] = action[7:14]

            data.ctrl[:] = 0.92 * data.ctrl + 0.08 * target_ctrl
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
