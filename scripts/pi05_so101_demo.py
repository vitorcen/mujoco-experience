"""π0 / PI05 × SO-ARM100/101 demo (REAL inference).

Loads the felixmayor/pi05_so101_orange_cube checkpoint via the bundled lerobot
in pi05_minimax_vla/ — that fork supports OpenPI-style safetensors checkpoints
which the upstream lerobot does not load out of the box.

Action pipeline:
  PI05 outputs a 32-dim action in degrees (PaliGemma padding). We slice to the
  first 6 dims, convert to radians, and write directly to data.ctrl[:6]
  (so_arm100 has 6 position-controlled actuators).

If model/weights/lerobot are unavailable, a procedural sine wave keeps the scene
moving so the visual pipeline still demonstrates.
"""
import time
import os
import re
import argparse
import numpy as np
import mujoco
import mujoco.viewer
from PIL import Image

# Make the bundled lerobot importable (pi05_minimax_vla ships its own fork).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PI05_PATH = os.path.join(_PROJECT_ROOT, "pi05_minimax_vla")
import sys
if _PI05_PATH not in sys.path:
    sys.path.insert(0, _PI05_PATH)

SCENE_XML_PATH = "scripts/so101_pi05_scene.xml"
INSTRUCTION = "pick up the orange cube"
DEFAULT_POLICY = "felixmayor/pi05_so101_orange_cube"
ARM_NQ = 6
CONTROL_DT = 0.1


class PI05Controller:
    def __init__(self, model_repo=None):
        self.policy = None
        self.normalizer = None
        self.tokenizer = None
        self.img_transform = None
        self.device = "cuda"
        if not model_repo:
            print("[info] no --policy passed, running procedural sine-wave mock.")
            return

        try:
            import json
            import torch
            from safetensors.torch import load_file
            from huggingface_hub import snapshot_download
            from lerobot.policies.pi05.modeling_pi05 import PI05Policy
            from lerobot.policies.pi05.configuration_pi05 import PI05Config
            from lerobot.configs.types import FeatureType, PolicyFeature
            from transformers import AutoTokenizer
            import torchvision.transforms as T

            print(f"Downloading PI05 checkpoint: {model_repo}")
            local_dir = snapshot_download(model_repo)

            # Build the PI05 config with the 32-dim padded features schema used
            # by OpenPI checkpoints (PaliGemma backbone + 32-dim action head).
            config = PI05Config()
            features = {
                "observation.images.image":  PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
                "observation.images.image2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
                "observation.state":         PolicyFeature(type=FeatureType.STATE,  shape=(32,)),
                "text":                      PolicyFeature(type=FeatureType.LANGUAGE, shape=(1,)),
                "action":                    PolicyFeature(type=FeatureType.ACTION, shape=(32,)),
            }
            config.features = features
            config.input_features  = {k: v for k, v in features.items() if v.type != FeatureType.ACTION}
            config.output_features = {k: v for k, v in features.items() if v.type == FeatureType.ACTION}

            self.policy = PI05Policy(config)
            state_dict = load_file(os.path.join(local_dir, "model.safetensors"))
            missing, unexpected = self.policy.load_state_dict(state_dict, strict=False)
            print(f"[OK] PI05 weights loaded (missing={len(missing)}, unexpected={len(unexpected)})")
            self.policy.eval()
            self.policy.to(self.device)

            # Normalizer from assets/.../norm_stats.json
            norm_path = None
            for root, _, files in os.walk(os.path.join(local_dir, "assets")):
                if "norm_stats.json" in files:
                    norm_path = os.path.join(root, "norm_stats.json")
                    break
            if norm_path:
                stats = json.load(open(norm_path))
                self.normalizer = self._build_normalizer(stats, torch)
                print(f"[OK] normalizer loaded from {norm_path}")
            else:
                print("[WARN] norm_stats.json not found — inference may be garbage.")

            # Tokenizer (paligemma-3b base; OpenPI checkpoints often skip the tokenizer config)
            self.tokenizer = AutoTokenizer.from_pretrained("google/paligemma-3b-pt-224",
                                                          padding_side="right")
            self.img_transform = T.Compose([
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # -> [-1, 1]
            ])
            print("[OK] PI05 pipeline ready.")
        except Exception as e:
            print(f"[WARN] PI05 load failed: {type(e).__name__}: {e}")
            print("        Falling back to procedural mock.")
            self.policy = None

    @staticmethod
    def _build_normalizer(stats, torch):
        class _N:
            def __init__(self, s):
                ns = s["norm_stats"]
                dev = "cuda" if torch.cuda.is_available() else "cpu"
                self.state_mean = torch.tensor(ns["state"]["mean"]).float().to(dev)
                self.state_std  = torch.tensor(ns["state"]["std"]).float().to(dev)
                self.act_mean   = torch.tensor(ns["actions"]["mean"]).float().to(dev)
                self.act_std    = torch.tensor(ns["actions"]["std"]).float().to(dev)
                self.state_std[self.state_std == 0] = 1.0
                self.act_std[self.act_std == 0] = 1.0
            def n_state(self, s):  return (s - self.state_mean) / self.state_std
            def dn_action(self, a): return a * self.act_std + self.act_mean
        return _N(stats)

    def predict(self, image_top_pil, image_wrist_pil, state_rad_6, instruction):
        """Return target joint positions in radians (length 6) plus a 'should_move' flag.
        Mock: sinusoidal joint dance."""
        if self.policy is None:
            t = time.time()
            return np.array([0.5 * np.sin(t * 0.5),
                             -1.5 + 0.3 * np.sin(t),
                              1.5 + 0.3 * np.cos(t),
                              0.5 * np.sin(t * 1.5),
                             -0.5 + 0.3 * np.sin(t * 0.8),
                              0.4 + 0.4 * np.sin(t * 2.0)])

        import torch
        with torch.inference_mode():
            img_top   = self.img_transform(image_top_pil).unsqueeze(0).to(self.device)
            img_wrist = self.img_transform(image_wrist_pil).unsqueeze(0).to(self.device)

            # State: 6 joints in radians -> degrees, pad to 32 dims, normalize
            state_vec = np.zeros(32, dtype=np.float32)
            state_vec[:6] = np.rad2deg(state_rad_6)
            state_t = torch.from_numpy(state_vec).float().to(self.device).unsqueeze(0)
            state_t = self.normalizer.n_state(state_t)

            # Tokenize text
            tok = self.tokenizer([instruction], return_tensors="pt", padding="max_length",
                                 max_length=64, truncation=True)
            tokens = tok["input_ids"].to(self.device)
            attn   = tok["attention_mask"].to(self.device).bool()

            batch = {
                "observation.images.image":  img_top,
                "observation.images.image2": img_wrist,
                "observation.state": state_t,
                "observation.language.tokens": tokens,
                "observation.language.attention_mask": attn,
            }
            action_out = self.policy.select_action(batch).squeeze(0)
            if action_out.dim() > 1:
                action_out = action_out[0]
            denorm = self.normalizer.dn_action(action_out)
            return np.deg2rad(denorm[:6].detach().cpu().numpy())


def load_scene_with_vfs(xml_path):
    arm_dir = os.path.join(_PROJECT_ROOT, "mujoco_menagerie", "trs_so_arm100")
    with open(os.path.join(_PROJECT_ROOT, xml_path)) as f:
        xml = f.read()
    xml = re.sub(r'<include\s+file="[^"]+so_arm100\.xml"\s*/>',
                 '<include file="so_arm100.xml"/>', xml)
    assets = {"so_arm100.xml": open(os.path.join(arm_dir, "so_arm100.xml"), "rb").read()}
    assets_dir = os.path.join(arm_dir, "assets")
    for name in os.listdir(assets_dir):
        fp = os.path.join(assets_dir, name)
        if os.path.isfile(fp):
            with open(fp, "rb") as f:
                assets[f"assets/{name}"] = f.read()
    return mujoco.MjModel.from_xml_string(xml, assets=assets)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default=DEFAULT_POLICY,
                        help=f"HF model id (default: {DEFAULT_POLICY}); pass empty to force mock")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--mock", action="store_true", help="skip policy load, mock only")
    args = parser.parse_args()

    print("--- PI05 × SO-ARM100/101 demo ---")
    model = load_scene_with_vfs(SCENE_XML_PATH)
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, height=224, width=224)
    # We render the same view twice if there's no wrist cam (the model expects two).
    cam_top = "top_cam" if model.camera("top_cam") else "vla_view"
    cam_wrist = "vla_view"

    pi05 = PI05Controller(None if args.mock else args.policy)

    try:
        mujoco.mj_resetDataKeyframe(model, data, model.key("pi05_home").id)
    except Exception:
        pass
    mujoco.mj_forward(model, data)

    def render(cam):
        renderer.update_scene(data, camera=cam)
        return Image.fromarray(renderer.render())

    if args.smoke:
        img_t = render(cam_top); img_w = render(cam_wrist)
        action = pi05.predict(img_t, img_w, data.qpos[:ARM_NQ].copy(), INSTRUCTION)
        print(f"SMOKE OK. target_joint_rad={np.round(action, 4).tolist()}")
        return

    print(f"\nInstruction: {INSTRUCTION!r}")
    print("Launching viewer...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.lookat[:] = [0.02, -0.15, 0.08]
        viewer.cam.distance = 0.7
        viewer.cam.azimuth = 210.0
        viewer.cam.elevation = -22.0

        last_t = 0.0
        target_ctrl = data.ctrl.copy()
        while viewer.is_running():
            now = time.time()
            if now - last_t > CONTROL_DT:
                last_t = now
                img_t = render(cam_top); img_w = render(cam_wrist)
                state_rad = data.qpos[:ARM_NQ].copy()
                joint_target = pi05.predict(img_t, img_w, state_rad, INSTRUCTION)
                target_ctrl[:ARM_NQ] = joint_target

            # Smooth toward target to absorb 10Hz steps
            data.ctrl[:ARM_NQ] = 0.85 * data.ctrl[:ARM_NQ] + 0.15 * target_ctrl[:ARM_NQ]

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
