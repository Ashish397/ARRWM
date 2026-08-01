"""Vista (OpenDriveLab, SVD-based driving WM) on OUR eval seeds.

Single-image conditioning: frame 0 of the real seed clip, center-cropped to
576x1024. Actions injected into value_dict (IMG mode is action-free upstream):
steer mode = 4 future keyframes of (speed m/s, steering-wheel angle deg/780).
nuScenes CAN convention: positive angle = LEFT. F = speed only; R = speed +
negative angle. Vista is a car model — no turn-in-place; R uses forward+steer.

Env: VS_WINDOWS "8:1", VS_DIRS "F:R", VS_SPEED (m/s, def 2.0), VS_ANGLE
(steering-wheel deg, def 260, sign handled per direction), VS_ROUNDS (def 4
-> 22*4+3=91 frames @10fps ~ 2x our standard clip), VS_STEPS (50), VS_OUT.
Run with cwd = third_party/Vista. Outputs <VS_OUT>/vista_rNN_D.mp4 @10fps.
"""
import os, sys, time
import numpy as np
import torch
import imageio
from PIL import Image
from pytorch_lightning import seed_everything

import init_proj_path  # noqa
from sample_utils import init_model, init_sampling, do_sample, init_embedder_options, set_lowvram_mode
from torchvision import transforms

ARR = os.environ.get("AF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WINDOWS = [int(x) for x in os.environ.get("VS_WINDOWS", "8:1").replace(":", ",").split(",")]
DIRS = os.environ.get("VS_DIRS", "F:R").replace(":", ",").split(",")
SPEED = float(os.environ.get("VS_SPEED", "2.0"))
ANGLE = float(os.environ.get("VS_ANGLE", "260.0"))
N_ROUNDS = int(os.environ.get("VS_ROUNDS", "4"))
N_FRAMES = 25
N_STEPS = int(os.environ.get("VS_STEPS", "50"))
CFG = 2.5
OUT = os.environ.get("VS_OUT", f"{ARR}/logs/eval_final/baseline_smoke/vista")

VERSION = {"config": "configs/inference/vista_sdpa.yaml", "ckpt": "ckpts/vista.safetensors"}


def action_dict(direction):
    if direction == "F":
        ang = 0.0
    elif direction == "R":
        ang = -ANGLE   # positive = left (nuScenes CAN); right = negative
    elif direction == "L":
        ang = ANGLE
    else:
        raise ValueError(direction)
    return {"speed": torch.tensor([SPEED] * 4),
            "angle": torch.tensor([ang] * 4) / 780}


def load_seed_frame(wi, th=576, tw=1024):
    r = imageio.get_reader(f"{ARR}/analysis/eval_final/seed65/seed65_r{wi:02d}.mp4")
    img = Image.fromarray(np.asarray(r.get_data(0))).convert("RGB")
    r.close()
    ow, oh = img.size
    if ow / oh > tw / th:
        w2 = int(tw / th * oh); img = img.crop(((ow - w2) // 2, 0, (ow + w2) // 2, oh))
    elif ow / oh < tw / th:
        h2 = int(th / tw * ow); img = img.crop((0, (oh - h2) // 2, ow, (oh + h2) // 2))
    img = img.resize((tw, th), resample=Image.LANCZOS)
    t = transforms.ToTensor()(img) * 2.0 - 1.0
    return t.to("cuda")


def main():
    os.makedirs(OUT, exist_ok=True)
    set_lowvram_mode(False)
    model = init_model(VERSION)
    unique_keys = set([x.input_key for x in model.conditioner.embedders])

    for wi in WINDOWS:
        img = load_seed_frame(wi)
        images = torch.stack([img] * N_FRAMES)
        for d in DIRS:
            outp = os.path.join(OUT, f"vista_r{wi:02d}_{d}.mp4")
            if os.path.exists(outp):
                print(f"[vista] exists {outp}"); continue
            seed_everything(23)
            value_dict = init_embedder_options(unique_keys)
            cond_img = img[None]
            value_dict["cond_frames_without_noise"] = cond_img
            value_dict["cond_aug"] = 0.0
            value_dict["cond_frames"] = cond_img
            for k, v in action_dict(d).items():
                value_dict[k] = v
            guider = "TrianglePredictionGuider" if N_ROUNDS > 1 else "VanillaCFG"
            sampler = init_sampling(guider=guider, steps=N_STEPS, cfg_scale=CFG, num_frames=N_FRAMES)
            uc_keys = ["cond_frames", "cond_frames_without_noise", "command", "trajectory", "speed", "angle", "goal"]
            t0 = time.time()
            out = do_sample(images, model, sampler, value_dict,
                            num_rounds=N_ROUNDS, num_frames=N_FRAMES,
                            force_uc_zero_embeddings=uc_keys,
                            initial_cond_indices=[0])
            dt = time.time() - t0
            samples, samples_z, inputs = out
            v = (samples.clamp(0, 1) * 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()
            imageio.mimwrite(outp, list(v), fps=10, quality=8, macro_block_size=1)
            print(f"[vista] saved {outp} frames={len(v)} gen_time={dt:.1f}s", flush=True)
    print("[vista] DONE")


if __name__ == "__main__":
    main()
