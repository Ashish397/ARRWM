"""Matrix-Game 2.0 (universal, 3-step distilled) on OUR eval seeds.

Single-image conditioning: frame 0 of the real seed clip
(analysis/eval_final/seed65/seed65_rNN.mp4), resize-cropped to 352x640.
Scripted actions per PIXEL frame: F = hold keyboard `forward`; R = hold mouse
yaw +0.1 (their turning channel; keyboard l/r is strafe) with no key pressed,
matching our turn-in-place eval semantics.

Env: MG_WINDOWS "8:1", MG_DIRS "F:R", MG_NUMLAT (latent frames, def 150 ->
597 pixel frames), MG_CKPT, MG_MODELDIR, MG_OUT, MG_CAM (yaw/frame, def 0.1).
Run with cwd = third_party/Matrix-Game/Matrix-Game-2.
Outputs <MG_OUT>/matrixgame_rNN_D.mp4 @25fps (native) + wall time printed.
"""
import os, time
import torch
import numpy as np
import imageio
from PIL import Image
from omegaconf import OmegaConf
from torchvision.transforms import v2
from torchvision.io import write_video
from einops import rearrange
from pipeline import CausalInferencePipeline
from wan.vae.wanx_vae import get_wanx_vae_wrapper
from demo_utils.vae_block3 import VAEDecoderWrapper
from utils.misc import set_seed
from utils.wan_wrapper import WanDiffusionWrapper
from safetensors.torch import load_file

ARR = os.environ.get("AF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEV = torch.device("cuda")
DTYPE = torch.bfloat16

WINDOWS = [int(x) for x in os.environ.get("MG_WINDOWS", "8:1").replace(":", ",").split(",")]
DIRS = os.environ.get("MG_DIRS", "F:R").replace(":", ",").split(",")
NUM_LAT = int(os.environ.get("MG_NUMLAT", "150"))
CAM = float(os.environ.get("MG_CAM", "0.1"))
MODELDIR = os.environ.get("MG_MODELDIR", "Matrix-Game-2.0")
CKPT = os.environ.get("MG_CKPT", f"{MODELDIR}/base_distilled_model/base_distill.safetensors")
OUT = os.environ.get("MG_OUT", f"{ARR}/logs/eval_final/baseline_smoke/matrixgame")


KDIM = int(os.environ.get("MG_KDIM", "4"))


def make_actions(direction, num_frames):
    # base ckpt uses 6 keyboard dims; assume [fwd, back, left, right, ...] and pad
    kb = torch.zeros(num_frames, KDIM)
    ms = torch.zeros(num_frames, 2)
    if direction == "F":
        kb[:, 0] = 1
    elif direction == "B":
        kb[:, 1] = 1
    elif direction == "R":
        ms[:, 1] = CAM
    elif direction == "L":
        ms[:, 1] = -CAM
    elif direction == "FR":
        kb[:, 0] = 1; ms[:, 1] = CAM
    elif direction == "FL":
        kb[:, 0] = 1; ms[:, 1] = -CAM
    elif direction == "BR":
        kb[:, 1] = 1; ms[:, 1] = CAM
    elif direction == "BL":
        kb[:, 1] = 1; ms[:, 1] = -CAM
    elif direction == "NOOP":
        # authored all-zero keyboard+mouse stream, held for the whole rollout.
        # Representable by the released conditioning tensors, but NOT part of
        # Matrix-Game's published benchmark protocol (no released GTA no-op).
        pass
    else:
        raise ValueError(f"unmapped direction {direction!r} would silently run a null action")
    return kb, ms


def resizecrop(image, th, tw):
    w, h = image.size
    if h / w > th / tw:
        new_w = int(w); new_h = int(new_w * th / tw)
    else:
        new_h = int(h); new_w = int(new_h * tw / th)
    left = (w - new_w) / 2; top = (h - new_h) / 2
    return image.crop((left, top, left + new_w, top + new_h))


def main():
    os.makedirs(OUT, exist_ok=True)
    set_seed(0)
    config = OmegaConf.load(os.environ.get("MG_CONFIG", "configs/inference_yaml/inference_universal.yaml"))
    mode = config.pop("mode")
    generator = WanDiffusionWrapper(**getattr(config, "model_kwargs", {}), is_causal=True)
    vae_decoder = VAEDecoderWrapper()
    vae_sd = torch.load(os.path.join(MODELDIR, "Wan2.1_VAE.pth"), map_location="cpu")
    vae_decoder.load_state_dict({k: v for k, v in vae_sd.items() if "decoder." in k or "conv2" in k})
    vae_decoder.to(DEV, torch.float16).requires_grad_(False).eval()
    pipeline = CausalInferencePipeline(config, generator=generator, vae_decoder=vae_decoder)
    print(f"[mg2] loading {CKPT}", flush=True)
    sd = load_file(CKPT)
    try:
        pipeline.generator.load_state_dict(sd)
    except RuntimeError:
        # base_model ckpt stores raw WanModel keys (no wrapper prefix)
        missing, unexpected = pipeline.generator.model.load_state_dict(sd, strict=False)
        print(f"[mg2] prefixless load: missing={len(missing)} unexpected={len(unexpected)}", flush=True)
        if missing:
            print("[mg2] missing e.g.:", missing[:6], flush=True)
        if unexpected:
            print("[mg2] unexpected e.g.:", unexpected[:6], flush=True)
        assert not missing, "base ckpt does not fit CausalWanModel"
    pipeline = pipeline.to(device=DEV, dtype=DTYPE)
    pipeline.vae_decoder.to(torch.float16)
    vae = get_wanx_vae_wrapper(MODELDIR, torch.float16)
    vae.requires_grad_(False).eval()
    vae = vae.to(DEV, DTYPE)

    frame_process = v2.Compose([
        v2.Resize(size=(352, 640), antialias=True),
        v2.ToTensor(),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    tiler_kwargs = {"tiled": True, "tile_size": [44, 80], "tile_stride": [23, 38]}
    num_frames = (NUM_LAT - 1) * 4 + 1

    for wi in WINDOWS:
        seedp = f"{ARR}/analysis/eval_final/seed65/seed65_r{wi:02d}.mp4"
        r = imageio.get_reader(seedp)
        img = Image.fromarray(np.asarray(r.get_data(0))).convert("RGB")
        r.close()
        image = resizecrop(img, 352, 640)
        image = frame_process(image)[None, :, None, :, :].to(dtype=DTYPE, device=DEV)
        padding_video = torch.zeros_like(image).repeat(1, 1, 4 * (NUM_LAT - 1), 1, 1)
        img_cond = torch.concat([image, padding_video], dim=2)
        img_cond = vae.encode(img_cond, device=DEV, **tiler_kwargs).to(DEV)
        mask_cond = torch.ones_like(img_cond)
        mask_cond[:, :, 1:] = 0
        cond_concat = torch.cat([mask_cond[:, :4], img_cond], dim=1)
        visual_context = vae.clip.encode_video(image)

        for d in DIRS:
            outp = os.path.join(OUT, f"matrixgame_r{wi:02d}_{d}.mp4")
            if os.path.exists(outp):
                print(f"[mg2] exists {outp}"); continue
            kb, ms = make_actions(d, num_frames)
            conditional_dict = {
                "cond_concat": cond_concat.to(device=DEV, dtype=DTYPE),
                "visual_context": visual_context.to(device=DEV, dtype=DTYPE),
                "keyboard_cond": kb[None].to(device=DEV, dtype=DTYPE),
                "mouse_cond": ms[None].to(device=DEV, dtype=DTYPE),
            }
            noise = torch.randn([1, 16, NUM_LAT, 44, 80], device=DEV, dtype=DTYPE,
                                generator=torch.Generator(DEV).manual_seed(1234))
            t0 = time.time()
            with torch.no_grad():
                videos = pipeline.inference(
                    noise=noise, conditional_dict=conditional_dict,
                    return_latents=False, mode=mode, profile=False)
            dt = time.time() - t0
            vt = torch.cat(videos, dim=1)
            vt = rearrange(vt, "B T C H W -> B T H W C")
            v = ((vt.float() + 1) * 127.5).clip(0, 255).cpu().to(torch.uint8)[0]
            write_video(outp, v, fps=25)
            print(f"[mg2] saved {outp} frames={v.shape[0]} gen_time={dt:.1f}s "
                  f"({v.shape[0]/dt:.1f} fps)", flush=True)
    print("[mg2] DONE")


if __name__ == "__main__":
    main()
