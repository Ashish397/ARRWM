"""Matrix-Game 2.0 (universal, 3-step distilled) on OUR eval seeds.

Single-image conditioning: canonical real-video frame 32, supplied as a
one-frame image whose ``_f0`` suffix means frame zero of this model input.
Scripted actions per PIXEL frame: F = hold keyboard `forward`; R = hold mouse
yaw +0.1 (their turning channel; keyboard l/r is strafe) with no key pressed,
matching our turn-in-place eval semantics.

Env: MG_WINDOWS "8:1", MG_DIRS "F:R", MG_NUMLAT (latent frames, def 150 ->
597 pixel frames), MG_CKPT, MG_MODELDIR, MG_OUT, MG_CAM (yaw/frame, def 0.1).
Run with cwd = third_party/Matrix-Game/Matrix-Game-2.
Outputs <MG_OUT>/matrixgame_rNN_D.mp4 @25fps (native) + wall time printed.

The vendor streaming entrypoint asks for an action once per three-latent-frame
block and writes that value over the corresponding pixel-frame interval.  A
held command is therefore exactly the constant pixel-frame tensor authored
here; the generation path below is the vendor ``CausalInferencePipeline``
used by ``inference.py`` rather than a reimplementation of its cache loop.
"""
import os, sys, time, json, hashlib

# This runner lives in ARRWM, while Matrix-Game has its own top-level
# ``pipeline``, ``wan`` and ``utils`` packages.  Put the vendor checkout at
# the front before importing any of them; otherwise ARRWM's ``utils`` package
# can be selected and paired with Matrix-Game's incompatible Wan sources.
VENDOR_ROOT = os.path.abspath(os.environ.get("MG_VENDOR_ROOT", os.getcwd()))
if not os.path.isfile(os.path.join(VENDOR_ROOT, "pipeline", "causal_inference.py")):
    raise RuntimeError(f"MG_VENDOR_ROOT is not a Matrix-Game checkout: {VENDOR_ROOT}")
sys.path.insert(0, VENDOR_ROOT)

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

WINDOWS = [int(x) if x.isdigit() else x for x in os.environ.get("MG_WINDOWS", "8:1").replace(":", ",").split(",")]
SEED_FMT = os.environ.get("MG_SEED_FMT", f"{ARR}/analysis/eval_final/seed65/seed65_r{{wi:02d}}.mp4")
DIRS = os.environ.get("MG_DIRS", "F:R").replace(":", ",").split(",")
NUM_LAT = int(os.environ.get("MG_NUMLAT", "150"))
CAM = float(os.environ.get("MG_CAM", "0.1"))
SEED = int(os.environ.get("MG_SEED", "0"))
MODELDIR = os.environ.get("MG_MODELDIR", "Matrix-Game-2.0")
CKPT = os.environ.get("MG_CKPT", f"{MODELDIR}/base_distilled_model/base_distill.safetensors")
OUT = os.environ.get("MG_OUT", f"{ARR}/logs/eval_final/baseline_smoke/matrixgame")


KDIM = int(os.environ.get("MG_KDIM", "4"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def make_actions(direction, num_frames):
    # The released universal distilled checkpoint declares exactly four
    # keyboard dimensions in this order: forward, back, left-strafe,
    # right-strafe.  Turns are the second mouse coordinate, as in the vendor
    # ``get_current_action(mode="universal")`` map.
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
    set_seed(SEED)
    config = OmegaConf.load(os.environ.get("MG_CONFIG", "configs/inference_yaml/inference_universal.yaml"))
    mode = config.pop("mode")
    if mode != "universal":
        raise RuntimeError(f"panel32 requires Matrix-Game universal mode, got {mode!r}")
    if int(config.num_frame_per_block) != 3:
        raise RuntimeError(
            f"released universal checkpoint requires 3 latent frames/block, got "
            f"{config.num_frame_per_block}"
        )
    if NUM_LAT % int(config.num_frame_per_block):
        raise RuntimeError(
            f"MG_NUMLAT={NUM_LAT} is not divisible by the native block length "
            f"{config.num_frame_per_block}"
        )
    model_config_path = os.path.join(
        VENDOR_ROOT, str(config.model_kwargs.model_config), "config.json"
    )
    with open(model_config_path) as f:
        released_model_config = json.load(f)
    action_config = released_model_config.get("action_config", {})
    if int(action_config.get("keyboard_dim_in", -1)) != KDIM:
        raise RuntimeError(
            f"MG_KDIM={KDIM} disagrees with released checkpoint config "
            f"keyboard_dim_in={action_config.get('keyboard_dim_in')!r}"
        )
    if not action_config.get("enable_keyboard") or not action_config.get("enable_mouse"):
        raise RuntimeError("released universal config does not enable keyboard and mouse")
    if CAM != 0.1:
        raise RuntimeError(
            f"panel32 uses the released horizontal-camera magnitude 0.1, got {CAM}"
        )
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
        assert not unexpected, "base ckpt contains unexpected CausalWanModel keys"
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
        seedp = SEED_FMT.format(wi=wi)
        wtag = f"r{wi:02d}" if isinstance(wi, int) else str(wi)
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
            outp = os.path.join(OUT, f"matrixgame_{wtag}_{d}.mp4")
            if os.path.exists(outp):
                print(f"[mg2] exists {outp}"); continue
            kb, ms = make_actions(d, num_frames)
            conditional_dict = {
                "cond_concat": cond_concat.to(device=DEV, dtype=DTYPE),
                "visual_context": visual_context.to(device=DEV, dtype=DTYPE),
                "keyboard_cond": kb[None].to(device=DEV, dtype=DTYPE),
                "mouse_cond": ms[None].to(device=DEV, dtype=DTYPE),
            }
            # Reset before every branch so different actions from the same
            # context share the released runner's seed-0 stochastic path.
            # This matches the vendor inference default and makes action
            # comparisons independent of loop/shard ordering.
            set_seed(SEED)
            noise = torch.randn(
                [1, 16, NUM_LAT, 44, 80], device=DEV, dtype=DTYPE
            )
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
            with open(outp + ".json", "w") as f:
                json.dump({"model": "Matrix-Game-2.0", "window": wtag, "direction": d,
                           "seed_image": os.path.abspath(seedp), "seed_sha256": sha256(seedp),
                           "generation_boundary_real_frame": int(os.environ.get("EVAL_REAL_FRAME", "0")),
                           "frames": int(v.shape[0]), "fps": 25, "latent_frames": NUM_LAT,
                           "sampling_seed": SEED,
                           "camera_yaw_per_frame": CAM, "keyboard_dims": KDIM,
                           "mode": mode,
                           "native_latent_frames_per_block": int(config.num_frame_per_block),
                           "inference_pipeline": "vendor CausalInferencePipeline",
                           "held_action_equivalence": (
                               "constant pixel-frame tensors equal repeated vendor "
                               "streaming actions for every three-latent-frame block"
                           ),
                           "checkpoint_path": os.path.abspath(CKPT),
                           "checkpoint_size_bytes": os.path.getsize(CKPT),
                           "vendor_root": VENDOR_ROOT,
                           "wall_seconds": dt}, f, indent=1)
            print(f"[mg2] saved {outp} frames={v.shape[0]} gen_time={dt:.1f}s "
                  f"({v.shape[0]/dt:.1f} fps)", flush=True)
    print("[mg2] DONE")


if __name__ == "__main__":
    main()
