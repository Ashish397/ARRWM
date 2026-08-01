"""WorldCam (Wan2.1-1.3B CS:GO camera-pose WM) on OUR eval seeds.

Conditioning: 65 REAL frames (analysis/eval_final/seed65/seed65_rNN.mp4,
decoded from the same ride+offset as the eval seeds; our seed windows are
non-moving so conditioning poses are identity). Generation poses: F = advance
along viewing dir, R = yaw in place — matching our eval action semantics.

Pose convention (from their example data): c2w OpenCV, per PIXEL frame,
intrinsics [fx fy cx cy] in 1920x1080 units (hardcoded original_video_wh).
ViPE per-clip scale is arbitrary -> WC_STEPT needs visual calibration.

Env: WC_WINDOWS "8:1", WC_DIRS "F:R", WC_STEPT (c2w units/frame, def 0.02),
WC_STEPR (deg/frame, def 1.5), WC_YAWSIGN (+1; flip if R turns left),
WC_ARSTEPS (def 50 -> 200 generated frames), WC_OUT, WC_PREFIX (def 1 = keep
their CS-game domain prefix). Run with cwd = third_party/WorldCam.
Outputs <WC_OUT>/worldcam_rNN_D.mp4 @16fps + per-video wall time printed.
"""
import os, sys, time, types

# diffsynth's downloader imports modelscope at module load; we never download.
_ms = types.ModuleType("modelscope")
_ms.snapshot_download = None
sys.modules.setdefault("modelscope", _ms)

import numpy as np
import torch
import imageio
from PIL import Image
from diffsynth import save_video
from diffsynth.pipelines.wan_video_new import WanVideoPipeline
from diffsynth.models import ModelManager, load_state_dict

ARR = os.environ.get("AF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WAN = os.path.join(os.environ["WAN_MODELS"], "Wan2.1-T2V-1.3B")
DIT_PATH = os.environ.get("WC_DIT", "weights/finetuned_dit.safetensors")

WINDOWS = [int(x) for x in os.environ.get("WC_WINDOWS", "8:1").replace(":", ",").split(",")]
DIRS = os.environ.get("WC_DIRS", "F:R").replace(":", ",").split(",")
STEP_T = float(os.environ.get("WC_STEPT", "0.02"))
STEP_R = np.radians(float(os.environ.get("WC_STEPR", "1.5"))) * float(os.environ.get("WC_YAWSIGN", "1"))
AR_STEPS = int(os.environ.get("WC_ARSTEPS", "50"))
OUT = os.environ.get("WC_OUT", f"{ARR}/logs/eval_final/baseline_smoke/worldcam")
COND_FRAMES = 65          # 17 latents; pipeline uses latents 1..16 (8 cond + 8 progressive)
# Overridable so the no-op run can strip motion words. The DEFAULT is the
# original string, so every earlier run reproduces byte-identically.
CAP = os.environ.get("WC_CAP",
      "a first-person view from a small delivery robot driving on a sidewalk in a "
      "residential street, houses and parked cars, daytime, clear weather.")
PREFIX = "<A first-person shooter CS game> " if os.environ.get("WC_PREFIX", "1") == "1" else ""
_NEG_DEFAULT = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
       "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，"
       "画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，"
       "杂乱的背景，三条腿，背景人很多，倒着走")
# NOTE the Wan default negative prompt lists 静态 (static), 静止 (motionless)
# and 静止不动的画面 (a still, unmoving picture) as things to AVOID. Under a
# null camera command that actively pushes the model off the behaviour being
# measured, so the no-op run overrides this to "".
NEG = os.environ.get("WC_NEG", _NEG_DEFAULT)


def make_poses(direction, n_pose, n_static):
    """c2w per pixel frame: identity over the conditioning span, then action."""
    c2w = np.eye(4)
    mats = []
    for i in range(n_pose):
        mats.append(c2w.copy())
        if i < n_static:
            continue
        d = np.eye(4)
        if direction == "F":
            d[2, 3] = STEP_T
        elif direction == "B":
            d[2, 3] = -STEP_T
        elif direction in ("L", "R", "FL", "FR", "BL", "BR"):
            a = -STEP_R if direction.endswith("L") else STEP_R
            d[0, 0] = np.cos(a); d[0, 2] = np.sin(a)
            d[2, 0] = -np.sin(a); d[2, 2] = np.cos(a)
            if direction.startswith("F"):
                d[2, 3] = STEP_T
            elif direction.startswith("B"):
                d[2, 3] = -STEP_T
        c2w = c2w @ d
    return np.stack(mats)


def main():
    os.makedirs(OUT, exist_ok=True)
    pipe = WanVideoPipeline(torch_dtype=torch.bfloat16, device="cuda")
    mm = ModelManager()
    for f in ("models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth", "diffusion_pytorch_model.safetensors"):
        mm.load_model(os.path.join(WAN, f), device="cpu", torch_dtype=pipe.torch_dtype)
    pipe.text_encoder = mm.fetch_model("wan_video_text_encoder")
    pipe.vae = mm.fetch_model("wan_video_vae")
    pipe.dit = mm.fetch_model("wan_video_dit")
    pipe.prompter.fetch_models(pipe.text_encoder)
    pipe.prompter.fetch_tokenizer(os.path.join(WAN, "google", "umt5-xxl"))
    print(f"[worldcam] loading finetuned DiT: {DIT_PATH}", flush=True)
    pipe.dit.load_state_dict(load_state_dict(DIT_PATH, device="cpu"), strict=True)
    pipe.text_encoder.to(pipe.device, dtype=pipe.torch_dtype)
    pipe.vae.to(pipe.device, dtype=pipe.torch_dtype)
    pipe.dit.to(pipe.device, dtype=pipe.torch_dtype)

    # poses: 1 (frame0, dropped) + 4*(16 cond+gen latents + AR_STEPS) + slack
    n_pose = 1 + 4 * (16 + AR_STEPS) + 40
    intr = np.tile(np.array([723.0, 723.0, 960.0, 540.0], dtype=np.float32), (n_pose, 1))

    for wi in WINDOWS:
        seedp = f"{ARR}/analysis/eval_final/seed65/seed65_r{wi:02d}.mp4"
        r = imageio.get_reader(seedp)
        frames = [Image.fromarray(np.asarray(f)) for f in r]
        r.close()
        frames = frames[:COND_FRAMES]
        assert len(frames) == COND_FRAMES, f"{seedp}: {len(frames)} frames"

        for d in DIRS:
            outp = os.path.join(OUT, f"worldcam_r{wi:02d}_{d}.mp4")
            if os.path.exists(outp):
                print(f"[worldcam] exists {outp}"); continue
            if os.environ.get("WC_FLOW_REC_BASE"):  # ARRWM flow-viz: per-dir record dir
                os.environ["WC_FLOW_REC"] = f"{os.environ['WC_FLOW_REC_BASE']}/r{wi:02d}_{d}"
            ext = make_poses(d, n_pose, n_static=COND_FRAMES)
            extrinsics = torch.from_numpy(ext).to("cuda", torch.float32)[None]
            intrinsics = torch.from_numpy(intr).to("cuda", torch.float32)[None]
            t0 = time.time()
            video = pipe(
                prompt=PREFIX + CAP,
                negative_prompt=NEG,
                input_video=frames,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
                cfg_scale=4,
                seed=0,
                tiled=True,
                num_ar_steps=AR_STEPS,
                attention_sink_inference=False,
            )
            dt = time.time() - t0
            save_video(video, outp, fps=int(os.environ.get("WC_FPS", "16")), quality=6)
            print(f"[worldcam] saved {outp} frames={len(video)} gen_time={dt:.1f}s "
                  f"({dt/max(1,len(video)-COND_FRAMES)*4:.2f}s/4frames)", flush=True)
    print("[worldcam] DONE")


if __name__ == "__main__":
    main()
