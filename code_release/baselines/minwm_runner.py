"""minWM (Wan2.1-1.3B Action2V, 4-step DMD) on OUR eval seeds with OUR actions.

Head-to-head vs our model: same seed starts (first 13 video frames -> 4 seed
latents, block-seeded via causal_inference's native multi-frame initial_latent),
actions translated throttle/steer -> camera trajectory (w/s translation 0.08/frame,
j/l yaw 3deg/frame), identity poses over the seed block.

Env: MW_WINDOWS "8,1,2" (window ids), MW_DIRS "F,B,L,R", MW_OUT.
Run with cwd = third_party/minWM/Wan21 (their relative imports).
Outputs <MW_OUT>/minwm_r{W}_{DIR}.mp4 @16fps.
"""
import os, sys, argparse
import numpy as np, torch, imageio
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.getcwd(), "Wan21"))
from pipeline.causal_inference import CausalInferencePipeline  # noqa
from pipeline.causal_diffusion_inference import CausalDiffusionInferencePipeline  # noqa
from wan_utils.wan_wrapper import WanVAEWrapper  # noqa
from torchvision.io import write_video

ARR = os.environ.get("AF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEV = "cuda"
STEP_T = float(os.environ.get("MW_STEPT", "0.08"))
STEP_R = np.radians(float(os.environ.get("MW_STEPR", "3.0")))
CAP = os.environ.get("MW_CAP",
      "A first-person view from a small delivery robot driving on a sidewalk "
      "street in a residential area, houses and parked cars, daytime.")
WINDOWS = [int(x) for x in os.environ.get("MW_WINDOWS", "8,1,2").replace(":", ",").split(",")]
DIRS = os.environ.get("MW_DIRS", "F,B,L,R").replace(":", ",").split(",")
OUT = os.environ.get("MW_OUT", f"{ARR}/logs/eval_final/minwm_smoke")
NUM_LAT = int(os.environ.get("MW_NUMLAT", "20"))   # total latent frames
SEED_LAT = 4


def make_viewmats(direction, n_total, n_seed):
    """w2c 4x4 poses: identity over seed, then per-frame step. Our steer = yaw."""
    c2w = np.eye(4)
    mats = []
    for i in range(n_total):
        mats.append(np.linalg.inv(c2w))
        if i < n_seed - 1:
            continue                                  # static during seed
        d = np.eye(4)
        if direction == "F":
            d[2, 3] = STEP_T
        elif direction == "B":
            d[2, 3] = -STEP_T
        elif direction in ("L", "R"):
            a = STEP_R if direction == "L" else -STEP_R
            d[0, 0] = np.cos(a); d[0, 2] = np.sin(a)
            d[2, 0] = -np.sin(a); d[2, 2] = np.cos(a)
        elif direction in ("FL", "FR"):
            a = STEP_R if direction == "FL" else -STEP_R
            d[0, 0] = np.cos(a); d[0, 2] = np.sin(a)
            d[2, 0] = -np.sin(a); d[2, 2] = np.cos(a)
            d[2, 3] = STEP_T
        elif direction in ("BL", "BR"):
            a = STEP_R if direction == "BL" else -STEP_R
            d[0, 0] = np.cos(a); d[0, 2] = np.sin(a)
            d[2, 0] = -np.sin(a); d[2, 2] = np.cos(a)
            d[2, 3] = -STEP_T
        c2w = c2w @ d
    return np.stack(mats).astype(np.float32)          # (T,4,4) w2c


def main():
    os.makedirs(OUT, exist_ok=True)
    STAGE = os.environ.get("MW_STAGE", "dmd")
    cfg_map = {"dmd": "causal_forcing_dmd_camera.yaml", "ar_tf": "ar_camera_tf.yaml"}
    config = OmegaConf.load(f"Wan21/configs/{cfg_map[STAGE]}")
    default_cfg = OmegaConf.load("Wan21/configs/default_config.yaml")
    config = OmegaConf.merge(default_cfg, config)

    sub = {"dmd": "dmd", "ar_tf": "ar_diffusion_tf"}[STAGE]
    ckpt = os.environ.get("MW_CKPT") or f"ckpts/Wan21/Action2V/{sub}/model.pt"
    print(f"[minwm] stage={STAGE} ckpt: {ckpt}", flush=True)
    PipeCls = CausalInferencePipeline if STAGE == "dmd" else CausalDiffusionInferencePipeline
    if os.environ.get("MW_GUIDE"):
        config.guidance_scale = float(os.environ["MW_GUIDE"])
    if os.environ.get("MW_ATTNSIZE"):
        # widen the rolling KV window so long rollouts never evict context
        # (kills the visible seam at latent 20 = frame 77 on 36-latent runs)
        config.model_kwargs.local_attn_size = int(os.environ["MW_ATTNSIZE"])
    pipeline = PipeCls(config, device=DEV)
    if os.environ.get("MW_STEPS"):
        pipeline.sampling_steps = int(os.environ["MW_STEPS"])
    import time as _t
    _t0 = _t.time()
    state = torch.load(ckpt, map_location="cpu")
    key = "generator_ema" if os.environ.get("MW_EMA") else "generator"
    sd = state[key] if key in state else state
    pipeline.generator.load_state_dict(sd, strict=True)
    pipeline = pipeline.to(device=DEV, dtype=torch.bfloat16)
    vae = pipeline.vae

    fx = fy = cx = cy = 0.5
    Ks_np = np.array([[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]] * NUM_LAT, dtype=np.float32)

    for wi in WINDOWS:
        src = f"{ARR}/logs/eval_final/A/pca8_8node/control_test/step05000_r{wi:02d}_F_raw.mp4"
        r = imageio.get_reader(src)
        frames = [np.asarray(r.get_data(i)) for i in range(13)]
        r.close()
        vid = torch.tensor(np.stack(frames)).permute(3, 0, 1, 2).unsqueeze(0)   # [1,C,T,H,W]
        vid = (vid.float() / 127.5 - 1.0).to(DEV, torch.bfloat16)
        with torch.no_grad():
            init_lat = vae.encode_to_latent(vid).to(DEV, torch.bfloat16)        # [1,4,16,60,104]
        print(f"[minwm] r{wi:02d} seed latents {tuple(init_lat.shape)}", flush=True)

        for d in DIRS:
            tag = os.environ.get("MW_TAG", "")
            outp = os.path.join(OUT, f"minwm{tag}_r{wi:02d}_{d}.mp4")
            if os.path.exists(outp):
                continue
            vm = torch.from_numpy(make_viewmats(d, NUM_LAT, SEED_LAT)).unsqueeze(0).to(DEV, torch.bfloat16)
            Ks = torch.from_numpy(Ks_np).unsqueeze(0).to(DEV, torch.bfloat16)
            noise = torch.randn([1, NUM_LAT - SEED_LAT, 16, 60, 104], device=DEV, dtype=torch.bfloat16,
                                generator=torch.Generator(DEV).manual_seed(1234))
            with torch.no_grad():
                video, _ = pipeline.inference(
                    noise=noise, text_prompts=[CAP], viewmats=vm, Ks=Ks,
                    return_latents=True, initial_latent=init_lat)
            v = (video[0].permute(0, 2, 3, 1).float().clamp(0, 1) * 255).to(torch.uint8).cpu()
            write_video(outp, v, fps=16)
            print(f"[minwm] saved {outp} frames={v.shape[0]} (+{_t.time()-_t0:.0f}s cum)", flush=True)
    print("[minwm] DONE")


if __name__ == "__main__":
    main()
