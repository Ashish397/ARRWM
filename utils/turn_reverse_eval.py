"""There-and-back turn-consistency eval.

Command a turn one way for the first half of the rollout, then the opposite way
for the second half (equal & opposite, throttle=0 => net heading & position ~0).
If the model is consistent, the LAST frame should return to the FIRST frame.

Reports, per seed ride:
  lpips_first_last : LPIPS(frame0, last)  -- closure (LOW = returned to start)
  lpips_first_mid  : LPIPS(frame0, midpoint) -- did it actually turn (HIGH = yes)
Saves all videos for eyeballing.

Env:
  TR_CONFIG, TR_CKPT, TR_STEP, TR_NVID (default 16)
  TR_CHUNKS  : rollout horizon in chunks (7 or 14)
  TR_PATTERN : 'RL' (right then left) or 'LR' (left then right)
  TR_STEER   : |steer| magnitude (default 0.7); TR_THROTTLE (default 0.0)
  TR_OUT
"""
import os, json
os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("RANK", "0"); os.environ.setdefault("LOCAL_RANK", "0")
import numpy as np, torch
from omegaconf import OmegaConf

CONFIG = os.environ.get("TR_CONFIG", "configs/causal_lora_diffusion_teacher_v14e_noatok.yaml")
CKPT = os.environ["TR_CKPT"]; STEP = int(os.environ.get("TR_STEP", "0"))
NVID = int(os.environ.get("TR_NVID", "16")); CHUNKS = int(os.environ.get("TR_CHUNKS", "7"))
PATTERN = os.environ.get("TR_PATTERN", "RL").upper()
STEER = float(os.environ.get("TR_STEER", "0.7")); THROTTLE = float(os.environ.get("TR_THROTTLE", "0.0"))
OUT = os.environ.get("TR_OUT", "logs/turnrev"); DEV = "cuda"
# right = +steer, left = -steer (arbitrary but consistent)
DIR1 = +1.0 if PATTERN == "RL" else -1.0


def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(CONFIG))
    os.environ["ARRWM_ACTION_ENCODER"] = str(cfg.get("teacher_action_encoder", "ss_vae"))
    cfg.logdir = os.path.abspath(OUT); cfg.auto_resume = False; cfg.control_test = False
    os.makedirs(OUT, exist_ok=True)
    for f in (".ride_manifest.pt", ".ride_manifest_shared.pt"):
        src = os.path.abspath(os.path.join("logs/v14e_noatok", f)); dst = os.path.join(OUT, f)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)

    from trainer.causal_diffusion_teacher_train import CausalLoRADiffusionTrainer, _frames_to_mp4_bytes
    from utils.causal_chain_rollout import stream_causal_chain
    from utils.zarr_dataset import ZarrRideDataset
    import pyiqa
    tr = CausalLoRADiffusionTrainer(cfg)
    tr.config.resume_from = CKPT; tr.start_step = 0; tr._maybe_resume()
    wrapper = tr.model.module if hasattr(tr.model, "module") else tr.model
    wrapper.eval()
    nfb = tr.num_frame_per_block
    ego = list(tr.action_dims) if tr.action_dims is not None else [0, 1]
    tot_f = nfb * (1 + CHUNKS)
    half = CHUNKS // 2
    lpips = pyiqa.create_metric("lpips", device=DEV)

    def lp(a, b):
        ta = torch.tensor(a).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.
        tb = torch.tensor(b).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.
        return float(lpips(ta, tb).item())

    # per-frame command: seed neutral; gen chunk c -> DIR1 for c<half else -DIR1
    z_cmd = torch.zeros(1, tot_f, len(ego), device=DEV, dtype=tr.dtype)
    z_cmd[..., 0] = THROTTLE
    for c in range(CHUNKS):
        s = DIR1 if c < half else -DIR1
        fs = nfb + c * nfb
        z_cmd[:, fs:fs + nfb, 1] = s * STEER

    n_rides = len(tr.eval_dataset); rows = []
    for r in range(NVID):
        ride = tr.eval_dataset[r % n_rides]
        zp = ride["zarr_path"]; n_lat = ride["n_latent_frames"]
        if n_lat < nfb:
            continue
        prompt = ride["prompt_embeds"].unsqueeze(0).to(DEV, dtype=tr.dtype)
        offset = max(0, min((r // n_rides) * tot_f, n_lat - nfb))
        seed = ZarrRideDataset.load_latent_chunk(zp, offset, offset + nfb).unsqueeze(0).to(DEV, torch.float32)
        with torch.no_grad():
            full = stream_causal_chain(wrapper, tr.action_projection, tr.action_token_projection,
                                       prompt, seed, z_cmd, gen_chunks=CHUNKS,
                                       eval_steps=int(getattr(cfg, "eval_inference_steps", 48)),
                                       dtype=tr.dtype, device=DEV, nfb=nfb)
            frames = tr._decode_latents(full)          # [T,H,W,3]
        T = frames.shape[0]
        rec = {"rank": r, "ride": os.path.basename(str(zp)), "T": int(T),
               "lpips_first_last": lp(frames[0], frames[-1]),
               "lpips_first_mid": lp(frames[0], frames[T // 2])}
        rows.append(rec)
        print(f"[turnrev] r{r} {rec['ride']} T={T} first-last={rec['lpips_first_last']:.3f} "
              f"first-mid={rec['lpips_first_mid']:.3f}", flush=True)
        mp4 = _frames_to_mp4_bytes(frames, fps=16.0)
        if mp4:
            open(os.path.join(OUT, f"turn{CHUNKS}{PATTERN}_step{STEP}_r{r}_{rec['ride'].replace('.zarr','')}.mp4"), "wb").write(mp4)
    fl = np.array([x["lpips_first_last"] for x in rows]); fm = np.array([x["lpips_first_mid"] for x in rows])
    print(f"\n[turnrev] === step{STEP} {os.path.basename(CKPT)} {CHUNKS}ch {PATTERN}: "
          f"first-last(closure,LOW=good)={fl.mean():.3f} | first-mid(turned?,HIGH=yes)={fm.mean():.3f} n={len(rows)} ===")
    json.dump({"config": CONFIG, "ckpt": CKPT, "step": STEP, "chunks": CHUNKS, "pattern": PATTERN,
               "steer": STEER, "throttle": THROTTLE, "rows": rows,
               "first_last_mean": float(fl.mean()), "first_mid_mean": float(fm.mean())},
              open(os.path.join(OUT, f"turnrev_{CHUNKS}{PATTERN}_step{STEP}.json"), "w"), indent=2)
    print(f"[turnrev] saved {OUT}/turnrev_{CHUNKS}{PATTERN}_step{STEP}.json")


if __name__ == "__main__":
    main()
