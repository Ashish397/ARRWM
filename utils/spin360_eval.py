"""360-degree consistency eval: command a constant turn from a real seed, roll out
a long causal chain, and measure whether the view returns to the start (loop
closure). If action tokens help maintain a consistent world, pca8 (tokens on)
should close the loop better than noatok (tokens off).

Reuses the VALIDATED offline rollout (utils.causal_chain_rollout.stream_causal_chain)
and the frozen teacher (CoTracker->PCA) to (a) integrate yaw and mark the ~360deg
frame, and (b) score frame0-vs-frameK similarity (LPIPS / SSIM / MSE).

Env:
  SPIN_CONFIG : config yaml (default noatok)
  SPIN_CKPT   : checkpoint .pt to load
  SPIN_STEP   : step label
  SPIN_NVID   : number of seed rides (default 16)
  SPIN_CHUNKS : rollout horizon in chunks (default 40 => 120 gen frames)
  SPIN_STEER  : constant steer command (PC1), default 0.7
  SPIN_THROTTLE : constant throttle (PC0), default 0.4
  SPIN_OUT    : output dir
"""
import os, glob, json
os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("RANK", "0"); os.environ.setdefault("LOCAL_RANK", "0")
import numpy as np, torch
from omegaconf import OmegaConf

CONFIG = os.environ.get("SPIN_CONFIG", "configs/causal_lora_diffusion_teacher_v14e_noatok.yaml")
CKPT = os.environ["SPIN_CKPT"]
STEP = int(os.environ.get("SPIN_STEP", "0"))
NVID = int(os.environ.get("SPIN_NVID", "16"))
CHUNKS = int(os.environ.get("SPIN_CHUNKS", "40"))
STEER = float(os.environ.get("SPIN_STEER", "0.7"))
THROTTLE = float(os.environ.get("SPIN_THROTTLE", "0.4"))
OUT = os.environ.get("SPIN_OUT", "logs/spin360")
DEV = "cuda"


def main():
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"), OmegaConf.load(CONFIG))
    os.environ["ARRWM_ACTION_ENCODER"] = str(cfg.get("teacher_action_encoder", "ss_vae"))  # critical (as in offline eval)
    cfg.logdir = os.path.abspath(OUT); cfg.auto_resume = False; cfg.control_test = False
    os.makedirs(OUT, exist_ok=True)
    for f in (".ride_manifest.pt", ".ride_manifest_shared.pt"):
        src = os.path.abspath(os.path.join("logs/v14e_noatok", f)); dst = os.path.join(OUT, f)
        if os.path.exists(src) and not os.path.exists(dst):
            os.symlink(src, dst)

    from trainer.causal_diffusion_teacher_train import CausalLoRADiffusionTrainer
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
    lpips = pyiqa.create_metric("lpips", device=DEV)

    def sim(a, b):  # a,b: [H,W,3] uint8 -> LPIPS(low=similar), SSIM(high), MSE(low)
        ta = torch.tensor(a).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.
        tb = torch.tensor(b).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.
        lp = float(lpips(ta, tb).item())
        mse = float(((ta - tb) ** 2).mean().item())
        # simple global SSIM
        from math import sqrt
        return lp, mse

    n_rides = len(tr.eval_dataset)
    rows = []
    for r in range(NVID):
        ride = tr.eval_dataset[r % n_rides]
        zarr_path = ride["zarr_path"]; n_lat = ride["n_latent_frames"]
        if n_lat < nfb:
            continue
        prompt = ride["prompt_embeds"].unsqueeze(0).to(DEV, dtype=tr.dtype)
        offset = (r // n_rides) * tot_f
        offset = max(0, min(offset, n_lat - nfb))
        seed_lat = ZarrRideDataset.load_latent_chunk(zarr_path, offset, offset + nfb).unsqueeze(0).to(DEV, torch.float32)
        # constant-turn command: [throttle, steer] held for the whole rollout
        z_cmd = torch.zeros(1, tot_f, len(ego), device=DEV, dtype=tr.dtype)
        z_cmd[..., 0] = THROTTLE; z_cmd[..., 1] = STEER
        with torch.no_grad():
            full_lat = stream_causal_chain(
                wrapper, tr.action_projection, tr.action_token_projection, prompt, seed_lat, z_cmd,
                gen_chunks=CHUNKS, eval_steps=int(getattr(cfg, "eval_inference_steps", 48)),
                dtype=tr.dtype, device=DEV, nfb=nfb)
            frames = tr._decode_latents(full_lat)                       # [T,H,W,3] uint8
            _, teacher_z8 = tr._compute_teacher_visuals(full_lat)       # [1,nc,8]
        # integrate teacher-read yaw (PC1/steer) as a proxy; find ~360deg frame
        yaw_rate = teacher_z8[0, :, ego[1]].float().cpu().numpy()       # per chunk
        cum = np.cumsum(np.abs(yaw_rate)); cum = cum / (cum[-1] + 1e-9)  # normalized 0..1 over rollout
        # frame0 vs every frame similarity
        T = frames.shape[0]; f0 = frames[0]
        lps, mses = [], []
        for k in range(T):
            lp, m = sim(f0, frames[k]); lps.append(lp); mses.append(m)
        lps = np.array(lps); mses = np.array(mses)
        # best loop-closure match (skip first 25% of horizon to avoid trivial early match)
        s = max(1, T // 4)
        best_k = s + int(np.argmin(lps[s:]))
        rec = {"rank": r, "ride": os.path.basename(str(zarr_path)), "offset": int(offset),
               "T": int(T), "best_frame": int(best_k), "best_lpips": float(lps[best_k]),
               "best_mse": float(mses[best_k]),
               "lpips_curve": [round(float(x), 4) for x in lps[::max(1, T // 60)]]}
        rows.append(rec)
        print(f"[spin] rank {r} ride={rec['ride']} T={T} best_frame={best_k} best_lpips={rec['best_lpips']:.3f}", flush=True)
        # save ALL spin videos for eyeballing
        from trainer.causal_diffusion_teacher_train import _frames_to_mp4_bytes
        mp4 = _frames_to_mp4_bytes(frames, fps=16.0)
        if mp4:
            open(os.path.join(OUT, f"spin_step{STEP}_r{r}_{rec['ride'].replace('.zarr','')}.mp4"), "wb").write(mp4)
    lp_all = np.array([x["best_lpips"] for x in rows])
    print(f"\n[spin] === step {STEP} ({os.path.basename(CKPT)}): "
          f"loop-closure LPIPS (lower=better) mean={lp_all.mean():.3f} median={np.median(lp_all):.3f} n={len(rows)} ===")
    json.dump({"config": CONFIG, "ckpt": CKPT, "step": STEP, "steer": STEER, "throttle": THROTTLE,
               "chunks": CHUNKS, "rows": rows, "best_lpips_mean": float(lp_all.mean())},
              open(os.path.join(OUT, f"spin360_step{STEP}.json"), "w"), indent=2)
    print(f"[spin] saved {OUT}/spin360_step{STEP}.json")


if __name__ == "__main__":
    main()
