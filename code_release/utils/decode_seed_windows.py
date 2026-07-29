"""Decode REAL seed clips from phase-A eval windows (same ride+offset as the
eval seeds) to mp4 — conditioning material for external baselines (WorldCam
needs 65 frames; single-image baselines take frame 0 of the same clip).

17 latents at the window offset -> 68 frames via WanVAE (first-latent
duplication trick, drop frame 0) -> first 65 kept.
Saves analysis/eval_final/seed65/seed65_rNN.mp4 @16fps.
Env: SW_WINDOWS "8:1" (phase-A window indices), SW_NLAT (default 17).
"""
import os, json
os.environ.setdefault("ARRWM_ACTION_ENCODER", "pca_raw")
import numpy as np, torch, imageio
from utils.zarr_dataset import ZarrRideDataset
from utils.wan_wrapper import WanVAEWrapper

ARR = os.environ.get("AF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{ARR}/analysis/eval_final/seed65"
WINDOWS = [int(x) for x in os.environ.get("SW_WINDOWS", "8:1").replace(":", ",").split(",")]
NLAT = int(os.environ.get("SW_NLAT", "17"))


def main():
    os.makedirs(OUT, exist_ok=True)
    wins = json.load(open(f"{ARR}/analysis/eval_final/phaseA_windows.json"))
    vae = WanVAEWrapper().eval().requires_grad_(False).to("cuda")
    for wi in WINDOWS:
        outp = os.path.join(OUT, f"seed65_r{wi:02d}.mp4")
        if os.path.exists(outp):
            print(f"[seed65] exists {outp}"); continue
        w = wins[wi]
        lat = ZarrRideDataset.load_latent_chunk(w["zarr_path"], w["offset"], w["offset"] + NLAT)
        lat = lat.unsqueeze(0).to("cuda").float()
        latwd = torch.cat([lat[:, 0:1], lat], dim=1)
        with torch.no_grad():
            px = vae.decode_to_pixel(latwd)[:, 1:, ...]
        v = (0.5 * (px.float() + 1)).clamp(0, 1)[0].cpu().numpy()
        v = (v * 255).astype(np.uint8)
        if v.shape[-1] != 3:
            v = v.transpose(0, 2, 3, 1)
        v = v[:65]
        imageio.mimwrite(outp, list(v), fps=16, quality=8, macro_block_size=1)
        print(f"[seed65] saved {outp} frames={len(v)} ride={w['ride']} off={w['offset']}", flush=True)
    print("[seed65] DONE")


if __name__ == "__main__":
    main()
