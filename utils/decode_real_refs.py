"""Decode REAL reference clips (27 latents = same length as eval rollouts) from
unseen daytime windows to mp4 — calibration anchors for detector tests (real
footage through the SAME WanVAE decode path as generated videos) and the
domain-matched real reference set NSG-VD wants.

Uses the same manifest + window selection machinery as the evals: picks 32
daytime unseen non-moving-ish windows NOT already used as eval seeds.
Saves analysis/eval_final/real_refs/real_{ride}_{off}.mp4
"""
import os, json
os.environ.setdefault("ARRWM_ACTION_ENCODER", "pca_raw")
import numpy as np, torch, imageio
from utils.zarr_dataset import ZarrRideDataset
from utils.wan_wrapper import WanVAEWrapper

OUT = "analysis/eval_final/real_refs"
os.makedirs(OUT, exist_ok=True)
TOT_F = 27
N = 32

def main():
    attrs = json.load(open("paper_assets/v14d_ride_attrs.json"))
    wA = json.load(open("analysis/eval_final/phaseA_windows.json"))
    wB = json.load(open("analysis/eval_final/phaseB_windows.json"))
    used = {w["ride"] for w in wA} | {w["ride"] for w in wB}
    man = torch.load("analysis/eval_final/.manifest_full.pt", map_location="cpu")
    rides = man["rides"] if isinstance(man, dict) else man
    tw = json.load(open("paper_assets/v14d_train_windows.json"))
    train = set(os.path.basename(x["zarr_path"]) for x in tw["windows"])

    def ok(r):
        b = os.path.basename(r["zarr_path"])
        tod = str(attrs.get(b.replace(".zarr", ""), {}).get("tod", "")).lower()
        return (b not in train and b not in used and tod in ("morning", "midday", "afternoon")
                and int(r["n_latent_frames"]) >= TOT_F * 3)

    pool = [r for r in rides if ok(r)]
    print(f"[refs] {len(pool)} candidate rides")
    vae = WanVAEWrapper().eval().requires_grad_(False).to("cuda")
    done = 0
    for r in pool:
        if done >= N:
            break
        zp = r["zarr_path"]; nlat = int(r["n_latent_frames"])
        off = (nlat - TOT_F) // 2
        try:
            lat = ZarrRideDataset.load_latent_chunk(zp, off, off + TOT_F).unsqueeze(0).to("cuda").float()
            latwd = torch.cat([lat[:, 0:1], lat], dim=1)
            px = vae.decode_to_pixel(latwd)[:, 1:, ...]
            v = (0.5 * (px.float() + 1)).clamp(0, 1)[0].cpu().numpy()
            v = (v * 255).astype(np.uint8)
            if v.shape[-1] != 3:
                v = v.transpose(0, 2, 3, 1)
        except Exception as e:
            print(f"skip {os.path.basename(zp)}: {str(e)[:80]}"); continue
        name = f"real_{os.path.basename(zp).replace('.zarr','')}_{off}.mp4"
        imageio.mimwrite(os.path.join(OUT, name), list(v), fps=16, quality=8, macro_block_size=1)
        done += 1
        if done % 8 == 0:
            print(f"[refs] {done}/{N}", flush=True)
    print(f"[refs] DONE: {done} real reference clips -> {OUT}")


if __name__ == "__main__":
    main()
