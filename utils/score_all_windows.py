"""Traverse the v14 ride manifest and score EVERY block-aligned 24-latent
window (3 context + 21) by motion + backward-ness, on GPU, sharded across
SLURM ranks. Output per-rank shards; merge separately.

Per window we record: zarr_path, start, n_latent_frames, turn (|z2|),
turn_var (std z2), motion (|z7|), score (turn + 0.7*motion + 0.5*turn_var,
same formula as harvest_backward_windows.py), mean_z7_future (for backward
selection; < thr => reverse-dominated).

This is the data-prep traversal for v14b: later we select the top-X% by
score (with non-max suppression) UNION the backward windows as the curated
high-motion + backward-augmented training pool.

Sharding: ride i handled by rank (i % WORLD). GPU = SLURM_LOCALID.
"""
import sys, os, json, argparse
sys.path.insert(0, '/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM')
os.chdir('/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM')
import numpy as np
import torch
from utils.zarr_dataset import ZarrRideDataset

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="logs/v14_balanced_weunz/.ride_manifest.pt")
ap.add_argument("--motion_root", default="/projects/u6ex/fbots/frodobots_motion")
ap.add_argument("--ss_vae", default="action_query/checkpoints/ss_vae_8free.pt")
ap.add_argument("--window", type=int, default=24)   # context(3)+chunk(21)
ap.add_argument("--cf", type=int, default=3)
ap.add_argument("--stride", type=int, default=3)    # block-aligned dense candidates
ap.add_argument("--bwd_thr", type=float, default=-0.15)
ap.add_argument("--outdir", default="paper_assets/window_scores")
args = ap.parse_args()

rank = int(os.environ.get("SLURM_PROCID", "0"))
world = int(os.environ.get("SLURM_NTASKS", "1"))
local = int(os.environ.get("SLURM_LOCALID", "0"))
dev = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[rank {rank}/{world}] local_gpu={local} dev={dev} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

man = torch.load(args.manifest, map_location='cpu', weights_only=False)
rides = man['rides'] if isinstance(man, dict) and 'rides' in man else man
print(f"[rank {rank}] manifest rides: {len(rides)}", flush=True)

ds = ZarrRideDataset.from_manifest(rides_data=rides, motion_root=args.motion_root,
                                   ss_vae_checkpoint=args.ss_vae, device=dev, ss_vae_device=dev)
N = len(ds)

my = list(range(rank, N, world))
print(f"[rank {rank}] handling {len(my)} rides", flush=True)

out_windows = []
done = 0
for i in my:
    try:
        r = ds[i]
        zp = r['zarr_path']; nl = int(r['n_latent_frames'])
        if nl < args.window:
            continue
        z = ds.encode_z_actions_window(zp, nl, 0, nl)   # [nl, 8]
        z2 = z[:, 2].float().cpu().numpy()   # steering / yaw
        z7 = z[:, 7].float().cpu().numpy()   # throttle (+fwd / -rev)
        for s in range(0, nl - args.window + 1, args.stride):
            f2 = z2[s + args.cf : s + args.window]
            f7 = z7[s + args.cf : s + args.window]
            if f7.size == 0:
                continue
            turn = float(np.abs(f2).mean()); turn_var = float(f2.std())
            motion = float(np.abs(f7).mean()); m7 = float(f7.mean())
            score = turn + 0.7 * motion + 0.5 * turn_var
            out_windows.append({
                "zarr_path": zp, "start": int(s), "n_latent_frames": nl,
                "turn": round(turn, 4), "turn_var": round(turn_var, 4),
                "motion": round(motion, 4), "score": round(score, 4),
                "mean_z7_future": round(m7, 4),
                "backward": bool(m7 < args.bwd_thr),
            })
        done += 1
        if done % 20 == 0:
            print(f"[rank {rank}] {done}/{len(my)} rides, {len(out_windows)} windows", flush=True)
    except Exception as e:
        print(f"[rank {rank}] ride {i} failed: {e}", flush=True)
        continue

os.makedirs(args.outdir, exist_ok=True)
outp = os.path.join(args.outdir, f"shard_{rank:03d}.json")
with open(outp, "w") as f:
    json.dump({"rank": rank, "world": world, "n_rides": done,
               "window": args.window, "cf": args.cf, "stride": args.stride,
               "bwd_thr": args.bwd_thr, "n_windows": len(out_windows),
               "windows": out_windows}, f)
nb = sum(w["backward"] for w in out_windows)
print(f"[rank {rank}] DONE: {done} rides, {len(out_windows)} windows ({nb} backward) -> {outp}", flush=True)
