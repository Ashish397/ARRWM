"""Build v14b's curated training pool from the weunz window scores.

Pipeline:
  1. Merge all score shards (paper_assets/window_scores_weunz/shard_*.json).
  2. Per ride, greedy non-max suppression so selected windows don't overlap
     (min start separation = --nms_sep, default = window_size 21, i.e. the
     same spacing the batcher's contiguous walk uses). This drops the
     near-duplicate overlapping windows the stride-3 scoring produced.
  3. The supervised pool = the MOST MOTION-Y windows: rank the NMS-survivors
     by score (turn + 0.7*motion + 0.5*turn_var) and keep the top --target_pool
     (dropping anything below --motion_min so stationary/idle never enters).
  4. Add the BACKWARD oversampler: all NMS-survivor backward windows
     (mean future z7 < bwd_thr), duplicated x--bwd_oversample so reverse
     egomotion reaches a healthy fraction of the pool.

Output: paper_assets/v14b_train_windows.json
  {"windows": [{"zarr_path", "start", "score", "motion", "backward"}, ...], ...}
Each duplicate of a backward window is a separate entry (the trainer maps each
to a forced-offset training sample; DDP shuffling spreads the copies).
"""
import sys, os, json, glob, argparse
sys.path.insert(0, '/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM')
os.chdir('/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM')
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--shards_dir", default="paper_assets/window_scores_weunz")
ap.add_argument("--out", default="paper_assets/v14b_train_windows.json")
ap.add_argument("--nms_sep", type=int, default=21, help="min start separation between kept windows of a ride")
ap.add_argument("--target_pool", type=int, default=60000, help="number of forward/high-motion windows to keep")
ap.add_argument("--motion_min", type=float, default=0.30, help="drop windows with motion (|z7|) below this (stationary)")
ap.add_argument("--bwd_oversample", type=int, default=6, help="duplicate each backward window this many times")
args = ap.parse_args()

shards = sorted(glob.glob(os.path.join(args.shards_dir, "shard_*.json")))
if not shards:
    raise SystemExit(f"no shards in {args.shards_dir} (has the scoring job finished?)")
W = []
for s in shards:
    d = json.load(open(s))
    W += d["windows"]
print(f"merged {len(shards)} shards -> {len(W)} scored windows (stride-3, overlapping)", flush=True)

# --- per-ride greedy NMS (highest score first, enforce min start separation) --
by_ride = {}
for w in W:
    by_ride.setdefault(w["zarr_path"], []).append(w)
kept = []
for zp, ws in by_ride.items():
    ws.sort(key=lambda w: -w["score"])
    chosen_starts = []
    for w in ws:
        s = w["start"]
        if all(abs(s - cs) >= args.nms_sep for cs in chosen_starts):
            chosen_starts.append(s)
            kept.append(w)
print(f"after per-ride NMS (sep={args.nms_sep}): {len(kept)} distinct windows across {len(by_ride)} rides", flush=True)

bwd = [w for w in kept if w["backward"]]
fwd = [w for w in kept if not w["backward"]]

# --- most-motion-y forward pool: drop stationary, keep top-N by score ---------
fwd = [w for w in fwd if w["motion"] >= args.motion_min]
fwd.sort(key=lambda w: -w["score"])
fwd_pool = fwd[: args.target_pool]
if fwd_pool:
    sc = np.array([w["score"] for w in fwd_pool]); mo = np.array([w["motion"] for w in fwd_pool])
    print(f"forward pool: {len(fwd_pool)} windows | score>= {sc.min():.3f} | motion med {np.median(mo):.3f}", flush=True)

# --- backward oversampler -----------------------------------------------------
bwd_pool = []
for w in bwd:
    bwd_pool += [w] * args.bwd_oversample
print(f"backward: {len(bwd)} distinct windows x{args.bwd_oversample} = {len(bwd_pool)} entries", flush=True)

pool = fwd_pool + bwd_pool
out_windows = [{"zarr_path": w["zarr_path"], "start": int(w["start"]),
                "n_latent_frames": int(w.get("n_latent_frames", 0)),
                "score": w["score"], "motion": w["motion"], "backward": w["backward"]}
               for w in pool]
n_bwd = sum(w["backward"] for w in out_windows)
print(f"\nFINAL POOL: {len(out_windows)} entries  "
      f"({len(fwd_pool)} forward + {len(bwd_pool)} backward, backward frac {100*n_bwd/max(len(out_windows),1):.1f}%)", flush=True)

os.makedirs(os.path.dirname(args.out), exist_ok=True)
json.dump({
    "n_windows": len(out_windows),
    "n_forward": len(fwd_pool), "n_backward_entries": len(bwd_pool),
    "n_backward_distinct": len(bwd),
    "params": {"nms_sep": args.nms_sep, "target_pool": args.target_pool,
               "motion_min": args.motion_min, "bwd_oversample": args.bwd_oversample},
    "windows": out_windows,
}, open(args.out, "w"))
print(f"wrote {args.out}", flush=True)
