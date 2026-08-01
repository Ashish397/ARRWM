"""Item 3 + 3.1: quantify backward (reverse) coverage in the WEU dataset AND
harvest a manifest of backward-dominated training windows that can be mixed
into weunz to boost reverse coverage for v15.

Signal: PCA action component (throttle), sign convention + = forward, - = reverse.
A "backward window" = a 24-latent training window (3 context + 21) whose FUTURE
(the 21 generated latents) has mean z7 below --thr.

Writes:
  assets/backward_windows.json  -- list of {zarr_path, start, n_latent_frames, mean_z7_future}
  prints the global z7 distribution + per-threshold backward fractions.
"""
import os
import sys, os, json, argparse
_ROOT = os.environ.get("AF_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)
import numpy as np
import torch; torch.set_num_threads(4)
from utils.zarr_dataset import ZarrRideDataset

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", default="logs/v14_balanced_weunz/.ride_manifest.pt")
ap.add_argument("--motion_root", default=os.path.join(os.environ.get("DATA_ROOT", ""), "frodobots_motion"))
ap.add_argument("--pca_basis", default="preprocessing/checkpoints/pca_basis.pt")
ap.add_argument("--window", type=int, default=24)   # context(3)+chunk(21)
ap.add_argument("--cf", type=int, default=3)
ap.add_argument("--stride", type=int, default=3)
ap.add_argument("--thr", type=float, default=-0.15)  # mean future z7 below this = backward window
ap.add_argument("--out", default="assets/backward_windows.json")
args = ap.parse_args()

man = torch.load(args.manifest, map_location='cpu', weights_only=False)
rides = man['rides'] if isinstance(man, dict) and 'rides' in man else man
print(f"manifest rides: {len(rides)}")
ds = ZarrRideDataset.from_manifest(rides_data=rides, motion_root=args.motion_root,
                                   pca_basis_checkpoint=args.pca_basis, device='cpu')
N = len(ds)

all_z7 = []
back_windows = []
interesting_by_ride = {}   # zarr -> best high-motion/turning window
n_windows_total = 0
ok = 0
for i in range(N):
    try:
        r = ds[i]
        zp = r['zarr_path']; nl = int(r['n_latent_frames'])
        if nl < args.window:
            continue
        z = ds.encode_z_actions_window(zp, nl, 0, nl)   # [nl, 8] (motion-capped internally if needed)
        z2 = z[:, 2].float().numpy()   # legacy slot 2
        z7 = z[:, 7].float().numpy()   # legacy slot 7
        all_z7.append(z7)
        ok += 1
        for s in range(0, nl - args.window + 1, args.stride):
            f2 = z2[s + args.cf : s + args.window]   # future turning
            f7 = z7[s + args.cf : s + args.window]   # future throttle
            if f7.size == 0:
                continue
            n_windows_total += 1
            m7 = float(f7.mean())
            if m7 < args.thr:
                back_windows.append({"zarr_path": zp, "start": int(s),
                                     "n_latent_frames": nl, "mean_z7_future": round(m7, 4)})
            # interesting = high turning (|z2|) and/or high motion (|z7|), with
            # within-window VARIATION (std) so the clip actually moves/turns.
            turn = float(np.abs(f2).mean()); turn_var = float(f2.std())
            motion = float(np.abs(f7).mean())
            score = turn + 0.7 * motion + 0.5 * turn_var   # reward turning + motion + change
            cur = interesting_by_ride.get(zp)
            if cur is None or score > cur["score"]:
                interesting_by_ride[zp] = {"zarr_path": zp, "start": int(s),
                                           "n_latent_frames": nl, "turn": round(turn, 3),
                                           "turn_var": round(turn_var, 3), "motion": round(motion, 3),
                                           "score": round(score, 3)}
    except Exception as e:
        continue

z7 = np.concatenate(all_z7) if all_z7 else np.array([0.0])
print(f"\nrides analyzed: {ok}/{N} | latent frames: {len(z7)} | candidate windows: {n_windows_total}")
print(f"z7 (throttle; +fwd/-rev): mean={z7.mean():+.3f} median={np.median(z7):+.3f} std={z7.std():.3f} min={z7.min():+.3f} max={z7.max():+.3f}")
for thr in [0.0, -0.05, -0.1, -0.15, -0.2, -0.3]:
    print(f"  frac latents z7 < {thr:+.2f} (backward): {100*(z7 < thr).mean():.2f}%")
print(f"  frac |z7| < 0.05 (~stationary): {100*(np.abs(z7) < 0.05).mean():.2f}%")
print(f"\nBACKWARD WINDOWS (mean future z7 < {args.thr}): {len(back_windows)} of {n_windows_total} "
      f"({100*len(back_windows)/max(n_windows_total,1):.2f}%)")
rides_with_back = len(set(w['zarr_path'] for w in back_windows))
print(f"  spanning {rides_with_back} distinct rides")

hist, edges = np.histogram(z7, bins=np.linspace(-1, 1, 21))
print("\nz7 histogram:")
for h, lo in zip(hist, edges[:-1]):
    print(f"  {lo:+.1f}: {'#'*int(60*h/max(hist.max(),1))} {100*h/len(z7):.1f}%")

os.makedirs(os.path.dirname(args.out), exist_ok=True)
with open(args.out, "w") as f:
    json.dump({"thr": args.thr, "window": args.window, "cf": args.cf, "stride": args.stride,
               "n_backward_windows": len(back_windows), "n_candidate_windows": n_windows_total,
               "windows": back_windows}, f, indent=1)
print(f"\nwrote {args.out}  ({len(back_windows)} backward windows)")

# Interesting (high turning/motion) windows, best-per-ride, sorted by score —
# for diverse, non-boring AR-rollout demo videos.
interesting = sorted(interesting_by_ride.values(), key=lambda w: -w["score"])
iout = "assets/interesting_windows.json"
with open(iout, "w") as f:
    json.dump({"n_rides": len(interesting), "windows": interesting}, f, indent=1)
print(f"wrote {iout}  ({len(interesting)} ride-diverse candidates; top score {interesting[0]['score'] if interesting else 'NA'})")
print("top 12 interesting windows (turn / motion / score):")
for w in interesting[:12]:
    print(f"  {os.path.basename(w['zarr_path'])} s{w['start']}  turn={w['turn']} motion={w['motion']} score={w['score']}")
