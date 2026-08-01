"""Select non-moving windows from UNSEEN rides for the two final evals.

Unseen = rides in frodobots_encoded whose basename was never in the 63,792-window
training manifest (and not an eval ride). We build a manifest (prompt_embeds +
n_latent_frames) for them, then scan windows measuring:
  ego   = mean over window of hypot(z0,z1)  (coherent throttle/steer = camera motion)
  scene = mean over window of load_motion_magnitudes (per-point activity = actors)

Pick:
  Phase A (action-injection): 32 NON-MOVING windows, one per distinct ride (diverse).
  Phase B (static rendering) : 64 NON-MOVING windows stratified by scene-change into
          4 bins (small/medium/big/massive), 16 each, distinct rides where possible.

Outputs (analysis/eval_final/):
  manifest_unseen.pt      - list of ride dicts actually used (seed source for generation)
  phaseA_windows.json     - [{ride, zarr_path, offset, ego, scene}] x32
  phaseB_windows.json     - [{ride, zarr_path, offset, ego, scene, bin}] x64
"""
import os, json
os.environ.setdefault("ARRWM_ACTION_ENCODER", "pca_raw")
import numpy as np, torch
from omegaconf import OmegaConf
from utils.zarr_dataset import build_ride_manifest, ZarrRideDataset

cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                      OmegaConf.load("configs/causal_lora_diffusion_teacher_v14e_16node.yaml"))
NFB = int(cfg.num_frame_per_block)                 # 3
GEN = int(cfg.get("control_gen_chunks", 8))        # 8
TOT_F = (1 + GEN) * NFB                             # 27
UNSEEN_ROOT = os.path.join(os.environ.get("DATA_ROOT", ""), "frodobots_encoded")
CAP = str(cfg.caption_root); MOT = str(cfg.motion_root)
CK = "preprocessing/checkpoints/pca_basis.pt"
OUT = "analysis/eval_final"; os.makedirs(OUT, exist_ok=True)
K_STARTS = 8            # window starts sampled per ride
NONMOVE_PCTL = 25       # ego percentile defining the "non-moving" pool
EGO_HARD = 0.12         # absolute ego ceiling for non-moving (squashed z units)


def main():
    tw = json.load(open("assets/train_windows.json"))
    train = set(os.path.basename(x["zarr_path"]) for x in tw["windows"])
    evalr = set(cfg.get("eval_ride_zarrs", []) or [])

    rides = build_ride_manifest(UNSEEN_ROOT, CAP, motion_root=MOT,
                                cache_path=f"{OUT}/.manifest_full.pt")
    unseen = [r for r in rides
              if os.path.basename(r["zarr_path"]) not in train
              and os.path.basename(r["zarr_path"]) not in evalr]
    print(f"[pool] {len(rides)} rides w/ caption+motion in {UNSEEN_ROOT}; {len(unseen)} UNSEEN")
    if len(unseen) < 64:
        print("[warn] fewer unseen rides than needed; will reuse rides across windows")

    ds = ZarrRideDataset.from_manifest(rides_data=unseen, motion_root=MOT, pca_basis_checkpoint=CK)

    cands = []   # dicts: ride, zarr_path, offset, ego, scene
    for r in unseen:
        zp, nlat = r["zarr_path"], int(r["n_latent_frames"])
        if nlat < TOT_F + NFB:
            continue
        try:
            mag = ds.load_motion_magnitudes(zp, nlat)              # [nlat]
        except Exception as e:
            continue
        starts = np.linspace(0, nlat - TOT_F, K_STARTS).astype(int)
        for off in sorted(set(int(s) for s in starts)):
            try:
                z = ds.encode_z_actions_window(zp, nlat, off, off + TOT_F).numpy()  # [TOT_F,8]
            except Exception:
                continue
            ego = float(np.hypot(z[:, 0], z[:, 1]).mean())
            scene = float(mag[off:off + TOT_F].mean())
            cands.append(dict(ride=os.path.basename(zp), zarr_path=zp, offset=int(off),
                              ego=round(ego, 4), scene=round(scene, 4)))
    print(f"[scan] {len(cands)} candidate windows over {len(unseen)} rides")
    if not cands:
        raise SystemExit("no candidates")

    egos = np.array([c["ego"] for c in cands])
    thr = min(EGO_HARD, float(np.percentile(egos, NONMOVE_PCTL)))
    nonmove = [c for c in cands if c["ego"] <= thr]
    print(f"[nonmove] ego<= {thr:.3f} -> {len(nonmove)} windows "
          f"(ego range {egos.min():.3f}..{egos.max():.3f})")

    # ---- Phase A: 32 most-non-moving, one per distinct ride ----
    seen, phaseA = set(), []
    for c in sorted(nonmove, key=lambda c: c["ego"]):
        if c["ride"] in seen:
            continue
        seen.add(c["ride"]); phaseA.append(c)
        if len(phaseA) == 32:
            break
    if len(phaseA) < 32:  # fall back to allowing repeat rides
        for c in sorted(nonmove, key=lambda c: c["ego"]):
            if c in phaseA:
                continue
            phaseA.append(c)
            if len(phaseA) == 32:
                break

    # ---- Phase B: 64 non-moving stratified by scene into 4 bins x16 ----
    scenes = np.array([c["scene"] for c in nonmove])
    q = np.quantile(scenes, [0.25, 0.5, 0.75])
    def binof(s): return int(np.digitize(s, q))         # 0..3 = small/med/big/massive
    BINNAMES = ["small", "medium", "big", "massive"]
    by_bin = {b: [] for b in range(4)}
    for c in sorted(nonmove, key=lambda c: c["scene"]):
        by_bin[binof(c["scene"])].append(c)
    phaseB = []
    for b in range(4):
        pool = by_bin[b]; seenb = set(); picked = []
        for c in pool:                                  # prefer distinct rides
            if c["ride"] in seenb:
                continue
            seenb.add(c["ride"]); picked.append({**c, "bin": BINNAMES[b]})
            if len(picked) == 16:
                break
        i = 0
        while len(picked) < 16 and i < len(pool):       # backfill w/ repeats
            c = pool[i]; i += 1
            if not any(p["offset"] == c["offset"] and p["ride"] == c["ride"] for p in picked):
                picked.append({**c, "bin": BINNAMES[b]})
        phaseB += picked
        print(f"[phaseB] bin {BINNAMES[b]:7} (scene~[{pool[0]['scene'] if pool else 0:.2f}.."
              f"{pool[-1]['scene'] if pool else 0:.2f}]): {len(picked)} windows")

    json.dump(phaseA, open(f"{OUT}/phaseA_windows.json", "w"), indent=1)
    json.dump(phaseB, open(f"{OUT}/phaseB_windows.json", "w"), indent=1)
    used = {c["zarr_path"] for c in phaseA} | {c["zarr_path"] for c in phaseB}
    torch.save([r for r in unseen if r["zarr_path"] in used], f"{OUT}/manifest_unseen.pt")
    print(f"[done] phaseA={len(phaseA)} phaseB={len(phaseB)} | "
          f"{len(used)} distinct rides -> {OUT}/manifest_unseen.pt")


if __name__ == "__main__":
    main()
