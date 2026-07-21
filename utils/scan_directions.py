"""Per-direction chunk census for dataset-imbalance reporting.

For every motion.npy, project PC0 (throttle: +fwd/-back) and PC1 (steer: +right/-left)
and classify each chunk. Reports for the 14e TRAINING rides (and whole dataset for
context), at mild (z>0.1) and clear (z>0.3) thresholds:
  marginal : forward / backward / right / left  (axis in that dir, ignoring the other)
  pure     : only ONE axis active (pure F/B/R/L)
  entangled: BOTH throttle & steer active (FR/FL/BR/BL)  <- how coupled the actions are
  stationary
"""
import os, glob, json, math
import numpy as np, torch
from multiprocessing import Pool

ck = torch.load("action_query/checkpoints/ss_vae_8free.pt", map_location="cpu")
MEAN = ck["pca_mean"].numpy().astype(np.float32)
COMP = ck["pca_comp"].numpy()                    # [16,200]
C0, C1 = COMP[0].astype(np.float32), COMP[1].astype(np.float32)
try:
    from utils.zarr_dataset import _PCA_RAW_SCALES
    S0, S1 = float(_PCA_RAW_SCALES[0]), float(_PCA_RAW_SCALES[1])
except Exception:
    S0, S1 = 93.7, 57.7
ROOT = "/projects/u6ex/fbots/frodobots_motion"
THS = [0.1, 0.3]
PTH = {th: (S0 * math.atanh(th), S1 * math.atanh(th)) for th in THS}   # (throttle_P, steer_P)
OUT = "analysis/eval_final/direction_census.json"

_tw = json.load(open("paper_assets/v14d_train_windows.json"))
TRAIN = set(os.path.basename(x["zarr_path"]).replace(".zarr", "") for x in _tw["windows"])
KEYS = ["n", "fwd", "bwd", "right", "left", "pureF", "pureB", "pureR", "pureL",
        "FR", "FL", "BR", "BL", "entangled", "stat"]


def base_of(path):
    d = os.path.basename(os.path.dirname(path))
    return d.split("_", 2)[-1] if d.startswith("ride_") else d


def scan(f):
    try:
        m = np.load(f, mmap_mode="r")
        if m.shape[1] * 2 != 200:
            return None
        flat = np.asarray(m[:, :, :2]).reshape(m.shape[0], -1).astype(np.float32) - MEAN
        P0 = flat @ C0; P1 = flat @ C1                       # throttle, steer (raw PCA)
        out = {"base": base_of(f), "intrain": int(base_of(f) in TRAIN)}
        for th in THS:
            t0, t1 = PTH[th]
            fwd, bwd = P0 > t0, P0 < -t0
            right, left = P1 > t1, P1 < -t1
            tA, sA = np.abs(P0) > t0, np.abs(P1) > t1        # throttle/steer active
            ent = tA & sA
            d = dict(n=len(P0),
                     fwd=int(fwd.sum()), bwd=int(bwd.sum()), right=int(right.sum()), left=int(left.sum()),
                     pureF=int((fwd & ~sA).sum()), pureB=int((bwd & ~sA).sum()),
                     pureR=int((right & ~tA).sum()), pureL=int((left & ~tA).sum()),
                     FR=int((fwd & right).sum()), FL=int((fwd & left).sum()),
                     BR=int((bwd & right).sum()), BL=int((bwd & left).sum()),
                     entangled=int(ent.sum()), stat=int((~tA & ~sA).sum()))
            out[f"th{th}"] = d
        return out
    except Exception:
        return None


def main():
    files = glob.glob(f"{ROOT}/**/motion.npy", recursive=True)
    print(f"[dir] {len(files)} motion.npy | throttle scale {S0}, steer scale {S1}", flush=True)
    with Pool(int(os.environ.get("SBF_WORKERS", "64"))) as p:
        res = [r for r in p.imap_unordered(scan, files, chunksize=16) if r]
    print(f"[dir] scanned {len(res)} rides\n")

    def report(rows, label):
        agg = {}
        for th in THS:
            a = {k: sum(r[f"th{th}"][k] for r in rows) for k in KEYS}
            agg[f"th{th}"] = a
            N = a["n"]
            def pc(x): return f"{x:>9,} ({100*x/max(1,N):5.2f}%)"
            print(f"=== {label}  (TH z>{th}) — {N:,} total chunks, {len(rows):,} rides ===")
            print(f"  MARGINAL   forward {pc(a['fwd'])}   backward {pc(a['bwd'])}")
            print(f"             right   {pc(a['right'])}   left     {pc(a['left'])}")
            print(f"  PURE       F {pc(a['pureF'])}  B {pc(a['pureB'])}")
            print(f"             R {pc(a['pureR'])}  L {pc(a['pureL'])}")
            print(f"  ENTANGLED  total {pc(a['entangled'])}  (FR {a['FR']:,}  FL {a['FL']:,}  BR {a['BR']:,}  BL {a['BL']:,})")
            print(f"  STATIONARY {pc(a['stat'])}\n")
        return agg

    intr = [r for r in res if r["intrain"]]
    out = {"train": report(intr, "14e TRAINING rides"),
           "all": report(res, "WHOLE DATASET")}
    json.dump(out, open(OUT, "w"), indent=1)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
