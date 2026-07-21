"""FAST backward census: read every motion.npy directly, project PC0 (throttle)
through the PCA, count backward. NO ride manifest / captions / zarr. Splits by
in-training vs not, and by motion group, to locate where backward footage lives.

motion.npy: [n_chunks, 100, 3] (dx,dy,vis). PC0>0 = forward, PC0<0 = backward
(same convention Phase A validated: F->+tz2, B->-tz2). Window grain = 9 chunks (=tot_f 27 latents).
"""
import os, glob, json, math
import numpy as np, torch
from multiprocessing import Pool

ck = torch.load("action_query/checkpoints/ss_vae_8free.pt", map_location="cpu")
MEAN = ck["pca_mean"].numpy().astype(np.float32)          # [200]
COMP0 = ck["pca_comp"].numpy()[0].astype(np.float32)      # [200] PC0 loading
try:
    from utils.zarr_dataset import _PCA_RAW_SCALES
    S0 = float(_PCA_RAW_SCALES[0])
except Exception:
    S0 = 93.7
THS = [0.1, 0.3]
PTH = {th: S0 * math.atanh(th) for th in THS}             # z<-th  <=>  P0 < -PTH[th]
ROOT = "/projects/u6ex/fbots/frodobots_motion"
CHUNKS_PER_WIN = 9
OUT = "analysis/eval_final/backward_census_fast.json"

_tw = json.load(open("paper_assets/v14d_train_windows.json"))
TRAIN = set(os.path.basename(x["zarr_path"]).replace(".zarr", "") for x in _tw["windows"])


def ride_basename(path):
    # .../output_rides_X/ride_NNNN_YYYYMMDDhhmmss/motion.npy -> YYYYMMDDhhmmss
    d = os.path.basename(os.path.dirname(path))
    return d.split("_", 2)[-1] if d.startswith("ride_") else d


def scan(f):
    try:
        m = np.load(f, mmap_mode="r")
        if m.shape[1] * 2 != 200:
            return None
        flat = np.asarray(m[:, :, :2]).reshape(m.shape[0], -1).astype(np.float32)
        P0 = (flat - MEAN) @ COMP0                        # [n_chunks]
        n = len(P0)
        base = ride_basename(f)
        grp = f.split("/output_rides_")[1].split("/")[0] if "/output_rides_" in f else "?"
        r = dict(base=base, grp=grp, n=n, intrain=int(base in TRAIN), minP=float(P0.min()))
        for th in THS:
            r[f"cb{th}"] = int((P0 < -PTH[th]).sum())     # backward chunks
            r[f"cf{th}"] = int((P0 > PTH[th]).sum())      # forward chunks
        nw = n // CHUNKS_PER_WIN
        if nw:
            wmean = P0[:nw * CHUNKS_PER_WIN].reshape(nw, CHUNKS_PER_WIN).mean(1)
            for th in THS:
                r[f"wb{th}"] = int((wmean < -PTH[th]).sum())
                r[f"wf{th}"] = int((wmean > PTH[th]).sum())
            r["nw"] = nw
        else:
            r["nw"] = 0
            for th in THS:
                r[f"wb{th}"] = r[f"wf{th}"] = 0
        return r
    except Exception:
        return None


def main():
    files = glob.glob(f"{ROOT}/**/motion.npy", recursive=True)
    print(f"[fast] {len(files)} motion.npy files | PC0 thresholds P0<{-PTH[0.1]:.1f}(z.1) {-PTH[0.3]:.1f}(z.3)", flush=True)
    with Pool(int(os.environ.get("SBF_WORKERS", "64"))) as p:
        res = [r for r in p.imap_unordered(scan, files, chunksize=16) if r]
    print(f"[fast] scanned {len(res)} rides ok\n")

    def agg(rows, label):
        if not rows:
            print(f"{label}: (none)"); return {}
        d = {"rides": len(rows), "chunks": sum(r["n"] for r in rows), "wins": sum(r["nw"] for r in rows)}
        for th in THS:
            d[f"chunk_back_{th}"] = sum(r[f"cb{th}"] for r in rows)
            d[f"chunk_fwd_{th}"] = sum(r[f"cf{th}"] for r in rows)
            d[f"win_back_{th}"] = sum(r[f"wb{th}"] for r in rows)
            d[f"rides_with_back_{th}"] = sum(1 for r in rows if r[f"cb{th}"] > 0)
        print(f"=== {label}: {d['rides']} rides, {d['chunks']} chunks, {d['wins']} windows ===")
        for th in THS:
            cb, ch = d[f"chunk_back_{th}"], d["chunks"]
            wb, w = d[f"win_back_{th}"], d["wins"]
            print(f"   z<-{th}: chunks back={cb} ({100*cb/max(1,ch):.2f}%) | "
                  f"windows back={wb} ({100*wb/max(1,w):.2f}%) | rides w/ back={d[f'rides_with_back_{th}']}")
        return d

    alld = agg(res, "WHOLE DATASET (7590)")
    intr = agg([r for r in res if r["intrain"]], "IN TRAINING (weunz used)")
    outr = agg([r for r in res if not r["intrain"]], "NOT IN TRAINING (unused)")
    # per motion group
    print("\n=== backward windows (z<-0.1) per motion group ===")
    grps = {}
    for r in res:
        grps.setdefault(r["grp"], []).append(r)
    for g in sorted(grps):
        rows = grps[g]; wb = sum(r["wb0.1"] for r in rows)
        print(f"   output_rides_{g}: {len(rows)} rides, {wb} backward windows")
    # top backward rides
    top = sorted(res, key=lambda r: r["wb0.1"], reverse=True)[:15]
    print("\n=== top-15 rides by backward windows (z<-0.1) ===")
    for r in top:
        print(f"   {r['base']}  back_win={r['wb0.1']:4} back_chunk={r['cb0.1']:5} "
              f"minP0={r['minP']:.0f}  {'IN-TRAIN' if r['intrain'] else 'unused'}")

    json.dump({"all": alld, "in_train": intr, "not_train": outr,
               "top": [{k: r[k] for k in ('base', 'grp', 'intrain', 'wb0.1', 'cb0.1', 'minP')} for r in top]},
              open(OUT, "w"), indent=1)
    print(f"\nsaved {OUT}")


if __name__ == "__main__":
    main()
