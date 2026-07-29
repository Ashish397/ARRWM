"""PAL4VST on blind100 (rerun to validate vs reference pal4vst_max/mean).
Per-video generated frames from ctx+1 (stride 8). Reuses pal_local.frame_score."""
import os
import numpy as np, imageio, pandas as pd
import blind100_common as bc
import pal_local as P

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "blind_pal.csv")


def gen_frames(path, ctx, stride=8):
    r = imageio.get_reader(path); n = r.count_frames()
    idx = list(range(ctx + 1, n - 1, stride))[:14]
    out = [np.asarray(r.get_data(int(i))) for i in idx]
    r.close(); return out


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for r in bc.refs():
        s = [P.frame_score(f) for f in gen_frames(r["path"], r["ctx"])]
        med, mean, mx = float(np.median(s)), float(np.mean(s)), float(np.max(s))
        rows.append(dict(blind_id=r["blind_id"], vid=r["vid"], model=r["model"],
                         scene=r["scene"], pal_max=mx, pal_mean=mean, pal_median=med))
        print(f"[pal] {r['blind_id']} {r['vid']:20s} max={mx:.4f} mean={mean:.4f}", flush=True)
        pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[pal] wrote {OUT}")


if __name__ == "__main__":
    main()
