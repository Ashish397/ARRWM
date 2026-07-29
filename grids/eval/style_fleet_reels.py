"""Style-shift (DINOv2 drift) validation reels on random fleet rollouts. High drift =
style shift (painted/game-like/re-rendered), low drift = clean. Clean sampling: real
context frame first, generated frames start ~0.5s in, 6s horizon cap.
Usage: python style_fleet_reels.py <seed> <tag> <int|ext|all>"""
import os, sys
import numpy as np, cv2, pandas as pd
import fleet_common as fc

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
MODE = sys.argv[3] if len(sys.argv) > 3 else "all"
OUT = os.path.join(HERE, "out", f"style_validation_fleet{TAG}.png")
PANW = 320


def strip(scene, model, drift, flag):
    n, fps = fc.meta(scene, model); ctx = fc.ctx_of(model)
    ref_i = max(0, ctx - 1)
    end = min(n - 1, ctx + int(round(6.0 * fps)))
    g0 = min(end - 1, ctx + int(round(fps * 0.5)))
    gidx = list(np.linspace(g0, end, 4).round().astype(int))
    frames = fc.frames_at(scene, model, [ref_i] + gidx)
    labs = ["REAL ctx"] + [f"gen {i}" for i in gidx]
    ps = []
    for f, lb in zip(frames, labs):
        im = cv2.cvtColor(cv2.resize(f, (PANW, int(PANW * f.shape[0] / f.shape[1]))), cv2.COLOR_RGB2BGR)
        h = np.full((18, im.shape[1], 3), 30, np.uint8)
        cv2.putText(h, lb, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
        ps.append(np.concatenate([h, im], 0))
    body = np.concatenate(ps, 1)
    col = (80, 80, 255) if flag else (120, 255, 120)
    tag = "STYLE SHIFT (high drift)" if flag else "clean (low drift)"
    band = np.full((30, body.shape[1], 3), 22, np.uint8)
    cv2.putText(band, f"{model}_{scene}   dino_drift={drift:.3f}  ->  {tag}", (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 1)
    return np.concatenate([band, body], 0)


def main():
    d = pd.read_csv(os.path.join(HERE, "out", "fleet_dino.csv"))
    if MODE == "int":
        d = d[d.model.str.startswith("ours_")]
    elif MODE == "ext":
        d = d[~d.model.str.startswith("ours_")]
    hi_thr, lo_thr = d.dino_drift.quantile(0.85), d.dino_drift.quantile(0.30)
    rng = np.random.RandomState(SEED)
    hi = d[d.dino_drift >= hi_thr].sample(3, random_state=rng)
    lo = d[d.dino_drift <= lo_thr].sample(3, random_state=rng)
    rows = pd.concat([hi, lo]).sort_values("dino_drift", ascending=False)
    imgs = []
    for _, r in rows.iterrows():
        try:
            imgs.append(strip(r.scene, r.model, r.dino_drift, r.dino_drift >= hi_thr))
        except Exception as e:
            print("skip", r.model, r.scene, str(e)[:50])
    W = max(i.shape[1] for i in imgs)
    imgs = [cv2.copyMakeBorder(i, 0, 0, 0, W - i.shape[1], cv2.BORDER_CONSTANT, value=(22, 22, 22)) for i in imgs]
    cv2.imwrite(OUT, np.concatenate(imgs, 0)); print("wrote", OUT)


if __name__ == "__main__":
    main()
