"""Conjuration (sudden-object-appearance) validation reels on random fleet rollouts.
High novel_sudden = an object materialised discontinuously between consecutive frames
(confabulation); low = scene evolves smoothly, no pop-in. No AUC for this axis - it is
validated visually, so we sample a DENSE strip (8 generated frames over the 6s window)
so the discontinuous appearance is catchable between adjacent panels.
Usage: python conjuration_fleet_reels.py <seed> <tag> <int|ext|all>"""
import os, sys
import numpy as np, cv2, pandas as pd
import fleet_common as fc

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
MODE = sys.argv[3] if len(sys.argv) > 3 else "all"
OUT = os.path.join(HERE, "out", f"conjuration_validation_fleet{TAG}.png")
PANW = 220
NGEN = 8


def strip(scene, model, nsud, nsp, flag):
    n, fps = fc.meta(scene, model); ctx = fc.ctx_of(model)
    ref_i = max(0, ctx - 1)
    end = min(n - 1, ctx + int(round(6.0 * fps)))
    gidx = list(np.linspace(ctx, end, NGEN).round().astype(int))
    frames = fc.frames_at(scene, model, [ref_i] + gidx)
    labs = ["REAL ctx"] + [f"gen {i}" for i in gidx]
    ps = []
    for f, lb in zip(frames, labs):
        im = cv2.cvtColor(cv2.resize(f, (PANW, int(PANW * f.shape[0] / f.shape[1]))), cv2.COLOR_RGB2BGR)
        h = np.full((16, im.shape[1], 3), 30, np.uint8)
        cv2.putText(h, lb, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1)
        ps.append(np.concatenate([h, im], 0))
    body = np.concatenate(ps, 1)
    col = (80, 80, 255) if flag else (120, 255, 120)
    tag = "CONJURED (sudden new object)" if flag else "clean (no pop-in)"
    band = np.full((30, body.shape[1], 3), 22, np.uint8)
    cv2.putText(band, f"{model}_{scene}   novel_sudden={nsud:.2f}  n_spikes={int(nsp)}  ->  {tag}", (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 1)
    return np.concatenate([band, body], 0)


def main():
    d = pd.read_csv(os.path.join(HERE, "out", "fleet_novelty.csv"))
    if MODE == "int":
        d = d[d.model.str.startswith("ours_")]
    elif MODE == "ext":
        d = d[~d.model.str.startswith("ours_")]
    rng = np.random.RandomState(SEED)
    hi = d[d.novel_sudden >= 0.9].sample(3, random_state=rng)      # clear conjuration
    lo = d[d.novel_sudden <= 0.05].sample(3, random_state=rng)     # clearly clean
    rows = pd.concat([hi, lo]).sort_values("novel_sudden", ascending=False)
    imgs = []
    for _, r in rows.iterrows():
        try:
            imgs.append(strip(r.scene, r.model, r.novel_sudden, r.n_spikes, r.novel_sudden >= 0.9))
        except Exception as e:
            print("skip", r.model, r.scene, str(e)[:50])
    W = max(i.shape[1] for i in imgs)
    imgs = [cv2.copyMakeBorder(i, 0, 0, 0, W - i.shape[1], cv2.BORDER_CONSTANT, value=(22, 22, 22)) for i in imgs]
    cv2.imwrite(OUT, np.concatenate(imgs, 0)); print("wrote", OUT)


if __name__ == "__main__":
    main()
