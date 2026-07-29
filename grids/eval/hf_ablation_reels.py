"""HF-degradation (Laplacian-variance drift B) validation reel on OUR ablation variants only
(external models do not have this failure mode). Reads the existing ablation HF data in
results_hazebase.csv (B = -x_blur), shows the real context frame then generated frames across the
6s horizon so the sharpness wash-out over the rollout is visible.
Usage: python hf_ablation_reels.py <seed>"""
import os, sys
import numpy as np, cv2, pandas as pd
import fleet_common as fc

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 3
OUT = os.path.join(HERE, "out", "hf_validation_internal.png")
PANW = 240


def strip(scene, model, B, degraded):
    n, fps = fc.meta(scene, model); ctx = fc.ctx_of(model)
    ref_i = max(0, ctx - 1)
    end = min(n - 1, ctx + int(round(6.0 * fps)))
    gidx = list(np.linspace(ctx + int(round(fps * 0.5)), end, 4).round().astype(int))
    frames = fc.frames_at(scene, model, [ref_i] + gidx)
    labs = ["REAL ctx"] + [f"gen {i}" for i in gidx]
    ps = []
    for f, lb in zip(frames, labs):
        im = cv2.cvtColor(cv2.resize(f, (PANW, int(PANW * f.shape[0] / f.shape[1]))), cv2.COLOR_RGB2BGR)
        h = np.full((18, im.shape[1], 3), 30, np.uint8)
        cv2.putText(h, lb, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
        ps.append(np.concatenate([h, im], 0))
    body = np.concatenate(ps, 1)
    col = (80, 80, 255) if degraded else (120, 255, 120)
    tag = "HF DEGRADED (detail washed out)" if degraded else "clean (detail preserved)"
    band = np.full((30, body.shape[1], 3), 22, np.uint8)
    cv2.putText(band, f"{model}_{scene}   B={B:.0f}  ->  {tag}", (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 1)
    return np.concatenate([band, body], 0)


def main():
    d = pd.read_csv(os.path.join(HERE, "results_hazebase.csv"))
    d["B"] = -d["x_blur"]
    d = d[~d.baseline_dirty.astype(bool)]                    # exclude lens-dirty sources
    d["model"] = "ours_" + d.variant.astype(str)
    hi_thr, lo_thr = d.B.quantile(0.90), d.B.quantile(0.35)
    rng = np.random.RandomState(SEED)
    hi = d[d.B >= hi_thr].sample(3, random_state=rng)
    lo = d[d.B <= lo_thr].sample(3, random_state=rng)
    rows = pd.concat([hi, lo]).sort_values("B", ascending=False)
    imgs = []
    for _, r in rows.iterrows():
        try:
            imgs.append(strip(r.grid, r.model, r.B, r.B >= hi_thr))
        except Exception as e:
            print("skip", r.model, r.grid, str(e)[:60])
    W = max(i.shape[1] for i in imgs)
    imgs = [cv2.copyMakeBorder(i, 0, 0, 0, W - i.shape[1], cv2.BORDER_CONSTANT, value=(22, 22, 22)) for i in imgs]
    cv2.imwrite(OUT, np.concatenate(imgs, 0)); print("wrote", OUT)


if __name__ == "__main__":
    main()
