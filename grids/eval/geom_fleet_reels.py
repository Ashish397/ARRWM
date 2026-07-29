"""Geometry-probe validation on a RANDOM fleet sample (no human labels; shows the
metric generalizing). Own clean sampling: last real context frame first, then
generated frames starting ~0.5s in (skip the handoff transition). Annotated with
p_uncanny + flag. CPU only."""
import os, sys
import numpy as np, cv2, pandas as pd
import fleet_common as fc

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
OUT = os.path.join(HERE, "out", f"geometry_validation_fleet{TAG}.png")
FPS = {"astra": 20, "matrixgame": 16, "minwm": 16, "worldcam": 24, "worldplay": 16, "yume": 16}
PANW = 320


def fps_of(m):
    return FPS.get(m.replace("ours_", "") if not m.startswith("ours_") else "ours", 16)


def strip(scene, model, scores):
    """scores: dict {vlm_label: p_uncanny}. First entry (Qwen) drives the verdict colour."""
    n, fps = fc.meta(scene, model); ctx = fc.ctx_of(model)
    ref_i = max(0, ctx - 1)
    end = min(n - 1, ctx + int(round(6.0 * fps)))       # 6s horizon cap
    g0 = min(end - 1, ctx + int(round(fps * 0.5)))      # skip ~0.5s handoff
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
    vals = list(scores.values())
    unc = vals[0]                                       # Qwen drives colour
    disagree = (max(vals) > 0.5) and (min(vals) <= 0.5)  # VLMs disagree on the flag
    col = (0, 200, 255) if disagree else ((80, 80, 255) if unc > 0.5 else (120, 255, 120))
    txt = "   ".join(f"{k}={v:.2f}" for k, v in scores.items())
    tag = "  [VLMs DISAGREE]" if disagree else ("  -> FLAGGED" if unc > 0.5 else "  -> clean")
    band = np.full((30, body.shape[1], 3), 22, np.uint8)
    cv2.putText(band, f"{model}_{scene}   {txt}{tag}", (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
    return np.concatenate([band, body], 0)


def main():
    d = pd.read_csv(os.path.join(HERE, "results_external_vlm.csv"))
    MODE = sys.argv[3] if len(sys.argv) > 3 else "all"     # int | ext | all
    if MODE == "int":
        d = d[d.model.str.startswith("ours_")]
    elif MODE == "ext":
        d = d[~d.model.str.startswith("ours_")]
    rng = np.random.RandomState(SEED)
    hi = d[d.p_uncanny > 0.6].sample(3, random_state=rng)
    lo = d[d.p_uncanny < 0.15].sample(3, random_state=rng)
    rows = pd.concat([hi, lo]).sort_values("p_uncanny", ascending=False)
    # optional other-VLM scores for the same rollouts
    extra = {}
    for vlm, lbl in [("internvl3_8b", "InternVL"), ("cosmos_reason1_7b", "Cosmos")]:
        p = os.path.join(HERE, "out", f"reel_uncanny_{vlm}.csv")
        if os.path.exists(p):
            extra[lbl] = pd.read_csv(p).set_index(["scene", "model"]).p_uncanny.to_dict()
    imgs = []
    for _, r in rows.iterrows():
        try:
            scores = {"Qwen": r.p_uncanny}
            for lbl, m in extra.items():
                if (r.scene, r.model) in m:
                    scores[lbl] = m[(r.scene, r.model)]
            imgs.append(strip(r.scene, r.model, scores))
        except Exception as e:
            print("skip", r.model, r.scene, str(e)[:50])
    W = max(i.shape[1] for i in imgs)
    imgs = [cv2.copyMakeBorder(i, 0, 0, 0, W - i.shape[1], cv2.BORDER_CONSTANT, value=(22, 22, 22)) for i in imgs]
    cv2.imwrite(OUT, np.concatenate(imgs, 0)); print("wrote", OUT)


if __name__ == "__main__":
    main()
