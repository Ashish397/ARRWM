"""Scene-relocation (ORB+RANSAC inliers) validation reels on random fleet rollouts.
Low inliers = relocated to a different place; high = same scene preserved. Deployed
flag: <50 inliers. Clean 6s-capped sampling, real context frame first.
Usage: python scene_fleet_reels.py <seed> <tag> <int|ext|all>"""
import os, sys
import numpy as np, cv2, pandas as pd
import fleet_common as fc

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 7
TAG = sys.argv[2] if len(sys.argv) > 2 else ""
MODE = sys.argv[3] if len(sys.argv) > 3 else "all"
OUT = os.path.join(HERE, "out", f"scene_validation_fleet{TAG}.png")
PANW = 320


def strip(scene, model, inl, relocated):
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
    col = (80, 80, 255) if relocated else (120, 255, 120)
    tag = "RELOCATED (few inliers)" if relocated else "in-scene (many inliers)"
    band = np.full((30, body.shape[1], 3), 22, np.uint8)
    cv2.putText(band, f"{model}_{scene}   ORB inliers={int(inl)}  ->  {tag}", (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 1)
    return np.concatenate([band, body], 0)


def main():
    d = pd.read_csv(os.path.join(HERE, "fleet_scene_consensus.csv"))
    if MODE == "int":
        d = d[d.model.str.startswith("ours_")]
    elif MODE == "ext":
        d = d[~d.model.str.startswith("ours_")]
    rng = np.random.RandomState(SEED)
    relo = d[d.consensus_inl < 50].sample(3, random_state=rng)         # deployed relocated threshold
    keep = d[d.consensus_inl > 200].sample(3, random_state=rng)        # clearly in-scene
    rows = pd.concat([relo, keep]).sort_values("consensus_inl")
    imgs = []
    for _, r in rows.iterrows():
        try:
            imgs.append(strip(r.scene, r.model, r.consensus_inl, r.consensus_inl < 50))
        except Exception as e:
            print("skip", r.model, r.scene, str(e)[:50])
    W = max(i.shape[1] for i in imgs)
    imgs = [cv2.copyMakeBorder(i, 0, 0, 0, W - i.shape[1], cv2.BORDER_CONSTANT, value=(22, 22, 22)) for i in imgs]
    cv2.imwrite(OUT, np.concatenate(imgs, 0)); print("wrote", OUT)


if __name__ == "__main__":
    main()
