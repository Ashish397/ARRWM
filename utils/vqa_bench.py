"""Off-the-shelf VIDEO quality / artifact models on r08 (BL + B), vs our ground truth.

1. DOVER (ICCV'23) technical + aesthetic branches — the standard learned NR-VQA.
   Run via its own evaluate_one_video.py logic (imported), weights from repo release.
2. VBench-style dimensions, reimplemented minimally to avoid the full harness:
   - motion_smoothness: optical-flow acceleration = || flow(t->t+1) - flow(t-1->t) ||
     mean over pixels/frames (VBench: lower accel = smoother; melt morphs -> jerky local flow).
   - temporal_flickering: mean |frame_t - frame_{t+1}| on low-motion regions.
   - subject_consistency-style: mean DINO CLS cosine between consecutive frames.

Validation: BL clean={pca8_8node,4node}; B: 16node worst AND pca8 mid/low.
Saves analysis/eval_final/vqa_bench_r08.csv
"""
import os, sys, subprocess
import numpy as np, pandas as pd, imageio, torch, cv2
import torch.nn.functional as F

DEV = "cuda" if torch.cuda.is_available() else "cpu"
RUNS = ["pca8_8node", "pca4", "pca2", "16node", "4node", "noatok"]
CLEAN_BL = {"pca8_8node", "4node"}
OUTD = "analysis/eval_final"


def vid_path(run, br):
    return f"logs/eval_final/A/{run}/control_test/step05000_r08_{br}_raw.mp4"


# ---------- DOVER via its CLI (simplest robust integration) ----------
def dover_scores(path):
    try:
        out = subprocess.run(
            [sys.executable, "evaluate_one_video.py", "-v", os.path.abspath(path)],
            capture_output=True, text=True, timeout=300,
            cwd=os.path.abspath("third_party/DOVER"))
        txt = out.stdout + out.stderr
        # DOVER prints normalized aesthetic/technical/overall; parse floats
        import re
        aes = re.search(r"aesthetic.*?([-0-9.]+)", txt, re.I)
        tec = re.search(r"technical.*?([-0-9.]+)", txt, re.I)
        ovr = re.search(r"(overall|fused).*?([-0-9.]+)", txt, re.I)
        return (float(aes.group(1)) if aes else np.nan,
                float(tec.group(1)) if tec else np.nan,
                float(ovr.group(2)) if ovr else np.nan, txt[-400:])
    except Exception as e:
        return (np.nan, np.nan, np.nan, str(e)[:200])


# ---------- VBench-style dims ----------
def read_all(path, stride=2):
    r = imageio.get_reader(path)
    fr = []
    i = 0
    while True:
        try:
            f = r.get_data(i)
        except Exception:
            break
        fr.append(np.asarray(f)); i += stride
    r.close(); return fr


def motion_dims(frames):
    """(motion_smoothness_accel, temporal_flicker) — VBench-style."""
    grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]
    flows = []
    for a, b in zip(grays[:-1], grays[1:]):
        flows.append(cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0))
    accel = [float(np.linalg.norm(f2 - f1, axis=-1).mean()) for f1, f2 in zip(flows[:-1], flows[1:])]
    # flicker on low-motion pixels only (mask fast motion like VBench)
    flick = []
    for (a, b), fl in zip(zip(frames[:-1], frames[1:]), flows):
        low = np.linalg.norm(fl, axis=-1) < 1.0
        if low.mean() > 0.05:
            d = np.abs(a.astype(float) - b.astype(float)).mean(-1)
            flick.append(float(d[low].mean()))
    return float(np.mean(accel)), float(np.mean(flick)) if flick else np.nan


DINO = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(DEV).eval()
DMEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
DSTD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)


def dino_consistency(frames):
    feats = []
    with torch.no_grad():
        for f in frames[::3]:
            t = torch.tensor(f).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.
            t = F.interpolate(t, size=(224, 224), mode="bilinear", align_corners=False)
            e = DINO((t - DMEAN) / DSTD)
            feats.append(e / e.norm(dim=-1, keepdim=True))
    cs = [float((a @ b.T).item()) for a, b in zip(feats[:-1], feats[1:])]
    return float(np.mean(cs)), float(np.min(cs))


def main():
    rows = []
    for br in ("BL", "B"):
        for run in RUNS:
            p = vid_path(run, br)
            aes, tec, ovr, _dbg = dover_scores(p)
            frames = read_all(p)
            accel, flick = motion_dims(frames)
            dmean, dmin = dino_consistency(frames)
            rows.append(dict(branch=br, run=run, dover_aes=round(aes, 4) if aes == aes else np.nan,
                             dover_tech=round(tec, 4) if tec == tec else np.nan,
                             dover_all=round(ovr, 4) if ovr == ovr else np.nan,
                             flow_accel=round(accel, 3), flicker=round(flick, 3),
                             dino_cons_mean=round(dmean, 4), dino_cons_min=round(dmin, 4)))
            print(rows[-1], flush=True)
    df = pd.DataFrame(rows); df.to_csv(f"{OUTD}/vqa_bench_r08.csv", index=False)

    print("\n=== VALIDATION ===")
    for br, crit in [("BL", "clean {pca8,4node} separate"), ("B", "16node worst + pca8 mid/low")]:
        s = df[df.branch == br]
        print(f" [{br}] ({crit})")
        for col in [c for c in s.columns if c not in ("branch", "run")]:
            if s[col].isna().all():
                continue
            # lower-better metrics: dover scores are higher-better; accel/flicker lower-better; dino cons higher-better
            lower_better = col in ("flow_accel", "flicker")
            o = s.sort_values(col, ascending=not lower_better)   # first = best/cleanest
            order = list(o.run)
            if br == "BL":
                ok = set(order[:2]) == CLEAN_BL
            else:
                ok = (order[-1] == "16node" and order.index("pca8_8node") <= 3)
            print(f"   {col:15}: " + " ".join(f"{r}:{x}" for r, x in zip(o.run, o[col])) + ("  -> PASS" if ok else "  -> fail"))


if __name__ == "__main__":
    main()
