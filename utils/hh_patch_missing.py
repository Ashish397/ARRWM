"""Patch the 8 missing head-to-head motion rows (7 matrixgame + 1 4node).

Reuses headtohead_extract's robust() + teacher read on exactly the missing
(model, window, dir) tuples and APPENDS to the model's existing CSV. The
original misses were per-video exceptions; failures here print the full error.
"""
import os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, "/scratch/u6ex/as1748.u6ex/ARRWM")
os.environ.setdefault("ARRWM_ACTION_ENCODER", "pca_raw")
from utils.ndof_following import load_pca, teacher_read_video, read_mp4
from utils.headtohead_extract import robust, vid_path

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
TARGETS = {
    f"{ARR}/analysis/eval_final/headtohead_matrixgame.csv": [
        ("matrixgame", 4, "L"), ("matrixgame", 6, "L"), ("matrixgame", 13, "L"),
        ("matrixgame", 19, "FL"), ("matrixgame", 24, "L"), ("matrixgame", 26, "L"),
        ("matrixgame", 29, "L")],
    f"{ARR}/analysis/eval_final/headtohead_motion.csv": [("4node", 6, "FL")],
}


def main():
    mean, comp_T, scales = load_pca()
    cot = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to("cuda").eval()
    for p in cot.parameters():
        p.requires_grad_(False)
    for csvp, items in TARGETS.items():
        prev = pd.read_csv(csvp)
        have = {(str(r.model), str(r.window), str(r["dir"])) for _, r in prev.iterrows()}
        rows = []
        for m, wi, d in items:
            if (m, str(wi), d) in have:
                print(f"[patch] {m} {wi} {d} already present, skip", flush=True)
                continue
            vp = vid_path(m, wi, d)
            try:
                vid = read_mp4(vp)
                frames = vid[0].permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
                rec, rot, tx = robust(list(frames))
                g8 = teacher_read_video(vid, cot, mean, comp_T, scales)[1:].cpu().numpy()
                gm = np.nanmean(g8, axis=0)
                rows.append(dict(model=m, window=wi, dir=d, recession=round(rec, 2),
                                 rot=round(rot, 2), tx=round(tx, 1),
                                 **{f"g{i}": round(float(gm[i]), 4) for i in range(8)}))
                print(f"[patch] {m} r{wi:02d} {d}: rec={rec:.1f} rot={rot:.1f}", flush=True)
            except Exception:
                import traceback
                print(f"[patch] FAILED {m} r{wi:02d} {d} ({vp}):", flush=True)
                traceback.print_exc()
        if rows:
            pd.concat([prev, pd.DataFrame(rows)], ignore_index=True).to_csv(csvp, index=False)
            print(f"[patch] {csvp}: +{len(rows)} rows -> {len(prev) + len(rows)}", flush=True)
    print("[patch] DONE", flush=True)


if __name__ == "__main__":
    main()
