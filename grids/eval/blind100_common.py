"""Shared helpers for scoring the blind100 eval set.

blind100/<blind_id>.mp4 are full-fleet rollouts (mixed fps + context length),
mapped to model/scene/direction in blind100_labels_and_scores.csv. Every video is
[CTX real context frames][generated]; per-model CTX from the fleet convention.
"""
import os
import pandas as pd

BLIND_DIR = os.path.expanduser("~/blind100")
PNG_DIR = os.path.expanduser("~/blind100_png")
LABELS = os.path.expanduser("~/blind100_labels_and_scores.csv")

# real context frames at the start of each model's saved video (fleet convention:
# ablations use the frame-12 boundary; externals per vlm_external.CTX_FRAMES).
CTX = {"astra": 4, "matrixgame": 1, "minwm": 13, "worldcam": 65,
       "worldplay": 1, "yume": 1,
       "16node": 12, "4node": 12, "pca8": 12, "pca4": 12, "pca2": 12,
       "noatok": 12, "noadaln": 12}


def refs():
    """List of dicts: blind_id, vid, model, scene, direction, path, ctx."""
    lab = pd.read_csv(LABELS)
    out = []
    for _, r in lab.iterrows():
        p = os.path.join(BLIND_DIR, f"{r.blind_id}.mp4")
        if not os.path.exists(p):
            print(f"[blind] MISSING {p}"); continue
        out.append(dict(blind_id=r.blind_id, vid=r.vid, model=r.model,
                        scene=int(r.scene), direction=r.direction, path=p,
                        ctx=CTX.get(r.model, 12)))
    return out
