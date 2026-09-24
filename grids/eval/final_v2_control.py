"""Native-spatial CoTracker/PCA readout via the AAAI teacher_read_video producer.

Temporal adapter constructs 12 seed frames plus 96 generated frames at 16 Hz
for each fixed six-second window. Matrix-Game's 25 fps frames are selected by
timestamp; spatial pixels are never resized. Both adaptation and raw indices
are saved, so these scores remain distinguishable from original AAAI inputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "code_release"))
import fleet30s_common as fc  # noqa: E402
import evaluation.ndof_following as teacher  # noqa: E402
teacher.CK = str(ROOT / "code_release" / "preprocessing" / "checkpoints" / "pca_basis.pt")
load_pca = teacher.load_pca
teacher_read_video = teacher.teacher_read_video

DIR = {"F": (1, 0), "FR": (1, 1), "R": (0, 1), "BR": (-1, 1),
       "B": (-1, 0), "BL": (-1, -1), "L": (0, -1), "FL": (1, -1)}


def hash_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


_FRAME_CACHE_PATH = None
_FRAME_CACHE = None


def frames_at(path, indices):
    """Decode a clip once, then reuse identical native pixels across windows."""
    global _FRAME_CACHE_PATH, _FRAME_CACHE
    if _FRAME_CACHE_PATH != path:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise ValueError(f"cannot open: {path}")
        frames = []
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()
        _FRAME_CACHE = np.stack(frames)
        _FRAME_CACHE_PATH = path
    if max(indices) >= len(_FRAME_CACHE):
        raise ValueError(f"undecodable index {max(indices)}: {path}")
    return _FRAME_CACHE[indices]


def sample_indices(ctx, fps, n, start_s):
    # Seed: final 12 real frames for any multi-frame context (13, 29, or 33
    # frames here), and a replicated conditioning still for one-image systems.
    # At later windows, take the 12 preceding frames.
    start = ctx + int(round(start_s * fps))
    if start_s == 0:
        seed = list(range(ctx-12, ctx)) if ctx >= 12 else [0] * 12
    else:
        seed = [max(ctx, start - int(round((12-j)*fps/16))) for j in range(12)]
    gen = [ctx + int(round((start_s+j/16)*fps)) for j in range(96)]
    if max(gen) >= n:
        raise IndexError(f"need frame {max(gen)} from {n} frames")
    return seed + gen


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--scenes", help="comma-separated scene keys")
    p.add_argument("--models", help="comma-separated models")
    p.add_argument("--shard-index", type=int)
    p.add_argument("--shard-count", type=int, default=1)
    a = p.parse_args()
    out = a.out.resolve()
    m = pd.read_csv(out / "video_manifest.csv")
    m = m[m.decoded_frames.notna()]
    if a.scenes:
        m = m[m.scene.isin(a.scenes.split(","))]
    if a.models:
        m = m[m.model.isin(a.models.split(","))]
    if a.shard_index is not None:
        if not 0 <= a.shard_index < a.shard_count:
            raise ValueError("invalid shard")
        m = m.iloc[a.shard_index::a.shard_count]
    suffix = f"_shard{a.shard_index}" if a.shard_index is not None else ""
    dest = out / f"control_windows{suffix}.csv"
    prior = pd.read_csv(dest) if dest.exists() else pd.DataFrame()
    rows = prior.to_dict("records")
    mean, comp_T, scales = load_pca()
    cot = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to("cuda").eval()
    for param in cot.parameters():
        param.requires_grad_(False)
    errors = []
    for k, r in enumerate(m.itertuples(), 1):
        digest = hash_file(r.path)
        for start_s in (0, 6, 9, 12, 18, 24):
            key = (r.scene, r.model, start_s)
            old = prior[(prior.scene == key[0]) & (prior.model == key[1]) &
                        (prior.window_start_s == key[2])] if len(prior) else pd.DataFrame()
            if len(old) and old.video_sha256.iloc[0] == digest:
                continue
            rows = [x for x in rows if (x["scene"], x["model"], x["window_start_s"]) != key]
            try:
                indices = sample_indices(int(r.context_frames), float(r.fps), int(r.decoded_frames), start_s)
                rgb = frames_at(r.path, indices)
                sampled = json.dumps(indices)
                vid = torch.from_numpy(rgb).to("cuda").float().permute(0, 3, 1, 2).unsqueeze(0)
                z = teacher_read_video(vid, cot, mean, comp_T, scales)[1:].cpu().numpy()
                g = np.nanmean(z, axis=0)
                if not np.isfinite(g).all():
                    raise ValueError("teacher returned non-finite value")
                magnitude = float(np.hypot(g[0], g[1]))
                direction = r.direction
                command = np.array(DIR[direction], float) if direction != "N" else None
                cosine = float(np.dot(g[:2], command)/(magnitude*np.linalg.norm(command))) if command is not None and magnitude else np.nan
                rows.append(dict(scene=r.scene, model=r.model, direction=direction,
                                 window_start_s=start_s, window_end_s=start_s+6,
                                 window_role="15s_endpoint" if start_s == 9 else "nonoverlapping",
                                 sampled_indices=sampled, native_width=int(r.width),
                                 native_height=int(r.height), native_fps=float(r.fps),
                                 video_sha256=digest, g0=float(g[0]), g1=float(g[1]),
                                 magnitude=magnitude, cosine=cosine,
                                 wrong_direction_60=int(cosine < 0.5) if direction != "N" else None,
                                 noop_motion_010=int(magnitude >= 0.1) if direction == "N" else None,
                                 producer="code_release/evaluation/ndof_following.teacher_read_video"))
                print(f"[{k}/{len(m)}] {r.scene} {r.model} {start_s}-{start_s+6}", flush=True)
            except Exception as ex:
                errors.append(dict(scene=r.scene, model=r.model, window_start_s=start_s, error=repr(ex)))
                pd.DataFrame(errors).to_csv(out / f"control_errors{suffix}.csv", index=False)
                print("ERROR", key, ex, flush=True)
        if rows:
            tmp = dest.with_suffix(".tmp")
            pd.DataFrame(rows).to_csv(tmp, index=False)
            os.replace(tmp, dest)
    if rows:
        assert not pd.DataFrame(rows).duplicated(["scene", "model", "window_start_s"]).any()
    if errors:
        raise RuntimeError(f"control producer failed on {len(errors)} windows; see {out / f'control_errors{suffix}.csv'}")


if __name__ == "__main__":
    main()
