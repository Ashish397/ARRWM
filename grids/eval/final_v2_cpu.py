"""Resumable producer-faithful CPU measurements on locally decodable ICLR clips.

Six-second anchor uses the AAAI producer's ctx+round(6*fps) frame. The
15/30-second extension uses the final frame in H seconds of generated video,
ctx+round(H*fps)-1. Long starting-view overlap is a diagnostic, not relocation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fleet30s_common as fc  # noqa: E402
from final_v2_iclr import frame_index, original_static_inl  # noqa: E402


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def lap_var(rgb):
    g = cv2.cvtColor(cv2.resize(rgb, (832, 448)), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def measure(r):
    scene, model, path = r.scene, r.model, r.path
    n, fps, ctx = int(r.decoded_frames), float(r.fps), int(r.context_frames)
    anchor = ctx + int(round(6 * fps))
    if anchor >= n:
        raise IndexError(f"AAAI 6 s endpoint unavailable: {scene} {model} {anchor}/{n}")
    ends = {6: anchor, 15: frame_index(ctx, fps, n, 15), 30: frame_index(ctx, fps, n, 30)}
    base_idx = list(range(ctx + int(round(fps)), ctx + int(round(fps)) + 4))
    idx = set(base_idx)
    for end in ends.values():
        idx.update(range(max(ctx, end - 14), end + 1, 2))
    windows = []
    for start_s in (0, 6, 9, 12, 18, 24):
        first = ctx + int(round(start_s * fps))
        last = ctx + int(round((start_s + 6) * fps)) - 1
        if last >= n:
            raise IndexError(f"six-second window unavailable: {scene} {model} {last}/{n}")
        windows.append((start_s, first, last))
        idx.update(range(max(first, last - 14), last + 1, 2))
    # This is the producer's imageio decode and Laplacian definition.
    indices = sorted(idx)
    frames = fc.frames_at(scene, model, indices)
    if len(frames) != len(indices):
        raise ValueError("missing imageio frames")
    sharp = {i: lap_var(f) for i, f in zip(indices, frames)}
    base = float(np.mean([sharp[i] for i in base_idx]))
    digest = file_hash(path)
    common = dict(scene=scene, model=model, family=r.family, direction=r.direction,
                  fps=fps, ctx=ctx, decoded_frames=n, video_sha256=digest,
                  producer_hf="code_release/evaluation/quality/fleet_hf.py",
                  producer_orb="code_release/evaluation/quality/fleet_static_inl.py")
    endpoints = []
    for horizon, end in ends.items():
        end_idx = list(range(max(ctx, end - 14), end + 1, 2))
        final = float(np.mean([sharp[i] for i in end_idx]))
        static = original_static_inl(path, ctx, end)
        endpoints.append(dict(**common, horizon_s=horizon, endpoint_index=end,
                              endpoint_timestamp_s=round((end-ctx)/fps, 5),
                              base_blur=round(base, 1), end_blur=round(final, 1),
                              d_blur=round(final-base, 1), starting_view_inliers=static,
                              control_near_static_500=int(static > 500) if horizon == 6 and r.direction != "N" else None,
                              active_static_600=int(static > 600) if horizon == 6 and r.direction != "N" else None,
                              endpoint_rule="AAAI_t_equals_6" if horizon == 6 else "last_frame_in_H_seconds"))
    wr = []
    for start_s, first, last in windows:
        end_idx = list(range(max(first, last - 14), last + 1, 2))
        final = float(np.mean([sharp[i] for i in end_idx]))
        inliers = original_static_inl(path, first, last)
        wr.append(dict(**common, window_start_s=start_s, window_end_s=start_s+6,
                       window_role="15s_endpoint" if start_s == 9 else "nonoverlapping",
                       first_index=first, last_index=last, last_timestamp_s=round((last-ctx)/fps, 5),
                       end_blur=round(final, 1), d_blur_from_early_base=round(final-base, 1),
                       local_orb_inliers=inliers,
                       local_near_static_500_exploratory=int(inliers > 500)))
    return endpoints, wr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scenes", help="comma-separated scene keys; omit for all local clips")
    ap.add_argument("--models", help="comma-separated models")
    ap.add_argument("--shard-index", type=int)
    ap.add_argument("--shard-count", type=int, default=1)
    a = ap.parse_args()
    out = a.out.resolve()
    m = pd.read_csv(out / "video_manifest.csv")
    m = m[m.local_video & m.decoded_frames.notna()].copy()
    if a.scenes:
        m = m[m.scene.isin(a.scenes.split(","))]
    if a.models:
        m = m[m.model.isin(a.models.split(","))]
    if a.shard_index is not None:
        if not 0 <= a.shard_index < a.shard_count:
            raise ValueError("invalid shard")
        m = m.iloc[a.shard_index::a.shard_count]
    suffix = f"_shard{a.shard_index}" if a.shard_index is not None else ""
    ep = out / f"cpu_endpoints{suffix}.csv"
    wp = out / f"cpu_windows{suffix}.csv"
    prior = pd.read_csv(ep) if ep.exists() else pd.DataFrame()
    erows = prior.to_dict("records")
    oldw = pd.read_csv(wp) if wp.exists() else pd.DataFrame()
    wrows = oldw.to_dict("records")
    errors = []
    for i, r in enumerate(m.itertuples(), 1):
        cached_e = (prior[(prior.scene == r.scene) & (prior.model == r.model)]
                    if len(prior) else pd.DataFrame())
        cached_w = (oldw[(oldw.scene == r.scene) & (oldw.model == r.model)]
                    if len(oldw) else pd.DataFrame())
        if len(cached_e) or len(cached_w):
            current_hash = file_hash(r.path)
            complete_e = (
                len(cached_e) == 3 and
                set(cached_e.horizon_s) == {6, 15, 30} and
                (cached_e.video_sha256 == current_hash).all()
            )
            complete_w = (
                len(cached_w) == 6 and
                set(cached_w.window_start_s) == {0, 6, 9, 12, 18, 24} and
                (cached_w.video_sha256 == current_hash).all()
            )
            if complete_e and complete_w:
                continue
            # A changed or partial cache must be replaced as one atomic video
            # unit; accepting endpoints without windows (or vice versa) makes
            # a resumed run impossible to complete.
            erows = [x for x in erows if (x["scene"], x["model"]) != (r.scene, r.model)]
            wrows = [x for x in wrows if (x["scene"], x["model"]) != (r.scene, r.model)]
        try:
            e, w = measure(r)
            erows.extend(e); wrows.extend(w)
            pd.DataFrame(erows).to_csv(ep, index=False)
            pd.DataFrame(wrows).to_csv(wp, index=False)
            print(f"[{i}/{len(m)}] {r.scene} {r.model}", flush=True)
        except Exception as ex:
            errors.append(dict(scene=r.scene, model=r.model, error=repr(ex)))
            pd.DataFrame(errors).to_csv(out / f"cpu_errors{suffix}.csv", index=False)
            print("ERROR", r.scene, r.model, ex, flush=True)
    d = pd.DataFrame(erows)
    if len(d):
        assert not d.duplicated(["scene", "model", "horizon_s"]).any()
        assert not pd.DataFrame(wrows).duplicated(["scene", "model", "window_start_s"]).any()
    print("endpoint rows", len(d), "window rows", len(wrows), "errors", len(errors))
    if errors:
        raise RuntimeError(f"CPU producer failed on {len(errors)} videos; see {out / f'cpu_errors{suffix}.csv'}")


if __name__ == "__main__":
    main()
