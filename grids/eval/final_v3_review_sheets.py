"""Build traceable visual review sheets for the four non-HF ICLR instruments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

SIZE = (416, 234)


def frame(path, index):
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise ValueError((path, index))
    return cv2.resize(bgr, SIZE, interpolation=cv2.INTER_AREA)


def label(image, text):
    header = np.full((34, image.shape[1], 3), 20, np.uint8)
    cv2.putText(header, text[:63], (8, 23), cv2.FONT_HERSHEY_SIMPLEX,
                0.56, (245, 245, 245), 1, cv2.LINE_AA)
    return np.concatenate([header, image])


def draw_row(r, metric, endpoints):
    clip = Path(r.path)
    ctx = int(r.context_frames)
    fps = float(r.fps)
    first = ctx
    middle = ctx + int(round(3 * fps))
    last = int(endpoints[(endpoints.scene == r.scene) &
                         (endpoints.model == r.model) &
                         (endpoints.horizon_s == 6)].endpoint_index.iloc[0])
    if metric == "conjuration" and r.conjuration_flag == 1:
        events = json.loads(r.events)
        positive = [e for e in events if e["score"] > 0]
        if positive:
            middle = max(first, min(last, int(positive[0]["birth"])))
    indices = [ctx-1, first, middle, last]
    middle_title = "detected birth" if metric == "conjuration" and r.conjuration_flag == 1 else "3 s"
    titles = ["real context", "generated start", middle_title, "6 s endpoint"]
    tiles = [label(frame(clip, idx), f"{title} | frame {idx}")
             for title, idx in zip(titles, indices)]
    body = np.concatenate(tiles, axis=1)
    header = np.full((37, body.shape[1], 3), 13, np.uint8)
    score = {"style": r.drift_from_real, "geometry": r.p_uncanny,
             "conjuration": r.top_score, "control": r.cosine}[metric]
    score_text = "none" if pd.isna(score) else f"{score:.4f}"
    cv2.putText(header, f"{r.review}: {r.model} {r.scene} | {metric} score={score_text}",
                (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.67, (255, 255, 255), 1,
                cv2.LINE_AA)
    return np.concatenate([header, body])


def select(d, metric):
    if metric == "style":
        pos = d[d.style_flag_072_descriptive == 1].nlargest(2, "drift_from_real")
        neg = d[d.style_flag_072_descriptive == 0].nlargest(2, "drift_from_real")
    elif metric == "geometry":
        pos = d[d.geometry_flag == 1].nlargest(2, "p_uncanny")
        neg = d[d.geometry_flag == 0].nlargest(2, "p_uncanny")
    elif metric == "conjuration":
        pos = d[d.conjuration_flag == 1].nlargest(2, "top_score")
        neg = d[d.conjuration_flag == 0].nlargest(2, "top_score")
    else:
        directional = d[(d.direction != "N") & d.cosine.notna()]
        pos = directional[directional.wrong_direction_60 == 1].nlargest(2, "cosine")
        neg = directional[directional.control_failure == 0].nsmallest(2, "cosine")
    return pd.concat([pos.assign(review="flagged"), neg.assign(review="hard negative")])


def main(out):
    m = pd.read_csv(out / "video_manifest.csv")
    q = pd.read_csv(out / "quality_endpoints_per_video.csv")
    endpoints = pd.read_csv(out / "cpu_endpoints_scored.csv")
    d = q[q.horizon_s == 6].merge(
        m[["scene", "model", "path", "local_video", "context_frames", "fps"]],
        on=["scene", "model"], validate="one_to_one")
    d = d[d.local_video]
    for metric in ("style", "geometry", "conjuration", "control"):
        selected = select(d, metric)
        if len(selected) < 4:
            raise AssertionError(f"insufficient local review examples: {metric}")
        rows = [draw_row(r, metric, endpoints) for r in selected.itertuples()]
        cv2.imwrite(str(out / f"{metric}_contact_sheet.png"), np.concatenate(rows))
        selected[["scene", "model", "review", "path", "video_sha256"]].to_csv(
            out / f"{metric}_contact_sheet_sources.csv", index=False)
        print(metric, len(selected))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    main(a.out.resolve())
