"""Render every long-horizon relocation candidate with native seam frames."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def read(path: str, indices: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = []
    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, min(count - 1, max(0, index)))
            ok, frame = cap.read()
            if not ok:
                raise ValueError((path, index))
            out.append(cv2.resize(frame, (256, 144)))
    finally:
        cap.release()
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--events", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--rows-per-page", type=int, default=7)
    p.add_argument("--abrupt-only", action="store_true")
    a = p.parse_args()
    events = pd.read_csv(a.events)
    if a.abrupt_only:
        if "abrupt_cut" not in events:
            raise ValueError("--abrupt-only requires an abrupt_cut column")
        events = events[events.abrupt_cut.eq(1)]
    events = events.sort_values(["model", "scene", "time_s"])
    manifest = pd.read_csv(a.manifest).set_index(["scene", "model"])
    a.out.mkdir(parents=True, exist_ok=True)
    rendered = []
    for number, row in enumerate(events.itertuples(), 1):
        meta = manifest.loc[(row.scene, row.model)]
        path, fps, context = str(meta.path), float(meta.fps), int(meta.context_frames)
        offsets = [-1.0, -0.25, -1.0 / fps, 0.0, 1.0 / fps, 0.25, 1.0]
        indices = [context + int(round((row.time_s + value) * fps)) for value in offsets]
        frames = read(path, indices)
        header = np.full((42, 256 * len(frames), 3), 255, np.uint8)
        title = (f"#{number:02d} {row.model} {row.scene} t={row.time_s:.2f}s  "
                 f"hist-ratio={row.seam_hist_ratio:.2f}")
        cv2.putText(header, title, (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                    0.43, (0, 0, 0), 1, cv2.LINE_AA)
        for i, value in enumerate(offsets):
            cv2.putText(header, f"{value:+.3f}s", (i * 256 + 5, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1, cv2.LINE_AA)
        rendered.append(np.vstack([header, np.hstack(frames)]))
    for page, start in enumerate(range(0, len(rendered), a.rows_per_page), 1):
        image = np.vstack(rendered[start:start + a.rows_per_page])
        cv2.imwrite(str(a.out / f"candidates_page{page}.png"), image)
    print(f"rendered={len(rendered)} pages={(len(rendered) + a.rows_per_page - 1) // a.rows_per_page}")


if __name__ == "__main__":
    main()
