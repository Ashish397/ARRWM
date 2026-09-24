"""Render event boundaries and direct-seed-loss rescues for pilot review."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def resolve(path: str, local_prefix: str, remote_prefix: str) -> str:
    if Path(path).exists():
        return path
    candidate = remote_prefix + path[len(local_prefix):]
    if Path(candidate).exists():
        return candidate
    raise FileNotFoundError(path)


def read(path: str, indices: list[int]) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, index))
            ok, frame = cap.read()
            if not ok:
                raise ValueError((path, index))
            frames.append(cv2.resize(frame, (320, 176)))
    finally:
        cap.release()
    return frames


def tile(frames: list[np.ndarray], title: str, labels: list[str]) -> np.ndarray:
    header = np.full((38, 320 * len(frames), 3), 255, np.uint8)
    cv2.putText(header, title, (5, 16), cv2.FONT_HERSHEY_SIMPLEX,
                0.42, (0, 0, 0), 1, cv2.LINE_AA)
    for i, label in enumerate(labels):
        cv2.putText(header, label, (i * 320 + 5, 33), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([header, np.hstack(frames)])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--direct", type=Path, required=True)
    p.add_argument("--events", type=Path)
    p.add_argument("--local-prefix", default="/home/ashish/ARRWM")
    p.add_argument("--remote-prefix", default="")
    a = p.parse_args()
    manifest = pd.read_csv(a.manifest).set_index(["scene", "model"])
    events_path = a.events or (a.out / "long_relocation_events.csv")
    events = pd.read_csv(events_path) if events_path.stat().st_size > 1 else pd.DataFrame()
    if len(events) and "relocation_flag" in events:
        events = events[events.relocation_flag.eq(1)]
    audit = a.out / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    event_tiles = []
    if len(events):
        chosen = events.sort_values(["model", "cross_inliers", "time_s"]).groupby(
            "model", as_index=False).head(2).head(30)
        for row in chosen.itertuples():
            meta = manifest.loc[(row.scene, row.model)]
            path = resolve(str(meta.path), a.local_prefix, a.remote_prefix)
            fps, context = float(meta.fps), int(meta.context_frames)
            offsets = [-1.0, -0.25, 0.0, 0.25, 1.0]
            indices = [context + int(round((row.time_s + off) * fps)) for off in offsets]
            frames = read(path, indices)
            title = (f"EVENT {row.model} {row.scene} t={row.time_s:.2f}s "
                     f"cross={row.cross_inliers} pre={row.pre_coherence:.0f} "
                     f"post={row.post_coherence:.0f}")
            event_tiles.append(tile(frames, title, [f"{off:+.2f}s" for off in offsets]))
        cv2.imwrite(str(audit / "long_relocation_events.png"), np.vstack(event_tiles))
        chosen.to_csv(audit / "long_relocation_events.csv", index=False)

    direct = pd.read_csv(a.direct)
    endpoint = pd.read_csv(a.out / "long_relocation_rows.csv")
    direct = direct[direct.horizon_s.eq(30) & direct.relocation_flag_50.eq(1)]
    endpoint = endpoint[endpoint.horizon_s.eq(30) & endpoint.long_relocation_flag.eq(0)]
    rescued = direct.merge(endpoint[["scene", "model"]], on=["scene", "model"])
    rescued = rescued.sort_values(["model", "panel_inliers"]).groupby(
        "model", as_index=False).head(2).head(30)
    rescue_tiles = []
    for row in rescued.itertuples():
        meta = manifest.loc[(row.scene, row.model)]
        path = resolve(str(meta.path), a.local_prefix, a.remote_prefix)
        fps, context = float(meta.fps), int(meta.context_frames)
        times = [0, 6, 12, 18, 24, 30]
        indices = [min(int(meta.decoded_frames) - 1,
                       context + int(round(time_s * fps))) for time_s in times]
        frames = read(path, indices)
        title = (f"CHAIN-PRESERVED {row.model} {row.scene}; "
                 f"direct h30 inliers={row.panel_inliers}")
        rescue_tiles.append(tile(frames, title, [f"{time_s}s" for time_s in times]))
    if rescue_tiles:
        cv2.imwrite(str(audit / "long_relocation_chain_preserved.png"),
                    np.vstack(rescue_tiles))
        rescued.to_csv(audit / "long_relocation_chain_preserved.csv", index=False)
    print(f"event examples={len(event_tiles)} chain-preserved={len(rescue_tiles)}")


if __name__ == "__main__":
    main()
