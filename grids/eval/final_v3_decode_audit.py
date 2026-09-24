"""Count decoded frames, read video headers, and verify DMD sidecars."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--shard-count", type=int, default=1)
    a = p.parse_args()
    m = pd.read_csv(a.out / "video_manifest.csv")
    m = m.iloc[a.shard_index::a.shard_count]
    rows = []
    dest = a.out / f"decoded_audit_shard{a.shard_index}.csv"
    for i, r in enumerate(m.itertuples(), 1):
        cap = cv2.VideoCapture(r.path)
        if not cap.isOpened():
            raise ValueError(f"cannot open: {r.path}")
        n_header = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        n = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            n += 1
        cap.release()
        sidecar_path = Path(r.path + ".json")
        sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.exists() else {}
        rows.append(dict(scene=r.scene, model=r.model, path=r.path,
                         header_frames=n_header, decoded_frames=n, fps=fps,
                         width=width, height=height,
                         sidecar_seed_frames=sidecar.get("seed_frames"),
                         sidecar_generated_frames=sidecar.get("generated_frames"),
                         expected_total=int(r.context_frames) + int(round(30*float(r.fps))),
                         complete=int(n == n_header == int(r.decoded_frames))))
        if i % 16 == 0 or i == len(m):
            pd.DataFrame(rows).to_csv(dest, index=False)
            print(f"[{i}/{len(m)}] {r.scene} {r.model} {n}", flush=True)
    assert all(x["complete"] for x in rows)


if __name__ == "__main__":
    main()
