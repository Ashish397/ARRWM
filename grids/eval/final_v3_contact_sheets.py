"""Make review sheets for scored ICLR clips without altering any source video."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

W, H = 416, 234


def frame(path, index):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"frame {index}: {path}")
    return cv2.resize(bgr, (W, H), interpolation=cv2.INTER_AREA)


def main(out):
    manifest = pd.read_csv(out / "video_manifest.csv")
    scores = pd.read_csv(out / "cpu_endpoints_scored.csv")
    h6 = scores[(scores.horizon_s == 6) & (scores.direction != "N")].copy()
    h6 = h6.merge(manifest[["scene", "model", "path", "local_video"]],
                  on=["scene", "model"], validate="one_to_one")
    h6 = h6[h6.local_video]
    positive = h6[h6.B_v2_iclrfive > 150].nlargest(2, "B_v2_iclrfive")
    borderline = h6[h6.B_v2_iclrfive <= 150].nlargest(2, "B_v2_iclrfive")
    picked = pd.concat([positive.assign(review="strong HF positive"),
                        borderline.assign(review="HF hard negative")])
    rows = []
    for r in picked.itertuples():
        e = scores[(scores.scene == r.scene) & (scores.model == r.model)]
        ids = [int(r.ctx)-1] + [int(e[e.horizon_s == h].endpoint_index.iloc[0])
                                 for h in (6, 15, 30)]
        tiles = []
        for idx, label in zip(ids, ("real context", "6 s", "15 s", "30 s")):
            img = frame(r.path, idx)
            header = np.full((38, W, 3), 20, dtype=np.uint8)
            cv2.putText(header, f"{label}  frame {idx}", (8, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (240, 240, 240), 1)
            tiles.append(np.concatenate([header, img]))
        bar = np.full((38, W*4, 3), 14, dtype=np.uint8)
        cv2.putText(bar, f"{r.review}: {r.model} {r.scene} | B={r.B_v2_iclrfive:.1f}",
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        rows.append(np.concatenate([bar, np.concatenate(tiles, axis=1)]))
    dest = out / "hf_contact_sheet.png"
    cv2.imwrite(str(dest), np.concatenate(rows))
    picked[["scene", "model", "review", "B_v2_iclrfive", "path", "video_sha256"]].to_csv(
        out / "hf_contact_sheet_sources.csv", index=False)
    print(dest)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    main(a.out.resolve())
