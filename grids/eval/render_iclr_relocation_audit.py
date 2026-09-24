"""Render contact sheets for low, boundary, and high relocation scores."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iclr_relocation import endpoint_index, resolve_path  # noqa: E402


def read(path: str, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError((path, index))
    return frame


def fit(image: np.ndarray, width=320, height=185) -> np.ndarray:
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)


def label(image: np.ndarray, text: str) -> np.ndarray:
    out = image.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (248, 248, 248), -1)
    cv2.putText(out, text, (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                (15, 15, 15), 1, cv2.LINE_AA)
    return out


def select(rows: pd.DataFrame, horizon: int, count: int) -> pd.DataFrame:
    d = rows[rows.horizon_s == horizon].copy()
    low = d.sort_values("panel_inliers").head(count).assign(audit_group="lowest")
    boundary = d.assign(distance=(d.panel_inliers - 50).abs()).sort_values(
        ["distance", "model", "scene"]).head(count).assign(audit_group="boundary")
    high = d.sort_values("panel_inliers", ascending=False).head(count).assign(
        audit_group="highest")
    return pd.concat([low, boundary, high], ignore_index=True).drop_duplicates(
        ["model", "scene", "horizon_s"], keep="first")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--seed-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--local-prefix", default="/home/ashish/ARRWM")
    p.add_argument("--remote-prefix", default="")
    p.add_argument("--count", type=int, default=5)
    a = p.parse_args()
    rows = pd.read_csv(a.rows)
    manifest = pd.read_csv(a.manifest).set_index(["scene", "model"])
    a.out.mkdir(parents=True, exist_ok=True)
    for horizon in (6, 30):
        chosen = select(rows, horizon, a.count)
        strips = []
        for r in chosen.itertuples():
            meta = manifest.loc[(r.scene, r.model)]
            candidate_path = resolve_path(str(meta.path), a.local_prefix, a.remote_prefix)
            seed_path = str(a.seed_dir / f"seed65_{r.uid}.mp4")
            seed_index = int(meta.context_frames) - 1
            seed = label(fit(read(seed_path, seed_index)),
                         f"conditioned real frame {seed_index}")
            candidate = label(fit(read(candidate_path, int(r.endpoint_index))),
                              f"{r.model} @ {horizon}s")
            if r.best_reference_type == "real":
                ref_index = int(str(r.best_reference).split("_")[-1])
                reference = read(seed_path, ref_index)
            else:
                peer = manifest.loc[(r.scene, r.best_reference)]
                peer_path = resolve_path(str(peer.path), a.local_prefix, a.remote_prefix)
                peer_index = endpoint_index(int(peer.context_frames), float(peer.fps),
                                            int(peer.decoded_frames), horizon)
                reference = read(peer_path, peer_index)
            reference = label(fit(reference),
                              f"best: {r.best_reference} ({r.panel_inliers} inliers)")
            triplet = np.hstack([seed, candidate, reference])
            header = np.full((34, triplet.shape[1], 3), 255, np.uint8)
            text = (f"{r.audit_group}: {r.scene} | flag={int(r.relocation_flag_50)} | "
                    f"real={r.real_best_inliers}, peer={r.peer_best_inliers}")
            cv2.putText(header, text, (7, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.56,
                        (0, 0, 0), 1, cv2.LINE_AA)
            strips.append(np.vstack([header, triplet]))
        sheet = np.vstack(strips)
        cv2.imwrite(str(a.out / f"relocation_audit_h{horizon}.png"), sheet)
        chosen.to_csv(a.out / f"relocation_audit_h{horizon}.csv", index=False)
        print("wrote", a.out / f"relocation_audit_h{horizon}.png")


if __name__ == "__main__":
    main()
