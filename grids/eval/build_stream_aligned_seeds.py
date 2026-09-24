"""Build the canonical evaluation seed from our native 9-latent prefix.

The original standalone seed renderer used Wan ``seed_first`` decoding.  That
adds a replicated latent and shifts its visible timeline by three pixel frames
relative to the native cached streaming decoder used by the world model.  The
evaluation boundary must instead be the final frame actually represented by
our three native seed chunks.  This script extracts that already-rendered,
action-independent 33-frame prefix from one complete checkpoint and writes a
lossless-H.264 canonical clip plus the common frame-32 PNG used by one-frame
baselines.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_frames(path: Path) -> int:
    return int(subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1",
        str(path),
    ], text=True).strip())


def read_frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    assert cap.isOpened(), path
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    assert ok and frame is not None, (path, index)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--windows", required=True, type=Path)
    parser.add_argument("--ours-dir", required=True, type=Path)
    parser.add_argument("--video-out", required=True, type=Path)
    parser.add_argument("--frame-out", required=True, type=Path)
    args = parser.parse_args()
    args.video_out.mkdir(parents=True, exist_ok=True)
    args.frame_out.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)

    windows = json.loads(args.windows.read_text())
    report = []
    for row in windows:
        uid = row["uid"]
        source = args.ours_dir / f"recovery_base_{uid}_F.mp4"
        assert source.exists(), source
        video = args.video_out / f"seed65_{uid}.mp4"
        temporary = video.with_suffix(".tmp.mp4")
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-threads", "1", "-i", str(source),
            "-vf", "trim=start_frame=0:end_frame=33,setpts=PTS-STARTPTS",
            "-frames:v", "33", "-r", "16", "-an", "-c:v", "libx264",
            "-preset", "veryfast", "-crf", "0", "-pix_fmt", "yuv420p",
            str(temporary),
        ], check=True)
        temporary.replace(video)
        assert count_frames(video) == 33, video

        frame_png = args.frame_out / f"seed65_{uid}_f0.png"
        temporary_png = frame_png.with_suffix(".tmp.png")
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-threads", "1", "-i", str(video),
            "-vf", "select=eq(n\\,32)", "-frames:v", "1", str(temporary_png),
        ], check=True)
        temporary_png.replace(frame_png)

        original_boundary = read_frame(source, 32)
        canonical_boundary = read_frame(video, 32)
        delta = np.abs(
            original_boundary.astype(np.int16) - canonical_boundary.astype(np.int16)
        )
        mae = float(delta.mean())
        assert mae < 3.0, (uid, mae, int(delta.max()))
        sidecar = {
            "uid": uid,
            "ride": row["ride"],
            "offset": int(row["offset"]),
            "source_video": str(source.resolve()),
            "source_sha256": sha256(source),
            "canonical_video": str(video.resolve()),
            "canonical_sha256": sha256(video),
            "frame32_png": str(frame_png.resolve()),
            "frame32_sha256": sha256(frame_png),
            "frames": 33,
            "fps": 16,
            "seed_latents": 9,
            "seed_chunks": 3,
            "generation_boundary_real_frame": 32,
            "decoder_path": "native cached Wan streaming prefix",
            "boundary_reencode_mae": mae,
        }
        Path(str(video) + ".json").write_text(json.dumps(sidecar, indent=2) + "\n")
        report.append(sidecar)
        print(f"STREAM_SEED uid={uid} frames=33 boundary_mae={mae:.4f}", flush=True)

    assert len(report) == 32
    summary = {
        "status": "pass",
        "seeds": len(report),
        "common_generation_boundary_real_frame": 32,
        "seed_latents": 9,
        "seed_chunks": 3,
        "maximum_boundary_reencode_mae": max(x["boundary_reencode_mae"] for x in report),
        "rows": report,
    }
    (args.video_out / "stream_seed_manifest.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
