"""Verify the real-video conditioning span of every row in an evaluation manifest.

This is deliberately independent of runner sidecars.  It decodes the first and
last conditioning frames from each submitted video and compares them with the
corresponding frames in the canonical seed65 source.  Consequently a correct
filename or sidecar cannot mask an off-by-one, wrong-scene, black, or truncated
conditioning prefix.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np
import pandas as pd


# Inclusive source-video bounds.  All spans end at the same real frame 32.
SOURCE_SPANS = {
    "lingbot": (32, 32),
    "dreamx": (32, 32),
    "yume5b": (32, 32),       # native single-image YUME condition
    "matrixgame2": (32, 32),
    "minwm": (4, 32),          # eight minWM latent frames / two native chunks
    "minwm_ode": (20, 32),     # four latent frames
    "ours_kl4rung": (0, 32),   # nine latent frames / three native chunks
    "ours_mse4rung": (0, 32),
    "ours_recovery_base": (0, 32),
    "ours_no_commit": (0, 32),
    "ours_no_aux": (0, 32),
    "ours_no_gan": (0, 32),
    "ours_no_carn": (0, 32),
    "ours_stat_mean_only": (0, 32),
    "ours_stat_nonmean_only": (0, 32),
    "ours_base_v2_BROKEN": (0, 32),
}


def read_frame(path: Path, index: int) -> np.ndarray:
    # Some cluster nodes transiently return EAGAIN while opening H.264 decoder
    # contexts.  Four workers keeps decoder pressure modest; retries distinguish
    # that resource condition from a persistently undecodable input.
    for attempt in range(3):
        cap = cv2.VideoCapture(str(path))
        opened = cap.isOpened()
        if opened:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = cap.read()
        else:
            ok, frame = False, None
        cap.release()
        if ok and frame is not None:
            return frame
        if attempt < 2:
            time.sleep(0.1 * (attempt + 1))
    raise AssertionError((path, index, "persistently undecodable"))


def read_frames_ffmpeg(
    path: Path, indices: list[int], width: int, height: int
) -> list[np.ndarray]:
    """Decode selected frames with one single-threaded ffmpeg process.

    OpenCV's bundled decoder chooses a large native thread pool per H.264
    stream on Isambard and can exhaust decoder resources even with only four
    Python workers.  Explicit single-threaded ffmpeg decoding is deterministic
    here and still tests the encoded media itself.
    """
    unique = list(dict.fromkeys(indices))
    expression = "+".join(f"eq(n\\,{index})" for index in unique)
    command = [
        "ffmpeg", "-v", "error", "-threads", "1",
        "-filter_threads", "1", "-filter_complex_threads", "1",
        "-i", str(path),
        "-vf", f"select={expression}", "-vsync", "0",
        "-frames:v", str(len(unique)), "-pix_fmt", "bgr24",
        "-c:v", "rawvideo", "-threads:v", "1", "-f", "rawvideo", "pipe:1",
    ]
    result = subprocess.run(command, capture_output=True, timeout=60)
    assert result.returncode == 0, (
        path, indices, result.stderr.decode("utf-8", errors="replace")[-1000:]
    )
    frame_bytes = width * height * 3
    expected_bytes = frame_bytes * len(unique)
    assert len(result.stdout) == expected_bytes, (
        path, indices, len(result.stdout), expected_bytes
    )
    decoded = {
        index: np.frombuffer(
            result.stdout[offset * frame_bytes:(offset + 1) * frame_bytes],
            dtype=np.uint8,
        ).reshape(height, width, 3).copy()
        for offset, index in enumerate(unique)
    }
    return [decoded[index] for index in indices]


def resize_like(reference: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    if reference.shape == candidate.shape:
        return reference
    return cv2.resize(
        reference,
        (candidate.shape[1], candidate.shape[0]),
        interpolation=cv2.INTER_AREA,
    )


def frame_mae(reference: np.ndarray, candidate: np.ndarray) -> float:
    reference = resize_like(reference, candidate)
    return float(np.abs(reference.astype(np.int16) - candidate.astype(np.int16)).mean())


def healthy(frame: np.ndarray) -> bool:
    return float(frame.mean()) > 3 and float(frame.std()) > 3 and int(frame.max()) > 10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--seed65-dir", type=Path)
    source.add_argument(
        "--eval-config", type=Path,
        help="panel32 resolver config containing canonical source clips and spans",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mae-threshold", type=float, default=40.0)
    parser.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    cv2.setNumThreads(1)

    manifest = pd.read_csv(args.manifest)
    assert not manifest.duplicated(["scene", "model"]).any()
    models = set(manifest.model)
    if args.eval_config is not None:
        config = json.loads(args.eval_config.resolve().read_text())
        configured_models = config["models"]
        spans = {
            model: tuple(map(int, configured_models[model]["source_frames_inclusive"]))
            for model in configured_models
        }
        source_clips = {str(k): Path(v) for k, v in config["source_clips"].items()}
        assert set(manifest.uid) <= set(source_clips), sorted(set(manifest.uid) - set(source_clips))
    else:
        spans = SOURCE_SPANS
        source_clips = {
            str(uid): args.seed65_dir / f"seed65_{uid}.mp4"
            for uid in manifest.uid.unique()
        }
    assert models <= set(spans), sorted(models - set(spans))
    contexts = int(manifest.uid.nunique())
    assert contexts > 0
    expected_per_model = contexts * 9
    assert (manifest.groupby("model").size() == expected_per_model).all()

    # Decode each selected canonical boundary reference once, not once per video.
    references: dict[tuple[str, int], np.ndarray] = {}
    for uid in sorted(manifest.uid.unique()):
        source_path = source_clips[uid]
        assert source_path.exists(), source_path
        needed = sorted({index for model in models for index in spans[model]})
        for index in needed:
            references[(uid, index)] = read_frame(source_path, index)

    def validate(row: object) -> dict[str, object]:
        model = str(row.model)
        uid = str(row.uid)
        start_source, end_source = spans[model]
        expected_context = end_source - start_source + 1
        context = int(row.context_frames)
        assert context == expected_context, (
            row.scene, model, context, expected_context
        )
        video = Path(str(row.path))
        assert video.exists(), video
        first, last = read_frames_ffmpeg(
            video, [0, context - 1], int(row.width), int(row.height)
        )
        assert healthy(first) and healthy(last), (
            video,
            "unhealthy conditioning frame",
            (float(first.mean()), float(first.std()), int(first.max())),
            (float(last.mean()), float(last.std()), int(last.max())),
        )
        first_mae = frame_mae(references[(uid, start_source)], first)
        last_mae = frame_mae(references[(uid, end_source)], last)
        assert first_mae < args.mae_threshold, (
            video, "conditioning-start mismatch", start_source, first_mae
        )
        assert last_mae < args.mae_threshold, (
            video, "conditioning-end mismatch", end_source, last_mae
        )
        return {
            "scene": row.scene,
            "model": model,
            "path": str(video),
            "source_start": start_source,
            "source_end": end_source,
            "context_frames": context,
            "first_frame_mae": first_mae,
            "last_frame_mae": last_mae,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(validate, manifest.itertuples(index=False)))
    detail = pd.DataFrame(rows).sort_values(["model", "scene"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.output, index=False)

    summary = (
        detail.groupby("model")
        .agg(
            videos=("scene", "size"),
            source_start=("source_start", "first"),
            source_end=("source_end", "first"),
            context_frames=("context_frames", "first"),
            maximum_first_frame_mae=("first_frame_mae", "max"),
            maximum_last_frame_mae=("last_frame_mae", "max"),
        )
        .reset_index()
    )
    assert (summary.videos == expected_per_model).all()
    report = {
        "status": "pass",
        "videos": len(detail),
        "models": len(summary),
        "mae_threshold": args.mae_threshold,
        "common_generation_boundary_real_frame": 32,
        "per_model": summary.to_dict(orient="records"),
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
