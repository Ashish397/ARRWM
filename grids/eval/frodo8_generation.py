#!/usr/bin/env python3
"""Fail-closed preparation and validation for the FrodoBots eight-context fleet.

The selected contexts are fixed by ``frodo8_manifest.json``.  ``prepare``
makes a runner resumable by retaining only complete, correctly attributed
clips for the requested UID shard; malformed outputs are moved to a recovery
directory.  ``validate`` checks all 72 model/action pairs, including the
decoded frame at the common real-frame-32 generation boundary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np


MODELS = ("lingbot", "dreamx", "matrixgame2", "minwm", "minwm_ode")
DIRS = ("F", "FR", "R", "BR", "B", "BL", "L", "FL", "N")
DREAMX_KEYS = {
    "F": "w", "FR": "wl", "R": "l", "BR": "sl", "B": "s",
    "BL": "sj", "L": "j", "FL": "wj", "N": " ",
}


def dreamx_context_seed(uid: str, base_seed: int = 1) -> int:
    digest = hashlib.sha256(uid.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % (2**31 - 1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path, source: Path | None = None) -> dict:
    manifest = json.loads(path.read_text())
    assert manifest["dataset"] == "FrodoBots", manifest.get("dataset")
    assert manifest["generation_boundary_real_frame"] == 32
    assert tuple(manifest["actions"]) == DIRS
    conditioning = manifest["conditioning"]
    assert conditioning["ours"] == {
        "latent_frames": 9, "source_pixel_frames_inclusive": [0, 32]}
    assert conditioning["minwm_dmd"] == {
        "latent_frames": 8, "source_pixel_frames_inclusive": [4, 32]}
    assert conditioning["minwm_ode"] == {
        "latent_frames": 4, "source_pixel_frames_inclusive": [20, 32]}
    assert conditioning["yume5b"] == {
        "source_pixel_frames_inclusive": [32, 32]}
    for model in ("lingbot", "dreamx", "matrixgame2"):
        assert conditioning[model] == {"source_pixel_frames_inclusive": [32, 32]}
    contexts = manifest["contexts"]
    indices = [0, 4, 8, 12, 16, 20, 24, 28]
    uids = ["u31", "u04", "a20", "m30", "m38", "m89", "m128", "b36"]
    assert [row["source_index"] for row in contexts] == indices
    assert [row["uid"] for row in contexts] == uids
    if source is not None:
        assert sha256(source) == manifest["selection"]["source_manifest_sha256"]
        rows = json.loads(source.read_text())
        assert len(rows) == 32
        for index, context in zip(indices, contexts):
            assert rows[index] == {k: context[k] for k in ("uid", "ride", "offset", "city", "tod")}
    return manifest


def disk_direction(model: str, direction: str) -> str:
    if direction == "N" and model in {"matrixgame2", "minwm", "minwm_ode"}:
        return "NOOP"
    return direction


def clip_path(model: str, output: Path, uid: str, direction: str) -> Path:
    direction = disk_direction(model, direction)
    templates = {
        "lingbot": "lingbot_{uid}_{direction}.mp4",
        "dreamx": "dreamx_{uid}_{direction}.mp4",
        "matrixgame2": "matrixgame_{uid}_{direction}.mp4",
        "minwm": "minwm_aligned32_{uid}_{direction}.mp4",
        "minwm_ode": "minwm_ode_{uid}_{direction}.mp4",
    }
    return output / templates[model].format(uid=uid, direction=direction)


def probe(path: Path) -> tuple[int, float]:
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames,r_frame_rate", "-of", "json",
        str(path),
    ], text=True)
    stream = json.loads(raw)["streams"][0]
    num, den = map(int, stream["r_frame_rate"].split("/"))
    return int(stream["nb_read_frames"]), num / den


def validate_sidecar(model: str, sidecar: Path, uid: str, direction: str,
                     seed_frame32: Path, seed_stream: Path) -> dict:
    meta = json.loads(sidecar.read_text())
    on_disk_direction = disk_direction(model, direction)
    assert meta["direction"] in {direction, on_disk_direction}, (sidecar, meta.get("direction"))
    if model == "lingbot":
        assert meta["model"] == "LingBot-World-V2-1.3B-Causal-Fast"
        assert meta["frames"] == 493 and meta["fps"] == 16
        assert meta["seed"] == 1
        assert abs(float(meta["step_t"]) - 0.25) < 1e-12
        assert abs(float(meta["step_r_deg"]) - 0.75) < 1e-12
    elif model == "dreamx":
        assert meta["model"] == "DreamX-World-5B"
        assert meta["latent_frames"] == 123 and meta["fps"] == 16
        assert meta["seed"] == dreamx_context_seed(uid)
        assert meta["seed_rule"] == (
            "base + first32bits(sha256(context_id)), modulo 2^31-1")
        assert meta["action_keys"] == DREAMX_KEYS[direction]
    elif model == "matrixgame2":
        assert meta["model"] == "Matrix-Game-2.0"
        assert meta["direction"] == on_disk_direction
        assert meta["frames"] == 753 and meta["fps"] == 25
        assert meta["latent_frames"] == 189 and meta["sampling_seed"] == 0
        assert float(meta["camera_yaw_per_frame"]) == 0.1
        assert meta["keyboard_dims"] == 4
    elif model == "minwm":
        assert meta["model"] == "minWM Wan2.1-1.3B Action2V 4-step DMD"
        assert meta["stage"] == "dmd" and meta["config"] == "causal_forcing_dmd_camera.yaml"
        assert meta["seed_start_frame"] == 4 and meta["seed_frames"] == 29
        assert meta["seed_latents"] == 8 and meta["total_latents"] == 128
        assert meta["generated_latents"] == 120 and meta["output_frames"] == 509
        assert meta["generated_frames"] == 480 and meta["fps"] == 16
        assert float(meta["generated_seconds"]) == 30.0
    else:
        assert meta["model"] == "minWM Wan2.1-1.3B Action2V causal ODE"
        assert meta["stage"] == "ode" and meta["config"] == "causal_ode_camera.yaml"
        assert meta["seed_start_frame"] == 20 and meta["seed_frames"] == 13
        assert meta["seed_latents"] == 4 and meta["total_latents"] == 124
        assert meta["generated_latents"] == 120 and meta["output_frames"] == 493
        assert meta["generated_frames"] == 480 and meta["fps"] == 16
        assert float(meta["generated_seconds"]) == 30.0

    if model in {"minwm", "minwm_ode"}:
        assert meta["generation_boundary_real_frame"] == 32
        assert meta["yaw_adapter"] == "released_minwm_yaw_negated_to_common_convention"
        seed = Path(meta["seed_video"])
        canonical = seed_stream / f"seed65_{uid}.mp4"
    else:
        assert meta["generation_boundary_real_frame"] == 32
        seed = Path(meta["seed_image"])
        canonical = seed_frame32 / f"seed65_{uid}_f0.png"
        assert meta["seed_sha256"] == sha256(canonical)
    assert seed.exists() and canonical.exists(), (seed, canonical)
    assert sha256(seed) == sha256(canonical), (sidecar, seed, canonical)
    return meta


def basic_valid(model: str, video: Path, uid: str, direction: str,
                seed_frame32: Path, seed_stream: Path) -> bool:
    try:
        assert video.exists()
        sidecar = Path(str(video) + ".json")
        assert sidecar.exists()
        validate_sidecar(model, sidecar, uid, direction, seed_frame32, seed_stream)
        expected = {
            "lingbot": (493, 16.0), "dreamx": (489, 16.0),
            "matrixgame2": (753, 25.0), "minwm": (509, 16.0),
            "minwm_ode": (493, 16.0),
        }[model]
        assert probe(video) == expected, (video, probe(video), expected)
        return True
    except (AssertionError, KeyError, ValueError, OSError, subprocess.SubprocessError,
            json.JSONDecodeError) as exc:
        print(f"[frodo8 invalid] {video}: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        return False


def related_paths(model: str, video: Path, uid: str, direction: str) -> list[Path]:
    paths = [video, Path(str(video) + ".json"), video.with_name(video.stem + "_latents.npz")]
    if model == "dreamx":
        # The stable DreamX path is a symlink to the vendor-named output.  If
        # the target is incomplete it must be quarantined too, otherwise the
        # vendor loop will see it and incorrectly skip regeneration.
        if video.is_symlink():
            paths.append(video.resolve(strict=False))
        parent = f"{uid}_{direction}_seed_frame32_seed65_{uid}_f0.mp4"
        paths.append(video.parent / parent)
    unique: list[Path] = []
    for path in paths:
        if path not in unique:
            unique.append(path)
    return unique


def prepare(args: argparse.Namespace, manifest: dict) -> None:
    selected = set(args.uids.split(",")) if args.uids else {
        row["uid"] for row in manifest["contexts"]
    }
    known = {row["uid"] for row in manifest["contexts"]}
    assert selected and selected <= known, (selected, known)
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.archive / args.model
    retained = resumable_vendor = quarantined = 0
    for uid in sorted(selected):
        for direction in DIRS:
            video = clip_path(args.model, args.output, uid, direction)
            if basic_valid(args.model, video, uid, direction,
                           args.seed_frame32, args.seed_stream):
                retained += 1
                continue
            if args.model == "dreamx":
                vendor = args.output / f"{uid}_{direction}_seed_frame32_seed65_{uid}_f0.mp4"
                try:
                    vendor_complete = probe(vendor) == (489, 16.0)
                except (OSError, subprocess.SubprocessError, ValueError, KeyError,
                        json.JSONDecodeError):
                    vendor_complete = False
                if vendor_complete:
                    # The vendor main loop skips this complete render, after
                    # which dreamx_runner recreates the stable link and exact
                    # sidecar.  Keep the expensive output and only clear a
                    # stale/broken stable view of it.
                    archive.mkdir(parents=True, exist_ok=True)
                    for path in (video, Path(str(video) + ".json")):
                        if path.exists() or path.is_symlink():
                            target = archive / path.name
                            if target.exists() or target.is_symlink():
                                target = archive / f"{int(time.time_ns())}_{path.name}"
                            shutil.move(str(path), str(target))
                    resumable_vendor += 1
                    continue
            candidates = [p for p in related_paths(args.model, video, uid, direction)
                          if p.exists() or p.is_symlink()]
            if not candidates:
                continue
            archive.mkdir(parents=True, exist_ok=True)
            for path in candidates:
                target = archive / path.name
                if target.exists() or target.is_symlink():
                    target = archive / f"{int(time.time_ns())}_{path.name}"
                shutil.move(str(path), str(target))
            quarantined += 1
    print(json.dumps({"model": args.model, "uids": sorted(selected),
                      "retained": retained, "resumable_vendor": resumable_vendor,
                      "quarantined_pairs": quarantined,
                      "archive": str(archive)}, indent=2))


def crop_resize(frame: np.ndarray, height: int, width: int) -> np.ndarray:
    source_h, source_w = frame.shape[:2]
    target_ratio = width / height
    source_ratio = source_w / source_h
    if source_ratio > target_ratio:
        new_w = int(round(source_h * target_ratio))
        left = (source_w - new_w) // 2
        frame = frame[:, left:left + new_w]
    elif source_ratio < target_ratio:
        new_h = int(round(source_w / target_ratio))
        top = (source_h - new_h) // 2
        frame = frame[top:top + new_h]
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def decode_frame(path: Path, index: int) -> np.ndarray:
    """Decode one exact frame, with ffmpeg as a fail-closed OpenCV fallback."""
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    if ok and frame is not None:
        return frame

    probe_raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(path),
    ], text=True)
    stream = json.loads(probe_raw)["streams"][0]
    width, height = int(stream["width"]), int(stream["height"])
    raw = subprocess.check_output([
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vf", f"select=eq(n\\,{index})", "-vsync", "0", "-frames:v", "1",
        "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
    ])
    expected = width * height * 3
    if len(raw) != expected:
        raise RuntimeError(
            f"could not decode frame {index} from {path}: "
            f"expected {expected} bytes, received {len(raw)}")
    return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)


def validate(args: argparse.Namespace, manifest: dict) -> None:
    all_contexts = [row["uid"] for row in manifest["contexts"]]
    contexts = args.uids.split(",") if args.uids else all_contexts
    assert contexts and len(contexts) == len(set(contexts)), contexts
    assert set(contexts) <= set(all_contexts), (contexts, all_contexts)
    context_frames = {"lingbot": 1, "dreamx": 1, "matrixgame2": 1,
                      "minwm": 29, "minwm_ode": 13}[args.model]
    expected_fps = {"lingbot": 16.0, "dreamx": 16.0, "matrixgame2": 25.0,
                    "minwm": 16.0, "minwm_ode": 16.0}[args.model]
    rows = []
    for uid in contexts:
        canonical_path = args.seed_frame32 / f"seed65_{uid}_f0.png"
        canonical = cv2.imread(str(canonical_path), cv2.IMREAD_COLOR)
        assert canonical is not None, canonical_path
        # Independently establish that the canonical PNG really is source
        # frame 32, rather than trusting its filename or generation sidecar.
        source32 = decode_frame(args.seed_stream / f"seed65_{uid}.mp4", 32)
        assert source32.shape == canonical.shape, uid
        delta = np.abs(source32.astype(np.int16) - canonical.astype(np.int16))
        assert float(delta.mean()) <= 1.1 and int(delta.max()) <= 4, (
            uid, float(delta.mean()), int(delta.max()))

        for direction in DIRS:
            video = clip_path(args.model, args.output, uid, direction)
            assert basic_valid(args.model, video, uid, direction,
                               args.seed_frame32, args.seed_stream), video
            frames, fps = probe(video)
            assert abs(fps - expected_fps) < 1e-9
            indices = (context_frames - 1,
                       context_frames + round(6 * fps) - 1,
                       context_frames + round(30 * fps) - 1)
            decoded = [decode_frame(video, index) for index in indices]
            boundary = decoded[0]
            canonical_scaled = crop_resize(canonical, boundary.shape[0], boundary.shape[1])
            mae = float(np.abs(boundary.astype(np.int16) -
                                canonical_scaled.astype(np.int16)).mean())
            assert mae < 40.0, (video, "frame-32 mismatch", mae)
            def black(frame: np.ndarray) -> bool:
                return float(frame.mean()) <= 3 or float(frame.std()) <= 3 or int(frame.max()) <= 10
            # A black conditioning frame means the runner is broken.  A later
            # black frame is a model result and must be reported, not hidden.
            assert not black(boundary), (video, "black conditioning frame")
            rows.append({"uid": uid, "direction": direction, "frames": frames,
                         "fps": fps, "context_frame32_mae": mae,
                         "black_at_6s": black(decoded[1]),
                         "black_at_30s": black(decoded[2])})
    assert len(rows) == len(contexts) * len(DIRS)
    report = {
        "model": args.model,
        "dataset": "FrodoBots",
        "uids": contexts,
        "actions": list(DIRS),
        "videos": len(rows),
        "generation_boundary_real_frame": 32,
        "maximum_context_frame32_mae": max(row["context_frame32_mae"] for row in rows),
        "black_at_6s": sum(row["black_at_6s"] for row in rows),
        "black_at_30s": sum(row["black_at_30s"] for row in rows),
        "black_examples_6s": [f'{row["uid"]}_{row["direction"]}' for row in rows
                              if row["black_at_6s"]],
        "black_examples_30s": [f'{row["uid"]}_{row["direction"]}' for row in rows
                               if row["black_at_30s"]],
    }
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("verify-manifest", "prepare", "validate"))
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-manifest", required=True, type=Path)
    parser.add_argument("--model", choices=MODELS)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed-frame32", type=Path)
    parser.add_argument("--seed-stream", type=Path)
    parser.add_argument("--uids", default="")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    manifest = load_manifest(args.manifest, args.source_manifest)
    if args.command == "verify-manifest":
        print(json.dumps({"manifest": str(args.manifest), "uids": [
            row["uid"] for row in manifest["contexts"]], "videos_per_model": 72}, indent=2))
        return
    for name in ("model", "output", "seed_frame32", "seed_stream"):
        assert getattr(args, name) is not None, f"--{name.replace('_', '-')} is required"
    if args.command == "prepare":
        assert args.archive is not None, "--archive is required for prepare"
        prepare(args, manifest)
    else:
        validate(args, manifest)


if __name__ == "__main__":
    main()
