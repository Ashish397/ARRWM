"""Fail-closed provenance and coverage checks for corrected external fleets."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np


DIRS = ("F", "FR", "R", "BR", "B", "BL", "L", "FL", "N")
DREAMX_KEYS = {
    "F": "w", "FR": "wl", "R": "l", "BR": "sl", "B": "s",
    "BL": "sj", "L": "j", "FL": "wj", "N": " ",
}
YUME_ACTIONS = {
    "F": ("W", "·", 4, 0, 0),
    "FR": ("W", "→", 4, 4, 4),
    "R": ("None", "→", 0, 4, 4),
    "BR": ("S", "→", 4, 4, 4),
    "B": ("S", "·", 4, 0, 0),
    "BL": ("S", "←", 4, 4, 4),
    "L": ("None", "←", 0, 4, 4),
    "FL": ("W", "←", 4, 4, 4),
    "N": ("None", "·", 0, 0, 0),
}


def dreamx_context_seed(uid: str, base_seed: int = 1) -> int:
    """Stable per-context seed shared by every action and shard ordering."""
    digest = hashlib.sha256(uid.encode("utf-8")).digest()
    return (base_seed + int.from_bytes(digest[:4], "big")) % (2**31 - 1)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def probe(path: Path) -> tuple[int, float]:
    raw = subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames,r_frame_rate", "-of", "json", str(path),
    ], text=True)
    stream = json.loads(raw)["streams"][0]
    num, den = map(int, stream["r_frame_rate"].split("/"))
    return int(stream["nb_read_frames"]), num / den


def decode_frames(path: Path, indices: tuple[int, ...]) -> list[np.ndarray]:
    """Decode exact frame indices with a single-threaded ffmpeg process."""
    metadata = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "json", str(path),
    ], text=True))["streams"][0]
    width, height = int(metadata["width"]), int(metadata["height"])
    unique = list(dict.fromkeys(indices))
    expression = "+".join(f"eq(n\\,{index})" for index in unique)
    result = subprocess.run([
        "ffmpeg", "-v", "error", "-threads", "1", "-filter_threads", "1",
        "-filter_complex_threads", "1", "-i", str(path),
        "-vf", f"select={expression}", "-vsync", "0",
        "-frames:v", str(len(unique)), "-pix_fmt", "bgr24",
        "-c:v", "rawvideo", "-threads:v", "1", "-f", "rawvideo", "pipe:1",
    ], capture_output=True, timeout=120)
    assert result.returncode == 0, (
        path, indices, result.stderr.decode("utf-8", errors="replace")[-1000:]
    )
    frame_bytes = width * height * 3
    assert len(result.stdout) == frame_bytes * len(unique), (
        path, indices, len(result.stdout), frame_bytes * len(unique)
    )
    decoded = {
        index: np.frombuffer(
            result.stdout[offset * frame_bytes:(offset + 1) * frame_bytes],
            dtype=np.uint8,
        ).reshape(height, width, 3).copy()
        for offset, index in enumerate(unique)
    }
    return [decoded[index] for index in indices]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--uids", required=True, type=Path)
    parser.add_argument("--seed-frame32", required=True, type=Path)
    parser.add_argument("--seed65-dir", required=True, type=Path)
    parser.add_argument("--minwm-ode-dir", type=Path)
    parser.add_argument("--matrix-runner", required=True, type=Path)
    parser.add_argument("--matrix-config", required=True, type=Path)
    parser.add_argument("--matrix-checkpoint", required=True, type=Path)
    parser.add_argument("--models", default="lingbot,dreamx,matrixgame2,minwm")
    parser.add_argument("--dreamx-dir", type=Path)
    parser.add_argument("--yume-dir", type=Path)
    parser.add_argument("--yume-runner", type=Path)
    parser.add_argument("--yume-vendor-entrypoint", type=Path)
    parser.add_argument("--yume-checkpoint", type=Path)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()
    cv2.setNumThreads(1)
    uid_payload = json.loads(args.uids.read_text())
    uid_rows = uid_payload["contexts"] if isinstance(uid_payload, dict) else uid_payload
    uids = [row["uid"] for row in uid_rows]
    expected = {(uid, direction) for uid in uids for direction in DIRS}
    expected_count = len(expected)
    selected_models = [x for x in args.models.split(",") if x]
    supported_models = {"lingbot", "dreamx", "matrixgame2", "minwm", "yume5b", "minwm_ode"}
    assert set(selected_models) <= supported_models, sorted(set(selected_models) - supported_models)
    if "dreamx" in selected_models:
        assert args.dreamx_dir is not None, "--dreamx-dir is required when validating DreamX"

    # Do not merely trust the filename of the shared one-frame condition.
    # Verify that every PNG is the decoded real-video frame 32.  PNGs were
    # extracted through ffmpeg; a second exact decode can differ by a few
    # integer RGB levels, hence the tight decode-tolerance rather than an
    # impossible byte comparison.
    for uid in uids:
        source = args.seed65_dir / f"seed65_{uid}.mp4"
        canonical = args.seed_frame32 / f"seed65_{uid}_f0.png"
        source_frame = decode_frames(source, (32,))[0]
        canonical_frame = cv2.imread(str(canonical), cv2.IMREAD_COLOR)
        assert canonical_frame is not None, (source, canonical)
        assert source_frame.shape == canonical_frame.shape, (source, canonical)
        delta = np.abs(source_frame.astype(np.int16) - canonical_frame.astype(np.int16))
        assert float(delta.mean()) <= 1.1 and int(delta.max()) <= 4, (
            uid, float(delta.mean()), int(delta.max())
        )

    specs = {
        "lingbot": ("lingbot_{uid}_{direction}.mp4", 481, 16.0, 1),
        "dreamx": ("dreamx_{uid}_{direction}.mp4", 481, 16.0, 1),
        "yume5b": ("yume5b_{uid}_{direction}.mp4", 481, 16.0, 1),
        "matrixgame2": ("matrixgame_{uid}_{direction}.mp4", 751, 25.0, 1),
        "minwm": ("minwm_aligned32_{uid}_{direction}.mp4", 509, 16.0, 29),
    }
    matrix_artifacts = {
        "runner": args.matrix_runner,
        "config": args.matrix_config,
        "checkpoint": args.matrix_checkpoint,
    }
    for name, path in matrix_artifacts.items():
        assert path.is_file(), (name, path)
    report = {
        "_matrix_provenance": {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in matrix_artifacts.items()
        }
    }
    if "yume5b" in selected_models:
        assert args.yume_dir is not None, "--yume-dir is required when validating YUME"
        yume_artifacts = {
            "runner": args.yume_runner,
            "vendor_entrypoint": args.yume_vendor_entrypoint,
            "checkpoint": args.yume_checkpoint,
        }
        for name, path in yume_artifacts.items():
            assert path is not None and path.is_file(), (name, path)
        report["_yume_provenance"] = {
            name: {"path": str(path.resolve()), "sha256": sha256(path)}
            for name, path in yume_artifacts.items()
        }
    png_hashes = {
        uid: sha256(args.seed_frame32 / f"seed65_{uid}_f0.png") for uid in uids
    }
    seed_video_hashes = {
        uid: sha256(args.seed65_dir / f"seed65_{uid}.mp4") for uid in uids
    }
    for model, (template, minimum, expected_fps, context_frames) in specs.items():
        if model not in selected_models:
            continue
        def validate_model_clip(
            item: tuple[str, str]
        ) -> tuple[str, str, int, float, bool, bool]:
            uid, direction = item
            disk_direction = "NOOP" if direction == "N" and model in {"matrixgame2", "minwm"} else direction
            if model == "yume5b":
                model_root = args.yume_dir
            elif model == "dreamx":
                model_root = args.dreamx_dir
            else:
                model_root = args.root / model
            video = model_root / template.format(uid=uid, direction=disk_direction)
            assert video.exists(), video
            sidecar = Path(str(video) + ".json")
            assert sidecar.exists(), sidecar
            meta = json.loads(sidecar.read_text())
            boundary = meta.get("generation_boundary_real_frame")
            if boundary is None and model == "minwm":
                boundary = int(meta["seed_start_frame"]) + int(meta["seed_frames"]) - 1
            assert boundary == 32, (sidecar, boundary)
            recorded_direction = meta.get("direction", meta.get("action"))
            assert recorded_direction in {direction, disk_direction}, (sidecar, recorded_direction)
            if model == "lingbot":
                assert meta.get("model") == "LingBot-World-V2-1.3B-Causal-Fast", sidecar
                assert meta.get("frames") == 493 and meta.get("fps") == 16, sidecar
                assert meta.get("seed") == 1, sidecar
                assert abs(float(meta.get("step_t")) - 0.25) < 1e-12, sidecar
                assert abs(float(meta.get("step_r_deg")) - 0.75) < 1e-12, sidecar
            elif model == "dreamx":
                assert meta.get("model") == "DreamX-World-5B", sidecar
                assert meta.get("latent_frames") == 123 and meta.get("fps") == 16, sidecar
                assert meta.get("seed") == dreamx_context_seed(uid), sidecar
                assert meta.get("seed_rule") == (
                    "base + first32bits(sha256(context_id)), modulo 2^31-1"
                ), sidecar
                assert meta.get("action_keys") == DREAMX_KEYS[direction], sidecar
            elif model == "yume5b":
                assert meta.get("model") == "YUME-5B-720P", sidecar
                assert meta.get("eval_model_key") == "yume5b", sidecar
                assert meta.get("total_frames") == 481 and meta.get("fps") == 16, sidecar
                assert meta.get("context_frames") == 1, sidecar
                assert meta.get("generated_frames") == 480, sidecar
                assert meta.get("generated_seconds") == 30.0, sidecar
                assert meta.get("generation_start_video_frame") == 1, sidecar
                assert meta.get("context_source_frames") == [32, 32], sidecar
                assert meta.get("rollout_segments") == 17, sidecar
                assert meta.get("base_seed") == 43, sidecar
                expected_action = YUME_ACTIONS[direction]
                observed_action = (
                    meta.get("yume_keyboard"), meta.get("yume_mouse_yaw"),
                    meta.get("yume_distance"), meta.get("yume_turn"),
                    meta.get("yume_rotation"),
                )
                assert observed_action == expected_action, (sidecar, observed_action, expected_action)
            elif model == "matrixgame2":
                assert meta.get("model") == "Matrix-Game-2.0", sidecar
                assert meta.get("direction") == disk_direction, (sidecar, meta.get("direction"))
                assert meta.get("frames") == 753 and meta.get("fps") == 25, sidecar
                assert meta.get("latent_frames") == 189, sidecar
                assert meta.get("sampling_seed") == 0, (sidecar, meta.get("sampling_seed"))
                assert meta.get("camera_yaw_per_frame") == 0.1, sidecar
                assert meta.get("keyboard_dims") == 4, sidecar
            if model == "yume5b":
                seed = Path(meta["context_source"])
                canonical = args.seed_frame32 / f"seed65_{uid}_f0.png"
                assert seed.exists() and canonical.exists(), sidecar
                assert sha256(seed) == png_hashes[uid], sidecar
                assert meta["context_source_sha256"] == png_hashes[uid], sidecar
            elif "seed_image" in meta:
                seed = Path(meta["seed_image"])
                canonical = args.seed_frame32 / f"seed65_{uid}_f0.png"
                assert seed.exists() and canonical.exists(), sidecar
                expected_hash = png_hashes[uid]
                assert sha256(seed) == expected_hash == meta["seed_sha256"], sidecar
            else:
                assert model == "minwm", (sidecar, "missing seed_image")
                assert meta["model"] == "minWM Wan2.1-1.3B Action2V 4-step DMD", sidecar
                assert meta["seed_start_frame"] == 4 and meta["seed_frames"] == 29, sidecar
                assert meta["seed_latents"] == 8, sidecar
                assert meta["total_latents"] == 128 and meta["generated_latents"] == 120, sidecar
                assert meta["output_frames"] == 509 and meta["generated_frames"] == 480, sidecar
                assert meta["fps"] == 16 and meta["generated_seconds"] == 30.0, sidecar
                assert meta["yaw_adapter"] == (
                    "released_minwm_yaw_negated_to_common_convention"
                ), sidecar
                seed = Path(meta["seed_video"])
                canonical = args.seed65_dir / f"seed65_{uid}.mp4"
                assert seed.exists() and canonical.exists(), sidecar
                assert sha256(seed) == seed_video_hashes[uid], sidecar
            frames, fps = probe(video)
            assert frames >= minimum and abs(fps - expected_fps) < 1e-6, (video, frames, fps)
            if model == "yume5b":
                assert frames == 481, (video, frames)
            # Test every clip, not only the pre-generation smoke.  This catches
            # a failed/black decode at the shared conditioning frame, six
            # generated seconds, or the final 30-second endpoint.
            indices = (
                context_frames - 1,
                context_frames + round(6 * fps) - 1,
                context_frames + round(30 * fps) - 1,
            )
            decoded = decode_frames(video, indices)
            health = []
            for index, frame in zip(indices, decoded):
                health.append((float(frame.mean()), float(frame.std()), int(frame.max())))
                if index == context_frames - 1:
                    frame32 = cv2.imread(
                        str(args.seed_frame32 / f"seed65_{uid}_f0.png"),
                        cv2.IMREAD_COLOR,
                    )
                    assert frame32 is not None
                    if frame32.shape != frame.shape:
                        frame32 = cv2.resize(
                            frame32, (frame.shape[1], frame.shape[0]),
                            interpolation=cv2.INTER_AREA,
                        )
                    mae = float(np.abs(
                        frame32.astype(np.int16) - frame.astype(np.int16)
                    ).mean())
                    # The model runners re-encode (and, for Matrix-Game,
                    # resize) the conditioning image.  This loose pixel check
                    # catches a wrong or black boundary frame while allowing
                    # those deterministic codec/preprocessing differences.
                    assert mae < 40.0, (video, "frame-32 mismatch", mae)
            # A black conditioning frame is a broken runner or wrong input.
            # Generated frames may genuinely collapse; record that model
            # failure rather than rejecting or concealing it at the fleet gate.
            mean, std, peak = health[0]
            assert mean > 3 and std > 3 and peak > 10, (video, health)
            is_black = lambda x: x[0] <= 3 or x[1] <= 3 or x[2] <= 10
            return uid, direction, frames, mae, is_black(health[1]), is_black(health[2])

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(validate_model_clip, sorted(expected)))
        assert len(rows) == expected_count
        report[model] = {"videos": len(rows), "minimum_frames": min(x[2] for x in rows),
                         "fps": expected_fps, "generation_boundary_real_frame": 32,
                         "maximum_context_frame32_mae": max(x[3] for x in rows),
                         "black_at_6s": sum(x[4] for x in rows),
                         "black_at_30s": sum(x[5] for x in rows),
                         "black_examples_6s": [f"{x[0]}_{x[1]}" for x in rows if x[4]],
                         "black_examples_30s": [f"{x[0]}_{x[1]}" for x in rows if x[5]]}

    if "minwm_ode" in selected_models:
        assert args.minwm_ode_dir is not None
        def validate_ode_clip(
            item: tuple[str, str]
        ) -> tuple[str, str, int, float, bool, bool]:
            uid, direction = item
            disk_direction = "NOOP" if direction == "N" else direction
            video = args.minwm_ode_dir / f"minwm_ode_{uid}_{disk_direction}.mp4"
            sidecar = Path(str(video) + ".json")
            assert video.exists() and sidecar.exists(), (video, sidecar)
            meta = json.loads(sidecar.read_text())
            assert meta["model"] == "minWM Wan2.1-1.3B Action2V causal ODE", sidecar
            assert meta["stage"] == "ode", sidecar
            assert meta["config"] == "causal_ode_camera.yaml", sidecar
            assert meta["seed_start_frame"] == 20 and meta["seed_frames"] == 13, sidecar
            assert meta["seed_start_frame"] + meta["seed_frames"] - 1 == 32, sidecar
            assert meta["seed_latents"] == 4, sidecar
            assert meta["yaw_adapter"] == (
                "released_minwm_yaw_negated_to_common_convention"
            ), sidecar
            frames, fps = probe(video)
            assert frames == 493 and abs(fps - 16.0) < 1e-6, (video, frames, fps)
            indices = (12, 12 + round(6 * fps), 12 + round(30 * fps))
            decoded = decode_frames(video, indices)
            health = []
            for frame in decoded:
                health.append((float(frame.mean()), float(frame.std()), int(frame.max())))
            frame = decoded[0]
            frame32 = cv2.imread(
                str(args.seed_frame32 / f"seed65_{uid}_f0.png"), cv2.IMREAD_COLOR
            )
            assert frame32 is not None and frame32.shape == frame.shape
            mae = float(np.abs(
                frame32.astype(np.int16) - frame.astype(np.int16)
            ).mean())
            assert mae < 40.0, (video, "frame-32 mismatch", mae)
            mean, std, peak = health[0]
            assert mean > 3 and std > 3 and peak > 10, (video, health)
            is_black = lambda x: x[0] <= 3 or x[1] <= 3 or x[2] <= 10
            return uid, direction, frames, mae, is_black(health[1]), is_black(health[2])

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            rows = list(pool.map(validate_ode_clip, sorted(expected)))
        assert len(rows) == expected_count
        report["minwm_ode"] = {
            "videos": len(rows),
            "minimum_frames": min(x[2] for x in rows),
            "fps": 16.0,
            "seed_latents": 4,
            "generation_boundary_real_frame": 32,
            "maximum_context_frame32_mae": max(x[3] for x in rows),
            "black_at_6s": sum(x[4] for x in rows),
            "black_at_30s": sum(x[5] for x in rows),
            "black_examples_6s": [f"{x[0]}_{x[1]}" for x in rows if x[4]],
            "black_examples_30s": [f"{x[0]}_{x[1]}" for x in rows if x[5]],
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
