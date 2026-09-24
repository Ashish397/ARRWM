"""Fail-closed validation of every generated panel32 model/action/context."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
from typing import Any

import cv2
import numpy as np


ACTIONS = ("F", "FR", "R", "BR", "B", "BL", "L", "FL", "N")
PROMPT = "A first-person view of an outdoor environment."
PROMPT_SHA = hashlib.sha256(PROMPT.encode()).hexdigest()


class ValidationError(RuntimeError):
    pass


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def probe(path: Path) -> tuple[int, float, int, int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValidationError(f"cannot open {path}")
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    if n < 1 or fps <= 0 or width < 1 or height < 1:
        raise ValidationError(f"invalid video header {path}: {(n, fps, width, height)}")
    return n, fps, width, height


def frame(path: Path, index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
    ok, image = cap.read()
    cap.release()
    if ok and image is not None:
        return image

    # OpenCV's FFmpeg backend can intermittently fail to initialise a colour
    # conversion context on the Isambard login/CPU nodes even though the same
    # file decodes correctly on a compute node.  Retry the exact frame through
    # a single-threaded ffmpeg process and decode its PNG payload.  This is a
    # decode fallback only: it neither changes the requested frame nor relaxes
    # any of the boundary checks below.
    command = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-threads", "1",
        "-i", str(path), "-vf", f"select=eq(n\\,{int(index)})", "-vsync", "0",
        "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "pipe:1",
    ]
    try:
        result = subprocess.run(
            command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=120,
        )
        image = cv2.imdecode(np.frombuffer(result.stdout, dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValidationError(
            f"cannot decode frame {index}: {path} (OpenCV and ffmpeg failed: {exc})"
        ) from exc
    if image is None:
        raise ValidationError(
            f"cannot decode frame {index}: {path} (ffmpeg returned no image)"
        )
    return image


def boundary_similarity(source: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    target = (640, 352)
    a = cv2.resize(source, target, interpolation=cv2.INTER_AREA)
    b = cv2.resize(candidate, target, interpolation=cv2.INTER_AREA)
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    mae = float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())
    ha = cv2.calcHist([a], [0, 1], None, [32, 32], [0, 256, 0, 256])
    hb = cv2.calcHist([b], [0, 1], None, [32, 32], [0, 256, 0, 256])
    hist = float(cv2.compareHist(ha, hb, cv2.HISTCMP_CORREL))
    orb = cv2.ORB_create(3000)
    k0, d0 = orb.detectAndCompute(ga, None)
    k1, d1 = orb.detectAndCompute(gb, None)
    inliers = 0
    if d0 is not None and d1 is not None and len(k0) >= 8 and len(k1) >= 8:
        matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(d0, d1)
        if len(matches) >= 8:
            p0 = np.float32([k0[m.queryIdx].pt for m in matches])
            p1 = np.float32([k1[m.trainIdx].pt for m in matches])
            _, mask = cv2.findHomography(p0, p1, cv2.RANSAC, 5.0)
            inliers = int(mask.sum()) if mask is not None else 0
    # The candidates include native resizes and VAE reconstructions.  This
    # gate catches a wrong/black boundary without demanding pixel identity.
    passed = inliers >= 8 or hist >= 0.70 or mae <= 35.0
    return {"orb_inliers": inliers, "histogram_correlation": hist,
            "mean_absolute_rgb_error": mae, "pass": bool(passed)}


def black_screen_stats(image: np.ndarray) -> dict[str, float | bool]:
    """Record all-black generated states as outcomes, never exclusions."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = float(grey.mean())
    std = float(grey.std())
    # The locked sources are visibly outdoor.  This deliberately narrow
    # diagnostic distinguishes a blank/collapsed render, but such renders stay
    # in the evaluation denominator and are scored by every applicable metric.
    return {"grey_mean": mean, "grey_std": std,
            "black_screen": bool(mean <= 3.0 and std <= 3.0)}


def sidecar_fields(payload: dict[str, Any]) -> dict[str, Any]:
    panel = payload.get("panel32") if isinstance(payload.get("panel32"), dict) else payload
    return {
        "context_id": panel.get("context_id", payload.get("window")),
        "action": panel.get("action", payload.get("action", payload.get("direction"))),
        "sampling_seed": panel.get("sampling_seed", payload.get("sampling_seed", payload.get("seed"))),
        "manifest_sha": panel.get("panel_manifest_sha256", payload.get("panel_manifest_sha256")),
        "boundary_sha": panel.get(
            "canonical_boundary_rgb24_sha256",
            payload.get("source_boundary_normalized_rgb24_sha256"),
        ),
        "prompt": panel.get(
            "neutral_prompt",
            payload.get("scene_prompt", payload.get("prompt", payload.get("caption"))),
        ),
        "prompt_sha": panel.get(
            "neutral_prompt_sha256",
            payload.get("scene_prompt_sha256", payload.get("prompt_sha256")),
        ),
    }


def validate_native_contract(
    model: str, payload: dict[str, Any], context_frames: int, disk_action: str
) -> None:
    """Reject native-context or input-adapter drift before expensive scoring."""
    if model == "matrixgame2":
        expected = {
            "model": "Matrix-Game-2.0",
            "inference_pipeline": "vendor CausalInferencePipeline",
            "mode": "universal",
            "native_latent_frames_per_block": 3,
            "keyboard_dims": 4,
            "camera_yaw_per_frame": 0.1,
            "latent_frames": 189,
            "frames": 753,
            "fps": 25,
            "generation_boundary_real_frame": 32,
            "sampling_seed": 0,
            "held_action_equivalence": (
                "constant pixel-frame tensors equal repeated vendor streaming "
                "actions for every three-latent-frame block"
            ),
        }
        for key, value in expected.items():
            if payload.get(key) != value:
                raise ValidationError(
                    f"{model}: native contract {key}={payload.get(key)!r}, "
                    f"expected {value!r}"
                )
        if context_frames != 1:
            raise ValidationError(
                f"{model}: config context_frames={context_frames}, expected 1"
            )
        if payload.get("direction") != disk_action:
            raise ValidationError(
                f"{model}: generated direction={payload.get('direction')!r}, "
                f"expected {disk_action!r}"
            )
        checkpoint = str(payload.get("checkpoint_path", ""))
        if not checkpoint.endswith("/base_distilled_model/base_distill.safetensors"):
            raise ValidationError(
                f"{model}: unexpected released checkpoint path {checkpoint!r}"
            )
        return
    if model not in ("minwm", "minwm_ode"):
        return
    expected = {
        "minwm": {
            "stage": "dmd", "config": "causal_forcing_dmd_camera.yaml",
            "seed_start_frame": 4, "seed_frames": 29, "seed_latents": 8,
            "total_latents": 128, "generated_latents": 120,
        },
        "minwm_ode": {
            "stage": "ode", "config": "causal_ode_camera.yaml",
            "seed_start_frame": 20, "seed_frames": 13, "seed_latents": 4,
            "total_latents": 124, "generated_latents": 120,
        },
    }[model]
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValidationError(
                f"{model}: native contract {key}={payload.get(key)!r}, expected {value!r}"
            )
    if context_frames != expected["seed_frames"]:
        raise ValidationError(
            f"{model}: config context_frames={context_frames}, expected {expected['seed_frames']}"
        )
    if payload.get("generation_boundary_real_frame") != 32:
        raise ValidationError(f"{model}: generation does not begin after source frame 32")
    if payload.get("direction") != disk_action:
        raise ValidationError(
            f"{model}: generated direction={payload.get('direction')!r}, expected {disk_action!r}"
        )
    if payload.get("yaw_adapter") != "released_minwm_yaw_negated_to_common_convention":
        raise ValidationError(f"{model}: missing corrected generation-time yaw adapter")
    if disk_action in ("FR", "BR", "BL", "FL") and payload.get(
        "camera_pose_update"
    ) != "released_rotate_then_local_translate":
        raise ValidationError(f"{model}: diagonal pose update is not release-faithful")


def validate(
    config_path: Path,
    models: list[str] | None = None,
    contexts: list[str] | None = None,
) -> dict[str, Any]:
    cfg = json.loads(config_path.read_text())
    selected = models or list(cfg["models"])
    unknown = set(selected) - set(cfg["models"])
    if unknown:
        raise ValidationError(f"unknown models: {sorted(unknown)}")
    selected_contexts = contexts or list(cfg["context_ids"])
    unknown_contexts = set(selected_contexts) - set(cfg["context_ids"])
    if unknown_contexts:
        raise ValidationError(f"unknown contexts: {sorted(unknown_contexts)}")
    if len(selected_contexts) != len(set(selected_contexts)):
        raise ValidationError("duplicate contexts requested")
    manifest_sha = cfg["panel_manifest_sha256"]
    provenance = json.loads(Path(cfg["source_provenance"]).read_text())
    sources = {row["context_id"]: row for row in provenance["rows"]}
    rows = []
    seeds: dict[tuple[str, str], set[Any]] = {}
    resolved_paths = set()
    errors = []
    for model in selected:
        record = cfg["models"][model]
        context_frames = int(record["context_frames"])
        for context_id in selected_contexts:
            source_record = sources[context_id]
            canonical = Path(source_record["canonical"]["stream"])
            reference = frame(canonical, 32)
            expected_boundary_sha = source_record["boundary"]["normalized_rgb24_sha256"]
            for action in ACTIONS:
                disk_action = record.get("noop_disk_action", "N") if action == "N" else action
                path = Path(record["path_template"].format(
                    context_id=context_id, action=disk_action
                ))
                try:
                    if not path.is_file():
                        raise ValidationError(f"missing video: {path}")
                    sidecar = Path(str(path) + ".json")
                    if not sidecar.is_file():
                        raise ValidationError(f"missing sidecar: {sidecar}")
                    resolved = str(path.resolve())
                    if resolved in resolved_paths:
                        raise ValidationError(f"duplicate resolved video path: {resolved}")
                    resolved_paths.add(resolved)
                    payload = json.loads(sidecar.read_text())
                    validate_native_contract(model, payload, context_frames, disk_action)
                    fields = sidecar_fields(payload)
                    if fields["context_id"] != context_id:
                        raise ValidationError(f"{path}: sidecar context mismatch {fields['context_id']!r}")
                    if fields["action"] not in (action, disk_action):
                        raise ValidationError(f"{path}: sidecar action mismatch {fields['action']!r}")
                    if fields["manifest_sha"] != manifest_sha:
                        raise ValidationError(f"{path}: panel manifest SHA mismatch")
                    if fields["boundary_sha"] != expected_boundary_sha:
                        raise ValidationError(f"{path}: canonical boundary SHA mismatch")
                    if model != "matrixgame2":
                        if fields["prompt"] != PROMPT and fields["prompt_sha"] != PROMPT_SHA:
                            raise ValidationError(f"{path}: neutral prompt mismatch")
                    n, fps, width, height = probe(path)
                    generated = n - context_frames
                    required = int(round(30 * fps))
                    if generated < required:
                        raise ValidationError(
                            f"{path}: only {generated} generated frames at {fps} fps; need {required}"
                        )
                    if not (math.isclose(fps, 16.0, abs_tol=0.05) or math.isclose(fps, 25.0, abs_tol=0.05)):
                        raise ValidationError(f"{path}: unexpected fps {fps}")
                    boundary = frame(path, context_frames - 1)
                    similarity = boundary_similarity(reference, boundary)
                    if not similarity["pass"]:
                        raise ValidationError(f"{path}: boundary mismatch {similarity}")
                    generated_checks = {}
                    for label, index in (
                            ("first", context_frames),
                            ("two_seconds", context_frames + int(round(2 * fps)) - 1)):
                        stats = black_screen_stats(frame(path, index))
                        generated_checks[label] = {"frame_index": index, **stats}
                    seed = fields["sampling_seed"]
                    if seed is None:
                        raise ValidationError(f"{path}: missing sampling seed")
                    seeds.setdefault((model, context_id), set()).add(str(seed))
                    rows.append({
                        "model": model, "context_id": context_id, "action": action,
                        "path": str(path), "sidecar": str(sidecar), "frames": n,
                        "fps": fps, "width": width, "height": height,
                        "context_frames": context_frames, "generated_frames": generated,
                        "sampling_seed": seed, "boundary_similarity": similarity,
                        "generated_frame_checks": generated_checks,
                    })
                except Exception as exc:
                    errors.append({"model": model, "context_id": context_id,
                                   "action": action, "path": str(path), "error": repr(exc)})
    for (model, context_id), values in seeds.items():
        if len(values) != 1:
            errors.append({"model": model, "context_id": context_id,
                           "error": f"action-dependent sampling seeds: {sorted(values)}"})
    expected = len(selected) * len(selected_contexts) * len(ACTIONS)
    if len(rows) != expected:
        errors.append({"error": f"validated {len(rows)}/{expected} required videos"})
    return {
        "status": "pass" if not errors else "fail",
        "config": str(config_path.resolve()), "models": selected,
        "contexts": selected_contexts,
        "expected_videos": expected, "validated_videos": len(rows),
        "action_invariant_seed_groups": len(seeds), "rows": rows, "errors": errors,
    }


def main() -> None:
    cv2.setNumThreads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--models", help="comma-separated subset")
    parser.add_argument("--contexts", help="comma-separated context subset")
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()
    result = validate(
        args.config.resolve(),
        args.models.split(",") if args.models else None,
        args.contexts.split(",") if args.contexts else None,
    )
    atomic_json(args.report.resolve(), result)
    print(json.dumps({key: value for key, value in result.items() if key != "rows"},
                     indent=2, sort_keys=True))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
