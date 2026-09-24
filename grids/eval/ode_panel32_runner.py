#!/usr/bin/env python3
"""Generate a locked mixed-panel action with a full-rank ODE student.

The runner is intentionally checkpoint-generic.  A caller supplies the model
ID, objective label, checkpoint, and portable base-model paths; the panel
contract remains fixed:

* all 32 full dataset-prefixed context IDs come from the panel manifest;
* canonical source frames 0--32 are the exact nine-latent/three-action bundle;
* all three real chunks clean-fill the cache before generation;
* one compass action is held for 40 three-latent chunks (480 pixel frames);
* noise is derived from the full context ID only, so action/model branches are
  matched and resume/shard order cannot change a sample;
* the neutral prompt and its embedding are loaded from the bundle index.

Four independent one-GPU processes should invoke this with shard indices 0--3.
The Isambard holder payload does that; this module itself never submits jobs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Iterator, Mapping

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from grids.eval.panel32_manifest import (  # noqa: E402
    ACTIONS as PANEL_ACTIONS,
    sha256_file,
    load_panel_manifest,
)
from grids.eval.panel32_seed_bundles import (  # noqa: E402
    LATENT_FRAMES as SEED_LATENT_FRAMES,
    NEUTRAL_PROMPT,
    PIXEL_FRAMES as SEED_PIXEL_FRAMES,
    SEED_SPANS,
)
from interactive.engine_api import (  # noqa: E402
    PHYSICAL_NULL_STEER,
    PHYSICAL_NULL_THROTTLE,
)


PROMPT = NEUTRAL_PROMPT
ACTION_ORDER = ("F", "FR", "R", "BR", "B", "BL", "L", "FL", "N")
ACTION_MAGNITUDE = 0.5
DIAGONAL_MAGNITUDE = ACTION_MAGNITUDE / (2.0 ** 0.5)
ACTION_VECTORS = {
    "F": (ACTION_MAGNITUDE, 0.0),
    "FR": (DIAGONAL_MAGNITUDE, DIAGONAL_MAGNITUDE),
    "R": (0.0, ACTION_MAGNITUDE),
    "BR": (-DIAGONAL_MAGNITUDE, DIAGONAL_MAGNITUDE),
    "B": (-ACTION_MAGNITUDE, 0.0),
    "BL": (-DIAGONAL_MAGNITUDE, -DIAGONAL_MAGNITUDE),
    "L": (0.0, -ACTION_MAGNITUDE),
    "FL": (DIAGONAL_MAGNITUDE, -DIAGONAL_MAGNITUDE),
    "N": (PHYSICAL_NULL_THROTTLE, PHYSICAL_NULL_STEER),
}

LATENTS_PER_CHUNK = 3
GENERATED_CHUNKS = 40
GENERATED_LATENT_FRAMES = GENERATED_CHUNKS * LATENTS_PER_CHUNK
PIXELS_PER_GENERATED_CHUNK = 12
GENERATED_PIXEL_FRAMES = GENERATED_CHUNKS * PIXELS_PER_GENERATED_CHUNK
TOTAL_LATENT_FRAMES = SEED_LATENT_FRAMES + GENERATED_LATENT_FRAMES
TOTAL_PIXEL_FRAMES = SEED_PIXEL_FRAMES + GENERATED_PIXEL_FRAMES
OUTPUT_FPS = 16.0
CACHE_CHUNKS = 6  # plus the query chunk = the trained seven-chunk/21-frame span
DENOISING_RUNGS = (1000.0, 625.0, 357.142857, 208.333333)
SCHEMA_VERSION = 1
MODEL_ID_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

# Every one of these changes generate_ar's output.  A holder environment can
# outlive an experiment, so inheriting one silently would invalidate matching.
FORBIDDEN_INFERENCE_ENV = (
    "ODE_ACTION_CFG", "ODE_EMDHEAD", "ODE_EMDREMAP", "ODE_EMDREMAP_ASENS",
    "ODE_FLOW_DET", "ODE_FLOW_REC", "ODE_FLOW_SEED", "ODE_GEXCL",
    "ODE_GEXCL_R0", "ODE_KLTS_CKPT", "ODE_KLTS_FIXED", "ODE_REPNUDGE_ETA",
    "ODE_SAMPLER", "ODE_SL_KMU", "ODE_SL_KSTD", "ODE_STAT_LOCK",
    "ODE_VRFM", "ODE_VRFM_ZDIM",
)


class ContractError(RuntimeError):
    """A panel input, checkpoint, or existing output violates the contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def prompt_sha256() -> str:
    return hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()


def sampling_seed(context_id: str, base_seed: int = 1234) -> int:
    """Stable full-ID seed; deliberately excludes model and action."""
    digest = hashlib.sha256(
        b"arrwm-panel32-ode-v1\0"
        + str(base_seed).encode("ascii")
        + b"\0"
        + context_id.encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def stable_path(output: Path, model_id: str, context_id: str, action: str) -> Path:
    return output / f"{model_id}_{context_id}_{action}.mp4"


def build_action_tensor(
    seed_actions: torch.Tensor, action: str, *, device: str | torch.device = "cpu"
) -> torch.Tensor:
    """Return [1, 129, 2]: three GT seed actions then one held command."""
    _require(action in ACTION_VECTORS, f"unknown action {action!r}")
    _require(
        isinstance(seed_actions, torch.Tensor)
        and tuple(seed_actions.shape) == (3, 2),
        f"seed_actions must have shape (3, 2), got {getattr(seed_actions, 'shape', None)}",
    )
    _require(bool(torch.isfinite(seed_actions).all()), "seed_actions are non-finite")
    result = torch.empty(
        (1, TOTAL_LATENT_FRAMES, 2), dtype=torch.float32, device=device
    )
    result[:, :SEED_LATENT_FRAMES] = seed_actions.float().to(device).repeat_interleave(
        LATENTS_PER_CHUNK, dim=0
    )
    result[:, SEED_LATENT_FRAMES:, 0] = ACTION_VECTORS[action][0]
    result[:, SEED_LATENT_FRAMES:, 1] = ACTION_VECTORS[action][1]
    return result


def load_source_rows(path: Path, panel) -> tuple[dict[str, Any], dict[str, dict]]:
    payload = _json(path)
    _require(payload.get("status") == "pass", "source provenance status is not pass")
    _require(payload.get("complete_panel") is True, "source provenance is incomplete")
    _require(payload.get("panel_id") == panel.panel_id, "source panel ID mismatch")
    _require(
        payload.get("manifest_sha256") == panel.sha256,
        "source panel-manifest SHA mismatch",
    )
    rows = payload.get("rows")
    _require(isinstance(rows, list) and len(rows) == 32, "source needs 32 rows")
    by_id = {
        row.get("context_id"): row for row in rows if isinstance(row, dict)
    }
    expected = {context.context_id for context in panel.contexts}
    _require(len(by_id) == 32 and set(by_id) == expected, "source context IDs mismatch")
    return payload, by_id


def load_bundle_index(
    path: Path, panel
) -> tuple[dict[str, Any], dict[str, dict], torch.Tensor, Path]:
    payload = _json(path)
    _require(
        payload.get("status") == "pass" and payload.get("contexts") == 32,
        "seed-bundle index is not a complete pass",
    )
    _require(payload.get("panel_id") == panel.panel_id, "bundle panel ID mismatch")
    _require(
        payload.get("panel_manifest_sha256") == panel.sha256,
        "bundle panel-manifest SHA mismatch",
    )
    _require(payload.get("seed_pixel_frames") == SEED_PIXEL_FRAMES,
             "bundle index seed pixel count is not 33")
    _require(payload.get("seed_latent_frames") == SEED_LATENT_FRAMES,
             "bundle index seed latent count is not 9")
    _require(
        payload.get("seed_action_spans_half_open") == [list(x) for x in SEED_SPANS],
        "bundle index seed-action spans changed",
    )
    _require(payload.get("neutral_prompt") == PROMPT, "bundle prompt text mismatch")
    _require(
        payload.get("neutral_prompt_sha256") == prompt_sha256(),
        "bundle prompt SHA mismatch",
    )
    prompt_path = Path(str(payload.get("neutral_prompt_embedding", ""))).resolve()
    _require(prompt_path.is_file(), f"neutral prompt bundle missing: {prompt_path}")
    _require(
        sha256_file(prompt_path) == payload.get("neutral_prompt_embedding_sha256"),
        "neutral prompt bundle changed",
    )
    prompt_record = torch.load(prompt_path, map_location="cpu", weights_only=False)
    _require(
        isinstance(prompt_record, dict) and prompt_record.get("prompt") == PROMPT,
        "neutral prompt bundle does not contain the exact prompt",
    )
    prompt_embeds = prompt_record.get("prompt_embeds")
    _require(
        isinstance(prompt_embeds, torch.Tensor)
        and tuple(prompt_embeds.shape) == (1, 512, 4096)
        and bool(torch.isfinite(prompt_embeds).all()),
        "neutral prompt embedding must be finite [1, 512, 4096]",
    )
    rows = payload.get("rows")
    _require(isinstance(rows, list) and len(rows) == 32, "bundle index needs 32 rows")
    by_id = {
        row.get("context_id"): row for row in rows if isinstance(row, dict)
    }
    expected = {context.context_id for context in panel.contexts}
    _require(len(by_id) == 32 and set(by_id) == expected, "bundle context IDs mismatch")
    return payload, by_id, prompt_embeds.float(), prompt_path


def load_context_bundle(
    context, index_row: Mapping[str, Any], source_row: Mapping[str, Any], panel
) -> tuple[dict[str, Any], Path]:
    _require(index_row.get("context_id") == context.context_id,
             f"{context.context_id}: bundle index ID mismatch")
    _require(index_row.get("dataset") == context.dataset,
             f"{context.context_id}: bundle index dataset mismatch")
    bundle_path = Path(str(index_row.get("bundle", ""))).resolve()
    _require(bundle_path.is_file(), f"missing seed bundle: {bundle_path}")
    _require(sha256_file(bundle_path) == index_row.get("bundle_sha256"),
             f"seed bundle changed: {bundle_path}")
    bundle = torch.load(bundle_path, map_location="cpu", weights_only=False)
    _require(isinstance(bundle, dict), f"{bundle_path}: bundle is not an object")
    _require(bundle.get("context_id") == context.context_id,
             f"{context.context_id}: bundle embedded ID mismatch")
    _require(bundle.get("dataset") == context.dataset,
             f"{context.context_id}: bundle embedded dataset mismatch")
    _require(bundle.get("panel_id") == panel.panel_id,
             f"{context.context_id}: bundle panel ID mismatch")
    _require(bundle.get("panel_manifest_sha256") == panel.sha256,
             f"{context.context_id}: bundle manifest SHA mismatch")
    _require(bundle.get("seed_pixel_frames") == SEED_PIXEL_FRAMES,
             f"{context.context_id}: bundle does not encode 33 pixels")
    _require(bundle.get("seed_latent_frames") == SEED_LATENT_FRAMES,
             f"{context.context_id}: bundle does not contain 9 latents")
    _require(bundle.get("seed_action_spans_half_open") == [list(x) for x in SEED_SPANS],
             f"{context.context_id}: bundle action spans changed")
    seed = bundle.get("seed")
    seed_actions = bundle.get("seed_actions")
    _require(isinstance(seed, torch.Tensor) and tuple(seed.shape) == (9, 16, 60, 104),
             f"{context.context_id}: seed shape is not [9,16,60,104]")
    _require(isinstance(seed_actions, torch.Tensor)
             and tuple(seed_actions.shape) == (3, 2),
             f"{context.context_id}: seed actions shape is not [3,2]")
    _require(bool(torch.isfinite(seed).all()) and bool(torch.isfinite(seed_actions).all()),
             f"{context.context_id}: non-finite seed bundle")

    canonical = source_row.get("canonical", {})
    boundary = source_row.get("boundary", {})
    _require(source_row.get("dataset") == context.dataset,
             f"{context.context_id}: source dataset mismatch")
    _require(source_row.get("source_id") == context.source_id,
             f"{context.context_id}: source ID mismatch")
    _require(source_row.get("panel_manifest_sha256") == panel.sha256,
             f"{context.context_id}: source row manifest SHA mismatch")
    _require(boundary.get("canonical_frame_index") == 32,
             f"{context.context_id}: source boundary is not canonical frame 32")
    _require(
        (canonical.get("frame_count"), canonical.get("fps"),
         canonical.get("width"), canonical.get("height"))
        == (33, 20.0, 832, 480),
        f"{context.context_id}: canonical stream contract changed",
    )
    stream = Path(str(canonical.get("stream", ""))).resolve()
    _require(stream.is_file(), f"{context.context_id}: canonical stream missing")
    _require(sha256_file(stream) == canonical.get("stream_sha256"),
             f"{context.context_id}: canonical stream SHA mismatch")
    _require(bundle.get("source_stream_sha256") == canonical.get("stream_sha256"),
             f"{context.context_id}: bundle/source stream mismatch")
    _require(index_row.get("source_stream_sha256") == canonical.get("stream_sha256"),
             f"{context.context_id}: bundle-index/source stream mismatch")
    _require(
        bundle.get("source_decoded_rgb24_sha256")
        == canonical.get("decoded_rgb24_sha256"),
        f"{context.context_id}: bundle/source decoded pixels mismatch",
    )
    return bundle, stream


def decode_canonical(stream: Path) -> np.ndarray:
    try:
        payload = subprocess.check_output([
            "ffmpeg", "-v", "error", "-threads", "1", "-i", str(stream),
            "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ])
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"cannot decode canonical stream {stream}: {exc}") from exc
    expected = SEED_PIXEL_FRAMES * 480 * 832 * 3
    _require(len(payload) == expected,
             f"{stream}: decoded {len(payload)} bytes, expected {expected}")
    return np.frombuffer(payload, dtype=np.uint8).reshape(33, 480, 832, 3).copy()


def iter_decoded_chunks(vae, latents: torch.Tensor) -> Iterator[np.ndarray]:
    """Causally decode 3-latent chunks: 9 pixels first, then 12 each."""
    _require(tuple(latents.shape[1:]) == (TOTAL_LATENT_FRAMES, 16, 60, 104),
             f"full latent shape is wrong: {tuple(latents.shape)}")
    vae.model.clear_cache()
    try:
        for chunk_index, start in enumerate(range(0, TOTAL_LATENT_FRAMES, 3)):
            chunk = latents[:, start:start + 3].float()
            pixels = vae.decode_to_pixel(chunk, use_cache=True)
            video = (0.5 * (pixels[0].float() + 1.0)).clamp(0, 1)
            frames = (video.permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
            expected = 9 if chunk_index == 0 else 12
            _require(
                tuple(frames.shape) == (expected, 480, 832, 3),
                f"decoded chunk {chunk_index} shape {frames.shape}, expected "
                f"({expected}, 480, 832, 3)",
            )
            yield frames
    finally:
        vae.model.clear_cache()


def encode_video(
    destination: Path,
    canonical_frames: np.ndarray,
    decoded_chunks: Iterator[np.ndarray],
) -> dict[str, Any]:
    """Write 33 exact source pixels plus only the 40 generated chunks."""
    _require(tuple(canonical_frames.shape) == (33, 480, 832, 3),
             f"canonical frame shape changed: {canonical_frames.shape}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.stem}.tmp.{os.getpid()}.mp4"
    )
    command = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", "832x480",
        "-r", str(OUTPUT_FPS), "-i", "pipe:0", "-an", "-c:v", "libx264",
        "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
        "-metadata", "comment=canonical_frames=0-32;generation_start=33",
        str(temporary),
    ]
    generated_hash = hashlib.sha256()
    decoded_seed_hash = hashlib.sha256()
    generated_frames = 0
    decoded_seed_frames = 0
    proc = None
    try:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        assert proc.stdin is not None and proc.stderr is not None
        proc.stdin.write(np.ascontiguousarray(canonical_frames).tobytes())
        for chunk_index, frames in enumerate(decoded_chunks):
            raw = np.ascontiguousarray(frames).tobytes()
            if chunk_index < 3:
                decoded_seed_hash.update(raw)
                decoded_seed_frames += int(frames.shape[0])
            else:
                proc.stdin.write(raw)
                generated_hash.update(raw)
                generated_frames += int(frames.shape[0])
        proc.stdin.close()
        stderr = proc.stderr.read()
        returncode = proc.wait(timeout=600)
        _require(returncode == 0,
                 f"ffmpeg failed for {destination}: {stderr.decode(errors='replace')}")
        _require(decoded_seed_frames == SEED_PIXEL_FRAMES,
                 f"decoded latent seed has {decoded_seed_frames} frames, expected 33")
        _require(generated_frames == GENERATED_PIXEL_FRAMES,
                 f"decoded generated span has {generated_frames} frames, expected 480")
        temporary.replace(destination)
    except Exception:
        if proc is not None and proc.poll() is None:
            proc.kill()
            proc.wait()
        temporary.unlink(missing_ok=True)
        raise
    return {
        "decoded_seed_rgb24_sha256": decoded_seed_hash.hexdigest(),
        "generated_rgb24_sha256": generated_hash.hexdigest(),
        "generated_frames": generated_frames,
    }


def probe_video(path: Path) -> dict[str, Any]:
    try:
        raw = subprocess.check_output([
            "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames",
            "-of", "json", str(path),
        ])
        stream = json.loads(raw)["streams"][0]
    except (OSError, subprocess.CalledProcessError, KeyError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot probe output {path}: {exc}") from exc
    numerator, denominator = stream["avg_frame_rate"].split("/", 1)
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "fps": float(numerator) / float(denominator),
        "frames": int(stream["nb_read_frames"]),
    }


def contract_fingerprint(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def existing_output_is_current(
    output: Path, sidecar: Path, expected_fingerprint: str
) -> bool:
    if not output.exists() and not sidecar.exists():
        return False
    _require(output.is_file() and sidecar.is_file(),
             f"partial existing output: {output} / {sidecar}")
    prior = _json(sidecar)
    _require(prior.get("contract_fingerprint") == expected_fingerprint,
             f"stale conflicting output exists: {output}")
    _require(prior.get("output_sha256") == sha256_file(output),
             f"existing output SHA mismatch: {output}")
    probe = probe_video(output)
    _require(
        (probe["frames"], probe["width"], probe["height"], probe["fps"])
        == (TOTAL_PIXEL_FRAMES, 832, 480, OUTPUT_FPS),
        f"existing output geometry changed: {output} {probe}",
    )
    return True


def check_inference_environment() -> None:
    leaked = [name for name in FORBIDDEN_INFERENCE_ENV if os.environ.get(name)]
    _require(not leaked, f"behaviour-changing ODE environment is set: {leaked}")
    span = os.environ.get("EVAL_SPAN_MATCH_TRAINING", "1")
    _require(span.lower() not in ("0", "false", "no", "off"),
             "EVAL_SPAN_MATCH_TRAINING must remain enabled")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-manifest", type=Path, required=True)
    parser.add_argument("--source-provenance", type=Path, required=True)
    parser.add_argument("--bundle-index", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--wan-model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--objective", required=True)
    parser.add_argument("--action", choices=ACTION_ORDER, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=1234)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> None:
    _require(tuple(PANEL_ACTIONS) == ACTION_ORDER, "panel action order changed")
    _require(MODEL_ID_RE.fullmatch(args.model_id) is not None,
             f"unsafe model ID {args.model_id!r}")
    _require(MODEL_ID_RE.fullmatch(args.objective) is not None,
             f"unsafe objective {args.objective!r}")
    _require(args.base_seed >= 0, "base seed must be non-negative")
    _require(args.shard_count > 0 and 0 <= args.shard_index < args.shard_count,
             "invalid shard index/count")
    check_inference_environment()

    panel = load_panel_manifest(args.panel_manifest, verify_sources=False)
    _, source_rows = load_source_rows(args.source_provenance, panel)
    bundle_payload, bundle_rows, prompt_embeds, prompt_path = load_bundle_index(
        args.bundle_index, panel
    )
    source_provenance_hash = sha256_file(args.source_provenance)
    _require(
        bundle_payload.get("source_provenance_sha256") == source_provenance_hash,
        "bundle/source-provenance SHA mismatch",
    )
    for path, label in (
        (args.checkpoint, "student checkpoint"),
        (args.teacher_checkpoint, "teacher checkpoint"),
        (args.config, "config"),
    ):
        _require(path.is_file(), f"missing {label}: {path}")
    model_dir = args.wan_model_root / "Wan2.1-T2V-1.3B"
    _require(model_dir.is_dir(), f"Wan model directory missing: {model_dir}")

    selected = list(panel.contexts)[args.shard_index::args.shard_count]
    _require(bool(selected), "empty context shard")
    if args.dry_run:
        print(json.dumps({
            "status": "dry-run-pass",
            "model_id": args.model_id,
            "objective": args.objective,
            "action": args.action,
            "contexts": [context.context_id for context in selected],
            "seed_pixel_frames": SEED_PIXEL_FRAMES,
            "seed_latent_frames": SEED_LATENT_FRAMES,
            "generated_chunks": GENERATED_CHUNKS,
            "generated_pixel_frames": GENERATED_PIXEL_FRAMES,
            "prompt": PROMPT,
        }, indent=2, sort_keys=True))
        return

    checkpoint_hash = sha256_file(args.checkpoint)
    teacher_hash = sha256_file(args.teacher_checkpoint)
    config_hash = sha256_file(args.config)
    bundle_index_hash = sha256_file(args.bundle_index)
    runner_hash = sha256_file(Path(__file__))
    args.output.mkdir(parents=True, exist_ok=True)

    # ODERegression's training-only guard must not observe stale holder vars.
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ["ODE_EVAL_BUILD"] = "1"
    os.environ.pop("AF_SNAPSHOT_STEPS", None)
    os.environ.pop("AF_EVAL_STEPS", None)

    from utils.eval_causal_AR import ODEChainPipeline

    pipeline = ODEChainPipeline("cuda")
    pipeline.build(
        config_path=str(args.config),
        teacher_checkpoint=str(args.teacher_checkpoint),
        wan_model_root=str(args.wan_model_root),
        build_evaluators=False,
    )
    student_step = pipeline.load_checkpoint(str(args.checkpoint), use_ema=False)
    _require(
        int(getattr(pipeline.ode_model, "num_frame_per_block", -1))
        == LATENTS_PER_CHUNK,
        "ODE model chunk size is not three latent frames",
    )
    _require(
        int(getattr(pipeline.ode_model, "raw_action_dim", -1)) == 2,
        "ODE model action dimension is not two",
    )
    rungs = torch.tensor(DENOISING_RUNGS, dtype=torch.float32, device="cuda")
    pipeline.denoising_step_list = rungs
    pipeline.ode_model.denoising_step_list = rungs.clone()
    base_dit = pipeline.wrapper.model
    if hasattr(base_dit, "get_base_model"):
        base_dit = base_dit.get_base_model()

    complete = 0
    try:
        for context in selected:
            source_row = source_rows[context.context_id]
            index_row = bundle_rows[context.context_id]
            bundle, stream = load_context_bundle(
                context, index_row, source_row, panel
            )
            output = stable_path(
                args.output, args.model_id, context.context_id, args.action
            )
            sidecar = Path(str(output) + ".json")
            seed_value = sampling_seed(context.context_id, args.base_seed)
            identity = {
                "schema_version": SCHEMA_VERSION,
                "model_id": args.model_id,
                "objective": args.objective,
                "context_id": context.context_id,
                "dataset": context.dataset,
                "action": args.action,
                "panel_manifest_sha256": panel.sha256,
                "source_provenance_sha256": source_provenance_hash,
                "bundle_index_sha256": bundle_index_hash,
                "bundle_sha256": index_row["bundle_sha256"],
                "prompt_embedding_sha256": bundle_payload[
                    "neutral_prompt_embedding_sha256"
                ],
                "checkpoint_sha256": checkpoint_hash,
                "teacher_checkpoint_sha256": teacher_hash,
                "config_sha256": config_hash,
                "sampling_seed": seed_value,
                "denoising_rungs": list(DENOISING_RUNGS),
                "runner_sha256": runner_hash,
                "prompt_sha256": prompt_sha256(),
                "action_vector": {
                    "throttle": ACTION_VECTORS[args.action][0],
                    "steer": ACTION_VECTORS[args.action][1],
                },
                "cache_fill": "three-real-chunk-clean-fill",
                "cache_chunks": CACHE_CHUNKS,
                "generated_chunks": GENERATED_CHUNKS,
                "output_fps": OUTPUT_FPS,
            }
            fingerprint = contract_fingerprint(identity)
            if existing_output_is_current(output, sidecar, fingerprint):
                print(f"[ode-panel32] retain {output.name}", flush=True)
                complete += 1
                continue

            canonical_frames = decode_canonical(stream)
            canonical_rgb = np.ascontiguousarray(canonical_frames).tobytes()
            canonical_hash = hashlib.sha256(canonical_rgb).hexdigest()
            _require(
                canonical_hash == source_row["canonical"]["decoded_rgb24_sha256"],
                f"{context.context_id}: canonical decoded RGB SHA mismatch",
            )
            seed_latents = bundle["seed"].float().unsqueeze(0).to("cuda")
            action_tensor = build_action_tensor(
                bundle["seed_actions"], args.action, device="cuda"
            )
            torch.manual_seed(seed_value)
            torch.cuda.manual_seed_all(seed_value)
            start = time.time()
            from utils.infinity_rope import infinity_rope_active, is_active

            with infinity_rope_active(True, base_dit):
                _require(is_active(), "Infinity-RoPE did not install")
                full_latents = pipeline.generate_ar(
                    prompt_embeds=prompt_embeds,
                    noisy_fa_full=action_tensor,
                    initial_latents=seed_latents,
                    num_gen_chunks=GENERATED_CHUNKS,
                    cache_chunks=CACHE_CHUNKS,
                    chunks_per_step=1,
                    context_noise_timestep=0.0,
                    ar_cache=False,
                    cache_refresh="append",
                    carn_seam_affine_lambda=0.0,
                )
            _require(
                tuple(full_latents.shape) == (1, 129, 16, 60, 104),
                f"{context.context_id}: generated latent shape {full_latents.shape}",
            )
            _require(
                bool(torch.equal(
                    full_latents[:, :9].cpu(),
                    seed_latents.to(dtype=pipeline.dtype).float().cpu(),
                )),
                f"{context.context_id}: generate_ar changed the canonical latent seed",
            )
            pixel_hashes = encode_video(
                output, canonical_frames, iter_decoded_chunks(pipeline.vae, full_latents)
            )
            probe = probe_video(output)
            _require(
                (probe["frames"], probe["width"], probe["height"], probe["fps"])
                == (TOTAL_PIXEL_FRAMES, 832, 480, OUTPUT_FPS),
                f"{context.context_id}: output contract failed: {probe}",
            )
            source = source_row.get("source", {})
            boundary = source_row["boundary"]
            sidecar_payload = {
                **identity,
                "contract_fingerprint": fingerprint,
                "model": "ARRWM four-rung ODE student",
                "student_checkpoint": str(args.checkpoint.resolve()),
                "student_checkpoint_embedded_step": student_step,
                "teacher_checkpoint": str(args.teacher_checkpoint.resolve()),
                "config": str(args.config.resolve()),
                "panel_id": panel.panel_id,
                "panel_manifest": str(panel.path),
                "source_provenance": str(args.source_provenance.resolve()),
                "source_id": context.source_id,
                "source_video_sha256": source.get("sha256"),
                "source_stream": str(stream),
                "source_stream_sha256": source_row["canonical"]["stream_sha256"],
                "source_decoded_rgb24_sha256": canonical_hash,
                "source_boundary_original_rgb24_sha256": boundary.get(
                    "original_rgb24_sha256"
                ),
                "source_boundary_normalized_rgb24_sha256": boundary.get(
                    "normalized_rgb24_sha256"
                ),
                "bundle": str(Path(index_row["bundle"]).resolve()),
                "neutral_prompt_bundle": str(prompt_path),
                "prompt": PROMPT,
                "prompt_sha256": prompt_sha256(),
                "seed_match_group": context.context_id,
                "sampling_seed_algorithm": "sha256(arrwm-panel32-ode-v1,NUL,base,NUL,full-context-id)",
                "sampling_seed_action_invariant": True,
                "sampling_seed_model_invariant": True,
                "action_vector": {
                    "throttle": ACTION_VECTORS[args.action][0],
                    "steer": ACTION_VECTORS[args.action][1],
                },
                "physical_noop": args.action == "N",
                "physical_noop_vector": {
                    "throttle": PHYSICAL_NULL_THROTTLE,
                    "steer": PHYSICAL_NULL_STEER,
                },
                "seed_source_frames_inclusive": [0, 32],
                "seed_pixel_frames": SEED_PIXEL_FRAMES,
                "seed_latent_frames": SEED_LATENT_FRAMES,
                "seed_action_spans_half_open": [list(x) for x in SEED_SPANS],
                "seed_actions": [
                    [float(value) for value in row]
                    for row in bundle["seed_actions"].float().tolist()
                ],
                "seed_actions_lineage": bundle.get("seed_actions_lineage"),
                "cache_fill": "three-real-chunk-clean-fill",
                "cache_chunks": CACHE_CHUNKS,
                "trained_attention_span_latent_frames": 21,
                "infinity_rope": True,
                "generated_chunks": GENERATED_CHUNKS,
                "generated_latent_frames": GENERATED_LATENT_FRAMES,
                "generated_pixel_frames": GENERATED_PIXEL_FRAMES,
                "generation_boundary_canonical_frame": 32,
                "first_generated_output_frame": 33,
                "total_pixel_frames": TOTAL_PIXEL_FRAMES,
                "fps": OUTPUT_FPS,
                "output_context_pixels": "exact canonical RGB frames 0-32 before H.264 encoding",
                **pixel_hashes,
                "generation_seconds": time.time() - start,
                "output_sha256": sha256_file(output),
            }
            atomic_json(sidecar, sidecar_payload)
            print(
                f"[ode-panel32] saved {output.name} canonical=33 generated=480 "
                f"seed={seed_value}",
                flush=True,
            )
            complete += 1
            del full_latents, seed_latents, action_tensor
            torch.cuda.empty_cache()
    finally:
        del pipeline
        torch.cuda.empty_cache()

    _require(complete == len(selected),
             f"completed {complete}/{len(selected)} selected contexts")
    print(
        f"ODE_PANEL32_COMPLETE model={args.model_id} objective={args.objective} "
        f"action={args.action} shard={args.shard_index}/{args.shard_count} "
        f"contexts={complete}",
        flush=True,
    )


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
