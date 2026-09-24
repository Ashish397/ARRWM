"""Prepare, package, and validate mixed-panel YUME-5B rollouts.

This adapter is intentionally separate from the Frodo8 runner and artifacts.
It consumes the locked ``panel32_manifest`` plus the provenance emitted by
``panel32_source.py``.  YUME receives only the canonical boundary PNG (frame
32) through its released single-image path.  The stable artifact contains
that one conditioning frame followed by exactly 480 generated frames at
16 fps.

Generation is still performed by YUME's released ``sample_5b.py``.  This file
only validates/stages inputs, records the exact contract, and packages its
final cumulative rollout.  Staging uses opaque deterministic tokens with an
explicit token-to-context map; dataset-prefixed context IDs are never parsed
by splitting a filename suffix.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np


ACTIONS = ("F", "FR", "R", "BR", "B", "BL", "L", "FL", "N")
ACTION_SPEC = {
    "F": {"keys": "W", "mouse": "·", "distance": 4, "turn": 0, "rotation": 0},
    "FR": {"keys": "W", "mouse": "→", "distance": 4, "turn": 4, "rotation": 4},
    "R": {"keys": "None", "mouse": "→", "distance": 0, "turn": 4, "rotation": 4},
    "BR": {"keys": "S", "mouse": "→", "distance": 4, "turn": 4, "rotation": 4},
    "B": {"keys": "S", "mouse": "·", "distance": 4, "turn": 0, "rotation": 0},
    "BL": {"keys": "S", "mouse": "←", "distance": 4, "turn": 4, "rotation": 4},
    "L": {"keys": "None", "mouse": "←", "distance": 0, "turn": 4, "rotation": 4},
    "FL": {"keys": "W", "mouse": "←", "distance": 4, "turn": 4, "rotation": 4},
    "N": {"keys": "None", "mouse": "·", "distance": 0, "turn": 0, "rotation": 0},
}
FPS = 16
CONTEXT_FRAMES = 1
GENERATED_FRAMES = 30 * FPS
TOTAL_FRAMES = CONTEXT_FRAMES + GENERATED_FRAMES
ROLLOUT_SEGMENTS = 17
FINAL_SEGMENT_INDEX = ROLLOUT_SEGMENTS - 1
BASE_SEED = 43
DEFAULT_PROMPT = (
    "A first-person view of an outdoor environment."
)
CONTRACT_SCHEMA_VERSION = 1
SIDECAR_SCHEMA_VERSION = 1


class ContractError(RuntimeError):
    """An input, prior artifact, or generation contract is inconsistent."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def cached_sha256(path: Path, cache_root: Path) -> str:
    stat = path.stat()
    key = hashlib.sha256(
        f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode("utf-8")
    ).hexdigest()
    cache_root.mkdir(parents=True, exist_ok=True)
    cache = cache_root / f"{key}.sha256"
    lock = cache_root / f"{key}.lock"
    with lock.open("w", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if not cache.exists():
            temporary = cache.with_name(f".{cache.name}.tmp.{os.getpid()}")
            temporary.write_text(sha256(path) + "\n", encoding="utf-8")
            temporary.replace(cache)
        value = cache.read_text(encoding="utf-8").strip()
    if len(value) != 64:
        raise ContractError(f"invalid cached SHA-256 for {path}: {value!r}")
    return value


def run_json(command: list[str]) -> dict[str, Any]:
    try:
        return json.loads(subprocess.check_output(command, text=True))
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        raise ContractError(f"command failed: {command}: {exc}") from exc


def probe(path: Path) -> dict[str, Any]:
    payload = run_json([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries",
        "stream=nb_read_frames,r_frame_rate,width,height,pix_fmt:format_tags",
        "-of", "json", str(path),
    ])
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise ContractError(f"expected one video/image stream in {path}")
    stream = streams[0]
    rate = stream.get("r_frame_rate", "0/1")
    numerator, denominator = map(int, rate.split("/"))
    frames = stream.get("nb_read_frames")
    return {
        "frames": int(frames) if frames not in (None, "N/A") else None,
        "fps": numerator / denominator,
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "pix_fmt": stream.get("pix_fmt"),
        "tags": payload.get("format", {}).get("tags", {}),
    }


def read_rgb(
    path: Path, frame: int, *, width: int, height: int, scale: bool = False
) -> bytes:
    filters = []
    if scale:
        filters.append(f"scale={width}:{height}:flags=bicubic")
    filters.append(f"select=eq(n\\,{frame})")
    try:
        payload = subprocess.check_output([
            "ffmpeg", "-v", "error", "-threads", "1", "-i", str(path),
            "-vf", ",".join(filters), "-frames:v", "1", "-f", "rawvideo",
            "-pix_fmt", "rgb24", "-",
        ])
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ContractError(f"cannot decode frame {frame} from {path}: {exc}") from exc
    expected = width * height * 3
    if len(payload) != expected:
        raise ContractError(
            f"decoded {len(payload)} bytes from {path} frame {frame}; expected {expected}"
        )
    return payload


def load_panel_module(repo_root: Path):
    root = str(repo_root.resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from grids.eval.panel32_manifest import load_panel_manifest
    except ImportError as exc:
        raise ContractError(
            f"cannot import grids.eval.panel32_manifest from {repo_root}"
        ) from exc
    return load_panel_manifest


def context_token(context_id: str) -> str:
    """Opaque vendor filename token; never recover IDs by parsing this token."""
    return "ctx-" + hashlib.sha256(context_id.encode("utf-8")).hexdigest()


def effective_seed(context_id: str, base_seed: int = BASE_SEED) -> int:
    # sample_5b.py derives its per-image seed from the staged image basename.
    token = context_token(context_id)
    offset = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:4], "big")
    return (base_seed + offset) % (2**31 - 1)


def stable_path(args: argparse.Namespace, context_id: str) -> Path:
    return args.output / f"yume5b_{context_id}_{args.action}.mp4"


def sidecar_path(video: Path) -> Path:
    return Path(str(video) + ".json")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"JSON root must be an object: {path}")
    return value


def assert_isolated_paths(args: argparse.Namespace) -> None:
    stage = args.stage.resolve()
    output = args.output.resolve()
    provenance = args.source_provenance.resolve()
    remote = args.remote_root.resolve()
    forbidden = {
        (remote / "aligned32_stage/yume5b_frodo8").resolve(),
        (remote / "aligned32_stage/fleet30s_aligned32/yume5b").resolve(),
    }
    _require(stage not in forbidden and output not in forbidden,
             "mixed-panel stage/output may not use a Frodo8 YUME path")
    _require(stage != output, "stage and stable output roots must differ")
    _require(not (stage == provenance.parent or output == provenance.parent),
             "generation paths may not overwrite the canonical source root")


def model_provenance(
    args: argparse.Namespace, *, hash_artifacts: bool = True
) -> dict[str, Any]:
    required = {
        "adapter": Path(__file__).resolve(),
        "diffusion_model": args.model_dir / "diffusion_pytorch_model.safetensors",
        "vae": args.model_dir / "Wan2.2_VAE.pth",
        "text_encoder": args.model_dir / "models_t5_umt5-xxl-enc-bf16.pth",
        "model_config": args.model_dir / "config.json",
        "inference_code": args.yume_root / "fastvideo/sample/sample_5b.py",
        "video_reader_fallback": args.yume_root / "fastvideo/utils/video_reader.py",
    }
    missing = [
        str(path) for path in required.values()
        if not path.is_file() or path.stat().st_size <= 0
    ]
    if missing:
        raise ContractError("missing YUME release assets: " + ", ".join(missing))
    hashes = {}
    for name, path in required.items():
        row: dict[str, Any] = {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
        }
        if hash_artifacts:
            row["sha256"] = cached_sha256(path, args.stage / "hashes")
        hashes[name] = row
    try:
        commit = subprocess.check_output(
            ["git", "-C", str(args.yume_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = "unknown"
    return {"official_repository_commit": commit, "artifacts": hashes}


def load_source_contract(args: argparse.Namespace) -> dict[str, Any]:
    load_panel_manifest = load_panel_module(args.repo_root)
    panel = load_panel_manifest(args.panel_manifest, verify_sources=False)
    summary = _load_json(args.source_provenance)
    _require(summary.get("status") == "pass", "source provenance status is not pass")
    _require(summary.get("complete_panel") is True,
             "source provenance is not a complete 32-context panel")
    _require(summary.get("panel_id") == panel.panel_id, "panel_id mismatch")
    _require(summary.get("manifest_sha256") == panel.sha256,
             "source provenance manifest SHA mismatch")
    _require(summary.get("extracted_contexts") == 32,
             "source provenance must contain exactly 32 extracted contexts")
    rows = summary.get("rows")
    _require(isinstance(rows, list) and len(rows) == 32,
             "source provenance rows must contain exactly 32 contexts")
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        _require(isinstance(row, dict), "source provenance row must be an object")
        context_id = row.get("context_id")
        _require(isinstance(context_id, str) and context_id not in by_id,
                 f"invalid or duplicate provenance context_id {context_id!r}")
        by_id[context_id] = row
    expected_ids = [context.context_id for context in panel.contexts]
    _require(set(by_id) == set(expected_ids),
             "manifest/provenance context ID sets differ")

    contexts = []
    for context in panel.contexts:
        row = by_id[context.context_id]
        _require(row.get("panel_manifest_sha256") == panel.sha256,
                 f"{context.context_id}: row manifest SHA mismatch")
        _require(row.get("dataset") == context.dataset,
                 f"{context.context_id}: dataset mismatch")
        _require(row.get("source_id") == context.source_id,
                 f"{context.context_id}: source_id mismatch")
        _require(row.get("source", {}).get("sha256") == context.source_sha256,
                 f"{context.context_id}: source SHA mismatch")
        boundary = row.get("boundary", {})
        canonical = row.get("canonical", {})
        wan = row.get("wan_interface", {})
        _require(boundary.get("canonical_frame_index") == 32,
                 f"{context.context_id}: canonical boundary is not frame 32")
        _require(
            (canonical.get("frame_count"), canonical.get("fps"),
             canonical.get("width"), canonical.get("height"))
            == (33, 20.0, 832, 480),
            f"{context.context_id}: canonical media contract mismatch",
        )
        _require((wan.get("pixel_frames"), wan.get("latent_frames")) == (33, 9),
                 f"{context.context_id}: Wan 33-pixel/9-latent contract mismatch")
        stream = Path(str(canonical.get("stream", ""))).resolve()
        boundary_png = Path(str(canonical.get("boundary_png", ""))).resolve()
        _require(stream.is_file() and boundary_png.is_file(),
                 f"{context.context_id}: canonical artifacts are missing")
        _require(sha256(stream) == canonical.get("stream_sha256"),
                 f"{context.context_id}: canonical stream SHA mismatch")
        _require(sha256(boundary_png) == canonical.get("boundary_png_sha256"),
                 f"{context.context_id}: boundary PNG SHA mismatch")
        stream_info = probe(stream)
        _require(
            (stream_info["frames"], stream_info["fps"], stream_info["width"],
             stream_info["height"]) == (33, 20.0, 832, 480),
            f"{context.context_id}: canonical stream probe mismatch {stream_info}",
        )
        boundary_rgb = read_rgb(boundary_png, 0, width=832, height=480)
        stream_boundary = read_rgb(stream, 32, width=832, height=480)
        _require(boundary_rgb == stream_boundary,
                 f"{context.context_id}: boundary PNG is not canonical frame 32")
        _require(sha256_bytes(boundary_rgb) == boundary.get("normalized_rgb24_sha256"),
                 f"{context.context_id}: normalized boundary RGB hash mismatch")
        contexts.append({
            "context_id": context.context_id,
            "dataset": context.dataset,
            "source_id": context.source_id,
            "source_video_sha256": context.source_sha256,
            "source_boundary_frame_index": boundary.get("source_frame_index"),
            "source_boundary_timestamp_s": boundary.get("source_timestamp_s"),
            "source_boundary_original_rgb24_sha256": boundary.get(
                "original_rgb24_sha256"
            ),
            "canonical_stream": str(stream),
            "canonical_stream_sha256": canonical["stream_sha256"],
            "canonical_decoded_rgb24_sha256": canonical.get("decoded_rgb24_sha256"),
            "canonical_boundary_png": str(boundary_png),
            "canonical_boundary_png_sha256": canonical["boundary_png_sha256"],
            "canonical_boundary_rgb24_sha256": boundary["normalized_rgb24_sha256"],
            "vendor_token": context_token(context.context_id),
            "effective_seed": effective_seed(context.context_id, args.base_seed),
        })
    return {
        "panel_id": panel.panel_id,
        "panel_manifest": str(panel.path),
        "panel_manifest_sha256": panel.sha256,
        "source_provenance": str(args.source_provenance.resolve()),
        "source_provenance_sha256": sha256(args.source_provenance),
        "contexts": contexts,
    }


def global_contract(args: argparse.Namespace, source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "model": "YUME-5B-720P",
        "eval_model_key": "yume5b",
        "panel_id": source["panel_id"],
        "panel_manifest": source["panel_manifest"],
        "panel_manifest_sha256": source["panel_manifest_sha256"],
        "source_provenance": source["source_provenance"],
        "source_provenance_sha256": source["source_provenance_sha256"],
        "context_count": 32,
        "context_frames": CONTEXT_FRAMES,
        "canonical_boundary_frame": 32,
        "generated_frames": GENERATED_FRAMES,
        "generated_seconds": 30,
        "fps": FPS,
        "total_frames": TOTAL_FRAMES,
        "rollout_segments": ROLLOUT_SEGMENTS,
        "base_seed": args.base_seed,
        "seed_rule": (
            "base_seed + first32bits(sha256(opaque_token)), modulo 2^31-1; "
            "opaque_token=ctx-hex(sha256(full_context_id)); matched across actions"
        ),
        "scene_prompt": args.scene_prompt,
        "scene_prompt_sha256": hashlib.sha256(
            args.scene_prompt.encode("utf-8")
        ).hexdigest(),
        "actions": list(ACTIONS),
        "action_specs": ACTION_SPEC,
    }


def lock_global_contract(args: argparse.Namespace, contract: Mapping[str, Any]) -> None:
    path = args.stage / "panel_contract.json"
    lock = args.stage / "panel_contract.lock"
    args.stage.mkdir(parents=True, exist_ok=True)
    with lock.open("w", encoding="utf-8") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        if path.exists():
            observed = _load_json(path)
            _require(observed == contract,
                     f"existing shared panel contract conflicts with this run: {path}")
        else:
            atomic_json(path, contract)


def expected_sidecar(
    args: argparse.Namespace, source: Mapping[str, Any], context: Mapping[str, Any]
) -> dict[str, Any]:
    spec = ACTION_SPEC[args.action]
    return {
        "schema_version": SIDECAR_SCHEMA_VERSION,
        "model": "YUME-5B-720P",
        "eval_model_key": "yume5b",
        "panel_id": source["panel_id"],
        "panel_manifest": source["panel_manifest"],
        "panel_manifest_sha256": source["panel_manifest_sha256"],
        "panel_source_provenance": source["source_provenance"],
        "panel_source_provenance_sha256": source["source_provenance_sha256"],
        "context_id": context["context_id"],
        "source_dataset": context["dataset"],
        "source_id": context["source_id"],
        "source_video_sha256": context["source_video_sha256"],
        "source_boundary_frame_index": context["source_boundary_frame_index"],
        "source_boundary_timestamp_s": context["source_boundary_timestamp_s"],
        "source_boundary_original_rgb24_sha256": context[
            "source_boundary_original_rgb24_sha256"
        ],
        "canonical_stream": context["canonical_stream"],
        "canonical_stream_sha256": context["canonical_stream_sha256"],
        "canonical_decoded_rgb24_sha256": context[
            "canonical_decoded_rgb24_sha256"
        ],
        "canonical_boundary_png": context["canonical_boundary_png"],
        "canonical_boundary_png_sha256": context[
            "canonical_boundary_png_sha256"
        ],
        "canonical_boundary_rgb24_sha256": context[
            "canonical_boundary_rgb24_sha256"
        ],
        "canonical_boundary_frame": 32,
        "action": args.action,
        "yume_keyboard": spec["keys"],
        "yume_mouse_yaw": spec["mouse"],
        "yume_distance": spec["distance"],
        "yume_turn": spec["turn"],
        "yume_rotation": spec["rotation"],
        "context_mode": "official native single-image conditioning",
        "context_frames": CONTEXT_FRAMES,
        "generation_start_video_frame": CONTEXT_FRAMES,
        "generated_frames": GENERATED_FRAMES,
        "generated_seconds": 30,
        "total_frames": TOTAL_FRAMES,
        "fps": FPS,
        "rollout_segments": ROLLOUT_SEGMENTS,
        "base_seed": args.base_seed,
        "vendor_input_token": context["vendor_token"],
        "sampling_seed": context["effective_seed"],
        "seed_match_group": context["context_id"],
        "seed_rule": (
            "base_seed + first32bits(sha256(vendor_input_token)), modulo 2^31-1; "
            "same context token and seed for every action"
        ),
        "scene_prompt": args.scene_prompt,
        "scene_prompt_sha256": hashlib.sha256(
            args.scene_prompt.encode("utf-8")
        ).hexdigest(),
    }


def validate_stable(
    args: argparse.Namespace, source: Mapping[str, Any], context: Mapping[str, Any]
) -> tuple[bool, str]:
    video = stable_path(args, str(context["context_id"]))
    sidecar = sidecar_path(video)
    if not video.exists() or not sidecar.exists():
        return False, "missing video or sidecar"
    try:
        info = probe(video)
        meta = _load_json(sidecar)
        for key, value in expected_sidecar(args, source, context).items():
            if meta.get(key) != value:
                return False, f"sidecar {key}={meta.get(key)!r}, expected {value!r}"
        model = meta.get("model_provenance")
        if not isinstance(model, dict) or not isinstance(model.get("artifacts"), dict):
            return False, "missing model provenance"
        for name in (
            "adapter", "diffusion_model", "vae", "text_encoder", "model_config",
            "inference_code", "video_reader_fallback",
        ):
            artifact = model["artifacts"].get(name)
            if not isinstance(artifact, dict):
                return False, f"missing model provenance artifact {name}"
            digest = artifact.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64:
                return False, f"invalid model provenance SHA for {name}"
        if (info["frames"], info["fps"]) != (TOTAL_FRAMES, float(FPS)):
            return False, f"media frames/fps {info['frames']}/{info['fps']}"
        if meta.get("stable_video_sha256") != sha256(video):
            return False, "stable video SHA mismatch"
        comment = info["tags"].get("comment", "")
        if "canonical_frame=32" not in comment or "generation_start_video_frame=1" not in comment:
            return False, "missing embedded boundary metadata"
        reference = read_rgb(
            Path(str(context["canonical_boundary_png"])), 0,
            width=info["width"], height=info["height"], scale=True,
        )
        packaged = read_rgb(
            video, 0, width=info["width"], height=info["height"]
        )
        mae = float(np.abs(
            np.frombuffer(reference, dtype=np.uint8).astype(np.int16)
            - np.frombuffer(packaged, dtype=np.uint8).astype(np.int16)
        ).mean())
        if mae > 8.0:
            return False, f"canonical frame-32 packaging MAE {mae:.4f}"
        # Decode exact first and last generated frames. Model collapse is a
        # measured result, but corrupt/truncated media is invalid.
        read_rgb(video, 1, width=info["width"], height=info["height"])
        read_rgb(video, TOTAL_FRAMES - 1, width=info["width"], height=info["height"])
    except Exception as exc:
        return False, str(exc)
    return True, "ok"


def plan(
    args: argparse.Namespace, *, include_model: bool, hash_model: bool = True
) -> dict[str, Any]:
    assert_isolated_paths(args)
    _require(
        args.scene_prompt == DEFAULT_PROMPT,
        f"panel scene prompt must be exactly {DEFAULT_PROMPT!r}",
    )
    source = load_source_contract(args)
    contexts = source["contexts"]
    pending = []
    for context in contexts:
        video = stable_path(args, context["context_id"])
        sidecar = sidecar_path(video)
        ok, reason = validate_stable(args, source, context)
        if ok:
            continue
        if video.exists() or sidecar.exists():
            raise ContractError(
                f"conflicting/partial stable artifact for {context['context_id']}: {reason}; "
                "use a clean isolated output root"
            )
        pending.append(context)
    payload = {
        **global_contract(args, source),
        "action": args.action,
        "action_spec": ACTION_SPEC[args.action],
        "output_root": str(args.output.resolve()),
        "stage_root": str(args.stage.resolve()),
        "contexts": contexts,
        "pending_context_ids": [row["context_id"] for row in pending],
        "valid_existing_context_ids": [
            row["context_id"] for row in contexts if row not in pending
        ],
    }
    if include_model:
        payload["model_provenance"] = model_provenance(
            args, hash_artifacts=hash_model
        )
    return payload


def dry_run(args: argparse.Namespace) -> None:
    payload = plan(args, include_model=True, hash_model=False)
    contract = global_contract(args, payload)
    contract_path = args.stage / "panel_contract.json"
    if contract_path.exists():
        _require(_load_json(contract_path) == contract,
                 f"existing shared panel contract conflicts: {contract_path}")
    print(json.dumps({
        "status": "pass",
        "mode": "dry-run (no files written)",
        "panel_id": payload["panel_id"],
        "action": args.action,
        "contexts": len(payload["contexts"]),
        "pending": len(payload["pending_context_ids"]),
        "prompt": payload["scene_prompt"],
        "output": payload["output_root"],
        "stage": payload["stage_root"],
    }, indent=2, sort_keys=True))


def prepare(args: argparse.Namespace) -> None:
    payload = plan(args, include_model=True)
    contract = global_contract(args, payload)
    lock_global_contract(args, contract)
    pending_ids = payload["pending_context_ids"]
    by_id = {row["context_id"]: row for row in payload["contexts"]}
    plan_hash = hashlib.sha256(
        json.dumps(
            {"action": args.action, "pending": pending_ids,
             "contract": contract}, sort_keys=True
        ).encode("utf-8")
    ).hexdigest()[:20]
    run_root = args.stage / "runs" / args.action / plan_hash
    inputs = run_root / "inputs"
    vendor = run_root / "vendor"
    inputs.mkdir(parents=True, exist_ok=True)
    vendor.mkdir(parents=True, exist_ok=True)

    items = []
    for context_id in pending_ids:
        context = by_id[context_id]
        token = context["vendor_token"]
        items.append({"vendor_token": token, "context_id": context_id, "padding": False})
    if items:
        padded_count = int(math.ceil(len(items) / 4.0) * 4)
        originals = list(items)
        for index in range(len(items), padded_count):
            source_item = originals[index % len(originals)]
            token = f"pad-{index:03d}-{source_item['vendor_token'][4:]}"
            items.append({
                "vendor_token": token,
                "context_id": source_item["context_id"],
                "padding": True,
            })
    for item in items:
        context = by_id[item["context_id"]]
        link = inputs / f"{item['vendor_token']}.png"
        target = Path(context["canonical_boundary_png"]).resolve()
        if link.is_symlink():
            _require(link.resolve() == target, f"conflicting staged link: {link}")
        elif link.exists():
            raise ContractError(f"staged input is not a symlink: {link}")
        else:
            link.symlink_to(target)
    unexpected = sorted(
        path.name for path in inputs.iterdir()
        if path.name not in {f"{item['vendor_token']}.png" for item in items}
    )
    _require(not unexpected, f"unexpected staged inputs in {inputs}: {unexpected}")
    payload.update({
        "status": "complete" if not pending_ids else "pending",
        "run_root": str(run_root.resolve()),
        "input_root": str(inputs.resolve()),
        "vendor_output": str(vendor.resolve()),
        "work_items": items,
    })
    action_manifest = args.stage / f"manifest_{args.action}.json"
    if action_manifest.exists():
        previous = _load_json(action_manifest)
        if previous != payload:
            atomic_json(action_manifest, payload)
    else:
        atomic_json(action_manifest, payload)
    print(json.dumps({
        "status": payload["status"], "action": args.action,
        "pending": len(pending_ids), "staged_items": len(items),
        "input_root": payload["input_root"],
        "vendor_output": payload["vendor_output"],
    }, indent=2, sort_keys=True))


def package_video(source: Path, generated: Path, destination: Path, panel_id: str) -> None:
    info = probe(generated)
    if info["frames"] is None or info["frames"] < GENERATED_FRAMES:
        raise ContractError(f"generated video is shorter than 480 frames: {generated} {info}")
    if not math.isclose(info["fps"], FPS):
        raise ContractError(f"generated video is not {FPS} fps: {generated} {info}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.stem}.tmp.{os.getpid()}.mp4")
    comment = (
        f"YUME-5B panel={panel_id}; context=canonical_frame_32; "
        "canonical_frame=32; generation_start_video_frame=1; "
        "generated_frames=480; fps=16"
    )
    try:
        subprocess.run([
            "ffmpeg", "-y", "-v", "error", "-threads", "4",
            "-loop", "1", "-framerate", str(FPS), "-i", str(source),
            "-i", str(generated), "-filter_complex",
            (
                "[0:v]trim=start_frame=0:end_frame=1,setpts=PTS-STARTPTS,"
                f"scale={info['width']}:{info['height']}:flags=bicubic,"
                "fps=16,setsar=1,format=yuv420p[ctx];"
                f"[1:v]trim=start_frame=0:end_frame={GENERATED_FRAMES},"
                "setpts=PTS-STARTPTS,fps=16,setsar=1,format=yuv420p[gen];"
                "[ctx][gen]concat=n=2:v=1:a=0[out]"
            ),
            "-map", "[out]", "-frames:v", str(TOTAL_FRAMES), "-an",
            "-c:v", "libx264", "-preset", "fast", "-crf", "12",
            "-pix_fmt", "yuv420p", "-r", str(FPS), "-movflags", "+faststart",
            "-metadata", "title=YUME-5B mixed panel aligned rollout",
            "-metadata", f"comment={comment}", str(temporary),
        ], check=True)
        temporary.replace(destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def finalize(args: argparse.Namespace) -> None:
    manifest_path = args.stage / f"manifest_{args.action}.json"
    manifest = _load_json(manifest_path)
    current = plan(args, include_model=True)
    for key in (
        "panel_id", "panel_manifest_sha256", "source_provenance_sha256",
        "scene_prompt", "scene_prompt_sha256", "base_seed", "action",
    ):
        _require(manifest.get(key) == current.get(key),
                 f"staged manifest {key} no longer matches current contract")
    contexts = {row["context_id"]: row for row in manifest["contexts"]}
    vendor = Path(manifest["vendor_output"])
    real_items = [item for item in manifest["work_items"] if not item["padding"]]
    for item in real_items:
        context_id = item["context_id"]
        context = contexts[context_id]
        already_valid, _ = validate_stable(args, manifest, context)
        if already_valid:
            print(
                f"YUME_PANEL32_RETAINED context_id={context_id} "
                f"action={args.action}"
            )
            continue
        token = item["vendor_token"]
        candidates = sorted(vendor.glob(f"{token}_*_{FINAL_SEGMENT_INDEX}.mp4"))
        if len(candidates) != 1:
            raise ContractError(
                f"expected exactly one final vendor video for token {token} "
                f"({context_id}), found {candidates}"
            )
        destination = stable_path(args, context_id)
        _require(not destination.exists() and not sidecar_path(destination).exists(),
                 f"refusing to overwrite invalid/partial stable artifact for {context_id}")
        package_video(
            Path(context["canonical_boundary_png"]), candidates[0], destination,
            manifest["panel_id"],
        )
        meta = {
            **expected_sidecar(args, manifest, context),
            "vendor_output": str(candidates[0].resolve()),
            "vendor_output_sha256": sha256(candidates[0]),
            "stable_video_sha256": sha256(destination),
            "packaging": "1 canonical conditioning frame + 480 generated frames; libx264 CRF 12",
            "model_provenance": manifest["model_provenance"],
        }
        atomic_json(sidecar_path(destination), meta)
        ok, reason = validate_stable(args, manifest, context)
        if not ok:
            raise ContractError(f"post-package validation failed for {context_id}: {reason}")
        print(
            f"YUME_PANEL32_PACKAGED context_id={context_id} action={args.action} "
            f"frames={TOTAL_FRAMES}"
        )


def validate_action(args: argparse.Namespace, *, require_complete: bool) -> bool:
    source = load_source_contract(args)
    rows = []
    for context in source["contexts"]:
        ok, reason = validate_stable(args, source, context)
        rows.append({"context_id": context["context_id"], "ok": ok, "reason": reason})
    passed = all(row["ok"] for row in rows)
    report = {
        "status": "pass" if passed else "incomplete",
        "panel_id": source["panel_id"],
        "action": args.action,
        "expected": 32,
        "valid": sum(row["ok"] for row in rows),
        "rows": rows,
    }
    args.stage.mkdir(parents=True, exist_ok=True)
    atomic_json(args.stage / f"validation_{args.action}.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))
    if require_complete and not passed:
        raise SystemExit(1)
    return passed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repo = Path(os.environ.get("AF_ROOT", Path(__file__).resolve().parents[2]))
    remote = Path(os.environ.get(
        "ARRWM_REMOTE_ROOT",
        "/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler",
    ))
    panel_root = Path(os.environ.get("PANEL32_STAGE", remote / "panel32_stage"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("dry-run", "prepare", "complete", "finalize", "validate")
    )
    parser.add_argument("--action", required=True, choices=ACTIONS)
    parser.add_argument("--panel-manifest", required=True, type=Path)
    parser.add_argument("--source-provenance", required=True, type=Path)
    parser.add_argument("--repo-root", type=Path, default=repo)
    parser.add_argument("--remote-root", type=Path, default=remote)
    parser.add_argument(
        "--yume-root", type=Path,
        default=Path(os.environ.get("YUME_ROOT", repo / "third_party/YUME")),
    )
    parser.add_argument(
        "--model-dir", type=Path,
        default=Path(os.environ.get("YUME_MODEL_DIR", remote / "yume/Yume-5B-720P")),
    )
    parser.add_argument(
        "--stage", type=Path,
        default=Path(os.environ.get("YUME_PANEL_STAGE", panel_root / "yume5b/work")),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path(os.environ.get("YUME_PANEL_OUT", panel_root / "yume5b/outputs")),
    )
    parser.add_argument(
        "--scene-prompt",
        default=os.environ.get("YUME_PANEL_SCENE_PROMPT", DEFAULT_PROMPT),
    )
    parser.add_argument(
        "--base-seed", type=int,
        default=int(os.environ.get("YUME_PANEL_BASE_SEED", str(BASE_SEED))),
    )
    args = parser.parse_args(argv)
    args.panel_manifest = args.panel_manifest.expanduser().resolve()
    args.source_provenance = args.source_provenance.expanduser().resolve()
    return args


def main() -> None:
    args = parse_args()
    if args.command == "dry-run":
        dry_run(args)
    elif args.command == "prepare":
        prepare(args)
    elif args.command == "complete":
        raise SystemExit(0 if validate_action(args, require_complete=False) else 1)
    elif args.command == "finalize":
        finalize(args)
    else:
        validate_action(args, require_complete=True)


if __name__ == "__main__":
    main()
