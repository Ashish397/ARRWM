"""Shared, provenance-locked input plans for external panel32 models.

This module does not run any model.  It translates the mixed-panel source
contract into explicit per-context records for LingBot, DreamX, Matrix-Game,
minWM, and YUME.  Existing model runners can consume the materialized
``frame32`` and ``streams`` directories without inferring identity from a
filename suffix.

The key native-context distinction is retained:

* LingBot, DreamX, Matrix-Game, and YUME condition on canonical frame 32.
* minWM DMD conditions on canonical frames 4--32 (29 pixels / 8 latents).

Every record carries the panel, original-source, canonical-stream, and
boundary-image hashes that must later be copied into the generated sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

try:
    from .panel32_manifest import ACTIONS, load_panel_manifest, sha256_file
except ImportError:  # direct execution
    from panel32_manifest import ACTIONS, load_panel_manifest, sha256_file  # type: ignore


NEUTRAL_PROMPT = "A first-person view of an outdoor environment."
MODELS = (
    "lingbot", "dreamx", "matrixgame2", "minwm", "minwm_ode", "yume5b"
)
MODEL_INPUT = {
    "lingbot": {
        "kind": "canonical_boundary_image", "source_frames_inclusive": [32, 32],
        "pixel_frames": 1, "latent_frames": None,
    },
    "dreamx": {
        "kind": "canonical_boundary_image", "source_frames_inclusive": [32, 32],
        "pixel_frames": 1, "latent_frames": None,
    },
    "matrixgame2": {
        "kind": "canonical_boundary_image", "source_frames_inclusive": [32, 32],
        "pixel_frames": 1, "latent_frames": None,
    },
    "minwm": {
        "kind": "canonical_video_span", "source_frames_inclusive": [4, 32],
        "pixel_frames": 29, "latent_frames": 8,
    },
    "minwm_ode": {
        "kind": "canonical_video_span", "source_frames_inclusive": [20, 32],
        "pixel_frames": 13, "latent_frames": 4,
    },
    "yume5b": {
        "kind": "canonical_boundary_image", "source_frames_inclusive": [32, 32],
        "pixel_frames": 1, "latent_frames": None,
    },
}
MODEL_PREFIX = {
    "lingbot": "lingbot",
    "dreamx": "dreamx",
    "matrixgame2": "matrixgame",
    "minwm": "minwm_aligned32",
    "minwm_ode": "minwm_ode",
    "yume5b": "yume5b",
}
NOOP_ON_DISK = {"matrixgame2", "minwm", "minwm_ode"}
SCHEMA_VERSION = 1


class PlanError(RuntimeError):
    """The panel source or requested model plan is inconsistent."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanError(message)


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PlanError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PlanError(f"JSON root must be an object: {path}")
    return value


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def context_token(context_id: str) -> str:
    return "p32-" + hashlib.sha256(context_id.encode("utf-8")).hexdigest()


def disk_action(model: str, action: str) -> str:
    return "NOOP" if action == "N" and model in NOOP_ON_DISK else action


def stable_filename(model: str, context_id: str, action: str) -> str:
    return f"{MODEL_PREFIX[model]}_{context_id}_{disk_action(model, action)}.mp4"


def sampling_seed(model: str, context_id: str, base_seed: int = 1) -> int:
    if model == "dreamx":
        offset = int.from_bytes(
            hashlib.sha256(context_id.encode("utf-8")).digest()[:4], "big"
        )
        return (base_seed + offset) % (2**31 - 1)
    if model == "yume5b":
        # YUME hashes the staged basename.  Its adapter uses the full SHA token
        # below, which is stable across action shards and resume order.
        token = "ctx-" + hashlib.sha256(context_id.encode("utf-8")).hexdigest()
        offset = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big")
        return (43 + offset) % (2**31 - 1)
    return {
        "lingbot": 1, "matrixgame2": 0, "minwm": 1234,
        "minwm_ode": 1234,
    }[model]


def load_panel_rows(
    manifest_path: Path, provenance_path: Path, *, verify_artifacts: bool = True
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    panel = load_panel_manifest(manifest_path, verify_sources=False)
    provenance = _json(provenance_path)
    _require(provenance.get("status") == "pass", "source provenance status is not pass")
    _require(provenance.get("complete_panel") is True,
             "source provenance is not a complete panel")
    _require(provenance.get("manifest_sha256") == panel.sha256,
             "panel manifest/provenance SHA mismatch")
    _require(provenance.get("panel_id") == panel.panel_id, "panel ID mismatch")
    raw_rows = provenance.get("rows")
    _require(isinstance(raw_rows, list) and len(raw_rows) == 32,
             "source provenance must contain 32 rows")
    by_id: dict[str, dict[str, Any]] = {}
    for row in raw_rows:
        _require(isinstance(row, dict), "source provenance row is not an object")
        context_id = row.get("context_id")
        _require(isinstance(context_id, str) and context_id not in by_id,
                 f"duplicate/invalid provenance context {context_id!r}")
        by_id[context_id] = row
    expected_ids = [context.context_id for context in panel.contexts]
    _require(set(by_id) == set(expected_ids), "manifest/provenance context sets differ")

    rows = []
    for context in panel.contexts:
        row = by_id[context.context_id]
        canonical = row.get("canonical", {})
        boundary = row.get("boundary", {})
        _require(row.get("dataset") == context.dataset, f"{context.context_id}: dataset mismatch")
        _require(row.get("source_id") == context.source_id,
                 f"{context.context_id}: source ID mismatch")
        _require(row.get("source", {}).get("sha256") == context.source_sha256,
                 f"{context.context_id}: original source SHA mismatch")
        _require(row.get("panel_manifest_sha256") == panel.sha256,
                 f"{context.context_id}: row manifest SHA mismatch")
        _require(boundary.get("canonical_frame_index") == 32,
                 f"{context.context_id}: canonical boundary is not frame 32")
        _require(
            (canonical.get("frame_count"), canonical.get("fps"),
             canonical.get("width"), canonical.get("height"))
            == (33, 20.0, 832, 480),
            f"{context.context_id}: canonical contract mismatch",
        )
        stream = Path(str(canonical.get("stream", ""))).resolve()
        image = Path(str(canonical.get("boundary_png", ""))).resolve()
        if verify_artifacts:
            _require(stream.is_file() and image.is_file(),
                     f"{context.context_id}: canonical artifact missing")
            _require(sha256_file(stream) == canonical.get("stream_sha256"),
                     f"{context.context_id}: canonical stream SHA mismatch")
            _require(sha256_file(image) == canonical.get("boundary_png_sha256"),
                     f"{context.context_id}: boundary image SHA mismatch")
        rows.append({
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
            "canonical_stream_sha256": canonical.get("stream_sha256"),
            "canonical_decoded_rgb24_sha256": canonical.get(
                "decoded_rgb24_sha256"
            ),
            "canonical_boundary_png": str(image),
            "canonical_boundary_png_sha256": canonical.get("boundary_png_sha256"),
            "canonical_boundary_rgb24_sha256": boundary.get(
                "normalized_rgb24_sha256"
            ),
        })
    return panel, provenance, rows


def runner_environment(
    model: str, action: str, contexts: list[str], frame32: Path, streams: Path,
    output: Path, prompt: str = NEUTRAL_PROMPT,
) -> dict[str, str]:
    joined = ",".join(contexts)
    selected_action = disk_action(model, action)
    common = {"EVAL_REAL_FRAME": "32"}
    if model == "lingbot":
        return {**common, "LB_WINDOWS": joined, "LB_SEED_DIR": str(frame32),
                "LB_DIRS": selected_action, "LB_CAP": prompt,
                "LB_OUT": str(output.resolve())}
    if model == "dreamx":
        return {**common, "DX_WINDOWS": joined, "DX_SEED_DIR": str(frame32),
                "DX_DIRS": selected_action, "DX_CAP": prompt,
                "DX_OUT": str(output.resolve())}
    if model == "matrixgame2":
        return {**common, "MG_WINDOWS": joined,
                "MG_DIRS": selected_action, "MG_OUT": str(output.resolve()),
                "MG_SEED_FMT": str(frame32 / "seed65_{wi}_f0.png")}
    if model in ("minwm", "minwm_ode"):
        is_ode = model == "minwm_ode"
        return {
            **common, "MW_WINDOWS": joined,
            "MW_DIRS": selected_action, "MW_OUT": str(output.resolve()),
            "MW_SEED_FMT": str(streams / "seed65_{wi}.avi"),
            "MW_SEED_START": "20" if is_ode else "4",
            "MW_SEED_LAT": "4" if is_ode else "8",
            "MW_NUMLAT": "124" if is_ode else "128",
            "MW_TAG": "_ode" if is_ode else "_aligned32",
            "MW_STAGE": "ode" if is_ode else "dmd",
            "MW_CPU_T5": "1", "MW_CHUNK_DECODE": "8",
            "MW_CAP": prompt,
        }
    return {
        **common, "YUME_PANEL_SCENE_PROMPT": prompt,
        "YUME_ACTION_PREFIX": "",
        "YUME_PANEL_ACTION": action,
        "YUME_PANEL_OUT": str(output.resolve()),
    }


def sidecar_provenance(
    plan: Mapping[str, Any], item: Mapping[str, Any]
) -> dict[str, Any]:
    """Return the common provenance block all model adapters must retain."""
    return {
        "schema_version": SCHEMA_VERSION,
        "panel_id": plan["panel_id"],
        "panel_manifest": plan["panel_manifest"],
        "panel_manifest_sha256": plan["panel_manifest_sha256"],
        "panel_source_provenance": plan["panel_source_provenance"],
        "panel_source_provenance_sha256": plan[
            "panel_source_provenance_sha256"
        ],
        "context_id": item["context_id"],
        "source_dataset": item["dataset"],
        "source_id": item["source_id"],
        "source_video_sha256": item["source_video_sha256"],
        "source_boundary_frame_index": item["source_boundary_frame_index"],
        "source_boundary_timestamp_s": item["source_boundary_timestamp_s"],
        "source_boundary_original_rgb24_sha256": item[
            "source_boundary_original_rgb24_sha256"
        ],
        "canonical_stream_sha256": item["canonical_stream_sha256"],
        "canonical_decoded_rgb24_sha256": item[
            "canonical_decoded_rgb24_sha256"
        ],
        "canonical_boundary_png_sha256": item[
            "canonical_boundary_png_sha256"
        ],
        "canonical_boundary_rgb24_sha256": item[
            "canonical_boundary_rgb24_sha256"
        ],
        "generation_boundary_canonical_frame": 32,
        "model_input_source_frames_inclusive": item[
            "source_frames_inclusive"
        ],
        "model_input_pixel_frames": item["input_pixel_frames"],
        "model_input_latent_frames": item["input_latent_frames"],
        "action": item["action"],
        "disk_action": item["disk_action"],
        "sampling_seed": item["sampling_seed"],
        "seed_match_group": item["seed_match_group"],
        "neutral_prompt": plan["neutral_prompt"],
        "neutral_prompt_sha256": plan["neutral_prompt_sha256"],
    }


def attach_sidecar_provenance(
    plan: Mapping[str, Any], item: Mapping[str, Any], sidecar: Path
) -> None:
    """Attach provenance without overwriting an inconsistent prior block."""
    payload = _json(sidecar)
    block = sidecar_provenance(plan, item)
    if "panel32" in payload:
        _require(payload["panel32"] == block,
                 f"conflicting existing panel32 provenance in {sidecar}")
        return
    payload["panel32"] = block
    _atomic_json(sidecar, payload)


def build_plan(
    manifest_path: Path, provenance_path: Path, model: str, action: str,
    stage: Path, output: Path, *, prompt: str = NEUTRAL_PROMPT,
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    _require(model in MODELS, f"unsupported model {model!r}")
    _require(action in ACTIONS, f"unsupported action {action!r}")
    _require(prompt == NEUTRAL_PROMPT,
             f"panel prompt must be exactly {NEUTRAL_PROMPT!r}")
    panel, _, rows = load_panel_rows(
        manifest_path, provenance_path, verify_artifacts=verify_artifacts
    )
    frame32 = stage.resolve() / "frame32"
    streams = stage.resolve() / "streams"
    input_contract = MODEL_INPUT[model]
    items = []
    for row in rows:
        context_id = row["context_id"]
        input_path = (
            frame32 / f"seed65_{context_id}_f0.png"
            if input_contract["kind"] == "canonical_boundary_image"
            else streams / f"seed65_{context_id}.avi"
        )
        items.append({
            **row,
            "task_token": context_token(context_id),
            "model": model,
            "action": action,
            "disk_action": disk_action(model, action),
            "input_kind": input_contract["kind"],
            "input_path": str(input_path),
            "source_frames_inclusive": input_contract["source_frames_inclusive"],
            "input_pixel_frames": input_contract["pixel_frames"],
            "input_latent_frames": input_contract["latent_frames"],
            "sampling_seed": sampling_seed(model, context_id),
            "seed_match_group": context_id,
            "stable_output": str(output.resolve() / stable_filename(
                model, context_id, action
            )),
        })
    _require(len(items) == 32, "internal error: plan does not contain 32 contexts")
    return {
        "schema_version": SCHEMA_VERSION,
        "panel_id": panel.panel_id,
        "panel_manifest": str(panel.path),
        "panel_manifest_sha256": panel.sha256,
        "panel_source_provenance": str(provenance_path.resolve()),
        "panel_source_provenance_sha256": sha256_file(provenance_path),
        "model": model,
        "action": action,
        "disk_action": disk_action(model, action),
        "neutral_prompt": prompt,
        "neutral_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "input_contract": input_contract,
        "generation_boundary_canonical_frame": 32,
        "generated_seconds": 30,
        "items": items,
        "runner_environment": runner_environment(
            model, action, [row["context_id"] for row in rows], frame32,
            streams, output, prompt
        ),
    }


def materialize(plan: Mapping[str, Any], stage: Path) -> None:
    frame32 = stage.resolve() / "frame32"
    streams = stage.resolve() / "streams"
    frame32.mkdir(parents=True, exist_ok=True)
    streams.mkdir(parents=True, exist_ok=True)
    for item in plan["items"]:
        links = (
            (frame32 / f"seed65_{item['context_id']}_f0.png",
             Path(item["canonical_boundary_png"]).resolve()),
            (streams / f"seed65_{item['context_id']}.avi",
             Path(item["canonical_stream"]).resolve()),
        )
        for link, target in links:
            if link.is_symlink():
                _require(link.resolve() == target, f"conflicting link {link}")
            elif link.exists():
                raise PlanError(f"refusing to replace non-symlink input {link}")
            else:
                link.symlink_to(target)
    _atomic_json(
        stage.resolve() / f"plan_{plan['model']}_{plan['action']}.json", plan
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-provenance", required=True, type=Path)
    parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--action", required=True, choices=ACTIONS)
    parser.add_argument("--stage", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--prompt", default=NEUTRAL_PROMPT)
    parser.add_argument("--materialize", action="store_true")
    args = parser.parse_args()
    plan = build_plan(
        args.manifest, args.source_provenance, args.model, args.action,
        args.stage, args.output, prompt=args.prompt,
    )
    if args.materialize:
        materialize(plan, args.stage)
    print(json.dumps(plan, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
