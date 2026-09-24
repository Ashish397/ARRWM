"""Reuse byte-identical v1 rollouts in the v2 panel with fresh provenance.

Only contexts whose complete source/canonical identity is unchanged are
eligible.  Videos are hard-linked when possible; sidecars are copied and
rebased onto the v2 source contract.  DMD/ODE rows additionally require the
new seed-bundle index so their bundle paths and hashes cannot be stale.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

try:
    from .panel32_manifest import ACTIONS, load_panel_manifest, sha256_file
    from .panel32_model_inputs import attach_sidecar_provenance
except ImportError:
    from panel32_manifest import ACTIONS, load_panel_manifest, sha256_file  # type: ignore
    from panel32_model_inputs import attach_sidecar_provenance  # type: ignore


EXTERNAL = ("lingbot", "dreamx", "matrixgame2", "minwm", "minwm_ode")
DMD = (
    "ours_recoverybase", "ours_nocarn", "ours_nocommit", "ours_noaux",
    "ours_nogan", "ours_meanenergy", "ours_vartv",
)
ODE = ("ours_kl4rung", "ours_mse4rung")
ALL_ROWS = EXTERNAL + ("yume5b",) + DMD + ODE
ODE_IDENTITY_KEYS = (
    "schema_version", "model_id", "objective", "context_id", "dataset",
    "action", "panel_manifest_sha256", "source_provenance_sha256",
    "bundle_index_sha256", "bundle_sha256", "prompt_embedding_sha256",
    "checkpoint_sha256", "teacher_checkpoint_sha256", "config_sha256",
    "sampling_seed", "denoising_rungs", "runner_sha256", "prompt_sha256",
    "action_vector", "cache_fill", "cache_chunks", "generated_chunks",
    "output_fps",
)


def require(value: bool, message: str) -> None:
    if not value:
        raise RuntimeError(message)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def fingerprint(value: Mapping[str, Any]) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(data).hexdigest()


def hardlink_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    require(not destination.exists(), f"destination already exists: {destination}")
    # DreamX's repaired v1 fleet uses relative stable symlinks.  Reusing the
    # symlink inode in another directory would change its target, so always
    # resolve to the authoritative media payload first.
    source = source.resolve(strict=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def row_map(provenance: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = provenance.get("rows")
    require(isinstance(rows, list) and len(rows) == 32, "bad source provenance")
    return {str(row["context_id"]): row for row in rows}


def unchanged_contexts(
    old: Mapping[str, Any], new: Mapping[str, Any], expected_count: int,
) -> set[str]:
    old_rows, new_rows = row_map(old), row_map(new)
    retained: set[str] = set()
    for context, current in new_rows.items():
        previous = old_rows.get(context)
        if previous is None:
            continue
        checks = (
            previous.get("dataset") == current.get("dataset"),
            previous.get("source_id") == current.get("source_id"),
            previous.get("source", {}).get("sha256")
            == current.get("source", {}).get("sha256"),
            previous.get("canonical", {}).get("stream_sha256")
            == current.get("canonical", {}).get("stream_sha256"),
            previous.get("canonical", {}).get("decoded_rgb24_sha256")
            == current.get("canonical", {}).get("decoded_rgb24_sha256"),
            previous.get("canonical", {}).get("boundary_png_sha256")
            == current.get("canonical", {}).get("boundary_png_sha256"),
            previous.get("boundary", {}).get("normalized_rgb24_sha256")
            == current.get("boundary", {}).get("normalized_rgb24_sha256"),
        )
        if all(checks):
            retained.add(context)
    require(
        len(retained) == expected_count,
        f"expected exactly {expected_count} unchanged contexts, got {len(retained)}",
    )
    return retained


def patch_common_source(
    meta: dict[str, Any], context: Mapping[str, Any], manifest: Path,
    manifest_hash: str, provenance: Path, provenance_hash: str,
    panel_id: str,
) -> None:
    source, canonical, boundary = (
        context["source"], context["canonical"], context["boundary"]
    )
    replacements = {
        "panel_id": panel_id,
        "panel_manifest": str(manifest.resolve()),
        "panel_manifest_sha256": manifest_hash,
        "panel_source_provenance": str(provenance.resolve()),
        "panel_source_provenance_sha256": provenance_hash,
        "source_provenance": str(provenance.resolve()),
        "source_provenance_sha256": provenance_hash,
        "source_dataset": context["dataset"],
        "source_id": context["source_id"],
        "source_video_sha256": source["sha256"],
        "source_boundary_frame_index": boundary["source_frame_index"],
        "source_boundary_timestamp_s": boundary["source_timestamp_s"],
        "source_boundary_original_rgb24_sha256": boundary[
            "original_rgb24_sha256"
        ],
        "source_boundary_normalized_rgb24_sha256": boundary[
            "normalized_rgb24_sha256"
        ],
        "canonical_stream": canonical["stream"],
        "canonical_stream_sha256": canonical["stream_sha256"],
        "canonical_decoded_rgb24_sha256": canonical[
            "decoded_rgb24_sha256"
        ],
        "canonical_boundary_png": canonical["boundary_png"],
        "canonical_boundary_png_sha256": canonical["boundary_png_sha256"],
        "canonical_boundary_rgb24_sha256": boundary[
            "normalized_rgb24_sha256"
        ],
        "source_stream": canonical["stream"],
        "source_stream_sha256": canonical["stream_sha256"],
        "source_decoded_rgb24_sha256": canonical[
            "decoded_rgb24_sha256"
        ],
    }
    for key, value in replacements.items():
        if key in meta:
            meta[key] = value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-root", type=Path, required=True)
    parser.add_argument("--new-root", type=Path, required=True)
    parser.add_argument("--old-provenance", type=Path, required=True)
    parser.add_argument("--new-provenance", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--phase", choices=("external", "all", "yume"), default="all")
    parser.add_argument("--bundle-index", type=Path)
    parser.add_argument("--old-bundle-index", type=Path)
    parser.add_argument("--expected-unchanged-contexts", type=int, default=14)
    parser.add_argument("--yume-retain-count", type=int, default=12)
    args = parser.parse_args()

    old_provenance = read_json(args.old_provenance)
    new_provenance = read_json(args.new_provenance)
    manifest = load_panel_manifest(args.manifest, verify_sources=False)
    require(new_provenance["manifest_sha256"] == manifest.sha256, "manifest mismatch")
    retained = unchanged_contexts(
        old_provenance, new_provenance, args.expected_unchanged_contexts
    )
    new_rows = row_map(new_provenance)
    provenance_hash = sha256_file(args.new_provenance)

    bundle_rows: dict[str, Any] = {}
    bundle_index_hash = ""
    if args.phase == "all":
        require(args.bundle_index is not None and args.bundle_index.is_file(),
                "all phase requires a bundle index")
        bundle = read_json(args.bundle_index)
        rows = bundle.get("rows") or bundle.get("contexts")
        require(isinstance(rows, list) and len(rows) == 32, "bad bundle index")
        bundle_rows = {str(row["context_id"]): row for row in rows}
        bundle_index_hash = sha256_file(args.bundle_index)
        require(
            args.old_bundle_index is not None and args.old_bundle_index.is_file(),
            "all phase requires the old bundle index for tensor-equivalence checks",
        )
        old_bundle = read_json(args.old_bundle_index)
        old_rows = old_bundle.get("rows") or old_bundle.get("contexts")
        require(isinstance(old_rows, list) and len(old_rows) == 32,
                "bad old bundle index")
        old_bundle_rows = {str(row["context_id"]): row for row in old_rows}
        import torch
        for context_id in retained:
            old_record = torch.load(
                old_bundle_rows[context_id]["bundle"], map_location="cpu",
                weights_only=False,
            )
            new_record = torch.load(
                bundle_rows[context_id]["bundle"], map_location="cpu",
                weights_only=False,
            )
            require(
                torch.equal(old_record["seed"], new_record["seed"]),
                f"seed tensor changed for reusable context {context_id}",
            )
            require(
                torch.equal(
                    old_record["seed_actions"], new_record["seed_actions"]
                ),
                f"seed actions changed for reusable context {context_id}",
            )

    # The external phase can run before GPU-built ARRWM bundles exist.  The
    # all phase deliberately means all ARRWM rows; it does not revisit files
    # already installed by the external phase.
    if args.phase == "all":
        rows = DMD + ODE
    elif args.phase == "yume":
        rows = ("yume5b",)
    else:
        rows = EXTERNAL + ("yume5b",)
    copied = 0
    expected = 0
    for row_name in rows:
        old_dir = args.old_root / "fleet30s" / row_name
        new_dir = args.new_root / "fleet30s" / row_name
        require(old_dir.is_dir(), f"missing old row: {old_dir}")
        plans: dict[str, dict[str, Any]] = {}
        if row_name in EXTERNAL:
            for action in ACTIONS:
                plan_path = args.new_root / "model_inputs" / f"plan_{row_name}_{action}.json"
                require(plan_path.is_file(), f"missing v2 input plan: {plan_path}")
                plans[action] = read_json(plan_path)

        # YUME launches four equal distributed ranks and therefore requires
        # the pending count to be divisible by four.  Reuse twelve, not all
        # fourteen, unchanged contexts so the final 20-context v2 workload is
        # balanced (five per rank).  Other runners permit uneven skip counts.
        row_retained = retained
        if row_name == "yume5b":
            require(
                0 <= args.yume_retain_count <= len(retained),
                "invalid YUME reuse count",
            )
            row_retained = set(sorted(retained)[-args.yume_retain_count:])
            require(
                len(row_retained) == args.yume_retain_count,
                "YUME reuse set has wrong context count",
            )
        expected += len(row_retained) * len(ACTIONS)

        for old_sidecar in sorted(old_dir.glob("*.mp4.json")):
            meta = read_json(old_sidecar)
            context_id = str(meta.get("context_id") or meta.get("window") or meta.get("panel32", {}).get("context_id"))
            if context_id not in row_retained:
                continue
            action = str(meta.get("action") or meta.get("direction") or meta.get("panel32", {}).get("action"))
            if action == "NOOP":
                action = "N"
            require(action in ACTIONS, f"cannot resolve action from {old_sidecar}")
            old_video = Path(str(old_sidecar)[:-5])
            require(old_video.is_file(), f"missing old video: {old_video}")
            new_video = new_dir / old_video.name
            new_sidecar = Path(str(new_video) + ".json")
            if new_video.is_symlink() and not new_video.exists():
                # Recover only the broken link this tool may have created in
                # the isolated v2 stage; no valid media is removed.
                new_video.unlink()
            if new_video.exists():
                require(new_video.is_file(), f"reuse target is not a file: {new_video}")
                require(
                    sha256_file(new_video) == sha256_file(old_video),
                    f"existing reuse video differs from v1: {new_video}",
                )
            else:
                hardlink_or_copy(old_video, new_video)
            if not new_sidecar.exists():
                shutil.copy2(old_sidecar, new_sidecar)
            require(new_sidecar.is_file(), f"reuse sidecar is not a file: {new_sidecar}")
            meta = read_json(new_sidecar)
            context = new_rows[context_id]

            if row_name in EXTERNAL:
                plan = plans[action]
                item = next(x for x in plan["items"] if x["context_id"] == context_id)
                # The shared helper correctly refuses to overwrite a
                # conflicting provenance block.  This script has already
                # proved exact source identity, so remove only that block and
                # let the helper install the complete v2 contract.
                meta.pop("panel32", None)
                atomic_json(new_sidecar, meta)
                attach_sidecar_provenance(plan, item, new_sidecar)
            else:
                patch_common_source(
                    meta, context, args.manifest, manifest.sha256,
                    args.new_provenance, provenance_hash, manifest.panel_id,
                )
                if row_name in DMD:
                    bundle = bundle_rows[context_id]
                    meta["bundle"] = str(Path(bundle["bundle"]).resolve())
                    meta["bundle_sha256"] = bundle["bundle_sha256"]
                elif row_name in ODE:
                    bundle = bundle_rows[context_id]
                    current_runner_hash = sha256_file(
                        Path(__file__).resolve().with_name("ode_panel32_runner.py")
                    )
                    require(
                        meta.get("runner_sha256") == current_runner_hash,
                        f"ODE runner changed; cannot reuse {old_video}",
                    )
                    meta["bundle"] = str(Path(bundle["bundle"]).resolve())
                    meta["bundle_index_sha256"] = bundle_index_hash
                    meta["bundle_sha256"] = bundle["bundle_sha256"]
                    identity = {key: meta[key] for key in ODE_IDENTITY_KEYS}
                    meta["contract_fingerprint"] = fingerprint(identity)
                atomic_json(new_sidecar, meta)
            copied += 1

    require(copied == expected, f"reused {copied}, expected {expected}")
    print(json.dumps({
        "status": "pass", "phase": args.phase, "rows": len(rows),
        "unchanged_contexts": sorted(retained), "reused_outputs": copied,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
