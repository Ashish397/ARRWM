"""Build deterministic CPU-only canonical sources for the mixed panel32.

For every selected source video this tool:

* resolves a locked boundary from either a zero-based decoded frame index or
  a timestamp (the latest source frame at or before that timestamp),
* samples 33 causal source frames ending on that exact boundary,
* applies one panel-wide spatial transform,
* writes a lossless FFV1/AVI stream at the declared canonical frame rate, and
* records source, boundary, sampling, transform, and output hashes.

The output is deliberately prior to model generation and prior to Wan VAE
encoding.  Thirty-three pixel frames are the exact temporal input for a later
nine-latent Wan encode.  Existing Frodo8 paths are not read or modified.
"""

from __future__ import annotations

import argparse
from bisect import bisect_right
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Iterable

try:
    from .panel32_manifest import (
        ContextSpec,
        PanelManifest,
        load_panel_manifest,
        sha256_file,
        validation_summary,
    )
except ImportError:  # Direct execution: python grids/eval/panel32_source.py
    from panel32_manifest import (  # type: ignore
        ContextSpec,
        PanelManifest,
        load_panel_manifest,
        sha256_file,
        validation_summary,
    )


PROVENANCE_SCHEMA_VERSION = 1


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _run_json(command: list[str]) -> dict[str, Any]:
    try:
        output = subprocess.check_output(command, text=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"command failed: {command}: {exc}") from exc
    return json.loads(output)


def _tool_version(tool: str) -> str:
    output = subprocess.check_output([tool, "-version"], text=True)
    return output.splitlines()[0]


def probe_source(path: Path, ffprobe: str = "ffprobe") -> dict[str, Any]:
    payload = _run_json([
        ffprobe, "-v", "error", "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,avg_frame_rate,r_frame_rate,duration,nb_frames:"
        "stream_tags=rotate:stream_side_data=rotation:"
        "frame=best_effort_timestamp_time",
        "-of", "json", str(path),
    ])
    streams = payload.get("streams", [])
    if len(streams) != 1:
        raise RuntimeError(f"expected one video stream in {path}, found {len(streams)}")
    stream = streams[0]
    timestamps = []
    for index, frame in enumerate(payload.get("frames", [])):
        value = frame.get("best_effort_timestamp_time")
        if value is None:
            raise RuntimeError(f"{path}: decoded frame {index} has no timestamp")
        timestamps.append(float(value))
    if not timestamps:
        raise RuntimeError(f"{path}: no decoded video-frame timestamps")
    if any(b < a for a, b in zip(timestamps, timestamps[1:])):
        raise RuntimeError(f"{path}: non-monotonic decoded timestamps")
    rotation = 0
    for side in stream.get("side_data_list", []):
        if side.get("rotation") is not None:
            rotation = int(side["rotation"])
            break
    if not rotation and stream.get("tags", {}).get("rotate") is not None:
        rotation = int(stream["tags"]["rotate"])
    coded_width, coded_height = int(stream["width"]), int(stream["height"])
    if abs(rotation) % 180 == 90:
        display_width, display_height = coded_height, coded_width
    else:
        display_width, display_height = coded_width, coded_height
    return {
        "coded_width": coded_width,
        "coded_height": coded_height,
        "display_width": display_width,
        "display_height": display_height,
        "rotation_degrees": rotation,
        "avg_frame_rate": stream.get("avg_frame_rate"),
        "r_frame_rate": stream.get("r_frame_rate"),
        "duration_s": float(stream["duration"]) if stream.get("duration") else None,
        "declared_frames": int(stream["nb_frames"]) if stream.get("nb_frames") else None,
        "decoded_frames": len(timestamps),
        "timestamps_s": timestamps,
    }


def resolve_boundary(context: ContextSpec, timestamps: list[float]) -> tuple[int, float]:
    if context.boundary_kind == "frame_index":
        index = int(context.boundary_value)
        if index >= len(timestamps):
            raise RuntimeError(
                f"{context.context_id}: boundary frame {index} outside {len(timestamps)} frames")
        return index, timestamps[index]
    requested = float(context.boundary_value)
    if len(timestamps) > 1:
        positive_steps = [b - a for a, b in zip(timestamps, timestamps[1:]) if b > a]
        final_tolerance = max(positive_steps) if positive_steps else 1e-6
    else:
        final_tolerance = 1e-6
    if requested > timestamps[-1] + final_tolerance + 1e-9:
        raise RuntimeError(
            f"{context.context_id}: boundary {requested:.9f}s is beyond source end "
            f"{timestamps[-1]:.9f}s")
    index = bisect_right(timestamps, requested + 1e-9) - 1
    if index < 0:
        raise RuntimeError(
            f"{context.context_id}: no source frame at/before boundary {requested:.9f}s")
    return index, timestamps[index]


def sampling_plan(
    timestamps: list[float], boundary_index: int, boundary_timestamp_s: float,
    frame_count: int, fps: float,
) -> tuple[list[float], list[int]]:
    target_times = [
        boundary_timestamp_s - (frame_count - 1 - index) / fps
        for index in range(frame_count)
    ]
    if target_times[0] < timestamps[0] - 1e-9:
        raise RuntimeError(
            f"source has insufficient causal history: need {target_times[0]:.9f}s, "
            f"first decoded timestamp is {timestamps[0]:.9f}s")
    source_indices = []
    for target in target_times:
        source_index = bisect_right(timestamps, target + 1e-9) - 1
        if source_index < 0:
            raise RuntimeError(f"no source frame at/before target {target:.9f}s")
        source_indices.append(source_index)
    source_indices[-1] = boundary_index
    if any(b < a for a, b in zip(source_indices, source_indices[1:])):
        raise RuntimeError("internal error: non-monotonic source sampling plan")
    return target_times, source_indices


def _select_expression(indices: Iterable[int]) -> str:
    return "+".join(f"eq(n\\,{index})" for index in indices)


def _spatial_filter(width: int, height: int, mode: str) -> str:
    if mode == "stretch":
        return f"scale={width}:{height}:flags=lanczos"
    aspect = width / height
    return (
        f"crop=w='min(iw,ih*{aspect:.12f})':"
        f"h='min(ih,iw/{aspect:.12f})':"
        "x='(iw-out_w)/2':y='(ih-out_h)/2',"
        f"scale={width}:{height}:flags=lanczos"
    )


def decode_unique_frames(
    source: Path, indices: list[int], width: int, height: int,
    *, ffmpeg: str = "ffmpeg", normalize: bool,
    resize_mode: str = "center_crop",
) -> dict[int, bytes]:
    unique = sorted(set(indices))
    filters = [f"select='{_select_expression(unique)}'"]
    output_width, output_height = width, height
    if normalize:
        filters.append(_spatial_filter(width, height, resize_mode))
    command = [
        ffmpeg, "-v", "error", "-threads", "1", "-i", str(source),
        "-map", "0:v:0", "-vf", ",".join(filters), "-vsync", "0",
        "-frames:v", str(len(unique)), "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ]
    payload = subprocess.check_output(command)
    frame_bytes = output_width * output_height * 3
    expected = len(unique) * frame_bytes
    if len(payload) != expected:
        raise RuntimeError(
            f"decoded {len(payload)} bytes from {source}; expected {expected} "
            f"for {len(unique)} frames at {output_width}x{output_height}")
    return {
        index: payload[offset * frame_bytes:(offset + 1) * frame_bytes]
        for offset, index in enumerate(unique)
    }


def decode_original_boundary(
    source: Path, index: int, display_width: int, display_height: int,
    *, ffmpeg: str = "ffmpeg",
) -> bytes:
    return decode_unique_frames(
        source, [index], display_width, display_height,
        ffmpeg=ffmpeg, normalize=False,
    )[index]


def _encode_ffv1_avi(
    frames: bytes, output: Path, width: int, height: int, fps: float,
    frame_count: int, ffmpeg: str,
) -> None:
    temporary = output.with_name(f".{output.stem}.tmp.{os.getpid()}.avi")
    command = [
        ffmpeg, "-y", "-v", "error", "-threads", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-video_size", f"{width}x{height}", "-framerate", f"{fps:.12g}",
        "-i", "-", "-frames:v", str(frame_count), "-an",
        "-c:v", "ffv1", "-level", "3", "-pix_fmt", "gbrp", str(temporary),
    ]
    subprocess.run(command, input=frames, check=True)
    temporary.replace(output)


def _encode_png(frame: bytes, output: Path, width: int, height: int, ffmpeg: str) -> None:
    temporary = output.with_name(f".{output.stem}.tmp.{os.getpid()}.png")
    command = [
        ffmpeg, "-y", "-v", "error", "-threads", "1",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-video_size", f"{width}x{height}",
        "-i", "-", "-frames:v", "1", "-compression_level", "9", str(temporary),
    ]
    subprocess.run(command, input=frame, check=True)
    temporary.replace(output)


def decode_canonical(path: Path, width: int, height: int, ffmpeg: str = "ffmpeg") -> bytes:
    payload = subprocess.check_output([
        ffmpeg, "-v", "error", "-threads", "1", "-i", str(path),
        "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ])
    frame_bytes = width * height * 3
    if len(payload) % frame_bytes:
        raise RuntimeError(f"canonical decode has partial RGB frame: {path}")
    return payload


def _atomic_json(path: Path, payload: MappingLike) -> None:
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


MappingLike = dict[str, Any]


def artifact_paths(output_root: Path, context_id: str) -> tuple[Path, Path, Path]:
    stream = output_root / "streams" / f"source33_{context_id}.avi"
    boundary = output_root / "boundaries" / f"source33_{context_id}_f32.png"
    provenance = output_root / "provenance" / f"source33_{context_id}.json"
    return stream, boundary, provenance


def _retained_provenance(
    panel: PanelManifest, context: ContextSpec, output_root: Path,
) -> dict[str, Any] | None:
    stream, boundary, provenance = artifact_paths(output_root, context.context_id)
    present = [path.exists() for path in (stream, boundary, provenance)]
    if not any(present):
        return None
    if not all(present):
        raise FileExistsError(
            f"partial existing artifact set for {context.context_id}; use a clean output root")
    try:
        row = json.loads(provenance.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FileExistsError(
            f"invalid existing provenance for {context.context_id}: {exc}") from exc
    expected = (
        row.get("panel_manifest_sha256") == panel.sha256
        and row.get("context_id") == context.context_id
        and row.get("source", {}).get("sha256") == context.source_sha256
        and row.get("canonical", {}).get("stream_sha256") == sha256_file(stream)
        and row.get("canonical", {}).get("boundary_png_sha256") == sha256_file(boundary)
    )
    if not expected:
        raise FileExistsError(
            f"stale/conflicting existing artifacts for {context.context_id}; use a clean output root")
    return row


def extract_context(
    panel: PanelManifest, context: ContextSpec, output_root: Path,
    *, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe",
) -> dict[str, Any]:
    retained = _retained_provenance(panel, context, output_root)
    if retained is not None:
        return retained
    observed_source_hash = sha256_file(context.source_video)
    if observed_source_hash != context.source_sha256:
        raise RuntimeError(
            f"{context.context_id}: source SHA mismatch: "
            f"{observed_source_hash} != {context.source_sha256}")
    probe = probe_source(context.source_video, ffprobe=ffprobe)
    timestamps = probe.pop("timestamps_s")
    boundary_index, boundary_timestamp = resolve_boundary(context, timestamps)
    canonical = panel.canonical
    target_times, source_indices = sampling_plan(
        timestamps, boundary_index, boundary_timestamp,
        canonical.frame_count, canonical.fps,
    )
    normalized = decode_unique_frames(
        context.source_video, source_indices, canonical.width, canonical.height,
        ffmpeg=ffmpeg, normalize=True, resize_mode=canonical.resize_mode,
    )
    canonical_frames = b"".join(normalized[index] for index in source_indices)
    frame_bytes = canonical.width * canonical.height * 3
    if len(canonical_frames) != canonical.frame_count * frame_bytes:
        raise RuntimeError(f"{context.context_id}: internal canonical frame assembly failure")
    normalized_boundary = normalized[boundary_index]
    original_boundary = decode_original_boundary(
        context.source_video, boundary_index,
        probe["display_width"], probe["display_height"], ffmpeg=ffmpeg,
    )

    stream, boundary_png, provenance = artifact_paths(output_root, context.context_id)
    for directory in (stream.parent, boundary_png.parent, provenance.parent):
        directory.mkdir(parents=True, exist_ok=True)
    _encode_ffv1_avi(
        canonical_frames, stream, canonical.width, canonical.height,
        canonical.fps, canonical.frame_count, ffmpeg,
    )
    _encode_png(normalized_boundary, boundary_png, canonical.width, canonical.height, ffmpeg)
    decoded = decode_canonical(stream, canonical.width, canonical.height, ffmpeg=ffmpeg)
    if decoded != canonical_frames:
        raise RuntimeError(f"{context.context_id}: FFV1 canonical stream failed lossless round trip")
    decoded_frames = len(decoded) // frame_bytes
    if decoded_frames != canonical.frame_count:
        raise RuntimeError(
            f"{context.context_id}: canonical stream has {decoded_frames} frames")
    decoded_boundary = decoded[canonical.boundary_frame_index * frame_bytes:
                               (canonical.boundary_frame_index + 1) * frame_bytes]
    if decoded_boundary != normalized_boundary:
        raise RuntimeError(f"{context.context_id}: canonical frame 32 is not source boundary")

    source_minus_target = [
        timestamps[index] - target for index, target in zip(source_indices, target_times)
    ]
    row: dict[str, Any] = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "panel_id": panel.panel_id,
        "panel_manifest": str(panel.path),
        "panel_manifest_sha256": panel.sha256,
        "context_id": context.context_id,
        "dataset": context.dataset,
        "source_id": context.source_id,
        "source": {
            "declared_video": context.declared_source_video,
            "resolved_video": str(context.source_video),
            "sha256": observed_source_hash,
            "uri": context.raw.get("source_uri"),
            "license": context.raw.get("license"),
            "selection_note": context.raw.get("selection_note"),
            "outdoor_verified": context.raw["outdoor_verified"],
            "outdoor_verification_note": context.raw["outdoor_verification_note"],
            "probe": probe,
        },
        "boundary": {
            "requested_kind": context.boundary_kind,
            "requested_value": context.boundary_value,
            "source_frame_index": boundary_index,
            "source_timestamp_s": boundary_timestamp,
            "original_rgb24_sha256": sha256_bytes(original_boundary),
            "normalized_rgb24_sha256": sha256_bytes(normalized_boundary),
            "canonical_frame_index": canonical.boundary_frame_index,
        },
        "sampling": {
            "policy": "latest decoded source frame at or before each causal target time",
            "target_timestamps_s": target_times,
            "source_frame_indices": source_indices,
            "source_frame_timestamps_s": [timestamps[index] for index in source_indices],
            "source_minus_target_s": source_minus_target,
            "maximum_source_minus_target_s": max(source_minus_target),
            "minimum_source_minus_target_s": min(source_minus_target),
        },
        "canonical": {
            "stream": str(stream.resolve()),
            "stream_sha256": sha256_file(stream),
            "decoded_rgb24_sha256": sha256_bytes(decoded),
            "boundary_png": str(boundary_png.resolve()),
            "boundary_png_sha256": sha256_file(boundary_png),
            "frame_count": canonical.frame_count,
            "fps": canonical.fps,
            "width": canonical.width,
            "height": canonical.height,
            "resize_mode": canonical.resize_mode,
            "codec": "FFV1 level 3 in AVI, gbrp, lossless RGB round-trip verified",
        },
        "wan_interface": {
            "pixel_frames": canonical.frame_count,
            "latent_frames": canonical.wan_latent_frames,
            "temporal_packing": "pixel_frames = 1 + 4 * (latent_frames - 1)",
            "status": "source prepared; Wan encoding intentionally not performed",
        },
        "tools": {
            "ffmpeg": _tool_version(ffmpeg),
            "ffprobe": _tool_version(ffprobe),
        },
    }
    _atomic_json(provenance, row)
    return row


def extract_panel(
    manifest_path: str | Path, output_root: str | Path,
    *, context_ids: Iterable[str] | None = None,
    ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe",
) -> dict[str, Any]:
    panel = load_panel_manifest(manifest_path, verify_sources=True)
    selected = list(context_ids) if context_ids is not None else [c.context_id for c in panel.contexts]
    if len(set(selected)) != len(selected):
        raise ValueError("selected context IDs contain duplicates")
    by_id = {row.context_id: row for row in panel.contexts}
    unknown = sorted(set(selected) - set(by_id))
    if unknown:
        raise ValueError(f"unknown context IDs: {unknown}")
    output = Path(output_root).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = [
        extract_context(panel, by_id[context_id], output, ffmpeg=ffmpeg, ffprobe=ffprobe)
        for context_id in selected
    ]
    summary: dict[str, Any] = {
        **validation_summary(panel),
        "provenance_schema_version": PROVENANCE_SCHEMA_VERSION,
        "output_root": str(output),
        "selected_contexts": selected,
        "extracted_contexts": len(rows),
        "complete_panel": len(rows) == len(panel.contexts),
        "rows": rows,
    }
    _atomic_json(output / "panel32_source_provenance.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate manifest and source hashes")
    validate.add_argument("--manifest", required=True, type=Path)
    validate.add_argument("--skip-source-hashes", action="store_true")
    extract = subparsers.add_parser("extract", help="extract canonical CPU source streams")
    extract.add_argument("--manifest", required=True, type=Path)
    extract.add_argument("--output", required=True, type=Path)
    extract.add_argument("--contexts", default="", help="optional comma-separated context IDs")
    extract.add_argument("--ffmpeg", default="ffmpeg")
    extract.add_argument("--ffprobe", default="ffprobe")
    args = parser.parse_args()
    if args.command == "validate":
        panel = load_panel_manifest(args.manifest, verify_sources=not args.skip_source_hashes)
        print(json.dumps(validation_summary(panel), indent=2, sort_keys=True))
        return
    selected = [item for item in args.contexts.split(",") if item] or None
    summary = extract_panel(
        args.manifest, args.output, context_ids=selected,
        ffmpeg=args.ffmpeg, ffprobe=args.ffprobe,
    )
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
