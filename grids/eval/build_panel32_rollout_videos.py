"""Build human-review timelines for the locked balanced-32 evaluation panel.

Each output contains the available real source footage within the requested
[-15,+30]-second interval at 20 fps.  A full-coverage file is exactly 45
seconds and places the model-generation boundary at review time 15.0 seconds.
Shorter published sources remain shorter: frames are never padded, frozen,
looped, or replaced by generated content.

FrodoBots is reconstructed from the stored real-video Wan latents used by the
evaluation.  Sekai is recovered with a targeted 45-second range request from
the locked YouTube source (not a whole-video or dataset download).  Ego4D and
SpatialVID use their locked official public clips, whose missing coverage is
made explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "grids/eval/panel32_locked_v2.json"
PROVENANCE = (
    ROOT / "analysis/panel32_v2/canonical/panel32_source_provenance.json"
)
OUTPUT = ROOT / "iclr/rollout_videos"
CACHE = ROOT / "analysis/panel32_v2/source_review_cache"
FRODO_ROOTS = (
    Path("/home/ashish/ARRWM_data/frodobots_encoded"),
    Path("/home/ashish/frodobots/frodobots_encoded"),
)

FPS = 20
WIDTH = 832
HEIGHT = 480
REVIEW_SECONDS = 45.0
BOUNDARY_SECONDS = 15.0
REVIEW_FRAMES = int(REVIEW_SECONDS * FPS)


class BuildError(RuntimeError):
    pass


def run(command: list[str], *, quiet: bool = False) -> None:
    kwargs: dict[str, Any] = {"check": True}
    if quiet:
        kwargs.update(stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    subprocess.run(command, **kwargs)


def json_load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BuildError(f"expected object in {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def probe(path: Path) -> dict[str, Any]:
    payload = subprocess.check_output([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries",
        "stream=width,height,avg_frame_rate,nb_read_frames:format=duration",
        "-of", "json", str(path),
    ], text=True)
    return json.loads(payload)


def duration(path: Path) -> float:
    return float(probe(path)["format"]["duration"])


def spatial_filter() -> str:
    aspect = WIDTH / HEIGHT
    return (
        f"crop=w='min(iw,ih*{aspect:.12f})':"
        f"h='min(ih,iw/{aspect:.12f})':"
        "x='(iw-out_w)/2':y='(ih-out_h)/2',"
        f"scale={WIDTH}:{HEIGHT}:flags=lanczos"
    )


def normalize_clip(
    source: Path,
    output: Path,
    *,
    boundary_png: Path,
    boundary_timestamp: float,
    trim_start: float = 0.0,
    trim_duration: float | None = None,
) -> tuple[float, int, float]:
    """Normalize only the available source interval, without any padding."""
    observed = duration(source)
    available = max(0.0, observed - trim_start)
    usable = min(available, trim_duration) if trim_duration is not None else available
    if usable <= 0:
        raise BuildError(f"{source}: no source footage in requested interval")
    frame_count = min(REVIEW_FRAMES, max(1, int(math.floor(usable * FPS + 1e-6))))
    output_duration = frame_count / FPS
    boundary_frame = int(round(boundary_timestamp * FPS))
    if not 0 <= boundary_frame < frame_count:
        raise BuildError(
            f"boundary frame {boundary_frame} outside {frame_count}-frame clip"
        )
    quantized_boundary = boundary_frame / FPS

    # The builder is intentionally restartable: completed review files are
    # validated and reused after a transient source-download failure.
    if output.exists():
        existing = probe(output)
        stream = existing["streams"][0]
        existing_frames = int(stream.get("nb_read_frames") or 0)
        existing_duration = float(existing["format"]["duration"])
        if (
            existing_frames == frame_count
            and int(stream["width"]) == WIDTH
            and int(stream["height"]) == HEIGHT
            and math.isclose(existing_duration, output_duration, abs_tol=1.0 / FPS)
        ):
            return output_duration, frame_count, quantized_boundary

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.mp4")
    video_filter = (
        f"[0:v]trim=start={trim_start:.9f}:duration={output_duration:.9f},"
        f"setpts=PTS-STARTPTS,{spatial_filter()},fps={FPS},format=yuv420p[src];"
        f"[1:v]scale={WIDTH}:{HEIGHT}:flags=lanczos,format=yuv420p[boundary];"
        f"[src][boundary]overlay=eof_action=repeat:"
        f"enable='eq(n\\,{boundary_frame})'[out]"
    )
    run([
        "ffmpeg", "-y", "-v", "error", "-threads", "4",
        "-i", str(source), "-loop", "1", "-framerate", str(FPS),
        "-i", str(boundary_png), "-filter_complex", video_filter,
        "-map", "[out]", "-frames:v", str(frame_count), "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "18",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
    ])
    temporary.replace(output)
    return output_duration, frame_count, quantized_boundary


def strip_fragment(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, parts.query, ""))


def sekai_original_excerpt_start(context: dict[str, Any]) -> float:
    note = str(context.get("selection_note", ""))
    match = re.search(r"only ([0-9.]+)--[0-9.]+ s downloaded", note)
    if match:
        return float(match.group(1))

    # Replacement clips retain the exact original-video timestamp in the
    # locked source URI even when the human-readable selection note only says
    # that a short review range was downloaded.  Prefer that explicit field
    # over inferring time from filenames or local clip duration.
    fragment = urlsplit(str(context.get("source_uri", ""))).fragment
    match = re.fullmatch(r"t=([0-9]+(?:\.[0-9]+)?)", fragment)
    if match:
        return float(match.group(1))
    raise BuildError(
        "cannot recover Sekai excerpt start from selection_note or source_uri: "
        f"{note!r}, {context.get('source_uri')!r}"
    )


def download_sekai_range(
    context: dict[str, Any], boundary_in_excerpt: float, cache: Path
) -> tuple[Path, float]:
    original_boundary = sekai_original_excerpt_start(context) + boundary_in_excerpt
    requested_start = original_boundary - BOUNDARY_SECONDS
    requested_end = original_boundary + (REVIEW_SECONDS - BOUNDARY_SECONDS)
    cache.mkdir(parents=True, exist_ok=True)
    stem = cache / f"{context['context_id']}_range"
    existing = sorted(cache.glob(f"{stem.name}.*"))
    existing = [path for path in existing if path.suffix not in (".part", ".ytdl")]
    if not existing:
        yt_dlp = shutil.which("yt-dlp")
        if yt_dlp is None:
            raise BuildError("yt-dlp executable is required for Sekai range recovery")
        template = str(stem) + ".%(ext)s"
        run([
            yt_dlp, "--no-playlist",
            "--extractor-args", "youtube:player_client=android",
            "--retries", "10", "--fragment-retries", "10",
            "--download-sections", f"*{requested_start:.6f}-{requested_end:.6f}",
            "--force-keyframes-at-cuts", "--merge-output-format", "mp4",
            "-f", (
                "bestvideo[vcodec^=avc1][height<=720]/"
                "best[vcodec^=avc1][height<=720]/best[height<=720]"
            ),
            "-o", template, strip_fragment(context["source_uri"]),
        ])
        existing = sorted(cache.glob(f"{stem.name}.*"))
        existing = [path for path in existing if path.suffix not in (".part", ".ytdl")]
    if len(existing) != 1:
        raise BuildError(f"expected one Sekai range for {context['context_id']}: {existing}")
    return existing[0], original_boundary


def find_frodo_zarr(ride: str) -> Path:
    for root in FRODO_ROOTS:
        candidate = root / ride
        if candidate.is_dir():
            return candidate
    raise BuildError(f"FrodoBots zarr not found: {ride}")


def decode_frodo_latents(
    context: dict[str, Any], cache: Path
) -> tuple[Path, float, float, bool]:
    """Decode the available true-latent interval around the eval boundary.

    Some retained FrodoBots zarrs contain exact-zero placeholders outside a
    shorter encoded source segment.  Those placeholders are not video.  We
    restrict the review clip to the contiguous nonzero run containing the
    evaluation boundary, rather than decoding the placeholders as flat frames.
    """
    import torch
    import zarr

    os.environ.setdefault("ARRWM_ACTION_ENCODER", "pca_raw")
    sys.path.insert(0, str(ROOT))
    import utils.wan_wrapper as wan_wrapper
    from utils.zarr_dataset import ZarrRideDataset

    ride, offset_text, _ = context["source_id"].split(":", 2)
    offset = int(offset_text)
    zarr_path = find_frodo_zarr(ride)
    boundary_latent = offset + 8
    latent_start_requested = max(
        0, boundary_latent - int(BOUNDARY_SECONDS * 5)
    )
    latent_end_requested = boundary_latent + int(30.0 * 5) + 1
    group = zarr.open_group(str(zarr_path), mode="r")
    # Dataset indices are post-head-drop; raw zarr index k+1 stores dataset
    # latent k (see ZarrRideDataset.load_latent_chunk).
    dataset_end = max(0, int(group["latents"].shape[0]) - 1)
    latent_end_requested = min(latent_end_requested, dataset_end)
    raw = np.asarray(
        group["latents"][
            latent_start_requested + 1 : latent_end_requested + 1
        ]
    )
    valid = np.any(raw != 0, axis=(1, 2, 3))
    boundary_local = boundary_latent - latent_start_requested
    if not 0 <= boundary_local < len(valid) or not bool(valid[boundary_local]):
        raise BuildError(f"{context['context_id']}: boundary latent is unavailable")
    valid_start = boundary_local
    while valid_start > 0 and bool(valid[valid_start - 1]):
        valid_start -= 1
    valid_end = boundary_local + 1
    while valid_end < len(valid) and bool(valid[valid_end]):
        valid_end += 1
    latent_start = latent_start_requested + valid_start
    latent_end = latent_start_requested + valid_end
    available = latent_end - latent_start
    if available <= 1:
        raise BuildError(f"{context['context_id']}: insufficient stored latents")
    boundary_timestamp = 4 * (boundary_latent - latent_start) / FPS

    cache.mkdir(parents=True, exist_ok=True)
    full_requested_interval = (
        latent_start == latent_start_requested
        and latent_end == latent_end_requested
    )
    range_tag = "full" if full_requested_interval else f"{latent_start}_{latent_end}"
    raw_output = cache / (
        f"{context['context_id']}_seeded_latent_{range_tag}_reconstruction.mp4"
    )
    expected_frames = 1 + 4 * (available - 1)
    if raw_output.exists():
        info = probe(raw_output)
        frames = int(info["streams"][0].get("nb_read_frames") or 0)
        if frames == expected_frames:
            return (
                raw_output,
                expected_frames / FPS,
                boundary_timestamp,
                not full_requested_interval,
            )

    wan_wrapper._default_wan_model_path = "/home/ashish/Wan2.1/"
    vae = wan_wrapper.WanVAEWrapper().eval().requires_grad_(False).to(
        "cuda", torch.bfloat16
    )
    scale = [vae.mean.to(device="cuda", dtype=torch.bfloat16),
             1.0 / vae.std.to(device="cuda", dtype=torch.bfloat16)]
    vae.model.clear_cache()
    temporary = raw_output.with_name(f".{raw_output.stem}.tmp.mp4")
    command = [
        "ffmpeg", "-y", "-v", "error", "-threads", "4",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-video_size", f"{WIDTH}x{HEIGHT}", "-framerate", str(FPS),
        "-i", "-", "-frames:v", str(expected_frames), "-an",
        "-c:v", "libx264", "-preset", "medium", "-crf", "16",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    assert process.stdin is not None
    written = 0
    try:
        for relative in range(0, available, 8):
            first = latent_start + relative
            count = min(8, available - relative)
            latent = ZarrRideDataset.load_latent_chunk(
                str(zarr_path), first, first + count
            ).unsqueeze(0).to("cuda", torch.bfloat16)
            z = latent.permute(0, 2, 1, 3, 4)
            with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                if relative == 0:
                    # This excerpt begins at an arbitrary continuation latent,
                    # not Wan's special one-frame video-head latent.  Seed the
                    # temporal decoder with a duplicate of its own first
                    # latent, as WanVAEWrapper(seed_first=True) does for a
                    # standalone clip, then decode the real excerpt through
                    # the warm cache.
                    dummy = vae.model.cached_decode(z[:, :, :1], scale)
                    del dummy
                pixel = vae.model.cached_decode(z, scale).float().clamp_(-1, 1)
            frames = ((pixel[0].permute(1, 2, 3, 0) + 1.0) * 127.5)
            if relative == 0:
                # Retain the same post-head-drop 20 Hz grid as evaluation:
                # one pixel frame for the first real latent, then four per
                # subsequent latent.  The discarded frames replace the old
                # flat special-head decode with a correctly seeded frame.
                frames = frames[3:]
            array = frames.clamp(0, 255).byte().cpu().numpy()
            process.stdin.write(array.tobytes())
            written += int(array.shape[0])
            del latent, z, pixel, frames, array
            torch.cuda.empty_cache()
    finally:
        process.stdin.close()
        return_code = process.wait()
        vae.model.clear_cache()
        del vae
        torch.cuda.empty_cache()
    if return_code != 0 or written != expected_frames:
        temporary.unlink(missing_ok=True)
        raise BuildError(
            f"{context['context_id']}: decoder wrote {written}/{expected_frames}, "
            f"ffmpeg rc={return_code}"
        )
    temporary.replace(raw_output)
    return (
        raw_output,
        expected_frames / FPS,
        boundary_timestamp,
        not full_requested_interval,
    )


def read_rgb_frame(path: Path, timestamp: float) -> np.ndarray:
    payload = subprocess.check_output([
        "ffmpeg", "-v", "error", "-ss", f"{timestamp:.9f}", "-i", str(path),
        "-frames:v", "1", "-vf", f"scale={WIDTH}:{HEIGHT}:flags=lanczos",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
    ])
    expected = WIDTH * HEIGHT * 3
    if len(payload) != expected:
        raise BuildError(f"cannot read boundary frame from {path}")
    return np.frombuffer(payload, dtype=np.uint8).reshape(HEIGHT, WIDTH, 3)


def boundary_mae(review: Path, boundary_png: Path, boundary_timestamp: float) -> float:
    observed = read_rgb_frame(review, boundary_timestamp).astype(np.float32)
    expected = read_rgb_frame(boundary_png, 0.0).astype(np.float32)
    return float(np.abs(observed - expected).mean())


def build(args: argparse.Namespace) -> None:
    manifest = json_load(args.manifest)
    provenance = json_load(args.provenance)
    rows = {row["context_id"]: row for row in provenance["rows"]}
    output = args.output.resolve()
    cache = args.cache.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []

    for number, context in enumerate(manifest["contexts"], 1):
        context_id = context["context_id"]
        dataset = context["dataset"]
        row = rows[context_id]
        source_boundary = float(row["boundary"]["source_timestamp_s"])
        boundary_png = Path(row["canonical"]["boundary_png"])
        source_path = (args.manifest.parent / context["source_video"]).resolve()
        output_path = output / f"{number:02d}_{context_id}_ground_truth.mp4"
        source_kind = "locked_public_source_clip"
        requested_source_start: float | None = None

        if dataset == "FrodoBots":
            (
                source_path,
                source_duration,
                frodo_boundary,
                clipped_to_encoded_run,
            ) = decode_frodo_latents(context, cache / "frodo")
            # At the beginning of a clipped encoded run, the VAE has no true
            # temporal predecessor.  Its single cache-warm-up pixel is flat;
            # omit that non-video frame and expose only reconstructed footage.
            frodo_trim_start = 1.0 / FPS if clipped_to_encoded_run else 0.0
            output_duration, output_frames, boundary_on_review = normalize_clip(
                source_path, output_path, boundary_png=boundary_png,
                boundary_timestamp=frodo_boundary - frodo_trim_start,
                trim_start=frodo_trim_start,
                trim_duration=min(
                    REVIEW_SECONDS, source_duration - frodo_trim_start
                ),
            )
            relative_interval = [
                round(-boundary_on_review, 6),
                round(output_duration - boundary_on_review, 6),
            ]
            source_kind = "stored_real_video_latent_reconstruction"
        elif dataset == "Sekai":
            source_path, original_boundary = download_sekai_range(
                context, source_boundary, cache / "sekai"
            )
            # yt-dlp's force-keyframes-at-cuts result begins at the requested
            # original boundary minus 15 seconds.
            output_duration, output_frames, boundary_on_review = normalize_clip(
                source_path, output_path, boundary_png=boundary_png,
                boundary_timestamp=BOUNDARY_SECONDS,
                trim_duration=REVIEW_SECONDS,
            )
            relative_interval = [
                round(-boundary_on_review, 6),
                round(output_duration - boundary_on_review, 6),
            ]
            source_kind = "targeted_original_youtube_range"
            requested_source_start = original_boundary - BOUNDARY_SECONDS
        else:
            source_total = duration(source_path)
            trim_start = max(0.0, source_boundary - BOUNDARY_SECONDS)
            trim_end = min(source_total, source_boundary + 30.0)
            output_duration, output_frames, boundary_on_review = normalize_clip(
                source_path, output_path, boundary_png=boundary_png,
                boundary_timestamp=source_boundary - trim_start,
                trim_start=trim_start,
                trim_duration=trim_end - trim_start,
            )
            relative_interval = [
                round(-boundary_on_review, 6),
                round(output_duration - boundary_on_review, 6),
            ]

        info = probe(output_path)
        observed_duration = float(info["format"]["duration"])
        observed_frames = int(info["streams"][0].get("nb_read_frames") or 0)
        if observed_frames != output_frames or not math.isclose(
            observed_duration, output_duration, abs_tol=1.0 / FPS
        ):
            raise BuildError(
                f"{output_path}: {observed_frames} frames, {observed_duration}s"
            )
        record = {
            "index": number,
            "context_id": context_id,
            "dataset": dataset,
            "output": output_path.name,
            "output_sha256": sha256(output_path),
            "duration_s": observed_duration,
            "fps": FPS,
            "frames": observed_frames,
            "generation_boundary_review_timestamp_s": boundary_on_review,
            "source_kind": source_kind,
            "source_used": str(source_path),
            "requested_original_source_start_s": requested_source_start,
            "available_relative_interval_s": relative_interval,
            "complete_requested_interval": (
                relative_interval[0] <= -15.0 + 1e-6
                and relative_interval[1] >= 30.0 - 1.0 / FPS
            ),
            "canonical_boundary_png": str(boundary_png),
            "boundary_frame_mae_to_canonical": boundary_mae(
                output_path, boundary_png, boundary_on_review
            ),
        }
        records.append(record)
        print(
            f"[{number:02d}/32] {context_id}: true source "
            f"relative {relative_interval[0]:.2f}--{relative_interval[1]:.2f}s; "
            f"boundary MAE {record['boundary_frame_mae_to_canonical']:.2f}",
            flush=True,
        )

    audit = {
        "schema_version": 1,
        "panel_id": manifest["panel_id"],
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "review_contract": {
            "maximum_duration_s": REVIEW_SECONDS,
            "fps": FPS,
            "maximum_frames": REVIEW_FRAMES,
            "width": WIDTH,
            "height": HEIGHT,
            "full_coverage_generation_boundary_timestamp_s": BOUNDARY_SECONDS,
            "requested_relative_interval_s": [-15.0, 30.0],
            "short_source_policy": "write only available true footage; never pad",
        },
        "records": records,
    }
    (output / "manifest.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    by_dataset: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        by_dataset.setdefault(record["dataset"], []).append(record)
    lines = [
        "# Ground-truth rollout review videos",
        "",
        "Every file contains only the available real footage within the requested",
        "relative interval [-15 s, +30 s]. Full-coverage files are exactly 45 seconds",
        "and generation begins at 15.0 s. Short official sources remain shorter;",
        "frames are never padded, frozen, looped, or replaced by model output. The",
        "per-file generation-boundary timestamp is recorded in `manifest.json`.",
        "",
        "FrodoBots files are VAE reconstructions of the stored real-video latents",
        "used by the evaluation. Sekai files are targeted ranges from the locked",
        "original YouTube videos. Ego4D and SpatialVID are the exact official public",
        "clips used to build the evaluation contexts.",
        "",
        "Coverage:",
        "",
    ]
    for dataset in ("FrodoBots", "Ego4D", "Sekai", "SpatialVID"):
        group = by_dataset[dataset]
        complete = sum(
            r["complete_requested_interval"]
            for r in group
        )
        lines.append(f"- {dataset}: {complete}/{len(group)} complete 45-second windows")
    lines.extend([
        "",
        "See `manifest.json` for per-file source coverage, hashes, and boundary",
        "alignment diagnostics.",
        "",
    ])
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--provenance", type=Path, default=PROVENANCE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--cache", type=Path, default=CACHE)
    args = parser.parse_args()
    build(args)


if __name__ == "__main__":
    main()
