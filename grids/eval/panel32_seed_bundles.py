"""Encode the locked panel32 sources for ARRWM without crossing frame 32.

The source builder writes one lossless 33-frame RGB stream per context.  This
tool consumes those exact pixels directly (there is no second crop/resample),
encodes them to nine Wan latents, and estimates the three causal seed actions
from the disjoint pixel spans ``[0:9]``, ``[9:21]``, and ``[21:33]``.  The
spans are important: the first Wan chunk decodes to nine pixels, while later
three-latent chunks decode to twelve.

The resulting bundles deliberately do not contain a scene-specific text
embedding.  The generation runner uses the same locked neutral prompt as all
other text-conditioned systems.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from grids.eval.panel32_manifest import load_panel_manifest, sha256_file  # noqa: E402


PIXEL_FRAMES = 33
WIDTH = 832
HEIGHT = 480
LATENT_FRAMES = 9
SEED_SPANS = ((0, 9), (9, 21), (21, 33))
NEUTRAL_PROMPT = "A first-person view of an outdoor environment."
PCA_SCALES = np.asarray(
    [93.7, 57.7, 22.5, 21.2, 18.1, 14.5, 12.6, 10.8], dtype=np.float64
)


class BundleError(RuntimeError):
    pass


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_source_summary(path: Path, panel) -> dict[str, dict[str, Any]]:
    try:
        summary = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BundleError(f"cannot read source provenance {path}: {exc}") from exc
    if summary.get("status") != "pass" or summary.get("complete_panel") is not True:
        raise BundleError("canonical source provenance is not a complete passing panel")
    if summary.get("panel_id") != panel.panel_id:
        raise BundleError("source provenance panel_id mismatch")
    if summary.get("manifest_sha256") != panel.sha256:
        raise BundleError("source provenance manifest SHA mismatch")
    rows = summary.get("rows")
    if not isinstance(rows, list) or len(rows) != 32:
        raise BundleError("source provenance must contain exactly 32 rows")
    by_id = {row.get("context_id"): row for row in rows if isinstance(row, dict)}
    expected = {row.context_id for row in panel.contexts}
    if len(by_id) != 32 or set(by_id) != expected:
        raise BundleError("source provenance context IDs do not match the panel")
    return by_id


def decode_stream(path: Path) -> bytes:
    try:
        payload = subprocess.check_output([
            "ffmpeg", "-v", "error", "-threads", "1", "-i", str(path),
            "-map", "0:v:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
        ])
    except (OSError, subprocess.CalledProcessError) as exc:
        raise BundleError(f"cannot decode canonical stream {path}: {exc}") from exc
    expected = PIXEL_FRAMES * WIDTH * HEIGHT * 3
    if len(payload) != expected:
        raise BundleError(
            f"{path}: decoded {len(payload)} RGB bytes, expected {expected}"
        )
    return payload


def pca_action(
    frames: np.ndarray,
    cotracker: torch.nn.Module,
    mean: torch.Tensor,
    components_t: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, Any]]:
    if frames.shape[0] < 2:
        raise BundleError("a seed-action span needs at least two frames")
    video = (
        torch.from_numpy(np.ascontiguousarray(frames))
        .permute(0, 3, 1, 2)
        .unsqueeze(0)
        .to("cuda", dtype=torch.float32)
    )
    with torch.inference_mode(), torch.amp.autocast(device_type="cuda", enabled=True):
        tracks, visibility = cotracker(video, grid_size=10)
    motion = (tracks[:, 1:] - tracks[:, :-1]).mean(dim=1).squeeze(0)
    vis = visibility
    if vis.ndim == 4:
        vis = vis[..., 0]
    vis = vis.to(dtype=motion.dtype).mean(dim=1).squeeze(0)
    field = torch.cat([motion, vis[:, None]], dim=-1)
    flat = field[:, :2].reshape(1, 200).float()
    raw = (flat - mean) @ components_t
    squashed = torch.tanh(raw[:, :8] / scales)[0]
    meta = {
        "tracked_points": int(field.shape[0]),
        "mean_visibility": float(field[:, 2].mean().item()),
        "raw_pca_first8": [float(value) for value in raw[0, :8].cpu()],
        "squashed_pca_first8": [float(value) for value in squashed.cpu()],
    }
    return squashed[:2].float().cpu(), meta


def build(args: argparse.Namespace) -> dict[str, Any]:
    panel = load_panel_manifest(args.panel_manifest, verify_sources=False)
    rows = load_source_summary(args.source_provenance, panel)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    seed_cache = output / ".seed_cache"
    seed_cache.mkdir(parents=True, exist_ok=True)

    import utils.wan_wrapper as wan_wrapper
    model_root = args.wan_model_root.resolve()
    wan_wrapper._default_wan_model_path = str(model_root) + "/"
    vae = wan_wrapper.WanVAEWrapper().to("cuda", torch.bfloat16).eval()

    staged: dict[str, dict[str, Any]] = {}
    for index, context in enumerate(panel.contexts, 1):
        row = rows[context.context_id]
        canonical = row.get("canonical", {})
        stream = Path(str(canonical.get("stream", ""))).resolve()
        if not stream.is_file():
            raise BundleError(f"missing canonical stream for {context.context_id}: {stream}")
        observed = sha256_file(stream)
        if observed != canonical.get("stream_sha256"):
            raise BundleError(f"canonical stream SHA mismatch for {context.context_id}")
        cache_path = seed_cache / f"{context.context_id}.pt"
        if cache_path.is_file():
            record = torch.load(cache_path, map_location="cpu", weights_only=False)
            if record.get("panel_manifest_sha256") != panel.sha256:
                raise BundleError(f"{context.context_id}: cached panel manifest mismatch")
            if record.get("source_stream_sha256") != observed:
                raise BundleError(f"{context.context_id}: cached source stream mismatch")
            if tuple(record["seed"].shape) != (LATENT_FRAMES, 16, 60, 104):
                raise BundleError(f"{context.context_id}: cached seed shape mismatch")
            staged[context.context_id] = record
            print(f"[panel32-seed] VAE {index}/32 {context.context_id} cached", flush=True)
            continue
        rgb = decode_stream(stream)
        array = np.frombuffer(rgb, dtype=np.uint8).reshape(
            PIXEL_FRAMES, HEIGHT, WIDTH, 3
        ).copy()
        pixels = (
            torch.from_numpy(array)
            .permute(3, 0, 1, 2)
            .unsqueeze(0)
            .to("cuda", dtype=torch.bfloat16)
        )
        pixels = pixels / 127.5 - 1.0
        with torch.inference_mode():
            latents = vae.encode_to_latent(pixels).float().cpu()
        if tuple(latents.shape) != (1, LATENT_FRAMES, 16, 60, 104):
            raise BundleError(
                f"{context.context_id}: unexpected Wan latent shape {tuple(latents.shape)}"
            )
        record = {
            "context_id": context.context_id,
            "dataset": context.dataset,
            "panel_id": panel.panel_id,
            "panel_manifest_sha256": panel.sha256,
            "source_provenance": str(args.source_provenance.resolve()),
            "source_stream": str(stream),
            "source_stream_sha256": observed,
            "source_decoded_rgb24_sha256": sha256_bytes(rgb),
            "seed": latents[0].to(torch.float16),
            "seed_pixel_frames": PIXEL_FRAMES,
            "seed_latent_frames": LATENT_FRAMES,
            "seed_action_spans_half_open": [list(span) for span in SEED_SPANS],
        }
        temporary = cache_path.with_name(f".{cache_path.name}.tmp.{os.getpid()}")
        torch.save(record, temporary)
        temporary.replace(cache_path)
        staged[context.context_id] = record
        print(f"[panel32-seed] VAE {index}/32 {context.context_id}", flush=True)

    del vae
    torch.cuda.empty_cache()

    text_encoder = wan_wrapper.WanTextEncoder().eval()
    with torch.inference_mode():
        prompt_embedding = text_encoder([NEUTRAL_PROMPT])["prompt_embeds"].float().cpu()
    if tuple(prompt_embedding.shape) != (1, 512, 4096):
        raise BundleError(
            f"unexpected neutral-prompt embedding shape {tuple(prompt_embedding.shape)}"
        )
    prompt_path = output / "neutral_prompt.pt"
    prompt_temporary = prompt_path.with_name(
        f".{prompt_path.name}.tmp.{os.getpid()}"
    )
    torch.save(
        {"prompt": NEUTRAL_PROMPT, "prompt_embeds": prompt_embedding},
        prompt_temporary,
    )
    prompt_temporary.replace(prompt_path)
    del text_encoder
    torch.cuda.empty_cache()

    pca_path = args.pca_checkpoint.resolve()
    blob = torch.load(pca_path, map_location="cpu", weights_only=False)
    mean = torch.as_tensor(np.asarray(blob["pca_mean"]), device="cuda", dtype=torch.float32)
    components_t = torch.as_tensor(
        np.asarray(blob["pca_comp"]).T, device="cuda", dtype=torch.float32
    )
    scales = torch.as_tensor(PCA_SCALES, device="cuda", dtype=torch.float32)
    cotracker = torch.hub.load(args.cotracker_repo, "cotracker3_offline", source="local")
    cotracker = cotracker.to("cuda").eval()
    for parameter in cotracker.parameters():
        parameter.requires_grad_(False)

    index_rows = []
    for index, context in enumerate(panel.contexts, 1):
        record = staged[context.context_id]
        rgb = decode_stream(Path(record["source_stream"]))
        frames = np.frombuffer(rgb, dtype=np.uint8).reshape(
            PIXEL_FRAMES, HEIGHT, WIDTH, 3
        ).copy()
        actions, action_meta = [], []
        for start, stop in SEED_SPANS:
            action, meta = pca_action(
                frames[start:stop], cotracker, mean, components_t, scales
            )
            actions.append(action)
            action_meta.append({"start": start, "stop": stop, **meta})
        seed_actions = torch.stack(actions)
        record["seed_actions"] = seed_actions
        # ``WorldModelEngine._load_seed_actions`` accepts the established
        # ``actions``/``z_actions`` keys when loading a dictionary.  Retain
        # the descriptive key as well, but make the bundle directly
        # consumable by the inference path rather than relying on an adapter
        # to rewrite it at launch time.
        record["actions"] = seed_actions
        record["seed_actions_lineage"] = {
            "producer": "frozen CoTracker3 offline 10x10 -> frozen pca_raw -> tanh/scales",
            "pca_checkpoint": str(pca_path),
            "pca_checkpoint_sha256": sha256_file(pca_path),
            "spans": action_meta,
        }
        bundle = output / f"{context.context_id}.pt"
        temporary = bundle.with_name(f".{bundle.name}.tmp.{os.getpid()}")
        torch.save(record, temporary)
        temporary.replace(bundle)
        index_rows.append({
            "context_id": context.context_id,
            "dataset": context.dataset,
            "bundle": str(bundle.resolve()),
            "bundle_sha256": sha256_file(bundle),
            "source_stream_sha256": record["source_stream_sha256"],
            "seed_shape": list(record["seed"].shape),
            "seed_actions": [[float(x) for x in row] for row in seed_actions],
        })
        print(
            f"[panel32-seed] action {index}/32 {context.context_id} "
            f"{index_rows[-1]['seed_actions']}", flush=True
        )

    summary = {
        "status": "pass",
        "panel_id": panel.panel_id,
        "panel_manifest": str(panel.path),
        "panel_manifest_sha256": panel.sha256,
        "source_provenance": str(args.source_provenance.resolve()),
        "source_provenance_sha256": sha256_file(args.source_provenance),
        "contexts": len(index_rows),
        "seed_pixel_frames": PIXEL_FRAMES,
        "seed_latent_frames": LATENT_FRAMES,
        "seed_action_spans_half_open": [list(span) for span in SEED_SPANS],
        "neutral_prompt": NEUTRAL_PROMPT,
        "neutral_prompt_sha256": sha256_bytes(NEUTRAL_PROMPT.encode("utf-8")),
        "neutral_prompt_embedding": str(prompt_path.resolve()),
        "neutral_prompt_embedding_sha256": sha256_file(prompt_path),
        "rows": index_rows,
    }
    atomic_json(output / "panel32_seed_bundles.json", summary)
    return summary


def validate(args: argparse.Namespace) -> dict[str, Any]:
    panel = load_panel_manifest(args.panel_manifest, verify_sources=False)
    summary_path = args.output.resolve() / "panel32_seed_bundles.json"
    summary = json.loads(summary_path.read_text())
    if summary.get("status") != "pass" or summary.get("contexts") != 32:
        raise BundleError("seed-bundle summary is not a complete pass")
    if summary.get("panel_manifest_sha256") != panel.sha256:
        raise BundleError("seed-bundle manifest SHA mismatch")
    if summary.get("neutral_prompt") != NEUTRAL_PROMPT:
        raise BundleError("seed-bundle neutral prompt mismatch")
    prompt_path = Path(summary.get("neutral_prompt_embedding", ""))
    if (
        not prompt_path.is_file()
        or sha256_file(prompt_path) != summary.get("neutral_prompt_embedding_sha256")
    ):
        raise BundleError("neutral prompt embedding is missing or changed")
    prompt = torch.load(prompt_path, map_location="cpu", weights_only=False)
    if prompt.get("prompt") != NEUTRAL_PROMPT:
        raise BundleError("neutral prompt embedding text mismatch")
    if tuple(prompt["prompt_embeds"].shape) != (1, 512, 4096):
        raise BundleError("neutral prompt embedding shape mismatch")
    rows = summary.get("rows", [])
    if {row.get("context_id") for row in rows} != {
        row.context_id for row in panel.contexts
    }:
        raise BundleError("seed-bundle contexts mismatch")
    for row in rows:
        path = Path(row["bundle"])
        if not path.is_file() or sha256_file(path) != row["bundle_sha256"]:
            raise BundleError(f"missing or changed seed bundle: {path}")
        bundle = torch.load(path, map_location="cpu", weights_only=False)
        if tuple(bundle["seed"].shape) != (9, 16, 60, 104):
            raise BundleError(f"bad seed shape: {path}")
        if tuple(bundle["seed_actions"].shape) != (3, 2):
            raise BundleError(f"bad seed actions: {path}")
        if not torch.isfinite(bundle["seed"]).all() or not torch.isfinite(
            bundle["seed_actions"]
        ).all():
            raise BundleError(f"non-finite seed bundle: {path}")
    return summary


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    for name in ("build", "validate"):
        child = sub.add_parser(name)
        child.add_argument("--panel-manifest", type=Path, required=True)
        child.add_argument("--source-provenance", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True)
        child.add_argument(
            "--wan-model-root", type=Path,
            default=Path("/home/ashish/Wan2.1"),
        )
        child.add_argument(
            "--pca-checkpoint", type=Path,
            default=ROOT / "code_release/preprocessing/checkpoints/pca_basis.pt",
        )
        child.add_argument(
            "--cotracker-repo", type=Path,
            default=Path.home() / ".cache/torch/hub/facebookresearch_co-tracker_main",
        )
    return result


def main() -> None:
    args = parser().parse_args()
    summary = build(args) if args.command == "build" else validate(args)
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
