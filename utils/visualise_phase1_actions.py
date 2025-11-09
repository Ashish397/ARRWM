#!/usr/bin/env python3
"""Generate a video sample from a trained checkpoint for phase-1 models."""

import argparse
import json
import sys
import subprocess
from pathlib import Path
from typing import Iterable, Optional

import torch
from omegaconf import OmegaConf
from torchvision.io import write_video

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.action_inference import ActionCausalInferencePipeline
from utils.memory import move_model_to_device_with_memory_preservation


DEFAULT_LOG_ROOT = Path("/home/ashish/log")
DEFAULT_OUTPUT_ROOT = Path("/home/ashish/vis")

ACTION_PRESETS: dict[str, tuple[float, float]] = {
    "noop": (0.0, 0.0),
    "straight": (1.0, 0.0),
    "back": (-1.0, 0.0),
    "straight_right": (0.5, 0.5),
    "straight_left": (0.5, -0.5),
    "back_right": (-0.5, 0.5),
    "back_left": (-0.5, -0.5),
    "right": (0.0, 0.5),
    "left": (0.0, -0.5),
}
DEFAULT_PROMPT_SWEEP_COUNT = 5

def _clean_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip FSDP wrapper prefixes so the plain module can load the weights."""
    cleaned = {}
    for key, value in state_dict.items():
        parts = key.split(".")
        filtered = [part for part in parts if part not in {"module", "_fsdp_wrapped_module"}]
        new_key = ".".join(filtered)
        cleaned[new_key] = value
    return cleaned


def _load_checkpoint_with_storage_fallback(path: Path, **torch_load_kwargs):
    """
    Load a checkpoint while handling typed storages that were saved as non-resizable.

    PyTorch 2.8 tightened storage resize behaviour which breaks checkpoints produced
    with older releases (or downstream forks) that saved storages with `resizable=False`.
    When that happens, fall back to cloning the underlying untyped storage before
    rebuilding tensors so we can materialise the tensor views without crashing.
    """
    try:
        return torch.load(path, **torch_load_kwargs)
    except RuntimeError as exc:
        msg = str(exc)
        if "not resizable" not in msg:
            raise
        print("[Info] Checkpoint uses non-resizable storages; cloning for compatibility.", file=sys.stderr)

    import torch._utils as torch_utils

    original_rebuild_tensor = torch_utils._rebuild_tensor

    def _rebuild_tensor_with_clone(storage, storage_offset, size, stride):
        untyped = storage._untyped_storage
        if not untyped.resizable():
            untyped = untyped.clone()
        tensor = torch.empty((0,), dtype=storage.dtype, device=untyped.device)
        return tensor.set_(untyped, storage_offset, size, stride)

    torch_utils._rebuild_tensor = _rebuild_tensor_with_clone
    try:
        return torch.load(path, **torch_load_kwargs)
    finally:
        torch_utils._rebuild_tensor = original_rebuild_tensor


def _parse_step_from_dirname(dirname: str) -> Optional[int]:
    try:
        return int(dirname.split("_")[-1])
    except ValueError:
        return None


def _choose_run_dir(log_root: Path, config_name: str, timestamp: Optional[str]) -> Path:
    if timestamp:
        candidate = log_root / f"{config_name}_{timestamp}"
        if candidate.is_dir():
            return candidate
        raise FileNotFoundError(f"Run directory not found: {candidate}")

    # Allow special values to mean "use the log root directly"
    if config_name in (".", "root", "self"):
        return log_root

    direct = log_root / config_name
    if direct.is_dir():
        return direct

    matches: list[Path] = []
    for path in log_root.glob(f"{config_name}_*"):
        if path.is_dir():
            matches.append(path)
    if not matches:
        # Fallback: if there are checkpoint files directly under log_root, use log_root
        if any(log_root.glob("checkpoint_model_*.pt")):
            return log_root
        raise FileNotFoundError(f"No runs found for pattern {config_name} under {log_root}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def _choose_checkpoint(run_dir: Path, requested_step: Optional[int]) -> Path:
    # Support two layouts:
    # 1) directories: checkpoint_model_XXXXXX/model.pt
    # 2) files:       checkpoint_model_XXXXXX.pt

    dir_candidates: list[Path] = [p for p in run_dir.glob("checkpoint_model_*") if p.is_dir()]
    file_candidates: list[Path] = [p for p in run_dir.glob("checkpoint_model_*.pt") if p.is_file()]

    # Filter by requested step if provided
    if requested_step is not None:
        name = f"checkpoint_model_{requested_step:06d}"
        dir_path = run_dir / name
        file_path = run_dir / f"{name}.pt"
        if dir_path.is_dir():
            ckpt_file = dir_path / "model.pt"
            if not ckpt_file.is_file():
                raise FileNotFoundError(f"Checkpoint file missing: {ckpt_file}")
            return ckpt_file
        if file_path.is_file():
            return file_path
        raise FileNotFoundError(f"Requested checkpoint step {requested_step} not found in {run_dir}")

    # Otherwise choose the latest by parsed step
    best_dir = max(dir_candidates, key=lambda p: _parse_step_from_dirname(p.name) or -1, default=None)
    best_file = max(file_candidates, key=lambda p: _parse_step_from_dirname(p.stem) or -1, default=None)

    # Compare steps
    best_dir_step = _parse_step_from_dirname(best_dir.name) if best_dir is not None else -1
    best_file_step = _parse_step_from_dirname(best_file.stem) if best_file is not None else -1

    if best_dir_step >= best_file_step and best_dir is not None:
        ckpt_file = best_dir / "model.pt"
        if not ckpt_file.is_file():
            raise FileNotFoundError(f"Checkpoint file missing: {ckpt_file}")
        return ckpt_file
    if best_file is not None:
        return best_file

    raise FileNotFoundError(f"No checkpoints found in {run_dir}")


def _list_rel_dirs(base_root: Path, output_bucket: Optional[int]) -> list[Path]:
    rel_dirs: list[Path] = []
    for parent in base_root.glob("output_rides_*"):
        if not parent.is_dir():
            continue
        if output_bucket is not None and not parent.name.endswith(str(output_bucket)):
            continue
        for sub in parent.iterdir():
            if sub.is_dir():
                rel_dirs.append(sub.relative_to(base_root))
    return sorted(rel_dirs)


def _select_rel_dir(
    base_root: Path,
    ride_name: Optional[str],
    output_bucket: Optional[int],
    output_index: int,
) -> Path:
    rel_dirs = _list_rel_dirs(base_root, output_bucket)
    if not rel_dirs:
        raise RuntimeError(f"No ride directories found under {base_root}")

    candidates: Iterable[Path]
    if ride_name:
        ride_name = ride_name.strip().rstrip("/")
        ride_matches = [
            rel
            for rel in rel_dirs
            if rel.name == ride_name or str(rel) == ride_name or rel.name.endswith(ride_name)
        ]
        if output_bucket is not None:
            bucket_name = f"output_rides_{output_bucket}"
            ride_matches = [rel for rel in ride_matches if rel.parts[0] == bucket_name]
        if not ride_matches:
            raise RuntimeError(f"Ride '{ride_name}' not found under {base_root}")
        candidates = ride_matches
    else:
        candidates = rel_dirs

    candidate_list = list(candidates)
    if not candidate_list:
        raise RuntimeError("No matching ride directories found after filtering.")

    index = max(0, min(output_index, len(candidate_list) - 1))
    return candidate_list[index]


def _load_caption(caption_root: Path, rel_dir: Path, *, text_pre_encoded: bool = False, encoded_suffix: str = '_encoded') -> tuple[str, Optional[torch.Tensor]]:
    caption_dir = caption_root / rel_dir
    if not caption_dir.is_dir():
        raise RuntimeError(f"Caption directory missing: {caption_dir}")

    prompt_embeds: Optional[torch.Tensor] = None
    if text_pre_encoded:
        encoded_candidates = sorted(caption_dir.glob(f"*{encoded_suffix}.json"))
        if not encoded_candidates:
            raise RuntimeError(f"No encoded caption JSON found for {rel_dir} in {caption_dir}")
        encoded_path = encoded_candidates[0]
        if len(encoded_candidates) > 1:
            raise RuntimeError(f"Multiple encoded captions found for {rel_dir}, using {encoded_path}")

        with encoded_path.open('r', encoding='utf-8') as efh:
            encoded_payload = json.load(efh)
        encoded_values = encoded_payload.get('caption_encoded')
        if encoded_values is None:
            raise RuntimeError(f"'caption_encoded' missing in {encoded_path}")
        prompt_embeds = torch.tensor(encoded_values, dtype=torch.float32)

    caption_candidates = sorted(caption_dir.glob('*InternVL3_8B.json'))
    caption_path = caption_candidates[0]
    if len(caption_candidates) > 1:
        raise RuntimeError(f"Multiple caption JSON found for {rel_dir}, using {caption_path}")

    with caption_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    caption = (data.get("combined_analysis") or "").strip()
    if not caption:
        raise RuntimeError(f"Caption text missing in {caption_path}")
    if prompt_embeds is not None:
        return caption, prompt_embeds
    else:
        return caption, None


def _load_generator_weights(
    pipeline: ActionCausalInferencePipeline,
    checkpoint_path: Path,
    prefer_ema: bool,
) -> None:
    ckpt = _load_checkpoint_with_storage_fallback(checkpoint_path, map_location="cpu")
    gen_key = None
    if prefer_ema and "generator_ema" in ckpt:
        gen_key = "generator_ema"
    elif "generator" in ckpt:
        gen_key = "generator"
    elif "model" in ckpt:
        gen_key = "model"

    if gen_key is None:
        raise KeyError(f"Generator weights not found in checkpoint {checkpoint_path}")

    state_dict = ckpt[gen_key]
    if not isinstance(state_dict, dict):
        raise TypeError(f"Unexpected state dict type for key '{gen_key}' in {checkpoint_path}")

    cleaned_state = _clean_state_dict_keys(state_dict)
    missing, unexpected = pipeline.generator.load_state_dict(cleaned_state, strict=False)
    if missing:
        print(f"[Warning] {len(missing)} generator parameters missing: {missing[:8]} ...", file=sys.stderr)
    if unexpected:
        print(f"[Warning] {len(unexpected)} unexpected generator parameters: {unexpected[:8]} ...", file=sys.stderr)

    if hasattr(pipeline, "action_projection") and pipeline.action_projection is not None:
        action_sd = ckpt.get("action_projection")
        if isinstance(action_sd, dict):
            pipeline.action_projection.load_state_dict(action_sd)


def _resolve_roots(config: OmegaConf, split: str) -> tuple[Optional[Path], Path]:
    split = split.lower()
    if split == "train":
        latent_attr = getattr(config, "real_latent_root", None)
        caption_root = Path(config.caption_root)
    elif split == "test":
        latent_attr = getattr(config, "val_real_latent_root", None)
        caption_root = Path(config.val_caption_root)
    else:
        raise ValueError(f"Unsupported split '{split}'. Expected 'train' or 'test'.")
    latent_root = Path(latent_attr) if latent_attr else None
    return latent_root, caption_root


def _extract_checkpoint_step(checkpoint_path: Path) -> str:
    candidates = [checkpoint_path.stem, checkpoint_path.parent.name]
    for name in candidates:
        step = _parse_step_from_dirname(name)
        if step is not None:
            return f"{step:06d}"
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a visualisation video from a phase-1 checkpoint.")
    parser.add_argument("--config-path", type=Path, required=True, help="Path to the YAML config used for the run.")
    parser.add_argument("--run", help="Config name / run prefix (e.g. dmd1-init-mse).", default=None)
    parser.add_argument("--timestamp", help="Optional timestamp suffix to disambiguate runs (e.g. 20251017-020749).")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT, help="Root directory containing run logs.")
    parser.add_argument("--checkpoint-step", type=int, help="Specific checkpoint step to load (e.g. 700).")
    parser.add_argument("--ride-name", help="Ride identifier (e.g. ride_26361_20240403000848).")
    parser.add_argument("--split", choices=("train", "test"), default="test", help="Dataset split to pull captions from.")
    parser.add_argument(
        "--output-index",
        type=int,
        default=None,
        help="Index within the filtered ride list when ride_name is ambiguous or omitted. "
        "If omitted the first five prompts are rendered sequentially.",
    )
    parser.add_argument("--output-ride", type=int, help="Restrict search to a specific output_rides_N bucket.")
    parser.add_argument("--num-frames", type=int, help="Number of frames to generate (default: config.num_training_frames).")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second for the saved video.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory to store the generated video.")
    parser.add_argument("--device", default='cuda:0', help="Computation device (e.g. cuda:0, cpu). Default: auto.")
    parser.add_argument("--no-ema", action="store_true", help="Disable EMA weights and use the raw generator parameters.")
    parser.add_argument("--seed", type=int, help="Optional random seed for noise sampling.")
    parser.add_argument(
        "--action",
        choices=tuple(ACTION_PRESETS.keys()),
        default=None,
        help="Predefined action sequence to condition the model: noop=(0,0), straight=(1,0), right=(1,0.5), left=(1,-0.5). "
        "Omit to render every preset action.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action is None:
        actions_to_run = list(ACTION_PRESETS.keys())
    else:
        actions_to_run = [args.action]

    log_root = args.log_root.expanduser()
    if not log_root.is_dir():
        raise FileNotFoundError(f"Log root does not exist: {log_root}")

    default_cfg_path = REPO_ROOT / 'configs' / 'default_config.yaml'
    if default_cfg_path.is_file():
        base_cfg = OmegaConf.load(default_cfg_path)
    else:
        base_cfg = OmegaConf.create({})
    user_cfg = OmegaConf.load(args.config_path)
    config = OmegaConf.merge(base_cfg, user_cfg)
    text_pre_encoded = bool(getattr(config, 'text_pre_encoded', False))

    run_name = args.run or config.get("config_name", None) or Path(args.config_path).stem
    run_dir = None
    checkpoint_path = None
    if args.run:
        if not run_name:
            raise ValueError("Could not determine run name; please provide --run explicitly.")
        run_dir = _choose_run_dir(log_root, run_name, args.timestamp)
        checkpoint_path = _choose_checkpoint(run_dir, args.checkpoint_step)
        print(f"[Info] Using run directory: {run_dir}")
        print(f"[Info] Loading checkpoint: {checkpoint_path}")
    else:
        generator_ckpt = getattr(config, "generator_ckpt", None)
        if generator_ckpt:
            ckpt_candidate = Path(str(generator_ckpt)).expanduser()
            if ckpt_candidate.is_dir():
                if ckpt_candidate.name.startswith("checkpoint_model_"):
                    maybe_file = ckpt_candidate / "model.pt"
                    if maybe_file.is_file():
                        checkpoint_path = maybe_file
                        run_dir = ckpt_candidate.parent
                else:
                    run_dir = ckpt_candidate
            elif ckpt_candidate.is_file():
                checkpoint_path = ckpt_candidate
                checkpoint_parent = ckpt_candidate.parent
                if checkpoint_parent.name.startswith("checkpoint_model_"):
                    run_dir = checkpoint_parent.parent
                else:
                    run_dir = checkpoint_parent
        weights_cfg = getattr(config, "weights", None)
        if checkpoint_path is None and weights_cfg is not None:
            weights_root = Path(str(weights_cfg.root)).expanduser()
            weights_run = str(getattr(weights_cfg, "run", run_name or ""))
            if not run_name:
                run_name = weights_run or run_name
            run_dir_candidate = weights_root / weights_run
            run_dir = run_dir_candidate
            target_step = args.checkpoint_step
            if target_step is None:
                default_step = getattr(weights_cfg, "default_step", None)
                if default_step is not None:
                    try:
                        target_step = int(str(default_step))
                    except ValueError:
                        target_step = None
            checkpoint_path = _choose_checkpoint(run_dir_candidate, target_step)
            print(f"[Info] Using weights directory: {run_dir_candidate}")
            print(f"[Info] Loading checkpoint: {checkpoint_path}")
        if checkpoint_path is None and run_name:
            try:
                run_dir = _choose_run_dir(log_root, run_name, args.timestamp)
                checkpoint_path = _choose_checkpoint(run_dir, args.checkpoint_step)
                print(f"[Info] Using run directory: {run_dir}")
                print(f"[Info] Loading checkpoint: {checkpoint_path}")
            except FileNotFoundError:
                pass
    _, caption_root = _resolve_roots(config, args.split)
    # Only captions are required for selecting rides and prompts
    if not caption_root.exists():
        raise FileNotFoundError(f"Caption root not found: {caption_root}")

    # Select ride directories relative to the caption root so we don't depend on latents.
    candidate_rel_dirs = _list_rel_dirs(caption_root, args.output_ride)
    if not candidate_rel_dirs:
        raise RuntimeError(f"No ride directories found under {caption_root}")

    if args.ride_name:
        ride_query = args.ride_name.strip().rstrip("/")
        ride_matches = [
            rel
            for rel in candidate_rel_dirs
            if rel.name == ride_query or str(rel) == ride_query or rel.name.endswith(ride_query)
        ]
        if not ride_matches:
            raise RuntimeError(f"Ride '{args.ride_name}' not found under {caption_root}")
        candidate_rel_dirs = ride_matches

    if not candidate_rel_dirs:
        raise RuntimeError("No matching ride directories found after filtering.")

    if args.output_index is None:
        sweep_count = min(DEFAULT_PROMPT_SWEEP_COUNT, len(candidate_rel_dirs))
        selected_indices = list(range(sweep_count))
        print(f"[Info] No --output-index provided; iterating first {sweep_count} prompt(s).")
    else:
        clamped_index = max(0, min(args.output_index, len(candidate_rel_dirs) - 1))
        if clamped_index != args.output_index:
            print(f"[Info] Clamped output index {args.output_index} to {clamped_index} within available range.")
        selected_indices = [clamped_index]

    prompt_jobs = []
    for order, idx in enumerate(selected_indices):
        rel_dir = candidate_rel_dirs[idx]
        caption, prompt_embeds = _load_caption(caption_root, rel_dir, text_pre_encoded=text_pre_encoded)
        prompt_text = " ".join(caption.splitlines())
        prompt_jobs.append(
            {
                "rel_dir": rel_dir,
                "caption": caption,
                "prompt_embeds": prompt_embeds,
                "prompt_text": prompt_text,
                "output_index": idx,
                "order": order,
            }
        )
        print(f"[Info] Selected ride #{order + 1}: {rel_dir} (index {idx}, caption length {len(caption)} characters)")
        if text_pre_encoded and prompt_embeds is not None:
            print(f"[Info] Loaded encoded caption embedding with shape {tuple(prompt_embeds.shape)}")
        print(f"[Info] Prompt: {prompt_text}")

    if not prompt_jobs:
        raise RuntimeError("Failed to assemble prompt jobs for rendering.")

    output_root = args.output_dir.expanduser()
    if output_root == DEFAULT_OUTPUT_ROOT:
        output_override = getattr(config, "output_folder", None)
        if output_override:
            output_root = Path(str(output_override)).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    if checkpoint_path is None:
        base_root = Path('/projects/u5as/frodobots')
        for job in prompt_jobs:
            rel_dir = job["rel_dir"]
            prompt_text = job["prompt_text"]
            output_index = job["output_index"]
            ride_label = rel_dir.name.replace("/", "_")
            bucket = rel_dir.parts[0]
            remainder = Path(*rel_dir.parts[1:]) if len(rel_dir.parts) > 1 else Path()

            # Look for video in recordings directory (following pre_encode.py pattern)
            ride_dir = base_root / args.split / bucket / remainder
            recordings_dir = ride_dir / "recordings"

            reference_video = None
            if recordings_dir.exists():
                # Look for combined MP4 file first
                mp4_files = list(recordings_dir.glob("*combined_audio_video.mp4"))
                if mp4_files:
                    reference_video = mp4_files[0]
                else:
                    # Look for front camera video playlist as fallback
                    m3u8_files = list(recordings_dir.glob("*uid_s_1000*video.m3u8"))
                    if m3u8_files:
                        reference_video = m3u8_files[0]

            if not reference_video or not reference_video.is_file():
                raise FileNotFoundError(f"Reference video not found under {recordings_dir}")

            out_path = output_root / f"reference_{ride_label}_idx{output_index}.mp4"
            if not out_path.exists():
                out_path.parent.mkdir(parents=True, exist_ok=True)

                # Convert M3U8 to MP4 using ffmpeg (similar to pre_encode.py)
                if reference_video.suffix == '.m3u8':
                    print(f"[Info] Converting M3U8 to MP4 (first 5 seconds): {reference_video}")
                    cmd = [
                        'ffmpeg', '-y', '-i', str(reference_video),
                        '-t', '5',  # Limit to first 5 seconds
                        '-c:v', 'libx264', '-preset', 'ultrafast',
                        '-pix_fmt', 'yuv420p',
                        str(out_path)
                    ]

                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        print(f"[Error] FFmpeg conversion failed: {result.stderr}")
                        raise RuntimeError(f"Failed to convert video: {result.stderr}")
                else:
                    # For MP4 files, also limit to first 5 seconds
                    print(f"[Info] Converting MP4 to first 5 seconds: {reference_video}")
                    cmd = [
                        'ffmpeg', '-y', '-i', str(reference_video),
                        '-t', '5',  # Limit to first 5 seconds
                        '-c:v', 'libx264', '-preset', 'ultrafast',
                        '-pix_fmt', 'yuv420p',
                        str(out_path)
                    ]

                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        print(f"[Error] FFmpeg conversion failed: {result.stderr}")
                        raise RuntimeError(f"Failed to convert video: {result.stderr}")

            # Save prompt as text file
            prompt_path = output_root / f"reference_{ride_label}_idx{output_index}_prompt.txt"
            with open(prompt_path, 'w') as f:
                f.write(prompt_text)
            print(f"[Info] Prompt saved to: {prompt_path}")

            print(f"[Info] Reference video ready at: {out_path}")
        return

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_grad_enabled(False)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    if checkpoint_path is None:
        raise RuntimeError("Checkpoint path could not be resolved; provide --run or configure weights/generator_ckpt.")
    config = OmegaConf.merge(config, OmegaConf.create({"generator_ckpt": str(checkpoint_path)}))

    action_dim = int(getattr(config, "action_dim", 2))
    pipeline = ActionCausalInferencePipeline(config, device=device, action_dim=action_dim)
    _load_generator_weights(pipeline, checkpoint_path, prefer_ema=not args.no_ema)

    use_bfloat16 = bool(getattr(config, "mixed_precision", False)) and device.type == "cuda"
    dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    pipeline = pipeline.to(device=device, dtype=dtype)
    pipeline.generator.to(device=device, dtype=dtype)
    pipeline.vae.to(device=device, dtype=dtype)
    if getattr(pipeline, 'text_encoder', None) is not None:
        pipeline.text_encoder.to(device=device)
    if getattr(pipeline, "action_projection", None) is not None:
        pipeline.action_projection.to(device=device, dtype=dtype)
    if device.type == "cuda":
        move_model_to_device_with_memory_preservation(pipeline.generator, target_device=device)
        move_model_to_device_with_memory_preservation(pipeline.vae, target_device=device)
        if getattr(pipeline, 'text_encoder', None) is not None and not text_pre_encoded:
            move_model_to_device_with_memory_preservation(pipeline.text_encoder, target_device=device)

    num_frames = args.num_frames or int(getattr(config, "num_output_frames", getattr(config, "num_training_frames", 21)))
    print(f"[Info] Generating {num_frames} frames on {device} with dtype {dtype}.")

    action_specs = [(name, ACTION_PRESETS[name]) for name in actions_to_run]
    action_list_str = ", ".join(name for name, _ in action_specs)
    print(f"[Info] Actions to render: {action_list_str}")

    run_label = run_dir.name if run_dir is not None else (run_name or "inference")
    step_label = _extract_checkpoint_step(checkpoint_path)
    total_prompts = len(prompt_jobs)

    for job_idx, job in enumerate(prompt_jobs, start=1):
        rel_dir = job["rel_dir"]
        caption = job["caption"]
        prompt_embeds = job["prompt_embeds"]
        output_index = job["output_index"]

        ride_label = rel_dir.name.replace("/", "_")
        print(f"[Info] Rendering prompt {job_idx}/{total_prompts}: {rel_dir} (idx {output_index})")

        base_noise = torch.randn(1, num_frames, 16, 60, 104, device=device, dtype=dtype)

        for action_name, pair in action_specs:
            print(f"[Info]  -> Action '{action_name}'")
            pair_tensor = torch.tensor(pair, device=device, dtype=dtype).flatten()
            if pair_tensor.numel() != action_dim:
                raise ValueError(
                    f"Preset '{action_name}' provides {pair_tensor.numel()} values but action_dim={action_dim}. "
                    "Update the preset or config to match."
                )
            action_features = pair_tensor.view(1, 1, action_dim).expand(1, num_frames, action_dim).contiguous()
            action_payload = {"action_features": action_features}

            video = pipeline.inference(
                noise=base_noise.clone(),
                text_prompts=[caption] if not text_pre_encoded else None,
                prompt_embeds=prompt_embeds if text_pre_encoded else None,
                action_inputs=action_payload,
            )
            if isinstance(video, tuple):
                video = video[0]

            video = video[0].permute(0, 2, 3, 1).clamp(0, 1).cpu()
            video_uint8 = (video * 255.0).round().to(torch.uint8)

            out_name = f"{run_label}_{ride_label}_step{step_label}_{action_name}_idx{output_index}.mp4"
            out_path = output_root / out_name
            write_video(str(out_path), video_uint8, fps=args.fps)
            if hasattr(pipeline.vae, "model"):
                pipeline.vae.model.clear_cache()
            print(f"[Info] Saved video to: {out_path}")


if __name__ == "__main__":
    main()
