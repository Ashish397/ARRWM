#!/usr/bin/env python3
"""Generate a video sample from a trained checkpoint for phase-1 models."""

import argparse
import json
import sys
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Optional

import torch
from omegaconf import OmegaConf
from torchvision.io import write_video

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline import CausalInferencePipeline
from utils.dataset import VideoLatentCaptionDataset


DEFAULT_LOG_ROOT = Path("/scratch/u5as/as1748.u5as/frodobots/dmd2/logs")
DEFAULT_OUTPUT_ROOT = Path("/scratch/u5as/as1748.u5as/frodobots/visualisations")

def _clean_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Strip FSDP wrapper prefixes so the plain module can load the weights."""
    replacements = ("_fsdp_wrapped_module.", "module.")
    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in replacements:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        cleaned[new_key] = value
    return cleaned


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

    direct = log_root / config_name
    if direct.is_dir():
        return direct

    matches: list[Path] = []
    for path in log_root.glob(f"{config_name}_*"):
        if path.is_dir():
            matches.append(path)
    if not matches:
        raise FileNotFoundError(f"No runs found for pattern {config_name} under {log_root}")
    return max(matches, key=lambda p: p.stat().st_mtime)


def _choose_checkpoint(run_dir: Path, requested_step: Optional[int]) -> Path:
    checkpoint_dirs: list[Path] = []
    for path in run_dir.glob("checkpoint_model_*"):
        if path.is_dir():
            checkpoint_dirs.append(path)
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoint_model_* directories found in {run_dir}")

    if requested_step is not None:
        name = f"checkpoint_model_{requested_step:06d}"
        candidate = run_dir / name
        if not candidate.is_dir():
            raise FileNotFoundError(f"Requested checkpoint step {requested_step} not found in {run_dir}")
        checkpoint_dirs = [candidate]

    selected = max(checkpoint_dirs, key=lambda p: _parse_step_from_dirname(p.name) or -1)
    ckpt_file = selected / "model.pt"
    if not ckpt_file.is_file():
        raise FileNotFoundError(f"Checkpoint file missing: {ckpt_file}")
    return ckpt_file


def _list_rel_dirs(latent_root: Path, output_bucket: Optional[int]) -> list[Path]:
    rel_dirs: list[Path] = []
    for parent in latent_root.glob("output_rides_*"):
        if not parent.is_dir():
            continue
        if output_bucket is not None and not parent.name.endswith(str(output_bucket)):
            continue
        for sub in parent.iterdir():
            if sub.is_dir():
                rel_dirs.append(sub.relative_to(latent_root))
    return sorted(rel_dirs)


def _select_rel_dir(
    latent_root: Path,
    ride_name: Optional[str],
    output_bucket: Optional[int],
    output_index: int,
) -> Path:
    rel_dirs = _list_rel_dirs(latent_root, output_bucket)
    if not rel_dirs:
        raise RuntimeError(f"No ride directories found under {latent_root}")

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
            raise RuntimeError(f"Ride '{ride_name}' not found under {latent_root}")
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
    pipeline: CausalInferencePipeline,
    checkpoint_path: Path,
    prefer_ema: bool,
) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
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


def _resolve_roots(config: OmegaConf, split: str) -> tuple[Path, Path]:
    split = split.lower()
    if split == "train":
        latent_root = Path(config.real_latent_root)
        caption_root = Path(config.caption_root)
    elif split == "test":
        latent_root = Path(config.val_real_latent_root)
        caption_root = Path(config.val_caption_root)
    else:
        raise ValueError(f"Unsupported split '{split}'. Expected 'train' or 'test'.")
    return latent_root, caption_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a visualisation video from a phase-1 checkpoint.")
    parser.add_argument("--config-path", type=Path, required=True, help="Path to the YAML config used for the run.")
    parser.add_argument("--run", help="Config name / run prefix (e.g. dmd1-init-mse).", default=None)
    parser.add_argument("--timestamp", help="Optional timestamp suffix to disambiguate runs (e.g. 20251017-020749).")
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT, help="Root directory containing run logs.")
    parser.add_argument("--checkpoint-step", type=int, help="Specific checkpoint step to load (e.g. 700).")
    parser.add_argument("--ride-name", help="Ride identifier (e.g. ride_26361_20240403000848).")
    parser.add_argument("--split", choices=("train", "test"), default="test", help="Dataset split to pull captions from.")
    parser.add_argument("--output-index", type=int, default=0, help="Index within the filtered ride list when ride_name is ambiguous or omitted.")
    parser.add_argument("--output-ride", type=int, help="Restrict search to a specific output_rides_N bucket.")
    parser.add_argument("--num-frames", type=int, help="Number of frames to generate (default: config.num_training_frames).")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second for the saved video.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Directory to store the generated video.")
    parser.add_argument("--device", default=None, help="Computation device (e.g. cuda:0, cpu). Default: auto.")
    parser.add_argument("--use-ema", action="store_true", help="Load generator_ema weights when available.")
    parser.add_argument("--seed", type=int, help="Optional random seed for noise sampling.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

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
    latent_root, caption_root = _resolve_roots(config, args.split)
    if not latent_root.exists():
        raise FileNotFoundError(f"Latent root not found: {latent_root}")
    if not caption_root.exists():
        raise FileNotFoundError(f"Caption root not found: {caption_root}")

    rel_dir = _select_rel_dir(latent_root, args.ride_name, args.output_ride, args.output_index)
    caption, prompt_embeds = _load_caption(caption_root, rel_dir, text_pre_encoded=text_pre_encoded)
    print(f"[Info] Selected ride: {rel_dir} (caption length {len(caption)} characters)")
    if text_pre_encoded and prompt_embeds is not None:
        print(f"[Info] Loaded encoded caption embedding with shape {tuple(prompt_embeds.shape)}")

    prompt_text = " ".join(caption.splitlines())
    print(f"[Info] Prompt: {prompt_text}")

    output_root = args.output_dir.expanduser()
    output_root.mkdir(parents=True, exist_ok=True)

    if not args.run:
        ride_label = rel_dir.name.replace("/", "_")
        base_root = Path('/projects/u5as/frodobots')
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
        
        out_path = output_root / f"reference_{ride_label}.mp4"
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
        prompt_path = output_root / f"reference_{ride_label}_prompt.txt"
        with open(prompt_path, 'w') as f:
            f.write(prompt_text)
        print(f"[Info] Prompt saved to: {prompt_path}")
        
        print(f"[Info] Reference video ready at: {out_path}")
        return

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)
    if args.seed is not None:
        torch.manual_seed(args.seed)

    assert checkpoint_path is not None and run_dir is not None
    config = OmegaConf.merge(config, OmegaConf.create({"generator_ckpt": str(checkpoint_path)}))

    pipeline = CausalInferencePipeline(config, device=device)
    _load_generator_weights(pipeline, checkpoint_path, prefer_ema=args.use_ema)

    use_bfloat16 = bool(getattr(config, "mixed_precision", False)) and device.type == "cuda"
    dtype = torch.bfloat16 if use_bfloat16 else torch.float32
    pipeline = pipeline.to(dtype=dtype)
    pipeline.generator.to(device=device, dtype=dtype)
    pipeline.vae.to(device=device, dtype=dtype)
    if getattr(pipeline, 'text_encoder', None) is not None:
        pipeline.text_encoder.to(device=device)

    num_frames = args.num_frames or int(getattr(config, "num_training_frames", 21))
    noise = torch.randn(1, num_frames, 16, 60, 104, device=device, dtype=dtype)
    print(f"[Info] Generating {num_frames} frames on {device} with dtype {dtype}.")

    video = pipeline.inference(
        noise=noise,
        text_prompts=[caption] if not text_pre_encoded else None,
        prompt_embeds=prompt_embeds if text_pre_encoded else None,
    )
    if isinstance(video, tuple):
        video = video[0]

    video = video[0].permute(0, 2, 3, 1).clamp(0, 1).cpu()
    video_uint8 = (video * 255.0).round().to(torch.uint8)

    ride_label = rel_dir.name.replace("/", "_")
    out_name = f"{run_dir.name}_{ride_label}_step{checkpoint_path.parent.name.split('_')[-1]}_idx{args.output_index}.mp4"
    out_path = output_root / out_name
    write_video(str(out_path), video_uint8, fps=args.fps)
    if hasattr(pipeline.vae, "model"):
        pipeline.vae.model.clear_cache()
    print(f"[Info] Saved video to: {out_path}")


if __name__ == "__main__":
    main()
