#!/usr/bin/env python3
"""Visualise LoRA finetuned checkpoints or decode reference latents."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
from omegaconf import OmegaConf
from torchvision.io import write_video

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.bidirectional_inference import BidirectionalInferencePipeline
from utils.wan_wrapper import WanDiffusionWrapper, WanVAEWrapper

import peft


# DEFAULT_RETRAIN_LOG_ROOT = Path("/scratch/u5as/as1748.u5as/frodobots/retrain/logs")
DEFAULT_OUTPUT_ROOT = Path("/scratch/u5as/as1748.u5as/frodobots/retrain_visualisations")
DEFAULT_REFERENCE_ROOT = Path("/projects/u5as/frodobots_encoded")
DEFAULT_ENCODED_SUFFIX = "_encoded"
DEFAULT_EMBEDDING_KEY = "caption_encoded"


@dataclass(frozen=True)
class PromptEntry:
    encoded_path: Path
    prompt_embeds: torch.Tensor
    caption_text: Optional[str]
    rel_bucket_path: Path


def _relative_from_bucket(path: Path) -> Path:
    """Slice the path so it starts at the first output_rides_* component."""
    for idx, part in enumerate(path.parts):
        if part.startswith("output_rides_"):
            return Path(*path.parts[idx:])
    raise ValueError(f"Unable to determine output bucket for {path}")


_SPLIT_NAMES = {"train", "test", "val", "validation", "dev"}


def _swap_split_and_bucket(path: Path) -> Optional[Path]:
    parts = list(path.parts)
    bucket_idx = next((i for i, part in enumerate(parts) if part.startswith("output_rides_")), None)
    split_idx = next((i for i, part in enumerate(parts) if part.lower() in _SPLIT_NAMES), None)
    if bucket_idx is None or split_idx is None:
        return None
    if bucket_idx > split_idx or bucket_idx == split_idx:
        return None

    new_parts = list(parts)
    split_part = new_parts.pop(split_idx)
    bucket_part = new_parts.pop(bucket_idx)
    new_parts.insert(bucket_idx, split_part)
    new_parts.insert(bucket_idx + 1, bucket_part)
    return Path(*new_parts)


def _resolve_input_root(path: Path) -> tuple[Path, bool]:
    expanded = path.expanduser()
    try:
        resolved = expanded.resolve()
    except FileNotFoundError:
        resolved = expanded

    if resolved.is_dir():
        return resolved, False

    swapped = _swap_split_and_bucket(resolved)
    if swapped is not None and swapped.is_dir():
        return swapped, True

    return resolved, False





def _instantiate_generator(config) -> WanDiffusionWrapper:
    """Construct a WanDiffusionWrapper matching the LoRA training setup."""
    model_name = getattr(config, "model_name", None) or getattr(config, "real_name", "Wan2.1-T2V-1.3B")
    wrapper_kwargs = dict(getattr(config, "model_kwargs", {}))
    generator = WanDiffusionWrapper(model_name=model_name, is_causal=False, **wrapper_kwargs)
    if getattr(config, "gradient_checkpointing", False):
        generator.enable_gradient_checkpointing()
    return generator



def _resolve_checkpoint(
    checkpoint_path: Optional[Path],
    log_root: Path,
    checkpoint_step: Optional[int],
) -> Path:
    if checkpoint_path:
        checkpoint_path = checkpoint_path.expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return checkpoint_path

    log_root = log_root.expanduser().resolve()
    if not log_root.is_dir():
        raise FileNotFoundError(f"Log directory not found: {log_root}")

    if checkpoint_step is not None:
        candidate = log_root / f"diffusion_lora_step{checkpoint_step:07d}.pt"
        if not candidate.is_file():
            raise FileNotFoundError(f"Checkpoint for step {checkpoint_step} not found in {log_root}")
        return candidate

    checkpoints = sorted(log_root.glob("diffusion_lora_step*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No diffusion_lora_step*.pt files discovered in {log_root}")
    return checkpoints[-1]


def _matching_caption_path(encoded_path: Path, encoded_suffix: str) -> Optional[Path]:
    name = encoded_path.name
    suffix = f"{encoded_suffix}.json"
    if not name.endswith(suffix):
        return None
    base = name[: -len(suffix)]
    return encoded_path.with_name(f"{base}.json")


def _load_prompt_entry(
    encoded_path: Path,
    encoded_suffix: str,
    embedding_key: str,
) -> PromptEntry:
    with encoded_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if embedding_key not in payload:
        raise KeyError(f"Key '{embedding_key}' missing in {encoded_path}")
    embeds = torch.tensor(payload[embedding_key], dtype=torch.float32)

    caption_text = None
    caption_path = _matching_caption_path(encoded_path, encoded_suffix)
    if caption_path and caption_path.is_file():
        with caption_path.open("r", encoding="utf-8") as cfh:
            raw = json.load(cfh)
        caption_text = (
            raw.get("combined_analysis")
            or raw.get("caption")
            or raw.get("caption_text")
            or raw.get("text")
        )
        if caption_text:
            caption_text = caption_text.strip()

    rel_path = _relative_from_bucket(encoded_path)
    return PromptEntry(
        encoded_path=encoded_path,
        prompt_embeds=embeds,
        caption_text=caption_text,
        rel_bucket_path=rel_path,
    )


def _gather_prompt_entries(
    caption_root: Path,
    encoded_suffix: str,
    embedding_key: str,
) -> list[PromptEntry]:
    encoded_files = sorted(caption_root.rglob(f"*{encoded_suffix}.json"))
    if not encoded_files:
        raise RuntimeError(f"No *{encoded_suffix}.json files found under {caption_root}")

    entries: list[PromptEntry] = []
    for encoded_path in encoded_files:
        try:
            entries.append(_load_prompt_entry(encoded_path, encoded_suffix, embedding_key))
        except Exception as exc:  # noqa: BLE001
            print(f"[Warning] Failed to load {encoded_path}: {exc}", file=sys.stderr)
    if not entries:
        raise RuntimeError("Failed to load any prompt embeddings.")
    return entries


def _collect_target_modules(transformer: torch.nn.Module) -> list[str]:
    target: set[str] = set()
    for module_name, module in transformer.named_modules():
        cls = module.__class__.__name__
        if cls in {"WanAttentionBlock", "CausalWanAttentionBlock"}:
            for full_name, submodule in module.named_modules(prefix=module_name):
                if isinstance(submodule, torch.nn.Linear):
                    target.add(full_name)
    if not target:
        raise RuntimeError("No Linear modules discovered for LoRA injection.")
    return sorted(target)


def _apply_lora_to_generator(generator: torch.nn.Module, adapter_cfg) -> torch.nn.Module:
    target_modules = adapter_cfg.get("target_modules")
    if not target_modules:
        target_modules = _collect_target_modules(generator)

    rank = int(adapter_cfg.get("rank", 16))
    alpha = int(adapter_cfg.get("alpha", rank))
    dropout = float(adapter_cfg.get("dropout", 0.0))
    init_type = adapter_cfg.get("init_lora_weights", "gaussian")

    peft_config = peft.LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        init_lora_weights=init_type,
    )
    lora_model = peft.get_peft_model(generator, peft_config)
    lora_model.print_trainable_parameters()
    return lora_model


def _load_lora_weights(generator: torch.nn.Module, checkpoint_path: Path) -> int:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if "lora" not in checkpoint:
        raise KeyError(f"'lora' entry missing in checkpoint {checkpoint_path}")
    peft.set_peft_model_state_dict(generator, checkpoint["lora"])
    step = int(checkpoint.get("step", -1))
    return step


def _prepare_device(device_arg: Optional[str]) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _ensure_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dim() == 2:
        return tensor.unsqueeze(0)
    if tensor.dim() == 3:
        return tensor
    raise ValueError(f"Unexpected prompt embedding shape {tuple(tensor.shape)}")


def _write_video_tensor(video: torch.Tensor, output_path: Path, fps: int) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_uint8 = (video.clamp(0, 1) * 255.0).round().to(torch.uint8)
    write_video(str(output_path), video_uint8, fps=fps)


def _format_caption(entry: PromptEntry) -> str:
    if entry.caption_text:
        return " ".join(entry.caption_text.split())
    return entry.encoded_path.stem.replace("_", " ")


def _decode_reference_latents(
    entries: Iterable[PromptEntry],
    reference_root: Path,
    output_root: Path,
    fps: int,
    device: torch.device,
    overwrite: bool,
) -> None:
    reference_root = reference_root.expanduser().resolve()
    vae = WanVAEWrapper().to(device)
    vae.eval()

    for idx, entry in enumerate(entries, start=1):
        latent_dir = reference_root / entry.rel_bucket_path.parent
        latent_files = sorted(latent_dir.glob("encoded_video_*.pt"))
        if not latent_files:
            print(f"[Warning] No encoded_video_*.pt files for {latent_dir}", file=sys.stderr)
            continue

        latents_list = []
        for latent_file in latent_files:
            tensor = torch.load(latent_file, map_location="cpu")
            if tensor.dim() == 4:
                tensor = tensor.unsqueeze(0)
            if tensor.dim() != 5:
                raise ValueError(f"Unexpected latent shape {tuple(tensor.shape)} in {latent_file}")
            latents_list.append(tensor)
        latents = torch.cat(latents_list, dim=1)
        latents = latents.to(device=device, dtype=torch.float32)

        decoded = vae.decode_to_pixel(latents)
        video = decoded[0].permute(0, 2, 3, 1)  # -> [T, H, W, C]
        rel_dir = entry.rel_bucket_path.parent
        output_path = output_root / rel_dir / f"{entry.encoded_path.stem}_reference.mp4"
        if output_path.exists() and not overwrite:
            print(f"[Info] Skipping existing {output_path}")
            continue
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Decode output is in [-1, 1].
        video_uint8 = ((video.clamp(-1, 1) + 1.0) * 127.5).round().to(torch.uint8)
        write_video(str(output_path), video_uint8, fps=fps)
        print(f"[{idx}] Saved reference video to {output_path}")


def _generate_videos(
    entries: list[PromptEntry],
    config: OmegaConf,
    checkpoint_path: Path,
    output_root: Path,
    fps: int,
    num_frames: Optional[int],
    seed: Optional[int],
    device: torch.device,
    overwrite: bool,
) -> None:
    torch.set_grad_enabled(False)
    if seed is not None:
        torch.manual_seed(seed)

    text_pre_encoded = bool(getattr(config, "text_pre_encoded", False))

    generator = _instantiate_generator(config)
    adapter_cfg = OmegaConf.to_container(config.adapter, resolve=True)
    if not isinstance(adapter_cfg, dict):
        raise TypeError("config.adapter must resolve to a mapping for LoRA configuration.")
    generator.model = _apply_lora_to_generator(generator.model, adapter_cfg)
    step = _load_lora_weights(generator.model, checkpoint_path)

    generator.to(device=device)
    generator.eval()

    pipeline = BidirectionalInferencePipeline(config, device=device, generator=generator)

    default_frames = int(getattr(config, "num_output_frames", getattr(config, "num_training_frames", 21)))
    target_frames = num_frames or default_frames
    latent_shape = (1, target_frames, 16, 60, 104)

    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    total_entries = len(entries)

    for idx, entry in enumerate(entries, start=1):
        print(f"[Info] ({idx}/{total_entries}) Generating {entry.encoded_path}")
        noise = torch.randn(latent_shape, device=device, dtype=torch.float32)

        if text_pre_encoded:
            prompt_embeds = _ensure_batch_dim(entry.prompt_embeds).to(device=device, dtype=torch.float32)
            generated_video, _ = pipeline.inference(
                noise=noise,
                prompt_embeds=prompt_embeds,
            )
            prompt_text = entry.caption_text or _format_caption(entry)
        else:
            prompt_text = entry.caption_text or _format_caption(entry)
            generated_video, _ = pipeline.inference(
                noise=noise,
                text_prompts=[prompt_text],
            )

        video = generated_video[0].permute(0, 2, 3, 1).cpu()

        rel_dir = entry.rel_bucket_path.parent
        checkpoint_suffix = checkpoint_path.stem
        video_name = f"{entry.encoded_path.stem}_{checkpoint_suffix}.mp4"
        output_path = output_root / rel_dir / video_name
        prompt_path = output_path.with_suffix(".txt")

        if output_path.exists() and not overwrite:
            print(f"[Info] Skipping existing {output_path}")
            continue

        _write_video_tensor(video, output_path, fps=fps)
        if device.type == "cuda":
            torch.cuda.synchronize()

        prompt_path.parent.mkdir(parents=True, exist_ok=True)
        with prompt_path.open("w", encoding="utf-8") as fh:
            fh.write((prompt_text or "") + "\n")

        tag = f"step {step}" if step >= 0 else checkpoint_suffix
        print(f"[{idx}] Saved generated video ({tag}) to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualise LoRA retrained checkpoints or decode reference latents.")
    parser.add_argument("--config-path", type=Path, help="YAML config used for training (required unless --reference).")
    parser.add_argument("--checkpoint", type=Path, help="Path to a diffusion_lora_step*.pt checkpoint.")
    parser.add_argument("--checkpoint-step", type=int, help="Select a checkpoint by step number from --logs-root.")
    parser.add_argument("--logs-root", type=Path, default='None', help="Directory containing retrain checkpoints.")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing encoded caption JSON files.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root directory to store videos.")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second for saved videos.")
    parser.add_argument("--num-frames", type=int, help="Number of frames to sample (defaults to config.num_training_frames).")
    parser.add_argument("--seed", type=int, help="Optional random seed.")
    parser.add_argument("--device", default=None, help="Computation device (e.g. cuda:0).")
    parser.add_argument("--reference", action="store_true", help="Decode reference latents instead of generating with diffusion.")
    parser.add_argument("--reference-root", type=Path, default=DEFAULT_REFERENCE_ROOT, help="Root for pre-encoded video latents.")
    parser.add_argument("--encoded-suffix", default=DEFAULT_ENCODED_SUFFIX, help="Suffix pattern for encoded caption JSON files.")
    parser.add_argument("--embedding-key", default=DEFAULT_EMBEDDING_KEY, help="JSON key containing the encoded caption tensor.")
    parser.add_argument("--max-samples", type=int, help="Limit the number of prompts to process.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip samples that already have an output video.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.reference and args.config_path is None:
        raise SystemExit("--config-path is required unless --reference is specified.")

    input_root, corrected_input = _resolve_input_root(args.input_dir)
    if corrected_input:
        print(f"[Info] Adjusted input directory to: {input_root}")
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_root}")
    input_root = input_root.resolve()

    entries = _gather_prompt_entries(input_root, args.encoded_suffix, args.embedding_key)
    if args.max_samples is not None:
        entries = entries[: max(0, args.max_samples)]

    device = _prepare_device(args.device)
    output_root = args.output_dir.expanduser().resolve()

    if args.reference:
        _decode_reference_latents(
            entries=entries,
            reference_root=args.reference_root,
            output_root=output_root,
            fps=args.fps,
            device=device,
            overwrite=not args.skip_existing,
        )
        return

    default_cfg_path = REPO_ROOT / "configs" / "default_config.yaml"
    if default_cfg_path.is_file():
        base_cfg = OmegaConf.load(default_cfg_path)
    else:
        base_cfg = OmegaConf.create({})
    user_cfg = OmegaConf.load(args.config_path)
    config = OmegaConf.merge(base_cfg, user_cfg)
    config = OmegaConf.merge(config, OmegaConf.create({"text_pre_encoded": True}))

    if not hasattr(config, "adapter"):
        raise AttributeError("The provided config does not define an adapter section required for LoRA inference.")

    checkpoint_path = _resolve_checkpoint(args.checkpoint, args.logs_root, args.checkpoint_step)
    print(f"[Info] Using checkpoint: {checkpoint_path}")

    _generate_videos(
        entries=entries,
        config=config,
        checkpoint_path=checkpoint_path,
        output_root=output_root,
        fps=args.fps,
        num_frames=args.num_frames,
        seed=args.seed,
        device=device,
        overwrite=not args.skip_existing,
    )


if __name__ == "__main__":
    main()
