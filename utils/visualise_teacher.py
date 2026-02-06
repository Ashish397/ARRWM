#!/usr/bin/env python3
"""Generate teacher videos using the same loading as distillation path."""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import peft
from omegaconf import OmegaConf
from torchvision.io import write_video

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.wan_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.misc import set_seed
from pipeline.bidirectional_inference import BidirectionalInferencePipeline


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
    """Gather prompt entries from encoded caption JSON files."""
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


def _load_checkpoint_with_storage_fallback(path, **torch_load_kwargs):
    """Load checkpoint handling non-resizable storages."""
    try:
        return torch.load(path, **torch_load_kwargs)
    except RuntimeError as exc:
        if "not resizable" not in str(exc):
            raise
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


def _build_lora_config(adapter_cfg, target_modules, trainable):
    """Build LoRA config matching distillation path."""
    if adapter_cfg["type"] != "lora":
        raise NotImplementedError(f'Adapter type {adapter_cfg["type"]} is not implemented')
    
    return peft.LoraConfig(
        r=adapter_cfg["rank"],
        lora_alpha=adapter_cfg["alpha"],
        lora_dropout=adapter_cfg["dropout"],
        target_modules=target_modules,
        inference_mode=not trainable,
    )


def _freeze_adapter_params(lora_model, adapter_name):
    """Freeze adapter parameters matching distillation path."""
    for name, param in lora_model.named_parameters():
        if f".{adapter_name}." in name or name.endswith(f".{adapter_name}"):
            param.requires_grad_(False)


def _configure_teacher_lora(transformer, teacher_lora_rank, teacher_adapter_name):
    """Configure LoRA for teacher model - matches distillation path exactly."""
    target_linear_modules = set()
    for name, module in transformer.named_modules():
        if module.__class__.__name__ in {"WanAttentionBlock", "CausalWanAttentionBlock"}:
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, torch.nn.Linear):
                    target_linear_modules.add(full_submodule_name)
    
    target_linear_modules = list(target_linear_modules)
    if not target_linear_modules:
        raise RuntimeError("Failed to locate Linear modules for teacher LoRA.")
    
    adapter_cfg = {
        "type": "lora",
        "rank": teacher_lora_rank,
        "alpha": teacher_lora_rank,
        "dropout": 0.0,
        "adapter_name": teacher_adapter_name,
        "verbose": False,
    }
    
    print(f"Teacher LoRA target modules: {len(target_linear_modules)} Linear layers (rank={teacher_lora_rank})")
    
    peft_config = _build_lora_config(adapter_cfg, target_linear_modules, trainable=False)
    lora_model = peft.get_peft_model(transformer, peft_config, adapter_name=teacher_adapter_name)
    _freeze_adapter_params(lora_model, teacher_adapter_name)
    
    print("Configured teacher LoRA adapter; parameters frozen for inference.")
    
    return lora_model


def _load_teacher_lora(lora_model, teacher_lora_weights, teacher_adapter_name):
    """Load pretrained teacher LoRA weights."""
    weights_path = Path(teacher_lora_weights).expanduser()
    if not weights_path.exists():
        raise FileNotFoundError(f"teacher_lora_weights path does not exist: {weights_path}")
    print(f"Loading pretrained teacher LoRA weights from {weights_path}")
    checkpoint = _load_checkpoint_with_storage_fallback(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "lora" in checkpoint:
            lora_state = checkpoint["lora"]
        elif "teacher_lora" in checkpoint:
            lora_state = checkpoint["teacher_lora"]
        else:
            lora_state = checkpoint
    else:
        raise ValueError(f"Unexpected format for teacher LoRA checkpoint: {type(checkpoint)}")
    peft.set_peft_model_state_dict(
        lora_model,
        lora_state,
        adapter_name=teacher_adapter_name,
    )
    _freeze_adapter_params(lora_model, teacher_adapter_name)
    
    print(
        f"Loaded pretrained teacher LoRA ({len(lora_state)} tensors) "
        f"into adapter '{teacher_adapter_name}'."
    )




def main():
    parser = argparse.ArgumentParser(description="Generate teacher videos using the same loading as distillation path")
    parser.add_argument("--config-path", type=Path, required=True, help="Path to config yaml file")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing encoded caption JSON files")
    parser.add_argument("--output-dir", type=Path, required=True, help="Root directory to store videos")
    parser.add_argument("--fps", type=int, default=16, help="Frames per second for saved videos")
    parser.add_argument("--num-frames", type=int, help="Number of frames to sample (defaults to config.num_training_frames)")
    parser.add_argument("--seed", type=int, help="Optional random seed")
    parser.add_argument("--device", default=None, help="Computation device (e.g. cuda:0)")
    parser.add_argument("--encoded-suffix", default="_encoded", help="Suffix pattern for encoded caption JSON files")
    parser.add_argument("--embedding-key", default="caption_encoded", help="JSON key containing the encoded caption tensor")
    parser.add_argument("--max-samples", type=int, help="Limit the number of prompts to process")
    parser.add_argument("--skip-existing", action="store_true", help="Skip samples that already have an output video")
    
    args = parser.parse_args()

    # Load config
    default_cfg_path = REPO_ROOT / "configs" / "default_config.yaml"
    if default_cfg_path.is_file():
        base_cfg = OmegaConf.load(default_cfg_path)
    else:
        base_cfg = OmegaConf.create({})
    user_cfg = OmegaConf.load(args.config_path)
    config = OmegaConf.merge(base_cfg, user_cfg)
    config = OmegaConf.merge(config, OmegaConf.create({"text_pre_encoded": True}))
    
    # Set seed
    if args.seed is not None:
        set_seed(args.seed)
    else:
        set_seed(getattr(config, "seed", 0))
    
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = torch.bfloat16 if config.mixed_precision else torch.float32

    # Initialize teacher model (real_score) - matches distillation path exactly
    real_model_name = getattr(config, "real_name", "Wan2.1-T2V-1.3B")
    teacher_model = WanDiffusionWrapper(model_name=real_model_name, is_causal=False)
    teacher_model.model.requires_grad_(False)
    teacher_model = teacher_model.to(device)
    teacher_model.eval()

    # Load teacher LoRA if configured - matches distillation path exactly
    teacher_lora_rank = getattr(config, "teacher_lora", None)
    teacher_lora_weights = getattr(config, "teacher_lora_weights", None)

    if teacher_lora_rank is not None and teacher_lora_weights:
        teacher_adapter_name = "default"
        teacher_model.model = _configure_teacher_lora(
            teacher_model.model, 
            int(teacher_lora_rank), 
            teacher_adapter_name
        )
        _load_teacher_lora(teacher_model.model, teacher_lora_weights, teacher_adapter_name)

    # Initialize text encoder and VAE
    text_encoder = WanTextEncoder(model_name=real_model_name).to(device).eval()
    vae = WanVAEWrapper(model_name=real_model_name).to(device).eval()
    
    # Create pipeline - use text_pre_encoded=True since we're using encoded prompts
    # The config already has text_pre_encoded=True from the merge above
    pipeline = BidirectionalInferencePipeline(
        args=config,
        device=device,
        generator=teacher_model,
        text_encoder=text_encoder,
        vae=vae,
    )
    
    # Load prompt entries
    input_root = args.input_dir.expanduser().resolve()
    if not input_root.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_root}")
    
    entries = _gather_prompt_entries(input_root, args.encoded_suffix, args.embedding_key)
    if args.max_samples is not None:
        entries = entries[:max(0, args.max_samples)]
    
    # Generate videos
    output_root = args.output_dir.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    
    default_frames = int(getattr(config, "num_output_frames", getattr(config, "num_training_frames", 21)))
    target_frames = args.num_frames or default_frames
    latent_shape = (1, target_frames, 16, 60, 104)
    
    total_entries = len(entries)
    
    for idx, entry in enumerate(entries, start=1):
        print(f"[Info] ({idx}/{total_entries}) Generating {entry.encoded_path}")
        
        # Generate noise
        noise = torch.randn(latent_shape, device=device, dtype=dtype)
        
        # Get prompt embeddings
        prompt_embeds = entry.prompt_embeds
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        
        # Generate video
        with torch.no_grad():
            generated_video, _ = pipeline.inference(
                noise=noise,
                prompt_embeds=prompt_embeds,
            )
        
        video = generated_video[0].permute(0, 2, 3, 1).cpu()
        
        # Determine output path
        rel_dir = entry.rel_bucket_path.parent
        video_name = f"{entry.encoded_path.stem}_teacher.mp4"
        output_path = output_root / rel_dir / video_name
        
        if output_path.exists() and args.skip_existing:
            print(f"[Info] Skipping existing {output_path}")
            continue
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save video
        video_uint8 = (video.clamp(0, 1) * 255.0).round().to(torch.uint8)
        write_video(str(output_path), video_uint8, fps=args.fps)
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        print(f"[{idx}] Saved teacher video to {output_path}")


if __name__ == "__main__":
    main()

