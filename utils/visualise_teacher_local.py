#!/usr/bin/env python3
"""Simple script to load LoRA weights and generate a video from the first prompt.

Memory optimization note: If you encounter OOM errors, try setting this before running:
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
This helps reduce memory fragmentation.
"""

import csv
import json
import os
import sys
from pathlib import Path

# Set memory allocation config before importing torch to reduce fragmentation
if "PYTORCH_CUDA_ALLOC_CONF" not in os.environ:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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


def _configure_teacher_lora(transformer, teacher_lora_rank, teacher_adapter_name):
    """Configure LoRA for teacher model."""
    target_linear_modules = set()
    for name, module in transformer.named_modules():
        if module.__class__.__name__ in {"WanAttentionBlock", "CausalWanAttentionBlock"}:
            for full_submodule_name, submodule in module.named_modules(prefix=name):
                if isinstance(submodule, torch.nn.Linear):
                    target_linear_modules.add(full_submodule_name)
    
    target_linear_modules = list(target_linear_modules)
    if not target_linear_modules:
        raise RuntimeError("Failed to locate Linear modules for teacher LoRA.")
    
    peft_config = peft.LoraConfig(
        r=teacher_lora_rank,
        lora_alpha=teacher_lora_rank,
        lora_dropout=0.0,
        target_modules=target_linear_modules,
        inference_mode=True,
    )
    
    print(f"Teacher LoRA target modules: {len(target_linear_modules)} Linear layers (rank={teacher_lora_rank})")
    
    lora_model = peft.get_peft_model(transformer, peft_config, adapter_name=teacher_adapter_name)
    
    # Freeze adapter parameters
    for name, param in lora_model.named_parameters():
        if f".{teacher_adapter_name}." in name or name.endswith(f".{teacher_adapter_name}"):
            param.requires_grad_(False)
    
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
    
    # Freeze adapter parameters again
    for name, param in lora_model.named_parameters():
        if f".{teacher_adapter_name}." in name or name.endswith(f".{teacher_adapter_name}"):
            param.requires_grad_(False)
    
    print(f"Loaded pretrained teacher LoRA ({len(lora_state)} tensors) into adapter '{teacher_adapter_name}'.")


def _load_actions(actions_path: Path) -> tuple[list[int], list[list[float]]]:
    """Load actions from CSV file, returning (frame_ids, values) lists."""
    if not actions_path.exists():
        return [], []
    
    try:
        with actions_path.open("r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            if reader.fieldnames is None or "frame_id" not in reader.fieldnames:
                raise ValueError("Missing 'frame_id' column")
            value_keys = [k for k in reader.fieldnames if k != "frame_id"]
            if not value_keys:
                raise ValueError("No action columns found")

            frames, values = [], []
            for row in reader:
                frames.append(int(row["frame_id"]))
                values.append([float(row[k]) for k in value_keys])
    except Exception as exc:
        print(f"Warning: Failed to read actions from {actions_path}: {exc}")
        return [], []
    
    return frames, values


def main():
    # ============================================================================
    # Configuration: Toggle between pre-encoded and raw text prompts
    # ============================================================================
    USE_PRE_ENCODED_PROMPTS = False  # Set to False to use raw text prompts instead
    
    # ============================================================================
    # Ride / prompt index configuration
    # ============================================================================
    FILE_INDEX = 0  # Index of the ride/prompt directory to use (0 = first ride, 1 = second ride, etc.)
    
    # Prompt Extension: DISABLED
    # ============================================================================
    # Note: Prompt extension (prompt rewriting/expansion via PromptExpander) is explicitly
    # disabled in this script to preserve exact camera motion instructions. The prompts
    # are used as-is without any augmentation or rewriting that could dilute camera instructions.
    # If you need prompt extension, it would need to be explicitly enabled here.
    
    # ============================================================================
    # Negative prompt and guidance configuration
    # ============================================================================
    NEGATIVE_PROMPT = "dolly in, push forward, moving forward, tracking shot, zoom, steady forward camera motion"
    GUIDANCE_SCALE = 3.0  # Set to 0.0 to disable guidance (faster but less prompt adherence)
    
    # Hardcoded paths
    lora_weights_path = "/home/ashish/isambard_weights/retrain_no_gan_256/logs/diffusion_lora_step0018600.pt"
    prompts_root = Path("/home/ashish/ARRWM/prompts")
    actions_root = Path("/home/ashish/frodobots/frodobots_actions/test")
    output_path = Path("/home/ashish/ARRWM/output_teacher_video.mp4")
    
    # Set seed
    set_seed(0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16
    
    # Clear CUDA cache before starting
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"Initial GPU memory: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated")
    
    print("Initializing teacher model...")
    teacher_model = WanDiffusionWrapper(model_name="Wan2.1-T2V-1.3B", is_causal=False)
    teacher_model.model.requires_grad_(False)
    
    # Enable gradient checkpointing to save memory
    if hasattr(teacher_model.model, 'gradient_checkpointing'):
        teacher_model.model.gradient_checkpointing = True
        print("Enabled gradient checkpointing for memory efficiency")
    
    teacher_model = teacher_model.to(device)
    teacher_model.eval()
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"After loading teacher model: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated")
    
    # Load teacher LoRA
    teacher_lora_rank = 256
    teacher_adapter_name = "default"
    print("Configuring teacher LoRA...")
    teacher_model.model = _configure_teacher_lora(
        teacher_model.model, 
        teacher_lora_rank, 
        teacher_adapter_name
    )
    _load_teacher_lora(teacher_model.model, lora_weights_path, teacher_adapter_name)
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"After loading LoRA: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated")
    
    # Initialize VAE
    print("Initializing VAE...")
    vae = WanVAEWrapper(model_name="Wan2.1-T2V-1.3B").to(device).eval()
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"After loading VAE: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated")
    
    # Handle text encoder based on prompt type
    text_encoder_for_pipeline = None
    if USE_PRE_ENCODED_PROMPTS:
        # Don't load text encoder since we're using pre-encoded prompts (saves ~2-3GB)
        print("Using pre-encoded prompts - text encoder not loaded")
    else:
        # Load text encoder temporarily to encode the prompt, then free it
        print("Initializing text encoder for raw text prompts...")
        text_encoder = WanTextEncoder(model_name="Wan2.1-T2V-1.3B").to(device).eval()
        
        if device.type == "cuda":
            torch.cuda.empty_cache()
            print(f"After loading text encoder: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated")
    
    # Select ride/prompt directory based on FILE_INDEX
    ride_dirs = sorted(
        [d for d in prompts_root.iterdir() if d.is_dir() and d.name.startswith("ride_")]
    )
    if not ride_dirs:
        raise FileNotFoundError(f"No ride_* directories found in {prompts_root}")
    if FILE_INDEX < 0 or FILE_INDEX >= len(ride_dirs):
        raise IndexError(
            f"FILE_INDEX {FILE_INDEX} is out of range. Found {len(ride_dirs)} ride directories."
        )
    prompt_dir = ride_dirs[FILE_INDEX]
    print(f"Using prompt directory {prompt_dir} (index {FILE_INDEX} of {len(ride_dirs)})")

    # Create minimal config
    # Note: We always use text_pre_encoded=True in the pipeline since we encode beforehand
    # sampling_steps: Number of denoising steps for the teacher model (UniPC scheduler)
    # denoising_step_list: Timesteps for student model (not used by teacher model)
    config = OmegaConf.create({
        "text_pre_encoded": True,  # Always True since we encode before pipeline
        "mixed_precision": True,
        "num_training_frames": 21,
        "num_output_frames": 21,
        "denoising_step_list": [1000, 750, 500, 250],  # For student model (not used by teacher)
        "warp_denoising_step": True,
        "num_train_timestep": 1000,
        "sampling_steps": 5,  # Reduced from default 50 for faster generation (lower quality)
        "timestep_shift": 5.0,
        "guidance_scale": GUIDANCE_SCALE,  # Classifier-free guidance scale (0.0 = disabled)
        "negative_prompt": NEGATIVE_PROMPT,  # Negative prompt for guidance
        "model_kwargs": {
            "timestep_shift": 5.0,
            "local_attn_size": 12,
            "sink_size": 3,
        },
    })
    
    # Create pipeline (text_encoder is None since we've already encoded)
    print("Creating pipeline...")
    pipeline = BidirectionalInferencePipeline(
        args=config,
        device=device,
        generator=teacher_model,
        text_encoder=text_encoder_for_pipeline,  # None since we use pre-encoded prompts
        vae=vae,
    )
    
    # Extract ride ID from prompt directory name (e.g., "ride_17836_20240203103305" -> "17836")
    ride_name = prompt_dir.name
    ride_id = ride_name.split("_")[1] if "_" in ride_name else None
    
    # Try to find and load actions
    actions_frames = []
    actions_values = []
    if ride_id and actions_root.exists():
        # Search for actions file in output_rides_* directories
        for output_rides_dir in sorted(actions_root.glob("output_rides_*")):
            ride_dir = output_rides_dir / ride_name
            actions_file = ride_dir / f"input_actions_{ride_id}.csv"
            if actions_file.exists():
                print(f"Loading actions from {actions_file}")
                actions_frames, actions_values = _load_actions(actions_file)
                if actions_frames:
                    print(f"Loaded {len(actions_frames)} action frames")
                    # Display first few actions
                    print("First 10 actions:")
                    print("Frame ID | Linear  | Angular")
                    print("-" * 30)
                    for i in range(min(10, len(actions_frames))):
                        linear = actions_values[i][0] if len(actions_values[i]) > 0 else 0.0
                        angular = actions_values[i][1] if len(actions_values[i]) > 1 else 0.0
                        print(f"  {actions_frames[i]:6d} | {linear:7.3f} | {angular:7.3f}")
                    if len(actions_frames) > 10:
                        print(f"... and {len(actions_frames) - 10} more actions")
                    break
        else:
            print(f"Warning: No actions file found for ride {ride_name} in {actions_root}")
    
    # Load and encode prompt based on type
    negative_prompt_embeds = None  # Initialize for both paths
    if USE_PRE_ENCODED_PROMPTS:
        # Load pre-encoded prompt from JSON
        encoded_files = sorted(prompt_dir.glob("*_encoded.json"))
        if not encoded_files:
            raise FileNotFoundError(f"No *_encoded.json files found in {prompt_dir}")
        
        first_encoded_file = encoded_files[0]
        print(f"Loading pre-encoded prompt from {first_encoded_file}")
        
        with open(first_encoded_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        
        if "caption_encoded" not in payload:
            raise KeyError(f"Key 'caption_encoded' missing in {first_encoded_file}")
        
        prompt_embeds = torch.tensor(payload["caption_encoded"], dtype=torch.float32)
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        
        print(f"Prompt embeddings shape: {prompt_embeds.shape}")
    else:
        # Load raw text prompt from JSON and encode it
        text_files = sorted(prompt_dir.glob("*.json"))
        # Filter out encoded files
        text_files = [f for f in text_files if not f.name.endswith("_encoded.json")]
        if not text_files:
            raise FileNotFoundError(f"No non-encoded JSON files found in {prompt_dir}")
        
        first_text_file = text_files[0]
        print(f"Loading raw text prompt from {first_text_file}")
        
        with open(first_text_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        
        # Try to get text from various possible keys
        text_prompt = (
            payload.get("combined_analysis") or
            payload.get("caption") or
            payload.get("caption_text") or
            payload.get("text") or
            (payload.get("chunks", [{}])[0].get("chunk_caption") if payload.get("chunks") else None)
        )
        
        if not text_prompt:
            raise KeyError(f"No text prompt found in {first_text_file}. Tried: combined_analysis, caption, caption_text, text, chunks[0].chunk_caption")
        
        # Clean up the text (remove extra whitespace)
        text_prompt = " ".join(text_prompt.split())
        
        # Add motion sentence to the prompt
        motion_sentence = "Fixed-position tripod camera. Instant whip pan right at the start (clockwise yaw), with motion blur during the pan, then hold. Rotation only: no dolly, no forward movement, no tracking, no zoom."
        text_prompt = f"{motion_sentence} {text_prompt}"
        
        print(f"Text prompt (with motion): {text_prompt[:200]}..." if len(text_prompt) > 200 else f"Text prompt (with motion): {text_prompt}")
        
        # Encode the text prompt and negative prompt using text encoder
        print("Encoding text prompt...")
        with torch.no_grad():
            conditional_dict = text_encoder([text_prompt])
            prompt_embeds = conditional_dict.get("prompt_embeds")
            if prompt_embeds is None:
                raise ValueError("WanTextEncoder did not return 'prompt_embeds'")
            
            # Encode negative prompt if guidance is enabled
            if GUIDANCE_SCALE > 0.0 and NEGATIVE_PROMPT:
                # Combine negative prompt with the text prompt
                negative_prompt_text = f"{NEGATIVE_PROMPT}"
                print(f"Encoding negative prompt (guidance_scale={GUIDANCE_SCALE})...")
                print(f"Negative prompt: {negative_prompt_text[:200]}..." if len(negative_prompt_text) > 200 else f"Negative prompt: {negative_prompt_text}")
                negative_dict = text_encoder([negative_prompt_text])
                negative_prompt_embeds = negative_dict.get("prompt_embeds")
                if negative_prompt_embeds is None:
                    raise ValueError("WanTextEncoder did not return 'prompt_embeds' for negative prompt")
        
        if prompt_embeds.dim() == 2:
            prompt_embeds = prompt_embeds.unsqueeze(0)
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        print(f"Prompt embeddings shape: {prompt_embeds.shape}")
        
        if negative_prompt_embeds is not None:
            if negative_prompt_embeds.dim() == 2:
                negative_prompt_embeds = negative_prompt_embeds.unsqueeze(0)
            negative_prompt_embeds = negative_prompt_embeds.to(device=device, dtype=dtype)
            print(f"Negative prompt embeddings shape: {negative_prompt_embeds.shape}")
        
        # Free text encoder from GPU memory
        print("Freeing text encoder from GPU memory...")
        # Move to CPU first to ensure memory is freed
        text_encoder.text_encoder = text_encoder.text_encoder.cpu()
        del text_encoder
        if device.type == "cuda":
            torch.cuda.empty_cache()
            print(f"After freeing text encoder: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated")
    
    # Store negative prompt embeddings in pipeline if available (for guidance)
    # This needs to be done after encoding but before inference
    if negative_prompt_embeds is not None:
        pipeline._negative_prompt_embeds = negative_prompt_embeds
        print(f"Stored negative prompt embeddings in pipeline (guidance_scale={GUIDANCE_SCALE})")
    elif GUIDANCE_SCALE > 0.0:
        print(f"Warning: guidance_scale={GUIDANCE_SCALE} but no negative prompt provided. Guidance will be disabled.")
    
    # Final memory check before inference
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"After loading prompt: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated")
    
    # Generate video
    print("Generating video...")
    num_frames = 21
    latent_shape = (1, num_frames, 16, 60, 104)
    noise = torch.randn(latent_shape, device=device, dtype=dtype)
    
    if device.type == "cuda":
        torch.cuda.empty_cache()
        print(f"Before inference: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated")
        print(f"Before inference (reserved): {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB reserved")
    
    # Track peak memory during inference
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    
    with torch.no_grad():
        # Always use prompt_embeds since we encode beforehand
        generated_video, _ = pipeline.inference(
            noise=noise,
            prompt_embeds=prompt_embeds,
        )
    
    # Check peak memory usage
    if device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved(device) / 1024**3
        print(f"Peak memory during inference: {peak_allocated:.2f} GB allocated, {peak_reserved:.2f} GB reserved")
        torch.cuda.empty_cache()
        print(f"After inference: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB allocated")
    
    video = generated_video[0].permute(0, 2, 3, 1).cpu()
    
    # Delete large tensors to free memory
    del generated_video, noise
    if USE_PRE_ENCODED_PROMPTS and prompt_embeds is not None:
        del prompt_embeds
    if device.type == "cuda":
        torch.cuda.empty_cache()
    
    # Save video
    print(f"Saving video to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_uint8 = (video.clamp(0, 1) * 255.0).round().to(torch.uint8)
    write_video(str(output_path), video_uint8, fps=16)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    print(f"Done! Video saved to {output_path}")


if __name__ == "__main__":
    main()
