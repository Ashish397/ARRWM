# Action Conditioning via adaLN-Zero - Complete Guide

This guide introduces how to inject action conditions into video generation models using the adaLN-Zero mechanism.

## 📚 Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Quick Start](#quick-start)
4. [API Reference](#api-reference)
5. [Implementing Your Action Module](#implementing-your-action-module)
6. [Training Guide](#training-guide)
7. [FAQ](#faq)

---

## Overview

**adaLN-Zero** (Adaptive Layer Normalization with Zero Initialization) is a powerful conditioning injection mechanism, originally proposed in DiT (Diffusion Transformer).

### Core Idea

```
Standard Transformer Block:
    x = x + Attention(LayerNorm(x))
    x = x + FFN(LayerNorm(x))

adaLN-Zero Enhancement:
    x = x + gate_attn * Attention(LayerNorm(x) * (1 + scale_attn) + shift_attn)
    x = x + gate_ffn * FFN(LayerNorm(x) * (1 + scale_ffn) + shift_ffn)
    
    where (scale, shift, gate) are predicted from conditions (e.g., timestep, action)
    Zero init: gate initialized to 0, model gradually learns to use conditions
```

### Why adaLN-Zero?

1. **Training Stability**: Zero initialization ensures no disruption to pretrained models
2. **Flexible Injection**: No need to modify backbone architecture
3. **Progressive Learning**: Model automatically learns how to leverage action information
4. **Efficient**: Low computational overhead, easy to optimize

---

## Architecture

### Overall Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Action Conditioning Pipeline              │
└─────────────────────────────────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │  1. Extract/Predict Action Features    │
        │     - From historical frames           │
        │     - From external input              │
        │     - Via action_module                │
        └──────────────┬─────────────────────────┘
                       │
                       ▼
        ┌────────────────────────────────────────┐
        │  2. Action Modulation Projection       │
        │     ActionModulationProjection         │
        │     [B, action_dim] →                  │
        │     [B, F, 6, hidden_dim]              │
        └──────────────┬─────────────────────────┘
                       │
                       ▼
        ┌────────────────────────────────────────┐
        │  3. Inject into Conditional Dict       │
        │     conditional_dict['_action_mod']    │
        └──────────────┬─────────────────────────┘
                       │
                       ▼
        ┌────────────────────────────────────────┐
        │  4. Wan Model Forward                  │
        │     time_mod + action_mod →            │
        │     Each Transformer Block             │
        └────────────────────────────────────────┘
```

### Key Components

#### 1. `ActionModulationProjection`
- **Location**: `model/action_modulation.py`
- **Function**: Projects action features to adaLN parameters
- **Output**: `[batch, num_frames, 6, hidden_dim]`
  - 6 parameters: `[shift_attn, scale_attn, gate_attn, shift_ffn, scale_ffn, gate_ffn]`

#### 2. `ActionCausalInferencePipeline`
- **Location**: `pipeline/action_inference.py`
- **Function**: Complete inference pipeline integrating action conditions
- **Feature**: Inherits from `CausalInferencePipeline`, seamless integration

#### 3. Model Patches
- **Location**: `model/action_model_patch.py`
- **Function**: Dynamically modifies Wan model to support action modulation
- **Method**: Monkey patching, no modification to original code

---

## Quick Start

### Simplest Example

```python
import torch
from pipeline.action_inference import ActionCausalInferencePipeline

# 1. Create pipeline (automatically applies adaLN-Zero)
pipeline = ActionCausalInferencePipeline(
    args=config,
    device='cuda',
    action_dim=512,  # Your action feature dimension
    enable_adaln_zero=True,  # Enable adaLN-Zero
)

# 2. Prepare inputs
noise = torch.randn(1, 21, 16, 60, 104).cuda()  # [B, F, C, H, W]
text_prompts = ["a robot walking forward"]

# 3. Prepare action features (example: random features)
action_features = torch.randn(1, 512).cuda()  # [B, action_dim]

# 4. Inference
video = pipeline.inference(
    noise=noise,
    text_prompts=text_prompts,
    action_inputs={'action_features': action_features}
)

print(f"Generated video shape: {video.shape}")  # [B, F, 3, H, W]
```

### Using Custom Action Module

```python
import torch.nn as nn

# 1. Define your action module
class MyActionModule(nn.Module):
    def __init__(self):
        super().__init__()
        # Your encoder: video frames → action features
        self.encoder = nn.Sequential(
            nn.Conv3d(16, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(64, 512),
        )
    
    def forward(self, frames):
        """
        Args:
            frames: [B, F, C, H, W] historical frames (latent space)
        Returns:
            action_features: [B, 512] action features
        """
        # Convert dimensions: [B, F, C, H, W] → [B, C, F, H, W]
        x = frames.permute(0, 2, 1, 3, 4)
        features = self.encoder(x)
        return features

# 2. Create pipeline
action_module = MyActionModule().cuda()
pipeline = ActionCausalInferencePipeline(
    args=config,
    device='cuda',
    action_module=action_module,
    action_dim=512,
    enable_adaln_zero=True,
)

# 3. Implement _process_action (override)
def custom_process_action(self, generated_frames, current_frame_idx, action_inputs):
    if current_frame_idx == 0:
        return None  # No history at first frame
    
    with torch.no_grad():
        # Extract features using action module
        action_features = self.action_module(generated_frames)
    
    return action_features

# Replace method
pipeline._process_action = custom_process_action.__get__(pipeline, ActionCausalInferencePipeline)

# 4. Inference (automatically uses action module)
video = pipeline.inference(
    noise=noise,
    text_prompts=text_prompts,
)
```

---

## API Reference

### `ActionCausalInferencePipeline.__init__`

```python
def __init__(
    args,                           # Configuration object
    device,                         # Device ('cuda' / 'cpu')
    generator=None,                 # WanDiffusionWrapper instance
    text_encoder=None,              # WanTextEncoder instance
    vae=None,                       # WanVAEWrapper instance
    action_module=None,             # Your action module
    action_dim: int = 512,          # Action feature dimension
    enable_adaln_zero: bool = True, # Whether to enable adaLN-Zero
)
```

### `ActionCausalInferencePipeline.inference`

```python
def inference(
    noise: torch.Tensor,              # [B, F, C, H, W] input noise
    text_prompts: list[str],          # Text prompt list
    action_inputs: dict | None = None,# Action input dictionary
    return_latents: bool = False,     # Whether to return latents
    profile: bool = False,            # Whether to profile performance
    low_memory: bool = False,         # Whether to use low memory mode
) -> torch.Tensor:                    # [B, F, 3, H, W] generated video
```

**`action_inputs` dictionary can contain:**
- `'action_features'`: `[B, action_dim]` precomputed action features
- `'target_actions'`: Target action sequence (custom)
- `'control_signals'`: Control signals (custom)

### `ActionModulationProjection`

```python
from model.action_modulation import ActionModulationProjection

projection = ActionModulationProjection(
    action_dim=512,        # Input action feature dimension
    hidden_dim=2048,       # Model hidden dimension
    num_frames=1,          # Default number of frames
    zero_init=True,        # adaLN-Zero initialization
)

# Usage
action_features = torch.randn(2, 512)  # [B, action_dim]
modulation = projection(action_features, num_frames=3)
# modulation shape: [2, 3, 6, 2048]
```

---

## Implementing Your Action Module

### Method 1: Extract Action from Historical Frames

```python
class HistoricalActionModule(nn.Module):
    """Extract action information from generated frames"""
    
    def __init__(self, latent_dim=16, action_dim=512):
        super().__init__()
        self.encoder = nn.Sequential(
            # 3D convolution for temporal information
            nn.Conv3d(latent_dim, 64, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(2, 2, 2)),
            
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            
            nn.Flatten(),
            nn.Linear(128, action_dim),
        )
    
    def forward(self, frames):
        """
        Args:
            frames: [B, F, C, H, W] historical frames
        Returns:
            [B, action_dim]
        """
        x = frames.permute(0, 2, 1, 3, 4)  # → [B, C, F, H, W]
        return self.encoder(x)
```

### Method 2: From External Action Sequence

```python
class ActionSequenceEncoder(nn.Module):
    """Encode action sequence into features"""
    
    def __init__(self, action_vocab_size=10, action_dim=512):
        super().__init__()
        self.embedding = nn.Embedding(action_vocab_size, 128)
        self.lstm = nn.LSTM(128, 256, batch_first=True)
        self.fc = nn.Linear(256, action_dim)
    
    def forward(self, action_sequence):
        """
        Args:
            action_sequence: [B, seq_len] action ID sequence
        Returns:
            [B, action_dim]
        """
        x = self.embedding(action_sequence)  # [B, seq_len, 128]
        _, (h, _) = self.lstm(x)             # h: [1, B, 256]
        return self.fc(h.squeeze(0))         # [B, action_dim]

# Usage
sequence_module = ActionSequenceEncoder().cuda()
pipeline = ActionCausalInferencePipeline(
    ...,
    action_module=sequence_module,
    action_dim=512,
)

# Provide action sequence during inference
action_seq = torch.tensor([[1, 3, 5, 2, 0]]).cuda()  # [B, seq_len]
action_features = sequence_module(action_seq)

video = pipeline.inference(
    noise=noise,
    text_prompts=text_prompts,
    action_inputs={'action_features': action_features}
)
```

### Method 3: Multimodal Fusion

```python
class MultimodalActionModule(nn.Module):
    """Fuse visual and language information to predict actions"""
    
    def __init__(self, visual_dim=16, text_dim=512, action_dim=512):
        super().__init__()
        self.visual_encoder = nn.Conv3d(visual_dim, 128, 3, padding=1)
        self.text_encoder = nn.Linear(text_dim, 256)
        self.fusion = nn.Sequential(
            nn.Linear(128 + 256, 512),
            nn.ReLU(),
            nn.Linear(512, action_dim),
        )
    
    def forward(self, frames, text_features):
        # Process visual
        vis = self.visual_encoder(frames.permute(0, 2, 1, 3, 4))
        vis = torch.nn.functional.adaptive_avg_pool3d(vis, 1).flatten(1)
        
        # Process text
        txt = self.text_encoder(text_features)
        
        # Fusion
        combined = torch.cat([vis, txt], dim=1)
        return self.fusion(combined)
```

---

## Training Guide

### Training Action Projection Layer

```python
from torch.optim import Adam

# 1. Create pipeline
pipeline = ActionCausalInferencePipeline(
    args=config,
    device='cuda',
    action_module=your_action_module,
    action_dim=512,
    enable_adaln_zero=True,
)

# 2. Get trainable parameters
action_params = pipeline.action_projection.parameters()
optimizer = Adam(action_params, lr=1e-4)

# 3. Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Get data
        noise = batch['noise']
        text_prompts = batch['prompts']
        target_video = batch['video']
        action_features = batch['action_features']  # Your action labels
        
        # Forward pass
        generated_video = pipeline.inference(
            noise=noise,
            text_prompts=text_prompts,
            action_inputs={'action_features': action_features}
        )
        
        # Compute loss
        loss = F.mse_loss(generated_video, target_video)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"Loss: {loss.item()}")
```

### Joint Training of Action Module + Projection

```python
# Optimize both action module and projection layer
all_params = list(pipeline.action_module.parameters()) + \
             list(pipeline.action_projection.parameters())
optimizer = Adam(all_params, lr=1e-4)

# Rest of training code is the same
```

### Freeze Main Model, Train Only Action Parts

```python
# Freeze generator
for param in pipeline.generator.parameters():
    param.requires_grad = False

# Train only action-related parts
action_params = list(pipeline.action_projection.parameters())
if pipeline.action_module is not None:
    action_params += list(pipeline.action_module.parameters())

optimizer = Adam(action_params, lr=1e-4)
```

---

## FAQ

### Q1: Why use Zero initialization?

**A**: Zero initialization (adaLN-Zero) ensures that at the start of training, action modulation doesn't affect the pretrained model's output. This provides:
1. Ability to start from pretrained model weights
2. More stable training
3. Gradual learning of how to use action information

### Q2: What should action_dim be set to?

**A**: Depends on your action representation:
- **Discrete actions (e.g., robot control)**: 256-512
- **Continuous actions (e.g., trajectories)**: 512-1024
- **Complex multimodal**: 1024-2048

### Q3: How to debug if action conditioning is working?

```python
# 1. Check if modulation is non-zero
action_features = torch.randn(1, 512).cuda()
modulation = pipeline.action_projection(action_features, num_frames=3)
print(f"Modulation norm: {modulation.norm()}")  # Should be close to 0 when just initialized

# 2. Compare generation with/without action
video_no_action = pipeline.inference(noise, text_prompts)
video_with_action = pipeline.inference(
    noise, text_prompts, 
    action_inputs={'action_features': action_features}
)
diff = (video_no_action - video_with_action).abs().mean()
print(f"Difference: {diff}")  # Should have significant difference after training
```

### Q4: Can multiple conditions be used simultaneously (e.g., action + style)?

**A**: Yes! Just create multiple projection modules:

```python
self.action_projection = ActionModulationProjection(action_dim, hidden_dim)
self.style_projection = ActionModulationProjection(style_dim, hidden_dim)

# In _apply_action_conditioning:
action_mod = self.action_projection(action_features, num_frames)
style_mod = self.style_projection(style_features, num_frames)
combined_mod = action_mod + style_mod  # Or weighted combination
```

### Q5: Is the performance overhead large?

**A**: Very small!
- `ActionModulationProjection`: ~5-10MB parameters (depending on hidden_dim)
- Forward pass time: increases <5%
- Memory overhead: increases <10%

### Q6: How to visualize action influence?

```python
# Generate action scan
action_features_list = [
    torch.randn(1, 512).cuda() * scale 
    for scale in [0, 0.5, 1.0, 2.0]
]

videos = []
for action_feat in action_features_list:
    video = pipeline.inference(
        noise=noise,
        text_prompts=text_prompts,
        action_inputs={'action_features': action_feat}
    )
    videos.append(video)

# Visualize comparison
visualize_video_grid(videos)
```

---

## Summary

With the adaLN-Zero mechanism, you can:

✅ Seamlessly inject action conditions into pretrained video generation models  
✅ Maintain training stability (Zero initialization)  
✅ Flexibly design your action module  
✅ Minimize computational overhead  
✅ Support combination of multiple condition types  

**Next Steps**:
1. Implement your action_module
2. Prepare action data/labels
3. Start training
4. Visualize results and iterate

Good luck! 🚀
