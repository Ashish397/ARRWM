# adaLN-Zero Action Conditioning - Quick Start

## 📁 Modified Files List

Created/modified files:

```
ARRWM/
├── model/
│   ├── action_modulation.py          # ✨ New: Action modulation projection layer
│   └── action_model_patch.py         # ✨ New: Model patches (dynamic injection)
├── pipeline/
│   └── action_inference.py           # ✅ Updated: Action-conditioned pipeline
├── utils/
│   └── action_wan_wrapper.py         # ✨ New: Action-aware wrapper (backup)
├── docs/
│   ├── action_conditioning_guide.md  # ✨ New: Complete documentation
│   └── ADALN_ZERO_QUICK_START.md     # ✨ New: Quick guide (this file)
└── test_action_pipeline.py           # ✨ New: Test script
```

## 🚀 60 Second Quick Start

### 1. Test Installation

```bash
cd /lus/lfs1aip2/home/u5as/tiankuo.u5as/ARRWM
python test_action_pipeline.py --mode simple
```

### 2. Minimal Usage

```python
from pipeline.action_inference import ActionCausalInferencePipeline
import torch

# Create pipeline (automatically enables adaLN-Zero)
pipeline = ActionCausalInferencePipeline(
    args=config,
    device='cuda',
    action_dim=512,
    enable_adaln_zero=True,
)

# Inference
action_features = torch.randn(1, 512).cuda()  # Your action features
video = pipeline.inference(
    noise=torch.randn(1, 21, 16, 60, 104).cuda(),
    text_prompts=["a robot walking"],
    action_inputs={'action_features': action_features}
)
```

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────┐
│                  Your Code                            │
│  ┌────────────────────────────────────────────┐     │
│  │  action_features = your_action_module(...) │     │
│  └────────────┬───────────────────────────────┘     │
└───────────────┼──────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────┐
│         ActionCausalInferencePipeline                │
│  ┌────────────────────────────────────────────┐     │
│  │  ActionModulationProjection                │     │
│  │  [B, action_dim] → [B, F, 6, hidden_dim]   │     │
│  └────────────┬───────────────────────────────┘     │
└───────────────┼──────────────────────────────────────┘
                ▼
         conditional_dict['_action_modulation'] = ...
                │
┌───────────────┼──────────────────────────────────────┐
│               ▼                                       │
│     WanDiffusionWrapper (patched)                    │
│       ├─→ Extract _action_modulation                 │
│       └─→ Pass to CausalWanModel                     │
└───────────────┼──────────────────────────────────────┘
                ▼
┌──────────────────────────────────────────────────────┐
│     CausalWanModel (patched)                         │
│       time_modulation + action_modulation            │
│                   │                                   │
│       ┌───────────┼───────────────┐                  │
│       ▼           ▼               ▼                  │
│   Block 0     Block 1   ...   Block N               │
│   (adaLN)     (adaLN)         (adaLN)               │
└──────────────────────────────────────────────────────┘
```

## 🎯 Three Usage Scenarios

### Scenario 1: Precomputed Action Features

```python
# Use case: Existing action labels, features extracted from other models
action_features = torch.load('action_features.pt')  # [B, 512]

video = pipeline.inference(
    noise=noise,
    text_prompts=prompts,
    action_inputs={'action_features': action_features}
)
```

### Scenario 2: Extract Action from Historical Frames

```python
# Use case: Autoregressive generation, predict action from history
import torch.nn as nn

class MyActionExtractor(nn.Module):
    def forward(self, historical_frames):
        # historical_frames: [B, F, C, H, W]
        # Your logic...
        return action_features  # [B, 512]

pipeline = ActionCausalInferencePipeline(
    args=config,
    device='cuda',
    action_module=MyActionExtractor(),
    action_dim=512,
)

# inference will automatically call action_module
video = pipeline.inference(noise, text_prompts)
```

### Scenario 3: Custom Action Processing

```python
# Override _process_action method
def custom_action_processing(self, generated_frames, current_frame_idx, action_inputs):
    # Your custom logic
    if current_frame_idx == 0:
        return None
    
    # Example: Use external data
    frame_actions = action_inputs['action_sequence'][current_frame_idx]
    return self.action_encoder(frame_actions)

pipeline._process_action = custom_action_processing.__get__(
    pipeline, ActionCausalInferencePipeline
)
```

## 🔧 Integration into Existing Training Code

### Modify `trainer/distillation.py`

At the `vis_pipeline` initialization location (around line 1499):

```python
# Original code
if 'switch' in self.config.distribution_loss:
    self.vis_pipeline = SwitchCausalInferencePipeline(...)
else:
    self.vis_pipeline = CausalInferencePipeline(...)

# Add action support
if 'action' in self.config.distribution_loss:
    from pipeline.action_inference import ActionCausalInferencePipeline
    
    # Optional: Create your action module
    action_module = YourActionModule().to(self.device)
    
    self.vis_pipeline = ActionCausalInferencePipeline(
        args=self.config,
        device=self.device,
        generator=self.model.generator,
        text_encoder=self.model.text_encoder,
        vae=self.model.vae,
        action_module=action_module,
        action_dim=512,  # Adjust to your needs
        enable_adaln_zero=True,
    )
elif 'switch' in self.config.distribution_loss:
    self.vis_pipeline = SwitchCausalInferencePipeline(...)
else:
    self.vis_pipeline = CausalInferencePipeline(...)
```

### Modify Configuration File

Create new config `configs/longlive_train_action.yaml`:

```yaml
# Copy content from longlive_train_init_real.yaml
# Then modify:
distribution_loss: dmd2real_action  # Add 'action' keyword

# Add action-related config
action_config:
  action_dim: 512
  enable_adaln_zero: true
  action_modulation_scale: 1.0
```

## 📊 Verify Action Conditioning is Working

### Method 1: Check Modulation Output

```python
# Before training (Zero initialization)
action_feat = torch.randn(1, 512).cuda()
mod = pipeline.action_projection(action_feat, num_frames=3)
print(f"Initial modulation norm: {mod.norm():.6f}")  # Should be ≈ 0

# After training
print(f"Trained modulation norm: {mod.norm():.6f}")  # Should be > 0
```

### Method 2: Compare Generation Results

```python
# Generate video without action
video_baseline = pipeline.inference(noise, prompts)

# Generate video with action
video_action = pipeline.inference(
    noise, prompts, 
    action_inputs={'action_features': action_feat}
)

# Calculate difference
diff = (video_baseline - video_action).abs().mean()
print(f"Video difference: {diff:.6f}")  # Should have significant difference after training
```

### Method 3: Visualize Action Scan

```python
import matplotlib.pyplot as plt

action_scales = [0, 0.5, 1.0, 2.0]
videos = []

for scale in action_scales:
    action_feat = torch.randn(1, 512).cuda() * scale
    video = pipeline.inference(
        noise, prompts,
        action_inputs={'action_features': action_feat}
    )
    videos.append(video)

# Visualize first frame
fig, axes = plt.subplots(1, len(action_scales), figsize=(16, 4))
for idx, (video, scale) in enumerate(zip(videos, action_scales)):
    axes[idx].imshow(video[0, 0].permute(1, 2, 0).cpu())
    axes[idx].set_title(f'Action Scale: {scale}')
plt.savefig('action_scan.png')
```

## 🐛 Troubleshooting

### Issue 1: Import Error

```bash
ImportError: cannot import name 'ActionModulationProjection'
```

**Solution**: Ensure you run from project root, or add to PYTHONPATH:
```bash
export PYTHONPATH=/lus/lfs1aip2/home/u5as/tiankuo.u5as/ARRWM:$PYTHONPATH
```

### Issue 2: Action Modulation Not Working

**Checklist**:
1. ✅ `enable_adaln_zero=True`
2. ✅ `action_inputs` correctly passed
3. ✅ `_action_modulation` in `conditional_dict`
4. ✅ Model patches applied (check logs)

```python
# Debug code
print("Pipeline has action_projection:", hasattr(pipeline, 'action_projection'))
print("Action projection params:", sum(p.numel() for p in pipeline.action_projection.parameters()))
```

### Issue 3: Training Instability

**Possible causes**:
- Action modulation scale too large
- Learning rate not appropriate

**Solutions**:
```python
# Reduce action modulation influence
pipeline = ActionCausalInferencePipeline(
    ...,
    # Add scale parameter (needs implementation in code)
)

# Or manually scale in _apply_action_conditioning
action_modulation = action_modulation * 0.1  # Reduce by 10x
```

## 📚 Further Reading

- **Complete Documentation**: `docs/action_conditioning_guide.md`
- **DiT Paper**: [Scalable Diffusion Models with Transformers](https://arxiv.org/abs/2212.09748)
- **adaLN-Zero Principle**: See paper Section 3.3

## 🎓 Learning Path

1. **Day 1**: Run `test_action_pipeline.py`, understand basic flow
2. **Day 2**: Read `docs/action_conditioning_guide.md`
3. **Day 3**: Implement simple action module
4. **Day 4**: Integrate into training code and start experiments
5. **Day 5+**: Iterate and optimize action representation

## 💡 Tips

1. **Start Simple**: Test entire flow with random action features first
2. **Visualize**: Frequently visualize generation results to observe action influence
3. **Gradually Increase Complexity**: Start by freezing main model, train only action parts
4. **Save Intermediate Results**: Save action features and modulation for debugging
5. **Monitor Gradients**: Ensure action_projection gradients are non-zero

## 🤝 Need Help?

- Check test script: `test_action_pipeline.py`
- Read complete documentation: `docs/action_conditioning_guide.md`
- Check example code: Code examples in documentation are all runnable

---

**Good luck! 🚀**

If you encounter issues, please check:
1. Log output (search for `[ActionPipeline]` or `[ActionPatch]`)
2. Model parameters correctly loaded
3. Device matching (CPU vs CUDA)
