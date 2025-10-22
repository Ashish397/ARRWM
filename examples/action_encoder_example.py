#!/usr/bin/env python3
"""
Example: Using ActionEncoder with ActionCausalInferencePipeline

This example shows how to:
1. Convert raw action values (e.g., [velocity, steering]) to action features
2. Use them with the action conditioning pipeline
"""

import sys
import os
# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import torch
from model.action_encoder import ActionEncoder, create_action_encoder
from pipeline.action_inference import ActionCausalInferencePipeline


def example_continuous_actions():
    """Example: Continuous actions (e.g., robot control)"""
    print("\n" + "="*70)
    print("Example 1: Continuous Actions (e.g., [velocity, steering])")
    print("="*70)
    
    # 1. Create action encoder
    action_encoder = ActionEncoder(
        action_dim=2,           # Your actions: [velocity, steering]
        feature_dim=512,        # Must match pipeline's action_dim
        use_sinusoidal=True,    # Use sinusoidal encoding (like timestep)
    )
    
    # 2. Your raw action values (what you have)
    raw_actions = torch.tensor([
        [0.5, 0.3],   # Sample 1: velocity=0.5, steering=0.3
        [0.8, -0.2],  # Sample 2: velocity=0.8, steering=-0.2
        [0.2, 0.0],   # Sample 3: velocity=0.2, steering=0.0
    ])
    print(f"\nRaw actions shape: {raw_actions.shape}")  # [3, 2]
    print(f"Raw actions:\n{raw_actions}")
    
    # 3. Encode to action features
    action_features = action_encoder(raw_actions)
    print(f"\nEncoded action features shape: {action_features.shape}")  # [3, 512]
    print(f"Feature range: [{action_features.min():.3f}, {action_features.max():.3f}]")
    
    # 4. Use with pipeline
    # (Assuming you have a pipeline already created)
    """
    video = pipeline.inference(
        noise=noise,
        text_prompts=["a robot moving"],
        action_inputs={'action_features': action_features}  # ← Pass encoded features
    )
    """
    
    print("\n✅ Success! Your raw actions [velocity, steering] are now encoded.")


def example_batch_processing():
    """Example: Processing a batch of actions over time"""
    print("\n" + "="*70)
    print("Example 2: Batch Processing (Multiple Frames)")
    print("="*70)
    
    # Create encoder
    action_encoder = ActionEncoder(
        action_dim=2,
        feature_dim=512,
        use_sinusoidal=True,
    )
    
    # Simulate actions for a video sequence
    batch_size = 2
    num_frames = 21
    
    # Your action trajectory (e.g., from a planner or dataset)
    action_trajectory = torch.randn(batch_size, num_frames, 2)  # [B, F, 2]
    print(f"\nAction trajectory shape: {action_trajectory.shape}")
    
    # Process each frame's action
    all_action_features = []
    for frame_idx in range(num_frames):
        # Get action for current frame
        frame_actions = action_trajectory[:, frame_idx, :]  # [B, 2]
        
        # Encode
        action_features = action_encoder(frame_actions)  # [B, 512]
        all_action_features.append(action_features)
    
    # Stack all features
    all_action_features = torch.stack(all_action_features, dim=1)  # [B, F, 512]
    print(f"All action features shape: {all_action_features.shape}")
    
    print("\n✅ Processed all frames!")


def example_with_pipeline_integration():
    """Example: Complete integration with ActionCausalInferencePipeline"""
    print("\n" + "="*70)
    print("Example 3: Complete Pipeline Integration")
    print("="*70)
    
    # Step 1: Create action encoder
    action_encoder = ActionEncoder(
        action_dim=2,           # Your action space
        feature_dim=512,        # Match pipeline's action_dim
        use_sinusoidal=True,
    ).cuda()
    
    print("Step 1: Action encoder created")
    
    # Step 2: Create pipeline with action conditioning
    """
    pipeline = ActionCausalInferencePipeline(
        args=config,
        device='cuda',
        action_dim=512,         # Must match action_encoder.feature_dim
        enable_adaln_zero=True,
    )
    print("Step 2: Pipeline created")
    """
    
    # Step 3: Prepare your data
    batch_size = 2
    raw_actions = torch.tensor([
        [0.7, 0.1],   # Video 1: action
        [0.3, -0.5],  # Video 2: action
    ]).cuda()
    
    print(f"\nStep 3: Raw actions prepared: {raw_actions.shape}")
    
    # Step 4: Encode actions
    action_features = action_encoder(raw_actions)
    print(f"Step 4: Actions encoded: {action_features.shape}")
    
    # Step 5: Generate video with action conditioning
    """
    video = pipeline.inference(
        noise=torch.randn(batch_size, 21, 16, 60, 104).cuda(),
        text_prompts=["a robot moving forward", "a robot turning left"],
        action_inputs={'action_features': action_features}  # ← Your encoded actions!
    )
    print(f"Step 5: Video generated: {video.shape}")
    """
    
    print("\n✅ Complete pipeline ready!")


def example_dynamic_actions():
    """Example: Dynamic actions that change during generation"""
    print("\n" + "="*70)
    print("Example 4: Dynamic Actions (Different for Each Block)")
    print("="*70)
    
    # This shows how you can use different actions for different parts of the video
    action_encoder = ActionEncoder(action_dim=2, feature_dim=512, use_sinusoidal=True)
    
    # Define action sequence (e.g., from a policy or planner)
    action_sequence = [
        [0.8, 0.0],    # Frames 0-6: move forward fast
        [0.5, 0.3],    # Frames 7-13: move forward while turning right
        [0.2, -0.5],   # Frames 14-20: slow down and turn left
    ]
    
    print("\nAction sequence:")
    for i, action in enumerate(action_sequence):
        print(f"  Block {i}: velocity={action[0]:.1f}, steering={action[1]:.1f}")
    
    # Encode all actions
    action_tensors = torch.tensor(action_sequence)  # [3, 2]
    action_features = action_encoder(action_tensors)  # [3, 512]
    
    print(f"\nEncoded features shape: {action_features.shape}")
    print("\n✅ Ready to use different actions for each generation block!")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print("ActionEncoder Usage Examples")
    print("="*70)
    
    example_continuous_actions()
    example_batch_processing()
    example_with_pipeline_integration()
    example_dynamic_actions()
    
    print("\n" + "="*70)
    print("Summary: How to Use Your Two-Number Actions")
    print("="*70)
    print("""
    1. Create ActionEncoder:
       action_encoder = ActionEncoder(action_dim=2, feature_dim=512)
    
    2. Your raw actions:
       raw_actions = [[velocity, steering], ...]  # Your two numbers
    
    3. Encode:
       action_features = action_encoder(torch.tensor(raw_actions))
    
    4. Use with pipeline:
       video = pipeline.inference(
           ...,
           action_inputs={'action_features': action_features}
       )
    
    That's it! 🎉
    """)


if __name__ == '__main__':
    main()

