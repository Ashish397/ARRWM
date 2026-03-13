#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
import argparse
from scipy import stats

#################################
# Configuration
#################################

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--noise_level", type=float, default=0.02, help="Noise level for data augmentation")
parser.add_argument("--name", type=str, default=None, help="Name of the model (new naming convention)")
parser.add_argument("--head_mode", type=str, default="regression", choices=["regression", "distribution"], 
                    help="Head mode: regression or distribution (for backwards compatibility)")
args = parser.parse_args()

noise_level = args.noise_level
checkpoint_name = args.name
head_mode = args.head_mode  # For backwards compatibility

# Directories
predictions_dir = Path("/home/u5dk/as1748.u5dk/ARRWM/test_predictions")
output_dir = Path("test_grid_outputs")
output_dir.mkdir(exist_ok=True)

#################################
# Analysis Functions
#################################

def create_results_table(prop_data, title="Results"):
    """Create a formatted table from proportion data."""
    df = pd.DataFrame(prop_data, index=['A', 'B', 'C', 'D'])
    return df

def print_results_table(df, title):
    """Print a nicely formatted results table."""
    print(f"\n{'=' * 80}")
    print(f"{title}")
    print(f"{'=' * 80}")
    print(df.to_string())
    print(f"{'=' * 80}\n")

# if __name__ == "__main__":
    # visualize()

# Load predictions from all 4 models
print("\n" + "=" * 80)
print("Loading predictions...")
print("=" * 80)

# Load all 3 checkpoints
models = ['xi', 'lambda', 'lambda']
steps = [8244, 5000, 6000]
noises = [0.05, 0.02, 0.02]
model_labels = ['xi_8244', 'lambda_5000', 'lambda_6000']

original_actions_dict = {}
predicted_actions_dict = {}
# Additional statistics dictionaries
predicted_log_std_dict = {}
predicted_std_dict = {}
predicted_unbounded_mean_dict = {}
predicted_unbounded_dict = {}  # For regression mode

for model_name, step, noise_level in zip(models, steps, noises):
    csv_path = predictions_dir / f"predictions_{model_name}_noise{noise_level}_step{step}.csv"
    if not csv_path.exists():
        print(f"Warning: {csv_path} not found, skipping...")
        continue
    print(f"Loading predictions from: {csv_path}")
    df = pd.read_csv(csv_path)
    oa = np.array([(row['original_linear'], row['original_angular']) for _, row in df.iterrows()])
    pa = np.array([(row['predicted_linear'], row['predicted_angular']) for _, row in df.iterrows()])
    original_actions_dict[model_name + '_' + str(step)] = oa
    predicted_actions_dict[model_name + '_' + str(step)] = pa
    
    # Load additional statistics if they exist
    key = model_name + '_' + str(step)
    if 'predicted_log_std_linear' in df.columns:
        # Distribution mode with predicted_ prefix
        predicted_log_std_dict[key] = np.array([(row['predicted_log_std_linear'], row['predicted_log_std_angular']) for _, row in df.iterrows()])
        predicted_std_dict[key] = np.array([(row['predicted_std_linear'], row['predicted_std_angular']) for _, row in df.iterrows()])
        predicted_unbounded_mean_dict[key] = np.array([(row['predicted_unbounded_mean_linear'], row['predicted_unbounded_mean_angular']) for _, row in df.iterrows()])
    elif 'log_std_linear' in df.columns:
        # Distribution mode without predicted_ prefix
        predicted_log_std_dict[key] = np.array([(row['log_std_linear'], row['log_std_angular']) for _, row in df.iterrows()])
        predicted_std_dict[key] = np.array([(row['std_linear'], row['std_angular']) for _, row in df.iterrows()])
    elif 'predicted_unbounded_linear' in df.columns:
        # Regression mode
        predicted_unbounded_dict[key] = np.array([(row['predicted_unbounded_linear'], row['predicted_unbounded_angular']) for _, row in df.iterrows()])
    # visualise(model_name, step)

# Print statistics for first model if available
if len(predicted_log_std_dict) > 0:
    first_key = list(predicted_log_std_dict.keys())[0]
    print(f"log std mean: {predicted_log_std_dict[first_key].mean()}")
    print(f"log std median: {np.median(predicted_log_std_dict[first_key])}")
    print(f"log std max: {predicted_log_std_dict[first_key].max()}")
    print(f"log std min: {predicted_log_std_dict[first_key].min()}")
    print(f"log std std: {predicted_log_std_dict[first_key].std()}")

if len(predicted_std_dict) > 0:
    first_key = list(predicted_std_dict.keys())[0]
    print(f"std mean: {predicted_std_dict[first_key].mean()}")
    print(f"std median: {np.median(predicted_std_dict[first_key])}")
    print(f"std max: {predicted_std_dict[first_key].max()}")
    print(f"std min: {predicted_std_dict[first_key].min()}")
    print(f"std std: {predicted_std_dict[first_key].std()}")

# Use the first model's original actions as ground truth (they should all be the same)
oa_all = original_actions_dict[models[0] + '_' + str(steps[0])]

if oa_all is None:
    print("Error: Could not load any original actions. Exiting analysis.")
    exit(1)

# Check which models were loaded
loaded_models = {k: v for k, v in predicted_actions_dict.items() if v is not None}
if not loaded_models:
    print("Error: Could not load any model predictions. Exiting analysis.")
    exit(1)

print(f"Loaded predictions from: {', '.join(loaded_models.keys())}")

# Ensure all predictions have the same length
lengths = [len(oa_all)]
for model_name, pred in loaded_models.items():
    if pred is not None:
        lengths.append(len(pred))
min_len = min(lengths)

# Truncate original actions to the same length
oa_all = oa_all[:min_len]

# Store data for each checkpoint
checkpoint_data = {}
for i, model_label in enumerate(model_labels):
    key = models[i] + '_' + str(steps[i])
    if key in predicted_actions_dict:
        pa = predicted_actions_dict[key][:min_len]
        checkpoint_data[model_label] = {
            'pa': pa,
            'oa': oa_all,
            'log_std': predicted_log_std_dict.get(key, None),
            'std': predicted_std_dict.get(key, None)
        }
        if checkpoint_data[model_label]['log_std'] is not None:
            checkpoint_data[model_label]['log_std'] = checkpoint_data[model_label]['log_std'][:min_len]
        if checkpoint_data[model_label]['std'] is not None:
            checkpoint_data[model_label]['std'] = checkpoint_data[model_label]['std'][:min_len]

# Plot all checkpoints in a single figure with 6 subplots
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, LinearSegmentedColormap, Normalize, ListedColormap

# Create figure with 3 rows (checkpoints) and 2 columns (angular, linear)
fig, axes = plt.subplots(3, 2, figsize=(16, 20))

# Helper function to calculate optimal log scale
def calculate_optimal_log_norm(x, y, gridsize=50):
    """Calculate optimal LogNorm for hexbin plot by computing counts first."""
    # Use a temporary invisible figure to compute hexbin counts
    fig_temp = plt.figure(figsize=(1, 1))
    hb_temp = plt.hexbin(x, y, gridsize=gridsize, mincnt=1)
    counts = hb_temp.get_array()
    plt.close(fig_temp)
    
    counts_positive = counts[counts > 0]
    if len(counts_positive) > 0:
        vmin = 1  # mincnt=1 ensures minimum is 1
        # Find optimal vmax: try different percentiles and pick one that maximizes log range
        percentiles = [90, 95, 98, 99, 99.5, 99.9]
        max_log_range = 0
        optimal_vmax = counts.max()
        for p in percentiles:
            candidate_vmax = np.percentile(counts_positive, p)
            log_range = np.log10(candidate_vmax) - np.log10(vmin)
            if log_range > max_log_range and candidate_vmax > vmin:
                max_log_range = log_range
                optimal_vmax = candidate_vmax
        vmax = max(optimal_vmax, vmin + 1)
        return LogNorm(vmin=vmin, vmax=vmax)
    return None

# Plot each checkpoint
for row_idx, model_label in enumerate(model_labels):
    if model_label not in checkpoint_data:
        continue
    
    data = checkpoint_data[model_label]
    pa = data['pa']
    oa = data['oa']
    
    # Angular plot (left column)
    ax_ang = axes[row_idx, 0]
    norm_ang = calculate_optimal_log_norm(pa[:,1], oa[:,1])
    hb_ang = ax_ang.hexbin(pa[:,1], oa[:,1], gridsize=50, cmap='viridis', mincnt=1, norm=norm_ang)
    ax_ang.set_xlabel('Predicted Angular')
    ax_ang.set_ylabel('Actual Angular')
    ax_ang.set_title(f'{model_label} - Angular (Log Scale)')
    plt.colorbar(hb_ang, ax=ax_ang, label='Count (log scale)')
    
    # Linear plot (right column)
    ax_lin = axes[row_idx, 1]
    norm_lin = calculate_optimal_log_norm(pa[:,0], oa[:,0])
    hb_lin = ax_lin.hexbin(pa[:,0], oa[:,0], gridsize=50, cmap='viridis', mincnt=1, norm=norm_lin)
    ax_lin.set_xlabel('Predicted Linear')
    ax_lin.set_ylabel('Actual Linear')
    ax_lin.set_title(f'{model_label} - Linear (Log Scale)')
    plt.colorbar(hb_lin, ax=ax_lin, label='Count (log scale)')

plt.tight_layout()
plt.savefig('latent_actions/all_checkpoints_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Plotted all checkpoints in single figure")

# Create uncertainty-based plots
from matplotlib.patches import RegularPolygon

def create_uncertainty_hexbin(ax, x, y, uncertainty, threshold, gridsize=50, mincnt=1):
    """
    Create hexbin plot colored by uncertainty threshold proportion.
    Red = all below threshold, Blue = all above threshold, Pink = equal split.
    Intensity is set by count (using alpha/transparency).
    """
    # Create binary indicator: 1 if above threshold, 0 if below
    above_threshold = (uncertainty > threshold).astype(float)
    
    # First, compute hexbin to get grid structure and counts
    fig_temp = plt.figure(figsize=(1, 1))
    hb_temp = plt.hexbin(x, y, gridsize=gridsize, mincnt=mincnt)
    counts = hb_temp.get_array()
    offsets = hb_temp.get_offsets()
    # Calculate extent from data
    x_margin = (x.max() - x.min()) * 0.05
    y_margin = (y.max() - y.min()) * 0.05
    extent = [x.min() - x_margin, x.max() + x_margin, 
              y.min() - y_margin, y.max() + y_margin]
    plt.close(fig_temp)
    
    # Compute sum of above_threshold values per hexagon
    fig_temp = plt.figure(figsize=(1, 1))
    hb_sum = plt.hexbin(x, y, C=above_threshold, gridsize=gridsize, mincnt=mincnt, reduce_C_function=np.sum)
    sums_above = hb_sum.get_array()
    plt.close(fig_temp)
    
    # Calculate RGB values using the specified formula with exponential/log scale intensity:
    # R = (num_samples_less_than_thresh / total_samples) * intensity
    # G = 0
    # B = (num_samples_more_than_thresh / total_samples) * intensity
    # where intensity is normalized using the same log scale as comparison plots
    valid_mask = counts >= mincnt
    num_samples_less_than_thresh = counts[valid_mask] - sums_above[valid_mask]
    num_samples_more_than_thresh = sums_above[valid_mask]
    total_samples = counts[valid_mask]
    
    # Calculate intensity using the same exponential/log scale as comparison plots
    if len(total_samples) > 0:
        counts_positive = total_samples[total_samples > 0]
        if len(counts_positive) > 0:
            # Find optimal vmax using percentiles (same as comparison plots)
            percentiles = [90, 95, 98, 99, 99.5, 99.9]
            max_log_range = 0
            optimal_vmax = counts_positive.max()
            vmin = 1  # mincnt=1 ensures minimum is 1
            
            for p in percentiles:
                candidate_vmax = np.percentile(counts_positive, p)
                log_range = np.log10(candidate_vmax) - np.log10(vmin)
                if log_range > max_log_range and candidate_vmax > vmin:
                    max_log_range = log_range
                    optimal_vmax = candidate_vmax
            
            vmax = max(optimal_vmax, vmin + 1)
            
            # Normalize counts using log scale with optimal range (same as comparison plots)
            counts_log = np.log10(total_samples + 1)
            intensity_norm = (counts_log - np.log10(vmin)) / (np.log10(vmax) - np.log10(vmin))
            intensity_norm = np.clip(intensity_norm, 0, 1)  # Clip to [0, 1]
            
            # Map intensity to a wider range to ensure variation: [0.2, 1.0]
            # This ensures low-count hexagons are noticeably less intense
            intensity_norm = 0.2 + 0.8 * intensity_norm
        else:
            intensity_norm = np.ones(len(total_samples))
    else:
        intensity_norm = np.array([])
    
    # Calculate R and B channels with exponential/log scale intensity:
    # Mix colors with white based on intensity - low intensity = more white, high intensity = full color
    # FLIPPED: Red = above threshold, Blue = below threshold
    # R = proportion_above * intensity + (1 - intensity) * white_component
    # B = proportion_below * intensity + (1 - intensity) * white_component
    # G = (1 - intensity) * white_component (to create white when intensity is low)
    if len(total_samples) > 0:
        # Calculate proportions (FLIPPED: red = above, blue = below)
        proportion_red = num_samples_more_than_thresh / total_samples  # Red = above threshold
        proportion_blue = num_samples_less_than_thresh / total_samples  # Blue = below threshold
        
        # Blend with white based on intensity: low intensity = more white, high intensity = full color
        # When intensity_norm is low, we want to add white (1, 1, 1), when high we want full color
        # intensity_norm is now in [0.2, 1.0] range
        white_blend = 1.0 - intensity_norm  # How much white to blend in
        R = proportion_red * intensity_norm + white_blend * 1.0
        B = proportion_blue * intensity_norm + white_blend * 1.0
        G = white_blend * 1.0  # Green component for white blending (allows white when intensity is low)
    else:
        R = np.array([])
        B = np.array([])
        G = np.array([])
    
    # Get hexagon positions
    valid_offsets = offsets[valid_mask]
    
    if len(valid_offsets) > 0:
        # Estimate hexagon size from grid
        if len(valid_offsets) > 1:
            # Calculate typical spacing
            dx = np.diff(np.sort(np.unique(valid_offsets[:, 0])))
            dy = np.diff(np.sort(np.unique(valid_offsets[:, 1])))
            dx_typical = dx[dx > 0][0] if len(dx[dx > 0]) > 0 else 1.0
            dy_typical = dy[dy > 0][0] if len(dy[dy > 0]) > 0 else 1.0
            hex_radius = min(dx_typical, dy_typical) * 0.866  # hexagon radius
        else:
            hex_radius = 0.1
        
        # Draw hexagons with RGB values calculated from formula
        for (gx, gy), r, g, b in zip(valid_offsets, R, G, B):
            color_rgb = (r, g, b, 1.0)  # Full opacity
            hexagon = RegularPolygon((gx, gy), numVertices=6, radius=hex_radius,
                                    orientation=np.pi/6, facecolor=color_rgb,
                                    edgecolor='none', linewidth=0)
            ax.add_patch(hexagon)
    
    # Create a dummy colormap for the colorbar (red to blue)
    colors_list = [(1.0, 0.0, 0.0), (1.0, 0.75, 0.8), (0.0, 0.0, 1.0)]
    cmap = LinearSegmentedColormap.from_list('red_pink_blue', colors_list, N=256)
    
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    
    # Return proportions and counts for colorbar (proportions are for display purposes)
    if len(total_samples) > 0:
        proportions = num_samples_more_than_thresh / total_samples
        return proportions, total_samples, cmap
    else:
        return np.array([]), np.array([]), cmap

# Create uncertainty threshold (use median of all uncertainties, or make it configurable)
all_uncertainties = []
for model_label in model_labels:
    if model_label in checkpoint_data and checkpoint_data[model_label]['std'] is not None:
        # Use the magnitude of uncertainty (norm of std vector)
        std = checkpoint_data[model_label]['std']
        uncertainty_magnitude = np.linalg.norm(std, axis=1)
        all_uncertainties.extend(uncertainty_magnitude)

if len(all_uncertainties) > 0:
    # uncertainty_threshold = np.median(all_uncertainties)
    uncertainty_threshold = 1
    print(f"Using uncertainty threshold (median): {uncertainty_threshold:.4f}")
else:
    uncertainty_threshold = 1.0
    print(f"Warning: No uncertainty data found, using default threshold: {uncertainty_threshold}")

# Create figure with 3 rows (checkpoints) and 2 columns (angular, linear)
fig, axes = plt.subplots(3, 2, figsize=(16, 20))

# Plot each checkpoint with uncertainty coloring
for row_idx, model_label in enumerate(model_labels):
    if model_label not in checkpoint_data:
        continue
    
    data = checkpoint_data[model_label]
    pa = data['pa']
    oa = data['oa']
    std = data['std']
    
    if std is None:
        print(f"Warning: No uncertainty data for {model_label}, skipping uncertainty plot")
        continue
    
    # Calculate uncertainty magnitude (norm of std vector)
    uncertainty = np.linalg.norm(std, axis=1)
    
    # Angular plot (left column)
    ax_ang = axes[row_idx, 0]
    result = create_uncertainty_hexbin(ax_ang, pa[:,1], oa[:,1], uncertainty, uncertainty_threshold, 
                                      gridsize=50, mincnt=1)
    if result is not None:
        proportions, counts, cmap = result
        ax_ang.set_xlabel('Predicted Angular')
        ax_ang.set_ylabel('Actual Angular')
        ax_ang.set_title(f'{model_label} - Angular (Uncertainty Threshold)')
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax_ang)
        cbar.set_label('Proportion Above Threshold (Red=High, Blue=Low, Pink=Equal)')
    
    # Linear plot (right column)
    ax_lin = axes[row_idx, 1]
    result = create_uncertainty_hexbin(ax_lin, pa[:,0], oa[:,0], uncertainty, uncertainty_threshold,
                                      gridsize=50, mincnt=1)
    if result is not None:
        proportions, counts, cmap = result
        ax_lin.set_xlabel('Predicted Linear')
        ax_lin.set_ylabel('Actual Linear')
        ax_lin.set_title(f'{model_label} - Linear (Uncertainty Threshold)')
        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax_lin)
        cbar.set_label('Proportion Above Threshold (Red=High, Blue=Low, Pink=Equal)')

plt.tight_layout()
plt.savefig('latent_actions/all_checkpoints_uncertainty.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Plotted all checkpoints with uncertainty coloring")

# Create video animation from uncertainty_min to uncertainty_max
if len(all_uncertainties) > 0:
    uncertainty_min = np.min(all_uncertainties)
    uncertainty_max = np.max(all_uncertainties)
    print(f"Creating video from uncertainty {uncertainty_min:.4f} to {uncertainty_max:.4f}")
    
    # Video parameters: 10 seconds total, 0.5 seconds per frame = 20 frames
    num_frames = 20
    frame_duration = 0.5  # seconds per frame
    uncertainty_thresholds = np.linspace(uncertainty_min, uncertainty_max, num_frames)
    
    # Create frames directory
    import os
    frames_dir = Path('latent_actions/uncertainty_video_frames')
    frames_dir.mkdir(exist_ok=True)
    
    # Generate frames
    print(f"Generating {num_frames} frames...")
    for frame_idx, threshold in enumerate(uncertainty_thresholds):
        print(f"  Frame {frame_idx + 1}/{num_frames}: threshold = {threshold:.4f}")
        
        # Create figure with 3 rows (checkpoints) and 2 columns (angular, linear)
        fig, axes = plt.subplots(3, 2, figsize=(16, 20))
        
        # Plot each checkpoint with uncertainty coloring
        for row_idx, model_label in enumerate(model_labels):
            if model_label not in checkpoint_data:
                continue
            
            data = checkpoint_data[model_label]
            pa = data['pa']
            oa = data['oa']
            std = data['std']
            
            if std is None:
                continue
            
            # Calculate uncertainty magnitude (norm of std vector)
            uncertainty = np.linalg.norm(std, axis=1)
            
            # Angular plot (left column)
            ax_ang = axes[row_idx, 0]
            result = create_uncertainty_hexbin(ax_ang, pa[:,1], oa[:,1], uncertainty, threshold, 
                                              gridsize=50, mincnt=1)
            if result is not None:
                proportions, counts, cmap = result
                ax_ang.set_xlabel('Predicted Angular')
                ax_ang.set_ylabel('Actual Angular')
                ax_ang.set_title(f'{model_label} - Angular (Threshold: {threshold:.3f})')
                # Add colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax_ang)
                cbar.set_label('Proportion Above Threshold (Red=High, Blue=Low, Pink=Equal)')
            
            # Linear plot (right column)
            ax_lin = axes[row_idx, 1]
            result = create_uncertainty_hexbin(ax_lin, pa[:,0], oa[:,0], uncertainty, threshold,
                                              gridsize=50, mincnt=1)
            if result is not None:
                proportions, counts, cmap = result
                ax_lin.set_xlabel('Predicted Linear')
                ax_lin.set_ylabel('Actual Linear')
                ax_lin.set_title(f'{model_label} - Linear (Threshold: {threshold:.3f})')
                # Add colorbar
                sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax_lin)
                cbar.set_label('Proportion Above Threshold (Red=High, Blue=Low, Pink=Equal)')
        
        # Add overall title with current threshold
        fig.suptitle(f'Uncertainty Threshold: {threshold:.4f}', fontsize=16, y=0.995)
        
        plt.tight_layout(rect=[0, 0, 1, 0.99])  # Leave space for suptitle
        plt.savefig(frames_dir / f'frame_{frame_idx:03d}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    # Create video from frames using imageio or ffmpeg
    try:
        import imageio
        print("Creating video using imageio...")
        frames = []
        for frame_idx in range(num_frames):
            frame_path = frames_dir / f'frame_{frame_idx:03d}.png'
            if frame_path.exists():
                frames.append(imageio.imread(frame_path))
        
        if len(frames) > 0:
            # Save video: 0.5 seconds per frame = 2 fps
            fps = 1.0 / frame_duration  # 2 fps
            video_path = Path('latent_actions/all_checkpoints_uncertainty_video.mp4')
            imageio.mimsave(video_path, frames, fps=fps, codec='libx264', quality=8)
            print(f"Video saved to: {video_path}")
        else:
            print("Error: No frames found to create video")
    except ImportError:
        print("imageio not available, trying ffmpeg...")
        import subprocess
        # Use ffmpeg to create video
        video_path = Path('latent_actions/all_checkpoints_uncertainty_video.mp4')
        ffmpeg_cmd = [
            'ffmpeg', '-y', '-framerate', str(1.0 / frame_duration),  # 2 fps
            '-i', str(frames_dir / 'frame_%03d.png'),
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p',
            str(video_path)
        ]
        try:
            subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
            print(f"Video saved to: {video_path}")
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Error: Could not create video. Please install imageio or ffmpeg.")
            print(f"Frames are saved in: {frames_dir}")
    
    print(f"Video creation complete!")
else:
    print("No uncertainty data available for video creation")