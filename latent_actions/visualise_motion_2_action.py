#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
import argparse
from scipy import stats

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

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

# Models to exclude from ensembling (e.g., ['pa_a', 'pa_c'] to exclude models A and C)
# pa_a (iota) is excluded because it has different label shapes than the other models
exclude_from_ensemble = []  # Set to ['pa_a', 'pa_c'] etc. to exclude specific models

# Grid parameters
GRID_MIN = -1.1
GRID_MAX = 1.1
GRID_SIZE = 11  # 11x11 grid
CELL_WIDTH = (GRID_MAX - GRID_MIN) / GRID_SIZE  # 0.2

#################################
# Helper Functions
#################################

def get_grid_index(value):
    """
    Convert a value to a grid index (0 to 10).
    Values are binned into cells of width 0.2.
    Standard binning uses [start, end) intervals.
    For the central cell to include both -0.1 and 0.1, we use:
    - Standard [start, end) for most cells
    - Upper boundaries (like 0.1, 0.3, etc.) go to the lower cell
    """
    # Clip values to grid range
    value = max(GRID_MIN, min(GRID_MAX, value))
    
    # Handle upper bound - values at GRID_MAX go to last cell
    if abs(value - GRID_MAX) < 1e-9:
        return GRID_SIZE - 1
    
    # Calculate index using standard binning: [start, end)
    # To make upper boundaries go to the lower cell, we subtract a tiny epsilon
    # This ensures values like 0.1, 0.3 go to the cell below
    epsilon = 1e-12
    raw_index = (value - GRID_MIN - epsilon) / CELL_WIDTH
    
    # For the central cell boundary (-0.1), we want it in cell 5, not cell 4
    # Check if value is exactly -0.1 (within floating point precision)
    if abs(value - (-0.1)) < 1e-9:
        return 5
    
    index = int(np.floor(raw_index))
    
    # Ensure index is in valid range
    return min(max(0, index), GRID_SIZE - 1)

def create_grid_from_actions(actions_list):
    """Create a grid from a list of (linear, angular) actions."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    
    for linear, angular in actions_list:
        if pd.isna(linear) or pd.isna(angular):
            continue
        
        x_idx = get_grid_index(float(angular))
        y_idx = get_grid_index(float(linear))
        grid[y_idx, x_idx] += 1
    
    return grid

def create_three_subplot_figure(original_grid, predicted_grid, matching_grid, output_file, title_suffix=""):
    """
    Create a figure with three subplots showing original, predicted, and matching distributions.
    """
    CELL_WIDTH = (GRID_MAX - GRID_MIN) / GRID_SIZE
    
    fig, axes = plt.subplots(1, 3, figsize=(30, 9))
    
    grids = [original_grid, predicted_grid, matching_grid]
    titles = [
        f"Original Distribution{title_suffix}",
        f"Model Distribution{title_suffix}",
        f"Matching Distribution{title_suffix}"
    ]
    
    extent = [GRID_MIN, GRID_MAX, GRID_MIN, GRID_MAX]
    vmax = max(g.max() for g in grids)
    
    for ax, grid, title in zip(axes, grids, titles):
        im = ax.imshow(grid, cmap='YlOrRd', aspect='auto', interpolation='nearest',
                       extent=extent, origin='lower', vmin=0, vmax=vmax)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Number of Actions', rotation=270, labelpad=20)
        
        # Set ticks and labels
        num_ticks = GRID_SIZE + 1
        tick_positions = np.linspace(GRID_MIN, GRID_MAX, num_ticks)
        tick_labels = [f'{GRID_MIN + i * CELL_WIDTH:.1f}' for i in range(num_ticks)]
        
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')
        ax.set_yticklabels(tick_labels)
        
        ax.set_xlabel('Right to Left (Angular/Sideways)', fontsize=12)
        ax.set_ylabel('Front to Back (Linear/Forward-Back)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Add text annotations
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                value = grid[i, j]
                if value > 0:
                    x_pos = GRID_MIN + (j + 0.5) * CELL_WIDTH
                    y_pos = GRID_MIN + (i + 0.5) * CELL_WIDTH
                    text_color = 'white' if value > grid.max() * 0.5 else 'black'
                    ax.text(x_pos, y_pos, f'{value:,}', ha='center', va='center',
                           color=text_color, fontsize=8, fontweight='bold')
        
        # Add grid lines
        grid_lines_x = [GRID_MIN + i * CELL_WIDTH for i in range(GRID_SIZE + 1)]
        grid_lines_y = [GRID_MIN + i * CELL_WIDTH for i in range(GRID_SIZE + 1)]
        for x in grid_lines_x:
            ax.axvline(x, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        for y in grid_lines_y:
            ax.axhline(y, color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
        
        # Add statistics
        total_actions = grid.sum()
        stats_text = f'Total: {total_actions:,}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def plot_error_violins(original_actions, predicted_actions, output_file, file_prefix):
    """
    Create violin plots showing the distribution of prediction errors (predicted - actual)
    for both linear and angular components.
    
    Args:
        original_actions: List of (linear, angular) tuples (GT)
        predicted_actions: List of (linear, angular) tuples (predictions)
        output_file: Path to save the plot
        file_prefix: Prefix for title
    """
    # Convert to numpy arrays
    gt_linear = np.array([a[0] for a in original_actions])
    gt_angular = np.array([a[1] for a in original_actions])
    pred_linear = np.array([a[0] for a in predicted_actions])
    pred_angular = np.array([a[1] for a in predicted_actions])
    
    # Compute differences (predicted - actual)
    linear_errors = pred_linear - gt_linear
    angular_errors = pred_angular - gt_angular
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Violin plot for linear errors
    parts1 = ax1.violinplot([linear_errors], positions=[0], widths=0.6, 
                             showmeans=True, showmedians=True, showextrema=True)
    ax1.set_ylabel('Error (Predicted - Actual)', fontsize=12)
    ax1.set_title(f'Linear Action Error Distribution\n{file_prefix}', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks([0])
    ax1.set_xticklabels(['Linear'])
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Zero Error')
    
    # Color the violin plot
    for pc in parts1['bodies']:
        pc.set_facecolor('skyblue')
        pc.set_alpha(0.7)
    
    # Add statistics text
    mean_linear = np.mean(linear_errors)
    std_linear = np.std(linear_errors)
    median_linear = np.median(linear_errors)
    mae_linear = np.mean(np.abs(linear_errors))
    stats_text_linear = f'Mean: {mean_linear:.4f}\nStd: {std_linear:.4f}\nMedian: {median_linear:.4f}\nMAE: {mae_linear:.4f}'
    ax1.text(0.02, 0.98, stats_text_linear, transform=ax1.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Violin plot for angular errors
    parts2 = ax2.violinplot([angular_errors], positions=[0], widths=0.6,
                            showmeans=True, showmedians=True, showextrema=True)
    ax2.set_ylabel('Error (Predicted - Actual)', fontsize=12)
    ax2.set_title(f'Angular Action Error Distribution\n{file_prefix}',
                 fontsize=14, fontweight='bold')
    ax2.set_xticks([0])
    ax2.set_xticklabels(['Angular'])
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7, label='Zero Error')
    
    # Color the violin plot
    for pc in parts2['bodies']:
        pc.set_facecolor('lightcoral')
        pc.set_alpha(0.7)
    
    # Add statistics text
    mean_angular = np.mean(angular_errors)
    std_angular = np.std(angular_errors)
    median_angular = np.median(angular_errors)
    mae_angular = np.mean(np.abs(angular_errors))
    stats_text_angular = f'Mean: {mean_angular:.4f}\nStd: {std_angular:.4f}\nMedian: {median_angular:.4f}\nMAE: {mae_angular:.4f}'
    ax2.text(0.02, 0.98, stats_text_angular, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

def categorize_action_for_violin(linear, angular):
    """
    Categorize an action into one of 9 categories for error analysis.
    Returns category index: 0-8
    0: abs(linear) > abs(angular) && 1.0 > linear > 0.5 (strong forward)
    1: abs(linear) > abs(angular) && -1.0 < linear < -0.5 (strong backward)
    2: abs(linear) > abs(angular) && 0.5 > linear > 0.1 (medium forward)
    3: abs(linear) > abs(angular) && -0.5 < linear < -0.1 (medium backward)
    4: abs(linear) < abs(angular) && 1.0 > angular > 0.5 (strong right)
    5: abs(linear) < abs(angular) && -1.0 < angular < -0.5 (strong left)
    6: abs(linear) < abs(angular) && 0.5 > angular > 0.1 (medium right)
    7: abs(linear) < abs(angular) && -0.5 < angular < -0.1 (medium left)
    8: abs(linear) < 0.1 && abs(angular) < 0.1 (stationary/noop)
    """
    abs_lin = abs(linear)
    abs_ang = abs(angular)
    
    # Stationary/noop - check this first
    if abs_lin < 0.1 and abs_ang < 0.1:
        return 8
    
    # Linear dominant
    if abs_lin > abs_ang:
        if 1.0 > linear > 0.5:
            return 0  # Strong forward
        elif -1.0 < linear < -0.5:
            return 1  # Strong backward
        elif 0.5 > linear > 0.1:
            return 2  # Medium forward
        elif -0.5 < linear < -0.1:
            return 3  # Medium backward
        else:
            # Edge case: linear dominant but outside these ranges
            return -1  # Unclassified
    
    # Angular dominant
    elif abs_lin < abs_ang:
        if 1.0 > angular > 0.5:
            return 4  # Strong right
        elif -1.0 < angular < -0.5:
            return 5  # Strong left
        elif 0.5 > angular > 0.1:
            return 6  # Medium right
        elif -0.5 < angular < -0.1:
            return 7  # Medium left
        else:
            # Edge case: angular dominant but outside these ranges
            return -1  # Unclassified
    
    # Equal case (abs_lin == abs_ang) - shouldn't happen often, but handle it
    else:
        return -1  # Unclassified

def plot_categorized_error_violins(original_actions, predicted_actions, output_file, file_prefix):
    """
    Create violin plots showing error distributions broken down by action category.
    Separate violins for each of 9 action categories.
    
    Args:
        original_actions: List of (linear, angular) tuples (GT)
        predicted_actions: List of (linear, angular) tuples (predictions)
        output_file: Path to save the plot
        file_prefix: Prefix for title
    """
    # Convert to numpy arrays
    gt_linear = np.array([a[0] for a in original_actions])
    gt_angular = np.array([a[1] for a in original_actions])
    pred_linear = np.array([a[0] for a in predicted_actions])
    pred_angular = np.array([a[1] for a in predicted_actions])
    
    # Compute differences (predicted - actual)
    linear_errors = pred_linear - gt_linear
    angular_errors = pred_angular - gt_angular
    
    # Categorize each action
    categories = []
    for i in range(len(original_actions)):
        cat = categorize_action_for_violin(gt_linear[i], gt_angular[i])
        categories.append(cat)
    categories = np.array(categories)
    
    # Category labels
    category_labels = [
        'Fwd Strong\n(0.5<lin<1.0)',
        'Bwd Strong\n(-1.0<lin<-0.5)',
        'Fwd Med\n(0.1<lin<0.5)',
        'Bwd Med\n(-0.5<lin<-0.1)',
        'Right Strong\n(0.5<ang<1.0)',
        'Left Strong\n(-1.0<ang<-0.5)',
        'Right Med\n(0.1<ang<0.5)',
        'Left Med\n(-0.5<ang<-0.1)',
        'Stationary\n(|lin|<0.1,|ang|<0.1)'
    ]
    
    # Filter out unclassified (-1)
    valid_mask = categories >= 0
    categories = categories[valid_mask]
    linear_errors = linear_errors[valid_mask]
    angular_errors = angular_errors[valid_mask]
    
    # Group errors by category
    linear_errors_by_cat = []
    angular_errors_by_cat = []
    valid_categories = []
    valid_labels = []
    
    for cat_idx in range(9):
        mask = categories == cat_idx
        if mask.sum() > 0:  # Only include categories that have samples
            linear_errors_by_cat.append(linear_errors[mask])
            angular_errors_by_cat.append(angular_errors[mask])
            valid_categories.append(cat_idx)
            valid_labels.append(category_labels[cat_idx])
    
    if len(linear_errors_by_cat) == 0:
        print("Warning: No valid categories found for categorized error violins")
        return
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Violin plot for linear errors by category
    positions = np.arange(len(linear_errors_by_cat))
    parts1 = ax1.violinplot(linear_errors_by_cat, positions=positions, widths=0.6,
                            showmeans=True, showmedians=True, showextrema=True)
    
    # Color the violins
    colors = plt.cm.Set3(np.linspace(0, 1, len(linear_errors_by_cat)))
    for pc, color in zip(parts1['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax1.set_ylabel('Error (Predicted - Actual)', fontsize=12)
    ax1.set_title(f'Linear Action Error by Category\n{file_prefix}', 
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(positions)
    ax1.set_xticklabels(valid_labels, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    # Add sample counts
    for i, (pos, errors) in enumerate(zip(positions, linear_errors_by_cat)):
        count = len(errors)
        mean_err = np.mean(errors)
        ax1.text(pos, ax1.get_ylim()[1] * 0.95, f'n={count}\nμ={mean_err:.3f}',
                ha='center', va='top', fontsize=8, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Violin plot for angular errors by category
    parts2 = ax2.violinplot(angular_errors_by_cat, positions=positions, widths=0.6,
                            showmeans=True, showmedians=True, showextrema=True)
    
    # Color the violins (same colors as linear)
    for pc, color in zip(parts2['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)
    
    ax2.set_ylabel('Error (Predicted - Actual)', fontsize=12)
    ax2.set_title(f'Angular Action Error by Category\n{file_prefix}',
                 fontsize=14, fontweight='bold')
    ax2.set_xticks(positions)
    ax2.set_xticklabels(valid_labels, rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.7)
    
    # Add sample counts
    for i, (pos, errors) in enumerate(zip(positions, angular_errors_by_cat)):
        count = len(errors)
        mean_err = np.mean(errors)
        ax2.text(pos, ax2.get_ylim()[1] * 0.95, f'n={count}\nμ={mean_err:.3f}',
                ha='center', va='top', fontsize=8,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

#################################
# Visualization Function
#################################

def visualise(model_name, step):
    print("=" * 80)
    print(f"Motion2Action Visualization Script - {model_name} step {step}")
    print("=" * 80)
    
    # Load predictions from CSV using new naming convention with step
    csv_filename = f"predictions_{model_name}_noise{noise_level}_step{step}.csv"
    csv_path = predictions_dir / csv_filename
    
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Predictions file not found: {csv_path}\n"
            f"Please run test_motion_2_action.py first to generate predictions."
        )
    
    print(f"\nLoading predictions from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Extract original and predicted actions
    original_actions = [(row['original_linear'], row['original_angular']) 
                        for _, row in df.iterrows()]
    predicted_actions = [(row['predicted_linear'], row['predicted_angular']) 
                        for _, row in df.iterrows()]
    
    print(f"Loaded {len(original_actions)} samples")
    
    # Create grids
    print("\nCreating distribution grids...")
    
    # Grid 1: Original data distribution (from input_actions CSV files - ground truth linear/angular)
    print("  Creating original data distribution grid (from input_actions files)...")
    original_grid = create_grid_from_actions(original_actions)
    
    # Grid 2: Model output distribution (model predictions)
    print("  Creating model output distribution grid (model predictions)...")
    predicted_grid = create_grid_from_actions(predicted_actions)
    
    # Grid 3: Positively matching distributions (where original and predicted match)
    print("  Creating matching distribution grid...")
    matching_actions = []
    for orig, pred in zip(original_actions, predicted_actions):
        orig_linear, orig_angular = orig
        pred_linear, pred_angular = pred
        
        # Check if they fall in the same grid cell
        orig_x = get_grid_index(orig_angular)
        orig_y = get_grid_index(orig_linear)
        pred_x = get_grid_index(pred_angular)
        pred_y = get_grid_index(pred_linear)
        
        if orig_x == pred_x and orig_y == pred_y:
            matching_actions.append(orig)
    
    matching_grid = create_grid_from_actions(matching_actions)
    
    # Create filenames with model_name and step
    file_prefix = f"{model_name}_noise{noise_level}_step{step}"
    title_suffix = f" ({model_name}, step={step}, noise={noise_level})"
    
    combined_filename = f"combined_{file_prefix}.grid.png"
    violin_filename = f"error_violins_{file_prefix}.png"
    categorized_violin_filename = f"categorized_error_violins_{file_prefix}.png"
    
    print(f"\nSaving grid plots to {output_dir}...")
    
    # Create combined three-subplot figure
    create_three_subplot_figure(original_grid, predicted_grid, matching_grid,
                                output_dir / combined_filename,
                                title_suffix=title_suffix)
    
    # Create error violin plots
    print("\nCreating error violin plots...")
    plot_error_violins(original_actions, predicted_actions,
                      output_dir / violin_filename, file_prefix)
    
    # Create categorized error violin plots
    print("\nCreating categorized error violin plots...")
    plot_categorized_error_violins(original_actions, predicted_actions,
                                   output_dir / categorized_violin_filename, file_prefix)
    
    # Print statistics
    print("\n" + "=" * 80)
    print("Grid Statistics:")
    print("=" * 80)
    print(f"Original distribution - Total actions: {original_grid.sum():,}")
    print(f"Model distribution - Total actions: {predicted_grid.sum():,}")
    print(f"Matching distribution - Total actions: {matching_grid.sum():,}")
    print(f"Match rate: {matching_grid.sum() / original_grid.sum() * 100:.2f}%")
    print(f"\nPlots saved to: {output_dir.absolute()}")
    print(f"  - {combined_filename}")
    print(f"  - {violin_filename}")
    print(f"  - {categorized_violin_filename}")
    print("=" * 80)

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

def ensemble_mean(predictions_list):
    """Ensemble using mean of all predictions."""
    return np.mean(predictions_list, axis=0)

def ensemble_median(predictions_list):
    """Ensemble using median of all predictions."""
    return np.median(predictions_list, axis=0)

def ensemble_mode(predictions_list):
    """Ensemble using mode (most common value) of all predictions."""
    # For continuous values, we need to discretize first
    # Round to 2 decimal places for mode calculation
    rounded = np.round(predictions_list, decimals=2)
    # Compute mode along the first axis (across models) for each sample and dimension
    # predictions_list shape: [4, N, 2]
    # We want mode for each of the N samples and 2 dimensions
    N = rounded.shape[1]
    mode_result = np.zeros((N, 2))
    for i in range(N):
        for j in range(2):
            mode_result[i, j] = stats.mode(rounded[:, i, j], keepdims=False)[0]
    return mode_result

def ensemble_weighted_average(predictions_list, weights=None):
    """Ensemble using weighted average. Default: equal weights."""
    if weights is None:
        weights = np.ones(len(predictions_list)) / len(predictions_list)
    else:
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
    
    weighted_sum = np.zeros_like(predictions_list[0])
    for pred, weight in zip(predictions_list, weights):
        weighted_sum += pred * weight
    return weighted_sum

# if __name__ == "__main__":
    # visualize()

# Load predictions from all 4 models
print("\n" + "=" * 80)
print("Loading predictions for ensemble analysis...")
print("=" * 80)

# Load all model predictions
# models = ['kappa', 'kappa', 'kappa', 'kappa']
# steps = [4000, 4333, 5000, 6000]

# models = ['lambda', 'lambda', 'lambda', 'lambda']
# steps = [4000, 4121, 5000, 6000]

# models = ['mu', 'mu', 'mu', 'mu']
# steps = [4000, 4537, 5000, 6000]

models = ['nu', 'xi', 'omicron', 'pi']
steps = [8000, 8244, 8000, 8000]
noises = [0.02, 0.05, 0.02, 0.05]

model_labels = ['A', 'B', 'C', 'D']
original_actions_dict = {}
predicted_actions_dict = {}

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
    # visualise(model_name, step)

# Use the first model's original actions as ground truth (they should all be the same)
oa_a = original_actions_dict[models[0] + '_' + str(steps[0])]
oa_b = original_actions_dict[models[1] + '_' + str(steps[1])]
assert (oa_a - oa_b).sum() == 0
oa_all = oa_b

# Load predictions and print which files were loaded
pa_a = predicted_actions_dict.get(models[0] + '_' + str(steps[0]))
pa_b = predicted_actions_dict.get(models[1] + '_' + str(steps[1]))
pa_c = predicted_actions_dict.get(models[2] + '_' + str(steps[2]))
pa_d = predicted_actions_dict.get(models[3] + '_' + str(steps[3]))

print("\nLoaded model predictions:")
if pa_a is not None:
    print(f"  pa_a (alpha): {predictions_dir / f'predictions_{models[0]}_noise{noises[0]}_step{steps[0]}.csv'}")
else:
    print(f"  pa_a (alpha): Not loaded")
if pa_b is not None:
    print(f"  pa_b (beta): {predictions_dir / f'predictions_{models[1]}_noise{noises[1]}_step{steps[1]}.csv'}")
else:
    print(f"  pa_b (beta): Not loaded")
if pa_c is not None:
    print(f"  pa_c (gamma): {predictions_dir / f'predictions_{models[2]}_noise{noises[2]}_step{steps[2]}.csv'}")
else:
    print(f"  pa_c (gamma): Not loaded")
if pa_d is not None:
    print(f"  pa_d (delta): {predictions_dir / f'predictions_{models[3]}_noise{noises[3]}_step{steps[3]}.csv'}")
else:
    print(f"  pa_d (delta): Not loaded")

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

# Truncate all arrays to the same length
oa_all = oa_all[:min_len]
if pa_a is not None:
    pa_a = pa_a[:min_len]
if pa_b is not None:
    pa_b = pa_b[:min_len]
if pa_c is not None:
    pa_c = pa_c[:min_len]
if pa_d is not None:
    pa_d = pa_d[:min_len]

# Calculate errors for individual models (only for loaded models)
delta_a = (pa_a - oa_all) if pa_a is not None else None
delta_b = (pa_b - oa_all) if pa_b is not None else None
delta_c = (pa_c - oa_all) if pa_c is not None else None
delta_d = (pa_d - oa_all) if pa_d is not None else None

norm_a = np.linalg.norm(delta_a, axis=1) if delta_a is not None else None
norm_b = np.linalg.norm(delta_b, axis=1) if delta_b is not None else None
norm_c = np.linalg.norm(delta_c, axis=1) if delta_c is not None else None
norm_d = np.linalg.norm(delta_d, axis=1) if delta_d is not None else None

#Get norms
# Find action type locations (using oa_all as ground truth)
stat_loc = np.logical_and(np.abs(oa_all[:,0]) < 0.1, np.abs(oa_all[:,1]) < 0.1)
forward_loc = np.logical_and(oa_all[:,0] > 0.1, np.abs(oa_all[:,0]) > np.abs(oa_all[:,1]))
backward_loc = np.logical_and(oa_all[:,0] < -0.1, np.abs(oa_all[:,0]) > np.abs(oa_all[:,1]))
right_loc = np.logical_and(oa_all[:,1] > 0.1, np.abs(oa_all[:,1]) > np.abs(oa_all[:,0]))
left_loc = np.logical_and(oa_all[:,1] < -0.1, np.abs(oa_all[:,1]) > np.abs(oa_all[:,0]))

# Calculate overall error proportions
error_thresholds = [0.1, 0.2, 0.3]
models_data = {}
if norm_a is not None:
    models_data['A'] = norm_a
if norm_b is not None:
    models_data['B'] = norm_b
if norm_c is not None:
    models_data['C'] = norm_c
if norm_d is not None:
    models_data['D'] = norm_d

# Create per-action-type error tables
action_types = {
    'Forward': forward_loc,
    'Backward': backward_loc,
    'Right': right_loc,
    'Left': left_loc,
    'Stationary': stat_loc
}

# Create overall error table (weighted equally across categories to avoid bias from imbalanced data)
overall_data = {}
for threshold in error_thresholds:
    row_data = []
    for label in ['A', 'B', 'C', 'D']:
        if label in models_data:
            norm = models_data[label]
            # Calculate error proportion for each category, then average with equal weights
            category_errors = []
            for action_name, action_mask in action_types.items():
                if action_mask.sum() > 0:  # Only include categories that exist in the data
                    category_errors.append(np.mean(norm[action_mask] < threshold))
            # Weight each category equally (not by frequency) to avoid bias
            row_data.append(np.mean(category_errors) if category_errors else np.nan)
        else:
            row_data.append(np.nan)
    overall_data[f'Error < {threshold}'] = row_data

overall_df = pd.DataFrame(overall_data, index=['A', 'B', 'C', 'D'])
print_results_table(overall_df, "Overall Error Proportions (weighted equally across categories)")

for action_name, action_mask in action_types.items():
    action_data = {}
    for threshold in error_thresholds:
        row_data = []
        for norm, label in [(norm_a, 'A'), (norm_b, 'B'), (norm_c, 'C'), (norm_d, 'D')]:
            if norm is not None and action_mask.sum() > 0:
                row_data.append(np.mean(norm[action_mask] < threshold))
            else:
                row_data.append(np.nan)
        action_data[f'Error < {threshold}'] = row_data
    action_df = pd.DataFrame(action_data, index=['A', 'B', 'C', 'D'])
    print_results_table(action_df, f"Error Proportions for {action_name} Actions (n={action_mask.sum()})")

# Category-based analysis
def categorize_action(linear, angular):
    """Categorize action: 1=forward, 2=backward, 3=right, 4=left, 5=stationary"""
    if np.abs(linear) < 0.1 and np.abs(angular) < 0.1:
        return 5
    elif np.abs(linear) > np.abs(angular):
        return 1 if linear > 0.1 else 2
    else:
        return 3 if angular > 0.1 else 4

oa_categories = np.array([categorize_action(oa_all[i,0], oa_all[i,1]) for i in range(len(oa_all))])
pa_a_categories = np.array([categorize_action(pa_a[i,0], pa_a[i,1]) for i in range(len(pa_a))]) if pa_a is not None else None
pa_b_categories = np.array([categorize_action(pa_b[i,0], pa_b[i,1]) for i in range(len(pa_b))]) if pa_b is not None else None
pa_c_categories = np.array([categorize_action(pa_c[i,0], pa_c[i,1]) for i in range(len(pa_c))]) if pa_c is not None else None
pa_d_categories = np.array([categorize_action(pa_d[i,0], pa_d[i,1]) for i in range(len(pa_d))]) if pa_d is not None else None

category_names = {1: 'Forward', 2: 'Backward', 3: 'Right', 4: 'Left', 5: 'Stationary'}

# Category accuracy table
category_data = {'Overall': [
    np.mean(oa_categories == pa_a_categories) if pa_a_categories is not None else np.nan,
    np.mean(oa_categories == pa_b_categories) if pa_b_categories is not None else np.nan,
    np.mean(oa_categories == pa_c_categories) if pa_c_categories is not None else np.nan,
    np.mean(oa_categories == pa_d_categories) if pa_d_categories is not None else np.nan
]}

for cat_id, cat_name in category_names.items():
    mask = oa_categories == cat_id
    if mask.sum() > 0:
        category_data[cat_name] = [
            np.mean(oa_categories[mask] == pa_a_categories[mask]) if pa_a_categories is not None else np.nan,
            np.mean(oa_categories[mask] == pa_b_categories[mask]) if pa_b_categories is not None else np.nan,
            np.mean(oa_categories[mask] == pa_c_categories[mask]) if pa_c_categories is not None else np.nan,
            np.mean(oa_categories[mask] == pa_d_categories[mask]) if pa_d_categories is not None else np.nan
        ]

category_df = pd.DataFrame(category_data, index=['A', 'B', 'C', 'D'])
print_results_table(category_df, "Category Classification Accuracy")

# # ========== ENSEMBLE ANALYSIS ==========
# print("\n" + "=" * 80)
# print("ENSEMBLE ANALYSIS")
# print("=" * 80)

# # Create ensemble predictions (only from loaded models, excluding specified ones)
# available_predictions = []
# pred_names = ['pa_a', 'pa_b', 'pa_c', 'pa_d']
# pred_values = [pa_a, pa_b, pa_c, pa_d]

# for pred_name, pred in zip(pred_names, pred_values):
#     if pred is not None and pred_name not in exclude_from_ensemble:
#         available_predictions.append(pred)
#     elif pred_name in exclude_from_ensemble:
#         print(f"Excluding {pred_name} from ensemble (as specified in exclude_from_ensemble)")

# if exclude_from_ensemble:
#     print(f"Models excluded from ensemble: {exclude_from_ensemble}")

# if len(available_predictions) == 0:
#     print("Warning: No predictions available for ensemble. Skipping ensemble analysis.")
# else:
#     all_predictions = np.stack(available_predictions, axis=0)  # [M, N, 2] where M is number of loaded models
    
#     pa_mean = ensemble_mean(all_predictions)
#     pa_median = ensemble_median(all_predictions)
#     pa_mode = ensemble_mode(all_predictions)
#     pa_weighted = ensemble_weighted_average(all_predictions)
    
#     # Calculate ensemble errors
#     delta_mean = pa_mean - oa_all
#     delta_median = pa_median - oa_all
#     delta_mode = pa_mode - oa_all
#     delta_weighted = pa_weighted - oa_all
    
#     norm_mean = np.linalg.norm(delta_mean, axis=1)
#     norm_median = np.linalg.norm(delta_median, axis=1)
#     norm_mode = np.linalg.norm(delta_mode, axis=1)
#     norm_weighted = np.linalg.norm(delta_weighted, axis=1)
    
#     # Overall ensemble error table (weighted equally across categories to avoid bias from imbalanced data)
#     ensemble_overall_data = {}
#     for threshold in error_thresholds:
#         # Calculate error proportion for each category, then average with equal weights for each ensemble method
#         category_errors_mean = []
#         category_errors_median = []
#         category_errors_mode = []
#         category_errors_weighted = []
        
#         for action_name, action_mask in action_types.items():
#             if action_mask.sum() > 0:  # Only include categories that exist in the data
#                 category_errors_mean.append(np.mean(norm_mean[action_mask] < threshold))
#                 category_errors_median.append(np.mean(norm_median[action_mask] < threshold))
#                 category_errors_mode.append(np.mean(norm_mode[action_mask] < threshold))
#                 category_errors_weighted.append(np.mean(norm_weighted[action_mask] < threshold))
        
#         # Weight each category equally (not by frequency) to avoid bias
#         ensemble_overall_data[f'Error < {threshold}'] = [
#             np.mean(category_errors_mean) if category_errors_mean else np.nan,
#             np.mean(category_errors_median) if category_errors_median else np.nan,
#             np.mean(category_errors_mode) if category_errors_mode else np.nan,
#             np.mean(category_errors_weighted) if category_errors_weighted else np.nan
#         ]

#     ensemble_overall_df = pd.DataFrame(ensemble_overall_data, index=['Mean', 'Median', 'Mode', 'Weighted'])
#     print_results_table(ensemble_overall_df, "Ensemble Overall Error Proportions (weighted equally across categories)")

#     # Per-action-type ensemble error tables
#     for action_name, action_mask in action_types.items():
#         ensemble_action_data = {}
#         for threshold in error_thresholds:
#             ensemble_action_data[f'Error < {threshold}'] = [
#                 np.mean(norm_mean[action_mask] < threshold) if action_mask.sum() > 0 else 0.0,
#                 np.mean(norm_median[action_mask] < threshold) if action_mask.sum() > 0 else 0.0,
#                 np.mean(norm_mode[action_mask] < threshold) if action_mask.sum() > 0 else 0.0,
#                 np.mean(norm_weighted[action_mask] < threshold) if action_mask.sum() > 0 else 0.0
#             ]
#         ensemble_action_df = pd.DataFrame(ensemble_action_data, index=['Mean', 'Median', 'Mode', 'Weighted'])
#         print_results_table(ensemble_action_df, f"Ensemble Error Proportions for {action_name} Actions (n={action_mask.sum()})")

#     # Ensemble category accuracy
#     pa_mean_categories = np.array([categorize_action(pa_mean[i,0], pa_mean[i,1]) for i in range(len(pa_mean))])
#     pa_median_categories = np.array([categorize_action(pa_median[i,0], pa_median[i,1]) for i in range(len(pa_median))])
#     pa_mode_categories = np.array([categorize_action(pa_mode[i,0], pa_mode[i,1]) for i in range(len(pa_mode))])
#     pa_weighted_categories = np.array([categorize_action(pa_weighted[i,0], pa_weighted[i,1]) for i in range(len(pa_weighted))])

#     ensemble_category_data = {'Overall': [
#         np.mean(oa_categories == pa_mean_categories),
#         np.mean(oa_categories == pa_median_categories),
#         np.mean(oa_categories == pa_mode_categories),
#         np.mean(oa_categories == pa_weighted_categories)
#     ]}

#     for cat_id, cat_name in category_names.items():
#         mask = oa_categories == cat_id
#         if mask.sum() > 0:
#             ensemble_category_data[cat_name] = [
#                 np.mean(oa_categories[mask] == pa_mean_categories[mask]),
#                 np.mean(oa_categories[mask] == pa_median_categories[mask]),
#                 np.mean(oa_categories[mask] == pa_mode_categories[mask]),
#                 np.mean(oa_categories[mask] == pa_weighted_categories[mask])
#             ]

#     ensemble_category_df = pd.DataFrame(ensemble_category_data, index=['Mean', 'Median', 'Mode', 'Weighted'])
#     print_results_table(ensemble_category_df, "Ensemble Category Classification Accuracy")

# print("\n" + "=" * 80)
# print("Analysis Complete!")
# print("=" * 80)