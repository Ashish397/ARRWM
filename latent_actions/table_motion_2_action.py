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
print("Loading predictions for ensemble analysis...")
print("=" * 80)

# Load all model predictions
# models = ['kappa', 'kappa', 'kappa', 'kappa']
# steps = [4000, 4333, 5000, 6000]

# models = ['lambda', 'lambda', 'lambda', 'lambda']
# steps = [4000, 4121, 5000, 6000]

# models = ['mu', 'mu', 'mu', 'mu']
# steps = [4000, 4537, 5000, 6000]

models = ['phi', 'chi', 'psi', 'omega']
steps = [9203, 9000, 9866, 9000]
noises = [0.02, 0.02, 0.05, 0.02]
# uncertainty_thresholds = [0.1, 0.2, 0.3, 0.4]

# model_labels = ['A', 'B', 'C', 'D']

# models = ['xi']
# steps = [8244]
# noises = [0.05]
# uncertainty_thresholds = [0.1]

model_labels = ['A', 'B', 'C', 'D']
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
        # Distribution mode
        predicted_log_std_dict[key] = np.array([(row['predicted_log_std_linear'], row['predicted_log_std_angular']) for _, row in df.iterrows()])
        predicted_std_dict[key] = np.array([(row['predicted_std_linear'], row['predicted_std_angular']) for _, row in df.iterrows()])
        predicted_unbounded_mean_dict[key] = np.array([(row['predicted_unbounded_mean_linear'], row['predicted_unbounded_mean_angular']) for _, row in df.iterrows()])
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