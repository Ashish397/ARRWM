#!/usr/bin/env python3
"""Noise analysis: secant decomposition of v12 rollout error.

Decomposes the prediction error e = L_a - L_g into:
  e_act   = component along the action-sensitive direction (secant from
            actual vs counterfactual predictions)
  e_noise = residual orthogonal to the action-sensitive direction

This uses ONLY precomputed artifacts — no extra model inference:
  - Actual-action predictions:       frodobots_lmdb/<ts>_w<idx>.pt
  - Counterfactual predictions:      frodobots_lmdb_counterfac/<ts>_w<idx>.pt
  - Ground-truth latents:            frodobots_encoded/<ts>.zarr
  - (Optional) noise zarrs:          frodobots_noise/<ts>.zarr

The counterfactual transform is: cf_z2 = -z2, cf_z7 = 1 - z7.
So delta_s = s_cf - s_a = [-2*z2, 1-2*z7] for each frame.

Each .pt contains a 7-step ODE trajectory [7, 21, 16, 60, 104] at steps
[0, 18, 36, 40, 44, 46, 48]. The final step (index 6, step 48) is pred_x0.

Modes:
  --mode offline   Latent metrics from precomputed noise zarrs vs GT (fast, no actions)
  --mode secant    Secant decomposition from actual + counterfactual .pt files

Secant decomposition (chunk 0):
  d_raw = L_c[0:3] - L_a[0:3]           raw counterfactual difference
  d = d_raw / (||delta_s|| + eps)        normalised secant direction
  alpha = <e, d> / (<d, d> + eps)        projection coefficient
  e_act = gamma * alpha * d              action-sensitive component (with shrinkage)
  e_noise = e - e_act                    residual / rollout noise

Action-sensitivity map from ODE trajectories:
  For each ODE step t, compute d_t = L_c^(t) - L_a^(t). Aggregate across
  steps with optional weighting (later steps = more resolved). Normalise to
  [0,1] to get a spatial-temporal mask S of "where action matters." Use S
  as weights for a weighted secant projection.

Caveats:
  - Counterfactual difference is not pure motion — it includes changed failure
    modes, different noise realisations in early ODE steps, etc.
  - The secant projection is an approximation. Shrinkage (gamma) and clipping
    guard against over-attribution.
  - Chunk 0 is the focus because it matches the true autoregressive rollout step.

Usage:
  python noise_analysis_backbone.py --mode offline --max_rides 5
  python noise_analysis_backbone.py --mode secant --gamma 0.5
  python noise_analysis_backbone.py --mode secant --gamma 0.5 --alpha_clip_pct 95
"""

import sys
import os
import argparse
import logging
import json
import numpy as np
import torch
from pathlib import Path
from collections import defaultdict

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (must match batch_eval_v12.py / eval_chain.py)
# ---------------------------------------------------------------------------
NUM_FRAMES = 21              # predicted frames per window (7 chunks)
NUM_FRAME_PER_BLOCK = 3      # frames per chunk
CONTEXT_FRAMES = 3           # clean context prepended (1 chunk)
NUM_CHUNKS = NUM_FRAMES // NUM_FRAME_PER_BLOCK  # 7
STREAM_LATENT_SPAN = CONTEXT_FRAMES + NUM_FRAMES  # 24

# ODE trajectory: 7 snapshots at these step indices
ODE_STEP_INDICES = [0, 18, 36, 40, 44, 46, 48]
FINAL_ODE_IDX = 6            # index into trajectory dim for step 48 (pred_x0)

GT_ROOT = "/projects/u6ex/fbots/frodobots_encoded"
NOISE_ROOT = "/projects/u6ex/fbots/frodobots_noise"
LMDB_ROOT = "/projects/u6ex/fbots/frodobots_lmdb"
LMDB_CF_ROOT = "/projects/u6ex/fbots/frodobots_lmdb_counterfac"
OUT_DIR = "vis/noise_analysis"


# ===================================================================
# Section 1: Latent-space metrics (no model needed)
# ===================================================================

def compute_latent_metrics(gt_window, pred_window):
    """Compare [F, C, H, W] latent tensors. Returns dict of scalars."""
    assert gt_window.shape == pred_window.shape
    diff = (pred_window.float() - gt_window.float())

    metrics = {
        "mse":       diff.pow(2).mean().item(),
        "mae":       diff.abs().mean().item(),
        "max_abs":   diff.abs().max().item(),
        "mse_per_channel": diff.pow(2).mean(dim=(0, 2, 3)).tolist(),
        "mse_per_frame": diff.pow(2).mean(dim=(1, 2, 3)).tolist(),
        "cosine_per_frame": [
            torch.nn.functional.cosine_similarity(
                gt_window[f].flatten().unsqueeze(0),
                pred_window[f].flatten().unsqueeze(0),
            ).item()
            for f in range(gt_window.shape[0])
        ],
        "snr_db": (10 * torch.log10(
            gt_window.float().pow(2).mean() / (diff.pow(2).mean() + 1e-10)
        )).item(),
    }
    chunk_mse = []
    for k in range(gt_window.shape[0] // NUM_FRAME_PER_BLOCK):
        s, e_ = k * NUM_FRAME_PER_BLOCK, (k + 1) * NUM_FRAME_PER_BLOCK
        chunk_mse.append(diff[s:e_].pow(2).mean().item())
    metrics["mse_per_chunk"] = chunk_mse
    return metrics


# ===================================================================
# Section 2: ODE trajectory analysis
# ===================================================================

def analyze_ode_trajectory(pt_data, gt_target):
    """Per-ODE-step error metrics. trajectory: [7,21,16,60,104]."""
    trajectory = pt_data["trajectory"]
    step_indices = pt_data["step_indices"]
    per_step = {}
    for i, step_idx in enumerate(step_indices):
        snap = trajectory[i].float()
        gt = gt_target.float()
        diff = snap - gt
        per_step[step_idx] = {
            "mse": diff.pow(2).mean().item(),
            "cosine": torch.nn.functional.cosine_similarity(
                snap.flatten().unsqueeze(0), gt.flatten().unsqueeze(0),
            ).item(),
        }
    return {"step_indices": step_indices, "per_step_metrics": per_step}


# ===================================================================
# Section 3: Action-sensitivity map from ODE trajectories
# ===================================================================

def build_action_sensitivity_map(traj_actual, traj_cf, step_weights=None):
    """Build spatial-temporal action-sensitivity map S from paired ODE trajectories.

    For each saved ODE step t, d_t = L_cf^(t) - L_actual^(t).
    S = weighted sum of |d_t| across steps, then normalised to [0, 1].

    Args:
        traj_actual: [7, 21, 16, 60, 104] float — actual-action ODE trajectory
        traj_cf:     [7, 21, 16, 60, 104] float — counterfactual ODE trajectory
        step_weights: optional [7] weights (default: uniform)

    Returns:
        S_full:   [21, 16, 60, 104] unnormalised sensitivity map
        S_normed: [21, 16, 60, 104] normalised to [0, 1]
        S_spatial: [21, 60, 104] channel-collapsed spatial map (mean over C)
    """
    n_steps = traj_actual.shape[0]
    if step_weights is None:
        step_weights = torch.ones(n_steps, dtype=torch.float32)
    step_weights = step_weights.float() / (step_weights.sum() + 1e-10)

    S = torch.zeros_like(traj_actual[0], dtype=torch.float32)
    for t in range(n_steps):
        d_t = (traj_cf[t].float() - traj_actual[t].float()).abs()
        S += step_weights[t] * d_t

    S_max = S.max()
    S_normed = S / (S_max + 1e-10)
    S_spatial = S_normed.mean(dim=1)  # collapse channels → [21, 60, 104]

    return S, S_normed, S_spatial


def build_later_weighted_step_weights():
    """Weights that emphasise later ODE steps (more resolved predictions).

    Steps: [0, 18, 36, 40, 44, 46, 48]
    Step 0 is pure noise (uninformative), later steps are increasingly resolved.
    """
    # Linearly increasing, step 0 gets 0 weight
    w = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=torch.float32)
    return w


# ===================================================================
# Section 4: Secant decomposition (chunk 0)
# ===================================================================

def compute_action_delta_chunk0(z_actual_chunk0):
    """Compute delta_s = s_cf - s_actual for chunk 0.

    Counterfactual transform: cf_z2 = -z2, cf_z7 = 1 - z7.
    So: delta_s_z2 = -z2 - z2 = -2*z2
        delta_s_z7 = (1 - z7) - z7 = 1 - 2*z7

    Args:
        z_actual_chunk0: [3, 2] or [2] actual actions for chunk 0
            index 0 = z2 (forward/back), index 1 = z7 (turn)

    Returns:
        delta_s: [2] action difference vector
        delta_s_norm: scalar norm
    """
    if z_actual_chunk0.dim() == 2:
        z_mean = z_actual_chunk0.float().mean(dim=0)  # average over 3 frames
    else:
        z_mean = z_actual_chunk0.float()

    delta_z2 = -2.0 * z_mean[0]
    delta_z7 = 1.0 - 2.0 * z_mean[1]
    delta_s = torch.stack([delta_z2, delta_z7])
    return delta_s, delta_s.norm()


def secant_projection_unweighted(e, d_raw, delta_s_norm, gamma=0.5, eps=1e-8):
    """Unweighted secant projection of error onto action-sensitive direction.

    d = d_raw / (||delta_s|| + eps)       normalised secant direction
    alpha = <e, d> / (<d, d> + eps)       raw projection coefficient
    alpha <- gamma * alpha                shrinkage
    e_act = alpha * d                     action-sensitive component
    e_noise = e - e_act                   residual

    All inputs/outputs are flat 1D tensors.
    """
    d = d_raw / (delta_s_norm + eps)

    d_dot_d = (d * d).sum()
    e_dot_d = (e * d).sum()
    alpha_raw = e_dot_d / (d_dot_d + eps)
    alpha = gamma * alpha_raw

    e_act = alpha * d
    e_noise = e - e_act

    return {
        "alpha_raw": alpha_raw.item(),
        "alpha": alpha.item(),
        "e_act": e_act,
        "e_noise": e_noise,
        "e_norm": e.norm().item(),
        "e_act_norm": e_act.norm().item(),
        "e_noise_norm": e_noise.norm().item(),
        "act_fraction": e_act.norm().item() / (e.norm().item() + eps),
        "noise_fraction": e_noise.norm().item() / (e.norm().item() + eps),
        "cosine_e_d": torch.nn.functional.cosine_similarity(
            e.unsqueeze(0), d.unsqueeze(0)).item() if d.norm() > eps else 0.0,
        "d_raw_norm": d_raw.norm().item(),
        "delta_s_norm": delta_s_norm.item(),
    }


def secant_projection_weighted(e, d_raw, delta_s_norm, S_chunk0, gamma=0.5, eps=1e-8):
    """Weighted secant projection using action-sensitivity map S as weights.

    Weighted inner product: <x, y>_S = sum(S * x * y)
    alpha = <e, d>_S / (<d, d>_S + eps)

    Args:
        e, d_raw: flat 1D tensors, same length as S_chunk0.flatten()
        S_chunk0: [3, 16, 60, 104] or flat, normalised sensitivity weights
    """
    S = S_chunk0.flatten().float()
    d = d_raw / (delta_s_norm + eps)

    d_dot_d_S = (S * d * d).sum()
    e_dot_d_S = (S * e * d).sum()
    alpha_raw = e_dot_d_S / (d_dot_d_S + eps)
    alpha = gamma * alpha_raw

    e_act = alpha * d
    e_noise = e - e_act

    return {
        "alpha_raw": alpha_raw.item(),
        "alpha": alpha.item(),
        "e_act": e_act,
        "e_noise": e_noise,
        "e_norm": e.norm().item(),
        "e_act_norm": e_act.norm().item(),
        "e_noise_norm": e_noise.norm().item(),
        "act_fraction": e_act.norm().item() / (e.norm().item() + eps),
        "noise_fraction": e_noise.norm().item() / (e.norm().item() + eps),
        "cosine_e_d": torch.nn.functional.cosine_similarity(
            e.unsqueeze(0), d.unsqueeze(0)).item() if d.norm() > eps else 0.0,
        "d_raw_norm": d_raw.norm().item(),
        "delta_s_norm": delta_s_norm.item(),
        "S_mean": S.mean().item(),
        "S_std": S.std().item(),
    }


# ===================================================================
# Section 5: Offline mode (pre-computed zarrs, no actions)
# ===================================================================

def find_matching_rides():
    gt_zarrs = {p.stem for p in Path(GT_ROOT).glob("*.zarr")}
    noise_zarrs = {p.stem for p in Path(NOISE_ROOT).glob("*.zarr")}
    common = sorted(gt_zarrs & noise_zarrs)
    log.info("Found %d GT, %d noise, %d overlap", len(gt_zarrs), len(noise_zarrs), len(common))
    return common


def process_ride_offline(ts):
    import zarr as zarr_lib
    gt_zarr = zarr_lib.open_group(os.path.join(GT_ROOT, f"{ts}.zarr"), mode="r")
    noise_zarr = zarr_lib.open_group(os.path.join(NOISE_ROOT, f"{ts}.zarr"), mode="r")

    gt_lat = gt_zarr["latents"]
    pred_lat = noise_zarr["latents"]
    n_gt, n_pred = gt_lat.shape[0], pred_lat.shape[0]
    city = noise_zarr.attrs.get("city", "unknown")
    n_windows = n_pred // NUM_FRAMES

    log.info("[%s] %s: %d GT, %d pred, %d windows", city, ts, n_gt, n_pred, n_windows)

    windows = []
    for w in range(n_windows):
        gt_offset = w * NUM_FRAMES
        gt_start = gt_offset + CONTEXT_FRAMES
        gt_end = gt_start + NUM_FRAMES
        if gt_end > n_gt:
            break
        p_start, p_end = w * NUM_FRAMES, (w + 1) * NUM_FRAMES

        gt_win = torch.from_numpy(gt_lat[gt_start:gt_end][:].astype(np.float32))
        pred_win = torch.from_numpy(pred_lat[p_start:p_end][:].astype(np.float32))
        lat_m = compute_latent_metrics(gt_win, pred_win)

        win_result = {"window_idx": w, "gt_offset": gt_offset, "latent_metrics": lat_m}

        pt_path = os.path.join(LMDB_ROOT, f"{ts}_w{w:04d}.pt")
        if os.path.exists(pt_path):
            pt_data = torch.load(pt_path, map_location="cpu", weights_only=False)
            win_result["ode_trajectory"] = analyze_ode_trajectory(pt_data, gt_win)

        windows.append(win_result)

    all_mse = [w["latent_metrics"]["mse"] for w in windows]
    all_snr = [w["latent_metrics"]["snr_db"] for w in windows]
    all_frame_mse = np.array([w["latent_metrics"]["mse_per_frame"] for w in windows])

    return {
        "ride_ts": ts, "city": city,
        "n_gt_frames": n_gt, "n_pred_frames": n_pred, "n_windows": len(windows),
        "aggregate": {
            "mean_mse": float(np.mean(all_mse)) if all_mse else 0,
            "std_mse": float(np.std(all_mse)) if all_mse else 0,
            "mean_snr_db": float(np.mean(all_snr)) if all_snr else 0,
            "avg_mse_per_frame": all_frame_mse.mean(axis=0).tolist() if len(all_frame_mse) else [],
            "avg_mse_per_chunk": [
                float(np.mean([w["latent_metrics"]["mse_per_chunk"][k] for w in windows]))
                for k in range(NUM_CHUNKS)
            ] if windows else [],
        },
        "windows": windows,
    }


def run_offline(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    rides = find_matching_rides()
    if args.max_rides:
        rides = rides[:args.max_rides]

    log.info("Processing %d rides (offline)...", len(rides))
    all_results = []
    for i, ts in enumerate(rides):
        try:
            result = process_ride_offline(ts)
            all_results.append(result)
            with open(os.path.join(OUT_DIR, f"{ts}.json"), "w") as f:
                json.dump(result, f, indent=2)
            if (i + 1) % 5 == 0 or (i + 1) == len(rides):
                log.info("Progress: %d/%d", i + 1, len(rides))
        except Exception as exc:
            log.warning("Failed %s: %s", ts, exc)
            import traceback; traceback.print_exc()

    if all_results:
        summary = {
            "n_rides": len(all_results),
            "cities": list(set(r["city"] for r in all_results)),
            "total_windows": sum(r["n_windows"] for r in all_results),
            "global_mean_mse": float(np.mean([r["aggregate"]["mean_mse"] for r in all_results])),
            "global_mean_snr_db": float(np.mean([r["aggregate"]["mean_snr_db"] for r in all_results])),
        }
        with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        log.info("Summary: MSE=%.6f SNR=%.2fdB (%d windows, %d rides)",
                 summary["global_mean_mse"], summary["global_mean_snr_db"],
                 summary["total_windows"], summary["n_rides"])


# ===================================================================
# Section 6: Secant mode
# ===================================================================

def find_paired_windows():
    """Find windows that exist in both normal and counterfactual LMDB dirs.

    Returns list of dicts with ts, window_idx, and paths to both .pt files.
    """
    normal_pts = {}
    for p in Path(LMDB_ROOT).glob("*.pt"):
        normal_pts[p.stem] = p

    paired = []
    for p in sorted(Path(LMDB_CF_ROOT).glob("*.pt")):
        stem = p.stem  # e.g. 20240316144555_w0003
        if stem in normal_pts:
            parts = stem.rsplit("_w", 1)
            ts = parts[0]
            widx = int(parts[1])
            paired.append({
                "ts": ts,
                "window_idx": widx,
                "normal_pt": str(normal_pts[stem]),
                "cf_pt": str(p),
            })

    log.info("Found %d paired windows across %d rides",
             len(paired), len(set(w["ts"] for w in paired)))
    return paired


def load_gt_chunk0(zarr_path, window_offset):
    """Load ground-truth target chunk 0: 3 frames starting at offset + CONTEXT_FRAMES."""
    import zarr as zarr_lib
    if "u6ej" in zarr_path:
        zarr_path = zarr_path.replace("/projects/u6ej/fbots/frodobots_encoded",
                                      "/projects/u6ex/fbots/frodobots_encoded")
    g = zarr_lib.open_group(zarr_path, mode="r")
    gt_start = window_offset + CONTEXT_FRAMES
    gt_end = gt_start + NUM_FRAME_PER_BLOCK
    return torch.from_numpy(g["latents"][gt_start:gt_end][:].astype(np.float32))


def load_actions_chunk0(zarr_path, n_lat, window_offset, z_ds, action_dims):
    """Load actual actions for chunk 0 via the ZarrRideDataset SS-VAE encoder.

    Returns z_actual_chunk0: [3, 2] (per-frame z2, z7 for chunk 0 of noisy side).
    """
    if "u6ej" in zarr_path:
        zarr_path = zarr_path.replace("/projects/u6ej/fbots/frodobots_encoded",
                                      "/projects/u6ex/fbots/frodobots_encoded")
    z_win = z_ds.encode_z_actions_window(
        zarr_path, n_lat, window_offset, window_offset + STREAM_LATENT_SPAN,
    )
    # Noisy side starts at CONTEXT_FRAMES; chunk 0 = first 3 frames of noisy side
    z_noisy_chunk0 = z_win[CONTEXT_FRAMES:CONTEXT_FRAMES + NUM_FRAME_PER_BLOCK, action_dims]
    return z_noisy_chunk0  # [3, 2]


def load_gt_full_window(zarr_path, window_offset):
    """Load full 21-frame GT target window starting at offset + CONTEXT_FRAMES."""
    import zarr as zarr_lib
    if "u6ej" in zarr_path:
        zarr_path = zarr_path.replace("/projects/u6ej/fbots/frodobots_encoded",
                                      "/projects/u6ex/fbots/frodobots_encoded")
    g = zarr_lib.open_group(zarr_path, mode="r")
    gt_start = window_offset + CONTEXT_FRAMES
    gt_end = gt_start + NUM_FRAMES
    if gt_end > g["latents"].shape[0]:
        return None
    return torch.from_numpy(g["latents"][gt_start:gt_end][:].astype(np.float32))


def process_window_secant(normal_pt_path, cf_pt_path, z_ds, action_dims,
                          gamma=0.5, use_weighted=True, save_vis_path=None):
    """Full secant decomposition for one paired window.

    Loads actual and counterfactual .pt files, GT from zarr, actions from SS-VAE.
    Computes chunk-0 unweighted and (optionally) weighted secant projections.

    If save_vis_path is set, saves a .pt tensor bundle for the vis script containing
    the full 21-frame latents (GT, actual, counterfactual) and chunk-0 decomposition.
    """
    d_normal = torch.load(normal_pt_path, map_location="cpu", weights_only=False)
    d_cf = torch.load(cf_pt_path, map_location="cpu", weights_only=False)

    zarr_path = d_normal["zarr_path"]
    offset = d_normal["window_offset"]
    n_lat = d_normal["n_latent_frames"]

    # Extract pred_x0 (final ODE step) for chunk 0
    # trajectory: [7, 21, 16, 60, 104], index 6 = step 48 = pred_x0
    L_a_chunk0 = d_normal["trajectory"][FINAL_ODE_IDX, :NUM_FRAME_PER_BLOCK].float()
    L_c_chunk0 = d_cf["trajectory"][FINAL_ODE_IDX, :NUM_FRAME_PER_BLOCK].float()

    # Ground truth chunk 0
    L_g_chunk0 = load_gt_chunk0(zarr_path, offset)

    # Actions for chunk 0
    z_actual_chunk0 = load_actions_chunk0(zarr_path, n_lat, offset, z_ds, action_dims)
    delta_s, delta_s_norm = compute_action_delta_chunk0(z_actual_chunk0)

    # Core vectors
    e = (L_a_chunk0 - L_g_chunk0).flatten()          # prediction error
    d_raw = (L_c_chunk0 - L_a_chunk0).flatten()       # counterfactual difference

    # --- Unweighted secant projection ---
    unw = secant_projection_unweighted(e, d_raw, delta_s_norm, gamma=gamma)

    result = {
        "ts": d_normal["ride_ts"],
        "city": d_normal["city"],
        "window_idx": d_normal["window_idx"],
        "window_offset": offset,
        "z2_chunk0": z_actual_chunk0[:, 0].mean().item(),
        "z7_chunk0": z_actual_chunk0[:, 1].mean().item(),
        "delta_s": delta_s.tolist(),
        "delta_s_norm": delta_s_norm.item(),
        "unweighted": {
            "alpha_raw": unw["alpha_raw"],
            "alpha": unw["alpha"],
            "act_fraction": unw["act_fraction"],
            "noise_fraction": unw["noise_fraction"],
            "cosine_e_d": unw["cosine_e_d"],
            "e_norm": unw["e_norm"],
            "e_act_norm": unw["e_act_norm"],
            "e_noise_norm": unw["e_noise_norm"],
            "d_raw_norm": unw["d_raw_norm"],
        },
    }

    # Per-channel and per-frame energy of e_noise (unweighted)
    e_noise_shaped = unw["e_noise"].reshape(NUM_FRAME_PER_BLOCK, 16, 60, 104)
    result["unweighted"]["e_noise_mse_per_channel"] = (
        e_noise_shaped.pow(2).mean(dim=(0, 2, 3)).tolist()
    )
    result["unweighted"]["e_noise_mse_per_frame"] = (
        e_noise_shaped.pow(2).mean(dim=(1, 2, 3)).tolist()
    )

    # Also store full-window latent metrics (pred_x0 vs GT, all 21 frames)
    L_a_full = d_normal["trajectory"][FINAL_ODE_IDX].float()
    gt_full_start = offset + CONTEXT_FRAMES
    gt_full_end = gt_full_start + NUM_FRAMES
    import zarr as zarr_lib
    zp = zarr_path
    if "u6ej" in zp:
        zp = zp.replace("/projects/u6ej/fbots/frodobots_encoded",
                         "/projects/u6ex/fbots/frodobots_encoded")
    g = zarr_lib.open_group(zp, mode="r")
    if gt_full_end <= g["latents"].shape[0]:
        L_g_full = torch.from_numpy(g["latents"][gt_full_start:gt_full_end][:].astype(np.float32))
        result["full_window_metrics"] = compute_latent_metrics(L_g_full, L_a_full)

    # --- Action-sensitivity map + weighted secant projection ---
    S_normed = None
    if use_weighted:
        traj_a = d_normal["trajectory"].float()   # [7, 21, 16, 60, 104]
        traj_c = d_cf["trajectory"].float()

        step_weights = build_later_weighted_step_weights()
        S_full, S_normed, S_spatial = build_action_sensitivity_map(
            traj_a, traj_c, step_weights=step_weights,
        )

        # Slice chunk 0
        S_chunk0 = S_normed[:NUM_FRAME_PER_BLOCK]  # [3, 16, 60, 104]

        wtd = secant_projection_weighted(
            e, d_raw, delta_s_norm, S_chunk0, gamma=gamma,
        )
        result["weighted"] = {
            "alpha_raw": wtd["alpha_raw"],
            "alpha": wtd["alpha"],
            "act_fraction": wtd["act_fraction"],
            "noise_fraction": wtd["noise_fraction"],
            "cosine_e_d": wtd["cosine_e_d"],
            "e_norm": wtd["e_norm"],
            "e_act_norm": wtd["e_act_norm"],
            "e_noise_norm": wtd["e_noise_norm"],
            "S_mean": wtd["S_mean"],
            "S_std": wtd["S_std"],
        }

        # Per-channel and per-frame energy of weighted e_noise
        e_noise_w_shaped = wtd["e_noise"].reshape(NUM_FRAME_PER_BLOCK, 16, 60, 104)
        result["weighted"]["e_noise_mse_per_channel"] = (
            e_noise_w_shaped.pow(2).mean(dim=(0, 2, 3)).tolist()
        )
        result["weighted"]["e_noise_mse_per_frame"] = (
            e_noise_w_shaped.pow(2).mean(dim=(1, 2, 3)).tolist()
        )

        # Sensitivity map summary stats
        result["sensitivity_map"] = {
            "S_full_mean": S_full.mean().item(),
            "S_full_std": S_full.std().item(),
            "S_chunk0_mean": S_chunk0.mean().item(),
            "S_chunk0_std": S_chunk0.std().item(),
            # Per-chunk sensitivity (which chunks are most action-sensitive?)
            "S_mean_per_chunk": [
                S_normed[k * NUM_FRAME_PER_BLOCK:(k + 1) * NUM_FRAME_PER_BLOCK].mean().item()
                for k in range(NUM_CHUNKS)
            ],
        }

    # ODE-step-level error analysis (optional)
    ode_per_step = {}
    for i, step_idx in enumerate(d_normal["step_indices"]):
        snap_a = d_normal["trajectory"][i, :NUM_FRAME_PER_BLOCK].float()
        diff_a = (snap_a - L_g_chunk0).flatten()
        ode_per_step[step_idx] = {
            "mse_vs_gt": diff_a.pow(2).mean().item(),
        }
        if use_weighted:
            snap_c = d_cf["trajectory"][i, :NUM_FRAME_PER_BLOCK].float()
            d_t = (snap_c - snap_a).flatten()
            ode_per_step[step_idx]["cf_diff_norm"] = d_t.norm().item()
    result["ode_per_step"] = ode_per_step

    # --- Save tensor bundle for vis script ---
    if save_vis_path is not None:
        L_a_full = d_normal["trajectory"][FINAL_ODE_IDX].float()  # [21, 16, 60, 104]
        L_c_full = d_cf["trajectory"][FINAL_ODE_IDX].float()
        L_g_full = load_gt_full_window(zarr_path, offset)

        vis_bundle = {
            # Full 21-frame latents (half precision to save space)
            "L_gt": L_g_full.half() if L_g_full is not None else None,
            "L_actual": L_a_full.half(),
            "L_counterfactual": L_c_full.half(),
            # Chunk-0 decomposition tensors [3, 16, 60, 104]
            "e_chunk0": e.reshape(NUM_FRAME_PER_BLOCK, 16, 60, 104).half(),
            "e_act_chunk0": unw["e_act"].reshape(NUM_FRAME_PER_BLOCK, 16, 60, 104).half(),
            "e_noise_chunk0": unw["e_noise"].reshape(NUM_FRAME_PER_BLOCK, 16, 60, 104).half(),
            "d_raw_chunk0": d_raw.reshape(NUM_FRAME_PER_BLOCK, 16, 60, 104).half(),
            # Sensitivity map (if computed)
            "S_normed": S_normed.half() if S_normed is not None else None,
            # Metadata
            "ts": d_normal["ride_ts"],
            "city": d_normal["city"],
            "window_idx": d_normal["window_idx"],
            "window_offset": offset,
            "zarr_path": zp,
            "act_fraction": unw["act_fraction"],
            "alpha": unw["alpha"],
            "gamma": gamma,
            "z2_chunk0": z_actual_chunk0[:, 0].mean().item(),
            "z7_chunk0": z_actual_chunk0[:, 1].mean().item(),
        }
        torch.save(vis_bundle, save_vis_path)
        log.info("  Saved vis bundle: %s", save_vis_path)

    return result


# ===================================================================
# Section 7: Global linear action-subspace fit (scaffold)
# ===================================================================

def fit_global_action_subspace(all_results, pca_dim=None):
    """Fit a linear map B such that d_raw ≈ B @ delta_s from many samples.

    Collects (delta_s, d_raw) pairs from all windows, optionally reduces
    d_raw via PCA first, then fits ridge regression from 2D delta_s to
    latent delta.

    This is a scaffold — the full implementation will need care with
    numerical stability and memory (each d_raw is ~300K floats).

    Args:
        all_results: list of per-window result dicts (must have delta_s, and
                     the normal/cf .pt paths for re-loading d_raw)
        pca_dim: if set, reduce d_raw to this dimensionality before fitting

    Returns:
        B: [D, 2] or [pca_dim, 2] linear map
        explained_variance: fraction of d_raw variance explained by B @ delta_s
    """
    # TODO: implement when we have enough data to make this meaningful.
    # Steps:
    #   1. Collect delta_s: [N, 2] and d_raw: [N, D] from all paired windows
    #   2. Optionally PCA d_raw to [N, pca_dim]
    #   3. Ridge regression: B = (X^T X + lambda I)^{-1} X^T Y
    #      where X = delta_s [N, 2], Y = d_raw [N, D]
    #   4. For each sample, d_hat = B @ delta_s, project e onto col(B)
    #   5. Return B and per-sample projections
    #
    # With ~665 paired windows and D = 3*16*60*104 = 299520, this is a
    # [665, 2] -> [665, 299520] regression. Straightforward with torch.linalg.lstsq
    # but may want PCA to e.g. 256 dims first for interpretability.
    log.info("Global action-subspace fit: scaffold only, not yet implemented.")
    return None, None


# ===================================================================
# Section 8: Secant mode main
# ===================================================================

def run_secant(args):
    import zarr as zarr_lib
    from omegaconf import OmegaConf
    from utils.zarr_dataset import ZarrRideDataset

    os.makedirs(OUT_DIR, exist_ok=True)

    cfg = OmegaConf.load("configs/causal_lora_diffusion_teacher.yaml")
    action_dims = list(cfg.get("action_dims", [2, 7]))
    motion_root = str(cfg.get("motion_root", ""))
    if "u6ej" in motion_root:
        motion_root = motion_root.replace("u6ej", "u6ex")
    ss_vae_ckpt = str(cfg.get("ss_vae_checkpoint", "action_query/checkpoints/ss_vae_8free.pt"))

    # Find paired windows
    paired = find_paired_windows()
    if not paired:
        log.error("No paired windows found. Check LMDB dirs.")
        return
    if args.max_windows:
        paired = paired[:args.max_windows]

    # Build ZarrRideDataset for action encoding (needs SS-VAE, runs on CPU/GPU)
    unique_zarps = {}
    for w in paired:
        d = torch.load(w["normal_pt"], map_location="cpu", weights_only=False)
        zp = d["zarr_path"]
        if "u6ej" in zp:
            zp = zp.replace("/projects/u6ej/fbots/frodobots_encoded",
                            "/projects/u6ex/fbots/frodobots_encoded")
        if zp not in unique_zarps:
            g = zarr_lib.open_group(zp, mode="r")
            unique_zarps[zp] = {
                "zarr_path": zp,
                "prompt_embeds": torch.zeros(1, 512, 4096),
                "attrs": dict(g.attrs),
                "n_latent_frames": g["latents"].shape[0],
            }

    ss_device = args.device if torch.cuda.is_available() else "cpu"
    z_ds = ZarrRideDataset.from_manifest(
        rides_data=list(unique_zarps.values()),
        motion_root=motion_root,
        ss_vae_checkpoint=ss_vae_ckpt,
        device="cpu",
        ss_vae_device=ss_device,
    )
    log.info("ZarrRideDataset ready (%d rides)", len(unique_zarps))

    vis_dir = os.path.join(OUT_DIR, "vis_bundles")
    if args.save_vis > 0:
        os.makedirs(vis_dir, exist_ok=True)

    log.info("Processing %d paired windows (gamma=%.2f, weighted=%s, save_vis=%d)...",
             len(paired), args.gamma, not args.no_weighted, args.save_vis)

    all_results = []
    for i, pw in enumerate(paired):
        try:
            save_path = None
            if i < args.save_vis:
                save_path = os.path.join(vis_dir, f"{pw['ts']}_w{pw['window_idx']:04d}.pt")

            result = process_window_secant(
                pw["normal_pt"], pw["cf_pt"],
                z_ds, action_dims,
                gamma=args.gamma,
                use_weighted=not args.no_weighted,
                save_vis_path=save_path,
            )
            all_results.append(result)

            if (i + 1) % 50 == 0 or (i + 1) == len(paired):
                recent = all_results[-min(50, len(all_results)):]
                avg_af = np.mean([r["unweighted"]["act_fraction"] for r in recent])
                log.info("Progress: %d/%d | recent avg act_fraction=%.4f", i + 1, len(paired), avg_af)

        except Exception as exc:
            log.warning("Failed %s w%d: %s", pw["ts"], pw["window_idx"], exc)
            import traceback; traceback.print_exc()

    if not all_results:
        log.error("No windows processed successfully.")
        return

    # --- Save per-window results ---
    out_path = os.path.join(OUT_DIR, "secant_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Per-window results saved to %s", out_path)

    # --- Aggregate statistics ---
    def _stats(vals):
        a = np.array(vals)
        return {
            "mean": float(np.mean(a)), "std": float(np.std(a)),
            "median": float(np.median(a)),
            "p5": float(np.percentile(a, 5)), "p25": float(np.percentile(a, 25)),
            "p75": float(np.percentile(a, 75)), "p95": float(np.percentile(a, 95)),
            "min": float(np.min(a)), "max": float(np.max(a)),
        }

    # Unweighted
    unw_af = [r["unweighted"]["act_fraction"] for r in all_results]
    unw_alpha_raw = [r["unweighted"]["alpha_raw"] for r in all_results]
    unw_alpha = [r["unweighted"]["alpha"] for r in all_results]
    unw_cosine = [r["unweighted"]["cosine_e_d"] for r in all_results]
    unw_e_norm = [r["unweighted"]["e_norm"] for r in all_results]
    unw_d_raw_norm = [r["unweighted"]["d_raw_norm"] for r in all_results]

    summary = {
        "n_windows": len(all_results),
        "gamma": args.gamma,
        "cities": list(set(r["city"] for r in all_results)),
        "unweighted": {
            "act_fraction": _stats(unw_af),
            "alpha_raw": _stats(unw_alpha_raw),
            "alpha": _stats(unw_alpha),
            "cosine_e_d": _stats(unw_cosine),
            "e_norm": _stats(unw_e_norm),
            "d_raw_norm": _stats(unw_d_raw_norm),
        },
    }

    # Alpha clipping stats (what fraction of alpha_raw is above p95?)
    if args.alpha_clip_pct:
        clip_val = np.percentile(np.abs(unw_alpha_raw), args.alpha_clip_pct)
        n_clipped = sum(1 for a in unw_alpha_raw if abs(a) > clip_val)
        summary["unweighted"]["alpha_clip"] = {
            "percentile": args.alpha_clip_pct,
            "clip_value": float(clip_val),
            "n_clipped": n_clipped,
            "frac_clipped": n_clipped / len(unw_alpha_raw),
        }

    # Weighted
    if not args.no_weighted and "weighted" in all_results[0]:
        wtd_af = [r["weighted"]["act_fraction"] for r in all_results]
        wtd_alpha_raw = [r["weighted"]["alpha_raw"] for r in all_results]
        wtd_alpha = [r["weighted"]["alpha"] for r in all_results]
        wtd_cosine = [r["weighted"]["cosine_e_d"] for r in all_results]

        summary["weighted"] = {
            "act_fraction": _stats(wtd_af),
            "alpha_raw": _stats(wtd_alpha_raw),
            "alpha": _stats(wtd_alpha),
            "cosine_e_d": _stats(wtd_cosine),
        }

    # Per-city breakdown
    by_city = defaultdict(list)
    for r in all_results:
        by_city[r["city"]].append(r)
    summary["per_city"] = {}
    for city, city_results in by_city.items():
        city_af = [r["unweighted"]["act_fraction"] for r in city_results]
        summary["per_city"][city] = {
            "n_windows": len(city_results),
            "act_fraction_mean": float(np.mean(city_af)),
            "act_fraction_std": float(np.std(city_af)),
        }

    # Sensitivity map per-chunk profile (averaged)
    if not args.no_weighted and "sensitivity_map" in all_results[0]:
        all_chunk_sens = np.array([
            r["sensitivity_map"]["S_mean_per_chunk"] for r in all_results
        ])
        summary["sensitivity_per_chunk"] = {
            "mean": all_chunk_sens.mean(axis=0).tolist(),
            "std": all_chunk_sens.std(axis=0).tolist(),
        }

    # Per-channel e_noise profile (averaged)
    all_ch = np.array([r["unweighted"]["e_noise_mse_per_channel"] for r in all_results])
    summary["e_noise_mse_per_channel"] = {
        "mean": all_ch.mean(axis=0).tolist(),
        "std": all_ch.std(axis=0).tolist(),
    }

    summary_path = os.path.join(OUT_DIR, "secant_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # --- Console report ---
    log.info("=" * 60)
    log.info("SECANT DECOMPOSITION SUMMARY (%d windows, gamma=%.2f)",
             summary["n_windows"], args.gamma)
    log.info("-" * 60)
    log.info("Unweighted secant projection:")
    log.info("  act_fraction:  %.4f +/- %.4f  (median %.4f, p5=%.4f, p95=%.4f)",
             summary["unweighted"]["act_fraction"]["mean"],
             summary["unweighted"]["act_fraction"]["std"],
             summary["unweighted"]["act_fraction"]["median"],
             summary["unweighted"]["act_fraction"]["p5"],
             summary["unweighted"]["act_fraction"]["p95"])
    log.info("  alpha_raw:     %.4f +/- %.4f  (median %.4f)",
             summary["unweighted"]["alpha_raw"]["mean"],
             summary["unweighted"]["alpha_raw"]["std"],
             summary["unweighted"]["alpha_raw"]["median"])
    log.info("  cosine(e, d):  %.4f +/- %.4f",
             summary["unweighted"]["cosine_e_d"]["mean"],
             summary["unweighted"]["cosine_e_d"]["std"])
    log.info("  e_norm:        %.4f +/- %.4f",
             summary["unweighted"]["e_norm"]["mean"],
             summary["unweighted"]["e_norm"]["std"])
    log.info("  d_raw_norm:    %.4f +/- %.4f",
             summary["unweighted"]["d_raw_norm"]["mean"],
             summary["unweighted"]["d_raw_norm"]["std"])

    if "weighted" in summary:
        log.info("Weighted secant projection (ODE sensitivity map):")
        log.info("  act_fraction:  %.4f +/- %.4f  (median %.4f)",
                 summary["weighted"]["act_fraction"]["mean"],
                 summary["weighted"]["act_fraction"]["std"],
                 summary["weighted"]["act_fraction"]["median"])
        log.info("  alpha_raw:     %.4f +/- %.4f",
                 summary["weighted"]["alpha_raw"]["mean"],
                 summary["weighted"]["alpha_raw"]["std"])

    for city, cs in summary.get("per_city", {}).items():
        log.info("  [%s] %d windows, act_fraction=%.4f +/- %.4f",
                 city, cs["n_windows"], cs["act_fraction_mean"], cs["act_fraction_std"])

    if "sensitivity_per_chunk" in summary:
        log.info("Sensitivity per chunk (avg): %s",
                 ["%.3f" % v for v in summary["sensitivity_per_chunk"]["mean"]])

    log.info("=" * 60)

    # --- Optional: global action-subspace fit scaffold ---
    if args.fit_global:
        fit_global_action_subspace(all_results)


# ===================================================================
# Section 9: Main
# ===================================================================

def main():
    parser = argparse.ArgumentParser(
        description="v12 noise analysis: secant decomposition from precomputed artifacts")
    parser.add_argument("--mode", choices=["offline", "secant"], default="offline",
                        help="offline=zarr metrics; secant=actual/cf decomposition")
    parser.add_argument("--max_rides", type=int, default=None)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="Shrinkage factor for secant projection (default 0.5)")
    parser.add_argument("--alpha_clip_pct", type=float, default=95.0,
                        help="Percentile for alpha clipping stats (default 95)")
    parser.add_argument("--no_weighted", action="store_true",
                        help="Skip weighted secant projection (ODE sensitivity map)")
    parser.add_argument("--save_vis", type=int, default=0,
                        help="Save tensor bundles for the first N windows (for vis_noise_secant.py)")
    parser.add_argument("--fit_global", action="store_true",
                        help="Run global linear action-subspace fit scaffold")
    parser.add_argument("--device", default="cuda:0",
                        help="Device for SS-VAE action encoding")
    args = parser.parse_args()

    if args.mode == "offline":
        run_offline(args)
    elif args.mode == "secant":
        run_secant(args)


if __name__ == "__main__":
    main()
