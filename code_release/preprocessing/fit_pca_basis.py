#!/usr/bin/env python3
"""Fit the frozen PCA egomotion basis on CoTracker motion frames.

Input per frame is the (100, 3) tracked-point field from motion.npy. We fit on
(dx, dy) only -- 200 dims (100 points x 2) -- and ignore visibility. The fit
yields N_COMPONENTS principal components, of which the top two are exposed to
the model as throttle and steer.

Run once to fit and save the basis; training loads the saved checkpoint and
never refits it.
"""

from __future__ import annotations

import os
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MOTION_ROOT = Path(os.path.join(os.environ.get("DATA_ROOT", ""), "frodobots_motion"))
N_POINTS = 100
N_FEAT = 3  # dx, dy, visibility (visibility ignored)
FLAT_DIM = N_POINTS * 2  # dx, dy only -> 200
# 16 components are fitted; the released basis (checkpoints/pca_basis.pt) has
# this shape. ACTION_DIMS of them are exposed to the model as the action vector,
# of which the top two are throttle and steer.
N_COMPONENTS = 16
ACTION_DIMS = 8
CHUNK_SIZE = 100_000
MAX_SAMPLES_DEFAULT = 12_000_000
CHECKPOINT_DIR = REPO_ROOT / "preprocessing" / "checkpoints"
DEFAULT_CKPT = CHECKPOINT_DIR / "pca_basis.pt"


def find_all_motion_files(motion_root=None):
    root = motion_root or MOTION_ROOT
    if not root.exists():
        raise FileNotFoundError(f"MOTION_ROOT does not exist: {root}")
    return sorted(root.glob("**/motion.npy"))


def get_valid_file_shapes(motion_files):
    valid = []
    N, feat = None, None
    for path in motion_files:
        try:
            motion = np.load(path, mmap_mode="r")
        except Exception:
            continue
        if motion.ndim != 3 or motion.shape[2] != 3:
            continue
        M, n_pts, three = motion.shape
        if N is None:
            N, feat = n_pts, three
        elif n_pts != N or three != feat:
            continue
        valid.append((path, M))
    return valid, (N, feat)


def load_chunk(valid_files, n_samples, rng):
    """Load one chunk of frames (S, N, 3), weighted by file size."""
    if not valid_files or n_samples <= 0:
        return None
    weights = np.array([M for _, M in valid_files], dtype=np.float64)
    weights /= weights.sum()
    n_per_file = rng.multinomial(n_samples, weights)
    out = []
    for (path, M), need in zip(valid_files, n_per_file):
        if need == 0 or M == 0:
            continue
        try:
            arr = np.load(path, mmap_mode="r")
        except Exception:
            continue
        if arr.shape[1] != N_POINTS or arr.shape[2] != N_FEAT:
            continue
        indices = rng.integers(0, M, size=min(need, M))
        rows = np.asarray(arr[indices], dtype=np.float32)
        rows = np.nan_to_num(rows, nan=0.0, posinf=0.0, neginf=0.0)
        out.append(rows)
    if not out:
        return None
    return np.concatenate(out, axis=0)[:n_samples]


def motion_to_flat(x):
    """(..., 100, 3) -> (..., 200). Uses dx, dy only; visibility ignored."""
    x = np.asarray(x, dtype=np.float64)
    return x[..., :, :2].reshape(*x.shape[:-2], FLAT_DIM)


def fit_pca(valid_files, max_samples, chunk_size, rng):
    """Load chunks into memory (up to max_samples), then mean + covariance + eigh. Pure NumPy.
    Fits on (dx, dy) only, 200 dims. PCA sees all data; no energy or standardise."""
    chunks = []
    n_seen = 0
    t0 = time.perf_counter()
    while n_seen < max_samples:
        take = min(chunk_size, max_samples - n_seen)
        chunk = load_chunk(valid_files, take, rng)
        if chunk is None or len(chunk) == 0:
            break
        flat = motion_to_flat(chunk)  # (S, 200)
        chunks.append(flat)
        n_seen += len(flat)
        print(f"  Loaded {n_seen} / {max_samples} frames ({time.perf_counter() - t0:.1f}s)")
    if not chunks:
        raise RuntimeError("No data loaded for PCA fit")
    data = np.concatenate(chunks, axis=0)[:max_samples]
    del chunks
    mean = data.mean(axis=0)
    centered = data - mean
    print("  Computing covariance...")
    cov = (centered.T @ centered) / max(centered.shape[0] - 1, 1)
    del centered, data
    print(f"  Eigen-decomposing ({FLAT_DIM}x{FLAT_DIM})...")
    w, v = np.linalg.eigh(cov)
    idx = np.argsort(w)[::-1][:N_COMPONENTS]
    components = v[:, idx].T
    # Variance explained (eigenvalues = variance along each component)
    total_var = w.sum()
    ev = w[idx]  # descending order
    ev_ratio = ev / total_var
    ev_ratio_cumsum = np.cumsum(ev_ratio)
    class PCAResult:
        mean_ = mean
        components_ = components
        explained_variance_ = ev
        explained_variance_ratio_ = ev_ratio
    return PCAResult(), ev_ratio, ev_ratio_cumsum


def transform(pca_mean, pca_components, x):
    """Transform motion to n_components. x: (..., 100, 3) -> (..., n_components). Uses dx, dy only; visibility ignored."""
    flat = motion_to_flat(x)  # (..., 200)
    prefix = flat.shape[:-1]
    flat = flat.reshape(-1, FLAT_DIM)
    centered = flat - pca_mean
    out = (centered @ pca_components.T)
    return out.reshape(*prefix, N_COMPONENTS)


def save_pca(out_path, pca):
    """Write the basis in the form the dataset and trainer load it.

    Key names match ``ZarrRideDataset``/``CausalDiffusionTeacherTrainer``:
    ``pca_mean`` (200,), ``pca_comp`` (N_COMPONENTS, 200), ``latent_ch``.
    The explained-variance ratio rides along so the PCA figure is
    reproducible from the same file.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    obj = {
        "pca_mean": torch.tensor(pca.mean_, dtype=torch.float32),
        "pca_comp": torch.tensor(pca.components_, dtype=torch.float32),
        "latent_ch": ACTION_DIMS,
    }
    if getattr(pca, "explained_variance_ratio_", None) is not None:
        obj["explained_variance_ratio"] = torch.tensor(
            pca.explained_variance_ratio_, dtype=torch.float32)
    torch.save(obj, out_path)
    print(f"Saved PCA basis to {out_path}")


def load_pca(ckpt_path):
    """Load the basis. Returns (mean, components, explained_variance_ratio)."""
    d = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    evr = d.get("explained_variance_ratio")
    return (np.asarray(d["pca_mean"]), np.asarray(d["pca_comp"]),
            None if evr is None else np.asarray(evr))


def main():
    parser = argparse.ArgumentParser(description="Fit PCA on motion: [100,3] -> n components. Save checkpoint for tsne_motion and kmeans_motion.")
    parser.add_argument("--max-samples", type=int, default=MAX_SAMPLES_DEFAULT,
                        help=f"Max frames to use for fitting (default {MAX_SAMPLES_DEFAULT})")
    parser.add_argument("--chunk-size", type=int, default=CHUNK_SIZE,
                        help=f"Frames per chunk (default {CHUNK_SIZE})")
    parser.add_argument("--out", type=Path, default=DEFAULT_CKPT,
                        help=f"Output .npz path (default {DEFAULT_CKPT})")
    parser.add_argument("--data-dir", type=Path, default=None,
                        help=f"Override motion data root (default {MOTION_ROOT})")
    parser.add_argument("--test", action="store_true",
                        help="After fitting, load and run a quick transform test")
    args = parser.parse_args()

    motion_root = args.data_dir or MOTION_ROOT

    print("Finding motion.npy files...")
    motion_files = find_all_motion_files(motion_root)
    print(f"  Found {len(motion_files)} files under {motion_root}")
    if not motion_files:
        print("No motion files found. Exiting.")
        sys.exit(1)

    valid_files, shape = get_valid_file_shapes(motion_files)
    if not valid_files or shape[0] is None:
        print("No valid motion files (need shape [M, 100, 3]). Exiting.")
        sys.exit(1)
    n_pts, feat = shape
    if n_pts != N_POINTS or feat != N_FEAT:
        print(f"Expected (N, feat) = ({N_POINTS}, {N_FEAT}), got ({n_pts}, {feat}). Exiting.")
        sys.exit(1)

    rng = np.random.default_rng(42)
    max_samples = min(args.max_samples, sum(M for _, M in valid_files))
    print(f"Fitting PCA (n_components={N_COMPONENTS}) on up to {max_samples} frames...")
    t0 = time.perf_counter()
    pca, ev_ratio, ev_ratio_cumsum = fit_pca(valid_files, max_samples, args.chunk_size, rng)
    elapsed = time.perf_counter() - t0
    print(f"Fit done in {elapsed:.1f}s")

    var_pct = 100.0 * ev_ratio_cumsum[-1]
    print(f"  Variance captured ({N_COMPONENTS} components): {var_pct:.2f}%")
    print("  Per-component explained variance ratio: ", np.round(ev_ratio, 4).tolist())
    print("  Cumulative:                            ", np.round(ev_ratio_cumsum, 4).tolist())

    save_pca(args.out, pca)

    if args.test:
        print("Running quick transform test...")
        mean, components, _ = load_pca(args.out)
        dummy = np.random.randn(4, N_POINTS, N_FEAT).astype(np.float32)
        out = transform(mean, components, dummy)
        assert out.shape == (4, N_COMPONENTS), out.shape
        out_n1 = out.reshape(4, N_COMPONENTS, 1)
        assert out_n1.shape == (4, N_COMPONENTS, 1)
        print(f"  transform batch (4, {N_POINTS}, {N_FEAT}) -> (4, {N_COMPONENTS}) ok")
    print("Done.")


if __name__ == "__main__":
    main()
