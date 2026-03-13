#!/usr/bin/env python3
"""PCA on motion frames: [100, 3] -> n components (output shape [n, 1] per sample).

Uses the same motion.npy data as beta_vae_motion. We fit on (dx, dy) only; visibility is ignored.
Input is (100, 3); internally we use 200 dims (100 points × 2).
Run this first to fit and save PCA; then use tsne_motion.py and kmeans_motion.py with the saved checkpoint.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
MOTION_ROOT = Path("/home/ashish/frodobots/frodobots_motion")
N_POINTS = 100
N_FEAT = 3  # dx, dy, visibility (visibility ignored)
FLAT_DIM = N_POINTS * 2  # dx, dy only -> 200
N_COMPONENTS = 12
CHUNK_SIZE = 100_000
MAX_SAMPLES_DEFAULT = 12_000_000
CHECKPOINT_DIR = REPO_ROOT / "action_query" / "checkpoints" / "pca_motion"
DEFAULT_CKPT = CHECKPOINT_DIR / "pca.npz"


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
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    kw = dict(
        mean=pca.mean_.astype(np.float32),
        components=pca.components_.astype(np.float32),
        n_components=N_COMPONENTS,
        input_shape=np.array([N_POINTS, N_FEAT]),  # input is (100,3); vis ignored
    )
    if hasattr(pca, "explained_variance_ratio_") and pca.explained_variance_ratio_ is not None:
        kw["explained_variance_ratio"] = pca.explained_variance_ratio_.astype(np.float32)
    if hasattr(pca, "explained_variance_") and pca.explained_variance_ is not None:
        kw["explained_variance"] = pca.explained_variance_.astype(np.float32)
    np.savez(out_path, **kw)
    print(f"Saved PCA to {out_path}")


def load_pca(npz_path):
    """Load PCA from .npz. Returns (mean, components, explained_variance).
    explained_variance may be None if not present in file."""
    data = np.load(npz_path)
    ev = data["explained_variance"] if "explained_variance" in data else None
    return data["mean"], data["components"], ev


def grid_energy(flat: np.ndarray) -> np.ndarray:
    """Per-sample energy: E = sum over 100 points of sqrt(dx^2 + dy^2). flat (n, 200) -> (n,)."""
    mag_per_pt = np.sqrt(flat[:, 0::2] ** 2 + flat[:, 1::2] ** 2)  # (n, 100)
    return mag_per_pt.sum(axis=1)


def motion_grid_type(flat: np.ndarray) -> np.ndarray:
    """Classify each grid: 0=left, 1=up, 2=right, 3=down, 4=converging, 5=diverging, 7=other (brown). Low energy (6=pink) set later from E."""
    n = flat.shape[0]
    V = flat.reshape(n, 10, 10, 2)
    mean_dx = flat[:, 0::2].mean(axis=1)
    mean_dy = flat[:, 1::2].mean(axis=1)
    gy = np.arange(10)
    gx = np.arange(10)
    cx, cy = 4.5, 4.5
    to_cell_x = gx - cx
    to_cell_y = gy - cy
    dx_cell = V[:, :, :, 0]
    dy_cell = V[:, :, :, 1]
    conv_score = (dx_cell * to_cell_x[np.newaxis, np.newaxis, :] +
                  dy_cell * to_cell_y[np.newaxis, :, np.newaxis]).sum(axis=(1, 2)) / 100.0
    out = np.full(n, -1, dtype=np.int32)
    t_dir = 0.02
    t_conv, t_div = 0.015, 0.015
    out[conv_score < -t_conv] = 4
    out[conv_score > t_div] = 5
    mask = out == -1
    out[mask] = 3
    out[mask & (mean_dx < -t_dir) & (np.abs(mean_dy) <= np.abs(mean_dx))] = 0
    out[mask & (mean_dy > t_dir) & (np.abs(mean_dx) <= np.abs(mean_dy))] = 1
    out[mask & (mean_dx > t_dir) & (np.abs(mean_dy) <= np.abs(mean_dx))] = 2
    out[mask & (mean_dy < -t_dir) & (np.abs(mean_dx) <= np.abs(mean_dy))] = 3
    out[out == -1] = 7   # other -> brown
    return out


def direction_from_visible(motion: np.ndarray, vis_thresh: float = 1.0, t_dir: float = 1.5) -> np.ndarray:
    """Left/right/up/down/other from mean dx,dy over points with visibility >= vis_thresh (default 1).

    motion: (n, 100, 3) with [:, :, 0]=dx, [:, :, 1]=dy, [:, :, 2]=visibility.
    Returns (n,) int32: 0=left (mean_dx < -t_dir), 2=right (mean_dx > t_dir),
    1=up (mean_dy > t_dir), 3=down (mean_dy < -t_dir), 7=other. Default t_dir=1.5. Priority: left, right, then up, down.
    """
    n = motion.shape[0]
    dx = np.asarray(motion[:, :, 0], dtype=np.float64)
    dy = np.asarray(motion[:, :, 1], dtype=np.float64)
    vis = np.asarray(motion[:, :, 2], dtype=np.float64) >= vis_thresh
    vis_count = np.maximum(vis.sum(axis=1), 1)
    mean_dx = (dx * vis).sum(axis=1) / vis_count
    mean_dy = (dy * vis).sum(axis=1) / vis_count
    out = np.full(n, 7, dtype=np.int32)  # other by default
    out[mean_dx < -t_dir] = 0   # left
    out[mean_dx > t_dir] = 2    # right
    out[(out == 7) & (mean_dy > t_dir)] = 1   # up (only if not already left/right)
    out[(out == 7) & (mean_dy < -t_dir)] = 3  # down
    return out


def direction_from_flat(flat: np.ndarray, t_dir: float = 10.0) -> np.ndarray:
    """Left/right/up/down/other from mean dx,dy over all grid points (no visibility filter).
    Use when only flat (n, 200) is available (e.g. when loading from cache).
    flat: (n, 200) with [:, 0::2]=dx, [:, 1::2]=dy. Returns same codes as direction_from_visible."""
    flat = np.asarray(flat, dtype=np.float64)
    mean_dx = flat[:, 0::2].mean(axis=1)  # (n,)
    mean_dy = flat[:, 1::2].mean(axis=1)
    out = np.full(flat.shape[0], 7, dtype=np.int32)
    out[mean_dx < -t_dir] = 0
    out[mean_dx > t_dir] = 2
    out[(out == 7) & (mean_dy > t_dir)] = 1
    out[(out == 7) & (mean_dy < -t_dir)] = 3
    return out


def smoothness_score_batch(flat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """flat (n, 200) -> (R_rel, M) each (n,). Normalized total variation; M is mean(||V||) per grid."""
    n = flat.shape[0]
    V = flat.reshape(n, 10, 10, 2)
    Dx = V[:, :, 1:, :] - V[:, :, :-1, :]   # (n, 10, 9, 2)
    Dy = V[:, 1:, :, :] - V[:, :-1, :, :]   # (n, 9, 10, 2)
    R = np.mean(np.linalg.norm(Dx, axis=-1), axis=(1, 2)) + np.mean(np.linalg.norm(Dy, axis=-1), axis=(1, 2))
    M = np.mean(np.linalg.norm(V, axis=-1), axis=(1, 2))
    R_rel = R / (M + 1e-8)
    return R_rel, M


def collect_transformed(valid_files, max_samples, chunk_size, rng, pca_mean, pca_components, n_components: int):
    """Stream chunks, transform with PCA; return (X, E, R_rel, M, motion_type, direction_visible, flat_all, n_collected).
    flat_all is (n, 200) raw motion for example-grid plotting."""
    chunks = []
    flat_chunks = []
    energy_chunks = []
    rrel_chunks = []
    mag_chunks = []
    motion_type_chunks = []
    direction_chunks = []
    n_seen = 0
    t0 = time.perf_counter()
    while n_seen < max_samples:
        take = min(chunk_size, max_samples - n_seen)
        chunk = load_chunk(valid_files, take, rng)
        if chunk is None or len(chunk) == 0:
            break
        flat = motion_to_flat(chunk)
        E = grid_energy(flat)
        R_rel, M = smoothness_score_batch(flat)
        motion_type_chunks.append(motion_grid_type(flat))
        direction_chunks.append(direction_from_visible(chunk, vis_thresh=1.0))
        flat_chunks.append(flat)
        centered = flat - pca_mean
        z = (centered @ pca_components.T)  # (S, n_components)
        chunks.append(z)
        energy_chunks.append(E)
        rrel_chunks.append(R_rel)
        mag_chunks.append(M)
        n_seen += len(z)
        if (n_seen // chunk_size) % 10 == 0 and n_seen >= chunk_size:
            print(f"  Transformed {n_seen} / {max_samples} frames ({time.perf_counter() - t0:.1f}s)")
    if not chunks:
        return (
            np.empty((0, n_components), dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty((0, FLAT_DIM), dtype=np.float64),
            0,
        )
    X = np.concatenate(chunks, axis=0)[:max_samples]
    E_all = np.concatenate(energy_chunks, axis=0)[:max_samples]
    R_rel_all = np.concatenate(rrel_chunks, axis=0)[:max_samples]
    M_all = np.concatenate(mag_chunks, axis=0)[:max_samples]
    motion_type_all = np.concatenate(motion_type_chunks, axis=0)[:max_samples]
    direction_all = np.concatenate(direction_chunks, axis=0)[:max_samples]
    flat_all = np.concatenate(flat_chunks, axis=0)[:max_samples]
    print(f"  Transformed {len(X)} frames ({time.perf_counter() - t0:.1f}s)")
    return X, E_all, R_rel_all, M_all, motion_type_all, direction_all, flat_all, len(X)


def whiten_pca(Z: np.ndarray, explained_variance: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale PCA coords so each component has unit variance (use for t-SNE / k-means)."""
    scale = np.sqrt(explained_variance + eps)
    return Z / scale


def unwhiten_pca(Z: np.ndarray, explained_variance: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Reverse whitening for centroid reconstruction."""
    scale = np.sqrt(explained_variance + eps)
    return Z * scale


def centroid_to_motion(centroid: np.ndarray, pca_mean: np.ndarray, pca_components: np.ndarray) -> np.ndarray:
    """Reconstruct (100, 2) motion from PCA-space centroid (n_components,)."""
    recon_flat = pca_mean + (centroid @ pca_components)
    return recon_flat.reshape(N_POINTS, 2)


def plot_example_grids_per_direction(
    flat: np.ndarray,
    direction: np.ndarray,
    out_dir: Path,
    n_examples: int = 9,
) -> None:
    """Plot up to n_examples (default 9) motion grids per direction; save to out_dir/example_<dir>.png.
    flat: (n, 200), direction: (n,) with 0=left, 1=up, 2=right, 3=down, 7=other."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
    except ImportError:
        print("  matplotlib not available, skipping example grids")
        return
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    g = 10
    names = {0: "left", 1: "up", 2: "right", 3: "down", 7: "other"}
    rng = np.random.default_rng(44)
    for dir_id, name in names.items():
        idx = np.where(direction == dir_id)[0]
        if len(idx) == 0:
            continue
        n_show = min(n_examples, len(idx))
        pick = rng.choice(idx, size=n_show, replace=False) if len(idx) >= n_show else idx
        motions = [flat[i].reshape(N_POINTS, 2) for i in pick]
        all_mag = np.concatenate([np.sqrt((m**2).sum(axis=1)) for m in motions])
        vmax = float(np.percentile(all_mag, 99)) or 1e-6
        norm = mcolors.Normalize(vmin=0.0, vmax=vmax)
        scale = vmax * 2.5
        rows = 3
        cols = 3
        fig, axes = plt.subplots(rows, cols, figsize=(9, 9))
        axes = axes.ravel()
        for i in range(rows * cols):
            if i < len(motions):
                motion_xy = motions[i]
                dx = motion_xy[:, 0].reshape(g, g)
                dy = motion_xy[:, 1].reshape(g, g)
                mag = np.sqrt(dx**2 + dy**2)
                Xg, Yg = np.meshgrid(np.arange(g), np.arange(g))
                axes[i].quiver(Xg, Yg, dx, dy, mag, scale=scale, scale_units="xy", cmap="viridis", norm=norm)
                axes[i].set_xlim(-0.5, g - 0.5)
                axes[i].set_ylim(g - 0.5, -0.5)
                axes[i].set_aspect("equal")
                axes[i].set_title(f"{name} #{i+1}")
            else:
                axes[i].axis("off")
        fig.suptitle(f"Example motion grids: {name}")
        fig.tight_layout()
        fig.savefig(out_dir / f"example_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {out_dir / f'example_{name}.png'}")


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

