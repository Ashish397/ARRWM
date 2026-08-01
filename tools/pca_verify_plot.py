"""Render the PCA-basis verification: shipped artifact vs a fresh refit.

Two questions, two panels — no dual axis:
  left   how closely each component direction reproduces (magnitude, 0..1)
  right  what the residual mean offset does to the action the model is
         conditioned on (polarity around zero)

    python tools/pca_verify_plot.py --shipped <pt> --refit <pt> --out <png>
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# Reference-palette categorical slots 1 and 2 (validated defaults).
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d8d8d4"
PCA_SCALES = np.array([93.7, 57.7, 22.5, 21.2, 18.1, 14.5, 12.6, 10.8])


def load(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    return (np.asarray(d["pca_mean"], np.float64),
            np.asarray(d["pca_comp"], np.float64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shipped", required=True)
    ap.add_argument("--refit", required=True)
    ap.add_argument("--motion-glob",
                    default="/projects/u6ex/fbots/frodobots_motion/*/*/motion.npy")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ma, Ca = load(a.shipped)
    mb, Cb = load(a.refit)
    n = min(8, len(Ca))
    cos = np.array([abs(Ca[i] @ Cb[i] /
                        (np.linalg.norm(Ca[i]) * np.linalg.norm(Cb[i])))
                    for i in range(n)])

    files = sorted(glob.glob(a.motion_glob))
    m = np.load(files[0], mmap_mode="r")[:2000]
    flat = np.asarray(m[:, :, :2], np.float64).reshape(len(m), 200)
    sign = np.sign(np.sum(Ca[:n] * Cb[:n], axis=1))
    za = np.tanh(((flat - ma) @ Ca.T)[:, :n] / PCA_SCALES[:n])
    zb = np.tanh(((flat - mb) @ Cb.T)[:, :n] / PCA_SCALES[:n]) * sign
    mae = np.abs(za - zb).mean(axis=0)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    names = [f"PC{i}" for i in range(n)]
    names[0] += "\nthrottle"
    names[1] += "\nsteer"

    # left: how well each direction reproduces. Zoomed to where the signal is.
    ax[0].bar(names, cos, color=BLUE, width=.62)
    ax[0].set_ylim(0.995, 1.0005)
    ax[0].axhline(1.0, color=MUTED, lw=1, ls=":")
    ax[0].set_ylabel("|cosine| vs shipped component")
    ax[0].set_title("Component directions reproduce", color=INK, fontsize=12)
    for i, v in enumerate(cos):                       # selective direct labels
        if i < 2 or v < 0.999:
            ax[0].text(i, min(v, 1.0) + 2e-5, f"{v:.5f}", ha="center",
                       fontsize=8, color=MUTED)

    # right: the consequence of the mean offset, in action units
    bars = ax[1].bar(names, mae, color=ORANGE, width=.62)
    ax[1].set_ylabel("MAE in squashed action units")
    ax[1].set_title("Residual mean offset shifts throttle", color=INK, fontsize=12)
    ax[1].text(0, mae[0], f" {mae[0]:.3f}", va="bottom", ha="center",
               fontsize=9, color=INK)
    bars[0].set_edgecolor(INK)
    bars[0].set_linewidth(1.2)

    for x in ax:
        x.grid(axis="y", color=GRID, lw=.8, alpha=.9)
        x.set_axisbelow(True)
        for s in ("top", "right"):
            x.spines[s].set_visible(False)
        x.tick_params(labelsize=9, colors=MUTED)

    fig.suptitle(
        f"PCA basis: shipped artifact vs fresh refit   "
        f"(||mean|| {np.linalg.norm(ma):.2f} vs {np.linalg.norm(mb):.2f})",
        fontsize=11, color=INK)
    fig.tight_layout()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.savefig(a.out, dpi=150)
    print(f"saved {a.out}")
    print(f"  min |cos| over top-{n}: {cos.min():.6f}")
    print(f"  throttle MAE {mae[0]:.4f} | steer MAE {mae[1]:.4f} | all-{n} {mae.mean():.4f}")


if __name__ == "__main__":
    main()
