import os
"""Paper figure: top-8 PCA components of the 10x10 CoTracker flow grid as
quiver fields, one panel per component, labelled with each component's
explained-variance share (from analysis/pca_evr.npy, measured on 300 rides /
115k chunks). No suptitle. Writes analysis/pca_components_flowfields.png.
"""
import numpy as np, torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CK = os.environ.get("PCA_BASIS", "preprocessing/checkpoints/pca_basis.pt")
GX = GY = 10


def main():
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    comp = np.asarray(ck["pca_comp"])            # [16,200]
    evr = np.load("analysis/pca_evr.npy")        # explained-variance ratios
    xs, ys = np.meshgrid(np.arange(GX), np.arange(GY))
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.4))
    for d in range(8):
        a = axes[d // 4][d % 4]
        fld = comp[d].reshape(GY, GX, 2)
        u, w = fld[..., 0], fld[..., 1]
        a.quiver(xs, ys, u, -w, angles="xy", scale_units="xy",
                 scale=np.abs(comp[d]).max() * 1.4, width=0.008, color="#1f77b4")
        a.set_title(f"PC{d}  ({100 * evr[d]:.1f}% var)", fontsize=12)
        a.set_xlim(-1, GX); a.set_ylim(GY, -1); a.set_aspect("equal")
        a.set_xticks([]); a.set_yticks([])
    fig.tight_layout()
    fig.savefig("analysis/pca_components_flowfields.png", dpi=160)
    print("saved analysis/pca_components_flowfields.png")


if __name__ == "__main__":
    main()
