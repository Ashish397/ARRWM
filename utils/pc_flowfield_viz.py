"""Per-PC flow-field visualization for the paper.

For each of the top-8 PCA components, reconstruct the 10x10 CoTracker flow grid
at action values v in {-1,-0.5,0,0.5,1}. The action is z = tanh(P/scale), so the
raw PCA coordinate for a given action value is P = scale*atanh(v); the flow field
is mean + P*component_d (only that PC active). One image per PC.
"""
import os
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

CK = "action_query/checkpoints/ss_vae_8free.pt"
SCALES = [93.7, 57.7, 22.5, 21.2, 18.1, 14.5, 12.6, 10.8]
# dominant physical DOF per PC (from divergence/curl/net-flow signatures)
LABELS = {
    0: "throttle / forward (flow divergence)",
    1: "steer / yaw (horizontal flow)",
    2: "pitch / vertical mode A",
    3: "pitch / vertical mode B",
    4: "mixed / weak",
    5: "pitch + zoom-out",
    6: "roll (flow curl)",
    7: "pitch (strongest vertical)",
}
VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0]
GX = GY = 10
OUT = "analysis/pc_viz"


def main():
    os.makedirs(OUT, exist_ok=True)
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    comp = np.asarray(ck["pca_comp"])        # [16,200]
    mean = np.asarray(ck["pca_mean"])        # [200]
    xs, ys = np.meshgrid(np.arange(GX), np.arange(GY))

    # common quiver scale so arrow lengths are comparable across values & PCs.
    # QSCALE larger => shorter arrows; 10x smaller than the original 0.6.
    QSCALE = 6.0
    for d in range(8):
        fig, axz = plt.subplots(1, 5, figsize=(22, 4.6))
        for j, v in enumerate(VALUES):
            a = axz[j]
            P = SCALES[d] * np.arctanh(np.clip(v, -0.995, 0.995))   # raw coord for action v
            flat = P * comp[d]                                     # ONLY PC d (no mean added)
            fld = flat.reshape(GY, GX, 2)
            u, w = fld[..., 0], fld[..., 1]
            mag = np.sqrt(u ** 2 + w ** 2)
            a.quiver(xs, ys, u, -w, mag, angles="xy", scale_units="xy", scale=QSCALE,
                     cmap="viridis", width=0.010, clim=(0, np.percentile(np.abs(comp[d]).reshape(-1) * SCALES[d] * 2.6, 95) + 1e-6))
            a.set_title(f"action = {v:+.1f}", fontsize=13)
            a.set_xlim(-1, GX); a.set_ylim(GY, -1); a.set_aspect("equal")
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle(f"PC{d} — {LABELS[d]}   (var-scale {SCALES[d]})   |   flow grid vs action value",
                     fontsize=15)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        p = f"{OUT}/PC{d}.png"
        fig.savefig(p, dpi=130); plt.close(fig)
        print(f"saved {p}")
    print(f"done -> {OUT}/PC0..PC7.png")


if __name__ == "__main__":
    main()
