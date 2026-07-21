"""Per-latent flow-field visualization for the SS-VAE (PCA+SS_VAE) encoder.

The SS-VAE action is z = tanh(mu) where mu = encoder(flow/scale). To show what
each latent dim looks like, we traverse one dim: set mu = onehot(d)*atanh(v) for
v in {-1,-0.5,0,0.5,1}, decode to a flow field, un-scale, and subtract the z=0
decode so only that dim's effect remains (the SS-VAE analog of dropping the mean).
One image per latent dim, arrows sized like the PCA figures.
"""
import os
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from action_query.ss_vae_model import load_ss_vae

CK = "action_query/checkpoints/ss_vae_8free.pt"
VALUES = [-1.0, -0.5, 0.0, 0.5, 1.0]
GX = GY = 10
QSCALE = 6.0                       # same as PCA figures (10x smaller than original)
OUT = "analysis/ssvae_viz"
N_SHOW = 8                         # action dims


def main():
    os.makedirs(OUT, exist_ok=True)
    model, scale = load_ss_vae(CK, device="cpu")
    model.eval()
    lat = model.latent_ch
    print(f"SS-VAE: latent_ch={lat} n_sup={model.n_sup} scale={scale:.4f}")
    xs, ys = np.meshgrid(np.arange(GX), np.arange(GY))

    @torch.no_grad()
    def decode(muvec):
        z = torch.tensor(muvec, dtype=torch.float32).reshape(1, lat, 1, 1)
        xhat = model.decoder(z)                       # [1,2,10,10]
        return (xhat.squeeze(0).permute(1, 2, 0).numpy()) * scale   # [10,10,2], un-scaled

    base = decode(np.zeros(lat))                      # z=0 baseline (subtract, like PCA no-mean)

    # SS-VAE dims span a ~100x magnitude range, so scale each dim's figure to its
    # own max arrow (~3 grid units). Arrow lengths are therefore NOT comparable
    # across dims; the per-dim peak magnitude is printed in each title.
    def dim_field(d, v):
        mu = np.zeros(lat); mu[d] = np.arctanh(np.clip(v, -0.995, 0.995))
        return decode(mu) - base

    for d in range(min(N_SHOW, lat)):
        peak = max(np.sqrt((dim_field(d, v) ** 2).sum(-1)).max() for v in VALUES if v != 0.0)
        qscale = max(peak / 3.0, 1e-4)
        fig, axz = plt.subplots(1, 5, figsize=(22, 4.6))
        for j, v in enumerate(VALUES):
            a = axz[j]
            fld = dim_field(d, v)                      # isolate dim d's effect
            u, w = fld[..., 0], fld[..., 1]
            mag = np.sqrt(u ** 2 + w ** 2)
            a.quiver(xs, ys, u, -w, mag, angles="xy", scale_units="xy", scale=qscale,
                     cmap="viridis", width=0.010)
            a.set_title(f"action = {v:+.1f}", fontsize=13)
            a.set_xlim(-1, GX); a.set_ylim(GY, -1); a.set_aspect("equal")
            a.set_xticks([]); a.set_yticks([])
        role = "supervised" if d < model.n_sup else "free"
        fig.suptitle(f"SS-VAE latent z{d} ({role})   |   decoded flow vs action value (z=0 baseline removed; peak |flow|={peak:.2f}, arrows scaled per-dim)",
                     fontsize=14)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        p = f"{OUT}/z{d}.png"
        fig.savefig(p, dpi=130); plt.close(fig)
        print(f"saved {p}")
    print(f"done -> {OUT}/z0..z{min(N_SHOW,lat)-1}.png")


if __name__ == "__main__":
    main()
