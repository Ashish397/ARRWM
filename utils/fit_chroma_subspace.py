"""Fit the decoder-visible CHROMA subspace of latent channel-means.

Lit-review-guided (chroma-only anchoring): regress per-chunk decoded
Cb/Cr means (NOT luminance Y) on per-chunk 16-channel latent means over
all recorded rollouts with videos. The row space of the fitted 2x16 map
is the chroma subspace; serve-time chroma-lock corrects latent means
only along its pseudo-inverse directions, leaving the ~14 motion/
luminance-carrying directions free.

Output: analysis/eval_final/flow_viz/chroma_subspace.npz
  Mc [2,16], b [2], plus fit R^2 per component.
"""
import glob, os
import numpy as np
import imageio.v2 as imageio

FV = "/scratch/u6ex/as1748.u6ex/ARRWM/analysis/eval_final/flow_viz"
RUNS = os.environ.get(
    "CS_RUNS",
    "pilot_gt0:pilot2_flip2:pilot3_flip2nr:pilot3_flip2nr_det:"
    "pilot3_flip2nr_inv:pilot3_flip2mts").split(":")
NFB, PXF = 3, 12          # latent frames per chunk; pixel frames per chunk


def rgb_to_cbcr_y(rgb):  # rgb in [0,1], returns (Y, Cb, Cr) means
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b
    return y.mean(), cb.mean(), cr.mean()


def main():
    X, Ycb, Ycr = [], [], []
    for run in RUNS:
        for d in sorted(glob.glob(f"{FV}/flow_{run}/r08_*")):
            tag = os.path.basename(d)                     # r08_F_s0
            vid = f"{FV}/.motion_check/{run}/{tag}.mp4"
            npz = f"{d}/steps.npz"
            if not (os.path.exists(vid) and os.path.exists(npz)):
                continue
            z = np.load(npz)
            sdt = z["sdt"]
            committed = {}
            for i, (c, r, t) in enumerate(sdt):
                if t > 0:
                    committed[int(c)] = i                 # last t>0 row per chunk wins
            rd = imageio.get_reader(vid)
            frames = np.stack([f for f in rd], 0).astype(np.float32) / 255.0
            rd.close()
            for c, i in committed.items():
                lat = z[f"x{i}"].astype(np.float32)       # [1, 3, 16, H, W]
                mu = lat.reshape(-1, NFB, 16, lat.shape[-2], lat.shape[-1]
                                 ).mean(axis=(0, 1, 3, 4))  # [16]
                lo = (1 + c) * PXF                        # skip 3-latent seed (12 px)
                blk = frames[lo:lo + PXF]
                if blk.shape[0] < PXF:
                    continue
                y, cb, cr = rgb_to_cbcr_y(blk)
                X.append(mu); Ycb.append(cb); Ycr.append(cr)
    X = np.stack(X); A = np.concatenate([X, np.ones((len(X), 1))], 1)
    print(f"[cs] fit on {len(X)} chunk samples from {len(RUNS)} runs")
    out, r2s = [], []
    for Y in (np.array(Ycb), np.array(Ycr)):
        w, *_ = np.linalg.lstsq(A, Y, rcond=None)
        pred = A @ w
        r2 = 1 - ((Y - pred) ** 2).sum() / ((Y - Y.mean()) ** 2).sum()
        out.append(w); r2s.append(r2)
    W = np.stack(out)                                     # [2, 17]
    Mc, b = W[:, :16], W[:, 16]
    print(f"[cs] R^2: Cb {r2s[0]:.3f}  Cr {r2s[1]:.3f}")
    np.savez(f"{FV}/chroma_subspace.npz", Mc=Mc, b=b,
             r2=np.array(r2s))
    print(f"[cs] saved {FV}/chroma_subspace.npz")


if __name__ == "__main__":
    main()
