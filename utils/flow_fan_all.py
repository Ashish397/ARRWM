"""All-models flow-fan grid, wedge-style: one action-residual fan per model.

Reads the existing flow_viz/world_lats_{run}.npz caches (32 windows x 8 dirs
of eval videos, VAE re-encoded; minwm labels already swapped to TRUE dirs).
Per (window, frame) subtract the across-action mean latent -> residual
trajectories over world time that all start at the origin (shared real-seed
trunk). Latent dims differ across models (56320/99840/225280) so a joint PCA
basis is impossible; instead each model gets its own 2-PC residual basis,
then the panel is rotated so the mean F endpoint points UP and reflected so
the mean R endpoint has x>0 (PCA orientation is arbitrary), and residuals are
divided by the model's mean endpoint-residual norm. Panels therefore share
compass orientation, origin and scale -> fan SHAPE and coherence compare
directly, like the wedge grids.

Two modes (FF_MODE):
  global  (flow_fan_all.png)     one PCA plane per model over ALL windows'
          residuals. Faithful to raw latent geometry, but since same-action
          residual directions barely align across contexts (world_cosine diag
          0.05-0.2) the across-window mean fans nearly cancel -> panels read
          as diffuse clouds.
  aligned (flow_fan_aligned.png, default) each window's fan is projected onto
          its OWN 2-PC plane, oriented F-up / R-right and scale-normalized
          BEFORE overlaying/averaging. Shows the model's characteristic fan
          shape up to the per-context orientation that we know is arbitrary;
          bundle tightness = how consistently the action organizes the
          rollout within a context.

Fits the 4GB login-node cgroup: npz members are decompressed one window at a
time; the global-mode PCA basis comes from a Gram matrix over a size-capped
frame subsample (residuals sum to zero across actions, so no centering
needed); per-run results are cached (.fan*_cache_{run}.npz) so plot restyles
are instant.

Faint lines = per-window residual trajectories, bold = across-window mean
per action, * = mean endpoint. Panel title carries the raw mean endpoint
residual magnitude.

Writes flow_viz/flow_fan_all.png. Env: FF_RUNS colon list (def all found),
FF_OUT (def analysis/eval_final/flow_viz).
"""
import os, glob, zipfile
import numpy as np
from numpy.lib import format as npf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
FV = f"{ARR}/analysis/eval_final/flow_viz"
OUT = os.environ.get("FF_OUT", FV)

DNAMES = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
COLORS = {"F": "#1f77b4", "FR": "#17becf", "R": "#2ca02c", "BR": "#bcbd22",
          "B": "#d62728", "BL": "#e377c2", "L": "#9467bd", "FL": "#8c564b"}
ORDER = ["pca8_8node", "pca4", "pca2", "16node", "4node", "noatok", "noadaln",
         "minwm", "matrixgame", "worldcam", "yume", "worldplay", "astra"]
MAX_X_BYTES = 400e6                      # float16 PCA sample cap (4GB cgroup)


def npz_shapes(path):
    """Member shapes from the zip headers, no decompression."""
    shapes = {}
    with zipfile.ZipFile(path) as zf:
        for name in zf.namelist():
            with zf.open(name) as fh:
                version = npf.read_magic(fh)
                shape, _, _ = npf._read_array_header(fh, version)
            shapes[name[:-4]] = shape
    return shapes


def complete_windows(shapes):
    ws = sorted({int(k.rsplit("_", 1)[0][1:]) for k in shapes})
    wins = [w for w in ws if all(f"w{w}_{d}" in shapes for d in DNAMES)]
    Fl = min(shapes[k][0] for k in shapes)
    return wins, Fl


def window_residual(z, w, Fl):
    tr = np.stack([z[f"w{w}_{d}"][:Fl].astype(np.float32) for d in DNAMES])
    tr -= tr.mean(0, keepdims=True)
    return tr                                                   # [8, Fl, D]


def orient(win2d, wins):
    """Rotate mean F endpoint to +y, reflect so mean R endpoint has x>0."""
    fx, fy = np.mean([win2d[(w, "F")][-1] for w in wins], 0)
    th = np.pi / 2 - np.arctan2(fy, fx)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]], np.float32)
    win2d = {k: p @ R.T for k, p in win2d.items()}
    if np.mean([win2d[(w, "R")][-1][0] for w in wins]) < 0:
        for p in win2d.values():
            p[:, 0] *= -1
    return win2d


def aligned_fan_for_run(run):
    """Per-window PCA + per-window orientation/scale, then overlay."""
    cachef = f"{FV}/.fanw_cache_{run}.npz"
    if os.path.exists(cachef):
        c = np.load(cachef)
        win2d = {(int(k.rsplit("_", 1)[0][1:]), k.rsplit("_", 1)[1]): c[k]
                 for k in c.files if k.startswith("w")}
        return c["mean2d"], win2d, float(c["scale"]), int(c["nw"]), int(c["Fl"])

    f = f"{FV}/world_lats_{run}.npz"
    shapes = npz_shapes(f)
    wins, Fl = complete_windows(shapes)

    win2d, scales = {}, []
    z = np.load(f)
    for w in wins:
        tr = window_residual(z, w, Fl)                          # [8, Fl, D]
        Xw = tr.reshape(8 * Fl, -1)
        G = Xw @ Xw.T                                           # tiny [8F, 8F]
        lam, u = np.linalg.eigh(G)
        P = Xw.T @ (u[:, -2:][:, ::-1] / np.sqrt(np.maximum(lam[-2:][::-1], 1e-6)))
        p2 = (Xw @ P).reshape(8, Fl, 2)
        s = float(np.mean(np.linalg.norm(tr[:, -1], axis=-1)))
        w2 = orient({(w, d): p2[i] for i, d in enumerate(DNAMES)}, [w])
        for k, p in w2.items():
            win2d[k] = p / s
        scales.append(s)
        del tr, Xw

    mean2d = np.stack([np.mean([win2d[(w, d)] for w in wins], 0) for d in DNAMES])
    scale = float(np.mean(scales))
    np.savez_compressed(cachef, mean2d=mean2d, scale=scale, nw=len(wins), Fl=Fl,
                        **{f"w{w}_{d}": p for (w, d), p in win2d.items()})
    return mean2d, win2d, scale, len(wins), Fl


def fan_for_run(run):
    """-> (mean2d [8,F,2], win2d {(w,d)->[F,2]}, raw mean endpoint norm, nw, Fl)."""
    cachef = f"{FV}/.fan_cache_{run}.npz"
    if os.path.exists(cachef):
        c = np.load(cachef)
        win2d = {(int(k.rsplit("_", 1)[0][1:]), k.rsplit("_", 1)[1]): c[k]
                 for k in c.files if k.startswith("w")}
        return c["mean2d"], win2d, float(c["scale"]), int(c["nw"]), int(c["Fl"])

    f = f"{FV}/world_lats_{run}.npz"
    shapes = npz_shapes(f)
    wins, Fl = complete_windows(shapes)
    D = shapes[next(iter(shapes))][1]

    # frame subsample for the basis (always include the endpoint),
    # capped so the float16 sample matrix stays well under the cgroup limit
    per_traj = max(2, int(MAX_X_BYTES / (2 * D * len(wins) * 8)))
    fsel = np.unique(np.linspace(1, Fl - 1, min(per_traj, Fl - 1)).astype(int))

    # pass 1: subsampled residual rows (float16) -> Gram-trick top-2 basis
    n = len(wins) * 8 * len(fsel)
    X = np.empty((n, D), np.float16)
    z = np.load(f)
    for wi, w in enumerate(wins):
        blk = window_residual(z, w, Fl)[:, fsel].reshape(8 * len(fsel), -1)
        X[wi * 8 * len(fsel):(wi + 1) * 8 * len(fsel)] = blk
        del blk
    G = np.zeros((n, n), np.float32)
    step = 512
    for i in range(0, n, step):
        xi = X[i:i + step].astype(np.float32)
        for j in range(i, n, step):
            xj = X[j:j + step].astype(np.float32)
            g = xi @ xj.T
            G[i:i + xi.shape[0], j:j + xj.shape[0]] = g
            if j > i:
                G[j:j + xj.shape[0], i:i + xi.shape[0]] = g.T
        del xi
    lam, u = np.linalg.eigh(G)
    lam2, u2 = lam[-2:][::-1], u[:, -2:][:, ::-1]               # top-2
    P = np.zeros((X.shape[1], 2), np.float32)
    for i in range(0, n, step):
        P += X[i:i + step].astype(np.float32).T @ (u2[i:i + step] / np.sqrt(np.maximum(lam2, 1e-6)))
    del X, G

    # pass 2: project full residual trajectories, keep 2D + endpoint norms
    win2d, endnorm = {}, []
    z = np.load(f)
    for w in wins:
        tr = window_residual(z, w, Fl)
        for i, d in enumerate(DNAMES):
            win2d[(w, d)] = tr[i] @ P
            endnorm.append(float(np.linalg.norm(tr[i, -1])))
        del tr

    win2d = orient(win2d, wins)
    scale = float(np.mean(endnorm))
    win2d = {k: p / scale for k, p in win2d.items()}
    mean2d = np.stack([np.mean([win2d[(w, d)] for w in wins], 0) for d in DNAMES])
    np.savez_compressed(cachef, mean2d=mean2d, scale=scale, nw=len(wins), Fl=Fl,
                        **{f"w{w}_{d}": p for (w, d), p in win2d.items()})
    return mean2d, win2d, scale, len(wins), Fl


def main():
    mode = os.environ.get("FF_MODE", "aligned")
    runs = os.environ["FF_RUNS"].split(":") if "FF_RUNS" in os.environ else \
        [os.path.basename(f)[len("world_lats_"):-len(".npz")]
         for f in sorted(glob.glob(f"{FV}/world_lats_*.npz"))]
    runs = [r for r in ORDER if r in runs] + [r for r in runs if r not in ORDER]

    fans = {}
    for run in runs:
        fans[run] = aligned_fan_for_run(run) if mode == "aligned" else fan_for_run(run)
        print(f"[fan] {mode} {run}: {fans[run][3]} windows x {fans[run][4]} frames, "
              f"mean end-residual {fans[run][2]:.1f}", flush=True)

    style = os.environ.get("FF_STYLE", "mean")     # mean trajectories | ellipse
    if mode == "aligned":
        # per-window normalization puts endpoints at ~1; fixed limit for all
        lim = 1.6
    elif style == "ellipse":
        # global-mode means nearly cancel; scale axes to the endpoint scatter
        lim = 1.1 * max(np.percentile(np.abs(np.stack(
            [p[-1] for p in w2d.values()])), 98) for _, w2d, *_ in fans.values())
    else:
        lim = 1.15 * max(np.abs(m).max() for m, *_ in fans.values())
    n = len(runs)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.8 * ncol, 3.9 * nrow))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[n:]:
        ax.set_visible(False)

    faint_alpha = (0.16 if style == "ellipse" else 0.22) if mode == "aligned" else \
        (0.10 if style == "ellipse" else 0.14)
    from matplotlib.patches import Ellipse
    for ax, run in zip(axes, runs):
        mean2d, win2d, scale, nw, Fl = fans[run]
        for (w, d), p in win2d.items():
            ax.plot(p[:, 0], p[:, 1], color=COLORS[d], alpha=faint_alpha, lw=0.5)
        for i, d in enumerate(DNAMES):
            if style == "ellipse":
                E = np.stack([win2d[k][-1] for k in win2d if k[1] == d])
                c = E.mean(0)
                lam, U = np.linalg.eigh(np.cov(E.T))
                ang = np.degrees(np.arctan2(U[1, -1], U[0, -1]))
                ax.add_patch(Ellipse(c, *(2 * np.sqrt(np.maximum(lam[::-1], 0))),
                                     angle=ang, facecolor=COLORS[d], alpha=0.22,
                                     edgecolor=COLORS[d], lw=1.6, zorder=4,
                                     label=d if run == runs[0] else None))
                ax.scatter(E[:, 0], E[:, 1], color=COLORS[d], s=7, alpha=0.6, zorder=4)
                ax.scatter(c[0], c[1], color=COLORS[d], s=70, marker="*",
                           edgecolor="black", linewidth=0.5, zorder=5)
            else:
                ax.plot(mean2d[i, :, 0], mean2d[i, :, 1], color=COLORS[d], lw=2.4,
                        label=d if run == runs[0] else None)
                ax.scatter(mean2d[i, -1, 0], mean2d[i, -1, 1], color=COLORS[d], s=70,
                           marker="*", edgecolor="black", linewidth=0.5, zorder=5)
        ax.scatter([0], [0], color="black", s=18, zorder=6)
        ax.axhline(0, color="gray", lw=0.4)
        ax.axvline(0, color="gray", lw=0.4)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"{run}  (|res|={scale:.0f})", fontsize=11)

    fig.legend(*axes[0].get_legend_handles_labels(), ncol=8, loc="lower center",
               fontsize=11, frameon=False)
    if mode == "aligned":
        agg = ("dots = per-window endpoints, ellipse = mean +- 1 sigma per action"
               if style == "ellipse" else
               "bold = across-window mean per action, * = mean endpoint")
        fig.suptitle("Action-residual flow fans over world time — each window's fan in "
                     "its OWN 2-PC plane, oriented F-up / R-right and unit-scaled, then "
                     "overlaid (raw mean endpoint residual in title)\n"
                     f"faint = 32 windows each, {agg}; tight same-color clusters = action "
                     "organizes every context the same way (up to orientation)", fontsize=12)
        dst = f"{OUT}/flow_fan_aligned_ellipse.png" if style == "ellipse" \
            else f"{OUT}/flow_fan_aligned.png"
    else:
        agg = ("dots = per-window endpoints, ellipse = mean +- 1 sigma per action"
               if style == "ellipse" else
               "bold = across-window mean per action, * = mean endpoint")
        fig.suptitle("Action-residual flow fans over world time — per-model PCA of "
                     "(latent - across-action mean), all panels rotated F-up / R-right "
                     "and scaled by mean endpoint residual (shown in title)\n"
                     f"faint = 32 windows each, {agg}; tight same-color clusters = "
                     "consistent action operator", fontsize=12)
        dst = f"{OUT}/flow_fan_all_ellipse.png" if style == "ellipse" \
            else f"{OUT}/flow_fan_all.png"
    fig.tight_layout(rect=[0, 0.04, 1, 0.94])
    fig.savefig(dst, dpi=130)
    print(f"[fan] saved {dst}", flush=True)


if __name__ == "__main__":
    main()
