"""Two stationarity compasses from the RANSAC similarity decomposition:
  FB/LR   : forward-back (scale) vs left-right (pan)  -> out/stationary_wedges_fblr.png
  Roll/UD : roll (rotation) vs up-down (pan)          -> out/stationary_wedges_rollud.png
Same style as stationary_wedges.py (green, no title, noadaln in / real out, astra/yume/noadaln clipped)."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
d = pd.read_csv(os.path.join(HERE, "out", "stationary_cotracker.csv"))
disp = {"pca8": "Default", "pca4": "pca4", "pca2": "pca2", "16node": "Batch64", "4node": "Batch16",
        "noatok": "No Action Tokens", "noadaln": "No AdaLN", "minwm": "minWM", "worldplay": "WorldPlay",
        "matrixgame": "Matrix-Game", "worldcam": "WorldCam", "astra": "Astra", "yume": "Yume"}
order = ["pca4", "4node", "noatok", "pca2", "16node", "pca8", "noadaln",
         "minwm", "matrixgame", "worldplay", "worldcam", "yume", "astra"]
CLIP = {"astra", "noadaln"}          # clipped; Yume defines the rim (max)
NB = 8
edges = np.linspace(-np.pi, np.pi, NB + 1)
theta = (edges[:-1] + edges[1:]) / 2


def sector_r(m, xcol, ycol, flip_y):
    g = d[d.model == m]
    x = g[xcol].values.astype(float)
    y = (-g[ycol].values if flip_y else g[ycol].values).astype(float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    ang = np.arctan2(y, x); mag = np.hypot(x, y)
    r = np.zeros(NB)
    for a, mg in zip(ang, mag):
        r[min(np.searchsorted(edges, a) - 1, NB - 1)] += mg
    return r


def make(xcol, ycol, flip_y, labels, magcol_fn, out):
    rs = {m: sector_r(m, xcol, ycol, flip_y) for m in order}
    RMAX = rs["yume"].max() * 1.02          # Yume is the rim; anything taller is clipped
    fig, axs = plt.subplots(2, 7, figsize=(22, 7.6), subplot_kw={"projection": "polar"})
    axf = axs.ravel()
    for ax, m in zip(axf, order):
        r = np.clip(rs[m], 0, RMAX)
        ax.bar(theta, r, width=2 * np.pi / NB, color="#2E8B57", alpha=1.0, edgecolor="k", lw=0.5)
        clip = "  (clipped)" if rs[m].max() > RMAX else ""
        ax.set_title(f"{disp[m]} — {magcol_fn(m):.1f}px{clip}", fontsize=10, pad=14)
        ax.set_xticks(np.deg2rad([0, 90, 180, 270])); ax.set_xticklabels(labels, fontsize=8)
        ax.set_yticklabels([]); ax.set_rmax(RMAX); ax.set_rticks([RMAX / 2, RMAX]); ax.grid(alpha=0.3)
    for ax in axf[len(order):]:
        ax.axis("off")
    plt.tight_layout(); plt.savefig(os.path.join(HERE, "out", out), dpi=110, bbox_inches="tight")
    print("wrote", out, f"(RMAX={RMAX:.0f})")


# median magnitude in the plotted plane, per model, for the panel title
def med_fblr(m):
    g = d[d.model == m]; return float(np.nanmedian(np.hypot(g.drift_dx, g.forward)))
def med_rollud(m):
    g = d[d.model == m]; return float(np.nanmedian(np.hypot(g.roll, g.drift_dy)))


make("drift_dx", "forward", False, ["R", "F", "L", "B"], med_fblr, "stationary_wedges_fblr.png")
make("roll", "drift_dy", True, ["CW", "U", "CCW", "D"], med_rollud, "stationary_wedges_rollud.png")
