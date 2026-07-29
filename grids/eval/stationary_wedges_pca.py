"""No-op drift wedges read through the frozen CoTracker->PCA teacher.

Drift is read with the same teacher the model was trained against: a 10x10 CoTracker
grid, mean per-frame flow over 12-frame chunks, projected onto the frozen PCA basis in
action_query/checkpoints/ss_vae_8free.pt. By that basis's convention PC0 is throttle
and PC1 is steer, so each scene gives one vector in the model's own action space:
x = PC1 (steer), y = PC0 (throttle). Vectors are taken RELATIVE to the real continuation
of the same scene, which removes the large common-mode offset of the basis (a static
scene projects to PC0 ~ -2 for every model, real footage included) and any
scene-specific content. A good no-op model has tiny wedges all round.

Display follows the fleet wedge convention -- non-parametric throughout, no fitted
variance model anywhere:

  pale translucent fill  raw per-scene density. One bar per rollout at alpha=0.05, all at
                         the same 36 deg width, so opacity accumulates where many scenes
                         land at a similar radius while a lone outlier stays nearly
                         invisible. It is a radial histogram, not a computed interval.
  solid outlined wedge   the median radius of the scenes in that sector, drawn unfilled on
                         top (zorder 4) so the density shows through.
  dashed arcs            the interquartile range: short arcs (+-14 deg) at the 25th and
                         75th percentile radii, drawn only where n >= 4.

Radius is |v| / vmax clipped at 1.0, with reference rings at 25/50/75/100% of the rim.
All panels share command units here, so radii ARE directly comparable across panels.

The rim is fixed rather than derived from a fleet-wide percentile, so that the panels of
interest stay legible: Yume and No AdaLN drift far enough that a percentile rim would
collapse every other model to a dot. Anything past the rim clips and the panel is labelled
as such -- those wedges are lower bounds, and the titles carry the true (unclipped) mean.
out/stationary_cotracker_pca.csv holds the raw values.

Unlike the commanded-response wedges there is no sign to encode by colour: direction is
carried by the sector angle itself, and the radius is already a magnitude.

Titles report the MEAN per-scene drift magnitude.
"""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "stationary_wedges_pca.png")
d = pd.read_csv(os.path.join(HERE, "out", "stationary_cotracker_pca.csv"))

disp = {"pca8": "Default", "pca4": "pca4", "pca2": "pca2", "16node": "Batch64",
        "4node": "Batch16", "noatok": "No Action Tokens", "noadaln": "No AdaLN",
        "minwm": "minWM", "worldplay": "WorldPlay", "matrixgame": "Matrix-Game",
        "worldcam": "WorldCam", "astra": "Astra", "yume": "Yume"}
order = ["pca4", "4node", "noatok", "pca2", "16node", "pca8", "noadaln",
         "minwm", "matrixgame", "worldplay", "worldcam", "yume", "astra"]
RIM = 30.0                            # outer rim of the wedges, in PCA drift units
NB = 8
BARW = np.deg2rad(36)                 # drawn wedge width, as in the fleet wedges
ARC = np.deg2rad(14)                  # IQR arc half-width
FILL, EDGE = "#2166ac", "#0b2f52"     # fill + darker outline variant

edges = np.linspace(-np.pi, np.pi, NB + 1)
theta = (edges[:-1] + edges[1:]) / 2

real = d[d.model == "real"].set_index("scene")[["pc0", "pc1"]]
m = d[d.model != "real"].join(real, on="scene", rsuffix="_r")
m["d0"] = m.pc0 - m.pc0_r             # throttle departure from the real continuation
m["d1"] = m.pc1 - m.pc1_r             # steer departure
m["mag"] = np.hypot(m.d1, m.d0)
m["sec"] = np.minimum(np.searchsorted(edges, np.arctan2(m.d0, m.d1)) - 1, NB - 1)

VMAX = RIM

ncol = 7
fig, axes = plt.subplots(2, ncol, figsize=(3.1 * ncol, 6.9), subplot_kw={"polar": True})
for ax, mm in zip(axes.ravel(), order):
    g = m[m.model == mm]
    clipped = bool((g.mag.values > VMAX).any())
    for s in range(NB):
        v = np.clip(g[g.sec == s].mag.values / VMAX, 0, 1.0)
        if not len(v):
            continue
        for r in v:                                                        # raw density
            ax.bar(theta[s], r, width=BARW, color=FILL, alpha=0.05, lw=0, zorder=2)
        ax.bar(theta[s], float(np.median(v)), width=BARW, fill=False,      # median
               edgecolor=EDGE, lw=1.6, zorder=4)
        if len(v) >= 4:                                                    # IQR arcs
            aa = np.linspace(theta[s] - ARC, theta[s] + ARC, 24)
            for q in (25, 75):
                ax.plot(aa, np.full_like(aa, np.percentile(v, q)), ls="--",
                        color=EDGE, lw=1.1, zorder=5)
    ax.set_title(f"{disp[mm]} — {g.mag.mean():.1f}" + ("  (clipped)" if clipped else ""),
                 fontsize=10)
    ax.set_xticks(np.deg2rad([0, 90, 180, 270]))
    ax.set_xticklabels(["+steer", "+throttle", "−steer", "−throttle"], fontsize=7)
    ax.set_rmax(1.0); ax.set_rticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["", "", "", f"{VMAX:.0f}"], fontsize=6)
    ax.grid(alpha=0.3)
for ax in axes.ravel()[len(order):]:
    ax.set_visible(False)
plt.tight_layout(); plt.savefig(OUT, dpi=110, bbox_inches="tight")
print(f"wrote {OUT}  (rim = {VMAX:.2f})")
