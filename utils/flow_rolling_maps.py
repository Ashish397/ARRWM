"""Optical-flow MAPS for the rolling-training runs' wandb rollout clips.

The latent-space flow suite (flow_fan_all.py FF_STYLE=ellipse, the preferred
aligned+ellipse fans) needs 8-action battery recordings + a GPU/VAE, which
the training-time wandb clips don't have. This adapts the suite's existing
VIDEO path (motion_check.py / r08_robust_motion.py Farneback conventions) to
render per-clip flow maps from pred_image_rollout mp4s, keeping the
aligned+ellipse presentation: every panel shares grid, arrow gain and color
scale, and each grid cell carries a 1-sigma temporal covariance ELLIPSE of
its flow (how steady that cell's motion is over the rollout).

Per clip -> analysis/flow_rolling/{run}_{step}.png with 3 panels:
  1) flow map: per-cell mean Farneback flow arrows over a faint mid-clip
     frame, colored by |flow| (shared vmax), + 1-sigma ellipses at arrow tips
  2) radial profile: mean radial flow v_r vs radius, split top/bottom half,
     + pure-zoom fit v_r = a*r with free center. zoomR2 -> 1 AND
     bottom/top ratio -> 1 = degenerate 2D-zoom signature; healthy forward
     egomotion keeps depth structure (ground streams faster than sky)
  3) time series: per-frame divergence fwd measure (motion_check formula),
     mean horizontal flow (steer) and mean |flow| across the rollout

Also writes flow_rolling_grid.png (runs x steps contact sheet of panel-1
maps, the aligned grid) and flow_rolling_metrics.csv.

fwd/steer definitions match motion_check.py (radial expansion +, scene-left
+). Flow is computed at FR_SCALE (def 0.5) resolution; magnitudes are in
px/frame at that scale. First FR_SKIP (def 14) frames = seed + transient.
Env: FR_OUT, FR_SKIP, FR_SCALE, FR_RUNS (colon list to restrict).
"""
import os, csv
import numpy as np
import cv2
cv2.setNumThreads(1)                    # login-node cgroup: no thread pool
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import cm, colors as mcolors

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
WB = f"{ARR}/wandb/wandb"
OUT = os.environ.get("FR_OUT", f"{ARR}/analysis/flow_rolling")
SKIP = int(os.environ.get("FR_SKIP", "14"))
SCALE = float(os.environ.get("FR_SCALE", "0.5"))
CELL = 30                               # grid cell (px at SCALE)
GAIN = 9.0                              # display px per flow px/frame
VMAX = 3.0                              # shared |flow| color scale
CMAP = cm.get_cmap("viridis")

# run -> (wandb run dir, steps to map).  rolllong: healthy 256-346 window +
# breakdown 466-541 window per the training-loss timeline.
RUNS = {
    "rollwarm_gan": ("run-20260820_125913-owqfk205",
                     [211, 271, 331, 391, 451, 496]),
    "rollcarn":     ("run-20260820_223449-mpcnyduh",
                     [211, 271, 331, 391, 421]),
    "rollcombo2":   ("run-20260821_011922-yxhnh783",
                     [211, 271, 331, 376]),
    "rolllong":     ("run-20260821_162525-4z2988ef",
                     [256, 286, 316, 346, 466, 481, 511, 526, 541]),
}
if os.environ.get("FR_RUNS"):
    keep = os.environ["FR_RUNS"].split(":")
    RUNS = {k: v for k, v in RUNS.items() if k in keep}


def clip_path(rdir, step):
    import glob
    g = glob.glob(f"{WB}/{rdir}/files/media/videos/sample/pred_image_rollout_{step}_*.mp4")
    return g[0] if g else None


def clip_flow(path):
    """-> frames(gray, SCALE), flows [T-1, H, W, 2] after global SKIP."""
    rd = imageio.get_reader(path)
    fr = [cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2GRAY) for f in rd]
    rd.close()
    fr = [cv2.resize(f, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
          for f in fr]
    fl = [cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0)
          for a, b in zip(fr[:-1], fr[1:])]
    return fr, np.stack(fl)


def radial_grids(H, W):
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rx, ry = (xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)
    rn = np.sqrt(rx ** 2 + ry ** 2) + 1e-6
    return xx, yy, rx, ry, rn


def zoom_fit(mf, xx, yy):
    """Fit mean flow ~ a*(p - c) (pure 2D zoom, free center). -> a, c, R2."""
    H, W = mf.shape[:2]
    x = (xx - W / 2).ravel(); y = (yy - H / 2).ravel()
    fx = mf[..., 0].ravel(); fy = mf[..., 1].ravel()
    n = x.size
    A = np.zeros((2 * n, 3)); b = np.concatenate([fx, fy])
    A[:n, 0] = x; A[:n, 1] = 1.0
    A[n:, 0] = y; A[n:, 2] = 1.0
    (a, bx, by), *_ = np.linalg.lstsq(A, b, rcond=None)
    pred = A @ np.array([a, bx, by])
    r2 = 1.0 - ((b - pred) ** 2).sum() / max(((b - b.mean()) ** 2).sum(), 1e-9)
    c = (W / 2 - bx / a, H / 2 - by / a) if abs(a) > 1e-6 else (np.nan, np.nan)
    return float(a), c, float(r2)


def cell_stats(flows):
    """-> centers [gy,gx,2], mean [gy,gx,2], cov [gy,gx,2,2] per CELL block."""
    T, H, W = flows.shape[:3]
    gy, gx = H // CELL, W // CELL
    fl = flows[:, :gy * CELL, :gx * CELL].reshape(T, gy, CELL, gx, CELL, 2)
    cm_ = fl.mean(axis=(2, 4))                       # [T, gy, gx, 2]
    mean = cm_.mean(0)
    d = cm_ - mean
    cov = np.einsum("tyxi,tyxj->yxij", d, d) / max(T - 1, 1)
    cy, cx = np.mgrid[0:gy, 0:gx]
    centers = np.stack([cx * CELL + CELL / 2, cy * CELL + CELL / 2], -1)
    return centers, mean, cov


def draw_map(ax, frame, centers, mean, cov, title):
    ax.imshow(frame, cmap="gray", alpha=0.45)
    for (px, py), mv, cv_ in zip(centers.reshape(-1, 2),
                                 mean.reshape(-1, 2), cov.reshape(-1, 2, 2)):
        mag = np.linalg.norm(mv)
        col = CMAP(min(mag / VMAX, 1.0))
        tip = (px + mv[0] * GAIN, py + mv[1] * GAIN)
        ax.annotate("", xy=tip, xytext=(px, py),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.4))
        lam, vec = np.linalg.eigh(cv_)
        ang = np.degrees(np.arctan2(vec[1, -1], vec[0, -1]))
        ax.add_patch(Ellipse(tip, *(2 * GAIN * np.sqrt(np.maximum(lam[::-1], 0))),
                             angle=ang, fill=False, color=col, lw=0.7, alpha=0.8))
    ax.set_xlim(0, frame.shape[1]); ax.set_ylim(frame.shape[0], 0)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=9)


def analyze(run, step, path, rows, panel_cache):
    fr, flows = clip_flow(path)
    H, W = fr[0].shape
    xx, yy, rx, ry, rn = radial_grids(H, W)
    fwd_t = ((flows[..., 0] * rx + flows[..., 1] * ry) / rn).mean(axis=(1, 2))
    ste_t = flows[..., 0].mean(axis=(1, 2))
    mag_t = np.linalg.norm(flows, axis=-1).mean(axis=(1, 2))
    post = flows[SKIP:]
    mf = post.mean(0)
    a, c, r2 = zoom_fit(mf, xx, yy)
    # top/bottom radial-coefficient asymmetry (depth structure probe)
    vr = (mf[..., 0] * rx + mf[..., 1] * ry) / rn                # radial comp
    rb = np.linspace(0.15, 1.2, 12)
    prof = {}
    for name, m in [("top", yy < H / 2), ("bottom", yy >= H / 2)]:
        pr = [vr[m & (rn >= r0) & (rn < r1)].mean()
              for r0, r1 in zip(rb[:-1], rb[1:])]
        prof[name] = np.array(pr)
    slope = {k: np.nansum(prof[k] * rb[:-1]) / np.nansum(rb[:-1] ** 2) for k in prof}
    bt = slope["bottom"] / slope["top"] if abs(slope["top"]) > 1e-4 else np.inf
    fwd, ste = float(fwd_t[SKIP:].mean()), float(ste_t[SKIP:].mean())
    rows.append([run, step, round(fwd, 3), round(ste, 3),
                 round(float(mag_t[SKIP:].mean()), 3), round(r2, 3),
                 round(float(a), 4), round(float(bt), 2)])
    centers, mean, cov = cell_stats(post)
    mid = fr[(SKIP + len(fr)) // 2]
    ttl = (f"{run} s{step}  fwd={fwd:+.2f} steer={ste:+.2f} "
           f"zoomR2={r2:.2f} b/t={bt:.1f}")
    panel_cache[(run, step)] = (mid, centers, mean, cov, ttl)

    fig = plt.figure(figsize=(16, 4.6))
    ax1 = fig.add_subplot(1, 3, 1); ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    draw_map(ax1, mid, centers, mean, cov, ttl)
    mid_r = (rb[:-1] + rb[1:]) / 2
    ax2.plot(mid_r, prof["top"], "o-", color="#1f77b4", label="top half (sky/far)")
    ax2.plot(mid_r, prof["bottom"], "o-", color="#d62728", label="bottom half (ground)")
    fit_r = np.linspace(0, 1.2, 20)
    ax2.plot(fit_r, a * fit_r * (W / 2), "--", color="k", lw=1,
             label=f"pure-zoom fit R2={r2:.2f}")
    ax2.axhline(0, color="gray", lw=0.5)
    ax2.set_xlabel("radius (norm)"); ax2.set_ylabel("radial flow (px/frame)")
    ax2.legend(fontsize=8); ax2.set_title("radial profile", fontsize=9)
    t = np.arange(len(fwd_t))
    ax3.plot(t, fwd_t, color="#2ca02c", label="fwd (divergence)")
    ax3.plot(t, ste_t, color="#9467bd", label="steer (mean x)")
    ax3.plot(t, mag_t, color="#7f7f7f", lw=0.8, label="|flow| mean")
    ax3.axvline(SKIP, color="k", lw=0.6, ls=":")
    ax3.axhline(0, color="gray", lw=0.5)
    ax3.set_xlabel("frame"); ax3.legend(fontsize=8)
    ax3.set_title("per-frame motion", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/{run}_{step}.png", dpi=110)
    plt.close(fig)
    print(f"[fr] {run} s{step}: fwd={fwd:+.2f} steer={ste:+.2f} "
          f"|f|={mag_t[SKIP:].mean():.2f} zoomR2={r2:.2f} a={a:+.4f} b/t={bt:.1f}")


def main():
    os.makedirs(OUT, exist_ok=True)
    rows, panels = [], {}
    for run, (rdir, steps) in RUNS.items():
        for s in steps:
            p = clip_path(rdir, s)
            if p is None:
                print(f"[fr] MISSING {run} s{s}"); continue
            analyze(run, s, p, rows, panels)
    with open(f"{OUT}/flow_rolling_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "step", "fwd_exp", "steer_px", "mean_flow_px",
                    "zoom_R2", "zoom_a", "bottom_top_ratio"])
        w.writerows(rows)
    # aligned contact grid
    ncol = max(len(v[1]) for v in RUNS.values())
    fig, axes = plt.subplots(len(RUNS), ncol,
                             figsize=(3.6 * ncol, 2.35 * len(RUNS)))
    axes = np.atleast_2d(axes)
    for i, (run, (rdir, steps)) in enumerate(RUNS.items()):
        for j in range(ncol):
            ax = axes[i, j]
            if j < len(steps) and (run, steps[j]) in panels:
                draw_map(ax, *panels[(run, steps[j])])
                ax.set_title(panels[(run, steps[j])][4], fontsize=6.5)
            else:
                ax.axis("off")
    sm = cm.ScalarMappable(norm=mcolors.Normalize(0, VMAX), cmap=CMAP)
    fig.colorbar(sm, ax=axes, fraction=0.012, pad=0.01,
                 label="|flow| px/frame (half-res)")
    fig.suptitle("Rolling-run rollout flow maps (aligned grid, 1-sigma temporal "
                 "ellipses at arrow tips)", fontsize=11)
    fig.savefig(f"{OUT}/flow_rolling_grid.png", dpi=110)
    plt.close(fig)
    print(f"[fr] wrote {OUT}/flow_rolling_grid.png + {len(rows)} clip maps")


if __name__ == "__main__":
    main()
