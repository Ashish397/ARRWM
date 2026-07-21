"""Commanded vs REALIZED egomotion of the pre-check student videos.

For each flow_viz/.motion_check/{run}/r08_{d}_s{sd}.mp4 (8-chunk student
rollouts): Farneback optical flow between consecutive frames, then
  fwd  = mean flow divergence (radial expansion from center; + = forward,
         - = backward/contraction)
  ste  = mean horizontal flow (+ = scene moves left = ego turns right)
averaged over frames (dropping the first SKIP frames = seed + transient).
Realized (fwd, ste) per commanded compass dir answers: does commanded-B
actually move backward, or is the flow-endpoint "mode misassignment" a
metric artifact? Prints a per-run table + polar-style scatter figure.

Writes flow_viz/motion_check.png + motion_check.csv.
Env: MC_RUNS colon list (def afall_freal_cd_vid:statwave_Freal_vid), MC_OUT.
"""
import os, glob
import numpy as np
import cv2
cv2.setNumThreads(1)                    # login-node cgroup: no thread pool
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
FV = f"{ARR}/analysis/eval_final/flow_viz"
OUT = os.environ.get("MC_OUT", FV)
RUNS = os.environ.get("MC_RUNS", "afall_freal_cd_vid:statwave_Freal_vid").split(":")
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
COLORS = {"F": "#1f77b4", "FR": "#17becf", "R": "#2ca02c", "BR": "#bcbd22",
          "B": "#d62728", "BL": "#e377c2", "L": "#9467bd", "FL": "#8c564b"}
SKIP = 14                        # seed frames (~13px for 3 latent) + transient


def video_motion(path):
    rd = imageio.get_reader(path)
    frames = [cv2.cvtColor(np.asarray(f), cv2.COLOR_RGB2GRAY) for f in rd]
    rd.close()
    if len(frames) <= SKIP + 1:
        return None
    H, W = frames[0].shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    rx, ry = (xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)
    rn = np.sqrt(rx ** 2 + ry ** 2) + 1e-6
    fwd, ste = [], []
    for a, b in zip(frames[SKIP:-1], frames[SKIP + 1:]):
        fl = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        # radial component (expansion>0 = forward), horizontal mean
        fwd.append(float(((fl[..., 0] * rx + fl[..., 1] * ry) / rn).mean()))
        ste.append(float(fl[..., 0].mean()))
    return float(np.mean(fwd)), float(np.mean(ste)), len(frames)


def main():
    rows = []
    fig, axes = plt.subplots(1, len(RUNS), figsize=(7.5 * len(RUNS), 7))
    axes = np.atleast_1d(axes)
    for ax, run in zip(axes, RUNS):
        print(f"[mc] {run}:")
        print(f"     {'dir':>3s} {'fwd(exp)':>10s} {'steer(px)':>10s}  per seed")
        for d in DIRS:
            pts = []
            for f in sorted(glob.glob(f"{FV}/.motion_check/{run}/r08_{d}_s*.mp4")):
                r = video_motion(f)
                if r is None:
                    print(f"[mc] UNREADABLE {f}", flush=True)
                    continue
                fwd, ste, nf = r
                pts.append((fwd, ste))
                rows.append((run, d, os.path.basename(f), fwd, ste, nf))
            if not pts:
                continue
            pts = np.array(pts)
            ax.scatter(pts[:, 1], pts[:, 0], color=COLORS[d], s=60,
                       edgecolor="black", linewidth=0.4)
            m = pts.mean(0)
            ax.annotate(d, (m[1], m[0]), fontsize=12, fontweight="bold",
                        color=COLORS[d], textcoords="offset points", xytext=(6, 4))
            print(f"     {d:>3s} {m[0]:10.3f} {m[1]:10.3f}  "
                  + " ".join(f"({p[0]:+.2f},{p[1]:+.2f})" for p in pts), flush=True)
        ax.axhline(0, color="gray", lw=0.6); ax.axvline(0, color="gray", lw=0.6)
        ax.set_xlabel("mean horizontal flow (steer; + = turning right)")
        ax.set_ylabel("mean radial flow (+ = forward, - = backward)")
        ax.set_title(run)
        ax.grid(alpha=0.3)
    fig.suptitle("Realized egomotion per commanded direction — 8-chunk student rollouts, "
                 "window r08 (labels at per-direction mean)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(f"{OUT}/motion_check.png", dpi=130)
    with open(f"{OUT}/motion_check.csv", "w") as fh:
        fh.write("run,dir,file,fwd,steer,frames\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")
    print(f"[mc] saved {OUT}/motion_check.png + motion_check.csv", flush=True)


if __name__ == "__main__":
    main()
