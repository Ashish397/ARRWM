"""3D 'grove of trees' over WORLD time, built from the EXISTING 256 eval videos.

No new generation: VAE-encodes each pca8 phase-A video (32 windows x 8 dirs)
back into latent space, one point per latent frame, z-axis = video time.
Each window is a tree: the real seed frames form a shared trunk, then the 8
action branches fan out as the rollout evolves. Tests the hypothesis that
actions connect different parts of flow space consistently:

  world_grove_3d.png / world_grove_rot.gif    raw joint PCA + z=latent frame
  world_overlay_3d.png / world_overlay_rot.gif  per-(tree, frame) across-action
                                              mean removed -> trees share origin
  world_cosine.png       8x8 cross-tree cosine of final-frame action residuals
  world_endpoints.png    final-frame latents in 2D, colored by action: do
                         same-action endpoints cluster ACROSS contexts (global
                         attractor) or stay grouped by context (relative op)?

Env: WG_RUN (def pca8_8node), WG_WINDOWS (def all 32), WG_OUT.
"""
import os
import numpy as np
import torch
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
RUN = os.environ.get("WG_RUN", "pca8_8node")
OUT = os.environ.get("WG_OUT", f"{ARR}/analysis/eval_final/flow_viz")
WINDOWS = [int(x) for x in os.environ["WG_WINDOWS"].split(",")] if "WG_WINDOWS" in os.environ else list(range(32))

DNAMES = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
COLORS = {"F": "#1f77b4", "FR": "#17becf", "R": "#2ca02c", "BR": "#bcbd22",
          "B": "#d62728", "BL": "#e377c2", "L": "#9467bd", "FL": "#8c564b"}

_MWSW = {"L": "R", "R": "L", "FL": "FR", "FR": "FL", "BL": "BR", "BR": "BL"}
_COMPARATORS = ("minwm", "matrixgame", "worldcam", "yume", "worldplay", "astra")


def vid_path(run, wi, d):
    # minwm disk labels are yaw-sign-flipped (SIFT-verified); swap to TRUE direction
    if run == "minwm":
        return f"{ARR}/logs/eval_final/A_minwm/minwm_r{wi:02d}_{_MWSW.get(d, d)}.mp4"
    if run in _COMPARATORS:
        return f"{ARR}/logs/eval_final/A_{run}/{run}_r{wi:02d}_{d}.mp4"
    return f"{ARR}/logs/eval_final/A/{run}/control_test/step05000_r{wi:02d}_{d}_raw.mp4"


def read_video(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return np.stack(frames)                                   # [T, H, W, 3] uint8


def encode_all():
    from utils.wan_wrapper import WanVAEWrapper
    vae = WanVAEWrapper().to("cuda", dtype=torch.float16).eval()
    lats = {}                                                  # (widx, dir) -> [F_lat, D] float16
    for widx in WINDOWS:
        for d in DNAMES:
            p = vid_path(RUN, widx, d)
            if not os.path.exists(p):
                print(f"[world] MISSING {p}", flush=True)
                continue
            fr = read_video(p)
            T = (fr.shape[0] - 1) // 4 * 4 + 1                 # Wan needs (T-1)%4==0
            px = torch.from_numpy(fr[:T]).cuda().permute(3, 0, 1, 2).float() / 127.5 - 1.0
            with torch.no_grad():
                lat = vae.encode_to_latent(px.unsqueeze(0).half())   # [1, F_lat, 16, H/8, W/8]
            lats[(widx, d)] = lat[0].flatten(1).cpu().numpy().astype(np.float16)
            print(f"[world] r{widx:02d} {d}: {fr.shape[0]}f -> {lats[(widx, d)].shape}", flush=True)
    np.savez_compressed(f"{OUT}/world_lats_{RUN}.npz",
                        **{f"w{w}_{d}": v for (w, d), v in lats.items()})
    return lats


def plot_endpoints(lats, mu, P, wpresent):
    # endpoint geometry: global attractor vs relative displacement
    ends = {k: (v[-1].astype(np.float32) - mu) @ P for k, v in lats.items()}
    fig, ax = plt.subplots(figsize=(10, 9))
    # each window's shared starting point (first latent frame = real seed) in
    # low-alpha grey: the origin/center each cluster-of-8 scatters around
    for widx in wpresent:
        k0 = next((widx, d) for d in DNAMES if (widx, d) in lats)
        p0 = (lats[k0][0].astype(np.float32) - mu) @ P
        ax.scatter(p0[0], p0[1], color="grey", s=70, alpha=0.35, zorder=1,
                   label="seed (start) point" if widx == wpresent[0] else None)
    for (widx, d), p in ends.items():
        ax.scatter(p[0], p[1], color=COLORS[d], s=45, edgecolor="black", linewidth=0.3,
                   zorder=3, label=d if widx == wpresent[0] else None)
    ax.set_title("Final-frame latents, colored by action (grey = each window's seed start)\n"
                 "clusters by COLOR = actions pull contexts to shared regions; "
                 "clusters by position-groups-of-8 = context dominates")
    ax.legend(ncol=4, fontsize=9); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout(); fig.savefig(f"{OUT}/world_endpoints_{RUN}.png", dpi=130)
    plt.close(fig)
    print(f"[world] saved world_endpoints_{RUN}.png", flush=True)


def plot_all(lats):
    Fl = min(v.shape[0] for v in lats.values())
    lats = {k: v[:Fl] for k, v in lats.items()}
    zs = np.arange(Fl)
    wpresent = sorted({w for (w, _) in lats})

    allpts = np.concatenate([v.astype(np.float32) for v in lats.values()], 0)
    mu = allpts.mean(0)
    U, S, V = torch.pca_lowrank(torch.from_numpy(allpts - mu), q=2, niter=6)
    P = V[:, :2].numpy()
    del allpts

    if os.environ.get("WG_ENDPOINTS_ONLY"):
        plot_endpoints(lats, mu, P, wpresent)
        return

    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection="3d")
    for (widx, d), tr in lats.items():
        p2 = (tr.astype(np.float32) - mu) @ P
        ax.plot(p2[:, 0], p2[:, 1], zs, color=COLORS[d], alpha=0.6, lw=1.0,
                label=d if widx == wpresent[0] else None)
        ax.scatter(p2[-1, 0], p2[-1, 1], Fl - 1, color=COLORS[d], s=35, marker="*",
                   edgecolor="black", linewidth=0.3, zorder=5)
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.set_zlabel("video time (latent frame)")
    ax.set_title(f"World-time grove — {RUN}, {len(wpresent)} contexts x 8 actions "
                 f"(existing eval videos, VAE re-encoded)\nshared real-seed trunk at bottom, "
                 f"action branches fan out over rollout time")
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/world_grove_3d_{RUN}.png", dpi=120)

    def spin(i):
        ax.view_init(elev=22, azim=i * 4)
        return []
    FuncAnimation(fig, spin, frames=90, blit=False).save(
        f"{OUT}/world_grove_rot_{RUN}.gif", writer=PillowWriter(fps=12), dpi=70)
    plt.close(fig)
    print("[world] saved world_grove_3d_" + RUN + ".png + world_grove_rot.gif", flush=True)

    # residual overlay
    res = {}
    for widx in wpresent:
        if any((widx, d) not in lats for d in DNAMES):
            continue
        tr = np.stack([lats[(widx, d)].astype(np.float32) for d in DNAMES])
        tr -= tr.mean(0, keepdims=True)
        for i, d in enumerate(DNAMES):
            res[(widx, d)] = tr[i]
    allr = np.concatenate(list(res.values()), 0)
    Ur, Sr, Vr = torch.pca_lowrank(torch.from_numpy(allr), q=2, niter=6)
    Pr = Vr[:, :2].numpy()
    del allr

    fig = plt.figure(figsize=(15, 12))
    ax = fig.add_subplot(111, projection="3d")
    for (widx, d), r in res.items():
        p2 = r @ Pr
        ax.plot(p2[:, 0], p2[:, 1], zs, color=COLORS[d], alpha=0.55, lw=1.0,
                label=d if widx == wpresent[0] else None)
        ax.scatter(p2[-1, 0], p2[-1, 1], Fl - 1, color=COLORS[d], s=35, marker="*",
                   edgecolor="black", linewidth=0.3, zorder=5)
    ax.set_xlabel("res PC1"); ax.set_ylabel("res PC2"); ax.set_zlabel("video time (latent frame)")
    ax.set_title("World-time action-residual overlay — all 32 trees share the origin\n"
                 "same color aligned across trees = action moves the world state consistently")
    ax.legend(ncol=4, fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/world_overlay_3d_{RUN}.png", dpi=120)

    def spin2(i):
        ax.view_init(elev=22, azim=i * 4)
        return []
    FuncAnimation(fig, spin2, frames=90, blit=False).save(
        f"{OUT}/world_overlay_rot_{RUN}.gif", writer=PillowWriter(fps=12), dpi=70)
    plt.close(fig)
    print("[world] saved world_overlay_3d_" + RUN + ".png + world_overlay_rot.gif", flush=True)

    # cross-tree cosine of final action residuals
    fin = {k: r[-1] for k, r in res.items()}
    nrm = {k: v / (np.linalg.norm(v) + 1e-8) for k, v in fin.items()}
    wres = sorted({w for (w, _) in res})
    C = np.zeros((8, 8)); n = np.zeros((8, 8))
    for a, da in enumerate(DNAMES):
        for b, db in enumerate(DNAMES):
            for i1, w1 in enumerate(wres):
                for w2 in wres[i1 + 1:]:
                    C[a, b] += float(nrm[(w1, da)] @ nrm[(w2, db)]); n[a, b] += 1
    C /= np.maximum(n, 1)
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(C, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(8), DNAMES); ax.set_yticks(range(8), DNAMES)
    for a in range(8):
        for b in range(8):
            ax.text(b, a, f"{C[a, b]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(f"World-time: cross-tree cosine of final action residuals ({RUN})")
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(f"{OUT}/world_cosine_{RUN}.png", dpi=130)
    plt.close(fig)
    diag = float(np.trace(C) / 8); off = float((C.sum() - np.trace(C)) / 56)
    print(f"[world] same-action cross-tree cosine {diag:.3f} | cross-action {off:.3f}", flush=True)

    plot_endpoints(lats, mu, P, wpresent)

    E = {k: v[-1].astype(np.float32) for k, v in lats.items()}
    def spread(groups):
        ds = []
        for g in groups:
            for i in range(len(g)):
                for j in range(i + 1, len(g)):
                    ds.append(np.linalg.norm(g[i] - g[j]))
        return float(np.mean(ds))
    by_action = [[E[(w, d)] for w in wpresent if (w, d) in E] for d in DNAMES]
    by_window = [[E[(w, d)] for d in DNAMES if (w, d) in E] for w in wpresent]
    print(f"[world] endpoint spread: within-action {spread(by_action):.1f} | "
          f"within-window {spread(by_window):.1f} | "
          f"global {spread([list(E.values())]):.1f}", flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    f = f"{OUT}/world_lats_{RUN}.npz"
    if os.path.exists(f):
        z = np.load(f)
        lats = {}
        for k in z.files:
            wpart, d = k.rsplit("_", 1)
            lats[(int(wpart[1:]), d)] = z[k]
        print(f"[world] reusing {f} ({len(lats)} videos)", flush=True)
    else:
        lats = encode_all()
    plot_all(lats)
