"""ADAPTED flow-tree AR timeline from TRAINING-SAMPLE rollout mp4s.

For runs that died without saving a trained checkpoint (e.g. dmd10k_rolllong)
the original probe chain (utils/flow_record_ode_student.py serving the ckpt
on the r08 8-direction battery) is impossible.  Closest equivalent: take the
run's own `samples/step_*_pred_image_rollout.mp4` training visualisations
(the student's multi-chunk AR rollout on its current training ride),
VAE-encode them back into Wan latent space, group latent frames into
3-frame chunks, and project each chunk onto the SAME PCA plane the
flow_tree_*_AR_timeline.mp4 videos use (flow_tree_timelines.fit_basis on the
teacher 20-step recordings).  The animation reveals the rollout
chunk-by-chunk with the same axes/limits/colors grammar; one line per
training step (colormap = training progress) replaces the original's one
line per action direction (the training rollouts follow the ride's logged
actions, so no 8-dir battery exists).

Env: SRT_RUN (label), SRT_SAMPLES (dir with step_*_pred_image_rollout.mp4),
SRT_STEPS (csv of training steps; default = up to 6 evenly spaced),
SRT_OUT (output mp4). GPU required (VAE encode).
"""
import os, re, glob
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
FV = f"{ARR}/analysis/eval_final/flow_viz"
RUN = os.environ["SRT_RUN"]
SAMPLES = os.environ["SRT_SAMPLES"]
OUT = os.environ.get("SRT_OUT", f"{ARR}/analysis/flow_rolling/{RUN}_AR_timeline.mp4")
NFB, C = 3, 16
GT_COL = "0.55"


def main():
    import sys
    sys.path.insert(0, ARR)
    from utils.flow_tree_timelines import fit_basis, gt_latents
    from utils.wan_wrapper import WanVAEWrapper

    m0, pcs = fit_basis(False)
    gt = gt_latents()
    gt_line = np.stack([((gt[NFB * c:NFB * (c + 1)].ravel() - m0) @ pcs)
                        for c in range(1, 7)])
    seed_pt = (gt[:NFB].ravel() - m0) @ pcs

    vids = sorted(glob.glob(f"{SAMPLES}/step_*_pred_image_rollout.mp4"))
    steps = [int(re.search(r"step_(\d+)_", os.path.basename(v)).group(1)) for v in vids]
    if os.environ.get("SRT_STEPS"):
        keep = [int(x) for x in os.environ["SRT_STEPS"].split(",")]
    else:
        idx = np.unique(np.linspace(0, len(steps) - 1, min(6, len(steps))).astype(int))
        keep = [steps[i] for i in idx]
    sel = [(s, v) for s, v in zip(steps, vids) if s in keep]
    print(f"[srt] {RUN}: encoding rollouts at training steps {[s for s, _ in sel]}")

    vae = WanVAEWrapper().to("cuda", torch.bfloat16)
    lines = {}
    with torch.no_grad():
        for s, v in sel:
            frames = np.stack(imageio.mimread(v, memtest=False))      # [T,H,W,3]
            px = torch.from_numpy(frames).float().div(127.5).sub(1.0)
            px = px.permute(3, 0, 1, 2).unsqueeze(0).to("cuda", torch.bfloat16)
            lat = vae.encode_to_latent(px)[0].float().cpu().numpy()   # [Tl,16,60,104]
            nch = lat.shape[0] // NFB
            pts = np.stack([((lat[NFB * c:NFB * (c + 1)].astype(np.float32).ravel()
                              - m0) @ pcs) for c in range(nch)])
            lines[s] = pts
            print(f"[srt]   step {s}: {frames.shape[0]}px f -> {lat.shape[0]} lat f "
                  f"-> {nch} chunks")
    del vae; torch.cuda.empty_cache()

    nch_max = max(len(p) for p in lines.values())
    cmap = plt.get_cmap("viridis")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    frames_out = []
    for j in range(1, nch_max + 1):
        fig, ax = plt.subplots(figsize=(7, 7))
        h_gt, = ax.plot(gt_line[:, 0], gt_line[:, 1], ":", color=GT_COL, lw=2.8,
                        zorder=10, alpha=0.2, label="GT r08 (plane ref)",
                        marker="s", ms=7, mfc=GT_COL, mec="0.25")
        h_s = ax.scatter([seed_pt[0]], [seed_pt[1]], marker="D", s=110,
                         color="#444444", zorder=12, label="r08 SEED (plane ref)")
        handles = [h_gt, h_s]
        for i, (s, pts) in enumerate(sorted(lines.items())):
            col = cmap(0.15 + 0.75 * i / max(1, len(lines) - 1))
            p = pts[:j]
            h, = ax.plot(p[:, 0], p[:, 1], color=col, lw=2.0, alpha=0.85,
                         label=f"train step {s}")
            handles.append(h)
            ax.plot(p[:-1, 0], p[:-1, 1], "o", color=col, ms=3.5, alpha=0.85)
            ax.plot(p[-1, 0], p[-1, 1], "o", mfc="none", mec=col, ms=10,
                    mew=1.6, alpha=0.95)
        ax.set_title(f"{RUN.upper()} AR TIMELINE (ADAPTED: training-sample "
                     f"rollouts, VAE re-encoded) | committed chunks: {j}",
                     fontsize=9.5)
        ax.set_xlim(-900, 900); ax.set_ylim(-900, 900); ax.set_aspect("equal")
        ax.axhline(0, color="k", lw=0.3); ax.axvline(0, color="k", lw=0.3)
        ax.legend(handles=handles, fontsize=6.5, ncol=2, loc="lower left")
        fig.tight_layout(); fig.canvas.draw()
        frames_out.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    frames_out.extend([frames_out[-1]] * 4)
    imageio.mimsave(OUT, frames_out, fps=5, quality=8)
    print(f"[srt] saved {OUT} ({len(frames_out)} frames)")


if __name__ == "__main__":
    main()
