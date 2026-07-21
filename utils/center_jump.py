"""Center-region SPAWN JUMP + full novelty trajectories, phase-A videos.

The suite's spawn_jump is the max single-step increase of the WHOLE-FRAME p95
DINO novelty — object spawns in the mid-view are diluted by the periphery, so
it barely discriminates. This recomputes, with the same DINOv2 seed-bank
recipe (bank: frames 0/3/6/9 + blur/brightness augs; samples: every 6th frame
from 12):

  c_t  = center-crop novelty per sampled frame   (crop [H/4:3H/4, W/3:2W/3])
  e_t  = max periphery-strip novelty per frame   (as utils/edge_novelty.py)
  center_jump = max single-step increase of c_t  <- abrupt mid-view appearance
  edge_jump   = max single-step increase of e_t
  center_nov  = max c_t (sanity: should reproduce suite center_nov)

Saves per-model npz of the FULL (c_t, e_t) trajectories for onset-time and
settling analysis without re-running DINO.

Only the FIRST CJ_MAX_SEC seconds (default 6.0) of each video are scored, so
models with longer rollouts (matrixgame 24s vs ours 6.8s) don't get extra time
to accumulate novelty — equal-time comparison.

Env: CJ_RUNS colon list, CJ_OUT csv, CJ_NPZ_DIR (default analysis/eval_final/nov_traj),
     CJ_MAX_SEC (default 6.0).
"""
import os
import numpy as np
import pandas as pd
import imageio
import torch
import torch.nn.functional as F

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
RUNS = os.environ.get("CJ_RUNS", "pca8_8node").split(":")
OUT = os.environ.get("CJ_OUT", f"{ARR}/analysis/eval_final/center_jump.csv")
NPZ_DIR = os.environ.get("CJ_NPZ_DIR", f"{ARR}/analysis/eval_final/nov_traj")
DIRS = ["F", "FR", "R", "BR", "B", "BL", "L", "FL"]
DEV = "cuda"
MAX_SEC = float(os.environ.get("CJ_MAX_SEC", "6.0"))
_MWSW = {"L": "R", "R": "L", "FL": "FR", "FR": "FL", "BL": "BR", "BR": "BL"}
os.makedirs(NPZ_DIR, exist_ok=True)


def vid_path(run, wi, d):
    if run == "minwm":
        return f"{ARR}/logs/eval_final/A_minwm/minwm_r{wi:02d}_{_MWSW.get(d, d)}.mp4"
    for m in ("matrixgame", "worldcam", "yume", "worldplay", "astra"):
        if run == m:
            return f"{ARR}/logs/eval_final/A_{m}/{m}_r{wi:02d}_{d}.mp4"
    return f"{ARR}/logs/eval_final/A/{run}/control_test/step05000_r{wi:02d}_{d}_raw.mp4"


dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(DEV).eval()
DM = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1)
DS = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
PATCH, STRIDE = 320, 160


def unfold(t):
    return F.unfold(t, PATCH, stride=STRIDE).transpose(1, 2).reshape(-1, 3, PATCH, PATCH)


@torch.no_grad()
def dfeat(ps):
    v = F.interpolate(ps, size=224, mode="bilinear", align_corners=False)
    e = dino((v - DM) / DS)
    return (e / e.norm(dim=-1, keepdim=True)).float()


def aug(x):
    outs = [x]
    for k in (9, 21, 41):
        w = torch.ones(3, 1, k, k, device=DEV) / (k * k)
        outs.append(F.conv2d(x, w, padding=k // 2, groups=3))
    outs += [(x * 0.7).clamp(0, 1), (x * 1.3).clamp(0, 1)]
    return outs


@torch.no_grad()
def trajectories(frames):
    t = lambda f: torch.from_numpy(np.ascontiguousarray(f)).permute(2, 0, 1)[None].to(DEV).float() / 255.0
    bank = torch.cat([dfeat(unfold(v)) for i in (0, 3, 6, 9) for v in aug(t(frames[i]))], 0)
    cs, es, idxs = [], [], []
    for i in range(12, len(frames) - 1, 6):
        fr = frames[i]
        H, W = fr.shape[:2]
        cf = fr[H // 4:3 * H // 4, W // 3:2 * W // 3]
        strips = [fr[:, :W // 6], fr[:, 5 * W // 6:], fr[:H // 6, :], fr[5 * H // 6:, :]]
        batch = torch.cat([F.interpolate(t(s), size=224, mode="bilinear", align_corners=False)
                           for s in [cf] + strips], 0)
        e = dfeat(batch)
        d = (1 - e @ bank.T).min(1).values          # [5]
        cs.append(float(d[0].item()))
        es.append(float(d[1:].max().item()))
        idxs.append(i)
    return np.array(cs), np.array(es), np.array(idxs)


def main():
    done = set()
    if os.path.exists(OUT):
        prev = pd.read_csv(OUT)
        done = set(zip(prev.run, prev.window, prev["dir"]))
    f = open(OUT, "a" if done else "w")
    if not done:
        f.write("run,window,dir,center_jump,edge_jump,center_nov,edge_nov\n")
    for run in RUNS:
        trajs = {}
        npz_path = f"{NPZ_DIR}/nov_traj_{run}.npz"
        if os.path.exists(npz_path):
            trajs = dict(np.load(npz_path))
        for wi in range(32):
            for d in DIRS:
                key = f"r{wi:02d}_{d}"
                if (run, f"r{wi:02d}", d) in done and key in trajs:
                    continue
                path = vid_path(run, wi, d)
                if not os.path.exists(path):
                    print(f"[cj] MISSING {path}", flush=True)
                    continue
                try:
                    r = imageio.get_reader(path)
                    fps = float(r.get_meta_data().get("fps", 16.0))
                    max_f = max(int(fps * MAX_SEC), 24)
                    frames = []
                    for fi, x in enumerate(r):
                        if fi >= max_f:
                            break
                        frames.append(np.asarray(x))
                    r.close()
                    cs, es, idxs = trajectories(frames)
                    if len(cs) < 2:
                        continue
                    trajs[key] = np.stack([idxs, cs, es]).astype(np.float32)
                    if (run, f"r{wi:02d}", d) not in done:
                        f.write(f"{run},r{wi:02d},{d},{np.diff(cs).max():.4f},"
                                f"{np.diff(es).max():.4f},{cs.max():.4f},{es.max():.4f}\n")
                        f.flush()
                except Exception as e:
                    print(f"[cj] FAIL {run} r{wi:02d} {d}: {e}", flush=True)
        np.savez_compressed(npz_path, **trajs)
        print(f"[cj] {run} done -> {npz_path}", flush=True)
    f.close()


if __name__ == "__main__":
    main()
