"""Stationary (no-op) drift read through the FROZEN CoTracker->PCA teacher.

The RANSAC variant (stationary_cotracker.py) fits an affine start->end and reports the
median inlier displacement in image pixels. That answers "how far did the background
slide", but in a space unrelated to the model's action interface. This variant instead
reads the generated span with the same teacher the model was trained against: a 10x10
CoTracker grid, mean per-frame flow over 12-frame chunks, flattened to 200-D and
projected onto the frozen PCA basis in action_query/checkpoints/ss_vae_8free.pt.

By that basis's convention PC0 is throttle (forward/backward dolly) and PC1 is steer
(left/right pan), so a good no-op rollout should sit near the origin in (PC1, PC0):
the model was commanded to do nothing and the teacher should read nothing.

Both raw projections and the tanh-squashed values the trainer conditions on are written,
along with PC2..PC7 for reference. Frames are resized to a common 832x448 so that the
pixel-scaled PCA magnitudes are comparable across models of different resolution.

Writes out/stationary_cotracker_pca.csv.
"""
import os, glob
import numpy as np, torch, cv2, imageio, pandas as pd

DIR = "/home/ashish/stationary_evaluation"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out", "stationary_cotracker_pca.csv")
CK = os.path.join(os.path.dirname(os.path.dirname(HERE)), "action_query", "checkpoints", "ss_vae_8free.pt")
CTX = {"astra": 4, "matrixgame": 1, "minwm": 13, "worldcam": 65, "worldplay": 1, "yume": 1}

GRID = 10                 # teacher grid -> N=100 points -> 200-D flow vector
N = GRID * GRID
OUT_CHUNK = 12            # frames pooled into one teacher read
COMPUTE_T = 48            # frames per CoTracker forward pass
SIZE = (832, 448)         # common resize; matches the ours_* training tile
SCALES = np.array([93.7, 57.7, 22.5, 21.2, 18.1, 14.5, 12.6, 10.8], np.float32)
DEV = "cuda"


def ctx_of(m):
    return CTX.get(m, 12)          # ours + real = 12


def read_gen(path, ctx):
    """All generated frames, consecutive (the teacher's flow is per-frame, so the
    frames must not be subsampled), resized to the common size."""
    r = imageio.get_reader(path)
    n = r.count_frames()
    fr = [cv2.resize(np.asarray(r.get_data(int(i))), SIZE) for i in range(min(ctx, n - 2), n)]
    r.close()
    return np.stack(fr)


def load_pca():
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    mean = torch.tensor(np.asarray(ck["pca_mean"]), dtype=torch.float32, device=DEV)
    comp_T = torch.tensor(np.asarray(ck["pca_comp"]).T, dtype=torch.float32, device=DEV)
    return mean, comp_T


@torch.no_grad()
def teacher_read(vid, cot, mean, comp_T):
    """vid [1,T,C,H,W] 0-255 -> raw PCA projections per chunk [n_chunks, 8]."""
    T_total = vid.shape[1]
    out = []
    for cs in range(0, T_total, COMPUTE_T):
        ch = vid[:, cs:min(cs + COMPUTE_T, T_total)]
        n_out = ch.shape[1] // OUT_CHUNK
        if n_out == 0:
            continue
        ch = ch[:, :n_out * OUT_CHUNK].clone()
        with torch.amp.autocast(device_type="cuda", enabled=True):
            tracks, _ = cot(ch, grid_size=GRID)
        tw = tracks.reshape(1, n_out, OUT_CHUNK, N, 2)
        mo = (tw[:, :, 1:] - tw[:, :, :-1]).mean(dim=2).squeeze(0)     # [n_out,N,2] px/frame
        flat = mo.reshape(mo.shape[0], 200).float()
        out.append((flat - mean) @ comp_T)
    return torch.cat(out, 0)[:, :8] if out else None


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    cot = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(DEV).eval()
    for p in cot.parameters():
        p.requires_grad_(False)
    mean, comp_T = load_pca()
    sc = torch.tensor(SCALES, device=DEV)

    files = sorted(glob.glob(os.path.join(DIR, "*.mp4")))
    rows = []
    for k, f in enumerate(files):
        base = os.path.basename(f)[:-4]
        model, scene = base.rsplit("_r", 1); scene = "r" + scene
        try:
            frames = read_gen(f, ctx_of(model))
            vid = torch.from_numpy(frames).permute(0, 3, 1, 2)[None].float().to(DEV)
            P = teacher_read(vid, cot, mean, comp_T)
            if P is None:
                print(f"[pca] {base} too short"); continue
            Z = torch.tanh(P / sc)
            p = P.mean(0).cpu().numpy()          # mean raw projection over the generated span
            z = Z.mean(0).cpu().numpy()          # mean squashed (what the model conditions on)
            pa = P.abs().mean(0).cpu().numpy()   # mean |projection| = drift magnitude per axis
            rows.append(dict(model=model, scene=scene, n_chunks=int(P.shape[0]),
                             pc0=round(float(p[0]), 3), pc1=round(float(p[1]), 3),
                             pc0_abs=round(float(pa[0]), 3), pc1_abs=round(float(pa[1]), 3),
                             z0=round(float(z[0]), 4), z1=round(float(z[1]), 4),
                             **{f"pc{i}": round(float(p[i]), 3) for i in range(2, 8)}))
        except Exception as e:
            print(f"[pca] {base} FAIL {str(e)[:70]}", flush=True); continue
        if (k + 1) % 40 == 0:
            pd.DataFrame(rows).to_csv(OUT, index=False)
            print(f"[pca] {k+1}/{len(files)}", flush=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"[pca] wrote {OUT} ({len(rows)})")


if __name__ == "__main__":
    main()
