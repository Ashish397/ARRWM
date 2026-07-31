"""Offline 8-DOF teacher-read extraction from saved control videos.

For each saved control_test rollout mp4 (gt/flip), re-run the FROZEN teacher
(CoTracker grid -> top-8 PCA, identical to the trainer's _compute_teacher_visuals)
to recover the generated video's action in ALL 8 PCA dims per chunk. Join with:
  - the commanded action (cz2/cz7) from the metrics JSONL,
  - the REAL ride's 8-dim action (encode_z_actions_window) for the GT reference.

Built-in validation: offline PC0/PC1 of the generated video MUST match the
live-logged tz2/tz7 for the same (step,rank,branch). If they correlate, dims
2..7 (roll=PC6, pitch=PC7) are trustworthy.

Output: one CSV row per (run, step, rank, branch, chunk):
  run,step,rank,branch,chunk,dir, cmd0,cmd1, g0..g7 (generated), r0..r7 (real ride)

Env:
  NDOF_RUNS   : comma list of run keys (default all)
  NDOF_STRIDE : only steps that are multiples of this (default 200)
  NDOF_LIMIT  : cap videos per run (0=all)
  NDOF_OUT    : output CSV (default analysis/ndof_following.csv)
"""
import os, glob, re, json, csv, time
import numpy as np, torch

RUN_DIRS = {
    "pca8_8node": "logs/v14e_pca8_raw",
    "16node":     "logs/v14e_16node",
    "4node":      "logs/v14e_4node",
    "pca4":       "logs/v14e_pca4",
    "pca2":       "logs/v14e_pca2",
    "noatok":     "logs/v14e_noatok",
    "noadaln":    "logs/v14e_noadaln",
}
CK = "preprocessing/checkpoints/pca_basis.pt"
SCALES = [93.7, 57.7, 22.5, 21.2, 18.1, 14.5, 12.6, 10.8]
NFB = 3
GEN_CHUNKS = 8
N_CHUNKS = 1 + GEN_CHUNKS          # 9; matches tot_f//nfb
GRID = 10
OUT_CHUNK = 12
COMPUTE_T = 48
N = GRID * GRID
EGO = [0, 1]                       # action_dims: PC0 throttle, PC1 steer
TH = 0.1
DEV = "cuda"

STRIDE = int(os.environ.get("NDOF_STRIDE", "200"))
LIMIT = int(os.environ.get("NDOF_LIMIT", "0"))
OUT = os.environ.get("NDOF_OUT", "analysis/ndof_following.csv")
ONLY = set(os.environ.get("NDOF_RUNS", "").split(",")) if os.environ.get("NDOF_RUNS") else None


def load_pca():
    ck = torch.load(CK, map_location="cpu", weights_only=False)
    mean = torch.tensor(np.asarray(ck["pca_mean"]), dtype=torch.float32, device=DEV)
    comp_T = torch.tensor(np.asarray(ck["pca_comp"]).T, dtype=torch.float32, device=DEV)  # [200,16]
    scales = torch.tensor(SCALES, dtype=torch.float32, device=DEV)
    return mean, comp_T, scales


def motion_to_z(est_motion, mean, comp_T, scales):
    raw_n = est_motion.shape[0]
    flat = est_motion[:, :, :2].reshape(raw_n, 200).float()
    P = (flat - mean) @ comp_T
    return torch.tanh(P[:, :8] / scales)          # [raw_n,8]


def reduce_to_segments(per_frame, n_seg):
    F_len = per_frame.shape[0]
    seg = max(1, F_len // n_seg)
    return torch.stack([per_frame[i * seg: min(i * seg + seg, F_len)].mean(0) for i in range(n_seg)])


def teacher_read_video(vid, cotracker, mean, comp_T, scales):
    """vid: [1,T,C,H,W] float 0-255 -> generated action [N_CHUNKS,8] (chunk-pooled)."""
    T_total = vid.shape[1]
    mws = []
    for cs in range(0, T_total, COMPUTE_T):
        ce = min(cs + COMPUTE_T, T_total)
        ch = vid[:, cs:ce]
        n_out = ch.shape[1] // OUT_CHUNK
        if n_out == 0:
            continue
        ch = ch[:, :n_out * OUT_CHUNK].clone()
        with torch.amp.autocast(device_type="cuda", enabled=True):
            tracks, vis = cotracker(ch, grid_size=GRID)
        tw = tracks.reshape(1, n_out, OUT_CHUNK, N, 2)
        vw = vis.reshape(1, n_out, OUT_CHUNK, N).unsqueeze(-1) if vis.dim() == 3 else vis.reshape(1, n_out, OUT_CHUNK, N, 1)
        mo = (tw[:, :, 1:] - tw[:, :, :-1]).mean(dim=2)
        vo = vw.to(dtype=mo.dtype).mean(dim=2)
        mws.append(torch.cat([mo, vo], dim=-1).squeeze(0))
    if not mws:
        return torch.zeros(N_CHUNKS, 8, device=DEV)
    est = torch.cat(mws, dim=0)
    z8 = motion_to_z(est, mean, comp_T, scales)
    return reduce_to_segments(z8, N_CHUNKS)       # [N_CHUNKS,8]


def read_mp4(path):
    from torchvision.io import read_video
    v, _, _ = read_video(path, pts_unit="sec", output_format="THWC")  # [T,H,W,C] uint8 RGB
    v = v.to(DEV).float().permute(0, 3, 1, 2).unsqueeze(0)            # [1,T,C,H,W] 0-255
    return v


def build_eval_ds():
    """Instantiate the eval dataset (for the real-ride reference) from a still-present
    shared manifest cache + eval_ride_zarrs from the config."""
    from omegaconf import OmegaConf
    from utils.zarr_dataset import ZarrRideDataset
    cfg = OmegaConf.merge(OmegaConf.load("configs/default_config.yaml"),
                          OmegaConf.load("configs/causal_lora_diffusion_teacher_v14e.yaml"))
    shared = None
    for d in ("logs/v14e_16node", "logs/v14e_pca8_raw", "logs/v14e_pca2"):
        p = os.path.join(d, ".ride_manifest_shared.pt")
        if os.path.exists(p):
            shared = p; break
    if shared is None:
        print("[warn] no shared manifest cache found -> real-ride reference disabled")
        return None, {}
    all_rides = torch.load(shared, map_location="cpu")
    from pathlib import Path as _P
    names = list(cfg.get("eval_ride_zarrs", []) or [])
    order = {n: i for i, n in enumerate(names)}
    picked = sorted([r for r in all_rides if _P(r["zarr_path"]).name in order], key=lambda r: order[_P(r["zarr_path"]).name])
    ds = ZarrRideDataset.from_manifest(rides_data=picked, motion_root=cfg.motion_root,
                                       pca_basis_checkpoint=CK)
    by_name = {_P(r["zarr_path"]).name: r for r in picked}
    print(f"[eval-ds] {len(picked)} rides for real-ride reference")
    return ds, by_name


def chunk_dir(c0, c1, flip):
    r0, r1 = (-c0, -c1) if flip else (c0, c1)
    fb = "F" if r0 > TH else ("B" if r0 < -TH else "")
    lr = "R" if r1 > TH else ("L" if r1 < -TH else "")
    return fb + lr


def main():
    mean, comp_T, scales = load_pca()
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(DEV).eval()
    for p in cotracker.parameters():
        p.requires_grad_(False)
    eval_ds, by_name = build_eval_ds()

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fout = open(OUT, "w", newline="")
    w = csv.writer(fout)
    w.writerow(["run", "step", "rank", "branch", "chunk", "dir", "cmd0", "cmd1"]
               + [f"g{d}" for d in range(8)] + [f"r{d}" for d in range(8)])

    val_g, val_c = [], []   # validation: offline PC0/1 vs logged tz2/tz7
    for run, base in RUN_DIRS.items():
        if ONLY and run not in ONLY:
            continue
        cdir = os.path.join(base, "control_test")
        if not os.path.isdir(cdir):
            continue
        # command lookup from JSONL: (step,rank,branch) -> (cz2[],cz7[],tz2[],tz7[],ride,offset)
        cmd = {}
        for f in glob.glob(f"{cdir}/metrics_r*.jsonl"):
            for ln in open(f):
                try:
                    d = json.loads(ln)
                except Exception:
                    continue
                s, r = d.get("step"), d.get("rank")
                for br in ("gt", "flip"):
                    b = d.get(br, {})
                    if b.get("cz2"):
                        cmd[(s, r, br)] = (b["cz2"], b["cz7"], b["tz2"], b["tz7"], d.get("ride"), d.get("offset"))
        vids = sorted(glob.glob(f"{cdir}/step*_r*_*_raw.mp4"))
        vids = [v for v in vids if (m := re.match(r"step0*(\d+)_r(\d+)_(gt|flip)_raw\.mp4", os.path.basename(v)))
                and int(m.group(1)) % STRIDE == 0]
        if LIMIT:
            vids = vids[:LIMIT]
        print(f"[{run}] {len(vids)} videos (stride {STRIDE})", flush=True)
        t0 = time.time()
        for i, v in enumerate(vids):
            m = re.match(r"step0*(\d+)_r(\d+)_(gt|flip)_raw\.mp4", os.path.basename(v))
            step, rank, br = int(m.group(1)), int(m.group(2)), m.group(3)
            key = (step, rank, br)
            if key not in cmd:
                continue
            cz2, cz7, tz2, tz7, ride, offset = cmd[key]
            try:
                vid = read_mp4(v)
                gen = teacher_read_video(vid, cotracker, mean, comp_T, scales)   # [9,8]
            except Exception as e:
                print(f"  [skip] {os.path.basename(v)}: {e}"); continue
            gg = gen[1:].detach().cpu().numpy()                                  # drop seed -> [8,8]
            # real-ride reference (8-dim), chunk-pooled, seed dropped
            rr = None
            if eval_ds is not None and ride in by_name:
                try:
                    zp = by_name[ride]["zarr_path"]; nlat = by_name[ride]["n_latent_frames"]
                    tot_f = NFB * N_CHUNKS
                    z = eval_ds.encode_z_actions_window(zp, nlat, offset, offset + tot_f)  # [tot_f,8]
                    z = z.reshape(N_CHUNKS, NFB, 8).mean(1)[1:].cpu().numpy()              # [8,8]
                    rr = z
                except Exception:
                    rr = None
            nc = min(len(gg), len(cz2))
            for k in range(nc):
                dn = chunk_dir(cz2[k], cz7[k], br == "flip")
                if not dn:
                    continue
                row = [run, step, rank, br, k, dn, round(cz2[k], 4), round(cz7[k], 4)]
                row += [round(float(gg[k, d]), 4) for d in range(8)]
                row += ([round(float(rr[k, d]), 4) for d in range(8)] if rr is not None else [""] * 8)
                w.writerow(row)
                val_g.append(float(gg[k, 0])); val_c.append(float(tz2[k]))     # PC0 vs logged tz2
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(vids)}  ({(time.time()-t0)/(i+1):.2f}s/vid)", flush=True)
        fout.flush()
    fout.close()
    # validation report
    if len(val_g) > 10:
        a, b = np.array(val_g), np.array(val_c)
        cc = float(np.corrcoef(a, b)[0, 1])
        print(f"\n[VALIDATION] offline PC0 vs logged tz2: corr={cc:.4f}  mae={np.mean(np.abs(a-b)):.4f}  n={len(a)}")
        print("  -> alignment GOOD (>0.9)" if cc > 0.9 else "  -> WARNING: alignment suspect (<0.9)")
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
