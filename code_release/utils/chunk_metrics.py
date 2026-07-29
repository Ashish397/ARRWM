"""UNIFIED per-chunk metrics for control videos -- run ONCE, derive any metric later.

For every saved control_test rollout mp4, in a single pass, emit one row per
generated chunk with EVERYTHING needed downstream:

  run, step, rank, branch, chunk, ride, offset(timestamp), dir
  cmd0,cmd1          : commanded action this chunk (throttle,steer; flipped for FLIP)
  g0..g7             : REALIZED / followed action (frozen CoTracker->top-8 PCA)
  r0..r7             : REAL-ride action reference, all 8 PCA dims (encode_z_actions_window)
  musiq,niqe,brisque : no-reference IQA of this chunk's frames
  mae,lpips          : full-reference recon vs the REAL ride chunk (GT branch only)

Any plot (gain, following, circular, roll/pitch, IQA-vs-step, recon) is then a
groupby on analysis/chunk_metrics.csv -- no re-running CoTracker/IQA.

Env:
  CM_RUNS   : comma list of run labels to restrict to (control mode)
  CM_STRIDE : step stride (default 200)
  CM_OUT    : output csv
  CM_DIRS   : comma list of "label:control_test_dir" to process INSTEAD of RUN_DIRS
              (used by the final inject/static evals whose branches are directions/static)
  CM_RECON  : comma list of branch names to full-ref recon vs the real ride (default "gt";
              set "static" for the Phase-B hold-still eval)
  CM_MANIFEST : optional .pt ride manifest (basename->zarr_path,n_latent_frames) to resolve
              the real-ride reference / recon latents for UNSEEN rides not in the eval set
"""
import os, glob, re, json, csv, time
os.environ.setdefault("WORLD_SIZE", "1"); os.environ.setdefault("RANK", "0"); os.environ.setdefault("LOCAL_RANK", "0")
import numpy as np, torch
from utils.ndof_following import (load_pca, teacher_read_video, read_mp4, build_eval_ds,
                                  RUN_DIRS, N_CHUNKS, NFB, chunk_dir)
from utils.zarr_dataset import ZarrRideDataset

DEV = "cuda"
STRIDE = int(os.environ.get("CM_STRIDE", "200"))
OUT = os.environ.get("CM_OUT", "analysis/chunk_metrics.csv")
ONLY = set(os.environ.get("CM_RUNS", "").split(",")) if os.environ.get("CM_RUNS") else None
RECON = set((os.environ.get("CM_RECON", "gt")).split(","))
TOT_F = NFB * N_CHUNKS
CK = "action_query/checkpoints/ss_vae_8free.pt"
VID_RE = re.compile(r"step0*(\d+)_r(\d+)_([A-Za-z]+)_raw\.mp4")

# processing targets: [(label, control_test_dir), ...]
if os.environ.get("CM_DIRS"):
    TARGETS = [(x.split(":", 1)[0], x.split(":", 1)[1]) for x in os.environ["CM_DIRS"].split(",") if ":" in x]
else:
    TARGETS = [(run, os.path.join(base, "control_test")) for run, base in RUN_DIRS.items()]

# optional manifest for unseen-ride real-latent lookup
MANIFEST = {}
if os.environ.get("CM_MANIFEST") and os.path.exists(os.environ["CM_MANIFEST"]):
    for r in torch.load(os.environ["CM_MANIFEST"], map_location="cpu"):
        MANIFEST[os.path.basename(r["zarr_path"])] = (r["zarr_path"], int(r["n_latent_frames"]))


def chunk_slice(frames, k, nchunks=N_CHUNKS):
    """frames [T,H,W,3] -> frames of GENERATED chunk k (0-indexed, seed chunk skipped)."""
    T = frames.shape[0]; seg = max(1, T // nchunks)
    s = (k + 1) * seg; e = min(s + seg, T)
    return frames[s:e]


def _r(x, n=4):
    return "" if (x is None or (isinstance(x, float) and np.isnan(x))) else round(float(x), n)


def main():
    mean, comp_T, scales = load_pca()
    cotracker = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(DEV).eval()
    for p in cotracker.parameters():
        p.requires_grad_(False)
    import pyiqa
    M = {name: pyiqa.create_metric(name, device=DEV) for name in ("musiq", "niqe", "brisque", "lpips")}
    from utils.wan_wrapper import WanVAEWrapper
    vae = WanVAEWrapper().eval().requires_grad_(False).to(DEV)
    eval_ds, by_name = build_eval_ds()

    def _t(fr):
        return torch.tensor(np.ascontiguousarray(fr)).permute(2, 0, 1).unsqueeze(0).float().to(DEV) / 255.

    def iqa(frames):
        if len(frames) == 0:
            return (np.nan, np.nan, np.nan)
        idx = np.unique(np.linspace(0, len(frames) - 1, min(4, len(frames))).astype(int))
        out = {}
        for name in ("musiq", "niqe", "brisque"):
            vals = []
            for i in idx:
                try:
                    vals.append(float(M[name](_t(frames[i])).item()))
                except Exception:
                    pass
            out[name] = float(np.mean(vals)) if vals else np.nan
        return (out["musiq"], out["niqe"], out["brisque"])

    def recon(gen_fr, real_fr):
        n = min(len(gen_fr), len(real_fr))
        if n == 0:
            return (np.nan, np.nan)
        idx = np.unique(np.linspace(0, n - 1, min(4, n)).astype(int))
        maes, lps = [], []
        for i in idx:
            g = _t(gen_fr[i]); r = _t(real_fr[i])
            if g.shape != r.shape:
                r = torch.nn.functional.interpolate(r, size=g.shape[-2:], mode="bilinear", align_corners=False)
            maes.append(float((g - r).abs().mean().item()))
            try:
                lps.append(float(M["lpips"](g, r).item()))
            except Exception:
                pass
        return (float(np.mean(maes)) if maes else np.nan, float(np.mean(lps)) if lps else np.nan)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fout = open(OUT, "w", newline=""); w = csv.writer(fout)
    w.writerow(["run", "step", "rank", "branch", "chunk", "ride", "offset", "dir", "cmd0", "cmd1"]
               + [f"g{d}" for d in range(8)] + [f"r{d}" for d in range(8)]
               + ["musiq", "niqe", "brisque", "mae", "lpips"])

    for run, cdir in TARGETS:
        if ONLY and run not in ONLY:
            continue
        if not os.path.isdir(cdir):
            continue
        cmd = {}
        for f in glob.glob(f"{cdir}/metrics_r*.jsonl"):
            for ln in open(f):
                try:
                    d = json.loads(ln)
                except Exception:
                    continue
                s, rk = d.get("step"), d.get("rank")
                for br, b in d.items():                          # any branch dict with cz2
                    if isinstance(b, dict) and b.get("cz2"):
                        cmd[(s, rk, br)] = (b["cz2"], b["cz7"], d.get("ride"), d.get("offset"))
        vids = sorted(glob.glob(f"{cdir}/step*_r*_*_raw.mp4"))
        vids = [v for v in vids if (m := VID_RE.match(os.path.basename(v)))
                and int(m.group(1)) % STRIDE == 0]
        print(f"[{run}] {len(vids)} videos (stride {STRIDE})", flush=True); t0 = time.time()
        for vi, v in enumerate(vids):
            m = VID_RE.match(os.path.basename(v))
            step, rank, br = int(m.group(1)), int(m.group(2)), m.group(3)
            if (step, rank, br) not in cmd:
                continue
            cz2, cz7, ride, offset = cmd[(step, rank, br)]
            try:
                vid = read_mp4(v)
                frames = vid[0].permute(0, 2, 3, 1).to(torch.uint8).cpu().numpy()
                gen8 = teacher_read_video(vid, cotracker, mean, comp_T, scales)[1:].cpu().numpy()
            except Exception as e:
                print(f"  skip {os.path.basename(v)}: {e}"); continue
            # resolve the real ride (eval-set by_name first, else the unseen manifest)
            zsrc = None
            if eval_ds is not None and ride in by_name:
                zsrc = (by_name[ride]["zarr_path"], by_name[ride]["n_latent_frames"], eval_ds)
            elif ride in MANIFEST:
                zsrc = (MANIFEST[ride][0], MANIFEST[ride][1], eval_ds)
            rr, real_frames = None, None
            if zsrc is not None:
                zp, nlat, _ds = zsrc
                try:
                    if _ds is not None and ride in by_name:
                        z = _ds.encode_z_actions_window(zp, nlat, offset, offset + TOT_F)
                        rr = z.reshape(N_CHUNKS, NFB, 8).mean(1)[1:].cpu().numpy()
                    if br in RECON:
                        rl = ZarrRideDataset.load_latent_chunk(zp, offset, offset + TOT_F).unsqueeze(0).to(DEV).float()
                        latwd = torch.cat([rl[:, 0:1], rl], dim=1)
                        px = vae.decode_to_pixel(latwd.float())[:, 1:, ...]
                        rv = (0.5 * (px.float() + 1)).clamp(0, 1)[0]
                        rv = (rv.cpu().numpy() * 255).astype(np.uint8)
                        if rv.shape[-1] != 3:
                            rv = rv.transpose(0, 2, 3, 1)
                        real_frames = rv
                except Exception:
                    rr = None
            nc = min(len(gen8), len(cz2))
            for k in range(nc):
                cf = chunk_slice(frames, k)
                mu, ni, bri = iqa(cf)
                mae = lp = np.nan
                if br in RECON and real_frames is not None:
                    mae, lp = recon(cf, chunk_slice(real_frames, k))
                dn = chunk_dir(cz2[k], cz7[k], br == "flip")
                row = [run, step, rank, br, k, ride, offset, dn, _r(cz2[k]), _r(cz7[k])]
                row += [_r(gen8[k, d]) for d in range(8)]
                row += ([_r(rr[k, d]) for d in range(8)] if rr is not None else [""] * 8)
                row += [_r(mu, 3), _r(ni, 3), _r(bri, 3), _r(mae), _r(lp)]
                w.writerow(row)
            if (vi + 1) % 50 == 0:
                fout.flush(); print(f"  {vi+1}/{len(vids)} ({(time.time()-t0)/(vi+1):.2f}s/vid)", flush=True)
        fout.flush()
    fout.close()
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
