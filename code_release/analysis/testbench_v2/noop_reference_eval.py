"""No-op (null-action) evaluation with REFERENCE-based metrics.

The only axis where we have ground truth: Phase-B rollouts command zero
action from near-stationary starts (ego ~0.023), so the real ride
continuation IS the reference, and the standard reference battery applies.

  decode : render reference mp4s from the encoded zarr rides through the
           SAME WanVAE decode path as generated rollouts —
           phaseB_windows.json (64) -> noop_refs/refB_r{ii}.mp4
           phaseA_windows.json (32) -> noop_refs/refA_r{ii}.mp4
           (A refs serve the external no-op sets, whose starts are the A scenes)
  eval   : per model — paired PSNR / SSIM / LPIPS per window (generation
           frames only, wall-clock aligned, 6s cap) + FVD (Kinetics r3d_18
           features, Frechet distance gen-set vs ref-set; small-n caveat:
           64 clips per side).

Dynamics binning: still ego does NOT mean still world — actors keep moving,
enter and leave the frame. Windows are binned by the REFERENCE's residual
motion energy (terciles over the ref set: static / mild / dynamic), and every
metric is reported per bin. The liveliness check is motion_ratio =
gen motion / ref motion: a model that freezes the world scores ~0 in the
dynamic bin even though frozen frames can WIN on PSNR/SSIM there — paired
pixel metrics reward dead worlds, which is exactly why the bin + ratio
columns are the ones to trust in dynamic scenes.

Ours: logs/eval_final/B/{run}/control_test/step05000_r{ii}_static_raw.mp4
External: A_{model}_nullact/*_r{NN}_NOOP.mp4 and A_matrixgame_noop (the
authored all-zero action stream) vs refA_r{NN}.

Role: reference-based measurement stage (FVD, PSNR, SSIM, LPIPS).
"""
import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch

ARR = os.environ.get("AF_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, ARR)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fleet  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
REF_DIR = os.path.join(ARR, "analysis", "eval_final", "noop_refs")
TOT_F = 27          # latent frames, matches rollout length
CTX_PX = 12         # first 12 pixel frames (~0.75s) are teacher-forced context
ABLATIONS = {"pca8_8node": "pca8", "pca4": "pca4", "pca2": "pca2",
             "16node": "16node", "4node": "4node", "noatok": "noatok",
             "noadaln": "noadaln"}


# ------------------------------------------------------------------- decode
def decode():
    os.environ.setdefault("ARRWM_ACTION_ENCODER", "pca_raw")
    import imageio
    from utils.zarr_dataset import ZarrRideDataset
    from utils.wan_wrapper import WanVAEWrapper
    os.makedirs(REF_DIR, exist_ok=True)
    vae = WanVAEWrapper().eval().requires_grad_(False).to(DEV)
    for phase, wfile in (("B", "phaseB_windows.json"), ("A", "phaseA_windows.json")):
        windows = json.load(open(os.path.join(ARR, "analysis", "eval_final", wfile)))
        for wi, w in enumerate(windows):
            out = os.path.join(REF_DIR, f"ref{phase}_r{wi:02d}.mp4")
            if os.path.exists(out):
                continue
            try:
                lat = ZarrRideDataset.load_latent_chunk(
                    w["zarr_path"], int(w["offset"]), int(w["offset"]) + TOT_F
                ).unsqueeze(0).to(DEV).float()
                latwd = torch.cat([lat[:, 0:1], lat], dim=1)
                px = vae.decode_to_pixel(latwd)[:, 1:, ...]
                v = (0.5 * (px.float() + 1)).clamp(0, 1)[0].cpu().numpy()
                v = (v * 255).astype(np.uint8)
                if v.shape[-1] != 3:
                    v = v.transpose(0, 2, 3, 1)
                imageio.mimwrite(out, list(v), fps=16, quality=8,
                                 macro_block_size=1)
            except Exception as e:  # noqa: BLE001
                print(f"[skip] {phase} w{wi}: {str(e)[:100]}", flush=True)
        print(f"phase {phase}: refs done", flush=True)


# --------------------------------------------------------------------- eval
class PairedMetrics:
    def __init__(self):
        import pyiqa
        self.lpips = pyiqa.create_metric("lpips", device=DEV)
        self.psnr = pyiqa.create_metric("psnr", device=DEV)
        self.ssim = pyiqa.create_metric("ssim", device=DEV)

    @torch.no_grad()
    def score(self, gen, ref):
        """gen/ref: uint8 [T,H,W,3], same T/H/W. Returns dict of means."""
        g = torch.from_numpy(gen).to(DEV).permute(0, 3, 1, 2).float() / 255.
        r = torch.from_numpy(ref).to(DEV).permute(0, 3, 1, 2).float() / 255.
        out = {}
        for name, m in (("lpips", self.lpips), ("psnr", self.psnr),
                        ("ssim", self.ssim)):
            vals = [float(m(g[i:i + 1], r[i:i + 1])) for i in range(len(g))]
            out[name] = float(np.mean(vals))
        return out


class Fvd:
    """Kinetics r3d_18 features + Frechet distance (repo convention)."""

    def __init__(self):
        from torchvision.models.video import r3d_18, R3D_18_Weights
        m = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
        m.fc = torch.nn.Identity()
        self.model = m.to(DEV).eval()
        self.mean = torch.tensor([0.43216, 0.394666, 0.37645], device=DEV)
        self.std = torch.tensor([0.22803, 0.22145, 0.216989], device=DEV)

    @torch.no_grad()
    def feat(self, frames):
        idx = np.linspace(0, len(frames) - 1, 16).astype(int)
        x = torch.from_numpy(frames[idx]).to(DEV).permute(0, 3, 1, 2).float() / 255.
        x = torch.nn.functional.interpolate(x, size=(112, 112), mode="bilinear",
                                            align_corners=False)
        x = (x - self.mean[:, None, None]) / self.std[:, None, None]
        return self.model(x.permute(1, 0, 2, 3)[None])[0].cpu().numpy()

    @staticmethod
    def frechet(fa, fb):
        from scipy import linalg
        mu1, mu2 = fa.mean(0), fb.mean(0)
        s1 = np.cov(fa, rowvar=False)
        s2 = np.cov(fb, rowvar=False)
        covmean = linalg.sqrtm(s1 @ s2)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(((mu1 - mu2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def motion_energy(frames):
    """Residual motion of a (still-ego) clip: mean absolute gray frame-diff
    over 0.25s steps, plus the fraction of pixels changing by >10 levels.
    Codec noise sits well below both."""
    import cv2
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in frames]).astype(np.int16)
    step = max(1, len(gray) // 24)
    d = np.abs(gray[step:] - gray[:-step])
    return float(d.mean()), float((d > 10).mean())


def load_aligned(gen_path, ref_path, max_gen_sec=6.0):
    """Load both videos, align by wall-clock on the ref's 16fps timeline,
    drop the context, cap at max_gen_sec of generation. Canonical resize
    comes from fleet.load_video.

    Also returns how much of the window the generation actually covers. A
    model that stops early (minwm ends at 4.75s) has its final frame
    re-used by the nearest-time mapping for the remainder, which silently
    manufactures a freeze: 33% of minwm's 6s window is one repeated frame,
    depressing its motion_ratio and inflating its LPIPS. Callers record
    `cover`/`pad_frac` so the artefact is visible in every table rather
    than buried in the aggregate."""
    g, gt, _ = fleet.load_video(gen_path)
    r, rt, _ = fleet.load_video(ref_path)
    t0 = CTX_PX / 16.0
    times = [t for t in rt if t0 <= t <= t0 + max_gen_sec]
    gi = np.array([int(np.argmin(np.abs(gt - t))) for t in times])
    ri = [int(np.argmin(np.abs(rt - t))) for t in times]
    pad = float((gi == gi.max()).mean()) if len(gi) else 1.0
    cover = float(min(1.0, max(0.0, (gt[-1] - t0) / max_gen_sec)))
    return g[gi], r[ri], cover, pad


def evaluate(only=None, max_gen_sec=6.0, tag="", starts="AB"):
    """Score every model, or -- with `only` (a regex on the model name) --
    just the matching ones and MERGE them into the existing out/*.csv,
    leaving the other models' published rows untouched. Dynamics terciles are
    a property of the reference set, so they are recomputed over the merged
    frame and come out identical."""
    import glob as _glob
    sel = re.compile(only) if only else None

    def want(model):
        return sel is None or sel.search(model) is not None

    pm = PairedMetrics()
    fvd = Fvd()
    rows, feats, ref_feats = [], {}, {}

    def add(model, start_set, vid, gen_path, ref_path):
        try:
            g, r, cover, pad = load_aligned(gen_path, ref_path, max_gen_sec)
            gm, gmf = motion_energy(g)
            rm, rmf = motion_energy(r)
            row = dict(model=model, start_set=start_set, vid=vid,
                       ref=os.path.basename(ref_path),
                       gen_motion=gm, ref_motion=rm,
                       gen_moving_frac=gmf, ref_moving_frac=rmf,
                       motion_ratio=gm / (rm + 1e-6),
                       cover=cover, pad_frac=pad, **pm.score(g, r))
            rows.append(row)
            key = (model, start_set)
            feats.setdefault(key, []).append(fvd.feat(g))
            if ref_path not in ref_feats:
                ref_feats[ref_path] = fvd.feat(r)
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {vid}: {str(e)[:100]}", flush=True)

    # ours: static rollouts — phase-B windows AND phase-A starts
    for start_set, root in (("B", os.path.join(ARR, "logs", "eval_final", "B")),
                            ("A", os.path.join(ARR, "logs", "eval_final", "B_Astarts"))):
        if start_set not in starts:
            continue
        n_win = 64 if start_set == "B" else 32
        for run, model in ABLATIONS.items():
            if not want(model):
                continue
            for wi in range(n_win):
                gp = os.path.join(root, run, "control_test",
                                  f"step05000_r{wi:02d}_static_raw.mp4")
                rp = os.path.join(REF_DIR, f"ref{start_set}_r{wi:02d}.mp4")
                if os.path.exists(gp) and os.path.exists(rp):
                    add(model, start_set, f"{model}_{start_set}_r{wi:02d}", gp, rp)
            print(f"{model}/{start_set} scanned", flush=True)

    # external no-op sets on the A starts. Matrix-Game uses an authored
    # all-zero stream (A_matrixgame_noop); the BL/BR workaround is discarded.
    ext_dirs = ((_glob.glob(os.path.join(ARR, "logs", "eval_final", "A_*_nullact"))
                + _glob.glob(os.path.join(ARR, "logs", "eval_final", "A_matrixgame_noop")))
                if "A" in starts else [])
    for d in sorted(ext_dirs):
        model = os.path.basename(d)[2:]          # e.g. worldplay_nullact / matrixgame_noop
        if not want(model):
            continue
        for gp in sorted(_glob.glob(os.path.join(d, "*.mp4"))):
            base = os.path.basename(gp)
            m = re.search(r"_r(\d+)_", base)
            if not m:
                continue
            scene = int(m.group(1))
            rp = os.path.join(REF_DIR, f"refA_r{scene:02d}.mp4")
            if os.path.exists(rp):
                add(model, "A", base.replace(".mp4", ""), gp, rp)

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    df = pd.DataFrame(rows)
    scored = set(df.model.unique())
    paired_p = os.path.join(out_dir, f"noop_paired{tag}.csv")
    if sel is not None and os.path.exists(paired_p):
        old = pd.read_csv(paired_p)
        keep = old[~old.model.isin(scored)].drop(columns=["dyn_bin"],
                                                 errors="ignore")
        df = pd.concat([keep, df], ignore_index=True)
        print(f"merged: kept {len(keep)} rows for "
              f"{sorted(keep.model.unique())}, rescored {sorted(scored)}",
              flush=True)
    # dynamics bins from the REFERENCE motion, terciles computed per start
    # set (per-window property, shared by all models on that window)
    for ss in df.start_set.unique():
        mask = df.start_set == ss
        ref_m = df[mask].drop_duplicates("ref").set_index("ref")["ref_motion"]
        t1, t2 = np.percentile(ref_m.values, [33.3, 66.7])
        df.loc[mask, "dyn_bin"] = pd.cut(df.loc[mask, "ref"].map(ref_m),
                                         [-np.inf, t1, t2, np.inf],
                                         labels=["static", "mild", "dynamic"])
    df.to_csv(paired_p, index=False)

    per_bin = (df.groupby(["model", "start_set", "dyn_bin"], observed=True)
               .agg(n=("vid", "count"), psnr=("psnr", "mean"),
                    ssim=("ssim", "mean"), lpips=("lpips", "mean"),
                    motion_ratio=("motion_ratio", "median")).round(3))
    per_bin.to_csv(os.path.join(out_dir, f"noop_by_dynamics{tag}.csv"))
    print(per_bin.to_string())

    summ = []
    for (model, ss), fl in feats.items():
        fa = np.stack(fl)
        g = df[(df.model == model) & (df.start_set == ss)]
        refs = np.stack([ref_feats[os.path.join(REF_DIR, r)]
                         for r in g["ref"].unique()])
        d = Fvd.frechet(fa, refs) if len(refs) >= 8 else np.nan
        gd = g[g.dyn_bin == "dynamic"]
        summ.append(dict(model=model, start_set=ss, n=len(g),
                         fvd=round(d, 1) if not np.isnan(d) else np.nan,
                         psnr=round(g.psnr.mean(), 2),
                         ssim=round(g.ssim.mean(), 3),
                         lpips=round(g.lpips.mean(), 3),
                         motion_ratio=round(float(g.motion_ratio.median()), 3),
                         motion_ratio_dynamic=round(
                             float(gd.motion_ratio.median()), 3) if len(gd) else np.nan))
    s = pd.DataFrame(summ)
    summ_p = os.path.join(out_dir, f"noop_summary{tag}.csv")
    if sel is not None and os.path.exists(summ_p):
        new_keys = set(zip(s.model, s.start_set))
        old = pd.read_csv(summ_p)
        old = old[[(m, ss) not in new_keys
                   for m, ss in zip(old.model, old.start_set)]]
        s = pd.concat([old, s], ignore_index=True)
    s = s.sort_values(["start_set", "lpips"])
    s.to_csv(summ_p, index=False)
    print(s.to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("decode", "eval", "all"))
    ap.add_argument("--only", default=None,
                    help="regex on model name: score only these and merge "
                         "into the existing out/noop_*.csv")
    ap.add_argument("--max-gen-sec", type=float, default=6.0,
                    help="length of the compared generation window. Use 4.0 "
                         "with --tag _4s for the matched-window control that "
                         "every model, including short-generating minwm, "
                         "covers without a repeated final frame.")
    ap.add_argument("--tag", default="",
                    help="suffix for the output csvs, e.g. _4s")
    ap.add_argument("--starts", default="AB", help="start sets to score")
    args = ap.parse_args()
    if args.cmd in ("decode", "all"):
        decode()
    if args.cmd in ("eval", "all"):
        evaluate(args.only, args.max_gen_sec, args.tag, args.starts)
