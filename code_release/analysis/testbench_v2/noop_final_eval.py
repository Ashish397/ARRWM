"""FINAL stationary (no-op) evaluation -- the main-paper behaviour plane.

A zero action command must satisfy two things at once:
  (1) the camera stays put, and
  (2) the independently moving world keeps moving.
This measures each one separately and places every model in that 2-D plane.

  Held-Still (%)              x-axis, higher better
  Residual scene-motion ratio y-axis, 1 = real

Pipeline (all constants are declared here and echoed into NOOP_FINAL.md):

SHARED SET      the 32 Phase-A contexts available for EVERY system, same scene
                ids throughout, each against its paired real continuation.
HORIZON         wall-clock [0.75 s, 4.75 s] -- exactly 4.0 s of generation
                measured from the true generation boundary (frame 12 of the
                16 fps references). Nothing is padded, repeated or
                extrapolated; every system covers the closed interval (minwm
                exactly, its final frame lands on t = 4.75 s).
TEMPORAL        resampled onto a common 16 fps grid, 65 timestamps
                t_k = 0.75 + k/16, by NEAREST source frame (never blended --
                temporal interpolation would ghost and corrupt optical flow).
                Per-system max temporal snap error is recorded; it is bounded
                by half a source frame interval and averages out over the 64
                pairs, since D and M are means over t.
SPATIAL         isotropic (aspect-preserving) resize onto an 832x480 canvas
                with symmetric letterbox padding -- never anisotropic stretch.
                Padded pixels are excluded from every flow statistic, and the
                valid mask is eroded by 8 px first so letterbox edges cannot
                leak into the flow field.
FLOW            RAFT-small, torchvision Raft_Small_Weights.DEFAULT.
                Per adjacent pair: dense F_t, then a RANSAC-fitted GLOBAL 2D
                AFFINE (primary) giving induced global flow G_t, residual
                R_t = F_t - G_t. A homography fit runs on the same flow for
                the appendix sensitivity check -- the main result never
                silently switches to it.
METRICS         D_i = mean_t median_x ||G_t(x)|| / dt        (px/s)
                M_i = mean_t Q0.90_x ||R_t(x)|| / dt         (px/s)
                tau_D = 95th pct of D over ALL real continuations (A+B, 96)
                held_i = 1[D_i <= tau_D];  Held-Still = 100 * mean_i held_i
                r_i = (M_gen + eps)/(M_real + eps),  eps = 0.5 px/s a priori
                Ratio = exp(mean_i log r_i) over the REFERENCE-DYNAMIC bin
                only (a static-scene denominator is noise and would make a
                frozen model look alive).
BASELINES       Real  -- the reference continuations, through this same
                         pipeline, never hand-placed.
                Freeze -- the final real CONTEXT frame (t = 0.6875 s) held for
                         the whole 4 s horizon, through the same pipeline.

Stages:  flow -> paired -> report   (or `all`).
Shard the flow stage with TB2_SHARD / TB2_NSHARD.

Role: source of the main-paper stationary numbers (flow / paired / report stages).
"""
import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ARR = os.environ.get("AF_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
OUT = os.path.join(HERE, "out")
FIG = os.path.join(OUT, "figs")
sys.path.insert(0, HERE)
sys.path.insert(0, ARR)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
REF_DIR = os.path.join(ARR, "analysis", "eval_final", "noop_refs")

# ------------------------------------------------------------------ constants
CANVAS_W, CANVAS_H = 832, 480     # isotropic-fit canvas
T0, T1 = 0.75, 4.75               # generation boundary -> +4.0 s
GRID_FPS = 16                     # common resampling grid
DT = 1.0 / GRID_FPS               # 0.0625 s between adjacent grid frames
N_T = int(round((T1 - T0) * GRID_FPS)) + 1        # 65 timestamps, 64 pairs
TIMES = T0 + np.arange(N_T) / GRID_FPS
CTX_LAST_T = 0.6875               # frame 11 = final real context frame
GRID = 8                          # correspondence grid stride (px)
MASK_ERODE = 8                    # px eroded off the valid mask before stats
RANSAC_THR = 2.0                  # px reprojection threshold
RANSAC_ITERS = 2000
RANSAC_CONF = 0.995
FB_THRESH = 1.5                   # px forward/backward consistency threshold
EPSILON = 0.5                     # px/s ratio regulariser, fixed A PRIORI
Q_MAIN = 0.90                     # residual quantile for the headline metric
Q_SENS = (0.80, 0.90, 0.95)
TAU_PCT = 95                      # percentile of real D that defines "still"
N_BOOT = 2000                     # >= 1000 paired scene-bootstrap samples
BOOT_SEED = 20260725
N_SCENES = 32
LPIPS_N = 16                      # uniformly spaced paired timestamps

ABLATIONS = {"pca8_8node": "pca8", "pca4": "pca4", "pca2": "pca2",
             "16node": "16node", "4node": "4node", "noatok": "noatok",
             "noadaln": "noadaln"}
EXT_DIRS = {"matrixgame": "A_matrixgame_noop"}     # rest are A_{name}_nullact
EXT = ["minwm", "worldcam", "worldplay", "yume", "astra", "matrixgame"]
# Neutral-prompt reruns: identical pose/action null, but with every motion word
# stripped from the text conditioning (WorldCam additionally had 静态/静止/
# 静止不动的画面 -- static/motionless -- in its NEGATIVE prompt, i.e. it was
# being actively penalised for holding still).
EXT_NP = ["minwm_np", "worldcam_np", "worldplay_np", "yume_np", "astra_np"]
EXT_DIRS.update({f"{m}_np": f"A_{m}_noop_neutral" for m in
                 ["minwm", "worldcam", "worldplay", "yume", "astra"]})


# ------------------------------------------------------------------- clip I/O
def _src_paths(system, scene):
    """Path for (system, scene). Returns None when the system has no video
    (real/freeze are synthesised from the references)."""
    if system in ABLATIONS.values():
        run = [k for k, v in ABLATIONS.items() if v == system][0]
        return os.path.join(ARR, "logs/eval_final/B_Astarts", run,
                            "control_test", f"step05000_r{scene:02d}_static_raw.mp4")
    if system in EXT or system in EXT_NP:
        d = EXT_DIRS.get(system, f"A_{system}_nullact")
        hits = sorted(glob.glob(os.path.join(ARR, "logs/eval_final", d,
                                             f"*_r{scene:02d}_*.mp4")))
        return hits[0] if hits else None
    if system in ("real", "freeze"):
        return os.path.join(REF_DIR, f"refA_r{scene:02d}.mp4")
    if system == "real_B":
        return os.path.join(REF_DIR, f"refB_r{scene:02d}.mp4")
    raise KeyError(system)


def _letterbox(img, cache={}):
    """Isotropic resize onto the canvas + symmetric padding. Returns the
    canvas image and the (cached) valid mask for this source size."""
    import cv2
    h, w = img.shape[:2]
    key = (w, h)
    if key not in cache:
        s = min(CANVAS_W / w, CANVAS_H / h)
        nw, nh = int(round(w * s)), int(round(h * s))
        x0, y0 = (CANVAS_W - nw) // 2, (CANVAS_H - nh) // 2
        m = np.zeros((CANVAS_H, CANVAS_W), bool)
        m[y0:y0 + nh, x0:x0 + nw] = True
        cache[key] = (nw, nh, x0, y0, m)
    nw, nh, x0, y0, m = cache[key]
    out = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    out[y0:y0 + nh, x0:x0 + nw] = cv2.resize(img, (nw, nh),
                                             interpolation=cv2.INTER_AREA)
    return out, m


def load_clip(system, scene):
    """-> frames uint8 [65,480,832,3] on the common grid, valid mask, meta.

    Raises if the source does not cover the closed horizon: nothing here is
    allowed to pad or extrapolate."""
    import av
    path = _src_paths(system, scene)
    if path is None or not os.path.exists(path):
        raise FileNotFoundError(f"{system} r{scene:02d}")
    container = av.open(path)
    st = container.streams.video[0]
    fps = float(st.average_rate)
    raw, times = [], []
    for i, fr in enumerate(container.decode(video=0)):
        t = i / fps
        if t > T1 + 1.0:
            break
        raw.append(fr.to_ndarray(format="rgb24"))
        times.append(t)
    container.close()
    times = np.asarray(times)
    if times[-1] + 1e-9 < T1:
        raise ValueError(f"{system} r{scene:02d} ends at {times[-1]:.4f}s "
                         f"< {T1}s -- would require padding")

    if system == "freeze":
        # the final real CONTEXT frame, held for the whole horizon
        j = int(np.argmin(np.abs(times - CTX_LAST_T)))
        f0, mask = _letterbox(raw[j])
        frames = np.repeat(f0[None], N_T, axis=0)
        return frames, mask, dict(native_fps=fps, snap_err=0.0,
                                  src_path=path, frozen_at=float(times[j]))

    idx = [int(np.argmin(np.abs(times - t))) for t in TIMES]
    snap = float(np.max(np.abs(times[idx] - TIMES)))
    outs = [_letterbox(raw[i]) for i in idx]
    frames = np.stack([o[0] for o in outs])
    return frames, outs[0][1], dict(native_fps=fps, snap_err=snap,
                                    src_path=path)


# ---------------------------------------------------------------- flow stage
class FlowDecomp:
    """RAFT-small + RANSAC global-motion fit, affine (primary) and
    homography (appendix sensitivity) from the SAME flow field."""

    def __init__(self, fb_check=True, batch=8):
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        self.weights = "Raft_Small_Weights.DEFAULT"
        self.model = raft_small(weights=Raft_Small_Weights.DEFAULT).to(DEV).eval()
        self.fb_check = fb_check
        self.batch = batch

    @staticmethod
    def _prep(a):
        x = torch.from_numpy(a).to(DEV).permute(0, 3, 1, 2).float() / 127.5 - 1.0
        return x

    @torch.no_grad()
    def _flow(self, a, b):
        return self.model(self._prep(a), self._prep(b))[-1]

    @torch.no_grad()
    def clip(self, frames, mask):
        """Per-pair global/residual decomposition over the whole clip."""
        import cv2
        from scipy.ndimage import binary_erosion
        m = binary_erosion(mask, np.ones((2 * MASK_ERODE + 1,) * 2))
        ys, xs = np.mgrid[GRID // 2:CANVAS_H:GRID, GRID // 2:CANVAS_W:GRID]
        keep = m[ys, xs]
        ys, xs = ys[keep], xs[keep]
        p1 = np.stack([xs, ys], 1).astype(np.float64)

        rows = []
        for s in range(0, N_T - 1, self.batch):
            e = min(s + self.batch, N_T - 1)
            a, b = frames[s:e], frames[s + 1:e + 1]
            fw = self._flow(a, b)
            bw = self._flow(b, a) if self.fb_check else None
            for j in range(e - s):
                f = fw[j].cpu().numpy()
                d = np.stack([f[0, ys, xs], f[1, ys, xs]], 1).astype(np.float64)
                ok = np.isfinite(d).all(1)
                if self.fb_check:
                    # warp the backward flow to the forward endpoints; a
                    # consistent pixel maps back to within FB_THRESH px
                    bwn = bw[j].cpu().numpy()
                    tx = np.clip(np.round(xs + d[:, 0]).astype(int), 0, CANVAS_W - 1)
                    ty = np.clip(np.round(ys + d[:, 1]).astype(int), 0, CANVAS_H - 1)
                    back = np.stack([bwn[0, ty, tx], bwn[1, ty, tx]], 1)
                    ok &= (np.hypot(*(d + back).T) <= FB_THRESH) & m[ty, tx]
                rows.append(self._one(p1, d, ok, cv2))
        return rows

    @staticmethod
    def _one(p1, d, ok, cv2):
        """Fit affine + homography on the kept correspondences."""
        out = dict(n_pts=int(ok.sum()), n_grid=len(p1))
        if ok.sum() < 32:
            return dict(out, ok=False)
        q1, q2 = p1[ok], (p1 + d)[ok]
        res = dict(out, ok=True)

        A, inl = cv2.estimateAffine2D(q1, q2, method=cv2.RANSAC,
                                      ransacReprojThreshold=RANSAC_THR,
                                      maxIters=RANSAC_ITERS,
                                      confidence=RANSAC_CONF)
        if A is None:
            res.update(fit_affine=False, D_affine=np.nan, inl_affine=0.0)
            for q in Q_SENS:
                res[f"M{int(q * 100)}_affine"] = np.nan
        else:
            proj = q1 @ A[:, :2].T + A[:, 2]
            g = np.hypot(*(proj - q1).T)          # induced global flow
            r = np.hypot(*(q2 - proj).T)          # residual
            res.update(fit_affine=True, D_affine=float(np.median(g)),
                       inl_affine=float(inl.mean()) if inl is not None else np.nan)
            for q in Q_SENS:
                res[f"M{int(q * 100)}_affine"] = float(np.quantile(r, q))

        H, inh = cv2.findHomography(q1, q2, cv2.RANSAC, RANSAC_THR,
                                    maxIters=RANSAC_ITERS, confidence=RANSAC_CONF)
        if H is None:
            res.update(fit_homog=False, D_homog=np.nan, M90_homog=np.nan,
                       inl_homog=0.0)
        else:
            ph = cv2.perspectiveTransform(q1[None].astype(np.float32),
                                          H)[0].astype(np.float64)
            res.update(fit_homog=True,
                       D_homog=float(np.median(np.hypot(*(ph - q1).T))),
                       M90_homog=float(np.quantile(np.hypot(*(q2 - ph).T), Q_MAIN)),
                       inl_homog=float(inh.mean()) if inh is not None else np.nan)
        return res


def systems_list(include_B=True):
    s = ["real", "freeze"] + list(ABLATIONS.values()) + EXT
    if include_B:
        s.append("real_B")
    return s


def stage_flow():
    import pandas as pd
    fd = FlowDecomp()
    sysl = systems_list()
    sh, ns = int(os.environ.get("TB2_SHARD", 0)), int(os.environ.get("TB2_NSHARD", 1))
    jobs = [(s, sc) for s in sysl
            for sc in range(64 if s == "real_B" else N_SCENES)]
    jobs = [j for i, j in enumerate(jobs) if i % ns == sh]
    print(f"[flow] {len(jobs)} clips (shard {sh}/{ns})", flush=True)

    per_clip, per_pair = [], []
    for n, (system, scene) in enumerate(jobs):
        try:
            frames, mask, meta = load_clip(system, scene)
        except Exception as e:  # noqa: BLE001
            print(f"[MISSING] {system} r{scene:02d}: {str(e)[:110]}", flush=True)
            per_clip.append(dict(system=system, scene=scene, missing=True,
                                 reason=str(e)[:200]))
            continue
        rows = fd.clip(frames, mask)
        df = pd.DataFrame(rows)
        good = df[df.ok.astype(bool)] if "ok" in df else df
        for r in rows:
            per_pair.append(dict(system=system, scene=scene, **r))
        if not len(good):
            print(f"[FITFAIL] {system} r{scene:02d}: no usable pair", flush=True)
            per_clip.append(dict(system=system, scene=scene, missing=True,
                                 reason="all frame pairs failed the fit"))
            continue
        n_fail = int((~good["fit_affine"].astype(bool)).sum()) \
            if "fit_affine" in good else 0
        rec = dict(system=system, scene=scene, missing=False,
                   native_fps=meta["native_fps"], snap_err=meta["snap_err"],
                   n_pairs=len(df), valid_frac=float(mask.mean()),
                   fit_fail_affine=n_fail,
                   inl_affine=float(good.inl_affine.mean()),
                   inl_homog=float(good.inl_homog.mean()),
                   # D and M are means over t of a per-frame statistic, /dt
                   D_affine=float(good.D_affine.mean()) / DT,
                   D_homog=float(good.D_homog.mean()) / DT,
                   M90_homog=float(good.M90_homog.mean()) / DT)
        for q in Q_SENS:
            k = f"M{int(q * 100)}_affine"
            rec[k] = float(good[k].mean()) / DT
        per_clip.append(rec)
        if (n + 1) % 25 == 0:
            print(f"[flow] {n + 1}/{len(jobs)}", flush=True)

    suf = f".shard{sh}" if ns > 1 else ""
    pd.DataFrame(per_clip).to_csv(os.path.join(OUT, f"noop_final_flow{suf}.csv"),
                                  index=False)
    pd.DataFrame(per_pair).to_csv(os.path.join(OUT, f"noop_final_pairs{suf}.csv"),
                                  index=False)
    print(f"[flow] wrote {len(per_clip)} clips, {len(per_pair)} pairs", flush=True)


# -------------------------------------------------------------- paired stage
def stage_paired():
    """FVD (R3D-18) + LPIPS on the SAME preprocessed clips.

    Both are computed inside the model's valid (unpadded) region: a
    letterboxed model would otherwise be scored against reference content
    that its own black bars cover. gen and ref are cropped to the identical
    box, so each model is compared like with like."""
    import pandas as pd
    import pyiqa
    from torchvision.models.video import r3d_18, R3D_18_Weights

    lpips = pyiqa.create_metric("lpips", device=DEV)
    r3d = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
    r3d.fc = torch.nn.Identity()
    r3d = r3d.to(DEV).eval()
    mean = torch.tensor([0.43216, 0.394666, 0.37645], device=DEV)
    std = torch.tensor([0.22803, 0.22145, 0.216989], device=DEV)

    @torch.no_grad()
    def feat(fr):
        i = np.linspace(0, len(fr) - 1, 16).astype(int)
        x = torch.from_numpy(fr[i]).to(DEV).permute(0, 3, 1, 2).float() / 255.
        x = torch.nn.functional.interpolate(x, size=(112, 112), mode="bilinear",
                                            align_corners=False)
        x = (x - mean[:, None, None]) / std[:, None, None]
        return r3d(x.permute(1, 0, 2, 3)[None])[0].cpu().numpy()

    li = np.linspace(0, N_T - 1, LPIPS_N).astype(int)
    rows, feats = [], {}
    for system in systems_list(include_B=False):
        for scene in range(N_SCENES):
            try:
                g, gm, _ = load_clip(system, scene)
                r, rm, _ = load_clip("real", scene)
            except Exception as e:  # noqa: BLE001
                print(f"[MISSING] {system} r{scene:02d}: {str(e)[:90]}", flush=True)
                continue
            ys, xs = np.where(gm & rm)          # common valid box
            y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
            gc, rc = g[:, y0:y1, x0:x1], r[:, y0:y1, x0:x1]
            with torch.no_grad():
                a = torch.from_numpy(gc[li]).to(DEV).permute(0, 3, 1, 2).float() / 255.
                b = torch.from_numpy(rc[li]).to(DEV).permute(0, 3, 1, 2).float() / 255.
                lp = float(np.mean([float(lpips(a[i:i + 1], b[i:i + 1]))
                                    for i in range(len(a))]))
            rows.append(dict(system=system, scene=scene, lpips=lp))
            feats.setdefault(system, []).append(feat(gc))
            feats.setdefault(f"__ref__{system}", []).append(feat(rc))
        print(f"[paired] {system} done", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(OUT, "noop_final_lpips.csv"),
                              index=False)
    np.savez(os.path.join(OUT, "noop_final_feats.npz"),
             **{k: np.stack(v) for k, v in feats.items()})
    print("[paired] wrote lpips + fvd features", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=("flow", "paired", "report", "all"))
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    if a.stage in ("flow", "all"):
        stage_flow()
    if a.stage in ("paired", "all"):
        stage_paired()
    if a.stage in ("report", "all"):
        import noop_final_report
        noop_final_report.main()
