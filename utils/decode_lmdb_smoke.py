#!/usr/bin/env python
"""Decode ODE-trajectory LMDB .pt windows to mp4 for visual smoke-verification.

Loads the first N clean .pt files from an LMDB gen output dir, VAE-decodes the
FINAL x0 snapshot (snapshot index -1 = the teacher's fully-denoised output, i.e.
the regression target the ODE student learns), and writes one mp4 per window
plus a contact-sheet PNG of the first frames into ``--out_dir``.

Usage:
  python utils/decode_lmdb_smoke.py --lmdb_dir <dir> --out_dir eval/ode_F_smoke --n 5
"""
import argparse
import os
import sys
from pathlib import Path

# Allow ``python utils/decode_lmdb_smoke.py`` (which puts utils/ on sys.path,
# breaking ``import utils.wan_wrapper``) by prepending the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch


def _save_mp4(frames_thwc_uint8, path, fps):
    """frames: [T, H, W, 3] uint8. Try imageio (ffmpeg), fall back to cv2."""
    try:
        import imageio.v2 as imageio
        imageio.mimwrite(path, list(frames_thwc_uint8), fps=fps, quality=8)
        return True
    except Exception as e:
        print(f"[decode] imageio failed ({e}); trying cv2", flush=True)
    try:
        import cv2
        T, H, W, _ = frames_thwc_uint8.shape
        vw = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H),
        )
        for f in frames_thwc_uint8:
            vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
        vw.release()
        return True
    except Exception as e:
        print(f"[decode] cv2 failed too ({e})", flush=True)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lmdb_dir", required=True)
    ap.add_argument("--out_dir", default="eval/ode_F_smoke")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--fps", type=int, default=12)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from utils.wan_wrapper import WanVAEWrapper
    vae = WanVAEWrapper().to(device).eval()
    vae_dtype = next(vae.parameters()).dtype

    files = sorted(Path(args.lmdb_dir).glob("*.pt"))[: args.n]
    if not files:
        raise SystemExit(f"No .pt files in {args.lmdb_dir}")
    print(f"[decode] {len(files)} windows from {args.lmdb_dir} -> {out_dir}", flush=True)

    first_frames = []
    for i, fp in enumerate(files):
        d = torch.load(fp, map_location="cpu", weights_only=False)
        traj = d["trajectory"]
        # trajectory is a list of (step_idx, latent[F,C,H,W]) OR a tensor
        # [T_snap, F, C, H, W]; the FINAL entry is the teacher x0.
        if isinstance(traj, (list, tuple)):
            final = traj[-1][1] if isinstance(traj[-1], (list, tuple)) else traj[-1]
        else:
            final = traj[-1]
        lat = final.to(device=device, dtype=vae_dtype).unsqueeze(0)  # [1,F,C,H,W]
        with torch.no_grad():
            pix = vae.decode_to_pixel(lat, seed_first=True)  # [1,F,3,H,W] in [-1,1]
        vid = (0.5 * (pix.float() + 1.0)).clamp(0.0, 1.0)[0]  # [F,3,H,W]
        vid = (vid.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)  # [F,H,W,3]
        ride = str(d.get("ride_ts", "?"))
        off = int(d.get("window_offset", i))
        bwd = bool(d.get("backward", d.get("counterfactual", False)))
        name = f"{i:02d}_{ride}_o{off}_{'bwd' if bwd else 'fwd'}.mp4"
        ok = _save_mp4(vid, str(out_dir / name), args.fps)
        first_frames.append(vid[0])
        print(f"[decode] {name}: frames={vid.shape[0]} {vid.shape[1]}x{vid.shape[2]} "
              f"mp4={'OK' if ok else 'FAIL'} ride={ride} off={off} bwd={bwd}", flush=True)

    # Contact sheet of first frames (robust visual check even if mp4 codec is absent).
    try:
        import imageio.v2 as imageio
        sheet = np.concatenate(first_frames, axis=1)  # side-by-side
        imageio.imwrite(str(out_dir / "first_frames_contact_sheet.png"), sheet)
        print(f"[decode] wrote contact sheet ({sheet.shape})", flush=True)
    except Exception as e:
        print(f"[decode] contact sheet skipped: {e}", flush=True)
    print("[decode] done.", flush=True)


if __name__ == "__main__":
    main()
