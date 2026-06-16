#!/usr/bin/env python3
"""Side-by-side FORWARD | REVERSE alignment-check video with action overlay.

Decodes a window of forward latents and the WALL-CLOCK-ALIGNED rear latents
(via the <ride>.align.npy map from build_rear_alignment.py), overlays the
egomotion z2 (steering/yaw) and z7 (throttle) on each panel — forward uses the
ride's z, reverse uses the SIGN-FLIPPED z (z2->-z2, z7->-z7), exactly as the
dual-view model will be conditioned — and writes an mp4 so we can visually
confirm the two cameras show the same moment.

Decode uses the project's dummy-leading-latent convention (utils/eval_chain.py:503)
and the cached decode path (use_cache=True) — Study 3's "cached decode".
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import zarr
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.wan_wrapper import WanVAEWrapper
from utils.zarr_dataset import ZarrRideDataset, _index_single_zarr


@torch.no_grad()
def decode(vae, lat_np, device):
    """lat_np: [n,16,60,104] -> uint8 frames [T,H,W,3] via dummy-prepend + cached decode."""
    lat = torch.from_numpy(lat_np.astype(np.float32))[None].to(device)  # [1,n,16,60,104]
    lat_wd = torch.cat([lat[:, 0:1], lat], dim=1)
    px = vae.decode_to_pixel(lat_wd, use_cache=True)[:, 1:]  # [1,T,3,H,W]
    vid = (0.5 * (px.float() + 1.0)).clamp(0, 1)[0].cpu().numpy()  # [T,3,H,W]
    vid = (vid.transpose(0, 2, 3, 1) * 255).astype(np.uint8)      # [T,H,W,3]
    return vid


def overlay(frames, z2, z7, label, color=(0, 255, 0)):
    """Draw panel label + per-frame steering arrow (z2) and throttle bar (z7).
    z2,z7: per-LATENT arrays; we map frame t -> latent via linear stretch."""
    out = []
    T, H, W, _ = frames.shape
    n = len(z2)
    for t in range(T):
        f = np.ascontiguousarray(frames[t])
        li = min(n - 1, int(t * n / max(1, T)))
        s, thr = float(z2[li]), float(z7[li])
        cv2.putText(f, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(f, f"z2(steer)={s:+.2f}  z7(thr)={thr:+.2f}", (8, H - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        # steering arrow (horizontal, centred)
        cx, cy = W // 2, 40
        cv2.arrowedLine(f, (cx, cy), (int(cx + s * 60), cy), (0, 200, 255), 2, tipLength=0.3)
        # throttle bar (vertical, right edge): up=forward(+), down=back(-)
        bx = W - 18
        cv2.line(f, (bx, cy), (bx, int(cy - thr * 60)), (0, 165, 255), 6)
        out.append(f)
    return np.stack(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ride_ts", required=True)
    ap.add_argument("--forward_root", default="/projects/u6ex/fbots/frodobots_encoded_weu")
    ap.add_argument("--rear_root", default="/projects/u6ex/fbots/frodobots_encoded_weu_rear")
    ap.add_argument("--caption_root", default="/projects/u6ex/fbots/frodobots_captions/train")
    ap.add_argument("--motion_root", default="/projects/u6ex/fbots/frodobots_motion")
    ap.add_argument("--ss_vae_checkpoint", default="action_query/checkpoints/ss_vae_8free.pt")
    ap.add_argument("--vae_root", default="/scratch/u6ex/as1748.u6ex/frodobots/Wan2.1-T2V-1.3B")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--n_latents", type=int, default=90)
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    fz = os.path.join(args.forward_root, f"{args.ride_ts}.zarr")
    rz = os.path.join(args.rear_root, f"{args.ride_ts}.zarr")
    amap = np.load(os.path.join(args.rear_root, f"{args.ride_ts}.align.npy"))

    Tf = int(zarr.open(fz, "r")["latents"].shape[0])
    # Clamp the window into the aligned span so we visualise where forward and
    # rear actually overlap (rear typically starts a few seconds after front).
    valid_idx = np.where(amap >= 0)[0]
    a0 = int(valid_idx[0]) if valid_idx.size else 0
    a1 = int(valid_idx[-1]) + 1 if valid_idx.size else Tf
    start = max(args.start, a0)
    end = min(start + args.n_latents, a1, Tf)
    sl = slice(start, end)
    fwd_lat = zarr.open(fz, "r")["latents"][sl]                     # [n,16,60,104]
    ridx = amap[sl]
    valid = ridx >= 0
    # gather rear latents at aligned indices; fill gaps with nearest valid
    rear_all = zarr.open(rz, "r")["latents"]
    safe = np.where(valid, ridx, 0)
    rev_lat = rear_all[:][safe]                                     # [n,16,60,104]
    print(f"ride {args.ride_ts}: forward[{start}:{end}] aligned rear {valid.sum()}/{len(valid)} "
          f"(rear idx {ridx[valid].min() if valid.any() else '-'}..{ridx[valid].max() if valid.any() else '-'})")

    # z (forward) + flipped (reverse), via ss_vae over the same window
    pe, attrs, n_lat = _index_single_zarr(Path(fz), Path(args.caption_root), Path(args.motion_root))
    ds = ZarrRideDataset.from_manifest(
        [{"zarr_path": fz, "prompt_embeds": pe, "attrs": attrs, "n_latent_frames": n_lat}],
        motion_root=args.motion_root, ss_vae_checkpoint=args.ss_vae_checkpoint, device=device)
    z = ds.encode_z_actions_window(fz, n_lat, start, end).cpu().numpy()   # [n,8]
    z2f, z7f = z[:, 2], z[:, 7]
    z2r, z7r = -z[:, 2], -z[:, 7]

    vae = WanVAEWrapper(model_root=args.vae_root).to(device).eval()
    vid_f = overlay(decode(vae, fwd_lat, device), z2f, z7f, "FORWARD", (0, 255, 0))
    vid_r = overlay(decode(vae, rev_lat, device), z2r, z7r, "REVERSE (z flipped)", (0, 128, 255))

    T = min(len(vid_f), len(vid_r))
    combo = np.concatenate([vid_f[:T], vid_r[:T]], axis=2)         # side-by-side on width
    out = args.out or f"eval/dual_view/{args.ride_ts}_dual.mp4"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    H, W = combo.shape[1:3]
    vw = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
    for f in combo:
        vw.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    vw.release()
    print(f"wrote {out}  ({T} frames, {W}x{H})")


if __name__ == "__main__":
    main()
