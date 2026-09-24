"""DINOv2 style trajectory using the deployed fleet_style_6s embedding method.

The 6 s anchor exactly uses its frame formula. Long windows use 16 samples from
their final one second, preserving temporal density across native FPS. Local
adjacent-window drift has no validated threshold.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fleet30s_common as fc  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
WINDOWS = (0, 6, 9, 12, 18, 24)


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(4*1024*1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--scenes")
    p.add_argument("--models")
    p.add_argument("--shard-index", type=int)
    p.add_argument("--shard-count", type=int, default=1)
    a = p.parse_args()
    out = a.out.resolve()
    m = pd.read_csv(out / "video_manifest.csv")
    m = m[m.decoded_frames.notna()]
    if a.scenes:
        m = m[m.scene.isin(a.scenes.split(","))]
    if a.models:
        m = m[m.model.isin(a.models.split(","))]
    if a.shard_index is not None:
        if not 0 <= a.shard_index < a.shard_count:
            raise ValueError("invalid shard")
        m = m.iloc[a.shard_index::a.shard_count]
    import timm
    dino = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True,
                             num_classes=0, img_size=224).cuda().eval()
    @torch.no_grad()
    def embed(frames):
        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float().div(255.0)
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)
        x = ((x-MEAN)/STD).cuda()
        return F.normalize(dino(x), dim=-1).mean(0)
    suffix = f"_shard{a.shard_index}" if a.shard_index is not None else ""
    dest = out / f"style_windows{suffix}.csv"
    old = pd.read_csv(dest) if dest.exists() else pd.DataFrame()
    rows = old.to_dict("records")
    for r in m.itertuples():
        h = digest(r.path)
        prior = old[(old.scene == r.scene)&(old.model == r.model)] if len(old) else pd.DataFrame()
        if len(prior)==len(WINDOWS) and (prior.video_sha256 == h).all():
            continue
        rows = [x for x in rows if (x["scene"],x["model"]) != (r.scene,r.model)]
        ctx, fps, n = int(r.context_frames), float(r.fps), int(r.decoded_frames)
        # In panel32 mode, references are decoded from the immutable canonical
        # source rather than a model's reconstructed output prefix.  The span
        # remains native: one-image systems use frame 32, minWM ODE uses its
        # final 13 real frames, and longer-context systems use the final 16.
        if fc.PANEL32_MODE:
            uid = r.uid
            ref_frames, ref_idx = fc.source_frames(uid, r.model, 16)
        else:
            ref_idx = list(range(max(0, ctx - 16), ctx))
            ref_frames = None
        ends = {}
        for s in WINDOWS:
            end = ctx + int(round((s+6)*fps)) - (0 if s == 0 else 1)
            if end >= n:
                raise IndexError((r.scene,r.model,s,end,n))
            if s == 0:
                indices = list(range(max(ctx,end-15),end+1))
            else:
                first = max(ctx+int(round((s+5)*fps)),end-int(round(fps))+1)
                indices = np.linspace(first,end,16).round().astype(int).tolist()
            ends[s] = indices
        ix = sorted(set(sum(ends.values(),[]))) if fc.PANEL32_MODE else sorted(set(ref_idx+sum(ends.values(),[])))
        fr = fc.frames_at(r.scene,r.model,ix)
        frame = dict(zip(ix,fr))
        ref = embed(ref_frames if ref_frames is not None else [frame[i] for i in ref_idx])
        prev = None
        for s in WINDOWS:
            e = embed([frame[i] for i in ends[s]])
            real = float(1-(ref@e))
            local = float(1-(prev@e)) if prev is not None else np.nan
            rows.append(dict(scene=r.scene,model=r.model,window_start_s=s,window_end_s=s+6,
                             window_role="15s_endpoint" if s==9 else "nonoverlapping",
                             end_indices=json.dumps(ends[s]), anchor_indices=json.dumps(ref_idx),
                             anchor_source=("canonical_source" if fc.PANEL32_MODE else "model_output_prefix"),
                             drift_from_real=round(real,4),local_adjacent_drift=round(local,4) if np.isfinite(local) else np.nan,
                             style_flag_072=int(round(real,4)>0.72) if s==0 else np.nan,
                             video_sha256=h,producer="code_release/evaluation/quality/fleet_style_6s.py",
                             local_threshold_status="unvalidated"))
            prev = e if s != 9 else prev  # overlapping 9-15 window does not replace 6-12 in local chain
        tmp = dest.with_suffix(".tmp")
        pd.DataFrame(rows).to_csv(tmp,index=False)
        os.replace(tmp, dest)
        print(r.scene,r.model,flush=True)
    if rows:
        assert not pd.DataFrame(rows).duplicated(["scene","model","window_start_s"]).any()


if __name__ == "__main__":
    main()
