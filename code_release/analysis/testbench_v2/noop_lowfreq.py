"""Reference-FREE low-frequency (mangle) + geometry scoring of the no-op
rollouts. A stationary command should be the EASY case (little warping), so
high mangle here is especially damning. Reuses the validated detector
classes; scores each no-op video with Qwen-melt, PAL4VST, depth roughness
(the 0.887-AUC ensemble ingredients) plus the geometry votes.

Sources:
  ablation A-starts : logs/eval_final/B_Astarts/{run}/control_test/
                      step05000_r{NN}_static_raw.mp4
  external no-op    : logs/eval_final/A_{model}_nullact/*_r{NN}_*.mp4

Role: reference-free cross-check of the no-op rollouts, complementing the reference-based stage.
"""
import glob
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ARR = os.environ.get("AF_ROOT", os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, HERE)
import fleet
import mangle_metrics as MM
import geometry_metrics as GM

ABL = {"pca8_8node": "pca8", "pca4": "pca4", "pca2": "pca2", "16node": "16node",
       "4node": "4node", "noatok": "noatok", "noadaln": "noadaln"}


def noop_refs():
    out = []
    for run, model in ABL.items():
        for p in sorted(glob.glob(os.path.join(
                ARR, "logs/eval_final/B_Astarts", run, "control_test",
                "step05000_r*_static_raw.mp4"))):
            sc = int(os.path.basename(p).split("_r")[1][:2])
            out.append(fleet.VideoRef(model, sc, "NOOP", p))
    ext_dirs = (glob.glob(os.path.join(ARR, "logs/eval_final/A_*_nullact"))
                + glob.glob(os.path.join(ARR, "logs/eval_final/A_matrixgame_noop")))
    for d in sorted(ext_dirs):
        model = os.path.basename(d)[2:]           # e.g. worldplay_nullact / matrixgame_noop
        for p in sorted(glob.glob(os.path.join(d, "*_r*_*.mp4"))):
            b = os.path.basename(p)
            sc = int(b.split("_r")[1][:2])
            out.append(fleet.VideoRef(model, sc, b.split("_")[-1][:-4], p))
    # the real continuations, scored through the identical detector path, so
    # the mangle THRESHOLD can be anchored to real video instead of a picked
    # constant (model name 'real_ref' -- a floor, not a competitor)
    for p in sorted(glob.glob(os.path.join(
            ARR, "analysis/eval_final/noop_refs/refA_r*.mp4"))):
        sc = int(os.path.basename(p).split("_r")[1][:2])
        out.append(fleet.VideoRef("real_ref", sc, "NOOP", p))
    return out


def main():
    refs = noop_refs()
    sh, ns = int(os.environ.get("TB2_SHARD", 0)), int(os.environ.get("TB2_NSHARD", 1))
    refs = [r for i, r in enumerate(refs) if i % ns == sh]
    # TB2_ONLY=<regex>: score just these models and merge into the existing
    # csv, so already-published rows are not re-derived.
    only = os.environ.get("TB2_ONLY")
    if only:
        import re
        rx = re.compile(only)
        refs = [r for r in refs if rx.search(r.model)]
    out = os.path.join(HERE, "out", f"noop_lowfreq.shard{sh}.csv" if ns > 1
                       else "noop_lowfreq.csv")

    pal, qwen, depth = MM.Pal4vst(), MM.QwenMelt(), GM.DepthField()
    print(f"{len(refs)} no-op videos", flush=True)
    rows = []
    for i, ref in enumerate(refs):
        try:
            frames, times, _ = fleet.load_video(ref.path)
            ts = fleet.gen_fraction_times(times, MM.N_FRAMES)
            gen = [fleet.frame_at(frames, times, t) for t in ts]
            _, base, end = fleet.windows(times)
            rows.append(dict(
                model=ref.model, scene=ref.scene, direction=ref.direction,
                vid=ref.vid,
                qwen_melt_pyes=qwen.video_score(gen),
                pal4vst_max=float(np.max([pal.frame_frac(f) for f in gen])),
                depth_rough_base=float(np.mean(
                    [depth.roughness(depth.disparity(frames[j]))
                     for j in base[np.linspace(0, len(base) - 1, 3).astype(int)]])),
            ))
        except Exception as e:  # noqa: BLE001
            print(f"[skip] {ref.vid}: {str(e)[:90]}", flush=True)
        if (i + 1) % 20 == 0:
            print(f"{i + 1}/{len(refs)}", flush=True)
            # checkpoint beside the real csv, never over it (merge mode would
            # otherwise clobber the kept rows on a crash)
            pd.DataFrame(rows).to_csv(out + ".partial", index=False)

    df = pd.DataFrame(rows)
    if only and os.path.exists(out):
        old = pd.read_csv(out)
        keep = old[~old.model.isin(set(df.model.unique()))]
        print(f"merged: kept {len(keep)} rows for {sorted(keep.model.unique())}",
              flush=True)
        df = pd.concat([keep, df], ignore_index=True)
    df.to_csv(out, index=False)
    print(f"wrote {len(df)} -> {out}")


if __name__ == "__main__":
    main()
