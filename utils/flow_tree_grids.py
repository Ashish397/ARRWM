"""Tile flow-tree AR-timeline mp4s into grids mirroring the video GRIDs.

Each video grid (.motion_check/GRID*_r08_*.mp4) compares N methods on the
same rollout; this builds the matching FLOW-TREE grid so the latent-space
picture can be read panel-for-panel against the pixels.

Out: flow_viz/FTGRID_{name}.mp4
Env: FTG_ONLY (colon list of grid names to build)
"""
import os
import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw

ARR = "/scratch/u6ex/as1748.u6ex/ARRWM"
FV = f"{ARR}/analysis/eval_final/flow_viz"
FP = dict(input_params=["-threads", "1"], output_params=["-threads", "1"])

# name -> (video-grid counterpart, [tree labels in panel order])
GRIDS = {
    "grid3_objectives":  ("GRID3_nrmom2",  ["teacher", "v1std", "nr", "nrmom2"]),
    "grid4_alldir8":     ("GRID4_alldir8", ["teacher", "v1std", "alldir8", "alldir8kl"]),
    "grid5_dir8nboard":  ("GRID5_dir8nboard", ["teacher", "alldir8n", "msecg8n", "kl8n"]),
    "grid5_serve":       ("GRID5_serve",   ["teacher", "alldir8n", "a8n_rm10", "a8n_rmauto2"]),
    "grid6_emdfamily":   ("GRID6_emdfamily", ["teacher", "emd1", "emd2", "emdc"]),
    "grid7_servetransport": ("GRID7_servetransport",
                             ["teacher", "nr", "nr_rm05", "nr_rm10"]),
    "heads_and_locks":   ("(no video grid)", ["teacher", "4rung", "lock", "emdz"]),
    "grid1_servesampler": ("GRID", ["teacher", "4rung", "nr", "det",
                                    "inv", "detinv", "hyb", "lock"]),
    "grid2_criticchroma": ("GRID2", ["teacher", "nr_chroma", "nr_hyb",
                                     "nrcg_chroma", "nrcg_hyb", "lock"]),
}


def tile(name, labels, counterpart):
    paths = [f"{FV}/flow_tree_{l}_AR_timeline.mp4" for l in labels]
    miss = [l for l, p in zip(labels, paths) if not os.path.exists(p)]
    if miss:
        print(f"[ftg] SKIP {name}: no tree for {miss}")
        return
    rds = [imageio.get_reader(p, format="ffmpeg", **FP) for p in paths]
    cols = 2 if len(labels) <= 4 else 3
    rows = (len(labels) + cols - 1) // cols
    out = f"{FV}/FTGRID_{name}.mp4"
    w = imageio.get_writer(out, fps=10, codec="libx264", quality=7,
                           macro_block_size=1, ffmpeg_params=["-threads", "1"])
    last = [None] * len(labels)
    H = W = None
    n = 0
    while True:
        alive, frames = 0, []
        for i, r in enumerate(rds):
            f = None
            if rds[i] is not None:
                try:
                    f = rds[i].get_next_data(); alive += 1
                except Exception:
                    rds[i].close(); rds[i] = None
            if f is None:
                f = last[i]
            if f is not None and H is None:
                H, W = f.shape[:2]
            if f is None:
                f = np.full((H or 704, W or 704, 3), 255, np.uint8)
            if f.shape[:2] != (H, W):
                f = np.asarray(Image.fromarray(f).resize((W, H)))
            last[i] = f
            frames.append(f)
        if alive == 0:
            break
        while len(frames) < rows * cols:                 # pad short grids
            frames.append(np.full((H, W, 3), 255, np.uint8))
        grid = np.concatenate(
            [np.concatenate(frames[r * cols:(r + 1) * cols], 1)
             for r in range(rows)], 0)
        im = Image.fromarray(grid); d = ImageDraw.Draw(im)
        d.text((8, 6), f"FLOW TREES — {name}   (pixels: {counterpart})",
               fill=(0, 0, 0))
        w.append_data(np.asarray(im)); n += 1
    w.close()
    for r in rds:
        if r is not None:
            r.close()
    print(f"[ftg] saved {out} ({n} frames, {rows}x{cols})")


def main():
    only = [x for x in os.environ.get("FTG_ONLY", "").split(":") if x]
    for name, (counterpart, labels) in GRIDS.items():
        if only and name not in only:
            continue
        tile(name, labels, counterpart)


if __name__ == "__main__":
    main()
